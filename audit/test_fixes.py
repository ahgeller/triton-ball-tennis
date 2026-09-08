"""Synthetic regression checks. No match media or labels are used."""
from pathlib import Path
import queue
import json
import tempfile
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from review_checks import Capture, metric_probe
from finetune.train_gridtracknet import better
from tennis_tracker.config import Config
from tennis_tracker.detectors import GridTrackNetBallDetector
from tennis_tracker.video_io import VideoWriter
from tennis_tracker.motion import preprocess_frame_cuda, refine_raw_motion_temporal_cpu, filter_boost_mask
from ball_in_play_selector.config import SelectorConfig
from ball_in_play_selector.physics import BallKalmanFilter, _predict_projectile, _predict_projectile_vel
from ball_in_play_selector.core import _motion_near
from finetune import label_tool, realign
from finetune import ft
from finetune.data_policy import is_verified
from validate_tracking import validate
from ball_in_play_selector.core import _result
from tennis_tracker.pipeline import _frame_result_to_json


class FixTests(unittest.TestCase):
    def test_manual_scaling_and_unused_filters_are_removed(self):
        for field in ("court_depth", "court_side", "motion_flicker_suppress",
                      "motion_raw_ball_color_gate", "motion_raw_component_filter"):
            self.assertNotIn(field, Config.__dataclass_fields__)

    def test_empty_motion_clears_reused_boost_buffer(self):
        buffers = SimpleNamespace(boost_mask_u8=np.full((10, 10), 255, np.uint8))
        result = filter_boost_mask(np.zeros((10, 10), np.uint8), 0, 600, Config(), buffers=buffers)
        self.assertFalse(np.any(result))

    def test_verified_policy_does_not_clear_grid_matches(self):
        policy = {"verified_video_min": 13}
        for name in ("video3", "video8", "video11", "video12", "grid_match13", "grid_match92"):
            self.assertFalse(is_verified(name, policy))
        for name in ("video13", "video53", "video100"):
            self.assertTrue(is_verified(name, policy))

    def test_shift_refuses_review_metadata_without_touching_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            path = folder / "example_ball.csv"
            original = "frame,ball_x,ball_y\nframe_001,20,30\n"
            path.write_text(original)
            path.with_suffix(".review.json").write_text(json.dumps({"reviewed": [1]}))
            with patch.object(ft, "LABELS", folder), self.assertRaises(ValueError):
                ft.shift_labels("example", 1, 1920, 1080, 10)
            self.assertEqual(path.read_text(), original)

    def test_localization_recall_counts_wrong_positions_and_misses(self):
        predictions = {"frames": [{"frame": 0, "present": True, "x": 1000, "y": 1000}]}
        labels = {"ball": [{"frame": i, "visible": True, "x": 10, "y": 10} for i in range(2)]}
        summary = validate(predictions, labels, 200, 2)["summary"]
        self.assertEqual(summary["presence_recall"], .5)
        self.assertEqual(summary["localization_recall_10px"], 0.)

    def test_export_preserves_measurement_after_position_changes(self):
        detection = SimpleNamespace(cx=100., cy=103., conf=.9, x1=98, y1=101, x2=102, y2=105)
        result = _result(100, 103, "det", detection=detection)
        result.cy = 100.5
        row = _frame_result_to_json(0, result)
        self.assertEqual(row["measurement"]["y"], 103.)
        self.assertEqual(row["y"], 100.5)
        self.assertEqual(row["position_kind"], "filtered")

    def test_duplicate_video_frames_keep_sequential_identity(self):
        class DuplicateCapture(Capture):
            def read(self):
                ok, frame = super().read()
                return ok, np.full_like(frame, 42) if ok else None
        source = SimpleNamespace(position=0, capture=DuplicateCapture(3), sig_of={}, end=None)
        tool = SimpleNamespace(source=source, signature=label_tool.Tool.signature)
        tool.remember = lambda index, sig: label_tool.Tool.remember(tool, index, sig)
        for index in range(3):
            self.assertIsNotNone(label_tool.Tool.decode_at(tool, index))
        self.assertEqual(source.position, 3)

    def test_human_pin_is_a_hard_constraint(self):
        n = 220
        bases = np.arange(n, dtype=np.int64) * 2
        lut_x = np.zeros(n * 2 + 3, dtype=np.float32)
        lut_y = np.zeros_like(lut_x)
        lut_x[bases] = 1000
        pinned = np.zeros(n, bool)
        pinned[n // 2] = True
        offsets = realign.solve(bases, np.zeros(n), np.zeros(n), lut_x, lut_y,
                                len(lut_x), 1., np.array([0, 1]), pinned)
        self.assertEqual(offsets[n // 2], 0)

    def test_projectile_equal_timestamps_and_filter_consistency(self):
        cfg30 = SelectorConfig(fps=30).auto_scale()
        cfg60 = SelectorConfig(fps=60).auto_scale()
        np.testing.assert_allclose(_predict_projectile((100, 100), (30, -10), 30, cfg30),
                                   _predict_projectile((100, 100), (15, -5), 60, cfg60), rtol=1e-10)
        np.testing.assert_allclose(_predict_projectile_vel((30, -10), 30, cfg30),
                                   np.array(_predict_projectile_vel((15, -5), 60, cfg60)) * 2, rtol=1e-10)
        kf = BallKalmanFilter(100, 100, cfg30)
        kf.kf.x[2:, 0] = [30, -10]
        predicted = kf.predict_dt(5)
        np.testing.assert_allclose(predicted, _predict_projectile((100, 100), (30, -10), 5, cfg30))
        for _ in range(5):
            actual = kf.predict()
        np.testing.assert_allclose(predicted, actual)
        self.assertTrue(np.all(np.linalg.eigvalsh(kf.kf.P) >= 0))

    def test_small_motion_component_uses_pixel_area(self):
        mask = np.zeros((100, 100), np.uint8)
        mask[49:52, 49:52] = 255
        result = _motion_near(mask, 50, 50, 175., 20., 4.)
        self.assertEqual((result["x"], result["y"], result["area"]), (50., 50., 9.))
        self.assertIsNone(_motion_near(mask, 0, 0, 175., 20., 4.))

    def test_motion_retains_non_yellow_evidence(self):
        frame = np.zeros((32, 32, 3), np.uint8)
        frame[10:20, 10:20, 2] = 255
        t = torch.from_numpy(frame).permute(2, 0, 1).float() / 255
        zeros = torch.zeros(32, 32)
        cfg = Config(motion_raw_temporal_gate=False, motion_raw_close_size=0,
                     motion_thresh=1., motion_v_min=0.)
        _, raw, _, _, _, _ = preprocess_frame_cuda(
            frame, zeros, zeros, zeros, zeros, cfg, frame_gpu_t=t,
            protect_mask_cuda_cached=zeros.bool(), skip_dim=True,
            need_cpu_frame=False, need_detector_boost=False)
        self.assertEqual(np.count_nonzero(raw), 100)
        expected = refine_raw_motion_temporal_cpu(raw, None, frame, None, cfg)
        np.testing.assert_array_equal(expected, raw)

    def test_metric_counts_localization_errors(self):
        stats = metric_probe(5, {i: [100, 100] if i == 0 else [120, 100] for i in range(5)})
        self.assertEqual(stats["visible"], 5)
        self.assertEqual(stats["recall"], .2)

    def test_metric_infers_tail(self):
        stats = metric_probe(7, {i: [100, 100] for i in range(7)})
        self.assertEqual(stats["visible"], 7)
        self.assertEqual(stats["recall"], 1.)

    def test_false_alarms_block_promotion(self):
        self.assertFalse(better({"recall": .91, "wrong": 0., "false_alarm": 1.},
                                {"recall": .90, "wrong": 0., "false_alarm": 0.}))

    def test_finalized_misses_are_immutable(self):
        for stride, count in ((1, 23), (2, 47)):
            with self.subTest(stride=stride):
                detector = object.__new__(GridTrackNetBallDetector)
                detector.cfg, detector.device = Config(), torch.device("cpu")
                detector.precomputed = [[] for _ in range(count)]
                detector._worker, detector._worker_error = None, None
                detector._frame_tensor = lambda frame, device: torch.tensor([float(frame[0, 0, 0])])
                detector.model = lambda batch: batch
                seen = set()
                def decode(batch, width, height, threshold):
                    decoded = []
                    for unit in batch:
                        for value in unit:
                            index = int(value)
                            decoded.append(((float(index), 20.) if index in seen else None, .9))
                            seen.add(index)
                    return decoded
                detector._decode = decode
                publications = []
                def publish(upto):
                    for previous in publications:
                        self.assertEqual(detector.precomputed[:len(previous)], previous)
                    publications.append([list(row) for row in detector.precomputed[:upto]])
                detector._publish = publish
                with patch.object(torch.cuda, "synchronize"):
                    detector._prepass(Capture(count), stride, 1920, 1080)
                self.assertIsNone(detector._worker_error)
                self.assertFalse(any(detector.precomputed))

    def test_writer_failure_does_not_block_full_queue(self):
        writer = object.__new__(VideoWriter)
        writer._proc = writer._cv = None
        writer._q = queue.Queue(maxsize=1)
        writer._q.put("queued")
        writer._thread_error = RuntimeError("encoder failed")
        writer._thread = threading.Thread(target=lambda: None)
        writer._thread.start()
        writer._thread.join()
        errors = []
        def close():
            try:
                writer.close()
            except RuntimeError as error:
                errors.append(str(error))
        closer = threading.Thread(target=close, daemon=True)
        closer.start()
        closer.join(1)
        self.assertFalse(closer.is_alive())
        self.assertIn("encoder failed", errors[0])


if __name__ == "__main__":
    unittest.main()
