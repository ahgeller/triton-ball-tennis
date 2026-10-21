"""Synthetic regression checks for the reviewed defects. No match media or labels are used."""
from pathlib import Path
import json
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import numpy as np

from tests._helpers import Capture, metric_probe
from finetune.train_gridtracknet import better
from tennis_tracker import detectors
from tennis_tracker.config import Config
from tennis_tracker.detectors import GridTrackNetBallDetector
from tennis_tracker.utils import ENCODER_OPENCV, EncoderChoice
from tennis_tracker.video_io import VideoWriter
from tennis_tracker.frame_window import FrameWindow
from tennis_tracker.motion import BackgroundModel, preprocess_frame_cuda, refine_raw_motion_temporal_cpu, filter_boost_mask
from ball_in_play_selector.config import SelectorConfig
from ball_in_play_selector.physics import BallKalmanFilter, predict_projectile, predict_projectile_velocity
from ball_in_play_selector.core import motion_near
from finetune import label_tool, realign
from finetune import label_audit
from finetune.data_policy import is_verified
from validate_tracking import validate
from ball_in_play_selector.core import frame_result
from tennis_tracker.export import frame_result_to_json

# Frame rates whose phase stride is 1 and 2 (see tennis_tracker.frame_rate).
_FPS_FOR_STRIDE = {1: 30.0, 2: 60.0}
_CPU = torch.device("cpu")


class CpuGridTrackNet(GridTrackNetBallDetector):
    """The real constructor, run on the CPU with an injected model, decoder and frame encoder.

    The constructor insists on CUDA and loads weights from disk; both are
    replaced for its duration only. ``on_publish`` sees every prefix the
    prepass declares final.
    """

    def __init__(self, cfg, model, decode, frame_tensor, on_publish=None):
        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch, "device", return_value=_CPU), \
             patch.object(torch.cuda, "synchronize"), \
             patch.object(detectors, "load_model", return_value=model):
            super().__init__("unused.npz", cfg)
        self._decode = decode
        self._frame_tensor = frame_tensor
        self._on_publish = on_publish

    def _publish(self, upto):
        super()._publish(upto)
        if self._on_publish is not None:
            self._on_publish(upto)


class FailingWriter(VideoWriter):
    """An OpenCV-backed writer whose encoder blocks on the first frame until released, then fails."""

    def __init__(self, path, cfg):
        self.entered, self.release, self.drained = threading.Event(), threading.Event(), threading.Event()
        super().__init__(path, 30.0, 64, 32, cfg, encoder=EncoderChoice(None, (ENCODER_OPENCV,)))

    def _raw_write(self, frame):
        self.entered.set()
        self.release.wait(1)
        raise RuntimeError("encoder failed")

    def _drain(self):
        try:
            super()._drain()
        finally:
            self.drained.set()


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
            with patch.object(label_audit, "LABELS", folder), self.assertRaises(ValueError):
                label_audit.shift_labels("example", 1, 1920, 1080, 10)
            self.assertEqual(path.read_text(), original)

    def test_localization_recall_counts_wrong_positions_and_misses(self):
        predictions = {"frames": [{"frame": 0, "present": True, "x": 1000, "y": 1000}]}
        labels = {"ball": [{"frame": i, "visible": True, "x": 10, "y": 10} for i in range(2)]}
        summary = validate(predictions, labels, 200, 2)["summary"]
        self.assertEqual(summary["presence_recall"], .5)
        self.assertEqual(summary["localization_recall_10px"], 0.)

    def test_export_preserves_measurement_after_position_changes(self):
        detection = SimpleNamespace(cx=100., cy=103., conf=.9, x1=98, y1=101, x2=102, y2=105)
        result = frame_result(100, 103, "det", detection=detection)
        result.cy = 100.5
        row = frame_result_to_json(0, result)
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
        np.testing.assert_allclose(predict_projectile((100, 100), (30, -10), 30, cfg30),
                                   predict_projectile((100, 100), (15, -5), 60, cfg60), rtol=1e-10)
        np.testing.assert_allclose(predict_projectile_velocity((30, -10), 30, cfg30),
                                   np.array(predict_projectile_velocity((15, -5), 60, cfg60)) * 2, rtol=1e-10)
        kf = BallKalmanFilter(100, 100, cfg30)
        kf.kf.x[2:, 0] = [30, -10]
        predicted = kf.predict_dt(5)
        np.testing.assert_allclose(predicted, predict_projectile((100, 100), (30, -10), 5, cfg30))
        for _ in range(5):
            actual = kf.predict()
        np.testing.assert_allclose(predicted, actual)
        self.assertTrue(np.all(np.linalg.eigvalsh(kf.kf.P) >= 0))

    def test_small_motion_component_uses_pixel_area(self):
        mask = np.zeros((100, 100), np.uint8)
        mask[49:52, 49:52] = 255
        result = motion_near(mask, 50, 50, 175., 20., 4.)
        self.assertEqual((result["x"], result["y"], result["area"]), (50., 50., 9.))
        self.assertIsNone(motion_near(mask, 0, 0, 175., 20., 4.))

    def test_motion_retains_non_yellow_evidence(self):
        frame = np.zeros((32, 32, 3), np.uint8)
        frame[10:20, 10:20, 2] = 255
        zeros = torch.zeros(32, 32)
        cfg = Config(motion_raw_temporal_gate=False, motion_raw_close_size=0,
                     motion_thresh=1., motion_v_min=0.)
        window = FrameWindow(torch.device("cpu"), True, frame, None)
        background = BackgroundModel(cfg, torch.device("cpu"))
        background.mean_v = background.mean_s = background.var_v = background.var_s = zeros
        raw = preprocess_frame_cuda(
            window, background, cfg, protect_mask=zeros.bool(), skip_dim=True,
            need_cpu_frame=False, need_detector_boost=False).raw_motion
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
                seen = set()
                def decode(batch, width, height, threshold):
                    decoded = []
                    for unit in batch:
                        for value in unit:
                            index = int(value)
                            decoded.append(((float(index), 20.) if index in seen else None, .9))
                            seen.add(index)
                    return decoded
                publications = []
                detector = None
                def publish(upto):
                    for previous in publications:
                        self.assertEqual(detector.precomputed[:len(previous)], previous)
                    publications.append([list(row) for row in detector.precomputed[:upto]])
                detector = CpuGridTrackNet(
                    Config(device="0"), model=lambda batch: batch, decode=decode,
                    frame_tensor=lambda frame, device: torch.tensor([float(frame[0, 0, 0])]),
                    on_publish=publish,
                )
                with patch.object(detectors.cv2, "VideoCapture", return_value=Capture(count)), \
                     patch.object(torch.cuda, "synchronize"):
                    results = detector.detect_clip(Path("synthetic.mp4"), _FPS_FOR_STRIDE[stride], 1920, 1080, count)
                self.assertEqual(len(results), count)
                self.assertFalse(any(results))
                self.assertEqual([detector.detections_at(index) for index in range(count)], [[]] * count)

    def test_writer_failure_does_not_block_full_queue(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = FailingWriter(str(Path(directory) / "out.avi"),
                                   Config(use_async_writer=True, async_queue=1))
            frame = np.zeros((32, 64, 3), np.uint8)
            writer.write(frame)                # taken by the drain thread, which then blocks
            self.assertTrue(writer.entered.wait(1))
            writer.write(frame)                # fills the one-slot queue
            writer.release.set()               # the encoder now fails with the queue still full
            self.assertTrue(writer.drained.wait(1))
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
