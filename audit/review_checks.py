"""Reproduce review findings without changing models, media, or labels.

These assertions document CURRENT defects, not desired behavior. Replace each
with a regression test for the corrected behavior when implementing its fix.
Run from the repository root with the tennis-analysis interpreter.
"""
from pathlib import Path
import json
import queue
import sys
import threading
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from finetune import train_gridtracknet as training
from finetune import label_tool, realign
from tennis_tracker.config import Config
from tennis_tracker.detectors import GridTrackNetBallDetector
from tennis_tracker.video_io import VideoWriter
from tennis_tracker.motion import preprocess_frame_cuda, refine_raw_motion_temporal_cpu
from ball_in_play_selector.config import SelectorConfig
from ball_in_play_selector.physics import BallKalmanFilter, _predict_projectile, _predict_projectile_vel
from ball_in_play_selector.core import _motion_near, _refine_trajectory
from ball_in_play_selector.models import FrameResult
from validate_tracking import validate


class Capture:
    def __init__(self, count):
        self.count, self.index = count, 0

    def read(self):
        if self.index >= self.count:
            return False, None
        frame = np.full((2, 2, 3), self.index, np.uint8)
        self.index += 1
        return True, frame

    def release(self):
        pass


def metric_probe(count, labels):
    with patch.object(training, "load_model", return_value=lambda _: None), \
         patch.object(training, "video_meta", return_value=(30, 1920, 1080, count)), \
         patch.object(training, "read_labels", return_value=labels), \
         patch.object(training.cv2, "VideoCapture", side_effect=lambda _: Capture(count)), \
         patch.object(training, "frame_tensor", return_value=torch.zeros(3, 1, 1)), \
         patch.object(training, "decode_predictions", return_value=[((100, 100), .9)] * 5):
        return training.detector_metrics(Path("unused"), [("probe", Path("unused"), Path("unused"), "custom")], torch.device("cpu"))


def main():
    findings = {}
    labels = {i: [100, 100] if i == 0 else [120, 100] for i in range(5)}
    stats = metric_probe(5, labels)
    assert stats["visible"] == 1 and stats["recall"] == 1.0
    findings["metric_denominator"] = {"reported": stats, "expected_visible": 5, "expected_recall": .2}

    stats = metric_probe(7, {i: [100, 100] for i in range(7)})
    assert stats["recall"] == 5 / 7
    findings["metric_tail"] = {"reported_recall": stats["recall"], "frames_never_inferred": [5, 6]}

    winner = training.better({"recall": .91, "wrong": 0., "false_alarm": 1.},
                             {"recall": .90, "wrong": 0., "false_alarm": 0.})
    assert winner
    findings["promotion_false_alarms"] = "100% false-alarm candidate beats 0% incumbent for +1% recall"

    cfg30 = SelectorConfig(fps=30).auto_scale()
    cfg60 = SelectorConfig(fps=60).auto_scale()
    v30 = _predict_projectile_vel((100, 0), 30, cfg30)[0] / 100
    v60 = _predict_projectile_vel((50, 0), 60, cfg60)[0] / 50
    assert abs(v30 - v60) > .2
    findings["fps_drag"] = {"one_second_speed_retained_30fps": v30, "one_second_speed_retained_60fps": v60}

    kf = BallKalmanFilter(100, 100, cfg30)
    kf.kf.x[2:, 0] = [30, -10]
    a = kf.predict_dt(5)
    b = _predict_projectile((100, 100), (30, -10), 5, cfg30)
    assert np.linalg.norm(np.array(a) - b) > 2
    findings["prediction_order"] = {"kalman": a, "projectile": b}

    mask = np.zeros((100, 100), np.uint8)
    mask[49:52, 49:52] = 255
    radius = .003 * np.hypot(1920, 1080)
    area = (2 * radius) ** 2
    assert _motion_near(mask, 50, 50, area, 20, 4) is None
    findings["small_blob_rejection"] = {"ball_pixels": 9, "contour_area": 4, "minimum_contour_area": .2 * area}

    results = [FrameResult(cx=float(i * 10), cy=100., conf=.9) for i in range(15)]
    results[7].cy = 103.
    _refine_trajectory(results, 2203.)
    assert results[7].source == "det" and abs(results[7].cy - 103.) > .01
    findings["raw_provenance"] = {"raw_y": 103., "export_y": results[7].cy, "source": results[7].source}

    stats = validate({"frames": [{"frame": 0, "present": True, "x": 900, "y": 900}]},
                     {"ball": [{"frame": 0, "visible": True, "x": 10, "y": 10}]}, 120, 1)["summary"]
    assert stats["recall"] == 1 and stats["within_20px"] == 0
    findings["presence_recall"] = {"recall": stats["recall"], "error_px": stats["mean_error_px"]}

    # No GPU needed: exercise the actual prepass publication and tail policy.
    detector = object.__new__(GridTrackNetBallDetector)
    detector.cfg, detector.device = Config(), torch.device("cpu")
    detector.precomputed = [[] for _ in range(23)]
    detector._worker, detector._worker_error = None, None
    detector._frame_tensor = lambda frame, device: torch.tensor([float(frame[0, 0, 0])])
    detector.model = lambda batch: batch
    calls = 0
    def decode(batch, width, height, threshold):
        nonlocal calls
        calls += 1
        return [(None if calls == 1 and int(i) == 18 else (float(i), 20.), .9)
                for unit in batch for i in unit]
    detector._decode = decode
    published = []
    detector._publish = lambda upto: published.append((upto, bool(detector.precomputed[18])))
    with patch.object(torch.cuda, "synchronize"):
        detector._prepass(Capture(23), 1, 1920, 1080)
    assert published[0] == (20, False) and detector.precomputed[18]
    findings["published_frame_mutation"] = {"publications": published, "changed_frame": 18}

    # Deterministically reproduce close() blocking after an async writer failure.
    writer = object.__new__(VideoWriter)
    writer._q = queue.Queue(maxsize=1)
    writer._q.put("queued frame")
    writer._thread_error = RuntimeError("simulated encoder failure")
    writer._thread = threading.Thread(target=lambda: None)
    writer._thread.start()
    writer._thread.join()
    errors = []
    entered = threading.Event()
    def close_writer():
        entered.set()
        try:
            writer.close()
        except RuntimeError as error:
            errors.append(str(error))
    closer = threading.Thread(target=close_writer, daemon=True)
    closer.start()
    entered.wait(1)
    closer.join(.2)
    blocked = closer.is_alive()
    writer._q.get()  # release the deliberately blocked probe; leave no worker behind
    closer.join(1)
    assert blocked and errors and not closer.is_alive()
    findings["writer_deadlock"] = "close blocks on full queue after consumer failure"

    # Identical decoded frames are normal in duplicated/frozen video; they are
    # not unique temporal identities. No GUI or real capture is opened here.
    class DuplicateCapture(Capture):
        def read(self):
            ok, frame = super().read()
            return ok, np.full_like(frame, 42) if ok else None
    source = SimpleNamespace(position=0, capture=DuplicateCapture(3), index_of={}, sig_of={})
    tool = SimpleNamespace(source=source, signature=label_tool.Tool.signature)
    tool.remember = lambda index, sig: label_tool.Tool.remember(tool, index, sig)
    first = label_tool.Tool.decode_at(tool, 0)
    second = label_tool.Tool.decode_at(tool, 1)
    assert first is not None and second is None and source.position == 1
    findings["duplicate_frame_identity"] = "correct sequential frame 1 rejected because its pixels equal frame 0"

    # Pins are finite penalties: long enough evidence can override even a
    # reviewed visible frame at rate=1, without exercising destructive apply.
    n = 220
    bases = np.arange(n, dtype=np.int64) * 2
    lut_x = np.zeros(n * 2 + 3, dtype=np.float32)
    lut_y = np.zeros_like(lut_x)
    lut_x[bases] = 1000
    pinned = np.zeros(n, bool)
    pinned[n // 2] = True
    offsets = realign.solve(bases, np.zeros(n), np.zeros(n), lut_x, lut_y,
                           len(lut_x), 1., np.array([0, 1]), pinned)
    assert offsets[n // 2] != 0
    findings["realign_soft_pins"] = {"reviewed_frame": int(bases[n // 2]), "chosen_offset": int(offsets[n // 2])}

    red = np.zeros((32, 32, 3), np.uint8)
    red[10:20, 10:20, 2] = 255
    frame_t = torch.from_numpy(red).permute(2, 0, 1).float() / 255
    zeros = torch.zeros(32, 32)
    color_cfg = Config(motion_raw_temporal_gate=False, motion_raw_ball_color_gate=True,
                       motion_raw_close_size=0, motion_thresh=1., motion_v_min=0.)
    _, cuda_raw, _, _, _, _ = preprocess_frame_cuda(
        red, zeros, zeros, zeros, zeros, color_cfg,
        frame_gpu_t=frame_t, protect_mask_cuda_cached=zeros.bool(),
        skip_dim=True, need_cpu_frame=False, need_detector_boost=False)
    cpu_raw = refine_raw_motion_temporal_cpu(cuda_raw.copy(), None, red, None, color_cfg)
    assert np.count_nonzero(cuda_raw) > 0 and np.count_nonzero(cpu_raw) == 0
    findings["cuda_color_gate_unused"] = {"cuda_path_red_pixels": int(np.count_nonzero(cuda_raw)),
                                           "cpu_path_red_pixels": int(np.count_nonzero(cpu_raw))}

    destination = ROOT / "audit" / "review_checks.json"
    destination.write_text(json.dumps(findings, indent=2), encoding="utf-8")
    print(json.dumps(findings, indent=2))
    print(f"Reproduced {len(findings)} current findings; saved {destination}")


if __name__ == "__main__":
    main()
