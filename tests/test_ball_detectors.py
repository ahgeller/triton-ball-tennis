"""The ball backends' pure helpers and the facade that re-exports them; no GPU or weights."""
import numpy as np
import pytest
import torch

from tennis_tracker import ball_detectors, detectors
from tennis_tracker.ball_detectors import (
    BallDetectorBackend,
    CachedDetections,
    GridTrackNetBallDetector,
    TensorRTRuntimeBallDetector,
    nms_xyxy,
    xywh_to_xyxy,
)
from tennis_tracker.config import Config


def test_xywh_to_xyxy_converts_centre_boxes_and_keeps_extra_columns():
    boxes = np.array([[10.0, 20.0, 4.0, 6.0, 0.9]], dtype=np.float32)
    out = xywh_to_xyxy(boxes)
    np.testing.assert_allclose(out, [[8.0, 17.0, 12.0, 23.0, 0.9]])
    np.testing.assert_allclose(boxes, [[10.0, 20.0, 4.0, 6.0, 0.9]])  # the input is not modified


def test_nms_keeps_the_best_of_overlapping_boxes_in_score_order():
    boxes = np.array([
        [0.0, 0.0, 10.0, 10.0],
        [1.0, 1.0, 11.0, 11.0],   # overlaps the first, lower score
        [50.0, 50.0, 60.0, 60.0],
    ], dtype=np.float32)
    scores = np.array([0.8, 0.6, 0.9], dtype=np.float32)
    assert nms_xyxy(boxes, scores, 0.45).tolist() == [2, 0]
    assert nms_xyxy(boxes, scores, 0.95).tolist() == [2, 0, 1]
    assert nms_xyxy(np.empty((0, 4)), np.empty((0,)), 0.45).tolist() == []


def test_backend_interface_and_cached_token():
    backend = BallDetectorBackend()
    assert not backend.supports_cuda_frame()
    with pytest.raises(NotImplementedError):
        backend.detect(np.zeros((4, 4, 3), np.uint8))
    assert GridTrackNetBallDetector.uses_raw_frames and not TensorRTRuntimeBallDetector.uses_raw_frames
    token = CachedDetections(3, [([1.0, 2.0, 3.0, 4.0], 0.5)])
    assert GridTrackNetBallDetector.detect_async_finish(None, token) == token.detections


def test_gridtracknet_constructor_requires_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError):
        GridTrackNetBallDetector("unused.npz", Config())


def test_detectors_facade_re_exports_the_ball_names():
    for name in (
        "Detection", "FrameInput", "BallDetectorBackend", "CachedDetections",
        "GridTrackNetBallDetector", "TensorRTBallPending", "TensorRTRuntimeBallDetector",
        "ball_engine_output_mode", "decode_ball_predictions",
    ):
        assert getattr(detectors, name) is getattr(ball_detectors, name), name
