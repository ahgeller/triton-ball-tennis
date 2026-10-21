"""The ball backends' interface; no GPU or weights."""
import numpy as np
import pytest
import torch

from tennis_tracker.ball_detectors import (
    BallDetectorBackend,
    CachedDetections,
    GridTrackNetBallDetector,
    TensorRTRuntimeBallDetector,
)
from tennis_tracker.config import Config


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
