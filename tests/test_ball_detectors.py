"""The ball backends' interface and the GridTrackNet prepass on a fake decoder; no GPU or weights."""
import threading
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import torch

from tennis_tracker.engines import ball_detectors
from tennis_tracker.engines.ball_detectors import (
    BallDetectorBackend,
    CachedDetections,
    GridTrackNetBallDetector,
    TensorRTRuntimeBallDetector,
)
from tennis_tracker.config import Config
from tests._helpers import Capture, CpuGridTrackNet


def test_ball_boxes_are_clipped_onto_the_frame():
    preds = torch.tensor([[[-2.0, 1.0, 5.0, 3.0, 0.9, 0.0]]])      # a postprocessed row past the left edge
    [(box, conf)] = ball_detectors.decode_ball_predictions(preds, (10.0, 10.0), 0.5, "postprocessed",
                                                           frame_size=(40, 25))
    assert box == [0.0, 10.0, 39.0, 24.0] and conf == pytest.approx(0.9)


def test_backend_interface_and_cached_token():
    backend = BallDetectorBackend()
    assert not backend.supports_cuda_frame()
    assert backend.prepare_video(Path("clip.mp4"), 30.0, 64, 32, 10) is None   # nothing to prepare per frame
    with pytest.raises(NotImplementedError):
        backend.detect(np.zeros((4, 4, 3), np.uint8), 0)
    assert GridTrackNetBallDetector.uses_raw_frames and not TensorRTRuntimeBallDetector.uses_raw_frames
    token = CachedDetections(3, [([1.0, 2.0, 3.0, 4.0], 0.5)])
    assert GridTrackNetBallDetector.detect_async_finish(None, token) == token.detections


def test_gridtracknet_constructor_requires_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError):
        GridTrackNetBallDetector("unused.npz", Config())


# --- the prepass on a fake decoder -------------------------------------------

class RaisingCapture(Capture):
    """A capture whose decoder fails on frame ``fail_at``."""

    def __init__(self, count, fail_at):
        super().__init__(count)
        self.fail_at = fail_at

    def read(self):
        if self.index == self.fail_at:
            raise MemoryError("decoder out of memory")
        return super().read()


def _bare_detector(cfg=None) -> GridTrackNetBallDetector:
    """A GridTrackNet backend without its constructor: the model, decoder and
    frame encoder are stand-ins that find a ball at (10, 20) on every frame."""
    detector = GridTrackNetBallDetector.__new__(GridTrackNetBallDetector)
    detector.cfg = cfg or Config(device="0")
    detector.device = torch.device("cpu")
    detector.model = lambda batch: batch
    detector._frame_tensor = lambda frame, device: torch.zeros((1, 3))
    detector._decode = lambda output, width, height, conf: [((10.0, 20.0), 0.9)] * (int(output.shape[0]) * 5)
    detector.precomputed = None
    detector._ready_upto = 0
    detector._done = False
    detector._ready = threading.Condition()
    detector._worker = None
    detector._worker_error = None
    return detector


def test_frame_index_is_required():
    detector = _bare_detector()
    detector.precomputed = [[]]
    detector._done = True
    with pytest.raises(ValueError, match="frame_index"):
        detector.detect(np.zeros((2, 2, 3), np.uint8), None)
    with pytest.raises(RuntimeError, match="prepare_video"):
        _bare_detector().detect(np.zeros((2, 2, 3), np.uint8), 0)


def test_prepass_grows_the_cache_past_an_under_reported_frame_count(monkeypatch, fake_cuda, capsys):
    detector = _bare_detector()
    monkeypatch.setattr(ball_detectors.cv2, "VideoCapture", lambda path: Capture(7))
    results = detector.detect_clip(Path("clip.mp4"), 30.0, 1920, 1080, total=3)
    assert len(results) == 7                      # the hint said 3; every decoded frame has an answer
    assert all(len(frame) == 1 for frame in results)
    assert [detector.detect(None, i)[0][1] for i in range(7)] == [0.9] * 7
    with pytest.raises(IndexError, match="decoded 7 frames; source frame 7"):
        detector.detections_at(7)
    out = capsys.readouterr().out
    assert "container reported 3 frames, 7 decoded" in out


def test_foreground_prepass_failure_raises_instead_of_returning_empty_frames(monkeypatch, fake_cuda):
    detector = _bare_detector()
    monkeypatch.setattr(ball_detectors.cv2, "VideoCapture", lambda path: RaisingCapture(7, fail_at=2))
    with pytest.raises(RuntimeError, match="prepass failed") as info:
        detector.detect_clip(Path("clip.mp4"), 30.0, 1920, 1080, total=7)
    assert isinstance(info.value.__cause__, MemoryError)


def test_background_prepass_failure_reaches_the_consumer(monkeypatch, fake_cuda):
    detector = _bare_detector(Config(device="0", gridtracknet_prepass_background=True))
    monkeypatch.setattr(ball_detectors.cv2, "VideoCapture", lambda path: RaisingCapture(7, fail_at=2))
    detector.prepare_video(Path("clip.mp4"), 30.0, 1920, 1080, total=7)
    with pytest.raises(RuntimeError, match="prepass failed"):
        detector.detections_at(5)


# --- the prepass through the real constructor on the CPU ----------------------

# Frame rates whose phase stride is 1 and 2 (see tennis_tracker.frame_rate).
_FPS_FOR_STRIDE = {1: 30.0, 2: 60.0}


@pytest.mark.parametrize("stride, count", [(1, 23), (2, 47)])
def test_finalized_misses_are_immutable(stride, count):
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
            assert detector.precomputed[:len(previous)] == previous
        publications.append([list(row) for row in detector.precomputed[:upto]])
    detector = CpuGridTrackNet(
        Config(device="0"), model=lambda batch: batch, decode=decode,
        frame_tensor=lambda frame, device: torch.tensor([float(frame[0, 0, 0])]),
        on_publish=publish,
    )
    with patch.object(cv2, "VideoCapture", return_value=Capture(count)), \
         patch.object(torch.cuda, "synchronize"):
        results = detector.detect_clip(Path("synthetic.mp4"), _FPS_FOR_STRIDE[stride], 1920, 1080, count)
    assert len(results) == count
    assert not any(results)
    assert [detector.detections_at(index) for index in range(count)] == [[]] * count
