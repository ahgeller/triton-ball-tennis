"""Synthetic fixtures shared by the regression tests. No match media or labels."""
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import torch

from finetune.training import train_gridtracknet as training
from tennis_tracker import ball_detectors
from tennis_tracker.ball_detectors import GridTrackNetBallDetector
from tennis_tracker.utils import ENCODER_OPENCV, EncoderChoice
from tennis_tracker.video_io import VideoWriter


def write_clip(path: Path, count: int, size=(64, 36), fps=30.0) -> Path:
    """A tiny FFV1 video of `count` frames, frame i filled with the value i (skips the test when
    this OpenCV build cannot write FFV1)."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), fps, size)
    if not writer.isOpened():
        pytest.skip("this OpenCV build cannot write FFV1")
    for index in range(count):
        writer.write(np.full((size[1], size[0], 3), index, np.uint8))
    writer.release()
    return path

# A detector output the metric probe returns for every five-frame unit.
_PROBE_POINT = (100, 100)
_PROBE_CONFIDENCE = 0.9
_PROBE_FPS = 30
_PROBE_SIZE = (1920, 1080)


class Capture:
    """A ``cv2.VideoCapture`` stand-in whose frame ``i`` is filled with the value ``i``."""

    def __init__(self, count: int):
        self.count, self.index = count, 0

    def isOpened(self) -> bool:  # noqa: N802 - mirrors cv2.VideoCapture
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.index >= self.count:
            return False, None
        frame = np.full((2, 2, 3), self.index, np.uint8)
        self.index += 1
        return True, frame

    def release(self) -> None:
        """Nothing to release; present because callers always release captures."""


def metric_probe(count: int, labels: Dict[int, Optional[List[float]]]) -> Dict[str, float]:
    """``detector_metrics`` on a ``count``-frame synthetic clip whose detector always fires at (100, 100)."""
    width, height = _PROBE_SIZE
    with patch.object(training, "load_model", return_value=lambda _: None), \
         patch.object(training, "video_meta", return_value=(_PROBE_FPS, width, height, count)), \
         patch.object(training, "training_targets", return_value=labels), \
         patch.object(training.cv2, "VideoCapture", side_effect=lambda _: Capture(count)), \
         patch.object(training, "frame_tensor", return_value=torch.zeros(3, 1, 1)), \
         patch.object(training, "decode_predictions",
                      return_value=[(_PROBE_POINT, _PROBE_CONFIDENCE)] * 5):
        return training.detector_metrics(
            Path("unused"), [("probe", Path("unused"), Path("unused"), "custom")], torch.device("cpu")
        )


class CpuGridTrackNet(GridTrackNetBallDetector):
    """The real constructor, run on the CPU with an injected model, decoder and frame encoder.

    The constructor insists on CUDA and loads weights from disk; both are
    replaced for its duration only. ``on_publish`` sees every prefix the
    prepass declares final.
    """

    def __init__(self, cfg, model, decode, frame_tensor, on_publish=None):
        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch, "device", return_value=torch.device("cpu")), \
             patch.object(torch.cuda, "synchronize"), \
             patch.object(ball_detectors, "load_model", return_value=model):
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
