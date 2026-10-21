"""Synthetic fixtures shared by the regression tests. No match media or labels."""
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import torch

from finetune.data.detector_metrics import DetectorMetrics
from finetune.data.workspace import VideoMeta
from finetune.training import metrics
from tennis_tracker.engines import ball_detectors
from tennis_tracker.engines.ball_detectors import GridTrackNetBallDetector
from tennis_tracker.config import Config
from tennis_tracker.motion import _PROBE_S_DIFF_SCALE, _clean_raw_motion_mask
from tennis_tracker.utils import ENCODER_OPENCV, EncoderChoice
from tennis_tracker.video_io import VideoWriter


def write_frames(path: Path, frames: Sequence[np.ndarray], fps: float = 30.0) -> Path:
    """Encode `frames` losslessly (FFV1 in AVI), so what a test measures is the synthetic picture
    itself and not an encoder's noise; skips the test when this OpenCV build cannot write FFV1."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), fps, (frames[0].shape[1], frames[0].shape[0]))
    if not writer.isOpened():
        pytest.skip("this OpenCV build cannot write FFV1")
    for image in frames:
        writer.write(image)
    writer.release()
    return path


def write_clip(path: Path, count: int, size=(64, 36), fps=30.0) -> Path:
    """A tiny video of `count` frames, frame i filled with the value i (see write_frames)."""
    return write_frames(path, [np.full((size[1], size[0], 3), index, np.uint8) for index in range(count)], fps)


def ball_frames(positions: Iterable[Tuple[int, int]], size=(360, 640)) -> List[np.ndarray]:
    """Grey frames with a 3x3 bright ball whose middle pixel is at each (column, row)."""
    frames = []
    for column, row in positions:
        image = np.full(size + (3,), 90, np.uint8)
        image[row - 1:row + 2, column - 1:column + 2] = (60, 230, 230)
        frames.append(image)
    return frames


# --------------------------------------------------------------------------- motion

def motion_probe_delta_from_frames(prev_bgr: np.ndarray, curr_bgr: np.ndarray) -> np.ndarray:
    """The C++ probe's frame delta on the CPU: max(gray, V, scaled-S) differences, blurred 3x3."""
    prev_hsv = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2HSV)
    curr_hsv = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2HSV)
    prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
    v_diff = cv2.absdiff(curr_hsv[:, :, 2], prev_hsv[:, :, 2])
    s_diff = cv2.absdiff(curr_hsv[:, :, 1], prev_hsv[:, :, 1])
    gray_diff = cv2.absdiff(curr_gray, prev_gray)
    s_scaled = np.minimum(s_diff.astype(np.float32) * _PROBE_S_DIFF_SCALE, 255.0).astype(np.uint8)
    combined = np.maximum(gray_diff, np.maximum(v_diff, s_scaled))
    return cv2.GaussianBlur(combined, (3, 3), 0)


def refine_raw_motion_temporal_cpu(
    raw_motion_u8: Optional[np.ndarray],
    prev_bgr: Optional[np.ndarray],
    curr_bgr: Optional[np.ndarray],
    next_bgr: Optional[np.ndarray],
    cfg: Config,
) -> Optional[np.ndarray]:
    """The CPU reference for the temporal gate ``motion._raw_motion_cuda`` applies: the
    ungated foreground kept where the C++ probe's movement test passes, then the raw-mask
    morphology.  Only the tests use it; the runtime runs the CUDA path."""
    if raw_motion_u8 is None:
        return None
    refined = raw_motion_u8
    if cfg.motion_raw_temporal_gate and prev_bgr is not None and curr_bgr is not None:
        hi, lo, very_hi = cfg.motion_raw_temporal_hi, cfg.motion_raw_temporal_lo, cfg.motion_raw_temporal_very_hi
        d_prev = motion_probe_delta_from_frames(prev_bgr, curr_bgr)
        if next_bgr is not None:
            d_next = motion_probe_delta_from_frames(curr_bgr, next_bgr)
            temporal = ((d_prev >= hi) & (d_next >= lo)) | ((d_next >= hi) & (d_prev >= lo)) | (d_prev >= very_hi)
        else:
            temporal = d_prev >= hi
        refined = cv2.bitwise_and(refined, temporal.astype(np.uint8) * 255)
    return _clean_raw_motion_mask(refined, cfg)


# --------------------------------------------------------------------------- the label tool

def fake_source(clip: str, count: int = 6, width: int = 640, height: int = 360, origin=(100.0, 80.0),
                step=(5.0, 2.0), conf: float = 0.0, **extra) -> SimpleNamespace:
    """A ``label_source.Source`` stand-in with no video: `count` rows of `clip` along a line from
    `origin` in steps of `step`, every row visible at `conf`, and the fields the label tool reads.
    `extra` adds or overrides fields (label_cli reads video, flagged and zoom)."""
    from finetune.labeling.label_store import Row

    rows = [Row(i, origin[0] + i * step[0], origin[1] + i * step[1], True, "csv", conf) for i in range(count)]
    for row in rows:
        row.clip = clip
    capture = SimpleNamespace(release=lambda: None, isOpened=lambda: False)
    fields = dict(clip=clip, rows=rows, width=width, height=height, frame_cache={}, dirty=False, cursor=0, start=0,
                  missing=set(), saved_cursor=0, capture=capture, position=0, release=lambda: None)
    return SimpleNamespace(**{**fields, **extra})


def fake_tool(*sources: SimpleNamespace, follow: bool = True):
    """A headless ``label_tool.Tool`` over `sources` (one fake_source by default) whose frames are
    blank arrays, one per (clip, frame) as the real cache hands back, centred on its first ball
    (`follow`); no window is opened."""
    from finetune.labeling.label_tool import Tool

    tool = Tool(list(sources or [fake_source("a")]))
    frames: Dict[Tuple[str, int], np.ndarray] = {}
    for source in tool.session.sources:
        source.frame = lambda index, source=source: frames.setdefault(
            (source.clip, index), np.zeros((source.height, source.width, 3), np.uint8))
    if follow:
        tool.follow_ball()
    return tool

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


def cpu_detector(decode, conf: float = 0.5, frame_tensor=None) -> "CpuGridTrackNet":
    """A GridTrackNet backend on the CPU whose model is the identity and whose decoder is
    ``decode(output, width, height, threshold)``; each frame's tensor is its first pixel's value
    unless ``frame_tensor`` says otherwise."""
    return CpuGridTrackNet(Config(device="0", conf=conf), model=lambda batch: batch, decode=decode,
                           frame_tensor=frame_tensor or (lambda frame, device: torch.tensor([float(frame[0, 0, 0])])))


def metric_probe(count: int, labels: Dict[int, Optional[List[float]]]) -> DetectorMetrics:
    """``score_weights`` on a ``count``-frame synthetic clip whose detector always fires at (100, 100),
    run through the real prepass (``GridTrackNetBallDetector.detect_clip``) on the CPU."""
    width, height = _PROBE_SIZE
    detector = cpu_detector(lambda output, w, h, threshold: [(_PROBE_POINT, _PROBE_CONFIDENCE)] * (5 * output.reshape(-1, 5).shape[0]))
    with patch.object(metrics, "ball_detector", return_value=detector), \
         patch.object(metrics, "video_meta", return_value=VideoMeta(_PROBE_FPS, width, height, count)), \
         patch.object(metrics, "training_targets", return_value=labels), \
         patch.object(ball_detectors.cv2, "VideoCapture", side_effect=lambda _: Capture(count)), \
         patch.object(torch.cuda, "synchronize"):
        return metrics.score_weights(
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
