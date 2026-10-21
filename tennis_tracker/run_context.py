"""The data the stages of ``pipeline.run`` hand to each other, and what it returns.

``RunContext`` is what ``_open_inputs`` builds once per run: the validated
config, the input clip's metadata, the CUDA device, the encoder choice and
the three detector backends.  ``Selection`` and ``CourtEvents`` are what the
selector and the court-event stages produce for pass 2 and the JSON export.
``RunResult`` is what ``run`` returns to a programmatic caller.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from ball_in_play_selector import FrameResult, Track

from .config import Config
from .ball_detectors import BallDetectorBackend
from .court_detector import CourtDetector
from .court_events import BounceEvent, ContactEvent, PlayerBoxes, RallyLeg
from .player_detector import PlayerDetector
from .utils import EncoderChoice

# A display-path point: (x, y, exact), exact when it is a track observation.
PathPoint = Tuple[float, float, bool]


class Timing:
    """Per-stage wall-clock seconds, kept only when ``cfg.info_timing`` is on.

    ``start()`` returns a timestamp (0.0 when disabled) and ``add(key, t0)``
    accumulates the time since it; both are no-ops when disabled so callers
    need no ``if info_timing`` guards.
    """

    def __init__(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self.values: Dict[str, float] = {}

    def start(self) -> float:
        return time.perf_counter() if self.enabled else 0.0

    def add(self, key: str, t0: float) -> float:
        """Accumulate the seconds since ``t0`` under ``key``; returns that delta."""
        if not self.enabled:
            return 0.0
        delta = time.perf_counter() - t0
        self.values[key] = self.values.get(key, 0.0) + delta
        return delta

    def set(self, key: str, seconds: float) -> None:
        if self.enabled:
            self.values[key] = float(seconds)

    def get(self, key: str) -> float:
        return float(self.values.get(key, 0.0))


@dataclass
class VideoInfo:
    path: Path
    fps: float
    width: int
    height: int
    frame_count_hint: int    # the container's frame count; the decoded count may differ


@dataclass
class RunContext:
    cfg: Config
    video: VideoInfo
    capture: Any                     # the cv2.VideoCapture pass 1 decodes from
    device: torch.device
    encoder: EncoderChoice
    detector: BallDetectorBackend
    court_detector: CourtDetector
    player_detector: PlayerDetector
    timing: Timing
    started_at: float = field(default_factory=time.time)
    init_t0: float = 0.0       # Timing.start() taken when the run began; pass 1 closes 'init_total'

    @property
    def needs_pass2_outputs(self) -> bool:
        cfg = self.cfg
        return bool(cfg.save_tracking_video or cfg.save_motion_debug or cfg.save_track_debug_video)

    def elapsed(self) -> float:
        return time.time() - self.started_at


@dataclass
class Selection:
    """The selector's answer for the clip plus what pass 2 derives from it."""

    per_frame: List[Optional[FrameResult]]
    chosen_track: Optional[Track]          # the stitched timeline chain
    all_tracks: List[Track]                # every scored track, for diagnostics
    court_polygon: Optional[np.ndarray]    # cv2 contour of the court, or None
    last_court_keypoints: Optional[List[float]]
    display_path: Dict[int, PathPoint]     # chosen track's path for the track-debug video

    @property
    def filled_frames(self) -> int:
        return sum(1 for r in self.per_frame if r is not None)


@dataclass
class CourtEvents:
    """Bounces, player contacts and rally legs for the court minimap (empty when it is off)."""

    bounces: List[BounceEvent] = field(default_factory=list)
    contacts: List[ContactEvent] = field(default_factory=list)
    rally_legs: List[RallyLeg] = field(default_factory=list)


@dataclass
class RunResult:
    """What ``pipeline.run`` returns: the selection, the per-frame court and
    player evidence, where the outputs went and how long it took, so a caller
    that drives the tracker from Python need not re-read the tracking JSON.
    """

    per_frame: List[Optional[FrameResult]]
    chosen_track: Optional[Track]                    # the stitched timeline chain
    all_tracks: List[Track]
    frame_count: int                                 # decoded source frames
    court_keypoints: List[Optional[List[float]]]     # per frame, as pass 1 saw them
    player_boxes: List[PlayerBoxes]                  # per frame, tracked player id -> box
    last_valid_court_keypoints: Optional[List[float]]
    elapsed_sec: float
    timing: Dict[str, float]                         # per-stage seconds, filled when cfg.info_timing is on
    pass2_frames_rendered: int
    outputs: Dict[str, Path]                         # files written, by role: tracking_json, tracking_video, ...

    @property
    def filled_frames(self) -> int:
        return sum(1 for r in self.per_frame if r is not None)

    def frame_rows(self) -> List[Dict[str, Any]]:
        """The per-frame rows as the tracking JSON's ``frames`` table holds them."""
        from .export import frame_result_to_json, json_safe   # export imports this module

        rows = []
        for i in range(self.frame_count):
            row = frame_result_to_json(i, self.per_frame[i] if i < len(self.per_frame) else None)
            if i < len(self.court_keypoints):
                row["court_keypoints"] = json_safe(self.court_keypoints[i])
            if i < len(self.player_boxes):
                row["player_boxes"] = json_safe(self.player_boxes[i])
            rows.append(row)
        return rows
