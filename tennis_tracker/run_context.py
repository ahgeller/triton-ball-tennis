"""The data the stages of ``pipeline.run`` hand to each other.

``RunContext`` is what ``_open_inputs`` builds once per run: the validated
config, the input clip's metadata, the CUDA device, the encoder choice and
the three detector backends.  ``Selection`` and ``CourtEvents`` are what the
selector and the court-event stages produce for pass 2 and the JSON export.
What ``run`` returns is ``export.RunResult``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ball_in_play_selector import FrameResult, Track

from .config import Config
from .engines.ball_detectors import BallDetectorBackend
from .engines.court_detector import CourtDetector
from .court.court_scene import BounceEvent, ContactEvent, RallyLeg
from .engines.player_detector import PlayerDetector
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
    capture: cv2.VideoCapture        # what pass 1 decodes from
    device: torch.device
    encoder: Optional[EncoderChoice]    # None when the run writes no video: no writer is opened
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
    """What the court-event stages found, for the minimap and the tracking JSON: the confirmed
    bounces, the model's low-score candidates, the kinematic bounces, the player contacts and
    the rally legs. ``computed`` is False, and every list empty, when the run did not ask for
    them (``Config.court_events`` or ``draw_court_minimap``)."""

    computed: bool = False
    bounces: List[BounceEvent] = field(default_factory=list)
    bounce_candidates: List[BounceEvent] = field(default_factory=list)
    kinematic_bounces: List[BounceEvent] = field(default_factory=list)
    contacts: List[ContactEvent] = field(default_factory=list)
    rally_legs: List[RallyLeg] = field(default_factory=list)
