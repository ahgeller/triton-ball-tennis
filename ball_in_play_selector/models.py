# Imports
import math
import json
import os
import numpy as np
import cv2
from types import SimpleNamespace
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass, field
try:
    from filterpy.kalman import KalmanFilter
except ImportError:
    KalmanFilter = None
    from filterpy.kalman import KalmanFilter

from .config import SelectorConfig
from .physics import BallKalmanFilter


@dataclass
class Detection:
    frame: int
    cx: float
    cy: float
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    area: float
    on_motion: bool = False  # overlaps boost/motion mask

@dataclass
class MotionTrack:
    """A continuous trajectory built purely from motion blobs over time."""
    track_id: int
    points: List[Tuple[int, float, float]] = field(default_factory=list)  # (frame, cx, cy)
    smoothed: List[Tuple[float, float]] = field(default_factory=list) # EMA smoothed (cx, cy)
    
    @property
    def start_frame(self) -> int:
        return self.points[0][0] if self.points else -1
        
    @property
    def end_frame(self) -> int:
        return self.points[-1][0] if self.points else -1

    def get_position_at(self, frame_idx: int) -> Optional[Tuple[float, float]]:
        """Return the smoothed position at exactly `frame_idx`, or None if no data."""
        for i, (f, _, _) in enumerate(self.points):
            if f == frame_idx and i < len(self.smoothed):
                return self.smoothed[i]
        return None

@dataclass
class Track:
    track_id: int
    cfg: SelectorConfig
    observations: List[Detection] = field(default_factory=list)
    # Prediction state
    last_pos: Tuple[float, float] = (0.0, 0.0)
    last_vel: Tuple[float, float] = (0.0, 0.0)
    last_frame: int = -1
    misses: int = 0
    alive: bool = True
    # FilterPy Kalman Filter
    kf: Optional['BallKalmanFilter'] = None
    # Scoring
    score: float = 0.0
    score_breakdown: Dict[str, float] = field(default_factory=dict)
    # Area Tracking for stable ROI search
    historical_areas: List[float] = field(default_factory=list)
    # Cached kinematics for repeated scoring passes
    _kin_cache_key: Optional[Tuple[Any, ...]] = field(default=None, init=False, repr=False)
    _cached_velocities: List[Tuple[float, float]] = field(default_factory=list, init=False, repr=False)
    _cached_speeds: List[float] = field(default_factory=list, init=False, repr=False)
    _cached_speed_stats: Tuple[float, float] = field(default=(0.0, 0.0), init=False, repr=False)

    def __post_init__(self):
        # We need cfg explicitly passed to initialize KF properly in update().
        # Due to dataclass ordering, we'll initialize KF on first update. 
        pass

    @property
    def num_obs(self):
        return len(self.observations)

    @property
    def span(self):
        if len(self.observations) < 2:
            return 0
        return self.observations[-1].frame - self.observations[0].frame + 1

    @property
    def first_frame(self):
        return self.observations[0].frame if self.observations else 0

    @property
    def last_obs_frame(self):
        return self.observations[-1].frame if self.observations else 0

    def predict(self, t: int) -> Tuple[float, float]:
        """Predict to frame t using Kalman Filter prediction."""
        dt = t - self.last_frame
        if dt <= 0:
            return self.last_pos
            
        if self.kf is not None:
            return self.kf.predict_dt(dt)
        return (self.last_pos[0], self.last_pos[1])

    def update(self, det: Detection):
        """Add observation, update KF."""
        if not self.observations:
            # First observation
            self.kf = BallKalmanFilter(det.cx, det.cy, self.cfg)
            self.last_vel = (0.0, 0.0)
        else:
            dt = det.frame - self.last_frame
            if dt > 0:
                # Bootstrap velocity from the first two observations so the filter
                # doesn't spend the early frames predicting with vx=vy=0.
                if len(self.observations) == 1 and self.kf is not None:
                    prev = self.observations[-1]
                    vx0 = (float(det.cx) - float(prev.cx)) / float(dt)
                    vy0 = (float(det.cy) - float(prev.cy)) / float(dt)
                    self.kf.kf.x[2, 0] = vx0
                    self.kf.kf.x[3, 0] = vy0
                    # Reduce initial velocity uncertainty a bit after bootstrapping.
                    self.kf.kf.P[2, 2] = min(float(self.kf.kf.P[2, 2]), 60.0)
                    self.kf.kf.P[3, 3] = min(float(self.kf.kf.P[3, 3]), 60.0)

                # Step the KF forward `dt` times before updating
                for _ in range(dt - 1):
                    self.kf.predict()
                
                # Predict to current frame, then update with measurement
                self.kf.predict()
                self.kf.update(det.cx, det.cy, conf=getattr(det, "conf", None))
                self.last_vel = self.kf.get_velocity()
                
            # Update historical area for stable rolling average (keep last 5)
            # Only add to history if it's a solid detection, not a spawned proxy
            if getattr(det, "conf", 0.0) > 0.1:
                self.historical_areas.append(det.area)
                if len(self.historical_areas) > 5:
                    self.historical_areas.pop(0)
                    
        self.last_pos = (det.cx, det.cy)
        self.last_frame = det.frame
        self.observations.append(det)
        self.misses = 0
        self._kin_cache_key = None
        
    def get_recent_average_area(self) -> float:
        """Return the mean area of the last up to 5 actual detections."""
        if not self.historical_areas:
            # Fallback to last known observation if history isn't populated
            return self.observations[-1].area if self.observations else 0.0
        return sum(self.historical_areas) / len(self.historical_areas)

@dataclass
class FrameResult:
    cx: Optional[float] = None
    cy: Optional[float] = None
    conf: float = 0.0
    interpolated: bool = False
    bbox: Optional[Tuple[float, float, float, float]] = None
    # Debug-only placeholder entries may carry guide-circle metadata without
    # representing a selected ball output for that frame.
    debug_only: bool = False
    # Debug: what decided this frame's position
    # 'det' = YOLO detection, 'motion' = motion blob tracking,
    # 'interp' = interpolation between detections,
    # 'carry' = predicted continuation during short loss,
    # 'guide' = chosen-track guide fallback
    source: str = 'det'
    # Debug: search region used for motion tracking
    search_cx: float = 0.0
    search_cy: float = 0.0
    search_radius: float = 0.0
    # Debug: guide-side detection gate/search circle used this frame (if any)
    guide_search_cx: float = 0.0
    guide_search_cy: float = 0.0
    guide_search_radius: float = 0.0
    guide_search_exact: bool = False
    guide_search_frozen: bool = False
    guide_search_hold: bool = False

