from typing import Optional, List, Sequence, Tuple, Dict, Literal
from dataclasses import dataclass, field

from .config import SelectorConfig
from .physics import BallKalmanFilter

# One ball candidate as the detector hands it over: an xyxy box in source
# pixels and its confidence.  ``association.build_detections`` turns the
# per-frame lists of these into ``Detection`` rows.
RawDetection = Tuple[Sequence[float], float]


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


# Every key scoring.score_tracks writes into Track.score_breakdown, with the
# value a reader gets when the track has not been scored (test fixtures build
# tracks by hand).  Read through Track.score_value so a typo is a KeyError
# instead of a silent default.
SCORE_BREAKDOWN_DEFAULTS: Dict[str, float] = {
    "total": 0.0,                   # final track score
    "num_obs": 0.0,
    "span": 0.0,                    # frames from first to last observation
    "inside_frac": 0.0,             # near-court: inside polygon or within court_expand_px
    "inside_strict_frac": 0.0,      # strictly inside the court polygon
    "motion_overlap": 0.0,          # observations on the boost/motion mask
    "motion_kin_frac": 0.0,         # physics.kinematic_motion_frac estimate
    "motion_evidence": 0.0,         # max(motion_overlap, motion_kin_frac)
    "avg_speed_pxpf": 0.0,
    "peak_speed_pxpf": 0.0,
    "extent_px": 0.0,               # diagonal of the box around all observations
    "jump_penalty": 0.0,
    "near_player_frac": 0.0,
    "near_player_court_frac": 0.0,
    "coverage_frac": 0.0,           # observations / total frames
    "span_frac": 0.0,               # span / total frames
    "start_frame": 0.0,
    "end_frame": 0.0,
    "start_frac": 0.0,              # first observation as a fraction of the clip
    "end_frac": 0.0,                # last observation as a fraction of the clip
    "obs_add": 0.0,                 # w_len * num_obs
    "span_add": 0.0,                # w_span * span_frac
    "additive_core": 0.0,           # obs_add + span_add
    "motion_mult": 0.0,
    "inside_mult": 0.0,
    "near_player_mult": 0.0,
    "movement_quality": 0.0,        # scoring._motion_consistency
    "timeline_tracks": 0.0,         # stitched chains only: tracks joined
}


@dataclass
class Track:
    track_id: int
    cfg: SelectorConfig
    observations: List[Detection] = field(default_factory=list)
    # Prediction state
    last_pos: Tuple[float, float] = (0.0, 0.0)
    last_vel: Tuple[float, float] = (0.0, 0.0)
    misses: int = 0
    alive: bool = True
    # Created on the first observation; None only while the track is empty.
    kf: Optional[BallKalmanFilter] = None
    # Scoring
    score: float = 0.0
    score_breakdown: Dict[str, float] = field(default_factory=dict)
    # Cached kinematics for repeated scoring passes
    # (num_obs, first frame, last frame, last cx, last cy); see scoring._track_kinematics
    _kin_cache_key: Optional[Tuple[int, int, int, float, float]] = field(default=None, init=False, repr=False)
    _cached_velocities: List[Tuple[float, float]] = field(default_factory=list, init=False, repr=False)
    _cached_speeds: List[float] = field(default_factory=list, init=False, repr=False)
    _cached_speed_stats: Tuple[float, float] = field(default=(0.0, 0.0), init=False, repr=False)

    @property
    def num_obs(self) -> int:
        return len(self.observations)

    @property
    def span(self) -> int:
        if len(self.observations) < 2:
            return 0
        return self.observations[-1].frame - self.observations[0].frame + 1

    @property
    def first_frame(self) -> int:
        return self.observations[0].frame if self.observations else 0

    @property
    def last_obs_frame(self) -> int:
        return self.observations[-1].frame if self.observations else 0

    def score_value(self, key: str) -> float:
        """A score_breakdown entry, or its documented default when unscored."""
        return float(self.score_breakdown.get(key, SCORE_BREAKDOWN_DEFAULTS[key]))

    def predict(self, t: int) -> Tuple[float, float]:
        """Predict to frame t using Kalman Filter prediction."""
        dt = t - self.last_obs_frame
        if dt <= 0:
            return self.last_pos

        if self.kf is not None:
            return self.kf.predict_dt(dt)
        return (self.last_pos[0], self.last_pos[1])

    def update(self, det: Detection) -> None:
        """Add observation, update KF."""
        if not self.observations:
            # First observation
            self.kf = BallKalmanFilter(det.cx, det.cy, self.cfg)
            self.last_vel = (0.0, 0.0)
        else:
            if self.kf is None:
                # update() creates the filter with the first observation; a track whose
                # observations were assigned directly (a stitched timeline chain, a test
                # fixture) has none to step, and is not meant to grow.
                raise ValueError(
                    f"track {self.track_id} has {len(self.observations)} observations but no Kalman "
                    "filter: only a track built through update() can be updated"
                )
            dt = det.frame - self.last_obs_frame
            if dt > 0:
                # Bootstrap velocity from the first two observations so the filter
                # doesn't spend the early frames predicting with vx=vy=0.
                if len(self.observations) == 1:
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
                self.kf.update(det.cx, det.cy, conf=det.conf)
                self.last_vel = self.kf.get_velocity()

        self.last_pos = (det.cx, det.cy)
        self.observations.append(det)
        self.misses = 0
        self._kin_cache_key = None


# What decided a frame's position:
#   'det'    = a detector observation on the selected track
#   'motion' = a motion blob near the predicted path (bounded correction)
#   'interp' = a fitted position: bounded fill between anchors, or a detector
#              point the trajectory refit relocated too far to still call raw
Source = Literal["det", "motion", "interp"]

# Where a frame's position stands relative to the detector's measurement:
#   'measurement' = the detector's own point
#   'filtered'    = a detected frame whose point the refit or the filter moved
#   'recovered'   = no detection this frame: a motion blob or a fitted fill
PositionKind = Literal["measurement", "filtered", "recovered"]


@dataclass
class FrameResult:
    """One frame's selected ball position; a frame without one is ``None``, not a row."""

    cx: float
    cy: float
    conf: float = 0.0
    interpolated: bool = False
    bbox: Optional[Tuple[float, float, float, float]] = None
    source: Source = "det"
    # Search region the selector used to look for a motion blob this frame
    # (centre = predicted point, radius 0 when no search happened).
    search_cx: float = 0.0
    search_cy: float = 0.0
    search_radius: float = 0.0
    # Short machine-readable explanation of `source` for the JSON export.
    source_reason: str = ""
    # Immutable detector location, separate from the selected/refitted coordinates.
    measurement: Optional[Tuple[float, float]] = None

    @property
    def position_kind(self) -> PositionKind:
        """Where ``(cx, cy)`` stands relative to the detector's measurement.

        A property, not a stored field: the trajectory refit moves ``cx, cy`` after the
        row is built, and a value recorded then would be stale.
        """
        if self.measurement is None:
            return "recovered"
        return "measurement" if self.measurement == (self.cx, self.cy) else "filtered"
