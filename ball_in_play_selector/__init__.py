"""Ball-in-play selector: turns per-frame detector candidates into one trajectory.

`select_ball_in_play` is the entry point.  The other names are the pieces
other packages and the tests reach for: the config and its reference
constants, the data classes, the projectile prior shared with the ROI
tracker, the track building/eligibility/timeline steps, the trajectory refit
and the packed-mask format.  Everything else in the modules is an
implementation detail.
"""

from .config import REFERENCE_DIAG_PX, REFERENCE_FPS, SelectorConfig, fps_scale
from .core import (
    direction_cosine,
    frame_result,
    motion_near,
    refine_trajectory,
    select_ball_in_play,
    selected_tracks,
    trajectory_observations,
)
from .models import SCORE_BREAKDOWN_DEFAULTS, Detection, FrameResult, Source, Track
from .physics import BallKalmanFilter, predict_projectile, predict_projectile_velocity
from .scoring import select_timeline_chain
from .tracking import build_tracks
from .utils import PackedMask, ensure_mask_u8

__all__ = [
    "select_ball_in_play",
    "SelectorConfig",
    "REFERENCE_DIAG_PX",
    "REFERENCE_FPS",
    "fps_scale",
    "Detection",
    "FrameResult",
    "Source",
    "Track",
    "SCORE_BREAKDOWN_DEFAULTS",
    "BallKalmanFilter",
    "predict_projectile",
    "predict_projectile_velocity",
    "build_tracks",
    "select_timeline_chain",
    "selected_tracks",
    "trajectory_observations",
    "direction_cosine",
    "motion_near",
    "frame_result",
    "refine_trajectory",
    "ensure_mask_u8",
    "PackedMask",
]
