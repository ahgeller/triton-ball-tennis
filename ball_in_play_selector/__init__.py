"""Ball-in-play selector: turns per-frame detector candidates into one trajectory.

`select_ball_in_play` is the entry point.  The other names are the pieces
other packages and the self-tests legitimately reach for: the projectile
prior shared with the ROI tracker, the track eligibility/timeline steps, the
trajectory refit and the data classes.  Underscore-prefixed helpers inside the
modules are implementation details.
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
from .physics import (
    BallKalmanFilter,
    kinematic_motion_frac,
    predict_projectile,
    predict_projectile_velocity,
    xy_dist,
)
from .scoring import is_near_player, score_tracks, select_timeline_chain, stitch_track_chain
from .tracking import build_detections, build_tracks
from .utils import PackedMask, cfg_diag, clamp01, ensure_mask_u8, fps_norm_pxpf

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
    "kinematic_motion_frac",
    "xy_dist",
    "build_detections",
    "build_tracks",
    "score_tracks",
    "is_near_player",
    "select_timeline_chain",
    "stitch_track_chain",
    "selected_tracks",
    "trajectory_observations",
    "direction_cosine",
    "motion_near",
    "frame_result",
    "refine_trajectory",
    "cfg_diag",
    "clamp01",
    "ensure_mask_u8",
    "PackedMask",
    "fps_norm_pxpf",
]
