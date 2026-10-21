"""Ball-in-play selector: turns per-frame detector candidates into one trajectory.

`select_ball_in_play` is the entry point.  The other names are what the
tracking runtime reaches for: the config and its reference diagonal, the
candidate and result types, the projectile prior shared with the ROI tracker
and the packed-mask format.  The internal steps (association, scoring, the
timeline chain, the refit) are imported from their modules.
"""

from .config import REFERENCE_DIAG_PX, SelectorConfig
from .core import select_ball_in_play
from .models import Detection, FrameResult, PositionKind, RawDetection, Source, Track
from .physics import predict_projectile
from .utils import PackedMask

__all__ = [
    "select_ball_in_play",
    "SelectorConfig",
    "REFERENCE_DIAG_PX",
    "RawDetection",
    "Detection",
    "FrameResult",
    "PositionKind",
    "Source",
    "Track",
    "predict_projectile",
    "PackedMask",
]
