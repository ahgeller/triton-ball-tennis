"""Frame-rate rules shared by the tracking runtime and the fine-tuning tools.

GridTrackNet was trained on 30 FPS footage: its five input frames are 1/30 s
apart.  A 60 FPS clip is two interleaved 30 FPS streams (the even and the odd
frames), so it is read with a source stride of 2 and each phase is a valid
model input on its own.  Other frame rates have no supported stride.
"""
from __future__ import annotations

from typing import Optional

THIRTY_FPS_RANGE = (22.0, 32.0)
SIXTY_FPS_RANGE = (57.0, 62.0)


def phase_stride(fps: float) -> Optional[int]:
    """Source frames per model frame: 1 at about 30 FPS, 2 at about 60 FPS, None otherwise."""
    fps = float(fps)
    if SIXTY_FPS_RANGE[0] <= fps <= SIXTY_FPS_RANGE[1]:
        return 2
    if THIRTY_FPS_RANGE[0] <= fps <= THIRTY_FPS_RANGE[1]:
        return 1
    return None


def require_phase_stride(fps: float) -> int:
    """``phase_stride`` for a frame rate the model supports; raises ValueError otherwise."""
    stride = phase_stride(fps)
    if stride is None:
        raise ValueError(f"GridTrackNet expects a 30 or 60 FPS video, got {float(fps):.3f} FPS")
    return stride
