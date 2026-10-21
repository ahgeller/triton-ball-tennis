from typing import Dict, Hashable, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np

from .config import REFERENCE_FPS, fps_scale


def pxpf_at_fps(base_pxpf_30fps: float, fps: float, *, clamp: bool) -> float:
    """A 30 FPS px/frame speed converted to ``fps``.

    With ``clamp`` the frame-rate ratio is bounded by ``config.fps_scale``
    (0.35 to 3.0), which the association and scoring thresholds want so an odd
    clip cannot scale them away; without it the ratio is exact, which the
    fill-speed floors in ``core`` want.
    """
    if clamp:
        return base_pxpf_30fps * fps_scale(fps)
    return base_pxpf_30fps * REFERENCE_FPS / max(float(fps), 1.0)

def clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


class PackedMask(NamedTuple):
    """A binary frame mask stored as the packed bits of one rectangle.

    The tracker keeps one raw-motion and one boost mask per frame for the
    whole clip; packing the ``(x1, y1, x2, y2)`` rectangle that holds the
    set pixels (or the ROI the motion was computed in) keeps that affordable.
    ``unpack`` restores the full ``height`` x ``width`` uint8 {0, 255} mask.
    A mask with no set pixels is never packed: ``pack`` returns None for it.
    """

    bits: np.ndarray
    height: int
    width: int
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def roi(self) -> Tuple[int, int, int, int]:
        """The packed rectangle as ``(x1, y1, x2, y2)``, exclusive on the far side."""
        return self.x1, self.y1, self.x2, self.y2

    @classmethod
    def pack(cls, mask_u8: Optional[np.ndarray],
             roi: Optional[Tuple[int, int, int, int]] = None) -> Optional["PackedMask"]:
        """Pack ``mask_u8`` (uint8 or bool, 2D); None when it is None or empty.

        ``roi`` is the rectangle to store; it is clipped to the frame.  When it
        is None or clips to nothing, the bounding box of the set pixels is used.
        """
        if mask_u8 is None:
            return None
        height, width = mask_u8.shape[:2]
        set_pixels = np.flatnonzero(mask_u8)
        if set_pixels.size == 0:
            return None
        rect = None
        if roi is not None and len(roi) == 4:
            x1, y1, x2, y2 = (int(v) for v in roi)
            x1, x2 = max(0, min(x1, width)), max(0, min(x2, width))
            y1, y2 = max(0, min(y1, height)), max(0, min(y2, height))
            if x2 > x1 and y2 > y1:
                rect = (x1, y1, x2, y2)
        if rect is None:
            rows, cols = np.divmod(set_pixels, width)
            rect = (int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1)
        x1, y1, x2, y2 = rect
        flat = (mask_u8[y1:y2, x1:x2].reshape(-1) > 0).astype(np.uint8)
        return cls(np.packbits(flat), int(height), int(width), x1, y1, x2, y2)

    def unpack(self) -> np.ndarray:
        """The full-frame uint8 {0, 255} mask."""
        out = np.zeros((self.height, self.width), dtype=np.uint8)
        rh, rw = self.y2 - self.y1, self.x2 - self.x1
        if rh > 0 and rw > 0:
            flat = np.unpackbits(self.bits, count=rh * rw)
            out[self.y1:self.y2, self.x1:self.x2] = flat.reshape(rh, rw).astype(np.uint8) * 255
        return out


# A per-frame motion mask as the selector accepts it.
MaskLike = Union[PackedMask, np.ndarray]
# One frame's player boxes (xyxy): tracked player id -> box, or a plain list.
FramePlayerBoxes = Union[Dict[Hashable, Optional[Sequence[float]]], Sequence[Optional[Sequence[float]]]]
PlayerBoxesByFrame = Sequence[Optional[FramePlayerBoxes]]


def ensure_mask_u8(mask_obj: Optional[MaskLike]) -> Optional[np.ndarray]:
    """Return uint8 mask {0,255}: None, a ``PackedMask`` unpacked, or the array as given."""
    if mask_obj is None:
        return None
    if isinstance(mask_obj, PackedMask):
        return mask_obj.unpack()
    return mask_obj
