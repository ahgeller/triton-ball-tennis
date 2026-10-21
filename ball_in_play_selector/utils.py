import math
from typing import NamedTuple, Optional, Tuple, Union

import numpy as np

from .config import SelectorConfig, fps_scale


def cfg_diag(cfg: SelectorConfig) -> float:
    if cfg.diag > 0.0:
        return cfg.diag
    return math.sqrt(cfg.width ** 2 + cfg.height ** 2)

def fps_norm_pxpf(base_pxpf_30fps: float, cfg: SelectorConfig) -> float:
    """Convert a 30fps px/frame threshold to the current fps domain."""
    return base_pxpf_30fps * fps_scale(cfg.fps)

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


def ensure_mask_u8(mask_obj: Optional[MaskLike]) -> Optional[np.ndarray]:
    """Return uint8 mask {0,255}: None, a ``PackedMask`` unpacked, or the array as given."""
    if mask_obj is None:
        return None
    if isinstance(mask_obj, PackedMask):
        return mask_obj.unpack()
    return mask_obj
