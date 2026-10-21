import math
import numpy as np

from .config import SelectorConfig, fps_scale


def _cfg_diag(cfg: SelectorConfig) -> float:
    if cfg.diag > 0.0:
        return cfg.diag
    return math.sqrt(cfg.width ** 2 + cfg.height ** 2)

def _fps_norm_pxpf(base_pxpf_30fps: float, cfg: SelectorConfig) -> float:
    """Convert a 30fps px/frame threshold to the current fps domain."""
    return base_pxpf_30fps * fps_scale(cfg.fps)

def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))

def _ensure_mask_u8(mask_obj):
    """Return uint8 mask {0,255}; supports packed (np.packbits) tuples."""
    if mask_obj is None:
        return None
    if isinstance(mask_obj, tuple) and len(mask_obj) == 8 and mask_obj[0] == "roi":
        _, packed, h, w, x1, y1, x2, y2 = mask_obj
        h = int(h)
        w = int(w)
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        out = np.zeros((h, w), dtype=np.uint8)
        rh = max(0, y2 - y1)
        rw = max(0, x2 - x1)
        if rh > 0 and rw > 0:
            flat = np.unpackbits(packed, count=rh * rw)
            out[y1:y2, x1:x2] = flat.reshape(rh, rw).astype(np.uint8) * 255
        return out
    if isinstance(mask_obj, tuple) and len(mask_obj) == 3:
        packed, h, w = mask_obj
        flat = np.unpackbits(packed, count=int(h) * int(w))
        return (flat.reshape(int(h), int(w)).astype(np.uint8) * 255)
    return mask_obj
