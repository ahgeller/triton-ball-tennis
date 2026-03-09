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


def _cfg_diag(cfg: SelectorConfig) -> float:
    if cfg.diag > 0.0:
        return cfg.diag
    return math.sqrt(cfg.width ** 2 + cfg.height ** 2)

def _fps_norm_pxpf(base_pxpf_30fps: float, cfg: SelectorConfig) -> float:
    """Convert a 30fps px/frame threshold to the current fps domain."""
    fps_scale = 30.0 / max(cfg.fps, 1.0)
    fps_scale = max(0.35, min(3.0, fps_scale))
    return base_pxpf_30fps * fps_scale

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

def _mask_has_motion_near(
    mask: Optional[np.ndarray],
    x: float,
    y: float,
    radius_px: int = 2
) -> bool:
    """True if mask has any active pixel near (x, y)."""
    if mask is None:
        return False
    h, w = mask.shape[:2]
    cx = int(round(float(x)))
    cy = int(round(float(y)))
    r = max(int(radius_px), 0)
    x1 = max(0, cx - r)
    y1 = max(0, cy - r)
    x2 = min(w - 1, cx + r)
    y2 = min(h - 1, cy + r)
    if x2 < x1 or y2 < y1:
        return False
    return bool(np.any(mask[y1:y2 + 1, x1:x2 + 1] > 0))

def build_court_homography(
    court_keypoints,
    court_w_m: float = 10.97,
    court_h_m: float = 23.77,
) -> Optional[Tuple[np.ndarray, np.ndarray, float, float]]:
    """
    Build a perspective homography from 4 court corners (pixels) to a
    normalized [0,1]^2 court canvas.  Returns (H, H_inv, court_w_m, court_h_m)
    or None when keypoints are unavailable.
    """
    if court_keypoints is None or len(court_keypoints) < 16:
        return None
    def _kp(kps, idx):
        i = idx * 2
        if i + 1 < len(kps):
            x, y = float(kps[i]), float(kps[i + 1])
            if x > 0 or y > 0:
                return np.float32([x, y])
        return None
    p0 = _kp(court_keypoints, 0)  # TL
    p3 = _kp(court_keypoints, 3)  # TR
    p4 = _kp(court_keypoints, 4)  # BL
    p7 = _kp(court_keypoints, 7)  # BR
    if any(p is None for p in (p0, p3, p4, p7)):
        return None
    src = np.float32([p0, p3, p4, p7])             # TL TR BL BR in pixels
    dst = np.float32([[0, 0], [1, 0], [0, 1], [1, 1]])  # normalised court
    H = cv2.getPerspectiveTransform(src, dst)
    H_inv = np.linalg.inv(H)
    return H, H_inv, float(court_w_m), float(court_h_m)

def court_px_per_meter(
    ball_cx: float,
    ball_cy: float,
    H: np.ndarray,
    H_inv: np.ndarray,
    court_w_m: float = 10.97,
) -> Optional[float]:
    """
    Estimate the pixel scale (px per real meter) at a given ball position
    by displacing 1 m sideways in court space and measuring the pixel shift.
    Returns None when the projection is unstable.
    """
    try:
        pt = np.float32([[[ball_cx, ball_cy]]])
        court_pt = cv2.perspectiveTransform(pt, H)[0, 0]          # (cx_n, cy_n)
        dx_norm = 1.0 / court_w_m                                  # 1 real m in normalised units
        shifted = np.float32([[[court_pt[0] + dx_norm, court_pt[1]]]])
        px_shifted = cv2.perspectiveTransform(shifted, H_inv)[0, 0]
        scale = float(np.linalg.norm(px_shifted - np.array([ball_cx, ball_cy])))
        return scale if np.isfinite(scale) and scale > 0.5 else None
    except Exception:
        return None

