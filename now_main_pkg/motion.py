# Imports
import argparse
import copy
import glob
import json
import math
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from collections import OrderedDict, namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import scipy.interpolate
from ball_in_play_selector import select_ball_in_play, FrameResult, _predict_projectile, SelectorConfig
HAS_NMS = False
_nms = None
try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except Exception:
    torch = None
    F = None
    HAS_TORCH = False

try:
    from boxmot import ByteTrack
except ImportError:
    print("[warning] boxmot not found. Player tracking will be disabled. Run 'pip install boxmot'")
    ByteTrack = None

from .config import Config
from .rendering import _build_court_side_guides, draw_court_side_guides


_MOTION_BUFFERS = {}
_KERNEL_CACHE = {}
_GAUSS_KERNEL_CACHE = {}
_KERNEL_3x3 = np.ones((3, 3), np.uint8)

class MotionBuffers:
    """Pre-allocated buffers to avoid per-frame memory allocation."""
    def __init__(self, h: int, w: int):
        self.h = h
        self.w = w
        # Pre-allocate arrays used every frame
        self.raw_motion_u8 = np.zeros((h, w), dtype=np.uint8)
        self.boost_mask_u8 = np.zeros((h, w), dtype=np.uint8)
        self.labels = np.zeros((h, w), dtype=np.int32)
        self.keep_lut = np.zeros(256, dtype=np.uint8)  # Max 255 blobs typically
        
    def reset_raw_motion(self):
        """Reset raw_motion_u8 buffer efficiently."""
        self.raw_motion_u8.fill(0)
        
    def reset_boost_mask(self):
        """Reset boost_mask_u8 buffer efficiently."""
        self.boost_mask_u8.fill(0)

def _get_motion_buffers(h: int, w: int) -> MotionBuffers:
    """Get or create pre-allocated buffers for given resolution."""
    key = (h, w)
    if key not in _MOTION_BUFFERS:
        _MOTION_BUFFERS[key] = MotionBuffers(h, w)
    return _MOTION_BUFFERS[key]

def _get_kernel(size: int) -> np.ndarray:
    k = _KERNEL_CACHE.get(size)
    if k is None:
        k = np.ones((size, size), np.uint8)
        _KERNEL_CACHE[size] = k
    return k

def _xywh_to_xyxy_np(boxes_xywh: np.ndarray) -> np.ndarray:
    out = boxes_xywh.copy()
    out[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] * 0.5
    out[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] * 0.5
    out[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] * 0.5
    out[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] * 0.5
    return out

def _nms_xyxy_np(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_thres: float) -> np.ndarray:
    if boxes_xyxy.size == 0 or scores.size == 0:
        return np.empty((0,), dtype=np.int32)
    x1 = boxes_xyxy[:, 0]
    y1 = boxes_xyxy[:, 1]
    x2 = boxes_xyxy[:, 2]
    y2 = boxes_xyxy[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = np.argsort(scores)[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter + 1e-6
        iou = inter / union
        order = rest[iou <= float(iou_thres)]
    return np.asarray(keep, dtype=np.int32)

def _motion_combined_from_hsv(hsv_prev: np.ndarray, hsv_curr: np.ndarray) -> np.ndarray:
    """Return per-pixel motion strength from HSV channel deltas."""
    v_diff = cv2.absdiff(hsv_curr[:, :, 2], hsv_prev[:, :, 2])
    s_diff = cv2.absdiff(hsv_curr[:, :, 1], hsv_prev[:, :, 1])
    h0 = hsv_prev[:, :, 0].astype(np.int16)
    h1 = hsv_curr[:, :, 0].astype(np.int16)
    h_diff_raw = np.abs(h1 - h0)
    h_diff = np.minimum(h_diff_raw, 180 - h_diff_raw).astype(np.uint8)

    combined = np.maximum(v_diff, np.maximum(
        (s_diff * 1.5).clip(0, 255).astype(np.uint8),
        (h_diff * 1.2).clip(0, 255).astype(np.uint8)))
    return cv2.GaussianBlur(combined, (3, 3), 0)

def compute_motion_sv_from_hsv(
    hsv_prev: np.ndarray, hsv_curr: np.ndarray, thresh: float
) -> np.ndarray:
    combined = _motion_combined_from_hsv(hsv_prev, hsv_curr)
    _, mask = cv2.threshold(combined, int(thresh), 255, cv2.THRESH_BINARY)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, _KERNEL_3x3)

def compute_motion_sv_3frame_hsv(
    hsv_prev,
    hsv_curr,
    hsv_next,
    thresh: float,
    soft_temporal: bool = False,
    temporal_lo_frac: float = 0.55,
    temporal_hi_mult: float = 1.35,
    v_min: float = 60.0,
) -> np.ndarray:
    c1 = _motion_combined_from_hsv(hsv_prev, hsv_curr)
    c2 = _motion_combined_from_hsv(hsv_curr, hsv_next)

    t = int(max(1, round(float(thresh))))
    if soft_temporal:
        lo = int(max(1, round(t * max(0.10, min(1.0, float(temporal_lo_frac))))))
        hi = int(max(t + 1, round(t * max(1.0, float(temporal_hi_mult)))))
        mask = (
            ((c1 >= t) & (c2 >= lo)) |
            ((c2 >= t) & (c1 >= lo)) |
            (c1 >= hi) |
            (c2 >= hi)
        ).astype(np.uint8) * 255
    else:
        m1 = (c1 >= t).astype(np.uint8) * 255
        m2 = (c2 >= t).astype(np.uint8) * 255
        mask = cv2.bitwise_and(m1, m2)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _KERNEL_3x3)
    bright = (hsv_curr[:, :, 2] > float(v_min)).astype(np.uint8) * 255
    return cv2.bitwise_and(mask, bright)

def _perspective_max_area(cy, cx, img_h, img_w, min_area, max_area, cfg):
    """Compute perspective-scaled max blob area at position (cx, cy)."""
    scale = 1.0
    if cfg.court_depth is not None and img_h > 1:
        y_norm = max(0.0, min(1.0, cy / float(img_h - 1)))
        s = max(0.0, min(1.0, cfg.y_scale_strength))
        if cfg.court_depth == "top_far":
            scale *= (1.0 - s) + s * y_norm
        elif cfg.court_depth == "bot_far":
            scale *= (1.0 - s) + s * (1.0 - y_norm)
    if cfg.court_side is not None and img_w > 1:
        x_norm = max(0.0, min(1.0, cx / float(img_w - 1)))
        s = max(0.0, min(1.0, cfg.x_scale_strength))
        if cfg.court_side == "center_near":
            dist = abs(x_norm - 0.5) * 2.0
            scale *= 1.0 - s * (dist ** 2)
        elif cfg.court_side == "left_far":
            scale *= 1.0 - s * ((1.0 - x_norm) ** 2)
        elif cfg.court_side == "right_far":
            scale *= 1.0 - s * (x_norm ** 2)
    return max(min_area, max_area * scale)

def filter_boost_mask(raw_motion, min_area, max_area, cfg, player_bboxes=None, buffers=None):
    """Filter motion mask - OPTIMIZED with pre-allocated buffers and single CC pass.
    
    Performance improvements:
    1. Single connectedComponents call (not 2-3)
    2. Uses pre-allocated buffers when available
    3. Early exit if no motion pixels
    4. Avoids intermediate array copies
    """
    img_h, img_w = raw_motion.shape[:2]
    
    # OPTIMIZATION: Early exit if no motion
    if not np.any(raw_motion):
        if buffers is not None:
            return buffers.boost_mask_u8  # Return pre-allocated empty buffer
        return np.zeros((img_h, img_w), dtype=np.uint8)
    
    has_perspective = cfg.court_depth is not None or cfg.court_side is not None
    preserve_tiny = getattr(cfg, 'blob_preserve_tiny', True)
    tiny_max_area = getattr(cfg, 'blob_tiny_max_area', 120)
    
    img_diag = math.hypot(float(img_w), float(img_h))
    ball_bypass_max_area = max(15, min(150, int(round(0.035 * img_diag))))
    if preserve_tiny:
        ball_bypass_max_area = max(ball_bypass_max_area, tiny_max_area)
    
    # Single CC call - the major bottleneck
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(raw_motion, connectivity=8)
    
    if num <= 1:
        if buffers is not None:
            return buffers.boost_mask_u8
        return np.zeros((img_h, img_w), dtype=np.uint8)
    
    if buffers is not None:
        result = buffers.boost_mask_u8
        result.fill(0)
    else:
        result = np.zeros((img_h, img_w), dtype=np.uint8)
        
    drawn_any = False
    
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
            
        # Determine max area based on position (perspective)
        if has_perspective:
            cy = stats[i, cv2.CC_STAT_TOP] + 0.5 * stats[i, cv2.CC_STAT_HEIGHT]
            cx = stats[i, cv2.CC_STAT_LEFT] + 0.5 * stats[i, cv2.CC_STAT_WIDTH]
            max_area_local = _perspective_max_area(cy, cx, img_h, img_w, float(min_area), float(max_area), cfg)
        else:
            max_area_local = float(max_area)
        
        # Area check
        if area < min_area or area > max_area_local:
            continue
        
        # Small ball blobs bypass aspect check
        survived = False
        if area <= ball_bypass_max_area:
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]
            aspect = max(bw, bh) / max(min(bw, bh), 1)
            aspect_thresh = 4.5 if area < 50 else 3.5
            if aspect <= aspect_thresh:
                survived = True
        else:
            # Larger blobs: aspect check only
            survived = True
            if cfg.blob_shape_filter:
                bw = stats[i, cv2.CC_STAT_WIDTH]
                bh = stats[i, cv2.CC_STAT_HEIGHT]
                aspect = max(bw, bh) / max(min(bw, bh), 1)
                if aspect > cfg.blob_max_aspect:
                    survived = False
                    
        if survived:
            # Draw a perfectly solid circle representing this blob's mass and position
            cx = int(centroids[i][0] + 0.5)
            cy = int(centroids[i][1] + 0.5)
            radius = int(math.sqrt(area / math.pi) + 0.5)
            cv2.circle(result, (cx, cy), max(radius, 1), 255, -1)
            drawn_any = True
            
    if not drawn_any:
        return np.zeros((img_h, img_w), dtype=np.uint8) if buffers is None else result
    
    return result

def suppress_flicker_components(
    mask_u8: Optional[np.ndarray],
    prev_mask_u8: Optional[np.ndarray],
    keep_mask_u8: Optional[np.ndarray],
    cfg: Config
) -> Optional[np.ndarray]:
    """Drop one-frame flicker blobs unless supported by history or predicted region."""
    if mask_u8 is None:
        return None
    if not cfg.motion_flicker_suppress:
        return mask_u8
    if mask_u8.max() == 0:
        return mask_u8

    h, w = mask_u8.shape[:2]
    prev_support = None
    if prev_mask_u8 is not None and prev_mask_u8.shape[:2] == (h, w) and prev_mask_u8.max() > 0:
        k = max(int(cfg.motion_flicker_prev_dilate), 1)
        prev_support = cv2.dilate(prev_mask_u8, _get_kernel(k), iterations=1) > 0

    keep_support = None
    if keep_mask_u8 is not None and keep_mask_u8.shape[:2] == (h, w):
        keep_support = keep_mask_u8 > 0

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num <= 1:
        return mask_u8

    out = mask_u8.copy()
    min_area = max(int(cfg.motion_flicker_min_area), 1)
    max_area = max(int(cfg.motion_flicker_max_area), min_area)
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            out[labels == i] = 0
            continue
        if area > max_area:
            continue
        comp = labels == i
        has_prev = bool(prev_support is not None and np.any(prev_support[comp]))
        has_keep = bool(keep_support is not None and np.any(keep_support[comp]))
        if not has_prev and not has_keep:
            out[comp] = 0
    return out

def build_player_protect_mask(h, w, player_bboxes, pad=0):
    """Mask of player-box regions that should remain color-neutral."""
    if not player_bboxes:
        return None
    mask = np.zeros((h, w), dtype=bool)
    for pb in player_bboxes:
        if pb is None or len(pb) < 4:
            continue
        x1, y1, x2, y2 = max(0, int(pb[0])-pad), max(0, int(pb[1])-pad), \
                          min(w-1, int(pb[2])+pad), min(h-1, int(pb[3])+pad)
        if x2 >= x1 and y2 >= y1:
            mask[y1:y2+1, x1:x2+1] = True
    return mask if np.any(mask) else None

def _court_kpt_xy(court_keypoints, idx):
    if court_keypoints is None:
        return None
    bi = idx * 2
    if len(court_keypoints) <= bi + 1:
        return None
    x = float(court_keypoints[bi])
    y = float(court_keypoints[bi + 1])
    if not np.isfinite(x) or not np.isfinite(y):
        return None
    if x <= 0.0 and y <= 0.0:
        return None
    return np.array([x, y], dtype=np.float32)

def _line_side_value(a: np.ndarray, v: np.ndarray, p: np.ndarray) -> float:
    d = p - a
    return float(v[0] * d[1] - v[1] * d[0])

def _clip_polygon_to_halfplane(poly, a, v, keep_sign):
    if not poly:
        return []

    def inside(pt):
        return _line_side_value(a, v, pt) * keep_sign >= -1e-6

    def intersect(p1, p2):
        d = p2 - p1
        denom = v[0] * d[1] - v[1] * d[0]
        if abs(float(denom)) < 1e-6:
            return p1
        t = ((a[0] - p1[0]) * v[1] - (a[1] - p1[1]) * v[0]) / denom
        t = float(max(0.0, min(1.0, t)))
        return p1 + t * d

    output = []
    prev = poly[-1]
    prev_in = inside(prev)
    for curr in poly:
        curr_in = inside(curr)
        if prev_in and curr_in:
            output.append(curr)
        elif prev_in and not curr_in:
            output.append(intersect(prev, curr))
        elif (not prev_in) and curr_in:
            output.append(intersect(prev, curr))
            output.append(curr)
        prev, prev_in = curr, curr_in
    return output

def _halfplane_mask_from_line(h, w, a, v, keep_sign):
    if float(np.hypot(v[0], v[1])) < 1e-6 or abs(float(keep_sign)) < 1e-6:
        return None
    rect = [
        np.array([0.0, 0.0], dtype=np.float32),
        np.array([float(w - 1), 0.0], dtype=np.float32),
        np.array([float(w - 1), float(h - 1)], dtype=np.float32),
        np.array([0.0, float(h - 1)], dtype=np.float32),
    ]
    clipped = _clip_polygon_to_halfplane(rect, a, v, keep_sign)
    if len(clipped) < 3:
        return None
    pts = np.round(np.asarray(clipped, dtype=np.float32)).astype(np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts.reshape(-1, 1, 2)], 255)
    return mask > 0

def _x_relation_mask_from_line(h, w, anchor, direction, relation):
    """Build mask using explicit left/right rule against line x(y).

    relation:
      "left"  -> keep pixels with x < line_x(y)
      "right" -> keep pixels with x > line_x(y)
    """
    dy = float(direction[1])
    dx = float(direction[0])
    if abs(dy) < 1e-6:
        return None

    ys = np.arange(h, dtype=np.float32)
    x_line = float(anchor[0]) + (ys - float(anchor[1])) * (dx / dy)
    xs = np.arange(w, dtype=np.float32)[None, :]

    if relation == "left":
        return xs < x_line[:, None]
    if relation == "right":
        return xs > x_line[:, None]
    return None

def build_court_side_protect_mask(h, w, court_keypoints):
    """Protect outside side regions using court-guided side lines.

    Left side:
      offset from kp4 to image-left by |kp5-kp4|, line parallel to kp4->kp0.
      Pixels left of this line are protected.
    Right side:
      offset from kp7 to image-right by |kp6-kp7|, line parallel to kp7->kp3.
      Pixels right of this line are protected.
    """
    guides = _build_court_side_guides(court_keypoints)
    if not guides:
        return None

    protect = None
    left_edge_ref = np.array([0.0, float(h) * 0.5], dtype=np.float32)
    right_edge_ref = np.array([float(w - 1), float(h) * 0.5], dtype=np.float32)

    for side in ("left", "right"):
        g = guides.get(side)
        if g is None:
            continue

        # Primary rule: explicit "left of left line / right of right line".
        relation = "left" if side == "left" else "right"
        m = _x_relation_mask_from_line(h, w, g["anchor"], g["dir"], relation)
        if m is not None:
            protect = m if protect is None else (protect | m)
            continue

        # Fallback for near-horizontal lines: keep the outside half-plane.
        inside_sign = _line_side_value(g["anchor"], g["dir"], g["base"])
        if abs(float(inside_sign)) < 1e-6:
            continue

        edge_ref = left_edge_ref if side == "left" else right_edge_ref
        edge_sign = _line_side_value(g["anchor"], g["dir"], edge_ref)

        # If midpoint lies on the line, sample corners of that edge.
        if abs(float(edge_sign)) < 1e-6:
            fallback_refs = (
                [np.array([0.0, 0.0], dtype=np.float32),
                 np.array([0.0, float(h - 1)], dtype=np.float32)]
                if side == "left" else
                [np.array([float(w - 1), 0.0], dtype=np.float32),
                 np.array([float(w - 1), float(h - 1)], dtype=np.float32)]
            )
            for ref in fallback_refs:
                edge_sign = _line_side_value(g["anchor"], g["dir"], ref)
                if abs(float(edge_sign)) >= 1e-6:
                    break

        # Enforce "outside" side: must be opposite the court-side base point.
        # If edge sample is ambiguous/wrong, fall back to the opposite of inside_sign.
        if abs(float(edge_sign)) < 1e-6 or edge_sign * inside_sign > 0:
            keep_sign = -inside_sign
        else:
            keep_sign = edge_sign

        m = _halfplane_mask_from_line(h, w, g["anchor"], g["dir"], keep_sign)
        if m is not None:
            protect = m if protect is None else (protect | m)

    return protect if protect is not None and np.any(protect) else None

def build_protect_mask(h, w, player_bboxes=None, court_keypoints=None, player_pad=0):
    # Player-area masking removed: keep only court-side protection.
    # This prevents motion/boost suppression around players during hits.
    return build_court_side_protect_mask(h, w, court_keypoints)

def apply_exclude_mask_u8(mask_u8, exclude_mask):
    """Zero out a uint8 mask where exclude_mask is True."""
    if mask_u8 is None or exclude_mask is None:
        return mask_u8
    if not np.any(exclude_mask):
        return mask_u8
    out = mask_u8.copy()
    out[exclude_mask] = 0
    return out

def _pack_mask_u8(mask_u8, roi=None):
    """Pack a uint8/bool 2D mask to bits for compact storage.

    ROI mode packs only a slice for reduced pass1 storage cost:
    - (packed, h, w) for full-frame packs
    - ("roi", packed, h, w, x1, y1, x2, y2) for ROI-sparse packs
    """
    if mask_u8 is None:
        return None
    h, w = mask_u8.shape[:2]

    if roi is not None and len(roi) == 4:
        x1, y1, x2, y2 = [int(v) for v in roi]
        x1 = max(0, min(x1, w))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h))
        y2 = max(0, min(y2, h))
        if x2 > x1 and y2 > y1:
            roi_flat = (mask_u8[y1:y2, x1:x2].reshape(-1) > 0).astype(np.uint8)
            roi_packed = np.packbits(roi_flat)
            return ("roi", roi_packed, int(h), int(w), x1, y1, x2, y2)

    flat = (mask_u8.reshape(-1) > 0).astype(np.uint8)
    packed = np.packbits(flat)
    return packed, int(h), int(w)

def _unpack_mask_u8(mask_obj):
    """Unpack a mask packed by _pack_mask_u8 back to uint8 {0,255}."""
    if mask_obj is None:
        return None
    if isinstance(mask_obj, tuple) and len(mask_obj) == 8 and mask_obj[0] == "roi":
        _, packed, h, w, x1, y1, x2, y2 = mask_obj
        h = int(h); w = int(w)
        x1 = int(x1); y1 = int(y1); x2 = int(x2); y2 = int(y2)
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

def preprocess_frame(frame, raw_motion, boost_mask, cfg,
                     player_bboxes=None, court_keypoints=None,
                     visualize=False, hsv_cached=None,
                     protect_mask_cached=None,
                     rois_dbg=None):
    # Visualize mode: overlay boost/motion directly, skip HSV work
    if visualize:
        out = frame.copy()
        if raw_motion is None and boost_mask is None:
            return out

        motion_vis = (raw_motion > 0) if (raw_motion is not None and cfg.debug_show_raw_motion) else np.zeros(out.shape[:2], dtype=bool)
        boost_vis = (boost_mask > 0) if boost_mask is not None else np.zeros_like(motion_vis)

        # Keep protected regions (players + side-outside court) visually neutral.
        protect_mask = protect_mask_cached if protect_mask_cached is not None else \
            build_protect_mask(
                frame.shape[0], frame.shape[1],
                player_bboxes=player_bboxes,
                court_keypoints=court_keypoints,
                player_pad=cfg.player_bbox_pad)
        if protect_mask is not None:
            motion_vis = motion_vis & (~protect_mask)
            boost_vis = boost_vis & (~protect_mask)

        out[motion_vis & (~boost_vis)] = (0, 0, 128)
        out[boost_vis] = (0, 255, 255)
        if court_keypoints is not None:
            draw_court_side_guides(out, court_keypoints, color=(0, 0, 255), thickness=2)
            
        return out

    hsv = hsv_cached.copy() if hsv_cached is not None else cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s_f = s.astype(np.float32)
    v_f = v.astype(np.float32)
    protect_mask = protect_mask_cached if protect_mask_cached is not None else \
        build_protect_mask(
            frame.shape[0], frame.shape[1],
            player_bboxes=player_bboxes,
            court_keypoints=court_keypoints,
            player_pad=cfg.player_bbox_pad)
    bm = None

    if boost_mask is not None:
        bm = boost_mask > 0
        if protect_mask is not None:
            bm = bm & (~protect_mask)
        s_f[bm] = np.clip(s_f[bm] * cfg.pre_sat_boost, 0, 255)
        v_f[bm] = np.clip(v_f[bm] * cfg.pre_val_boost, 0, 255)

    if raw_motion is not None:
        if cfg.motion_dilate > 1:
            k = _get_kernel(cfg.motion_dilate)
            motion_keep = cv2.dilate(raw_motion, k, iterations=1)
        else:
            motion_keep = raw_motion
        static = (motion_keep == 0) & ~((s <= 110) & (v >= 120))
        if protect_mask is not None:
            static = static & (~protect_mask)
        v_f[static] = np.clip(v_f[static] * cfg.dim_static, 0, 255)
        s_copy = s.copy()
        s_copy[static] = np.clip(
            s_copy[static].astype(np.float32) * cfg.static_sat_scale, 0, 255).astype(np.uint8)
        s = s_copy

    v = v_f.astype(np.uint8)
    if cfg.pre_hue_shift > 0.0 and bm is not None:
        h = h.copy()
        h_f = h.astype(np.float32)
        h_f[bm] = np.clip(h_f[bm] * (1.0 - cfg.pre_hue_shift) + 30 * cfg.pre_hue_shift, 0, 179)
        h = h_f.astype(np.uint8)

    return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)

def _hsv_to_bgr_cuda(h_t, s_t, v_t):
    """Convert HSV tensors (all 0-1 range) to BGR tensor (0-255 uint8)."""
    h6 = h_t * 6.0
    c = v_t * s_t
    x = c * (1.0 - torch.abs((h6 % 2.0) - 1.0))
    m = v_t - c
    z = torch.zeros_like(h_t)
    sec = h6.long().clamp(0, 5)
    s0 = sec == 0
    s1 = sec == 1
    s2 = sec == 2
    s3 = sec == 3
    s4 = sec == 4
    s5 = sec == 5
    ro = torch.where(s0 | s5, c, torch.where(s1 | s4, x, z))
    go = torch.where(s1 | s2, c, torch.where(s0 | s3, x, z))
    bo = torch.where(s3 | s4, c, torch.where(s2 | s5, x, z))
    bgr = torch.stack([bo + m, go + m, ro + m], dim=0)
    return torch.clamp(bgr * 255, 0, 255).byte().permute(1, 2, 0).contiguous()

def _dilate_motion_cuda(raw_motion, dilate_k):
    """Dilate boolean motion mask on GPU via max_pool2d."""
    if dilate_k > 1:
        mm = raw_motion.float().unsqueeze(0).unsqueeze(0)
        mm = F.max_pool2d(mm, dilate_k, stride=1, padding=dilate_k // 2)
        return mm.squeeze() > 0.5
    return raw_motion

def _gaussian_blur_2d(t, kernel_size=3, sigma=1.0):
    """
    Applies a 2D Gaussian blur to a PyTorch tensor (B, C, H, W).
    Useful for smoothing out H.264 compression artifacts before differencing.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    c = t.shape[1]
    cache_key = (t.dtype, t.device, kernel_size, sigma, c)
    if cache_key not in _GAUSS_KERNEL_CACHE:
        x = torch.arange(kernel_size, dtype=t.dtype, device=t.device) - kernel_size // 2
        kernel_1d = torch.exp(- (x ** 2) / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
        kernel_2d = kernel_2d.expand(c, 1, kernel_size, kernel_size).contiguous()
        _GAUSS_KERNEL_CACHE[cache_key] = kernel_2d
    padding = kernel_size // 2
    return F.conv2d(t, _GAUSS_KERNEL_CACHE[cache_key], padding=padding, groups=c)

@torch.no_grad()
def preprocess_frame_cuda(frame, prev_v, prev_s, master_var_v, master_var_s, cfg,
                          player_bboxes=None, court_keypoints=None,
                          protect_mask_cached=None, rois=None,
                          skip_dim=False,
                          return_cuda_frame=False,
                          need_cpu_frame=True,
                          frame_gpu_t=None,
                          curr_v_cached=None,
                          curr_s_cached=None,
                          protect_mask_cuda_cached=None,
                          perf: Optional[Dict[str, float]] = None):
    """CUDA preprocessing with S+V motion detection.
    
    roi: optional (x1, y1, x2, y2) — if provided, only compute motion/CC
         inside this region. Outside the ROI, pixels are treated as static.
    """
    if torch is None or F is None:
        return frame, None, None, None, None, None

    perf_t0 = time.perf_counter() if perf is not None else 0.0
    perf_d2h = 0.0
    perf_cc = 0.0

    def _finalize_perf():
        if perf is None:
            return
        total = time.perf_counter() - perf_t0
        perf["pre_cuda_total"] = perf.get("pre_cuda_total", 0.0) + total
        perf["pre_cuda_raw_d2h"] = perf.get("pre_cuda_raw_d2h", 0.0) + perf_d2h
        perf["pre_cuda_cc_filter"] = perf.get("pre_cuda_cc_filter", 0.0) + perf_cc

    def _emit(pre_cuda_hwc, raw_u8, boost_u8, cv, cs):
        pre_np = pre_cuda_hwc.cpu().numpy() if (need_cpu_frame and pre_cuda_hwc is not None) else None
        pre_cuda = pre_cuda_hwc if return_cuda_frame else None
        return pre_np, raw_u8, boost_u8, cv.detach(), cs.detach(), pre_cuda

    device = torch.device("cuda")
    if frame_gpu_t is not None:
        t = frame_gpu_t
    else:
        t = torch.from_numpy(frame).to(device=device, dtype=torch.float32) / 255.0
        t = t.permute(2, 0, 1).contiguous()

    b, g, r = t[0], t[1], t[2]
    delta = None
    if curr_v_cached is not None and curr_s_cached is not None:
        curr_v = curr_v_cached
        curr_s = curr_s_cached
        maxc = curr_v
    else:
        maxc = torch.max(t, dim=0).values
        minc = torch.min(t, dim=0).values
        delta = maxc - minc
        curr_v = maxc
        curr_s = torch.where(maxc > 1e-6, delta / (maxc + 1e-6), torch.zeros_like(maxc))
        
        # Apply slight Gaussian Blur to V and S to smear out compression block breathing
        vs_tensor = torch.stack([curr_v, curr_s]).unsqueeze(0)  # Shape: (1, 2, H, W)
        vs_blurred = _gaussian_blur_2d(vs_tensor, kernel_size=3, sigma=0.8).squeeze(0)
        curr_v = vs_blurred[0]
        curr_s = vs_blurred[1]

    protect_t = protect_mask_cuda_cached
    if protect_t is None:
        protect_np = protect_mask_cached if protect_mask_cached is not None else \
            build_protect_mask(
                frame.shape[0], frame.shape[1],
                player_bboxes=player_bboxes,
                court_keypoints=court_keypoints,
                player_pad=cfg.player_bbox_pad)
        if protect_np is not None:
            protect_t = torch.from_numpy(protect_np).to(device)

    raw_motion = None
    if prev_v is not None and prev_s is not None and master_var_v is not None and master_var_s is not None:
        thr = float(cfg.motion_thresh) / 255.0
        thr_sq = thr ** 2
        k_sq = float(getattr(cfg, 'motion_k_std', 3.5)) ** 2
        v_min = float(cfg.motion_v_min) / 255.0
        
        # Blur for motion difference calculation to suppress compression block breathing
        vs_mot = torch.stack([curr_v, curr_s, prev_v, prev_s]).unsqueeze(0)
        vs_mot_blur = _gaussian_blur_2d(vs_mot, kernel_size=3, sigma=0.8).squeeze(0)
        mot_curr_v, mot_curr_s, mot_bg_v, mot_bg_s = vs_mot_blur[0], vs_mot_blur[1], vs_mot_blur[2], vs_mot_blur[3]
        
        # If ROIs are specified, only compute motion diff inside the ROI regions
        if rois is not None:
            raw_motion = torch.zeros_like(curr_v, dtype=torch.bool)
            for roi in rois:
                rx1, ry1, rx2, ry2 = roi
                v_curr_roi = mot_curr_v[ry1:ry2, rx1:rx2]
                v_bg_roi = mot_bg_v[ry1:ry2, rx1:rx2]
                v_var = master_var_v[ry1:ry2, rx1:rx2]
                
                s_curr_roi = mot_curr_s[ry1:ry2, rx1:rx2]
                s_bg_roi = mot_bg_s[ry1:ry2, rx1:rx2]
                s_var = master_var_s[ry1:ry2, rx1:rx2]

                v_diff_sq = (v_curr_roi - v_bg_roi)**2
                s_diff_sq = (s_curr_roi - s_bg_roi)**2
                
                v_thresh_sq = torch.clamp(v_var * k_sq, min=thr_sq)
                s_thresh_sq = torch.clamp(s_var * k_sq, min=thr_sq) * 1.5  # Penalize saturation noise

                motion_roi = ((v_diff_sq > v_thresh_sq) | (s_diff_sq > s_thresh_sq)) & (v_curr_roi > v_min)
                
                # Soft Morphological Opening inside ROI (Soft Erode then Dilate) 
                # 1. Soft Erode: Use average pooling to count neighbors. Kill strictly isolated 1-pixel noise (sum < 1.5).
                mf = motion_roi.float().unsqueeze(0).unsqueeze(0)
                mf_sum = F.avg_pool2d(mf, 3, stride=1, padding=1) * 9.0
                mf = (mf_sum >= 1.5).float()
                # 2. Dilate SECOND: This restores the size of the REAL motion blobs that survived
                mf = F.max_pool2d(mf, 3, stride=1, padding=1)    # Dilate
                motion_roi_clean = (mf.squeeze(0).squeeze(0) > 0.5)

                raw_motion[ry1:ry2, rx1:rx2] = raw_motion[ry1:ry2, rx1:rx2] | motion_roi_clean
        else:
            # Full-frame motion (original path)
            v_diff_sq = (mot_curr_v - mot_bg_v)**2
            s_diff_sq = (mot_curr_s - mot_bg_s)**2
            
            v_thresh_sq = torch.clamp(master_var_v * k_sq, min=thr_sq)
            s_thresh_sq = torch.clamp(master_var_s * k_sq, min=thr_sq) * 1.5
            
            motion = ((v_diff_sq > v_thresh_sq) | (s_diff_sq > s_thresh_sq)) & (mot_curr_v > v_min)
            
            # Soft Morphological Opening (Soft Erode then Dilate)
            mf = motion.float().unsqueeze(0).unsqueeze(0)
            mf_sum = F.avg_pool2d(mf, 3, stride=1, padding=1) * 9.0
            mf = (mf_sum >= 1.5).float()
            mf = F.max_pool2d(mf, 3, stride=1, padding=1)    # Dilate
            raw_motion = (mf.squeeze(0).squeeze(0) > 0.5)

    raw_motion_u8 = boost_mask_u8 = None
    boost_has_blobs = False
    if raw_motion is not None:
        if rois is not None:
            h_u8 = int(curr_v.shape[0])
            w_u8 = int(curr_v.shape[1])
            raw_motion_u8 = np.zeros((h_u8, w_u8), dtype=np.uint8)
            boost_mask_u8 = np.zeros((h_u8, w_u8), dtype=np.uint8)

            for roi in rois:
                rx1, ry1, rx2, ry2 = roi
                t_d2h = time.perf_counter() if perf is not None else 0.0
                raw_motion_roi_u8 = (
                    raw_motion[ry1:ry2, rx1:rx2].contiguous().byte().cpu().numpy() * 255
                )
                if perf is not None:
                    perf_d2h += (time.perf_counter() - t_d2h)
                
                # Apply mask piece where ROIs might overlap
                np.maximum(raw_motion_u8[ry1:ry2, rx1:rx2], raw_motion_roi_u8, out=raw_motion_u8[ry1:ry2, rx1:rx2])
                
                if raw_motion_roi_u8.max() > 0:
                    t_cc = time.perf_counter() if perf is not None else 0.0
                    filtered_roi = filter_boost_mask(
                        raw_motion_roi_u8, cfg.boost_min_blob_area, cfg.boost_max_blob_area, cfg,
                        player_bboxes=player_bboxes)
                    if perf is not None:
                        perf_cc += (time.perf_counter() - t_cc)
                    
                    np.maximum(boost_mask_u8[ry1:ry2, rx1:rx2], filtered_roi, out=boost_mask_u8[ry1:ry2, rx1:rx2])

            boost_has_blobs = bool(boost_mask_u8.max() > 0)
            if not boost_has_blobs:
                boost_mask_u8 = None
        else:
            t_d2h = time.perf_counter() if perf is not None else 0.0
            raw_motion_u8 = (raw_motion.byte().cpu().numpy() * 255)
            if perf is not None:
                perf_d2h += (time.perf_counter() - t_d2h)
            t_cc = time.perf_counter() if perf is not None else 0.0
            boost_mask_u8 = filter_boost_mask(
                raw_motion_u8, cfg.boost_min_blob_area, cfg.boost_max_blob_area, cfg,
                player_bboxes=player_bboxes)
            if perf is not None:
                perf_cc += (time.perf_counter() - t_cc)
            boost_has_blobs = bool(boost_mask_u8 is not None and boost_mask_u8.max() > 0)

    # Early exit: no ball-sized blobs → just dim static regions
    if not boost_has_blobs:
        if raw_motion is not None and not skip_dim:
            motion_keep = _dilate_motion_cuda(raw_motion, cfg.motion_dilate)
            static = ~motion_keep
            if protect_t is not None:
                static = static & (~protect_t)
            # Reuse already-uploaded CHW tensor instead of re-uploading CPU frame.
            t_out = t * 255.0
            static_3d = static.unsqueeze(0).expand_as(t_out)
            t_out = torch.where(static_3d, t_out * cfg.dim_static, t_out)
            t_out = torch.clamp(t_out, 0, 255).byte().permute(1, 2, 0).contiguous()
            out = _emit(t_out, raw_motion_u8, boost_mask_u8, curr_v, curr_s)
            _finalize_perf()
            return out
        if return_cuda_frame:
            frame_cuda = torch.clamp(t * 255.0, 0, 255).byte().permute(1, 2, 0).contiguous()
            out = _emit(frame_cuda, raw_motion_u8, boost_mask_u8, curr_v, curr_s)
            _finalize_perf()
            return out
        out = (frame, raw_motion_u8, boost_mask_u8, curr_v.detach(), curr_s.detach(), None)
        _finalize_perf()
        return out

    # Full HSV manipulation (only when ball-sized blobs found)
    if delta is None:
        minc = torch.min(t, dim=0).values
        delta = maxc - minc

    h_t = torch.zeros_like(maxc)
    mask = delta > 1e-6
    h_r = ((g - b) / (delta + 1e-6)) % 6.0
    h_g = ((b - r) / (delta + 1e-6)) + 2.0
    h_b = ((r - g) / (delta + 1e-6)) + 4.0
    h_t = torch.where((maxc == r) & mask, h_r, h_t)
    h_t = torch.where((maxc == g) & mask, h_g, h_t)
    h_t = torch.where((maxc == b) & mask, h_b, h_t)
    h_t = (h_t / 6.0) % 1.0
    s_t = curr_s.clone()
    v_t = curr_v.clone()

    bm = None
    if boost_mask_u8 is not None:
        bm = torch.from_numpy(boost_mask_u8 > 0).to(device)
        if protect_t is not None:
            bm = bm & (~protect_t)
        s_t = torch.where(bm, torch.clamp(s_t * cfg.pre_sat_boost, 0, 1), s_t)
        v_t = torch.where(bm, torch.clamp(v_t * cfg.pre_val_boost, 0, 1), v_t)

    if raw_motion is not None:
        motion_keep = _dilate_motion_cuda(raw_motion, cfg.motion_dilate)
        static = ~motion_keep
        white = (s_t <= 70.0 / 255.0) & (v_t >= 170.0 / 255.0)
        static = static & (~white)
        if protect_t is not None:
            static = static & (~protect_t)
        v_t = torch.where(static, torch.clamp(v_t * cfg.dim_static, 0, 1), v_t)
        s_t = torch.where(static, torch.clamp(s_t * cfg.static_sat_scale, 0, 1), s_t)

    if cfg.pre_hue_shift > 0.0 and bm is not None:
        ht = 30.0 / 179.0
        h_t = torch.where(bm, torch.clamp(
            h_t * (1 - cfg.pre_hue_shift) + ht * cfg.pre_hue_shift, 0, 1), h_t)

    bgr = _hsv_to_bgr_cuda(h_t, s_t, v_t)
    out = _emit(bgr, raw_motion_u8, boost_mask_u8, curr_v, curr_s)
    _finalize_perf()
    return out

    def preprocess_frame(self, frame_bgr: np.ndarray):
        h0, w0 = frame_bgr.shape[:2]
        t = torch.from_numpy(frame_bgr).to(device=self.device, dtype=torch.float32)
        t = t.permute(2, 0, 1).unsqueeze(0).contiguous()
        t = t[:, [2, 1, 0], :, :] / 255.0
        if h0 != self.input_h or w0 != self.input_w:
            t = F.interpolate(t, size=(self.input_h, self.input_w), mode="bilinear", align_corners=False)
        if self.fp16:
            t = t.half()
        scale = (w0 / float(self.input_w), h0 / float(self.input_h))
        return t, scale

