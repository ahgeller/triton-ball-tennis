"""Motion evidence for pass 1: the background model, raw and boost motion
masks on CUDA, the court-side protect mask and the detector-input enhancement."""
from __future__ import annotations

import math
import time
from typing import Dict, List, NamedTuple, Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .config import Config
from .court_geometry import build_court_side_guides
from .frame_window import FrameWindow


_KERNEL_CACHE = {}
_KERNEL_3x3 = np.ones((3, 3), np.uint8)

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

def _motion_probe_delta_from_frames(prev_bgr: np.ndarray, curr_bgr: np.ndarray) -> np.ndarray:
    """C++ probe-style frame delta: max(gray, V, scaled-S), blurred 3x3."""
    prev_hsv = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2HSV)
    curr_hsv = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2HSV)
    prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
    v_diff = cv2.absdiff(curr_hsv[:, :, 2], prev_hsv[:, :, 2])
    s_diff = cv2.absdiff(curr_hsv[:, :, 1], prev_hsv[:, :, 1])
    gray_diff = cv2.absdiff(curr_gray, prev_gray)
    s_scaled = np.minimum(s_diff.astype(np.float32) * 1.25, 255.0).astype(np.uint8)
    combined = np.maximum(gray_diff, np.maximum(v_diff, s_scaled))
    return cv2.GaussianBlur(combined, (3, 3), 0)


def _clean_raw_motion_mask(mask_u8: Optional[np.ndarray], cfg: Config) -> Optional[np.ndarray]:
    """Apply the active raw-mask morphology without appearance or size heuristics."""
    if mask_u8 is None or mask_u8.size == 0:
        return mask_u8
    if cv2.countNonZero(mask_u8) == 0:
        return mask_u8

    out = mask_u8
    close_size = int(cfg.motion_raw_close_size or 0)
    open_size = int(cfg.motion_raw_open_size or 0)
    if close_size > 1:
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, _get_kernel(close_size))
    if open_size > 1:
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, _get_kernel(open_size))

    return out

def refine_raw_motion_temporal_cpu(
    raw_motion_u8: Optional[np.ndarray],
    prev_bgr: Optional[np.ndarray],
    curr_bgr: Optional[np.ndarray],
    next_bgr: Optional[np.ndarray],
    cfg: Config,
) -> Optional[np.ndarray]:
    """Clean raw foreground with the C++ probe's temporal movement test.

    CPU reference for the CUDA gate in ``preprocess_frame_cuda``; the runtime
    itself only uses the CUDA path (see tests/test_fixes.py).
    """
    if raw_motion_u8 is None:
        return None
    refined = raw_motion_u8
    if bool(cfg.motion_raw_temporal_gate) and prev_bgr is not None and curr_bgr is not None:
        hi = float(cfg.motion_raw_temporal_hi)
        lo = float(cfg.motion_raw_temporal_lo)
        very_hi = float(cfg.motion_raw_temporal_very_hi)
        d_prev = _motion_probe_delta_from_frames(prev_bgr, curr_bgr)
        if next_bgr is not None:
            d_next = _motion_probe_delta_from_frames(curr_bgr, next_bgr)
            temporal = (
                ((d_prev >= hi) & (d_next >= lo)) |
                ((d_next >= hi) & (d_prev >= lo)) |
                (d_prev >= very_hi)
            )
        else:
            temporal = d_prev >= hi
        refined = cv2.bitwise_and(refined, temporal.astype(np.uint8) * 255)
    return _clean_raw_motion_mask(refined, cfg)

def filter_boost_mask(raw_motion, min_area, max_area, cfg, buffers=None):
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
            buffers.boost_mask_u8.fill(0)
            return buffers.boost_mask_u8  # Return pre-allocated empty buffer
        return np.zeros((img_h, img_w), dtype=np.uint8)
    
    preserve_tiny = cfg.blob_preserve_tiny
    tiny_max_area = cfg.blob_tiny_max_area
    
    img_diag = math.hypot(float(img_w), float(img_h))
    ball_bypass_max_area = max(15, min(150, int(round(0.035 * img_diag))))
    if preserve_tiny:
        ball_bypass_max_area = max(ball_bypass_max_area, tiny_max_area)
    
    # Single CC call - the major bottleneck
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(raw_motion, connectivity=8)
    
    if num <= 1:
        if buffers is not None:
            buffers.boost_mask_u8.fill(0)
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
            
        # Area check
        if area < min_area or area > max_area:
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
            # Draw a solid circle representing this blob's position.
            # Cap radius to ~12px so the boost mask doesn't balloon far beyond the
            # actual ball - large blobs (motion blur, shadows) would otherwise create
            # huge yellow regions in the debug view and over-expand the detector input.
            cx = int(centroids[i][0] + 0.5)
            cy = int(centroids[i][1] + 0.5)
            radius = min(int(math.sqrt(area / math.pi) + 0.5), 12)
            cv2.circle(result, (cx, cy), max(radius, 1), 255, -1)
            drawn_any = True
            
    if not drawn_any:
        return np.zeros((img_h, img_w), dtype=np.uint8) if buffers is None else result
    
    return result


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
      offset from kp2 to image-left by |kp5-kp2|, line parallel to kp2->kp0.
      Pixels left of this line are protected.
    Right side:
      offset from kp3 to image-right by |kp7-kp3|, line parallel to kp3->kp1.
      Pixels right of this line are protected.
    """
    guides = build_court_side_guides(court_keypoints)
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

def _motion_probe_delta_cuda(prev_frame_t, curr_frame_t, prev_v_t, prev_s_t, curr_v_t, curr_s_t):
    """C++ probe-style frame delta in 0-255 units on CUDA."""
    prev_gray = prev_frame_t[0] * 0.114 + prev_frame_t[1] * 0.587 + prev_frame_t[2] * 0.299
    curr_gray = curr_frame_t[0] * 0.114 + curr_frame_t[1] * 0.587 + curr_frame_t[2] * 0.299
    gray_diff = torch.abs(curr_gray - prev_gray) * 255.0
    v_diff = torch.abs(curr_v_t - prev_v_t) * 255.0
    s_diff = torch.abs(curr_s_t - prev_s_t) * (255.0 * 1.25)
    delta = torch.maximum(gray_diff, torch.maximum(v_diff, s_diff))
    return F.avg_pool2d(delta.unsqueeze(0).unsqueeze(0), 3, stride=1, padding=1).squeeze(0).squeeze(0)


class BackgroundModel:
    """Running mean and variance of the S and V planes, on CUDA.

    A pixel counts as motion when its squared distance from the mean exceeds
    ``motion_thresh^2 + motion_k_std^2 * variance`` (see
    ``preprocess_frame_cuda``).  Static pixels adapt at ``background_alpha``;
    pixels that were motion on the previous frame at ``motion_freeze_alpha``.
    Call ``update`` once per frame before preprocessing it and
    ``remember_motion`` with that frame's raw mask afterwards.
    """

    def __init__(self, cfg: Config, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.mean_v: Optional[torch.Tensor] = None
        self.mean_s: Optional[torch.Tensor] = None
        self.var_v: Optional[torch.Tensor] = None
        self.var_s: Optional[torch.Tensor] = None
        self._prev_motion: Optional[torch.Tensor] = None    # bool, last frame's raw motion

    @property
    def ready(self) -> bool:
        return self.mean_v is not None

    @torch.no_grad()
    def update(self, curr_v: torch.Tensor, curr_s: torch.Tensor) -> None:
        """Fold the current frame's V and S planes into the model."""
        if not self.ready:
            self.mean_v = curr_v.clone()
            self.mean_s = curr_s.clone()
            thr = float(self.cfg.motion_thresh) / 255.0
            self.var_v = torch.full_like(self.mean_v, thr ** 2)
            self.var_s = torch.full_like(self.mean_s, thr ** 2)
            return
        v_diff_sq = (curr_v - self.mean_v) ** 2
        s_diff_sq = (curr_s - self.mean_s) ** 2
        if self._prev_motion is not None:
            alpha = torch.where(self._prev_motion, float(self.cfg.motion_freeze_alpha),
                                float(self.cfg.background_alpha))
        else:
            alpha = float(self.cfg.background_alpha)
        self.var_v.mul_(1.0 - alpha).add_(alpha * v_diff_sq)
        self.var_s.mul_(1.0 - alpha).add_(alpha * s_diff_sq)
        self.mean_v.mul_(1.0 - alpha).add_(curr_v * alpha)
        self.mean_s.mul_(1.0 - alpha).add_(curr_s * alpha)

    def remember_motion(self, raw_motion_u8: Optional[np.ndarray]) -> None:
        """Keep this frame's raw motion on the GPU for the next update's slow-adaptation mask."""
        if raw_motion_u8 is None:
            self._prev_motion = None
        else:
            self._prev_motion = torch.from_numpy(raw_motion_u8).to(self.device, non_blocking=True) > 0


class PreprocessResult(NamedTuple):
    """What ``preprocess_frame_cuda`` produces for one frame.

    ``detector_frame`` is the CPU BGR detector input: the enhanced frame when
    one was built and ``need_cpu_frame`` asked for it, the source frame when
    nothing was applied, else None.  ``detector_frame_cuda`` is the enhanced
    frame as an HWC uint8 CUDA tensor when ``return_cuda_frame`` asked for
    it.  ``raw_motion`` and ``boost_mask`` are full-frame uint8 {0, 255}
    masks, None when the background model was not ready or no ball-sized
    blob survived.
    """

    detector_frame: Optional[np.ndarray]
    detector_frame_cuda: Optional[torch.Tensor]
    raw_motion: Optional[np.ndarray]
    boost_mask: Optional[np.ndarray]


Roi = Tuple[int, int, int, int]
# Saturation noise is penalised by widening its threshold this much.
_S_THRESHOLD_SCALE = 1.5
# Pixels this bright and unsaturated are white court lines: never dimmed.
_WHITE_MAX_S = 70.0 / 255.0
_WHITE_MIN_V = 170.0 / 255.0
# The boost hue (yellow-green) that the hue shift pulls boosted pixels toward, as a 0-1 hue.
_BOOST_HUE = 30.0 / 179.0


def _raw_motion_cuda(window: FrameWindow, background: BackgroundModel, cfg: Config,
                     rois: Optional[List[Roi]]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """``(gated, ungated)`` boolean raw-motion masks for the current frame.

    ``ungated`` is the background-model foreground; ``gated`` additionally
    passes the frame-to-frame temporal test when it is enabled and a previous
    frame exists.  Both are None until the background model is ready; they are
    the same tensor when the gate did not run.
    """
    if not background.ready:
        return None, None
    curr = window.curr_gpu
    curr_v, curr_s = curr.v, curr.s
    thr_sq = (float(cfg.motion_thresh) / 255.0) ** 2
    k_sq = float(cfg.motion_k_std) ** 2
    v_min = float(cfg.motion_v_min) / 255.0

    def foreground(y1: int, y2: int, x1: int, x2: int) -> torch.Tensor:
        v_diff_sq = (curr_v[y1:y2, x1:x2] - background.mean_v[y1:y2, x1:x2]) ** 2
        s_diff_sq = (curr_s[y1:y2, x1:x2] - background.mean_s[y1:y2, x1:x2]) ** 2
        v_thresh_sq = thr_sq + background.var_v[y1:y2, x1:x2] * k_sq
        s_thresh_sq = (thr_sq + background.var_s[y1:y2, x1:x2] * k_sq) * _S_THRESHOLD_SCALE
        return ((v_diff_sq > v_thresh_sq) | (s_diff_sq > s_thresh_sq)) & (curr_v[y1:y2, x1:x2] > v_min)

    if rois is not None:
        raw_motion = torch.zeros_like(curr_v, dtype=torch.bool)
        for rx1, ry1, rx2, ry2 in rois:
            raw_motion[ry1:ry2, rx1:rx2] = raw_motion[ry1:ry2, rx1:rx2] | foreground(ry1, ry2, rx1, rx2)
    else:
        raw_motion = foreground(0, int(curr_v.shape[0]), 0, int(curr_v.shape[1]))

    prev = window.prev_gpu
    if not (bool(cfg.motion_raw_temporal_gate) and prev is not None):
        return raw_motion, raw_motion
    hi = float(cfg.motion_raw_temporal_hi)
    lo = float(cfg.motion_raw_temporal_lo)
    very_hi = float(cfg.motion_raw_temporal_very_hi)
    nxt = window.next_gpu

    def temporal(y1: int, y2: int, x1: int, x2: int) -> torch.Tensor:
        d_prev = _motion_probe_delta_cuda(
            prev.chw[:, y1:y2, x1:x2], curr.chw[:, y1:y2, x1:x2],
            prev.v[y1:y2, x1:x2], prev.s[y1:y2, x1:x2],
            curr_v[y1:y2, x1:x2], curr_s[y1:y2, x1:x2],
        )
        if nxt is None:
            return d_prev >= hi
        d_next = _motion_probe_delta_cuda(
            curr.chw[:, y1:y2, x1:x2], nxt.chw[:, y1:y2, x1:x2],
            curr_v[y1:y2, x1:x2], curr_s[y1:y2, x1:x2],
            nxt.v[y1:y2, x1:x2], nxt.s[y1:y2, x1:x2],
        )
        return (
            ((d_prev >= hi) & (d_next >= lo)) |
            ((d_next >= hi) & (d_prev >= lo)) |
            (d_prev >= very_hi)
        )

    if rois is not None:
        temporal_motion = torch.zeros_like(raw_motion)
        for rx1, ry1, rx2, ry2 in rois:
            temporal_motion[ry1:ry2, rx1:rx2] = temporal_motion[ry1:ry2, rx1:rx2] | temporal(ry1, ry2, rx1, rx2)
    else:
        temporal_motion = temporal(0, int(curr_v.shape[0]), 0, int(curr_v.shape[1]))
    # Ungated motion drives the detector-input boost (recall: helps the detector
    # recover misses). Gated motion drives the selector and the static dimming
    # (precision: rejects false positives).
    return raw_motion & temporal_motion, raw_motion


class _BoostMasks(NamedTuple):
    raw_u8: Optional[np.ndarray]
    selector_boost_u8: Optional[np.ndarray]    # narrow, from the gated mask
    detector_boost_u8: Optional[np.ndarray]    # wide, from the ungated mask (or the narrow one)
    has_blobs: bool


def _boost_masks(raw_motion: Optional[torch.Tensor], raw_motion_ungated: Optional[torch.Tensor],
                 cfg: Config, rois: Optional[List[Roi]], need_detector_boost: bool,
                 perf: Optional[Dict[str, float]]) -> _BoostMasks:
    """Download the raw mask(s) and keep only ball-sized components.

    Two boost masks have different roles: the narrow one from the gated mask
    feeds the selector's motion-blob picker (precision), the wide one from the
    ungated mask drives the detector-input brightening (recall: when the
    temporal gate would suppress a real ball blob, the wider mask still
    highlights it so the detector can find it).
    """
    if raw_motion is None:
        return _BoostMasks(None, None, None, False)
    boost_source = raw_motion_ungated if need_detector_boost and raw_motion_ungated is not None else raw_motion
    wide_distinct = boost_source is not raw_motion
    min_area, max_area = cfg.boost_min_blob_area, cfg.boost_max_blob_area

    def download(tensor: torch.Tensor) -> np.ndarray:
        t_d2h = time.perf_counter() if perf is not None else 0.0
        out = tensor.contiguous().byte().cpu().numpy() * 255
        if perf is not None:
            perf["pre_cuda_raw_d2h"] = perf.get("pre_cuda_raw_d2h", 0.0) + (time.perf_counter() - t_d2h)
        return out

    def components(mask_u8: np.ndarray, timed: bool) -> np.ndarray:
        t_cc = time.perf_counter() if (perf is not None and timed) else 0.0
        out = filter_boost_mask(mask_u8, min_area, max_area, cfg)
        if perf is not None and timed:
            perf["pre_cuda_cc_filter"] = perf.get("pre_cuda_cc_filter", 0.0) + (time.perf_counter() - t_cc)
        return out

    if rois is not None:
        h_u8, w_u8 = int(raw_motion.shape[0]), int(raw_motion.shape[1])
        raw_u8 = np.zeros((h_u8, w_u8), dtype=np.uint8)
        narrow_u8 = np.zeros((h_u8, w_u8), dtype=np.uint8)
        wide_u8 = np.zeros((h_u8, w_u8), dtype=np.uint8) if wide_distinct else None
        for rx1, ry1, rx2, ry2 in rois:
            raw_roi = download(raw_motion[ry1:ry2, rx1:rx2])
            wide_roi = download(boost_source[ry1:ry2, rx1:rx2]) if wide_distinct else None
            raw_roi = _clean_raw_motion_mask(raw_roi, cfg)
            np.maximum(raw_u8[ry1:ry2, rx1:rx2], raw_roi, out=raw_u8[ry1:ry2, rx1:rx2])
            if raw_roi.max() > 0:
                np.maximum(narrow_u8[ry1:ry2, rx1:rx2], components(raw_roi, timed=True),
                           out=narrow_u8[ry1:ry2, rx1:rx2])
            if wide_distinct and wide_roi is not None and wide_roi.max() > 0:
                np.maximum(wide_u8[ry1:ry2, rx1:rx2], components(wide_roi, timed=False),
                           out=wide_u8[ry1:ry2, rx1:rx2])
        narrow = narrow_u8 if narrow_u8.max() > 0 else None
        if not wide_distinct:
            wide = narrow
        else:
            wide = wide_u8 if wide_u8.max() > 0 else None
        return _BoostMasks(raw_u8, narrow, wide, bool(wide is not None or narrow is not None))

    raw_u8 = _clean_raw_motion_mask(download(raw_motion), cfg)
    wide_src = download(boost_source) if wide_distinct else None
    narrow = components(raw_u8, timed=True)
    if wide_distinct and wide_src is not None:
        wide = components(wide_src, timed=False)
        if wide is not None and wide.max() == 0:
            wide = None
    else:
        wide = narrow
    has_blobs = bool((narrow is not None and narrow.max() > 0) or (wide is not None and wide.max() > 0))
    return _BoostMasks(raw_u8, narrow, wide, has_blobs)


def _bgr_u8_hwc(t_chw: torch.Tensor) -> torch.Tensor:
    """A CHW float [0,1] frame as the HWC uint8 tensor the detector input writer takes."""
    return torch.clamp(t_chw * 255.0, 0, 255).byte().permute(1, 2, 0).contiguous()


def _dim_static_bgr(t_chw: torch.Tensor, raw_motion: torch.Tensor, protect: Optional[torch.Tensor],
                    cfg: Config) -> torch.Tensor:
    """The frame with every pixel away from motion scaled by ``dim_static``, as HWC uint8."""
    static = ~_dilate_motion_cuda(raw_motion, cfg.motion_dilate)
    if protect is not None:
        static = static & (~protect)
    t_out = t_chw * 255.0
    static_3d = static.unsqueeze(0).expand_as(t_out)
    t_out = torch.where(static_3d, t_out * cfg.dim_static, t_out)
    return torch.clamp(t_out, 0, 255).byte().permute(1, 2, 0).contiguous()


def _enhance_bgr(t_chw: torch.Tensor, curr_v: torch.Tensor, curr_s: torch.Tensor,
                 raw_motion: Optional[torch.Tensor], boost_u8: Optional[np.ndarray],
                 protect: Optional[torch.Tensor], cfg: Config) -> torch.Tensor:
    """Boost S/V (and shift hue) inside the boost mask and dim static pixels, as HWC uint8."""
    b, g, r = t_chw[0], t_chw[1], t_chw[2]
    maxc = curr_v
    minc = torch.min(t_chw, dim=0).values
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
    if boost_u8 is not None:
        bm = torch.from_numpy(boost_u8 > 0).to(t_chw.device)
        if protect is not None:
            bm = bm & (~protect)
        s_t = torch.where(bm, torch.clamp(s_t * cfg.pre_sat_boost, 0, 1), s_t)
        v_t = torch.where(bm, torch.clamp(v_t * cfg.pre_val_boost, 0, 1), v_t)

    if raw_motion is not None:
        static = ~_dilate_motion_cuda(raw_motion, cfg.motion_dilate)
        white = (s_t <= _WHITE_MAX_S) & (v_t >= _WHITE_MIN_V)
        static = static & (~white)
        if protect is not None:
            static = static & (~protect)
        v_t = torch.where(static, torch.clamp(v_t * cfg.dim_static, 0, 1), v_t)
        s_t = torch.where(static, torch.clamp(s_t * cfg.static_sat_scale, 0, 1), s_t)

    if cfg.pre_hue_shift > 0.0 and bm is not None:
        h_t = torch.where(bm, torch.clamp(
            h_t * (1 - cfg.pre_hue_shift) + _BOOST_HUE * cfg.pre_hue_shift, 0, 1), h_t)
    return _hsv_to_bgr_cuda(h_t, s_t, v_t)


@torch.no_grad()
def preprocess_frame_cuda(window: FrameWindow, background: BackgroundModel, cfg: Config, *,
                          rois: Optional[List[Roi]] = None,
                          protect_mask: Optional[torch.Tensor] = None,
                          skip_dim: bool = False,
                          return_cuda_frame: bool = False,
                          need_cpu_frame: bool = True,
                          need_detector_boost: bool = True,
                          perf: Optional[Dict[str, float]] = None) -> PreprocessResult:
    """Motion masks for the current frame of ``window`` and, if wanted, the enhanced detector input.

    ``rois`` limits the motion computation to those ``(x1, y1, x2, y2)``
    regions (None = the whole frame); ``protect_mask`` is a boolean CUDA mask
    of pixels that are never dimmed or boosted (the court-side mask).  With
    ``skip_dim`` the frame is not dimmed when no ball-sized blob was found.
    ``need_cpu_frame`` / ``return_cuda_frame`` choose which detector inputs to
    materialise; ``need_detector_boost`` builds the wide boost from ungated
    motion.  ``perf`` accumulates ``pre_cuda_total`` / ``pre_cuda_raw_d2h`` /
    ``pre_cuda_cc_filter`` seconds when given.
    """
    perf_t0 = time.perf_counter() if perf is not None else 0.0
    frame = window.curr
    curr = window.curr_gpu
    t = curr.chw

    raw_motion, raw_motion_ungated = _raw_motion_cuda(window, background, cfg, rois)
    masks = _boost_masks(raw_motion, raw_motion_ungated, cfg, rois, need_detector_boost, perf)

    enhanced: Optional[torch.Tensor] = None       # HWC uint8 on CUDA, None = the source frame
    if skip_dim and not need_cpu_frame and not return_cuda_frame:
        pass
    elif not masks.has_blobs:
        # No ball-sized blob: at most dim the static background.
        if raw_motion is not None and not skip_dim:
            enhanced = _dim_static_bgr(t, raw_motion, protect_mask, cfg)
        elif return_cuda_frame:
            enhanced = _bgr_u8_hwc(t)
    else:
        boost_src = masks.detector_boost_u8 if masks.detector_boost_u8 is not None else masks.selector_boost_u8
        enhanced = _enhance_bgr(t, curr.v, curr.s, raw_motion, boost_src, protect_mask, cfg)

    if enhanced is None:
        result = PreprocessResult(frame, None, masks.raw_u8, masks.selector_boost_u8)
    else:
        result = PreprocessResult(enhanced.cpu().numpy() if need_cpu_frame else None,
                                  enhanced if return_cuda_frame else None,
                                  masks.raw_u8, masks.selector_boost_u8)
    if perf is not None:
        perf["pre_cuda_total"] = perf.get("pre_cuda_total", 0.0) + (time.perf_counter() - perf_t0)
    return result
