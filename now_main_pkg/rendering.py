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

# Trail rendering constants
COLOR_DET = (0, 255, 0)
COLOR_MOTION = (0, 165, 255)
COLOR_INTERP = (0, 255, 255)  # yellow: prolonged guessed/stuck
COLOR_CARRY = (255, 0, 0)     # blue: short gap-connection
COLOR_RAW = (0, 180, 180)
COLOR_SEARCH = (255, 0, 255)
COLOR_GUIDE = (255, 255, 255)
COLOR_GUIDE_INTERP = (180, 220, 255)
COLOR_GAP = (0, 0, 0)
GAP_END_TRIM_PX = 6
ENABLE_GAP_CONNECTORS = False

_SOURCE_BASE_COLOR: Dict[str, Tuple[int, int, int]] = {
    "det": (40, 255, 40),
    "motion": (0, 185, 255),
    "carry": COLOR_CARRY,
    "interp": COLOR_INTERP,
    "guide": COLOR_GUIDE,
    "gap": COLOR_GAP,
}


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


def _build_court_side_guides(court_keypoints):
    p0 = _court_kpt_xy(court_keypoints, 0)
    p3 = _court_kpt_xy(court_keypoints, 3)
    p4 = _court_kpt_xy(court_keypoints, 4)
    p5 = _court_kpt_xy(court_keypoints, 5)
    p6 = _court_kpt_xy(court_keypoints, 6)
    p7 = _court_kpt_xy(court_keypoints, 7)
    if any(p is None for p in (p0, p3, p4, p5, p6, p7)):
        return None

    guides = {}

    # Left: use |5-4| as offset, extend parallel to 4->0.
    d_left = float(np.hypot(*(p5 - p4)))
    v_left = p0 - p4
    if d_left >= 1.0 and float(np.hypot(v_left[0], v_left[1])) >= 1e-6:
        guides["left"] = {
            "base": p4,
            "anchor": p4 + np.array([-d_left, 0.0], dtype=np.float32),
            "dir": v_left,
        }

    # Right: use |6-7| as offset, extend parallel to 7->3.
    d_right = float(np.hypot(*(p6 - p7)))
    v_right = p3 - p7
    if d_right >= 1.0 and float(np.hypot(v_right[0], v_right[1])) >= 1e-6:
        guides["right"] = {
            "base": p7,
            "anchor": p7 + np.array([d_right, 0.0], dtype=np.float32),
            "dir": v_right,
        }

    return guides or None

def _clip_infinite_line_to_frame(anchor, direction, w, h):
    n = float(np.hypot(direction[0], direction[1]))
    if n < 1e-6:
        return None
    L = float(max(w, h) * 4.0 + 1000.0)
    p1 = (int(round(anchor[0] - direction[0] * L)),
          int(round(anchor[1] - direction[1] * L)))
    p2 = (int(round(anchor[0] + direction[0] * L)),
          int(round(anchor[1] + direction[1] * L)))
    ok, q1, q2 = cv2.clipLine((0, 0, int(w), int(h)), p1, p2)
    if not ok:
        return None
    return q1, q2

def draw_court_side_guides(frame, court_keypoints, color=(0, 0, 255), thickness=2):
    """Draw offset side guides: base horizontal offsets + extended side lines."""
    guides = _build_court_side_guides(court_keypoints)
    if not guides:
        return frame

    h, w = frame.shape[:2]
    for side in ("left", "right"):
        g = guides.get(side)
        if g is None:
            continue
        base = tuple(np.round(g["base"]).astype(int))
        anchor = tuple(np.round(g["anchor"]).astype(int))
        cv2.line(frame, base, anchor, color, max(1, thickness), cv2.LINE_AA)
        clipped = _clip_infinite_line_to_frame(g["anchor"], g["dir"], w, h)
        if clipped is not None:
            cv2.line(frame, clipped[0], clipped[1], color, thickness, cv2.LINE_AA)
    return frame

def _build_display_guide(
    track,
    total_frames: int,
    max_interp_gap: int,
    frame_w: int,
    frame_h: int
) -> dict:
    """Build a per-frame guide map from chosen track for guide-debug rendering."""
    guide = {}
    if track is None or getattr(track, "num_obs", 0) <= 0:
        return guide
    diag = float(np.hypot(float(frame_w), float(frame_h)))
    max_step = max(10.0, 0.15 * diag)
    static_speed = max(2.0, 0.010 * diag)
    max_resid = max(14.0, 0.08 * diag)

    # De-duplicate by frame (keep highest-confidence observation).
    best_by_frame = {}
    for o in track.observations:
        if o is None:
            continue
        f = int(o.frame)
        prev = best_by_frame.get(f)
        if prev is None or float(getattr(o, "conf", 0.0)) > float(getattr(prev, "conf", 0.0)):
            best_by_frame[f] = o

    obs = [best_by_frame[f] for f in sorted(best_by_frame.keys()) if 0 <= f < total_frames]
    if not obs:
        return guide
    if len(obs) >= 3:
        keep = [obs[0]]
        for i in range(1, len(obs) - 1):
            a = keep[-1]
            b = obs[i]
            c = obs[i + 1]
            dt_ab = max(int(b.frame - a.frame), 1)
            dt_bc = max(int(c.frame - b.frame), 1)
            step_ab = float(np.hypot(float(b.cx - a.cx), float(b.cy - a.cy))) / dt_ab
            step_bc = float(np.hypot(float(c.cx - b.cx), float(c.cy - b.cy))) / dt_bc

            jump_to_static = (
                step_ab > max_step and
                step_bc <= static_speed and
                not bool(getattr(b, "on_motion", False))
            )
            if jump_to_static:
                continue

            dt_ac = int(c.frame - a.frame)
            if dt_ac > 0 and dt_ac <= max(2, int(max_interp_gap)):
                t = (float(b.frame) - float(a.frame)) / float(dt_ac)
                ix = float(a.cx) + (float(c.cx) - float(a.cx)) * t
                iy = float(a.cy) + (float(c.cy) - float(a.cy)) * t
                resid = float(np.hypot(float(b.cx - ix), float(b.cy - iy)))
                shortcut = float(np.hypot(float(c.cx - a.cx), float(c.cy - a.cy))) / dt_ac
                if (step_ab > max_step and step_bc > max_step and
                        resid > max_resid and shortcut < 0.70 * max(step_ab, step_bc)):
                    continue

            keep.append(b)

        last = obs[-1]
        if int(last.frame) != int(keep[-1].frame):
            keep.append(last)
        obs = keep

    for o in obs:
        guide[int(o.frame)] = (float(o.cx), float(o.cy), True)

    max_gap = max(1, int(max_interp_gap))
    for i in range(1, len(obs)):
        a = obs[i - 1]
        b = obs[i]
        gap = int(b.frame) - int(a.frame)
        if gap <= 1 or gap > max_gap:
            continue
        seg_speed = float(np.hypot(float(b.cx - a.cx), float(b.cy - a.cy))) / max(gap, 1)
        if seg_speed > max_step * 1.10:
            continue
        for f in range(int(a.frame) + 1, int(b.frame)):
            if f in guide:
                continue
            t = (f - float(a.frame)) / float(gap)
            cx = float(a.cx) + (float(b.cx) - float(a.cx)) * t
            cy = float(a.cy) + (float(b.cy) - float(a.cy)) * t
            guide[f] = (cx, cy, False)

    # Conservative tail extension: only when the observed tail is moving.
    if len(obs) >= 2:
        last = obs[-1]
        prev = obs[-2]
        dt_last = max(int(last.frame - prev.frame), 1)
        vx = (float(last.cx) - float(prev.cx)) / dt_last
        vy = (float(last.cy) - float(prev.cy)) / dt_last
        tail_speed = float(np.hypot(vx, vy))
        if tail_speed > static_speed * 1.35 and bool(getattr(last, "on_motion", False) or getattr(prev, "on_motion", False)):
            tail_horizon = min(max_gap, max(2, int(max_interp_gap)))
            end_f = min(total_frames - 1, int(last.frame) + tail_horizon)
            for f in range(int(last.frame) + 1, end_f + 1):
                if f in guide:
                    continue
                dtf = f - int(last.frame)
                guide[f] = (float(last.cx) + vx * dtf, float(last.cy) + vy * dtf, False)

    return guide

def _drop_unattached_soft_runs(
    per_frame: List[Optional[FrameResult]],
    cfg: Config,
    frame_w: int,
    frame_h: int
) -> Tuple[int, int]:
    """Drop detached soft runs (carry and short/random motion fragments)."""
    if not per_frame:
        return 0, 0

    diag = float(np.hypot(float(frame_w), float(frame_h)))
    attach_px = max(8.0, float(cfg.carry_attach_max_frac) * diag)
    # Slightly looser than carry to allow orange continuity across short weak spans.
    motion_attach_px = attach_px * 1.55
    removed_carry = 0
    removed_motion = 0
    n = len(per_frame)

    def _dist(a: Optional[FrameResult], b: Optional[FrameResult]) -> float:
        if a is None or b is None or a.cx is None or a.cy is None or b.cx is None or b.cy is None:
            return 1e9
        return float(np.hypot(float(a.cx) - float(b.cx), float(a.cy) - float(b.cy)))

    i = 0
    while i < n:
        r = per_frame[i]
        src = str(r.source) if r is not None else ""
        if src not in ("carry", "motion"):
            i += 1
            continue

        s = i
        while i + 1 < n:
            nxt = per_frame[i + 1]
            if nxt is None or str(nxt.source) != src:
                break
            i += 1
        e = i

        left_i = s - 1
        while left_i >= 0 and per_frame[left_i] is None:
            left_i -= 1
        right_i = e + 1
        while right_i < n and per_frame[right_i] is None:
            right_i += 1

        left_r = per_frame[left_i] if left_i >= 0 else None
        right_r = per_frame[right_i] if right_i < n else None
        left_src = str(left_r.source) if left_r is not None else ""
        right_src = str(right_r.source) if right_r is not None else ""
        start_r = per_frame[s]
        end_r = per_frame[e]
        run_len = e - s + 1

        keep = False
        if src == "carry":
            # Prefer forward attachment, but allow a short terminal tail carry
            # when it cleanly continues from a trusted left anchor.
            if (right_r is not None and right_src not in ("", "carry") and
                    _dist(end_r, right_r) <= attach_px):
                keep = True
            elif right_r is None:
                left_attached = (
                    left_r is not None and
                    left_src in ("det", "motion", "guide", "interp", "carry") and
                    _dist(start_r, left_r) <= attach_px * 1.35
                )
                tail_max_len = max(2, int(getattr(cfg, "guide_interp_max_gap", 12)) + 4)
                keep = left_attached and (run_len <= tail_max_len)
        else:  # motion
            # Motion may bridge det->det or det->carry, but reject detached stubs.
            left_attached = (left_src in ("det", "carry", "motion", "guide", "interp") and
                             _dist(start_r, left_r) <= motion_attach_px)
            right_attached = (right_src in ("det", "carry", "motion", "guide", "interp") and
                              _dist(end_r, right_r) <= motion_attach_px)
            keep = left_attached or right_attached
            # Single-frame motion is noisy: still allow if attached to either trusted side.
            if run_len == 1:
                keep = (
                    (left_src in ("det", "motion", "guide") and
                     _dist(start_r, left_r) <= motion_attach_px) or
                    (right_src in ("det", "motion", "guide") and
                     _dist(end_r, right_r) <= motion_attach_px)
                )

        if not keep:
            for k in range(s, e + 1):
                if per_frame[k] is not None and str(per_frame[k].source) == src:
                    per_frame[k] = None
                    if src == "carry":
                        removed_carry += 1
                    else:
                        removed_motion += 1

        i += 1

    return removed_carry, removed_motion

def _trail_base_color(src):
    return _SOURCE_BASE_COLOR.get(src, (255, 220, 80))

def _trail_smooth_alpha(src):
    # Higher alpha follows the raw point more closely.
    if src == "det":
        return 0.48
    if src == "motion":
        return 0.44
    if src == "interp":
        return 0.38
    if src == "guide":
        return 0.36
    if src == "carry":
        return 0.34
    return 0.40

def _trail_jump_fracs(src, cfg: Config):
    if src == "det":
        return cfg.trail_hard_switch_x_frac, cfg.trail_hard_switch_y_frac
    if src == "motion":
        return min(cfg.trail_hard_switch_x_frac, 0.24), min(cfg.trail_hard_switch_y_frac, 0.24)
    if src == "interp":
        return 0.16, 0.16
    if src == "guide":
        return 0.13, 0.13
    if src == "carry":
        return 0.12, 0.12
    return 0.18, 0.18

def _get_track_color(track_id: int) -> Tuple[int, int, int]:
    """Generate a stable, distinct BGR color from track ID."""
    import hashlib
    h = hashlib.md5(str(track_id).encode()).hexdigest()
    # Use parts of the hash for R, G, B
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    
    # Simple strategy to avoid very dark colors: ensure max component is high
    m = max(r, g, b)
    if m < 100:
        scale = 150.0 / max(m, 1)
        r = min(255, int(r * scale))
        g = min(255, int(g * scale))
        b = min(255, int(b * scale))
        
    return (b, g, r)  # OpenCV uses BGR

def _is_soft_source(src):
    return src in ("interp", "carry", "guide")

def _trail_prev2(trail_list):
    # Return previous and second-previous non-gap trail points.
    prev = None
    prev2 = None
    for j in range(len(trail_list) - 1, -1, -1):
        p = trail_list[j]
        if p is None:
            continue
        if prev is None:
            prev = p
        else:
            prev2 = p
            break
    return prev, prev2

def _trail_direction_break(prev2, prev, curr_xy):
    # Break smoothing segment on strong direction reversal (bounce/hit).
    v1x = prev[0] - prev2[0]
    v1y = prev[1] - prev2[1]
    v2x = curr_xy[0] - prev[0]
    v2y = curr_xy[1] - prev[1]
    m1 = float(np.hypot(v1x, v1y))
    m2 = float(np.hypot(v2x, v2y))
    if m1 < 6.0 or m2 < 6.0:
        return False
    cosang = (v1x * v2x + v1y * v2y) / max(m1 * m2, 1e-6)
    if cosang < -0.32:
        return True
    if np.sign(v1y) != np.sign(v2y) and abs(v1y) > 12.0 and abs(v2y) > 12.0:
        return True
    return False

def _court_axis_spans(kps, w, h):
    # x span uses keypoint 4->7, y span uses keypoint 0->4.
    if kps is None or len(kps) < 16:
        return float(w), float(h)
    try:
        x0, y0 = float(kps[0]), float(kps[1])
        x4, y4 = float(kps[8]), float(kps[9])
        x7 = float(kps[14])
        x_span = abs(x7 - x4)
        y_span = abs(y4 - y0)
        if x_span < 1.0:
            x_span = float(w)
        if y_span < 1.0:
            y_span = float(h)
        x_span = float(np.clip(x_span, 0.35 * w, w))
        y_span = float(np.clip(y_span, 0.25 * h, h))
        return x_span, y_span
    except Exception:
        return float(w), float(h)

def _kpt_xy(kps, idx):
    base = int(idx) * 2
    if kps is None or base + 1 >= len(kps):
        return None
    x = float(kps[base])
    y = float(kps[base + 1])
    if x <= 0 and y <= 0:
        return None
    return x, y

def _homography_apply(H: np.ndarray, x: float, y: float) -> Optional[Tuple[float, float]]:
    """Apply a 3x3 homography to a 2D point."""
    den = float(H[2, 0] * x + H[2, 1] * y + H[2, 2])
    if abs(den) < 1e-8:
        return None
    ox = float(H[0, 0] * x + H[0, 1] * y + H[0, 2]) / den
    oy = float(H[1, 0] * x + H[1, 1] * y + H[1, 2]) / den
    return ox, oy

def _build_ordered_court_polygon(kps, w, h):
    # Explicit corner order: top-left -> bottom-left -> bottom-right -> top-right.
    corner_ids = (0, 4, 7, 3)

    def _poly_from_flat(flat_kps):
        pts = []
        for corner_idx in corner_ids:
            p = _kpt_xy(flat_kps, corner_idx)
            if p is None:
                return None
            pts.append(p)
        poly = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
        if abs(cv2.contourArea(poly)) < 10.0:
            return None
        return poly

    def _poly_score(poly):
        if poly is None:
            return -1e9
        p = poly[:, 0, :]
        tl, bl, br, tr = p[0], p[1], p[2], p[3]
        score = 0.0
        if bl[1] > tl[1]:
            score += 2.0
        if br[1] > tr[1]:
            score += 2.0
        top_w = float(np.linalg.norm(tr - tl))
        bot_w = float(np.linalg.norm(br - bl))
        if bot_w >= top_w:
            score += 1.5
        area = abs(float(cv2.contourArea(poly)))
        score += min(area / max(float(w * h), 1.0), 1.0) * 2.0
        return score

    candidates = []
    poly_raw = _poly_from_flat(kps)
    if poly_raw is not None:
        candidates.append(poly_raw)

    if kps is not None and len(kps) >= 28:
        arr = np.asarray(kps, dtype=np.float32).reshape(-1, 2)
        if arr.shape[0] == 14:
            from .detectors import CourtDetector
            remap = CourtDetector._MODEL_TO_SEMANTIC_14
            remapped = np.zeros_like(arr)
            for model_i, semantic_i in enumerate(remap):
                remapped[semantic_i] = arr[model_i]
            poly_remap = _poly_from_flat(remapped.reshape(-1).tolist())
            if poly_remap is not None:
                candidates.append(poly_remap)

    if not candidates:
        return None
    return max(candidates, key=_poly_score)

def _build_ground_projection_model(kps, w, h):
    """Build court-plane homography from ordered court corners."""
    ordered_poly = _build_ordered_court_polygon(kps, w, h)
    if ordered_poly is None:
        return None
    c = ordered_poly[:, 0, :]
    tl = (float(c[0][0]), float(c[0][1]))
    bl = (float(c[1][0]), float(c[1][1]))
    br = (float(c[2][0]), float(c[2][1]))
    tr = (float(c[3][0]), float(c[3][1]))

    src = np.array([tl, tr, br, bl], dtype=np.float32)  # TL, TR, BR, BL
    dst = np.array(
        [[0.0, 0.0],
         [1.0, 0.0],
         [1.0, 1.0],
         [0.0, 1.0]],
        dtype=np.float32,
    )
    try:
        H_img_to_court = cv2.getPerspectiveTransform(src, dst).astype(np.float64)
        H_court_to_img = cv2.getPerspectiveTransform(dst, src).astype(np.float64)
    except Exception:
        return None

    return {"H_i2c": H_img_to_court, "H_c2i": H_court_to_img}

def _draw_homography_net_line(frame, ground_model):
    """Draw net-bottom line using court-plane homography (v = 0.5)."""
    if frame is None or ground_model is None:
        return
    H = ground_model.get("H_c2i")
    if H is None:
        return
    p_l = _homography_apply(H, 0.0, 0.5)
    p_r = _homography_apply(H, 1.0, 0.5)
    if p_l is None or p_r is None:
        return

    h, w = frame.shape[:2]
    x1 = int(np.clip(round(p_l[0]), 0, w - 1))
    y1 = int(np.clip(round(p_l[1]), 0, h - 1))
    x2 = int(np.clip(round(p_r[0]), 0, w - 1))
    y2 = int(np.clip(round(p_r[1]), 0, h - 1))
    if abs(x2 - x1) + abs(y2 - y1) < 2:
        return

    # Dark outline + bright green line for visibility.
    cv2.line(frame, (x1, y1), (x2, y2), (15, 15, 15), 4, cv2.LINE_AA)
    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA)

def _build_court_polygon(all_court_kps, w, h):
    """Find last valid keypoints and build an ordered court polygon."""
    last_kps = None
    for kps in reversed(all_court_kps):
        if kps and len(kps) >= 8:
            last_kps = kps
            break
    if not last_kps:
        print("[court polygon] No court keypoints found")
        return None, last_kps

    ordered_poly = _build_ordered_court_polygon(last_kps, w, h)
    if ordered_poly is not None:
        court_poly = ordered_poly
        p0 = court_poly[0, 0].tolist()
        p1 = court_poly[1, 0].tolist()
        p2 = court_poly[2, 0].tolist()
        p3 = court_poly[3, 0].tolist()
        print(
            f"[court polygon] Ordered corners [0,4,7,3]: "
            f"TL={p0}, BL={p1}, BR={p2}, TR={p3}"
        )
        bx, by, bw, bh = cv2.boundingRect(court_poly)
        print(f"[court polygon] bbox=({bx},{by},{bw}x{bh})")
        test_cx = bx + bw / 2.0
        test_cy = by + bh / 2.0
        test_dist = cv2.pointPolygonTest(court_poly, (test_cx, test_cy), True)
        print(
            f"[court polygon] Center ({test_cx:.0f},{test_cy:.0f}) dist={test_dist:.1f} "
            "(positive=inside)"
        )
        return court_poly, last_kps

    pts = [
        (last_kps[i], last_kps[i + 1])
        for i in range(0, len(last_kps) - 1, 2)
        if last_kps[i] > 0 or last_kps[i + 1] > 0
    ]
    if len(pts) >= 4:
        court_poly = cv2.convexHull(np.array(pts, dtype=np.float32).reshape(-1, 1, 2))
        bx, by, bw, bh = cv2.boundingRect(court_poly)
        print(f"[court polygon] Fallback hull ({len(pts)} points), bbox=({bx},{by},{bw}x{bh})")
        return court_poly, last_kps

    print(f"[court polygon] Not enough valid points ({len(pts)}), skipping")
    return None, last_kps

def _print_selector_track_summary(all_tracks, chosen_track, total_frames: int, limit: int = 0):
    """Print all selector tracks and clearly indicate which one was chosen."""
    if not all_tracks:
        print("[selector] Track table: none")
        return

    chosen_id = int(chosen_track.track_id) if chosen_track is not None else None
    ordered = sorted(all_tracks, key=lambda t: float(getattr(t, "score", 0.0)), reverse=True)
    if limit and limit > 0:
        view = ordered[:limit]
    else:
        view = ordered

    print("[selector] Track table (sorted by score)")
    print("  sel  id    score   obs  span   t0%   t1%   in%  mot% nearP%")
    for trk in view:
        sb = trk.score_breakdown if getattr(trk, "score_breakdown", None) else {}
        start_frac = float(sb.get("start_frac", 0.0))
        end_frac = float(sb.get("end_frac", 0.0))
        if total_frames > 1 and (start_frac <= 0.0 and end_frac <= 0.0):
            ff = int(getattr(trk, "first_frame", 0))
            lf = int(getattr(trk, "last_obs_frame", ff))
            start_frac = max(0.0, min(1.0, ff / float(total_frames - 1)))
            end_frac = max(0.0, min(1.0, lf / float(total_frames - 1)))

        inside_frac = float(sb.get("inside_strict_frac", sb.get("inside_frac", -1.0)))
        motion_frac = float(sb.get("motion_frac", -1.0))
        near_player_frac = float(sb.get("near_player_frac", -1.0))
        inside_s = f"{inside_frac * 100:5.1f}" if inside_frac >= 0.0 else "   --"
        motion_s = f"{motion_frac * 100:5.1f}" if motion_frac >= 0.0 else "   --"
        near_s = f"{near_player_frac * 100:5.1f}" if near_player_frac >= 0.0 else "   --"

        sel = "*" if int(getattr(trk, "track_id", -1)) == chosen_id else " "
        print(
            f"  {sel:>3} {int(getattr(trk, 'track_id', -1)):>3} "
            f"{float(getattr(trk, 'score', 0.0)):>8.1f} "
            f"{int(getattr(trk, 'num_obs', 0)):>5} "
            f"{int(getattr(trk, 'span', 0)):>5} "
            f"{start_frac * 100:>5.1f} "
            f"{end_frac * 100:>5.1f} "
            f"{inside_s:>5} {motion_s:>5} {near_s:>6}"
        )

    if limit and limit > 0 and len(ordered) > len(view):
        print(f"[selector] Showing top {len(view)} / {len(ordered)} tracks")

    if chosen_track is None:
        print("[selector] Chosen track: none")
    else:
        sb = chosen_track.score_breakdown if getattr(chosen_track, "score_breakdown", None) else {}
        t0 = float(sb.get("start_frac", 0.0))
        t1 = float(sb.get("end_frac", 0.0))
        if total_frames > 1 and (t0 <= 0.0 and t1 <= 0.0):
            ff = int(getattr(chosen_track, "first_frame", 0))
            lf = int(getattr(chosen_track, "last_obs_frame", ff))
            t0 = max(0.0, min(1.0, ff / float(total_frames - 1)))
            t1 = max(0.0, min(1.0, lf / float(total_frames - 1)))
        print(
            f"[selector] Chosen track: id={int(chosen_track.track_id)} "
            f"score={float(chosen_track.score):.1f} "
            f"obs={int(chosen_track.num_obs)} span={int(chosen_track.span)} "
            f"t={t0 * 100:.1f}%->{t1 * 100:.1f}%"
        )

def _print_timing_summary(
    timing: Dict[str, float],
    total_elapsed: float,
    pass1_frames: int,
    pass2_frames: int,
):
    print("\n[info] Stage timing summary")
    rows = [
        ("init_total", "init", 0),
        ("pass1_total", "pass1 total", pass1_frames),
        ("pass1_aux_detect", "pass1 aux detect", pass1_frames),
        ("pass1_preprocess", "pass1 preprocess", pass1_frames),
        ("pass1_pre_mask_build", "pass1 pre(mask-build)", pass1_frames),
        ("pass1_pre_postmask", "pass1 pre(post/flicker)", pass1_frames),
        ("pass1_pre_cuda_total", "pass1 pre(cuda total)", pass1_frames),
        ("pass1_pre_cuda_gpu_other", "pass1 pre(cuda gpu/other)", pass1_frames),
        ("pass1_pre_cuda_d2h", "pass1 pre(cuda d2h)", pass1_frames),
        ("pass1_pre_cuda_cc", "pass1 pre(cuda cc)", pass1_frames),
        ("pass1_ball_detect", "pass1 ball detect", pass1_frames),
        ("pass1_store", "pass1 store/debug", pass1_frames),
        ("pass1_slide", "pass1 decode/slide", pass1_frames),
        ("selector_total", "selector total", pass1_frames),
        ("selector_build_poly", "selector build poly", pass1_frames),
        ("selector_select", "selector select", pass1_frames),
        ("selector_post", "selector post", pass1_frames),
        ("pass2_total", "pass2 total", pass2_frames),
        ("pass2_read", "pass2 frame read", pass2_frames),
        ("pass2_render", "pass2 render", pass2_frames),
        ("pass2_write_main", "pass2 write main", pass2_frames),
        ("pass2_write_guide", "pass2 write guide", pass2_frames),
        ("pass2_write_debug", "pass2 write debug", pass2_frames),
    ]
    denom_total = max(float(total_elapsed), 1e-9)
    for key, label, nframes in rows:
        sec = float(timing.get(key, 0.0))
        if sec <= 0.0:
            continue
        pct = 100.0 * sec / denom_total
        line = f"[info]   {label:<20} {sec:7.3f}s ({pct:5.1f}%)"
        if nframes > 0:
            line += f", {1000.0 * sec / max(int(nframes), 1):6.2f} ms/frame"
        print(line)

