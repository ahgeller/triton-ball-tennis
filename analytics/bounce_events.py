#!/usr/bin/env python3
"""2D bounce/hit event detection + court mapping from tracking JSON.

Key idea: full monocular 3D reconstruction is under-determined, but a bounce
happens ON the court plane (z = 0), so the court homography alone converts the
bounce pixel to metric court coordinates. This sidecar therefore never touches
3D: it finds bounce instants on the selected 2D trajectory (local maxima of
image-y with a real down->up vertical velocity flip), separates them from hits
(player proximity + horizontal reversal / speed change), then maps each bounce
through the homography for court position and in/out calls.

Depends only on numpy + cv2 + the tracking JSON. It does not import the runtime.

Usage:
    python analytics/bounce_events.py --tracking-json out_tracking.json --output-json bounces.json
    python analytics/bounce_events.py --tracking-json out.json --output-json b.json --annotations validation/annotations/clip.json
    python analytics/bounce_events.py --tracking-json out.json --output-json b.json --render-video bounces.mp4

Coordinate convention for court output (matches the 2D homography, not the 3D
sidecar): u = across court 0..1 (left->right doubles sideline, far side on top
of image), v = along court 0..1 (far baseline -> near baseline).  Meters use
the doubles court: 10.97 m x 23.77 m.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

COURT_W_M = 10.97
COURT_L_M = 23.77
SINGLES_INSET_M = 1.372          # doubles alley width per side
SERVICE_FROM_NET_M = 6.40
NET_V = 0.5                      # net at v = 0.5 in normalized court space

# How much to trust a trajectory point by selector source (same spirit as the
# 3D sidecar's SOURCE_SIGMA_PX).
SOURCE_WEIGHT = {"det": 1.0, "motion": 0.75, "guide": 0.4, "carry": 0.3, "interp": 0.35}
SOFT_SOURCES = {"carry", "guide", "interp"}

# Raw court-model keypoint order: corners live at indices (0, 4, 7, 3) =
# TL, BL, BR, TR (mirrors tennis_tracker.rendering._build_ordered_court_polygon).
CORNER_IDS_RAW = (0, 4, 7, 3)
# Fallback if a JSON carries semantically remapped 14-pt keypoints instead.
MODEL_TO_SEMANTIC_14 = (0, 4, 6, 1, 2, 5, 7, 3, 8, 12, 9, 11, 13, 10)
CORNER_IDS_SEMANTIC = (0, 2, 3, 1)  # TL, BL, BR, TR in semantic order


def _import_cv2():
    import cv2
    return cv2


# ----------------------------------------------------------------------------
# Trajectory extraction
# ----------------------------------------------------------------------------

class Point:
    __slots__ = ("frame", "t", "x", "y", "source", "conf", "weight", "player_boxes")

    def __init__(self, frame: int, t: float, x: float, y: float, source: str,
                 conf: float, player_boxes: List[List[float]]):
        self.frame = frame
        self.t = t
        self.x = x
        self.y = y
        self.source = source
        self.conf = conf
        self.weight = SOURCE_WEIGHT.get(source, 0.3)
        self.player_boxes = player_boxes


def _extract_points(tracking: Dict[str, Any], fps: float) -> List[Point]:
    points: List[Point] = []
    for row in tracking.get("frames") or []:
        if not row.get("present"):
            continue
        x, y = row.get("x"), row.get("y")
        if x is None or y is None:
            continue
        frame = int(row["frame"])
        boxes: List[List[float]] = []
        pb = row.get("player_boxes")
        if isinstance(pb, dict):
            boxes = [list(map(float, b[:4])) for b in pb.values()
                     if isinstance(b, (list, tuple)) and len(b) >= 4]
        elif isinstance(pb, (list, tuple)):
            boxes = [list(map(float, b[:4])) for b in pb
                     if isinstance(b, (list, tuple)) and len(b) >= 4]
        points.append(Point(frame, frame / fps, float(x), float(y),
                            str(row.get("source") or ""), float(row.get("conf") or 0.0), boxes))
    points.sort(key=lambda p: p.frame)
    return points


def _split_segments(points: List[Point], max_gap_frames: int, jump_px: float) -> List[List[Point]]:
    segments: List[List[Point]] = []
    current: List[Point] = []
    for p in points:
        if current:
            gap = p.frame - current[-1].frame
            dist = math.hypot(p.x - current[-1].x, p.y - current[-1].y)
            if gap > max_gap_frames or (gap <= 2 and dist > jump_px):
                segments.append(current)
                current = []
        current.append(p)
    if current:
        segments.append(current)
    return [s for s in segments if len(s) >= 7]


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return values.astype(np.float64)
    kernel = np.ones(int(window), dtype=np.float64) / float(window)
    pad = window // 2
    padded = np.pad(values.astype(np.float64), pad, mode="edge")
    out = np.convolve(padded, kernel, mode="same")[pad:pad + len(values)]
    return out


def _linear_velocity(ts: np.ndarray, vals: np.ndarray, weights: np.ndarray) -> Optional[float]:
    """Weighted least-squares slope of vals over ts (px/sec)."""
    if len(ts) < 2:
        return None
    w = np.clip(weights, 0.05, None)
    t_mean = float(np.average(ts, weights=w))
    v_mean = float(np.average(vals, weights=w))
    denom = float(np.sum(w * (ts - t_mean) ** 2))
    if denom <= 1e-12:
        return None
    return float(np.sum(w * (ts - t_mean) * (vals - v_mean)) / denom)


def _near_player(x: float, y: float, boxes: Sequence[Sequence[float]], margin: float) -> bool:
    for box in boxes or []:
        x1, y1, x2, y2 = box[:4]
        if x1 - margin <= x <= x2 + margin and y1 - margin <= y <= y2 + margin:
            return True
    return False


# ----------------------------------------------------------------------------
# Court homography (mirrors runtime corner logic; standalone on purpose)
# ----------------------------------------------------------------------------

def _kp_xy(kps: Sequence[float], idx: int) -> Optional[Tuple[float, float]]:
    i = idx * 2
    if i + 1 >= len(kps):
        return None
    x, y = float(kps[i]), float(kps[i + 1])
    if abs(x) <= 1e-6 and abs(y) <= 1e-6:
        return None
    return x, y


def _poly_from_ids(kps: Sequence[float], ids: Sequence[int]) -> Optional[np.ndarray]:
    pts = []
    for idx in ids:
        p = _kp_xy(kps, idx)
        if p is None:
            return None
        pts.append(p)
    poly = np.asarray(pts, dtype=np.float64)  # TL, BL, BR, TR
    # Shoelace area sanity check
    x, y = poly[:, 0], poly[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    if area < 10.0:
        return None
    return poly


def _poly_plausibility(poly: np.ndarray, width: int, height: int) -> float:
    tl, bl, br, tr = poly
    score = 0.0
    if bl[1] > tl[1]:
        score += 2.0
    if br[1] > tr[1]:
        score += 2.0
    top_w = float(np.linalg.norm(tr - tl))
    bot_w = float(np.linalg.norm(br - bl))
    if bot_w >= top_w:
        score += 1.5
    x, y = poly[:, 0], poly[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    score += min(area / max(float(width * height), 1.0), 1.0) * 2.0
    return score


def build_court_homography(kps: Optional[Sequence[float]], width: int, height: int):
    """Return dict with H_i2c/H_c2i mapping image px <-> normalized court, or None."""
    if not kps or len(kps) < 8:
        return None
    cv2 = _import_cv2()
    candidates: List[np.ndarray] = []
    poly_raw = _poly_from_ids(kps, CORNER_IDS_RAW)
    if poly_raw is not None:
        candidates.append(poly_raw)
    arr = np.asarray(kps, dtype=np.float64)
    if arr.size >= 28:
        pts = arr[:28].reshape(-1, 2)
        remapped = np.zeros_like(pts)
        for model_i, semantic_i in enumerate(MODEL_TO_SEMANTIC_14):
            remapped[semantic_i] = pts[model_i]
        poly_sem = _poly_from_ids(remapped.reshape(-1).tolist(), CORNER_IDS_SEMANTIC)
        if poly_sem is not None:
            candidates.append(poly_sem)
    if not candidates:
        return None
    poly = max(candidates, key=lambda p: _poly_plausibility(p, width, height))
    tl, bl, br, tr = poly
    src = np.asarray([tl, tr, br, bl], dtype=np.float32)
    dst = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    try:
        H_i2c = cv2.getPerspectiveTransform(src, dst).astype(np.float64)
        H_c2i = cv2.getPerspectiveTransform(dst, src).astype(np.float64)
    except Exception:
        return None
    return {"H_i2c": H_i2c, "H_c2i": H_c2i, "corners_px": poly.tolist()}


def _apply_h(H: np.ndarray, x: float, y: float) -> Optional[Tuple[float, float]]:
    den = float(H[2, 0] * x + H[2, 1] * y + H[2, 2])
    if abs(den) < 1e-9:
        return None
    return (
        float(H[0, 0] * x + H[0, 1] * y + H[0, 2]) / den,
        float(H[1, 0] * x + H[1, 1] * y + H[1, 2]) / den,
    )


def court_position(homography, x_px: float, y_px: float) -> Optional[Dict[str, Any]]:
    if homography is None:
        return None
    uv = _apply_h(homography["H_i2c"], x_px, y_px)
    if uv is None:
        return None
    u, v = uv
    x_m = u * COURT_W_M
    y_m = v * COURT_L_M
    inset = SINGLES_INSET_M
    near_side = v > NET_V
    # Signed distances to boundaries (positive = inside)
    d_doubles_x = min(x_m, COURT_W_M - x_m)
    d_singles_x = min(x_m - inset, (COURT_W_M - inset) - x_m)
    d_baseline = min(y_m, COURT_L_M - y_m)
    in_doubles = d_doubles_x >= 0.0 and d_baseline >= 0.0
    in_singles = d_singles_x >= 0.0 and d_baseline >= 0.0
    # Service boxes (relevant half between net and service line)
    service_near_y = NET_V * COURT_L_M + SERVICE_FROM_NET_M
    service_far_y = NET_V * COURT_L_M - SERVICE_FROM_NET_M
    in_service_band = (service_far_y <= y_m <= service_near_y)
    service_box = None
    if in_service_band and in_singles:
        half = "near" if near_side else "far"
        lr = "left" if x_m < COURT_W_M / 2.0 else "right"
        service_box = f"{half}_{lr}"
    nearest_line_m = min(abs(d_doubles_x), abs(d_singles_x), abs(d_baseline))
    return {
        "u": u, "v": v,
        "x_m": x_m, "y_m": y_m,
        "near_side": bool(near_side),
        "in_doubles_court": bool(in_doubles),
        "in_singles_court": bool(in_singles),
        "service_box": service_box,
        "nearest_line_distance_m": float(nearest_line_m),
        "long_m": float(max(0.0, -d_baseline)),
        "wide_singles_m": float(max(0.0, -d_singles_x)),
        "wide_doubles_m": float(max(0.0, -d_doubles_x)),
    }


# ----------------------------------------------------------------------------
# Event detection
# ----------------------------------------------------------------------------

def detect_events(
    points: List[Point],
    fps: float,
    width: int,
    height: int,
    homography=None,
    vel_window_sec: float = 0.13,
    smooth_window: int = 5,
    min_down_speed_frac: float = 0.0009,
    min_up_speed_frac: float = 0.00035,
    player_margin_px: float = 90.0,
    min_separation_sec: float = 0.20,
    max_gap_frames: int = 12,
    jump_px_frac: float = 0.12,
) -> List[Dict[str, Any]]:
    """Find bounce/hit events on the selected 2D trajectory.

    A bounce is a local maximum of (smoothed) image-y where the weighted
    vertical velocity flips from clearly-down to clearly-up, away from any
    player box. Near a player box, direction/speed discontinuities are hits.
    """
    diag = math.hypot(float(width), float(height))
    min_down = min_down_speed_frac * diag * fps    # px/sec
    min_up = min_up_speed_frac * diag * fps
    win = max(2, int(round(vel_window_sec * fps)))  # frames per side
    events: List[Dict[str, Any]] = []

    for seg in _split_segments(points, max_gap_frames, jump_px_frac * diag):
        ts = np.asarray([p.t for p in seg])
        xs = np.asarray([p.x for p in seg])
        ys = np.asarray([p.y for p in seg])
        ws = np.asarray([p.weight * max(p.conf, 0.15) for p in seg])
        ys_s = _smooth(ys, smooth_window)

        for i in range(2, len(seg) - 2):
            # Local maximum of image y (ball at lowest visual point)
            if not (ys_s[i] >= ys_s[i - 1] and ys_s[i] >= ys_s[i + 1]):
                continue
            lo = max(0, i - win)
            hi = min(len(seg), i + win + 1)
            if i - lo < 2 or hi - i - 1 < 2:
                continue
            vy_pre = _linear_velocity(ts[lo:i + 1], ys_s[lo:i + 1], ws[lo:i + 1])
            vy_post = _linear_velocity(ts[i:hi], ys_s[i:hi], ws[i:hi])
            vx_pre = _linear_velocity(ts[lo:i + 1], xs[lo:i + 1], ws[lo:i + 1])
            vx_post = _linear_velocity(ts[i:hi], xs[i:hi], ws[i:hi])
            if vy_pre is None or vy_post is None:
                continue
            if vy_pre < min_down or vy_post > -min_up:
                continue

            p = seg[i]
            near_player = _near_player(p.x, p.y, p.player_boxes, player_margin_px)
            vx_reversed = (
                vx_pre is not None and vx_post is not None
                and vx_pre * vx_post < 0
                and abs(vx_pre) > 0.25 * min_down and abs(vx_post) > 0.25 * min_down
            )
            speed_pre = math.hypot(vx_pre or 0.0, vy_pre)
            speed_post = math.hypot(vx_post or 0.0, vy_post)
            speed_jump = speed_post > 1.8 * speed_pre + min_down

            # Sub-frame apex refinement: parabola through the 3 samples around i.
            frame_f = float(p.frame)
            x_f, y_f = p.x, p.y
            y3 = ys_s[i - 1:i + 2]
            denom = float(y3[0] - 2.0 * y3[1] + y3[2])
            if abs(denom) > 1e-9:
                delta = 0.5 * float(y3[0] - y3[2]) / denom
                delta = max(-1.0, min(1.0, delta))
                frame_f = p.frame + delta * max(1, seg[i + 1].frame - seg[i].frame)
                if delta >= 0:
                    x_f = p.x + delta * (seg[i + 1].x - p.x)
                    y_f = ys_s[i] + delta * (ys_s[i + 1] - ys_s[i])
                else:
                    x_f = p.x + delta * (p.x - seg[i - 1].x)
                    y_f = ys_s[i] + delta * (ys_s[i] - ys_s[i - 1])

            local_sources = {seg[j].source for j in range(max(0, i - 2), min(len(seg), i + 3))}
            soft_only = local_sources <= SOFT_SOURCES
            flip_strength = min(1.0, (vy_pre - vy_post) / max(3.0 * min_down, 1e-6))
            source_support = max(SOURCE_WEIGHT.get(s, 0.3) for s in local_sources)
            confidence = 0.30 + 0.45 * flip_strength + 0.25 * source_support
            if soft_only:
                confidence = min(confidence, 0.45)

            is_hit = near_player and (vx_reversed or speed_jump)
            ev: Dict[str, Any] = {
                "type": "hit" if is_hit else "bounce",
                "frame": int(p.frame),
                "frame_subpixel": float(frame_f),
                "time_sec": float(frame_f / fps),
                "x_px": float(x_f),
                "y_px": float(y_f),
                "confidence": float(max(0.05, min(0.95, confidence))),
                "near_player": bool(near_player),
                "soft_sources_only": bool(soft_only),
                "metrics": {
                    "vy_pre_px_s": float(vy_pre),
                    "vy_post_px_s": float(vy_post),
                    "vx_pre_px_s": None if vx_pre is None else float(vx_pre),
                    "vx_post_px_s": None if vx_post is None else float(vx_post),
                    "vx_reversed": bool(vx_reversed),
                    "speed_jump": bool(speed_jump),
                },
            }
            if not is_hit:
                ev["court"] = court_position(homography, float(x_f), float(y_f))
            events.append(ev)

    # Non-max suppression in time: keep the highest-confidence event per window.
    events.sort(key=lambda e: e["time_sec"])
    kept: List[Dict[str, Any]] = []
    for ev in sorted(events, key=lambda e: -e["confidence"]):
        if all(abs(ev["time_sec"] - k["time_sec"]) >= min_separation_sec for k in kept):
            kept.append(ev)
    kept.sort(key=lambda e: e["frame"])
    return kept


# ----------------------------------------------------------------------------
# Scoring against hand-labeled events
# ----------------------------------------------------------------------------

def score_against_annotations(
    events: List[Dict[str, Any]],
    annotations: Dict[str, Any],
    tolerance_frames: int = 3,
) -> Dict[str, Any]:
    labeled = [e for e in (annotations.get("events") or [])
               if str(e.get("type", "")).startswith("bounce") or e.get("type") == "hit"]
    out: Dict[str, Any] = {"tolerance_frames": tolerance_frames, "by_type": {}}
    for ev_type in ("bounce", "hit"):
        truth = sorted(int(e["frame"]) for e in labeled if str(e["type"]).startswith(ev_type))
        preds = sorted(int(e["frame"]) for e in events if e["type"] == ev_type)
        matched_truth = set()
        matched_pred = set()
        for t in truth:
            best = None
            for p in preds:
                if p in matched_pred or abs(p - t) > tolerance_frames:
                    continue
                if best is None or abs(p - t) < abs(best - t):
                    best = p
            if best is not None:
                matched_truth.add(t)
                matched_pred.add(best)
        tp = len(matched_truth)
        out["by_type"][ev_type] = {
            "labeled": len(truth),
            "predicted": len(preds),
            "matched": tp,
            "recall": tp / max(len(truth), 1),
            "precision": tp / max(len(preds), 1),
            "missed_frames": [t for t in truth if t not in matched_truth],
            "false_positive_frames": [p for p in preds if p not in matched_pred],
        }
    return out


# ----------------------------------------------------------------------------
# Optional overlay rendering
# ----------------------------------------------------------------------------

def render_overlay(
    tracking: Dict[str, Any],
    events: List[Dict[str, Any]],
    homography,
    input_video: str,
    output_video: str,
    linger_frames: int = 45,
) -> None:
    cv2 = _import_cv2()
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {input_video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    by_frame: Dict[int, List[Dict[str, Any]]] = {}
    for ev in events:
        by_frame.setdefault(int(ev["frame"]), []).append(ev)
    active: List[Tuple[int, Dict[str, Any]]] = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for ev in by_frame.get(frame_idx, []):
            active.append((frame_idx, ev))
        active = [(f0, ev) for f0, ev in active if frame_idx - f0 <= linger_frames]
        for f0, ev in active:
            x, y = int(round(ev["x_px"])), int(round(ev["y_px"]))
            is_bounce = ev["type"] == "bounce"
            color = (0, 200, 255) if is_bounce else (255, 120, 0)
            cv2.circle(frame, (x, y), 12, (10, 10, 10), 4, cv2.LINE_AA)
            cv2.circle(frame, (x, y), 12, color, 2, cv2.LINE_AA)
            label = ev["type"].upper()
            court = ev.get("court")
            if is_bounce and court:
                verdict = "IN" if court["in_singles_court"] else (
                    "IN(dbl)" if court["in_doubles_court"] else "OUT")
                label += f" {verdict}"
            cv2.putText(frame, label, (x + 16, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 10, 10), 3, cv2.LINE_AA)
            cv2.putText(frame, label, (x + 16, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
        writer.write(frame)
        frame_idx += 1
    cap.release()
    writer.release()


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def run(
    tracking_json: str,
    output_json: str,
    annotations_json: Optional[str] = None,
    render_video: Optional[str] = None,
    input_video: Optional[str] = None,
    tolerance_frames: int = 3,
) -> Dict[str, Any]:
    with Path(tracking_json).open("r", encoding="utf-8") as f:
        tracking = json.load(f)
    video = tracking.get("video") or {}
    fps = float(video.get("fps") or 30.0)
    width = int(video.get("width") or 1920)
    height = int(video.get("height") or 1080)

    homography = build_court_homography(
        tracking.get("last_valid_court_keypoints"), width, height)
    points = _extract_points(tracking, fps)
    events = detect_events(points, fps, width, height, homography=homography)

    result: Dict[str, Any] = {
        "schema_version": "bounce_events_v1",
        "video": video,
        "homography_available": homography is not None,
        "court_corners_px": None if homography is None else homography["corners_px"],
        "summary": {
            "trajectory_points": len(points),
            "bounce_count": sum(1 for e in events if e["type"] == "bounce"),
            "hit_count": sum(1 for e in events if e["type"] == "hit"),
            "bounces_in_singles": sum(
                1 for e in events
                if e["type"] == "bounce" and (e.get("court") or {}).get("in_singles_court")),
            "bounces_out": sum(
                1 for e in events
                if e["type"] == "bounce" and e.get("court")
                and not e["court"]["in_doubles_court"]),
        },
        "events": events,
    }

    if annotations_json:
        with Path(annotations_json).open("r", encoding="utf-8") as f:
            result["annotation_score"] = score_against_annotations(
                events, json.load(f), tolerance_frames)

    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    if render_video:
        src = input_video or video.get("input")
        if not src or not Path(src).exists():
            print(f"[bounce_events] render skipped: input video not found ({src})")
        else:
            render_overlay(tracking, events, homography, str(src), render_video)
            print(f"[bounce_events] overlay video: {render_video}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Detect bounce/hit events from tracking JSON and map bounces to court coordinates.")
    parser.add_argument("--tracking-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--annotations", default=None,
                        help="Optional annotation JSON with an 'events' array for precision/recall scoring.")
    parser.add_argument("--tolerance-frames", type=int, default=3)
    parser.add_argument("--render-video", default=None, help="Optional overlay MP4 path.")
    parser.add_argument("--input-video", default=None,
                        help="Video override for --render-video; defaults to tracking JSON video.input.")
    args = parser.parse_args()

    result = run(
        args.tracking_json,
        args.output_json,
        annotations_json=args.annotations,
        render_video=args.render_video,
        input_video=args.input_video,
        tolerance_frames=args.tolerance_frames,
    )
    s = result["summary"]
    print(f"[bounce_events] points={s['trajectory_points']} bounces={s['bounce_count']} "
          f"hits={s['hit_count']} in_singles={s['bounces_in_singles']} out={s['bounces_out']}")
    if "annotation_score" in result:
        for ev_type, sc in result["annotation_score"]["by_type"].items():
            print(f"[bounce_events][score] {ev_type}: recall={sc['recall']:.2f} "
                  f"precision={sc['precision']:.2f} ({sc['matched']}/{sc['labeled']} matched)")
    print(f"[bounce_events] output: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
