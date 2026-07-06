#!/usr/bin/env python3
"""Standalone motion-candidate benchmark for tennis-ball tracking.

This script intentionally sits outside the production pipeline. It compares
cheap motion candidate generators on the same video frames, optionally scoring
their blobs against hand annotations or an existing tracking JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import cv2
    import numpy as np
except ModuleNotFoundError:
    cv2 = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]


ALL_METHODS = ("adaptive_sv", "delta3")
METHOD_COLORS = {
    "adaptive_sv": (0, 200, 255),
    "delta3": (0, 255, 120),
}
METHOD_LABELS = {
    "adaptive_sv": "adaptive_sv=orange",
    "delta3": "delta3=green",
}


def _require_cv_runtime() -> None:
    if cv2 is None or np is None:
        raise SystemExit(
            "OpenCV/NumPy runtime is unavailable in this Python environment. "
            "Activate the project environment first, for example: conda activate tennis-tracker"
        )


@dataclass
class ReferencePoint:
    frame: int
    x: float
    y: float
    kind: str


@dataclass
class Candidate:
    frame: int
    method: str
    x: float
    y: float
    bbox: Tuple[int, int, int, int]
    area: float
    aspect_ratio: float
    fill_ratio: float
    compactness: float
    color_score: float
    score: float
    distance_to_reference: Optional[float] = None
    nearest_player_distance: Optional[float] = None


@dataclass
class FusedCandidate:
    frame: int
    x: float
    y: float
    bbox: Tuple[int, int, int, int]
    methods: List[str]
    member_count: int
    support_count: int
    score: float
    mask_support_score: float
    candidate_quality_score: float
    kalman_gate_score: float
    velocity_score: float
    player_penalty: float
    distance_to_prediction: Optional[float] = None
    gate_radius: Optional[float] = None
    velocity_cosine: Optional[float] = None
    speed_ratio: Optional[float] = None
    distance_to_reference: Optional[float] = None
    nearest_player_distance: Optional[float] = None
    accepted: bool = False


class AdaptiveSVState:
    """Small CPU version of the repo's S+V adaptive background idea."""

    def __init__(
        self,
        thresh: float,
        k_std: float,
        v_min: float,
        alpha: float,
        motion_alpha: float,
    ) -> None:
        self.thresh = float(thresh)
        self.k_std = float(k_std)
        self.v_min = float(v_min)
        self.alpha = float(alpha)
        self.motion_alpha = float(motion_alpha)
        self.bg_v: Optional[np.ndarray] = None
        self.bg_s: Optional[np.ndarray] = None
        self.var_v: Optional[np.ndarray] = None
        self.var_s: Optional[np.ndarray] = None

    def apply(self, frame_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2].astype(np.float32)
        s = hsv[:, :, 1].astype(np.float32)

        if self.bg_v is None:
            self.bg_v = v.copy()
            self.bg_s = s.copy()
            init_var = max(self.thresh, 1.0) ** 2
            self.var_v = np.full_like(v, init_var, dtype=np.float32)
            self.var_s = np.full_like(s, init_var, dtype=np.float32)
            return np.zeros(v.shape, dtype=np.uint8)

        assert self.bg_s is not None and self.var_v is not None and self.var_s is not None
        dv = (v - self.bg_v) ** 2
        ds = (s - self.bg_s) ** 2
        k2 = self.k_std * self.k_std
        thresh2 = self.thresh * self.thresh
        v_thresh = thresh2 + self.var_v * k2
        s_thresh = (thresh2 + self.var_s * k2) * 1.5
        raw = ((dv > v_thresh) | (ds > s_thresh)) & (v > self.v_min)
        mask = raw.astype(np.uint8) * 255

        alpha = np.full_like(v, self.alpha, dtype=np.float32)
        alpha[raw] = self.motion_alpha
        inv_alpha = 1.0 - alpha
        self.var_v = self.var_v * inv_alpha + dv * alpha
        self.var_s = self.var_s * inv_alpha + ds * alpha
        self.bg_v = self.bg_v * inv_alpha + v * alpha
        self.bg_s = self.bg_s * inv_alpha + s * alpha
        return mask


class SimpleFusionState:
    """Small constant-velocity state used only by this experiment."""

    def __init__(self) -> None:
        self.initialized = False
        self.x = 0.0
        self.y = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.last_frame = -1
        self.uncertainty = 80.0
        self.misses = 0

    def seed(self, x: float, y: float, frame_idx: int) -> None:
        self.initialized = True
        self.x = float(x)
        self.y = float(y)
        self.vx = 0.0
        self.vy = 0.0
        self.last_frame = int(frame_idx)
        self.uncertainty = 60.0
        self.misses = 0

    def predict(self, frame_idx: int, base_gate: float) -> Dict[str, Any]:
        if not self.initialized:
            return {
                "active": False,
                "x": None,
                "y": None,
                "vx": 0.0,
                "vy": 0.0,
                "speed": 0.0,
                "gate_radius": float(base_gate),
            }
        dt = max(1, int(frame_idx) - int(self.last_frame))
        px = self.x + self.vx * dt
        py = self.y + self.vy * dt
        speed = math.hypot(self.vx, self.vy)
        gate = max(float(base_gate), min(260.0, self.uncertainty + speed * dt * 0.35))
        return {
            "active": True,
            "x": float(px),
            "y": float(py),
            "vx": float(self.vx),
            "vy": float(self.vy),
            "speed": float(speed),
            "gate_radius": float(gate),
        }

    def update(self, x: float, y: float, frame_idx: int, alpha: float = 0.62) -> None:
        if not self.initialized:
            self.seed(x, y, frame_idx)
            return
        dt = max(1, int(frame_idx) - int(self.last_frame))
        px = self.x + self.vx * dt
        py = self.y + self.vy * dt
        obs_vx = (float(x) - self.x) / float(dt)
        obs_vy = (float(y) - self.y) / float(dt)
        a = max(0.0, min(1.0, float(alpha)))
        self.x = a * float(x) + (1.0 - a) * px
        self.y = a * float(y) + (1.0 - a) * py
        self.vx = a * obs_vx + (1.0 - a) * self.vx
        self.vy = a * obs_vy + (1.0 - a) * self.vy
        residual = math.hypot(float(x) - px, float(y) - py)
        self.uncertainty = max(28.0, min(180.0, 0.55 * self.uncertainty + 0.45 * residual))
        self.last_frame = int(frame_idx)
        self.misses = 0

    def miss(self) -> None:
        if not self.initialized:
            return
        self.misses += 1
        self.uncertainty = min(260.0, self.uncertainty * 1.18 + 8.0)


def _positive_int(value: str) -> int:
    out = int(value)
    if out < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return out


def _parse_methods(value: str) -> List[str]:
    requested = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not requested or "all" in requested:
        return list(ALL_METHODS)
    unknown = sorted(set(requested) - set(ALL_METHODS))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown method(s): {', '.join(unknown)}. Use any of: {', '.join(ALL_METHODS)}, all"
        )
    return requested


def _parse_float_list(value: str) -> List[float]:
    out: List[float] = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        out.append(float(item))
    if not out:
        raise argparse.ArgumentTypeError("expected at least one number")
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark tennis-ball motion candidate generators outside the main pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Input video path.")
    parser.add_argument("--output-json", default="output_videos/motion_speed_probe.json",
                        help="Summary/report JSON path.")
    parser.add_argument("--output-csv", default=None,
                        help="Optional flat candidate CSV path.")
    parser.add_argument("--debug-video", default=None,
                        help="Optional overlay MP4 showing top candidates.")
    parser.add_argument("--debug-view", default="diagnostic",
                        choices=["diagnostic", "motion"],
                        help="Debug-video style. 'motion' draws only raw method masks, no tracking or candidates.")
    parser.add_argument("--methods", type=_parse_methods, default=list(ALL_METHODS),
                        help="Comma-separated methods: adaptive_sv, delta3, all.")
    parser.add_argument("--start-frame", type=_positive_int, default=0,
                        help="First frame to process.")
    parser.add_argument("--max-frames", type=int, default=300,
                        help="Maximum frames to process; <=0 means until EOF.")
    parser.add_argument("--downscale", type=float, default=1.0,
                        help="Process at this scale, report coordinates in original pixels.")
    parser.add_argument("--tracking-json", default=None,
                        help="Optional production tracking JSON for reference points/player boxes.")
    parser.add_argument("--annotations", default=None,
                        help="Optional hand annotation JSON. Takes priority over tracking JSON for scoring.")
    parser.add_argument("--hit-radius", type=float, default=24.0,
                        help="Candidate/reference distance counted as a hit, in original pixels.")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Candidates retained per method/frame in JSON and overlay.")
    parser.add_argument("--skip-candidates", action="store_true",
                        help="Skip contour candidate extraction; useful for pure motion-mask video inspection.")
    parser.add_argument("--min-area", type=float, default=6.0,
                        help="Minimum blob area in original pixels squared.")
    parser.add_argument("--max-area", type=float, default=2500.0,
                        help="Maximum blob area in original pixels squared.")
    parser.add_argument("--max-dim", type=float, default=120.0,
                        help="Maximum blob width/height in original pixels; <=0 disables.")
    parser.add_argument("--max-aspect", type=float, default=5.0,
                        help="Maximum blob aspect ratio.")
    parser.add_argument("--min-fill", type=float, default=0.10,
                        help="Minimum area / bounding-rectangle fill ratio.")
    parser.add_argument("--ideal-area", type=float, default=180.0,
                        help="Soft scoring target area in original pixels squared.")
    parser.add_argument("--open-size", type=int, default=3,
                        help="Morphological opening kernel size; <=1 disables.")
    parser.add_argument("--close-size", type=int, default=0,
                        help="Morphological closing kernel size; <=1 disables.")
    parser.add_argument("--min-component-area", type=float, default=18.0,
                        help="Drop motion-mask components smaller than this original-pixel area; <=0 disables.")
    parser.add_argument("--flicker-clean-frame-frac", type=float, default=0.0,
                        help="When a mask covers at least this frame fraction, remove small scattered components; <=0 disables.")
    parser.add_argument("--flicker-clean-component-count", type=int, default=0,
                        help="Also trigger flicker cleanup when a mask has more components than this; <=0 disables.")
    parser.add_argument("--flicker-clean-component-area", type=float, default=160.0,
                        help="Original-pixel component area retained during flicker cleanup.")
    parser.add_argument("--flicker-suppress-frame-frac", type=float, default=0.0,
                        help="Zero a method mask when it still covers at least this frame fraction after cleanup; <=0 disables.")
    parser.add_argument("--mask-keep-largest-components", type=int, default=0,
                        help="Keep only this many largest connected mask components per method; <=0 disables.")
    parser.add_argument("--prefilter", default="gaussian",
                        choices=["none", "gaussian", "median"],
                        help="Denoise only the motion-analysis frame copy before differencing/background updates.")
    parser.add_argument("--prefilter-ksize", type=int, default=3,
                        help="Odd kernel size for gaussian/median prefilter.")
    parser.add_argument("--prefilter-sigma", type=float, default=0.0,
                        help="Gaussian sigma; 0 lets OpenCV infer from kernel.")
    parser.add_argument("--diff-thresh", type=float, default=18.0,
                        help="Threshold for delta3 frame differencing.")
    parser.add_argument("--diff-lo-frac", type=float, default=0.55,
                        help="Low threshold fraction for soft delta3 support.")
    parser.add_argument("--diff-very-hi", type=float, default=36.0,
                        help="Very high one-sided threshold for delta3 support.")
    parser.add_argument("--motion-v-min", type=float, default=40.0,
                        help="Minimum V brightness for delta/adaptive motion.")
    parser.add_argument("--adaptive-thresh", type=float, default=11.0,
                        help="Adaptive S+V additive threshold.")
    parser.add_argument("--adaptive-k-std", type=float, default=3.0,
                        help="Adaptive S+V variance multiplier.")
    parser.add_argument("--adaptive-alpha", type=float, default=0.02,
                        help="Adaptive background update rate at static pixels.")
    parser.add_argument("--adaptive-motion-alpha", type=float, default=0.015,
                        help="Adaptive background update rate at motion pixels.")
    parser.add_argument("--player-margin", type=float, default=30.0,
                        help="Distance in pixels counted as near a player box for diagnostics.")
    parser.add_argument("--fusion", action="store_true",
                        help="Cluster candidates across methods and score them with a Kalman-style motion gate.")
    parser.add_argument("--fusion-cluster-radius", type=float, default=64.0,
                        help="Maximum center distance for merging method candidates into one fused cluster.")
    parser.add_argument("--fusion-top-k", type=int, default=8,
                        help="Fused candidates retained per frame in JSON and overlay.")
    parser.add_argument("--fusion-gate-radius", type=float, default=90.0,
                        help="Base Kalman/prediction gate radius in original pixels.")
    parser.add_argument("--fusion-min-score", type=float, default=18.0,
                        help="Minimum fused score needed to update the experiment tracker.")
    parser.add_argument("--fusion-seed-reference", action="store_true",
                        help="Seed/reseed the experiment tracker from annotation/tracking JSON when available.")
    parser.add_argument("--fusion-guide-reference", action="store_true",
                        help="Use annotation/tracking JSON as the per-frame prediction guide when available.")
    parser.add_argument("--fusion-seed-after-misses", type=int, default=12,
                        help="When reference seeding is enabled, reseed after this many missed fusion frames.")
    parser.add_argument("--fusion-hit-radii", type=_parse_float_list, default=[24.0, 50.0, 80.0],
                        help="Comma-separated radii used to summarize fused ROI usefulness.")
    parser.add_argument("--mask-overlay-alpha", type=float, default=0.18,
                        help="Debug-video alpha for colored motion-mask pixels; 0 disables mask overlay.")
    parser.add_argument("--motion-background", default="black",
                        choices=["black", "translucent"],
                        help="Background for --debug-view motion.")
    parser.add_argument("--motion-alpha", type=float, default=0.95,
                        help="Motion color strength for translucent motion view.")
    parser.add_argument("--motion-base-alpha", type=float, default=0.25,
                        help="Original-frame brightness retained for translucent motion view.")
    parser.add_argument("--motion-visual-dilate", type=int, default=2,
                        help="Visualization-only dilation iterations so tiny fast motion is easy to see.")
    parser.add_argument("--no-progress", action="store_true",
                        help="Suppress progress prints.")
    return parser


def _load_json(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _annotation_rows(payload: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    rows = payload.get("ball", payload.get("frames", []))
    if isinstance(rows, dict):
        for frame_s, value in rows.items():
            row = dict(value or {})
            row["frame"] = int(frame_s)
            yield row
    elif isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict):
                yield row


def _is_ignored(frame: int, ranges: Sequence[Dict[str, Any]]) -> bool:
    for item in ranges:
        start = int(item.get("start", item.get("from", -1)))
        end = int(item.get("end", item.get("to", start)))
        if start <= frame <= end:
            return True
    return False


def _load_references(
    tracking_payload: Optional[Dict[str, Any]],
    annotation_payload: Optional[Dict[str, Any]],
) -> Tuple[Dict[int, ReferencePoint], str]:
    refs: Dict[int, ReferencePoint] = {}
    if annotation_payload:
        ignore_ranges = annotation_payload.get("ignore_ranges", [])
        for row in _annotation_rows(annotation_payload):
            frame = int(row.get("frame", -1))
            if frame < 0 or _is_ignored(frame, ignore_ranges):
                continue
            visible = bool(row.get("visible", True))
            if not visible or row.get("x") is None or row.get("y") is None:
                continue
            refs[frame] = ReferencePoint(
                frame=frame,
                x=float(row["x"]),
                y=float(row["y"]),
                kind="annotation",
            )
        return refs, "annotations"

    if tracking_payload:
        frames = tracking_payload.get("frames", [])
        if isinstance(frames, list):
            iterable = enumerate(frames)
        elif isinstance(frames, dict):
            iterable = ((int(k), v) for k, v in frames.items())
        else:
            iterable = []
        for idx, row in iterable:
            if not isinstance(row, dict):
                continue
            frame = int(row.get("frame", idx))
            if not bool(row.get("present", False)):
                continue
            if row.get("x") is None or row.get("y") is None:
                continue
            refs[frame] = ReferencePoint(
                frame=frame,
                x=float(row["x"]),
                y=float(row["y"]),
                kind="tracking_json",
            )
        return refs, "tracking_json"

    return refs, "none"


def _numeric_box(value: Any) -> Optional[Tuple[float, float, float, float]]:
    if isinstance(value, dict):
        for keys in (("x1", "y1", "x2", "y2"), ("left", "top", "right", "bottom")):
            if all(k in value for k in keys):
                return tuple(float(value[k]) for k in keys)  # type: ignore[return-value]
        value = value.get("bbox", value.get("box"))
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        try:
            return float(value[0]), float(value[1]), float(value[2]), float(value[3])
        except (TypeError, ValueError):
            return None
    return None


def _iter_player_boxes(value: Any) -> Iterable[Tuple[float, float, float, float]]:
    box = _numeric_box(value)
    if box is not None:
        yield box
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from _iter_player_boxes(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_player_boxes(item)


def _load_player_boxes_by_frame(
    tracking_payload: Optional[Dict[str, Any]]
) -> Dict[int, List[Tuple[float, float, float, float]]]:
    out: Dict[int, List[Tuple[float, float, float, float]]] = {}
    if not tracking_payload:
        return out
    frames = tracking_payload.get("frames", [])
    if isinstance(frames, list):
        iterable = enumerate(frames)
    elif isinstance(frames, dict):
        iterable = ((int(k), v) for k, v in frames.items())
    else:
        iterable = []
    for idx, row in iterable:
        if not isinstance(row, dict):
            continue
        frame = int(row.get("frame", idx))
        boxes = list(_iter_player_boxes(row.get("player_boxes", [])))
        if boxes:
            out[frame] = boxes
    return out


def _nearest_player_distance(
    x: float,
    y: float,
    boxes: Sequence[Tuple[float, float, float, float]],
) -> Optional[float]:
    best: Optional[float] = None
    for x1, y1, x2, y2 in boxes:
        dx = max(float(x1) - x, 0.0, x - float(x2))
        dy = max(float(y1) - y, 0.0, y - float(y2))
        dist = math.hypot(dx, dy)
        if best is None or dist < best:
            best = dist
    return best


def _resize_frame(frame: Optional[np.ndarray], scale: float) -> Optional[np.ndarray]:
    if frame is None:
        return None
    if abs(scale - 1.0) < 1e-9:
        return frame
    h, w = frame.shape[:2]
    new_w = max(2, int(round(w * scale)))
    new_h = max(2, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(frame, (new_w, new_h), interpolation=interp)


def _odd_kernel(value: int, minimum: int = 3) -> int:
    k = max(int(value), int(minimum))
    if k % 2 == 0:
        k += 1
    return k


def _prefilter_frame(frame: Optional[np.ndarray], args: argparse.Namespace) -> Optional[np.ndarray]:
    if frame is None:
        return None
    mode = str(getattr(args, "prefilter", "none") or "none").lower()
    if mode == "none":
        return frame
    if mode == "gaussian":
        k = _odd_kernel(int(getattr(args, "prefilter_ksize", 3)), minimum=3)
        return cv2.GaussianBlur(frame, (k, k), float(getattr(args, "prefilter_sigma", 0.0)))
    if mode == "median":
        k = _odd_kernel(int(getattr(args, "prefilter_ksize", 3)), minimum=3)
        return cv2.medianBlur(frame, k)
    return frame


def _remove_small_components(mask: np.ndarray, min_area: float) -> np.ndarray:
    if min_area <= 0.0 or mask is None or cv2.countNonZero(mask) <= 0:
        return mask
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return mask
    keep = np.zeros(num, dtype=np.uint8)
    for idx in range(1, num):
        if float(stats[idx, cv2.CC_STAT_AREA]) >= float(min_area):
            keep[idx] = 255
    return keep[labels]


def _clean_mask(
    mask: Optional[np.ndarray],
    open_size: int,
    close_size: int,
    min_component_area: float = 0.0,
) -> Optional[np.ndarray]:
    if mask is None:
        return None
    out = (mask > 0).astype(np.uint8) * 255
    out = _remove_small_components(out, float(min_component_area))
    did_morph = False
    if open_size > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k)
        did_morph = True
    if close_size > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k)
        did_morph = True
    if did_morph:
        out = _remove_small_components(out, float(min_component_area))
    return out


def _apply_flicker_guard(
    mask: Optional[np.ndarray],
    args: argparse.Namespace,
    scale_x: float,
    scale_y: float,
) -> Optional[np.ndarray]:
    if mask is None or cv2.countNonZero(mask) <= 0:
        return mask
    total_px = max(int(mask.shape[0] * mask.shape[1]), 1)
    clean_frac = max(0.0, float(getattr(args, "flicker_clean_frame_frac", 0.0)))
    suppress_frac = max(0.0, float(getattr(args, "flicker_suppress_frame_frac", 0.0)))
    max_components = max(0, int(getattr(args, "flicker_clean_component_count", 0)))

    mask_pixels = int(cv2.countNonZero(mask))
    frame_frac = float(mask_pixels / total_px)
    if suppress_frac > 0.0 and frame_frac >= suppress_frac:
        return np.zeros_like(mask)

    should_check_components = max_components > 0 or (clean_frac > 0.0 and frame_frac >= clean_frac)
    if not should_check_components:
        return mask

    num, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8) * 255, connectivity=8)
    component_count = max(0, int(num) - 1)
    should_clean = (clean_frac > 0.0 and frame_frac >= clean_frac) or (
        max_components > 0 and component_count > max_components
    )
    if not should_clean:
        return mask

    min_area_orig = max(0.0, float(getattr(args, "flicker_clean_component_area", 160.0)))
    min_area_proc = min_area_orig / max(scale_x * scale_y, 1e-9)
    keep = np.zeros(num, dtype=np.uint8)
    for idx in range(1, num):
        if float(stats[idx, cv2.CC_STAT_AREA]) >= min_area_proc:
            keep[idx] = 255
    cleaned = keep[labels]

    if suppress_frac > 0.0:
        cleaned_frac = float(cv2.countNonZero(cleaned) / total_px)
        if cleaned_frac >= suppress_frac:
            return np.zeros_like(mask)
    return cleaned


def _keep_largest_components(mask: Optional[np.ndarray], keep_count: int) -> Optional[np.ndarray]:
    keep_count = int(keep_count)
    if keep_count <= 0 or mask is None or cv2.countNonZero(mask) <= 0:
        return mask
    num, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8) * 255, connectivity=8)
    if num <= keep_count + 1:
        return mask
    component_ids = list(range(1, num))
    component_ids.sort(key=lambda idx: int(stats[idx, cv2.CC_STAT_AREA]), reverse=True)
    keep = np.zeros(num, dtype=np.uint8)
    for idx in component_ids[:keep_count]:
        keep[idx] = 255
    return keep[labels]


def _delta_strength(prev_bgr: np.ndarray, curr_bgr: np.ndarray) -> np.ndarray:
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


def _delta2_mask(
    prev_bgr: Optional[np.ndarray],
    curr_bgr: np.ndarray,
    thresh: float,
    v_min: float,
) -> np.ndarray:
    if prev_bgr is None:
        return np.zeros(curr_bgr.shape[:2], dtype=np.uint8)
    strength = _delta_strength(prev_bgr, curr_bgr)
    bright = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2HSV)[:, :, 2] > float(v_min)
    return ((strength >= float(thresh)) & bright).astype(np.uint8) * 255


def _delta3_mask(
    prev_bgr: Optional[np.ndarray],
    curr_bgr: np.ndarray,
    next_bgr: Optional[np.ndarray],
    thresh: float,
    lo_frac: float,
    very_hi: float,
    v_min: float,
) -> np.ndarray:
    if prev_bgr is None or next_bgr is None:
        return np.zeros(curr_bgr.shape[:2], dtype=np.uint8)
    d_prev = _delta_strength(prev_bgr, curr_bgr)
    d_next = _delta_strength(curr_bgr, next_bgr)
    hi = float(thresh)
    lo = max(1.0, hi * max(0.05, min(1.0, float(lo_frac))))
    bright = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2HSV)[:, :, 2] > float(v_min)
    mask = (
        ((d_prev >= hi) & (d_next >= lo)) |
        ((d_next >= hi) & (d_prev >= lo)) |
        (d_prev >= float(very_hi)) |
        (d_next >= float(very_hi))
    ) & bright
    return mask.astype(np.uint8) * 255


def _foreground_mask(back_sub: Any, frame_bgr: np.ndarray, learning_rate: float) -> np.ndarray:
    fg = back_sub.apply(frame_bgr, learningRate=float(learning_rate))
    # If shadow detection is enabled, OpenCV marks shadows as 127. Keep only sure foreground.
    return (fg > 200).astype(np.uint8) * 255


def _color_score(frame_bgr: np.ndarray, contour: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return 0.0
    crop = frame_bgr[y:y + h, x:x + w]
    if crop.size == 0:
        return 0.0
    local = contour.copy()
    local[:, :, 0] -= x
    local[:, :, 1] -= y
    comp_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(comp_mask, [local], -1, 255, -1)
    denom = int(cv2.countNonZero(comp_mask))
    if denom <= 0:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    support = (
        (hsv[:, :, 0].astype(np.float32) >= 18.0) &
        (hsv[:, :, 0].astype(np.float32) <= 75.0) &
        (hsv[:, :, 1].astype(np.float32) >= 0.12 * 255.0) &
        (hsv[:, :, 2].astype(np.float32) >= 0.18 * 255.0)
    ).astype(np.uint8) * 255
    return float(cv2.countNonZero(cv2.bitwise_and(support, comp_mask)) / max(denom, 1))


def _candidate_score(
    area: float,
    aspect: float,
    fill: float,
    compactness: float,
    color_score: float,
    ideal_area: float,
    min_area: float,
    max_area: float,
) -> float:
    area = max(float(area), 1.0)
    ideal = max(float(ideal_area), 1.0)
    spread = max(math.log(max(float(max_area), ideal) / max(float(min_area), 1.0)), 1e-6)
    area_term = 15.0 * max(0.0, 1.0 - abs(math.log(area / ideal)) / spread)
    return float(
        30.0 * max(0.0, min(1.0, color_score)) +
        20.0 * max(0.0, min(1.0, compactness)) +
        15.0 * max(0.0, min(1.0, fill)) +
        area_term -
        5.0 * max(0.0, aspect - 1.0)
    )


def _extract_candidates(
    mask: Optional[np.ndarray],
    frame_bgr: np.ndarray,
    method: str,
    frame_idx: int,
    scale_x: float,
    scale_y: float,
    args: argparse.Namespace,
    reference: Optional[ReferencePoint],
    player_boxes: Sequence[Tuple[float, float, float, float]],
) -> Tuple[List[Candidate], int]:
    if mask is None:
        return [], 0
    mask_pixels = int(cv2.countNonZero(mask))
    if mask_pixels <= 0:
        return [], 0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: List[Candidate] = []
    for contour in contours:
        area_proc = float(cv2.contourArea(contour))
        if area_proc <= 0.0:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            continue
        area_orig = area_proc * scale_x * scale_y
        bw_orig = float(w) * scale_x
        bh_orig = float(h) * scale_y
        if area_orig < float(args.min_area) or area_orig > float(args.max_area):
            continue
        if float(args.max_dim) > 0.0 and max(bw_orig, bh_orig) > float(args.max_dim):
            continue
        aspect = float(max(w, h) / max(min(w, h), 1))
        if aspect > float(args.max_aspect):
            continue
        fill = float(area_proc / max(float(w * h), 1.0))
        if fill < float(args.min_fill):
            continue
        moments = cv2.moments(contour)
        if abs(float(moments.get("m00", 0.0))) <= 1e-9:
            continue
        cx_proc = float(moments["m10"] / moments["m00"])
        cy_proc = float(moments["m01"] / moments["m00"])
        cx = cx_proc * scale_x
        cy = cy_proc * scale_y
        perimeter = float(cv2.arcLength(contour, True))
        compactness = (
            float(4.0 * math.pi * area_proc / max(perimeter * perimeter, 1e-9))
            if perimeter > 0.0 else 0.0
        )
        cscore = _color_score(frame_bgr, contour, (x, y, w, h))
        score = _candidate_score(
            area_orig, aspect, fill, compactness, cscore,
            args.ideal_area, args.min_area, args.max_area
        )
        dist_ref = None
        if reference is not None:
            dist_ref = float(math.hypot(cx - reference.x, cy - reference.y))
        nearest_player = _nearest_player_distance(cx, cy, player_boxes) if player_boxes else None
        bbox_orig = (
            int(round(x * scale_x)),
            int(round(y * scale_y)),
            int(round((x + w) * scale_x)),
            int(round((y + h) * scale_y)),
        )
        candidates.append(Candidate(
            frame=int(frame_idx),
            method=str(method),
            x=float(cx),
            y=float(cy),
            bbox=bbox_orig,
            area=float(area_orig),
            aspect_ratio=aspect,
            fill_ratio=fill,
            compactness=compactness,
            color_score=cscore,
            score=score,
            distance_to_reference=dist_ref,
            nearest_player_distance=nearest_player,
        ))
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:max(1, int(args.top_k))], mask_pixels


def _method_weight(method: str) -> float:
    weights = {
        "adaptive_sv": 1.15,
        "delta2": 1.05,
        "delta3": 1.25,
        "knn": 0.70,
        "mog2": 0.45,
    }
    return float(weights.get(method, 1.0))


def _clip_bbox_union(boxes: Sequence[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
    if not boxes:
        return (0, 0, 0, 0)
    return (
        int(min(b[0] for b in boxes)),
        int(min(b[1] for b in boxes)),
        int(max(b[2] for b in boxes)),
        int(max(b[3] for b in boxes)),
    )


def _cluster_candidates(
    per_method_candidates: Dict[str, List[Candidate]],
    cluster_radius: float,
) -> List[List[Candidate]]:
    all_candidates: List[Candidate] = []
    for candidates in per_method_candidates.values():
        all_candidates.extend(candidates)
    all_candidates.sort(key=lambda c: c.score * _method_weight(c.method), reverse=True)

    clusters: List[List[Candidate]] = []
    centers: List[Tuple[float, float]] = []
    radius = max(1.0, float(cluster_radius))
    for cand in all_candidates:
        best_idx = -1
        best_dist = radius
        for i, (cx, cy) in enumerate(centers):
            dist = math.hypot(float(cand.x) - cx, float(cand.y) - cy)
            if dist <= best_dist:
                best_dist = dist
                best_idx = i
        if best_idx < 0:
            clusters.append([cand])
            centers.append((float(cand.x), float(cand.y)))
            continue
        clusters[best_idx].append(cand)
        weights = [max(1.0, float(c.score)) * _method_weight(c.method) for c in clusters[best_idx]]
        total_w = max(sum(weights), 1e-9)
        centers[best_idx] = (
            sum(float(c.x) * w for c, w in zip(clusters[best_idx], weights)) / total_w,
            sum(float(c.y) * w for c, w in zip(clusters[best_idx], weights)) / total_w,
        )
    return clusters


def _score_fused_candidates(
    per_method_candidates: Dict[str, List[Candidate]],
    frame_idx: int,
    state: SimpleFusionState,
    prediction: Dict[str, Any],
    reference: Optional[ReferencePoint],
    player_boxes: Sequence[Tuple[float, float, float, float]],
    args: argparse.Namespace,
) -> List[FusedCandidate]:
    clusters = _cluster_candidates(per_method_candidates, float(args.fusion_cluster_radius))
    fused: List[FusedCandidate] = []
    pred_active = bool(prediction.get("active"))
    pred_x = prediction.get("x")
    pred_y = prediction.get("y")
    pred_vx = float(prediction.get("vx", 0.0) or 0.0)
    pred_vy = float(prediction.get("vy", 0.0) or 0.0)
    gate_radius = float(prediction.get("gate_radius", args.fusion_gate_radius) or args.fusion_gate_radius)

    for members in clusters:
        if not members:
            continue
        weights = [max(1.0, float(c.score)) * _method_weight(c.method) for c in members]
        total_w = max(sum(weights), 1e-9)
        cx = sum(float(c.x) * w for c, w in zip(members, weights)) / total_w
        cy = sum(float(c.y) * w for c, w in zip(members, weights)) / total_w
        methods = sorted({str(c.method) for c in members})
        bbox = _clip_bbox_union([c.bbox for c in members])
        nearest_player = _nearest_player_distance(cx, cy, player_boxes) if player_boxes else None
        dist_ref = None
        if reference is not None:
            dist_ref = float(math.hypot(cx - reference.x, cy - reference.y))

        method_score = sum(_method_weight(m) for m in methods)
        mask_support_score = 12.0 * method_score + 2.0 * min(len(members), 6)
        candidate_quality_score = min(45.0, sum(max(0.0, c.score) for c in members) / max(len(members), 1) * 0.55)

        dist_pred = None
        kalman_gate_score = 0.0
        if pred_active and pred_x is not None and pred_y is not None:
            dist_pred = float(math.hypot(cx - float(pred_x), cy - float(pred_y)))
            gate = max(gate_radius, 1.0)
            if dist_pred <= gate:
                kalman_gate_score = 60.0 * (1.0 - dist_pred / gate)
            else:
                kalman_gate_score = -min(75.0, 38.0 * ((dist_pred - gate) / gate))

        velocity_score = 0.0
        velocity_cos = None
        speed_ratio = None
        if state.initialized and int(state.last_frame) < int(frame_idx):
            dt = max(1, int(frame_idx) - int(state.last_frame))
            obs_vx = (cx - float(state.x)) / float(dt)
            obs_vy = (cy - float(state.y)) / float(dt)
            obs_speed = math.hypot(obs_vx, obs_vy)
            pred_speed = math.hypot(pred_vx, pred_vy)
            if obs_speed > 1e-6 and pred_speed > 1.5:
                velocity_cos = float((obs_vx * pred_vx + obs_vy * pred_vy) / max(obs_speed * pred_speed, 1e-9))
                velocity_score += 22.0 * max(0.0, min(1.0, (velocity_cos + 0.20) / 1.20))
                if velocity_cos < -0.20:
                    velocity_score -= 32.0
                speed_ratio = float(obs_speed / max(pred_speed, 1e-6))
                log_err = abs(math.log(max(speed_ratio, 1e-4)))
                velocity_score += 18.0 * max(0.0, 1.0 - log_err / math.log(3.5))
                if speed_ratio < 0.15 or speed_ratio > 4.5:
                    velocity_score -= 18.0
            elif obs_speed > 2.0:
                velocity_score += min(12.0, obs_speed * 0.4)

        player_penalty = 0.0
        if nearest_player is not None:
            margin = max(1.0, float(args.player_margin))
            if nearest_player <= 0.0:
                player_penalty = 18.0
            elif nearest_player < margin:
                player_penalty = 14.0 * (1.0 - nearest_player / margin)

        score = mask_support_score + candidate_quality_score + kalman_gate_score + velocity_score - player_penalty
        fused.append(FusedCandidate(
            frame=int(frame_idx),
            x=float(cx),
            y=float(cy),
            bbox=bbox,
            methods=methods,
            member_count=int(len(members)),
            support_count=int(len(methods)),
            score=float(score),
            mask_support_score=float(mask_support_score),
            candidate_quality_score=float(candidate_quality_score),
            kalman_gate_score=float(kalman_gate_score),
            velocity_score=float(velocity_score),
            player_penalty=float(player_penalty),
            distance_to_prediction=dist_pred,
            gate_radius=float(gate_radius) if pred_active else None,
            velocity_cosine=velocity_cos,
            speed_ratio=speed_ratio,
            distance_to_reference=dist_ref,
            nearest_player_distance=nearest_player,
        ))

    fused.sort(key=lambda c: c.score, reverse=True)
    return fused[:max(1, int(getattr(args, "fusion_top_k", args.top_k)))]


def _p90(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), 90.0))


def _init_stats() -> Dict[str, Any]:
    return {
        "frames": 0,
        "mask_frames": 0,
        "mask_pixels_total": 0,
        "candidate_frames": 0,
        "candidate_frames_without_reference": 0,
        "total_candidates": 0,
        "reference_frames": 0,
        "hit_frames": 0,
        "miss_frames": 0,
        "near_player_candidates": 0,
        "mask_ms_total": 0.0,
        "extract_ms_total": 0.0,
        "best_distances": [],
    }


def _summarize_stats(stats: Dict[str, Dict[str, Any]], hit_radius: float) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for method, st in stats.items():
        frames = max(int(st["frames"]), 1)
        ref_frames = int(st["reference_frames"])
        best_distances = [float(v) for v in st["best_distances"]]
        summary[method] = {
            "frames": int(st["frames"]),
            "mask_frames": int(st["mask_frames"]),
            "candidate_frames": int(st["candidate_frames"]),
            "candidate_frames_without_reference": int(st["candidate_frames_without_reference"]),
            "total_candidates": int(st["total_candidates"]),
            "avg_candidates_per_frame": float(st["total_candidates"] / frames),
            "avg_mask_pixels_per_frame": float(st["mask_pixels_total"] / frames),
            "reference_frames": ref_frames,
            "hit_radius_px": float(hit_radius),
            "hit_frames": int(st["hit_frames"]),
            "miss_frames": int(st["miss_frames"]),
            "hit_rate": (float(st["hit_frames"] / ref_frames) if ref_frames > 0 else None),
            "mean_best_distance_px": (
                float(np.mean(best_distances)) if best_distances else None
            ),
            "p90_best_distance_px": _p90(best_distances),
            "near_player_candidates": int(st["near_player_candidates"]),
            "avg_mask_ms": float(st["mask_ms_total"] / frames),
            "avg_extract_ms": float(st["extract_ms_total"] / frames),
        }
    return summary


def _init_fusion_stats(radii: Sequence[float]) -> Dict[str, Any]:
    return {
        "frames": 0,
        "cluster_frames": 0,
        "accepted_frames": 0,
        "total_clusters": 0,
        "reference_frames": 0,
        "best_distances": [],
        "accepted_distances": [],
        "hit_counts": {str(float(r)): 0 for r in radii},
        "accepted_hit_counts": {str(float(r)): 0 for r in radii},
    }


def _summarize_fusion_stats(stats: Dict[str, Any], radii: Sequence[float]) -> Dict[str, Any]:
    frames = max(int(stats.get("frames", 0)), 1)
    ref_frames = int(stats.get("reference_frames", 0))
    best_distances = [float(v) for v in stats.get("best_distances", [])]
    accepted_distances = [float(v) for v in stats.get("accepted_distances", [])]
    return {
        "frames": int(stats.get("frames", 0)),
        "cluster_frames": int(stats.get("cluster_frames", 0)),
        "accepted_frames": int(stats.get("accepted_frames", 0)),
        "total_clusters": int(stats.get("total_clusters", 0)),
        "avg_clusters_per_frame": float(stats.get("total_clusters", 0) / frames),
        "reference_frames": ref_frames,
        "hit_rates": {
            str(float(r)): (
                float(stats["hit_counts"].get(str(float(r)), 0) / ref_frames)
                if ref_frames > 0 else None
            )
            for r in radii
        },
        "accepted_hit_rates": {
            str(float(r)): (
                float(stats["accepted_hit_counts"].get(str(float(r)), 0) / ref_frames)
                if ref_frames > 0 else None
            )
            for r in radii
        },
        "mean_best_distance_px": float(np.mean(best_distances)) if best_distances else None,
        "p90_best_distance_px": _p90(best_distances),
        "mean_accepted_distance_px": float(np.mean(accepted_distances)) if accepted_distances else None,
        "p90_accepted_distance_px": _p90(accepted_distances),
    }


def _overlay_motion_masks(
    out: np.ndarray,
    per_method_masks: Dict[str, Optional[np.ndarray]],
    alpha: float,
) -> np.ndarray:
    alpha = max(0.0, min(0.65, float(alpha)))
    if alpha <= 0.0:
        return out
    h, w = out.shape[:2]
    overlay = out.copy()
    for method, mask in per_method_masks.items():
        if mask is None or cv2.countNonZero(mask) <= 0:
            continue
        if mask.shape[:2] != (h, w):
            mask_show = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            mask_show = mask
        color = METHOD_COLORS.get(method, (0, 255, 255))
        overlay[mask_show > 0] = color
    return cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0.0)


def _mask_for_motion_view(
    mask: Optional[np.ndarray],
    width: int,
    height: int,
    visual_dilate: int,
) -> np.ndarray:
    if mask is None or cv2.countNonZero(mask) <= 0:
        return np.zeros((height, width), dtype=np.uint8)
    if mask.shape[:2] != (height, width):
        out = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    else:
        out = mask.copy()
    out = (out > 0).astype(np.uint8) * 255
    if visual_dilate > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        out = cv2.dilate(out, kernel, iterations=int(visual_dilate))
    return out


def _draw_motion_only_frame(
    frame: np.ndarray,
    frame_idx: int,
    per_method_masks: Dict[str, Optional[np.ndarray]],
    args: argparse.Namespace,
) -> np.ndarray:
    """Render only the physical motion masks, with no tracking/debug boxes."""
    height, width = frame.shape[:2]
    visual_dilate = max(0, int(getattr(args, "motion_visual_dilate", 0)))
    display_masks = {
        method: _mask_for_motion_view(mask, width, height, visual_dilate)
        for method, mask in per_method_masks.items()
    }

    motion_colors = np.zeros((height, width, 3), dtype=np.uint8)
    for method, mask in display_masks.items():
        if cv2.countNonZero(mask) <= 0:
            continue
        motion_colors[mask > 0] = METHOD_COLORS.get(method, (0, 255, 255))

    if len(display_masks) > 1:
        overlap_count = np.zeros((height, width), dtype=np.uint8)
        for mask in display_masks.values():
            overlap_count += (mask > 0).astype(np.uint8)
        overlap = overlap_count > 1
        motion_colors[overlap] = (255, 255, 255)

    background = str(getattr(args, "motion_background", "black")).lower()
    if background == "translucent":
        base_alpha = max(0.0, min(1.0, float(getattr(args, "motion_base_alpha", 0.25))))
        motion_alpha = max(0.0, min(1.0, float(getattr(args, "motion_alpha", 0.95))))
        out = cv2.convertScaleAbs(frame, alpha=base_alpha, beta=0.0)
        out = cv2.addWeighted(out, 1.0, motion_colors, motion_alpha, 0.0)
    else:
        out = motion_colors

    legend = " | ".join(METHOD_LABELS.get(method, f"{method}=color") for method in per_method_masks)
    lines: List[Tuple[str, Tuple[int, int, int]]] = [
        (f"frame {frame_idx} | motion only | background={background}", (255, 255, 255)),
        (f"{legend} | overlap = white", (230, 230, 230)),
    ]
    for method in per_method_masks:
        color = METHOD_COLORS.get(method, (0, 255, 255))
        raw_count = 0
        if per_method_masks[method] is not None:
            raw_count = int(cv2.countNonZero(per_method_masks[method]))
        lines.append((f"{method}: motion_px={raw_count}", color))
    _draw_info_panel(out, lines)
    return out


def _draw_info_panel(out: np.ndarray, lines: Sequence[Tuple[str, Tuple[int, int, int]]]) -> None:
    if not lines:
        return
    panel_w = min(out.shape[1] - 20, 620)
    panel_h = min(out.shape[0] - 20, 26 + 20 * len(lines))
    panel = out.copy()
    cv2.rectangle(panel, (10, 10), (10 + panel_w, 10 + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(panel, 0.58, out, 0.42, 0.0, out)
    y = 34
    for text, color in lines:
        cv2.putText(out, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, color, 1, cv2.LINE_AA)
        y += 20


def _draw_debug_frame(
    frame: np.ndarray,
    frame_idx: int,
    per_method_candidates: Dict[str, List[Candidate]],
    per_method_masks: Dict[str, Optional[np.ndarray]],
    method_payloads: Dict[str, Any],
    fused_candidates: Sequence[FusedCandidate],
    chosen_fused: Optional[FusedCandidate],
    prediction: Dict[str, Any],
    reference: Optional[ReferencePoint],
    hit_radius: float,
    mask_alpha: float = 0.18,
    fusion_trail: Optional[Sequence[Tuple[float, float]]] = None,
) -> np.ndarray:
    out = frame.copy()
    out = _overlay_motion_masks(out, per_method_masks, mask_alpha)

    if prediction.get("active") and prediction.get("x") is not None and prediction.get("y") is not None:
        px, py = int(round(float(prediction["x"]))), int(round(float(prediction["y"])))
        gate = int(round(float(prediction.get("gate_radius", 0.0) or 0.0)))
        if gate > 0:
            cv2.circle(out, (px, py), gate, (255, 80, 0), 1, cv2.LINE_AA)
        cv2.drawMarker(out, (px, py), (255, 80, 0), cv2.MARKER_TILTED_CROSS, 18, 2, cv2.LINE_AA)
        cv2.putText(out, "pred/gate", (px + 8, py + 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 80, 0), 1, cv2.LINE_AA)

    if reference is not None:
        rx, ry = int(round(reference.x)), int(round(reference.y))
        cv2.circle(out, (rx, ry), int(max(3, round(hit_radius))), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.drawMarker(out, (rx, ry), (255, 255, 255), cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
        cv2.putText(out, reference.kind, (rx + 8, ry - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)

    if fusion_trail:
        pts = [(int(round(x)), int(round(y))) for x, y in fusion_trail[-80:]]
        for a, b in zip(pts, pts[1:]):
            cv2.line(out, a, b, (0, 255, 180), 1, cv2.LINE_AA)

    for method, candidates in per_method_candidates.items():
        color = METHOD_COLORS.get(method, (0, 255, 255))
        for rank, cand in enumerate(candidates[:3], start=1):
            x1, y1, x2, y2 = cand.bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
            cv2.circle(out, (int(round(cand.x)), int(round(cand.y))), 3, color, -1, cv2.LINE_AA)
            if rank == 1:
                label = f"{method}:{rank}"
                if cand.distance_to_reference is not None:
                    label += f" {cand.distance_to_reference:.0f}px"
                cv2.putText(out, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, color, 1, cv2.LINE_AA)

    for rank, cand in enumerate(fused_candidates[:5], start=1):
        x1, y1, x2, y2 = cand.bbox
        color = (0, 255, 180) if cand.accepted else (0, 180, 255)
        thickness = 3 if cand.accepted else 1
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        cv2.circle(out, (int(round(cand.x)), int(round(cand.y))), 5, color, -1, cv2.LINE_AA)
        label = f"F{rank} {cand.score:.0f} {'+'.join(cand.methods)}"
        if cand.distance_to_prediction is not None:
            label += f" pred{cand.distance_to_prediction:.0f}"
        if cand.distance_to_reference is not None:
            label += f" ref{cand.distance_to_reference:.0f}"
        cv2.putText(out, label, (x1, min(out.shape[0] - 8, y2 + 16)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, color, 1, cv2.LINE_AA)

    lines: List[Tuple[str, Tuple[int, int, int]]] = []
    lines.append((f"frame {frame_idx} | fusion={'ON' if fused_candidates else 'no clusters'}", (255, 255, 255)))
    if prediction.get("active"):
        lines.append((
            f"pred=({prediction.get('x'):.0f},{prediction.get('y'):.0f}) "
            f"v=({prediction.get('vx'):.1f},{prediction.get('vy'):.1f}) "
            f"gate={prediction.get('gate_radius'):.0f}",
            (255, 180, 80),
        ))
    else:
        lines.append(("pred=inactive (waiting for seed/accepted fused candidate)", (255, 180, 80)))
    if chosen_fused is not None:
        lines.append((
            f"ACCEPT F score={chosen_fused.score:.1f} methods={'+'.join(chosen_fused.methods)} "
            f"support={chosen_fused.support_count} members={chosen_fused.member_count}",
            (0, 255, 180),
        ))
        lines.append((
            f"  gate={chosen_fused.kalman_gate_score:.1f} vel={chosen_fused.velocity_score:.1f} "
            f"mask={chosen_fused.mask_support_score:.1f} quality={chosen_fused.candidate_quality_score:.1f} "
            f"player_pen={chosen_fused.player_penalty:.1f}",
            (0, 230, 180),
        ))
    elif fused_candidates:
        best = fused_candidates[0]
        lines.append((f"BEST rejected score={best.score:.1f} methods={'+'.join(best.methods)}", (0, 180, 255)))
    if reference is not None:
        lines.append((f"reference {reference.kind}=({reference.x:.0f},{reference.y:.0f})", (255, 255, 255)))
    for method, payload in method_payloads.items():
        color = METHOD_COLORS.get(method, (0, 255, 255))
        best_dist = payload.get("best_distance_to_reference")
        dist_txt = "n/a" if best_dist is None else f"{float(best_dist):.0f}px"
        lines.append((
            f"{method}: blobs={payload.get('candidate_count', 0)} "
            f"mask_px={payload.get('mask_pixels', 0)} best_ref={dist_txt}",
            color,
        ))
    _draw_info_panel(out, lines[:15])
    return out


def _write_csv(path: str, frame_rows: Sequence[Dict[str, Any]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "frame", "method", "rank", "x", "y", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
        "area", "aspect_ratio", "fill_ratio", "compactness", "color_score", "score",
        "distance_to_reference", "nearest_player_distance",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in frame_rows:
            methods = row.get("methods", {})
            for method, payload in methods.items():
                for rank, cand in enumerate(payload.get("candidates", []), start=1):
                    bbox = cand.get("bbox", [None, None, None, None])
                    writer.writerow({
                        "frame": row.get("frame"),
                        "method": method,
                        "rank": rank,
                        "x": cand.get("x"),
                        "y": cand.get("y"),
                        "bbox_x1": bbox[0],
                        "bbox_y1": bbox[1],
                        "bbox_x2": bbox[2],
                        "bbox_y2": bbox[3],
                        "area": cand.get("area"),
                        "aspect_ratio": cand.get("aspect_ratio"),
                        "fill_ratio": cand.get("fill_ratio"),
                        "compactness": cand.get("compactness"),
                        "color_score": cand.get("color_score"),
                        "score": cand.get("score"),
                        "distance_to_reference": cand.get("distance_to_reference"),
                        "nearest_player_distance": cand.get("nearest_player_distance"),
                    })


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    _require_cv_runtime()
    methods = list(args.methods)
    if args.downscale <= 0.0:
        raise SystemExit("--downscale must be > 0")

    tracking_payload = _load_json(args.tracking_json)
    annotation_payload = _load_json(args.annotations)
    references, reference_source = _load_references(tracking_payload, annotation_payload)
    player_boxes_by_frame = _load_player_boxes_by_frame(tracking_payload)

    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise SystemExit(f"Unable to open input video: {args.input}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    start_frame = max(0, int(args.start_frame))
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    ok, curr = cap.read()
    if not ok or curr is None:
        cap.release()
        raise SystemExit("No frames available at requested start frame.")
    if width <= 0 or height <= 0:
        height, width = curr.shape[:2]
    ok_next, next_frame = cap.read()
    if not ok_next:
        next_frame = None

    writer = None
    if args.debug_video:
        debug_path = Path(args.debug_video)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(debug_path), fourcc, fps, (width, height))

    adaptive = AdaptiveSVState(
        thresh=args.adaptive_thresh,
        k_std=args.adaptive_k_std,
        v_min=args.motion_v_min,
        alpha=args.adaptive_alpha,
        motion_alpha=args.adaptive_motion_alpha,
    )
    stats = {method: _init_stats() for method in methods}
    fusion_enabled = bool(args.fusion)
    fusion_radii = [float(r) for r in args.fusion_hit_radii]
    fusion_state = SimpleFusionState()
    fusion_stats = _init_fusion_stats(fusion_radii)
    fusion_trail: List[Tuple[float, float]] = []
    last_reference_for_velocity: Optional[ReferencePoint] = None
    frame_rows: List[Dict[str, Any]] = []
    prev: Optional[np.ndarray] = None
    local_count = 0
    max_frames = int(args.max_frames)
    t_start = time.perf_counter()

    while curr is not None and (max_frames <= 0 or local_count < max_frames):
        frame_idx = start_frame + local_count
        proc_prev = _resize_frame(prev, float(args.downscale))
        proc_curr = _resize_frame(curr, float(args.downscale))
        proc_next = _resize_frame(next_frame, float(args.downscale))
        assert proc_curr is not None
        motion_prev = _prefilter_frame(proc_prev, args)
        motion_curr = _prefilter_frame(proc_curr, args)
        motion_next = _prefilter_frame(proc_next, args)
        assert motion_curr is not None
        scale_x = width / float(proc_curr.shape[1])
        scale_y = height / float(proc_curr.shape[0])
        min_component_area_proc = (
            float(args.min_component_area) / max(scale_x * scale_y, 1e-9)
            if float(args.min_component_area) > 0.0 else 0.0
        )

        reference = references.get(frame_idx)
        player_boxes = player_boxes_by_frame.get(frame_idx, [])
        row_methods: Dict[str, Any] = {}
        per_method_candidates: Dict[str, List[Candidate]] = {}
        per_method_masks: Dict[str, Optional[np.ndarray]] = {}

        for method in methods:
            st = stats[method]
            st["frames"] += 1
            mask_t0 = time.perf_counter()
            if method == "adaptive_sv":
                mask = adaptive.apply(motion_curr)
            elif method == "delta3":
                mask = _delta3_mask(
                    motion_prev, motion_curr, motion_next,
                    args.diff_thresh, args.diff_lo_frac, args.diff_very_hi, args.motion_v_min
                )
            else:
                mask = np.zeros(proc_curr.shape[:2], dtype=np.uint8)
            mask = _clean_mask(
                mask,
                int(args.open_size),
                int(args.close_size),
                min_component_area=min_component_area_proc,
            )
            mask = _apply_flicker_guard(mask, args, scale_x, scale_y)
            mask = _keep_largest_components(mask, int(args.mask_keep_largest_components))
            per_method_masks[method] = mask
            mask_ms = (time.perf_counter() - mask_t0) * 1000.0
            st["mask_ms_total"] += mask_ms

            extract_t0 = time.perf_counter()
            if bool(args.skip_candidates):
                candidates = []
                mask_pixels = 0 if mask is None else int(cv2.countNonZero(mask))
                extract_ms = 0.0
            else:
                candidates, mask_pixels = _extract_candidates(
                    mask, proc_curr, method, frame_idx, scale_x, scale_y,
                    args, reference, player_boxes
                )
                extract_ms = (time.perf_counter() - extract_t0) * 1000.0
            st["extract_ms_total"] += extract_ms
            st["mask_pixels_total"] += mask_pixels
            if mask_pixels > 0:
                st["mask_frames"] += 1
            if candidates:
                st["candidate_frames"] += 1
                st["total_candidates"] += len(candidates)
                if reference is None:
                    st["candidate_frames_without_reference"] += 1
            if reference is not None:
                st["reference_frames"] += 1
                if candidates:
                    best_dist = min(
                        float(c.distance_to_reference)
                        for c in candidates
                        if c.distance_to_reference is not None
                    )
                    st["best_distances"].append(best_dist)
                    if best_dist <= float(args.hit_radius):
                        st["hit_frames"] += 1
                    else:
                        st["miss_frames"] += 1
                else:
                    st["miss_frames"] += 1
            st["near_player_candidates"] += sum(
                1 for c in candidates
                if c.nearest_player_distance is not None and c.nearest_player_distance <= float(args.player_margin)
            )

            per_method_candidates[method] = candidates
            row_methods[method] = {
                "mask_pixels": int(mask_pixels),
                "mask_ms": float(mask_ms),
                "extract_ms": float(extract_ms),
                "candidate_count": int(len(candidates)),
                "best_distance_to_reference": (
                    None if not candidates or reference is None else
                    min(c.distance_to_reference for c in candidates if c.distance_to_reference is not None)
                ),
                "hit": (
                    None if reference is None else
                    bool(candidates and min(
                        c.distance_to_reference for c in candidates if c.distance_to_reference is not None
                    ) <= float(args.hit_radius))
                ),
                "candidates": [asdict(c) for c in candidates],
            }

        fused_candidates: List[FusedCandidate] = []
        chosen_fused: Optional[FusedCandidate] = None
        prediction = fusion_state.predict(frame_idx, float(args.fusion_gate_radius))
        if fusion_enabled:
            if (
                bool(args.fusion_seed_reference) and reference is not None and
                (
                    not fusion_state.initialized or
                    int(fusion_state.misses) >= int(args.fusion_seed_after_misses)
                )
            ):
                fusion_state.seed(reference.x, reference.y, frame_idx)
            prediction = fusion_state.predict(frame_idx, float(args.fusion_gate_radius))
            if bool(args.fusion_guide_reference) and reference is not None:
                guide_vx = 0.0
                guide_vy = 0.0
                if last_reference_for_velocity is not None and frame_idx > last_reference_for_velocity.frame:
                    dt_ref = max(1, int(frame_idx) - int(last_reference_for_velocity.frame))
                    guide_vx = (float(reference.x) - float(last_reference_for_velocity.x)) / float(dt_ref)
                    guide_vy = (float(reference.y) - float(last_reference_for_velocity.y)) / float(dt_ref)
                prediction = {
                    "active": True,
                    "x": float(reference.x),
                    "y": float(reference.y),
                    "vx": float(guide_vx),
                    "vy": float(guide_vy),
                    "speed": float(math.hypot(guide_vx, guide_vy)),
                    "gate_radius": float(args.fusion_gate_radius),
                    "source": str(reference.kind),
                }
            fused_candidates = _score_fused_candidates(
                per_method_candidates,
                frame_idx,
                fusion_state,
                prediction,
                reference,
                player_boxes,
                args,
            )
            fusion_stats["frames"] += 1
            fusion_stats["total_clusters"] += len(fused_candidates)
            if fused_candidates:
                fusion_stats["cluster_frames"] += 1
            if reference is not None:
                fusion_stats["reference_frames"] += 1
                if fused_candidates:
                    best_dist = min(
                        float(c.distance_to_reference)
                        for c in fused_candidates
                        if c.distance_to_reference is not None
                    )
                    fusion_stats["best_distances"].append(best_dist)
                    for radius in fusion_radii:
                        key = str(float(radius))
                        if best_dist <= float(radius):
                            fusion_stats["hit_counts"][key] += 1

            if fused_candidates and float(fused_candidates[0].score) >= float(args.fusion_min_score):
                chosen_fused = fused_candidates[0]
                chosen_fused.accepted = True
                fusion_state.update(chosen_fused.x, chosen_fused.y, frame_idx)
                fusion_trail.append((chosen_fused.x, chosen_fused.y))
                if len(fusion_trail) > 300:
                    fusion_trail = fusion_trail[-300:]
                fusion_stats["accepted_frames"] += 1
                if reference is not None and chosen_fused.distance_to_reference is not None:
                    acc_dist = float(chosen_fused.distance_to_reference)
                    fusion_stats["accepted_distances"].append(acc_dist)
                    for radius in fusion_radii:
                        key = str(float(radius))
                        if acc_dist <= float(radius):
                            fusion_stats["accepted_hit_counts"][key] += 1
            else:
                fusion_state.miss()
        if reference is not None:
            last_reference_for_velocity = reference

        frame_rows.append({
            "frame": int(frame_idx),
            "reference": None if reference is None else asdict(reference),
            "methods": row_methods,
            "fusion": {
                "enabled": bool(fusion_enabled),
                "prediction": prediction,
                "chosen": None if chosen_fused is None else asdict(chosen_fused),
                "candidates": [asdict(c) for c in fused_candidates],
            },
        })

        if writer is not None:
            if str(args.debug_view) == "motion":
                writer.write(_draw_motion_only_frame(curr, frame_idx, per_method_masks, args))
            else:
                writer.write(_draw_debug_frame(
                    curr,
                    frame_idx,
                    per_method_candidates,
                    per_method_masks,
                    row_methods,
                    fused_candidates,
                    chosen_fused,
                    prediction,
                    reference,
                    float(args.hit_radius),
                    mask_alpha=float(args.mask_overlay_alpha),
                    fusion_trail=fusion_trail,
                ))

        local_count += 1
        if not args.no_progress and local_count % 100 == 0:
            elapsed = time.perf_counter() - t_start
            print(f"[probe] {local_count} frames, {local_count / max(elapsed, 1e-9):.1f} fps")

        prev = curr
        curr = next_frame
        ok_next, next_frame = cap.read()
        if not ok_next:
            next_frame = None

    elapsed = time.perf_counter() - t_start
    cap.release()
    if writer is not None:
        writer.release()

    summary = _summarize_stats(stats, float(args.hit_radius))
    fusion_summary = _summarize_fusion_stats(fusion_stats, fusion_radii) if fusion_enabled else None
    output = {
        "schema_version": 1,
        "input": str(args.input),
        "video": {
            "fps": fps,
            "width": width,
            "height": height,
            "total_frames": total_frames,
            "start_frame": start_frame,
            "frames_processed": local_count,
        },
        "reference_source": reference_source,
        "methods": methods,
        "elapsed_sec": float(elapsed),
        "effective_fps": float(local_count / max(elapsed, 1e-9)),
        "params": {
            "downscale": float(args.downscale),
            "hit_radius": float(args.hit_radius),
            "top_k": int(args.top_k),
            "skip_candidates": bool(args.skip_candidates),
            "min_area": float(args.min_area),
            "max_area": float(args.max_area),
            "max_dim": float(args.max_dim),
            "max_aspect": float(args.max_aspect),
            "min_fill": float(args.min_fill),
            "min_component_area": float(args.min_component_area),
            "flicker_clean_frame_frac": float(args.flicker_clean_frame_frac),
            "flicker_clean_component_count": int(args.flicker_clean_component_count),
            "flicker_clean_component_area": float(args.flicker_clean_component_area),
            "flicker_suppress_frame_frac": float(args.flicker_suppress_frame_frac),
            "mask_keep_largest_components": int(args.mask_keep_largest_components),
            "prefilter": str(args.prefilter),
            "prefilter_ksize": int(args.prefilter_ksize),
            "prefilter_sigma": float(args.prefilter_sigma),
            "diff_thresh": float(args.diff_thresh),
            "adaptive_thresh": float(args.adaptive_thresh),
            "debug_view": str(args.debug_view),
            "motion_background": str(args.motion_background),
            "motion_alpha": float(args.motion_alpha),
            "motion_base_alpha": float(args.motion_base_alpha),
            "motion_visual_dilate": int(args.motion_visual_dilate),
            "fusion": bool(fusion_enabled),
            "fusion_cluster_radius": float(args.fusion_cluster_radius),
            "fusion_gate_radius": float(args.fusion_gate_radius),
            "fusion_min_score": float(args.fusion_min_score),
            "fusion_seed_reference": bool(args.fusion_seed_reference),
            "fusion_guide_reference": bool(args.fusion_guide_reference),
            "fusion_hit_radii": fusion_radii,
        },
        "summary": summary,
        "fusion_summary": fusion_summary,
        "frames": frame_rows,
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    if args.output_csv:
        _write_csv(args.output_csv, frame_rows)

    print(f"[probe] wrote {out_path}")
    if args.output_csv:
        print(f"[probe] wrote {args.output_csv}")
    if args.debug_video:
        print(f"[probe] wrote {args.debug_video}")
    for method in methods:
        ms = summary[method]
        hit_rate = ms["hit_rate"]
        hit_text = "n/a" if hit_rate is None else f"{100.0 * hit_rate:.1f}%"
        print(
            f"[probe] {method}: hit={hit_text}, "
            f"cands/frame={ms['avg_candidates_per_frame']:.2f}, "
            f"mask={ms['avg_mask_ms']:.2f}ms, extract={ms['avg_extract_ms']:.2f}ms"
        )
    if fusion_summary is not None:
        hit_bits = []
        for radius, rate in fusion_summary["hit_rates"].items():
            hit_bits.append(f"{radius}px={'n/a' if rate is None else f'{100.0 * rate:.1f}%'}")
        print(
            f"[probe] fusion: accepted={fusion_summary['accepted_frames']}/{fusion_summary['frames']}, "
            f"clusters/frame={fusion_summary['avg_clusters_per_frame']:.2f}, "
            f"hits({', '.join(hit_bits)})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
