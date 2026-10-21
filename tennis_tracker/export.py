"""The tracking JSON: per-frame results, tracks, court/player data and optional motion diagnostics."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from ball_in_play_selector import FrameResult, PackedMask, Track

from .pass1 import Pass1Result
from .run_context import RunContext, RunResult

# Motion-diagnostics search fallbacks, as fractions of the frame diagonal with a pixel floor.
_SEARCH_FROM_POSITION_FRAC, _SEARCH_FROM_POSITION_MIN_PX = 0.035, 24.0
_SEARCH_FROM_PREVIOUS_FRAC, _SEARCH_FROM_PREVIOUS_MIN_PX = 0.045, 28.0
_BLOB_TOO_FAR_MIN_GATE_PX = 18.0
# Candidate ranking weights (debug only): compactness, fill, area (saturating), distance penalty, aspect penalty.
_SCORE_COMPACTNESS, _SCORE_FILL, _SCORE_AREA, _SCORE_AREA_SATURATION_PX = 50.0, 25.0, 15.0, 600.0
_SCORE_DISTANCE_CAP_PX, _SCORE_DISTANCE_DIVISOR, _SCORE_ASPECT_PENALTY = 300.0, 6.0, 4.0
_MAX_CANDIDATES_PER_FRAME = 5
_SCHEMA_VERSION = 1


def json_safe(value: Any) -> Any:
    """``value`` with numpy scalars, paths and tuples turned into JSON-serialisable types."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    return str(value)


def frame_result_to_json(frame_idx: int, result: Optional[FrameResult]) -> Dict[str, Any]:
    row: Dict[str, Any] = {"frame": int(frame_idx), "present": False}
    if result is None:
        return row
    row.update({
        "present": result.cx is not None and result.cy is not None,
        "x": None if result.cx is None else float(result.cx),
        "y": None if result.cy is None else float(result.cy),
        "conf": float(result.conf),
        "source": str(result.source),
        "interpolated": bool(result.interpolated),
        "bbox": json_safe(result.bbox),
        "measurement": (
            {"x": float(result.measurement[0]), "y": float(result.measurement[1]),
             "conf": float(result.conf), "bbox": json_safe(result.bbox)}
            if result.measurement is not None else None
        ),
        "position_kind": (
            "measurement" if result.measurement == (result.cx, result.cy)
            else "filtered" if result.measurement is not None else "recovered"
        ),
        "search": {
            "x": float(result.search_cx),
            "y": float(result.search_cy),
            "radius": float(result.search_radius),
        },
        "selection": {"source": str(result.source), "reason": str(result.source_reason)},
    })
    return row


def track_to_json(track: Optional[Track]) -> Optional[Dict[str, Any]]:
    if track is None:
        return None
    observations = [{
        "frame": int(obs.frame),
        "x": float(obs.cx),
        "y": float(obs.cy),
        "bbox": [float(obs.x1), float(obs.y1), float(obs.x2), float(obs.y2)],
        "conf": float(obs.conf),
        "area": float(obs.area),
        "on_motion": bool(obs.on_motion),
    } for obs in (track.observations or [])]
    return {
        "track_id": int(track.track_id),
        "score": float(track.score),
        "num_obs": int(track.num_obs),
        "span": int(track.span),
        "first_frame": int(track.first_frame),
        "last_obs_frame": int(track.last_obs_frame),
        "score_breakdown": json_safe(track.score_breakdown or {}),
        "observations": observations,
    }


# ---------------------------------------------------------------------------
# Motion diagnostics (opt-in; nothing in the normal path reads them)
# ---------------------------------------------------------------------------

def _extract_motion_candidates(
    mask: Optional[PackedMask],
    mask_source: str,
    frame_idx: int,
    search_x: Optional[float],
    search_y: Optional[float],
    selected_x: Optional[float],
    selected_y: Optional[float],
    max_candidates: int = _MAX_CANDIDATES_PER_FRAME,
) -> List[Dict[str, Any]]:
    """Blobs of ``mask`` ranked for the diagnostics: compact, filled and near the search point first."""
    if mask is None:
        return []
    contours, _ = cv2.findContours(mask.unpack(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: List[Dict[str, Any]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area <= 0.0:
            continue
        x, y, w_box, h_box = cv2.boundingRect(contour)
        if w_box <= 0 or h_box <= 0:
            continue
        moments = cv2.moments(contour)
        if abs(float(moments.get("m00", 0.0))) <= 1e-9:
            continue
        cx = float(moments["m10"] / moments["m00"])
        cy = float(moments["m01"] / moments["m00"])
        perimeter = float(cv2.arcLength(contour, True))
        compactness = (
            float(4.0 * math.pi * area / max(perimeter * perimeter, 1e-9))
            if perimeter > 0.0 else 0.0
        )
        aspect_ratio = float(max(w_box, h_box) / max(min(w_box, h_box), 1))
        fill_ratio = float(area / max(float(w_box * h_box), 1.0))

        dist_to_search = None
        if search_x is not None and search_y is not None:
            dist_to_search = float(math.hypot(cx - float(search_x), cy - float(search_y)))
        dist_to_selected = None
        if selected_x is not None and selected_y is not None:
            dist_to_selected = float(math.hypot(cx - float(selected_x), cy - float(selected_y)))

        score = 0.0
        score += _SCORE_COMPACTNESS * max(0.0, min(1.0, compactness))
        score += _SCORE_FILL * max(0.0, min(1.0, fill_ratio))
        score += _SCORE_AREA * min(area, _SCORE_AREA_SATURATION_PX) / _SCORE_AREA_SATURATION_PX
        if dist_to_search is not None:
            score -= min(dist_to_search, _SCORE_DISTANCE_CAP_PX) / _SCORE_DISTANCE_DIVISOR
        if aspect_ratio > 1.0:
            score -= max(0.0, aspect_ratio - 1.0) * _SCORE_ASPECT_PENALTY

        candidates.append({
            "frame": int(frame_idx),
            "mask": str(mask_source),
            "x": cx,
            "y": cy,
            "bbox": [int(x), int(y), int(x + w_box), int(y + h_box)],
            "area": area,
            "compactness": compactness,
            "aspect_ratio": aspect_ratio,
            "fill_ratio": fill_ratio,
            "distance_to_search": dist_to_search,
            "distance_to_selected": dist_to_selected,
            "score": float(score),
        })

    candidates.sort(key=lambda item: float(item["score"]), reverse=True)
    return candidates[:max(1, int(max_candidates))]


def _choose_motion_mask(boost_mask: Optional[PackedMask],
                        raw_motion: Optional[PackedMask]) -> Tuple[Optional[PackedMask], str]:
    """The boost mask when it has pixels, else the raw mask, else nothing (a packed mask is never empty)."""
    if boost_mask is not None:
        return boost_mask, "boost"
    if raw_motion is not None:
        return raw_motion, "raw"
    return None, "none"


def _result_position(result: Optional[FrameResult]) -> Tuple[Optional[float], Optional[float]]:
    if result is None or result.cx is None or result.cy is None:
        return None, None
    return float(result.cx), float(result.cy)


def _result_search(
    result: Optional[FrameResult],
    prev_pos: Optional[Tuple[float, float]],
    diag: float,
) -> Tuple[Optional[float], Optional[float], float, str]:
    """Where a motion blob would have been looked for this frame, and why that point."""
    if result is not None:
        sx, sy, sr = float(result.search_cx), float(result.search_cy), float(result.search_radius)
        if sr > 0.0 and (sx != 0.0 or sy != 0.0):
            return sx, sy, sr, "result_search"
        rx, ry = _result_position(result)
        if rx is not None and ry is not None:
            return rx, ry, max(_SEARCH_FROM_POSITION_MIN_PX, _SEARCH_FROM_POSITION_FRAC * diag), "result_position"
    if prev_pos is not None:
        return (float(prev_pos[0]), float(prev_pos[1]),
                max(_SEARCH_FROM_PREVIOUS_MIN_PX, _SEARCH_FROM_PREVIOUS_FRAC * diag), "previous_position")
    return None, None, 0.0, "none"


def _motion_diagnostic_reason(
    selected_source: Optional[str],
    has_detections: bool,
    mask_source: str,
    candidates: List[Dict[str, Any]],
    search_x: Optional[float],
    search_y: Optional[float],
    search_radius: float,
) -> str:
    if selected_source == "motion":
        return "accepted_motion"
    if has_detections:
        return "detection_available"
    if mask_source == "none":
        return "no_motion_mask"
    if not candidates:
        return "no_blob"
    if search_x is None or search_y is None:
        return "no_search_anchor"
    best_dist = candidates[0].get("distance_to_search")
    if best_dist is not None:
        gate = max(_BLOB_TOO_FAR_MIN_GATE_PX, float(search_radius) if search_radius > 0.0 else 0.0)
        if float(best_dist) > gate:
            return "blob_too_far_from_search"
    if selected_source == "interp":
        return "interp_selected_over_motion_candidate"
    if selected_source:
        return "non_motion_selected_over_motion_candidate"
    return "lost_despite_motion_candidate"


def build_motion_diagnostics(
    per_frame: List[Optional[FrameResult]],
    detections_by_frame: List[List[Tuple[list, float]]],
    boost_masks: List[Optional[PackedMask]],
    raw_motions: List[Optional[PackedMask]],
    width: int,
    height: int,
    max_candidates_per_frame: int = _MAX_CANDIDATES_PER_FRAME,
) -> Dict[str, Any]:
    """Per-frame record of why motion evidence was or was not used, for frames without detections."""
    diag = math.sqrt(float(width) ** 2 + float(height) ** 2)
    total_frames = max(len(per_frame), len(detections_by_frame), len(boost_masks), len(raw_motions))
    reason_counts: Dict[str, int] = {}
    mask_counts: Dict[str, int] = {}
    selected_source_counts: Dict[str, int] = {}
    diagnostic_frames: List[Dict[str, Any]] = []
    prev_pos: Optional[Tuple[float, float]] = None
    detector_gap_frames = 0
    candidate_gap_frames = 0

    for frame_idx in range(total_frames):
        result = per_frame[frame_idx] if frame_idx < len(per_frame) else None
        selected_x, selected_y = _result_position(result)
        selected_source = None
        if result is not None:
            selected_source = str(result.source or "")
            if selected_source:
                selected_source_counts[selected_source] = selected_source_counts.get(selected_source, 0) + 1

        dets = detections_by_frame[frame_idx] if frame_idx < len(detections_by_frame) else []
        has_detections = bool(dets)
        if not has_detections:
            detector_gap_frames += 1

        boost = boost_masks[frame_idx] if frame_idx < len(boost_masks) else None
        raw = raw_motions[frame_idx] if frame_idx < len(raw_motions) else None
        chosen_mask, mask_source = _choose_motion_mask(boost, raw)
        mask_counts[mask_source] = mask_counts.get(mask_source, 0) + 1

        search_x, search_y, search_radius, search_source = _result_search(result, prev_pos, diag)
        candidates = _extract_motion_candidates(
            chosen_mask, mask_source, frame_idx, search_x, search_y, selected_x, selected_y,
            max_candidates=max_candidates_per_frame,
        )
        if candidates and not has_detections and selected_source != "motion":
            candidate_gap_frames += 1

        reason = _motion_diagnostic_reason(
            selected_source, has_detections, mask_source, candidates, search_x, search_y, search_radius)
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

        should_export_frame = (
            selected_source == "motion" or
            (not has_detections and selected_source != "motion") or
            bool(candidates and not has_detections)
        )
        if should_export_frame:
            diagnostic_frames.append({
                "frame": int(frame_idx),
                "selected_source": selected_source,
                "reason": reason,
                "has_detections": bool(has_detections),
                "mask_source": mask_source,
                "has_boost_mask": boost is not None,
                "has_raw_motion": raw is not None,
                "search": {
                    "x": search_x,
                    "y": search_y,
                    "radius": float(search_radius),
                    "source": search_source,
                },
                "selected_position": {
                    "x": selected_x,
                    "y": selected_y,
                },
                "candidate_count_exported": int(len(candidates)),
                "candidates": candidates,
            })

        if selected_x is not None and selected_y is not None:
            prev_pos = (selected_x, selected_y)

    return {
        "schema_version": _SCHEMA_VERSION,
        "summary": {
            "frames_analyzed": int(total_frames),
            "detector_gap_frames": int(detector_gap_frames),
            "detector_gap_frames_with_motion_candidate": int(candidate_gap_frames),
            "reason_counts": dict(sorted(reason_counts.items())),
            "mask_source_counts": dict(sorted(mask_counts.items())),
            "selected_source_counts": dict(sorted(selected_source_counts.items())),
            "max_candidates_per_frame": int(max_candidates_per_frame),
        },
        "frames": diagnostic_frames,
    }


def write_tracking_json(ctx: RunContext, pass1: Pass1Result, result: RunResult) -> None:
    """Write ``cfg.tracking_json`` for this run: what ``result`` holds, plus the motion
    diagnostics from ``pass1`` when they were asked for."""
    cfg, video = ctx.cfg, ctx.video
    total_frames = result.frame_count
    filled = result.filled_frames
    out_path = Path(cfg.tracking_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    chosen = result.chosen_track
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "video": {
            "input": str(cfg.input_video),
            "output": str(cfg.output_video),
            "fps": float(video.fps),
            "width": int(video.width),
            "height": int(video.height),
            "total_frames": int(total_frames),
        },
        "summary": {
            "filled_frames": int(filled),
            "filled_percent": float(100.0 * filled / max(1, total_frames)),
            "elapsed_sec": float(result.elapsed_sec),
            "effective_fps": float(total_frames / max(result.elapsed_sec, 1e-9)),
            "pass2_frames_rendered": int(result.pass2_frames_rendered),
            "chosen_track_id": None if chosen is None else int(chosen.track_id),
            "track_count": int(len(result.all_tracks)),
        },
        "config": json_safe(vars(cfg)),
        "timing": json_safe(result.timing),
        "last_valid_court_keypoints": json_safe(result.last_valid_court_keypoints),
        "chosen_track": track_to_json(chosen),
        "tracks": [track_to_json(t) for t in result.all_tracks],
        "motion_diagnostics": (
            build_motion_diagnostics(
                result.per_frame, pass1.detections, pass1.boost_masks, pass1.raw_motions,
                int(video.width), int(video.height),
            )
            if cfg.motion_diagnostics else {}
        ),
        "frames": result.frame_rows(),
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
