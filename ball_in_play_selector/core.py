"""Compact ball selector: tracks, evidence gates, bounded fill.

The detector/runtime owns candidate generation.  This module owns the final
trajectory and deliberately has no hidden handoff state.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import cv2

from .config import SelectorConfig
from .models import FrameResult, Track
from .physics import _predict_projectile
from .scoring import _select_timeline_chain, _stitch_track_chain, score_tracks
from .tracking import build_detections, build_tracks
from .utils import _ensure_mask_u8


def _selected_tracks(tracks: List[Track], fps: float) -> List[Track]:
    """Keep established shots, reacquisitions, and slow in-court rolls."""
    minimum_observations = max(3, int(round(float(fps) / 3.0)))
    reacquire_observations = max(10, int(round(float(fps) * 0.16)))
    selected = []
    for track in tracks:
        score = track.score_breakdown or {}
        motion_fraction = float(score.get("motion_frac_raw", 0.0))
        confidences = sorted(float(item.conf) for item in track.observations)
        median_confidence = confidences[len(confidences) // 2] if confidences else 0.0
        score_density = track.score / max(track.num_obs, 1)
        established = (
            track.num_obs >= minimum_observations
            and motion_fraction >= 0.5
            and score_density >= 0.5
        )
        reacquired = (
            track.num_obs >= reacquire_observations
            and motion_fraction >= 0.25
            and median_confidence >= 0.55
            and float(score.get("inside_strict_frac", 1.0)) <= 0.1
            and score_density >= 0.5
        )
        rolling = (
            track.num_obs >= minimum_observations
            and motion_fraction >= 0.15
            and median_confidence >= 0.80
            and float(score.get("inside_strict_frac", 0.0)) >= 0.5
            and float(score.get("extent_px", 0.0)) >= 0.02 * track.cfg.diag
            and score_density >= 0.12
        )
        if established or reacquired or rolling:
            selected.append(track)
    return sorted(selected, key=lambda track: (track.first_frame, track.last_obs_frame))


def _trajectory_observations(track: Track, fps: float):
    """Drop lone weak detections that cannot anchor a visible trajectory."""
    observations = track.observations
    neighbor_frames = max(2, int(round(float(fps) * 0.05)))
    kept = []
    for index, detection in enumerate(observations):
        previous_gap = (
            detection.frame - observations[index - 1].frame
            if index else neighbor_frames + 1
        )
        next_gap = (
            observations[index + 1].frame - detection.frame
            if index + 1 < len(observations) else neighbor_frames + 1
        )
        if (
            detection.conf < 0.5
            and not detection.on_motion
            and previous_gap > neighbor_frames
            and next_gap > neighbor_frames
        ):
            continue
        kept.append(detection)
    return kept


def _motion_near(
    mask_obj: Any,
    x: float,
    y: float,
    reference_area: float,
    gate: float,
    area_ratio_max: float,
) -> Optional[Dict[str, Any]]:
    mask = _ensure_mask_u8(mask_obj)
    if mask is None or not cv2.countNonZero(mask):
        return None

    best = None
    best_distance = float(gate) + 1.0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area <= 0.0 or not 0.2 * reference_area <= area <= area_ratio_max * reference_area:
            continue
        moments = cv2.moments(contour)
        if not moments["m00"]:
            continue
        cx = float(moments["m10"] / moments["m00"])
        cy = float(moments["m01"] / moments["m00"])
        distance = math.hypot(cx - x, cy - y)
        if distance <= gate and distance < best_distance:
            bx, by, bw, bh = cv2.boundingRect(contour)
            best = {"x": cx, "y": cy, "bbox": (bx, by, bx + bw, by + bh), "area": area}
            best_distance = distance
    return best


def _motion_candidate(
    frame: int,
    x: float,
    y: float,
    area: float,
    boost_masks,
    raw_motions,
    gate: float,
    area_ratio_max: float,
):
    for masks in (boost_masks, raw_motions):
        if masks is None or frame >= len(masks):
            continue
        candidate = _motion_near(masks[frame], x, y, max(area, 1.0), gate, area_ratio_max)
        if candidate is not None:
            return candidate
    return None


def _result(x, y, source, detection=None, motion=None) -> FrameResult:
    bbox = None
    confidence = 0.0
    if detection is not None:
        bbox = (detection.x1, detection.y1, detection.x2, detection.y2)
        confidence = float(detection.conf)
    elif motion is not None:
        bbox = motion["bbox"]
    return FrameResult(
        cx=float(x),
        cy=float(y),
        conf=confidence,
        bbox=bbox,
        interpolated=source != "det",
        source=source,
        search_cx=float(x),
        search_cy=float(y),
        source_reason={
            "det": "selected_track_observation",
            "motion": "bounded_motion_correction",
            "interp": "bounded_linear_fill",
            "carry": "evidence_bounded_tail",
        }[source],
    )


def select_ball_in_play(
    detections_by_frame,
    fps,
    width,
    height,
    court_polygon=None,
    boost_masks=None,
    raw_motions=None,
    player_boxes_by_frame=None,
    court_keypoints=None,
    debug=False,
    **_,
):
    """Return a complete, schema-compatible trajectory selection."""
    total = len(detections_by_frame)
    cfg = SelectorConfig(fps=fps, width=width, height=height).auto_scale()
    resolution_scale = cfg.diag / 2203.0
    fill_frames = max(3, int(round(float(fps) * 0.30)))
    motion_gate = max(12.0, 40.0 * resolution_scale)
    motion_snap_gate = max(6.0, 13.0 * resolution_scale)
    detections = build_detections(detections_by_frame, boost_masks)
    tracks = score_tracks(
        build_tracks(detections, cfg),
        cfg,
        court_poly=court_polygon,
        player_boxes_by_frame=player_boxes_by_frame,
        total_frames=total,
    )
    selected = _select_timeline_chain(_selected_tracks(tracks, fps), cfg, total)
    chosen = _stitch_track_chain(selected, cfg)
    selected_observations = {
        id(track): _trajectory_observations(track, fps) for track in selected
    }
    if chosen is not None:
        chosen.observations = _trajectory_observations(chosen, fps)
    results: List[Optional[FrameResult]] = [None] * total

    # Stronger tracks win overlapping detector observations.
    for track in sorted(selected, key=lambda item: item.score, reverse=True):
        for detection in selected_observations[id(track)]:
            current = results[detection.frame]
            if current is None or detection.conf > current.conf:
                results[detection.frame] = _result(
                    detection.cx, detection.cy, "det", detection=detection
                )

    # Fill internal paths only when a future real detection anchors the endpoint.
    for track in sorted(selected, key=lambda item: item.score, reverse=True):
        observations = selected_observations[id(track)]
        for left, right in zip(observations, observations[1:]):
            gap = right.frame - left.frame
            if gap <= 1 or gap - 1 > fill_frames:
                continue
            strong_anchors = left.conf >= 0.55 and right.conf >= 0.55
            offset_x = offset_y = 0.0
            offset_active = False
            for frame in range(left.frame + 1, right.frame):
                if results[frame] is not None:
                    continue
                alpha = (frame - left.frame) / float(gap)
                base_x = left.cx + alpha * (right.cx - left.cx)
                base_y = left.cy + alpha * (right.cy - left.cy)
                base_y += (
                    0.5
                    * cfg.gravity_px_per_frame2
                    * (frame - left.frame)
                    * (frame - right.frame)
                )
                area = left.area + alpha * (right.area - left.area)
                motion = _motion_candidate(
                    frame, base_x, base_y, area, boost_masks, raw_motions,
                    gate=motion_snap_gate,
                    area_ratio_max=4.0,
                )
                if motion is None and not strong_anchors:
                    offset_x = offset_y = 0.0
                    offset_active = False
                    continue
                if motion is not None:
                    target_x = 0.75 * (motion["x"] - base_x)
                    target_y = 0.75 * (motion["y"] - base_y)
                    if offset_active:
                        offset_x = 0.5 * (offset_x + target_x)
                        offset_y = 0.5 * (offset_y + target_y)
                    else:
                        offset_x, offset_y = target_x, target_y
                        offset_active = True
                elif offset_active:
                    offset_x *= 0.5
                    offset_y *= 0.5
                endpoint_weight = 0.5 if frame == right.frame - 1 else 1.0
                x = base_x + endpoint_weight * offset_x
                y = base_y + endpoint_weight * offset_y
                results[frame] = _result(
                    x, y, "motion" if motion else "interp", motion=motion
                )

    # Continue a segment only after nearby motion proves the tail still exists.
    for track in sorted(selected, key=lambda item: item.score, reverse=True):
        observations = selected_observations[id(track)]
        if len(observations) < 2:
            continue
        left, right = observations[-2:]
        dt = max(1, right.frame - left.frame)
        vx = (right.cx - left.cx) / dt
        vy = (right.cy - left.cy) / dt
        next_track = min(
            (other for other in selected if other.first_frame > right.frame),
            key=lambda other: other.first_frame,
            default=None,
        )
        next_start = next_track.first_frame if next_track is not None else total
        if next_track is not None and next_start <= right.frame + fill_frames + 1:
            next_dt = next_start - right.frame
            predicted_x, predicted_y = _predict_projectile(
                (right.cx, right.cy),
                (vx, vy),
                next_dt,
                cfg,
            )
            next_observation = next_track.observations[0]
            if math.hypot(
                predicted_x - next_observation.cx,
                predicted_y - next_observation.cy,
            ) > max(motion_gate, 0.055 * cfg.diag):
                continue
        offset_x = offset_y = 0.0
        offset_active = False
        # A tail cannot help association here; tracks were already built above.
        # Emit it only while the current frame still contains nearby motion.
        for step in range(1, fill_frames + 1):
            frame = right.frame + step
            if frame >= min(total, next_start):
                break
            base_x, base_y = _predict_projectile(
                (right.cx, right.cy),
                (vx, vy),
                step,
                cfg,
            )
            evidence = _motion_candidate(
                frame, base_x, base_y, right.area, boost_masks, raw_motions,
                gate=max(motion_gate, 0.5 * motion_gate + 2.0 * resolution_scale * step),
                area_ratio_max=12.0,
            )
            if evidence is None:
                offset_x = offset_y = 0.0
                offset_active = False
                continue
            if results[frame] is not None:
                continue
            motion = _motion_candidate(
                frame, base_x, base_y, right.area, boost_masks, raw_motions,
                gate=motion_snap_gate,
                area_ratio_max=4.0,
            )
            if motion is not None:
                target_x = 0.75 * (motion["x"] - base_x)
                target_y = 0.75 * (motion["y"] - base_y)
                if offset_active:
                    offset_x = 0.5 * (offset_x + target_x)
                    offset_y = 0.5 * (offset_y + target_y)
                else:
                    offset_x = 0.5 * target_x
                    offset_y = 0.5 * target_y
                    offset_active = True
            elif offset_active:
                offset_x *= 0.5
                offset_y *= 0.5
            x = base_x + offset_x
            y = base_y + offset_y
            results[frame] = _result(
                x, y, "motion" if motion else "carry", motion=motion
            )

    if debug:
        print(f"[clean selector] selected tracks: {[track.track_id for track in selected]}")
    # ponytail: one missing row is safer than connecting two balls across the court.
    jump_gate = max(120.0, 0.055 * cfg.diag)
    previous = None
    for frame, result in enumerate(results):
        if result is not None and not (0.0 <= result.cx < width and 0.0 <= result.cy < height):
            results[frame] = None
            result = None
        if result is not None and previous is not None:
            if math.hypot(result.cx - previous.cx, result.cy - previous.cy) > jump_gate:
                results[frame] = None
                result = None
        previous = result
    return results, chosen, tracks, []
