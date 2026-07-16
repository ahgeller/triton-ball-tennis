"""Compact ball selector: tracks, evidence gates, bounded fill.

The detector/runtime owns candidate generation.  This module owns the final
trajectory and deliberately has no hidden handoff state.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from .config import SelectorConfig
from .models import FrameResult, Track
from .physics import _predict_projectile
from .scoring import _is_near_player, _select_timeline_chain, _stitch_track_chain, score_tracks
from .tracking import build_detections, build_tracks
from .utils import _ensure_mask_u8


def _selected_tracks(tracks: List[Track], fps: float) -> List[Track]:
    """Keep established shots, reacquisitions, and slow in-court rolls."""
    minimum_observations = max(3, int(round(float(fps) * 0.25)))
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


def _direction_cosine(first, second, third) -> float:
    first_dt = max(1, second.frame - first.frame)
    second_dt = max(1, third.frame - second.frame)
    ax = (second.cx - first.cx) / first_dt
    ay = (second.cy - first.cy) / first_dt
    bx = (third.cx - second.cx) / second_dt
    by = (third.cy - second.cy) / second_dt
    scale = math.hypot(ax, ay) * math.hypot(bx, by)
    return -1.0 if scale < 1e-6 else (ax * bx + ay * by) / scale


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


_REFINE_SOURCE_WEIGHT = {"det": 1.0, "motion": 0.30, "interp": 0.15, "carry": 0.10}


def _find_trajectory_kinks(x: np.ndarray, y: np.ndarray) -> List[int]:
    """Locate contact/bounce vertices so smoothing never rounds a real hit."""
    m = len(x)
    baseline = 3
    min_speed = 2.5
    max_cos = math.cos(math.radians(30.0))
    turns = np.zeros(m)
    for i in range(baseline, m - baseline):
        vin_x = (x[i] - x[i - baseline]) / baseline
        vin_y = (y[i] - y[i - baseline]) / baseline
        vout_x = (x[i + baseline] - x[i]) / baseline
        vout_y = (y[i + baseline] - y[i]) / baseline
        speed_in = math.hypot(vin_x, vin_y)
        speed_out = math.hypot(vout_x, vout_y)
        if speed_in < min_speed or speed_out < min_speed:
            continue
        cosine = (vin_x * vout_x + vin_y * vout_y) / (speed_in * speed_out)
        if cosine < max_cos:
            turns[i] = 1.0 - cosine
    kinks = []
    for i in range(baseline, m - baseline):
        if turns[i] > 0.0 and turns[i] == turns[max(0, i - baseline):i + baseline + 1].max():
            if not kinks or i - kinks[-1] > baseline:
                kinks.append(i)
    return kinks


def _find_spikes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Mark short detector excursions that immediately return to the path."""
    m = len(x)
    spikes = np.zeros(m, dtype=bool)
    if m < 4:
        return spikes
    steps = np.hypot(np.diff(x), np.diff(y))

    def local_step(i: int) -> float:
        lo = max(0, i - 5)
        hi = min(len(steps), i + 5)
        return float(np.median(steps[lo:hi])) if hi > lo else 0.0

    for length in (1, 2, 3):
        for i in range(1, m - length):
            a_x, a_y = x[i - 1], y[i - 1]
            b_x, b_y = x[i + length], y[i + length]
            gate = max(10.0, 3.0 * local_step(i))
            deviation = min(
                math.hypot(
                    x[i + j] - (a_x + (b_x - a_x) * (j + 1) / (length + 1)),
                    y[i + j] - (a_y + (b_y - a_y) * (j + 1) / (length + 1)),
                )
                for j in range(length)
            )
            if deviation > gate:
                spikes[i:i + length] = True
    return spikes


def _refine_run(results, start: int, end: int) -> None:
    """Robust weighted local polynomial refit of one continuous trajectory run."""
    points = results[start:end]
    m = len(points)
    x = np.array([item.cx for item in points])
    y = np.array([item.cy for item in points])
    weights = np.array([
        (1.0 if item.conf >= 0.70 else 0.6) if item.source == "det"
        else _REFINE_SOURCE_WEIGHT.get(item.source, 0.15)
        for item in points
    ])
    is_strong_det = np.array([
        item.source == "det" and item.conf >= 0.55 for item in points
    ])

    spikes = _find_spikes(x, y)
    weights[spikes] = 0.02
    is_strong_det[spikes] = False
    clean_x = x.copy()
    clean_y = y.copy()
    if spikes.any() and not spikes.all():
        good = np.flatnonzero(~spikes)
        bad = np.flatnonzero(spikes)
        clean_x[bad] = np.interp(bad, good, x[good])
        clean_y[bad] = np.interp(bad, good, y[good])

    kinks = _find_trajectory_kinks(clean_x, clean_y)
    boundaries = [0] + kinks + [m - 1]
    segment_lo = np.zeros(m, dtype=int)
    segment_hi = np.full(m, m - 1, dtype=int)
    for lo, hi in zip(boundaries, boundaries[1:]):
        segment_lo[lo:hi + 1] = lo
        segment_hi[lo:hi + 1] = hi
    for kink in kinks:
        segment_lo[kink] = kink
        segment_hi[kink] = kink

    window = 10
    huber_px = 4.0
    new_x = x.copy()
    new_y = y.copy()
    for i in range(m):
        lo = max(int(segment_lo[i]), i - window)
        hi = min(int(segment_hi[i]), i + window)
        count = hi - lo + 1
        if count < 3:
            continue
        tt = np.arange(lo, hi + 1, dtype=np.float64) - i
        xx = x[lo:hi + 1]
        yy = y[lo:hi + 1]
        base_w = weights[lo:hi + 1].copy()
        if base_w.sum() <= 0.0:
            continue
        degree = 2 if count >= 6 else 1
        robust = np.ones(count)
        fit_x = fit_y = None
        for _ in range(2):
            sqrt_w = np.sqrt(base_w * robust)
            design = np.vander(tt, degree + 1)
            lhs = design * sqrt_w[:, None]
            rhs = np.stack([xx * sqrt_w, yy * sqrt_w], axis=1)
            coeffs, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
            fitted = design @ coeffs
            residual = np.hypot(fitted[:, 0] - xx, fitted[:, 1] - yy)
            robust = np.minimum(1.0, huber_px / np.maximum(residual, 1e-9))
            fit_x, fit_y = coeffs[-1, 0], coeffs[-1, 1]
        local = i - lo
        if spikes[i]:
            bound = None
        elif is_strong_det[i]:
            bound = 3.0
            det_mask = base_w >= 0.6
            det_mask[local] = False
            if det_mask.any():
                other_residual = float(np.median(residual[det_mask]))
                if residual[local] > max(9.0, 4.0 * other_residual):
                    bound = None
        elif points[i].source == "motion":
            bound = 6.0
        else:
            bound = 12.0
        dx = fit_x - x[i]
        dy = fit_y - y[i]
        distance = math.hypot(dx, dy)
        if bound is not None and distance > bound:
            scale = bound / distance
            dx *= scale
            dy *= scale
        new_x[i] = x[i] + dx
        new_y[i] = y[i] + dy

    for index, item in enumerate(points):
        item.cx = float(new_x[index])
        item.cy = float(new_y[index])


def _refine_trajectory(results) -> None:
    frame = 0
    while frame < len(results):
        if results[frame] is None:
            frame += 1
            continue
        start = frame
        while frame < len(results) and results[frame] is not None:
            frame += 1
        if frame - start >= 9:
            _refine_run(results, start, frame)


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
        for index, (left, right) in enumerate(zip(observations, observations[1:])):
            gap = right.frame - left.frame
            if gap <= 1 or gap - 1 > fill_frames:
                continue
            speed = math.hypot(right.cx - left.cx, right.cy - left.cy) / gap
            minimum_speed = 10.0 * 30.0 / max(float(fps), 1.0)
            direction_supported = (
                index > 0
                and _direction_cosine(observations[index - 1], left, right) >= 0.8
            ) or (
                index + 2 < len(observations)
                and _direction_cosine(left, right, observations[index + 2]) >= 0.8
            )
            strong_anchors = left.conf >= 0.5 and right.conf >= 0.5
            pair_supported = (
                strong_anchors
                and direction_supported
                and (gap == 2 or speed >= minimum_speed)
            )
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
                motion_supported = (
                    motion is not None
                    and (direction_supported or gap == 2)
                    and (gap == 2 or speed >= minimum_speed)
                )
                if (
                    _is_near_player(base_x, base_y, player_boxes_by_frame, frame, margin=0)
                    and speed < 15.0 * 30.0 / max(float(fps), 1.0)
                ):
                    pair_supported = False
                    motion_supported = False
                if not pair_supported and not motion_supported:
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
        top_reentry = (
            dt > fill_frames
            and left.cy < 0.05 * height
            and right.cy < 0.15 * height
        )
        if not top_reentry and math.hypot(vx, vy) < 10.0 * 30.0 / max(float(fps), 1.0):
            continue
        # A tail cannot help association here; tracks were already built above.
        # Emit it only while every current frame contains nearby strict motion.
        reentry_x, reentry_y = right.cx, right.cy
        for step in range(1, fill_frames + 1):
            frame = right.frame + step
            if frame >= min(total, next_start):
                break
            if top_reentry:
                base_x, base_y = reentry_x, reentry_y
            else:
                base_x, base_y = _predict_projectile(
                    (right.cx, right.cy),
                    (vx, vy),
                    step,
                    cfg,
                )
            motion = _motion_candidate(
                frame, base_x, base_y, right.area, boost_masks, raw_motions,
                gate=motion_snap_gate,
                area_ratio_max=4.0,
            )
            if motion is None:
                break
            if results[frame] is None:
                results[frame] = _result(motion["x"], motion["y"], "motion", motion=motion)
            if top_reentry:
                reentry_x, reentry_y = motion["x"], motion["y"]

    # One missing source frame between strong, fast detector anchors is safer
    # to close than leaving the bottom of a bounce visibly disconnected.
    minimum_speed = 10.0 * 30.0 / max(float(fps), 1.0)
    for frame in range(1, total - 1):
        left, right = results[frame - 1], results[frame + 1]
        speed = (
            math.hypot(right.cx - left.cx, right.cy - left.cy) / 2.0
            if left is not None and right is not None
            else 0.0
        )
        if (
            results[frame] is None
            and left is not None and right is not None
            and left.source == right.source == "det"
            and left.conf >= 0.5 and right.conf >= 0.5
            and minimum_speed <= speed <= 0.015 * cfg.diag
            and not _is_near_player(
                0.5 * (left.cx + right.cx),
                0.5 * (left.cy + right.cy),
                player_boxes_by_frame,
                frame,
                margin=0,
            )
        ):
            results[frame] = _result(
                0.5 * (left.cx + right.cx),
                0.5 * (left.cy + right.cy),
                "interp",
            )

    _refine_trajectory(results)

    if debug:
        print(f"[clean selector] selected tracks: {[track.track_id for track in selected]}")
    # ponytail: one missing row is safer than connecting two balls across the court.
    jump_gate = max(120.0, 0.055 * cfg.diag)
    previous = None
    for frame, result in enumerate(results):
        if (
            result is not None
            and result.source != "det"
            and not (0.0 <= result.cx < width and 0.0 <= result.cy < height)
        ):
            results[frame] = None
            result = None
        if result is not None and previous is not None:
            if (
                result.source != "det"
                and math.hypot(result.cx - previous.cx, result.cy - previous.cy) > jump_gate
            ):
                results[frame] = None
                result = None
        previous = result
    return results, chosen, tracks, []
