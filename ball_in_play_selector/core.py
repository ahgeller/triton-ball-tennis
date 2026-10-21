"""Compact ball selector: tracks, evidence gates, bounded fill.

The detector/runtime owns candidate generation.  This module owns the final
trajectory and deliberately has no hidden handoff state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import REFERENCE_DIAG_PX, SelectorConfig
from .models import Detection, FrameResult, RawDetection, Source, Track
from .physics import predict_projectile
from .scoring import is_near_player, score_tracks, select_timeline_chain, stitch_track_chain
from .association import build_detections, build_tracks
from .utils import MaskLike, PlayerBoxesByFrame, ensure_mask_u8, pxpf_at_fps

# -- Tuned constants (30 FPS, 1080p reference unless stated) ----------------
# Longest internal gap, and longest tail, the selector will fill: 0.30 s.
_FILL_WINDOW_SEC = 0.30
_FILL_WINDOW_MIN_FRAMES = 3
# A fill is only plausible for a ball moving at least this fast (px/frame at
# 30 FPS); slower "gaps" are a static object the detector flickers on.
_MIN_FILL_SPEED_PXPF = 10.0
# Beside a player a fill needs a faster ball still: a slow point there is
# usually the racket or a hand.
_NEAR_PLAYER_FILL_SPEED_PXPF = 15.0
# Incoming and outgoing segments must agree in direction this well (cosine)
# for a straight fill to be believable.
_FILL_DIRECTION_COS_MIN = 0.8
# Detector confidence both anchors of a fill must reach.
_STRONG_ANCHOR_CONF = 0.5
# A motion blob may be at most 4x the interpolated detection area.
_FILL_AREA_RATIO_MAX = 4.0
# How far a blob pulls the interpolated point towards itself, how fast that
# pull decays on blobless frames, and how much of it the frame before the
# right anchor keeps (the anchor itself is exact).
_BLOB_PULL = 0.75
_BLOB_PULL_DECAY = 0.5
_ENDPOINT_PULL_WEIGHT = 0.5
# A tail may run into the next track only if its projectile prediction lands
# within 5.5% of the diagonal (~121 px at 1080p) of that track's start.
_HANDOVER_GATE_DIAG_FRAC = 0.055
# A ball that left over the top edge (last two points in the top 5% / 15% of
# the frame) is waited for at its exit point instead of being projected.
_TOP_EXIT_FRAC = 0.05
_TOP_REENTRY_FRAC = 0.15
# One missing frame between two strong detections is closed when the implied
# step is below 1.5% of the diagonal per frame.
_SINGLE_HOLE_MAX_STEP_DIAG_FRAC = 0.015
# A non-detector point further than max(120 px, 5.5% of the diagonal) from
# its predecessor is dropped: better one missing row than a cross-court line.
_JUMP_GATE_MIN_PX = 120.0
_JUMP_GATE_DIAG_FRAC = 0.055
# Shortest continuous run the trajectory refit will touch.
_MIN_REFIT_RUN = 9
# Refit weight per source: a detector point anchors the fit (a weak one a
# little less), a motion blob is a coarse observation, a fitted fill carries
# almost no information of its own, and a spike almost none at all.
_REFINE_SOURCE_WEIGHT = {"det": 1.0, "motion": 0.30, "interp": 0.15}
_REFINE_WEAK_DET_WEIGHT = 0.6
_REFINE_SPIKE_WEIGHT = 0.02
# Detector confidence that makes a point a full-weight anchor, and the one
# past which it counts as a strong detection whose residual bounds the refit.
_REFINE_FULL_WEIGHT_CONF = 0.70
_REFINE_STRONG_DET_CONF = 0.55
# Every point is fitted over at most this many neighbours on each side, with a
# Huber loss that down-weights residuals beyond this many pixels.
_REFINE_WINDOW = 10
_REFINE_HUBER_PX = 4.0
# How far the refit may move a point: a strong detection, a motion blob, a fill.
_REFIT_BOUND_STRONG_DET_PX = 3.0
_REFIT_BOUND_MOTION_PX = 6.0
_REFIT_BOUND_FILL_PX = 12.0
# A strong detection whose residual exceeds max(9 px, 4x the median residual
# of the other detector points) is off the path, so its move is unbounded.
_REFIT_OUTLIER_MIN_PX = 9.0
_REFIT_OUTLIER_RESIDUAL_MULT = 4.0
# Past this much correction a "det" point is no longer where the detector
# fired, so it stops being reported as one (~44 px at 1080p).
_REFINE_DET_RELABEL_FRAC = 0.02


def selected_tracks(tracks: List[Track]) -> List[Track]:
    """Keep established shots, reacquisitions, airborne balls, and slow in-court rolls.

    Every threshold comes from each track's own ``cfg`` (already scaled to the
    clip's frame rate and resolution).
    """
    selected = []
    for track in tracks:
        cfg = track.cfg
        # Fraction of observations on the motion mask.
        motion_overlap = track.score_value("motion_overlap")
        # max(mask overlap, kinematic estimate) -- see scoring.score_tracks
        motion_evidence = track.score_value("motion_evidence")
        confidences = sorted(float(item.conf) for item in track.observations)
        median_confidence = confidences[len(confidences) // 2] if confidences else 0.0
        score_density = track.score / max(track.num_obs, 1)
        # A ball the detector can see is seen on most frames; ten detections
        # sprinkled over seventy frames is a bystander, not a reacquisition.
        observation_density = track.num_obs / max(track.span, 1)
        inside_strict = track.score_value("inside_strict_frac")
        extent_px = track.score_value("extent_px")
        established = (
            track.num_obs >= cfg.select_min_obs_frames
            and motion_overlap >= cfg.select_established_motion_frac
            and score_density >= cfg.select_score_density_min
        )
        reacquired = (
            track.num_obs >= cfg.select_reacquire_obs_frames
            and motion_overlap >= cfg.select_reacquire_motion_frac
            and median_confidence >= cfg.select_reacquire_conf
            and inside_strict <= cfg.select_outside_strict_max
            and score_density >= cfg.select_score_density_min
            and observation_density >= cfg.select_obs_density_min
        )
        # A ball in flight is unmistakable from its own kinematics even when the
        # boost mask barely fires on it (small, distant, or far-court), and it
        # crosses the baseline, so it sits in neither court-fraction band above.
        airborne = (
            track.num_obs >= cfg.select_min_obs_frames
            and motion_evidence >= cfg.select_kinematic_motion_frac
            and median_confidence >= cfg.select_reacquire_conf
            and extent_px >= cfg.select_kinematic_extent_frac * cfg.diag
            and score_density >= cfg.select_score_density_min
            and observation_density >= cfg.select_obs_density_min
        )
        rolling = (
            track.num_obs >= cfg.select_min_obs_frames
            and motion_overlap >= cfg.select_rolling_motion_frac
            and median_confidence >= cfg.select_rolling_conf
            and inside_strict >= cfg.select_inside_strict_min
            and extent_px >= cfg.select_rolling_extent_frac * cfg.diag
            and score_density >= cfg.select_rolling_score_density_min
        )
        if established or reacquired or airborne or rolling:
            selected.append(track)
    return sorted(selected, key=lambda track: (track.first_frame, track.last_obs_frame))


def _isolation_frames(fps: float) -> int:
    """Gap beyond which one detection has no same-track corroboration (~0.1 s)."""
    return max(_neighbor_frames(fps) + 1, int(round(float(fps) * 0.1)))


def _neighbor_frames(fps: float) -> int:
    return max(2, int(round(float(fps) * 0.05)))


def trajectory_observations(track: Track, fps: float) -> List[Detection]:
    """Drop lone detections that cannot anchor a visible trajectory.

    A weak, motionless detection needs a neighbour within ~0.05 s.  Any
    detection separated from the rest of its track by more than ~0.1 s on
    both sides is unverifiable: association can reconnect a static object
    (a van, a shoe) after a long miss, and a lone endpoint like that must
    not draw a path or a dot.

    One exception, because this rule was eating the very anchors the long-gap
    fill then demands as corroboration: a detection that is both confident and
    sitting on the motion mask is corroborated by the mask rather than by its
    neighbours, so it survives.
    """
    observations = track.observations
    cfg = track.cfg
    neighbor_frames = _neighbor_frames(fps)
    isolation_frames = _isolation_frames(fps)
    kept = []
    for index, detection in enumerate(observations):
        has_previous = index > 0
        has_next = index + 1 < len(observations)
        previous_gap = (
            detection.frame - observations[index - 1].frame
            if has_previous else neighbor_frames + 1
        )
        next_gap = (
            observations[index + 1].frame - detection.frame
            if has_next else neighbor_frames + 1
        )
        if (
            detection.conf < 0.5
            and not detection.on_motion
            and previous_gap > neighbor_frames
            and next_gap > neighbor_frames
        ):
            continue
        corroborated_by_mask = (
            detection.conf >= cfg.isolation_keep_conf and detection.on_motion
        )
        previous_isolated = has_previous and previous_gap > isolation_frames
        next_isolated = has_next and next_gap > isolation_frames
        isolated = (
            (previous_isolated or not has_previous)
            and (next_isolated or not has_next)
            and (has_previous or has_next)
        )
        if isolated and not corroborated_by_mask and len(observations) > 1:
            continue
        kept.append(detection)
    return kept


class MotionBlob(NamedTuple):
    """A connected component of a motion mask: its centroid, xyxy box and pixel area."""

    x: float
    y: float
    bbox: Tuple[int, int, int, int]
    area: float


def motion_near(
    mask_obj: Optional[MaskLike],
    x: float,
    y: float,
    reference_area: float,
    gate: float,
    area_ratio_max: float,
) -> Optional[MotionBlob]:
    """Nearest motion component within `gate` px of (x, y), or None.

    Returns the closest component whose pixel area lies in
    ``[2, area_ratio_max * reference_area]``.
    """
    mask = ensure_mask_u8(mask_obj)
    if mask is None or not cv2.countNonZero(mask):
        return None

    best = None
    best_distance = float(gate) + 1.0
    count, _, stats, centers = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for index in range(1, count):
        bx, by, bw, bh, pixels = stats[index]
        area = float(pixels)
        # GridTrackNet's bbox is synthetic, not a measured ball diameter.
        # Keep small components; location/color/temporal evidence supplies the gate.
        if not 2 <= area <= area_ratio_max * reference_area:
            continue
        cx, cy = map(float, centers[index])
        distance = math.hypot(cx - x, cy - y)
        if distance <= gate and distance < best_distance:
            best = MotionBlob(cx, cy, (int(bx), int(by), int(bx + bw), int(by + bh)), area)
            best_distance = distance
    return best


def _motion_search_radius(cfg: SelectorConfig, steps: int, speed: float) -> float:
    """How far from the predicted point a motion blob may still be the ball.

    Grows with the number of frames the prediction has run unanchored and with
    the ball's own speed: a 3% prediction error on an 80 px/frame ball is
    already outside any fixed radius that is safe for a slow one.
    """
    base = max(cfg.motion_search_min_px, cfg.motion_search_base_frac * cfg.diag)
    radius = (
        base
        + cfg.motion_search_growth_px * max(0, steps - 1)
        + cfg.motion_search_vel_mult * max(0.0, speed)
    )
    return min(radius, cfg.motion_search_max_frac * cfg.diag)


def _motion_candidate(
    frame: int,
    x: float,
    y: float,
    area: float,
    boost_masks: Optional[Sequence[Optional[MaskLike]]],
    raw_motions: Optional[Sequence[Optional[MaskLike]]],
    gate: float,
    area_ratio_max: float,
) -> Optional[MotionBlob]:
    for masks in (boost_masks, raw_motions):
        if masks is None or frame >= len(masks):
            continue
        candidate = motion_near(masks[frame], x, y, max(area, 1.0), gate, area_ratio_max)
        if candidate is not None:
            return candidate
    return None


def _segment_speed(first: Detection, second: Detection) -> float:
    return math.hypot(second.cx - first.cx, second.cy - first.cy) / max(1, second.frame - first.frame)


def _segment_cosine(a: Detection, b: Detection, c: Detection, d: Detection) -> float:
    """Cosine between the incoming segment a->b and the outgoing segment c->d."""
    ax, ay = b.cx - a.cx, b.cy - a.cy
    bx, by = d.cx - c.cx, d.cy - c.cy
    scale = math.hypot(ax, ay) * math.hypot(bx, by)
    return 0.0 if scale < 1e-6 else (ax * bx + ay * by) / scale


def direction_cosine(first: Detection, second: Detection, third: Detection) -> float:
    """Cosine between the per-frame velocities first->second and second->third."""
    first_dt = max(1, second.frame - first.frame)
    second_dt = max(1, third.frame - second.frame)
    ax = (second.cx - first.cx) / first_dt
    ay = (second.cy - first.cy) / first_dt
    bx = (third.cx - second.cx) / second_dt
    by = (third.cy - second.cy) / second_dt
    scale = math.hypot(ax, ay) * math.hypot(bx, by)
    return -1.0 if scale < 1e-6 else (ax * bx + ay * by) / scale


_SOURCE_REASON: Dict[str, str] = {
    "det": "selected_track_observation",
    "motion": "bounded_motion_correction",
    "interp": "bounded_linear_fill",
}


def frame_result(
    x: float,
    y: float,
    source: Source,
    detection: Optional[Detection] = None,
    motion: Optional[MotionBlob] = None,
    search_radius: float = 0.0,
) -> FrameResult:
    """Build one output row; `detection` or `motion` supplies bbox/confidence."""
    bbox = None
    confidence = 0.0
    if detection is not None:
        bbox = (detection.x1, detection.y1, detection.x2, detection.y2)
        confidence = float(detection.conf)
    elif motion is not None:
        bbox = motion.bbox
    return FrameResult(
        cx=float(x),
        cy=float(y),
        conf=confidence,
        bbox=bbox,
        interpolated=source != "det",
        source=source,
        measurement=(float(detection.cx), float(detection.cy)) if detection is not None else None,
        search_cx=float(x),
        search_cy=float(y),
        # Reported so the pipeline diagnostics judge a blob against the radius
        # the selector actually used, not an unrelated one of their own.
        search_radius=float(search_radius),
        source_reason=_SOURCE_REASON[source],
    )


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


def _refit_bound(
    index: int,
    spikes: np.ndarray,
    is_strong_det: np.ndarray,
    source: str,
    residual: np.ndarray,
    local: int,
    base_w: np.ndarray,
) -> Optional[float]:
    """How far the refit may move point `index`; None means unbounded."""
    if spikes[index]:
        return None
    if is_strong_det[index]:
        det_mask = base_w >= _REFINE_WEAK_DET_WEIGHT     # the detector points of the window
        det_mask[local] = False
        if det_mask.any():
            other_residual = float(np.median(residual[det_mask]))
            if residual[local] > max(_REFIT_OUTLIER_MIN_PX, _REFIT_OUTLIER_RESIDUAL_MULT * other_residual):
                return None
        return _REFIT_BOUND_STRONG_DET_PX
    if source == "motion":
        return _REFIT_BOUND_MOTION_PX
    return _REFIT_BOUND_FILL_PX


def _refine_weights(points: Sequence[FrameResult]) -> Tuple[np.ndarray, np.ndarray]:
    """Per-point fit weights and the strong-detection flags, before spikes are marked."""
    weights = np.array([
        (1.0 if item.conf >= _REFINE_FULL_WEIGHT_CONF else _REFINE_WEAK_DET_WEIGHT) if item.source == "det"
        else _REFINE_SOURCE_WEIGHT[item.source]
        for item in points
    ])
    is_strong_det = np.array([
        item.source == "det" and item.conf >= _REFINE_STRONG_DET_CONF for item in points
    ])
    return weights, is_strong_det


def _despiked(x: np.ndarray, y: np.ndarray, spikes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """The path with the spike points linearly bridged, for finding kinks."""
    clean_x = x.copy()
    clean_y = y.copy()
    if spikes.any() and not spikes.all():
        good = np.flatnonzero(~spikes)
        bad = np.flatnonzero(spikes)
        clean_x[bad] = np.interp(bad, good, x[good])
        clean_y[bad] = np.interp(bad, good, y[good])
    return clean_x, clean_y


def _segment_bounds(m: int, kinks: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Per point, the first and last index of its segment; a kink is a segment of its own."""
    boundaries = [0] + list(kinks) + [m - 1]
    segment_lo = np.zeros(m, dtype=int)
    segment_hi = np.full(m, m - 1, dtype=int)
    for lo, hi in zip(boundaries, boundaries[1:]):
        segment_lo[lo:hi + 1] = lo
        segment_hi[lo:hi + 1] = hi
    for kink in kinks:
        segment_lo[kink] = kink
        segment_hi[kink] = kink
    return segment_lo, segment_hi


def _local_fit(
    tt: np.ndarray, xx: np.ndarray, yy: np.ndarray, base_w: np.ndarray, degree: int
) -> Tuple[float, float, np.ndarray]:
    """Two rounds of a Huber-reweighted weighted polynomial fit; the fitted
    position at ``tt == 0`` and the last round's residuals."""
    count = len(tt)
    robust = np.ones(count)
    fit_x = fit_y = 0.0
    residual = np.zeros(count)
    for _ in range(2):
        sqrt_w = np.sqrt(base_w * robust)
        design = np.vander(tt, degree + 1)
        lhs = design * sqrt_w[:, None]
        rhs = np.stack([xx * sqrt_w, yy * sqrt_w], axis=1)
        coeffs, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
        fitted = design @ coeffs
        residual = np.hypot(fitted[:, 0] - xx, fitted[:, 1] - yy)
        robust = np.minimum(1.0, _REFINE_HUBER_PX / np.maximum(residual, 1e-9))
        fit_x, fit_y = coeffs[-1, 0], coeffs[-1, 1]
    return fit_x, fit_y, residual


def _refit_positions(
    points: Sequence[FrameResult], x: np.ndarray, y: np.ndarray, weights: np.ndarray,
    is_strong_det: np.ndarray, spikes: np.ndarray, kinks: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Every point refitted within its segment and window, moved at most its bound."""
    m = len(points)
    segment_lo, segment_hi = _segment_bounds(m, kinks)
    new_x = x.copy()
    new_y = y.copy()
    for i in range(m):
        lo = max(int(segment_lo[i]), i - _REFINE_WINDOW)
        hi = min(int(segment_hi[i]), i + _REFINE_WINDOW)
        count = hi - lo + 1
        if count < 3:
            continue
        base_w = weights[lo:hi + 1].copy()
        if base_w.sum() <= 0.0:
            continue
        tt = np.arange(lo, hi + 1, dtype=np.float64) - i
        fit_x, fit_y, residual = _local_fit(tt, x[lo:hi + 1], y[lo:hi + 1], base_w, 2 if count >= 6 else 1)
        local = i - lo
        bound = _refit_bound(i, spikes, is_strong_det, points[i].source, residual, local, base_w)
        dx = fit_x - x[i]
        dy = fit_y - y[i]
        distance = math.hypot(dx, dy)
        if bound is not None and distance > bound:
            scale = bound / distance
            dx *= scale
            dy *= scale
        new_x[i] = x[i] + dx
        new_y[i] = y[i] + dy
    return new_x, new_y


def _refine_run(results: List[FrameResult], start: int, end: int, diag: float = REFERENCE_DIAG_PX) -> None:
    """Robust weighted local polynomial refit of one continuous trajectory run."""
    points = results[start:end]
    x = np.array([item.cx for item in points])
    y = np.array([item.cy for item in points])
    weights, is_strong_det = _refine_weights(points)
    spikes = _find_spikes(x, y)
    weights[spikes] = _REFINE_SPIKE_WEIGHT
    is_strong_det[spikes] = False
    kinks = _find_trajectory_kinks(*_despiked(x, y, spikes))
    new_x, new_y = _refit_positions(points, x, y, weights, is_strong_det, spikes, kinks)

    # Repairing a short target switch needs a large correction, so the move
    # itself stays unbounded. Large relocations are labeled as fitted recovery;
    # smaller detector-supported adjustments retain their source. The immutable
    # measurement coordinates remain available separately in either case.
    relabel_px = max(12.0, _REFINE_DET_RELABEL_FRAC * float(diag))
    for index, item in enumerate(points):
        if item.source == "det" and item.measurement is None:
            item.measurement = (float(item.cx), float(item.cy))
        moved = math.hypot(new_x[index] - item.cx, new_y[index] - item.cy)
        item.cx = float(new_x[index])
        item.cy = float(new_y[index])
        if item.source == "det" and moved > relabel_px:
            item.source = "interp"
            item.interpolated = True
            item.source_reason = "refit_relocated_detection"


def refine_trajectory(results: List[Optional[FrameResult]], diag: float = REFERENCE_DIAG_PX) -> None:
    """Refit every continuous run of at least 9 rows in place (see _refine_run)."""
    frame = 0
    while frame < len(results):
        if results[frame] is None:
            frame += 1
            continue
        start = frame
        while frame < len(results) and results[frame] is not None:
            frame += 1
        if frame - start >= _MIN_REFIT_RUN:
            _refine_run(results, start, frame, diag)


@dataclass(frozen=True)
class _Scene:
    """Inputs shared by the fill phases of select_ball_in_play."""
    cfg: SelectorConfig
    fps: float
    total: int
    width: int
    height: int
    boost_masks: Optional[Sequence[Optional[MaskLike]]]
    raw_motions: Optional[Sequence[Optional[MaskLike]]]
    player_boxes_by_frame: Optional[PlayerBoxesByFrame]
    fill_frames: int        # longest gap or tail the selector fills
    min_fill_speed: float   # px/frame at this fps

    def motion_candidate(
        self, frame: int, x: float, y: float, area: float, steps: int, speed: float
    ) -> Tuple[Optional[MotionBlob], float]:
        """Nearest usable blob to (x, y) at `frame`, with the radius searched."""
        radius = _motion_search_radius(self.cfg, steps, speed)
        blob = _motion_candidate(
            frame, x, y, area, self.boost_masks, self.raw_motions,
            gate=radius, area_ratio_max=_FILL_AREA_RATIO_MAX,
        )
        return blob, radius


def _place_detections(
    results: List[Optional[FrameResult]], selected: List[Track],
    selected_observations: Dict[int, List[Detection]],
) -> None:
    """Stronger tracks win overlapping detector observations."""
    for track in sorted(selected, key=lambda item: item.score, reverse=True):
        for detection in selected_observations[id(track)]:
            current = results[detection.frame]
            if current is None or detection.conf > current.conf:
                results[detection.frame] = frame_result(
                    detection.cx, detection.cy, "det", detection=detection
                )


def _gap_is_supported(
    scene: _Scene, observations: List[Detection], index: int, speed: float, direction_supported: bool
) -> bool:
    """Whether a straight fill between observations[index] and [index+1] is a safe claim."""
    left, right = observations[index], observations[index + 1]
    gap = right.frame - left.frame
    strong_anchors = left.conf >= _STRONG_ANCHOR_CONF and right.conf >= _STRONG_ANCHOR_CONF
    # A long fill (> ~0.1 s) is a physics claim: both anchors must be
    # corroborated by a same-track neighbour, and the ball may not change
    # speed by more than the configured ratio across the gap.  A free-flying
    # ball cannot triple its speed in the same direction; a detector jump to
    # a static object after a long miss looks exactly like that.
    isolation_frames = _isolation_frames(scene.fps)
    if gap > isolation_frames:
        corroborated = (
            index > 0
            and left.frame - observations[index - 1].frame <= isolation_frames
            and index + 2 < len(observations)
            and observations[index + 2].frame - right.frame <= isolation_frames
        )
        speed_consistent = all(
            scene.cfg.fill_speed_ratio_min
            <= speed / max(_segment_speed(first, second), 1e-6)
            <= scene.cfg.fill_speed_ratio_max
            for first, second in (
                (observations[index - 1], left) if index > 0 else (None, None),
                (right, observations[index + 2]) if index + 2 < len(observations) else (None, None),
            )
            if first is not None
        )
        strong_anchors = strong_anchors and corroborated and speed_consistent
    return (
        strong_anchors
        and direction_supported
        and (gap == 2 or speed >= scene.min_fill_speed)
    )


def _fill_gap(
    scene: _Scene, results: List[Optional[FrameResult]], observations: List[Detection], index: int
) -> None:
    """Fill the frames between observations[index] and [index+1] where evidence allows."""
    left, right = observations[index], observations[index + 1]
    gap = right.frame - left.frame
    speed = math.hypot(right.cx - left.cx, right.cy - left.cy) / gap
    direction_supported = (
        index > 0
        and direction_cosine(observations[index - 1], left, right) >= _FILL_DIRECTION_COS_MIN
    ) or (
        index + 2 < len(observations)
        and direction_cosine(left, right, observations[index + 2]) >= _FILL_DIRECTION_COS_MIN
    )
    # Opposite incoming and outgoing directions mean a hit happened inside
    # the gap; a straight fill would cut the vertex off.  That rules out
    # interpolation, but not a motion blob: a blob is an observation of where
    # the ball actually was, and bounces are exactly where the detector loses it.
    vertex_in_gap = (
        gap >= 3
        and index > 0
        and index + 2 < len(observations)
        and _segment_cosine(observations[index - 1], left, right, observations[index + 2]) < 0.0
    )
    # Judged once for the whole gap, but re-read per frame below: a single
    # frame vetoed beside a player must not disqualify the rest of the gap.
    gap_pair_supported = _gap_is_supported(scene, observations, index, speed, direction_supported)
    near_player_slow = speed < pxpf_at_fps(_NEAR_PLAYER_FILL_SPEED_PXPF, scene.fps, clamp=False)
    cfg = scene.cfg
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
        motion, search_radius = scene.motion_candidate(
            frame, base_x, base_y, area, frame - left.frame, speed
        )
        pair_supported = gap_pair_supported and not vertex_in_gap
        motion_supported = (
            motion is not None
            and (direction_supported or gap == 2)
            and (gap == 2 or speed >= scene.min_fill_speed)
        )
        if (
            is_near_player(base_x, base_y, scene.player_boxes_by_frame, frame, margin=0)
            and near_player_slow
        ):
            pair_supported = False
            # A motion blob between adjacent sampled detections is direct
            # evidence, even when the ball is slow beside a player.
            if gap != 2:
                motion_supported = False
        if not pair_supported and not motion_supported:
            offset_x = offset_y = 0.0
            offset_active = False
            continue
        if motion is not None:
            target_x = _BLOB_PULL * (motion.x - base_x)
            target_y = _BLOB_PULL * (motion.y - base_y)
            if offset_active:
                offset_x = _BLOB_PULL_DECAY * (offset_x + target_x)
                offset_y = _BLOB_PULL_DECAY * (offset_y + target_y)
            else:
                offset_x, offset_y = target_x, target_y
                offset_active = True
        elif offset_active:
            offset_x *= _BLOB_PULL_DECAY
            offset_y *= _BLOB_PULL_DECAY
        endpoint_weight = _ENDPOINT_PULL_WEIGHT if frame == right.frame - 1 else 1.0
        x = base_x + endpoint_weight * offset_x
        y = base_y + endpoint_weight * offset_y
        results[frame] = frame_result(
            x, y, "motion" if motion else "interp", motion=motion,
            search_radius=search_radius,
        )


def _fill_internal_gaps(
    scene: _Scene, results: List[Optional[FrameResult]], selected: List[Track],
    selected_observations: Dict[int, List[Detection]],
) -> None:
    """Fill internal paths only when a future real detection anchors the endpoint."""
    for track in sorted(selected, key=lambda item: item.score, reverse=True):
        observations = selected_observations[id(track)]
        for index in range(len(observations) - 1):
            gap = observations[index + 1].frame - observations[index].frame
            if gap <= 1 or gap - 1 > scene.fill_frames:
                continue
            _fill_gap(scene, results, observations, index)


def _extend_tail(
    scene: _Scene, results: List[Optional[FrameResult]], selected: List[Track],
    observations: List[Detection],
) -> None:
    """Continue one track's end only while nearby motion proves the ball is still there."""
    cfg = scene.cfg
    left, right = observations[-2:]
    dt = max(1, right.frame - left.frame)
    vx = (right.cx - left.cx) / dt
    vy = (right.cy - left.cy) / dt
    next_track = min(
        (other for other in selected if other.first_frame > right.frame),
        key=lambda other: other.first_frame,
        default=None,
    )
    next_start = next_track.first_frame if next_track is not None else scene.total
    if next_track is not None and next_start <= right.frame + scene.fill_frames + 1:
        next_dt = next_start - right.frame
        predicted_x, predicted_y = predict_projectile((right.cx, right.cy), (vx, vy), next_dt, cfg)
        next_observation = next_track.observations[0]
        if math.hypot(
            predicted_x - next_observation.cx,
            predicted_y - next_observation.cy,
        ) > _HANDOVER_GATE_DIAG_FRAC * cfg.diag:
            return
    top_reentry = (
        dt > scene.fill_frames
        and left.cy < _TOP_EXIT_FRAC * scene.height
        and right.cy < _TOP_REENTRY_FRAC * scene.height
    )
    if not top_reentry and math.hypot(vx, vy) < scene.min_fill_speed:
        return
    # A tail cannot help association here; tracks were already built above.
    # Emit it only while every current frame contains nearby strict motion.
    reentry_x, reentry_y = right.cx, right.cy
    tail_speed = math.hypot(vx, vy)
    misses = 0
    for step in range(1, scene.fill_frames + 1):
        frame = right.frame + step
        if frame >= min(scene.total, next_start):
            break
        if top_reentry:
            base_x, base_y = reentry_x, reentry_y
        else:
            base_x, base_y = predict_projectile((right.cx, right.cy), (vx, vy), step, cfg)
        motion, search_radius = scene.motion_candidate(
            frame, base_x, base_y, right.area, step, tail_speed
        )
        if motion is None:
            # A single blobless frame is a dropped mask, not proof the ball
            # is gone; keep coasting on the prediction for a moment. The
            # run still ends the moment the evidence runs out for good.
            misses += 1
            if misses > cfg.motion_tail_miss_budget:
                break
            continue
        misses = 0
        if results[frame] is None:
            results[frame] = frame_result(
                motion.x, motion.y, "motion", motion=motion,
                search_radius=search_radius,
            )
        if top_reentry:
            reentry_x, reentry_y = motion.x, motion.y


def _close_single_frame_holes(scene: _Scene, results: List[Optional[FrameResult]]) -> None:
    """One missing source frame between strong, fast detector anchors is safer
    to close than leaving the bottom of a bounce visibly disconnected."""
    max_step = _SINGLE_HOLE_MAX_STEP_DIAG_FRAC * scene.cfg.diag
    for frame in range(1, scene.total - 1):
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
            and left.conf >= _STRONG_ANCHOR_CONF and right.conf >= _STRONG_ANCHOR_CONF
            and scene.min_fill_speed <= speed <= max_step
            and not is_near_player(
                0.5 * (left.cx + right.cx),
                0.5 * (left.cy + right.cy),
                scene.player_boxes_by_frame,
                frame,
                margin=0,
            )
        ):
            results[frame] = frame_result(
                0.5 * (left.cx + right.cx),
                0.5 * (left.cy + right.cy),
                "interp",
            )


def _drop_implausible_fills(scene: _Scene, results: List[Optional[FrameResult]]) -> None:
    """Remove fitted rows that left the frame or jumped across the court.

    One missing row is safer than connecting two balls across the court.
    """
    jump_gate = max(_JUMP_GATE_MIN_PX, _JUMP_GATE_DIAG_FRAC * scene.cfg.diag)
    previous = None
    for frame, result in enumerate(results):
        if (
            result is not None
            and result.source != "det"
            and not (0.0 <= result.cx < scene.width and 0.0 <= result.cy < scene.height)
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


def select_ball_in_play(
    detections_by_frame: Sequence[Sequence[RawDetection]],
    fps: float,
    width: int,
    height: int,
    court_polygon: Optional[np.ndarray] = None,
    boost_masks: Optional[Sequence[Optional[MaskLike]]] = None,
    raw_motions: Optional[Sequence[Optional[MaskLike]]] = None,
    player_boxes_by_frame: Optional[PlayerBoxesByFrame] = None,
    debug: bool = False,
) -> Tuple[List[Optional[FrameResult]], Optional[Track], List[Track]]:
    """Select one ball trajectory from per-frame detector candidates.

    ``detections_by_frame[t]`` holds ``(bbox_xyxy, confidence)`` pairs for
    source frame ``t``.  ``boost_masks`` / ``raw_motions`` are per-frame motion
    masks (``utils.PackedMask``, a full-frame uint8 array, or None),
    ``player_boxes_by_frame[t]`` a dict or list of player boxes.

    Returns ``(per_frame, chosen, tracks)``: one ``FrameResult`` or ``None`` per
    source frame, the stitched chosen track (``None`` when nothing qualified),
    and every scored track for diagnostics.
    """
    total = len(detections_by_frame)
    cfg = SelectorConfig(fps=fps, width=width, height=height)
    scene = _Scene(
        cfg=cfg, fps=fps, total=total, width=width, height=height,
        boost_masks=boost_masks, raw_motions=raw_motions,
        player_boxes_by_frame=player_boxes_by_frame,
        fill_frames=max(_FILL_WINDOW_MIN_FRAMES, int(round(float(fps) * _FILL_WINDOW_SEC))),
        min_fill_speed=pxpf_at_fps(_MIN_FILL_SPEED_PXPF, fps, clamp=False),
    )
    detections = build_detections(detections_by_frame, boost_masks)
    tracks = score_tracks(
        build_tracks(detections, cfg),
        cfg,
        court_poly=court_polygon,
        player_boxes_by_frame=player_boxes_by_frame,
        total_frames=total,
    )
    selected = select_timeline_chain(selected_tracks(tracks), cfg, total)
    chosen = stitch_track_chain(selected, cfg)
    selected_observations = {
        id(track): trajectory_observations(track, fps) for track in selected
    }
    if chosen is not None:
        chosen.observations = trajectory_observations(chosen, fps)
    results: List[Optional[FrameResult]] = [None] * total

    _place_detections(results, selected, selected_observations)
    _fill_internal_gaps(scene, results, selected, selected_observations)
    for track in sorted(selected, key=lambda item: item.score, reverse=True):
        observations = selected_observations[id(track)]
        if len(observations) >= 2:
            _extend_tail(scene, results, selected, observations)
    _close_single_frame_holes(scene, results)
    refine_trajectory(results, cfg.diag)

    if debug:
        print(f"[clean selector] selected tracks: {[track.track_id for track in selected]}")
    _drop_implausible_fills(scene, results)
    return results, chosen, tracks
