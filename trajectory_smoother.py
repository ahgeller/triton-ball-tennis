"""
Trajectory post-processor for tennis ball tracking.

Takes the raw per_frame results from the selector (patchwork of det/carry/motion/interp)
and produces one clean, physically-consistent trajectory per shot segment.

Pipeline:
  1. Extract continuous runs of tracked frames
  2. Detect bounces/hits within runs (velocity sign reversals)
  3. Segment at bounce/hit points
  4. Fit smooth spline through detection points in each segment
  5. Replace ALL positions (det, carry, motion, interp) with fitted values
  6. Tag bounce frames with metadata
  7. Estimate ball speed in real-world units (km/h) where court homography available

Usage:
    from trajectory_smoother import smooth_trajectory, BounceEvent

    per_frame, bounces, speeds = smooth_trajectory(
        per_frame, fps, frame_w, frame_h,
        court_keypoints=last_kps,
    )
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

try:
    from scipy.interpolate import UnivariateSpline
    from scipy.signal import savgol_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from ball_in_play_selector import (
        FrameResult, SelectorConfig, build_court_homography, court_px_per_meter
    )
except ImportError:
    FrameResult = None
    SelectorConfig = None
    build_court_homography = None
    court_px_per_meter = None


# ═══════════════════════════════════════════════════════════════════
# Data Types
# ═══════════════════════════════════════════════════════════════════

@dataclass
class BounceEvent:
    """Detected bounce or hit event."""
    frame: int
    cx: float
    cy: float
    speed_before: float = 0.0   # px/frame before bounce
    speed_after: float = 0.0    # px/frame after bounce
    speed_kmh: float = 0.0      # real-world speed estimate (km/h)
    is_bounce: bool = True      # True=bounce, False=hit
    court_x: float = 0.0       # real-world court position (meters)
    court_y: float = 0.0


@dataclass
class SegmentInfo:
    """One continuous trajectory segment between bounces/hits."""
    start_frame: int
    end_frame: int
    det_frames: List[int] = field(default_factory=list)
    det_xs: List[float] = field(default_factory=list)
    det_ys: List[float] = field(default_factory=list)
    all_frames: List[int] = field(default_factory=list)
    smoothed_xs: List[float] = field(default_factory=list)
    smoothed_ys: List[float] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════
# Bounce / Hit Detection
# ═══════════════════════════════════════════════════════════════════

def _compute_velocities(
    per_frame: List[Optional['FrameResult']],
) -> Tuple[List[Optional[float]], List[Optional[float]]]:
    """Compute per-frame vx, vy from consecutive non-None positions."""
    n = len(per_frame)
    vx = [None] * n
    vy = [None] * n

    prev_f = -1
    prev_x = prev_y = 0.0
    for i, r in enumerate(per_frame):
        if r is None or r.cx is None or r.cy is None:
            continue
        if prev_f >= 0:
            dt = i - prev_f
            if 0 < dt <= 4:  # only compute velocity for small gaps
                vx[i] = (r.cx - prev_x) / dt
                vy[i] = (r.cy - prev_y) / dt
        prev_f = i
        prev_x = float(r.cx)
        prev_y = float(r.cy)

    return vx, vy


def detect_bounces(
    per_frame: List[Optional['FrameResult']],
    fps: float,
    min_speed_px: float = 3.0,
    court_polygon=None,
) -> List[BounceEvent]:
    """Detect bounce events from velocity sign reversals.
    
    A bounce is: vy flips from positive (downward) to negative (upward),
    speed is above minimum, and the ball is near/on the court.
    """
    vx_list, vy_list = _compute_velocities(per_frame)
    n = len(per_frame)
    bounces = []

    for i in range(2, n - 1):
        vy_before = vy_list[i]
        vy_after = vy_list[i + 1] if i + 1 < n else None

        if vy_before is None or vy_after is None:
            continue

        # Bounce: moving down (vy > 0 in image coords) then up (vy < 0)
        if vy_before > min_speed_px * 0.5 and vy_after < -min_speed_px * 0.3:
            r = per_frame[i]
            if r is None or r.cx is None:
                continue

            speed_before = math.sqrt(
                (vx_list[i] or 0.0) ** 2 + vy_before ** 2
            )
            speed_after = math.sqrt(
                (vx_list[i + 1] or 0.0) ** 2 + vy_after ** 2
            )

            if speed_before < min_speed_px:
                continue

            bounces.append(BounceEvent(
                frame=i,
                cx=float(r.cx),
                cy=float(r.cy),
                speed_before=speed_before,
                speed_after=speed_after,
                is_bounce=True,
            ))

    # Deduplicate bounces within 3 frames of each other
    if len(bounces) > 1:
        deduped = [bounces[0]]
        for b in bounces[1:]:
            if b.frame - deduped[-1].frame > 3:
                deduped.append(b)
            elif b.speed_before > deduped[-1].speed_before:
                deduped[-1] = b
        bounces = deduped

    return bounces


def detect_hits(
    per_frame: List[Optional['FrameResult']],
    player_boxes_by_frame: Optional[List] = None,
    min_speed_change: float = 5.0,
) -> List[BounceEvent]:
    """Detect hit events from sudden velocity changes near players.
    
    A hit is: large speed change AND ball is near a player bounding box.
    """
    if player_boxes_by_frame is None:
        return []

    vx_list, vy_list = _compute_velocities(per_frame)
    n = len(per_frame)
    hits = []

    for i in range(2, n - 2):
        vx_b = vx_list[i]
        vy_b = vy_list[i]
        vx_a = vx_list[i + 1]
        vy_a = vy_list[i + 1]

        if vx_b is None or vy_b is None or vx_a is None or vy_a is None:
            continue

        speed_b = math.sqrt(vx_b ** 2 + vy_b ** 2)
        speed_a = math.sqrt(vx_a ** 2 + vy_a ** 2)

        # Direction change: dot product of velocity vectors
        dot = vx_b * vx_a + vy_b * vy_a
        mag = speed_b * speed_a
        if mag < 1e-6:
            continue
        cos_angle = dot / mag

        # Hit: large direction change OR large speed change
        speed_change = abs(speed_a - speed_b)
        if speed_change < min_speed_change and cos_angle > -0.3:
            continue

        r = per_frame[i]
        if r is None or r.cx is None:
            continue

        # Check if near a player
        near_player = False
        if i < len(player_boxes_by_frame):
            pboxes = player_boxes_by_frame[i]
            if isinstance(pboxes, dict):
                pboxes = list(pboxes.values())
            for pb in (pboxes or []):
                if pb is None or len(pb) < 4:
                    continue
                # Expand player box by 50px for racket reach
                if (float(pb[0]) - 50 <= r.cx <= float(pb[2]) + 50 and
                        float(pb[1]) - 50 <= r.cy <= float(pb[3]) + 50):
                    near_player = True
                    break

        if not near_player:
            continue

        hits.append(BounceEvent(
            frame=i,
            cx=float(r.cx),
            cy=float(r.cy),
            speed_before=speed_b,
            speed_after=speed_a,
            is_bounce=False,
        ))

    return hits


# ═══════════════════════════════════════════════════════════════════
# Segmentation
# ═══════════════════════════════════════════════════════════════════

def _extract_runs(
    per_frame: List[Optional['FrameResult']],
    max_internal_gap: int = 6,
) -> List[Tuple[int, int]]:
    """Extract continuous runs of tracked frames.
    
    A run is a contiguous sequence of non-None frames, allowing small
    internal gaps (where carry/interp bridge).
    """
    n = len(per_frame)
    runs = []
    run_start = -1
    gap_count = 0

    for i in range(n):
        r = per_frame[i]
        if r is not None and r.cx is not None:
            if run_start < 0:
                run_start = i
            gap_count = 0
        else:
            if run_start >= 0:
                gap_count += 1
                if gap_count > max_internal_gap:
                    run_end = i - gap_count
                    if run_end > run_start + 2:
                        runs.append((run_start, run_end))
                    run_start = -1
                    gap_count = 0

    if run_start >= 0:
        run_end = n - 1
        # Trim trailing Nones
        while run_end > run_start and (per_frame[run_end] is None or per_frame[run_end].cx is None):
            run_end -= 1
        if run_end > run_start + 2:
            runs.append((run_start, run_end))

    return runs


def _segment_run(
    per_frame: List[Optional['FrameResult']],
    run_start: int,
    run_end: int,
    events: List[BounceEvent],
) -> List[SegmentInfo]:
    """Split a run at bounce/hit events into segments."""
    # Event frames within this run
    event_frames = sorted(set(
        e.frame for e in events
        if run_start <= e.frame <= run_end
    ))

    # Build segment boundaries
    boundaries = [run_start] + event_frames + [run_end]
    # Deduplicate and sort
    boundaries = sorted(set(boundaries))

    segments = []
    for i in range(len(boundaries) - 1):
        sf = boundaries[i]
        ef = boundaries[i + 1]
        if ef - sf < 2:
            continue

        seg = SegmentInfo(start_frame=sf, end_frame=ef)
        for f in range(sf, ef + 1):
            r = per_frame[f] if f < len(per_frame) else None
            if r is not None and r.cx is not None:
                seg.all_frames.append(f)
                if r.source == 'det':
                    seg.det_frames.append(f)
                    seg.det_xs.append(float(r.cx))
                    seg.det_ys.append(float(r.cy))
        segments.append(seg)

    return segments


# ═══════════════════════════════════════════════════════════════════
# Spline Fitting
# ═══════════════════════════════════════════════════════════════════

def _fit_segment(
    seg: SegmentInfo,
    per_frame: List[Optional['FrameResult']],
    smoothing_factor: float = 0.8,
) -> SegmentInfo:
    """Fit a smooth spline through detection points in a segment.
    
    Uses all available positions (det, motion, carry) as data points,
    but weights detection points higher for anchor accuracy.
    """
    if not HAS_SCIPY:
        # No scipy — just copy positions through unchanged
        seg.smoothed_xs = [float(per_frame[f].cx) for f in seg.all_frames]
        seg.smoothed_ys = [float(per_frame[f].cy) for f in seg.all_frames]
        return seg

    if len(seg.all_frames) < 4:
        # Too few points for spline — use raw positions
        seg.smoothed_xs = [float(per_frame[f].cx) for f in seg.all_frames]
        seg.smoothed_ys = [float(per_frame[f].cy) for f in seg.all_frames]
        return seg

    # Collect all positions with weights
    frames = np.array(seg.all_frames, dtype=np.float64)
    xs = np.array([float(per_frame[f].cx) for f in seg.all_frames], dtype=np.float64)
    ys = np.array([float(per_frame[f].cy) for f in seg.all_frames], dtype=np.float64)

    # Weight: detections get weight 1.0, carry/motion get 0.3
    weights = np.array([
        1.0 if per_frame[f].source == 'det' else 0.3
        for f in seg.all_frames
    ], dtype=np.float64)

    # Normalize frame indices to [0, 1] for numerical stability
    t_min = frames[0]
    t_max = frames[-1]
    t_range = max(t_max - t_min, 1.0)
    t_norm = (frames - t_min) / t_range

    n_pts = len(frames)

    # Choose smoothing: more points = more smoothing allowed
    # s parameter controls the tradeoff between closeness and smoothness
    s_val = n_pts * smoothing_factor

    try:
        # Fit x(t) and y(t) separately with weighted cubic spline
        spline_x = UnivariateSpline(t_norm, xs, w=weights, k=3, s=s_val)
        spline_y = UnivariateSpline(t_norm, ys, w=weights, k=3, s=s_val)

        # Evaluate at all frame positions (including carry/interp frames)
        all_t = np.array(seg.all_frames, dtype=np.float64)
        all_t_norm = (all_t - t_min) / t_range

        seg.smoothed_xs = spline_x(all_t_norm).tolist()
        seg.smoothed_ys = spline_y(all_t_norm).tolist()

    except Exception:
        # Spline failed — use Savitzky-Golay as fallback
        window = min(len(xs) - 1, 11)
        if window % 2 == 0:
            window -= 1
        window = max(window, 3)
        poly_order = min(2, window - 1)

        try:
            seg.smoothed_xs = savgol_filter(xs, window, poly_order).tolist()
            seg.smoothed_ys = savgol_filter(ys, window, poly_order).tolist()
        except Exception:
            seg.smoothed_xs = xs.tolist()
            seg.smoothed_ys = ys.tolist()

    return seg


# ═══════════════════════════════════════════════════════════════════
# Speed Estimation
# ═══════════════════════════════════════════════════════════════════

def _compute_speeds(
    per_frame: List[Optional['FrameResult']],
    fps: float,
    court_keypoints=None,
) -> List[Optional[float]]:
    """Compute per-frame speed in km/h using court homography.
    
    Returns list parallel to per_frame with speed in km/h or None.
    Falls back to raw px/frame if no homography available.
    """
    n = len(per_frame)
    speeds = [None] * n

    # Try to get court homography for real-world scaling
    hom = None
    H = H_inv = None
    court_w = 10.97  # meters
    if build_court_homography is not None and court_keypoints is not None:
        hom = build_court_homography(court_keypoints)
        if hom is not None:
            H, H_inv, court_w, _ = hom

    prev_f = -1
    prev_x = prev_y = 0.0
    for i, r in enumerate(per_frame):
        if r is None or r.cx is None or r.cy is None:
            continue
        if prev_f >= 0:
            dt_frames = i - prev_f
            if 0 < dt_frames <= 3:
                dx = float(r.cx) - prev_x
                dy = float(r.cy) - prev_y
                px_per_frame = math.sqrt(dx * dx + dy * dy) / dt_frames

                if H is not None and court_px_per_meter is not None:
                    # Convert to real-world speed
                    scale = court_px_per_meter(
                        float(r.cx), float(r.cy), H, H_inv, court_w
                    )
                    if scale is not None and scale > 0:
                        m_per_frame = px_per_frame / scale
                        m_per_sec = m_per_frame * fps
                        kmh = m_per_sec * 3.6
                        speeds[i] = kmh
                        continue

                # Fallback: rough estimate assuming court is ~600px tall = 23.77m
                # This is very approximate but better than nothing
                rough_scale = 23.77 / 600.0  # meters per pixel (rough)
                m_per_frame = px_per_frame * rough_scale
                m_per_sec = m_per_frame * fps
                speeds[i] = m_per_sec * 3.6

        prev_f = i
        prev_x = float(r.cx)
        prev_y = float(r.cy)

    return speeds


# ═══════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════

def smooth_trajectory(
    per_frame: List[Optional['FrameResult']],
    fps: float,
    frame_w: int,
    frame_h: int,
    court_keypoints=None,
    court_polygon=None,
    player_boxes_by_frame=None,
    smoothing_factor: float = 0.8,
    enable_bounce_detect: bool = True,
    enable_speed: bool = True,
    enable_smoothing: bool = True,
) -> Tuple[
    List[Optional['FrameResult']],
    List[BounceEvent],
    List[Optional[float]],
]:
    """Post-process selector results into clean trajectories.
    
    Args:
        per_frame: selector output (list of FrameResult or None)
        fps: video frame rate
        frame_w, frame_h: frame dimensions
        court_keypoints: flat list of court keypoint coordinates
        court_polygon: cv2 contour for court boundary
        player_boxes_by_frame: per-frame player bounding boxes
        smoothing_factor: spline smoothness (0=tight fit, 2=very smooth)
        enable_bounce_detect: detect and tag bounce events
        enable_speed: compute per-frame speed estimates
        enable_smoothing: apply spline smoothing to trajectories
    
    Returns:
        (smoothed_per_frame, bounce_events, per_frame_speeds_kmh)
    """
    diag = math.sqrt(frame_w ** 2 + frame_h ** 2)
    min_speed = max(2.0, 0.003 * diag)  # minimum speed to consider for bounces

    # Step 1: Detect bounces
    bounces = []
    if enable_bounce_detect:
        bounces = detect_bounces(per_frame, fps, min_speed_px=min_speed,
                                 court_polygon=court_polygon)
        hits = detect_hits(per_frame, player_boxes_by_frame,
                          min_speed_change=min_speed * 1.5)
        all_events = sorted(bounces + hits, key=lambda e: e.frame)
        print(f"[smoother] Detected {len(bounces)} bounces, {len(hits)} hits")
    else:
        all_events = []

    # Step 2: Extract continuous runs and segment at events
    runs = _extract_runs(per_frame)
    print(f"[smoother] Found {len(runs)} continuous tracking runs")

    total_smoothed = 0
    if enable_smoothing and HAS_SCIPY:
        # Step 3: For each run, segment and fit splines
        for run_start, run_end in runs:
            segments = _segment_run(per_frame, run_start, run_end, all_events)

            for seg in segments:
                if len(seg.all_frames) < 3:
                    continue

                seg = _fit_segment(seg, per_frame, smoothing_factor)

                # Step 4: Write smoothed positions back into per_frame
                for idx, f in enumerate(seg.all_frames):
                    if idx >= len(seg.smoothed_xs):
                        break
                    r = per_frame[f]
                    if r is None:
                        continue
                    r.cx = seg.smoothed_xs[idx]
                    r.cy = seg.smoothed_ys[idx]
                    total_smoothed += 1

        print(f"[smoother] Smoothed {total_smoothed} frame positions across "
              f"{sum(len(_segment_run(per_frame, s, e, all_events)) for s, e in runs)} segments")
    elif not HAS_SCIPY:
        print("[smoother] scipy not available — skipping trajectory smoothing")

    # Step 5: Compute speeds
    speeds = [None] * len(per_frame)
    if enable_speed:
        speeds = _compute_speeds(per_frame, fps, court_keypoints)
        valid_speeds = [s for s in speeds if s is not None and s > 5.0]
        if valid_speeds:
            avg_speed = sum(valid_speeds) / len(valid_speeds)
            max_speed = max(valid_speeds)
            print(f"[smoother] Speed: avg={avg_speed:.0f} km/h, max={max_speed:.0f} km/h "
                  f"({len(valid_speeds)} frames with valid speed)")

    # Step 6: Annotate bounce events with speed
    for b in bounces:
        fi = b.frame
        if fi < len(speeds) and speeds[fi] is not None:
            b.speed_kmh = speeds[fi]

        # Court position via homography
        if build_court_homography is not None and court_keypoints is not None:
            hom = build_court_homography(court_keypoints)
            if hom is not None:
                H, H_inv, cw, ch = hom
                try:
                    pt = np.array([b.cx, b.cy, 1.0])
                    world = H @ pt
                    if abs(world[2]) > 1e-6:
                        b.court_x = float(world[0] / world[2])
                        b.court_y = float(world[1] / world[2])
                except Exception:
                    pass

    return per_frame, bounces, speeds
