# Imports
import math
import json
import os
import numpy as np
import cv2
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass, field
try:
    from filterpy.kalman import KalmanFilter
except ImportError:
    KalmanFilter = None
    from filterpy.kalman import KalmanFilter

from .config import SelectorConfig
from .models import Detection, MotionTrack, Track, FrameResult
from .utils import _cfg_diag, _fps_norm_pxpf, _clamp01, _ensure_mask_u8, _mask_has_motion_near, build_court_homography, court_px_per_meter
from .physics import _kinematic_motion_frac, _xy_dist, _predict_projectile, _predict_projectile_vel, BallKalmanFilter
from .scoring import score_tracks, select_best_track, _track_speed_stats, _track_movement_score, _track_extent, _is_near_player


def _guide_static_speed_thresh(cfg: SelectorConfig, diag: float) -> float:
    """Static-speed threshold in px/frame, capped to avoid over-scaling with resolution."""
    diag_based = float(cfg.guide_spawn_static_speed_frac) * float(diag) * 0.95
    fps_capped = _fps_norm_pxpf(4.0, cfg)
    return max(1.2, min(diag_based, fps_capped))

def _guide_det_gate_ok(
    x: float,
    y: float,
    gx: float,
    gy: float,
    guide_exact: bool,
    cfg: SelectorConfig,
    diag: float
) -> bool:
    """Guide is used only to gate candidate detections near the chosen path."""
    gate = (cfg.guide_gate_exact_frac if guide_exact else cfg.guide_gate_soft_frac) * diag
    d = math.sqrt((x - gx) ** 2 + (y - gy) ** 2)
    return d <= gate

def _det_hard_continuity_ok(
    x: float,
    y: float,
    last_det_pos: Optional[Tuple[float, float]],
    last_vel: Tuple[float, float],
    frames_since_det: int,
    cfg: SelectorConfig,
    diag: float
) -> bool:
    """Hard-reject abrupt det jumps that break local trajectory continuity."""
    if last_det_pos is None:
        return True

    dt = max(int(frames_since_det), 1)
    max_step = max(
        20.0,
        cfg.det_hard_step_frac * diag *
        (1.0 + cfg.det_hard_step_growth_per_frame * (dt - 1))
    )

    d_raw = _xy_dist(x, y, last_det_pos[0], last_det_pos[1])
    if d_raw <= max_step:
        return True

    pred_x = last_det_pos[0] + last_vel[0] * dt
    pred_y = last_det_pos[1] + last_vel[1] * dt
    d_pred = _xy_dist(x, y, pred_x, pred_y)
    return d_pred <= max_step * 1.15

def _filter_guide_observations(
    obs: List["Detection"],
    cfg: SelectorConfig,
    diag: float
) -> Tuple[List["Detection"], int]:
    """Drop short-lived guide spikes and jumps to static balls.
    
    IMPROVED: More aggressive detection of jumps to stationary balls.
    """
    n = len(obs)
    if n < 3:
        return obs, 0

    max_step = max(10.0, cfg.guide_spike_max_step_frac * diag)
    max_resid = max(16.0, cfg.guide_spike_resid_frac * diag)
    max_neighbor_gap = max(int(cfg.guide_spike_max_neighbor_gap), 2)
    static_speed = max(2.0, _guide_static_speed_thresh(cfg, diag))
    lookahead_obs = max(int(cfg.guide_spawn_lookahead_obs), 1)
    max_motion_frac = max(0.0, min(1.0, float(cfg.guide_spawn_max_motion_frac)))
    
    # IMPROVED: Also detect sudden deceleration to static (velocity discontinuity)
    min_moving_speed = static_speed * 2.0  # Ball was moving significantly

    keep: List["Detection"] = [obs[0]]
    dropped = 0
    for i in range(1, n - 1):
        a = keep[-1]
        b = obs[i]
        c = obs[i + 1]

        dt_ab = max(int(b.frame - a.frame), 1)
        dt_bc = max(c.frame - b.frame, 1)
        dt_ac = c.frame - a.frame

        step_ab = _xy_dist(a.cx, a.cy, b.cx, b.cy) / dt_ab
        step_bc = _xy_dist(b.cx, b.cy, c.cx, c.cy) / dt_bc

        # Guard against cross-court "spawned" static false positives:
        # a large jump into a low-motion local window should be ignored.
        jump_to_static = False
        
        # IMPROVED: Check for jump to static with multiple conditions
        b_on_motion = bool(getattr(b, "on_motion", False))
        
        # Case 1: Large jump to non-motion point
        if step_ab > max_step and not b_on_motion:
            end_i = min(n, i + 1 + lookahead_obs)
            win = obs[i:end_i]
            win_motion = (
                sum(1 for o in win if bool(getattr(o, "on_motion", False))) /
                max(len(win), 1)
            )
            win_speeds: List[float] = []
            for j in range(i, end_i - 1):
                o0 = obs[j]
                o1 = obs[j + 1]
                dt_j = max(int(o1.frame - o0.frame), 1)
                win_speeds.append(_xy_dist(o0.cx, o0.cy, o1.cx, o1.cy) / dt_j)
            win_avg_speed = (
                float(sum(win_speeds)) / max(len(win_speeds), 1)
                if win_speeds else step_bc
            )
            if win_avg_speed <= static_speed and win_motion <= max_motion_frac:
                jump_to_static = True
        
        # IMPROVED: Case 2 - Sudden deceleration to static (moving -> stopped)
        # If the incoming speed was high and the outgoing speed is near zero,
        # this is likely a jump to a stationary ball
        if not jump_to_static and step_ab > min_moving_speed and step_bc < static_speed:
            # Check if b is on motion mask - if not, it's likely a static ball
            if not b_on_motion:
                # Use the forward velocity from a to predict where b should be
                if i >= 2:
                    prev_obs = obs[i - 1]  # the point before a in the raw list
                    # But 'a' is keep[-1], so estimate velocity from keep
                    if len(keep) >= 2:
                        a_prev = keep[-2]
                        dt_prev = max(int(a.frame - a_prev.frame), 1)
                        fwd_vx = (a.cx - a_prev.cx) / dt_prev
                        fwd_vy = (a.cy - a_prev.cy) / dt_prev
                    else:
                        fwd_vx, fwd_vy = 0.0, 0.0
                else:
                    fwd_vx, fwd_vy = 0.0, 0.0
                expected_x = a.cx + fwd_vx * dt_ab
                expected_y = a.cy + fwd_vy * dt_ab
                deviation = _xy_dist(b.cx, b.cy, expected_x, expected_y)
                # If we deviated significantly AND landed at a non-moving spot
                if deviation > max_step * 0.5:
                    jump_to_static = True
        
        # IMPROVED: Case 3 - Position cluster check
        # If this observation is in a tight cluster with nearby observations but no motion,
        # AND the ball doesn't move away quickly after, it's a static ball cluster.
        if not jump_to_static and not b_on_motion:
            # Check if nearby observations (in time) cluster spatially
            cluster_start = max(0, i - 2)
            cluster_end = min(n, i + 3)
            cluster_obs = obs[cluster_start:cluster_end]
            if len(cluster_obs) >= 3:
                xs = [o.cx for o in cluster_obs]
                ys = [o.cy for o in cluster_obs]
                cluster_extent = math.sqrt((max(xs) - min(xs))**2 + (max(ys) - min(ys))**2)
                cluster_motion = sum(1 for o in cluster_obs if bool(getattr(o, "on_motion", False)))
                # Small cluster with no motion = static ball — but only if the ball
                # doesn't escape quickly afterwards (which would indicate a real bounce).
                if cluster_extent < max_step * 0.5 and cluster_motion == 0:
                    # Check if the point AFTER the cluster moves away fast (real bounce)
                    escape_check_end = min(n, cluster_end + 2)
                    escape_obs = obs[cluster_end:escape_check_end]
                    last_cluster = cluster_obs[-1]
                    escapes = any(
                        _xy_dist(last_cluster.cx, last_cluster.cy, eo.cx, eo.cy) /
                        max(int(eo.frame - last_cluster.frame), 1) > static_speed * 2.0
                        for eo in escape_obs
                    )
                    if not escapes:
                        jump_to_static = True

        is_spike = False
        if dt_ac > 0 and dt_ac <= max_neighbor_gap:
            t = (b.frame - a.frame) / float(dt_ac)
            ix = a.cx + (c.cx - a.cx) * t
            iy = a.cy + (c.cy - a.cy) * t
            resid = _xy_dist(b.cx, b.cy, ix, iy)
            shortcut = _xy_dist(a.cx, a.cy, c.cx, c.cy) / dt_ac
            is_spike = (
                step_ab > max_step and
                step_bc > max_step and
                resid > max_resid and
                shortcut < 0.70 * max(step_ab, step_bc)
            )

        if is_spike or jump_to_static:
            dropped += 1
            continue
        keep.append(b)

    # Last point guard for end-of-track jumps into static clutter.
    # IMPROVED: More aggressive check
    last = obs[-1]
    prev = keep[-1]
    dt_last = max(int(last.frame - prev.frame), 1)
    step_last = _xy_dist(prev.cx, prev.cy, last.cx, last.cy) / dt_last
    drop_last = False
    last_on_motion = bool(getattr(last, "on_motion", False))
    
    # Drop if: large jump to non-motion point
    if step_last > max_step * 1.15 and not last_on_motion:
        drop_last = True
    # IMPROVED: Also drop if sudden stop at non-motion location after fast movement
    elif len(keep) >= 2 and not last_on_motion:
        prev2 = keep[-2]
        dt_prev = max(int(prev.frame - prev2.frame), 1)
        prev_step = _xy_dist(prev2.cx, prev2.cy, prev.cx, prev.cy) / dt_prev
        # Was moving fast, suddenly stopped at non-motion location
        if prev_step > static_speed * 2.0 and step_last < static_speed:
            drop_last = True
    
    if drop_last:
        dropped += 1
    else:
        keep.append(last)

    if len(keep) < 2:
        return obs[:2], dropped
    return keep, dropped

def _trim_leading_static_guide_obs(
    obs: List["Detection"],
    cfg: SelectorConfig,
    diag: float
) -> Tuple[List["Detection"], int]:
    """Trim a static prefix so guide cannot bootstrap from parked balls."""
    if not obs:
        return obs, 0
    if len(obs) == 1:
        # Single-point guide is not useful and is often static clutter.
        return [], 1

    static_speed = _guide_static_speed_thresh(cfg, diag)
    move_speed = static_speed * 1.25
    first_moving_idx = None
    n = len(obs)

    for i, o in enumerate(obs):
        moving = bool(getattr(o, "on_motion", False))
        if i > 0:
            p = obs[i - 1]
            dt = max(int(o.frame - p.frame), 1)
            moving = moving or (_xy_dist(o.cx, o.cy, p.cx, p.cy) / dt > move_speed)
        if i + 1 < n:
            q = obs[i + 1]
            dt = max(int(q.frame - o.frame), 1)
            moving = moving or (_xy_dist(q.cx, q.cy, o.cx, o.cy) / dt > move_speed)
        if moving:
            first_moving_idx = i
            break

    if first_moving_idx is None:
        return [], n
    if first_moving_idx <= 0:
        return obs, 0
    return obs[first_moving_idx:], first_moving_idx

def _prune_static_guide_runs(
    obs: List["Detection"],
    cfg: SelectorConfig,
    diag: float
) -> Tuple[List["Detection"], int]:
    """Remove observations that cluster in a small spatial area without motion.

    Uses spatial grid clustering instead of consecutive-run detection so that
    static ball observations interleaved with real ones (from track merging)
    are still caught and removed.
    """
    n = len(obs)
    print(f"[prune_static] Input: {n} observations")
    if n < 4:
        return obs, 0

    cell_size = max(8.0, 0.012 * diag)  # ~22px on 1080p
    min_cluster_obs = 3
    max_cluster_motion_frac = 0.15

    # Grid-bucket all observations by spatial cell.
    from collections import defaultdict
    cells: dict = defaultdict(list)
    for i, o in enumerate(obs):
        cx_cell = int(o.cx / cell_size)
        cy_cell = int(o.cy / cell_size)
        cells[(cx_cell, cy_cell)].append(i)

    # Merge neighboring cells into connected clusters.
    drop = set()
    visited_cells = set()
    for seed_key in list(cells.keys()):
        if seed_key in visited_cells:
            continue
        # BFS to find all connected cells.
        cluster_indices = []
        queue = [seed_key]
        cluster_cells = set()
        while queue:
            ck = queue.pop()
            if ck in visited_cells:
                continue
            if ck not in cells:
                continue
            visited_cells.add(ck)
            cluster_cells.add(ck)
            cluster_indices.extend(cells[ck])
            # Check 4-connected neighbors.
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nk = (ck[0] + dx, ck[1] + dy)
                if nk in cells and nk not in visited_cells:
                    queue.append(nk)

        unique_indices = sorted(set(cluster_indices))
        if len(unique_indices) < min_cluster_obs:
            continue
        # Check spatial extent.
        xs = [float(obs[i].cx) for i in unique_indices]
        ys = [float(obs[i].cy) for i in unique_indices]
        extent = float(np.hypot(max(xs) - min(xs), max(ys) - min(ys)))
        if extent > 2.5 * cell_size:
            print(f"[prune_static]   Cluster skipped: extent={extent:.1f} > limit={2.5 * cell_size:.1f}")
            continue
        # Check motion fraction.
        motion_count = sum(
            1 for i in unique_indices
            if bool(getattr(obs[i], "on_motion", False))
        )
        motion_frac = motion_count / max(len(unique_indices), 1)
        if motion_frac > max_cluster_motion_frac:
            print(f"[prune_static]   Cluster skipped: motion_frac={motion_frac:.2f} > {max_cluster_motion_frac}")
            continue
        # Static cluster — mark non-motion obs for removal.
        cx_avg = sum(xs) / len(xs)
        cy_avg = sum(ys) / len(ys)
        print(f"[prune_static]   DROPPING cluster: {len(unique_indices)} obs at ({cx_avg:.0f},{cy_avg:.0f}), extent={extent:.1f}, motion={motion_frac:.2f}")
        for i in unique_indices:
            if not bool(getattr(obs[i], "on_motion", False)):
                drop.add(i)

    kept = [o for idx, o in enumerate(obs) if idx not in drop]
    dropped = len(drop)
    print(f"[prune_static] Result: dropped={dropped}, kept={len(kept)}")
    if len(kept) < 2:
        return obs, 0
    return kept, dropped

def build_detections(
    detections_by_frame: List[List[Tuple[list, float]]],
    raw_motions: Optional[List[Optional[np.ndarray]]] = None
) -> List[List[Detection]]:
    """Convert raw YOLO outputs to Detection objects."""
    all_dets = []
    has_motion = bool(raw_motions)
    for t, frame_dets in enumerate(detections_by_frame):
        frame_list = []
        mask = None
        if has_motion and t < len(raw_motions):
            bm = raw_motions[t]
            if bm is not None:
                mask = _ensure_mask_u8(bm)
        for bbox, conf in frame_dets:
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            area = max((x2 - x1) * (y2 - y1), 1.0)

            on_motion = False
            if mask is not None:
                my = max(0, min(int(cy), mask.shape[0] - 1))
                mx = max(0, min(int(cx), mask.shape[1] - 1))
                # Check 3x3 patch instead of single center pixel to avoid
                # sub-pixel alignment misses on moving balls.
                h_mask, w_mask = mask.shape[:2]
                for dy in (-1, 0, 1):
                    if on_motion:
                        break
                    py = max(0, min(my + dy, h_mask - 1))
                    for dx in (-1, 0, 1):
                        px = max(0, min(mx + dx, w_mask - 1))
                        if mask[py, px] > 0:
                            on_motion = True
                            break

            frame_list.append(Detection(
                frame=t, cx=cx, cy=cy,
                x1=x1, y1=y1, x2=x2, y2=y2,
                conf=conf, area=area, on_motion=on_motion
            ))
        all_dets.append(frame_list)
    return all_dets

def build_motion_tracks(
    boost_masks_packed: List[Optional[np.ndarray]],
    cfg: SelectorConfig
) -> List[MotionTrack]:
    """
    Extract continuous centroid tracks from per-frame motion blobs.

    Association uses velocity-predicted matching with a radius that scales with
    track speed and gap, so fast balls don't break the track. Per-track median
    area gives a shape-consistency check so tracks don't drift onto unrelated
    blobs (rackets, edges). Output centroids are raw — no EMA — so the trail
    doesn't lag behind the actual ball.
    """
    import cv2
    import math

    diag = math.hypot(float(cfg.width), float(cfg.height))
    base_radius = max(20.0, 0.012 * diag)        # ~26 px on 1080p
    max_radius = max(80.0, 0.045 * diag)         # ~99 px on 1080p
    MAX_GAP = 4                                  # max frames since last point before track is dead
    MIN_TRACK_LEN = 4

    # Each active track: dict with points, areas, median_area, vel
    active_tracks: List[Dict[str, Any]] = []
    completed_tracks: List[Dict[str, Any]] = []

    for fi, pk in enumerate(boost_masks_packed):
        mask = _ensure_mask_u8(pk)
        current_blobs: List[Dict[str, float]] = []
        if mask is not None:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                if area < 15 or area > 500:
                    continue
                bx, by, bw, bh = cv2.boundingRect(c)
                if bw <= 0 or bh <= 0:
                    continue
                aspect = max(bw, bh) / max(min(bw, bh), 1)
                if aspect > 3.0:
                    continue
                fill_ratio = area / max(bw * bh, 1)
                if fill_ratio < 0.25:
                    continue
                M = cv2.moments(c)
                if M["m00"] > 0:
                    current_blobs.append({
                        "cx": M["m10"] / M["m00"],
                        "cy": M["m01"] / M["m00"],
                        "area": float(area),
                    })

        # Sort tracks by recency so freshest tracks claim blobs first.
        active_tracks.sort(key=lambda t: -t["points"][-1][0])

        matched_blob_idx = set()
        for trk in active_tracks:
            last_fi, last_x, last_y = trk["points"][-1]
            gap = fi - last_fi
            if gap <= 0 or gap > MAX_GAP:
                continue

            vx, vy = trk["vel"]
            pred_x = last_x + vx * gap
            pred_y = last_y + vy * gap

            # Adaptive radius: base + speed*gap + per-gap slack. Caps at max_radius.
            speed = math.hypot(vx, vy)
            radius = base_radius + speed * gap * 0.9 + max(0, gap - 1) * 14.0
            if radius > max_radius:
                radius = max_radius

            track_med_area = trk.get("median_area")
            best_dist = radius
            best_idx = -1
            for i, b in enumerate(current_blobs):
                if i in matched_blob_idx:
                    continue
                dist = math.hypot(b["cx"] - pred_x, b["cy"] - pred_y)
                if dist > radius:
                    continue
                # Shape consistency vs track's running area median.
                if track_med_area and track_med_area > 0:
                    ratio = b["area"] / track_med_area
                    if ratio < 0.30 or ratio > 3.5:
                        continue
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i

            if best_idx >= 0:
                b = current_blobs[best_idx]
                trk["points"].append((fi, b["cx"], b["cy"]))
                trk["areas"].append(b["area"])
                if len(trk["areas"]) > 12:
                    trk["areas"] = trk["areas"][-12:]
                sa = sorted(trk["areas"])
                trk["median_area"] = sa[len(sa) // 2]
                # Velocity from last two real points (skip gap normalization on dt).
                f1, x1, y1 = trk["points"][-2]
                f2, x2, y2 = trk["points"][-1]
                dt = max(1, f2 - f1)
                trk["vel"] = ((x2 - x1) / dt, (y2 - y1) / dt)
                matched_blob_idx.add(best_idx)

        # Unmatched blobs spawn new tracks.
        for i, b in enumerate(current_blobs):
            if i in matched_blob_idx:
                continue
            active_tracks.append({
                "points": [(fi, b["cx"], b["cy"])],
                "areas": [b["area"]],
                "median_area": b["area"],
                "vel": (0.0, 0.0),
            })

        # Retire tracks that have aged out.
        kept = []
        for trk in active_tracks:
            last_fi = trk["points"][-1][0]
            if fi - last_fi > MAX_GAP:
                if len(trk["points"]) >= MIN_TRACK_LEN:
                    completed_tracks.append(trk)
            else:
                kept.append(trk)
        active_tracks = kept

    for trk in active_tracks:
        if len(trk["points"]) >= MIN_TRACK_LEN:
            completed_tracks.append(trk)

    # No EMA: smoothed mirrors raw centroids so the trail doesn't lag.
    result = []
    for tid, trk in enumerate(completed_tracks):
        pts = trk["points"]
        smoothed = [(p[1], p[2]) for p in pts]
        result.append(MotionTrack(track_id=tid, points=pts, smoothed=smoothed))

    print(f"[selector] Extracted {len(result)} continuous motion tracks.")
    return result

def _association_plausible(
    trk: "Track",
    det: Detection,
    dt: int,
    cfg: SelectorConfig,
    diag: float
) -> bool:
    """Reject implausible associations so bad jumps form new tracks instead."""
    if dt <= 0:
        return False
    if trk.num_obs < 2:
        return True

    last_obs = trk.observations[-1]
    area_ratio = det.area / max(last_obs.area, 1.0)
    if area_ratio < cfg.assoc_area_ratio_min or area_ratio > cfg.assoc_area_ratio_max:
        return False

    obs_vx = (det.cx - trk.last_pos[0]) / dt
    obs_vy = (det.cy - trk.last_pos[1]) / dt
    obs_speed = math.sqrt(obs_vx * obs_vx + obs_vy * obs_vy)

    exp_vx, exp_vy = trk.last_vel
    exp_speed = math.sqrt(exp_vx * exp_vx + exp_vy * exp_vy)
    min_speed_for_dir = _fps_norm_pxpf(cfg.assoc_min_speed_for_dir, cfg)

    if exp_speed >= min_speed_for_dir:
        speed_ratio = obs_speed / max(exp_speed, 1e-6)
        if speed_ratio < cfg.assoc_speed_ratio_min or speed_ratio > cfg.assoc_speed_ratio_max:
            return False
        dot = obs_vx * exp_vx + obs_vy * exp_vy
        cos_sim = dot / max(obs_speed * exp_speed, 1e-6)
        if cos_sim < cfg.assoc_dir_cos_min:
            return False

    dvx = obs_vx - exp_vx
    dvy = obs_vy - exp_vy
    accel = math.sqrt(dvx * dvx + dvy * dvy)
    if accel > max(8.0, cfg.assoc_accel_frac * diag):
        return False

    return True

def build_tracks(
    all_dets: List[List[Detection]],
    cfg: SelectorConfig
) -> List[Track]:
    """Greedy association with stale reconnect + anti-jump trail guards."""
    active: List[Track] = []
    finished: List[Track] = []
    next_id = 0
    diag = _cfg_diag(cfg)
    dt_cap = max(int(cfg.assoc_gate_growth_cap_frames), 1)
    stitch_enable = bool(getattr(cfg, "trail_stitch_enable", True))
    stitch_gap_frames = max(1, int(getattr(cfg, "trail_stitch_gap_frames", 30)))
    stitch_min_stale = max(1, int(getattr(cfg, "trail_stitch_min_stale_frames", 3)))
    stitch_min_obs = max(1, int(getattr(cfg, "trail_stitch_min_obs", 3)))
    stitch_dist_px = max(8.0, float(getattr(cfg, "trail_stitch_dist_frac", 0.06)) * diag)
    stitch_max_step_px = max(10.0, float(getattr(cfg, "trail_stitch_max_step_frac", 0.11)) * diag)
    stitch_pred_resid_px = max(10.0, float(getattr(cfg, "trail_stitch_pred_resid_frac", 0.08)) * diag)
    jump_break_px = max(10.0, float(getattr(cfg, "trail_jump_break_frac", 0.06)) * diag)
    jump_break_growth_px = max(0.0, float(getattr(cfg, "trail_jump_break_growth_frac", 0.020)) * diag)

    for t, frame_dets in enumerate(all_dets):
        # Predict positions for all active tracks
        predictions = {}
        for trk in active:
            predictions[trk.track_id] = trk.predict(t)

        # Compute cost matrix: (track_idx, det_idx) -> distance
        assignments = []
        for ti, trk in enumerate(active):
            if not trk.alive:
                continue
            pred = predictions[trk.track_id]
            dt = t - trk.last_frame
            dt_eff = min(max(dt, 1), dt_cap)
            gate = cfg.base_gate_px + cfg.gate_growth_px * dt_eff
            speed_limit = cfg.max_speed_px_per_frame * dt_eff

            for di, det in enumerate(frame_dets):
                # Distance from predicted position
                dist_pred = math.sqrt((pred[0] - det.cx) ** 2 + (pred[1] - det.cy) ** 2)
                # Distance from last known position (handles direction reversals)
                dist_raw = math.sqrt((trk.last_pos[0] - det.cx) ** 2 +
                                     (trk.last_pos[1] - det.cy) ** 2)
                dist = min(dist_pred, dist_raw)
                if dist <= gate and dist_raw <= speed_limit:
                    # If a track is already missing for a few frames, require stricter
                    # prediction consistency to avoid grabbing a different ball instantly.
                    if trk.misses > 0:
                        reacq_gate = min(gate, stitch_dist_px)
                        if dist > reacq_gate:
                            continue
                        step = dist / float(max(dt_eff, 1))
                        if step > stitch_max_step_px:
                            continue
                        pred_gate = stitch_pred_resid_px * (1.0 + 0.25 * float(max(0, dt_eff - 1)))
                        if dist_pred > pred_gate:
                            continue

                    jump_gate = jump_break_px + jump_break_growth_px * float(max(0, dt_eff - 1))
                    if dist_raw > jump_gate:
                        continue

                    if not _association_plausible(trk, det, max(dt, 1), cfg, diag):
                        continue
                    score = dist + (0.15 * dist_pred if trk.misses > 0 else 0.0)
                    assignments.append((score, ti, di))

        # Greedy assignment: nearest first
        assignments.sort(key=lambda x: x[0])
        used_tracks = set()
        used_dets = set()
        for dist, ti, di in assignments:
            if ti in used_tracks or di in used_dets:
                continue
            active[ti].update(frame_dets[di])
            used_tracks.add(ti)
            used_dets.add(di)

        # Unmatched tracks: increment misses
        for ti, trk in enumerate(active):
            if ti not in used_tracks and trk.alive:
                trk.misses += 1
                if trk.misses > cfg.max_gap_frames:
                    trk.alive = False

        # Unmatched detections: start new tracks
        for di, det in enumerate(frame_dets):
            if di not in used_dets:
                stitched = False

                # Reconnect only to genuinely stale tracks, with motion/prediction bounds.
                if stitch_enable and finished:
                    best_idx = -1
                    best_score = 1e12
                    for fi_idx, trk_prev in enumerate(finished):
                        if trk_prev.num_obs < stitch_min_obs:
                            continue
                        gap = int(t - trk_prev.last_frame)
                        if gap < stitch_min_stale or gap > stitch_gap_frames:
                            continue

                        raw_dist = math.sqrt(
                            (trk_prev.last_pos[0] - det.cx) ** 2 +
                            (trk_prev.last_pos[1] - det.cy) ** 2
                        )
                        if raw_dist > stitch_dist_px:
                            continue

                        step = raw_dist / float(max(gap, 1))
                        if step > stitch_max_step_px:
                            continue

                        pred_x, pred_y = trk_prev.predict(t)
                        pred_resid = math.sqrt((pred_x - det.cx) ** 2 + (pred_y - det.cy) ** 2)
                        pred_gate = stitch_pred_resid_px * (1.0 + 0.25 * float(max(0, gap - 1)))
                        if pred_resid > pred_gate:
                            continue

                        jump_gate = jump_break_px + jump_break_growth_px * float(max(0, gap - 1))
                        if raw_dist > jump_gate:
                            continue

                        if not _association_plausible(trk_prev, det, max(gap, 1), cfg, diag):
                            continue

                        score = 0.60 * pred_resid + 0.40 * raw_dist
                        if score < best_score:
                            best_score = score
                            best_idx = fi_idx

                    if best_idx >= 0:
                        trk = finished.pop(best_idx)
                        trk.alive = True
                        trk.misses = 0
                        trk.update(det)
                        active.append(trk)
                        stitched = True

                if not stitched:
                    trk = Track(track_id=next_id, cfg=cfg)
                    next_id += 1
                    trk.update(det)
                    active.append(trk)

        # Move dead tracks to finished
        still_active = []
        for trk in active:
            if trk.alive:
                still_active.append(trk)
            else:
                if trk.num_obs >= 2:
                    finished.append(trk)
        active = still_active

    # Finalize remaining active tracks
    for trk in active:
        if trk.num_obs >= 2:
            finished.append(trk)

    return finished

def merge_tracks(
    tracks: List[Track],
    cfg: SelectorConfig,
    court_poly=None,
    player_boxes_by_frame=None
) -> List[Track]:
    """Merge only forward fragmented tracks likely from the same ball.

    Important behavior:
    - Only ended->new merges are allowed (no overlap/rewind merges).
    - This keeps concurrent/alternate-ball candidates separate.
    - If a track ends and a plausible continuation appears later, they can reconnect.
    """
    if len(tracks) < 2:
        return tracks

    diag = _cfg_diag(cfg)
    # Keep reconnect conservative: avoid fusing unrelated balls across long gaps.
    merge_max_gap_frames = max(int(round(cfg.max_gap_frames * 1.25)), cfg.max_gap_frames)
    merge_max_dist = 0.12 * diag         # tighter baseline merge radius
    merge_player_dist = 0.16 * diag      # still wider near player, but no longer permissive
    merge_small_gap_frames = max(2, int(round(float(cfg.fps) * 0.08)))
    merge_small_gap_jump = 0.10 * diag
    merge_pred_resid_base = 0.075 * diag
    merge_min_speed_for_dir = _fps_norm_pxpf(1.8, cfg)
    min_speed_threshold = _fps_norm_pxpf(2.0, cfg)  # 30fps-ref px/frame

    # Pre-filter: mark tracks as static or moving
    for trk in tracks:
        trk._avg_speed, _ = _track_speed_stats(trk)
        trk._is_static = trk._avg_speed < min_speed_threshold

    # Sort by first frame
    tracks.sort(key=lambda t: t.first_frame)

    merged = True
    while merged:
        merged = False
        i = 0
        while i < len(tracks):
            trk_a = tracks[i]

            # Skip static tracks as merge sources
            if trk_a._is_static and trk_a.num_obs < 10:
                i += 1
                continue

            best_j = -1
            best_score = float('inf')

            a_end_frame = trk_a.last_obs_frame
            a_end_pos = (trk_a.observations[-1].cx, trk_a.observations[-1].cy)
            a_vel = trk_a.last_vel

            for j in range(i + 1, len(tracks)):
                trk_b = tracks[j]

                # Skip merging static tiny tracks
                if trk_b._is_static and trk_b.num_obs < 5:
                    continue

                b_start_frame = trk_b.first_frame
                b_start_pos = (trk_b.observations[0].cx, trk_b.observations[0].cy)

                # Time check
                time_gap = b_start_frame - a_end_frame
                # Strict forward-only merge: B must start after A ended.
                if time_gap < 1:
                    continue
                if time_gap > merge_max_gap_frames:
                    continue

                # Spatial distance (raw + predicted)
                raw_dist = math.sqrt((a_end_pos[0] - b_start_pos[0]) ** 2 +
                                     (a_end_pos[1] - b_start_pos[1]) ** 2)
                dt = max(time_gap, 1)
                pred_x = a_end_pos[0] + a_vel[0] * dt
                pred_y = a_end_pos[1] + a_vel[1] * dt
                pred_dist = math.sqrt((pred_x - b_start_pos[0]) ** 2 +
                                      (pred_y - b_start_pos[1]) ** 2)
                dist = min(raw_dist, pred_dist)

                # Check merge conditions
                max_dist = merge_max_dist

                # Widen gate if near a player (ball being hit/served)
                near_player_a = _is_near_player(
                    a_end_pos[0], a_end_pos[1],
                    player_boxes_by_frame, a_end_frame)
                near_player_b = _is_near_player(
                    b_start_pos[0], b_start_pos[1],
                    player_boxes_by_frame, b_start_frame)
                if near_player_a or near_player_b:
                    max_dist = merge_player_dist

                # Both moving on court? Extra generous
                if (not trk_a._is_static and not trk_b._is_static and
                        court_poly is not None):
                    da = cv2.pointPolygonTest(court_poly,
                         (float(a_end_pos[0]), float(a_end_pos[1])), True)
                    db = cv2.pointPolygonTest(court_poly,
                         (float(b_start_pos[0]), float(b_start_pos[1])), True)
                    if da >= -cfg.court_expand_px and db >= -cfg.court_expand_px:
                        max_dist = merge_player_dist  # both near court = generous

                # Very short-gap merges must be spatially tight in both raw and predicted space.
                if time_gap <= merge_small_gap_frames:
                    if raw_dist > merge_small_gap_jump and pred_dist > merge_small_gap_jump:
                        continue

                # For larger gaps, require predicted endpoint compatibility.
                pred_resid_gate = max(
                    18.0,
                    merge_pred_resid_base * (1.0 + 0.15 * (dt - 1))
                )
                if pred_dist > pred_resid_gate and raw_dist > max_dist:
                    continue

                # Velocity-continuity guard: prevent joining tracks that imply implausible direction/speed shifts.
                obs_vx = (b_start_pos[0] - a_end_pos[0]) / dt
                obs_vy = (b_start_pos[1] - a_end_pos[1]) / dt
                obs_speed = math.sqrt(obs_vx * obs_vx + obs_vy * obs_vy)
                exp_speed = math.sqrt(a_vel[0] * a_vel[0] + a_vel[1] * a_vel[1])
                if exp_speed >= merge_min_speed_for_dir:
                    speed_ratio = obs_speed / max(exp_speed, 1e-6)
                    if speed_ratio < 0.22 or speed_ratio > 2.20:
                        continue
                    if obs_speed > 1e-6:
                        # Removed direction reversal penalty so racket hits (cos_sim ~ -1.0) can merge
                        pass

                if dist <= max_dist and dist < best_score:
                    best_j = j
                    best_score = dist
                # Fallback: if tracks are temporally very close and the raw distance
                # is plausible for the observed speed, allow the merge even if
                # it exceeds the normal distance gate. This catches cases where the
                # ball is moving fast and the prediction drifts, but the tracks
                # clearly belong to the same ball.
                elif time_gap <= 5 and raw_dist < best_score:
                    speed_est = max(obs_speed, exp_speed, 1.0)
                    expected_travel = speed_est * dt * 1.8  # allow 80% overshoot
                    if raw_dist <= expected_travel and raw_dist <= max_dist * 2.0:
                        best_j = j
                        best_score = raw_dist

            if best_j >= 0:
                trk_b = tracks[best_j]
                # Merge B into A
                a_frames = set(o.frame for o in trk_a.observations)
                new_obs = [o for o in trk_b.observations if o.frame not in a_frames]
                trk_a.observations.extend(new_obs)
                trk_a.observations.sort(key=lambda o: o.frame)
                # Update state
                last = trk_a.observations[-1]
                trk_a.last_pos = (last.cx, last.cy)
                trk_a.last_frame = last.frame
                if len(trk_a.observations) >= 2:
                    prev = trk_a.observations[-2]
                    ddt = last.frame - prev.frame
                    if ddt > 0:
                        trk_a.last_vel = ((last.cx - prev.cx) / ddt,
                                          (last.cy - prev.cy) / ddt)
                trk_a._avg_speed, _ = _track_speed_stats(trk_a)
                trk_a._is_static = trk_a._avg_speed < min_speed_threshold
                tracks.pop(best_j)
                merged = True
            else:
                i += 1

    return tracks

def _is_track_moving_for_merge(trk: Track, cfg: SelectorConfig, diag: float) -> bool:
    """Strict moving predicate for movement-based merge pass."""
    if trk.num_obs < 3:
        return False
    sb = trk.score_breakdown if trk.score_breakdown else {}
    avg_speed_fallback, peak_speed_fallback = _track_speed_stats(trk)
    avg_speed = float(sb.get("avg_speed_pxpf", avg_speed_fallback))
    peak_speed = float(sb.get("peak_speed_pxpf", peak_speed_fallback))
    extent_px = float(sb.get("extent_px", _track_extent(trk)))

    min_avg = _fps_norm_pxpf(cfg.movement_merge_min_avg_speed, cfg)
    min_peak = _fps_norm_pxpf(cfg.movement_merge_min_peak_speed, cfg)
    min_extent = max(12.0, cfg.movement_merge_min_extent_frac * diag)

    moving_speed_ok = (avg_speed >= min_avg) or (peak_speed >= min_peak)
    moving_extent_ok = extent_px >= min_extent
    return moving_speed_ok and moving_extent_ok

def _merge_high_movement_tracks(
    tracks: List[Track],
    cfg: SelectorConfig
) -> Tuple[List[Track], int]:
    """Merge candidate tracks that are moving and exceed movement-score threshold."""
    if len(tracks) < 2:
        return tracks, 0

    diag = _cfg_diag(cfg)
    move_min = max(float(cfg.movement_merge_min_score), 0.0)
    moving = [
        t for t in tracks
        if _is_track_moving_for_merge(t, cfg, diag)
        and _track_movement_score(t, cfg, diag) >= move_min
    ]
    if len(moving) < 2:
        return tracks, 0

    anchor = max(moving, key=lambda t: t.score)
    merged = Track(track_id=anchor.track_id, cfg=cfg)

    # Keep one observation per frame, preferring higher-confidence detections.
    best_obs_by_frame: Dict[int, Detection] = {}
    for trk in moving:
        for obs in trk.observations:
            prev = best_obs_by_frame.get(obs.frame)
            if prev is None or obs.conf > prev.conf:
                best_obs_by_frame[obs.frame] = obs

    merged_obs = [best_obs_by_frame[f] for f in sorted(best_obs_by_frame.keys())]
    if len(merged_obs) < 2:
        return tracks, 0

    merged.observations = merged_obs
    merged.last_pos = (merged_obs[-1].cx, merged_obs[-1].cy)
    merged.last_frame = merged_obs[-1].frame
    if len(merged_obs) >= 2:
        prev = merged_obs[-2]
        last = merged_obs[-1]
        dt = max(last.frame - prev.frame, 1)
        merged.last_vel = ((last.cx - prev.cx) / dt, (last.cy - prev.cy) / dt)

    moving_ids = {id(t) for t in moving}
    kept = [t for t in tracks if id(t) not in moving_ids]
    kept.append(merged)
    return kept, len(moving)

def _build_track_guide(
    track: Optional[Track],
    total_frames: int,
    cfg: SelectorConfig,
    max_interp_gap: int = 20,
    apply_filters: bool = True,
    filter_static_jumps: bool = False,
    all_dets: Optional[List[List["Detection"]]] = None,
) -> Tuple[Dict[int, Tuple[float, float, bool]], int]:
    """
    Build per-frame guide positions from a chosen track.
    
    Gap-fill uses physics-based KF prediction (gravity + drag) instead of
    linear interpolation, producing parabolic arcs during carry frames.
    
    apply_filters: full filter suite (legacy mode).
    filter_static_jumps: lighter filter — only removes static-ball snaps and
        position spikes, without trimming the leading edge. Used in trail_only
        mode to prevent guide from jumping to parked balls without losing
        early-rally coverage.
    
    Returns: (frame_idx -> (cx, cy, is_exact_observation), dropped_spike_count)
    """
    guide: Dict[int, Tuple[float, float, bool]] = {}
    if track is None or track.num_obs == 0:
        return guide, 0

    diag = _cfg_diag(cfg)
    obs_raw = sorted(track.observations, key=lambda o: o.frame)
    obs = list(obs_raw)
    dropped = 0
    if apply_filters:
        obs, dropped = _filter_guide_observations(obs, cfg, diag)
        obs, dropped_static_prefix = _trim_leading_static_guide_obs(obs, cfg, diag)
        dropped += dropped_static_prefix
        obs, dropped_static_runs = _prune_static_guide_runs(obs, cfg, diag)
        dropped += dropped_static_runs
    elif filter_static_jumps:
        # Lightweight: only remove spikes and static-ball snaps.
        # Do NOT trim the leading edge — that would lose early rally coverage.
        obs, dropped = _filter_guide_observations(obs, cfg, diag)
        # Still prune tight spatial clusters with no motion (parked balls that
        # got merged into the stitched track via gap-fill observations).
        obs, dropped_static_runs = _prune_static_guide_runs(obs, cfg, diag)
        dropped += dropped_static_runs
    if not obs:
        return guide, dropped

    # Tail rescue: if static-run pruning clipped a moving tail, re-attach only
    # trajectory-consistent and non-static late observations.
    if obs_raw and obs[-1].frame < obs_raw[-1].frame:
        static_speed = _guide_static_speed_thresh(cfg, diag)
        max_tail_step = max(20.0, 0.085 * diag)
        tail_candidates = [o for o in obs_raw if o.frame > obs[-1].frame]
        prev_o = obs[-1]
        rescued = []
        for o in tail_candidates:
            dt = max(int(o.frame - prev_o.frame), 1)
            step = _xy_dist(prev_o.cx, prev_o.cy, o.cx, o.cy) / dt
            if step > max_tail_step:
                continue
            rescued.append(o)
            prev_o = o
        if rescued:
            obs.extend(rescued)

    # Place exact observations
    for o in obs:
        if 0 <= o.frame < total_frames:
            guide[o.frame] = (o.cx, o.cy, True)

    # ── Physics-based gap fill using BallKalmanFilter ──
    # For each gap between consecutive observations, build a temporary KF
    # seeded from the pre-gap observations, then predict forward with gravity
    # and drag to produce parabolic arcs instead of straight lines.
    has_kf = KalmanFilter is not None
    kf_gaps_filled = 0
    linear_gaps_filled = 0
    for i in range(1, len(obs)):
        a = obs[i - 1]
        b = obs[i]
        gap = b.frame - a.frame
        if gap <= 1 or gap > max_interp_gap:
            continue

        if has_kf and gap >= 2:
            # Build a temporary KF from recent observations before the gap.
            # Seed it with the last 2+ observations so it has good velocity.
            seed_start = max(0, i - 5)
            seed_obs = obs[seed_start:i]
            if len(seed_obs) >= 2:
                kf_tmp = BallKalmanFilter(float(seed_obs[0].cx), float(seed_obs[0].cy), cfg)
                # Feed observations to build velocity estimate
                for si in range(1, len(seed_obs)):
                    so = seed_obs[si]
                    dt_seed = so.frame - seed_obs[si - 1].frame
                    for _ in range(max(dt_seed, 1)):
                        kf_tmp.predict()
                    kf_tmp.update(float(so.cx), float(so.cy), conf=getattr(so, "conf", None))

                # Now predict forward through the gap
                for f in range(a.frame + 1, b.frame):
                    if not (0 <= f < total_frames):
                        continue
                    if f in guide:
                        continue
                    dt_gap = f - a.frame
                    px, py = kf_tmp.predict_dt(dt_gap)
                    guide[f] = (px, py, False)
                kf_gaps_filled += 1
                continue

        # Fallback for short gaps (1-2 frames) or when filterpy unavailable:
        # linear interpolation is fine for very short gaps
        for f in range(a.frame + 1, b.frame):
            if not (0 <= f < total_frames):
                continue
            t = (f - a.frame) / float(gap)
            cx = a.cx + (b.cx - a.cx) * t
            cy = a.cy + (b.cy - a.cy) * t
            if f not in guide:
                guide[f] = (cx, cy, False)
        linear_gaps_filled += 1

    if kf_gaps_filled > 0 or linear_gaps_filled > 0:
        print(f"[guide] Gap fill: {kf_gaps_filled} physics (KF), {linear_gaps_filled} linear"
              f" | has_kf={has_kf}, gravity={cfg.gravity_px_per_frame2:.3f} px/f²"
              f" | gravity_enabled={cfg.gravity_enabled}")

    # ── Snap interpolated (non-exact) gap frames to real detections ──
    # KF gap-fill produces smooth arcs, but if the ball was actually detected
    # in those frames (e.g. right after a hit, before the next chain segment's
    # first observation), snap the guide to the real detection instead of the
    # physics prediction. This fixes the "missing early trajectory" problem where
    # a detection appears in the ROI but no green trail is drawn because the frame
    # was covered only by a KF prediction, not a chain-track observation.
    if all_dets is not None:
        snap_radius = max(20.0, 0.06 * diag)  # conservative: must be genuinely close
        snapped = 0
        for f, gval in list(guide.items()):
            gx, gy, g_exact = gval
            if g_exact:
                continue  # already an exact observation, don't override
            if not (0 <= f < len(all_dets)):
                continue
            frame_dets = all_dets[f]
            if not frame_dets:
                continue
            # Find the closest detection on motion within the snap radius.
            best_d: Optional["Detection"] = None
            best_dist = snap_radius
            for d in frame_dets:
                if not bool(getattr(d, "on_motion", False)):
                    continue  # only snap to moving detections — avoids static balls
                dd = _xy_dist(float(d.cx), float(d.cy), gx, gy)
                if dd < best_dist:
                    best_dist = dd
                    best_d = d
            if best_d is not None:
                guide[f] = (float(best_d.cx), float(best_d.cy), True)
                snapped += 1
        if snapped > 0:
            print(f"[guide] Snapped {snapped} gap-fill frames to real on-motion detections")

    # ── Physics-based tail extension ──
    # Use KF prediction instead of constant-velocity extrapolation.
    if apply_filters and len(obs) >= 2:
        last = obs[-1]
        prev = obs[-2]
        dt_tail = max(int(last.frame - prev.frame), 1)
        vx = (last.cx - prev.cx) / dt_tail
        vy = (last.cy - prev.cy) / dt_tail
        speed = math.sqrt(vx * vx + vy * vy)
        static_speed = _guide_static_speed_thresh(cfg, diag)
        moving_tail = speed > static_speed * 0.90
        if moving_tail and 0 <= int(last.frame) < total_frames - 1:
            tail_horizon = min(
                max_interp_gap,
                max(int(getattr(cfg, "carry_interp_frames", 3)) + 14, 12)
            )
            end_f = min(total_frames - 1, int(last.frame) + int(tail_horizon))

            if has_kf and len(obs) >= 3:
                # Build KF from last few observations for physics tail
                seed_start = max(0, len(obs) - 6)
                seed_obs = obs[seed_start:]
                kf_tail = BallKalmanFilter(float(seed_obs[0].cx), float(seed_obs[0].cy), cfg)
                for si in range(1, len(seed_obs)):
                    so = seed_obs[si]
                    dt_seed = so.frame - seed_obs[si - 1].frame
                    for _ in range(max(dt_seed, 1)):
                        kf_tail.predict()
                    kf_tail.update(float(so.cx), float(so.cy), conf=getattr(so, "conf", None))
                for f in range(int(last.frame) + 1, end_f + 1):
                    if f in guide:
                        continue
                    dt_ext = f - int(last.frame)
                    px, py = kf_tail.predict_dt(dt_ext)
                    guide[f] = (px, py, False)
            else:
                # Fallback: constant velocity
                for f in range(int(last.frame) + 1, end_f + 1):
                    if f in guide:
                        continue
                    dti = f - int(last.frame)
                    guide[f] = (last.cx + vx * dti, last.cy + vy * dti, False)

    return guide, dropped