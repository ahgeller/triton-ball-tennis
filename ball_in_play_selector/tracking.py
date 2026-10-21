import math
import numpy as np
from typing import Optional, List, Tuple

from .config import SelectorConfig
from .models import Detection, Track
from .utils import _cfg_diag, _fps_norm_pxpf, _ensure_mask_u8
from .physics import _xy_dist


def _static_speed_threshold(cfg: SelectorConfig, diag: float) -> float:
    """Static-speed threshold in px/frame, capped to avoid over-scaling with resolution."""
    diag_based = float(cfg.assoc_static_speed_frac) * float(diag) * 0.95
    fps_capped = _fps_norm_pxpf(4.0, cfg)
    return max(1.2, min(diag_based, fps_capped))


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
        if frame_dets and has_motion and t < len(raw_motions):
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

    recent = trk.observations[-8:]
    if len(recent) == 8 and bool(det.on_motion):
        recent_dt = max(recent[-1].frame - recent[0].frame, 1)
        recent_speed = sum(
            _xy_dist(a.cx, a.cy, b.cx, b.cy)
            for a, b in zip(recent, recent[1:])
        ) / recent_dt
        recent_motion = sum(bool(obs.on_motion) for obs in recent) / len(recent)
        static_speed = _static_speed_threshold(cfg, diag)
        if recent_speed <= static_speed and recent_motion < 0.25 and obs_speed > 2.0 * static_speed:
            return False

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
    stitch_enable = bool(cfg.trail_stitch_enable)
    stitch_gap_frames = max(1, int(cfg.trail_stitch_gap_frames))
    stitch_min_stale = max(1, int(cfg.trail_stitch_min_stale_frames))
    stitch_min_obs = max(1, int(cfg.trail_stitch_min_obs))
    stitch_dist_px = max(8.0, float(cfg.trail_stitch_dist_frac) * diag)
    stitch_max_step_px = max(10.0, float(cfg.trail_stitch_max_step_frac) * diag)
    stitch_pred_resid_px = max(10.0, float(cfg.trail_stitch_pred_resid_frac) * diag)
    jump_break_px = max(10.0, float(cfg.trail_jump_break_frac) * diag)
    jump_break_growth_px = max(0.0, float(cfg.trail_jump_break_growth_frac) * diag)

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
                        recent = trk_prev.observations[-8:]
                        recent_dt = max(recent[-1].frame - recent[0].frame, 1)
                        recent_distance = sum(
                            _xy_dist(a.cx, a.cy, b.cx, b.cy)
                            for a, b in zip(recent, recent[1:])
                        )
                        recent_speed = recent_distance / float(recent_dt)
                        recent_motion = sum(bool(obs.on_motion) for obs in recent) / len(recent)
                        if (
                            recent_speed < _fps_norm_pxpf(1.0, cfg)
                            and recent_motion < 0.25
                        ):
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

