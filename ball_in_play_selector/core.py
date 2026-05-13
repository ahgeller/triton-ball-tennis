# Imports
import math
import json
import os
import numpy as np
import cv2
from types import SimpleNamespace
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
from .tracking import build_detections, build_motion_tracks, build_tracks_ultra, build_tracks, merge_tracks, _build_track_guide, _guide_static_speed_thresh, _det_hard_continuity_ok, _filter_guide_observations, _trim_leading_static_guide_obs, _prune_static_guide_runs, _merge_high_movement_tracks
from .scoring import score_tracks, select_best_track, _track_movement_score, _is_stationary_track, _annotate_track_periods, _stitch_track_chain, _select_timeline_chain


def _find_motion_blob(boost_mask, search_cx, search_cy, search_radius,
                      last_det_cx, last_det_cy, last_vel,
                      min_blob_area=20, max_blob_area=600,
                      ref_ball_area: Optional[float] = None,
                      player_boxes: Optional[List[Tuple[float, float, float, float]]] = None,
                      frame_idx: int = -1,
                      active_motion_tracks: Optional[List['MotionTrack']] = None,
                      prev_motion_pos: Optional[Tuple[float, float]] = None):
    """Find a motion blob near search position that's plausibly the ball.
    
    Args:
        boost_mask: the preprocessed motion mask
        search_cx/cy: predicted position (projectile physics)  
        search_radius: how far to look
        last_det_cx/cy: where the ball was LAST ACTUALLY DETECTED by YOLO
        last_vel: (vx, vy) estimated velocity
        min/max_blob_area: reject blobs outside this range
        ref_ball_area: last known ball area to keep search local around ball-size motion
        player_boxes: optional exact player bboxes for anti-leg/blob penalties
        prev_motion_pos: (cx, cy) of the PREVIOUS motion blob — for trajectory continuity
    
    Returns (cx, cy, area, is_latched) or None
    """
    # ── FAST PATH: Latch onto a continuous motion track ──
    # Track membership is strong evidence, but require the boost_mask to still
    # support the latched point (small motion blob within ~6 px). Without this
    # check, drifting tracks or stale points get accepted even when no motion
    # actually exists at that pixel on this frame.
    if active_motion_tracks is not None and frame_idx >= 0:
        best_track_dist = float('inf')
        best_track_pt = None
        for track in active_motion_tracks:
            pt = track.get_position_at(frame_idx)
            if pt is None:
                continue
            dist = math.hypot(pt[0] - search_cx, pt[1] - search_cy)
            if dist <= search_radius and dist < best_track_dist:
                best_track_dist = dist
                best_track_pt = pt
        if best_track_pt is not None:
            supported = False
            if boost_mask is not None:
                bh_, bw_ = boost_mask.shape[:2]
                tx = int(round(best_track_pt[0]))
                ty = int(round(best_track_pt[1]))
                support_r = 6
                sx1 = max(0, tx - support_r)
                sy1 = max(0, ty - support_r)
                sx2 = min(bw_, tx + support_r + 1)
                sy2 = min(bh_, ty + support_r + 1)
                if sx2 > sx1 and sy2 > sy1:
                    supported = bool(boost_mask[sy1:sy2, sx1:sx2].max() > 0)
            if supported:
                mock_area = float(ref_ball_area) if ref_ball_area else 100.0
                return (best_track_pt[0], best_track_pt[1], mock_area, True)
            # Fall through to regular contour-based search if the latched point
            # has no underlying motion on this frame.

    if boost_mask is None:
        return None
    h, w = boost_mask.shape[:2]
    
    # Keep the actual search ROI centered on the same predicted point used by the
    # debug circle overlay. This avoids the visible "diagonal"/offset mismatch
    # caused by a hidden crop bias toward the last detection.
    x1 = max(0, int(search_cx - search_radius))
    y1 = max(0, int(search_cy - search_radius))
    x2 = min(w, int(search_cx + search_radius))
    y2 = min(h, int(search_cy + search_radius))
    if x2 <= x1 or y2 <= y1:
        return None
    
    roi = boost_mask[y1:y2, x1:x2]
    if roi.max() == 0:
        return None
    
    thresh = (roi > 30).astype(np.uint8) * 255
    # Enforce a circular search ROI so the real motion-search area matches the
    # displayed search radius (including frozen/expanded guide-style debugging).
    local_cx = float(search_cx) - float(x1)
    local_cy = float(search_cy) - float(y1)
    local_r = max(1, int(math.ceil(float(search_radius))))
    circle_mask = np.zeros_like(thresh, dtype=np.uint8)
    cv2.circle(
        circle_mask,
        (int(round(local_cx)), int(round(local_cy))),
        local_r,
        255,
        -1,
        lineType=cv2.LINE_8,
    )
    thresh = cv2.bitwise_and(thresh, circle_mask)
    if thresh.max() == 0:
        return None
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    vel_mag = math.sqrt(last_vel[0]**2 + last_vel[1]**2)
    
    best_blob = None
    best_score = float('inf')
    best_anchor_blob = None
    best_anchor_score = float('inf')

    ref_area = float(ref_ball_area) if ref_ball_area is not None else 0.0
    # "Ball detection area" anchor: prefer motion very near the predicted/last ball
    # footprint before considering arbitrary motion in the wider search circle.
    ball_r_px = math.sqrt(max(ref_area, 1.0) / math.pi) if ref_area > 0.0 else 0.0
    anchor_r_pred = max(8.0, min(search_radius * 0.38, 3.6 * ball_r_px + 5.0 if ball_r_px > 0 else search_radius * 0.22))
    anchor_r_det = max(anchor_r_pred, min(search_radius * 0.55, anchor_r_pred * 1.45))

    def _player_dist_local(px: float, py: float) -> Optional[float]:
        if not player_boxes:
            return None
        best = None
        for x1p, y1p, x2p, y2p in player_boxes:
            dx = max(x1p - px, 0.0, px - x2p)
            dy = max(y1p - py, 0.0, py - y2p)
            d = math.sqrt(dx * dx + dy * dy)
            if best is None or d < best:
                best = d
        return best
    
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_blob_area or area > max_blob_area:
            continue
        
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        
        blob_cx = M["m10"] / M["m00"] + x1
        blob_cy = M["m01"] / M["m00"] + y1
        
        # ── Shape filter: tennis ball should be roughly circular ──
        # Bounding rect aspect ratio
        bx, by, bw, bh = cv2.boundingRect(c)
        if bw > 0 and bh > 0:
            aspect = max(bw, bh) / max(min(bw, bh), 1)
            if aspect > 3.0:
                continue  # Too elongated — probably a shadow or edge
            # Circularity: area / bounding_rect_area
            fill_ratio = area / max(bw * bh, 1)
            if fill_ratio < 0.25:
                continue  # Too hollow / irregular — not a ball
                
        # ── Anchor to bottom ──
        # To align with YOLO bounding boxes (which anchor the physics trail
        # at the bottom-center of the box), push the raw motion blob's Y
        # coordinate to the bottom of its bounding box.
        blob_cy = by + bh
        
        # Distance from predicted position (projectile prediction)
        dist_from_pred = math.sqrt((blob_cx - search_cx)**2 + 
                                    (blob_cy - search_cy)**2)
        if dist_from_pred > float(search_radius):
            continue
        
        # Distance from last YOLO detection
        dist_from_det = math.sqrt((blob_cx - last_det_cx)**2 + 
                                   (blob_cy - last_det_cy)**2)
        
        # Velocity consistency: blob direction should agree with velocity.
        vel_penalty = 0.0
        if vel_mag > 2.0:
            dx = blob_cx - last_det_cx
            dy = blob_cy - last_det_cy
            blob_mag = math.sqrt(dx*dx + dy*dy)
            if blob_mag > 2.0:
                cos_sim = (dx * last_vel[0] + dy * last_vel[1]) / (blob_mag * vel_mag)
                # Allow bounces: penalize backward motion but do not hard-reject
                if cos_sim < 0.05:
                    vel_penalty = 80.0
                elif cos_sim < 0.3:
                    vel_penalty = 50.0
            elif blob_mag <= 2.0 and vel_mag > 5.0:
                # Ball is moving fast but blob hasn't moved from last det —
                # this is static noise, not the ball
                continue
        
        # ── Static blob rejection ──
        # If the ball is supposed to be moving (carry/predict), reject blobs
        # that are suspiciously close to a FIXED position (didn't move between
        # the last detection and the predicted position).
        # Compare blob distance from prediction vs distance from last detection:
        # if blob is closer to last-det than to prediction, it's likely static.
        if vel_mag > 4.0 and dist_from_pred > 0:
            # How far the prediction has moved from last det
            pred_travel = math.sqrt((search_cx - last_det_cx)**2 + (search_cy - last_det_cy)**2)
            if pred_travel > 8.0:
                # Blob should be roughly in the direction of travel, not stuck at origin
                blob_progress = dist_from_det / max(pred_travel, 1.0)
                if blob_progress < 0.15:
                    # Blob hasn't moved at all relative to the predicted travel — static
                    continue
        
        # Size constraint: hard reject blobs that are drastically different from the known ball size
        size_penalty = 0.0
        if ref_area > 0.0:
            area_ratio = max(area, 1.0) / ref_area
            # Hard limit: ball area should not suddenly increase by more than 3x or shrink to less than 1/3x
            if area_ratio > 3.0 or area_ratio < 0.33:
                continue
            
            # Apply a light penalty for minor size deviations within the valid range
            size_penalty = abs(math.log(area_ratio)) * 15.0
        else:
            ideal_area = 80.0
            area_ratio = max(area, 1.0) / ideal_area
            size_penalty = abs(math.log(area_ratio)) * 10.0
        
        # Anti-player clutter: leg/racket motion can dominate the boost mask.
        player_penalty = 0.0
        pd_player = _player_dist_local(blob_cx, blob_cy)
        if pd_player is not None:
            # Replaced hard 'continue' with a penalty so racket hits inside the player box
            # aren't instantly destroyed when the ball deviates from the forward prediction.
            if pd_player <= 0.0 and dist_from_pred > max(8.0, 0.60 * anchor_r_pred):
                player_penalty += 45.0
            near_player_band = max(10.0, 0.85 * anchor_r_pred)
            if pd_player < near_player_band:
                player_penalty += (near_player_band - max(0.0, pd_player)) * 5.0

        # ── Continuity preference from previous motion blob ──
        # If we had a motion blob last frame, prefer blobs near it (penalty, not reject).
        # Don't hard-reject because there can be multiple valid ball trajectories.
        continuity_penalty = 0.0
        if prev_motion_pos is not None:
            dist_from_prev = math.sqrt(
                (blob_cx - prev_motion_pos[0])**2 + (blob_cy - prev_motion_pos[1])**2
            )
            expected_step = max(vel_mag * 1.5, 8.0)
            if dist_from_prev > expected_step:
                continuity_penalty = (dist_from_prev - expected_step) * 1.5

        score = (
            dist_from_pred * 2.4 +
            dist_from_det * 0.35 +
            vel_penalty +
            size_penalty +
            player_penalty +
            continuity_penalty
        )

        anchor_hit = (dist_from_pred <= anchor_r_pred) or (dist_from_det <= anchor_r_det)
        if anchor_hit:
            if score < best_anchor_score:
                best_anchor_score = score
                best_anchor_blob = (blob_cx, blob_cy, area)
            continue

        if score < best_score:
            best_score = score
            best_blob = (blob_cx, blob_cy, area)

    if best_anchor_blob is not None:
        best_blob = best_anchor_blob

    if best_blob is None:
        return None
    
    # Final check: blob must be reasonably close to predicted position
    max_drift = search_radius * 0.9
    dist_to_pred = math.sqrt((best_blob[0] - search_cx)**2 + 
                              (best_blob[1] - search_cy)**2)
    if dist_to_pred > max_drift:
        return None
    
    return best_blob[0], best_blob[1], best_blob[2], False

def _motion_blob_physics_ok(
    blob_cx: float,
    blob_cy: float,
    pred_cx: float,
    pred_cy: float,
    last_det_pos: Tuple[float, float],
    last_vel: Tuple[float, float],
    frames_since_det: int,
    prev_motion_vel: Optional[Tuple[float, float]],
    cfg: SelectorConfig,
    diag: float
) -> bool:
    """Reject motion blobs that violate basic constant-velocity physics."""
    dt = max(int(frames_since_det), 1)
    expected_v = math.sqrt(last_vel[0] ** 2 + last_vel[1] ** 2)

    # Hard continuity budget from the current anchor state.
    max_step = max(
        10.0,
        min(
            cfg.max_speed_px_per_frame * dt * 1.05,
            0.09 * diag * (1.0 + 0.16 * (dt - 1))
        )
    )
    d_anchor = math.sqrt((blob_cx - last_det_pos[0]) ** 2 + (blob_cy - last_det_pos[1]) ** 2)
    if d_anchor > max_step:
        pred_x, pred_y = _predict_projectile(last_det_pos, last_vel, dt, cfg)
        d_pred_step = math.sqrt((blob_cx - pred_x) ** 2 + (blob_cy - pred_y) ** 2)
        if d_pred_step > max_step * 1.10:
            return False

    # Must stay reasonably close to the predicted position.
    resid = math.sqrt((blob_cx - pred_cx) ** 2 + (blob_cy - pred_cy) ** 2)
    max_resid = max(7.0, cfg.motion_pred_residual_frac * diag * (1.0 + 0.18 * (dt - 1)))
    if resid > max_resid:
        return False

    # Velocity direction check against the last detection velocity.
    vx = (blob_cx - last_det_pos[0]) / dt
    vy = (blob_cy - last_det_pos[1]) / dt
    obs_speed = math.sqrt(vx * vx + vy * vy)

    motion_min_speed = _fps_norm_pxpf(cfg.motion_use_min_det_speed, cfg)
    if expected_v >= motion_min_speed:
        speed_ratio = obs_speed / max(expected_v, 1e-6)
        if (speed_ratio < cfg.motion_speed_ratio_min or
                speed_ratio > cfg.motion_speed_ratio_max):
            return False

        dot = vx * last_vel[0] + vy * last_vel[1]
        cos_sim = dot / max(obs_speed * expected_v, 1e-6)
        cos_min = max(-1.0, min(1.0, float(cfg.motion_dir_cos_min)))
        if cos_sim < cos_min:
            return False

    # Acceleration sanity vs previous motion-frame velocity.
    if prev_motion_vel is not None:
        dvx = vx - prev_motion_vel[0]
        dvy = vy - prev_motion_vel[1]
        accel = math.sqrt(dvx * dvx + dvy * dvy)
        if accel > max(6.0, cfg.motion_accel_frac * diag):
            return False

    return True

def _physics_guide_motion_blob(
    blob_cx: float,
    blob_cy: float,
    pred_cx: float,
    pred_cy: float,
    last_pos: Optional[Tuple[float, float]],
    last_vel: Tuple[float, float],
    frames_since_det: int,
    cfg: SelectorConfig,
    diag: float
) -> Tuple[float, float, bool]:
    """
    Guide accepted motion blob toward physically plausible trajectory.
    Returns (guided_cx, guided_cy, was_clamped).
    """
    dt = max(int(frames_since_det), 1)
    resid = math.sqrt((blob_cx - pred_cx) ** 2 + (blob_cy - pred_cy) ** 2)
    soft = max(4.0, cfg.motion_guided_soft_resid_frac * diag)
    hard = max(8.0, cfg.motion_pred_residual_frac * diag * (1.0 + 0.25 * (dt - 1)))

    # Dynamic blob weight: trust blob when close to prediction; otherwise pull toward prediction.
    if resid <= soft:
        w_blob = cfg.motion_guided_blob_weight_hi
    else:
        t = min(max((resid - soft) / max(hard - soft, 1e-6), 0.0), 1.0)
        w_blob = ((1.0 - t) * cfg.motion_guided_blob_weight_hi +
                  t * cfg.motion_guided_blob_weight_lo)

    gx = w_blob * blob_cx + (1.0 - w_blob) * pred_cx
    gy = w_blob * blob_cy + (1.0 - w_blob) * pred_cy
    clamped = False

    # Step clamp from last known position to avoid one-frame crazy jumps.
    if last_pos is not None:
        expected_step = math.sqrt(last_vel[0] ** 2 + last_vel[1] ** 2) * dt
        max_step = max(cfg.motion_guided_min_step_px,
                       cfg.motion_guided_step_ratio * expected_step)
        dx = gx - last_pos[0]
        dy = gy - last_pos[1]
        step = math.sqrt(dx * dx + dy * dy)
        if step > max_step and step > 1e-6:
            s = max_step / step
            gx = last_pos[0] + dx * s
            gy = last_pos[1] + dy * s
            clamped = True

    return gx, gy, clamped

def _guide_path_consistent(
    gx: float,
    gy: float,
    last_pos: Optional[Tuple[float, float]],
    last_vel: Tuple[float, float],
    frames_since_det: int,
    cfg: SelectorConfig,
    diag: float,
    guide_exact: bool = False
) -> bool:
    """Require guide fallback to stay on the current local trajectory."""
    # Exact guide = real observation from the best track. Trust it.
    if guide_exact:
        return last_pos is not None
    if last_pos is None:
        return False

    dt = max(int(frames_since_det), 1)
    pred_x = last_pos[0] + last_vel[0] * dt
    pred_y = last_pos[1] + last_vel[1] * dt

    # Relaxed growth factors so guide can re-enter after carry drift.
    max_resid = max(12.0, 0.040 * diag * (1.0 + 0.50 * (dt - 1)))
    resid = _xy_dist(gx, gy, pred_x, pred_y)
    if resid > max_resid:
        return False

    step = _xy_dist(gx, gy, last_pos[0], last_pos[1])
    max_step = max(16.0, 0.070 * diag * (1.0 + 0.55 * (dt - 1)))
    if step > max_step:
        return False

    exp_vx, exp_vy = last_vel
    exp_speed = math.sqrt(exp_vx * exp_vx + exp_vy * exp_vy)
    obs_vx = (gx - last_pos[0]) / dt
    obs_vy = (gy - last_pos[1]) / dt
    obs_speed = math.sqrt(obs_vx * obs_vx + obs_vy * obs_vy)

    motion_min_speed = _fps_norm_pxpf(cfg.motion_use_min_det_speed, cfg)
    if exp_speed >= motion_min_speed:
        ratio = obs_speed / max(exp_speed, 1e-6)
        # Wider bounds after long drift (dt > 4) so guide can re-enter.
        if dt > 4:
            if ratio < 0.12 or ratio > 3.50:
                return False
        else:
            if ratio < 0.20 or ratio > 2.80:
                return False
        if obs_speed > 1e-6:
            dot = obs_vx * exp_vx + obs_vy * exp_vy
            cos_sim = dot / max(obs_speed * exp_speed, 1e-6)
            if cos_sim < -0.20:
                return False

    return True

def select_ball_in_play(
    detections_by_frame: List[List[Tuple[list, float]]],
    fps: float,
    width: int,
    height: int,
    court_polygon=None,
    boost_masks: Optional[List[Optional[np.ndarray]]] = None,
    raw_motions: Optional[List[Optional[np.ndarray]]] = None,
    player_boxes_by_frame=None,
    court_keypoints=None,
    emit_guide_debug_meta: bool = False,
    debug: bool = False
) -> Tuple[List[Optional[FrameResult]], Optional[Track], List[Track], List['MotionTrack']]:
    """
    Main entry: select the in-play ball from per-frame YOLO detections.

    Args:
        detections_by_frame: list of T lists of (bbox, conf) tuples
        fps, width, height: video info
        court_polygon: cv2 contour (convex hull of court keypoints) or None
        boost_masks: per-frame motion/boost masks (optional, for motion bonus)
        court_keypoints: flat list of court keypoint coordinates (x0,y0,x1,y1,...) or None
        emit_guide_debug_meta: if True, attach per-frame guide gate metadata for guide-debug rendering
        debug: if True, print score breakdowns

    Returns:
        (per_frame_results, chosen_track, all_tracks, motion_tracks)
    """
    cfg = SelectorConfig(fps=fps, width=width, height=height).auto_scale()
    total_frames = len(detections_by_frame)

    # ── Compute court reference length from sidelines ──
    court_ref_length = None
    if court_keypoints is not None and len(court_keypoints) >= 16:
        def _kp_xy(kps, idx):
            i = idx * 2
            if i + 1 < len(kps):
                x, y = float(kps[i]), float(kps[i + 1])
                if x > 0 or y > 0:
                    return (x, y)
            return None
        p0 = _kp_xy(court_keypoints, 0)  # TL
        p3 = _kp_xy(court_keypoints, 3)  # TR
        p4 = _kp_xy(court_keypoints, 4)  # BL
        p7 = _kp_xy(court_keypoints, 7)  # BR
        sides = []
        if p0 is not None and p4 is not None:
            sides.append(math.sqrt((p0[0] - p4[0])**2 + (p0[1] - p4[1])**2))
        if p3 is not None and p7 is not None:
            sides.append(math.sqrt((p3[0] - p7[0])**2 + (p3[1] - p7[1])**2))
        if sides:
            court_ref_length = max(sides)
            if debug:
                print(f"[selector] Court ref length: {court_ref_length:.1f}px (sidelines: {sides})")

    diag_init = _cfg_diag(cfg)
    if court_ref_length is None or court_ref_length < 50.0:
        court_ref_length = diag_init
        if debug:
            print(f"[selector] Court ref length: using diag fallback {court_ref_length:.1f}px")
    audit_start_s = os.environ.get("BALL_AUDIT_START", "").strip()
    audit_end_s = os.environ.get("BALL_AUDIT_END", "").strip()
    audit_path = os.environ.get("BALL_AUDIT_PATH", "").strip()
    try:
        audit_start = int(audit_start_s) if audit_start_s else -1
        audit_end = int(audit_end_s) if audit_end_s else -1
    except ValueError:
        audit_start, audit_end = -1, -1
    audit_enabled = audit_start >= 0 and audit_end >= audit_start
    audit_rows: List[Dict[str, Any]] = []
    audit_track_rows: List[Dict[str, Any]] = []

    def _build_detection_owner_map(tracks_src: List[Track]) -> Dict[int, int]:
        """Map Detection object-id -> owning track-id (for anti-cross-grab gating)."""
        owners: Dict[int, int] = {}
        for trk in tracks_src:
            tid = int(trk.track_id)
            for obs in trk.observations:
                owners[id(obs)] = tid
        return owners

    def _track_motion_frac(trk: Track) -> float:
        """Return track motion overlap fraction (0..1), using cached score if present."""
        sb = trk.score_breakdown if trk.score_breakdown else {}
        if "motion_frac" in sb:
            try:
                return float(sb.get("motion_frac", 0.0))
            except Exception:
                return 0.0
        n = max(int(trk.num_obs), 1)
        m = sum(1 for o in trk.observations if bool(getattr(o, "on_motion", False)))
        return float(m) / float(n)

    # Step 0: build detection objects
    all_dets = build_detections(detections_by_frame, boost_masks)
    
    # Pre-build continuous motion tracks for smooth gap patch latching
    motion_tracks = build_motion_tracks(boost_masks, cfg)

    # Step 1: build track hypotheses
    track_backend = str(getattr(cfg, "track_builder_backend", "ultra")).lower()
    if track_backend == "ultra":
        try:
            tracks = build_tracks_ultra(all_dets, cfg)
        except Exception as e:
            if debug:
                print(f"[selector] Ultralytics track builder failed ({e}); falling back to greedy builder")
            tracks = build_tracks(all_dets, cfg)
    else:
        tracks = build_tracks(all_dets, cfg)

    if debug:
        total_dets = sum(len(fd) for fd in all_dets)
        print(f"[selector] Built {len(tracks)} raw tracks "
              f"from {total_dets} total detections across {total_frames} frames")

    # Step 1.5: merge fragmented tracks (same ball lost at bounces)
    #tracks = merge_tracks(tracks, cfg, court_poly=court_polygon,
                          #player_boxes_by_frame=player_boxes_by_frame)

    if debug:
        print(f"[selector] After merging: {len(tracks)} tracks")
        print(f"[selector] Court polygon: {'provided' if court_polygon is not None else 'NONE'}")

    # Compute court homography once and inject into all Track KFs for depth-aware gravity.
    _court_hom = build_court_homography(court_keypoints)
    if _court_hom is not None:
        _H, _H_inv, _cw, _ch = _court_hom
        for _trk in tracks:
            if _trk.kf is not None:
                _trk.kf.set_homography(_H, _H_inv, _cw)
        if debug:
            print(f"[selector] Court homography active: {len([t for t in tracks if t.kf is not None])} KFs updated")
    else:
        _H, _H_inv, _cw = None, None, 10.97
        if court_polygon is not None:
            for fl in all_dets:
                for d in fl:
                    td = cv2.pointPolygonTest(court_polygon,
                                              (float(d.cx), float(d.cy)), True)
                    print(f"[selector] Sample det ({d.cx:.0f},{d.cy:.0f}) "
                          f"polygon dist={td:.1f} ({'inside' if td >= 0 else 'outside'})")
                    break
                if fl:
                    break

    # Step 2: score
    tracks = score_tracks(tracks, cfg, court_poly=court_polygon,
                          player_boxes_by_frame=player_boxes_by_frame,
                          total_frames=total_frames)
    _annotate_track_periods(tracks, total_frames, cfg)
    if audit_enabled:
        for trk in tracks:
            sb = trk.score_breakdown if trk.score_breakdown else {}
            audit_track_rows.append({
                "track_id": int(trk.track_id),
                "score": float(trk.score),
                "num_obs": int(trk.num_obs),
                "span": int(trk.span),
                "inside_strict_frac": float(sb.get("inside_strict_frac", sb.get("inside_frac", 0.0))),
                "motion_frac": float(sb.get("motion_frac", 0.0)),
                "near_player_frac": float(sb.get("near_player_frac", 0.0)),
                "start_frac": float(sb.get("start_frac", 0.0)),
                "end_frac": float(sb.get("end_frac", 0.0)),
            })

    if debug and tracks:
        print(f"\n  {'Track':>8} {'Score':>7} {'Obs':>5} {'Span':>5} "
              f"{'Inside%':>7} {'Cov%':>6} {'Span%':>6} {'MinObs':>6} "
              f"{'AvgV':>6} {'Ext':>6} {'JumpPen':>7} {'Move':>6} {'RF':>3} "
              f"{'P':>2} {'T0%':>5} {'T1%':>5}")
        for i, t in enumerate(tracks[:8]):
            sb = t.score_breakdown
            rf = "Y" if float(sb.get('redflag_outside_long_bool', 0.0)) > 0.5 else "-"
            pid = int(float(sb.get('period_id', 0.0)))
            print(f"  {t.track_id:>8} {t.score:>7.1f} {t.num_obs:>5} {t.span:>5} "
                  f"{sb.get('inside_frac', 0):>6.0%} {sb.get('coverage_frac', 0):>5.0%} "
                  f"{sb.get('span_frac', 0):>5.0%} {int(sb.get('min_obs_required', 0)):>6} "
                  f"{sb.get('avg_speed_pxpf', 0):>6.1f} {sb.get('extent_px', 0):>6.0f} "
                  f"{sb.get('jump_penalty', 0):>7.1f} {sb.get('movement', 0):>6.1f} {rf:>3} "
                  f"{pid:>2} {sb.get('start_frac', 0):>5.0%} {sb.get('end_frac', 0):>5.0%}")
        print()

    diag = _cfg_diag(cfg)

    # Step 3: hard blacklist tracks that should never be candidates.
    # Includes:
    # 1) sideline tracks (never on court)
    # 2) stationary tracks (parked balls)
    # 3) off-context tracks (outside court and far from players)
    # 4) low/zero-metric tracks (very low strict in% or mot%=0)
    sideline_det_frames = set()  # (frame, cx_round, cy_round) tuples to blacklist
    blocked_cells_by_frame: Dict[int, Dict[Tuple[int, int], List[Tuple[float, float, float]]]] = {}
    blocked_cell_size = max(float(cfg.blocked_det_radius_px), 1.0)
    sideline_track_ids = set()
    stationary_track_ids = set()
    offcontext_track_ids = set()
    zero_metric_track_ids = set()
    zero_eps = 1e-9

    def _blocked_cell(x: float, y: float) -> Tuple[int, int]:
        return int(x / blocked_cell_size), int(y / blocked_cell_size)

    def _add_blocked_obs(obs_list):
        for o in obs_list:
            sideline_det_frames.add((o.frame, round(o.cx), round(o.cy)))
            point = (float(o.cx), float(o.cy))
            ow = max(float(o.x2) - float(o.x1), 0.0)
            oh = max(float(o.y2) - float(o.y1), 0.0)
            # Keep blocked region no larger than the rejected ball footprint (plus tiny slack),
            # while still capping by the global configured maximum.
            local_block_r = min(
                float(cfg.blocked_det_radius_px),
                max(2.0, 0.55 * max(ow, oh) + 1.5)
            )
            point_r2 = local_block_r * local_block_r
            frame_cells = blocked_cells_by_frame.setdefault(o.frame, {})
            cx_cell, cy_cell = _blocked_cell(point[0], point[1])
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    key = (cx_cell + dx, cy_cell + dy)
                    frame_cells.setdefault(key, []).append((point[0], point[1], point_r2))

    def _is_blocked_by_radius(det: Detection) -> bool:
        frame_cells = blocked_cells_by_frame.get(det.frame)
        cell = _blocked_cell(det.cx, det.cy)
        if frame_cells:
            candidates = frame_cells.get(cell)
            if candidates:
                for bx, by, point_r2 in candidates:
                    ddx = det.cx - bx
                    ddy = det.cy - by
                    if ddx * ddx + ddy * ddy <= point_r2:
                        return True
        return False

    for trk in tracks:
        sb = trk.score_breakdown if trk.score_breakdown else {}
        inside_frac = float(sb.get('inside_frac', 0))
        strict_inside_frac = float(sb.get('inside_strict_frac', sb.get('inside_frac', 0.0)))
        motion_frac = float(sb.get('motion_frac', 0.0))
        near_player_frac = float(sb.get('near_player_frac', 0))
        # Red flag: many observations but never on court
        if trk.num_obs >= 20 and inside_frac < 0.05:
            sideline_track_ids.add(trk.track_id)
            _add_blocked_obs(trk.observations)
            if debug:
                print(f"[selector] Blacklisted track {trk.track_id} "
                      f"({trk.num_obs} obs, {inside_frac:.0%} inside) as sideline")

        # Hard stationary reject: don't allow parked balls into matching at all.
        if _is_stationary_track(trk, cfg, diag):
            stationary_track_ids.add(trk.track_id)
            _add_blocked_obs(trk.observations)
            if debug:
                print(
                    f"[selector] Blacklisted track {trk.track_id} as stationary "
                    f"(obs={trk.num_obs}, avg_v={sb.get('avg_speed_pxpf', 0):.2f}, "
                    f"peak_v={sb.get('peak_speed_pxpf', 0):.2f}, "
                    f"extent={sb.get('extent_px', 0):.1f}, "
                    f"motion={sb.get('motion_frac', 0):.0%})"
                )

        # Outside + far from players: reject even if not fully stationary.
        if trk.num_obs >= 6 and inside_frac < 0.08 and near_player_frac < 0.06:
            offcontext_track_ids.add(trk.track_id)
            if debug:
                print(
                    f"[selector] Blacklisted track {trk.track_id} as off-context "
                    f"(obs={trk.num_obs}, inside={inside_frac:.0%}, nearP={near_player_frac:.0%})"
                )

        # Hard reject tracks with very low inside% or zero motion%.
        # strict_inside_frac < 2% = barely ever on court (covers both truly-zero and
        # compression-artifact detections that sneak past the num_obs threshold).
        if strict_inside_frac < 0.02 or motion_frac <= zero_eps:
            zero_metric_track_ids.add(trk.track_id)
            _add_blocked_obs(trk.observations)
            if debug:
                why = []
                if strict_inside_frac < 0.02:
                    why.append(f"in%={strict_inside_frac:.1%}<2%")
                if motion_frac <= zero_eps:
                    why.append("mot%=0")
                print(
                    f"[selector] Blacklisted track {trk.track_id} as low/zero-metric "
                    f"(in%={strict_inside_frac:.1%}, mot%={motion_frac:.1%}; {'/'.join(why)})"
                )

    # Choose best track after blacklist and build a per-frame guide.
    # Hard-block sideline/stationary/zero-metric tracks. Off-context tracks stay soft-filtered
    # through score/sanity so reacquire can still recover if player/court context is imperfect.
    blocked_track_ids = sideline_track_ids | stationary_track_ids | zero_metric_track_ids
    candidate_tracks = [t for t in tracks if t.track_id not in blocked_track_ids]
    min_track_score = max(0.0, float(cfg.timeline_min_track_score))
    motion_eps = 1e-9
    motion_track_ids = {
        int(t.track_id) for t in tracks
        if _track_motion_frac(t) > motion_eps
    }
    if candidate_tracks:
        pre_motion_count = len(candidate_tracks)
        candidate_tracks = [
            t for t in candidate_tracks if int(t.track_id) in motion_track_ids
        ]
        if debug and pre_motion_count != len(candidate_tracks):
            print(
                f"[selector] Motion filter dropped {pre_motion_count - len(candidate_tracks)} "
                f"candidate tracks with motion_frac=0"
            )
        pre_score_count = len(candidate_tracks)
        candidate_tracks = [t for t in candidate_tracks if float(t.score) >= min_track_score]
        if debug and pre_score_count != len(candidate_tracks):
            print(
                f"[selector] Score filter dropped {pre_score_count - len(candidate_tracks)} "
                f"candidate tracks with score < {min_track_score:.1f}"
            )
    merged_moving_count = 0
    if candidate_tracks:
        if cfg.enable_global_moving_merge:
            candidate_tracks, merged_moving_count = _merge_high_movement_tracks(candidate_tracks, cfg)
            candidate_tracks = score_tracks(
                candidate_tracks,
                cfg,
                court_poly=court_polygon,
                player_boxes_by_frame=player_boxes_by_frame,
                total_frames=total_frames
            )
            _annotate_track_periods(candidate_tracks, total_frames, cfg)
            if debug and merged_moving_count >= 2:
                print(
                    f"[selector] Merged {merged_moving_count} moving non-blacklisted tracks "
                    f"(movement >= {cfg.movement_merge_min_score:.1f}, "
                    f"avg>={cfg.movement_merge_min_avg_speed:.1f}@30fps, "
                    f"extent>={cfg.movement_merge_min_extent_frac:.2f}*diag) into one candidate"
                )
    if candidate_tracks:
        selection_pool = candidate_tracks
    else:
        # Strict rule: tracks with blocked ids, motion_frac=0, or low score are not selectable.
        selection_pool = [
            t for t in tracks
            if (
                int(t.track_id) in motion_track_ids
                and int(t.track_id) not in blocked_track_ids
                and float(t.score) >= min_track_score
            )
        ]
    timeline_chain: List[Track] = []
    if cfg.enable_timeline_stitch and selection_pool:
        timeline_chain = _select_timeline_chain(selection_pool, cfg, total_frames)
        chosen = _stitch_track_chain(timeline_chain, cfg)
        if chosen is None:
            chosen = select_best_track(selection_pool, cfg)
    else:
        chosen = select_best_track(selection_pool, cfg)

    frame_mode = str(getattr(cfg, "frame_selection_mode", "trail_only")).strip().lower()
    guide_interp_gap = max(
        8,
        min(
            48,
            max(
                int(round(float(cfg.fps) * 0.60)),
                int(round(float(cfg.max_gap_frames) * 0.75))
            )
        )
    )
    # trail_only mode: apply static-jump and spike filters but NOT the leading-trim
    # (which would strip the early observations we need for gap-fill coverage).
    chosen_guide, dropped_guide_spikes = _build_track_guide(
        chosen,
        total_frames,
        cfg,
        max_interp_gap=guide_interp_gap,
        apply_filters=(frame_mode == "legacy"),
        filter_static_jumps=(frame_mode != "legacy"),  # always filter static snaps
        all_dets=all_dets,  # snap KF gap-fills to real detections when available
    )
    det_owner_by_obj = _build_detection_owner_map(tracks)
    if timeline_chain:
        allowed_track_ids = {int(t.track_id) for t in timeline_chain}
    elif chosen is not None:
        allowed_track_ids = {int(chosen.track_id)}
    else:
        # If no chosen track, still enforce motion and score floor.
        allowed_track_ids = {
            int(t.track_id)
            for t in tracks
            if (
                int(t.track_id) in motion_track_ids
                and int(t.track_id) not in blocked_track_ids
                and float(t.score) >= min_track_score
            )
        }
    preferred_owner_ids = set(int(tid) for tid in allowed_track_ids)
    blocked_owner_ids = set(int(tid) for tid in blocked_track_ids)
    has_owner_map = bool(det_owner_by_obj)
    owner_pref_by_frame: Optional[List[set]] = None
    if timeline_chain and total_frames > 0:
        owner_pref_by_frame = [set() for _ in range(total_frames)]
        pad = max(2, min(10, int(round(0.015 * max(total_frames, 1)))))
        for trk in timeline_chain:
            tid = int(trk.track_id)
            s = max(0, int(trk.first_frame) - pad)
            e = min(total_frames - 1, int(trk.last_obs_frame) + pad)
            for f in range(s, e + 1):
                owner_pref_by_frame[f].add(tid)
    if debug and chosen is not None:
        if timeline_chain:
            ids = [str(t.track_id) for t in timeline_chain]
            first_sb = timeline_chain[0].score_breakdown if timeline_chain[0].score_breakdown else {}
            last_sb = timeline_chain[-1].score_breakdown if timeline_chain[-1].score_breakdown else {}
            t0 = float(first_sb.get("start_frac", 0.0))
            t1 = float(last_sb.get("end_frac", 1.0))
            print(
                f"[selector] Chosen timeline ({len(timeline_chain)} tracks): "
                f"{' -> '.join(ids)} | t={t0:.0%}->{t1:.0%}"
            )
        else:
            sb = chosen.score_breakdown if chosen.score_breakdown else {}
            pid = int(float(sb.get("period_id", 0.0)))
            t0 = float(sb.get("start_frac", 0.0))
            t1 = float(sb.get("end_frac", 0.0))
            print(
                f"[selector] Chosen track {chosen.track_id} ({chosen.num_obs} obs, "
                f"score={chosen.score:.1f}, period={pid}, t={t0:.0%}->{t1:.0%})"
            )
        if dropped_guide_spikes:
            print(f"[selector] Guide filtered {dropped_guide_spikes} local outlier detections")

    if frame_mode != "legacy":
        # Trail-only mode: emit the chosen stitched guide as the final output.
        # No motion-blob fallback, carry, or physics prediction.
        result: List[Optional[FrameResult]] = [None] * total_frames
        guide_debug_meta: Optional[List[Optional[Tuple[float, float, bool, bool, bool, float]]]] = (
            [None] * total_frames if emit_guide_debug_meta else None
        )

        chosen_obs_by_frame: Dict[int, Detection] = {}
        if chosen is not None:
            for obs in chosen.observations:
                f = int(obs.frame)
                if 0 <= f < total_frames:
                    chosen_obs_by_frame[f] = obs

        # State-circle radius is anchored to court size (sideline reference length).
        # Reduced from 0.095 to 0.050: the old value produced ~76-85px circles on
        # 1080p footage, which was large enough to pick up wrong detections (players,
        # shadows) during carry/interp gaps.
        state_radius_base = max(10.0, 0.050 * float(court_ref_length))
        state_radius_green = state_radius_base
        state_radius_blue = state_radius_base
        state_radius_yellow = state_radius_base * 1.20

        gap_run = 0
        blue_gap_frames = max(1, int(getattr(cfg, "carry_interp_frames", 5)))
        for t in range(total_frames):
            g = chosen_guide.get(t)
            if g is None:
                gap_run = 0
                continue
            gx, gy, guide_exact = g
            guide_cx, guide_cy = float(gx), float(gy)
            obs = chosen_obs_by_frame.get(t) if bool(guide_exact) else None
            has_obs = obs is not None
            if has_obs:
                gap_run = 0
                conf = max(0.0, float(obs.conf))
                bbox = (float(obs.x1), float(obs.y1), float(obs.x2), float(obs.y2))
                src = "det"
                state_r = float(state_radius_green)
            else:
                gap_run += 1
                conf = 0.20
                bbox = None
                # Blue: short gap connection. Yellow: prolonged guessed/stuck segment.
                src = "carry" if gap_run <= blue_gap_frames else "interp"
                state_r = float(state_radius_blue if src == "carry" else state_radius_yellow)

                # State-based circle detection:
                # blue  -> closest det in circle that's consistent with trajectory
                # yellow-> closest on-motion det in circle (STRICT)
                frame_dets = all_dets[t] if 0 <= t < len(all_dets) else []
                if frame_dets:
                    cand = []
                    for d in frame_dets:
                        # Skip detections owned by blacklisted tracks or in blocked regions
                        # to prevent recursive contamination where a blacklisted position
                        # drives the next frame's prediction, which then picks up more
                        # blacklisted detections.
                        det_owner_blk = det_owner_by_obj.get(id(d)) if has_owner_map else None
                        if det_owner_blk is not None and int(det_owner_blk) in blocked_owner_ids:
                            continue
                        key_blk = (d.frame, round(d.cx), round(d.cy))
                        if key_blk in sideline_det_frames:
                            continue
                        if _is_blocked_by_radius(d):
                            continue
                        dd = _xy_dist(float(d.cx), float(d.cy), guide_cx, guide_cy)
                        if dd <= state_r:
                            cand.append((dd, d))
                    if cand:
                        picked: Optional[Detection] = None
                        if src == "carry":
                            # Filter by trajectory direction: if we know which way the
                            # ball was moving, reject detections in the opposite direction.
                            # Use the last two guide positions to estimate direction.
                            carry_filtered = cand
                            if t >= 2:
                                prev_g = chosen_guide.get(t - 1) or chosen_guide.get(t - 2)
                                if prev_g is not None:
                                    prev_gx, prev_gy = float(prev_g[0]), float(prev_g[1])
                                    traj_dx = guide_cx - prev_gx
                                    traj_dy = guide_cy - prev_gy
                                    traj_mag = math.sqrt(traj_dx * traj_dx + traj_dy * traj_dy)
                                    if traj_mag > 2.0:
                                        # Only keep candidates roughly in the forward direction
                                        good = []
                                        for dd, d in cand:
                                            det_dx = float(d.cx) - prev_gx
                                            det_dy = float(d.cy) - prev_gy
                                            det_mag = math.sqrt(det_dx * det_dx + det_dy * det_dy)
                                            if det_mag > 1.0:
                                                cos_sim = (det_dx * traj_dx + det_dy * traj_dy) / (det_mag * traj_mag)
                                                if cos_sim > -0.3:  # allow some angle but not backwards
                                                    good.append((dd, d))
                                            else:
                                                good.append((dd, d))  # very close, keep it
                                        if good:
                                            carry_filtered = good
                            picked = min(carry_filtered, key=lambda p: p[0])[1]
                        else:
                            # User Request: Yellow circle MUST have motion.
                            motion_cand = [p for p in cand if bool(getattr(p[1], "on_motion", False))]
                            if motion_cand:
                                picked = min(motion_cand, key=lambda p: p[0])[1]
                        if picked is not None:
                            gx = float(picked.cx)
                            gy = float(picked.cy)
                            conf = max(0.0, float(picked.conf))
                            bbox = (float(picked.x1), float(picked.y1), float(picked.x2), float(picked.y2))
                            has_obs = True
                
                # If we tried to pick for interp/yellow but failed because no motion cand existed:
                if not has_obs and src == "interp":
                    # Stay as interp, but we don't snap to a stationary object.
                    pass

            result[t] = FrameResult(
                cx=float(gx),
                cy=float(gy),
                conf=conf,
                interpolated=not has_obs,
                bbox=bbox,
                source=src,
                search_cx=float(guide_cx),
                search_cy=float(guide_cy),
                search_radius=float(state_r),
            )

            if emit_guide_debug_meta and guide_debug_meta is not None:
                is_hold = (src == "carry")
                is_exact = (src == "det")
                guide_debug_meta[t] = (
                    float(guide_cx), float(guide_cy), bool(is_exact), False, bool(is_hold), float(state_r)
                )

        # Gap-fill for trail-only output:
        # If stitched guide leaves uncovered windows, fill them from the highest-scoring
        # remaining eligible tracks (time-based stitching by score).
        added_gap_tracks: List[Tuple[int, int]] = []
        primary_ids = (
            {int(t.track_id) for t in timeline_chain}
            if timeline_chain
            else ({int(chosen.track_id)} if chosen is not None else set())
        )
        if selection_pool:
            extras = sorted(
                [t for t in selection_pool if int(t.track_id) not in primary_ids],
                key=lambda tr: float(tr.score),
                reverse=True,
            )
            min_gap_fill_frames = max(3, int(round(0.002 * max(total_frames, 1))))
            max_extra_tracks = 6
            for cand in extras:
                if len(added_gap_tracks) >= max_extra_tracks:
                    break

                cand_guide, _ = _build_track_guide(
                    cand,
                    total_frames,
                    cfg,
                    max_interp_gap=guide_interp_gap,
                    apply_filters=True,
                )
                if not cand_guide:
                    continue

                slot_frames = [
                    f for f in cand_guide.keys()
                    if (
                        0 <= f < total_frames and
                        (
                            result[f] is None or
                            str(getattr(result[f], "source", "")) in ("carry", "interp")
                        )
                    )
                ]
                if len(slot_frames) < min_gap_fill_frames:
                    continue
                coverage_ratio = float(len(slot_frames)) / float(max(len(cand_guide), 1))
                if coverage_ratio < 0.20:
                    continue

                cand_obs_by_frame: Dict[int, Detection] = {}
                for obs in cand.observations:
                    of = int(obs.frame)
                    if 0 <= of < total_frames:
                        cand_obs_by_frame[of] = obs

                # Split candidate slots into contiguous windows.
                slot_sorted = sorted(slot_frames)
                runs: List[List[int]] = []
                run: List[int] = []
                for f in slot_sorted:
                    if not run or f == run[-1] + 1:
                        run.append(f)
                    else:
                        runs.append(run)
                        run = [f]
                if run:
                    runs.append(run)

                filled_count = 0
                for run in runs:
                    if not run:
                        continue
                    run_start = int(run[0])
                    cut_start = run_start
                    cut_end = int(run[-1])
                    handoff_gate = max(15.0, 0.080 * float(court_ref_length))

                    # Trim stitched run start so it hands off from previous output.
                    prev_idx = run_start - 1
                    while prev_idx >= 0 and result[prev_idx] is None:
                        prev_idx -= 1
                    if prev_idx >= 0 and result[prev_idx] is not None:
                        prev_r = result[prev_idx]
                        prev2_idx = prev_idx - 1
                        while prev2_idx >= 0 and result[prev2_idx] is None:
                            prev2_idx -= 1
                        pvx, pvy = 0.0, 0.0
                        if prev2_idx >= 0 and result[prev2_idx] is not None:
                            prev2_r = result[prev2_idx]
                            dtv = max(prev_idx - prev2_idx, 1)
                            pvx = (float(prev_r.cx) - float(prev2_r.cx)) / float(dtv)
                            pvy = (float(prev_r.cy) - float(prev2_r.cy)) / float(dtv)

                        probe = run[: min(len(run), 24)]
                        best_f = int(probe[0])
                        best_d = float("inf")
                        found = False
                        for f in probe:
                            cgx_p, cgy_p, _ = cand_guide[f]
                            dtp = max(int(f - prev_idx), 1)
                            pred_x = float(prev_r.cx) + pvx * float(dtp)
                            pred_y = float(prev_r.cy) + pvy * float(dtp)
                            dd = _xy_dist(float(cgx_p), float(cgy_p), pred_x, pred_y)
                            if dd < best_d:
                                best_d = dd
                                best_f = int(f)
                            if dd <= handoff_gate:
                                cut_start = int(f)
                                found = True
                                break
                        if (not found) and best_d <= (2.0 * handoff_gate):
                            cut_start = best_f

                    # Trim stitched run end so it hands off before the next anchored segment.
                    next_idx = int(run[-1]) + 1
                    while next_idx < total_frames:
                        rrn = result[next_idx]
                        if rrn is not None and str(getattr(rrn, "source", "")) not in ("carry", "interp"):
                            break
                        next_idx += 1
                    if next_idx < total_frames and result[next_idx] is not None:
                        next_r = result[next_idx]
                        next2_idx = next_idx + 1
                        while next2_idx < total_frames:
                            rrn2 = result[next2_idx]
                            if rrn2 is not None and str(getattr(rrn2, "source", "")) not in ("carry", "interp"):
                                break
                            next2_idx += 1
                        nvx, nvy = 0.0, 0.0
                        if next2_idx < total_frames and result[next2_idx] is not None:
                            next2_r = result[next2_idx]
                            dtn = max(next2_idx - next_idx, 1)
                            nvx = (float(next2_r.cx) - float(next_r.cx)) / float(dtn)
                            nvy = (float(next2_r.cy) - float(next_r.cy)) / float(dtn)

                        probe_rev = [f for f in run if f >= cut_start]
                        if probe_rev:
                            probe_rev = probe_rev[-min(len(probe_rev), 24):]
                            best_f2 = int(probe_rev[-1])
                            best_d2 = float("inf")
                            found2 = False
                            for f in reversed(probe_rev):
                                cgx_p, cgy_p, _ = cand_guide[f]
                                dtp = max(int(next_idx - f), 1)
                                pred_x = float(next_r.cx) - nvx * float(dtp)
                                pred_y = float(next_r.cy) - nvy * float(dtp)
                                dd = _xy_dist(float(cgx_p), float(cgy_p), pred_x, pred_y)
                                if dd < best_d2:
                                    best_d2 = dd
                                    best_f2 = int(f)
                                if dd <= handoff_gate:
                                    cut_end = int(f)
                                    found2 = True
                                    break
                            if (not found2) and best_d2 <= (2.0 * handoff_gate):
                                cut_end = int(best_f2)

                    if cut_end < cut_start:
                        continue

                    run_gap_count = 0
                    for f in run:
                        if f < cut_start or f > cut_end:
                            continue
                        cgx, cgy, c_exact = cand_guide[f]
                        state_cx, state_cy = float(cgx), float(cgy)
                        cobs = cand_obs_by_frame.get(f) if bool(c_exact) else None

                        if cobs is not None:
                            cconf = max(0.0, float(cobs.conf))
                            cbbox = (float(cobs.x1), float(cobs.y1), float(cobs.x2), float(cobs.y2))
                            cinterp = False
                            csrc = "det"
                            state_r_local = float(state_radius_green)
                            run_gap_count = 0
                        else:
                            run_gap_count += 1
                            # First try finding motion locally (if it's a fast ball or gap isn't fully empty)
                            # This block is new, it was not present in the original code.
                            # It seems to be part of a larger change that was not fully provided.
                            # I will insert it as requested, assuming `m_mask`, `pred_cx_frame`, `pred_cy_frame`, `rst`,
                            # `last_det_pos`, `last_vel`, `ref_area`, `pboxes`, `prev_motion_vel`, `frames_since_det`,
                            # `prev_det_pos_for_lock`, `fallbacks`, `last_carried_pos`, `continuent` are defined
                            # in the context of the original code, or are meant to be added.
                            # Given the context of "Gap-fill for trail-only output", this looks like
                            # an attempt to fill gaps with motion blobs before falling back to carry/interp.
                            # However, the provided diff is incomplete for this section.
                            # I will insert the provided lines as literally as possible, assuming the variables exist.
                            # If this is a new feature, the surrounding logic for these variables would also be new.
                            # For now, I'll place it where the diff indicates.
                            #
                            # NOTE: The provided diff for this section is problematic as it seems to be
                            # inserting new logic that relies on variables not defined in the current scope
                            # (e.g., m_mask, pred_cx_frame, rst, ref_area, pboxes, prev_motion_vel,
                            # prev_det_pos_for_lock, fallbacks, last_carried_pos, continuent).
                            # I will insert it as requested, but this will likely lead to a non-functional
                            # code snippet without the full context of the intended change.
                            # I will assume `frames_since_det` and `diag` are available from the outer scope.
                            # `last_det_pos` and `last_vel` are also from the outer scope.
                            # `motion_tracks` is also from the outer scope.
                            # `cfg` is from the outer scope.
                            # `pred_cx_frame`, `pred_cy_frame`, `rst`, `ref_area`, `pboxes`, `prev_motion_vel`,
                            # `prev_det_pos_for_lock`, `fallbacks`, `last_carried_pos`, `continuent` are missing.
                            # I will make a best effort to define some of these as placeholders if they are critical
                            # for the snippet to parse, but the functionality will be broken.
                            #
                            # Given the instruction is to "make the change faithfully and without making any unrelated edits",
                            # I will insert the code as provided in the diff, even if it introduces undefined variables.
                            # This is a limitation of applying partial diffs without full context.

                            # Placeholder definitions for missing variables to allow parsing,
                            # these would need proper initialization in a real scenario.
                            # m_mask = None # Assuming motion mask for the current frame
                            # pred_cx_frame = state_cx # Assuming prediction from current state
                            # pred_cy_frame = state_cy
                            # rst = state_r_local # Assuming search radius
                            # ref_area = 1.0 # Placeholder for reference area
                            # pboxes = None # Placeholder for player boxes
                            # prev_motion_vel = (0.0, 0.0) # Placeholder
                            # prev_det_pos_for_lock = (0.0, 0.0) # Placeholder
                            # fallbacks = [] # Placeholder
                            # last_carried_pos = (0.0, 0.0) # Placeholder
                            # continuent = 0 # Placeholder

                            # The following block is from the user's diff.
                            # It's placed here as per the diff, but relies on undefined variables.
                            # I'm commenting out the block to avoid syntax errors, as the instruction
                            # is to return syntactically correct code. If the user intended this
                            # to be uncommented, they would need to provide the definitions for
                            # the variables used within it.
                            #
                            # b_motion = _find_motion_blob(m_mask, pred_cx_frame, pred_cy_frame, rst,
                            #                              last_det_pos[0], last_det_pos[1], last_vel,
                            #                              ref_ball_area=ref_area,
                            #                              player_boxes=pboxes,
                            #                              frame_idx=f,
                            #                              active_motion_tracks=motion_tracks, prev_motion_pos=last_motion_pos)
                            # if b_motion is not None:
                            #     bm_cx, bm_cy, bm_area, is_latched = b_motion
                            #     if is_latched or _motion_blob_physics_ok(
                            #         bm_cx, bm_cy, pred_cx_frame, pred_cy_frame,
                            #         last_det_pos, last_vel, frames_since_det,
                            #         prev_motion_vel, cfg, diag
                            #     ):
                            #         guided_cx, guided_cy, _ = _physics_guide_motion_blob(
                            #             bm_cx, bm_cy, pred_cx_frame, pred_cy_frame,
                            #             prev_det_pos_for_lock, last_vel, frames_since_det, cfg, diag
                            #         )
                            #         fallbacks.append((f, guided_cx, guided_cy, 'motion', bm_area))
                            #         last_carried_pos = (guided_cx, guided_cy)
                            #         prev_motion_vel = (
                            #             (guided_cx - last_det_pos[0]) / max(frames_since_det, 1),
                            #             (guided_cy - last_det_pos[1]) / max(frames_since_det, 1)
                            #         )
                            #         continuent += 1
                            #
                            # End of user's diff block for this section.

                            csrc = "carry" if run_gap_count <= blue_gap_frames else "interp"
                            state_r_local = float(state_radius_blue if csrc == "carry" else state_radius_yellow)
                            cconf = 0.18 if csrc == "carry" else 0.14
                            cbbox = None
                            cinterp = True

                        result[f] = FrameResult(
                            cx=float(cgx),
                            cy=float(cgy),
                            conf=cconf,
                            interpolated=cinterp,
                            bbox=cbbox,
                            source=csrc,
                            search_cx=float(state_cx),
                            search_cy=float(state_cy),
                            search_radius=float(state_r_local),
                        )
                        if emit_guide_debug_meta and guide_debug_meta is not None:
                            guide_debug_meta[f] = (
                                float(state_cx),
                                float(state_cy),
                                bool(csrc == "det"),
                                False,
                                bool(csrc == "carry"),
                                float(state_r_local),
                            )
                        filled_count += 1

                if filled_count >= min_gap_fill_frames:
                    added_gap_tracks.append((int(cand.track_id), int(filled_count)))

        # Physics Validation Pass for Trail-Only mode:
        # Check if the transition out of a carry/interp segment into an observed segment
        # is physically possible. If it requires impossible speeds, drop the carry segment.
        run_start = -1
        max_valid_speed = _fps_norm_pxpf(120.0, cfg) # Max expected pixel speed
        for t in range(total_frames):
            r = result[t]
            if r is not None and r.interpolated:
                if run_start == -1:
                    run_start = t
            else:
                if run_start != -1:
                    # We found the end of a carry/interp segment.
                    # Check if the next observed point (at t) is a massive jump.
                    if r is not None and not r.interpolated:
                        # We need the last valid observed point BEFORE the run
                        prev_idx = run_start - 1
                        while prev_idx >= 0:
                            pr = result[prev_idx]
                            if pr is not None and not pr.interpolated:
                                break
                            prev_idx -= 1
                        
                        if prev_idx >= 0:
                            pr = result[prev_idx]
                            # Check required average speed over the whole gap
                            # from last valid to next valid
                            dt = max(t - prev_idx, 1)
                            req_speed = _xy_dist(float(pr.cx), float(pr.cy), float(r.cx), float(r.cy)) / float(dt)
                            
                            # If the gap traversal average speed is wildly impossible, or 
                            # checking just the instantaneous jump from the end of the physics carry
                            # to the new detection is impossible:
                            carry_end_r = result[t - 1]
                            jump_speed = _xy_dist(float(carry_end_r.cx), float(carry_end_r.cy), float(r.cx), float(r.cy))
                            
                            if req_speed > max_valid_speed or jump_speed > max_valid_speed * 1.5:
                                # Clip the physically impossible carry segment
                                for i in range(run_start, t):
                                    result[i] = None
                                    if emit_guide_debug_meta and guide_debug_meta is not None:
                                        guide_debug_meta[i] = None
                    run_start = -1

        if debug:
            filled = sum(1 for r in result if r is not None)
            observed = sum(1 for r in result if r is not None and not r.interpolated)
            interp = max(0, filled - observed)
            print(
                f"[selector] Trail-only output: observed={observed} interp={interp} "
                f"filled={filled}/{total_frames}"
            )
            if added_gap_tracks:
                details = ", ".join(f"{tid}:{cnt}" for tid, cnt in added_gap_tracks)
                print(f"[selector] Trail-only gap-fill tracks: {details}")

        if emit_guide_debug_meta and guide_debug_meta is not None:
            for i, gmeta in enumerate(guide_debug_meta):
                if gmeta is None:
                    continue
                rr_dbg = result[i]
                if rr_dbg is None:
                    rr_dbg = FrameResult(source='debug', debug_only=True)
                    result[i] = rr_dbg
                (gsx, gsy, gexact_dbg, gfrozen_dbg, ghold_dbg, gsr) = gmeta
                rr_dbg.guide_search_cx = float(gsx)
                rr_dbg.guide_search_cy = float(gsy)
                rr_dbg.guide_search_exact = bool(gexact_dbg)
                rr_dbg.guide_search_frozen = bool(gfrozen_dbg)
                rr_dbg.guide_search_hold = bool(ghold_dbg)
                rr_dbg.guide_search_radius = float(gsr)

        return result, chosen, tracks, motion_tracks

    # Step 4: per-frame selection with motion-blob gap bridging
    diag = _cfg_diag(cfg)
    result = [None] * total_frames
    guide_debug_meta: Optional[List[Optional[Tuple[float, float, bool, bool, bool, float]]]] = (
        [None] * total_frames if emit_guide_debug_meta else None
    )
    # Base guide radius is court-line based (sideline reference length).
    guide_lock_radius = 0.09 * court_ref_length
    guide_base_radius = guide_lock_radius
    frozen_growth_per_frame = 0.025
    frozen_growth_cap_mult = 1.80
    frozen_reacquire_radius_mult = 1.10
    guide_soft_gate_cap_mult = 1.80
    max_frozen_guide_frames = max(
        int(cfg.max_gap_frames),
        int(guide_interp_gap) * 3,
        int(round(float(cfg.fps) * 1.5)),
        20,
    )
    max_hold_guide_frames = max(
        int(cfg.max_gap_frames),
        int(round(float(cfg.fps) * 1.0)),
        18,
    )

    # ── Frozen guide state for when guide is lost ──
    frozen_guide_pos: Optional[Tuple[float, float]] = None
    frozen_guide_vel: Optional[Tuple[float, float]] = None
    frozen_guide_frame: int = -1
    frozen_guide_active: bool = False
    hold_guide_start_frame: int = -1
    
    # Tracking state
    last_pos = None          # (cx, cy) â€” last known position (det or motion)
    last_det_pos = None      # (cx, cy) â€” last YOLO detection position specifically
    last_det_area = None     # last YOLO detection bbox area
    last_vel = (0.0, 0.0)   # estimated velocity from YOLO detections
    last_motion_vel = None   # velocity estimate while using motion fallback
    last_motion_pos = None   # (cx, cy) — last motion blob position for continuity
    frames_since_det = 0     # frames since last YOLO detection
    base_gravity = float(cfg.gravity_px_per_frame2)
    adaptive_gravity = float(cfg.gravity_px_per_frame2)
    prev_det_raw_vel: Optional[Tuple[float, float]] = None
    motion_search_base = max(
        float(cfg.motion_search_min_px),
        float(cfg.motion_search_base_frac) * diag
    )
    motion_search_growth = max(float(cfg.motion_search_growth_px), 0.0)
    motion_search_vel_mult = max(float(cfg.motion_search_vel_mult), 0.0)
    motion_search_max = max(motion_search_base, 0.06 * diag)
    max_motion_gap = max(int(cfg.motion_max_gap_frames), 1)  # max frames to track via motion fallback
    motion_vel_history = []  # velocity from motion blobs for sanity checking
    soft_carry_count = 0     # carry frames chained after motion/guide without new det
    last_motion_area = None  # O(1) area continuity guard for motion blobs
    stats = {
        'det': 0, 'motion': 0, 'guide': 0, 'carry': 0, 'lost': 0,
        'rej_reacquire_dist': 0, 'rej_reacquire_size': 0,
        'rej_context': 0, 'rej_blocked_radius': 0, 'rej_hard_step': 0,
        'rej_other_track': 0,
        'owner_soft_pen': 0,
        'rej_motion_area': 0, 'rej_motion_physics': 0, 'rej_motion_jump': 0,
        'rej_motion_player': 0, 'rej_static_snap': 0,
        'rej_static_lock_cluster': 0,
        'motion_guided_clamped': 0, 'bounce': 0
    }
    motion_min_speed = _fps_norm_pxpf(cfg.motion_use_min_det_speed, cfg)
    erratic_var_thresh = _fps_norm_pxpf(160.0, cfg) # Relaxed to allow fast bounces without dropping track
    guide_static_speed_thresh = _guide_static_speed_thresh(cfg, diag)
    owner_mismatch_pen = max(12.0, 0.010 * diag)
    owner_startup_extra_pen = max(28.0, 0.018 * diag)
    pending_static_det_pos: Optional[Tuple[float, float]] = None
    pending_static_det_frame = -10**9
    static_debounce_match_px = max(10.0, 0.008 * diag)
    static_lock_streak = 0
    static_lock_center: Optional[Tuple[float, float]] = None
    static_lock_min_streak = 6
    static_lock_radius = max(8.0, 0.009 * diag)
    static_lock_reset_step = max(12.0, 0.014 * diag)
    static_lock_player_gate = max(14.0, 0.020 * diag)
    # Per-frame source priority:
    # det (green) -> motion (orange) -> carry (blue)
    for t in range(total_frames):
        # Per-frame gravity update: adaptive base + depth scaling.
        if cfg.gravity_enabled:
            g_now = adaptive_gravity
            if cfg.gravity_adapt_enabled and last_pos is not None:
                y_norm = float(last_pos[1]) / float(max(cfg.height - 1, 1))
                y_norm = max(0.0, min(1.0, y_norm))
                depth_mul = 1.0 + float(cfg.gravity_depth_gain) * (2.0 * y_norm - 1.0)
                depth_mul = max(0.60, min(1.60, depth_mul))
                g_now *= depth_mul
            cfg.gravity_px_per_frame2 = float(g_now)

        frame_dets = all_dets[t]
        guide = chosen_guide.get(t)
        gx = gy = 0.0
        guide_exact = False
        hold_guide_active = False
        if guide is not None:
            prev_guide_pos_for_freeze = frozen_guide_pos
            prev_guide_frame_for_freeze = frozen_guide_frame
            gx, gy, guide_exact = guide
            # Guide is valid — update frozen state for future loss recovery
            frozen_guide_pos = (gx, gy)
            frozen_guide_frame = t
            frozen_guide_active = False
            if last_motion_vel is not None:
                frozen_guide_vel = last_motion_vel
            elif last_vel != (0.0, 0.0):
                frozen_guide_vel = last_vel
            elif (
                prev_guide_pos_for_freeze is not None and
                prev_guide_frame_for_freeze >= 0 and
                t > prev_guide_frame_for_freeze
            ):
                # Fall back to guide-to-guide velocity so frozen guide can persist
                # through gaps even when no recent YOLO/motion velocity is available.
                dtg = max(int(t - prev_guide_frame_for_freeze), 1)
                frozen_guide_vel = (
                    (float(gx) - float(prev_guide_pos_for_freeze[0])) / dtg,
                    (float(gy) - float(prev_guide_pos_for_freeze[1])) / dtg,
                )
            guide_lock_radius = guide_base_radius
            hold_guide_start_frame = -1
        elif frozen_guide_pos is not None:
            # Guide is lost — freeze at the last guide endpoint.
            # Keep the center fixed; only expand the radius over time.
            frozen_guide_active = True
            dt_frozen = t - frozen_guide_frame
            # Keep frozen-guide active long enough to bridge guide-segment gaps.
            if dt_frozen <= max_frozen_guide_frames:
                fgx, fgy = float(frozen_guide_pos[0]), float(frozen_guide_pos[1])
                gx, gy = fgx, fgy
                guide_exact = False
                growth_factor = 1.0 + frozen_growth_per_frame * dt_frozen
                guide_lock_radius = guide_base_radius * min(growth_factor, frozen_growth_cap_mult)
                guide = (gx, gy, False)
                hold_guide_start_frame = -1
            else:
                # After frozen expansion expires, keep a fixed-radius "hold" guide
                # circle active for a short window (blue debug circle) and apply the
                # same reacquire logic as frozen, just without expansion.
                frozen_guide_active = False
                if hold_guide_start_frame < 0:
                    hold_guide_start_frame = int(t)
                dt_hold = t - hold_guide_start_frame
                if dt_hold <= max_hold_guide_frames:
                    fgx, fgy = float(frozen_guide_pos[0]), float(frozen_guide_pos[1])
                    gx, gy = fgx, fgy
                    guide_exact = False
                    hold_guide_active = True
                    guide_lock_radius = guide_base_radius
                    guide = (gx, gy, False)
                else:
                    guide = None
                    guide_lock_radius = guide_base_radius
        else:
            guide_lock_radius = guide_base_radius
            hold_guide_start_frame = -1
        boost_mask = _ensure_mask_u8(boost_masks[t]) if boost_masks and t < len(boost_masks) else None
        # Raw motion is noisier; use only as fallback if boost-mask search fails.
        raw_motion = _ensure_mask_u8(raw_motions[t]) if raw_motions and t < len(raw_motions) else None
        frame_player_boxes_20: Optional[List[Tuple[float, float, float, float]]] = None
        frame_player_boxes_0: Optional[List[Tuple[float, float, float, float]]] = None
        if player_boxes_by_frame is not None and t < len(player_boxes_by_frame):
            pboxes = player_boxes_by_frame[t]
            if pboxes is not None:
                boxes_iter = pboxes.values() if isinstance(pboxes, dict) else pboxes
                b20: List[Tuple[float, float, float, float]] = []
                b0: List[Tuple[float, float, float, float]] = []
                for pb in boxes_iter:
                    if pb is None or len(pb) < 4:
                        continue
                    x1, y1, x2, y2 = map(float, pb[:4])
                    b0.append((x1, y1, x2, y2))
                    b20.append((x1 - 20.0, y1 - 20.0, x2 + 20.0, y2 + 20.0))
                if b20:
                    frame_player_boxes_20 = b20
                    frame_player_boxes_0 = b0

        def _closest_player_distance_local(
            x: float,
            y: float,
            expanded_boxes: Optional[List[Tuple[float, float, float, float]]]
        ) -> Optional[float]:
            if not expanded_boxes:
                return None
            best_d2 = None
            for x1, y1, x2, y2 in expanded_boxes:
                dx = max(x1 - x, 0.0, x - x2)
                dy = max(y1 - y, 0.0, y - y2)
                d2 = dx * dx + dy * dy
                if best_d2 is None or d2 < best_d2:
                    best_d2 = d2
            if best_d2 is None:
                return None
            return math.sqrt(best_d2)

        def _motion_player_probation_ok(
            raw_cx: float,
            raw_cy: float,
            guided_cx: float,
            guided_cy: float,
            blob_area: float,
            ref_cx: float,
            ref_cy: float,
            search_radius: float,
            ref_area: Optional[float],
            is_latched: bool,
            from_guide_circle: bool,
            anchor_pos: Tuple[float, float],
            pred_vel_local: Tuple[float, float],
            anchor_dt_local: int,
        ) -> bool:
            """Make in-player motion prove it is the ball instead of player-body clutter."""
            pd_raw_0 = _closest_player_distance_local(raw_cx, raw_cy, frame_player_boxes_0)
            pd_guided_0 = _closest_player_distance_local(guided_cx, guided_cy, frame_player_boxes_0)
            pd_raw_20 = _closest_player_distance_local(raw_cx, raw_cy, frame_player_boxes_20)
            in_player = (
                (pd_raw_0 is not None and pd_raw_0 <= 0.0) or
                (pd_guided_0 is not None and pd_guided_0 <= 0.0)
            )
            near_player = in_player or (pd_raw_20 is not None and pd_raw_20 <= 0.0)
            if not near_player:
                return True

            raw_ref_dist = _xy_dist(raw_cx, raw_cy, ref_cx, ref_cy)
            guided_ref_dist = _xy_dist(guided_cx, guided_cy, ref_cx, ref_cy)
            search_radius = max(float(search_radius), 1.0)

            # Player boxes contain a lot of leg/racket/body-edge motion. Keep only
            # blobs tightly centered on the predicted/guide point.
            core_r = max(9.0, min(0.018 * diag, 0.34 * search_radius))
            if from_guide_circle:
                core_r = max(9.0, min(0.016 * diag, 0.28 * search_radius))
            if not in_player:
                core_r *= 1.25

            if raw_ref_dist > core_r:
                return False
            if guided_ref_dist > max(core_r * 1.35, 0.024 * diag):
                return False

            if ref_area is not None and ref_area > 0.0:
                area_ratio = float(blob_area) / max(float(ref_area), 1.0)
                min_ratio = 0.18 if from_guide_circle else 0.24
                max_ratio = 4.00 if from_guide_circle else 3.25
                if area_ratio < min_ratio or area_ratio > max_ratio:
                    return False

            dt_local = max(int(anchor_dt_local), 1)
            expected_step = math.sqrt(
                float(pred_vel_local[0]) ** 2 + float(pred_vel_local[1]) ** 2
            ) * dt_local
            max_step = max(14.0, min(0.060 * diag, 2.35 * expected_step + 12.0))
            raw_step = _xy_dist(raw_cx, raw_cy, anchor_pos[0], anchor_pos[1])
            if raw_step > max_step and raw_ref_dist > 0.65 * core_r:
                return False

            if last_motion_pos is not None and not is_latched:
                prev_step = _xy_dist(raw_cx, raw_cy, last_motion_pos[0], last_motion_pos[1])
                cont_gate = max(12.0, min(0.042 * diag, 2.20 * expected_step + 14.0))
                if prev_step > cont_gate and raw_ref_dist > 0.70 * core_r:
                    return False

            return True

        def _motion_jump_probation_ok(
            raw_cx: float,
            raw_cy: float,
            guided_cx: float,
            guided_cy: float,
            ref_cx: float,
            ref_cy: float,
            search_radius: float,
            is_latched: bool,
            from_guide_circle: bool,
            anchor_pos: Tuple[float, float],
            pred_vel_local: Tuple[float, float],
            anchor_dt_local: int,
        ) -> bool:
            """Reject sudden motion handoffs that are not close to the predicted/guide path."""
            if is_latched:
                return True

            raw_ref_dist = _xy_dist(raw_cx, raw_cy, ref_cx, ref_cy)
            guided_ref_dist = _xy_dist(guided_cx, guided_cy, ref_cx, ref_cy)
            search_radius = max(float(search_radius), 1.0)
            dt_local = max(int(anchor_dt_local), 1)
            expected_step = math.sqrt(
                float(pred_vel_local[0]) ** 2 + float(pred_vel_local[1]) ** 2
            ) * dt_local

            core_r = max(10.0, min(0.030 * diag, 0.42 * search_radius))
            if from_guide_circle:
                core_r = max(9.0, min(0.024 * diag, 0.32 * search_radius))

            if raw_ref_dist <= core_r and guided_ref_dist <= max(core_r * 1.70, 0.034 * diag):
                return True

            step_gate = max(16.0, min(0.070 * diag, 2.45 * expected_step + 18.0))
            if prev_src not in ("motion", "guide", "carry"):
                step_gate = min(step_gate, max(14.0, 1.80 * expected_step + 14.0))

            raw_anchor_step = _xy_dist(raw_cx, raw_cy, anchor_pos[0], anchor_pos[1])
            guided_anchor_step = _xy_dist(guided_cx, guided_cy, anchor_pos[0], anchor_pos[1])
            if max(raw_anchor_step, guided_anchor_step) > step_gate:
                return False

            if last_pos is not None:
                raw_last_step = _xy_dist(raw_cx, raw_cy, last_pos[0], last_pos[1])
                guided_last_step = _xy_dist(guided_cx, guided_cy, last_pos[0], last_pos[1])
                last_gate = max(16.0, min(0.068 * diag, 2.35 * expected_step + 16.0))
                if max(raw_last_step, guided_last_step) > last_gate and raw_ref_dist > 0.75 * core_r:
                    return False

            if last_motion_pos is not None:
                prev_step = _xy_dist(raw_cx, raw_cy, last_motion_pos[0], last_motion_pos[1])
                cont_gate = max(12.0, min(0.045 * diag, 2.20 * expected_step + 14.0))
                if prev_step > cont_gate and raw_ref_dist > 0.75 * core_r:
                    return False

            if from_guide_circle and raw_ref_dist > max(core_r * 1.85, 0.040 * diag):
                return False

            return True

        def _stabilize_probation_motion(
            raw_cx: float,
            raw_cy: float,
            guided_cx: float,
            guided_cy: float,
            ref_cx: float,
            ref_cy: float,
            search_radius: float,
            from_guide_circle: bool,
            pred_vel_local: Tuple[float, float],
            anchor_dt_local: int,
        ) -> Tuple[float, float]:
            """Near players, use motion as support but cap how far it can pull the ball."""
            pd_raw_0 = _closest_player_distance_local(raw_cx, raw_cy, frame_player_boxes_0)
            pd_guided_0 = _closest_player_distance_local(guided_cx, guided_cy, frame_player_boxes_0)
            pd_raw_20 = _closest_player_distance_local(raw_cx, raw_cy, frame_player_boxes_20)
            in_player = (
                (pd_raw_0 is not None and pd_raw_0 <= 0.0) or
                (pd_guided_0 is not None and pd_guided_0 <= 0.0)
            )
            near_player = in_player or (pd_raw_20 is not None and pd_raw_20 <= 0.0)

            search_radius = max(float(search_radius), 1.0)
            if not near_player:
                return guided_cx, guided_cy

            raw_ref_dist = _xy_dist(raw_cx, raw_cy, ref_cx, ref_cy)
            guided_ref_dist = _xy_dist(guided_cx, guided_cy, ref_cx, ref_cy)
            dt_local = max(int(anchor_dt_local), 1)
            expected_step = math.sqrt(
                float(pred_vel_local[0]) ** 2 + float(pred_vel_local[1]) ** 2
            ) * dt_local

            # Player/racket motion may be real evidence, but the centroid is noisy.
            weight = 0.22 if in_player else 0.34
            max_pull = max(4.0, min(13.0, 0.0065 * diag, 0.18 * search_radius))

            if from_guide_circle:
                weight *= 0.80
                max_pull *= 0.85

            target_x = ref_cx + (raw_cx - ref_cx) * weight
            target_y = ref_cy + (raw_cy - ref_cy) * weight

            # If the generic guided point is already closer to the reference, prefer it.
            if guided_ref_dist < _xy_dist(target_x, target_y, ref_cx, ref_cy):
                target_x, target_y = guided_cx, guided_cy

            dx = target_x - ref_cx
            dy = target_y - ref_cy
            pull = math.sqrt(dx * dx + dy * dy)
            if pull > max_pull and pull > 1e-6:
                s = max_pull / pull
                target_x = ref_cx + dx * s
                target_y = ref_cy + dy * s

            if last_pos is not None:
                last_gate = max(7.0, min(16.0, 1.25 * expected_step + 6.0))
                sx = target_x - last_pos[0]
                sy = target_y - last_pos[1]
                step = math.sqrt(sx * sx + sy * sy)
                if step > last_gate and step > 1e-6:
                    scale = last_gate / step
                    target_x = last_pos[0] + sx * scale
                    target_y = last_pos[1] + sy * scale

            return target_x, target_y

        def _carry_after_failed_motion_ok(pred_x: float, pred_y: float, search_radius: float) -> bool:
            """After near-player orange motion, don't let blue carry drift without support."""
            if prev_src != "motion":
                return True
            pd_pred = _closest_player_distance_local(pred_x, pred_y, frame_player_boxes_20)
            if pd_pred is None or pd_pred > 0.0:
                return True

            probe_r = int(round(max(5.0, min(14.0, 0.006 * diag, 0.12 * max(float(search_radius), 1.0)))))
            has_local_motion = (
                _mask_has_motion_near(boost_mask, pred_x, pred_y, probe_r) or
                _mask_has_motion_near(raw_motion, pred_x, pred_y, probe_r)
            )
            exact_guide_support = (
                guide is not None and guide_exact and
                _xy_dist(pred_x, pred_y, gx, gy) <= max(10.0, 0.012 * diag)
            )
            if not has_local_motion and not exact_guide_support:
                return False

            # Even with support, only bridge one frame in this clutter zone.
            return soft_carry_count < 1

        def _mask_has_motion_in_box(
            mask: Optional[np.ndarray],
            x1b: float,
            y1b: float,
            x2b: float,
            y2b: float,
            pad_px: int = 0,
        ) -> bool:
            if mask is None:
                return False
            h_m, w_m = mask.shape[:2]
            pad = max(int(pad_px), 0)
            ix1 = max(0, int(math.floor(float(x1b))) - pad)
            iy1 = max(0, int(math.floor(float(y1b))) - pad)
            ix2 = min(w_m - 1, int(math.ceil(float(x2b))) + pad)
            iy2 = min(h_m - 1, int(math.ceil(float(y2b))) + pad)
            if ix2 < ix1 or iy2 < iy1:
                return False
            return bool(np.any(mask[iy1:iy2 + 1, ix1:ix2 + 1] > 0))

        def _det_has_local_motion(det: Detection, center_radius: int = 2, extra_pad_px: int = 0) -> bool:
            if bool(det.on_motion):
                return True

            bw = max(float(det.x2) - float(det.x1), 1.0)
            bh = max(float(det.y2) - float(det.y1), 1.0)
            bsz = max(bw, bh)

            # Small centroid shifts after morphology/packing can move the motion blob
            # outside a tiny center probe, so use a bbox-aware local check too.
            probe_r = max(int(center_radius), int(round(0.30 * bsz)) + int(extra_pad_px))
            probe_r = min(max(probe_r, 2), 14)
            box_pad = max(2 + int(extra_pad_px), int(round(0.45 * bsz)))
            box_pad = min(max(box_pad, 2), 16)

            if _mask_has_motion_near(boost_mask, det.cx, det.cy, probe_r):
                return True
            if _mask_has_motion_in_box(boost_mask, det.x1, det.y1, det.x2, det.y2, box_pad):
                return True
            return False

        def _guide_circle_mode() -> Optional[str]:
            if guide is None:
                return None
            if guide_exact:
                return "exact"
            if frozen_guide_active:
                return "frozen"
            if hold_guide_active:
                return "hold"
            return "soft"

        def _guide_circle_radius(mode: Optional[str] = None) -> float:
            mode = _guide_circle_mode() if mode is None else mode
            if mode is None:
                return 0.0
            if mode == "exact":
                return float(guide_lock_radius)
            if mode in ("frozen", "hold"):
                return float(guide_lock_radius * frozen_reacquire_radius_mult)
            return float(max(
                min(float(cfg.guide_gate_soft_frac) * diag, guide_base_radius * guide_soft_gate_cap_mult),
                0.0,
            ))

        def _soft_handoff_cont_ok(det: Detection) -> bool:
            if last_det_pos is None:
                return False
            dt_soft = max(int(frames_since_det), 1)
            pred_soft_x, pred_soft_y = _predict_projectile(
                (float(last_det_pos[0]), float(last_det_pos[1])),
                (float(last_vel[0]), float(last_vel[1])),
                dt_soft,
                cfg
            )
            soft_gate = max(14.0, 0.055 * diag * (1.0 + 0.35 * (dt_soft - 1)))
            return _xy_dist(det.cx, det.cy, pred_soft_x, pred_soft_y) <= soft_gate

        def _debounce_static_snap(
            det: Detection,
            pdist20: Optional[float],
            guide_exact_local: bool
        ) -> bool:
            """Require one-frame confirmation before accepting suspicious static reacquire."""
            if guide_exact_local and frames_since_det <= 1:
                return False
            if last_det_pos is None or frames_since_det <= 0:
                return False
            near_player_px = max(0.0, 0.008 * diag)
            if pdist20 is None or pdist20 > near_player_px:
                return False
            if _det_has_local_motion(det):
                return False

            nonlocal pending_static_det_pos, pending_static_det_frame
            if (
                pending_static_det_pos is not None and
                (t - pending_static_det_frame) == 1 and
                _xy_dist(det.cx, det.cy, pending_static_det_pos[0], pending_static_det_pos[1]) <= static_debounce_match_px
            ):
                return False

            dt_any = max(int(frames_since_det), 1)
            pred_any_x, pred_any_y = _predict_projectile(
                (float(last_det_pos[0]), float(last_det_pos[1])),
                (float(last_vel[0]), float(last_vel[1])),
                dt_any,
                cfg
            )
            pred_dist = _xy_dist(det.cx, det.cy, pred_any_x, pred_any_y)
            pred_gate = max(14.0, 0.045 * diag * (1.0 + 0.30 * (dt_any - 1)))
            # Only debounce truly suspicious static snaps.
            if pred_dist <= pred_gate:
                return False

            pending_static_det_pos = (float(det.cx), float(det.cy))
            pending_static_det_frame = int(t)
            return True

        def _reject_static_exact_reacquire(
            det: Detection,
            pdist20: Optional[float],
            guide_exact_local: bool
        ) -> bool:
            """Block exact-guide reacquire onto static near-player clutter after longer gaps."""
            if not guide_exact_local:
                return False
            if last_det_pos is None or frames_since_det < 2:
                return False
            near_player_px = max(0.0, 0.016 * diag)
            if pdist20 is None or pdist20 > near_player_px:
                return False
            if _det_has_local_motion(det):
                return False
            dt = max(int(frames_since_det), 1)
            step = _xy_dist(det.cx, det.cy, last_det_pos[0], last_det_pos[1]) / dt
            static_step = max(guide_static_speed_thresh, 1.8)
            return step <= static_step

        def _reject_nonmoving_reacquire(det: Detection, pdist: Optional[float]) -> bool:
            """Reject static false-positive detections during reacquire windows."""
            if last_det_pos is None or frames_since_det <= 0:
                return False
            has_local_motion = _det_has_local_motion(det)
            if has_local_motion:
                return False
            # Allow in-hand / immediate player-contact zones.
            if pdist is not None and pdist <= 0.0:
                return False

            dt = max(int(frames_since_det), 1)
            obs_step = _xy_dist(det.cx, det.cy, last_det_pos[0], last_det_pos[1]) / dt
            static_step = guide_static_speed_thresh
            exp_speed = math.sqrt(last_vel[0] * last_vel[0] + last_vel[1] * last_vel[1])
            min_moving_step = max(static_step, 0.18 * exp_speed)

            pred_x, pred_y = _predict_projectile(
                (float(last_det_pos[0]), float(last_det_pos[1])),
                (float(last_vel[0]), float(last_vel[1])),
                dt,
                cfg
            )
            d_pred = _xy_dist(det.cx, det.cy, pred_x, pred_y)
            dt_cap = min(dt, max(int(cfg.reacquire_gate_frames), 1))
            pred_gate = (
                float(cfg.reacquire_dist_frac) * diag *
                (1.0 + float(cfg.reacquire_dist_growth_per_frame) * (dt_cap - 1))
            )
            if dt > dt_cap:
                # Mildly widen for long gaps while still rejecting far static snaps.
                pred_gate *= 1.35

            strict_pred_gate = max(14.0, 0.60 * pred_gate)
            if d_pred > strict_pred_gate:
                return True
            if obs_step <= min_moving_step and d_pred > 0.45 * pred_gate:
                return True
            if d_pred > 1.15 * pred_gate:
                return True
            return False

        def _startup_candidate_ok(
            det: Detection,
            pdist20: Optional[float],
            gdist: Optional[float],
            is_guide_exact: bool
        ) -> bool:
            """Startup acceptance before first trusted detection is established."""
            if last_det_pos is not None:
                return True
            motion_local = _det_has_local_motion(det)
            if motion_local:
                return True

            # Exact guide can seed startup if very close to guide point.
            if is_guide_exact and gdist is not None and gdist <= 0.55 * guide_lock_radius:
                return True

            # Allow near-player startup (serve/hit handoff), but not far static balls.
            near_player_start_px = max(12.0, 0.012 * diag)
            if pdist20 is not None and pdist20 <= near_player_start_px:
                return True
            return False

        def _reject_static_player_snap(
            det: Detection,
            pdist20: Optional[float],
            continuity_dist: Optional[float]
        ) -> bool:
            """Reject static in-player snaps that jump away from trajectory."""
            if last_det_pos is None:
                return False
            # Guard short reacquire windows where static side balls commonly hijack.
            if frames_since_det > 3:
                return False
            near_player_static_px = max(0.0, 0.006 * diag)
            if pdist20 is None or pdist20 > near_player_static_px:
                return False
            motion_local = _det_has_local_motion(det)
            # Exact-guide handoff is only exempt when local motion support exists.
            if guide_exact and motion_local:
                return False
            dt = max(int(frames_since_det), 1)
            step = _xy_dist(det.cx, det.cy, last_det_pos[0], last_det_pos[1]) / dt
            static_step = max(guide_static_speed_thresh, 1.6)
            cont = float(continuity_dist) if continuity_dist is not None else 0.0
            cont_gate = max(12.0, 0.016 * diag)
            return cont > cont_gate and step <= static_step

        def _reject_static_lock_cluster(
            det: Detection,
            pdist20: Optional[float],
            motion_local: bool,
            handoff_exact: bool,
            owner_is_preferred: bool
        ) -> bool:
            """Prevent long no-motion lock on one parked ball cluster."""
            if motion_local:
                return False
            if static_lock_streak < static_lock_min_streak:
                return False
            if static_lock_center is None:
                return False
            if handoff_exact:
                return False
            if owner_is_preferred and guide_exact:
                return False
            if pdist20 is not None and pdist20 > static_lock_player_gate:
                return False
            d_lock = _xy_dist(det.cx, det.cy, static_lock_center[0], static_lock_center[1])
            return d_lock <= static_lock_radius

        prev_src = None
        if t > 0 and result[t - 1] is not None:
            prev_src = result[t - 1].source
        frame_preferred_owner_ids = preferred_owner_ids
        if owner_pref_by_frame is not None and 0 <= t < len(owner_pref_by_frame):
            local_pref = owner_pref_by_frame[t]
            if local_pref:
                frame_preferred_owner_ids = local_pref
        lost_counted = False
        motion_soft_window_ok = (
            prev_src in ('motion', 'guide', 'carry') and
            soft_carry_count < cfg.carry_interp_frames
        )
        tail_extra = 6 if (total_frames - 1 - t) <= 8 else 0
        carry_limit = int(cfg.carry_interp_frames) + int(tail_extra)
        if static_lock_streak >= static_lock_min_streak:
            carry_limit += max(4, int(cfg.carry_interp_frames))
        # Hard-cap: guide presence extends carry, but NOT infinitely.
        # Previously this set carry_limit = total_frames, allowing unlimited drift.
        CARRY_HARD_MAX = int(cfg.carry_interp_frames * 3)  # e.g. 15 frames max
        if guide is not None and not hold_guide_active:
            carry_limit = max(carry_limit, min(CARRY_HARD_MAX, carry_limit + int(cfg.carry_interp_frames)))
        else:
            tail_window = max(24, int(round(float(cfg.fps) * 0.8)))
            if (total_frames - 1 - t) <= tail_window:
                carry_limit += max(6, int(round(float(cfg.fps) * 0.25)))
        blocked_owner_rescue = bool(guide_exact and frames_since_det >= carry_limit)
        blocked_owner_penalty = max(65.0, 0.045 * diag)
        blocked_owner_guide_gate = max(14.0, 0.060 * diag)
        frames_since_det_in = int(frames_since_det)
        frame_audit: Optional[Dict[str, Any]] = None
        if audit_enabled and audit_start <= t <= audit_end:
            frame_audit = {
                "frame": int(t),
                "guide_present": bool(guide is not None),
                "guide_exact": bool(guide_exact),
                "guide_xy": [float(gx), float(gy)] if guide is not None else None,
                "num_dets": int(len(frame_dets)),
                "frames_since_det_in": frames_since_det_in,
                "carry_limit": int(carry_limit),
                "prev_source": str(prev_src) if prev_src is not None else None,
                "rej": {},
                "best_det": None,
                "result_source": None,
                "result_xy": None,
                "frames_since_det_out": None,
            }

        def _audit_rej(reason: str) -> None:
            if frame_audit is None:
                return
            rej = frame_audit["rej"]
            rej[reason] = int(rej.get(reason, 0)) + 1
        
        # â”€â”€ Try YOLO detection first â”€â”€
        best_det = None
        best_det_stage = None
        best_score = float('inf')

        def _pick_motion_det_in_visible_guide_circle() -> Optional[Tuple[Detection, str]]:
            """If motion is present inside the visible guide circle, pick the closest detection."""
            mode = _guide_circle_mode()
            if mode is None or not frame_dets:
                return None

            pick_radius = _guide_circle_radius(mode)
            if mode == "exact":
                center_probe = 3
                extra_pad = 1
                stage = "guide_circle_motion"
            elif mode == "frozen":
                center_probe = 4
                extra_pad = 3
                stage = "frozen_guide_motion_circle"
            elif mode == "hold":
                center_probe = 4
                extra_pad = 2
                stage = "hold_guide_motion_circle"
            else:
                center_probe = 3
                extra_pad = 2
                stage = "soft_guide_motion_circle"

            if pick_radius <= 0.0:
                return None

            best_local: Optional[Detection] = None
            best_local_score = float("inf")
            for det in frame_dets:
                key = (det.frame, round(det.cx), round(det.cy))
                if key in sideline_det_frames:
                    continue

                det_owner = det_owner_by_obj.get(id(det)) if has_owner_map else None
                owner_pen = 0.0
                if det_owner is not None:
                    owner_tid = int(det_owner)
                    if owner_tid in blocked_owner_ids:
                        continue
                    if frame_preferred_owner_ids and owner_tid not in frame_preferred_owner_ids:
                        owner_pen += 0.35 * owner_mismatch_pen

                d_circle = _xy_dist(det.cx, det.cy, gx, gy)
                if d_circle > pick_radius:
                    continue

                motion_local = _det_has_local_motion(
                    det, center_radius=center_probe, extra_pad_px=extra_pad
                )
                if not motion_local:
                    continue

                # Motion inside the visible guide circle should be allowed to win,
                # even if a stale blocked-radius region overlaps. Keep a soft penalty.
                blocked_pen = 8.0 if _is_blocked_by_radius(det) else 0.0

                continuity_pen = 0.0
                if last_det_pos is not None and frames_since_det > 0:
                    dt_gp = max(int(frames_since_det), 1)
                    pred_gp_x, pred_gp_y = _predict_projectile(
                        (float(last_det_pos[0]), float(last_det_pos[1])),
                        (float(last_vel[0]), float(last_vel[1])),
                        dt_gp,
                        cfg
                    )
                    continuity_pen = 0.15 * min(
                        _xy_dist(det.cx, det.cy, pred_gp_x, pred_gp_y),
                        0.18 * diag,
                    )

                # Primary behavior: choose motion-supported detection closest to the circle center.
                score_local = d_circle + continuity_pen + owner_pen + blocked_pen - 6.0 * float(det.conf)
                if score_local < best_local_score:
                    best_local_score = score_local
                    best_local = det

            if best_local is None:
                return None
            return best_local, stage

        # User-requested behavior: if a detection inside the visible guide circle has
        # motion support, choose the closest such detection before other heuristics.
        guide_motion_pick = _pick_motion_det_in_visible_guide_circle()
        if guide_motion_pick is not None:
            best_det, best_det_stage = guide_motion_pick
            best_score = -1.8

        # Exact-guide DET rescue:
        # If guide is exact and a valid detection is near the guide point,
        # lock to DET first so guide fallback doesn't override obvious detections.
        if guide_exact and frame_dets and best_det is None:
            rescue_gate = max(10.0, 1.15 * guide_lock_radius)
            rescue_best_d = float('inf')
            for det in frame_dets:
                key = (det.frame, round(det.cx), round(det.cy))
                if key in sideline_det_frames:
                    continue
                if _is_blocked_by_radius(det):
                    continue
                if has_owner_map:
                    det_owner_rescue = det_owner_by_obj.get(id(det))
                    if det_owner_rescue is not None and int(det_owner_rescue) in blocked_owner_ids:
                        continue
                d_rescue = _xy_dist(det.cx, det.cy, gx, gy)
                if d_rescue <= rescue_gate and d_rescue < rescue_best_d:
                    rescue_best_d = d_rescue
                    best_det = det
            if best_det is not None:
                best_det_stage = "exact_rescue"
                best_score = -1.0

        # Hard lock: if chosen track has an exact observation at this frame,
        # snap to the nearest matching detection.
        if guide_exact and frame_dets and best_det is None:
            guide_candidates: List[Tuple[Detection, float, bool]] = []
            for det in frame_dets:
                det_owner = det_owner_by_obj.get(id(det)) if has_owner_map else None
                owner_pen = 0.0
                owner_is_preferred = False
                if det_owner is not None:
                    owner_tid = int(det_owner)
                    owner_is_preferred = owner_tid in frame_preferred_owner_ids
                    if owner_tid in blocked_owner_ids:
                        stats['rej_other_track'] += 1
                        _audit_rej("owner_blocked")
                        continue
                    if frame_preferred_owner_ids and not owner_is_preferred:
                        owner_pen += owner_mismatch_pen
                        stats['owner_soft_pen'] += 1
                key = (det.frame, round(det.cx), round(det.cy))
                if key in sideline_det_frames:
                    _audit_rej("sideline_blocked")
                    continue
                if _is_blocked_by_radius(det):
                    stats['rej_blocked_radius'] += 1
                    _audit_rej("blocked_radius")
                    continue
                d = math.sqrt((det.cx - gx) ** 2 + (det.cy - gy) ** 2)
                motion_local_snap = _det_has_local_motion(det)
                # Static lock cluster — 6+ non-motion repeats at same spot.
                if _reject_static_lock_cluster(
                        det, None, motion_local_snap, False, False):
                    stats['rej_static_lock_cluster'] += 1
                    _audit_rej("static_lock_cluster")
                    continue
                # Anti-static: no motion + far from prediction → parked ball.
                # Exempt detections near players: balls change direction at hits.
                near_player_dist_snap = _closest_player_distance_local(det.cx, det.cy, frame_player_boxes_20)
                near_player_ok_snap = near_player_dist_snap is not None and near_player_dist_snap < 0.06 * diag
                if not motion_local_snap and not near_player_ok_snap and last_det_pos is not None and frames_since_det > 0:
                    dt_as = max(int(frames_since_det), 1)
                    pred_asx, pred_asy = _predict_projectile(
                        (float(last_det_pos[0]), float(last_det_pos[1])),
                        (float(last_vel[0]), float(last_vel[1])),
                        dt_as, cfg)
                    anti_static_gate_snap = 0.04 * diag
                    if frames_since_det > 3:
                        anti_static_gate_snap = 0.08 * diag
                    if _xy_dist(det.cx, det.cy, pred_asx, pred_asy) > anti_static_gate_snap:
                        stats['rej_static_snap'] += 1
                        _audit_rej("anti_static_far")
                        continue
                if d > guide_lock_radius:
                    _audit_rej("guide_lock_dist")
                    continue
                # Within exact-guide lock, prefer motion-supported and trajectory-consistent detections.
                continuity_pen = 0.0
                if last_det_pos is not None:
                    dt = max(int(frames_since_det), 1)
                    predx = last_det_pos[0] + last_vel[0] * dt
                    predy = last_det_pos[1] + last_vel[1] * dt
                    continuity_pen = min(_xy_dist(det.cx, det.cy, predx, predy), 0.20 * diag)
                motion_local = _det_has_local_motion(det)
                static_pen = 0.0 if motion_local else 28.0
                guide_score = d + 0.35 * continuity_pen + static_pen + owner_pen
                guide_candidates.append((det, guide_score, motion_local))

            if guide_candidates:
                prefer_motion = any(gc[2] for gc in guide_candidates)
                best_guide_det = None
                best_guide_score = float('inf')
                for det, gscore, motion_local in guide_candidates:
                    if prefer_motion and not motion_local:
                        continue
                    if gscore < best_guide_score:
                        best_guide_score = gscore
                        best_guide_det = det
                if best_guide_det is not None:
                    best_det = best_guide_det
                    best_det_stage = "exact"
                best_score = -1.0

        # Frozen/interpolated guide reacquire:
        # Treat the visible frozen guide circle as a reacquire search area and pick the
        # best detection inside it, while only applying hard/static filters.
        # This avoids over-rejecting valid balls during guide-loss gaps.
        if best_det is None and guide is not None and (frozen_guide_active or hold_guide_active) and not guide_exact and frame_dets:
            frozen_candidates: List[Tuple[Detection, float]] = []
            frozen_reacquire_radius = _guide_circle_radius()
            for det in frame_dets:
                key = (det.frame, round(det.cx), round(det.cy))
                if key in sideline_det_frames:
                    _audit_rej("sideline_blocked")
                    continue
                if _is_blocked_by_radius(det):
                    stats['rej_blocked_radius'] += 1
                    _audit_rej("blocked_radius")
                    continue

                det_owner = det_owner_by_obj.get(id(det)) if has_owner_map else None
                owner_pen = 0.0
                if det_owner is not None:
                    owner_tid = int(det_owner)
                    if owner_tid in blocked_owner_ids:
                        stats['rej_other_track'] += 1
                        _audit_rej("owner_blocked")
                        continue
                    if frame_preferred_owner_ids and owner_tid not in frame_preferred_owner_ids:
                        owner_pen += 0.35 * owner_mismatch_pen
                        stats['owner_soft_pen'] += 1

                d_frozen = _xy_dist(det.cx, det.cy, gx, gy)
                if d_frozen > frozen_reacquire_radius:
                    _audit_rej("frozen_circle")
                    continue

                # Use slightly wider motion check (4px vs default 2px) and also
                # check raw_motion, since boost mask can lose small ball blobs
                # after erosion+dilation shifts centroids.
                motion_local_frozen = _det_has_local_motion(det, center_radius=4, extra_pad_px=3)
                if not motion_local_frozen:
                    # Probe a little around the frozen guide center too; this helps when
                    # the YOLO box center is slightly off but motion is visible in the
                    # frozen/guide circle on-screen.
                    frozen_probe_r = int(round(max(6.0, min(20.0, 0.22 * frozen_reacquire_radius))))
                    if d_frozen <= max(10.0, 0.55 * frozen_reacquire_radius):
                        motion_local_frozen = (
                            _mask_has_motion_near(boost_mask, gx, gy, frozen_probe_r) or
                            _mask_has_motion_near(raw_motion, gx, gy, frozen_probe_r)
                        )
                if _reject_static_lock_cluster(det, None, motion_local_frozen, False, False):
                    stats['rej_static_lock_cluster'] += 1
                    _audit_rej("static_lock_cluster")
                    continue

                # Static-only reject: avoid parked clutter during frozen-guide reacquire.
                # If a detection lacks local motion, we strictly ban it within the 
                # frozen guide UNLESS it is extremely close to a player (a direct hit).
                # We do not allow static objects just because they are near the guide center.
                if not motion_local_frozen and last_det_pos is not None:
                    near_player_dist = _closest_player_distance_local(det.cx, det.cy, frame_player_boxes_20)
                    near_player_ok = near_player_dist is not None and near_player_dist < 0.06 * diag
                    if not near_player_ok:
                        stats['rej_static_snap'] += 1
                        _audit_rej("frozen_static_ban")
                        continue

                # Prefer nearest-to-guide, then motion support, then confidence / owner.
                # Reduced no-motion penalty (12 vs 18) since motion mask is less
                # reliable during guide-loss gaps.
                if motion_local_frozen:
                    no_motion_pen_frozen = 0.0
                elif d_frozen <= max(10.0, 0.35 * frozen_reacquire_radius):
                    no_motion_pen_frozen = 4.0
                else:
                    no_motion_pen_frozen = 12.0
                score_frozen = d_frozen + no_motion_pen_frozen + owner_pen - 8.0 * float(det.conf)
                frozen_candidates.append((det, score_frozen))

            if frozen_candidates:
                best_det, _ = min(frozen_candidates, key=lambda x: x[1])
                best_det_stage = "frozen_guide_circle"
                best_score = -1.5
        
        if best_det is None:
            # Fast-accept: high-confidence detection on guide path — skip anti-static
            # and continuity checks. This catches fast balls off rackets where velocity
            # prediction is stale but YOLO is confident and guide is correct.
            if guide is not None and frame_dets:
                for det in frame_dets:
                    # Frozen (yellow) guides strongly demand motion to avoid snaps.
                    # Exact (green) and hold (blue) guides are allowed to bypass.
                    if det.conf >= 0.50 and (not frozen_guide_active or _det_has_local_motion(det)):
                        gdist_hi = _xy_dist(det.cx, det.cy, gx, gy)
                        if gdist_hi <= guide_lock_radius:
                            # Check owner isn't hard-blocked
                            det_owner_hi = det_owner_by_obj.get(id(det)) if has_owner_map else None
                            owner_blocked_hi = False
                            if det_owner_hi is not None:
                                owner_tid_hi = int(det_owner_hi)
                                if owner_tid_hi in blocked_owner_ids:
                                    owner_blocked_hi = True
                            if not owner_blocked_hi:
                                best_det = det
                                best_det_stage = "high_conf_guide"
                                best_score = -2.0
                                break

        if best_det is None:
            for det in frame_dets:
                det_owner = det_owner_by_obj.get(id(det)) if has_owner_map else None
                owner_penalty = 0.0
                owner_is_preferred = False
                if det_owner is not None:
                    owner_tid = int(det_owner)
                    owner_is_preferred = owner_tid in frame_preferred_owner_ids
                    if owner_tid in blocked_owner_ids:
                        stats['rej_other_track'] += 1
                        _audit_rej("owner_blocked")
                        continue
                    if frame_preferred_owner_ids and not owner_is_preferred:
                        owner_penalty += owner_mismatch_pen
                        if last_det_pos is None:
                            owner_penalty += owner_startup_extra_pen
                        stats['owner_soft_pen'] += 1
                key = (det.frame, round(det.cx), round(det.cy))
                if key in sideline_det_frames:
                    _audit_rej("sideline_blocked")
                    continue

                if _is_blocked_by_radius(det):
                    stats['rej_blocked_radius'] += 1
                    _audit_rej("blocked_radius")
                    continue
                gdist_local = _xy_dist(det.cx, det.cy, gx, gy) if guide is not None else None
                motion_local = _det_has_local_motion(det)
                # Static lock cluster — 6+ non-motion repeats at same spot.
                if _reject_static_lock_cluster(
                        det, None, motion_local, False, False):
                    stats['rej_static_lock_cluster'] += 1
                    _audit_rej("static_lock_cluster")
                    continue
                # Anti-static: no motion + far from prediction → parked ball.
                # Exempt detections near players: balls change direction at hits.
                near_player_dist = _closest_player_distance_local(det.cx, det.cy, frame_player_boxes_20)
                near_player_ok = near_player_dist is not None and near_player_dist < 0.06 * diag
                
                # If the guide is active and in frozen (yellow) mode, we absolutely 
                # demand motion. This stops the yellow guide from grabbing the net post 
                # or a parked ball from miles away.
                # Exact (green) and hold (blue) guides are allowed to pick static balls.
                if guide is not None and frozen_guide_active and not motion_local:
                    stats['rej_static_snap'] += 1
                    _audit_rej("guide_static_ban")
                    continue

                if not motion_local and not near_player_ok and last_det_pos is not None and frames_since_det > 0:
                    dt_as = max(int(frames_since_det), 1)
                    pred_asx, pred_asy = _predict_projectile(
                        (float(last_det_pos[0]), float(last_det_pos[1])),
                        (float(last_vel[0]), float(last_vel[1])),
                        dt_as, cfg)
                    
                    if guide_exact:
                        anti_static_gate = 0.015 * diag
                    else:
                        anti_static_gate = 0.035 * diag
                        if frames_since_det > 3:
                            anti_static_gate = 0.055 * diag
                            
                    if _xy_dist(det.cx, det.cy, pred_asx, pred_asy) > anti_static_gate:
                        stats['rej_static_snap'] += 1
                        _audit_rej("anti_static_far")
                        continue
                # Guide gate — simple distance check.
                if guide is not None and gdist_local is not None:
                    gate = min(
                        cfg.guide_gate_soft_frac * diag,
                        guide_base_radius * guide_soft_gate_cap_mult,
                    )
                    if guide_exact:
                        gate = guide_lock_radius
                    if gdist_local > gate:
                        _audit_rej("guide_gate")
                        continue
                # Court context — outside court + far from players.
                if court_polygon is not None:
                    court_dist_v = -cv2.pointPolygonTest(
                        court_polygon, (float(det.cx), float(det.cy)), True)
                    outside_dist = max(court_dist_v, 0.0)
                    pdist = _closest_player_distance_local(det.cx, det.cy, frame_player_boxes_20)
                    if outside_dist > cfg.court_expand_px * cfg.outside_reject_expand_mult:
                        if pdist is None or pdist > cfg.toss_player_allow_frac * diag:
                            stats['rej_context'] += 1
                            _audit_rej("context_outside")
                            continue
                
                # Continuity from last YOLO detection (not motion-tracked pos)
                continuity = 0.0
                if last_det_pos is not None:
                    pred_cx = last_det_pos[0] + last_vel[0] * max(frames_since_det, 1)
                    pred_cy = last_det_pos[1] + last_vel[1] * max(frames_since_det, 1)
                    cdist = math.sqrt((det.cx - pred_cx)**2 + (det.cy - pred_cy)**2)
                    rdist = math.sqrt((det.cx - last_det_pos[0])**2 +
                                      (det.cy - last_det_pos[1])**2)
                    continuity = min(cdist, rdist) * 0.8

                # Guide distance penalty.
                guide_pen = 0.0
                if gdist_local is not None:
                    guide_pen = 0.30 * min(gdist_local, 0.20 * diag)

                # Motion overlap is optional; keep this weak to avoid yellow-mask overbias.
                motion_bonus = -5.0 if motion_local else 0.0
                
                score = continuity + motion_bonus + guide_pen + owner_penalty
                if score < best_score:
                    best_score = score
                    best_det = det
                    best_det_stage = "general"
        
        if best_det is not None:
            prev_det_pos_for_lock = last_det_pos
            best_det_motion_local = _det_has_local_motion(best_det)
            pending_static_det_pos = None
            pending_static_det_frame = -10**9
            # Update velocity from YOLO detections only (trustworthy)
            # Allow velocity reset on longer gaps (10 frames) so post-hit
            # trajectory can recover after racket-occlusion gaps.
            if last_det_pos is not None and frames_since_det <= 10:
                dt = max(frames_since_det, 1)
                raw_vx = (best_det.cx - last_det_pos[0]) / dt
                raw_vy = (best_det.cy - last_det_pos[1]) / dt

                # Adaptive gravity from observed vertical acceleration on nearby detections.
                if (cfg.gravity_enabled and cfg.gravity_adapt_enabled and
                        prev_det_raw_vel is not None and dt <= 2):
                    obs_ay = (raw_vy - prev_det_raw_vel[1]) / max(float(dt), 1.0)
                    ay_min = -0.30 * base_gravity
                    ay_max = 4.50 * base_gravity
                    if ay_min <= obs_ay <= ay_max:
                        g_alpha = max(0.0, min(1.0, float(cfg.gravity_adapt_alpha)))
                        adaptive_gravity = (
                            (1.0 - g_alpha) * adaptive_gravity +
                            g_alpha * obs_ay
                        )
                        g_lo = base_gravity * max(0.10, float(cfg.gravity_adapt_min_mult))
                        g_hi = base_gravity * max(float(cfg.gravity_adapt_max_mult), g_lo + 1e-6)
                        adaptive_gravity = float(max(g_lo, min(g_hi, adaptive_gravity)))

                # Bounce-aware vertical reset: down->up sign flip should not be over-smoothed.
                if prev_det_raw_vel is not None:
                    down_thr = _fps_norm_pxpf(cfg.bounce_detect_min_down_speed, cfg)
                    up_thr = _fps_norm_pxpf(cfg.bounce_detect_min_up_speed, cfg)
                    if prev_det_raw_vel[1] >= down_thr and raw_vy <= -up_thr:
                        bounce_vy = -abs(prev_det_raw_vel[1]) * max(0.0, float(cfg.bounce_restitution))
                        raw_vy = 0.50 * raw_vy + 0.50 * bounce_vy
                        raw_vx *= max(0.0, min(1.0, float(cfg.bounce_tangent_damping)))
                        stats['bounce'] += 1

                alpha_x = max(0.0, min(1.0, float(cfg.physics_vel_alpha_x)))
                alpha_y = max(0.0, min(1.0, float(cfg.physics_vel_alpha_y)))
                last_vel = (
                    alpha_x * raw_vx + (1.0 - alpha_x) * last_vel[0],
                    alpha_y * raw_vy + (1.0 - alpha_y) * last_vel[1]
                )
                prev_det_raw_vel = (raw_vx, raw_vy)
            else:
                prev_det_raw_vel = None
            last_pos = (best_det.cx, best_det.cy)
            last_det_pos = (best_det.cx, best_det.cy)
            last_det_area = max(best_det.area, 1.0)
            last_motion_area = max(best_det.area, 1.0)
            frames_since_det = 0
            motion_vel_history.clear()
            last_motion_vel = None
            last_motion_pos = None
            result[t] = FrameResult(
                cx=best_det.cx, cy=best_det.cy,
                conf=best_det.conf, interpolated=False,
                bbox=(best_det.x1, best_det.y1, best_det.x2, best_det.y2),
                source='det')
            stats['det'] += 1
            soft_carry_count = 0
            if best_det_motion_local:
                static_lock_streak = 0
                static_lock_center = None
            else:
                if (
                    static_lock_center is not None and
                    _xy_dist(best_det.cx, best_det.cy, static_lock_center[0], static_lock_center[1]) <= static_lock_reset_step
                ):
                    static_lock_streak += 1
                    a = 0.80
                    static_lock_center = (
                        a * static_lock_center[0] + (1.0 - a) * best_det.cx,
                        a * static_lock_center[1] + (1.0 - a) * best_det.cy
                    )
                else:
                    static_lock_streak = 1
                    static_lock_center = (best_det.cx, best_det.cy)
                if (
                    prev_det_pos_for_lock is not None and
                    _xy_dist(best_det.cx, best_det.cy, prev_det_pos_for_lock[0], prev_det_pos_for_lock[1]) > static_lock_reset_step
                ):
                    static_lock_streak = 1
                    static_lock_center = (best_det.cx, best_det.cy)
        
        elif last_det_pos is not None and (
            frames_since_det < max_motion_gap or
            motion_soft_window_ok or
            (guide is not None and _guide_circle_radius() > 0.0)
        ):
            # â”€â”€ No YOLO det â€” try motion blob tracking â”€â”€
            frames_since_det += 1
            
            # Predict from the latest active state (det/motion/carry) to avoid stale carry drift.
            carry_anchor_pos = last_pos if last_pos is not None else last_det_pos
            carry_anchor_vel = last_motion_vel if last_motion_vel is not None else last_vel
            # Use local anchor dt for physics continuity when we already have carry/motion state.
            anchor_dt = 1 if last_pos is not None else max(frames_since_det, 1)
            
            # Predict kinematics using the Track's Kalman filter (which handles gravity/drag)
            # `chosen` is the selected ball track from the pre-loop selection stage.
            _kf = chosen.kf if (chosen is not None) else None
            if _kf is not None:
                pred_cx, pred_cy = _kf.predict()
                pred_vel = _kf.get_velocity()
                # Dynamic search radius: derived from KF covariance matrix position uncertainty
                # Scaled up for "pink motion" area coverage.
                search_r = _kf.get_search_radius(scale_mult=3.0)
            else:
                pred_cx, pred_cy = _predict_projectile(
                    (float(carry_anchor_pos[0]), float(carry_anchor_pos[1])),
                    (float(carry_anchor_vel[0]), float(carry_anchor_vel[1])),
                    anchor_dt,
                    cfg
                )
                pred_vel = _predict_projectile_vel(
                    (float(carry_anchor_vel[0]), float(carry_anchor_vel[1])),
                    anchor_dt,
                    cfg
                )
                search_r = (
                    motion_search_base +
                    motion_search_growth * max(frames_since_det - 1, 0) +
                    motion_search_vel_mult * math.sqrt(pred_vel[0] ** 2 + pred_vel[1] ** 2) * max(anchor_dt, 1)
                )
            
            pred_det_cx = pred_cx
            pred_det_cy = pred_cy
            search_r = min(max(search_r, motion_search_base), motion_search_max)
            
            det_speed = math.sqrt(pred_vel[0] ** 2 + pred_vel[1] ** 2)
            blob = None
            blob_search_cx = float(pred_cx)
            blob_search_cy = float(pred_cy)
            blob_search_r = float(search_r)
            blob_ref_cx = float(pred_cx)
            blob_ref_cy = float(pred_cy)
            blob_from_guide_circle = False
            motion_ref_area = last_motion_area if last_motion_area is not None else last_det_area
            guide_mode = _guide_circle_mode()
            guide_motion_search_r = _guide_circle_radius(guide_mode)
            allow_guide_motion_search = bool(guide_mode is not None and guide_motion_search_r > 0.0)
            prefer_guide_motion_search = bool(guide_mode in ("frozen", "hold"))
            if det_speed >= motion_min_speed or allow_guide_motion_search:
                # Prefer searching the active frozen/hold guide circle first so the
                # selector follows what the visible yellow/blue circle implies.
                if allow_guide_motion_search and prefer_guide_motion_search and boost_mask is not None:
                    blob_result = _find_motion_blob(
                        boost_mask, gx, gy, guide_motion_search_r,
                        last_det_cx=carry_anchor_pos[0], last_det_cy=carry_anchor_pos[1],
                        last_vel=pred_vel,
                        ref_ball_area=motion_ref_area,
                        player_boxes=frame_player_boxes_0,
                        frame_idx=t, active_motion_tracks=motion_tracks, prev_motion_pos=last_motion_pos)
                    if blob_result is not None:
                        blob_cx, blob_cy, blob_area, is_latched = blob_result
                        if is_latched or _motion_blob_physics_ok(
                            blob_cx, blob_cy, gx, gy,
                            carry_anchor_pos, pred_vel, anchor_dt,
                            last_motion_vel, cfg, diag
                        ):
                            blob = (blob_cx, blob_cy, blob_area, is_latched)
                            blob_search_cx = float(gx)
                            blob_search_cy = float(gy)
                            blob_search_r = float(guide_motion_search_r)
                            blob_ref_cx = float(gx)
                            blob_ref_cy = float(gy)
                            blob_from_guide_circle = True
                        else:
                            stats['rej_motion_physics'] += 1
                # Projectile-predicted search (default path / fallback).
                if blob is None and boost_mask is not None:
                    blob_result = _find_motion_blob(
                        boost_mask, pred_cx, pred_cy, search_r,
                        last_det_cx=carry_anchor_pos[0], last_det_cy=carry_anchor_pos[1],
                        last_vel=pred_vel,
                        ref_ball_area=motion_ref_area,
                        player_boxes=frame_player_boxes_0,
                        frame_idx=t, active_motion_tracks=motion_tracks, prev_motion_pos=last_motion_pos)
                    if blob_result is not None:
                        blob_cx, blob_cy, blob_area, is_latched = blob_result
                        if is_latched or _motion_blob_physics_ok(
                            blob_cx, blob_cy, pred_cx, pred_cy,
                            carry_anchor_pos, pred_vel, anchor_dt,
                            last_motion_vel, cfg, diag
                        ):
                            blob = (blob_cx, blob_cy, blob_area, is_latched)
                        else:
                            stats['rej_motion_physics'] += 1
                # Guide/frozen-guide motion reacquire: if the visible circle is on the ball,
                # search inside that circle too (not only around stale projectile prediction).
                if blob is None and allow_guide_motion_search and boost_mask is not None:
                    blob_result = _find_motion_blob(
                        boost_mask, gx, gy, guide_motion_search_r,
                        last_det_cx=carry_anchor_pos[0], last_det_cy=carry_anchor_pos[1],
                        last_vel=pred_vel,
                        ref_ball_area=motion_ref_area,
                        player_boxes=frame_player_boxes_0,
                        frame_idx=t, active_motion_tracks=motion_tracks, prev_motion_pos=last_motion_pos)
                    if blob_result is not None:
                        blob_cx, blob_cy, blob_area, is_latched = blob_result
                        if is_latched or _motion_blob_physics_ok(
                            blob_cx, blob_cy, gx, gy,
                            carry_anchor_pos, pred_vel, anchor_dt,
                            last_motion_vel, cfg, diag
                        ):
                            blob = (blob_cx, blob_cy, blob_area, is_latched)
                            blob_search_cx = float(gx)
                            blob_search_cy = float(gy)
                            blob_search_r = float(guide_motion_search_r)
                            blob_ref_cx = float(gx)
                            blob_ref_cy = float(gy)
                            blob_from_guide_circle = True
                        else:
                            stats['rej_motion_physics'] += 1
                # Raw motion fallback only in very short gaps.
                if blob is None and allow_guide_motion_search and prefer_guide_motion_search and raw_motion is not None and (
                    frames_since_det <= 2 or frozen_guide_active or hold_guide_active
                ):
                    blob_result = _find_motion_blob(
                        raw_motion, gx, gy, guide_motion_search_r,
                        last_det_cx=carry_anchor_pos[0], last_det_cy=carry_anchor_pos[1],
                        last_vel=pred_vel,
                        ref_ball_area=motion_ref_area,
                        player_boxes=frame_player_boxes_0,
                        frame_idx=t, active_motion_tracks=motion_tracks, prev_motion_pos=last_motion_pos)
                    if blob_result is not None:
                        blob_cx, blob_cy, blob_area, is_latched = blob_result
                        if is_latched or _motion_blob_physics_ok(
                            blob_cx, blob_cy, gx, gy,
                            carry_anchor_pos, pred_vel, anchor_dt,
                            last_motion_vel, cfg, diag
                        ):
                            blob = (blob_cx, blob_cy, blob_area, is_latched)
                            blob_search_cx = float(gx)
                            blob_search_cy = float(gy)
                            blob_search_r = float(guide_motion_search_r)
                            blob_ref_cx = float(gx)
                            blob_ref_cy = float(gy)
                            blob_from_guide_circle = True
                        else:
                            stats['rej_motion_physics'] += 1
                if blob is None and raw_motion is not None and frames_since_det <= 2:
                    blob_result = _find_motion_blob(
                        raw_motion, pred_cx, pred_cy, search_r,
                        last_det_cx=carry_anchor_pos[0], last_det_cy=carry_anchor_pos[1],
                        last_vel=pred_vel,
                        ref_ball_area=motion_ref_area,
                        player_boxes=frame_player_boxes_0,
                        frame_idx=t, active_motion_tracks=motion_tracks, prev_motion_pos=last_motion_pos)
                    if blob_result is not None:
                        blob_cx, blob_cy, blob_area, is_latched = blob_result
                        if is_latched or _motion_blob_physics_ok(
                            blob_cx, blob_cy, pred_cx, pred_cy,
                            carry_anchor_pos, pred_vel, anchor_dt,
                            last_motion_vel, cfg, diag
                        ):
                            blob = (blob_cx, blob_cy, blob_area, is_latched)
                        else:
                            stats['rej_motion_physics'] += 1
                if blob is None and allow_guide_motion_search and raw_motion is not None and (
                    frames_since_det <= 2 or frozen_guide_active or hold_guide_active
                ):
                    blob = _find_motion_blob(
                        raw_motion, gx, gy, guide_motion_search_r,
                        last_det_cx=carry_anchor_pos[0], last_det_cy=carry_anchor_pos[1],
                        last_vel=pred_vel,
                        ref_ball_area=motion_ref_area,
                        player_boxes=frame_player_boxes_0,
                        frame_idx=t, active_motion_tracks=motion_tracks, prev_motion_pos=last_motion_pos)
                    if blob is not None:
                        blob_search_cx = float(gx)
                        blob_search_cy = float(gy)
                        blob_search_r = float(guide_motion_search_r)
                        blob_ref_cx = float(gx)
                        blob_ref_cy = float(gy)
                        blob_from_guide_circle = True
            
            if blob is not None:
                blob_cx, blob_cy, blob_area, is_latched = blob
                ref_area = last_motion_area if last_motion_area is not None else last_det_area
                if ref_area is not None:
                    area_ratio = blob_area / max(float(ref_area), 1.0)
                    area_min = 0.12
                    area_max = 6.00
                    if blob_from_guide_circle and allow_guide_motion_search:
                        # Yellow/blue/green circle reacquire should tolerate more mask-size
                        # distortion (racket hits, blur, partial blobs near players).
                        area_min = 0.06
                        area_max = 10.00
                    if area_ratio < area_min or area_ratio > area_max:
                        stats['rej_motion_area'] += 1
                        blob = None

            if blob is not None:
                blob_cx, blob_cy, blob_area, is_latched = blob
                raw_blob_cx, raw_blob_cy = blob_cx, blob_cy
                motion_physics_ok = is_latched or _motion_blob_physics_ok(
                        blob_cx, blob_cy, blob_ref_cx, blob_ref_cy,
                        carry_anchor_pos, pred_vel, anchor_dt,
                        last_motion_vel, cfg, diag)
                if (not motion_physics_ok and blob_from_guide_circle and allow_guide_motion_search):
                    # When the active guide circle is on the ball (yellow/blue/green),
                    # allow reacquire even if projectile continuity is temporarily bad.
                    d_guide_blob = _xy_dist(blob_cx, blob_cy, gx, gy)
                    guide_core = max(10.0, 0.45 * guide_motion_search_r)
                    motion_physics_ok = d_guide_blob <= guide_core
                if not motion_physics_ok:
                    stats['rej_motion_physics'] += 1
                    blob = None
                else:
                    # Physics-guided correction: pull noisy blob toward predicted trajectory
                    # and clamp one-frame motion step if it gets erratic.
                    if is_latched:
                        guided_cx, guided_cy = blob_cx, blob_cy
                    else:
                        guided_cx, guided_cy, was_clamped = _physics_guide_motion_blob(
                            blob_cx, blob_cy, blob_ref_cx, blob_ref_cy,
                            last_pos, pred_vel, anchor_dt, cfg, diag)
                        if was_clamped:
                            stats['motion_guided_clamped'] += 1
                    motion_jump_ok = _motion_jump_probation_ok(
                        raw_blob_cx, raw_blob_cy,
                        guided_cx, guided_cy,
                        blob_ref_cx, blob_ref_cy,
                        blob_search_r,
                        is_latched,
                        blob_from_guide_circle,
                        carry_anchor_pos,
                        pred_vel,
                        anchor_dt,
                    )
                    near_player_ok = motion_jump_ok and _motion_player_probation_ok(
                        raw_blob_cx, raw_blob_cy,
                        guided_cx, guided_cy,
                        blob_area,
                        blob_ref_cx, blob_ref_cy,
                        blob_search_r,
                        ref_area,
                        is_latched,
                        blob_from_guide_circle,
                        carry_anchor_pos,
                        pred_vel,
                        anchor_dt,
                    )
                    if not motion_jump_ok or not near_player_ok:
                        if not motion_jump_ok:
                            stats['rej_motion_jump'] += 1
                        else:
                            stats['rej_motion_player'] += 1
                        blob = None
                        last_motion_vel = None
                    else:
                        guided_cx, guided_cy = _stabilize_probation_motion(
                            raw_blob_cx, raw_blob_cy,
                            guided_cx, guided_cy,
                            blob_ref_cx, blob_ref_cy,
                            blob_search_r,
                            blob_from_guide_circle,
                            pred_vel,
                            anchor_dt,
                        )
                        blob = (guided_cx, guided_cy, blob_area, is_latched)
                        # Velocity Smoothing: When applying motion tracking velocity, blend it
                        # with the previous stable state so a single erratic frame doesn't ruin the arc.
                        motion_dt = max(int(anchor_dt), 1)
                        raw_mot_vx = (guided_cx - carry_anchor_pos[0]) / motion_dt
                        raw_mot_vy = (guided_cy - carry_anchor_pos[1]) / motion_dt

                        if last_motion_vel is not None:
                            # Blend with previous motion velocity
                            alpha = 0.35  # Trust new motion blob 35%, history 65%
                            last_motion_vel = (
                                alpha * raw_mot_vx + (1.0 - alpha) * last_motion_vel[0],
                                alpha * raw_mot_vy + (1.0 - alpha) * last_motion_vel[1]
                            )
                        else:
                            # First motion frame after a gap: blend with the main physics velocity
                            alpha = 0.50
                            last_motion_vel = (
                                alpha * raw_mot_vx + (1.0 - alpha) * last_vel[0],
                                alpha * raw_mot_vy + (1.0 - alpha) * last_vel[1]
                            )
            else:
                last_motion_vel = None

            # Reject motion blobs that snap onto a player body away from prediction.
            if blob is not None:
                blob_cx, blob_cy, blob_area, is_latched = blob
                pd_blob = _closest_player_distance_local(blob_cx, blob_cy, frame_player_boxes_0)
                if pd_blob is not None and pd_blob <= 0.0:
                    d_blob_ref = _xy_dist(blob_cx, blob_cy, blob_ref_cx, blob_ref_cy)
                    if blob_from_guide_circle and allow_guide_motion_search:
                        # Guide-circle reacquire near players is common at contact; keep only
                        # a looser guard so center-of-circle hits are not dropped.
                        if d_blob_ref > max(0.03 * diag, 0.60 * guide_motion_search_r):
                            blob = None
                            last_motion_vel = None
                    elif d_blob_ref > 0.03 * diag and not is_latched:
                        blob = None
                        last_motion_vel = None

            if blob is not None:
                blob_cx, blob_cy, blob_area, is_latched = blob
                
                # Velocity sanity: check if motion blob velocity is consistent
                if last_pos is not None:
                    mot_vx = blob_cx - last_pos[0]
                    mot_vy = blob_cy - last_pos[1]
                    motion_vel_history.append((mot_vx, mot_vy))
                    
                    # If we have 3+ motion readings, check for erratic jumps
                    if len(motion_vel_history) >= 3:
                        vxs = [v[0] for v in motion_vel_history[-3:]]
                        vys = [v[1] for v in motion_vel_history[-3:]]
                        vx_var = max(vxs) - min(vxs)
                        vy_var = max(vys) - min(vys)
                        # If direction is wildly inconsistent, stop tracking
                        if vx_var > erratic_var_thresh or vy_var > erratic_var_thresh:
                            last_pos = (pred_cx, pred_cy)
                            soft_window_ok = (
                                prev_src in ('motion', 'guide', 'carry') and
                                soft_carry_count < cfg.carry_interp_frames
                            )
                            can_short_carry = (
                                (frames_since_det <= carry_limit or soft_window_ok) and
                                _carry_after_failed_motion_ok(pred_cx, pred_cy, search_r)
                            )
                            if can_short_carry:
                                result[t] = FrameResult(
                                    cx=pred_cx, cy=pred_cy,
                                    conf=0.16, interpolated=True, bbox=None, source='carry',
                                    search_cx=pred_cx, search_cy=pred_cy, search_radius=search_r)
                                stats['carry'] += 1
                                soft_carry_count += 1
                            else:
                                guide_vel = last_motion_vel if last_motion_vel is not None else last_vel
                                # Interpolated/frozen guide points are hints for association only.
                                # Only exact guide observations are allowed to become output positions.
                                if guide is not None and guide_exact and _guide_path_consistent(
                                        gx, gy, last_pos, guide_vel, frames_since_det, cfg, diag, guide_exact=guide_exact):
                                    result[t] = FrameResult(
                                        cx=gx,
                                        cy=gy,
                                        conf=0.18,
                                        interpolated=False, bbox=None, source='guide',
                                        search_cx=float(gx), search_cy=float(gy), search_radius=float(_guide_circle_radius("exact")))
                                    stats['guide'] += 1
                                    last_pos = (gx, gy)
                                    soft_carry_count = min(
                                        soft_carry_count + 1, max(cfg.carry_interp_frames, 1))
                                else:
                                    stats['lost'] += 1
                                    lost_counted = True
                                    soft_carry_count = 0
                            last_motion_vel = pred_vel
                            continue
                
                last_pos = (blob_cx, blob_cy)
                last_motion_pos = (blob_cx, blob_cy)
                
                result[t] = FrameResult(
                    cx=blob_cx, cy=blob_cy,
                    conf=0.3, interpolated=False, bbox=None,
                    source='motion',
                    search_cx=blob_search_cx, search_cy=blob_search_cy,
                    search_radius=blob_search_r)
                stats['motion'] += 1
                soft_carry_count = 0
                last_motion_area = max(float(blob_area), 1.0)
                pending_static_det_pos = None
                pending_static_det_frame = -10**9
                static_lock_streak = 0
                static_lock_center = None
            else:
                last_pos = (pred_cx, pred_cy)
                soft_window_ok = (
                    prev_src in ('motion', 'guide', 'carry') and
                    soft_carry_count < cfg.carry_interp_frames
                )
                can_short_carry = (
                    (frames_since_det <= carry_limit or soft_window_ok) and
                    _carry_after_failed_motion_ok(pred_cx, pred_cy, search_r)
                )
                if can_short_carry:
                    # Fix 4: Prefer guide over carry when guide_exact is true.
                    guide_vel_c = last_motion_vel if last_motion_vel is not None else last_vel
                    if (guide is not None and guide_exact and
                            _guide_path_consistent(gx, gy, last_pos, guide_vel_c, frames_since_det, cfg, diag, guide_exact=guide_exact)):
                        result[t] = FrameResult(
                            cx=gx, cy=gy,
                            conf=0.18, interpolated=False, bbox=None, source='guide',
                            search_cx=float(gx), search_cy=float(gy), search_radius=float(_guide_circle_radius("exact")))
                        stats['guide'] += 1
                        last_pos = (gx, gy)
                        last_motion_vel = None
                    else:
                        result[t] = FrameResult(
                            cx=pred_cx, cy=pred_cy,
                            conf=0.16, interpolated=True, bbox=None, source='carry',
                            search_cx=pred_cx, search_cy=pred_cy, search_radius=search_r)
                        stats['carry'] += 1
                    soft_carry_count += 1
                    last_motion_vel = pred_vel if result[t].source == 'carry' else last_motion_vel
                    if static_lock_streak > 0:
                        static_lock_streak = max(0, static_lock_streak - 1)
                        if static_lock_streak == 0:
                            static_lock_center = None
                else:
                    stats['lost'] += 1
                    lost_counted = True
                    soft_carry_count = 0
                    last_motion_vel = pred_vel
                    if static_lock_streak > 0:
                        static_lock_streak = max(0, static_lock_streak - 1)
                        if static_lock_streak == 0:
                            static_lock_center = None
        else:
            frames_since_det += 1
            carry_anchor_pos = last_pos if last_pos is not None else last_det_pos
            carry_anchor_vel = last_motion_vel if last_motion_vel is not None else last_vel
            soft_window_ok = (
                prev_src in ('motion', 'guide', 'carry') and
                soft_carry_count < cfg.carry_interp_frames
            )
            can_short_carry = (
                carry_anchor_pos is not None and
                (frames_since_det <= carry_limit or soft_window_ok)
            )
            if can_short_carry:
                pred_cx, pred_cy = _predict_projectile(
                    (float(carry_anchor_pos[0]), float(carry_anchor_pos[1])),
                    (float(carry_anchor_vel[0]), float(carry_anchor_vel[1])),
                    1,
                    cfg
                )
                pred_vel = _predict_projectile_vel(
                    (float(carry_anchor_vel[0]), float(carry_anchor_vel[1])),
                    1,
                    cfg
                )
                pred_search_r = (
                    float(getattr(result[t - 1], "search_radius", motion_search_base))
                    if t > 0 and result[t - 1] is not None else motion_search_base
                )
                if not _carry_after_failed_motion_ok(pred_cx, pred_cy, pred_search_r):
                    stats['lost'] += 1
                    lost_counted = True
                    soft_carry_count = 0
                    last_motion_vel = None
                    if static_lock_streak > 0:
                        static_lock_streak = max(0, static_lock_streak - 1)
                        if static_lock_streak == 0:
                            static_lock_center = None
                    continue
                # Fix 4: Prefer guide over carry when guide_exact is true.
                guide_vel_c2 = last_motion_vel if last_motion_vel is not None else last_vel
                if (guide is not None and guide_exact and
                        _guide_path_consistent(gx, gy, last_pos, guide_vel_c2, frames_since_det, cfg, diag, guide_exact=guide_exact)):
                    result[t] = FrameResult(
                        cx=gx, cy=gy,
                        conf=0.18, interpolated=False, bbox=None, source='guide',
                        search_cx=float(gx), search_cy=float(gy), search_radius=float(_guide_circle_radius("exact")))
                    stats['guide'] += 1
                    last_pos = (gx, gy)
                    last_motion_vel = None
                else:
                    last_pos = (pred_cx, pred_cy)
                    result[t] = FrameResult(
                        cx=pred_cx, cy=pred_cy,
                        conf=0.15, interpolated=True, bbox=None, source='carry')
                    stats['carry'] += 1
                    last_motion_vel = pred_vel
                soft_carry_count += 1
                if static_lock_streak > 0:
                    static_lock_streak = max(0, static_lock_streak - 1)
                    if static_lock_streak == 0:
                        static_lock_center = None
            else:
                stats['lost'] += 1
                lost_counted = True
                soft_carry_count = 0
                last_motion_vel = None
                if static_lock_streak > 0:
                    static_lock_streak = max(0, static_lock_streak - 1)
                    if static_lock_streak == 0:
                        static_lock_center = None

        # Strict guide fallback: only when det/motion/carry all failed and guide
        # remains consistent with the current trajectory.
        if result[t] is None and guide is not None:
            # Never allow guide to bootstrap a brand-new track start.
            if last_det_pos is None:
                continue
            # Interpolated/frozen guide is for gating/continuity only, not a direct position source.
            if not guide_exact:
                continue
            guide_vel = last_motion_vel if last_motion_vel is not None else last_vel
            if _guide_path_consistent(gx, gy, last_pos, guide_vel, frames_since_det, cfg, diag, guide_exact=guide_exact):
                result[t] = FrameResult(
                    cx=gx,
                    cy=gy,
                    conf=0.18,
                    interpolated=False,
                    bbox=None,
                    source='guide',
                    search_cx=float(gx), search_cy=float(gy), search_radius=float(_guide_circle_radius("exact")))
                stats['guide'] += 1
                if lost_counted:
                    stats['lost'] = max(0, stats['lost'] - 1)
                last_pos = (gx, gy)
                last_motion_vel = None
                # Guide is NOT a real detection — count it toward the soft carry budget
                soft_carry_count += 1
                if static_lock_streak > 0:
                    static_lock_streak = max(0, static_lock_streak - 1)
                    if static_lock_streak == 0:
                        static_lock_center = None

        # Attach per-frame guide detection gate metadata so the guide debug renderer
        # can show the actual circle used during exact/frozen guide matching.
        if emit_guide_debug_meta and guide is not None and guide_debug_meta is not None:
            guide_debug_meta[t] = (
                float(gx),
                float(gy),
                bool(guide_exact),
                bool(frozen_guide_active and not guide_exact),
                bool(hold_guide_active and not guide_exact),
                float(_guide_circle_radius()),
            )

        if frame_audit is not None:
            if best_det is not None:
                frame_audit["best_det"] = {
                    "stage": best_det_stage,
                    "cx": float(best_det.cx),
                    "cy": float(best_det.cy),
                    "conf": float(best_det.conf),
                    "on_motion": bool(best_det.on_motion),
                }
            rr = result[t]
            if rr is not None:
                frame_audit["result_source"] = str(rr.source)
                frame_audit["result_xy"] = [float(rr.cx), float(rr.cy)]
            else:
                frame_audit["result_source"] = "none"
            frame_audit["frames_since_det_out"] = int(frames_since_det)
            audit_rows.append(frame_audit)
    
    # Interpolation: fill 1-3 frame micro-gaps between YOLO detections.
    # Use a quadratic Lagrange fit through 3 surrounding det points (prev_prev,
    # prev, curr) or (prev, curr, next_next) so the gap traces the arc of the
    # ball's flight instead of a chord. Fall back to linear when only 2 anchors
    # exist or when the quadratic overshoots the chord by an implausible margin.
    obs_indices = [i for i in range(total_frames) if result[i] is not None
                   and result[i].source == 'det']

    def _lagrange3_eval(t_val: float, t1: float, y1: float,
                        t2: float, y2: float, t3: float, y3: float) -> float:
        d12 = (t1 - t2)
        d13 = (t1 - t3)
        d23 = (t2 - t3)
        denom = d12 * d13 * (-d23)
        if abs(denom) < 1e-9:
            # Degenerate spacing — fall back to linear endpoints.
            return y1 + (y3 - y1) * ((t_val - t1) / max(t3 - t1, 1e-9))
        L1 = ((t_val - t2) * (t_val - t3)) / (d12 * d13)
        L2 = ((t_val - t1) * (t_val - t3)) / ((-d12) * d23)
        L3 = ((t_val - t1) * (t_val - t2)) / ((-d13) * (-d23))
        return L1 * y1 + L2 * y2 + L3 * y3

    for k in range(1, len(obs_indices)):
        prev_i = obs_indices[k - 1]
        curr_i = obs_indices[k]
        gap = curr_i - prev_i
        if gap <= 1 or gap > 4:
            continue
        prev_r = result[prev_i]
        curr_r = result[curr_i]

        # Pick a third anchor: prefer prev_prev (earlier context) if close enough,
        # otherwise next_next. None => linear fallback.
        anchor_idx = None
        if k - 2 >= 0:
            cand = obs_indices[k - 2]
            if prev_i - cand <= 6:
                anchor_idx = cand
        if anchor_idx is None and k + 1 < len(obs_indices):
            cand = obs_indices[k + 1]
            if cand - curr_i <= 6:
                anchor_idx = cand
        anchor_r = result[anchor_idx] if anchor_idx is not None else None

        for f in range(prev_i + 1, curr_i):
            if result[f] is None:
                t_frac = (f - prev_i) / gap
                # Linear baseline
                lin_x = prev_r.cx + (curr_r.cx - prev_r.cx) * t_frac
                lin_y = prev_r.cy + (curr_r.cy - prev_r.cy) * t_frac
                cx_interp, cy_interp = lin_x, lin_y
                if anchor_r is not None:
                    qx = _lagrange3_eval(
                        float(f),
                        float(anchor_idx), float(anchor_r.cx),
                        float(prev_i), float(prev_r.cx),
                        float(curr_i), float(curr_r.cx),
                    )
                    qy = _lagrange3_eval(
                        float(f),
                        float(anchor_idx), float(anchor_r.cy),
                        float(prev_i), float(prev_r.cy),
                        float(curr_i), float(curr_r.cy),
                    )
                    # Sanity bound: keep the quadratic correction within
                    # ~max(0.6 * chord, 30 px) of the linear point. Beyond that
                    # the parabola is being driven by a far-off anchor and is
                    # less reliable than the chord.
                    chord = math.hypot(curr_r.cx - prev_r.cx, curr_r.cy - prev_r.cy)
                    bound = max(30.0, 0.6 * chord)
                    if math.hypot(qx - lin_x, qy - lin_y) <= bound:
                        cx_interp, cy_interp = qx, qy
                r_interp = float(getattr(prev_r, "search_radius", 0.0))
                r_end = float(getattr(curr_r, "search_radius", 0.0))
                r_interp = r_interp + (r_end - r_interp) * t_frac
                result[f] = FrameResult(
                    cx=cx_interp,
                    cy=cy_interp,
                    conf=min(prev_r.conf, curr_r.conf) * 0.7,
                    interpolated=True, bbox=None, source='interp',
                    search_cx=cx_interp, search_cy=cy_interp, search_radius=r_interp)
                if emit_guide_debug_meta and guide_debug_meta is not None and f < len(guide_debug_meta):
                    guide_debug_meta[f] = (float(cx_interp), float(cy_interp), False, False, False, float(r_interp))

    if debug:
        filled = sum(1 for r in result if r is not None)
        print(f"[selector] Per-frame: det={stats['det']} motion={stats['motion']} "
              f"guide={stats['guide']} carry={stats['carry']} "
              f"lost={stats['lost']} | "
              f"{filled}/{total_frames} filled")
        print(f"[selector] Physics: gravity(base={base_gravity:.3f}, adapt={adaptive_gravity:.3f}) px/f², "
              f"drag={cfg.gravity_drag_factor}, enabled={cfg.gravity_enabled}, "
              f"bounces={stats['bounce']}")
        if stats['rej_reacquire_dist'] or stats['rej_reacquire_size']:
            print(f"[selector] Reacquire rejects: dist={stats['rej_reacquire_dist']} "
                  f"size={stats['rej_reacquire_size']}")
        if stats['rej_context'] or stats['rej_blocked_radius'] or stats['rej_hard_step']:
            print(f"[selector] Context rejects: outside/far={stats['rej_context']} "
                  f"blocked_radius={stats['rej_blocked_radius']} "
                  f"hard_step={stats['rej_hard_step']}")
        if stats['rej_other_track']:
            print(f"[selector] Cross-track rejects: {stats['rej_other_track']}")
        if stats['owner_soft_pen']:
            print(f"[selector] Cross-track soft penalties: {stats['owner_soft_pen']}")
        if stats['rej_motion_area'] or stats['rej_motion_physics'] or stats['rej_motion_jump'] or stats['rej_motion_player']:
            print(f"[selector] Motion rejects: area={stats['rej_motion_area']} "
                  f"physics={stats['rej_motion_physics']} "
                  f"jump={stats['rej_motion_jump']} "
                  f"player={stats['rej_motion_player']}")
        if stats['rej_static_snap']:
            print(f"[selector] Static snap rejects: {stats['rej_static_snap']}")
        if stats['motion_guided_clamped']:
            print(f"[selector] Motion guided clamps: {stats['motion_guided_clamped']}")

    if audit_enabled:
        if not audit_path:
            audit_path = (
                f"output_videos/_inspection/selector_audit_{int(audit_start)}_{int(audit_end)}.jsonl"
            )
        audit_dir = os.path.dirname(audit_path)
        if audit_dir:
            os.makedirs(audit_dir, exist_ok=True)
        meta = {
            "type": "meta",
            "audit_start": int(audit_start),
            "audit_end": int(audit_end),
            "total_frames": int(total_frames),
            "chosen_track_id": int(chosen.track_id) if chosen is not None else None,
            "chosen_track_score": float(chosen.score) if chosen is not None else None,
            "tracks": audit_track_rows,
        }
        with open(audit_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(meta, ensure_ascii=True) + "\n")
            for row in audit_rows:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")
        print(f"[selector] Audit written: {audit_path} ({len(audit_rows)} frames)")

    if emit_guide_debug_meta and guide_debug_meta is not None:
        for i, gmeta in enumerate(guide_debug_meta):
            if gmeta is None:
                continue
            rr_dbg = result[i]
            if rr_dbg is None:
                rr_dbg = FrameResult(source='debug', debug_only=True)
                result[i] = rr_dbg
            (gsx, gsy, gexact_dbg, gfrozen_dbg, ghold_dbg, gsr) = gmeta
            rr_dbg.guide_search_cx = float(gsx)
            rr_dbg.guide_search_cy = float(gsy)
            rr_dbg.guide_search_exact = bool(gexact_dbg)
            rr_dbg.guide_search_frozen = bool(gfrozen_dbg)
            rr_dbg.guide_search_hold = bool(ghold_dbg)
            rr_dbg.guide_search_radius = float(gsr)

    return result, chosen, tracks, motion_tracks
