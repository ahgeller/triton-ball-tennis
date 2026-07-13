import math
import cv2
from typing import Optional, List, Tuple, Dict

from .config import SelectorConfig
from .models import Detection, Track
from .utils import _cfg_diag, _fps_norm_pxpf
from .physics import _kinematic_motion_frac, _xy_dist


def _track_kinematics(trk: Track) -> Tuple[List[Tuple[float, float]], List[float], float, float]:
    """Return cached (velocities, speeds, avg_speed, peak_speed) for a track."""
    obs = trk.observations
    if len(obs) < 2:
        trk._kin_cache_key = (len(obs),)
        trk._cached_velocities = []
        trk._cached_speeds = []
        trk._cached_speed_stats = (0.0, 0.0)
        return trk._cached_velocities, trk._cached_speeds, 0.0, 0.0

    key = (
        len(obs),
        int(obs[0].frame),
        int(obs[-1].frame),
        float(obs[-1].cx),
        float(obs[-1].cy),
    )
    if trk._kin_cache_key == key:
        avg_speed, peak_speed = trk._cached_speed_stats
        return trk._cached_velocities, trk._cached_speeds, avg_speed, peak_speed

    velocities: List[Tuple[float, float]] = []
    speeds: List[float] = []
    total_dist = 0.0
    total_dt = 0
    peak = 0.0
    for i in range(1, len(obs)):
        dt = int(obs[i].frame - obs[i - 1].frame)
        if dt <= 0:
            continue
        dx = float(obs[i].cx - obs[i - 1].cx)
        dy = float(obs[i].cy - obs[i - 1].cy)
        vx = dx / dt
        vy = dy / dt
        dist = math.sqrt(dx * dx + dy * dy)
        spd = dist / dt
        velocities.append((vx, vy))
        speeds.append(spd)
        total_dist += dist
        total_dt += dt
        if spd > peak:
            peak = spd

    avg = total_dist / max(total_dt, 1)
    trk._kin_cache_key = key
    trk._cached_velocities = velocities
    trk._cached_speeds = speeds
    trk._cached_speed_stats = (avg, peak)
    return velocities, speeds, avg, peak

def _track_speed_stats(trk: Track) -> Tuple[float, float]:
    """Returns (avg_speed, peak_speed) in px/frame."""
    _, _, avg, peak = _track_kinematics(trk)
    return avg, peak

def _track_extent(trk: Track) -> float:
    """Spatial extent (diag of bbox containing all obs centers)."""
    if trk.num_obs < 2:
        return 0.0
    xs = [o.cx for o in trk.observations]
    ys = [o.cy for o in trk.observations]
    return math.sqrt((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2)

def _is_stationary_track(trk: Track, cfg: SelectorConfig, diag: float) -> bool:
    """Hard filter for parked/static balls so they never enter final matching."""
    if trk.num_obs < cfg.stationary_exclude_min_obs:
        return False

    sb = trk.score_breakdown if trk.score_breakdown else {}
    avg_speed_fallback, peak_speed_fallback = _track_speed_stats(trk)
    avg_speed = float(sb.get("avg_speed_pxpf", avg_speed_fallback))
    peak_speed = float(sb.get("peak_speed_pxpf", peak_speed_fallback))
    extent_px = float(sb.get("extent_px", _track_extent(trk)))
    if "motion_frac" in sb:
        motion_frac = float(sb["motion_frac"])
    else:
        motion_frac = sum(1 for o in trk.observations if o.on_motion) / max(trk.num_obs, 1)

    max_avg_speed = _fps_norm_pxpf(cfg.stationary_exclude_max_avg_speed, cfg)
    max_peak_speed = _fps_norm_pxpf(cfg.stationary_exclude_max_peak_speed, cfg)

    if avg_speed > max_avg_speed:
        return False
    if peak_speed > max_peak_speed:
        return False
    if extent_px > cfg.stationary_exclude_max_extent_frac * diag:
        return False
    if motion_frac > cfg.stationary_exclude_max_motion_frac:
        return False
    return True

def _is_near_player(x, y, player_boxes_by_frame, frame_idx, margin=80):
    """Check if (x,y) is near any player bbox at given frame."""
    if player_boxes_by_frame is None or frame_idx >= len(player_boxes_by_frame):
        return False
    pboxes = player_boxes_by_frame[frame_idx]
    if pboxes is None:
        return False
    # pboxes can be a dict (slot->bbox) or list of bboxes
    boxes = pboxes.values() if isinstance(pboxes, dict) else pboxes
    for pb in boxes:
        if pb is None or len(pb) < 4:
            continue
        # Expand player bbox upward generously (serve toss above head)
        px1, py1, px2, py2 = float(pb[0]), float(pb[1]), float(pb[2]), float(pb[3])
        ph = py2 - py1
        if (px1 - margin <= x <= px2 + margin and
                py1 - ph * 2 - margin <= y <= py2 + margin):
            return True
    return False

def _closest_player_distance(x, y, player_boxes_by_frame, frame_idx, margin=20):
    """Distance from point to nearest (expanded) player bbox edge; 0 if inside."""
    if player_boxes_by_frame is None or frame_idx >= len(player_boxes_by_frame):
        return None
    pboxes = player_boxes_by_frame[frame_idx]
    if pboxes is None:
        return None
    boxes = pboxes.values() if isinstance(pboxes, dict) else pboxes
    best = None
    for pb in boxes:
        if pb is None or len(pb) < 4:
            continue
        x1, y1, x2, y2 = map(float, pb[:4])
        x1 -= margin
        y1 -= margin
        x2 += margin
        y2 += margin
        dx = max(x1 - x, 0.0, x - x2)
        dy = max(y1 - y, 0.0, y - y2)
        d = math.sqrt(dx * dx + dy * dy)
        if best is None or d < best:
            best = d
    return best

def _track_movement_score(trk: Track, cfg: SelectorConfig, diag: float) -> float:
    """Movement score used for track-level merge eligibility."""
    sb = trk.score_breakdown if trk.score_breakdown else {}
    if "movement" in sb:
        return float(sb["movement"])

    speed_ref = max(0.010 * diag, 1.0)
    peak_ref = max(0.022 * diag, 1.0)
    extent_ref = max(0.18 * diag, 1.0)
    avg_speed, peak_speed = _track_speed_stats(trk)
    extent = _track_extent(trk)
    motion_frac = sum(1 for o in trk.observations if o.on_motion) / max(trk.num_obs, 1)
    motion_smooth = _motion_consistency(trk)

    speed_norm = min(avg_speed / speed_ref, 1.0)
    peak_norm = min(peak_speed / peak_ref, 1.0)
    extent_norm = min(extent / extent_ref, 1.0)
    movement_quality = (
        0.34 * speed_norm +
        0.18 * peak_norm +
        0.28 * extent_norm +
        0.15 * motion_smooth +
        0.05 * motion_frac
    )
    return cfg.w_movement * movement_quality

def _annotate_track_periods(
    tracks: List[Track],
    total_frames: int,
    cfg: SelectorConfig
) -> None:
    """Attach temporal-period metadata to each track for cut-aware debugging."""
    if not tracks:
        return

    denom = max(total_frames - 1, 1)
    split_gap = max(1, int(round(max(0.0, cfg.period_split_gap_frac) * max(total_frames, 1))))

    ordered = sorted(tracks, key=lambda t: (t.first_frame, t.last_obs_frame))
    period_id = 0
    period_end = ordered[0].last_obs_frame

    for trk in ordered:
        start_f = int(trk.first_frame)
        end_f = int(trk.last_obs_frame)

        if start_f - period_end > split_gap:
            period_id += 1
            period_end = end_f
        else:
            period_end = max(period_end, end_f)

        sb = trk.score_breakdown if trk.score_breakdown else {}
        sb["period_id"] = float(period_id)
        sb["start_frame"] = float(start_f)
        sb["end_frame"] = float(end_f)
        sb["start_frac"] = start_f / float(denom)
        sb["end_frac"] = end_f / float(denom)
        trk.score_breakdown = sb

def _select_timeline_chain(
    tracks: List[Track],
    cfg: SelectorConfig,
    total_frames: int
) -> List[Track]:
    """Pick an auto timeline of tracks maximizing score with time-progression continuity."""
    if not tracks:
        return []

    min_score = max(0.0, float(cfg.timeline_min_track_score))
    eligible = [t for t in tracks if float(t.score) >= min_score]
    if not eligible:
        return []
    if len(eligible) == 1:
        return [eligible[0]]

    diag = _cfg_diag(cfg)
    tden = max(total_frames, 1)
    max_gap = max(1, int(round(max(0.0, cfg.timeline_max_gap_frac) * tden)))
    overlap_tol = max(0, int(round(max(0.0, cfg.timeline_overlap_tol_frac) * tden)))
    small_gap = max(0, int(round(max(0.0, cfg.timeline_small_gap_frac) * tden)))
    jump_reject = max(0.0, float(cfg.timeline_small_gap_jump_reject_frac))
    medium_gap = max(small_gap + 1, int(round(small_gap * 2.0)))

    ordered = sorted(eligible, key=lambda t: (t.first_frame, t.last_obs_frame, -t.score))
    n = len(ordered)
    cov_w = max(0.0, float(cfg.timeline_coverage_weight))
    score_w = max(0.0, float(cfg.timeline_score_weight))
    base = []
    for t in ordered:
        span_frames = max(int(t.last_obs_frame) - int(t.first_frame) + 1, 1)
        span_frac = span_frames / float(tden)
        base_score = max(0.0, float(t.score))
        base.append(cov_w * span_frac + score_w * base_score)
    dp = base[:]
    prev = [-1] * n

    for j in range(n):
        tj = ordered[j]
        sj = int(tj.first_frame)
        pj = tj.score_breakdown if tj.score_breakdown else {}
        pjid = int(float(pj.get("period_id", 0.0)))
        j_start = tj.observations[0]

        for i in range(j):
            ti = ordered[i]
            si = int(ti.first_frame)
            ei = int(ti.last_obs_frame)
            if sj <= si:
                continue

            gap = sj - ei - 1
            if gap > max_gap:
                continue
            if gap < -overlap_tol:
                continue

            i_end = ti.observations[-1]
            dt = max(gap + 1, 1)
            shared_frames = set(obs.frame for obs in ti.observations) & set(
                obs.frame for obs in tj.observations
            )
            if gap < 0 and shared_frames:
                ti_by_frame = {obs.frame: obs for obs in ti.observations}
                tj_by_frame = {obs.frame: obs for obs in tj.observations}
                shared_distances = sorted(
                    _xy_dist(
                        ti_by_frame[frame].cx,
                        ti_by_frame[frame].cy,
                        tj_by_frame[frame].cx,
                        tj_by_frame[frame].cy,
                    )
                    for frame in shared_frames
                )
                d = shared_distances[len(shared_distances) // 2]
                if d > max(12.0, 0.08 * diag):
                    continue
            else:
                pred_x = i_end.cx + ti.last_vel[0] * dt
                pred_y = i_end.cy + ti.last_vel[1] * dt
                d_raw = _xy_dist(i_end.cx, i_end.cy, j_start.cx, j_start.cy)
                d_pred = _xy_dist(pred_x, pred_y, j_start.cx, j_start.cy)
                d = min(d_raw, d_pred)
            dfrac = d / max(diag, 1.0)

            hard_discontinuity = (
                (gap <= small_gap and dfrac > jump_reject)
                or (gap <= medium_gap and dfrac > (jump_reject + 0.08))
                or dfrac > 0.70
            )
            # Simultaneous distant tracks are alternatives. Sequential distant
            # tracks are separate rallies/clips: keep both, but pay enough that
            # short noise cannot create a reset.
            if hard_discontinuity and gap < 0:
                continue

            if hard_discontinuity:
                trans = -6.0 * float(cfg.timeline_switch_penalty)
            else:
                trans = -float(cfg.timeline_switch_penalty)
                if gap < 0:
                    trans -= float(cfg.timeline_overlap_penalty) * (abs(gap) / float(tden))
                trans -= float(cfg.timeline_jump_penalty) * min(dfrac, 1.5)
            if gap > 0:
                trans -= float(cfg.timeline_gap_penalty) * (gap / float(tden))

            pi = ti.score_breakdown if ti.score_breakdown else {}
            piid = int(float(pi.get("period_id", 0.0)))
            if piid != pjid:
                trans -= float(cfg.timeline_period_switch_penalty)

            # -- Velocity-continuity penalty --
            # If the outgoing velocity of track i and incoming velocity of track j
            # are directionally inconsistent, penalize the transition. This catches
            # cases where the DP picks a path through an unrelated static or
            # crossing detection cluster just because it's spatially nearby.
            exp_vx, exp_vy = ti.last_vel
            exp_speed = math.sqrt(exp_vx * exp_vx + exp_vy * exp_vy)
            obs_vx = (j_start.cx - i_end.cx) / max(dt, 1)
            obs_vy = (j_start.cy - i_end.cy) / max(dt, 1)
            obs_speed = math.sqrt(obs_vx * obs_vx + obs_vy * obs_vy)
            fps_ref_speed = max(diag * 0.015, 2.0)  # ~1% diag/frame = meaningful motion
            if (
                not hard_discontinuity
                and gap >= 0
                and exp_speed > fps_ref_speed
                and obs_speed > fps_ref_speed
            ):
                dot = obs_vx * exp_vx + obs_vy * exp_vy
                cos_sim = dot / max(obs_speed * exp_speed, 1e-6)
                # Removed direction reversal penalties because racket hits and bounces reverse direction (cos_sim -> -1.0)
                # Also penalize large speed-ratio changes that suggest different balls
                speed_ratio = obs_speed / max(exp_speed, 1e-6)
                if speed_ratio > 3.5 or speed_ratio < 0.15:
                    trans -= float(cfg.timeline_jump_penalty) * 0.5

            # -- Tiny-bridge penalty --
            # A track with very few observations spanning a tiny window that creates
            # a large dead zone after it (the next viable track is far away) is likely
            # noise. Apply a small penalty proportional to how tiny the bridge is
            # relative to the gap it opens up on the far side.
            tj_span = max(int(tj.last_obs_frame) - int(tj.first_frame) + 1, 1)
            if tj_span < max(6, int(round(0.005 * tden))) and tj.num_obs < 10:
                trans -= float(cfg.timeline_switch_penalty) * 0.4

            cand = dp[i] + base[j] + trans
            if cand > dp[j]:
                dp[j] = cand
                prev[j] = i

    end_idx = max(range(n), key=lambda k: dp[k])
    idx_chain = []
    k = end_idx
    while k >= 0:
        idx_chain.append(k)
        k = prev[k]
    idx_chain.reverse()

    chain = [ordered[k] for k in idx_chain]
    if not chain:
        return [max(ordered, key=lambda t: t.score)]
    return chain

def _stitch_track_chain(chain: List[Track], cfg: SelectorConfig) -> Optional[Track]:
    """Convert a timeline chain into one stitched track for guide building."""
    if not chain:
        return None
    if len(chain) == 1:
        return chain[0]

    kept_chain: List[Track] = []
    for trk in chain:
        if not kept_chain:
            kept_chain.append(trk)
            continue
        # Keep timeline segments selected by the DP chain to maximize game coverage.
        kept_chain.append(trk)

    if not kept_chain:
        return chain[0]
    if len(kept_chain) == 1:
        return kept_chain[0]

    stitched = Track(track_id=chain[0].track_id, cfg=cfg)
    best_obs_by_frame: Dict[int, Detection] = {}
    for trk in kept_chain:
        for obs in trk.observations:
            prev = best_obs_by_frame.get(obs.frame)
            if prev is None or obs.conf > prev.conf:
                best_obs_by_frame[obs.frame] = obs

    obs = [best_obs_by_frame[f] for f in sorted(best_obs_by_frame.keys())]
    if len(obs) < 2:
        return chain[0]

    stitched.observations = obs
    stitched.last_pos = (obs[-1].cx, obs[-1].cy)
    stitched.last_frame = obs[-1].frame
    prev = obs[-2]
    dt = max(obs[-1].frame - prev.frame, 1)
    stitched.last_vel = ((obs[-1].cx - prev.cx) / dt, (obs[-1].cy - prev.cy) / dt)
    stitched.score = sum(float(t.score) for t in kept_chain)

    sb0 = kept_chain[0].score_breakdown if kept_chain[0].score_breakdown else {}
    sbn = kept_chain[-1].score_breakdown if kept_chain[-1].score_breakdown else {}
    stitched.score_breakdown = {
        "total": stitched.score,
        "timeline_tracks": float(len(kept_chain)),
        "start_frame": float(kept_chain[0].first_frame),
        "end_frame": float(kept_chain[-1].last_obs_frame),
        "start_frac": float(sb0.get("start_frac", 0.0)),
        "end_frac": float(sbn.get("end_frac", 1.0)),
        "period_id": float(sb0.get("period_id", 0.0)),
    }
    return stitched

def _court_fractions(track: Track, court_poly, expand_px: float):
    """Compute court occupancy fractions.

    Returns:
      near_court_frac: inside polygon or within expand_px margin (used for scoring).
      far_outside_frac: farther than 2x expand_px outside polygon.
      strict_inside_frac: strictly inside polygon (dist >= 0).
    """
    if court_poly is None or len(track.observations) == 0:
        return 0.5, 0.5, 0.5  # neutral

    near_court = 0
    far_outside = 0
    strict_inside = 0
    for obs in track.observations:
        dist = cv2.pointPolygonTest(court_poly, (float(obs.cx), float(obs.cy)), True)
        # dist > 0 = inside polygon, dist < 0 = outside, magnitude = distance to edge
        if dist >= 0.0:
            strict_inside += 1
        if dist >= -expand_px:  # inside or within margin
            near_court += 1
        elif -dist > expand_px * 2:  # really far
            far_outside += 1

    n = len(track.observations)
    return near_court / n, far_outside / n, strict_inside / n

def _motion_consistency(track: Track) -> float:
    """Score 0-1 for how smooth/consistent the track motion is.
    High jitter = low score. Smooth arcs = high score."""
    obs = track.observations
    if len(obs) < 3:
        return 0.5

    velocities, _, _, _ = _track_kinematics(track)

    if len(velocities) < 2:
        return 0.5

    # Measure direction changes (angle between consecutive velocity vectors)
    angle_changes = []
    for i in range(1, len(velocities)):
        v1 = velocities[i-1]
        v2 = velocities[i]
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
        if mag1 < 0.5 or mag2 < 0.5:
            continue  # near-stationary, skip
        cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2 + 1e-9)))
        angle_changes.append(abs(math.acos(cos_angle)))

    if not angle_changes:
        return 0.5

    # Penalize frequent sharp direction changes (>90 deg)
    # Tennis ball has smooth arcs with direction changes at bounces/hits
    # But random jitter has constant tiny direction changes
    sharp = sum(1 for a in angle_changes if a > math.pi * 0.6)
    smooth_frac = 1.0 - sharp / len(angle_changes)
    return max(0.0, min(1.0, smooth_frac))

def _jump_penalty(track: Track) -> float:
    """Sum of unusually large jumps between consecutive observations."""
    obs = track.observations
    if len(obs) < 2:
        return 0.0
    penalty = 0.0
    _, speeds, _, _ = _track_kinematics(track)

    if not speeds:
        return 0.0

    # Penalize jumps above 2x median speed
    median_speed = sorted(speeds)[len(speeds) // 2]
    threshold = max(median_speed * 2.5, 5.0)
    for s in speeds:
        if s > threshold:
            penalty += (s - threshold)
    return penalty

def score_tracks(
    tracks: List[Track],
    cfg: SelectorConfig,
    court_poly=None,
    player_boxes_by_frame=None,
    total_frames: int = 0
) -> List[Track]:
    """Score each track, sort by score descending.

    Positive-only scoring:
      base_add = obs_add + span_add
      score = base_add * inside_mult * motion_mult * near_player_mult

    with:
      - obs_add increases with observations
      - span_add increases with temporal span
      - inside_mult = in% (strict-inside court fraction)
      - motion_mult = mot% (motion overlap fraction)
      - near_player_mult = weaker multiplier from nearP%
    """
    if not tracks:
        return tracks

    for trk in tracks:
        obs = trk.observations
        n = trk.num_obs
        span = trk.span

        coverage_frac = (n / max(total_frames, 1)) if total_frames > 0 else 0.0
        span_frac = (span / max(total_frames, 1)) if total_frames > 0 else 0.0
        min_obs = max(cfg.min_track_obs_abs,
                      int(cfg.min_track_obs_frac * max(total_frames, 1)))

        inside_frac, _, strict_inside_frac = _court_fractions(
            trk, court_poly, cfg.court_expand_px
        )
        motion_frac_raw = sum(1 for o in obs if o.on_motion) / max(n, 1)

        near_player_frac = 0.0
        near_player_court_frac = 0.0
        if player_boxes_by_frame is not None and n > 0:
            near_player_count = 0
            near_player_court_count = 0
            for o in obs:
                near_player = _is_near_player(
                    o.cx, o.cy, player_boxes_by_frame, o.frame, margin=60
                )
                if near_player:
                    near_player_count += 1
                    if court_poly is None:
                        near_player_court_count += 1
                    else:
                        cdist = cv2.pointPolygonTest(
                            court_poly, (float(o.cx), float(o.cy)), True
                        )
                        if cdist >= -cfg.court_expand_px:
                            near_player_court_count += 1
            near_player_frac = near_player_count / max(n, 1)
            near_player_court_frac = near_player_court_count / max(n, 1)

        avg_speed, peak_speed = _track_speed_stats(trk)
        extent = _track_extent(trk)
        jump_penalty_raw = _jump_penalty(trk)
        motion_smooth = _motion_consistency(trk)
        motion_kin_frac = _kinematic_motion_frac(
            avg_speed, peak_speed, extent, near_player_frac, cfg, _cfg_diag(cfg)
        )
        motion_frac = max(motion_frac_raw, motion_kin_frac)

        # Positive additive terms requested by user.
        obs_add = cfg.w_len * n
        span_add = cfg.w_span * span_frac
        base_add = obs_add + span_add

        # Multipliers in [0, 1], with near-player intentionally weaker.
        # Use near_court_frac (inside OR within margin) instead of strict_inside_frac
        # so near-edge detections don't destroy track scores.
        inside_mult = max(0.0, min(1.0, inside_frac))
        motion_mult = max(0.0, min(1.0, motion_frac))
        near_player_mult = 0.70 + 0.30 * max(0.0, min(1.0, near_player_frac))

        raw_total_score = base_add * inside_mult * motion_mult * near_player_mult
        total_score = max(0.0, raw_total_score)

        trk.score = total_score
        trk.score_breakdown = {
            "total": float(total_score),
            "total_raw": float(raw_total_score),
            "score_clamped_nonnegative": 0.0,
            "num_obs": n,
            "span": span,
            "inside_frac": float(inside_frac),
            "inside_strict_frac": float(strict_inside_frac),
            "redflag_outside_long_bool": 0.0,
            "motion_frac": float(motion_frac),
            "motion_frac_raw": float(motion_frac_raw),
            "motion_kin_frac": float(motion_kin_frac),
            "avg_speed_pxpf": float(avg_speed),
            "peak_speed_pxpf": float(peak_speed),
            "extent_px": float(extent),
            "movement": float(raw_total_score),
            "jump_penalty": float(jump_penalty_raw),
            "near_player_frac": float(near_player_frac),
            "near_player_court_frac": float(near_player_court_frac),
            "coverage_frac": float(coverage_frac),
            "span_frac": float(span_frac),
            "min_obs_required": float(min_obs),
            "start_frame": float(trk.first_frame),
            "end_frame": float(trk.last_obs_frame),
            "start_frac": (trk.first_frame / float(max(total_frames - 1, 1))) if total_frames > 0 else 0.0,
            "end_frac": (trk.last_obs_frame / float(max(total_frames - 1, 1))) if total_frames > 0 else 0.0,
            "obs_add": float(obs_add),
            "span_add": float(span_add),
            "additive_core": float(base_add),
            "motion_mult": float(motion_mult),
            "inside_mult": float(inside_mult),
            "near_player_mult": float(near_player_mult),
            "multiplied_core": float(raw_total_score),
            "movement_quality": float(motion_smooth),
        }

    tracks.sort(key=lambda t: t.score, reverse=True)
    return tracks

def _passes_sanity(trk: Track, cfg: SelectorConfig) -> bool:
    """Reject tracks that are clearly junk.

    The scoring should do the heavy lifting.
    """
    if trk.num_obs < 3:
        return False
    sb = trk.score_breakdown if trk.score_breakdown else {}
    avg_speed = float(sb.get("avg_speed_pxpf", 0.0))
    extent_px = float(sb.get("extent_px", 0.0))
    inside_frac = float(sb.get("inside_frac", 0.5))
    motion_frac = float(sb.get("motion_frac", 0.0))
    near_player_frac = float(sb.get("near_player_frac", 0.0))
    near_player_court_frac = float(sb.get("near_player_court_frac", 0.0))
    min_obs_required = int(float(sb.get("min_obs_required", 0.0)))
    redflag_outside_long = float(sb.get("redflag_outside_long_bool", 0.0)) > 0.5
    diag = _cfg_diag(cfg)

    very_static_thresh = _fps_norm_pxpf(1.1, cfg)
    weak_motion_thresh = _fps_norm_pxpf(1.6, cfg)

    # Hard reject very static compact tracks (typical parked balls).
    if avg_speed < very_static_thresh and extent_px < 0.05 * diag:
        return False

    # Hard reject tracks mostly outside court unless they clearly interact with players.
    if inside_frac < 0.10 and near_player_frac < 0.12:
        return False
    # Hard reject long tracks that are effectively never on court.
    if redflag_outside_long:
        return False

    # Reject tracks that almost never satisfy the intended context.
    if near_player_court_frac < 0.02 and inside_frac < 0.35 and near_player_frac < 0.12:
        return False

    # Reject weak/no-motion/no-player context tracks.
    if motion_frac < 0.04 and avg_speed < weak_motion_thresh and near_player_frac < 0.08:
        return False

    # In longer videos, avoid selecting tiny late/early bursts as the main track.
    if min_obs_required > 0 and trk.num_obs < min_obs_required:
        if inside_frac < 0.85 or near_player_court_frac < 0.35:
            return False
    return True

def select_best_track(
    tracks: List[Track],
    cfg: SelectorConfig
) -> Optional[Track]:
    """Pick the single highest-score track (test mode)."""
    if not tracks:
        return None
    return max(tracks, key=lambda t: float(t.score))
