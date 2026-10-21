from typing import Dict, Optional, Tuple
import cv2
import numpy as np
from .config import Config
from .court_geometry import build_court_side_guides, clip_infinite_line_to_frame, homography_apply

# Trail rendering constants
COLOR_DET = (0, 255, 0)
COLOR_MOTION = (0, 165, 255)
COLOR_PHYSICS = (255, 255, 0)
COLOR_INTERP = COLOR_PHYSICS
COLOR_CARRY = COLOR_PHYSICS
COLOR_RAW = (0, 180, 180)
COLOR_SEARCH = (255, 0, 255)
COLOR_GUIDE = (255, 255, 255)
COLOR_GUIDE_INTERP = (180, 220, 255)
COLOR_GAP = (0, 0, 0)
GAP_END_TRIM_PX = 6
ENABLE_GAP_CONNECTORS = False

_SOURCE_BASE_COLOR: Dict[str, Tuple[int, int, int]] = {
    "det": (40, 255, 40),
    "motion": (0, 185, 255),
    "carry": COLOR_PHYSICS,
    "interp": COLOR_PHYSICS,
    "guide": COLOR_GUIDE,
    "gap": COLOR_GAP,
}


def draw_court_side_guides(frame, court_keypoints, color=(0, 0, 255), thickness=2):
    """Draw offset side guides: base horizontal offsets + extended side lines."""
    guides = build_court_side_guides(court_keypoints)
    if not guides:
        return frame

    h, w = frame.shape[:2]
    for side in ("left", "right"):
        g = guides.get(side)
        if g is None:
            continue
        base = tuple(np.round(g["base"]).astype(int))
        anchor = tuple(np.round(g["anchor"]).astype(int))
        cv2.line(frame, base, anchor, color, max(1, thickness), cv2.LINE_AA)
        clipped = clip_infinite_line_to_frame(g["anchor"], g["dir"], w, h)
        if clipped is not None:
            cv2.line(frame, clipped[0], clipped[1], color, thickness, cv2.LINE_AA)
    return frame

def _build_display_guide(
    track,
    total_frames: int,
    max_interp_gap: int,
    frame_w: int,
    frame_h: int
) -> dict:
    """Build a per-frame guide map from chosen track for guide-debug rendering."""
    guide = {}
    if track is None or getattr(track, "num_obs", 0) <= 0:
        return guide
    diag = float(np.hypot(float(frame_w), float(frame_h)))
    max_step = max(10.0, 0.15 * diag)
    static_speed = max(2.0, 0.010 * diag)
    max_resid = max(14.0, 0.08 * diag)

    # De-duplicate by frame (keep highest-confidence observation).
    best_by_frame = {}
    for o in track.observations:
        if o is None:
            continue
        f = int(o.frame)
        prev = best_by_frame.get(f)
        if prev is None or float(getattr(o, "conf", 0.0)) > float(getattr(prev, "conf", 0.0)):
            best_by_frame[f] = o

    obs = [best_by_frame[f] for f in sorted(best_by_frame.keys()) if 0 <= f < total_frames]
    if not obs:
        return guide
    if len(obs) >= 3:
        keep = [obs[0]]
        for i in range(1, len(obs) - 1):
            a = keep[-1]
            b = obs[i]
            c = obs[i + 1]
            dt_ab = max(int(b.frame - a.frame), 1)
            dt_bc = max(int(c.frame - b.frame), 1)
            step_ab = float(np.hypot(float(b.cx - a.cx), float(b.cy - a.cy))) / dt_ab
            step_bc = float(np.hypot(float(c.cx - b.cx), float(c.cy - b.cy))) / dt_bc

            jump_to_static = (
                step_ab > max_step and
                step_bc <= static_speed and
                not bool(getattr(b, "on_motion", False))
            )
            if jump_to_static:
                continue

            dt_ac = int(c.frame - a.frame)
            if dt_ac > 0 and dt_ac <= max(2, int(max_interp_gap)):
                t = (float(b.frame) - float(a.frame)) / float(dt_ac)
                ix = float(a.cx) + (float(c.cx) - float(a.cx)) * t
                iy = float(a.cy) + (float(c.cy) - float(a.cy)) * t
                resid = float(np.hypot(float(b.cx - ix), float(b.cy - iy)))
                shortcut = float(np.hypot(float(c.cx - a.cx), float(c.cy - a.cy))) / dt_ac
                if (step_ab > max_step and step_bc > max_step and
                        resid > max_resid and shortcut < 0.70 * max(step_ab, step_bc)):
                    continue

            keep.append(b)

        last = obs[-1]
        if int(last.frame) != int(keep[-1].frame):
            keep.append(last)
        obs = keep

    for o in obs:
        guide[int(o.frame)] = (float(o.cx), float(o.cy), True)

    max_gap = max(1, int(max_interp_gap))
    for i in range(1, len(obs)):
        a = obs[i - 1]
        b = obs[i]
        gap = int(b.frame) - int(a.frame)
        if gap <= 1 or gap > max_gap:
            continue
        seg_speed = float(np.hypot(float(b.cx - a.cx), float(b.cy - a.cy))) / max(gap, 1)
        if seg_speed > max_step * 1.10:
            continue
        for f in range(int(a.frame) + 1, int(b.frame)):
            if f in guide:
                continue
            t = (f - float(a.frame)) / float(gap)
            cx = float(a.cx) + (float(b.cx) - float(a.cx)) * t
            cy = float(a.cy) + (float(b.cy) - float(a.cy)) * t
            guide[f] = (cx, cy, False)

    # Conservative tail extension: only when the observed tail is moving.
    if len(obs) >= 2:
        last = obs[-1]
        prev = obs[-2]
        dt_last = max(int(last.frame - prev.frame), 1)
        vx = (float(last.cx) - float(prev.cx)) / dt_last
        vy = (float(last.cy) - float(prev.cy)) / dt_last
        tail_speed = float(np.hypot(vx, vy))
        if tail_speed > static_speed * 1.35 and bool(getattr(last, "on_motion", False) or getattr(prev, "on_motion", False)):
            tail_horizon = min(max_gap, max(2, int(max_interp_gap)))
            end_f = min(total_frames - 1, int(last.frame) + tail_horizon)
            for f in range(int(last.frame) + 1, end_f + 1):
                if f in guide:
                    continue
                dtf = f - int(last.frame)
                guide[f] = (float(last.cx) + vx * dtf, float(last.cy) + vy * dtf, False)

    return guide

def _trail_base_color(src):
    return _SOURCE_BASE_COLOR.get(src, (255, 220, 80))

def _trail_jump_fracs(src, cfg: Config):
    if src == "det":
        return cfg.trail_hard_switch_x_frac, cfg.trail_hard_switch_y_frac
    if src == "motion":
        return min(cfg.trail_hard_switch_x_frac, 0.24), min(cfg.trail_hard_switch_y_frac, 0.24)
    if src == "interp":
        return 0.16, 0.16
    if src == "guide":
        return 0.13, 0.13
    if src == "carry":
        return 0.12, 0.12
    return 0.18, 0.18

def _get_track_color(track_id: int) -> Tuple[int, int, int]:
    """Generate a stable, distinct BGR color from track ID."""
    import hashlib
    h = hashlib.md5(str(track_id).encode()).hexdigest()
    # Use parts of the hash for R, G, B
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    
    # Simple strategy to avoid very dark colors: ensure max component is high
    m = max(r, g, b)
    if m < 100:
        scale = 150.0 / max(m, 1)
        r = min(255, int(r * scale))
        g = min(255, int(g * scale))
        b = min(255, int(b * scale))
        
    return (b, g, r)  # OpenCV uses BGR

def _is_soft_source(src):
    return src in ("interp", "carry", "guide")

def _trail_prev2(trail_list):
    # Return previous and second-previous non-gap trail points.
    prev = None
    prev2 = None
    for j in range(len(trail_list) - 1, -1, -1):
        p = trail_list[j]
        if p is None:
            continue
        if prev is None:
            prev = p
        else:
            prev2 = p
            break
    return prev, prev2

def _trail_direction_break(prev2, prev, curr_xy):
    # Break smoothing segment on strong direction reversal (bounce/hit).
    v1x = prev[0] - prev2[0]
    v1y = prev[1] - prev2[1]
    v2x = curr_xy[0] - prev[0]
    v2y = curr_xy[1] - prev[1]
    m1 = float(np.hypot(v1x, v1y))
    m2 = float(np.hypot(v2x, v2y))
    if m1 < 6.0 or m2 < 6.0:
        return False
    cosang = (v1x * v2x + v1y * v2y) / max(m1 * m2, 1e-6)
    if cosang < -0.32:
        return True
    if np.sign(v1y) != np.sign(v2y) and abs(v1y) > 12.0 and abs(v2y) > 12.0:
        return True
    return False

def _draw_homography_net_line(frame, ground_model):
    """Draw net-bottom line using court-plane homography (v = 0.5)."""
    if frame is None or ground_model is None:
        return
    H = ground_model.get("H_c2i")
    if H is None:
        return
    p_l = homography_apply(H, 0.0, 0.5)
    p_r = homography_apply(H, 1.0, 0.5)
    if p_l is None or p_r is None:
        return

    h, w = frame.shape[:2]
    x1 = int(np.clip(round(p_l[0]), 0, w - 1))
    y1 = int(np.clip(round(p_l[1]), 0, h - 1))
    x2 = int(np.clip(round(p_r[0]), 0, w - 1))
    y2 = int(np.clip(round(p_r[1]), 0, h - 1))
    if abs(x2 - x1) + abs(y2 - y1) < 2:
        return

    # Dark outline + bright green line for visibility.
    cv2.line(frame, (x1, y1), (x2, y2), (15, 15, 15), 4, cv2.LINE_AA)
    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA)

def _print_selector_track_summary(all_tracks, chosen_track, total_frames: int, limit: int = 0):
    """Print all selector tracks and clearly indicate which one was chosen."""
    if not all_tracks:
        print("[selector] Track table: none")
        return

    chosen_id = int(chosen_track.track_id) if chosen_track is not None else None
    ordered = sorted(all_tracks, key=lambda t: float(getattr(t, "score", 0.0)), reverse=True)
    if limit and limit > 0:
        view = ordered[:limit]
    else:
        view = ordered

    print("[selector] Track table (sorted by score)")
    print("  sel  id    score   obs  span   t0%   t1%   in%  mot% nearP%")
    for trk in view:
        sb = trk.score_breakdown if getattr(trk, "score_breakdown", None) else {}
        start_frac = float(sb.get("start_frac", 0.0))
        end_frac = float(sb.get("end_frac", 0.0))
        if total_frames > 1 and (start_frac <= 0.0 and end_frac <= 0.0):
            ff = int(getattr(trk, "first_frame", 0))
            lf = int(getattr(trk, "last_obs_frame", ff))
            start_frac = max(0.0, min(1.0, ff / float(total_frames - 1)))
            end_frac = max(0.0, min(1.0, lf / float(total_frames - 1)))

        inside_frac = float(sb.get("inside_strict_frac", sb.get("inside_frac", -1.0)))
        motion_frac = float(sb.get("motion_frac", -1.0))
        near_player_frac = float(sb.get("near_player_frac", -1.0))
        inside_s = f"{inside_frac * 100:5.1f}" if inside_frac >= 0.0 else "   --"
        motion_s = f"{motion_frac * 100:5.1f}" if motion_frac >= 0.0 else "   --"
        near_s = f"{near_player_frac * 100:5.1f}" if near_player_frac >= 0.0 else "   --"

        sel = "*" if int(getattr(trk, "track_id", -1)) == chosen_id else " "
        print(
            f"  {sel:>3} {int(getattr(trk, 'track_id', -1)):>3} "
            f"{float(getattr(trk, 'score', 0.0)):>8.1f} "
            f"{int(getattr(trk, 'num_obs', 0)):>5} "
            f"{int(getattr(trk, 'span', 0)):>5} "
            f"{start_frac * 100:>5.1f} "
            f"{end_frac * 100:>5.1f} "
            f"{inside_s:>5} {motion_s:>5} {near_s:>6}"
        )

    if limit and limit > 0 and len(ordered) > len(view):
        print(f"[selector] Showing top {len(view)} / {len(ordered)} tracks")

    if chosen_track is None:
        print("[selector] Chosen track: none")
    else:
        sb = chosen_track.score_breakdown if getattr(chosen_track, "score_breakdown", None) else {}
        t0 = float(sb.get("start_frac", 0.0))
        t1 = float(sb.get("end_frac", 0.0))
        if total_frames > 1 and (t0 <= 0.0 and t1 <= 0.0):
            ff = int(getattr(chosen_track, "first_frame", 0))
            lf = int(getattr(chosen_track, "last_obs_frame", ff))
            t0 = max(0.0, min(1.0, ff / float(total_frames - 1)))
            t1 = max(0.0, min(1.0, lf / float(total_frames - 1)))
        print(
            f"[selector] Chosen track: id={int(chosen_track.track_id)} "
            f"score={float(chosen_track.score):.1f} "
            f"obs={int(chosen_track.num_obs)} span={int(chosen_track.span)} "
            f"t={t0 * 100:.1f}%->{t1 * 100:.1f}%"
        )

def _print_timing_summary(
    timing: Dict[str, float],
    total_elapsed: float,
    pass1_frames: int,
    pass2_frames: int,
):
    print("\n[info] Stage timing summary")
    rows = [
        ("init_total", "init", 0),
        ("pass1_total", "pass1 total", pass1_frames),
        ("pass1_aux_detect", "pass1 aux detect", pass1_frames),
        ("pass1_preprocess", "pass1 preprocess", pass1_frames),
        ("pass1_pre_mask_build", "pass1 pre(mask-build)", pass1_frames),
        ("pass1_pre_postmask", "pass1 pre(post/flicker)", pass1_frames),
        ("pass1_pre_cuda_total", "pass1 pre(cuda total)", pass1_frames),
        ("pass1_pre_cuda_gpu_other", "pass1 pre(cuda gpu/other)", pass1_frames),
        ("pass1_pre_cuda_d2h", "pass1 pre(cuda d2h)", pass1_frames),
        ("pass1_pre_cuda_cc", "pass1 pre(cuda cc)", pass1_frames),
        ("pass1_ball_detect", "pass1 ball detect", pass1_frames),
        ("pass1_store", "pass1 store/debug", pass1_frames),
        ("pass1_slide", "pass1 decode/slide", pass1_frames),
        ("selector_total", "selector total", pass1_frames),
        ("selector_build_poly", "selector build poly", pass1_frames),
        ("selector_select", "selector select", pass1_frames),
        ("selector_post", "selector post", pass1_frames),
        ("pass2_total", "pass2 total", pass2_frames),
        ("pass2_read", "pass2 frame read", pass2_frames),
        ("pass2_render", "pass2 render", pass2_frames),
        ("pass2_write_main", "pass2 write main", pass2_frames),
        ("pass2_write_guide", "pass2 write guide", pass2_frames),
        ("pass2_write_debug", "pass2 write debug", pass2_frames),
    ]
    denom_total = max(float(total_elapsed), 1e-9)
    for key, label, nframes in rows:
        sec = float(timing.get(key, 0.0))
        if sec <= 0.0:
            continue
        pct = 100.0 * sec / denom_total
        line = f"[info]   {label:<20} {sec:7.3f}s ({pct:5.1f}%)"
        if nframes > 0:
            line += f", {1000.0 * sec / max(int(nframes), 1):6.2f} ms/frame"
        print(line)

