"""Console summaries printed at the end of a run; nothing here draws."""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

from ball_in_play_selector import Track


# ---------------------------------------------------------------------------
# Console summaries
# ---------------------------------------------------------------------------

def _track_time_fracs(trk: Track, total_frames: int) -> Tuple[float, float]:
    sb = trk.score_breakdown if trk.score_breakdown else {}
    start_frac = float(sb.get("start_frac", 0.0))
    end_frac = float(sb.get("end_frac", 0.0))
    if total_frames > 1 and (start_frac <= 0.0 and end_frac <= 0.0):
        ff = int(trk.first_frame)
        lf = int(trk.last_obs_frame)
        start_frac = max(0.0, min(1.0, ff / float(total_frames - 1)))
        end_frac = max(0.0, min(1.0, lf / float(total_frames - 1)))
    return start_frac, end_frac


def print_selector_track_summary(all_tracks: Sequence[Track], chosen_track: Optional[Track], total_frames: int,
                                 limit: int = 0) -> None:
    """Print all selector tracks and clearly indicate which one was chosen."""
    if not all_tracks:
        print("[selector] Track table: none")
        return

    chosen_id = int(chosen_track.track_id) if chosen_track is not None else None
    ordered = sorted(all_tracks, key=lambda t: float(t.score), reverse=True)
    view = ordered[:limit] if limit and limit > 0 else ordered

    print("[selector] Track table (sorted by score)")
    print("  sel  id    score   obs  span   t0%   t1%   in%  mot% nearP%")
    for trk in view:
        print(_track_row(trk, int(trk.track_id) == chosen_id, total_frames))

    if limit and limit > 0 and len(ordered) > len(view):
        print(f"[selector] Showing top {len(view)} / {len(ordered)} tracks")

    if chosen_track is None:
        print("[selector] Chosen track: none")
    else:
        t0, t1 = _track_time_fracs(chosen_track, total_frames)
        print(
            f"[selector] Chosen track: id={int(chosen_track.track_id)} "
            f"score={float(chosen_track.score):.1f} "
            f"obs={int(chosen_track.num_obs)} span={int(chosen_track.span)} "
            f"t={t0 * 100:.1f}%->{t1 * 100:.1f}%"
        )


def _percent_or_dash(fraction: float) -> str:
    """A 0..1 fraction as a percentage column; a negative one (not scored) as ``--``."""
    return f"{fraction * 100:5.1f}" if fraction >= 0.0 else "   --"


def _track_row(trk: Track, chosen: bool, total_frames: int) -> str:
    """One line of the track table."""
    sb = trk.score_breakdown if trk.score_breakdown else {}
    start_frac, end_frac = _track_time_fracs(trk, total_frames)
    inside_s = _percent_or_dash(float(sb.get("inside_strict_frac", sb.get("inside_frac", -1.0))))
    motion_s = _percent_or_dash(float(sb.get("motion_evidence", -1.0)))
    near_s = _percent_or_dash(float(sb.get("near_player_frac", -1.0)))
    sel = "*" if chosen else " "
    return (
        f"  {sel:>3} {int(trk.track_id):>3} "
        f"{float(trk.score):>8.1f} "
        f"{int(trk.num_obs):>5} "
        f"{int(trk.span):>5} "
        f"{start_frac * 100:>5.1f} "
        f"{end_frac * 100:>5.1f} "
        f"{inside_s:>5} {motion_s:>5} {near_s:>6}"
    )


def print_timing_summary(timing: Dict[str, float], total_elapsed: float,
                         pass1_frames: int, pass2_frames: int) -> None:
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
        ("pass2_write_track_debug", "pass2 write track-debug", pass2_frames),
        ("pass2_write_debug", "pass2 write debug", pass2_frames),
    ]
    denom_total = max(float(total_elapsed), 1e-9)
    for key, label, nframes in rows:
        sec = float(timing.get(key, 0.0))
        if sec <= 0.0:
            continue
        pct = 100.0 * sec / denom_total
        line_text = f"[info]   {label:<20} {sec:7.3f}s ({pct:5.1f}%)"
        if nframes > 0:
            line_text += f", {1000.0 * sec / max(int(nframes), 1):6.2f} ms/frame"
        print(line_text)
