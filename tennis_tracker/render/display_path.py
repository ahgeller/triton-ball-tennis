"""The chosen track's display path for the track-debug video.

``build_display_path`` maps frame index to ``(x, y, exact)``: the track's
observations (exact), linear fills across short gaps and a short tail
extension while the observed tail is moving (not exact).  Drawing only; the
tracking JSON never reads it, and ``trails.DisplayPathTrail`` paints it.
"""
from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
from ball_in_play_selector import Detection, Track

# A display-path point: (x, y, exact), exact when it is a track observation.
PathPoint = Tuple[float, float, bool]

_PATH_MAX_STEP_FRAC, _PATH_MAX_STEP_MIN_PX = 0.15, 10.0
_PATH_STATIC_SPEED_FRAC, _PATH_STATIC_SPEED_MIN_PX = 0.010, 2.0
_PATH_MAX_RESID_FRAC, _PATH_MAX_RESID_MIN_PX = 0.08, 14.0
_PATH_SHORTCUT_RATIO = 0.70
_PATH_SEGMENT_SPEED_SLACK = 1.10
_PATH_TAIL_SPEED_SCALE = 1.35


class _PathLimits(NamedTuple):
    """Pixel thresholds of the display path, scaled from the frame diagonal."""
    max_step: float       # px/frame above which a step is a jump
    static_speed: float   # px/frame below which the ball is standing still
    max_resid: float      # px off the a->c chord that makes b an excursion
    max_gap: int          # frames a linear fill or tail may span

    @classmethod
    def for_frame(cls, frame_w: int, frame_h: int, max_interp_gap: int) -> "_PathLimits":
        diag = float(np.hypot(frame_w, frame_h))
        return cls(
            max_step=max(_PATH_MAX_STEP_MIN_PX, _PATH_MAX_STEP_FRAC * diag),
            static_speed=max(_PATH_STATIC_SPEED_MIN_PX, _PATH_STATIC_SPEED_FRAC * diag),
            max_resid=max(_PATH_MAX_RESID_MIN_PX, _PATH_MAX_RESID_FRAC * diag),
            max_gap=max(1, max_interp_gap),
        )


def _distance(a: Detection, b: Detection) -> float:
    return float(np.hypot(b.cx - a.cx, b.cy - a.cy))


def _best_observation_per_frame(track: Track, total_frames: int) -> List[Detection]:
    """The track's observations inside the clip, one per frame (the most confident), in frame order."""
    best_by_frame: Dict[int, Detection] = {}
    for o in track.observations:
        prev = best_by_frame.get(o.frame)
        if prev is None or o.conf > prev.conf:
            best_by_frame[o.frame] = o
    return [best_by_frame[f] for f in sorted(best_by_frame) if 0 <= f < total_frames]


def _drop_display_outliers(obs: List[Detection], lim: _PathLimits, max_interp_gap: int) -> List[Detection]:
    """Drop observations that would draw a spike: a jump onto a static point, or a short
    excursion off the chord between its neighbours.  The first and last observations stay."""
    if len(obs) < 3:
        return obs
    resid_gap = max(2, max_interp_gap)
    keep = [obs[0]]
    for i in range(1, len(obs) - 1):
        a, b, c = keep[-1], obs[i], obs[i + 1]
        step_ab = _distance(a, b) / max(b.frame - a.frame, 1)
        step_bc = _distance(b, c) / max(c.frame - b.frame, 1)
        if step_ab > lim.max_step and step_bc <= lim.static_speed and not b.on_motion:
            continue
        dt_ac = c.frame - a.frame
        if 0 < dt_ac <= resid_gap:
            t = (b.frame - a.frame) / dt_ac
            resid = float(np.hypot(b.cx - (a.cx + (c.cx - a.cx) * t), b.cy - (a.cy + (c.cy - a.cy) * t)))
            shortcut = _distance(a, c) / dt_ac
            if (step_ab > lim.max_step and step_bc > lim.max_step
                    and resid > lim.max_resid and shortcut < _PATH_SHORTCUT_RATIO * max(step_ab, step_bc)):
                continue
        keep.append(b)
    if obs[-1].frame != keep[-1].frame:
        keep.append(obs[-1])
    return keep


def _fill_display_gaps(path: Dict[int, PathPoint], obs: List[Detection], lim: _PathLimits) -> None:
    """Linear fills across gaps of at most ``lim.max_gap`` frames, unless the ball would have jumped."""
    for a, b in zip(obs, obs[1:]):
        gap = b.frame - a.frame
        if gap <= 1 or gap > lim.max_gap:
            continue
        if _distance(a, b) / gap > lim.max_step * _PATH_SEGMENT_SPEED_SLACK:
            continue
        for f in range(a.frame + 1, b.frame):
            if f not in path:
                t = (f - a.frame) / gap
                path[f] = (a.cx + (b.cx - a.cx) * t, a.cy + (b.cy - a.cy) * t, False)


def _extend_display_tail(path: Dict[int, PathPoint], obs: List[Detection], total_frames: int,
                         lim: _PathLimits) -> None:
    """Continue the last step's velocity for up to ``lim.max_gap`` frames while the tail is moving."""
    if len(obs) < 2:
        return
    prev, last = obs[-2], obs[-1]
    dt_last = max(last.frame - prev.frame, 1)
    vx = (last.cx - prev.cx) / dt_last
    vy = (last.cy - prev.cy) / dt_last
    tail_speed = float(np.hypot(vx, vy))
    if tail_speed > lim.static_speed * _PATH_TAIL_SPEED_SCALE and (last.on_motion or prev.on_motion):
        end_f = min(total_frames - 1, last.frame + lim.max_gap)
        for f in range(last.frame + 1, end_f + 1):
            if f not in path:
                dtf = f - last.frame
                path[f] = (last.cx + vx * dtf, last.cy + vy * dtf, False)


def build_display_path(track: Optional[Track], total_frames: int, max_interp_gap: int,
                       frame_w: int, frame_h: int) -> Dict[int, PathPoint]:
    """The chosen track's display path for the track-debug video.

    Maps frame index to ``(x, y, exact)``: the track's observations (exact),
    linear fills across gaps of at most ``max_interp_gap`` frames, and a short
    tail extension while the observed tail is moving (not exact).  Drawing
    only; the tracking JSON never reads it.
    """
    path: Dict[int, PathPoint] = {}
    if track is None or track.num_obs <= 0:
        return path
    lim = _PathLimits.for_frame(frame_w, frame_h, max_interp_gap)
    obs = _drop_display_outliers(_best_observation_per_frame(track, total_frames), lim, max_interp_gap)
    for o in obs:
        path[o.frame] = (float(o.cx), float(o.cy), True)
    _fill_display_gaps(path, obs, lim)
    _extend_display_tail(path, obs, total_frames, lim)
    return path
