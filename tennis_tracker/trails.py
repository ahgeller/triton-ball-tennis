"""The trails of the tracking video and its track-debug twin: the chosen ball's recent path
(``BallTrail``), the chosen track's display path (``DisplayPathTrail``) and every selector
track's dots and short trails (``CandidateTrackTrails``).  Each one advances once per frame in
pass 2 and draws onto the ``rendering.Canvas``.
"""
from __future__ import annotations

import hashlib
import math
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from ball_in_play_selector import FrameResult, Track

from .config import Config
from .court_geometry import court_axis_spans
from .rendering import (
    COLOR_DISPLAY_PATH, COLOR_DISPLAY_PATH_INTERP, COLOR_INTERP, COLOR_OUTLINE, Canvas, Color, Point,
)

# Trail colour per FrameResult.source, before the age fade.
_SOURCE_BASE_COLOR: Dict[str, Color] = {
    "det": (40, 255, 40),
    "motion": (0, 185, 255),
    "interp": COLOR_INTERP,
}
_UNKNOWN_SOURCE_COLOR: Color = (255, 220, 80)
# Trail geometry.
TRAIL_SECONDS = 0.25
TRAIL_MIN_LEN = 10
DISPLAY_PATH_TRAIL_MIN_LEN = 40
DISPLAY_PATH_TRAIL_EXTRA = 20
_TRAIL_FADE_BASE, _TRAIL_FADE_RANGE = 0.55, 0.45    # alpha = base + range * age fraction
_TRAIL_THICKNESS = 3
_TRAIL_MIN_THICKNESS = 2
_TRAIL_HARD_MIN_X_PX, _TRAIL_HARD_MIN_Y_PX = 34.0, 26.0
_TRAIL_BRIDGE_FRAC, _TRAIL_BRIDGE_MIN_PX = 0.035, 18.0
_TRAIL_TRANSITION_FRAC, _TRAIL_TRANSITION_MIN_PX = 0.020, 14.0
_TRAIL_DIRECTION_BRIDGE_SCALE = 1.4
_MOTION_JUMP_FRAC_CAP = 0.24
_INTERP_JUMP_FRAC = 0.16
_DEFAULT_JUMP_FRAC = 0.18
_DIRECTION_BREAK_MIN_STEP_PX = 6.0
_DIRECTION_BREAK_COS = -0.32
_DIRECTION_BREAK_MIN_DY_PX = 12.0
_DISPLAY_PATH_MAX_SEGMENT_FRAC, _DISPLAY_PATH_MAX_SEGMENT_MIN_PX = 0.15, 200.0
# Candidate-track overlay.
CANDIDATE_TRAIL_LEN = 120     # long enough to show full rally segments
_CANDIDATE_LABEL_OFFSET: Point = (6, 6)
_TRACK_COLOR_MIN_COMPONENT, _TRACK_COLOR_TARGET = 100, 150.0


def _trail_base_color(src: str) -> Color:
    return _SOURCE_BASE_COLOR.get(src, _UNKNOWN_SOURCE_COLOR)


def _trail_jump_fracs(src: Optional[str], cfg: Config) -> Tuple[float, float]:
    """Court-span fractions a trail may jump between frames before it breaks, per source."""
    if src == "det":
        return cfg.trail_hard_switch_x_frac, cfg.trail_hard_switch_y_frac
    if src == "motion":
        return (min(cfg.trail_hard_switch_x_frac, _MOTION_JUMP_FRAC_CAP),
                min(cfg.trail_hard_switch_y_frac, _MOTION_JUMP_FRAC_CAP))
    if src == "interp":
        return _INTERP_JUMP_FRAC, _INTERP_JUMP_FRAC
    return _DEFAULT_JUMP_FRAC, _DEFAULT_JUMP_FRAC


def _trail_direction_break(prev2: Sequence[float], prev: Sequence[float], curr_xy: Sequence[float]) -> bool:
    """True on a strong direction reversal (a bounce or hit) between three trail points."""
    v1x = prev[0] - prev2[0]
    v1y = prev[1] - prev2[1]
    v2x = curr_xy[0] - prev[0]
    v2y = curr_xy[1] - prev[1]
    m1 = float(np.hypot(v1x, v1y))
    m2 = float(np.hypot(v2x, v2y))
    if m1 < _DIRECTION_BREAK_MIN_STEP_PX or m2 < _DIRECTION_BREAK_MIN_STEP_PX:
        return False
    cosang = (v1x * v2x + v1y * v2y) / max(m1 * m2, 1e-6)
    if cosang < _DIRECTION_BREAK_COS:
        return True
    if (np.sign(v1y) != np.sign(v2y) and abs(v1y) > _DIRECTION_BREAK_MIN_DY_PX
            and abs(v2y) > _DIRECTION_BREAK_MIN_DY_PX):
        return True
    return False


# A trail point: (x, y, source, conf); None marks a break between segments.
TrailPoint = Tuple[int, int, str, float]


class BallTrail:
    """The chosen ball's recent path, broken into segments at jumps and reversals.

    ``advance`` appends this frame's position (or a break) and ``draw`` paints
    the segments, newest brightest and thickest.
    """

    def __init__(self, cfg: Config, fps: float, frame_w: int, frame_h: int) -> None:
        self.cfg = cfg
        self.frame_w, self.frame_h = frame_w, frame_h
        self.max_len = max(TRAIL_MIN_LEN, int(round(fps * TRAIL_SECONDS)))
        self.points: List[Optional[TrailPoint]] = []
        self._pending_lost_gap = False

    def _last_two(self) -> Tuple[Optional[TrailPoint], Optional[TrailPoint]]:
        prev = prev2 = None
        for p in reversed(self.points):
            if p is None:
                continue
            if prev is None:
                prev = p
            else:
                prev2 = p
                break
        return prev, prev2

    def _breaks_before(self, x: int, y: int, src: str, court_kps: Optional[Sequence[float]]) -> bool:
        """Whether the segment ends before (x, y): too far from the last point, or a reversal."""
        prev, prev2 = self._last_two()
        if prev is None:
            return False
        prev_src = prev[2]
        x_span, y_span = court_axis_spans(court_kps, self.frame_w, self.frame_h)
        dx, dy = abs(x - prev[0]), abs(y - prev[1])
        jx_c, jy_c = _trail_jump_fracs(src, self.cfg)
        jx_p, jy_p = _trail_jump_fracs(prev_src, self.cfg)
        hard_x_px = max(max(jx_c, jx_p) * x_span, _TRAIL_HARD_MIN_X_PX)
        hard_y_px = max(max(jy_c, jy_p) * y_span, _TRAIL_HARD_MIN_Y_PX)
        dist = float(np.hypot(dx, dy))
        pair_diag = float(np.hypot(x_span, y_span))
        source_transition = src != prev_src
        bridge_px = max(_TRAIL_BRIDGE_FRAC * pair_diag, _TRAIL_BRIDGE_MIN_PX)
        transition_px = max(_TRAIL_TRANSITION_FRAC * pair_diag, _TRAIL_TRANSITION_MIN_PX)

        # A source transition connects only within a reasonable distance; too
        # tight leaves gaps everywhere, too wide connects to noise.
        if source_transition:
            if dist > bridge_px:
                return True
            if prev_src == "interp" and src == "det" and dist > transition_px:
                return True
        if dx > hard_x_px or dy > hard_y_px:
            # A small source-transition mismatch (e.g. a reacquire) is allowed.
            return not (source_transition and dist <= bridge_px)
        if prev2 is not None and _trail_direction_break(prev2, prev, (x, y)):
            # Reversals only break confident segments.
            return not (src == "interp" or prev_src == "interp"
                        or (source_transition and dist <= bridge_px * _TRAIL_DIRECTION_BRIDGE_SCALE))
        return False

    def advance(self, result: Optional[FrameResult], court_kps: Optional[Sequence[float]]) -> None:
        if result is None:
            # Nothing selected: break the segment before the next recovered point.
            if self.points and self.points[-1] is not None:
                self._pending_lost_gap = True
            self.points.append(None)
        else:
            x, y = int(result.cx), int(result.cy)
            src = result.source
            if not self._pending_lost_gap and self._breaks_before(x, y, src, court_kps):
                self.points.append(None)
            # Recovering from a fully-lost run hard-breaks the segment.
            if self._pending_lost_gap and self._last_two()[0] is not None:
                self.points.append(None)
            self._pending_lost_gap = False
            # The selector already smooths soft points: draw its path exactly so
            # estimates meet detections instead of lagging behind them.
            self.points.append((x, y, src, float(result.conf)))
        while len(self.points) > self.max_len:
            self.points.pop(0)

    def draw(self, canvas: Canvas, draw_main: bool) -> None:
        """Paint the segments: the debug frame always, the main frame when ``draw_main``."""
        n = len(self.points)
        for i in range(1, n):
            a, b = self.points[i - 1], self.points[i]
            if a is None or b is None:
                continue
            src = b[2]
            p0, p1 = (a[0], a[1]), (b[0], b[1])
            base = _trail_base_color(src)
            alpha = _TRAIL_FADE_BASE + _TRAIL_FADE_RANGE * (i / n)
            color = (int(base[0] * alpha), int(base[1] * alpha), int(base[2] * alpha))
            if canvas.debug is not None:
                th = max(_TRAIL_MIN_THICKNESS, int(_TRAIL_THICKNESS * alpha))
                cv2.line(canvas.debug, p0, p1, COLOR_OUTLINE, th + 1, cv2.LINE_AA)
                cv2.line(canvas.debug, p0, p1, color, th, cv2.LINE_AA)
            if draw_main:
                th = max(_TRAIL_MIN_THICKNESS, int(round(_TRAIL_THICKNESS * alpha)))
                cv2.line(canvas.main, p0, p1, COLOR_OUTLINE, th + 1, cv2.LINE_AA)
                cv2.line(canvas.main, p0, p1, color, th, cv2.LINE_AA)


# A display-path trail point: (x, y, exact); None marks a gap.
PathTrailPoint = Tuple[int, int, bool]


class DisplayPathTrail:
    """The chosen track's display path as drawn in the track-debug video.

    The path is only drawn on frames without a selected result, so the white
    line shows where the chosen track says the ball is while the selector has
    nothing; ``exact`` points (observations) are brighter and thicker.
    """

    def __init__(self, fps: float, frame_w: int) -> None:
        trail_max = max(TRAIL_MIN_LEN, int(round(fps * TRAIL_SECONDS)))
        self.max_len = max(DISPLAY_PATH_TRAIL_MIN_LEN, trail_max + DISPLAY_PATH_TRAIL_EXTRA)
        self.max_segment_px = max(_DISPLAY_PATH_MAX_SEGMENT_MIN_PX, frame_w * _DISPLAY_PATH_MAX_SEGMENT_FRAC)
        self.points: List[Optional[PathTrailPoint]] = []

    def advance(self, path_point: Optional[Tuple[float, float, bool]]) -> None:
        if path_point is not None:
            x, y, exact = path_point
            self.points.append((int(round(x)), int(round(y)), bool(exact)))
        elif self.points and self.points[-1] is not None:
            self.points.append(None)
        while len(self.points) > self.max_len:
            self.points.pop(0)

    def draw(self, debug: np.ndarray, path_point: Optional[Tuple[float, float, bool]]) -> None:
        """The path segments and the current point; call only when no result was selected."""
        for i in range(1, len(self.points)):
            p0, p1 = self.points[i - 1], self.points[i]
            if p0 is None or p1 is None:
                continue
            if math.hypot(p1[0] - p0[0], p1[1] - p0[1]) > self.max_segment_px:
                continue
            seg_exact = bool(p0[2] and p1[2])
            seg_color = COLOR_DISPLAY_PATH if seg_exact else COLOR_DISPLAY_PATH_INTERP
            seg_th = 2 if seg_exact else 1
            cv2.line(debug, (p0[0], p0[1]), (p1[0], p1[1]), COLOR_OUTLINE, seg_th + 1, cv2.LINE_AA)
            cv2.line(debug, (p0[0], p0[1]), (p1[0], p1[1]), seg_color, seg_th, cv2.LINE_AA)
        if path_point is not None:
            x, y, exact = path_point
            centre = (int(round(x)), int(round(y)))
            color = COLOR_DISPLAY_PATH if exact else COLOR_DISPLAY_PATH_INTERP
            radius = 6 if exact else 4
            cv2.circle(debug, centre, radius + 2, COLOR_OUTLINE, 2)
            cv2.circle(debug, centre, radius, color, -1)


def track_color(track_id: int) -> Color:
    """A stable, distinct BGR colour from a track id (never very dark)."""
    h = hashlib.md5(str(track_id).encode(), usedforsecurity=False).hexdigest()
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    m = max(r, g, b)
    if m < _TRACK_COLOR_MIN_COMPONENT:
        scale = _TRACK_COLOR_TARGET / max(m, 1)
        r = min(255, int(r * scale))
        g = min(255, int(g * scale))
        b = min(255, int(b * scale))
    return (b, g, r)


def _draw_trail_segments(debug: np.ndarray, trail: Sequence[Optional[Tuple[float, float]]], color: Color) -> None:
    """One thin polyline per unbroken stretch of a candidate trail (None marks a gap)."""
    segment: List[Tuple[float, float]] = []
    for p in list(trail) + [None]:
        if p is None:
            for i in range(1, len(segment)):
                p0 = (int(segment[i - 1][0]), int(segment[i - 1][1]))
                p1 = (int(segment[i][0]), int(segment[i][1]))
                cv2.line(debug, p0, p1, color, 1, cv2.LINE_AA)
            segment = []
        else:
            segment.append(p)


class CandidateTrackTrails:
    """Every selector track in the track-debug video: coloured dots, ids, scores and short trails."""

    def __init__(self, tracks: Sequence[Track], chosen_track: Optional[Track]) -> None:
        self.tracks = sorted(tracks, key=lambda t: float(t.score), reverse=True)
        self.chosen_id = int(chosen_track.track_id) if chosen_track is not None else None
        # track id -> {frame: (cx, cy)}
        self.positions: Dict[int, Dict[int, Tuple[float, float]]] = {
            int(t.track_id): {o.frame: (o.cx, o.cy) for o in t.observations} for t in self.tracks}
        self.scores = {int(t.track_id): float(t.score) for t in self.tracks}
        self.trails: Dict[int, List[Optional[Tuple[float, float]]]] = {int(t.track_id): [] for t in self.tracks}

    def _draw_dot(self, debug: np.ndarray, tid: int, pt: Tuple[float, float], color: Color) -> None:
        """The track's position this frame: a dot (filled when chosen) with its id and score."""
        cx, cy = int(pt[0]), int(pt[1])
        is_chosen = tid == self.chosen_id
        dot_r = 5 if is_chosen else 3
        dot_th = -1 if is_chosen else 2
        cv2.circle(debug, (cx, cy), dot_r + 2, COLOR_OUTLINE, -1)   # dark outline for legibility
        cv2.circle(debug, (cx, cy), dot_r, color, dot_th)
        label = f"{tid}({self.scores[tid]:.0f})"
        origin = (cx + _CANDIDATE_LABEL_OFFSET[0], cy + _CANDIDATE_LABEL_OFFSET[1])
        cv2.putText(debug, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_OUTLINE, 2)
        cv2.putText(debug, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

    def advance_and_draw(self, debug: np.ndarray, frame_idx: int) -> None:
        for trk in self.tracks:
            tid = int(trk.track_id)
            pt = self.positions[tid].get(frame_idx)
            trail = self.trails[tid]
            if pt is not None:
                trail.append((pt[0], pt[1]))
            elif trail and trail[-1] is not None:
                trail.append(None)          # a break for the gap
            while len(trail) > CANDIDATE_TRAIL_LEN:
                trail.pop(0)

            color = track_color(tid)
            if len(trail) > 1:
                _draw_trail_segments(debug, trail, color)
            if pt is not None:
                self._draw_dot(debug, tid, pt, color)
