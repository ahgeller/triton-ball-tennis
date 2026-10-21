"""Drawing for the tracking video and its track-debug twin.

Everything here draws onto frames; the geometry comes from
``court_geometry`` and the per-frame state (trails, marker size) lives in
small classes that pass 2 advances once per frame.
"""
from __future__ import annotations

import hashlib
import math
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

import cv2
import numpy as np
from ball_in_play_selector import Detection, FrameResult, Track

from .config import Config
from .court_geometry import (
    LINE_PAIRS_14, GroundModel, build_court_side_guides, clip_infinite_line_to_frame, court_axis_spans,
    homography_apply,
)
from .court_overlay import PlayerBoxes

Color = Tuple[int, int, int]
Point = Tuple[int, int]

# Colours (BGR).
COLOR_DET: Color = (0, 255, 0)
COLOR_MOTION: Color = (0, 165, 255)
COLOR_INTERP: Color = (255, 255, 0)
COLOR_RAW: Color = (0, 180, 180)
COLOR_SEARCH: Color = (255, 0, 255)
COLOR_DISPLAY_PATH: Color = (255, 255, 255)
COLOR_DISPLAY_PATH_INTERP: Color = (180, 220, 255)
COLOR_OUTLINE: Color = (15, 15, 15)          # dark halo under markers and text
COLOR_WHITE: Color = (255, 255, 255)
COLOR_PLAYER_BOX: Color = (0, 0, 255)
COLOR_COURT_POLYGON: Color = (100, 100, 100)
COLOR_ROI: Color = (0, 255, 0)
COLOR_NET_LINE: Color = (0, 255, 0)
COLOR_COURT_LINE: Color = (0, 255, 0)
COLOR_COURT_INDEX: Color = (0, 0, 255)
# Fewest court keypoints that are joined by lines.
_COURT_MIN_LINE_POINTS = 4

# Trail colour per FrameResult.source, before the age fade.
_SOURCE_BASE_COLOR: Dict[str, Color] = {
    "det": (40, 255, 40),
    "motion": (0, 185, 255),
    "interp": COLOR_INTERP,
}
_UNKNOWN_SOURCE_COLOR: Color = (255, 220, 80)
# Marker colour per FrameResult.source in the motion-debug video.
MOTION_DEBUG_SOURCE_COLOR: Dict[str, Color] = {
    "det": (0, 220, 60),
    "motion": (0, 160, 255),
    "interp": (0, 220, 220),
}
HUD_LABEL: Dict[str, str] = {"det": "DET", "motion": "MOT"}
HUD_LABEL_DEFAULT = "PHY"

# Marker geometry.
_DET_MARKER_DEFAULT_RADIUS = 6
_ESTIMATE_MARKER_MIN_RADIUS, _ESTIMATE_MARKER_MAX_RADIUS = 4, 12
_LABEL_OFFSET: Point = (10, -5)
_ANCHOR_RADIUS_EMA_KEEP, _ANCHOR_RADIUS_EMA_NEW = 0.80, 0.20   # running ball-radius estimate weights
_ANCHOR_RADIUS_CAP_SCALE = 3               # cap = ball_marker_max_radius * this
SEARCH_MARKER_SIZE = 12
_SEARCH_CENTRE_MAX_STEP_PX = 64.0          # display smoothing of the motion-search centre
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
# Display path (track-debug video) construction.
_PATH_MAX_STEP_FRAC, _PATH_MAX_STEP_MIN_PX = 0.15, 10.0
_PATH_STATIC_SPEED_FRAC, _PATH_STATIC_SPEED_MIN_PX = 0.010, 2.0
_PATH_MAX_RESID_FRAC, _PATH_MAX_RESID_MIN_PX = 0.08, 14.0
_PATH_SHORTCUT_RATIO = 0.70
_PATH_SEGMENT_SPEED_SLACK = 1.10
_PATH_TAIL_SPEED_SCALE = 1.35
# Candidate-track overlay.
CANDIDATE_TRAIL_LEN = 120     # long enough to show full rally segments
_CANDIDATE_LABEL_OFFSET: Point = (6, 6)
_TRACK_COLOR_MIN_COMPONENT, _TRACK_COLOR_TARGET = 100, 150.0
# HUD.
_HUD_FRAME_POS: Point = (10, 25)
_HUD_TEXT_POS: Point = (60, 25)
_HUD_FONT_SCALE = 0.5


# ---------------------------------------------------------------------------
# Canvas: the output frame and its optional debug twin
# ---------------------------------------------------------------------------

class Canvas:
    """The pass-2 output frame and, when the track-debug video is on, its debug twin.

    Each drawing primitive is applied to every target it is given, so a
    marker that belongs on both frames is written once.  ``both`` is the
    main frame plus the debug frame (when present); ``debug_only`` is the
    debug frame alone; ``select`` picks per flag.
    """

    def __init__(self, main: np.ndarray, debug: Optional[np.ndarray]) -> None:
        self.main = main
        self.debug = debug

    @property
    def both(self) -> Tuple[np.ndarray, ...]:
        return (self.main, self.debug) if self.debug is not None else (self.main,)

    @property
    def debug_only(self) -> Tuple[np.ndarray, ...]:
        return (self.debug,) if self.debug is not None else ()

    def select(self, main: bool, debug: bool = True) -> Tuple[np.ndarray, ...]:
        targets = []
        if main:
            targets.append(self.main)
        if debug and self.debug is not None:
            targets.append(self.debug)
        return tuple(targets)


def line(targets: Iterable[np.ndarray], p0: Point, p1: Point, color: Color, thickness: int,
         line_type: int = cv2.LINE_8) -> None:
    for img in targets:
        cv2.line(img, p0, p1, color, thickness, line_type)


def circle(targets: Iterable[np.ndarray], centre: Point, radius: int, color: Color, thickness: int,
           line_type: int = cv2.LINE_8) -> None:
    for img in targets:
        cv2.circle(img, centre, radius, color, thickness, line_type)


def rectangle(targets: Iterable[np.ndarray], p0: Point, p1: Point, color: Color, thickness: int) -> None:
    for img in targets:
        cv2.rectangle(img, p0, p1, color, thickness)


def text(targets: Iterable[np.ndarray], label: str, origin: Point, scale: float, color: Color,
         thickness: int = 1) -> None:
    for img in targets:
        cv2.putText(img, label, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def cross(targets: Iterable[np.ndarray], centre: Point, color: Color, size: int) -> None:
    for img in targets:
        cv2.drawMarker(img, centre, color, cv2.MARKER_CROSS, size, 1)


# ---------------------------------------------------------------------------
# Static layers: court, players, raw detections, ROIs
# ---------------------------------------------------------------------------

def draw_court_side_guides(frame: np.ndarray, court_keypoints: Optional[Sequence[float]],
                           color: Color = COLOR_PLAYER_BOX, thickness: int = 2) -> np.ndarray:
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


def draw_homography_net_line(frame: np.ndarray, ground_model: Optional[GroundModel]) -> None:
    """Draw the net line (unit-court v = 0.5) through the court-plane homography."""
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
    cv2.line(frame, (x1, y1), (x2, y2), COLOR_OUTLINE, 4, cv2.LINE_AA)
    cv2.line(frame, (x1, y1), (x2, y2), COLOR_NET_LINE, 2, cv2.LINE_AA)


def _detected_court_point(kps: Sequence[float], index: int) -> Optional[Point]:
    """Keypoint ``index`` as integer pixels, or None when it is at ``(0, 0)`` or below (not detected)."""
    x, y = int(kps[2 * index]), int(kps[2 * index + 1])
    return (x, y) if (x > 0 or y > 0) else None


def _court_line_pairs(n: int) -> Sequence[Tuple[int, int]]:
    """Which keypoints to join: the semantic 14-point layout, else a chain through consecutive points."""
    if n == 14:
        return LINE_PAIRS_14
    if n >= _COURT_MIN_LINE_POINTS:
        return [(j, j + 1) for j in range(n - 1)]
    return ()


def _draw_court_points(frame: np.ndarray, kps: Sequence[float], cfg: Config) -> None:
    for index in range(len(kps) // 2):
        point = _detected_court_point(kps, index)
        if point is None:
            continue
        cv2.circle(frame, point, 5, COLOR_COURT_LINE, -1)
        if cfg.court_draw_indices:
            cv2.putText(frame, str(index), (point[0], point[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_COURT_INDEX, 2)


def _draw_court_lines(frame: np.ndarray, kps: Sequence[float]) -> None:
    for a, b in _court_line_pairs(len(kps) // 2):
        pa, pb = _detected_court_point(kps, a), _detected_court_point(kps, b)
        if pa is not None and pb is not None:
            cv2.line(frame, pa, pb, COLOR_COURT_LINE, 2)


def draw_court(frame: np.ndarray, keypoints: Optional[Sequence[float]], cfg: Config) -> np.ndarray:
    """Draw court keypoints as green dots and, unless ``cfg.court_points_only``, the lines between them.

    With 14 points the lines follow the detector's semantic layout
    (``LINE_PAIRS_14``); with at least four they chain consecutive points.
    A point at ``(0, 0)`` or below is not drawn.  ``cfg.court_draw_indices``
    labels each dot with its index.  Returns ``frame``, drawn on in place.
    """
    if keypoints is None:
        return frame
    _draw_court_points(frame, keypoints, cfg)
    if not cfg.court_points_only:
        _draw_court_lines(frame, keypoints)
    return frame


def draw_court_layer(targets: Iterable[np.ndarray], cfg: Config, court_kps: Optional[Sequence[float]],
                     ground_model: Optional[GroundModel], court_polygon: Optional[np.ndarray]) -> None:
    """Court keypoints and lines (when ``cfg.draw_court``), the net line and the court polygon outline."""
    targets = tuple(targets)
    if cfg.draw_court and court_kps:
        for img in targets:
            draw_court(img, court_kps, cfg)
    if ground_model is not None:
        for img in targets:
            draw_homography_net_line(img, ground_model)
    if court_polygon is not None:
        for img in targets:
            cv2.drawContours(img, [court_polygon.astype(np.int32)], 0, COLOR_COURT_POLYGON, 1)


def draw_player_boxes(targets: Iterable[np.ndarray], player_boxes: PlayerBoxes) -> None:
    targets = tuple(targets)
    for pbox in player_boxes.values():
        if pbox is None or len(pbox) < 4:
            continue
        px1, py1, px2, py2 = map(int, pbox)
        rectangle(targets, (px1, py1), (px2, py2), COLOR_PLAYER_BOX, 2)


def draw_raw_detections(targets: Iterable[np.ndarray], dets: Sequence[Tuple[Sequence[float], float]]) -> None:
    """Every detector proposal as a thin box with its bottom edge emphasised."""
    targets = tuple(targets)
    for ball_box, _conf in dets:
        ix1, iy1, ix2, iy2 = map(int, ball_box)
        rectangle(targets, (ix1, iy1), (ix2, iy2), COLOR_RAW, 1)
        line(targets, (ix1, iy2), (ix2, iy2), COLOR_PLAYER_BOX, 1)


def draw_motion_rois(targets: Iterable[np.ndarray], rois: Optional[Sequence[Sequence[int]]]) -> None:
    targets = tuple(targets)
    for rx1, ry1, rx2, ry2 in rois or ():
        rectangle(targets, (int(rx1), int(ry1)), (int(rx2), int(ry2)), COLOR_ROI, 2)
        text(targets, "ROI", (int(rx1), max(15, int(ry1) - 5)), 0.4, COLOR_ROI)


# ---------------------------------------------------------------------------
# The selected result: marker, search region, HUD
# ---------------------------------------------------------------------------

class MarkerSizer:
    """Running estimate of the ball's radius from detector boxes, for marker sizes."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.radius = float(max(cfg.ball_marker_min_radius, 4))
        self._cap = float(max(cfg.ball_marker_max_radius * _ANCHOR_RADIUS_CAP_SCALE, cfg.ball_marker_min_radius))

    def observe_box(self, box_w: float, box_h: float) -> int:
        """Fold a detector box in; returns the marker radius for that detection."""
        cfg = self.cfg
        det_radius = 0.5 * max(box_w, box_h)
        self.radius = float(np.clip(
            _ANCHOR_RADIUS_EMA_KEEP * self.radius + _ANCHOR_RADIUS_EMA_NEW * det_radius,
            cfg.ball_marker_min_radius, self._cap))
        ball_px = max(int(round(box_w)), int(round(box_h)), 1)
        return int(np.clip(round(ball_px * cfg.ball_marker_box_scale),
                           cfg.ball_marker_min_radius, cfg.ball_marker_max_radius))

    @property
    def estimate_radius(self) -> int:
        """Marker radius for a position without a detector box (motion/interp)."""
        return int(np.clip(round(self.radius), _ESTIMATE_MARKER_MIN_RADIUS, _ESTIMATE_MARKER_MAX_RADIUS))


class SearchCentreSmoother:
    """Display-only smoothing of the motion-search centre (the magenta cross).

    The selector's search centre can jump between frames; the overlay caps the
    jump so the cross does not flicker.  This never feeds back into selection.
    """

    def __init__(self) -> None:
        self._last: Optional[Point] = None
        self._last_frame = -1

    def smooth(self, frame_idx: int, scx: int, scy: int) -> Point:
        if self._last is not None and self._last_frame == frame_idx - 1:
            pscx, pscy = self._last
            dx, dy = scx - pscx, scy - pscy
            d = float((dx * dx + dy * dy) ** 0.5)
            if d > _SEARCH_CENTRE_MAX_STEP_PX and d > 1e-6:
                s = _SEARCH_CENTRE_MAX_STEP_PX / d
                scx = int(round(pscx + dx * s))
                scy = int(round(pscy + dy * s))
        self._last = (scx, scy)
        self._last_frame = frame_idx
        return scx, scy

    def reset(self) -> None:
        self._last = None
        self._last_frame = -1


def draw_selected_result(canvas: Canvas, result: FrameResult, frame_idx: int, sizer: MarkerSizer,
                         search_smoother: SearchCentreSmoother, draw_search_regions: bool) -> None:
    """The chosen position, colour-coded by source, with its search region on the debug frame
    (and on the main frame when ``draw_search_regions``)."""
    rcx, rcy = int(result.cx), int(result.cy)
    label_at = (rcx + _LABEL_OFFSET[0], rcy + _LABEL_OFFSET[1])
    both = canvas.both
    if result.source == "det":
        marker_radius = _DET_MARKER_DEFAULT_RADIUS
        if result.bbox:
            bx1f, by1f, bx2f, by2f = map(float, result.bbox)
            bx1, by1, bx2, by2 = map(int, (bx1f, by1f, bx2f, by2f))
            # High-contrast box: dark halo + bright box + emphasised bottom edge.
            rectangle(both, (bx1, by1), (bx2, by2), COLOR_OUTLINE, 4)
            rectangle(both, (bx1, by1), (bx2, by2), COLOR_DET, 2)
            line(both, (bx1, by2), (bx2, by2), COLOR_WHITE, 2)
            marker_radius = sizer.observe_box(max(bx2f - bx1f, 1.0), max(by2f - by1f, 1.0))
        # The marker sits on the measured centre, not a bottom anchor.
        circle(both, (rcx, rcy), marker_radius + 2, COLOR_OUTLINE, 2)
        circle(both, (rcx, rcy), marker_radius, COLOR_DET, -1)
        text(both, "DET", label_at, 0.4, COLOR_DET)
    elif result.source == "motion":
        radius = sizer.estimate_radius
        circle(both, (rcx, rcy), radius, COLOR_MOTION, -1)
        text(both, "MOT", label_at, 0.4, COLOR_MOTION)
        if result.search_radius > 0:
            scx, scy = search_smoother.smooth(frame_idx, int(result.search_cx), int(result.search_cy))
            targets = canvas.select(main=draw_search_regions)
            circle(targets, (scx, scy), int(result.search_radius), COLOR_SEARCH, 1)
            line(targets, (scx, scy), (rcx, rcy), COLOR_SEARCH, 1)
            cross(targets, (scx, scy), COLOR_SEARCH, SEARCH_MARKER_SIZE)
        else:
            search_smoother.reset()
    elif result.source == "interp":
        search_smoother.reset()
        radius = sizer.estimate_radius
        circle(both, (rcx, rcy), radius, COLOR_INTERP, -1)
        text(both, "PHY", label_at, 0.4, COLOR_INTERP)

    # The selector's search region for det/interp frames (court-scaled radius).
    if result.source in ("det", "interp") and float(result.search_radius) > 0.0:
        scx = int(round(float(result.search_cx)))
        scy = int(round(float(result.search_cy)))
        sr = int(round(float(result.search_radius)))
        scol = COLOR_DET if result.source == "det" else COLOR_INTERP
        targets = canvas.select(main=draw_search_regions)
        circle(targets, (scx, scy), sr, COLOR_OUTLINE, 2, cv2.LINE_AA)
        circle(targets, (scx, scy), sr, scol, 1, cv2.LINE_AA)


def draw_hud(canvas: Canvas, frame_idx: int, result: Optional[FrameResult],
             chosen_track: Optional[Track]) -> None:
    """Frame number on both frames; source/confidence on both when there is a result;
    the chosen track id on the debug frame."""
    frame_label = f"F{frame_idx}"
    text(canvas.both, frame_label, _HUD_FRAME_POS, _HUD_FONT_SCALE, COLOR_WHITE)
    if result is not None:
        chosen_id = int(chosen_track.track_id) if chosen_track else -1
        source_label = HUD_LABEL.get(result.source, HUD_LABEL_DEFAULT)
        text(canvas.both, f"ID:{chosen_id} {source_label} c={result.conf:.2f}", _HUD_TEXT_POS,
             _HUD_FONT_SCALE, COLOR_WHITE)
        # The debug frame's label is drawn a second time: OpenCV text overdraw
        # thickens the glyph edges, and the track-debug video has always looked so.
        text(canvas.debug_only, frame_label, _HUD_FRAME_POS, _HUD_FONT_SCALE, COLOR_WHITE)
    if chosen_track is not None:
        text(canvas.debug_only, f"track={chosen_track.track_id}", _HUD_TEXT_POS, _HUD_FONT_SCALE, COLOR_WHITE)


# ---------------------------------------------------------------------------
# Trails
# ---------------------------------------------------------------------------

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
                segment: List[Tuple[float, float]] = []
                for p in trail + [None]:
                    if p is None:
                        for i in range(1, len(segment)):
                            p0 = (int(segment[i - 1][0]), int(segment[i - 1][1]))
                            p1 = (int(segment[i][0]), int(segment[i][1]))
                            cv2.line(debug, p0, p1, color, 1, cv2.LINE_AA)
                        segment = []
                    else:
                        segment.append(p)

            if pt is not None:
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


# ---------------------------------------------------------------------------
# The chosen track's display path
# ---------------------------------------------------------------------------

# A display-path point: (x, y, exact), exact when it is a track observation.
PathPoint = Tuple[float, float, bool]


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
