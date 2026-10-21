"""The motion-debug video: masks, the selected ball and raw-motion trails painted on each frame.

Drawing only.  Nothing here feeds the tracker, the ROI tracker or the selector.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from ball_in_play_selector import FrameResult

from ..config import Config
from ..court.court_drawing import draw_court_side_guides
from .rendering import MOTION_DEBUG_SOURCE_COLOR, SEARCH_MARKER_SIZE

try:
    import scipy.interpolate
    import scipy.ndimage
except ImportError:          # optional: trails are then drawn as straight segments
    scipy = None

Color = Tuple[int, int, int]
Box = Tuple[int, int, int, int]
# A raw-motion trail sample: (frame, x, y, radius).
TrailSample = Tuple[int, float, float, float]

COLOR_RAW_ONLY: Color = (0, 180, 0)
COLOR_RAW_AND_BOOST: Color = (0, 255, 255)
COLOR_BOOST_ONLY: Color = (255, 220, 0)
COLOR_COMPONENT_BOX: Color = (255, 255, 0)
COLOR_ROI: Color = (0, 255, 0)
COLOR_LEGEND_BG: Color = (0, 0, 0)
COLOR_MOTION_PIXEL: Color = (0, 0, 128)
COLOR_BOOST_PIXEL: Color = (0, 255, 255)
COLOR_SIDE_GUIDE: Color = (0, 0, 255)
COLOR_SEARCH: Color = (255, 0, 220)
COLOR_TRAIL: Color = (0, 140, 255)
_LEGEND_WIDTH_PX, _LEGEND_HEIGHT_PX = 900, 58
# Display-only component box limits; these never filter tracking evidence.
_COMPONENT_MIN_AREA, _COMPONENT_MAX_AREA = 2, 260
_COMPONENT_MAX_DIM, _COMPONENT_MAX_ASPECT, _COMPONENT_MIN_FILL = 38, 5.5, 0.14
_COMPONENT_BOX_LIMIT = 100
# Ball overlay.
_ESTIMATED_BOX_MIN_RADIUS = 6
_SEARCH_RADIUS_FALLBACK_SCALE, _SEARCH_RADIUS_FALLBACK_MIN = 2.5, 20
# Raw-motion trails.
_BLOB_MIN_AREA = 1.0
_BLOB_RADIUS_MIN, _BLOB_RADIUS_MAX = 4.0, 10.0
_ROI_FALLBACK_RADIUS_FRAC = 0.12
_MATCH_MAX_DIST_FRAC, _MATCH_MAX_DIST_MIN_PX = 0.025, 40.0   # unrelated regions must not merge
_MATCH_MAX_GAP_FRAMES = 3
_TRAIL_MAX_AGE_FRAMES = 150
_TRAIL_PRUNE_EVERY = 100
_TRAIL_MIN_SAMPLES = 2
_UNIQUE_POINT_MIN_DIST_PX = 2.0
_SPLINE_MIN_POINTS = 4
_SPLINE_SMOOTH_SIGMA = 1.5
_SPLINE_MIN_SAMPLES, _SPLINE_SAMPLES_PER_POINT = 100, 5
_STRAIGHT_SEGMENT_MAX_PX = 60.0
_SPLINE_SEGMENT_MAX_PX = 80.0
_TRAIL_THICKNESS = 4
_HEAD_RADIUS_EMA_KEEP, _HEAD_RADIUS_EMA_NEW = 0.90, 0.10
_HEAD_RADIUS_SAMPLES = 10


def _probe_component_boxes(mask_u8: Optional[np.ndarray], limit: int = _COMPONENT_BOX_LIMIT) -> List[Box]:
    """Compact components of ``mask_u8`` as (x, y, w, h) boxes, for the probe overlay."""
    if mask_u8 is None or mask_u8.size == 0 or cv2.countNonZero(mask_u8) == 0:
        return []
    num, _, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    boxes: List[Box] = []
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        if (area < _COMPONENT_MIN_AREA or area > _COMPONENT_MAX_AREA
                or bw > _COMPONENT_MAX_DIM or bh > _COMPONENT_MAX_DIM):
            continue
        aspect = max(bw, bh) / max(min(bw, bh), 1)
        fill = area / max(bw * bh, 1)
        if aspect > _COMPONENT_MAX_ASPECT or fill < _COMPONENT_MIN_FILL:
            continue
        boxes.append((x, y, bw, bh))
        if len(boxes) >= int(limit):
            break
    return boxes


def _draw_probe_motion_overlay(frame: np.ndarray, raw_motion: Optional[np.ndarray],
                               boost_mask: Optional[np.ndarray], protect_mask: Optional[np.ndarray],
                               rois: Optional[Sequence[Sequence[int]]]) -> np.ndarray:
    """Probe-style debug view: direct raw/boost pixels plus compact component boxes."""
    out = frame.copy()
    raw_mask = raw_motion > 0 if raw_motion is not None else np.zeros(out.shape[:2], dtype=bool)
    boost_vis = boost_mask > 0 if boost_mask is not None else np.zeros(out.shape[:2], dtype=bool)
    if protect_mask is not None:
        raw_mask = raw_mask & (~protect_mask)
        boost_vis = boost_vis & (~protect_mask)

    out[raw_mask & (~boost_vis)] = COLOR_RAW_ONLY
    out[raw_mask & boost_vis] = COLOR_RAW_AND_BOOST
    out[boost_vis & (~raw_mask)] = COLOR_BOOST_ONLY

    for x, y, bw, bh in _probe_component_boxes(raw_motion):
        cv2.rectangle(out, (x, y), (x + bw, y + bh), COLOR_COMPONENT_BOX, 1, cv2.LINE_AA)
    for rx1, ry1, rx2, ry2 in rois or ():
        cv2.rectangle(out, (int(rx1), int(ry1)), (int(rx2), int(ry2)), COLOR_ROI, 1, cv2.LINE_AA)

    cv2.rectangle(out, (0, 0), (min(out.shape[1], _LEGEND_WIDTH_PX), _LEGEND_HEIGHT_PX), COLOR_LEGEND_BG, cv2.FILLED)
    cv2.putText(out, "Raw motion probe view: green=refined raw, yellow=raw+boost, cyan boxes=compact raw components",
                (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(out, "Debug drawing only; tracker/ROI/selector still use the existing pipeline masks",
                (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1, cv2.LINE_AA)
    return out


def draw_motion_masks(frame: np.ndarray, raw_motion: Optional[np.ndarray], boost_mask: Optional[np.ndarray],
                      cfg: Config, court_keypoints: Optional[Sequence[float]] = None,
                      protect_mask: Optional[np.ndarray] = None,
                      rois: Optional[Sequence[Sequence[int]]] = None) -> np.ndarray:
    """A copy of ``frame`` with the raw and boost masks painted on it.

    With ``cfg.debug_probe_motion_style`` this is the probe overlay (raw and
    boost pixels, compact component boxes, ROIs, legend); otherwise boost
    pixels are yellow, raw motion (when ``cfg.debug_show_raw_motion``) dark
    red, and the court side guides are drawn.
    """
    if raw_motion is None and boost_mask is None:
        return frame.copy()
    if cfg.debug_probe_motion_style:
        return _draw_probe_motion_overlay(frame, raw_motion, boost_mask, protect_mask, rois)
    out = frame.copy()
    motion_vis = ((raw_motion > 0) if (raw_motion is not None and cfg.debug_show_raw_motion)
                  else np.zeros(out.shape[:2], dtype=bool))
    boost_vis = (boost_mask > 0) if boost_mask is not None else np.zeros_like(motion_vis)
    if protect_mask is not None:
        motion_vis = motion_vis & (~protect_mask)
        boost_vis = boost_vis & (~protect_mask)
    out[motion_vis & (~boost_vis)] = COLOR_MOTION_PIXEL
    out[boost_vis] = COLOR_BOOST_PIXEL
    if court_keypoints is not None:
        draw_court_side_guides(out, court_keypoints, color=COLOR_SIDE_GUIDE, thickness=2)
    return out


def draw_selected_ball(vis: np.ndarray, result: FrameResult, anchor_radius: float) -> None:
    """The selected position: its detector box (or an estimated box) and, off detections, the search region."""
    fx, fy = result.cx, result.cy
    src = str(result.source)
    color = MOTION_DEBUG_SOURCE_COLOR[src]
    if not (fx > 0 or fy > 0):
        return
    cx_i, cy_i = int(round(fx)), int(round(fy))
    if result.bbox is not None and len(result.bbox) == 4:
        bx1, by1, bx2, by2 = (int(round(v)) for v in result.bbox)
        cv2.rectangle(vis, (bx1, by1), (bx2, by2), color, 2)
        cv2.putText(vis, src[:3].upper(), (bx1, max(by1 - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    else:
        br = max(int(round(anchor_radius)), _ESTIMATED_BOX_MIN_RADIUS)
        cv2.rectangle(vis, (cx_i - br, cy_i - br), (cx_i + br, cy_i + br), color, 1)
        cv2.putText(vis, src[:3].upper(), (cx_i - br, max(cy_i - br - 3, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)
    cv2.circle(vis, (cx_i, cy_i), 4, color, -1)
    if src != "det":
        _draw_search_region(vis, result, anchor_radius, (fx, fy), (cx_i, cy_i))


def _draw_search_region(vis: np.ndarray, result: FrameResult, anchor_radius: float, centre: Tuple[float, float],
                        centre_px: Tuple[int, int]) -> None:
    """The search region of a non-detection frame: the selector's radius when it searched, else
    an estimate from the ball size, about the search centre (the ball when there was none)."""
    sr = float(result.search_radius or 0.0)
    scx = float(result.search_cx or centre[0])
    scy = float(result.search_cy or centre[1])
    draw_sr = (int(round(sr)) if sr > 1
               else max(int(round(anchor_radius * _SEARCH_RADIUS_FALLBACK_SCALE)), _SEARCH_RADIUS_FALLBACK_MIN))
    draw_scx = int(round(scx)) if scx > 0 else centre_px[0]
    draw_scy = int(round(scy)) if scy > 0 else centre_px[1]
    cv2.circle(vis, (draw_scx, draw_scy), draw_sr, COLOR_SEARCH, 1, cv2.LINE_AA)
    cv2.drawMarker(vis, (draw_scx, draw_scy), COLOR_SEARCH, cv2.MARKER_CROSS, SEARCH_MARKER_SIZE, 1)
    cv2.putText(vis, f"r={draw_sr}", (draw_scx + draw_sr + 3, draw_scy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_SEARCH, 1)


def _aged_color_and_thickness(age: float) -> Tuple[Color, int]:
    ratio = max(0.0, 1.0 - (age / float(_TRAIL_MAX_AGE_FRAMES)))
    color = (int(COLOR_TRAIL[0] * ratio), int(COLOR_TRAIL[1] * ratio), int(COLOR_TRAIL[2] * ratio))
    return color, max(1, int(_TRAIL_THICKNESS * ratio))


def _draw_straight_trail(vis: np.ndarray, frame_index: int, points: Sequence[TrailSample]) -> None:
    for i in range(1, len(points)):
        color, th = _aged_color_and_thickness(frame_index - points[i][0])
        pt1 = (int(points[i - 1][1]), int(points[i - 1][2]))
        pt2 = (int(points[i][1]), int(points[i][2]))
        if math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1]) < _STRAIGHT_SEGMENT_MAX_PX:
            cv2.line(vis, pt1, pt2, color, th, cv2.LINE_AA)


# A spline fitted through trail samples: x, y and age (frames) at each fine sample.
SplineTrail = Tuple[np.ndarray, np.ndarray, np.ndarray]


def _fit_spline_trail(frame_index: int, points: Sequence[TrailSample]) -> Optional[SplineTrail]:
    """A smoothed B-spline through the samples, or None when scipy cannot fit one
    (too few distinct points, or a degenerate set)."""
    pts_arr = np.array([[pt[1], pt[2]] for pt in points])
    # A 1D Gaussian smooth removes the pixel jitter that makes the spline wave.
    x_smooth = scipy.ndimage.gaussian_filter1d(pts_arr[:, 0], sigma=_SPLINE_SMOOTH_SIGMA)
    y_smooth = scipy.ndimage.gaussian_filter1d(pts_arr[:, 1], sigma=_SPLINE_SMOOTH_SIGMA)
    try:
        # s > 0 lets the spline relax instead of forcing it through every point.
        tck, u = scipy.interpolate.splprep([x_smooth, y_smooth], s=len(x_smooth))
    except (TypeError, ValueError):      # splprep: "m > k must hold", "Error on input data"
        return None
    num_samples = max(_SPLINE_MIN_SAMPLES, len(points) * _SPLINE_SAMPLES_PER_POINT)
    u_fine = np.linspace(0, 1.0, num_samples)
    x_fine, y_fine = scipy.interpolate.splev(u_fine, tck)
    ages_fine = np.interp(u_fine, u, [frame_index - pt[0] for pt in points])
    return x_fine, y_fine, ages_fine


def _draw_spline_trail(vis: np.ndarray, spline: SplineTrail) -> None:
    x_fine, y_fine, ages_fine = spline
    for i in range(1, len(x_fine)):
        color, th = _aged_color_and_thickness(ages_fine[i])
        pt1 = (int(x_fine[i - 1]), int(y_fine[i - 1]))
        pt2 = (int(x_fine[i]), int(y_fine[i]))
        if math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1]) < _SPLINE_SEGMENT_MAX_PX:
            cv2.line(vis, pt1, pt2, color, th, cv2.LINE_AA)


class RawMotionTrails:
    """Comet trails of boost-mask blobs inside the motion ROIs (non-probe style only).

    Blobs are associated frame to frame by nearest distance into trails that
    fade with age; the head of a live trail is drawn as a dot.  The spline
    smoothing needs scipy; without it (or when a fit fails) segments are straight.
    """

    def __init__(self, frame_w: int) -> None:
        self.max_match_dist = max(_MATCH_MAX_DIST_MIN_PX, frame_w * _MATCH_MAX_DIST_FRAC)
        self.trails: List[List[TrailSample]] = []
        self._warned_no_scipy = False

    @staticmethod
    def _blobs(rois: Optional[Sequence[Sequence[int]]], boost_mask: Optional[np.ndarray]) -> List[Tuple[float, float, float]]:
        """Boost-blob anchors (bottom-centre, like the detector box) per ROI, or the ROI centre without blobs."""
        blobs: List[Tuple[float, float, float]] = []
        for rx1, ry1, rx2, ry2 in rois or ():
            irx1, iry1, irx2, iry2 = int(rx1), int(ry1), int(rx2), int(ry2)
            found = False
            if boost_mask is not None and boost_mask.size > 0:
                roi_mask = boost_mask[iry1:iry2, irx1:irx2]
                if roi_mask.size > 0 and roi_mask.max() > 0:
                    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in contours:
                        area = cv2.contourArea(cnt)
                        if area < _BLOB_MIN_AREA:
                            continue
                        bx, by, bw, bh = cv2.boundingRect(cnt)
                        cx = irx1 + bx + bw / 2.0
                        cy = iry1 + by + bh
                        radius = min(_BLOB_RADIUS_MAX, max(_BLOB_RADIUS_MIN, math.sqrt(area / math.pi)))
                        blobs.append((float(cx), float(cy), float(radius)))
                        found = True
            if not found:
                radius = min(_BLOB_RADIUS_MAX, max(_BLOB_RADIUS_MIN, min(rx2 - rx1, ry2 - ry1) * _ROI_FALLBACK_RADIUS_FRAC))
                blobs.append((float((rx1 + rx2) / 2.0), float((ry1 + ry2) / 2.0), float(radius)))
        return blobs

    def advance(self, frame_index: int, rois: Optional[Sequence[Sequence[int]]],
                boost_mask: Optional[np.ndarray]) -> None:
        unmatched = self._blobs(rois, boost_mask)
        for trail in self.trails:
            last_frame, last_x, last_y, _ = trail[-1]
            if frame_index - last_frame > _MATCH_MAX_GAP_FRAMES:
                continue
            best_dist, best = self.max_match_dist, None
            for blob in unmatched:
                dist = math.hypot(blob[0] - last_x, blob[1] - last_y)
                if dist < best_dist:
                    best_dist, best = dist, blob
            if best is not None:
                trail.append((frame_index, best[0], best[1], best[2]))
                unmatched.remove(best)
        for blob in unmatched:
            self.trails.append([(frame_index, blob[0], blob[1], blob[2])])
        if frame_index % _TRAIL_PRUNE_EVERY == 0:
            self.trails = [t for t in self.trails if frame_index - t[-1][0] <= _TRAIL_MAX_AGE_FRAMES]

    @staticmethod
    def _spline_points(valid: Sequence[TrailSample]) -> List[TrailSample]:
        """The samples with those too dense for the spline dropped, the current head always kept."""
        unique = [valid[0]]
        for pt in valid[1:]:
            if math.hypot(pt[1] - unique[-1][1], pt[2] - unique[-1][2]) > _UNIQUE_POINT_MIN_DIST_PX:
                unique.append(pt)
        if valid[-1] not in unique:
            unique.append(valid[-1])
        return unique

    def _draw_trail(self, vis: np.ndarray, frame_index: int, unique: Sequence[TrailSample]) -> None:
        """A spline through the samples when scipy can fit one, else straight segments."""
        if scipy is None and not self._warned_no_scipy:
            print("[debug] scipy not installed: raw-motion trails are drawn as straight segments")
            self._warned_no_scipy = True
        spline = None
        if scipy is not None and len(unique) >= _SPLINE_MIN_POINTS:
            spline = _fit_spline_trail(frame_index, unique)
        if spline is None:
            _draw_straight_trail(vis, frame_index, unique)
        else:
            _draw_spline_trail(vis, spline)

    @staticmethod
    def _draw_head(vis: np.ndarray, valid: Sequence[TrailSample]) -> None:
        """The dot at a live trail's head, sized by a running average of its recent blob radii."""
        radius = valid[-_HEAD_RADIUS_SAMPLES:][0][3]
        for pt in valid[-_HEAD_RADIUS_SAMPLES:][1:]:
            radius = _HEAD_RADIUS_EMA_NEW * pt[3] + _HEAD_RADIUS_EMA_KEEP * radius
        head = (int(valid[-1][1]), int(valid[-1][2]))
        cv2.circle(vis, head, max(2, int(radius)), COLOR_TRAIL, -1)

    def draw(self, vis: np.ndarray, frame_index: int) -> None:
        for trail in self.trails:
            valid = [pt for pt in trail if frame_index - pt[0] <= _TRAIL_MAX_AGE_FRAMES]
            if len(valid) < _TRAIL_MIN_SAMPLES:
                continue       # isolated dots and very short noise trails
            self._draw_trail(vis, frame_index, self._spline_points(valid))
            if frame_index - valid[-1][0] == 0:
                self._draw_head(vis, valid)
