"""The motion-debug video: masks and the selected ball painted on each frame (the raw-motion
trails drawn over them are ``motion_trails``).

Drawing only.  Nothing here feeds the tracker, the ROI tracker or the selector.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from ball_in_play_selector import FrameResult

from ..config import Config
from ..court.court_drawing import draw_court_side_guides
from .rendering import MOTION_DEBUG_SOURCE_COLOR, SEARCH_MARKER_SIZE

Color = Tuple[int, int, int]
Box = Tuple[int, int, int, int]

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
_LEGEND_WIDTH_PX, _LEGEND_HEIGHT_PX = 900, 58
# Display-only component box limits; these never filter tracking evidence.
_COMPONENT_MIN_AREA, _COMPONENT_MAX_AREA = 2, 260
_COMPONENT_MAX_DIM, _COMPONENT_MAX_ASPECT, _COMPONENT_MIN_FILL = 38, 5.5, 0.14
_COMPONENT_BOX_LIMIT = 100
# Ball overlay.
_ESTIMATED_BOX_MIN_RADIUS = 6
_SEARCH_RADIUS_FALLBACK_SCALE, _SEARCH_RADIUS_FALLBACK_MIN = 2.5, 20


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
