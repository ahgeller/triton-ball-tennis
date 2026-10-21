"""Drawing the court onto frames: keypoints and their lines, the side guides, the net line
through the homography and the court polygon.  The geometry itself comes from
``court_geometry``; the colours and canvas primitives from ``rendering``.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import Config
from .court_geometry import (
    LINE_PAIRS_14, GroundModel, build_court_side_guides, clip_infinite_line_to_frame, homography_apply,
)
from .rendering import (
    COLOR_COURT_INDEX, COLOR_COURT_LINE, COLOR_COURT_POLYGON, COLOR_NET_LINE, COLOR_OUTLINE, COLOR_PLAYER_BOX, Color,
    Point,
)

# Fewest court keypoints that are joined by lines.
_COURT_MIN_LINE_POINTS = 4


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
