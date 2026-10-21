"""Court geometry derived from the court detector's keypoints.

Keypoint lookup, the offset side guides, the court polygon, the ground-plane
homography and the quarter-pixel cache key live here so that the motion,
court-event and rendering modules share one definition without depending on
the drawing layer.  Nothing in this module draws.

Keypoints are a flat ``[x0, y0, x1, y1, ...]`` sequence in image pixels in the
detector's semantic order (see ``CourtDetector``): 0/1 far baseline corners,
2/3 near baseline corners, 4/6 far singles, 5/7 near singles, 8/9 far service
line, 10/11 near service line, 12/13 centre marks.  A point at ``(0, 0)`` or
below is "not detected".
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

CourtKeypoints = Sequence[float]
Point = Tuple[float, float]
# A side guide: ``base`` is the court corner, ``anchor`` the corner shifted
# outward by the alley width, ``dir`` the direction of the outer sideline.
SideGuide = Dict[str, np.ndarray]
GroundModel = Dict[str, np.ndarray]  # "H_i2c": image -> unit court, "H_c2i": the inverse

# Keypoint indices in the detector's semantic order.
KP_FAR_LEFT = 0
KP_FAR_RIGHT = 1
KP_NEAR_LEFT = 2
KP_NEAR_RIGHT = 3
KP_NEAR_LEFT_SINGLES = 5
KP_NEAR_RIGHT_SINGLES = 7
# The four corners that bound the court polygon: top-left, bottom-left,
# bottom-right, top-right.
CORNER_ORDER = (KP_FAR_LEFT, KP_NEAR_LEFT, KP_NEAR_RIGHT, KP_FAR_RIGHT)
# Keypoint index pairs that form the court lines, in the semantic order above.
LINE_PAIRS_14: Tuple[Tuple[int, int], ...] = (
    (0, 4), (4, 6), (6, 1),
    (0, 2), (1, 3),
    (2, 5), (5, 7), (7, 3),
    (4, 8), (8, 10), (10, 5),
    (6, 9), (9, 11), (11, 7),
    (8, 12), (12, 9),
    (10, 13), (13, 11),
    (12, 13),
)
# Values needed for the eight corner/singles points every geometry helper reads.
MIN_KEYPOINT_VALUES = 16
# Court keypoints are cached at this resolution: a quarter of a pixel.
KEYPOINT_KEY_SCALE = 4.0

# ITF court dimensions (metres) and their fractions of the unit court, whose
# x runs 0..1 across the doubles width and y runs 0..1 along the length with
# the net at NET_V.
COURT_LENGTH_M = 23.77
DOUBLES_WIDTH_M = 10.97
SINGLES_WIDTH_M = 8.23
SERVICE_LINE_FROM_NET_M = 6.40
SINGLES_INSET = (DOUBLES_WIDTH_M - SINGLES_WIDTH_M) / (2.0 * DOUBLES_WIDTH_M)
SERVICE_DEPTH = SERVICE_LINE_FROM_NET_M / COURT_LENGTH_M
NET_V = 0.5
# Unit-court corners in the same TL, TR, BR, BL order as the image corners
# handed to getPerspectiveTransform.
_UNIT_COURT_CORNERS = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)

# A court polygon with less area than this (px^2) is degenerate.
_MIN_POLYGON_AREA_PX = 10.0
# A side guide needs at least this alley width (px) and a non-zero direction.
_MIN_ALLEY_WIDTH_PX = 1.0
_MIN_DIRECTION_NORM = 1e-6
_MIN_HOMOGRAPHY_DENOMINATOR = 1e-8
# An "infinite" line is drawn this far beyond the frame before clipping.
_LINE_EXTENSION_PX = 1000.0
# Axis spans are clipped to these fractions of the frame when the court
# keypoints give an implausibly small court.
_MIN_X_SPAN_FRAC = 0.35
_MIN_Y_SPAN_FRAC = 0.25
_MIN_SPAN_PX = 1.0


def court_keypoint(kps: Optional[CourtKeypoints], idx: int, *, require_finite: bool) -> Optional[Point]:
    """Keypoint ``idx`` as ``(x, y)``, or None when it is missing or not detected.

    Missing means the sequence is None or too short; not detected means both
    coordinates are at or below zero (the detector's "absent" value).  With
    ``require_finite`` a NaN or infinite coordinate is also None; without it
    the values are returned as they are.
    """
    if kps is None:
        return None
    base = int(idx) * 2
    if base + 1 >= len(kps):
        return None
    x = float(kps[base])
    y = float(kps[base + 1])
    if require_finite and not (np.isfinite(x) and np.isfinite(y)):
        return None
    if x <= 0.0 and y <= 0.0:
        return None
    return x, y


def court_keypoints_key(keypoints: Optional[CourtKeypoints]) -> Optional[Tuple[int, ...]]:
    """Quarter-pixel cache key of the first eight court keypoints.

    None when fewer than ``MIN_KEYPOINT_VALUES`` values are present or any of
    them is non-finite, so callers need no try/except around the conversion.
    """
    if keypoints is None or len(keypoints) < MIN_KEYPOINT_VALUES:
        return None
    values = np.asarray(keypoints[:MIN_KEYPOINT_VALUES], dtype=np.float64)
    if not np.isfinite(values).all():
        return None
    return tuple(int(round(float(value) * KEYPOINT_KEY_SCALE)) for value in values)


def _keypoint_array(kps: Optional[CourtKeypoints], idx: int) -> Optional[np.ndarray]:
    point = court_keypoint(kps, idx, require_finite=True)
    return None if point is None else np.array(point, dtype=np.float32)


def build_court_side_guides(court_keypoints: Optional[CourtKeypoints]) -> Optional[Dict[str, SideGuide]]:
    """Lines one alley width outside each outer sideline, keyed "left"/"right".

    Each guide is parallel to its outer sideline and offset horizontally from
    the near corner by the near-side alley width (corner to singles mark).
    None when the six keypoints it needs are not all detected, or when
    neither side yields a usable line.
    """
    far_left = _keypoint_array(court_keypoints, KP_FAR_LEFT)
    far_right = _keypoint_array(court_keypoints, KP_FAR_RIGHT)
    near_left = _keypoint_array(court_keypoints, KP_NEAR_LEFT)
    near_right = _keypoint_array(court_keypoints, KP_NEAR_RIGHT)
    near_left_singles = _keypoint_array(court_keypoints, KP_NEAR_LEFT_SINGLES)
    near_right_singles = _keypoint_array(court_keypoints, KP_NEAR_RIGHT_SINGLES)
    if any(p is None for p in (far_left, far_right, near_left, near_right,
                               near_left_singles, near_right_singles)):
        return None

    guides: Dict[str, SideGuide] = {}
    for side, base, far, singles, outward in (
        ("left", near_left, far_left, near_left_singles, -1.0),
        ("right", near_right, far_right, near_right_singles, 1.0),
    ):
        alley = float(np.hypot(*(singles - base)))
        direction = far - base
        if alley >= _MIN_ALLEY_WIDTH_PX and float(np.hypot(direction[0], direction[1])) >= _MIN_DIRECTION_NORM:
            guides[side] = {
                "base": base,
                "anchor": base + np.array([outward * alley, 0.0], dtype=np.float32),
                "dir": direction,
            }
    return guides or None


def clip_infinite_line_to_frame(anchor: np.ndarray, direction: np.ndarray,
                                w: int, h: int) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """The segment of the line through ``anchor`` along ``direction`` that lies
    inside a ``w`` x ``h`` frame, as integer end points; None if it misses."""
    if float(np.hypot(direction[0], direction[1])) < _MIN_DIRECTION_NORM:
        return None
    reach = float(max(w, h) * 4.0 + _LINE_EXTENSION_PX)
    p1 = (int(round(anchor[0] - direction[0] * reach)), int(round(anchor[1] - direction[1] * reach)))
    p2 = (int(round(anchor[0] + direction[0] * reach)), int(round(anchor[1] + direction[1] * reach)))
    ok, q1, q2 = cv2.clipLine((0, 0, int(w), int(h)), p1, p2)
    if not ok:
        return None
    return q1, q2


def court_axis_spans(kps: Optional[CourtKeypoints], w: int, h: int) -> Tuple[float, float]:
    """Pixel extents of the court along x (near baseline) and y (left sideline).

    Falls back to the frame size when the keypoints are missing, and clips to
    at least ``_MIN_X_SPAN_FRAC`` / ``_MIN_Y_SPAN_FRAC`` of the frame so a bad
    detection cannot make the trail guards absurdly tight.
    """
    if kps is None or len(kps) < MIN_KEYPOINT_VALUES:
        return float(w), float(h)
    far_left_y = float(kps[2 * KP_FAR_LEFT + 1])
    near_left_x, near_left_y = float(kps[2 * KP_NEAR_LEFT]), float(kps[2 * KP_NEAR_LEFT + 1])
    near_right_x = float(kps[2 * KP_NEAR_RIGHT])
    x_span = abs(near_right_x - near_left_x)
    y_span = abs(near_left_y - far_left_y)
    if x_span < _MIN_SPAN_PX:
        x_span = float(w)
    if y_span < _MIN_SPAN_PX:
        y_span = float(h)
    x_span = float(np.clip(x_span, _MIN_X_SPAN_FRAC * w, w))
    y_span = float(np.clip(y_span, _MIN_Y_SPAN_FRAC * h, h))
    return x_span, y_span


def homography_apply(H: np.ndarray, x: float, y: float) -> Optional[Point]:
    """Apply a 3x3 homography to a 2D point; None at a vanishing denominator."""
    den = float(H[2, 0] * x + H[2, 1] * y + H[2, 2])
    if abs(den) < _MIN_HOMOGRAPHY_DENOMINATOR:
        return None
    ox = float(H[0, 0] * x + H[0, 1] * y + H[0, 2]) / den
    oy = float(H[1, 0] * x + H[1, 1] * y + H[1, 2]) / den
    return ox, oy


def build_ordered_court_polygon(kps: Optional[CourtKeypoints]) -> Optional[np.ndarray]:
    """The four court corners as an OpenCV contour in ``CORNER_ORDER``
    (TL, BL, BR, TR), or None when a corner is missing or the area is degenerate."""
    pts: List[Point] = []
    for corner_idx in CORNER_ORDER:
        p = court_keypoint(kps, corner_idx, require_finite=False)
        if p is None:
            return None
        pts.append(p)
    poly = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
    if abs(cv2.contourArea(poly)) < _MIN_POLYGON_AREA_PX:
        return None
    return poly


def build_ground_projection_model(kps: Optional[CourtKeypoints]) -> Optional[GroundModel]:
    """Homographies between the image and the unit court, from the four corners."""
    ordered_poly = build_ordered_court_polygon(kps)
    if ordered_poly is None:
        return None
    tl, bl, br, tr = (tuple(map(float, corner)) for corner in ordered_poly[:, 0, :])
    src = np.array([tl, tr, br, bl], dtype=np.float32)
    try:
        image_to_court = cv2.getPerspectiveTransform(src, _UNIT_COURT_CORNERS).astype(np.float64)
        court_to_image = cv2.getPerspectiveTransform(_UNIT_COURT_CORNERS, src).astype(np.float64)
    except cv2.error:
        return None
    return {"H_i2c": image_to_court, "H_c2i": court_to_image}


def last_valid_court_keypoints(kps_by_frame: Sequence[Optional[CourtKeypoints]],
                               min_values: int = MIN_KEYPOINT_VALUES) -> Optional[CourtKeypoints]:
    """The latest frame's keypoints that hold at least ``min_values`` values, or None.

    The one definition of "the court as last seen" that the court polygon, the
    tracking JSON and pass 2 share.
    """
    for kps in reversed(kps_by_frame):
        if kps is not None and len(kps) >= min_values:
            return kps
    return None


def build_court_polygon(all_court_kps: Sequence[Optional[CourtKeypoints]]) -> Tuple[Optional[np.ndarray], Optional[CourtKeypoints]]:
    """The court polygon from the last frame with valid keypoints, and those keypoints.

    Prefers the ordered four-corner polygon; falls back to the convex hull of
    every detected point.  Reports what it built on stdout.
    """
    last_kps = last_valid_court_keypoints(all_court_kps)
    if last_kps is None:
        print("[court polygon] No court keypoints found")
        return None, None

    court_poly = build_ordered_court_polygon(last_kps)
    if court_poly is not None:
        tl, bl, br, tr = (court_poly[i, 0].tolist() for i in range(4))
        print(f"[court polygon] Ordered corners [0,2,3,1]: TL={tl}, BL={bl}, BR={br}, TR={tr}")
        bx, by, bw, bh = cv2.boundingRect(court_poly)
        print(f"[court polygon] bbox=({bx},{by},{bw}x{bh})")
        test_cx = bx + bw / 2.0
        test_cy = by + bh / 2.0
        test_dist = cv2.pointPolygonTest(court_poly, (test_cx, test_cy), True)
        print(f"[court polygon] Center ({test_cx:.0f},{test_cy:.0f}) dist={test_dist:.1f} (positive=inside)")
        return court_poly, last_kps

    pts = [
        (last_kps[i], last_kps[i + 1])
        for i in range(0, len(last_kps) - 1, 2)
        if last_kps[i] > 0 or last_kps[i + 1] > 0
    ]
    if len(pts) >= 4:
        court_poly = cv2.convexHull(np.array(pts, dtype=np.float32).reshape(-1, 1, 2))
        bx, by, bw, bh = cv2.boundingRect(court_poly)
        print(f"[court polygon] Fallback hull ({len(pts)} points), bbox=({bx},{by},{bw}x{bh})")
        return court_poly, last_kps

    print(f"[court polygon] Not enough valid points ({len(pts)}), skipping")
    return None, last_kps
