"""The optional top-right court minimap: completed rally legs and player positions.

Drawing only. The legs come from ``rally_legs.assemble_rally_legs`` and the
image-to-court homography from ``court_geometry.build_ground_projection_model``.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import cv2
import numpy as np

from .court_geometry import (
    COURT_LENGTH_M,
    DOUBLES_WIDTH_M,
    NET_V,
    SERVICE_DEPTH,
    SINGLES_INSET,
    homography_apply,
)
from .court_scene import Color, ContactEvent, CourtPoint, PlayerBoxes, RallyLeg

# --- Minimap drawing ----------------------------------------------------------
_PANEL_BG: Color = (42, 42, 42)
_PANEL_COURT: Color = (28, 72, 42)
_PANEL_LINE: Color = (235, 235, 235)
_PANEL_NET: Color = (90, 180, 255)
_PANEL_BORDER: Color = (230, 230, 230)
_SHADOW: Color = (12, 12, 12)
_MARKER_OUTLINE: Color = (15, 15, 15)
_INFERRED_BOUNCE_COLOR: Color = (0, 165, 255)
_CONFIRMED_BOUNCE_COLOR: Color = (0, 255, 255)
_DEFAULT_PHASE_COLOR: Color = (70, 220, 255)
_PHASE_COLORS: Dict[str, Color] = {
    "serve": (255, 210, 60),
    "rally": _DEFAULT_PHASE_COLOR,
    "fault": (70, 70, 255),
}
_PLAYER_COLORS: Tuple[Color, ...] = (
    (255, 110, 60), (70, 90, 255), (255, 80, 220), (80, 230, 120),
)
_PANEL_POINT_BOUNDS = (-0.08, 1.08)
_PANEL_OPACITY = 0.90
_FRAME_OPACITY = 0.10  # video showing through the panel
_ARC_STEPS = 10  # dots along the drawn leg chord
_DASH_PX = 7.0


class _PanelRect(NamedTuple):
    left: int
    top: int
    right: int
    bottom: int


def _panel_point(
    point: Tuple[float, float],
    left: int,
    top: int,
    right: int,
    bottom: int,
) -> Tuple[int, int]:
    low, high = _PANEL_POINT_BOUNDS
    u = float(np.clip(point[0], low, high))
    v = float(np.clip(point[1], low, high))
    return (
        int(round(left + u * (right - left))),
        int(round(top + v * (bottom - top))),
    )


def _draw_dashed_line(
    image: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
    color: Color,
    thickness: int = 1,
) -> None:
    distance = float(np.hypot(end[0] - start[0], end[1] - start[1]))
    if distance < 1.0:
        return
    steps = max(1, int(distance // _DASH_PX))
    for index in range(steps):
        if index % 2:
            continue
        a = index / steps
        b = min(1.0, (index + 1) / steps)
        p0 = (int(round(start[0] + a * (end[0] - start[0]))),
              int(round(start[1] + a * (end[1] - start[1]))))
        p1 = (int(round(start[0] + b * (end[0] - start[0]))),
              int(round(start[1] + b * (end[1] - start[1]))))
        cv2.line(image, p0, p1, color, thickness, cv2.LINE_AA)


def _visible_completed_legs(
    rally_legs: Sequence[RallyLeg],
    contact_events: Optional[Iterable[ContactEvent]],
    current_frame: int,
    fps: float,
) -> List[RallyLeg]:
    """The latest completed leg, until about a second passes or play moves on."""
    completed = [
        leg for leg in rally_legs
        if leg["end_frame"] <= current_frame
    ]
    if not completed:
        return []

    latest = completed[-1]
    if current_frame - latest["end_frame"] > int(round(float(fps))):
        return []
    if any(
        latest["end_frame"] < int(event.frame) <= current_frame
        for event in contact_events or ()
    ):
        return []

    newest_started = next(
        (leg for leg in reversed(rally_legs) if leg["start_frame"] <= current_frame),
        None,
    )
    if (
        newest_started is not None
        and newest_started["start_frame"] > latest["end_frame"]
        and newest_started["end_frame"] > current_frame
    ):
        return []
    return [latest]


def _minimap_layout(frame_shape: Tuple[int, ...]) -> Optional[Tuple[int, int, int, int, _PanelRect]]:
    """Return (x0, y0, panel_w, panel_h, court rect) or None if it does not fit."""
    frame_h, frame_w = frame_shape[:2]
    panel_w = int(np.clip(round(frame_w * 0.10), 130, 180))
    panel_h = int(round(panel_w * 2.08))
    margin = max(8, int(round(frame_w * 0.006)))
    x0 = frame_w - panel_w - margin
    y0 = margin
    if x0 < 0 or y0 + panel_h > frame_h:
        return None
    court_w = panel_w - 36
    court_h = int(round(court_w * COURT_LENGTH_M / DOUBLES_WIDTH_M))
    left = (panel_w - court_w) // 2
    top = (panel_h - court_h) // 2
    return x0, y0, panel_w, panel_h, _PanelRect(left, top, left + court_w, top + court_h)


def _draw_court_lines(panel: np.ndarray, rect: _PanelRect) -> None:
    def line(a: CourtPoint, b: CourtPoint, color: Color, thickness: int) -> None:
        cv2.line(panel, _panel_point(a, *rect), _panel_point(b, *rect),
                 color, thickness, cv2.LINE_AA)

    corners = (rect.left, rect.top), (rect.right, rect.bottom)
    cv2.rectangle(panel, *corners, _PANEL_COURT, -1)
    cv2.rectangle(panel, *corners, _PANEL_LINE, 1, cv2.LINE_AA)
    for u in (SINGLES_INSET, 1.0 - SINGLES_INSET):
        line((u, 0.0), (u, 1.0), _PANEL_LINE, 1)
    line((0.0, NET_V), (1.0, NET_V), _PANEL_NET, 2)
    for v in (NET_V - SERVICE_DEPTH, NET_V + SERVICE_DEPTH):
        line((SINGLES_INSET, v), (1.0 - SINGLES_INSET, v), _PANEL_LINE, 1)
    line((0.5, NET_V - SERVICE_DEPTH), (0.5, NET_V + SERVICE_DEPTH), _PANEL_LINE, 1)
    cv2.putText(panel, "COURT", (7, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                _PANEL_LINE, 1, cv2.LINE_AA)


def _draw_leg_landing(
    panel: np.ndarray, leg: RallyLeg, end_point: Tuple[int, int], color: Color
) -> None:
    if leg["outcome"] == "bounce":
        if leg["quality"] == "inferred":
            x, y = end_point
            diamond = np.array(((x, y - 5), (x + 5, y), (x, y + 5), (x - 5, y)), np.int32)
            cv2.polylines(panel, [diamond], True, _INFERRED_BOUNCE_COLOR, 2, cv2.LINE_AA)
            cv2.putText(panel, "?", (x + 5, y - 3), cv2.FONT_HERSHEY_SIMPLEX,
                        0.30, _INFERRED_BOUNCE_COLOR, 1, cv2.LINE_AA)
        else:
            cv2.circle(panel, end_point, 5, _MARKER_OUTLINE, 2, cv2.LINE_AA)
            cv2.circle(panel, end_point, 4, _CONFIRMED_BOUNCE_COLOR, 2, cv2.LINE_AA)
    elif leg["outcome"] == "volley":
        cv2.putText(panel, "V", (end_point[0] + 4, end_point[1] - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)


def _draw_leg(panel: np.ndarray, leg: RallyLeg, rect: _PanelRect) -> None:
    start_point = _panel_point(leg["start"], *rect)
    end_point = _panel_point(leg["end"], *rect)
    color = _PHASE_COLORS.get(leg["phase"], _DEFAULT_PHASE_COLOR)
    cv2.line(panel, start_point, end_point, _SHADOW, 3, cv2.LINE_AA)
    if leg["quality"] == "inferred" or leg.get("start_quality") == "inferred":
        _draw_dashed_line(panel, start_point, end_point, color, 2)
    else:
        cv2.line(panel, start_point, end_point, color, 2, cv2.LINE_AA)

    # q(s)=4s(1-s) is a relative height cue only. Court coordinates stay
    # on the straight, physically valid plan-view chord.
    for step in range(1, _ARC_STEPS):
        fraction = step / float(_ARC_STEPS)
        relative_height = 4.0 * fraction * (1.0 - fraction)
        point = (
            int(round(start_point[0] + fraction * (end_point[0] - start_point[0]))),
            int(round(start_point[1] + fraction * (end_point[1] - start_point[1]))),
        )
        radius = 1 + int(round(relative_height * (2 if leg["phase"] == "serve" else 1)))
        cv2.circle(panel, point, radius, color, -1, cv2.LINE_AA)

    _draw_leg_landing(panel, leg, end_point, color)


def _draw_phase_label(panel: np.ndarray, leg: RallyLeg, panel_w: int) -> None:
    label = {"serve": "SERVE", "fault": "FAULT"}.get(leg["phase"], "RALLY")
    if leg["outcome"] == "volley":
        label = "VOLLEY"
    cv2.putText(panel, label, (panel_w - 48, 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.30, _PHASE_COLORS.get(leg["phase"], _DEFAULT_PHASE_COLOR),
                1, cv2.LINE_AA)


def _draw_player_dots(
    panel: np.ndarray,
    H: np.ndarray,
    player_boxes: Optional[PlayerBoxes],
    rect: _PanelRect,
) -> None:
    for index, (_player_id, box) in enumerate(
        sorted((player_boxes or {}).items(), key=lambda item: str(item[0]))
    ):
        if box is None or len(box) < 4:
            continue
        x1, _, x2, y2 = map(float, box[:4])
        projected = homography_apply(H, 0.5 * (x1 + x2), y2)
        if projected is None:
            continue
        center = _panel_point(projected, *rect)
        color = _PLAYER_COLORS[index % len(_PLAYER_COLORS)]
        cv2.circle(panel, center, 5, _MARKER_OUTLINE, -1, cv2.LINE_AA)
        cv2.circle(panel, center, 3, color, -1, cv2.LINE_AA)


def draw_court_minimap(
    frame: Optional[np.ndarray],
    ground_model: Optional[Mapping[str, np.ndarray]],
    player_boxes: Optional[PlayerBoxes],
    rally_legs: Sequence[RallyLeg],
    current_frame: int,
    fps: float,
    contact_events: Optional[Iterable[ContactEvent]] = (),
) -> None:
    """Draw the optional top-right 2D court minimap in-place."""
    if frame is None or ground_model is None:
        return
    H = ground_model.get("H_i2c")
    if H is None:
        return
    layout = _minimap_layout(frame.shape)
    if layout is None:
        return
    x0, y0, panel_w, panel_h, rect = layout

    panel = np.full((panel_h, panel_w, 3), _PANEL_BG, dtype=np.uint8)
    _draw_court_lines(panel, rect)
    visible_legs = _visible_completed_legs(rally_legs, contact_events, current_frame, fps)
    for leg in visible_legs:
        _draw_leg(panel, leg, rect)
    if visible_legs:
        _draw_phase_label(panel, visible_legs[-1], panel_w)
    _draw_player_dots(panel, H, player_boxes, rect)

    roi = frame[y0:y0 + panel_h, x0:x0 + panel_w]
    cv2.addWeighted(panel, _PANEL_OPACITY, roi, _FRAME_OPACITY, 0.0, roi)
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w - 1, y0 + panel_h - 1),
                  _PANEL_BORDER, 1, cv2.LINE_AA)
