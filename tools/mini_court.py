"""Reusable top-right mini-court overlay used by ball-tracking tools.

Standalone — no dependency on the archived 3D mapping pipeline. Renders the
doubles court outline + singles tramlines + net + service boxes inside a
fixed panel in the upper-right corner of a frame, and provides a helper for
mapping court meters <-> panel pixels (so a labeling tool can convert mouse
clicks on the mini-court back to court (X, Y) in meters).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Optional, Tuple


COURT_WIDTH_M = 10.97          # doubles
COURT_LENGTH_M = 23.77
SINGLES_WIDTH_M = 8.23
NET_LINE_Y_M = COURT_LENGTH_M / 2.0
SERVICE_LINE_OFFSET_M = 6.4    # distance from net to each service line


@dataclass
class MiniCourtLayout:
    panel_x1: int
    panel_y1: int
    panel_x2: int
    panel_y2: int
    court_x1: int
    court_y1: int
    court_x2: int
    court_y2: int
    scale: float
    ox: float
    oy: float

    def world_to_panel(self, xm: float, ym: float) -> Tuple[int, int]:
        return (
            int(round(self.ox + xm * self.scale)),
            int(round(self.oy + ym * self.scale)),
        )

    def panel_to_world(self, px: int, py: int) -> Optional[Tuple[float, float]]:
        if self.scale <= 0:
            return None
        xm = (px - self.ox) / self.scale
        ym = (py - self.oy) / self.scale
        return xm, ym

    def contains(self, px: int, py: int) -> bool:
        return self.panel_x1 <= px <= self.panel_x2 and self.panel_y1 <= py <= self.panel_y2


def compute_layout(
    frame_w: int,
    frame_h: int,
    panel_w: int = 260,
    panel_h: int = 440,
    margin: int = 18,
    pad: int = 14,
    header_h: int = 24,
    footer_h: int = 30,
) -> MiniCourtLayout:
    panel_x1 = frame_w - panel_w - margin
    panel_y1 = margin
    panel_x2 = panel_x1 + panel_w
    panel_y2 = panel_y1 + panel_h
    court_x1 = panel_x1 + pad
    court_y1 = panel_y1 + pad + header_h
    court_x2 = panel_x2 - pad
    court_y2 = panel_y2 - pad - footer_h
    avail_w = court_x2 - court_x1
    avail_h = court_y2 - court_y1
    scale = min(avail_w / COURT_WIDTH_M, avail_h / COURT_LENGTH_M)
    draw_w = COURT_WIDTH_M * scale
    draw_h = COURT_LENGTH_M * scale
    cx = (court_x1 + court_x2) / 2.0
    cy = (court_y1 + court_y2) / 2.0
    ox = cx - draw_w / 2.0
    oy = cy - draw_h / 2.0
    return MiniCourtLayout(
        panel_x1=panel_x1, panel_y1=panel_y1, panel_x2=panel_x2, panel_y2=panel_y2,
        court_x1=court_x1, court_y1=court_y1, court_x2=court_x2, court_y2=court_y2,
        scale=scale, ox=ox, oy=oy,
    )


def draw_mini_court(
    frame,
    cv2,
    layout: Optional[MiniCourtLayout] = None,
    bg_color: Tuple[int, int, int] = (15, 25, 25),
    bg_alpha: float = 0.85,
    border_color: Tuple[int, int, int] = (220, 220, 220),
    line_color: Tuple[int, int, int] = (255, 255, 255),
    singles_color: Tuple[int, int, int] = (220, 220, 220),
    net_color: Tuple[int, int, int] = (0, 220, 255),
    service_color: Tuple[int, int, int] = (180, 180, 180),
) -> MiniCourtLayout:
    """Draw the dark panel + court markings in the top-right of `frame`.
    Returns the layout so callers can place additional markers."""
    H, W = frame.shape[:2]
    if layout is None:
        layout = compute_layout(W, H)
    overlay = frame.copy()
    cv2.rectangle(overlay, (layout.panel_x1, layout.panel_y1),
                  (layout.panel_x2, layout.panel_y2), bg_color, -1)
    cv2.addWeighted(overlay, bg_alpha, frame, 1.0 - bg_alpha, 0, frame)
    cv2.rectangle(frame, (layout.panel_x1, layout.panel_y1),
                  (layout.panel_x2, layout.panel_y2), border_color, 2)
    cv2.rectangle(frame, layout.world_to_panel(0, 0),
                  layout.world_to_panel(COURT_WIDTH_M, COURT_LENGTH_M),
                  line_color, 2)
    if COURT_WIDTH_M > SINGLES_WIDTH_M + 0.1:
        inset = (COURT_WIDTH_M - SINGLES_WIDTH_M) / 2.0
        cv2.line(frame, layout.world_to_panel(inset, 0),
                 layout.world_to_panel(inset, COURT_LENGTH_M), singles_color, 1)
        cv2.line(frame, layout.world_to_panel(COURT_WIDTH_M - inset, 0),
                 layout.world_to_panel(COURT_WIDTH_M - inset, COURT_LENGTH_M),
                 singles_color, 1)
    cv2.line(frame, layout.world_to_panel(0, NET_LINE_Y_M),
             layout.world_to_panel(COURT_WIDTH_M, NET_LINE_Y_M), net_color, 2)
    svc_top = NET_LINE_Y_M - SERVICE_LINE_OFFSET_M
    svc_bot = NET_LINE_Y_M + SERVICE_LINE_OFFSET_M
    if svc_top > 0:
        cv2.line(frame, layout.world_to_panel(0, svc_top),
                 layout.world_to_panel(COURT_WIDTH_M, svc_top), service_color, 1)
    if svc_bot < COURT_LENGTH_M:
        cv2.line(frame, layout.world_to_panel(0, svc_bot),
                 layout.world_to_panel(COURT_WIDTH_M, svc_bot), service_color, 1)
    cv2.line(frame, layout.world_to_panel(COURT_WIDTH_M / 2.0, svc_top),
             layout.world_to_panel(COURT_WIDTH_M / 2.0, svc_bot), service_color, 1)
    return layout


def draw_point(frame, cv2, layout: MiniCourtLayout, xm: float, ym: float,
               color: Tuple[int, int, int] = (0, 0, 255), radius: int = 6,
               ring: Tuple[int, int, int] = (255, 255, 255)) -> None:
    p = layout.world_to_panel(xm, ym)
    cv2.circle(frame, p, radius, color, -1)
    cv2.circle(frame, p, radius + 1, ring, 1)


def draw_polyline(frame, cv2, layout: MiniCourtLayout,
                  points_world_xy: List[Tuple[float, float]],
                  color: Tuple[int, int, int] = (80, 180, 255),
                  thickness: int = 2) -> None:
    pts = [layout.world_to_panel(x, y) for x, y in points_world_xy]
    for p0, p1 in zip(pts, pts[1:]):
        cv2.line(frame, p0, p1, color, thickness, cv2.LINE_AA)
