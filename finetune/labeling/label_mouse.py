"""The label tool's mouse: what a press, a wheel turn or a move does in the single-frame view,
and which events the tool must not act on (its own wheel posts, anything while it is moving the
pointer, a click ahead of the displayed view). The grid and the sheet take their own clicks
(``label_grid``, ``label_sheet``); the HUD chips and the inset take theirs (``label_frame_view``)."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import cv2

if TYPE_CHECKING:
    from finetune.labeling.label_session import LabelSession
    from finetune.labeling.label_tool import Tool


class MouseHandler:
    """``on_mouse`` is the highgui callback the tool registers on its window."""

    def __init__(self, session: "LabelSession", tool: "Tool") -> None:
        self.session = session
        self.tool = tool

    def on_mouse(self, event: int, x: int, y: int, flags: int, param: object = None) -> None:
        tool, session = self.tool, self.session
        if event == cv2.EVENT_MOUSEWHEEL and tool.viewport.absorb_wheel(x, y, flags, session.view_w):
            return                                  # a wheel the tool sent itself, not the hand's
        self.note_pointer(x, y)
        if self.guarded(event):
            if event == cv2.EVENT_MOUSEWHEEL:
                self.settle_wheel(flags)
            return
        if tool.sheet_view.active:
            tool.sheet_view.on_mouse(event, x, y)
            return
        if event == cv2.EVENT_LBUTTONDBLCLK:
            # The second press of a quick double-click comes through as this instead of a press. With
            # the pointer parked on the same screen pixel frame after frame, click-click on two
            # consecutive frames is exactly that, and every second click used to be thrown away.
            event = cv2.EVENT_LBUTTONDOWN
        if tool.grid_view.active:
            tool.grid_view.on_mouse(event, x, y)
            return
        self.frame_event(event, x, y, flags)

    def frame_event(self, event: int, x: int, y: int, flags: int) -> None:
        """An event on the single-frame view: the HUD's chips, the inset, a click placing the ball,
        a right click re-centring, or the wheel zooming about the pointer."""
        tool, session = self.tool, self.session
        if y >= session.view_h:                     # the HUD: its buttons, and nothing else
            if event == cv2.EVENT_LBUTTONDOWN:
                tool.frame_view.click_hud(x, y)
            return
        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN) and tool.frame_view.inset_hit(x, y):
            tool.frame_view.click_inset(x, y)
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.place_click(x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            session.centre_on(*session.to_frame(x + 0.5, y + 0.5))   # a click in the black beyond the
        elif event == cv2.EVENT_MOUSEWHEEL:                           # frame is kept on the frame
            tool.mouse.wheel_at = time.monotonic()
            session.zoom_at(self.wheel_notches(flags), x + 0.5, y + 0.5)
            tool.viewport.repay_wheel(flags >> 16)

    def note_pointer(self, x: int, y: int) -> None:
        """Every event says where the pointer is: where the last warp landed, which the placer's
        check_placement reads, so this comes before any guard."""
        self.tool.mouse.report(x, y)

    def settle_wheel(self, flags: int) -> None:
        """A wheel the tool does not act on still zoomed Qt's own viewport: cancel that, or the
        next viewport check finds it and rebuilds the window."""
        self.tool.viewport.repay_wheel(flags >> 16)

    def guarded(self, event: int) -> bool:
        """True for a mouse event the tool must not act on (a pure test; the wheels the tool posted
        itself are swallowed before it): anything but a move while the tool is moving the pointer,
        a wheel on the grid or the sheet, and a wheel or click ahead of the displayed view - a
        queued click must never edit the next frame before it is displayed."""
        tool = self.tool
        if tool.mouse.calibrating:
            return event != cv2.EVENT_MOUSEMOVE
        if tool.overview:
            return event == cv2.EVENT_MOUSEWHEEL
        return event != cv2.EVENT_MOUSEMOVE and not tool.view_ready()

    def place_click(self, x: int, y: int) -> None:
        """A left click on the picture: the ball is there, and the next frame comes up."""
        session = self.session
        if session.frame_missing():
            return                                  # nothing to see, so nothing to label
        fx, fy = session.to_frame(x + 0.5, y + 0.5)  # highgui floors the pointer to a pixel: aim at
        if not (-2 <= fx < session.width + 2 and -2 <= fy < session.height + 2):   # its middle
            return                                # a click in the black beyond the frame
        if self.tool.set_point(fx, fy):
            self.tool.move(1)

    @staticmethod
    def wheel_notches(flags: int) -> float:
        """Wheel movement in notches, sign included.

        highgui packs the wheel delta into the top 16 bits of flags as a signed short, so an
        arithmetic shift gets it back; cv2.getMouseWheelDelta is C++-only in this build. One
        detent is 120, but a free-spinning wheel or a precision trackpad sends less than that,
        and following those fractions is most of what makes a zoom feel continuous.
        """
        delta = (flags >> 16) / 120.0
        if delta == 0:                       # a build that does not pack a delta: one notch it is
            delta = 1.0 if flags > 0 else -1.0
        return max(-4.0, min(4.0, delta))
