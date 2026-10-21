"""The label tool's mouse: what a press, a wheel turn or a move does in the single-frame view,
and which events the tool must not act on (its own wheel posts, anything while it is moving the
pointer, a click ahead of the displayed view). The grid and the sheet take their own clicks
(``label_views``); the HUD chips and the inset take theirs (``label_frame_view``)."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import cv2

if TYPE_CHECKING:
    from finetune.labeling.label_tool import Tool


class MouseHandler:
    """``on_mouse`` is the highgui callback the tool registers on its window."""

    def __init__(self, tool: "Tool") -> None:
        self.tool = tool

    def on_mouse(self, event: int, x: int, y: int, flags: int, param: object = None) -> None:
        tool = self.tool
        if self.guarded(event, x, y, flags):
            return
        if tool.sheet:
            tool.sheet_mouse(event, x, y)
            return
        if event == cv2.EVENT_LBUTTONDBLCLK:
            # The second press of a quick double-click comes through as this instead of a press. With
            # the pointer parked on the same screen pixel frame after frame, click-click on two
            # consecutive frames is exactly that, and every second click used to be thrown away.
            event = cv2.EVENT_LBUTTONDOWN
        if tool.grid:
            tool.grid_mouse(event, x, y)
            return
        if y >= tool.view_h:                        # the HUD: its buttons, and nothing else
            if event == cv2.EVENT_LBUTTONDOWN:
                tool.frame_view.click_hud(x, y)
            return
        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN) and tool.frame_view.inset_hit(x, y):
            tool.frame_view.click_inset(x, y)
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.place_click(x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            frame_x, frame_y = tool.to_frame(x + 0.5, y + 0.5)
            tool.center = (min(max(frame_x, 0.0), tool.width - 1.0),   # a click in the black beyond
                           min(max(frame_y, 0.0), tool.height - 1.0))  # the frame: keep it on frame
            tool.centre_view()
        elif event == cv2.EVENT_MOUSEWHEEL:
            tool.mouse.wheel_at = time.monotonic()
            tool.zoom_at(self.wheel_notches(flags), x + 0.5, y + 0.5)
            tool.viewport.repay_wheel(flags >> 16)

    def guarded(self, event: int, x: int, y: int, flags: int) -> bool:
        """True for a mouse event the tool must not act on: a wheel it posted itself, anything
        while it is moving the pointer, a wheel or click ahead of the displayed view."""
        tool = self.tool
        if event == cv2.EVENT_MOUSEWHEEL and tool.viewport.absorb_wheel(x, y, flags, tool.view_w):
            return True                             # a wheel the tool sent itself, not the hand's
        tool.mouse.report(x, y)                     # where the warp landed; check_placement reads it
        if event == cv2.EVENT_MOUSEWHEEL and (tool.mouse.calibrating or tool.overview or not tool.view_ready()):
            # A wheel the tool does not act on still zoomed Qt's own viewport: cancel that, or the
            # next viewport check finds it and rebuilds the window.
            tool.viewport.repay_wheel(flags >> 16)
            return True
        if tool.mouse.calibrating and event != cv2.EVENT_MOUSEMOVE:
            return True                             # the tool is moving the pointer itself just now
        # a queued click must never edit the next frame before it is displayed
        return not tool.overview and event != cv2.EVENT_MOUSEMOVE and not tool.view_ready()

    def place_click(self, x: int, y: int) -> None:
        """A left click on the picture: the ball is there, and the next frame comes up."""
        tool = self.tool
        if tool.frame_missing():
            return                                  # nothing to see, so nothing to label
        fx, fy = tool.to_frame(x + 0.5, y + 0.5)  # highgui floors the pointer to a pixel: aim at
        if not (-2 <= fx < tool.width + 2 and -2 <= fy < tool.height + 2):   # its middle
            return                                # a click in the black beyond the frame
        if tool.set_point(fx, fy):
            tool.move(1)

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
