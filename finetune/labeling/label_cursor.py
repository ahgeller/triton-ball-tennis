"""The OS mouse pointer: ``CursorPlacer`` puts it on a point of the displayed image (the ball,
so a click is a one-pixel move) and keeps the map from view pixels to screen pixels true, since
OpenCV's Qt window never says where it draws the picture. A Windows nicety: without ``user32``
(``label_window``) every placement is skipped and the tool works without it."""

from __future__ import annotations

import time
from typing import Optional, Tuple

from finetune.labeling import label_window as win

WHEEL_QUIET = 0.25        # seconds after a wheel notch during which the mouse is left alone
PLACEMENT_WAIT = 0.30     # seconds a warp's own move report is waited for
PROBE_WAIT = 0.25         # seconds a calibration probe waits for highgui to report the pointer
MAP_RETRY_S = 2.0         # a map is not re-measured more often than this
WARP_TRIES = 3            # placements mended by nudging the offset before the map is re-measured
CALIBRATE_NEAR, CALIBRATE_FAR = 0.2, 0.8   # where in the client area the two calibration probes go
MIN_SPREAD_PX = 5         # the two probes must read this far apart to solve a scale
SCALE_RANGE = (0.05, 20.0)   # a solved scale outside this is a hand on the mouse, not a measurement


class CursorPlacer:
    """Puts the OS mouse pointer on a point of the displayed image, and keeps the map from view
    pixels to screen pixels true. `host` is the tool: it supplies the view size (view_w, view_h,
    hud_h) and pump(ms), which lets highgui deliver the pointer reports this reads."""

    def __init__(self, host) -> None:
        self.host = host
        self.cursor_map: Optional[Tuple[float, float, float, float]] = None   # (scale_x, scale_y, offset_x, offset_y) view px -> client px
        self.map_stale = True         # until calibrate has measured the real one
        self.map_time = 0.0           # when the map was last measured
        self.pointer: Optional[Tuple[int, int]] = None   # last place highgui reported the mouse, in view pixels
        self.pointer_seq = 0          # bumped on every report, so a fresh one can be waited for
        self.aim = None               # the last warp, until check_placement has confirmed it
        self.calibrating = False      # while the pointer is being walked round for a measurement
        self.wheel_at = 0.0           # when the wheel last turned; the mouse is left alone just after

    def report(self, x: int, y: int) -> None:
        """highgui reported the pointer at a view pixel (on_mouse)."""
        self.pointer = (x, y)
        self.pointer_seq += 1

    def forget_window(self) -> None:
        """The window was rebuilt: measure it from scratch."""
        self.cursor_map = None
        self.map_stale = True

    def scrolling(self) -> bool:
        """True just after a wheel notch. A wheel arrives in big discrete steps, and moving the mouse
        between them moves what the next notch zooms about - which is what made the wheel feel jumpy
        next to a trackpad, where the deltas are small and continuous. So leave the mouse alone until
        the scrolling stops, then put it back on the ball."""
        return time.monotonic() - self.wheel_at < WHEEL_QUIET

    def screen_for(self, view_x: float, view_y: float) -> Optional[Tuple[int, int]]:
        """The screen pixel a point of the displayed image sits on, or None if it cannot be said.

        The window origin is read fresh every time, so dragging the window elsewhere keeps working;
        only the scale and offset inside the client area are the measured part.
        """
        geometry = win.window_client()
        if geometry is None or self.cursor_map is None:
            return None
        scale_x, scale_y, offset_x, offset_y = self.cursor_map
        return (int(round(geometry[0] + offset_x + view_x * scale_x)),
                int(round(geometry[1] + offset_y + view_y * scale_y)))

    def warp(self, view_x: float, view_y: float, tries: int = 0) -> bool:
        """Move the OS mouse cursor to a point of the displayed image. True if it was placed.

        The point is pulled inside the image rather than refused: a ball on the very edge of the
        frame still gets the pointer as near to it as the window allows, instead of leaving it
        wherever the last frame happened to end.

        Where we aimed is kept for check_placement, which reads back where highgui says the
        pointer landed. That read-back is what makes the placement dependable: the map is a
        measurement, and one taken while another window covered ours, or before the window was
        moved to a screen at a different DPI, is wrong in a way nothing else here would notice.
        """
        view_x = min(max(view_x, 0.0), self.host.view_w - 1.0)
        view_y = min(max(view_y, 0.0), self.host.view_h - 1.0)
        target = self.screen_for(view_x, view_y)
        if target is None:
            return False
        if tries == 0 and win.cursor_screen() == target:
            # Already sitting there - which is every frame once the ball is kept dead centre. The OS
            # sends no move event for a warp to the pointer's own pixel, so there is nothing to
            # check and no reason to hold prefetch back for the 0.3 s it used to wait in vain.
            self.aim = None
            return True
        seq = self.pointer_seq                       # taken first: the report can beat this line
        win._user32.SetCursorPos(*target)
        self.aim = (view_x, view_y, target, seq, time.monotonic(), tries)
        return True

    def check_placement(self) -> None:
        """Confirm the last warp landed where it was aimed, and mend the map when it did not.

        The warp's own move event comes back through highgui a few milliseconds later, so this
        costs nothing: it reads what the main loop's waitKey has already delivered. A disagreement
        means the offset half of the map is off - learn it from the error and warp again. Two
        frames of that beats any single measurement, and it recovers on its own from the one case
        that used to strand the pointer for a whole session: a calibration that could not see the
        pointer move, gave up, and left the map on a guess that is wrong here by a factor of two.

        A report that took too long, or a pointer no longer where we put it, is a hand on the
        mouse. Leave it alone: the next frame will place the cursor anyway.
        """
        if self.aim is None:
            return
        view_x, view_y, target, seq, aimed_at, tries = self.aim
        if self.pointer_seq == seq:                       # highgui has not reported yet
            if time.monotonic() - aimed_at > PLACEMENT_WAIT:
                self.aim = None
                if not win.inside_client(target) and time.monotonic() - self.map_time > MAP_RETRY_S:
                    self.map_stale = True   # warped clean off the window: that map is not a map
            return
        seen = self.pointer
        self.aim = None
        if seen is None or time.monotonic() - aimed_at > PLACEMENT_WAIT:
            return
        where = win.cursor_screen()
        if where is None or abs(where[0] - target[0]) > 2 or abs(where[1] - target[1]) > 2:
            return                                        # the pointer has moved on since: not ours
        self.mend_map(seen[0] - view_x, seen[1] - view_y, view_x, view_y, tries)

    def mend_map(self, error_x: float, error_y: float, view_x: float, view_y: float, tries: int) -> None:
        """Learn the map's offset from where a warp landed, and warp again; after WARP_TRIES the
        map is measured afresh instead."""
        if abs(error_x) < 1.0 and abs(error_y) < 1.0:
            return                                        # dead on: nothing to mend
        scale_x, scale_y, offset_x, offset_y = self.cursor_map
        self.cursor_map = (scale_x, scale_y,
                           offset_x - error_x * scale_x, offset_y - error_y * scale_y)
        if tries < WARP_TRIES:
            self.warp(view_x, view_y, tries + 1)
        elif time.monotonic() - self.map_time > MAP_RETRY_S:
            self.map_stale = True            # offsets alone are not closing it: measure again

    def probe(self, screen_x: int, screen_y: int) -> Optional[Tuple[int, int]]:
        """Put the pointer on a screen pixel and return the view point highgui says that is, or None.

        Probing by screen pixel rather than through the map is what lets a badly wrong map be
        measured at all. A point taken from the client rectangle is on the window whatever the map
        says, so highgui always has something to report; a point aimed through a map that is out by
        the DPI factor can land clean off the window, report nothing, and leave the broken map in
        place for the rest of the session - which is how the pointer used to end up somewhere other
        than the ball and stay there.

        The pointer is jogged a few pixels first and that jog waited out, because SetCursorPos to
        where the pointer already is sends no message at all: without the jog a measurement could
        return the reading before it, and be solved into a map that is wrong everywhere.

        Worth waiting a while for the answer: right after a window appears or is resized the first
        move event can take well over 100 ms to come through, and treating that as a failure used
        to throw away the whole measurement.
        """
        seq = self.pointer_seq
        win._user32.SetCursorPos(screen_x + 6, screen_y + 6)
        if self.await_pointer(seq, PROBE_WAIT) is None:
            return None               # the jog's report could still arrive and pass for the reading
        seq = self.pointer_seq
        win._user32.SetCursorPos(screen_x, screen_y)
        return self.await_pointer(seq, PROBE_WAIT)

    def await_pointer(self, seq: int, seconds: float) -> Optional[Tuple[int, int]]:
        """Pump highgui until it reports a pointer position newer than `seq`. None if it never does."""
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.host.pump(5)
            if self.pointer_seq != seq:
                return self.pointer
        return None

    def calibrate(self) -> None:
        """Measure the map from a view pixel to a client pixel, instead of assuming one.

        highgui draws our image inside the client area at a scale and offset it never reports: the
        monitor DPI when the window sizes itself, whatever fits once you resize or full-screen it,
        plus a band it keeps at the top. Measured on this machine that was 2.000x at +1,+67 in a
        fitted window and 2.400x at +2,+61 full screen - guessing any of it is hopeless.

        So: put the pointer on two known pixels of the client rectangle, ask highgui which pixels
        of the image it thinks those are, and solve the straight line through the two pairs. Runs
        again whenever the window changes size, because every one of those numbers moves with it,
        and whenever a placement lands somewhere the offset nudging cannot mend.
        """
        if win._user32 is None:
            return
        geometry = win.window_client()
        if geometry is None:
            return
        origin_x, origin_y, client_w, client_h = geometry
        host = self.host
        if self.cursor_map is None:    # something to aim with until the measurement comes back
            guess = min(client_w / max(host.view_w, 1), client_h / max(host.view_h + host.hud_h, 1))
            self.cursor_map = (guess, guess, 0.0, 0.0)
        self.map_stale = False
        self.map_time = time.monotonic()
        near = (int(client_w * CALIBRATE_NEAR), int(client_h * CALIBRATE_NEAR))   # well inside the picture either way,
        far = (int(client_w * CALIBRATE_FAR), int(client_h * CALIBRATE_FAR))      # and far enough apart to solve cleanly
        self.calibrating = True       # a click while the pointer is being walked round is not a label
        try:
            seen_near = self.probe(origin_x + near[0], origin_y + near[1])
            seen_far = self.probe(origin_x + far[0], origin_y + far[1])
        finally:
            self.calibrating = False
        # A measurement needs highgui to see the pointer move, which it cannot while another window
        # covers ours. Whatever goes wrong from here, keep the map we came in with: a stale
        # measurement is close, and check_placement goes on nudging whatever we keep towards the
        # truth on every frame.
        if seen_near is None or seen_far is None:
            return
        solved = list(self.cursor_map)
        for axis in (0, 1):
            # Two (client pixel put there, view pixel reported) pairs, and the map is the straight
            # line through them - no part of it inherited from the map being replaced.
            spread = seen_far[axis] - seen_near[axis]
            if abs(spread) < MIN_SPREAD_PX:
                return                                      # the pointer barely moved: keep the map
            true_scale = (far[axis] - near[axis]) / spread
            if not SCALE_RANGE[0] <= true_scale <= SCALE_RANGE[1]:
                return                                      # a stray hand on the mouse: keep the map
            solved[axis] = true_scale
            solved[2 + axis] = near[axis] - seen_near[axis] * true_scale
        self.cursor_map = tuple(solved)
