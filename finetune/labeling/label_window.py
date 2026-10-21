"""The label tool's window plumbing: where the window is on screen, the OS mouse pointer, and the
two things OpenCV's Qt window does on its own that have to be undone - its viewport zoom
(``ViewportGuard``) and the fact that it never says where it draws the picture (``CursorPlacer``
measures that so the pointer can be parked on the ball).

Cursor placement and window sizing are Windows niceties: without ``user32`` everything here
answers "cannot say" and the tool works without them.
"""
from __future__ import annotations

import ctypes
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2

WINDOW = "label_tool"
WM_MOUSEWHEEL = 0x020A
SPI_GETWORKAREA = 0x0030
SWP_NOZORDER = 0x0004
VK_SHIFT = 0x10
KEY_DOWN = 0x8000              # the high bit of GetKeyState
VK_LEFT, VK_UP, VK_RIGHT, VK_DOWN = 37, 38, 39, 40      # what waitKeyEx gives for the arrows here
KEYS_NEXT = {VK_RIGHT, 2555904, 83, 0x01000014}         # Win32 / Qt (this build) / GTK / Qt raw
KEYS_PREV = {VK_LEFT, 2424832, 81, 0x01000012}
KEY_TAB = 9
KEYS_QUIT = (ord("q"), 27)                 # q / ESC
KEYS_CONFIRM = (ord(" "), 13, ord("c"))    # SPACE / ENTER / c
KEYS_ZOOM_IN, KEYS_ZOOM_OUT = (ord("+"), ord("=")), (ord("-"), ord("_"))
KEYS_STEP = {ord(digit): int(digit) for digit in "12468"}     # frames n/d/p/a move at a time
NUDGE_OF = dict(zip("ijkl", ((0, -1), (-1, 0), (0, 1), (1, 0))))   # up / left / down / right
KEYS_NUDGE = {ord(key): step for letter, step in NUDGE_OF.items() for key in (letter, letter.upper())}
WHEEL_QUIET = 0.25        # seconds after a wheel notch during which the mouse is left alone
VIEWPORT_CHECK = 3.0      # seconds between checks that the window's own zoom is still at x1
PROBE_TIMEOUT = 2.0       # a probe unanswered this long was lost (window hidden, being resized)
PROBE_TOLERANCE_PX = 4.0  # a probe reading may be off by this (whole-pixel readings) ...
PROBE_TOLERANCE_SHARE = 0.008   # ... or this share of the view width, whichever is more
RESET_WINDOW_S = 60.0     # window rebuilds are counted over this long ...
RESET_LIMIT = 3           # ... and more than this many means the check is measuring something else
PLACEMENT_WAIT = 0.30     # seconds a warp's own move report is waited for
PROBE_WAIT = 0.25         # seconds a calibration probe waits for highgui to report the pointer
MAP_RETRY_S = 2.0         # a map is not re-measured more often than this
WARP_TRIES = 3            # placements mended by nudging the offset before the map is re-measured
CALIBRATE_NEAR, CALIBRATE_FAR = 0.2, 0.8   # where in the client area the two calibration probes go
MIN_SPREAD_PX = 5         # the two probes must read this far apart to solve a scale
SCALE_RANGE = (0.05, 20.0)   # a solved scale outside this is a hand on the mouse, not a measurement


class _POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class _RECT(ctypes.Structure):
    _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                ("right", ctypes.c_long), ("bottom", ctypes.c_long)]


try:  # cursor placement and window sizing are Windows niceties; everything else works without them
    _user32 = ctypes.windll.user32
except (AttributeError, OSError):
    _user32 = None


def find_window() -> Optional[int]:
    """The tool's window handle, or None without user32 or a window."""
    if _user32 is None:
        return None
    return _user32.FindWindowW(None, WINDOW) or None


def work_area() -> Optional[Tuple[int, int, int, int]]:
    """The desktop minus the taskbar, (width, height, left, top) in physical screen pixels, or
    None if the OS will not say."""
    if _user32 is None:
        return None
    rect = _RECT()
    if not _user32.SystemParametersInfoW(SPI_GETWORKAREA, 0, ctypes.byref(rect), 0):
        return None
    return rect.right - rect.left, rect.bottom - rect.top, rect.left, rect.top


def window_rect() -> Optional[Tuple[int, int, int, int]]:
    """(left, top, right, bottom) of the window including its frame, in screen pixels, or None."""
    hwnd = find_window()
    if not hwnd:
        return None
    rect = _RECT()
    if not _user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        return None
    return rect.left, rect.top, rect.right, rect.bottom


def place_window(left: int, top: int, width: int, height: int) -> None:
    hwnd = find_window()
    if hwnd:
        _user32.SetWindowPos(hwnd, 0, left, top, width, height, SWP_NOZORDER)


def window_client() -> Optional[Tuple[int, int, int, int]]:
    """(origin_x, origin_y, width, height) of the window's client area, in physical screen pixels.

    Everything about placing the mouse has to come from here. cv2.getWindowImageRect answers in
    *image* pixels while SetCursorPos works in screen pixels, and highgui puts an image inside the
    client area at a scale of its own choosing - the monitor DPI when the window sizes itself to
    the image, whatever fits once you resize it. Measuring beats assuming.
    """
    hwnd = find_window()
    if not hwnd:
        return None
    origin, client = _POINT(), _RECT()
    if not (_user32.ClientToScreen(hwnd, ctypes.byref(origin))
            and _user32.GetClientRect(hwnd, ctypes.byref(client))):
        return None
    if client.right <= 0 or client.bottom <= 0:
        return None
    return origin.x, origin.y, client.right, client.bottom


def cursor_screen() -> Optional[Tuple[int, int]]:
    """Where the OS says the pointer is, in physical screen pixels, or None.

    SetCursorPos is not a promise: another process can move the pointer in the same millisecond,
    and a target off the edge of the desktop is clamped. Reading it back is how a warp that did
    not take - or a hand on the mouse - is told apart from a map that needs mending.
    """
    if _user32 is None:
        return None
    point = _POINT()
    if not _user32.GetCursorPos(ctypes.byref(point)):
        return None
    return point.x, point.y


def inside_client(point: Tuple[int, int]) -> bool:
    """Is this screen pixel inside our window's client area? True when the OS will not say."""
    geometry = window_client()
    if geometry is None:
        return True
    origin_x, origin_y, client_w, client_h = geometry
    return (origin_x <= point[0] < origin_x + client_w
            and origin_y <= point[1] < origin_y + client_h)


def window_chrome() -> Tuple[int, int]:
    """(width, height) the frame and title bar add on top of the client area."""
    hwnd = find_window()
    if not hwnd:
        return 0, 0
    window, client = _RECT(), _RECT()
    if not (_user32.GetWindowRect(hwnd, ctypes.byref(window)) and _user32.GetClientRect(hwnd, ctypes.byref(client))):
        return 0, 0
    return (window.right - window.left) - client.right, (window.bottom - window.top) - client.bottom


def shift_held() -> bool:
    """Is SHIFT down right now? waitKey cannot say: this build's highgui (Qt) hands every letter back
    in lower case whatever the modifiers, so the fine nudge has to ask the OS instead."""
    if _user32 is None:
        return False
    return bool(_user32.GetKeyState(VK_SHIFT) & KEY_DOWN)


# Keys with no character of their own come back from this build as their bare virtual-key code,
# and some of those codes are also printable keys the tool binds: F1..F12 are 112..123, which read
# as p, q, r, s, t, u, v ...; Insert/Delete are 45/46 ('-', '.'); PgUp/PgDn/End/Home are 33..36.
# The arrows (37..40) are wanted and handled as arrows. A letter is typed on a different key, so
# asking the OS whether the virtual key itself is down tells the two apart.
NON_CHARACTER_KEYS = set(range(0x70, 0x7C)) | {0x2D, 0x2E} | set(range(0x21, 0x25))


def non_character_key(code: int) -> bool:
    """Is `code` an F-key, Insert, Delete or a page key that only looks like a printable one?"""
    return (_user32 is not None and code in NON_CHARACTER_KEYS
            and bool(_user32.GetKeyState(code) & KEY_DOWN))


def post_wheel(delta: int, screen: Tuple[int, int]) -> bool:
    """Post a wheel notch of `delta` to the window at a screen point, as if the hand had turned it."""
    hwnd = find_window()
    if not hwnd:
        return False
    wparam = (delta & 0xFFFF) << 16
    lparam = ((screen[1] & 0xFFFF) << 16) | (screen[0] & 0xFFFF)
    return bool(_user32.PostMessageW(hwnd, WM_MOUSEWHEEL, wparam, lparam))


# ---------------------------------------------------------------- the window's own zoom
# This build's highgui is Qt's, and its viewport zooms itself on every wheel notch - x1.1 a notch,
# about the pointer, Ctrl or no Ctrl - before or after handing the notch to on_mouse, and nothing
# in the cv2 API resets it. So every notch used to zoom twice: the tool's view about the ball,
# and the window about the pointer. The ball stayed under the pointer while scrolling, so it
# looked fine, but the next frame was drawn centred in a picture the window now showed off-centre
# and cropped: the ball turned up in a corner, with a little navigator box top-right. Measured:
# a notch of delta d scales the window by 1 + d/1200, and it never goes below x1, so posting the
# opposite notch back straight away lands it on exactly x1 again (1.1 x 0.9 < 1, clamped).

@dataclass
class ViewportProbe:
    """An in-flight check of the window's own zoom: two 1/120-notch wheels posted at known screen
    points, whose readings say which picture pixels sit there."""
    sent: float
    screens: List[Tuple[int, int]]
    want: List[Tuple[float, float]]
    seen: List[Tuple[int, int]] = field(default_factory=list)
    stage: int = 0            # readings received (0-2), then repayments seen (2-4)
    void: bool = False        # the view changed size under it: judge nothing


class ViewportGuard:
    """Keeps the Qt viewport at x1: pays every zoom-in notch back, swallows the wheels it posted
    itself, checks the picture's position every few seconds, and rations the window rebuilds
    that a failed check asks for."""

    def __init__(self) -> None:
        self.wheel_debts: List[Tuple[int, float]] = []   # (delta, when) of wheels posted to cancel Qt's own zoom
        self.probe: Optional[ViewportProbe] = None
        self.probe_at = 0.0           # when the last probe was sent
        self.reset_needed = False     # a probe failed: the main loop should rebuild the window
        self.window_resets = 0
        self.reset_times: List[float] = []   # recent rebuilds, so a check that keeps failing gives up

    def absorb_wheel(self, x: int, y: int, flags: int, view_w: int) -> bool:
        """Swallow the wheel events the tool sent itself. True when this one was ours."""
        delta = flags >> 16
        now = time.monotonic()
        self.wheel_debts = [debt for debt in self.wheel_debts if now - debt[1] < 1.0]
        if self.probe is not None:
            if self.probe_answer(x, y, delta, view_w):
                return True
            if now - self.probe.sent > 1.0:
                self.probe = None                            # never answered: forget it
        if delta < 0 and self.wheel_debts and self.wheel_debts[0][0] == delta:
            self.wheel_debts.pop(0)
            return True
        return False

    def probe_answer(self, x: int, y: int, delta: int, view_w: int) -> bool:
        """Take a wheel event as the probe in flight's next answer: a reading (stage 0-1), then
        the repayments (2-3), the last of which judges the window. False when it is not one."""
        probe = self.probe
        if probe.stage < 2 and delta == 1:              # a reading came back
            probe.seen.append((x, y))
            probe.stage += 1
            if probe.stage == 2:                         # both read: pay the two 1/1200ths back
                for screen in probe.screens:
                    post_wheel(-1, screen)
            return True
        if 2 <= probe.stage < 4 and delta == -1:
            probe.stage += 1
            if probe.stage == 4:
                self.probe = None
                if not probe.void and self.probe_failed(probe, view_w):
                    self.reset_needed = True     # rebuilt from the main loop, not from here
            return True
        return False

    @staticmethod
    def probe_failed(probe: ViewportProbe, view_w: int) -> bool:
        """Whether either reading sits further from where it should be than whole-pixel readings allow."""
        tolerance = max(PROBE_TOLERANCE_PX, PROBE_TOLERANCE_SHARE * view_w)
        return any(abs(seen[0] - want[0]) > tolerance or abs(seen[1] - want[1]) > tolerance
                   for seen, want in zip(probe.seen, probe.want))

    def repay_wheel(self, delta: int) -> None:
        """A zoom-in notch just zoomed the window too: post the opposite one at the same spot. A
        zoom-out notch needs nothing - the window is already at x1 and will not go lower."""
        if delta <= 0:
            return
        where = cursor_screen()
        if where is None or not post_wheel(-delta, where):
            return
        self.wheel_debts.append((-delta, time.monotonic()))

    def void_probe(self) -> None:
        """The window or picture changed size under a probe in flight: its expected readings no
        longer hold. It still runs to the end, so its own wheels are swallowed, but judges nothing."""
        if self.probe is not None:
            self.probe.void = True

    def check(self, view_w: int, image_h: int) -> None:
        """Every few idle seconds, ask the window which picture pixels sit at the middle of its client
        area and a quarter of the way across. With the aspect ratio kept and the window at x1 the
        picture is centred and scaled to fit, so both answers are known in advance; a pan moves the
        first, a zoom about the centre moves the second. Anything else means the window's own zoom
        has crept in some way not caught above, and the window is rebuilt. Each question is a
        1/120-notch wheel, since a zero delta is dropped, and both are paid back at once."""
        now = time.monotonic()
        if self.probe is not None and now - self.probe.sent > PROBE_TIMEOUT:
            self.probe = None             # never answered (window hidden, being resized): forget it
        if _user32 is None or self.probe is not None or now - self.probe_at < VIEWPORT_CHECK:
            return
        geometry = window_client()
        if geometry is None:
            return
        origin_x, origin_y, client_w, client_h = geometry
        fit = min(client_w / max(view_w, 1), client_h / max(image_h, 1))
        screens = [(origin_x + client_w // 2, origin_y + client_h // 2),
                   (origin_x + client_w // 4, origin_y + client_h // 2)]
        want = [(view_w / 2.0, image_h / 2.0),
                (view_w / 2.0 - (client_w // 2 - client_w // 4) / fit, image_h / 2.0)]
        self.probe_at = now
        if post_wheel(1, screens[0]) and post_wheel(1, screens[1]):
            self.probe = ViewportProbe(sent=now, screens=screens, want=want)

    def allow_reset(self) -> bool:
        """Count a window rebuild; False when they come too often, in which case the check is
        switched off for the rest of the session instead."""
        self.reset_needed = False
        now = time.monotonic()
        self.reset_times = [t for t in self.reset_times if now - t < RESET_WINDOW_S] + [now]
        if len(self.reset_times) > RESET_LIMIT:
            # A check that fails again straight after a rebuild is measuring something other than
            # Qt's zoom; rebuilding every few seconds forever is far worse than not checking.
            self.probe_at = float("inf")
            self.probe = None
            return False
        self.window_resets += 1
        return True

    def forget(self) -> None:
        """After a rebuild: whatever was owed to or asked of the old window went with it."""
        self.wheel_debts = []
        self.probe = None


# ---------------------------------------------------------------- the OS pointer

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
        geometry = window_client()
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
        if tries == 0 and cursor_screen() == target:
            # Already sitting there - which is every frame once the ball is kept dead centre. The OS
            # sends no move event for a warp to the pointer's own pixel, so there is nothing to
            # check and no reason to hold prefetch back for the 0.3 s it used to wait in vain.
            self.aim = None
            return True
        seq = self.pointer_seq                       # taken first: the report can beat this line
        _user32.SetCursorPos(*target)
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
                if not inside_client(target) and time.monotonic() - self.map_time > MAP_RETRY_S:
                    self.map_stale = True   # warped clean off the window: that map is not a map
            return
        seen = self.pointer
        self.aim = None
        if seen is None or time.monotonic() - aimed_at > PLACEMENT_WAIT:
            return
        where = cursor_screen()
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
        _user32.SetCursorPos(screen_x + 6, screen_y + 6)
        if self.await_pointer(seq, PROBE_WAIT) is None:
            return None               # the jog's report could still arrive and pass for the reading
        seq = self.pointer_seq
        _user32.SetCursorPos(screen_x, screen_y)
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
        if _user32 is None:
            return
        geometry = window_client()
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


def highgui_backend() -> str:
    """The GUI line of OpenCV's build information: QT6, WIN32UI, GTK3, ..."""
    for line in cv2.getBuildInformation().splitlines():
        if line.strip().startswith("GUI:"):
            return line.split(":", 1)[1].strip()
    return "unknown"
