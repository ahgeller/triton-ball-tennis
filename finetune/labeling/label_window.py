"""The label tool's window plumbing: where the window is on screen, the key tables, and the
viewport zoom OpenCV's Qt window does on its own, which has to be undone (``ViewportGuard``).
``label_cursor`` parks the OS pointer on the ball with the ``user32`` calls from here.

Window sizing and the pointer are Windows niceties: without ``user32`` everything here answers
"cannot say" and the tool works without them.
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
VIEWPORT_CHECK = 3.0      # seconds between checks that the window's own zoom is still at x1
PROBE_TIMEOUT = 2.0       # a probe unanswered this long was lost (window hidden, being resized)
PROBE_TOLERANCE_PX = 4.0  # a probe reading may be off by this (whole-pixel readings) ...
PROBE_TOLERANCE_SHARE = 0.008   # ... or this share of the view width, whichever is more
RESET_WINDOW_S = 60.0     # window rebuilds are counted over this long ...
RESET_LIMIT = 3           # ... and more than this many means the check is measuring something else


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


def highgui_backend() -> str:
    """The GUI line of OpenCV's build information: QT6, WIN32UI, GTK3, ..."""
    for line in cv2.getBuildInformation().splitlines():
        if line.strip().startswith("GUI:"):
            return line.split(":", 1)[1].strip()
    return "unknown"
