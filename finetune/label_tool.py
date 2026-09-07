"""Correct the detector's draft labels as fast as possible.

    python finetune/label_tool.py <clip>                 # videos/<clip>.mp4 + labels/<clip>_ball.csv
    python finetune/label_tool.py video22..video53      # a whole run, one clip after another
    python finetune/label_tool.py video22..video53 --uncertain   # only the frames worth a look

Several clips run as one session: the last frame of one leads straight into the first of the next,
n and p cross the join, zoom carries over, and every edit is written back to the clip the frame
came from. The HUD names the clip you are in.

Every label frame (every 2nd frame of a 60 FPS clip, every frame at 30 FPS)
comes up with the detector's guess drawn and — on Windows — the mouse cursor
already sitting on it.  If it is right, press SPACE (or ENTER / c).  If it is
wrong, click where the ball is.  Both move to the next frame.  The view keeps
the ball in the middle and the mouse goes with it, so the pointer is always
sitting on the next guess.  The guess is dead centre on every frame, even one
at the very edge of the picture — the view runs off the frame there and that
strip is drawn black; zoom stays where you set it.  The window is
resizable — drag it, maximise it, or press m for full screen — and the picture
is redrawn at whatever size you give it.

Controls:
  left click               ball is here -> save and go to next frame
  SPACE / ENTER / c        guess is good -> next frame
  v                        ball not visible on this frame -> next frame
  r                        accept the whole run of confident guesses up to the next iffy one
  mouse wheel, + / -       smooth zoom in / out about the mouse / the ball (kept across frames)
  right click              re-centre the view here
  i / j / k / l            nudge the point one pixel up / left / down / right (stays on frame)
  n / d / right arrow      next frame        p / a / left arrow   previous frame
  1 / 2 / 4 / 6 / 8        how many frames those four move at a time - the x1..x8 buttons on the
                           bar do the same. Holding d or a at x8 covers ground eight times as
                           fast; SPACE, v and a click always move exactly one frame
  f / b                    jump to next / previous unreviewed uncertain frame - that includes any
                           frame flagged by ft.py check --audit --mark, drawn in magenta. SPACE
                           clears a flag as well as a click does: it is good where it is
  u                        undo last change on this frame
  g / click the badge      auto-centre on / off. On (the default) every frame puts the guess in
                           the middle of the screen with the mouse on it; off leaves the view and
                           the mouse alone, for looking round a frame
  m                        full screen on / off (the window also maximises normally)
  t                        toggle trail       s   save        q / ESC   save and quit

--uncertain walks only the frames that want attention - no confident guess, or flagged by
ft.py check --audit --mark - and skips everything the detector already got right. n/p/SPACE step
between those frames; the rest of the labels are untouched and still saved.

Output: labels/<clip>_ball.csv (frame,ball_x,ball_y).  Review progress lives
in labels/<clip>_ball.review.json so you can quit and resume.  Everything is
autosaved every few seconds and on any exit; a frame you already dealt with
shows a REVIEWED (space) or EDITED (click / v / nudge) badge in the top-right
corner when you come back to it, and the top-left corner counts how many
frames are left to do.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

WORKSPACE = Path(__file__).resolve().parent
VIDEOS = WORKSPACE / "videos"
LABELS = WORKSPACE / "labels"
WINDOW = "label_tool"
HUD = 58
CONFIDENT_CONF = 0.70
MAX_UPSCALE = 2.0         # cap on drawing the frame bigger than it is: past this it only costs time
UI_REFERENCE = 1280.0     # view width the HUD text sizes were chosen for; bigger views scale up
ZOOM_STEP = 1.18          # per wheel notch / keypress; multiplicative so every step feels the same size
ZOOM_EASE = 0.34          # share of the remaining gap closed each redraw: a glide, not a jump
ZOOM_MAX = 12.0
WHEEL_QUIET = 0.25        # seconds after a wheel notch during which the mouse is left alone
FORWARD_DECODE = 90       # frames worth decoding through rather than seeking (a seek costs far more)
PREFETCH = 6              # label frames decoded ahead while the tool waits for the next keypress
SEEK_BACKUP = 6           # frames to land before a seek target, so a known frame can re-anchor us
CACHE_BYTES = 250_000_000  # decoded frames kept in memory, as a budget: 1080p frames are 6 MB each
AUTOSAVE_SECONDS = 5.0    # write labels + review progress this often while there are unsaved changes

try:  # cursor placement and window sizing are Windows niceties; everything else works without them
    import ctypes
    _user32 = ctypes.windll.user32

    class _POINT(ctypes.Structure):
        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

    class _RECT(ctypes.Structure):
        _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                    ("right", ctypes.c_long), ("bottom", ctypes.c_long)]
except (AttributeError, ImportError, OSError):
    _user32 = None


def work_area():
    """The desktop minus the taskbar, in physical screen pixels, or None if the OS will not say."""
    if _user32 is None:
        return None
    rect = _RECT()
    if not _user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rect), 0):   # SPI_GETWORKAREA
        return None
    return rect.right - rect.left, rect.bottom - rect.top, rect.left, rect.top


def window_client():
    """(origin_x, origin_y, width, height) of the window's client area, in physical screen pixels.

    Everything about placing the mouse has to come from here. cv2.getWindowImageRect answers in
    *image* pixels while SetCursorPos works in screen pixels, and highgui puts an image inside the
    client area at a scale of its own choosing - the monitor DPI when the window sizes itself to
    the image, whatever fits once you resize it. Measuring beats assuming.
    """
    if _user32 is None:
        return None
    hwnd = _user32.FindWindowW(None, WINDOW)
    if not hwnd:
        return None
    origin, client = _POINT(), _RECT()
    if not (_user32.ClientToScreen(hwnd, ctypes.byref(origin))
            and _user32.GetClientRect(hwnd, ctypes.byref(client))):
        return None
    if client.right <= 0 or client.bottom <= 0:
        return None
    return origin.x, origin.y, client.right, client.bottom


def cursor_screen():
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


def inside_client(point) -> bool:
    """Is this screen pixel inside our window's client area? True when the OS will not say."""
    geometry = window_client()
    if geometry is None:
        return True
    origin_x, origin_y, client_w, client_h = geometry
    return (origin_x <= point[0] < origin_x + client_w
            and origin_y <= point[1] < origin_y + client_h)


def window_chrome():
    """(width, height) the frame and title bar add on top of the client area."""
    if _user32 is None:
        return 0, 0
    hwnd = _user32.FindWindowW(None, WINDOW)
    if not hwnd:
        return 0, 0
    window, client = _RECT(), _RECT()
    if not (_user32.GetWindowRect(hwnd, ctypes.byref(window)) and _user32.GetClientRect(hwnd, ctypes.byref(client))):
        return 0, 0
    return (window.right - window.left) - client.right, (window.bottom - window.top) - client.bottom


class Row:
    __slots__ = ("frame", "x", "y", "visible", "source", "conf", "reviewed", "edited", "suspect",
                 "history", "clip")

    def __init__(self, frame: int, x: float, y: float, visible: bool, source: str, conf: float):
        self.frame, self.x, self.y, self.visible = frame, x, y, visible
        self.clip = ""                # which clip this row belongs to, set by Source
        self.source, self.conf = source, conf
        self.reviewed = False
        self.edited = False
        self.suspect = False          # flagged by ft.py check --audit --mark; cleared once you edit it
        self.history: List[tuple] = []

    @property
    def settled(self) -> bool:
        """You have looked at this frame and said something about it - either way it is dealt with."""
        return self.reviewed or self.edited

    @property
    def uncertain(self) -> bool:
        """Worth stopping on: no confident guess, or something already found this frame doubtful."""
        return (self.suspect and not self.settled) or not (self.visible and self.conf >= CONFIDENT_CONF)

    def snapshot(self) -> None:
        self.history.append((self.x, self.y, self.visible))

    def undo(self) -> None:
        if self.history:
            self.x, self.y, self.visible = self.history.pop()
            self.edited = bool(self.history)


def hit(box, x: int, y: int) -> bool:
    """Did a click at (x, y) land on this HUD button? Boxes are in display pixels."""
    return bool(box) and box[0] <= x <= box[2] and box[1] <= y <= box[3]


def fit_text(text: str, width: int, scale: float, thin: int) -> str:
    """`text` cut to what fits in `width` pixels. Guessing from the average character width gets it
    about right in one go; the loop is there for the handful of pixels that guess is out by."""
    measure = lambda s: cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, thin)[0][0]
    full = measure(text)
    if full <= width:
        return text
    text = text[:max(0, int(len(text) * width / full))]
    while text and measure(text) > width:
        text = text[:-1]
    return text


def is_visible(x: float, y: float, width: int, height: int) -> bool:
    return not (x >= width * 0.95 and y <= height * 0.05)


def load_rows(csv_path: Path, width: int, height: int, origin: str = "det") -> List[Row]:
    """origin names where a row with no source column of its own came from: "csv" for labels that
    arrived with the clip, "det" for ones pretrack.py guessed. Only pretrack writes a conf column,
    and the tool drops it on save, so without this a hand-labelled clip read back as "det 1.00"."""
    rows = []
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        for record in csv.DictReader(handle):
            frame = int(re.fullmatch(r"frame_(\d+)", record["frame"].strip()).group(1))
            x, y = float(record["ball_x"]), float(record["ball_y"])
            visible = is_visible(x, y, width, height)
            conf = record.get("conf")
            rows.append(Row(frame, x if visible else width / 2.0, y if visible else height / 2.0, visible,
                            record.get("source") or (origin if visible else "none"),
                            float(conf) if conf not in (None, "") else (1.0 if visible else 0.0)))
    rows.sort(key=lambda row: row.frame)
    return rows


def save_rows(csv_path: Path, rows: List[Row], width: int, height: int) -> None:
    temporary = csv_path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "ball_x", "ball_y"])
        for row in rows:
            if row.visible:
                writer.writerow([f"frame_{row.frame:03d}", f"{row.x:.2f}", f"{row.y:.2f}"])
            else:
                writer.writerow([f"frame_{row.frame:03d}", width - 1, 0])
    temporary.replace(csv_path)


def save_suspects(path: Path, rows: List[Row]) -> None:
    """Drop the frames you have settled from the flag file, and remove it once none are left.

    A flag is a question, and confirming the label answers it just as well as changing the label
    does. Leaving answered ones behind would keep the clip marked BAD* for work already done.
    """
    if not path.is_file():
        return
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    settled = {row.frame for row in rows if row.settled}
    kept = {kind: [f for f in blob.get(kind, []) if int(f) not in settled]
            for kind in ("unplaceable", "check")}
    if not (kept["unplaceable"] or kept["check"]):
        path.unlink(missing_ok=True)
        return
    if all(kept[kind] == blob.get(kind, []) for kind in kept):
        return
    blob.update(kept)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(blob, indent=1), encoding="utf-8")
    temporary.replace(path)


def save_review(path: Path, rows: List[Row], cursor: int, zoom: float) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps({
        "cursor": cursor,
        "zoom": zoom,
        "reviewed": [row.frame for row in rows if row.reviewed],
        "edited": [row.frame for row in rows if row.edited],
    }), encoding="utf-8")
    temporary.replace(path)


def load_suspects(path: Path, rows: List[Row]) -> int:
    """Mark the frames a previous audit flagged. They are the ones worth your time first."""
    if not path.is_file():
        return 0
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
        frames = set(int(f) for f in blob.get("unplaceable", [])) | \
                 set(int(f) for f in blob.get("check", blob.get("frames", [])))
    except (OSError, ValueError):
        return 0
    for row in rows:
        row.suspect = row.frame in frames
    return sum(row.suspect for row in rows)


def load_review(path: Path, rows: List[Row]):
    if not path.is_file():
        return None
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    reviewed, edited = set(state.get("reviewed", [])), set(state.get("edited", []))
    for row in rows:
        row.reviewed = row.frame in reviewed
        row.edited = row.frame in edited
    return state


class Source:
    """One clip in a run: its video, its rows, and the files its work goes back to.

    Everything that is about a particular video lives here - the capture, where it is positioned,
    the frame fingerprints, the decoded-frame cache - so that working through several clips in one
    sitting cannot mix them up. Edits always go back to the clip the row came from.
    """

    def __init__(self, clip: str, video: Path, rows: List[Row], width: int, height: int,
                 out_csv: Path, review: Path):
        self.clip, self.video, self.rows = clip, video, rows
        self.width, self.height = width, height
        self.out_csv, self.review = out_csv, review
        self.suspects = review.with_name(review.name.replace("_ball.review.json", "_ball.suspect.json"))
        for row in rows:
            row.clip = clip
        self.capture = cv2.VideoCapture(str(video))
        self.frame_cache: Dict[int, np.ndarray] = {}
        self.cache_limit = max(24, min(96, CACHE_BYTES // max(width * height * 3, 1)))
        self.position = 0             # next frame the capture will decode, so stepping can avoid seeks
        self.index_of: Dict[bytes, int] = {}   # frame fingerprint -> its real index, see decode_at
        self.sig_of: Dict[int, bytes] = {}     # ... and back again, to check a frame is the one wanted
        self.warned_seek = False
        self.dirty = False
        self.cursor = 0               # where you were in this clip, for the review file
        self.start = 0                # where its rows begin in the run's flat row list
        self.zoom = 2.0


class Tool:
    def __init__(self, sources: List["Source"]):
        self.sources = sources
        self.rows: List[Row] = []
        for source in sources:
            source.start = len(self.rows)
            self.rows.extend(source.rows)
        self.by_clip = {source.clip: source for source in sources}
        self.width, self.height = sources[0].width, sources[0].height
        self.base_scale = 1.0                  # replaced by fit_view once the window has a size
        self.view_w, self.view_h = self.width, self.height
        self.ui = 1.0                          # HUD/marker scale, so text stays readable when enlarged
        self.hud_h = HUD
        self.client = None                     # last client size seen; a change re-fits and re-measures
        self.cursor_map = None                 # (scale_x, scale_y, offset_x, offset_y) view px -> client px
        self.map_stale = True                  # until calibrate_cursor has measured the real one
        self.fullscreen = False
        self.zoom = 1.0
        self.zoom_target = 1.0
        self.anchor = None            # (frame point, view point) the zoom glide holds still
        self.center = (self.width / 2.0, self.height / 2.0)
        self.origin = (0.0, 0.0)
        self.cursor = 0
        self.walk: List[int] = []     # row indices navigation visits; empty means every row
        self.walk_pos = 0
        self.trail = True
        self.follow = True            # auto-centre: the view and the mouse go to the guess
        self.step = 1                 # frames n/d/p/a move at a time; SPACE and a click stay at 1
        self.centre_button = None     # where the HUD badge that toggles that is, in display pixels
        self.step_buttons = []        # ... and the x1..x8 ones, as (box, frames) pairs
        self.dirty = False
        self.pending_cursor_move = False
        self.pointer = None           # last place highgui reported the mouse, in view pixels
        self.pointer_seq = 0          # bumped on every report, so a fresh one can be waited for
        self.aim = None               # the last warp, until check_placement has confirmed it
        self.map_time = 0.0           # when the map was last measured
        self.wheel_at = 0.0           # when the wheel last turned; the mouse is left alone just after
        self.last_save = time.monotonic()

    # ---------------------------------------------------------------- the clip we are in
    @property
    def source(self) -> "Source":
        return self.by_clip[self.rows[self.cursor].clip]

    @property
    def clip(self) -> str:
        return self.source.clip

    # ---------------------------------------------------------------- geometry
    @property
    def scale(self) -> float:
        return self.base_scale * self.zoom

    def open_window(self) -> None:
        """Create a resizable window filling the desktop. WINDOW_NORMAL, not WINDOW_AUTOSIZE: autosize
        lets highgui pick the size from the image and the monitor DPI, which on a 200% display made a
        1080p clip a 3226x2095 window on a 2880x1800 desktop, and left no way to maximise it."""
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        cv2.imshow(WINDOW, np.zeros((self.height, self.width, 3), dtype=np.uint8))
        cv2.waitKey(1)
        area = work_area()
        hwnd = _user32.FindWindowW(None, WINDOW) if _user32 else None
        if area is None or not hwnd:
            return
        work_w, work_h, work_x, work_y = area
        chrome_w, chrome_h = window_chrome()
        _user32.SetWindowPos(hwnd, 0, work_x, work_y, work_w, work_h, 0x0004)   # SWP_NOZORDER
        cv2.waitKey(1)

    def fit_view(self, client_w: int, client_h: int) -> None:
        """Draw at the size the window actually is, so enlarging it gains detail instead of blur."""
        self.ui = max(1.0, min(client_w / UI_REFERENCE, 2.5))
        self.hud_h = int(round(HUD * self.ui))
        room_h = max(120, client_h - self.hud_h)
        self.base_scale = max(0.2, min(client_w / self.width, room_h / self.height, MAX_UPSCALE))
        self.view_w = max(160, int(round(self.width * self.base_scale)))
        self.view_h = max(90, int(round(self.height * self.base_scale)))
        self.centre_view()

    def track_window(self) -> None:
        """Notice the window being resized, maximised or full-screened, and follow it: both the render
        size and the view-pixel-to-screen-pixel map depend on how big the client area is."""
        geometry = window_client()
        if geometry is None:
            return
        size = (geometry[2], geometry[3])
        if size == self.client:
            return
        self.client = size
        self.fit_view(*size)
        self.map_stale = True

    def toggle_fullscreen(self) -> None:
        self.fullscreen = not self.fullscreen
        cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL)
        cv2.waitKey(1)

    def centre_view(self) -> None:
        """Work out the corner to draw from: whatever we want in the middle, put in the middle.

        Nothing is pulled back to keep the picture full. A ball at the very edge of the frame is
        centred like any other and the view simply runs off the side, which renders as black - the
        frame has nothing there to show anyway. Holding the view inside the frame is what dragged
        the ball left on one frame and right on the next as a rally crossed the court, and up
        against the top on a lob; at a big zoom that is most frames. Dead centre every time means
        the ball, and the mouse sitting on it, never move from the middle of the screen.

        self.center is what we want in the middle and is left exactly as asked for, so a later
        change of zoom re-derives this from the ball rather than from a corner it worked out
        earlier: that is what used to walk the view down off a ball near the top as it zoomed.
        """
        self.origin = (self.center[0] - self.view_w / self.scale / 2.0,
                       self.center[1] - self.view_h / self.scale / 2.0)

    def to_view(self, x: float, y: float):
        return int(round((x - self.origin[0]) * self.scale)), int(round((y - self.origin[1]) * self.scale))

    def to_frame(self, vx: float, vy: float):
        return self.origin[0] + vx / self.scale, self.origin[1] + vy / self.scale

    def toggle_follow(self) -> None:
        """Auto-centring on or off, from the g key or a click on the HUD badge.

        Off leaves the view and the mouse exactly where they are on every frame, for reading a
        frame or working somewhere other than the middle of the screen; back on snaps to the ball
        straight away rather than waiting for the next frame.
        """
        self.follow = not self.follow
        if self.follow:
            self.follow_ball()

    def follow_ball(self) -> None:
        """Centre the view on this frame's ball, so it is always in the middle and the mouse - which
        goes to the same place - never has to be hunted for.

        Dropping the zoom anchor here matters. It belongs to the frame you were on when you turned
        the wheel, and a glide still in flight would otherwise keep hauling the view back to that
        frame's ball - so scrolling and then pressing SPACE straight away left you looking at, and
        with the mouse parked on, the previous detection.
        """
        self.anchor = None
        if not self.follow:
            return                    # auto-centring is off: leave the view and the mouse be
        row = self.rows[self.cursor]
        if row.visible:
            self.center = (row.x, row.y)
        self.centre_view()
        self.pending_cursor_move = True

    # ---------------------------------------------------------------- frames
    @staticmethod
    def signature(image) -> bytes:
        """A cheap fingerprint of a frame, enough to recognise one we have decoded before."""
        return hashlib.blake2b(cv2.resize(image, (96, 54), interpolation=cv2.INTER_AREA).tobytes(),
                               digest_size=8).digest()

    def decode_at(self, index: int):
        """Decode exactly `index`, or None if we ended up somewhere else.

        Neither counter can be trusted. On a clip cut and merged from two recordings, a seek past
        the join lands one frame early while CAP_PROP_POS_FRAMES still reports the frame asked for
        (POS_MSEC comes back negative, so that is no help either). The old code took the seek on
        faith, so one bad seek silently shifted everything decoded after it: the tool drew frame
        N-1 while the row being edited was N, and clicks made exactly on the ball were stored
        against the next row.

        Walk sequentially from the start to establish source identity. Fingerprints only
        check consistency at an already known index: duplicate images cannot identify
        a timestamp. A failed decode is never returned as a valid annotation image.
        """
        source = self.source
        if source.position < 0 or index < source.position:
            self.seek_and_resync(index)
        while 0 <= source.position < index:
            if not source.capture.grab():
                source.position = -1
                return None
            source.position += 1
        ok, image = source.capture.read()
        if not ok:
            source.position = -1
            return None
        at, source.position = source.position, source.position + 1
        signature = self.signature(image)
        if source.sig_of.get(index) not in (None, signature) and at == index:
            source.position = -1                         # not the frame we know sits here
            return None
        self.remember(at, signature)
        return image if at == index else None

    def seek_and_resync(self, index: int) -> None:
        """Reopen at the start; decode_at walks forward by source frame count.

        Duplicate pixels cannot identify a unique timestamp, and compressed-video
        seeks can report a false index. Only sequential decode establishes it.
        """
        source = self.source
        # ponytail: uncached backward jumps decode O(index) frames; use a verified
        # decoder index if this becomes slow on long clips. Cached jumps stay fast.
        source.capture.release()
        source.capture = cv2.VideoCapture(str(source.video))
        source.position = 0

    def remember(self, index: int, signature: bytes) -> None:
        source = self.source
        source.sig_of[index] = signature

    def frame(self, index: int) -> np.ndarray:
        """Decode one frame, walking forward where possible: seeking a long GOP costs far more than
        decoding through it, and stepping frame by frame is what this tool does all day."""
        source = self.source
        cached = source.frame_cache.get(index)
        if cached is not None:
            return cached
        image = self.decode_at(index)
        for _ in range(3):                      # lost the thread: re-seek, now knowing where we are
            if image is not None:
                break
            image = self.decode_at(index)
        if image is None:
            raise RuntimeError(f"Cannot verify frame {index} of {source.clip}; no label placeholder was cached")
        if len(source.frame_cache) > source.cache_limit:
            source.frame_cache.pop(next(iter(source.frame_cache)))
        source.frame_cache[index] = image
        return image

    def prefetch(self) -> bool:
        """Decode one not-yet-cached frame ahead of the cursor. Called while the tool is idle, so the
        next SPACE lands on a frame that is already in memory. True when there was work to do."""
        upcoming = (self.walk[self.walk_pos + 1:self.walk_pos + 1 + PREFETCH] if self.walk
                    else range(self.cursor + 1, min(self.cursor + 1 + PREFETCH, len(self.rows))))
        for index in upcoming:
            if self.rows[index].clip != self.rows[self.cursor].clip:
                break                                    # do not disturb the next clip's capture yet
            if self.rows[index].frame not in self.source.frame_cache:
                self.frame(self.rows[index].frame)
                return True
        return False

    def render(self) -> np.ndarray:
        self.centre_view()
        row = self.rows[self.cursor]
        image = self.frame(row.frame)
        x0, y0 = self.origin
        # Resize with exact sub-pixel origin handling via warpAffine.
        matrix = np.array([[self.scale, 0.0, -x0 * self.scale], [0.0, self.scale, -y0 * self.scale]], dtype=np.float64)
        display = cv2.warpAffine(image, matrix, (self.view_w, self.view_h),
                                 flags=cv2.INTER_LINEAR if self.zoom < 8 else cv2.INTER_NEAREST)
        ui = self.ui
        thin = max(1, int(round(ui)))
        if self.trail:
            for previous in self.rows[max(0, self.cursor - 12):self.cursor]:
                if previous.visible:
                    cv2.circle(display, self.to_view(previous.x, previous.y), max(3, int(3 * ui)),
                               (200, 200, 60), thin, cv2.LINE_AA)
        if row.visible:
            colour = ((0, 0, 255) if row.edited else (0, 220, 0) if row.reviewed
                      else (255, 0, 255) if row.suspect else (0, 255, 255))
            cx, cy = self.to_view(row.x, row.y)
            radius = int(max(8, 6 * self.zoom) * ui)
            cv2.circle(display, (cx, cy), radius, colour, thin, cv2.LINE_AA)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                cv2.line(display, (cx + dx * (radius - int(3 * ui)), cy + dy * (radius - int(3 * ui))),
                         (cx + dx * (radius + int(8 * ui)), cy + dy * (radius + int(8 * ui))), colour, thin)
        if row.settled or row.suspect:
            badge = "EDITED" if row.edited else "REVIEWED" if row.reviewed else "CHECK THIS"
            fill = (60, 140, 255) if row.edited else (70, 190, 70) if row.reviewed else (200, 60, 200)
            (text_w, text_h), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * ui, 2)
            left = self.view_w - text_w - int(26 * ui)
            cv2.rectangle(display, (left, int(10 * ui)), (self.view_w - int(10 * ui), int(24 * ui) + text_h), fill, -1)
            cv2.putText(display, badge, (left + int(8 * ui), int(17 * ui) + text_h), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6 * ui, (0, 0, 0), 2, cv2.LINE_AA)
        reviewed = sum(r.reviewed for r in self.rows)
        remaining = len(self.rows) - reviewed
        counter = "ALL DONE" if remaining == 0 else f"{remaining} LEFT"
        fill, text_colour = ((70, 190, 70), (0, 0, 0)) if remaining == 0 else ((50, 50, 50), (255, 255, 255))
        (text_w, text_h), _ = cv2.getTextSize(counter, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * ui, 2)
        cv2.rectangle(display, (int(10 * ui), int(10 * ui)), (int(26 * ui) + text_w, int(24 * ui) + text_h), fill, -1)
        cv2.putText(display, counter, (int(18 * ui), int(17 * ui) + text_h), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * ui,
                    text_colour, 2, cv2.LINE_AA)
        hud = np.zeros((self.hud_h, self.view_w, 3), dtype=np.uint8)
        uncertain_left = sum(1 for r in self.rows if r.uncertain and not r.reviewed)
        flagged_left = sum(1 for r in self.rows if r.suspect and not r.settled)
        state = "NOT VISIBLE" if not row.visible else f"({row.x:.1f}, {row.y:.1f})"
        tag = "EDITED" if row.edited else "ok" if row.reviewed else "unreviewed"
        source = self.source
        within = f"{self.cursor - source.start + 1}/{len(source.rows)}"
        place = (f"{source.clip} [{within}]" if len(self.sources) == 1 else
                 f"{source.clip} [{within}]  clip {self.sources.index(source) + 1}/{len(self.sources)}"
                 f"  run {self.cursor + 1}/{len(self.rows)}")
        if self.walk:
            left = sum(1 for i in self.walk if not self.rows[i].settled)
            place += f"  UNCERTAIN ONLY {self.walk_pos + 1}/{len(self.walk)} ({left} left)"
        line1 = (f"{place}  frame {row.frame}  {state}  "
                 f"from {row.source} {row.conf:.2f}  {tag}  zoom x{self.zoom:.1f}{'  *unsaved*' if self.dirty else ''}")
        line2 = (f"reviewed {reviewed}/{len(self.rows)}  uncertain left {uncertain_left}  "
                 f"{('flagged left ' + str(flagged_left) + '   ') if flagged_left else '  '}"
                 "SPACE=good  v=invisible  r=accept run  f=uncertain  q=quit")
        # Buttons along the right end of the bar, laid out right to left and measured before the
        # two text lines, so a long clip name or a narrow window cuts the text short instead of
        # running under them. Every box is kept for on_mouse: they are all clickable.
        pad = int(6 * ui)
        (_, glyph_h), _ = cv2.getTextSize("Xy8", cv2.FONT_HERSHEY_SIMPLEX, 0.42 * ui, thin)
        top = max(0, (self.hud_h - glyph_h - pad) // 2)
        bottom = min(self.hud_h - 1, top + glyph_h + pad)

        def chip(right_edge: int, text: str, lit: bool):
            (text_w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42 * ui, thin)
            left_edge = max(0, right_edge - text_w - 2 * pad)
            cv2.rectangle(hud, (left_edge, top), (right_edge, bottom),
                          (70, 190, 70) if lit else (55, 55, 55), -1)
            cv2.putText(hud, text, (left_edge + pad, bottom - pad // 2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42 * ui, (0, 0, 0) if lit else (190, 190, 190), thin, cv2.LINE_AA)
            return left_edge, (left_edge, self.view_h + top, right_edge, self.view_h + bottom)

        edge, self.centre_button = chip(self.view_w - pad,
                                        f"g  AUTO-CENTRE {'ON' if self.follow else 'OFF'}", self.follow)
        self.step_buttons = []
        for jump in (8, 6, 4, 2, 1):                     # right to left, so x1 comes out leftmost
            edge, box = chip(edge - pad, f"x{jump}", jump == self.step)
            self.step_buttons.append((box, jump))
        caption = "a/d step"
        (caption_w, _), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.4 * ui, thin)
        edge = max(0, edge - pad - caption_w)
        cv2.putText(hud, caption, (edge, bottom - pad // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * ui,
                    (150, 150, 150), thin, cv2.LINE_AA)
        room = max(0, edge - int(16 * ui))
        cv2.putText(hud, fit_text(line1, room, 0.55 * ui, thin), (int(8 * ui), int(22 * ui)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55 * ui, (255, 255, 255), thin, cv2.LINE_AA)
        cv2.putText(hud, fit_text(line2, room, 0.48 * ui, thin), (int(8 * ui), int(46 * ui)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48 * ui, (180, 180, 180), thin, cv2.LINE_AA)
        return np.vstack([display, hud])

    def screen_for(self, view_x: float, view_y: float):
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
        view_x = min(max(view_x, 0.0), self.view_w - 1.0)
        view_y = min(max(view_y, 0.0), self.view_h - 1.0)
        target = self.screen_for(view_x, view_y)
        if target is None:
            return False
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
            if time.monotonic() - aimed_at > 0.30:
                self.aim = None
                if not inside_client(target) and time.monotonic() - self.map_time > 2.0:
                    self.map_stale = True   # warped clean off the window: that map is not a map
            return
        seen = self.pointer
        self.aim = None
        if seen is None or time.monotonic() - aimed_at > 0.30:
            return
        where = cursor_screen()
        if where is None or abs(where[0] - target[0]) > 2 or abs(where[1] - target[1]) > 2:
            return                                        # the pointer has moved on since: not ours
        error_x, error_y = seen[0] - view_x, seen[1] - view_y
        if abs(error_x) < 1.0 and abs(error_y) < 1.0:
            return                                        # dead on: nothing to mend
        scale_x, scale_y, offset_x, offset_y = self.cursor_map
        self.cursor_map = (scale_x, scale_y,
                           offset_x - error_x * scale_x, offset_y - error_y * scale_y)
        if tries < 3:
            self.warp(view_x, view_y, tries + 1)
        elif time.monotonic() - self.map_time > 2.0:
            self.map_stale = True            # offsets alone are not closing it: measure again

    def probe(self, screen_x: int, screen_y: int):
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
        self.await_pointer(seq, 0.10)
        seq = self.pointer_seq
        _user32.SetCursorPos(screen_x, screen_y)
        return self.await_pointer(seq, 0.25)

    def await_pointer(self, seq: int, seconds: float):
        """Pump highgui until it reports a pointer position newer than `seq`. None if it never does."""
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            cv2.waitKey(5)
            if self.pointer_seq != seq:
                return self.pointer
        return None

    def scrolling(self) -> bool:
        """True just after a wheel notch. A wheel arrives in big discrete steps, and moving the mouse
        between them moves what the next notch zooms about - which is what made the wheel feel jumpy
        next to a trackpad, where the deltas are small and continuous. So leave the mouse alone until
        the scrolling stops, then put it back on the ball."""
        return time.monotonic() - self.wheel_at < WHEEL_QUIET

    def place_cursor(self) -> None:
        """Put the OS mouse cursor on the current guess so a click is a one-pixel move.

        follow_ball has already put the ball in the middle of the view, so this normally lands the
        pointer dead centre. A frame with no ball has nothing to aim at, and the pointer goes to
        the centre anyway rather than being left wherever the last frame happened to leave it.
        """
        self.pending_cursor_move = False
        if _user32 is None or not self.follow:
            return
        if self.cursor_map is None or self.map_stale:
            self.calibrate_cursor()                  # never measured, or the window changed size
        row = self.rows[self.cursor]
        if not row.visible:
            self.warp(self.view_w / 2.0, self.view_h / 2.0)
            return
        view_x, view_y = self.to_view(row.x, row.y)
        if not (0 <= view_x < self.view_w and 0 <= view_y < self.view_h):
            self.center = (row.x, row.y)         # nothing should leave the ball off screen, but if
            self.centre_view()                    # anything does, do not leave the mouse behind
            view_x, view_y = self.to_view(row.x, row.y)
        self.warp(view_x, view_y)

    def calibrate_cursor(self) -> None:
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
        if self.cursor_map is None:    # something to aim with until the measurement comes back
            guess = min(client_w / max(self.view_w, 1), client_h / max(self.view_h + self.hud_h, 1))
            self.cursor_map = (guess, guess, 0.0, 0.0)
        self.map_stale = False
        self.map_time = time.monotonic()
        near = (int(client_w * 0.2), int(client_h * 0.2))    # well inside the picture either way,
        far = (int(client_w * 0.8), int(client_h * 0.8))     # and far enough apart to solve cleanly
        seen_near = self.probe(origin_x + near[0], origin_y + near[1])
        seen_far = self.probe(origin_x + far[0], origin_y + far[1])
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
            if abs(spread) < 5:
                return                                      # the pointer barely moved: keep the map
            true_scale = (far[axis] - near[axis]) / spread
            if not 0.05 <= true_scale <= 20.0:
                return                                      # a stray hand on the mouse: keep the map
            solved[axis] = true_scale
            solved[2 + axis] = near[axis] - seen_near[axis] * true_scale
        self.cursor_map = tuple(solved)

    # ---------------------------------------------------------------- edits
    def on_mouse(self, event, x, y, flags, param):
        self.pointer = (x, y)                       # where the warp landed; check_placement reads it
        self.pointer_seq += 1
        if y >= self.view_h:                        # the HUD: its buttons, and nothing else
            if event == cv2.EVENT_LBUTTONDOWN:
                if hit(self.centre_button, x, y):
                    self.toggle_follow()
                for box, jump in self.step_buttons:
                    if hit(box, x, y):
                        self.step = jump
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            fx, fy = self.to_frame(x, y)
            self.set_point(fx, fy)
            self.move(1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            frame_x, frame_y = self.to_frame(x, y)
            self.center = (min(max(frame_x, 0.0), self.width - 1.0),   # a click in the black beyond
                           min(max(frame_y, 0.0), self.height - 1.0))  # the frame: keep it on frame
            self.centre_view()
        elif event == cv2.EVENT_MOUSEWHEEL:
            self.wheel_at = time.monotonic()
            self.zoom_at(self.wheel_notches(flags), float(x), float(y))

    @staticmethod
    def wheel_notches(flags) -> float:
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

    def zoom_at(self, notches: float, vx: float, vy: float) -> None:
        """Aim the zoom at a new level about the view point (vx, vy). The glide there happens in
        step_zoom; whatever sits under (vx, vy) stays under it the whole way."""
        target = min(max(self.zoom_target * ZOOM_STEP ** notches, 1.0), ZOOM_MAX)
        if abs(target - self.zoom_target) < 1e-6:
            return
        self.zoom_target = target
        self.anchor = (self.to_frame(vx, vy), (vx, vy))

    def step_zoom(self) -> bool:
        """Ease one redraw's worth towards the target zoom. True while there is still moving to do."""
        gap = self.zoom_target - self.zoom
        if abs(gap) < 5e-3:
            if self.zoom != self.zoom_target:
                self.zoom = self.zoom_target
                self.hold_anchor()
                return True
            return False
        self.zoom += gap * ZOOM_EASE
        self.hold_anchor()
        return True

    def hold_anchor(self) -> None:
        """Re-centre so the anchored frame point stays under the same view point at the new zoom."""
        if self.anchor is None:
            return
        (frame_x, frame_y), (view_x, view_y) = self.anchor
        self.center = (frame_x + (self.view_w / 2.0 - view_x) / self.scale,
                       frame_y + (self.view_h / 2.0 - view_y) / self.scale)
        self.centre_view()

    def zoom_key(self, notches: float) -> None:
        """Keyboard zoom pivots on the ball so it stays put on screen (view centre when invisible)."""
        row = self.rows[self.cursor]
        if row.visible:
            vx, vy = self.to_view(row.x, row.y)
            self.zoom_at(notches, float(vx), float(vy))
        else:
            self.zoom_at(notches, self.view_w / 2.0, self.view_h / 2.0)
        self.pending_cursor_move = True

    def set_point(self, x: float, y: float) -> None:
        row = self.rows[self.cursor]
        row.snapshot()
        row.x = float(min(max(x, 0.0), self.width - 1.0))
        row.y = float(min(max(y, 0.0), self.height - 1.0))
        row.visible = True
        row.reviewed = row.edited = True
        self.dirty = self.source.dirty = True

    def nudge(self, dx: float, dy: float) -> None:
        row = self.rows[self.cursor]
        if row.visible:
            self.set_point(row.x + dx, row.y + dy)
            self.pending_cursor_move = True

    def mark_invisible(self) -> None:
        row = self.rows[self.cursor]
        row.snapshot()
        row.visible = False
        row.reviewed = row.edited = True
        self.dirty = self.source.dirty = True

    def confirm(self) -> None:
        self.rows[self.cursor].reviewed = True
        self.dirty = self.source.dirty = True

    def only_uncertain(self) -> int:
        """Restrict stepping to the frames that want attention. Returns how many there are.

        A filter on where you go, never on what is saved: every row stays in its clip's list, so
        the frames you skip past are written back exactly as they were.
        """
        self.walk = [i for i, row in enumerate(self.rows) if row.uncertain and not row.settled]
        self.walk_pos = 0
        if self.walk:
            self.cursor = self.walk[0]
        return len(self.walk)

    def move(self, delta: int) -> None:
        if self.walk:
            self.walk_pos = min(max(self.walk_pos + delta, 0), len(self.walk) - 1)
            self.cursor = self.walk[self.walk_pos]
        else:
            self.cursor = min(max(self.cursor + delta, 0), len(self.rows) - 1)
        self.follow_ball()

    def accept_run(self) -> int:
        """Mark every confident guess from here up to the next iffy one as reviewed, and stop there.

        f already skips past confident frames, but it leaves them unreviewed, so the counter never
        moves and the same frames come round again. This is that skip with the tick applied.
        """
        accepted = 0
        while self.cursor < len(self.rows) - 1 and not self.rows[self.cursor].uncertain:
            if not self.rows[self.cursor].reviewed:
                self.rows[self.cursor].reviewed = True
                accepted += 1
            self.cursor += 1
        if accepted:
            self.dirty = self.source.dirty = True
        self.follow_ball()
        return accepted

    def jump(self, direction: int) -> None:
        if self.walk:                                  # already walking the uncertain ones
            position = self.walk_pos + direction
            while 0 <= position < len(self.walk):
                if not self.rows[self.walk[position]].settled:
                    self.walk_pos, self.cursor = position, self.walk[position]
                    self.follow_ball()
                    return
                position += direction
            self.walk_pos = len(self.walk) - 1 if direction > 0 else 0
            self.cursor = self.walk[self.walk_pos]
            self.follow_ball()
            return
        index = self.cursor + direction
        while 0 <= index < len(self.rows):
            row = self.rows[index]
            if row.uncertain and not row.reviewed:
                self.cursor = index
                self.follow_ball()
                return
            index += direction
        self.cursor = len(self.rows) - 1 if direction > 0 else 0
        self.follow_ball()

    def save(self, quiet: bool = False) -> None:
        """Write back every clip that has changed - and only those, so an untouched clip is left alone."""
        written = []
        for source in self.sources:
            if not source.dirty:
                continue
            if source.start <= self.cursor < source.start + len(source.rows):
                source.cursor = self.cursor - source.start
            save_rows(source.out_csv, source.rows, source.width, source.height)
            save_review(source.review, source.rows, source.cursor, self.zoom)
            save_suspects(source.suspects, source.rows)
            source.dirty = False
            written.append(source)
        self.dirty = False
        self.last_save = time.monotonic()
        if not quiet and written:
            for source in written:
                print(f"[save] {source.out_csv.name} ({sum(r.visible for r in source.rows)} visible, "
                      f"{sum(r.reviewed for r in source.rows)}/{len(source.rows)} reviewed)")

    # ---------------------------------------------------------------- loop
    def run(self) -> None:
        self.open_window()
        self.track_window()
        self.center = (self.width / 2.0, self.height / 2.0)
        self.centre_view()
        cv2.imshow(WINDOW, self.render())
        cv2.waitKey(1)
        self.calibrate_cursor()
        self.follow_ball()
        try:
            self.loop()
        finally:
            # Any way out - q, the window's X, Ctrl+C, a crash - keeps the work done so far.
            self.save()
            cv2.destroyAllWindows()

    def loop(self) -> None:
        while True:
            self.track_window()
            was_gliding = self.zoom != self.zoom_target
            gliding = self.step_zoom()
            cv2.imshow(WINDOW, self.render())
            if was_gliding and not gliding:
                self.pending_cursor_move = True       # zoom has settled: put the mouse back on the ball
            if self.pending_cursor_move and not self.scrolling():
                # Placed once per frame change, not on every redraw. follow_ball has centred the ball
                # and the glide has no anchor, so the ball stays under the cursor while the zoom
                # finishes - repeating the warp only fought a hand that was on the wheel.
                cv2.waitKey(1)
                self.place_cursor()
            key = cv2.waitKeyEx(8 if gliding else 30)
            self.check_placement()          # did that warp land on the ball? mend the map if not
            if self.dirty and time.monotonic() - self.last_save > AUTOSAVE_SECONDS:
                self.save(quiet=True)
            if key == -1:
                try:
                    if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                        return
                except cv2.error:
                    return
                if not gliding and self.aim is None:
                    self.prefetch()          # decode the frames SPACE is about to ask for, but not
                                             # while a warp is still waiting to be checked: a decode
                                             # runs long enough to age the answer out
                continue
            code = key & 0xFF if key < 0x10000 else key
            if code in (ord("q"), 27):
                break
            elif code in (ord(" "), 13, ord("c")):
                self.confirm()
                self.move(1)
            elif code == ord("v"):
                self.mark_invisible()
                self.move(1)
            elif code == ord("r"):
                accepted = self.accept_run()
                print(f"[run ] accepted {accepted} confident frame(s); stopped at frame "
                      f"{self.rows[self.cursor].frame}")
            elif code in (ord("n"), ord("d")) or key in (2555904, 83):
                self.move(self.step)
            elif code in (ord("p"), ord("a")) or key in (2424832, 81):
                self.move(-self.step)
            elif code in (ord("1"), ord("2"), ord("4"), ord("6"), ord("8")):
                self.step = code - ord("0")
            elif code == ord("f"):
                self.jump(1)
            elif code == ord("b"):
                self.jump(-1)
            elif code == ord("u"):
                self.rows[self.cursor].undo()
                self.dirty = self.source.dirty = True
                self.follow_ball()
            elif code in (ord("i"), ord("j"), ord("k"), ord("l")):
                dx, dy = {ord("i"): (0, -1), ord("j"): (-1, 0), ord("k"): (0, 1), ord("l"): (1, 0)}[code]
                self.nudge(dx, dy)
            elif code in (ord("+"), ord("=")):
                self.zoom_key(1.0)
            elif code in (ord("-"), ord("_")):
                self.zoom_key(-1.0)
            elif code == ord("m"):
                self.toggle_fullscreen()
            elif code == ord("t"):
                self.trail = not self.trail
            elif code == ord("g"):
                self.toggle_follow()
            elif code == ord("s"):
                self.save()


def expand(names: List[str], labels: Path) -> List[str]:
    """Turn what was typed into a clip list. 'a..b' means every labelled clip from a to b in order."""
    available = sorted((path.name[: -len("_ball.csv")] for path in labels.glob("*_ball.csv")), key=natural_key)
    out: List[str] = []
    for name in names:
        if ".." in name:
            first, last = (part.strip() for part in name.split("..", 1))
            if first not in available or last not in available:
                raise SystemExit(f"{first}..{last}: no labels for one of those in {labels}")
            lo, hi = available.index(first), available.index(last)
            out.extend(available[min(lo, hi):max(lo, hi) + 1])
        else:
            out.append(name)
    seen, unique = set(), []
    for name in out:                                  # keep the order, drop repeats
        if name not in seen:
            seen.add(name)
            unique.append(name)
    return unique


def natural_key(text: str) -> list:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def build_source(clip: str, videos: Path, labels: Path):
    """Everything one clip needs, or None with a message if it is not ready."""
    found = [p for p in videos.glob(f"{clip}.*") if p.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi", ".m4v"}]
    if not found:
        print(f"[skip] {clip}: no video in {videos}")
        return None
    video = found[0]
    capture = cv2.VideoCapture(str(video))
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    final, draft = labels / f"{clip}_ball.csv", labels / f"{clip}_ball.csv.draft"
    from_file = final if final.is_file() else draft
    if not from_file.is_file():
        print(f"[skip] {clip}: no labels; run  python finetune/pretrack.py --clips {clip}")
        return None
    rows = load_rows(from_file, width, height, "det" if draft.is_file() else "csv")
    review = labels / f"{clip}_ball.review.json"
    source = Source(clip, video, rows, width, height, final, review)
    flagged = load_suspects(source.suspects, rows)
    state = load_review(review, rows)
    if state is not None:
        source.cursor = min(int(state.get("cursor", 0)), len(rows) - 1)
        source.zoom = float(state.get("zoom", 2.0))
    source.flagged = flagged
    return source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clips", nargs="+", help="Clip stem(s), or a range like video22..video53")
    parser.add_argument("--videos", type=Path, default=VIDEOS)
    parser.add_argument("--labels", type=Path, default=LABELS)
    parser.add_argument("--zoom", type=float, default=None, help="Starting zoom (default: last used, else 2)")
    parser.add_argument("--uncertain", action="store_true",
                        help="Only stop on frames that want attention: no confident guess, or flagged "
                             "by ft.py check --audit --mark. Every other label is left alone.")
    args = parser.parse_args()

    names = expand(args.clips, args.labels)
    sources = [source for source in (build_source(name, args.videos, args.labels) for name in names)
               if source is not None]
    if not sources:
        return 1
    sizes = {(source.width, source.height) for source in sources}
    if len(sizes) > 1:
        print(f"[stop] these clips are not all the same size ({sorted(sizes)}); run them separately")
        return 1

    tool = Tool(sources)
    if args.uncertain:
        if not tool.only_uncertain():
            print("[done] nothing uncertain left in " + (names[0] if len(names) == 1 else
                                                         f"{names[0]}..{names[-1]}") +
                  " - every frame here has either a confident guess or your tick on it.")
            return 0
    elif len(sources) == 1:
        tool.cursor = sources[0].cursor                       # resume exactly where you left off
    else:                                                     # a run: carry on at the first frame not done
        tool.cursor = next((i for i, row in enumerate(tool.rows) if not row.settled), 0)
    tool.zoom = args.zoom or sources[0].zoom
    tool.zoom = min(max(tool.zoom, 1.0), ZOOM_MAX)
    tool.zoom_target = tool.zoom

    total = sum(len(source.rows) for source in sources)
    flagged = sum(getattr(source, "flagged", 0) for source in sources)
    if len(sources) == 1:
        source = sources[0]
        print(f"[open] {source.video.name}: {len(source.rows)} label frames, "
              f"{sum(r.uncertain for r in source.rows)} uncertain")
    else:
        done = sum(1 for row in tool.rows if row.settled)
        print(f"[open] {len(sources)} clips as one run: {names[0]} .. {names[-1]}, {total} label frames, "
              f"{done} already done. Starting at {tool.rows[tool.cursor].clip} "
              f"frame {tool.rows[tool.cursor].frame}.")
        print("       n/p and SPACE carry straight on into the next clip; each clip saves to its own file.")
    if args.uncertain:
        print(f"[only ] stepping through {len(tool.walk)} uncertain frame(s) of {total}; "
              f"n/p/SPACE move between those, everything else is left exactly as it is.")
    elif flagged:
        print(f"[flag] {flagged} frame(s) were marked doubtful by the last audit - press f to walk them; "
              f"they are drawn in magenta. Fixing or confirming one clears it.")
    tool.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
