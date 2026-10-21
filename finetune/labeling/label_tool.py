"""Correct the detector's draft labels as fast as possible.

    python finetune/labeling/label_tool.py <clip>                 # videos/<clip>.mp4 + labels/<clip>_ball.csv
    python finetune/labeling/label_tool.py video22..video53      # a whole run, one clip after another
    python finetune/labeling/label_tool.py video22..video53 --uncertain   # only the frames worth a look

Every label frame comes up with the mark drawn and the mouse on it: SPACE if the mark is on the
ball, a click where the ball is if not, v if there is no ball. Every key, the grid and sheet
views, and what the marks, hints and badges mean: finetune/labeling/LABEL_TOOL.md (the bar at the bottom
of the window names the keys that matter on the frame you are on). Output is
labels/<clip>_ball.csv, with progress in labels/<clip>_ball.review.json, autosaved.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Dict, Hashable, List, Optional, Tuple

import cv2
import numpy as np

if __name__ == "__main__":          # run as a script: make the repository root importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from finetune.labeling import label_window as win
from finetune.labeling.label_draw import (FILTERS, ZOOM_STEP, draw_ticks, enhance, fit_text, hit, label_mark_colour,
                                          mark_plan)
from finetune.labeling.label_hints import Suggestions
from finetune.labeling.label_source import Source, build_source, expand   # noqa: F401 - the tool's parts, by name
from finetune.labeling.label_store import SETTINGS_NAME, Row, load_settings, save_flags, save_review, save_rows, save_settings
from finetune.labeling.label_views import GRID_SIZES, GridView, SheetView
from finetune.labeling.label_window import (KEY_TAB, KEYS_CONFIRM, KEYS_NEXT, KEYS_NUDGE, KEYS_PREV, KEYS_QUIT, KEYS_STEP,
                                            KEYS_ZOOM_IN, KEYS_ZOOM_OUT, WINDOW)
from finetune.data import workspace as ws
from finetune.data.labels import LabelFormatError, is_visible
from finetune.labeling.refine import Suggestion
from finetune.data.scoring import REFERENCE_HEIGHT

HUD = 58
MAX_UPSCALE = 2.0         # cap on drawing the frame bigger than it is: past this it only costs time
UI_REFERENCE = 1280.0     # view width the HUD text sizes were chosen for; bigger views scale up
ZOOM_EASE = 0.34          # share of the remaining gap closed each redraw: a glide, not a jump
ZOOM_MAX = 12.0
PREFETCH = 6              # label frames decoded ahead while the tool waits for the next keypress
INSET_SHARE = 0.22        # width of the overview inset as a share of the view
GHOST_REACH = 3           # rows either side searched for a ball to interpolate a missing guess from
SNAP_PX = 4               # a wheel turned with the pointer this close to the ball zooms about the ball
AUTOSAVE_SECONDS = 5.0    # write labels + review progress this often while there are unsaved changes


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
        self.mouse = win.CursorPlacer(self)    # parks the OS pointer on the ball (Windows)
        self.viewport = win.ViewportGuard()    # keeps the Qt window's own zoom at x1
        self.hints = Suggestions(sources)      # refine.suggest's measurements, kept current in the background
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
        self.last_save = time.monotonic()
        self.grid_view = GridView(self)   # TAB's page of crops; see label_views
        self.sheet_view = SheetView(self) # w's trajectory sheet

        self.last_clip = None
        self.presented = None
        self.filter = 0               # index into FILTERS; h cycles it, the HUD chip shows it
        self.inset = True             # the small whole-frame overview drawn while zoomed in
        self.inset_box = None         # where it is on screen, so a click in it can re-centre there
        self.inset_scale = 1.0        # frame px -> inset px
        self.filter_button = None     # HUD chips for those two, like centre_button
        self.inset_button = None
        self.thumbs: Dict[tuple, np.ndarray] = {}   # (clip, frame, width) -> small copy, for the inset
        self.ghost = None             # a frame with no guess: where the neighbours say the ball should be
        self.settings_path: Optional[Path] = None
        self.normal_rect = None       # the window's rect before going full screen, to rebuild into
        self.pending_key = -1         # a key pumped out of the queue by a short waitKey, kept for loop
        self.disputed_key, self.disputed = None, 0   # disputed_count's cache
        self.shown_offers: Dict[int, tuple] = {}   # row index -> the suggestion drawn as the mark on
                                                   # offer in the last render; what SPACE takes
        self.render_key = None        # what the last single-frame render was drawn from, see render_state
        self.render_cache = None
        self.render_fresh = True      # did the last render() draw anything new?
        self.shown = None             # the image last handed to imshow, so an unchanged one is not re-sent
        self.direction = 1            # which way the last move went, so prefetch runs ahead of a or d
        self.quit_warned = False      # q with unsaved labels warns once, then quits

    # ---------------------------------------------------------------- the clip we are in
    @property
    def source(self) -> "Source":
        return self.by_clip[self.rows[self.cursor].clip]

    @property
    def clip(self) -> str:
        return self.source.clip

    @property
    def overview(self) -> bool:
        """The grid or the trajectory sheet: views with no single frame under a parked mouse."""
        return self.grid or self.sheet

    # The views' state, by the names the tool's own code and the tests read; the views own it.
    grid = property(lambda self: self.grid_view.active, lambda self, on: setattr(self.grid_view, "active", on))
    sheet = property(lambda self: self.sheet_view.active, lambda self, on: setattr(self.sheet_view, "active", on))
    grid_size = property(lambda self: self.grid_view.size, lambda self, n: setattr(self.grid_view, "size", n))
    grid_zoom = property(lambda self: self.grid_view.zoom, lambda self, z: setattr(self.grid_view, "zoom", z))
    grid_indices = property(lambda self: self.grid_view.indices)
    grid_tiles = property(lambda self: self.grid_view.tiles)
    sheet_start = property(lambda self: self.sheet_view.start, lambda self, i: setattr(self.sheet_view, "start", i))
    sheet_points = property(lambda self: self.sheet_view.points)

    # ---------------------------------------------------------------- geometry
    @property
    def scale(self) -> float:
        return self.base_scale * self.zoom

    def create_window(self) -> None:
        """WINDOW_NORMAL, not WINDOW_AUTOSIZE: autosize lets highgui pick the size from the image and
        the monitor DPI, which on a 200% display made a 1080p clip a 3226x2095 window on a 2880x1800
        desktop, and left no way to maximise it. WINDOW_GUI_NORMAL: the plain window, without the Qt
        toolbar whose magnifier buttons zoom the window itself (see viewport_wheel) and without the
        status bar, so the whole client area is picture."""
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        cv2.setMouseCallback(WINDOW, self.on_mouse)

    def open_window(self) -> None:
        """Create a resizable window filling the desktop."""
        self.create_window()
        cv2.imshow(WINDOW, np.zeros((self.height, self.width, 3), dtype=np.uint8))
        self.shown = None             # the window shows a blank, not the last render
        cv2.waitKey(1)
        area = win.work_area()
        if area is None:
            return
        work_w, work_h, work_x, work_y = area
        win.place_window(work_x, work_y, work_w, work_h)
        cv2.waitKey(1)

    def reset_window(self) -> None:
        """Throw the window away and make a fresh one in the same place: the only way to put the Qt
        viewport's own zoom and pan back to x1 from outside, since highgui offers no call for it.
        Reached when the viewport check finds the picture is not where it should be."""
        if not self.viewport.allow_reset():
            print("[view] the window zoom check keeps failing; it is off for the rest of this session")
            return
        if self.fullscreen and self.normal_rect is not None:
            rect = self.normal_rect   # not the monitor's rect
        else:
            rect = win.window_rect()
        cv2.destroyWindow(WINDOW)
        cv2.waitKey(1)
        self.create_window()
        self.shown = None             # a new, empty window: whatever was shown went with the old one
        self.show()
        cv2.waitKey(1)
        if rect is not None:
            left, top, right, bottom = rect
            win.place_window(left, top, right - left, bottom - top)
        if self.fullscreen:
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.waitKey(1)
        self.client = None            # measure the new window from scratch: size, map, everything
        self.mouse.forget_window()
        self.viewport.forget()
        self.pending_cursor_move = True
        print("[view] the window's own zoom had drifted off x1; window rebuilt")

    def fit_view(self, client_w: int, client_h: int) -> None:
        """Draw at the size the window actually is, so enlarging it gains detail instead of blur."""
        self.anchor = None
        self.mouse.aim = None
        old = (self.view_w, self.view_h, self.hud_h)
        self.ui = max(1.0, min(client_w / UI_REFERENCE, 2.5))
        self.hud_h = int(round(HUD * self.ui))
        room_h = max(120, client_h - self.hud_h)
        self.base_scale = max(0.2, min(client_w / self.width, room_h / self.height, MAX_UPSCALE))
        # The view is the whole client area, not just the frame at that scale: highgui keeps the
        # image's aspect ratio and fills whatever is left over in white, which framed a capped or
        # differently-shaped frame in bright bars. Beyond the frame the view is black.
        self.view_w = max(160, int(client_w))
        self.view_h = max(90, int(room_h))
        self.centre_view()
        if (self.view_w, self.view_h, self.hud_h) != old:
            self.mouse.map_stale = True   # a new image size means highgui fits it differently: re-measure
            self.viewport.void_probe()

    def track_window(self) -> None:
        """Notice the window being resized, maximised or full-screened, and follow it: both the render
        size and the view-pixel-to-screen-pixel map depend on how big the client area is."""
        geometry = win.window_client()
        if geometry is None:
            return
        size = (geometry[2], geometry[3])
        if size == self.client:
            return
        self.client = size
        self.fit_view(*size)
        self.mouse.map_stale = True

    def toggle_fullscreen(self) -> None:
        if not self.fullscreen:
            rect = win.window_rect()
            if rect is not None:
                self.normal_rect = rect
        self.viewport.void_probe()
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

    # Frame coordinates put pixel k's centre at k + 0.5 (what the click mapping, GridTrackNet's grid
    # decode and training's encode all assume). View pixel u covers [u, u + 1), so a frame point maps
    # to the view pixel whose span holds it (floor, not round: round's ties go to even, which put a
    # ball dead centre a pixel off); render's warp is built on the same convention.
    def to_view(self, x: float, y: float) -> Tuple[int, int]:
        return (math.floor((x - self.origin[0]) * self.scale),
                math.floor((y - self.origin[1]) * self.scale))

    def to_frame(self, vx: float, vy: float) -> Tuple[float, float]:
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
        if self.last_clip != self.clip:
            if self.last_clip is not None:
                self.by_clip[self.last_clip].release()   # let go of the clip being left, decoder included
            self.last_clip = self.clip
            self.width, self.height = self.source.width, self.source.height
            self.fit_view(*(self.client or (self.view_w, self.view_h + self.hud_h)))
        self.anchor = None
        self.mouse.aim = None  # a pending OS cursor report belongs to the previous view
        self.zoom = self.zoom_target  # finish the old frame's animation before centering the next
        if self.overview:
            self.pending_cursor_move = False
            return
        row = self.rows[self.cursor]
        self.ghost = None if row.visible else self.guess_for(self.cursor)
        if not self.follow:
            return                    # auto-centring is off: leave the view and the mouse be
        point = self.target_point()   # the mark on offer, the ball, or the neighbours' guess
        if point is not None:
            self.center = point
        self.centre_view()
        self.pending_cursor_move = True

    def guess_for(self, index: int) -> Optional[Tuple[float, float]]:
        """Where a frame with no detection probably has its ball: a straight line between the nearest
        visible labels either side, weighted by how far away each is. One side only is still worth
        aiming at when it is the row next door. None when there is nothing nearby to go on.

        Only a hint - it is drawn dashed, the mouse is parked on it, and nothing is written until you
        click. It saves the hunt on the frames the detector missed, which are most of the ones the
        --uncertain walk stops on.
        """
        clip = self.rows[index].clip
        before = after = None
        for step in range(1, GHOST_REACH + 1):
            if before is None and index - step >= 0:
                row = self.rows[index - step]
                if row.clip == clip and row.visible:
                    before = row
            if after is None and index + step < len(self.rows):
                row = self.rows[index + step]
                if row.clip == clip and row.visible:
                    after = row
        frame = self.rows[index].frame
        if before is not None and after is not None and after.frame > before.frame:
            share = (frame - before.frame) / (after.frame - before.frame)
            return (before.x + (after.x - before.x) * share, before.y + (after.y - before.y) * share)
        near = before if after is None else after if before is None else None
        if near is not None and abs(self.rows.index(near) - index) == 1:
            return (near.x, near.y)
        return None

    def target_point(self) -> Optional[Tuple[float, float]]:
        """The frame point this frame is about - where the view centres and the mouse parks: the
        suggestion on offer, else the ball, else the ghost guess when there is no ball.

        Only the suggestions known at this moment count. One that arrives for the frame on screen a
        little later (the clip's first pass, or a re-measure after an edit) is drawn as the mark on
        offer where it is, but the view and the mouse stay put: snapping them to a point that moved
        by a fraction of a pixel would jog the picture and pull the mouse out from under a hand on
        its way to click. SPACE takes the drawn mark exactly, wherever the mouse sits."""
        row = self.rows[self.cursor]
        offer = self.offer_for(self.cursor)
        if offer is not None:
            return (offer["x"], offer["y"])
        if row.visible:
            return (row.x, row.y)
        return self.ghost

    # ---------------------------------------------------------------- frames
    def frame(self, index: int) -> np.ndarray:
        """The picture of source frame `index` of the clip we are in (see Source.frame)."""
        return self.source.frame(index)

    def clip_bounds(self, index: int) -> Tuple[int, int]:
        """(first, last + 1) row indices of the clip row `index` belongs to."""
        source = self.by_clip[self.rows[index].clip]
        return source.start, source.start + len(source.rows)

    # ---------------------------------------------------------------- the grid and the sheet
    # The two overview views own their state, picture and keys (label_views); these are the
    # actions by name, the surface the key table and the tests use.
    def toggle_grid(self) -> None:
        self.grid_view.toggle()

    def prepare_grid(self, start: int) -> None:
        self.grid_view.prepare(start)

    def grid_page(self, direction: int) -> None:
        self.grid_view.page(direction)

    def grid_select(self, slot: int) -> None:
        self.grid_view.select(slot)

    def grid_step(self, direction: int) -> None:
        self.grid_view.step(direction)

    def toggle_grid_size(self) -> None:
        self.grid_view.toggle_size()

    def confirm_grid(self) -> None:
        self.grid_view.confirm_page()

    def render_grid(self) -> np.ndarray:
        return self.grid_view.render()

    def grid_mouse(self, event: int, x: int, y: int) -> None:
        self.grid_view.on_mouse(event, x, y)

    def toggle_sheet(self) -> None:
        self.sheet_view.toggle()

    def sheet_page(self, direction: int) -> None:
        self.sheet_view.page(direction)

    def render_sheet(self) -> np.ndarray:
        return self.sheet_view.render()

    def sheet_mouse(self, event: int, x: int, y: int) -> None:
        self.sheet_view.on_mouse(event, x, y)


    def view_ready(self) -> bool:
        return self.presented == (self.cursor, self.origin, self.scale)

    def show(self) -> None:
        image = self.render()
        presented = (self.cursor, self.origin, self.scale)
        if self.render_fresh or image is not self.shown:   # unchanged and already on screen: skip
            cv2.imshow(WINDOW, image)
            self.shown = image
        self.presented = presented


    def frame_missing(self) -> bool:
        """True when the frame under the cursor has no picture, so nothing should be said about it."""
        return self.rows[self.cursor].frame in self.source.missing

    def prefetch(self) -> bool:
        """Decode not-yet-cached frames ahead of the cursor. Called while the tool is idle, so the
        next SPACE lands on a frame that is already in memory. True when there was work to do.

        Keeps going for about a screen refresh rather than one frame per idle tick: a run of SPACE
        presses at two or three a second was outrunning one decode per 30 ms wait."""
        s = self.step                                    # n/d/p/a land every step-th row
        if self.direction >= 0:
            upcoming = (self.walk[self.walk_pos + s::s][:PREFETCH] if self.walk
                        else range(self.cursor + s, min(self.cursor + s * PREFETCH + 1, len(self.rows)), s))
        else:                                            # heading back: run ahead of `a` instead
            upcoming = ((self.walk[self.walk_pos - s::-s][:PREFETCH] if self.walk_pos - s >= 0 else [])
                        if self.walk else range(self.cursor - s, max(-1, self.cursor - s * PREFETCH - 1), -s))
        started, worked = time.monotonic(), False
        for index in upcoming:
            if self.rows[index].clip != self.rows[self.cursor].clip:
                break                                    # do not disturb the next clip's capture yet
            frame = self.rows[index].frame
            if FILTERS[self.filter] == "motion" and index > 0 and self.rows[index - 1].clip == self.rows[index].clip:
                before = self.rows[index - 1].frame      # the motion view diffs against the row before;
                if (before not in self.source.frame_cache and before not in self.source.missing
                        and before >= self.source.position >= 0):
                    self.frame(before)                   # forward of the capture, so no seek
            if frame in self.source.frame_cache or frame in self.source.missing:
                continue
            self.frame(frame)
            worked = True
            if time.monotonic() - started > 0.02:
                break
        return worked

    def previous_image(self) -> Optional[np.ndarray]:
        """The picture of the label row before this one in the same clip, if it is already in the
        cache - it usually is, having just been looked at. Never decoded on purpose: a frame behind
        the capture costs a re-read from the start of the clip, and the motion view is a hint."""
        if self.cursor == 0:
            return None
        before = self.rows[self.cursor - 1]
        if before.clip != self.rows[self.cursor].clip:
            return None
        return self.source.frame_cache.get(before.frame)

    def thumbnail(self, frame: int, image: np.ndarray, width: int, height: int) -> np.ndarray:
        key = (self.clip, frame, width)
        thumb = self.thumbs.get(key)
        if thumb is None:
            thumb = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            if len(self.thumbs) >= 64:
                self.thumbs.pop(next(iter(self.thumbs)))
            self.thumbs[key] = thumb
        return thumb

    def draw_inset(self, display: np.ndarray, image: np.ndarray, row: Row, ui: float) -> None:
        """A small copy of the whole frame in the bottom-right corner while zoomed in, with the part on
        screen boxed and the ball dotted. When the detector's guess is nowhere near, this is where the
        real ball is found without zooming out: a click on it brings that spot into the zoomed view."""
        self.inset_box = None
        if not self.inset or self.zoom < 1.5:
            return
        width = int(self.view_w * INSET_SHARE)
        height = max(1, int(round(width * self.height / self.width)))
        margin = int(10 * ui)
        left, top = self.view_w - width - margin, self.view_h - height - margin
        if left < 0 or top < 0:
            return
        panel = self.thumbnail(row.frame, image, width, height).copy()
        scale = width / self.width
        self.inset_scale = scale
        x0, y0 = self.origin
        x1, y1 = x0 + self.view_w / self.scale, y0 + self.view_h / self.scale
        cv2.rectangle(panel, (int(x0 * scale), int(y0 * scale)), (int(x1 * scale) - 1, int(y1 * scale) - 1),
                      (255, 255, 255), 1)
        point = self.target_point()
        if point is not None:
            cv2.circle(panel, (int(point[0] * scale), int(point[1] * scale)), max(2, int(2 * ui)),
                       (0, 255, 255) if row.visible else (255, 200, 0), -1, cv2.LINE_AA)
        display[top:top + height, left:left + width] = panel
        cv2.rectangle(display, (left - 1, top - 1), (left + width, top + height), (110, 110, 110), 1)
        self.inset_box = (left, top, left + width - 1, top + height - 1)

    def render_state(self, image: np.ndarray, before: Optional[np.ndarray]) -> Tuple[Hashable, ...]:
        """Everything the single-frame view is drawn from. render redraws only when this changes,
        so the loop can call it every 30 ms for free. IF YOU MAKE render, draw_inset OR THE HUD READ
        SOMETHING NEW, ADD IT HERE - a screen that does not change when it should is far worse than
        the CPU this saves. Any write to any Row bumps Row.revision, which covers the labels, review
        flags, trail and every counter in the HUD; the frame images are keyed by identity, which
        changes when one is decoded, evicted or replaced by a placeholder."""
        return (Row.revision, self.cursor, self.origin, self.scale, self.zoom, self.view_w, self.view_h,
                self.hud_h, self.ui, self.width, self.height, self.filter, self.inset, self.trail,
                self.follow, self.step, self.dirty, id(self.walk), len(self.walk), self.walk_pos,
                self.ghost, len(self.sources), id(image), id(before), self.hints.version)

    def render(self) -> np.ndarray:
        if self.sheet:
            return self.render_sheet()
        if self.grid:
            return self.render_grid()
        self.centre_view()
        row = self.rows[self.cursor]
        image = self.frame(row.frame)
        mode = FILTERS[self.filter]
        before = self.previous_image() if mode == "motion" else None
        key = self.render_state(image, before)
        if key == self.render_key:
            self.render_fresh = False
            return self.render_cache
        self.render_fresh = True
        x0, y0 = self.origin
        # Resize with exact sub-pixel origin handling via warpAffine. warpAffine puts pixel centres on
        # whole numbers while frame coordinates put them at k + 0.5 (see to_view): the half-pixel
        # terms line the drawn picture up with where a click is stored.
        s = self.scale
        matrix = np.array([[s, 0.0, s * (0.5 - x0) - 0.5], [0.0, s, s * (0.5 - y0) - 0.5]], dtype=np.float64)
        interpolation = cv2.INTER_LINEAR if self.zoom < 8 else cv2.INTER_NEAREST
        # One canvas for picture and HUD, the picture warped straight into its top: building them
        # apart and stacking them copied the whole view once more per redraw. A fresh canvas each
        # time, so an image handed out earlier (the cache, a caller's reference) is never drawn over.
        canvas = np.empty((self.view_h + self.hud_h, self.view_w, 3), np.uint8)
        display = canvas[:self.view_h]
        cv2.warpAffine(image, matrix, (self.view_w, self.view_h), dst=display, flags=interpolation)
        if mode != "off":
            previous = None
            if before is not None:
                previous = cv2.warpAffine(before, matrix, (self.view_w, self.view_h), flags=interpolation)
            display[:] = enhance(display, mode, previous)
        ui = self.ui
        thin = max(1, int(round(ui)))
        if self.trail:
            for previous in self.rows[max(0, self.cursor - 12):self.cursor]:
                if previous.visible and previous.clip == row.clip:
                    cv2.circle(display, self.to_view(previous.x, previous.y), max(3, int(3 * ui)),
                               (200, 200, 60), thin, cv2.LINE_AA)
        radius = int(max(8, 6 * self.zoom) * ui)
        hint = self.suggestion_for(self.cursor)
        offer = self.offer_for(self.cursor)
        self.shown_offers = {} if offer is None else {self.cursor: (offer["x"], offer["y"])}
        # The suggestion on offer is the mark SPACE takes, drawn as the label would be, in orange -
        # bright when it is sure, dull when only a guess; the draft it replaces is ticked in grey.
        plan = mark_plan(row, hint, offer, label_mark_colour(row))
        for point, colour in plan.ticks:
            draw_ticks(display, *self.to_view(*point), colour, ui)
        if plan.mark is not None:
            cx, cy = self.to_view(*plan.mark)
            cv2.circle(display, (cx, cy), radius, plan.colour, thin, cv2.LINE_AA)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                cv2.line(display, (cx + dx * (radius - int(3 * ui)), cy + dy * (radius - int(3 * ui))),
                         (cx + dx * (radius + int(8 * ui)), cy + dy * (radius + int(8 * ui))), plan.colour, thin)
        elif self.ghost is not None:
            # Dashed, so it cannot be mistaken for a label: the neighbours' idea of where the ball is
            gx, gy = self.to_view(*self.ghost)
            for start in range(0, 360, 45):
                cv2.ellipse(display, (gx, gy), (radius, radius), 0, start, start + 25, (255, 200, 0), thin, cv2.LINE_AA)
            cv2.putText(display, "guess - click to place", (gx + radius + int(6 * ui), gy + int(5 * ui)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45 * ui, (255, 200, 0), thin, cv2.LINE_AA)
        self.draw_inset(display, image, row, ui)
        if row.settled or row.suspect:
            badge = "CHECK THIS" if row.suspect else "EDITED" if row.edited else "REVIEWED"
            fill = (200, 60, 200) if row.suspect else (60, 140, 255) if row.edited else (70, 190, 70)
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
        self.draw_hud(canvas, row, hint, offer)
        self.render_key, self.render_cache = key, canvas
        return canvas

    def draw_hud(self, canvas: np.ndarray, row: Row, hint: Optional[Suggestion], offer: Optional[Suggestion]) -> None:
        """The two text lines and the clickable chips along the bottom of the single-frame view."""
        ui = self.ui
        thin = max(1, int(round(ui)))
        reviewed = sum(r.reviewed for r in self.rows)
        hud = canvas[self.view_h:]
        hud[:] = 0
        uncertain_left = sum(1 for r in self.rows if r.uncertain and not r.reviewed)
        flagged_left = sum(1 for r in self.rows if r.suspect)
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
        if offer is not None:
            sure = "sure" if offer["trusted"] else "UNSURE, look closely"
            if row.visible:
                gap = math.hypot(offer["x"] - row.x, offer["y"] - row.y) * REFERENCE_HEIGHT / self.height
                state = f"HINT ({offer['x']:.1f}, {offer['y']:.1f}) {sure}  grey draft {gap:.1f}px off"
            else:
                state = f"HINT ({offer['x']:.1f}, {offer['y']:.1f}) {sure}  draft: no ball"
        elif hint is not None and not row.visible:
            state += "  hint: the video shows a ball here (y takes it)"
        elif hint is not None:
            gap = math.hypot(hint["x"] - row.x, hint["y"] - row.y) * REFERENCE_HEIGHT / self.height
            state += f"  hint {gap:.1f}px" if gap >= 0.05 else "  hint: here"
            verdict = self.hint_verdict(self.cursor)
            if verdict is not None:
                state += "  hint ok" if verdict == "agrees" else "  hint differs"
            else:
                state += " (unsure)"
        line1 = (f"{place}  frame {row.frame}  {state}  "
                 f"from {row.source} {row.conf:.2f}  {tag}  zoom x{self.zoom:.1f}{'  *unsaved*' if self.dirty else ''}")
        disputed = self.disputed_count()
        keys = ("SPACE=take hint  x=keep draft  v=invisible" if offer is not None else
                "SPACE=good  y=take hint  v=invisible")
        line2 = (f"reviewed {reviewed}/{len(self.rows)}  uncertain left {uncertain_left}  "
                 f"{('flagged left ' + str(flagged_left) + '   ') if flagged_left else '  '}"
                 f"{('hint differs ' + str(disputed) + ' (e)   ') if disputed else ''}"
                 f"{keys}  r=accept run  f=uncertain  q=quit")
        # Buttons along the right end of the bar, laid out right to left and measured before the
        # two text lines, so a long clip name or a narrow window cuts the text short instead of
        # running under them. Every box is kept for on_mouse: they are all clickable.
        pad = int(6 * ui)
        (_, glyph_h), _ = cv2.getTextSize("Xy8", cv2.FONT_HERSHEY_SIMPLEX, 0.42 * ui, thin)
        top = max(0, (self.hud_h - glyph_h - pad) // 2)
        bottom = min(self.hud_h - 1, top + glyph_h + pad)

        def chip(right_edge: int, text: str, lit: bool) -> Tuple[int, Tuple[int, int, int, int]]:
            (text_w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42 * ui, thin)
            left_edge = max(0, right_edge - text_w - 2 * pad)
            cv2.rectangle(hud, (left_edge, top), (right_edge, bottom),
                          (70, 190, 70) if lit else (55, 55, 55), -1)
            cv2.putText(hud, text, (left_edge + pad, bottom - pad // 2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42 * ui, (0, 0, 0) if lit else (190, 190, 190), thin, cv2.LINE_AA)
            return left_edge, (left_edge, self.view_h + top, right_edge, self.view_h + bottom)

        # Lit green means on; the words stay short so the text lines keep their room
        edge, self.centre_button = chip(self.view_w - pad, "g  CENTRE", self.follow)
        edge, self.inset_button = chip(edge - pad, "o  INSET", self.inset)
        edge, self.filter_button = chip(edge - pad, f"h  {FILTERS[self.filter].upper() if self.filter else 'FILTER'}",
                                        self.filter != 0)
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

    def pump(self, ms: int) -> None:
        """Let highgui deliver events for `ms` without losing a keypress: Qt's waitKey hands back,
        and so takes off the queue, any key it meets, so a short wait that ignored it used to drop
        a SPACE pressed while the pointer was being placed. The key is kept for the main loop."""
        key = cv2.waitKeyEx(ms)
        if key != -1 and self.pending_key == -1:
            self.pending_key = key

    def place_cursor(self) -> None:
        """Put the OS mouse cursor on the current guess so a click is a one-pixel move.

        follow_ball has already put the ball in the middle of the view, so this normally lands the
        pointer dead centre. A frame with no ball has nothing to aim at, and the pointer goes to
        the centre anyway rather than being left wherever the last frame happened to leave it.
        """
        self.pending_cursor_move = False
        if win._user32 is None or not self.follow:
            return
        if self.mouse.cursor_map is None or self.mouse.map_stale:
            self.mouse.calibrate()                   # never measured, or the window changed size
        point = self.target_point()                  # the ball, or the ghost guess standing in for it
        if point is None:
            self.mouse.warp(self.view_w / 2.0, self.view_h / 2.0)
            return
        view_x, view_y = self.to_view(*point)
        if not (0 <= view_x < self.view_w and 0 <= view_y < self.view_h):
            self.center = point                   # nothing should leave the ball off screen, but if
            self.centre_view()                    # anything does, do not leave the mouse behind
            view_x, view_y = self.to_view(*point)
        self.mouse.warp(view_x, view_y)

    # ---------------------------------------------------------------- edits
    def on_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_MOUSEWHEEL and self.viewport.absorb_wheel(x, y, flags, self.view_w):
            return                                  # a wheel the tool sent itself, not the hand's
        self.mouse.report(x, y)                     # where the warp landed; check_placement reads it
        if event == cv2.EVENT_MOUSEWHEEL and (self.mouse.calibrating or self.overview or not self.view_ready()):
            # A wheel the tool does not act on still zoomed Qt's own viewport: cancel that, or the
            # next viewport check finds it and rebuilds the window.
            self.viewport.repay_wheel(flags >> 16)
            return
        if self.mouse.calibrating and event != cv2.EVENT_MOUSEMOVE:
            return                                  # the tool is moving the pointer itself just now
        if not self.overview and event != cv2.EVENT_MOUSEMOVE and not self.view_ready():
            return  # a queued click must never edit the next frame before it is displayed
        if self.sheet:
            self.sheet_mouse(event, x, y)
            return
        if event == cv2.EVENT_LBUTTONDBLCLK:
            # The second press of a quick double-click comes through as this instead of a press. With
            # the pointer parked on the same screen pixel frame after frame, click-click on two
            # consecutive frames is exactly that, and every second click used to be thrown away.
            event = cv2.EVENT_LBUTTONDOWN
        if self.grid:
            self.grid_mouse(event, x, y)
            return
        if y >= self.view_h:                        # the HUD: its buttons, and nothing else
            if event == cv2.EVENT_LBUTTONDOWN:
                if hit(self.centre_button, x, y):
                    self.toggle_follow()
                elif hit(self.filter_button, x, y):
                    self.cycle_filter()
                elif hit(self.inset_button, x, y):
                    self.toggle_inset()
                for box, jump in self.step_buttons:
                    if hit(box, x, y):
                        self.step = jump
            return
        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN) and hit(self.inset_box, x, y):
            # Either button in the overview: bring that part of the frame into the zoomed view. The
            # left button does not place a label from a thumbnail - nothing there is precise enough.
            left, top = self.inset_box[0], self.inset_box[1]
            self.center = (min(max((x + 0.5 - left) / self.inset_scale, 0.0), self.width - 1.0),
                           min(max((y + 0.5 - top) / self.inset_scale, 0.0), self.height - 1.0))
            self.centre_view()
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.frame_missing():
                return                              # nothing to see, so nothing to label
            fx, fy = self.to_frame(x + 0.5, y + 0.5)  # highgui floors the pointer to a pixel: aim at
            if not (-2 <= fx < self.width + 2 and -2 <= fy < self.height + 2):   # its middle
                return                                # a click in the black beyond the frame
            if self.set_point(fx, fy):
                self.move(1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            frame_x, frame_y = self.to_frame(x + 0.5, y + 0.5)
            self.center = (min(max(frame_x, 0.0), self.width - 1.0),   # a click in the black beyond
                           min(max(frame_y, 0.0), self.height - 1.0))  # the frame: keep it on frame
            self.centre_view()
        elif event == cv2.EVENT_MOUSEWHEEL:
            self.mouse.wheel_at = time.monotonic()
            self.zoom_at(self.wheel_notches(flags), x + 0.5, y + 0.5)
            self.viewport.repay_wheel(flags >> 16)

    def cycle_filter(self) -> None:
        self.filter = (self.filter + 1) % len(FILTERS)
        self.grid_view.invalidate()
        self.remember_settings()

    def toggle_inset(self) -> None:
        self.inset = not self.inset
        self.remember_settings()

    def remember_settings(self) -> None:
        if self.settings_path is not None:
            save_settings(self.settings_path, {"filter": FILTERS[self.filter], "inset": self.inset,
                                               "grid_size": self.grid_size,
                                               "trail": self.trail})

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

    def zoom_at(self, notches: float, vx: float, vy: float) -> None:
        """Aim the zoom at a new level about the view point (vx, vy). The glide there happens in
        step_zoom; whatever sits under (vx, vy) stays under it the whole way.

        With the pointer on or next to the ball - where the tool itself parks it - the zoom pivots on
        the ball exactly, not on the pixel highgui rounded the pointer to. That rounding is under a
        pixel, but a pivot a pixel off the ball moves the ball by that pixel times the zoom ratio:
        going from x1 to x12 walked it a dozen pixels away from the centre, and from the pointer.
        """
        target = min(max(self.zoom_target * ZOOM_STEP ** notches, 1.0), ZOOM_MAX)
        if abs(target - self.zoom_target) < 1e-6:
            return
        self.zoom_target = target
        point = self.target_point()
        if point is not None:
            ball_x = (point[0] - self.origin[0]) * self.scale
            ball_y = (point[1] - self.origin[1]) * self.scale
            if abs(ball_x - vx) <= SNAP_PX and abs(ball_y - vy) <= SNAP_PX:
                self.anchor = (point, (ball_x, ball_y))
                return
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
        point = self.target_point()
        if point is not None:         # exact, not rounded to a view pixel: see zoom_at
            self.zoom_at(notches, (point[0] - self.origin[0]) * self.scale,
                         (point[1] - self.origin[1]) * self.scale)
        else:
            self.zoom_at(notches, self.view_w / 2.0, self.view_h / 2.0)
        self.pending_cursor_move = True

    # ---------------------------------------------------------------- suggestions
    def suggestion_for(self, index: int) -> Optional[Suggestion]:
        """The suggestion worth showing for a row, or None (see Suggestions.for_row)."""
        row = self.rows[index]
        return self.hints.for_row(row, self.by_clip[row.clip].height)

    def offer_for(self, index: int) -> Optional[Suggestion]:
        """The suggestion drawn as the mark SPACE takes on an unreviewed row (see Suggestions.offer_for)."""
        row = self.rows[index]
        return self.hints.offer_for(row, self.by_clip[row.clip].height)

    def hint_verdict(self, index: int) -> Optional[str]:
        """"agrees" / "differs" / None for a row's label against a trusted suggestion."""
        row = self.rows[index]
        return self.hints.verdict(row, self.by_clip[row.clip].height)

    def disputed_count(self) -> int:
        """How many frames the suggestions disagree with, cached on the state it depends on."""
        key = (Row.revision, self.hints.version)
        if self.disputed_key != key:
            self.disputed_key = key
            self.disputed = sum(1 for i in range(len(self.rows)) if self.hint_verdict(i) == "differs")
        return self.disputed

    def jump_disputed(self, direction: int) -> bool:
        """Go to the next (or previous) frame the suggestions disagree with. False if there is none."""
        index = self.cursor + direction
        while 0 <= index < len(self.rows):
            if self.hint_verdict(index) == "differs":
                self.direction = direction
                self.cursor = index
                if self.walk and index in self.walk:
                    self.walk_pos = self.walk.index(index)
                self.follow_ball()
                return True
            index += direction
        return False

    def take_suggestion(self) -> bool:
        """Move this frame's label to its suggestion. True when there was one to take."""
        hint = self.suggestion_for(self.cursor)
        if hint is None or not self.set_point(hint["x"], hint["y"]):
            return False
        self.rows[self.cursor].source = "refined"   # kept apart from clicks in the review file
        return True

    def set_point(self, x: float, y: float) -> bool:
        """Put this frame's ball at (x, y). False when nothing was stored."""
        if self.frame_missing():
            return False
        x = float(min(max(x, 0.0), self.width - 1.0))
        y = float(min(max(y, 0.0), self.height - 1.0))
        if not is_visible(x, y, self.width, self.height):
            # The label CSV marks "no ball" by a point in the top-right corner, and every reader in
            # the workspace takes any point there to mean that. Stored, this would read back as absent.
            print(f"[warn] frame {self.rows[self.cursor].frame}: ({x:.0f}, {y:.0f}) is in the top-right "
                  f"corner the label files reserve for 'no ball', so it cannot be saved as a ball there")
            return False
        row = self.rows[self.cursor]
        row.snapshot()
        row.suspect = False           # an edit answers an audit flag
        row.x, row.y = x, y
        if row.source == "refined":
            row.source = "click"      # placed by hand now, not taken from a suggestion
        row.visible = True
        row.reviewed = row.edited = True
        self.dirty = self.source.dirty = True
        self.source.cursor = self.cursor - self.source.start
        return True

    def nudge(self, dx: float, dy: float) -> None:
        row = self.rows[self.cursor]
        if row.visible:
            self.set_point(row.x + dx, row.y + dy)
            self.pending_cursor_move = True

    def mark_invisible(self) -> None:
        if self.frame_missing():
            return
        row = self.rows[self.cursor]
        row.snapshot()
        row.suspect = False
        row.visible = False
        row.reviewed = row.edited = True
        self.dirty = self.source.dirty = True

    def confirm(self, take_offer: bool = True) -> None:
        """SPACE: the mark on screen is right. When that mark was a suggestion on offer, the label
        moves there (recorded as source "refined", like y); otherwise the label is confirmed where
        it is. Only an offer that was actually drawn counts - one that arrived between the draw and
        the key press is not taken on a frame you judged without it. `take_offer=False` confirms
        the label as it is (x, and r's unseen frames)."""
        if self.frame_missing():
            return                    # a black placeholder is not something to say "good" to
        offer = self.shown_offers.get(self.cursor) if take_offer else None
        if offer is not None:
            if self.set_point(*offer):
                self.rows[self.cursor].source = "refined"
            return                    # could not be stored (set_point said why): nothing confirmed
        row = self.rows[self.cursor]
        if not row.reviewed or row.suspect:
            row.snapshot()
        row.suspect = False           # "it is right where it is" answers an audit flag
        row.reviewed = True
        self.dirty = self.source.dirty = True

    def only_uncertain(self) -> int:
        """Restrict stepping to the frames that want attention. Returns how many there are.

        A filter on where you go, never on what is saved: every row stays in its clip's list, so
        the frames you skip past are written back exactly as they were.
        """
        self.walk = [i for i, row in enumerate(self.rows) if row.suspect or (row.uncertain and not row.settled)]
        self.walk_pos = 0
        if self.walk:
            self.cursor = self.walk[0]
        return len(self.walk)

    def move(self, delta: int) -> None:
        if delta:
            self.direction = 1 if delta > 0 else -1
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
        while not self.rows[self.cursor].uncertain:
            if not self.rows[self.cursor].reviewed and not self.frame_missing():
                self.confirm(take_offer=False)    # the detector's guesses, as asked; the frames
                accepted += 1                     # were not looked at, so no offer was judged
            previous = self.cursor
            self.move(1)
            if self.cursor == previous:
                break
        self.follow_ball()
        return accepted

    def jump(self, direction: int) -> None:
        self.direction = direction
        if self.walk:                                  # already walking the uncertain ones
            position = self.walk_pos + direction
            while 0 <= position < len(self.walk):
                if self.rows[self.walk[position]].suspect or not self.rows[self.walk[position]].settled:
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
            if row.suspect or (row.uncertain and not row.reviewed):
                self.cursor = index
                self.follow_ball()
                return
            index += direction
        self.cursor = len(self.rows) - 1 if direction > 0 else 0
        self.follow_ball()

    def save(self, quiet: bool = False) -> None:
        """Write back every clip that has changed - and only those, so an untouched clip is left alone.

        The clip you are in gets its review file refreshed even with no edits, so quitting after
        only stepping through frames still resumes where you stopped rather than where you last
        clicked."""
        written, failed = [], []
        for source in self.sources:
            here = source.start <= self.cursor < source.start + len(source.rows)
            if here:
                source.cursor = self.cursor - source.start
            try:
                if not source.dirty:
                    if here and source.cursor != source.saved_cursor:
                        save_review(source.review, source.rows, source.cursor, self.zoom)
                        source.saved_cursor = source.cursor
                    continue
                save_rows(source.out_csv, source.rows, source.width, source.height)
                save_review(source.review, source.rows, source.cursor, self.zoom)
                save_flags(source.suspects, source.rows)
            except (OSError, LabelFormatError) as error:
                # Another program has the file open, most likely (or a sidecar was damaged under
                # us). Keep the edits in memory and keep going: the next autosave tries again, and
                # one stuck clip must not stop the others.
                print(f"[warn] could not save {source.clip}: {error} - close the file wherever it is open; "
                      f"the edits are kept and saving is retried")
                failed.append(source)
                continue
            source.saved_cursor = source.cursor
            source.dirty = False
            written.append(source)
        self.dirty = bool(failed)
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
        self.follow_ball()            # first, so a run resumed mid-way draws with that clip's size
        self.show()
        cv2.waitKey(1)
        if not self.grid:
            self.mouse.calibrate()
        self.hints.start(lambda: self.source)
        try:
            self.loop()
        finally:
            self.hints.closing = True
            # Any way out - q, the window's X, Ctrl+C, a crash - keeps the work done so far.
            try:
                self.save()
                self.rescue_unsaved()
            finally:
                self.remember_settings()
                cv2.destroyAllWindows()

    def rescue_unsaved(self) -> None:
        """Labels that still could not be written at exit go to <clip>_ball.unsaved.csv beside the
        real file, so closing the tool never throws edits away. Nothing reads that file; copy it
        over the label file once whatever held it open has let go."""
        for source in self.sources:
            if not source.dirty:
                continue
            rescue = ws.unsaved_csv(source.clip, source.out_csv.parent)
            try:
                save_rows(rescue, source.rows, source.width, source.height)
                print(f"[save] {source.out_csv.name} could not be written; your labels are in {rescue.name}")
            except OSError as error:
                # The last resort has failed and the tool is closing: there is nowhere left to put
                # these rows, so say so plainly and go on to rescue the other clips of the run.
                print(f"[error] {source.clip}: unsaved labels lost, {rescue.name} failed too: {error}")

    def loop(self) -> None:
        while True:
            if self.viewport.reset_needed:
                self.reset_window()
            self.track_window()
            was_gliding = self.zoom != self.zoom_target
            gliding = self.step_zoom()
            self.show()
            if was_gliding and not gliding:
                self.pending_cursor_move = True       # zoom has settled: put the mouse back on the ball
            if not self.overview and self.pending_cursor_move and not self.mouse.scrolling():
                # Placed once per frame change, not on every redraw. follow_ball has centred the ball
                # and the glide has no anchor, so the ball stays under the cursor while the zoom
                # finishes - repeating the warp only fought a hand that was on the wheel.
                self.pump(1)
                if self.view_ready():
                    self.place_cursor()
            if self.pending_key != -1:
                key, self.pending_key = self.pending_key, -1
            else:
                key = cv2.waitKeyEx(8 if gliding else 30)
            if not self.overview and self.view_ready():
                self.mouse.check_placement()    # only reconcile reports for the displayed view
            if self.dirty and time.monotonic() - self.last_save > AUTOSAVE_SECONDS:
                self.save(quiet=True)
            if key == -1:
                try:
                    if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                        return
                except cv2.error:
                    return
                if not self.overview and not gliding and self.mouse.aim is None:
                    self.prefetch()          # decode the frames SPACE is about to ask for, but not
                                             # while a warp is still waiting to be checked: a decode
                                             # runs long enough to age the answer out
                if not gliding and not self.mouse.scrolling() and not self.mouse.calibrating:
                    self.viewport.check(self.view_w, self.view_h + self.hud_h)   # whole picture, at x1?
                continue
            if self.handle_key(key):
                break

    # ---------------------------------------------------------------- keys
    def handle_key(self, key: int) -> bool:
        """One key from waitKeyEx; True when the tool should quit. The guards and q are here; what
        a key does belongs to the view showing it (SheetView, GridView, plain_key), and the grid and
        the single frame then share common_key."""
        code = key & 0xFF if key < 0x10000 else key
        if win.non_character_key(code):
            return False          # F1 is not p, F2 is not q: see NON_CHARACTER_KEYS
        if not self.overview and not self.view_ready() and code not in KEYS_QUIT:
            return False  # a mouse callback may have advanced while waitKey dispatched events
        if code in KEYS_QUIT:
            return self.quit_key()
        direction = self.key_direction(key, code)
        if self.sheet:
            self.sheet_view.handle_key(code, direction)
        elif self.grid:
            self.grid_view.handle_key(code, direction) or self.common_key(code)
        else:
            self.plain_key(code, direction) or self.common_key(code)
        return False

    @staticmethod
    def key_direction(key: int, code: int) -> int:
        """1 for a forward key (n, d, right arrow), -1 for a backward one (p, a, left arrow), else 0.
        Arrows: this build hands back the bare virtual-key codes (37/39), older Win32 builds pack
        them above bit 16, GTK and Qt have codes of their own. Match them all; none of them
        collide with a printable key the tool uses."""
        if code in (ord("n"), ord("d")) or key in KEYS_NEXT or code in KEYS_NEXT:
            return 1
        if code in (ord("p"), ord("a")) or key in KEYS_PREV or code in KEYS_PREV:
            return -1
        return 0

    def quit_key(self) -> bool:
        """q / ESC: save, and quit - unless labels could not be saved and that has not been said."""
        self.save()
        if self.dirty and not self.quit_warned:
            self.quit_warned = True
            print("[warn] some labels could not be saved (see above). Close the file in the other "
                  "program and press q again; quitting anyway keeps them in <clip>_ball.unsaved.csv")
            return False
        return True


    def plain_key(self, code: int, direction: int) -> bool:
        """What a key does in the single-frame view; False for a key it shares with the grid."""
        if code in KEYS_CONFIRM:
            self.confirm()
            self.move(1)
        elif code == ord("y"):
            if self.take_suggestion():
                self.move(1)
        elif code == ord("x"):
            self.confirm(take_offer=False)    # the draft is right, not the hint on offer
            self.move(1)
        elif code == ord("v"):
            self.mark_invisible()
            self.move(1)
        elif code == ord("r"):
            accepted = self.accept_run()
            print(f"[run ] accepted {accepted} confident frame(s); stopped at frame "
                  f"{self.rows[self.cursor].frame}")
        elif code == ord("e"):
            self.seek_disputed()
        elif code in (ord("f"), ord("b")):
            self.jump(1 if code == ord("f") else -1)
        elif direction:
            self.move(direction * self.step)
        elif code in KEYS_STEP:
            self.step = KEYS_STEP[code]
        elif code in KEYS_ZOOM_IN + KEYS_ZOOM_OUT:
            self.zoom_key(1.0 if code in KEYS_ZOOM_IN else -1.0)
        else:
            return False
        return True

    def common_key(self, code: int) -> None:
        """The keys the grid and the single frame read the same way."""
        toggles = {ord("w"): self.toggle_sheet, KEY_TAB: self.toggle_grid, ord("z"): self.toggle_grid_size,
                   ord("t"): self.toggle_trail, ord("g"): self.toggle_follow, ord("h"): self.cycle_filter,
                   ord("o"): self.toggle_inset, ord("m"): self.toggle_fullscreen, ord("s"): self.save}
        if code in toggles:
            toggles[code]()
        elif code == ord("u"):
            self.rows[self.cursor].undo()
            self.dirty = self.source.dirty = True
            self.follow_ball()
        elif code in KEYS_NUDGE:
            dx, dy = KEYS_NUDGE[code]
            step = .25 if (chr(code).isupper() or win.shift_held()) else 1.
            self.nudge(dx * step, dy * step)

    def seek_disputed(self) -> bool:
        """e / SHIFT+e: the next / previous frame the suggestions disagree with. False, with a
        note, when there is none that way."""
        if self.jump_disputed(-1 if win.shift_held() else 1):
            return True
        print("[hint] no more frames the suggestions disagree with that way")
        return False

    def toggle_trail(self) -> None:
        self.trail = not self.trail
        self.remember_settings()

def build_parser() -> argparse.ArgumentParser:
    """The command line; the folder defaults are the workspace's as it is now (``TENNIS_WORKSPACE``)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clips", nargs="+", help="Clip stem(s), or a range like video22..video53")
    parser.add_argument("--videos", type=Path, default=ws.VIDEOS)
    parser.add_argument("--labels", type=Path, default=ws.LABELS)
    parser.add_argument("--zoom", type=float, default=None, help="Starting zoom (default: last used, else 2)")
    parser.add_argument("--grid", action="store_true", help="Review a page of enlarged ball crops together (z: 8 or 16); TAB toggles this view")
    parser.add_argument("--uncertain", action="store_true",
                        help="Only stop on frames that want attention: no confident guess, or flagged "
                             "by ft.py check --audit --mark. Every other label is left alone.")
    return parser


def main() -> int:
    backend = win.highgui_backend()
    if not backend.upper().startswith("QT"):
        # The view, wheel and key handling are built around Qt's window: on another backend the
        # zoom check never passes and the window is rebuilt over and over.
        print(f"[error] this OpenCV draws its windows with {backend}, but the label tool needs the Qt "
              f"build. Install conda-forge's instead of pip's:\n"
              f"        pip uninstall opencv-python && conda install -c conda-forge opencv",
              file=sys.stderr)
        return 2
    args = build_parser().parse_args()

    try:
        names = expand(args.clips, args.labels)
    except ValueError as error:
        print(f"[error] {error}", file=sys.stderr)
        return 2
    sources = [source for source in (build_source(name, args.videos, args.labels) for name in names)
               if source is not None]
    if not sources:
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
    tool.settings_path = args.labels / SETTINGS_NAME
    settings = load_settings(tool.settings_path)
    if settings.get("filter") in FILTERS:
        tool.filter = FILTERS.index(settings["filter"])
    tool.inset = bool(settings.get("inset", tool.inset))
    tool.trail = bool(settings.get("trail", tool.trail))
    if settings.get("grid_size") in GRID_SIZES:
        tool.grid_size = settings["grid_size"]

    total = sum(len(source.rows) for source in sources)
    flagged = sum(source.flagged for source in sources)
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
    if args.grid:
        tool.toggle_grid()
    tool.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
