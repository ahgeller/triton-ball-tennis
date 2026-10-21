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
from finetune.labeling.label_draw import FILTERS, ZOOM_STEP
from finetune.labeling.label_frame_view import FrameView
from finetune.labeling.label_hints import Suggestions
from finetune.labeling.label_keys import KeyHandler
from finetune.labeling.label_mouse import MouseHandler
from finetune.labeling.label_source import Prefetcher, Source, build_source, expand   # noqa: F401 - the tool's parts, by name
from finetune.labeling.label_store import Row, rescue_unsaved, save_flags, save_review, save_rows, save_settings
from finetune.labeling.label_views import GridView, SheetView
from finetune.labeling.label_window import WINDOW
from finetune.data.labels import LabelFormatError, is_visible
from finetune.labeling.refine import Suggestion

HUD = 58
MAX_UPSCALE = 2.0         # cap on drawing the frame bigger than it is: past this it only costs time
UI_REFERENCE = 1280.0     # view width the HUD text sizes were chosen for; bigger views scale up
ZOOM_EASE = 0.34          # share of the remaining gap closed each redraw: a glide, not a jump
ZOOM_MAX = 12.0
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
        self.frame_view = FrameView(self)      # the single-frame picture and HUD; see label_frame_view
        self.pointer = MouseHandler(self)      # what clicks and the wheel do; see label_mouse
        self.keys = KeyHandler(self)           # what the keys do; see label_keys
        self.prefetcher = Prefetcher(self)     # decodes ahead of the cursor while idle; see label_source
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
        self.dirty = False
        self.pending_cursor_move = False
        self.last_save = time.monotonic()
        self.grid_view = GridView(self)   # TAB's page of crops; see label_views
        self.sheet_view = SheetView(self) # w's trajectory sheet

        self.last_clip = None
        self.presented = None
        self.filter = 0               # index into FILTERS; h cycles it, the HUD chip shows it
        self.inset = True             # the small whole-frame overview drawn while zoomed in
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

    @property
    def filter_name(self) -> str:
        """The view filter in force, by its name in FILTERS (h cycles it)."""
        return FILTERS[self.filter]

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
        before, after = self.visible_neighbours(index)
        frame = self.rows[index].frame
        if before is not None and after is not None and after.frame > before.frame:
            share = (frame - before.frame) / (after.frame - before.frame)
            return (before.x + (after.x - before.x) * share, before.y + (after.y - before.y) * share)
        near = before if after is None else after if before is None else None
        if near is not None and abs(self.rows.index(near) - index) == 1:
            return (near.x, near.y)
        return None

    def visible_neighbours(self, index: int) -> Tuple[Optional[Row], Optional[Row]]:
        """The nearest rows with a ball within GHOST_REACH either side of `index`, in its clip."""
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
        return before, after

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

    def render_state(self, image: np.ndarray, before: Optional[np.ndarray]) -> Tuple[Hashable, ...]:
        """Everything the single-frame view is drawn from. render redraws only when this changes,
        so the loop can call it every 30 ms for free. IF YOU MAKE render OR FrameView (the picture,
        the inset, the HUD) READ SOMETHING NEW, ADD IT HERE - a screen that does not change when it should is far worse than
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
        mode = self.filter_name
        before = self.previous_image() if mode == "motion" else None
        key = self.render_state(image, before)
        if key == self.render_key:
            self.render_fresh = False
            return self.render_cache
        self.render_fresh = True
        hint = self.suggestion_for(self.cursor)
        offer = self.offer_for(self.cursor)
        self.shown_offers = {} if offer is None else {self.cursor: (offer["x"], offer["y"])}
        canvas = self.frame_view.draw(image, before, mode, row, hint, offer)
        self.render_key, self.render_cache = key, canvas
        return canvas

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
        """The window's mouse callback (label_mouse.MouseHandler)."""
        self.pointer.on_mouse(event, x, y, flags, param)

    def cycle_filter(self) -> None:
        self.filter = (self.filter + 1) % len(FILTERS)
        self.grid_view.invalidate()
        self.remember_settings()

    def toggle_inset(self) -> None:
        self.inset = not self.inset
        self.remember_settings()

    def remember_settings(self) -> None:
        if self.settings_path is not None:
            save_settings(self.settings_path, {"filter": self.filter_name, "inset": self.inset,
                                               "grid_size": self.grid_size,
                                               "trail": self.trail})

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
        """At exit, the labels of every clip that still could not be written go beside the real
        file (label_store.rescue_unsaved)."""
        rescue_unsaved(self.sources)

    def loop(self) -> None:
        while True:
            if self.viewport.reset_needed:
                self.reset_window()
            self.track_window()
            gliding = self.redraw()
            key = self.next_key(gliding)
            if not self.overview and self.view_ready():
                self.mouse.check_placement()    # only reconcile reports for the displayed view
            if self.dirty and time.monotonic() - self.last_save > AUTOSAVE_SECONDS:
                self.save(quiet=True)
            if key == -1:
                if self.idle(gliding):
                    return
                continue
            if self.handle_key(key):
                break

    def redraw(self) -> bool:
        """One redraw: a step of the zoom glide, the picture, and the mouse put on the ball once
        the view it belongs to is up. True while the zoom is still gliding."""
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
        return gliding

    def next_key(self, gliding: bool) -> int:
        """A key pumped out of the queue earlier, else what highgui delivers in one short wait."""
        if self.pending_key != -1:
            key, self.pending_key = self.pending_key, -1
            return key
        return cv2.waitKeyEx(8 if gliding else 30)

    def idle(self, gliding: bool) -> bool:
        """What a tick with no key does: notice a closed window (True: the loop ends), decode
        ahead and check the window's own zoom."""
        try:
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                return True
        except cv2.error:
            return True
        if not self.overview and not gliding and self.mouse.aim is None:
            self.prefetcher.run()    # decode the frames SPACE is about to ask for, but not
                                     # while a warp is still waiting to be checked: a decode
                                     # runs long enough to age the answer out
        if not gliding and not self.mouse.scrolling() and not self.mouse.calibrating:
            self.viewport.check(self.view_w, self.view_h + self.hud_h)   # whole picture, at x1?
        return False

    # ---------------------------------------------------------------- keys (label_keys has the table)
    def handle_key(self, key: int) -> bool:
        """One key from waitKeyEx; True when the tool should quit (label_keys.KeyHandler)."""
        return self.keys.handle(key)

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

def main() -> int:
    """The command line (label_cli); this file stays the script people run."""
    # label_cli builds a Tool, so it imports this module: imported here, when the script runs,
    # rather than at load, or the two would import each other half-made.
    from finetune.labeling.label_cli import main as cli_main
    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
