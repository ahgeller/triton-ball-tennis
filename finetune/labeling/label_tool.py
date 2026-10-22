"""Correct the detector's draft labels as fast as possible.

    python finetune/labeling/label_tool.py <clip>                 # videos/<clip>.mp4 + labels/<clip>_ball.csv
    python finetune/labeling/label_tool.py video22..video53      # a whole run, one clip after another
    python finetune/labeling/label_tool.py video22..video53 --uncertain   # only the frames worth a look

Every label frame comes up with the mark drawn and the mouse on it: SPACE if the mark is on the
ball, a click where the ball is if not, v if there is no ball. Every key, the grid and sheet
views, and what the marks, hints and badges mean: finetune/labeling/LABEL_TOOL.md (the bar at the bottom
of the window names the keys that matter on the frame you are on). Output is
labels/<clip>_ball.csv, with progress in labels/<clip>_ball.review.json and the landings marked
with / in labels/<clip>_bounces.csv, autosaved.

``Tool`` is the actions - the edits and the navigation the keys name (``label_keys``), the window,
the picture and the loop; the state they act on is one ``LabelSession`` (``label_session``), which
the views and handlers read and write directly.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Hashable, List, Optional, Tuple

import cv2
import numpy as np

if __name__ == "__main__":          # run as a script: make the repository root importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from finetune.data.bounce_labels import between_frames
from finetune.labeling import label_window as win
from finetune.labeling.label_cursor import CursorPlacer
from finetune.labeling.label_draw import FILTERS
from finetune.labeling.label_frame_view import FrameView
from finetune.labeling.label_grid import GridView
from finetune.labeling.label_keys import KeyHandler
from finetune.labeling.label_mouse import MouseHandler
from finetune.labeling.label_session import LabelSession
from finetune.labeling.label_sheet import SheetView
from finetune.labeling.label_source import Prefetcher, Source
from finetune.labeling.label_store import (Row, rescue_unsaved, save_flags, save_landings, save_review, save_rows,
                                           save_settings)
from finetune.labeling.label_window import WINDOW
from finetune.data.labels import LabelFormatError, is_visible

AUTOSAVE_SECONDS = 5.0    # write labels + review progress this often while there are unsaved changes


class Tool:
    def __init__(self, sources: List[Source]):
        self.session = LabelSession(sources)
        session = self.session
        self.client = None                     # last client size seen; a change re-fits and re-measures
        self.mouse = CursorPlacer(session, self.pump)   # parks the OS pointer on the ball (Windows); see label_cursor
        self.viewport = win.ViewportGuard()    # keeps the Qt window's own zoom at x1
        self.frame_view = FrameView(session, self)      # the single-frame picture and HUD; see label_frame_view
        self.grid_view = GridView(session, self)        # TAB's page of crops; see label_grid
        self.sheet_view = SheetView(session, self)      # w's trajectory sheet; see label_sheet
        self.pointer = MouseHandler(session, self)      # what clicks and the wheel do; see label_mouse
        self.keys = KeyHandler(session, self)           # what the keys do; see label_keys
        self.prefetcher = Prefetcher(session)           # decodes ahead of the cursor while idle; see label_source
        self.fullscreen = False
        self.pending_cursor_move = False
        self.last_save = time.monotonic()
        self.last_clip = None
        self.presented = None
        self.settings_path: Optional[Path] = None
        self.normal_rect = None       # the window's rect before going full screen, to rebuild into
        self.pending_key = -1         # a key pumped out of the queue by a short waitKey, kept for loop
        self.render_key = None        # what the last single-frame render was drawn from, see render_state
        self.render_cache = None
        self.shown = None             # the image last handed to imshow, so an unchanged one is not re-sent
        self.quit_warned = False      # q with unsaved labels warns once, then quits

    @property
    def overview(self) -> bool:
        """The grid or the trajectory sheet: views with no single frame under a parked mouse."""
        return self.grid_view.active or self.sheet_view.active

    # ---------------------------------------------------------------- the window
    def create_window(self) -> None:
        """WINDOW_NORMAL, not WINDOW_AUTOSIZE: autosize lets highgui pick the size from the image and
        the monitor DPI, which on a 200% display made a 1080p clip a 3226x2095 window on a 2880x1800
        desktop, and left no way to maximise it. WINDOW_GUI_NORMAL: the plain window, without the Qt
        toolbar whose magnifier buttons zoom the window itself (see viewport_wheel) and without the
        status bar, so the whole client area is picture."""
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        cv2.setMouseCallback(WINDOW, self.pointer.on_mouse)

    def open_window(self) -> None:
        """Create a resizable window filling the desktop."""
        self.create_window()
        cv2.imshow(WINDOW, np.zeros((self.session.height, self.session.width, 3), dtype=np.uint8))
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
        """Draw at the size the window actually is (LabelSession.fit); a new size means highgui
        fits the picture differently, so the pointer map is measured again."""
        self.session.anchor = None
        self.mouse.aim = None
        if self.session.fit(client_w, client_h):
            self.mouse.map_stale = True
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

    # ---------------------------------------------------------------- following the ball
    def toggle_follow(self) -> None:
        """Auto-centring on or off, from the g key or a click on the HUD badge.

        Off leaves the view and the mouse exactly where they are on every frame, for reading a
        frame or working somewhere other than the middle of the screen; back on snaps to the ball
        straight away rather than waiting for the next frame.
        """
        self.session.follow = not self.session.follow
        if self.session.follow:
            self.follow_ball()

    def follow_ball(self) -> None:
        """Centre the view on this frame's ball, so it is always in the middle and the mouse - which
        goes to the same place - never has to be hunted for.

        Dropping the zoom anchor here matters. It belongs to the frame you were on when you turned
        the wheel, and a glide still in flight would otherwise keep hauling the view back to that
        frame's ball - so scrolling and then pressing SPACE straight away left you looking at, and
        with the mouse parked on, the previous detection.
        """
        session = self.session
        if self.last_clip != session.source.clip:
            if self.last_clip is not None:
                session.by_clip[self.last_clip].release()   # let go of the clip being left, decoder included
            self.last_clip = session.source.clip
            session.width, session.height = session.source.width, session.source.height
            self.fit_view(*(self.client or (session.view_w, session.view_h + session.hud_h)))
        session.anchor = None
        self.mouse.aim = None         # a pending OS cursor report belongs to the previous view
        session.zoom = session.zoom_target  # finish the old frame's animation before centering the next
        if self.overview:
            self.pending_cursor_move = False
            return
        row = session.rows[session.cursor]
        session.ghost = None if row.visible else session.guess_for(session.cursor)
        if not session.follow:
            return                    # auto-centring is off: leave the view and the mouse be
        point = session.target_point()   # the mark on offer, the ball, or the neighbours' guess
        if point is not None:
            session.center = point
        session.centre_view()
        self.pending_cursor_move = True

    def place_cursor(self) -> None:
        """Put the OS mouse cursor on the current guess so a click is a one-pixel move.

        follow_ball has already put the ball in the middle of the view, so this normally lands the
        pointer dead centre. A frame with no ball has nothing to aim at, and the pointer goes to
        the centre anyway rather than being left wherever the last frame happened to leave it.
        """
        session = self.session
        self.pending_cursor_move = False
        if win._user32 is None or not session.follow:
            return
        if self.mouse.cursor_map is None or self.mouse.map_stale:
            self.mouse.calibrate()                   # never measured, or the window changed size
        point = session.target_point()               # the ball, or the ghost guess standing in for it
        if point is None:
            self.mouse.warp(session.view_w / 2.0, session.view_h / 2.0)
            return
        view_x, view_y = session.to_view(*point)
        if not (0 <= view_x < session.view_w and 0 <= view_y < session.view_h):
            session.center = point                # nothing should leave the ball off screen, but if
            session.centre_view()                 # anything does, do not leave the mouse behind
            view_x, view_y = session.to_view(*point)
        self.mouse.warp(view_x, view_y)

    def zoom_key(self, notches: float) -> None:
        """Keyboard zoom pivots on the ball so it stays put on screen (view centre when invisible)."""
        session = self.session
        point = session.target_point()
        if point is not None:         # exact, not rounded to a view pixel: see LabelSession.zoom_at
            session.zoom_at(notches, (point[0] - session.origin[0]) * session.scale,
                            (point[1] - session.origin[1]) * session.scale)
        else:
            session.zoom_at(notches, session.view_w / 2.0, session.view_h / 2.0)
        self.pending_cursor_move = True

    # ---------------------------------------------------------------- the picture
    def view_ready(self) -> bool:
        session = self.session
        return self.presented == (session.cursor, session.origin, session.scale)

    def show(self) -> None:
        image = self.render()
        if image is not self.shown:   # unchanged and already on screen: skip
            cv2.imshow(WINDOW, image)
            self.shown = image
        self.presented = (self.session.cursor, self.session.origin, self.session.scale)

    def render_state(self, image: np.ndarray, before: Optional[np.ndarray]) -> Tuple[Hashable, ...]:
        """Everything the single-frame view is drawn from. render redraws only when this changes,
        so the loop can call it every 30 ms for free. IF YOU MAKE render OR FrameView (the picture,
        the inset, the HUD) READ SOMETHING NEW, ADD IT HERE - a screen that does not change when it should is far worse than
        the CPU this saves. Any write to any Row bumps Row.revision, which covers the labels, review
        flags, trail and every counter in the HUD; the frame images are keyed by identity, which
        changes when one is decoded, evicted or replaced by a placeholder. The landings are the
        clip's, not the rows', so what the view reads of them is keyed by hand: the landing this
        frame wears and its point, whether one is waiting for its click, and how many there are."""
        s = self.session
        landings = s.source.landings
        landing = s.landing_for(s.cursor)
        return (Row.revision, s.cursor, s.origin, s.scale, s.zoom, s.view_w, s.view_h, s.hud_h, s.ui, s.width,
                s.height, s.filter, s.inset, s.trail, s.follow, s.step, s.dirty, id(s.walk), len(s.walk), s.walk_pos,
                s.ghost, len(s.sources), id(image), id(before), s.hints.version,
                landing, landings.get(landing) if landing is not None else None, s.awaiting_landing, len(landings))

    def render(self) -> np.ndarray:
        if self.sheet_view.active:
            return self.sheet_view.render()
        if self.grid_view.active:
            return self.grid_view.render()
        session = self.session
        session.centre_view()
        row = session.rows[session.cursor]
        image = session.source.frame(row.frame)
        mode = session.filter_name
        before = session.previous_image() if mode == "motion" else None
        key = self.render_state(image, before)
        if key == self.render_key:
            return self.render_cache
        hint = session.suggestion_for(session.cursor)
        offer = session.offer_for(session.cursor)
        session.shown_offers = {} if offer is None else {session.cursor: (offer["x"], offer["y"])}
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

    # ---------------------------------------------------------------- the toggles
    def cycle_filter(self) -> None:
        self.session.filter = (self.session.filter + 1) % len(FILTERS)
        self.grid_view.invalidate()
        self.remember_settings()

    def toggle_inset(self) -> None:
        self.session.inset = not self.session.inset
        self.remember_settings()

    def toggle_trail(self) -> None:
        self.session.trail = not self.session.trail
        self.remember_settings()

    def remember_settings(self) -> None:
        if self.settings_path is not None:
            save_settings(self.settings_path, {"filter": self.session.filter_name, "inset": self.session.inset,
                                               "grid_size": self.grid_view.size,
                                               "trail": self.session.trail})

    # ---------------------------------------------------------------- edits
    def take_suggestion(self) -> bool:
        """Move this frame's label to its suggestion. True when there was one to take."""
        session = self.session
        hint = session.suggestion_for(session.cursor)
        if hint is None or not self.set_point(hint["x"], hint["y"]):
            return False
        session.rows[session.cursor].source = "refined"   # kept apart from clicks in the review file
        return True

    def set_point(self, x: float, y: float) -> bool:
        """Put this frame's ball at (x, y). False when nothing was stored."""
        session = self.session
        if session.frame_missing():
            return False
        x = float(min(max(x, 0.0), session.width - 1.0))
        y = float(min(max(y, 0.0), session.height - 1.0))
        if not is_visible(x, y, session.width, session.height):
            # The label CSV marks "no ball" by the top-right pixel, and every reader in the workspace
            # takes a point on it to mean that. Stored, this would read back as absent.
            print(f"[warn] frame {session.rows[session.cursor].frame}: ({x:.0f}, {y:.0f}) is the top-right "
                  f"pixel the label files reserve for 'no ball', so it cannot be saved as a ball there")
            return False
        row = session.rows[session.cursor]
        row.snapshot()
        row.suspect = False           # an edit answers an audit flag
        row.x, row.y = x, y
        if row.source == "refined":
            row.source = "click"      # placed by hand now, not taken from a suggestion
        row.visible = True
        row.reviewed = row.edited = True
        session.dirty = session.source.dirty = True
        session.source.cursor = session.cursor - session.source.start
        return True

    def nudge(self, dx: float, dy: float) -> None:
        row = self.session.rows[self.session.cursor]
        if row.visible:
            self.set_point(row.x + dx, row.y + dy)
            self.pending_cursor_move = True

    def mark_invisible(self) -> None:
        session = self.session
        if session.frame_missing():
            return
        row = session.rows[session.cursor]
        row.snapshot()
        row.suspect = False
        row.visible = False
        row.reviewed = row.edited = True
        session.dirty = session.source.dirty = True

    def toggle_landing(self) -> None:
        """/: the ball is on the ground on this frame - or, pressed again, it is not. Pressed on the
        row after (or before) a frame marked that way, the ball bounced between the two: that mark
        becomes one landing halfway between them (3.5 between frames 3 and 4; on a 60 FPS clip
        labelled every other frame, the unlabelled frame between), and both frames wear it. / on
        either of the two frames removes it again.

        A landing with no ball label under it - one between two frames, or one on a frame whose
        row says there is no ball - has nothing to take its point from, so in the single-frame
        view the tool then waits for a click on where the ball touched the ground
        (session.awaiting_landing, taken by MouseHandler.place_click; any other key leaves the
        landing without a point). The grid has no picture to click on and never waits.

        A landing is the clip's (Source.landings), not part of the row's label, so u leaves it
        alone: the key is its own undo. A landing without a point stores none: the court-event
        scorer takes the ball label of its frame, else of the frame before it, else of the frame
        after (so the whole frame between two labelled every other frame takes the one before),
        or the midpoint of the two a half frame lies between (bounce_labels.resolve_points), so it
        follows the labels wherever they are moved."""
        session = self.session
        session.awaiting_landing = None
        if session.frame_missing():
            return                    # a black placeholder is not a frame the ball can be seen on
        index = session.cursor
        row = session.rows[index]
        landings = session.source.landings
        worn = session.landing_for(index)
        if worn is not None:
            del landings[worn]        # on this frame, or between it and the next: unmarked either way
        else:
            placed = row.frame
            for other in session.neighbour_frames(index):   # the previous row first: the one just marked
                if other in landings:
                    del landings[other]
                    placed = between_frames(row.frame, other)
                    break
            landings[placed] = None
            if not self.overview and (placed != row.frame or not row.visible):
                session.awaiting_landing = placed   # no ball label under it: ask where it touched the ground
        session.dirty = session.source.dirty = True

    def place_landing(self, x: float, y: float) -> None:
        """The click a landing with no ball label under it waits for (toggle_landing): the ball
        touched the ground at (x, y), kept on the frame. The frame stays up, so the spot can be
        seen marked."""
        session = self.session
        landing, session.awaiting_landing = session.awaiting_landing, None
        if landing is None or landing not in session.source.landings:
            return                    # removed since, or never there: nothing to place
        session.source.landings[landing] = (float(min(max(x, 0.0), session.width - 1.0)),
                                            float(min(max(y, 0.0), session.height - 1.0)))
        session.dirty = session.source.dirty = True

    def confirm(self, take_offer: bool = True) -> None:
        """SPACE: the mark on screen is right. When that mark was a suggestion on offer, the label
        moves there (recorded as source "refined", like y); otherwise the label is confirmed where
        it is. Only an offer that was actually drawn counts - one that arrived between the draw and
        the key press is not taken on a frame you judged without it. `take_offer=False` confirms
        the label as it is (x, and r's unseen frames)."""
        session = self.session
        if session.frame_missing():
            return                    # a black placeholder is not something to say "good" to
        offer = session.shown_offers.get(session.cursor) if take_offer else None
        if offer is not None:
            if self.set_point(*offer):
                session.rows[session.cursor].source = "refined"
            return                    # could not be stored (set_point said why): nothing confirmed
        row = session.rows[session.cursor]
        if not row.reviewed or row.suspect:
            row.snapshot()
        row.suspect = False           # "it is right where it is" answers an audit flag
        row.reviewed = True
        session.dirty = session.source.dirty = True

    def undo(self) -> None:
        """u: the last change on this frame comes back off its history."""
        session = self.session
        session.rows[session.cursor].undo()
        session.dirty = session.source.dirty = True
        self.follow_ball()

    # ---------------------------------------------------------------- navigation
    def move(self, delta: int) -> None:
        self.session.step_cursor(delta)
        self.follow_ball()

    def jump(self, direction: int) -> None:
        """f / b: the next / previous frame that wants attention (LabelSession.next_uncertain)."""
        session = self.session
        session.direction = direction
        session.set_cursor(session.next_uncertain(direction))
        self.follow_ball()

    def jump_disputed(self, direction: int) -> bool:
        """Go to the next (or previous) frame the suggestions disagree with. False if there is none."""
        session = self.session
        index = session.next_disputed(direction)
        if index is None:
            return False
        session.direction = direction
        session.set_cursor(index)
        self.follow_ball()
        return True

    def seek_disputed(self) -> bool:
        """e / SHIFT+e: the next / previous frame the suggestions disagree with. False, with a
        note, when there is none that way."""
        if self.jump_disputed(-1 if win.shift_held() else 1):
            return True
        print("[hint] no more frames the suggestions disagree with that way")
        return False

    def accept_run(self) -> int:
        """Mark every confident guess from here up to the next iffy one as reviewed, and stop there.

        f already skips past confident frames, but it leaves them unreviewed, so the counter never
        moves and the same frames come round again. This is that skip with the tick applied.
        """
        session = self.session
        accepted = 0
        while not session.rows[session.cursor].uncertain:
            if not session.rows[session.cursor].reviewed and not session.frame_missing():
                self.confirm(take_offer=False)    # the detector's guesses, as asked; the frames
                accepted += 1                     # were not looked at, so no offer was judged
            previous = session.cursor
            self.move(1)
            if session.cursor == previous:
                break
        self.follow_ball()
        return accepted

    # ---------------------------------------------------------------- saving
    def save(self, quiet: bool = False) -> None:
        """Write back every clip that has changed - and only those, so an untouched clip is left alone:
        its labels, its review and flag sidecars and its landings (label_store.save_landings).

        The clip you are in gets its review file refreshed even with no edits, so quitting after
        only stepping through frames still resumes where you stopped rather than where you last
        clicked."""
        session = self.session
        written, failed = [], []
        for source in session.sources:
            here = source.start <= session.cursor < source.start + len(source.rows)
            if here:
                source.cursor = session.cursor - source.start
            try:
                if not source.dirty:
                    if here and source.cursor != source.saved_cursor:
                        save_review(source.review, source.rows, source.cursor, session.zoom)
                        source.saved_cursor = source.cursor
                    continue
                save_rows(source.out_csv, source.rows, source.width, source.height)
                save_review(source.review, source.rows, source.cursor, session.zoom)
                save_flags(source.suspects, source.rows)
                save_landings(source.bounces, source.landings)
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
        session.dirty = bool(failed)
        self.last_save = time.monotonic()
        if not quiet and written:
            for source in written:
                print(f"[save] {source.out_csv.name} ({sum(r.visible for r in source.rows)} visible, "
                      f"{sum(r.reviewed for r in source.rows)}/{len(source.rows)} reviewed"
                      f"{f', {len(source.landings)} landing(s)' if source.landings else ''})")

    def rescue_unsaved(self) -> None:
        """At exit, the labels of every clip that still could not be written go beside the real
        file (label_store.rescue_unsaved)."""
        rescue_unsaved(self.session.sources)

    # ---------------------------------------------------------------- loop
    def run(self) -> None:
        session = self.session
        self.open_window()
        self.track_window()
        session.center = (session.width / 2.0, session.height / 2.0)
        session.centre_view()
        self.follow_ball()            # first, so a run resumed mid-way draws with that clip's size
        self.show()
        cv2.waitKey(1)
        if not self.grid_view.active:
            self.mouse.calibrate()
        session.hints.start(lambda: session.source)
        try:
            self.loop()
        finally:
            session.hints.closing = True
            # Any way out - q, the window's X, Ctrl+C, a crash - keeps the work done so far.
            try:
                self.save()
                self.rescue_unsaved()
            finally:
                self.remember_settings()
                cv2.destroyAllWindows()

    def loop(self) -> None:
        while True:
            if self.viewport.reset_needed:
                self.reset_window()
            self.track_window()
            gliding = self.redraw()
            key = self.next_key(gliding)
            if not self.overview and self.view_ready():
                self.mouse.check_placement()    # only reconcile reports for the displayed view
            if self.session.dirty and time.monotonic() - self.last_save > AUTOSAVE_SECONDS:
                self.save(quiet=True)
            if key == -1:
                if self.idle(gliding):
                    return
                continue
            if self.keys.handle(key):
                break

    def redraw(self) -> bool:
        """One redraw: a step of the zoom glide, the picture, and the mouse put on the ball once
        the view it belongs to is up. True while the zoom is still gliding."""
        session = self.session
        was_gliding = session.zoom != session.zoom_target
        gliding = session.step_zoom()
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
            self.viewport.check(self.session.view_w, self.session.view_h + self.session.hud_h)   # whole picture, at x1?
        return False


def main() -> int:
    """The command line (label_cli); this file stays the script people run."""
    from finetune.labeling.label_cli import main as cli_main
    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
