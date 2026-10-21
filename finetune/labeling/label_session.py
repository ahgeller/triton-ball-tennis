"""One labelling sitting's state, with one owner: the clips of the run and their rows, where the
cursor is and the walk it may be restricted to, the view's geometry, the toggles, what is unsaved,
and the suggestions with the rules for what each row is shown. ``Tool`` (``label_tool``) acts on
it - the edits, the navigation the keys name, the window and the loop - and the views and handlers
(``label_grid``, ``label_sheet``, ``label_frame_view``, ``label_mouse``, ``label_keys``,
``Prefetcher``) read and write it directly. Nothing about the window or the picture lives here.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from finetune.labeling.label_draw import FILTERS, ZOOM_STEP
from finetune.labeling.label_hints import Suggestions
from finetune.labeling.label_source import Source
from finetune.labeling.label_store import Row
from finetune.refiner.refine import Point, Suggestion

HUD = 58
MAX_UPSCALE = 2.0         # cap on drawing the frame bigger than it is: past this it only costs time
UI_REFERENCE = 1280.0     # view width the HUD text sizes were chosen for; bigger views scale up
ZOOM_EASE = 0.34          # share of the remaining gap closed each redraw: a glide, not a jump
ZOOM_MAX = 12.0
GHOST_REACH = 3           # rows either side searched for a ball to interpolate a missing guess from
SNAP_PX = 4               # a wheel turned with the pointer this close to the ball zooms about the ball

Anchor = Tuple[Point, Tuple[float, float]]   # (frame point, view point) a zoom glide holds still


class LabelSession:
    """The state of one sitting over a run of clips; see the module docstring for who reads it."""

    def __init__(self, sources: List[Source]):
        self.sources = sources
        self.rows: List[Row] = []
        for source in sources:
            source.start = len(self.rows)
            self.rows.extend(source.rows)
        self.by_clip = {source.clip: source for source in sources}
        self.hints = Suggestions(sources)      # refine.suggest's measurements, kept current in the background
        # the cursor
        self.cursor = 0
        self.walk: List[int] = []     # row indices navigation visits; empty means every row
        self.walk_pos = 0
        self.step = 1                 # frames n/d/p/a move at a time; SPACE and a click stay at 1
        self.direction = 1            # which way the last move went, so prefetch runs ahead of a or d
        # the view: the frame's size, the window's, and where the picture sits in it
        self.width, self.height = sources[0].width, sources[0].height
        self.base_scale = 1.0                  # replaced by fit once the window has a size
        self.view_w, self.view_h = self.width, self.height
        self.ui = 1.0                          # HUD/marker scale, so text stays readable when enlarged
        self.hud_h = HUD
        self.zoom = 1.0
        self.zoom_target = 1.0
        self.anchor: Optional[Anchor] = None   # what the zoom glide holds still
        self.center = (self.width / 2.0, self.height / 2.0)
        self.origin = (0.0, 0.0)
        # the toggles
        self.trail = True
        self.follow = True            # auto-centre: the view and the mouse go to the guess
        self.filter = 0               # index into FILTERS; h cycles it, the HUD chip shows it
        self.inset = True             # the small whole-frame overview drawn while zoomed in
        # what the frame on screen is about
        self.dirty = False            # unsaved edits in some clip
        self.ghost: Optional[Point] = None     # a frame with no guess: where the neighbours say the ball is
        self.shown_offers: Dict[int, Point] = {}   # row index -> the suggestion drawn as the mark on
                                                   # offer in the last render; what SPACE takes
        self.disputed_key, self.disputed = None, 0   # disputed_count's cache

    # ---------------------------------------------------------------- the clip we are in
    @property
    def source(self) -> Source:
        return self.by_clip[self.rows[self.cursor].clip]

    @property
    def clip(self) -> str:
        return self.source.clip

    def clip_bounds(self, index: int) -> Tuple[int, int]:
        """(first, last + 1) row indices of the clip row `index` belongs to."""
        source = self.by_clip[self.rows[index].clip]
        return source.start, source.start + len(source.rows)

    def frame(self, index: int) -> np.ndarray:
        """The picture of source frame `index` of the clip we are in (see Source.frame)."""
        return self.source.frame(index)

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

    # ---------------------------------------------------------------- the cursor
    def set_cursor(self, index: int) -> None:
        """Put the cursor on row `index`, keeping the walk's position in step with it when the row
        is on the walk: the one place the two are written together."""
        self.cursor = index
        if self.walk and index in self.walk:
            self.walk_pos = self.walk.index(index)

    def step_cursor(self, delta: int) -> None:
        """Move `delta` rows along the walk (or the run), stopping at either end."""
        if delta:
            self.direction = 1 if delta > 0 else -1
        if self.walk:
            self.walk_pos = min(max(self.walk_pos + delta, 0), len(self.walk) - 1)
            self.cursor = self.walk[self.walk_pos]
        else:
            self.cursor = min(max(self.cursor + delta, 0), len(self.rows) - 1)

    def next_uncertain(self, direction: int) -> int:
        """The next row that way wanting attention - an open audit flag, or uncertain and not yet
        reviewed (every row of the walk is uncertain already) - or the end that way of the run, or
        of the walk when there is one, when no row does."""
        along = self.walk or range(len(self.rows))      # the rows navigation visits, in order
        at = self.walk_pos if self.walk else self.cursor
        for position in range(at + direction, len(along) if direction > 0 else -1, direction):
            row = self.rows[along[position]]
            if row.suspect or (not row.reviewed and (self.walk or row.uncertain)):
                return along[position]
        return along[-1] if direction > 0 else along[0]

    def next_disputed(self, direction: int) -> Optional[int]:
        """The next row that way whose label a trusted suggestion disagrees with; None if there is none."""
        index = self.cursor + direction
        while 0 <= index < len(self.rows):
            if self.hint_verdict(index) == "differs":
                return index
            index += direction
        return None

    def only_uncertain(self) -> int:
        """Restrict stepping to the frames that want attention. Returns how many there are.

        A filter on where you go, never on what is saved: every row stays in its clip's list, so
        the frames you skip past are written back exactly as they were.
        """
        self.walk = [i for i, row in enumerate(self.rows) if row.suspect or (row.uncertain and not row.reviewed)]
        self.walk_pos = 0
        if self.walk:
            self.cursor = self.walk[0]
        return len(self.walk)

    # ---------------------------------------------------------------- geometry
    @property
    def scale(self) -> float:
        return self.base_scale * self.zoom

    @property
    def filter_name(self) -> str:
        """The view filter in force, by its name in FILTERS (h cycles it)."""
        return FILTERS[self.filter]

    def fit(self, client_w: int, client_h: int) -> bool:
        """Size the view to the window's client area, so enlarging the window gains detail instead
        of blur. True when the view's size changed."""
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
        return (self.view_w, self.view_h, self.hud_h) != old

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

    def centre_on(self, x: float, y: float) -> None:
        """Put frame point (x, y), kept on the frame, in the middle of the view."""
        self.center = (min(max(x, 0.0), self.width - 1.0), min(max(y, 0.0), self.height - 1.0))
        self.centre_view()

    # Frame coordinates put pixel k's centre at k + 0.5 (what the click mapping, GridTrackNet's grid
    # decode and training's encode all assume). View pixel u covers [u, u + 1), so a frame point maps
    # to the view pixel whose span holds it (floor, not round: round's ties go to even, which put a
    # ball dead centre a pixel off); render's warp is built on the same convention.
    def to_view(self, x: float, y: float) -> Tuple[int, int]:
        return (math.floor((x - self.origin[0]) * self.scale),
                math.floor((y - self.origin[1]) * self.scale))

    def to_frame(self, vx: float, vy: float) -> Tuple[float, float]:
        return self.origin[0] + vx / self.scale, self.origin[1] + vy / self.scale

    # ---------------------------------------------------------------- zoom
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

    # ---------------------------------------------------------------- where the ball is
    def guess_for(self, index: int) -> Optional[Point]:
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

    def target_point(self) -> Optional[Point]:
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
