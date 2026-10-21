"""The label tool's grid view (TAB): a page of enlarged ball crops, all of one clip, each a label
frame about its ball, owning its own state, picture, clicks and keys. There is no single frame
under a parked mouse here, which is what sets it apart from the tool's own view.

It reads the session's rows, cursor and view geometry and moves the cursor through the session;
the tool it asks for the edits (confirm, set_point, take_suggestion...) and to follow the ball.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import cv2
import numpy as np

from finetune.labeling.label_draw import (FILTERS, ZOOM_STEP, draw_ticks, draw_tile_mark, enhance, fit_text, hit, mark_plan)
from finetune.labeling.label_window import KEYS_CONFIRM, KEYS_ZOOM_IN, KEYS_ZOOM_OUT
from finetune.refiner.refine import Point, Suggestion

if TYPE_CHECKING:
    from finetune.labeling.label_session import LabelSession
    from finetune.labeling.label_tool import Tool

GRID_SIZES = (8, 16)

Tile = Tuple[int, Tuple[int, int, int, int], float, float, float]   # row index, box, ox, oy, scale


class GridView:
    """A page of GRID_SIZES tiles, each a label frame enlarged about its ball, all of one clip.

    Holds which rows are on the page, where each tile was drawn (so a click lands on the right
    frame), the page size and the tiles' zoom, and the last canvas with the state it was drawn
    from, so an unchanged page costs nothing to show again."""

    def __init__(self, session: "LabelSession", tool: "Tool") -> None:
        self.session = session
        self.tool = tool
        self.active = False
        self.indices: List[int] = []            # the rows on this page, in order
        self.centers: Dict[int, Point] = {}     # row index -> the frame point its tile is centred on
        self.tiles: List[Tile] = []             # what the last render drew, for clicks and confirm_page
        self.cache: Optional[np.ndarray] = None
        self.cache_key: Optional[tuple] = None
        self.zoom = 1.0
        self.size = GRID_SIZES[0]

    def invalidate(self) -> None:
        """The next render draws afresh (the picture filter changed)."""
        self.cache_key = None

    # ---------------------------------------------------------------- which rows
    def candidates(self) -> List[int]:
        """The rows a page is filled from: the walk when there is one, else every row."""
        return self.session.walk if self.session.walk else list(range(len(self.session.rows)))

    def toggle(self) -> None:
        self.tool.sheet_view.active = False
        self.active = not self.active
        if self.active:
            self.prepare(self.session.cursor)
        self.tool.follow_ball()

    def prepare(self, start: int) -> None:
        """Fill the page from `start`: the next rows of the walk (or of every row) within the clip."""
        session = self.session
        candidates = self.candidates()
        position = candidates.index(start) if start in candidates else 0
        clip = session.rows[candidates[position]].clip
        self.indices = []
        for index in candidates[position:position + self.size]:
            if session.rows[index].clip != clip:
                break
            self.indices.append(index)
        session.set_cursor(self.indices[0])
        self.centers = {i: (session.rows[i].x, session.rows[i].y) for i in self.indices}
        self.tiles = []
        self.cache_key = None
        self.tool.follow_ball()

    def page(self, direction: int) -> None:
        candidates = self.candidates()
        edge = self.indices[-1] if direction > 0 else self.indices[0]
        position = candidates.index(edge) + (1 if direction > 0 else -self.size)
        if position >= len(candidates):
            return
        self.prepare(candidates[max(0, position)])

    def select(self, slot: int) -> None:
        """Make the slot-th tile (0-based, clamped to the page) the one the keys act on."""
        if self.indices:
            self.session.set_cursor(self.indices[min(max(slot, 0), len(self.indices) - 1)])

    def step(self, direction: int) -> None:
        """Move the selection one tile along, turning the page at either end."""
        cursor = self.session.cursor
        slot = self.indices.index(cursor) + direction if cursor in self.indices else 0
        if 0 <= slot < len(self.indices):
            self.select(slot)
            return
        before = self.indices[0]
        self.page(direction)
        if self.indices[0] != before and direction < 0:
            self.select(len(self.indices) - 1)   # came back a page: land on its last tile

    def toggle_size(self) -> None:
        self.size = GRID_SIZES[(GRID_SIZES.index(self.size) + 1) % len(GRID_SIZES)] \
            if self.size in GRID_SIZES else GRID_SIZES[0]
        if self.active:
            selected = self.session.cursor
            self.prepare(self.indices[0])
            if selected in self.indices:
                self.select(self.indices.index(selected))
        self.tool.remember_settings()

    def confirm_page(self) -> None:
        """SPACE: confirm every tile on the page as drawn - only a page actually rendered, never a
        prefetched or unseen one - then turn the page."""
        if [tile[0] for tile in self.tiles] != self.indices:
            return
        for index in self.indices:
            self.session.set_cursor(index)
            self.tool.confirm()
        self.page(1)

    # ---------------------------------------------------------------- the picture
    def render(self) -> np.ndarray:
        session = self.session
        rows = session.rows
        key = (session.view_w, session.view_h, self.zoom, self.size, session.cursor, session.dirty, session.hints.version,
               tuple((i, rows[i].x, rows[i].y, rows[i].visible, rows[i].reviewed, rows[i].edited) for i in self.indices))
        if key == self.cache_key:
            return self.cache
        canvas = np.zeros((session.view_h + session.hud_h, session.view_w, 3), np.uint8)
        tiles: List[Tile] = []
        offers: Dict[int, Point] = {}
        ui = session.ui
        band = int(round(28 * ui))                       # title strip above each tile
        # A page is one clip (prepare stops at a clip's end) and the cursor is on it, so
        # session.source.frame reads every tile from the right clip without touching the cursor.
        for slot, index in enumerate(self.indices):
            row = rows[index]
            rows_of = max(1, self.size // 4)
            left, right = slot % 4 * session.view_w // 4, (slot % 4 + 1) * session.view_w // 4
            top, bottom = slot // 4 * session.view_h // rows_of, (slot // 4 + 1) * session.view_h // rows_of
            y0 = top + band
            offer = session.offer_for(index)
            if offer is not None:
                offers[index] = (offer["x"], offer["y"])
            tile, ox, oy, scale = self.tile(index, right - left, bottom - y0, offer)
            canvas[y0:bottom, left:right] = tile
            title = self.title(index, slot, offer)
            color = (70, 210, 70) if row.reviewed else (0, 210, 255)
            cv2.putText(canvas, fit_text(title, right - left - 12, .5 * ui, max(1, int(ui))),
                        (left + 6, top + int(round(20 * ui))), cv2.FONT_HERSHEY_SIMPLEX, .5 * ui,
                        color, max(1, int(ui)))
            cv2.rectangle(canvas, (left, top), (right-1, bottom-1), (255, 180, 50) if index == session.cursor else (65, 65, 65), 2)
            tiles.append((index, (left, y0, right-1, bottom-1), ox, oy, scale))
        caption =(f"{session.source.clip} | click=center  right click=full frame  SPACE=confirm these {len(tiles)}"
                   f"{' (orange crosses are hints on offer)' if offers else ''}  a/d=page  TAB=single")
        thin = max(1, int(ui))
        cv2.putText(canvas, fit_text(caption, session.view_w - 12, .48 * ui, thin),
                    (6, session.view_h + int(round(21 * ui))), cv2.FONT_HERSHEY_SIMPLEX, .48 * ui, (255, 255, 255), thin)
        keys = ("1-9 or ,/. select | y: take hint | x: keep draft | i/j/k/l: 1px (SHIFT 0.25px) | u: undo | "
                f"v: absent | +/- zoom | z: {GRID_SIZES[(GRID_SIZES.index(self.size) + 1) % len(GRID_SIZES)]} tiles | q: save/quit")
        cv2.putText(canvas, fit_text(keys, session.view_w - 12, .42 * ui, thin),
                    (6, session.view_h + int(round(44 * ui))), cv2.FONT_HERSHEY_SIMPLEX, .42 * ui, (200, 200, 200), thin)
        self.tiles = tiles
        session.shown_offers = offers
        self.cache_key, self.cache = key, canvas
        return canvas

    def tile(self, index: int, width: int, height: int, offer: Optional[Suggestion]) -> Tuple[np.ndarray, float, float, float]:
        """One tile of `width` x `height`: the row's frame about its ball (the whole frame when
        there is none), filtered and marked. (the picture, its frame origin x and y, its scale)."""
        session = self.session
        row = session.rows[index]
        ui = session.ui
        image = session.source.frame(row.frame)
        scale = width / 96.0 * self.zoom if row.visible else min(width / session.width, height / session.height)
        cx, cy = self.centers[index] if row.visible else (session.width / 2, session.height / 2)
        ox, oy = cx - width / (2 * scale), cy - height / (2 * scale)
        # Pixel centres sit at k + 0.5 in frame coordinates, as in the single-frame view (see LabelSession.to_view).
        matrix = np.array([[scale, 0, scale * (0.5 - ox) - 0.5],
                           [0, scale, scale * (0.5 - oy) - 0.5]], np.float64)
        tile = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_NEAREST)
        if FILTERS[session.filter] not in ("off", "motion"):    # motion needs the frame before
            tile = enhance(tile, FILTERS[session.filter])
        to_tile = lambda point: (math.floor((point[0] - ox) * scale), math.floor((point[1] - oy) * scale))
        plan = mark_plan(row, session.suggestion_for(index), offer, (255, 255, 255))
        for point, colour in plan.ticks:
            draw_ticks(tile, *to_tile(point), colour, ui)
        if plan.mark is not None:
            draw_tile_mark(tile, *to_tile(plan.mark), plan.colour, ui)
        return tile, ox, oy, scale

    def title(self, index: int, slot: int, offer: Optional[Suggestion]) -> str:
        """A tile's strip: its slot number (the key that selects it), frame, state and hint."""
        row = self.session.rows[index]
        title = (f"{slot + 1}  " if slot < 9 else "") + \
            f"f{row.frame}  {'EDITED' if row.edited else 'OK' if row.reviewed else 'CHECK'}"
        if not row.visible:
            title += "  absent"
        if offer is not None:
            title += "  HINT sure" if offer["trusted"] else "  HINT unsure"
        else:
            verdict = self.session.hint_verdict(index)
            if verdict is not None:
                title += "  hint ok" if verdict == "agrees" else "  hint differs"
        return title

    # ---------------------------------------------------------------- the hand
    def on_mouse(self, event: int, x: int, y: int) -> None:
        if event not in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_LBUTTONDBLCLK, cv2.EVENT_RBUTTONDOWN):
            return
        session = self.session
        for index, box, ox, oy, scale in self.tiles:
            if hit(box, x, y):
                session.set_cursor(index)
                if event == cv2.EVENT_RBUTTONDOWN:
                    self.active = False
                    self.tool.follow_ball()
                else:                 # +0.5: highgui floors the pointer to a pixel; aim at its middle
                    fx, fy = ox + (x + 0.5 - box[0]) / scale, oy + (y + 0.5 - box[1]) / scale
                    if 0 <= fx < session.width and 0 <= fy < session.height:
                        was_visible = session.rows[index].visible
                        if self.tool.set_point(fx, fy) and not was_visible:
                            # The tile was showing the whole frame; zoom it on the ball just placed.
                            self.centers[index] = (session.rows[index].x, session.rows[index].y)
                return

    def handle_key(self, code: int, direction: int) -> bool:
        """What a key does on a grid page; False for a key the grid reads like the single frame."""
        if self.edit_key(code):
            return True
        if code in KEYS_CONFIRM:
            self.confirm_page()
        elif direction:
            self.page(direction)
        elif code in (ord(","), ord(".")):
            self.step(-1 if code == ord(",") else 1)
        elif ord("1") <= code <= ord("9"):
            self.select(code - ord("1"))
        elif code in KEYS_ZOOM_IN + KEYS_ZOOM_OUT:
            factor = ZOOM_STEP if code in KEYS_ZOOM_IN else 1 / ZOOM_STEP
            self.zoom = min(4., max(.5, self.zoom * factor))
        elif code != ord("r"):            # r: only the displayed page can be batch-confirmed (SPACE)
            return False
        return True

    def edit_key(self, code: int) -> bool:
        """The grid keys that say something about the selected tile's label, or go to a frame
        worth a look: y, x, v, e, f and b. False for any other key."""
        tool = self.tool
        if code == ord("y"):
            if tool.take_suggestion():
                self.step(1)
        elif code == ord("x"):
            tool.confirm(take_offer=False)    # the draft is right, not the hint on offer
            self.step(1)
        elif code == ord("v"):
            tool.mark_invisible()
        elif code == ord("e"):
            if tool.seek_disputed():
                self.prepare(self.session.cursor)   # bring the page along, not just the cursor
        elif code in (ord("f"), ord("b")):
            tool.jump(1 if code == ord("f") else -1)
            self.prepare(self.session.cursor)
        else:
            return False
        return True
