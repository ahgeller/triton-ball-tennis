"""The label tool's grid view (TAB): a page of enlarged ball crops, all of one clip, each a label
frame about its ball, owning its own state, picture, clicks and keys. There is no single frame
under a parked mouse here, which is what sets it apart from the tool's own view.

It draws from the tool's rows, cursor and window geometry and hands edits back to it. Its keys
name the tool's actions (``tool.grid_page``, ``tool.confirm_grid``...): that is the one surface
the key table in the tests pins, and the tool hands the grid's actions straight back here.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import cv2
import numpy as np

from finetune.labeling.label_draw import (FILTERS, ZOOM_STEP, draw_ticks, draw_tile_mark, enhance, fit_text, hit, mark_plan)
from finetune.labeling.label_window import KEYS_CONFIRM, KEYS_ZOOM_IN, KEYS_ZOOM_OUT
from finetune.labeling.refine import Point, Suggestion

if TYPE_CHECKING:
    from finetune.labeling.label_tool import Tool

GRID_SIZES = (8, 16)

Tile = Tuple[int, Tuple[int, int, int, int], float, float, float]   # row index, box, ox, oy, scale


class GridView:
    """A page of GRID_SIZES tiles, each a label frame enlarged about its ball, all of one clip.

    Holds which rows are on the page, where each tile was drawn (so a click lands on the right
    frame), the page size and the tiles' zoom, and the last canvas with the state it was drawn
    from, so an unchanged page costs nothing to show again."""

    def __init__(self, tool: "Tool") -> None:
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
    def toggle(self) -> None:
        tool = self.tool
        tool.sheet_view.active = False
        self.active = not self.active
        if self.active:
            self.prepare(tool.cursor)
        tool.follow_ball()

    def prepare(self, start: int) -> None:
        """Fill the page from `start`: the next rows of the walk (or of every row) within the clip."""
        tool = self.tool
        candidates = tool.walk if tool.walk else list(range(len(tool.rows)))
        position = candidates.index(start) if start in candidates else 0
        clip = tool.rows[candidates[position]].clip
        self.indices = []
        for index in candidates[position:position + self.size]:
            if tool.rows[index].clip != clip:
                break
            self.indices.append(index)
        tool.cursor = self.indices[0]
        if tool.walk:
            tool.walk_pos = tool.walk.index(tool.cursor)
        self.centers = {i: (tool.rows[i].x, tool.rows[i].y) for i in self.indices}
        self.tiles = []
        self.cache_key = None
        tool.follow_ball()

    def page(self, direction: int) -> None:
        tool = self.tool
        candidates = tool.walk if tool.walk else list(range(len(tool.rows)))
        edge = self.indices[-1] if direction > 0 else self.indices[0]
        position = candidates.index(edge) + (1 if direction > 0 else -self.size)
        if position >= len(candidates):
            return
        self.prepare(candidates[max(0, position)])

    def select(self, slot: int) -> None:
        """Make the slot-th tile (0-based, clamped to the page) the one the keys act on."""
        if not self.indices:
            return
        tool = self.tool
        tool.cursor = self.indices[min(max(slot, 0), len(self.indices) - 1)]
        if tool.walk and tool.cursor in tool.walk:
            tool.walk_pos = tool.walk.index(tool.cursor)

    def step(self, direction: int) -> None:
        """Move the selection one tile along, turning the page at either end."""
        tool = self.tool
        slot = self.indices.index(tool.cursor) + direction if tool.cursor in self.indices else 0
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
            selected = self.tool.cursor
            self.prepare(self.indices[0])
            if selected in self.indices:
                self.select(self.indices.index(selected))
        self.tool.remember_settings()

    def confirm_page(self) -> None:
        """SPACE: confirm every tile on the page as drawn - only a page actually rendered, never a
        prefetched or unseen one - then turn the page."""
        if [tile[0] for tile in self.tiles] != self.indices:
            return
        tool = self.tool
        for index in self.indices:
            tool.cursor = index
            tool.confirm()
        self.page(1)

    # ---------------------------------------------------------------- the picture
    def render(self) -> np.ndarray:
        tool = self.tool
        rows = tool.rows
        key = (tool.view_w, tool.view_h, self.zoom, self.size, tool.cursor, tool.dirty, tool.hints.version,
               tuple((i, rows[i].x, rows[i].y, rows[i].visible, rows[i].reviewed, rows[i].edited) for i in self.indices))
        if key == self.cache_key:
            tool.render_fresh = False
            return self.cache
        tool.render_fresh = True
        canvas = np.zeros((tool.view_h + tool.hud_h, tool.view_w, 3), np.uint8)
        tiles: List[Tile] = []
        offers: Dict[int, tuple] = {}
        ui = tool.ui
        band = int(round(28 * ui))                       # title strip above each tile
        # A page is one clip (prepare stops at a clip's end) and the cursor is on it, so
        # tool.frame reads every tile from the right clip without touching the cursor.
        for slot, index in enumerate(self.indices):
            row = rows[index]
            rows_of = max(1, self.size // 4)
            left, right = slot % 4 * tool.view_w // 4, (slot % 4 + 1) * tool.view_w // 4
            top, bottom = slot // 4 * tool.view_h // rows_of, (slot // 4 + 1) * tool.view_h // rows_of
            y0 = top + band
            offer = tool.offer_for(index)
            if offer is not None:
                offers[index] = (offer["x"], offer["y"])
            tile, ox, oy, scale = self.tile(index, right - left, bottom - y0, offer)
            canvas[y0:bottom, left:right] = tile
            title = self.title(index, slot, offer)
            color = (70, 210, 70) if row.settled else (0, 210, 255)
            cv2.putText(canvas, fit_text(title, right - left - 12, .5 * ui, max(1, int(ui))),
                        (left + 6, top + int(round(20 * ui))), cv2.FONT_HERSHEY_SIMPLEX, .5 * ui,
                        color, max(1, int(ui)))
            cv2.rectangle(canvas, (left, top), (right-1, bottom-1), (255, 180, 50) if index == tool.cursor else (65, 65, 65), 2)
            tiles.append((index, (left, y0, right-1, bottom-1), ox, oy, scale))
        caption =(f"{tool.clip} | click=center  right click=full frame  SPACE=confirm these {len(tiles)}"
                   f"{' (orange crosses are hints on offer)' if offers else ''}  a/d=page  TAB=single")
        thin = max(1, int(ui))
        cv2.putText(canvas, fit_text(caption, tool.view_w - 12, .48 * ui, thin),
                    (6, tool.view_h + int(round(21 * ui))), cv2.FONT_HERSHEY_SIMPLEX, .48 * ui, (255, 255, 255), thin)
        keys = ("1-9 or ,/. select | y: take hint | x: keep draft | i/j/k/l: 1px (SHIFT 0.25px) | u: undo | "
                f"v: absent | +/- zoom | z: {GRID_SIZES[(GRID_SIZES.index(self.size) + 1) % len(GRID_SIZES)]} tiles | q: save/quit")
        cv2.putText(canvas, fit_text(keys, tool.view_w - 12, .42 * ui, thin),
                    (6, tool.view_h + int(round(44 * ui))), cv2.FONT_HERSHEY_SIMPLEX, .42 * ui, (200, 200, 200), thin)
        self.tiles = tiles
        tool.shown_offers = offers
        self.cache_key, self.cache = key, canvas
        return canvas

    def tile(self, index: int, width: int, height: int, offer: Optional[Suggestion]) -> Tuple[np.ndarray, float, float, float]:
        """One tile of `width` x `height`: the row's frame about its ball (the whole frame when
        there is none), filtered and marked. (the picture, its frame origin x and y, its scale)."""
        tool = self.tool
        row = tool.rows[index]
        ui = tool.ui
        image = tool.frame(row.frame)
        scale = width / 96.0 * self.zoom if row.visible else min(width / tool.width, height / tool.height)
        cx, cy = self.centers[index] if row.visible else (tool.width / 2, tool.height / 2)
        ox, oy = cx - width / (2 * scale), cy - height / (2 * scale)
        # Pixel centres sit at k + 0.5 in frame coordinates, as in the single-frame view (see Tool.to_view).
        matrix = np.array([[scale, 0, scale * (0.5 - ox) - 0.5],
                           [0, scale, scale * (0.5 - oy) - 0.5]], np.float64)
        tile = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_NEAREST)
        if FILTERS[tool.filter] not in ("off", "motion"):    # motion needs the frame before
            tile = enhance(tile, FILTERS[tool.filter])
        to_tile = lambda point: (math.floor((point[0] - ox) * scale), math.floor((point[1] - oy) * scale))
        plan = mark_plan(row, tool.suggestion_for(index), offer, (255, 255, 255))
        for point, colour in plan.ticks:
            draw_ticks(tile, *to_tile(point), colour, ui)
        if plan.mark is not None:
            draw_tile_mark(tile, *to_tile(plan.mark), plan.colour, ui)
        return tile, ox, oy, scale

    def title(self, index: int, slot: int, offer: Optional[Suggestion]) -> str:
        """A tile's strip: its slot number (the key that selects it), frame, state and hint."""
        row = self.tool.rows[index]
        title = (f"{slot + 1}  " if slot < 9 else "") + \
            f"f{row.frame}  {'EDITED' if row.edited else 'OK' if row.reviewed else 'CHECK'}"
        if not row.visible:
            title += "  absent"
        if offer is not None:
            title += "  HINT sure" if offer["trusted"] else "  HINT unsure"
        else:
            verdict = self.tool.hint_verdict(index)
            if verdict is not None:
                title += "  hint ok" if verdict == "agrees" else "  hint differs"
        return title

    # ---------------------------------------------------------------- the hand
    def on_mouse(self, event: int, x: int, y: int) -> None:
        if event not in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_LBUTTONDBLCLK, cv2.EVENT_RBUTTONDOWN):
            return
        tool = self.tool
        for index, box, ox, oy, scale in self.tiles:
            if hit(box, x, y):
                tool.cursor = index
                if tool.walk:
                    tool.walk_pos = tool.walk.index(index)
                if event == cv2.EVENT_RBUTTONDOWN:
                    self.active = False
                    tool.follow_ball()
                else:                 # +0.5: highgui floors the pointer to a pixel; aim at its middle
                    fx, fy = ox + (x + 0.5 - box[0]) / scale, oy + (y + 0.5 - box[1]) / scale
                    if 0 <= fx < tool.width and 0 <= fy < tool.height:
                        was_visible = tool.rows[index].visible
                        if tool.set_point(fx, fy) and not was_visible:
                            # The tile was showing the whole frame; zoom it on the ball just placed.
                            self.centers[index] = (tool.rows[index].x, tool.rows[index].y)
                return

    def handle_key(self, code: int, direction: int) -> bool:
        """What a key does on a grid page; False for a key the grid reads like the single frame."""
        tool = self.tool
        if self.edit_key(code):
            return True
        if code in KEYS_CONFIRM:
            tool.confirm_grid()
        elif direction:
            tool.grid_page(direction)
        elif code in (ord(","), ord(".")):
            tool.grid_step(-1 if code == ord(",") else 1)
        elif ord("1") <= code <= ord("9"):
            tool.grid_select(code - ord("1"))
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
                tool.grid_step(1)
        elif code == ord("x"):
            tool.confirm(take_offer=False)    # the draft is right, not the hint on offer
            tool.grid_step(1)
        elif code == ord("v"):
            tool.mark_invisible()
        elif code == ord("e"):
            if tool.seek_disputed():
                tool.prepare_grid(tool.cursor)   # bring the page along, not just the cursor
        elif code in (ord("f"), ord("b")):
            tool.jump(1 if code == ord("f") else -1)
            tool.prepare_grid(tool.cursor)
        else:
            return False
        return True
