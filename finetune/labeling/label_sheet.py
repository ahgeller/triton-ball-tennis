"""The label tool's trajectory sheet (w): the whole frame, dimmed, with every label of a stretch
of label frames drawn on it and joined in order, so a label off the ball's flight is plain at a
glance. It only finds frames: a click opens one, and nothing here edits a label. Like the grid
(``label_grid``) it owns its state, picture, clicks and keys, and names the tool's actions.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, List, Optional, Tuple

import cv2
import numpy as np

from finetune.labeling.label_draw import HINT_TRUSTED, HINT_UNSURE, fit_text
from finetune.labeling.label_store import Row
from finetune.labeling.label_window import KEY_TAB

if TYPE_CHECKING:
    from finetune.labeling.label_tool import Tool

SHEET_ROWS = 60           # label frames on one trajectory sheet (w)


class SheetView:
    """The whole frame, dimmed, with every label of a stretch of SHEET_ROWS label frames drawn on
    it, joined in order, and the suggestions as small orange dots. The labels trace the ball's
    flight, so one that is off it - or a run of them following a player - is plain at a glance,
    and the orange path says where the video itself puts the ball. It only finds frames: a click
    on a point opens that frame, and nothing here edits a label.

    A motion trace was tried as the background and dropped: the players' movement swamped it,
    and the ball's streak is a pixel or two wide at this scale, under the labels anyway."""

    def __init__(self, tool: "Tool") -> None:
        self.tool = tool
        self.active = False
        self.start = 0                          # the first row on the sheet
        self.points: List[Tuple[int, int, int]] = []   # (row index, view x, view y) of each label drawn
        self.cache_key: Optional[tuple] = None
        self.cache: Optional[np.ndarray] = None

    def toggle(self) -> None:
        tool = self.tool
        self.active = not self.active
        if self.active:
            tool.grid_view.active = False
            first, end = tool.clip_bounds(tool.cursor)
            self.start = max(first, min(tool.cursor - SHEET_ROWS // 2, end - SHEET_ROWS))
            self.cache_key = None
        tool.follow_ball()

    def page(self, direction: int) -> None:
        tool = self.tool
        first, end = tool.clip_bounds(self.start)
        start = max(first, min(self.start + direction * SHEET_ROWS, end - SHEET_ROWS))
        if start != self.start:
            self.start = start
            tool.cursor = start       # leaving the sheet lands in the stretch it shows
            tool.follow_ball()

    def on_mouse(self, event: int, x: int, y: int) -> None:
        tool = self.tool
        if event == cv2.EVENT_RBUTTONDOWN:
            self.toggle()
        elif event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_LBUTTONDBLCLK) and self.points:
            index, px, py = min(self.points, key=lambda p: (p[1] - x) ** 2 + (p[2] - y) ** 2)
            if math.hypot(px - x, py - y) <= 14 * tool.ui:
                tool.cursor = index
                if tool.walk and index in tool.walk:
                    tool.walk_pos = tool.walk.index(index)
                self.active = False
                tool.follow_ball()

    def handle_key(self, code: int, direction: int) -> None:
        """The trajectory sheet only finds frames: its own few keys, nothing else."""
        tool = self.tool
        if code in (ord("w"), KEY_TAB):
            tool.toggle_sheet()
            if code == KEY_TAB:
                tool.toggle_grid()
        elif direction:
            tool.sheet_page(direction)

    def render(self) -> np.ndarray:
        tool = self.tool
        rows = tool.rows
        end = tool.clip_bounds(self.start)[1]
        indices = list(range(self.start, min(self.start + SHEET_ROWS, end)))
        key = (tuple(indices), Row.revision, tool.hints.version, tool.view_w, tool.view_h, tool.hud_h,
               tool.cursor, tool.ui)
        if key == self.cache_key:
            tool.render_fresh = False
            return self.cache
        tool.render_fresh = True
        selected = tool.cursor
        base = tool.frame(rows[selected].frame)
        ui, thin = tool.ui, max(1, int(round(tool.ui)))
        scale = min(tool.view_w / tool.width, tool.view_h / tool.height)
        shown_w, shown_h = max(1, int(round(tool.width * scale))), max(1, int(round(tool.height * scale)))
        scale = shown_w / tool.width                     # what resize really does, for the points
        off_x, off_y = (tool.view_w - shown_w) // 2, (tool.view_h - shown_h) // 2
        picture = cv2.convertScaleAbs(base, alpha=0.45)
        canvas = np.zeros((tool.view_h + tool.hud_h, tool.view_w, 3), np.uint8)
        canvas[off_y:off_y + shown_h, off_x:off_x + shown_w] = cv2.resize(
            picture, (shown_w, shown_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        display = canvas[:tool.view_h]
        to_sheet = lambda x, y: (math.floor(x * scale + off_x), math.floor(y * scale + off_y))
        for index in indices:                            # the suggestions first, under the labels
            hint = tool.suggestion_for(index)
            if hint is not None:
                cv2.circle(display, to_sheet(hint["x"], hint["y"]), max(2, int(round(2 * ui))),
                           HINT_TRUSTED if hint["trusted"] else HINT_UNSURE, -1, cv2.LINE_AA)
        self.points = self.trace(display, indices, to_sheet)
        differs = self.draw_points(display, selected)
        shown = [rows[i] for i in indices]
        line1 = (f"{rows[indices[0]].clip}  TRAJECTORY  frames {shown[0].frame}-{shown[-1].frame}  "
                 f"{sum(r.visible for r in shown)} with a ball  reviewed {sum(r.reviewed for r in shown)}/{len(shown)}"
                 + (f"  hint differs {differs} (red)" if differs else ""))
        line2 = "click a point: open that frame   a/d: stretch before/after   w or right click: back   q: save/quit"
        hud = canvas[tool.view_h:]
        cv2.putText(hud, fit_text(line1, tool.view_w - int(16 * ui), 0.55 * ui, thin), (int(8 * ui), int(22 * ui)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55 * ui, (255, 255, 255), thin, cv2.LINE_AA)
        cv2.putText(hud, fit_text(line2, tool.view_w - int(16 * ui), 0.48 * ui, thin), (int(8 * ui), int(46 * ui)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48 * ui, (180, 180, 180), thin, cv2.LINE_AA)
        self.cache_key, self.cache = key, canvas
        return canvas

    def trace(self, display: np.ndarray, indices: List[int], to_sheet) -> List[Tuple[int, int, int]]:
        """Join the labels of `indices` in order with a grey line, broken at every frame with no
        ball; (row index, view x, view y) of each label drawn."""
        points, last = [], None
        thin = max(1, int(round(self.tool.ui)))
        for index in indices:
            row = self.tool.rows[index]
            if not row.visible:
                last = None
                continue
            point = to_sheet(row.x, row.y)
            if last is not None:
                cv2.line(display, last, point, (200, 200, 200), thin, cv2.LINE_AA)
            last = point
            points.append((index,) + point)
        return points

    def draw_points(self, display: np.ndarray, selected: int) -> int:
        """A dot per label on the trace (green reviewed, yellow not), ringed red where a trusted
        suggestion differs, white on the selected frame, every tenth captioned with its frame
        number; how many differ."""
        tool = self.tool
        ui, thin = tool.ui, max(1, int(round(tool.ui)))
        differs = 0
        for index, px, py in self.points:
            row = tool.rows[index]
            colour = (70, 210, 70) if row.reviewed else (0, 220, 255)
            cv2.circle(display, (px, py), max(2, int(round(3 * ui))), colour, -1, cv2.LINE_AA)
            if tool.hint_verdict(index) == "differs":
                differs += 1
                cv2.circle(display, (px, py), int(round(8 * ui)), (0, 0, 255), thin, cv2.LINE_AA)
            if index == selected:
                cv2.circle(display, (px, py), int(round(12 * ui)), (255, 255, 255), thin, cv2.LINE_AA)
            if (index - self.start) % 10 == 0:
                cv2.putText(display, str(row.frame), (px + int(6 * ui), py - int(6 * ui)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42 * ui, (255, 255, 255), thin, cv2.LINE_AA)
        return differs
