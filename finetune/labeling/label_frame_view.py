"""The label tool's single-frame view: the frame warped to the view's zoom and origin with the
trail, the mark and the overview inset drawn on it, and the HUD bar under it with its two text
lines and clickable chips.

``FrameView`` draws from the session's rows, cursor, view geometry and suggestions (the tool's
``render`` decides when a redraw is due, see ``Tool.render_state``) and asks the tool for the
toggles a chip click names. The chip and inset boxes it records while drawing are what a click on
the bar or the inset is matched against.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import cv2
import numpy as np

from finetune.labeling.label_draw import (FILTERS, GHOST, LANDING, ChipRow, draw_badge, draw_ghost, draw_mark, draw_ticks,
                                          enhance, fit_text, hit, label_mark_colour, mark_plan, review_badge)
from finetune.labeling.label_store import Row
from finetune.refiner.refine import Suggestion
from finetune.data.detector_metrics import REFERENCE_HEIGHT

if TYPE_CHECKING:
    from finetune.labeling.label_session import LabelSession
    from finetune.labeling.label_tool import Tool

INSET_SHARE = 0.22        # width of the overview inset as a share of the view
THUMBNAILS_KEPT = 64      # small copies of frames kept for the inset


class FrameView:
    """The single-frame view's picture and HUD, drawn for the tool's current frame."""

    def __init__(self, session: "LabelSession", tool: "Tool") -> None:
        self.session = session
        self.tool = tool
        self.inset_box = None         # where the inset is on screen, so a click in it can re-centre there
        self.inset_scale = 1.0        # frame px -> inset px
        self.centre_button = None     # where the HUD chip that toggles auto-centring is, in display pixels
        self.step_buttons = []        # ... and the x1..x8 ones, as (box, frames) pairs
        self.filter_button = None     # ... and the filter and inset chips
        self.inset_button = None
        self.thumbs: Dict[tuple, np.ndarray] = {}   # (clip, frame, width) -> small copy, for the inset

    # ---------------------------------------------------------------- the picture
    def draw(self, image: np.ndarray, before: Optional[np.ndarray], mode: str, row: Row,
             hint: Optional[Suggestion], offer: Optional[Suggestion]) -> np.ndarray:
        """A fresh canvas: the warped frame with its marks, the inset, the badges and the HUD."""
        canvas = self.warp_picture(image, before, mode)
        display = canvas[:self.session.view_h]
        self.draw_marks(display, row, hint, offer)
        self.draw_inset(display, image, row, self.session.ui)
        self.draw_badges(display, row)
        self.draw_hud(canvas, row, hint, offer)
        return canvas

    def warp_picture(self, image: np.ndarray, before: Optional[np.ndarray], mode: str) -> np.ndarray:
        """A fresh canvas for picture and HUD with the frame drawn into its top at the view's
        scale and origin, filtered by `mode` (the motion filter diffs against `before`)."""
        session = self.session
        x0, y0 = session.origin
        # Resize with exact sub-pixel origin handling via warpAffine. warpAffine puts pixel centres on
        # whole numbers while frame coordinates put them at k + 0.5 (see Tool.to_view): the half-pixel
        # terms line the drawn picture up with where a click is stored.
        s = session.scale
        matrix = np.array([[s, 0.0, s * (0.5 - x0) - 0.5], [0.0, s, s * (0.5 - y0) - 0.5]], dtype=np.float64)
        interpolation = cv2.INTER_LINEAR if session.zoom < 8 else cv2.INTER_NEAREST
        # One canvas for picture and HUD, the picture warped straight into its top: building them
        # apart and stacking them copied the whole view once more per redraw. A fresh canvas each
        # time, so an image handed out earlier (the cache, a caller's reference) is never drawn over.
        canvas = np.empty((session.view_h + session.hud_h, session.view_w, 3), np.uint8)
        display = canvas[:session.view_h]
        cv2.warpAffine(image, matrix, (session.view_w, session.view_h), dst=display, flags=interpolation)
        if mode != "off":
            previous = None
            if before is not None:
                previous = cv2.warpAffine(before, matrix, (session.view_w, session.view_h), flags=interpolation)
            display[:] = enhance(display, mode, previous)
        return canvas

    def draw_marks(self, display: np.ndarray, row: Row, hint: Optional[Suggestion], offer: Optional[Suggestion]) -> None:
        """The trail, the mark (the suggestion on offer or the label, with the other ticked beside
        it) or, on a frame with no ball and no offer, the dashed guess."""
        session = self.session
        ui = session.ui
        thin = max(1, int(round(ui)))
        if session.trail:
            for previous in session.rows[max(0, session.cursor - 12):session.cursor]:
                if previous.visible and previous.clip == row.clip:
                    cv2.circle(display, session.to_view(previous.x, previous.y), max(3, int(3 * ui)),
                               (200, 200, 60), thin, cv2.LINE_AA)
        radius = int(max(8, 6 * session.zoom) * ui)
        # The suggestion on offer is the mark SPACE takes, drawn as the label would be, in orange -
        # bright when it is sure, dull when only a guess; the draft it replaces is ticked in grey.
        plan = mark_plan(row, hint, offer, label_mark_colour(row))
        for point, colour in plan.ticks:
            draw_ticks(display, *session.to_view(*point), colour, ui)
        if plan.mark is not None:
            draw_mark(display, *session.to_view(*plan.mark), radius, plan.colour, ui)
        elif session.ghost is not None:
            draw_ghost(display, *session.to_view(*session.ghost), radius, ui)

    def thumbnail(self, frame: int, image: np.ndarray, width: int, height: int) -> np.ndarray:
        key = (self.session.source.clip, frame, width)
        thumb = self.thumbs.get(key)
        if thumb is None:
            thumb = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            if len(self.thumbs) >= THUMBNAILS_KEPT:
                self.thumbs.pop(next(iter(self.thumbs)))
            self.thumbs[key] = thumb
        return thumb

    def draw_inset(self, display: np.ndarray, image: np.ndarray, row: Row, ui: float) -> None:
        """A small copy of the whole frame in the bottom-right corner while zoomed in, with the part on
        screen boxed and the ball dotted. When the detector's guess is nowhere near, this is where the
        real ball is found without zooming out: a click on it brings that spot into the zoomed view."""
        session = self.session
        self.inset_box = None
        if not session.inset or session.zoom < 1.5:
            return
        width = int(session.view_w * INSET_SHARE)
        height = max(1, int(round(width * session.height / session.width)))
        margin = int(10 * ui)
        left, top = session.view_w - width - margin, session.view_h - height - margin
        if left < 0 or top < 0:
            return
        panel = self.thumbnail(row.frame, image, width, height).copy()
        scale = width / session.width
        self.inset_scale = scale
        x0, y0 = session.origin
        x1, y1 = x0 + session.view_w / session.scale, y0 + session.view_h / session.scale
        cv2.rectangle(panel, (int(x0 * scale), int(y0 * scale)), (int(x1 * scale) - 1, int(y1 * scale) - 1),
                      (255, 255, 255), 1)
        point = session.target_point()
        if point is not None:
            cv2.circle(panel, (int(point[0] * scale), int(point[1] * scale)), max(2, int(2 * ui)),
                       (0, 255, 255) if row.visible else GHOST, -1, cv2.LINE_AA)
        display[top:top + height, left:left + width] = panel
        cv2.rectangle(display, (left - 1, top - 1), (left + width, top + height), (110, 110, 110), 1)
        self.inset_box = (left, top, left + width - 1, top + height - 1)

    def draw_badges(self, display: np.ndarray, row: Row) -> None:
        """The row's review badge at the top right, the LANDING badge to its left on a frame marked
        as one (/), and the frames-left counter at the top left."""
        session = self.session
        edge = session.view_w
        badge = review_badge(row)
        if badge is not None:
            edge = draw_badge(display, badge[0], badge[1], (0, 0, 0), session.ui, right_edge=edge)[0]
        if row.frame in session.source.landings:
            draw_badge(display, "LANDING", LANDING, (0, 0, 0), session.ui, right_edge=edge)
        remaining = len(session.rows) - sum(r.reviewed for r in session.rows)
        counter = "ALL DONE" if remaining == 0 else f"{remaining} LEFT"
        fill, text_colour = ((70, 190, 70), (0, 0, 0)) if remaining == 0 else ((50, 50, 50), (255, 255, 255))
        draw_badge(display, counter, fill, text_colour, session.ui)

    # ---------------------------------------------------------------- the HUD
    def hud_place(self) -> str:
        """Where in the clip, the run and the uncertain walk the cursor is."""
        session = self.session
        source = session.source
        within = f"{session.cursor - source.start + 1}/{len(source.rows)}"
        place = (f"{source.clip} [{within}]" if len(session.sources) == 1 else
                 f"{source.clip} [{within}]  clip {session.sources.index(source) + 1}/{len(session.sources)}"
                 f"  run {session.cursor + 1}/{len(session.rows)}")
        if session.walk:
            left = sum(1 for i in session.walk if not session.rows[i].reviewed)
            place += f"  UNCERTAIN ONLY {session.walk_pos + 1}/{len(session.walk)} ({left} left)"
        return place

    def hud_state(self, row: Row, hint: Optional[Suggestion], offer: Optional[Suggestion]) -> str:
        """The label, and what the suggestion says about it."""
        session = self.session
        state = "NOT VISIBLE" if not row.visible else f"({row.x:.1f}, {row.y:.1f})"
        if offer is not None:
            sure = "sure" if offer["trusted"] else "UNSURE, look closely"
            if row.visible:
                gap = math.hypot(offer["x"] - row.x, offer["y"] - row.y) * REFERENCE_HEIGHT / session.height
                return f"HINT ({offer['x']:.1f}, {offer['y']:.1f}) {sure}  grey draft {gap:.1f}px off"
            return f"HINT ({offer['x']:.1f}, {offer['y']:.1f}) {sure}  draft: no ball"
        if hint is not None and not row.visible:
            return state + "  hint: the video shows a ball here (y takes it)"
        if hint is not None:
            gap = math.hypot(hint["x"] - row.x, hint["y"] - row.y) * REFERENCE_HEIGHT / session.height
            state += f"  hint {gap:.1f}px" if gap >= 0.05 else "  hint: here"
            verdict = session.hint_verdict(session.cursor)
            if verdict is not None:
                state += "  hint ok" if verdict == "agrees" else "  hint differs"
            else:
                state += " (unsure)"
        return state

    def hud_lines(self, row: Row, hint: Optional[Suggestion], offer: Optional[Suggestion]) -> Tuple[str, str]:
        """The two text lines of the HUD: this frame (its landing mark and the clip's landing count
        among its state), and the counts with the keys that matter."""
        session = self.session
        landings = session.source.landings
        tag = "EDITED" if row.edited else "ok" if row.reviewed else "unreviewed"
        if row.frame in landings:
            tag += "  LANDING"
        line1 = (f"{self.hud_place()}  frame {row.frame}  {self.hud_state(row, hint, offer)}  "
                 f"from {row.source} {row.conf:.2f}  {tag}  zoom x{session.zoom:.1f}  landings {len(landings)}"
                 f"{'  *unsaved*' if session.dirty else ''}")
        keys = ("SPACE=take hint  x=keep draft  v=invisible" if offer is not None else
                "SPACE=good  y=take hint  v=invisible")
        return line1, f"{self.hud_counts()}{keys}  /=landing  r=accept run  f=uncertain  q=quit"

    def hud_counts(self) -> str:
        """How much of the run is reviewed, uncertain, flagged and disputed, as the HUD says it."""
        rows = self.session.rows
        reviewed = sum(r.reviewed for r in rows)
        uncertain_left = sum(1 for r in rows if r.uncertain and not r.reviewed)
        flagged_left = sum(1 for r in rows if r.suspect)
        disputed = self.session.disputed_count()
        return (f"reviewed {reviewed}/{len(rows)}  uncertain left {uncertain_left}  "
                f"{('flagged left ' + str(flagged_left) + '   ') if flagged_left else '  '}"
                f"{('hint differs ' + str(disputed) + ' (e)   ') if disputed else ''}")

    def draw_hud(self, canvas: np.ndarray, row: Row, hint: Optional[Suggestion], offer: Optional[Suggestion]) -> None:
        """The two text lines and the clickable chips along the bottom of the single-frame view."""
        session = self.session
        ui = session.ui
        thin = max(1, int(round(ui)))
        hud = canvas[session.view_h:]
        hud[:] = 0
        line1, line2 = self.hud_lines(row, hint, offer)
        # Buttons along the right end of the bar, laid out right to left and measured before the
        # two text lines, so a long clip name or a narrow window cuts the text short instead of
        # running under them. Every box is kept for click_hud: they are all clickable.
        chips = ChipRow(hud, session.hud_h, session.view_h, ui)
        pad = chips.pad
        # Lit green means on; the words stay short so the text lines keep their room
        edge, self.centre_button = chips.chip(session.view_w - pad, "g  CENTRE", session.follow)
        edge, self.inset_button = chips.chip(edge - pad, "o  INSET", session.inset)
        edge, self.filter_button = chips.chip(edge - pad, f"h  {FILTERS[session.filter].upper() if session.filter else 'FILTER'}",
                                              session.filter != 0)
        self.step_buttons = []
        for jump in (8, 6, 4, 2, 1):                     # right to left, so x1 comes out leftmost
            edge, box = chips.chip(edge - pad, f"x{jump}", jump == session.step)
            self.step_buttons.append((box, jump))
        edge = chips.caption(edge, "a/d step")
        room = max(0, edge - int(16 * ui))
        cv2.putText(hud, fit_text(line1, room, 0.55 * ui, thin), (int(8 * ui), int(22 * ui)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55 * ui, (255, 255, 255), thin, cv2.LINE_AA)
        cv2.putText(hud, fit_text(line2, room, 0.48 * ui, thin), (int(8 * ui), int(46 * ui)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48 * ui, (180, 180, 180), thin, cv2.LINE_AA)

    # ---------------------------------------------------------------- clicks on what was drawn
    def click_hud(self, x: int, y: int) -> None:
        """A left click on the HUD bar: the chip under it, if any."""
        if hit(self.centre_button, x, y):
            self.tool.toggle_follow()
        elif hit(self.filter_button, x, y):
            self.tool.cycle_filter()
        elif hit(self.inset_button, x, y):
            self.tool.toggle_inset()
        for box, jump in self.step_buttons:
            if hit(box, x, y):
                self.session.step = jump

    def inset_hit(self, x: int, y: int) -> bool:
        return hit(self.inset_box, x, y)

    def click_inset(self, x: int, y: int) -> None:
        """Either button in the overview: bring that part of the frame into the zoomed view. The
        left button does not place a label from a thumbnail - nothing there is precise enough."""
        session = self.session
        left, top = self.inset_box[0], self.inset_box[1]
        session.centre_on((x + 0.5 - left) / self.inset_scale, (y + 0.5 - top) / self.inset_scale)
