"""What every view of the label tool is drawn with: the picture filters, the ball marks and
their colours, text that is cut to fit, and the zoom step the views share. The single-frame
view (``label_tool``), the grid and the trajectory sheet (``label_views``) all draw from here;
only the mapping to their own pixels is each view's own."""
from __future__ import annotations

import math
from typing import List, NamedTuple, Optional, Tuple

import cv2
import numpy as np

from finetune.labeling.label_hints import OFFER_MIN_PX
from finetune.labeling.label_store import Row
from finetune.labeling.refine import Point, Suggestion

ZOOM_STEP = 1.18          # per wheel notch / keypress; multiplicative so every step feels the same size
FILTERS = ("off", "dim white", "ball pop", "motion")   # h cycles these; see enhance()
HINT_TRUSTED = (0, 150, 255)     # BGR: the suggestion's mark when it is trusted ...
HINT_UNSURE = (70, 110, 150)     # ... and when it is only a guess
DRAFT_MARK = (160, 160, 160)     # the draft label's ticks while the suggestion is the mark on offer
GHOST = (255, 200, 0)            # the dashed guess on a frame with no ball, and its dot in the inset

Colour = Tuple[int, int, int]


def enhance(display: np.ndarray, mode: str, previous: Optional[np.ndarray] = None) -> np.ndarray:
    """Redraw the visible picture so a small ball is easier to pick out. Applied to the zoomed view,
    not the whole frame, so it costs the same at any clip size.

    dim white  - bright, colourless pixels (court lines, white kit, sky) are pulled down to a dull
                 blue-grey; a yellow ball keeps its colour and stops hiding on a line
    ball pop   - only tennis-ball yellows keep their colour, everything else goes to a dim grey
    motion     - what changed since the previous label frame is painted in magenta over a dimmed
                 picture: a moving ball lights up, the court and the crowd do not
    """
    # Everything below stays in OpenCV calls: the same masks written with numpy indexing took
    # 100 ms a redraw on a 2800-pixel-wide view, which is a third of the tool's frame rate.
    if mode == "dim white":
        hsv = cv2.cvtColor(display, cv2.COLOR_BGR2HSV)
        white = cv2.inRange(hsv, (0, 0, 166), (180, 54, 255))
        dimmed = cv2.multiply(display, (0.62, 0.48, 0.38, 1.0))   # BGR: a dull blue-grey
        out = display.copy()
        cv2.copyTo(dimmed, white, out)
        return out
    if mode == "ball pop":
        hsv = cv2.cvtColor(display, cv2.COLOR_BGR2HSV)
        ball = cv2.inRange(hsv, (18, 55, 110), (52, 255, 255))
        out = cv2.cvtColor(cv2.convertScaleAbs(cv2.cvtColor(display, cv2.COLOR_BGR2GRAY), alpha=0.45),
                           cv2.COLOR_GRAY2BGR)
        cv2.copyTo(display, ball, out)
        return out
    if mode == "motion" and previous is not None and previous.shape == display.shape:
        change = cv2.cvtColor(cv2.absdiff(display, previous), cv2.COLOR_BGR2GRAY)
        change = cv2.GaussianBlur(change, (3, 3), 0)
        # The noise floor is measured on the frame pair itself (from a 1-in-64 sample, so it is
        # cheap): compression noise and a slightly unsteady camera stay dark, a ball that moved
        # saturates. Both steps saturate at 0 and 255 - not convertScaleAbs, whose absolute value
        # turned every unchanged pixel into a faint magenta.
        floor = int(max(12.0, float(np.percentile(change[::8, ::8], 99.5)) + 8.0))
        strength = cv2.multiply(cv2.subtract(change, floor), 4.0)
        out = cv2.convertScaleAbs(display, alpha=0.55)
        nothing = np.zeros_like(strength)
        return cv2.add(out, cv2.merge([strength, nothing, strength]))
    return display


def hit(box, x: int, y: int) -> bool:
    """Did a click at (x, y) land on this HUD button? Boxes are in display pixels."""
    return bool(box) and box[0] <= x <= box[2] and box[1] <= y <= box[3]


def hint_colour(hint: Suggestion) -> Colour:
    """Orange when the suggestion is trusted, dull when it is only a guess."""
    return HINT_TRUSTED if hint["trusted"] else HINT_UNSURE


def label_mark_colour(row: Row) -> Colour:
    """The label's mark in the single-frame view: magenta flagged, red edited, green reviewed, yellow unreviewed."""
    return ((255, 0, 255) if row.suspect else (0, 0, 255) if row.edited
            else (0, 220, 0) if row.reviewed else (0, 255, 255))


class MarkPlan(NamedTuple):
    """What a row's frame is marked with, in frame coordinates: the mark (None on a no-ball row
    without an offer), its colour, and the ticked points drawn beside it first."""
    mark: Optional[Point]
    colour: Optional[Colour]
    ticks: List[Tuple[Point, Colour]]


def mark_plan(row: Row, hint: Optional[Suggestion], offer: Optional[Suggestion], label_colour: Colour) -> MarkPlan:
    """The mark is the suggestion on offer - in orange, the draft label it replaces ticked grey - or
    else the label in `label_colour`, with a suggestion that differs from it ticked orange beside
    it. Both renderers draw from this; only the mapping to their pixels is their own."""
    if offer is not None:
        ticks = [((row.x, row.y), DRAFT_MARK)] if row.visible else []
        return MarkPlan((offer["x"], offer["y"]), hint_colour(offer), ticks)
    if not row.visible:
        return MarkPlan(None, None, [])
    ticks = []
    if hint is not None and math.hypot(hint["x"] - row.x, hint["y"] - row.y) >= OFFER_MIN_PX:
        ticks.append(((hint["x"], hint["y"]), hint_colour(hint)))
    return MarkPlan((row.x, row.y), label_colour, ticks)


def draw_ticks(image: np.ndarray, hx: int, hy: int, colour: Colour, ui: float) -> None:
    """Four short ticks on the diagonals around image pixel (hx, hy), leaving the centre itself clear."""
    near, far = int(round(3 * ui)), int(round(7 * ui))
    thin = max(1, int(round(ui)))
    for dx, dy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        start, end = (hx + dx * near, hy + dy * near), (hx + dx * far, hy + dy * far)
        cv2.line(image, start, end, (0, 0, 0), thin + 2, cv2.LINE_AA)
        cv2.line(image, start, end, colour, thin, cv2.LINE_AA)


def draw_mark(image: np.ndarray, cx: int, cy: int, radius: int, colour: Colour, ui: float) -> None:
    """The label's mark in the single-frame view: a thin circle with four ticks pointing in from
    just outside it, leaving the ball itself clear."""
    thin = max(1, int(round(ui)))
    cv2.circle(image, (cx, cy), radius, colour, thin, cv2.LINE_AA)
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        cv2.line(image, (cx + dx * (radius - int(3 * ui)), cy + dy * (radius - int(3 * ui))),
                 (cx + dx * (radius + int(8 * ui)), cy + dy * (radius + int(8 * ui))), colour, thin)


def draw_tile_mark(tile: np.ndarray, px: int, py: int, colour: Colour, ui: float) -> None:
    """A grid tile's mark: four ticks pointing in at the ball pixel, drawn on the tile itself so
    one near its edge cannot spill onto a neighbour, and leaving the ball itself unobstructed."""
    inner, outer = int(round(8 * ui)), int(round(17 * ui))
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        start, end = (px + dx * inner, py + dy * inner), (px + dx * outer, py + dy * outer)
        cv2.line(tile, start, end, (0, 0, 0), max(3, int(round(3 * ui))))
        cv2.line(tile, start, end, colour, max(1, int(round(ui))))


def draw_ghost(image: np.ndarray, gx: int, gy: int, radius: int, ui: float) -> None:
    """The neighbours' idea of where the ball is: dashed, so it cannot be mistaken for a label."""
    thin = max(1, int(round(ui)))
    for start in range(0, 360, 45):
        cv2.ellipse(image, (gx, gy), (radius, radius), 0, start, start + 25, GHOST, thin, cv2.LINE_AA)
    cv2.putText(image, "guess - click to place", (gx + radius + int(6 * ui), gy + int(5 * ui)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45 * ui, GHOST, thin, cv2.LINE_AA)


def draw_badge(image: np.ndarray, text: str, fill: Colour, text_colour: Colour, ui: float,
               right_edge: Optional[int] = None) -> None:
    """A filled box with bold text along the top of the view: at the top-left corner, or ending a
    little short of `right_edge` when one is given."""
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * ui, 2)
    if right_edge is None:
        box = ((int(10 * ui), int(10 * ui)), (int(26 * ui) + text_w, int(24 * ui) + text_h))
        origin = (int(18 * ui), int(17 * ui) + text_h)
    else:
        left = right_edge - text_w - int(26 * ui)
        box = ((left, int(10 * ui)), (right_edge - int(10 * ui), int(24 * ui) + text_h))
        origin = (left + int(8 * ui), int(17 * ui) + text_h)
    cv2.rectangle(image, box[0], box[1], fill, -1)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * ui, text_colour, 2, cv2.LINE_AA)


def review_badge(row: Row) -> Optional[Tuple[str, Colour]]:
    """(text, fill) of the badge a row's frame wears: an open audit flag, an edit, a review; None
    for an untouched row."""
    if row.suspect:
        return "CHECK THIS", (200, 60, 200)
    if row.edited:
        return "EDITED", (60, 140, 255)
    if row.reviewed:
        return "REVIEWED", (70, 190, 70)
    return None


class ChipRow:
    """Clickable chips laid along a HUD strip right to left, each measured before it is drawn so
    the text to their left can be cut short of them. `boxes` are in display pixels (the strip
    sits `top` pixels below the view's own origin)."""

    def __init__(self, hud: np.ndarray, hud_h: int, y0: int, ui: float) -> None:
        self.hud, self.y0, self.ui = hud, y0, ui
        self.thin = max(1, int(round(ui)))
        self.pad = int(6 * ui)
        (_, glyph_h), _ = cv2.getTextSize("Xy8", cv2.FONT_HERSHEY_SIMPLEX, 0.42 * ui, self.thin)
        self.top = max(0, (hud_h - glyph_h - self.pad) // 2)
        self.bottom = min(hud_h - 1, self.top + glyph_h + self.pad)

    def chip(self, right_edge: int, text: str, lit: bool) -> Tuple[int, Tuple[int, int, int, int]]:
        """Draw one chip ending at `right_edge`; (its left edge, its box for a click test)."""
        ui, pad, thin = self.ui, self.pad, self.thin
        (text_w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42 * ui, thin)
        left_edge = max(0, right_edge - text_w - 2 * pad)
        cv2.rectangle(self.hud, (left_edge, self.top), (right_edge, self.bottom),
                      (70, 190, 70) if lit else (55, 55, 55), -1)
        cv2.putText(self.hud, text, (left_edge + pad, self.bottom - pad // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42 * ui, (0, 0, 0) if lit else (190, 190, 190), thin, cv2.LINE_AA)
        return left_edge, (left_edge, self.y0 + self.top, right_edge, self.y0 + self.bottom)

    def caption(self, right_edge: int, text: str) -> int:
        """A dim caption ending at `right_edge`; the edge to its left."""
        (caption_w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4 * self.ui, self.thin)
        edge = max(0, right_edge - self.pad - caption_w)
        cv2.putText(self.hud, text, (edge, self.bottom - self.pad // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * self.ui,
                    (150, 150, 150), self.thin, cv2.LINE_AA)
        return edge


def fit_text(text: str, width: int, scale: float, thin: int) -> str:
    """`text` cut to what fits in `width` pixels. Guessing from the average character width gets it
    about right in one go; the loop is there for the handful of pixels that guess is out by."""
    measure = lambda s: cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, thin)[0][0]
    full = measure(text)
    if full <= width:
        return text
    original = text
    text = text[:max(0, int(len(text) * width / full))]
    while text and measure(text) > width:
        text = text[:-1]
    # End on a whole word ("q=qui" reads as a typo), unless the cut already fell between words.
    if original[len(text):len(text) + 1] not in (" ", "") and " " in text.strip():
        text = text[:text.rstrip().rfind(" ")]
    return text.rstrip()
