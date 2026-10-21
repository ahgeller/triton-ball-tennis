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
