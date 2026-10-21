"""finetune/labeling/label_draw.py on blank arrays and synthetic rows: the picture filters, the
mark plan, the ticks, marks, ghost, badges and chips, and text cut to fit."""
import cv2
import numpy as np
import pytest

from finetune.labeling.label_draw import (DRAFT_MARK, FILTERS, GHOST, HINT_TRUSTED, HINT_UNSURE, ChipRow, draw_badge,
                                          draw_ghost, draw_mark, draw_ticks, enhance, fit_text, hint_colour, hit,
                                          label_mark_colour, mark_plan, review_badge)
from finetune.labeling.label_hints import OFFER_MIN_PX
from finetune.labeling.label_store import Row


def blank(height=90, width=160, value=0):
    return np.full((height, width, 3), value, np.uint8)


def hint(x, y, trusted=True):
    return {"x": x, "y": y, "trusted": trusted, "ball_here": True}


# --------------------------------------------------------------------------- filters

def test_off_returns_the_picture_and_every_filter_keeps_its_shape():
    picture = blank(value=120)
    assert enhance(picture, "off") is picture
    for mode in FILTERS[1:]:
        out = enhance(picture, mode, picture)
        assert out.shape == picture.shape and out.dtype == np.uint8


def test_dim_white_dims_white_and_keeps_yellow():
    picture = blank(value=255)
    picture[:, :80] = (40, 230, 230)                       # a yellow ball-ish left half (BGR)
    out = enhance(picture, "dim white")
    assert tuple(out[0, 120]) < (255, 255, 255)            # white pulled down
    assert tuple(out[0, 40]) == (40, 230, 230)             # yellow untouched


def test_ball_pop_greys_everything_but_ball_yellows():
    picture = blank(value=200)
    picture[:, :80] = (40, 230, 230)
    out = enhance(picture, "ball pop")
    assert tuple(out[0, 40]) == (40, 230, 230)
    assert out[0, 120, 0] == out[0, 120, 1] == out[0, 120, 2] < 200


def test_motion_paints_what_changed_in_magenta_and_needs_a_matching_previous():
    before, after = blank(value=60), blank(value=60)
    after[40:50, 70:80] = 255
    out = enhance(after, "motion", before)
    moved, still = out[45, 75], out[5, 5]
    assert still[0] == still[1] == still[2] < 60                    # dimmed, no cast
    assert moved[0] > moved[1] and moved[2] > moved[1]              # magenta over the change
    assert enhance(after, "motion", None) is after
    assert enhance(after, "motion", blank(height=10, width=10)) is after


# --------------------------------------------------------------------------- the mark plan

def test_hit_and_colours():
    assert hit((10, 10, 20, 20), 15, 15) and not hit((10, 10, 20, 20), 21, 15) and not hit(None, 0, 0)
    assert hint_colour(hint(0, 0, True)) == HINT_TRUSTED and hint_colour(hint(0, 0, False)) == HINT_UNSURE
    row = Row(0, 10.0, 10.0, True, "csv", 0.0)
    assert label_mark_colour(row) == (0, 255, 255)
    row.reviewed = True
    assert label_mark_colour(row) == (0, 220, 0)
    row.edited = True
    assert label_mark_colour(row) == (0, 0, 255)
    row.suspect = True
    assert label_mark_colour(row) == (255, 0, 255)


def test_an_offer_is_the_mark_and_the_draft_is_ticked_grey():
    row = Row(0, 10.0, 10.0, True, "det", 0.4)
    plan = mark_plan(row, hint(12.0, 10.0), hint(12.0, 10.0), (1, 2, 3))
    assert plan.mark == (12.0, 10.0) and plan.colour == HINT_TRUSTED and plan.ticks == [((10.0, 10.0), DRAFT_MARK)]
    absent = Row(0, 10.0, 10.0, False, "none", 0.0)
    assert mark_plan(absent, hint(12.0, 10.0), hint(12.0, 10.0), (1, 2, 3)).ticks == []


def test_without_an_offer_the_label_is_the_mark_and_a_far_hint_is_ticked():
    row = Row(0, 10.0, 10.0, True, "csv", 0.0)
    assert mark_plan(row, None, None, (1, 2, 3)) == ((10.0, 10.0), (1, 2, 3), [])
    near = mark_plan(row, hint(10.0 + OFFER_MIN_PX / 2, 10.0, False), None, (1, 2, 3))
    assert near.ticks == []
    far = mark_plan(row, hint(10.0 + OFFER_MIN_PX, 10.0, False), None, (1, 2, 3))
    assert far.ticks == [((10.0 + OFFER_MIN_PX, 10.0), HINT_UNSURE)]
    absent = Row(0, 10.0, 10.0, False, "none", 0.0)
    assert mark_plan(absent, hint(12.0, 10.0), None, (1, 2, 3)).mark is None


# --------------------------------------------------------------------------- the marks

def test_ticks_and_mark_leave_the_centre_clear():
    image = blank()
    draw_ticks(image, 80, 45, (0, 0, 255), 1.0)
    assert not image[45, 80].any() and image[45 + 5, 80 + 5].any()
    image = blank()
    draw_mark(image, 80, 45, 12, (0, 255, 0), 1.0)
    assert not image[45, 80].any() and image[45, 80 + 12].any() and image[45, 80 + 12 + 6].any()


def test_ghost_is_dashed_and_captioned():
    image = blank(height=120, width=320)
    draw_ghost(image, 60, 60, 12, 1.0)
    ring = image[46:75, 46:75]
    assert not image[60, 60].any() and ring.any()
    assert (ring[..., 0] > ring[..., 2]).any() and not (ring[..., 2] > ring[..., 0]).any()   # GHOST is blue-heavy
    assert image[55:70, 80:200].any()                      # the caption to the right
    assert GHOST[0] > GHOST[2]


@pytest.mark.parametrize("ui", [1.0, 1.35, 2.5])
def test_badges_sit_at_the_corners(ui):
    image = blank(height=120, width=800)
    draw_badge(image, "REVIEWED", (70, 190, 70), (0, 0, 0), ui, right_edge=800)
    draw_badge(image, "3 LEFT", (50, 50, 50), (255, 255, 255), ui)
    assert tuple(image[int(10 * ui), int(10 * ui)]) == (50, 50, 50)          # the counter's box
    assert tuple(image[int(10 * ui), 800 - int(10 * ui)]) == (70, 190, 70)   # the badge's right edge
    assert not image[int(10 * ui), 400].any()                                # room between them


def test_review_badge_names_the_most_telling_state():
    row = Row(0, 1.0, 1.0, True, "csv", 0.0)
    assert review_badge(row) is None
    row.reviewed = True
    assert review_badge(row) == ("REVIEWED", (70, 190, 70))
    row.edited = True
    assert review_badge(row) == ("EDITED", (60, 140, 255))
    row.suspect = True
    assert review_badge(row) == ("CHECK THIS", (200, 60, 200))


def test_chips_lay_out_right_to_left_and_report_clickable_boxes():
    hud = blank(height=58, width=400)
    chips = ChipRow(hud, 58, 300, 1.0)
    edge, box = chips.chip(400 - chips.pad, "g  CENTRE", True)
    assert box[2] == 400 - chips.pad and box[0] == edge and box[1] == 300 + chips.top
    assert tuple(hud[chips.top + 1, edge + 1]) == (70, 190, 70)
    edge2, box2 = chips.chip(edge - chips.pad, "x1", False)
    assert box2[2] == edge - chips.pad and edge2 < edge and tuple(hud[chips.top + 1, edge2 + 1]) == (55, 55, 55)
    assert chips.caption(edge2, "a/d step") < edge2


# --------------------------------------------------------------------------- fit_text

def test_fit_text_cuts_on_a_word_and_keeps_what_fits():
    text = "reviewed 10/20  uncertain left 3  SPACE=good  y=take hint  v=invisible  q=quit"
    full = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
    assert fit_text(text, full, 0.5, 1) == text
    cut = fit_text(text, full // 2, 0.5, 1)
    assert text.startswith(cut) and cut and not cut.endswith(" ")
    assert cv2.getTextSize(cut, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] <= full // 2
    assert text[len(cut)] == " "                            # the cut fell between words
    assert fit_text(text, 0, 0.5, 1) == ""
