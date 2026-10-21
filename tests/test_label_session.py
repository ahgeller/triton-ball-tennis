"""finetune/labeling/label_session.py: the one owner of the label tool's state - the cursor and
the walk kept in step, the frames navigation stops on, the view's geometry and zoom glide, and
where a frame's ball is taken to be. Synthetic clips; no window, no file."""
import pytest

from finetune.labeling.label_session import GHOST_REACH, ZOOM_MAX, LabelSession
from tests._helpers import fake_source


def session_over(*sources):
    return LabelSession(list(sources or [fake_source("a", 12)]))


# --------------------------------------------------------------------------- the cursor and the walk

def test_set_cursor_keeps_the_walk_position_in_step_only_for_rows_on_the_walk():
    session = session_over()
    session.set_cursor(7)
    assert (session.cursor, session.walk_pos) == (7, 0)                      # no walk: nothing to keep in step
    session.walk, session.walk_pos = [2, 5, 8], 0
    session.set_cursor(8)
    assert (session.cursor, session.walk_pos) == (8, 2)
    session.set_cursor(3)                                                    # off the walk: the cursor moves, the walk stays
    assert (session.cursor, session.walk_pos) == (3, 2)


def test_step_cursor_walks_the_walk_or_the_run_and_stops_at_the_ends():
    session = session_over()
    session.step_cursor(3)
    assert (session.cursor, session.direction) == (3, 1)
    session.step_cursor(-20)
    assert (session.cursor, session.direction) == (0, -1)
    session.step_cursor(0)
    assert session.direction == -1                                           # a step of nothing says nothing
    session.walk, session.walk_pos = [2, 5, 8], 1
    session.step_cursor(1)
    assert (session.cursor, session.walk_pos) == (8, 2)
    session.step_cursor(5)
    assert (session.cursor, session.walk_pos) == (8, 2)


def test_next_uncertain_finds_flags_and_unreviewed_uncertain_rows_or_the_end():
    session = session_over()
    for row in session.rows:
        row.conf = 0.9                                                       # every draft confident
    session.rows[4].suspect = True
    session.rows[9].conf = 0.1
    session.rows[9].reviewed = True                                          # uncertain, but dealt with
    assert session.next_uncertain(1) == 4
    session.cursor = 4
    assert session.next_uncertain(1) == 11                                   # nothing more that way: the run's end
    assert session.next_uncertain(-1) == 0
    session.walk, session.walk_pos = [2, 4, 9], 0
    assert session.next_uncertain(1) == 4                                    # along the walk: a flag reopens a row
    session.cursor, session.walk_pos = 4, 1
    assert session.next_uncertain(1) == 9                                    # the walk's end, reviewed or not


def test_only_uncertain_restricts_the_walk_to_the_rows_wanting_attention():
    session = session_over()
    for row in session.rows:
        row.conf = 0.9
    session.rows[3].conf, session.rows[6].conf, session.rows[6].reviewed = 0.2, 0.2, True
    session.rows[10].suspect = True
    assert session.only_uncertain() == 2
    assert session.walk == [3, 10] and (session.cursor, session.walk_pos) == (3, 0)
    assert len(session.rows) == 12                                           # a filter on where you go, never on the rows


# --------------------------------------------------------------------------- the view

def test_fit_sizes_the_view_to_the_window_and_says_when_it_changed():
    session = session_over()
    assert session.fit(1280, 720)
    assert (session.view_w, session.view_h) == (1280, 720 - session.hud_h) and session.ui == 1.0
    assert not session.fit(1280, 720)                                        # the same size: nothing changed
    assert session.fit(2560, 1440) and session.ui == 2.0 and session.hud_h == 116
    assert session.to_view(*session.to_frame(10.5, 20.5)) == (10, 20)       # view pixel u covers [u, u + 1)


def test_centre_on_keeps_the_centre_on_the_frame():
    session = session_over()
    session.centre_on(-50.0, 1000.0)
    assert session.center == (0.0, session.height - 1.0)
    assert session.origin == (0.0 - session.view_w / 2.0, session.height - 1.0 - session.view_h / 2.0)


def test_zoom_glides_towards_its_target_holding_the_anchored_point_still():
    session = session_over()
    session.fit(1280, 720)
    session.center = (session.rows[0].x, session.rows[0].y)
    session.centre_view()
    vx, vy = session.to_view(session.rows[0].x, session.rows[0].y)
    session.zoom_at(2.0, vx + 0.5, vy + 0.5)                                 # the pointer on the ball: pivots on it exactly
    assert session.anchor[0] == (session.rows[0].x, session.rows[0].y)
    assert 1.0 < session.zoom_target <= ZOOM_MAX
    steps = 0
    while session.step_zoom():
        steps += 1
        assert session.to_view(session.rows[0].x, session.rows[0].y) == (vx, vy)
    assert steps > 1 and session.zoom == session.zoom_target
    session.zoom_at(-50.0, 0.0, 0.0)
    assert session.zoom_target == 1.0                                        # never below x1
    session.zoom_at(100.0, 0.0, 0.0)
    assert session.zoom_target == ZOOM_MAX


# --------------------------------------------------------------------------- where the ball is

def test_guess_for_interpolates_between_the_nearest_balls_either_side():
    session = session_over()
    rows = session.rows
    rows[5].visible = False
    assert session.guess_for(5) == pytest.approx(((rows[4].x + rows[6].x) / 2, (rows[4].y + rows[6].y) / 2))
    for index in range(1, 1 + GHOST_REACH + 1):
        rows[index].visible = False                                          # rows 1..4: too far from a ball after
    assert session.guess_for(1) == (rows[0].x, rows[0].y)                    # one side only, but the row next door
    assert session.guess_for(2) is None
    assert session.visible_neighbours(11)[1] is None


def test_target_point_prefers_the_offer_then_the_ball_then_the_ghost():
    session = session_over()
    row = session.rows[0]
    assert session.target_point() == (row.x, row.y)
    session.hints.found["a"] = {0: {"x": row.x + 1.0, "y": row.y, "seed": (row.x, row.y), "from": "label", "move": 1.5,
                                    "path": 0.3, "blobs": 1, "ratio": 1.0, "contrast": 40.0, "trusted": True, "ball_here": True}}
    assert session.target_point() == (row.x + 1.0, row.y)
    assert session.offer_for(0) is not None and session.hint_verdict(0) == "agrees"
    row.reviewed = True                                                      # your label is the mark again
    assert session.target_point() == (row.x, row.y)
    row.visible = False
    session.ghost = (3.0, 4.0)
    assert session.target_point() == (3.0, 4.0)
    assert session.next_disputed(1) is None and session.disputed_count() == 0
