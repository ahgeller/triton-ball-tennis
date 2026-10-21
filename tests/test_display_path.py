"""display_path.build_display_path keeps observations and linear fills apart (exact vs. not)."""
from types import SimpleNamespace

from ball_in_play_selector import Detection

from tennis_tracker.render.display_path import build_display_path
from tennis_tracker.render.trails import _DIRECTION_BREAK_MIN_STEP_PX, _trail_direction_break

WIDTH, HEIGHT, FRAMES = 1280, 720, 10


def det(frame, cx, cy, conf=0.9, on_motion=False):
    return Detection(frame, cx, cy, cx - 4, cy - 4, cx + 4, cy + 4, conf, 64.0, on_motion)


def track(*observations):
    return SimpleNamespace(num_obs=len(observations), observations=list(observations))


def test_no_track_or_no_observations_gives_an_empty_path():
    assert build_display_path(None, FRAMES, 5, WIDTH, HEIGHT) == {}
    assert build_display_path(track(), FRAMES, 5, WIDTH, HEIGHT) == {}


def test_short_gap_is_filled_linearly_and_marked_as_not_exact():
    path = build_display_path(track(det(0, 100.0, 50.0), det(3, 130.0, 50.0)), FRAMES, 5, WIDTH, HEIGHT)
    assert sorted(path) == [0, 1, 2, 3]
    assert path[0] == (100.0, 50.0, True)
    assert path[3] == (130.0, 50.0, True)
    assert path[1] == (110.0, 50.0, False)
    assert path[2] == (120.0, 50.0, False)


def test_gap_beyond_max_interp_gap_stays_absent():
    path = build_display_path(track(det(0, 100.0, 50.0), det(3, 130.0, 50.0)), FRAMES, 2, WIDTH, HEIGHT)
    assert sorted(path) == [0, 3]
    assert all(exact for _, _, exact in path.values())


def test_same_frame_keeps_the_higher_confidence_observation():
    path = build_display_path(track(det(0, 100.0, 50.0, conf=0.5), det(0, 105.0, 55.0, conf=0.9)),
                              FRAMES, 5, WIDTH, HEIGHT)
    assert path == {0: (105.0, 55.0, True)}


def test_observations_outside_the_clip_are_dropped():
    path = build_display_path(track(det(-1, 1.0, 1.0), det(2, 100.0, 50.0), det(FRAMES, 1.0, 1.0)),
                              FRAMES, 5, WIDTH, HEIGHT)
    assert path == {2: (100.0, 50.0, True)}


def test_direction_break_needs_a_reversal_of_real_steps():
    assert _trail_direction_break((0, 0), (20, 0), (0, 0)) is True
    assert _trail_direction_break((0, 0), (20, 0), (40, 0)) is False
    small = _DIRECTION_BREAK_MIN_STEP_PX - 1.0
    assert _trail_direction_break((0, 0), (small, 0), (0, 0)) is False
    assert _trail_direction_break((0, 0), (20, 0), (20 - small, 0)) is False
