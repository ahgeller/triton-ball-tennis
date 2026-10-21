"""tennis_tracker.trails: the chosen ball's trail breaks where it should, the display-path trail
keeps exact and filled points apart, and the candidate overlay draws every track. Synthetic
frames and results only."""
from types import SimpleNamespace

import numpy as np
from ball_in_play_selector import Detection, FrameResult

from tennis_tracker.config import Config
from tennis_tracker.rendering import Canvas
from tennis_tracker.trails import (
    CANDIDATE_TRAIL_LEN, TRAIL_MIN_LEN, TRAIL_SECONDS, BallTrail, CandidateTrackTrails, DisplayPathTrail,
    _trail_jump_fracs, track_color,
)

WIDTH, HEIGHT, FPS = 640, 360, 30.0


def result(cx, cy, source="det", conf=0.9):
    return FrameResult(cx=cx, cy=cy, conf=conf, source=source)


def blank():
    return np.zeros((HEIGHT, WIDTH, 3), np.uint8)


# --------------------------------------------------------------------------- BallTrail

def test_trail_length_follows_the_frame_rate_with_a_floor():
    assert BallTrail(Config(), 60.0, WIDTH, HEIGHT).max_len == max(TRAIL_MIN_LEN, round(60.0 * TRAIL_SECONDS))
    assert BallTrail(Config(), 10.0, WIDTH, HEIGHT).max_len == TRAIL_MIN_LEN


def test_a_steady_path_is_one_segment_and_a_jump_breaks_it():
    trail = BallTrail(Config(), FPS, WIDTH, HEIGHT)
    for i in range(4):
        trail.advance(result(100.0 + 8 * i, 200.0), None)
    assert None not in trail.points and len(trail.points) == 4
    trail.advance(result(500.0, 60.0), None)              # across the frame in one step
    assert trail.points[-2] is None and trail.points[-1][:2] == (500, 60)


def test_losing_the_ball_hard_breaks_the_segment_and_the_trail_is_bounded():
    trail = BallTrail(Config(), FPS, WIDTH, HEIGHT)
    trail.advance(result(100.0, 100.0), None)
    trail.advance(None, None)
    trail.advance(None, None)
    trail.advance(result(104.0, 100.0), None)
    # One break per lost frame, and recovering from a lost run breaks once more before the point.
    assert [p is None for p in trail.points] == [False, True, True, True, False]
    for i in range(3 * trail.max_len):
        trail.advance(result(104.0 + i, 100.0), None)
    assert len(trail.points) == trail.max_len


def test_source_specific_jump_fractions():
    cfg = Config()
    assert _trail_jump_fracs("det", cfg) == (cfg.trail_hard_switch_x_frac, cfg.trail_hard_switch_y_frac)
    assert _trail_jump_fracs("motion", cfg) <= _trail_jump_fracs("det", cfg)
    assert _trail_jump_fracs("interp", cfg) == _trail_jump_fracs("interp", cfg)
    assert _trail_jump_fracs(None, cfg) == _trail_jump_fracs("other", cfg)


def test_drawing_paints_the_debug_frame_and_the_main_frame_only_when_asked():
    trail = BallTrail(Config(), FPS, WIDTH, HEIGHT)
    for i in range(4):
        trail.advance(result(100.0 + 8 * i, 200.0), None)
    main, debug = blank(), blank()
    trail.draw(Canvas(main, debug), draw_main=False)
    assert debug.any() and not main.any()
    trail.draw(Canvas(main, debug), draw_main=True)
    assert main.any()


# --------------------------------------------------------------------------- DisplayPathTrail

def test_display_path_trail_rounds_points_and_records_gaps_once():
    trail = DisplayPathTrail(FPS, WIDTH)
    trail.advance((10.4, 20.6, True))
    trail.advance(None)
    trail.advance(None)
    trail.advance((30.0, 40.0, False))
    assert trail.points == [(10, 21, True), None, (30, 40, False)]


def test_display_path_trail_draws_exact_segments_brighter_than_fills():
    trail = DisplayPathTrail(FPS, WIDTH)
    for x in (100.0, 110.0, 120.0):
        trail.advance((x, 100.0, True))
    exact = blank()
    trail.draw(exact, (120.0, 100.0, True))
    filled = DisplayPathTrail(FPS, WIDTH)
    for x in (100.0, 110.0, 120.0):
        filled.advance((x, 100.0, False))
    soft = blank()
    filled.draw(soft, (120.0, 100.0, False))
    assert exact.any() and soft.any()
    assert int(exact.sum()) > int(soft.sum())        # thicker, whiter line and a bigger dot


def test_display_path_trail_skips_a_segment_that_spans_the_frame():
    trail = DisplayPathTrail(FPS, WIDTH)
    trail.advance((10.0, 100.0, True))
    trail.advance((WIDTH - 10.0, 100.0, True))
    debug = blank()
    trail.draw(debug, None)
    assert not debug[100, WIDTH // 2].any()


# --------------------------------------------------------------------------- CandidateTrackTrails

def det(frame, cx, cy):
    return Detection(frame, cx, cy, cx - 4, cy - 4, cx + 4, cy + 4, 0.9, 64.0, False)


def track(track_id, score, *observations):
    return SimpleNamespace(track_id=track_id, score=score, observations=list(observations))


def test_track_colour_is_stable_and_never_dark():
    assert track_color(7) == track_color(7)
    assert all(max(track_color(i)) >= 100 for i in range(50))


def test_candidate_trails_follow_each_track_and_mark_the_chosen_one():
    a = track(1, 10.0, det(0, 50.0, 50.0), det(1, 60.0, 50.0), det(3, 80.0, 50.0))
    b = track(2, 30.0, det(0, 300.0, 200.0))
    trails = CandidateTrackTrails([a, b], chosen_track=b)
    assert [int(t.track_id) for t in trails.tracks] == [2, 1]        # best score first
    assert trails.chosen_id == 2
    debug = blank()
    for frame in range(4):
        trails.advance_and_draw(debug, frame)
    assert trails.trails[1] == [(50.0, 50.0), (60.0, 50.0), None, (80.0, 50.0)]
    assert trails.trails[2] == [(300.0, 200.0), None]
    assert debug[50, 55].any() and debug[200, 300].any()


def test_candidate_trails_are_bounded():
    long = track(1, 1.0, *(det(f, 10.0 + f, 10.0) for f in range(CANDIDATE_TRAIL_LEN + 20)))
    trails = CandidateTrackTrails([long], chosen_track=None)
    debug = blank()
    for frame in range(CANDIDATE_TRAIL_LEN + 20):
        trails.advance_and_draw(debug, frame)
    assert len(trails.trails[1]) == CANDIDATE_TRAIL_LEN
