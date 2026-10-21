"""tennis_tracker.court_drawing on blank frames: which keypoints are joined, which are skipped,
and what the court layer adds for each of its inputs."""
import numpy as np

from tennis_tracker.config import Config
from tennis_tracker.court_drawing import (
    _court_line_pairs, _detected_court_point, draw_court, draw_court_layer, draw_court_side_guides,
    draw_homography_net_line,
)
from tennis_tracker.court_geometry import LINE_PAIRS_14

WIDTH, HEIGHT = 320, 180


def blank():
    return np.zeros((HEIGHT, WIDTH, 3), np.uint8)


def test_line_pairs_follow_the_layout_for_14_points_and_chain_otherwise():
    assert _court_line_pairs(14) is LINE_PAIRS_14
    assert _court_line_pairs(5) == [(0, 1), (1, 2), (2, 3), (3, 4)]
    assert _court_line_pairs(3) == ()


def test_a_keypoint_at_the_origin_or_below_is_not_detected():
    assert _detected_court_point([0.0, 0.0, 10.0, 20.0], 0) is None
    assert _detected_court_point([-5.0, -5.0, 10.0, 20.0], 0) is None
    assert _detected_court_point([0.0, 0.0, 10.0, 20.0], 1) == (10, 20)


def test_draw_court_dots_and_lines_and_honours_points_only():
    keypoints = [20.0, 20.0, 120.0, 20.0, 120.0, 120.0, 20.0, 120.0]
    frame = draw_court(blank(), keypoints, Config())
    assert frame[20, 20].any() and frame[20, 70].any()          # a dot and the line between the first two
    points_only = draw_court(blank(), keypoints, Config(court_points_only=True))
    assert points_only[20, 20].any() and not points_only[20, 70].any()
    assert not draw_court(blank(), None, Config()).any()


def test_indices_are_drawn_only_when_asked():
    keypoints = [60.0, 60.0, 160.0, 60.0, 160.0, 120.0, 60.0, 120.0]
    plain = draw_court(blank(), keypoints, Config(court_points_only=True))
    labelled = draw_court(blank(), keypoints, Config(court_points_only=True, court_draw_indices=True))
    assert int(labelled.sum()) > int(plain.sum())


def test_court_layer_draws_each_input_it_is_given():
    keypoints = [20.0, 20.0, 120.0, 20.0, 120.0, 120.0, 20.0, 120.0]
    off = blank()
    draw_court_layer([off], Config(draw_court=False), keypoints, None, None)
    assert not off.any()
    polygon = np.array([[10, 10], [100, 10], [100, 100], [10, 100]], dtype=np.float32)
    outlined = blank()
    draw_court_layer([outlined], Config(draw_court=False), None, None, polygon)
    assert outlined[10, 50].any()
    both = [blank(), blank()]
    draw_court_layer(both, Config(), keypoints, None, None)
    assert both[0].any() and np.array_equal(both[0], both[1])


def test_net_line_needs_a_homography_and_side_guides_need_keypoints():
    frame = blank()
    draw_homography_net_line(frame, None)
    draw_homography_net_line(frame, {"H_c2i": None})
    assert not frame.any()
    assert not draw_court_side_guides(blank(), None).any()
    identity = {"H_c2i": np.array([[WIDTH, 0.0, 0.0], [0.0, HEIGHT, 0.0], [0.0, 0.0, 1.0]])}
    drawn = blank()
    draw_homography_net_line(drawn, identity)
    assert drawn[HEIGHT // 2, WIDTH // 2].any()              # v = 0.5 maps to the middle row
