"""Court keypoint geometry: lookup rules, side guides, polygon, homography, cache key."""
import math

import numpy as np
import pytest

from tennis_tracker.court_geometry import (
    CORNER_ORDER,
    MIN_KEYPOINT_VALUES,
    build_court_polygon,
    build_court_side_guides,
    build_ground_projection_model,
    build_ordered_court_polygon,
    clip_infinite_line_to_frame,
    court_axis_spans,
    court_keypoint,
    court_keypoints_key,
    homography_apply,
    last_valid_court_keypoints,
)

# The harness court in a 640x360 frame (tests/test_pipeline_run.py).
KEYPOINTS = [
    180.0, 90.0, 460.0, 90.0, 80.0, 330.0, 560.0, 330.0,
    210.0, 90.0, 120.0, 330.0, 430.0, 90.0, 520.0, 330.0,
    187.5, 150.0, 452.5, 150.0, 146.25, 260.0, 493.75, 260.0,
    320.0, 150.0, 320.0, 260.0,
]


def test_court_keypoint_validity_rules():
    assert court_keypoint(KEYPOINTS, 3, require_finite=True) == (560.0, 330.0)
    assert court_keypoint(None, 0, require_finite=True) is None
    assert court_keypoint(KEYPOINTS[:5], 2, require_finite=True) is None      # too short
    assert court_keypoint([0.0, 0.0, 5.0, 5.0], 0, require_finite=True) is None   # not detected
    assert court_keypoint([0.0, 4.0], 0, require_finite=True) == (0.0, 4.0)       # one zero is fine
    assert court_keypoint([math.nan, 4.0], 0, require_finite=True) is None
    nan_x, _ = court_keypoint([math.nan, 4.0], 0, require_finite=False)
    assert math.isnan(nan_x)


def test_court_keypoints_key_is_quarter_pixel_and_rejects_bad_sets():
    assert court_keypoints_key(None) is None
    assert court_keypoints_key(KEYPOINTS[:MIN_KEYPOINT_VALUES - 1]) is None
    bad = list(KEYPOINTS)
    bad[3] = math.inf
    assert court_keypoints_key(bad) is None
    key = court_keypoints_key(KEYPOINTS)
    assert len(key) == MIN_KEYPOINT_VALUES and key[:2] == (720, 360)
    assert court_keypoints_key([v + 0.1 for v in KEYPOINTS]) == key        # within a quarter pixel
    assert court_keypoints_key([v + 0.3 for v in KEYPOINTS]) != key


def test_side_guides_sit_one_alley_width_outside_each_sideline():
    guides = build_court_side_guides(KEYPOINTS)
    assert set(guides) == {"left", "right"}
    left, right = guides["left"], guides["right"]
    np.testing.assert_allclose(left["base"], (80.0, 330.0))
    np.testing.assert_allclose(left["anchor"], (40.0, 330.0))   # alley = |kp5 - kp2| = 40 px, outward
    np.testing.assert_allclose(left["dir"], (100.0, -240.0))    # along kp2 -> kp0
    np.testing.assert_allclose(right["anchor"], (600.0, 330.0))
    assert build_court_side_guides(KEYPOINTS[:10]) is None
    collapsed = list(KEYPOINTS)
    collapsed[10:12] = collapsed[4:6]       # left singles mark on the corner: no alley
    collapsed[14:16] = collapsed[6:8]
    assert build_court_side_guides(collapsed) is None


def test_clip_infinite_line_to_frame():
    seg = clip_infinite_line_to_frame(np.array([40.0, 330.0]), np.array([100.0, -240.0]), 640, 360)
    assert seg is not None
    for x, y in seg:
        assert 0 <= x <= 640 and 0 <= y <= 360
    assert clip_infinite_line_to_frame(np.array([0.0, 0.0]), np.array([0.0, 0.0]), 640, 360) is None


def test_polygon_corner_order_and_degenerate_rejection():
    poly = build_ordered_court_polygon(KEYPOINTS)
    assert poly.shape == (4, 1, 2)
    expected = [(KEYPOINTS[2 * i], KEYPOINTS[2 * i + 1]) for i in CORNER_ORDER]
    np.testing.assert_allclose(poly[:, 0, :], expected)
    assert build_ordered_court_polygon([1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]) is None  # collinear
    assert build_ordered_court_polygon(KEYPOINTS[:6]) is None


def test_ground_model_maps_corners_to_the_unit_court_and_back():
    model = build_ground_projection_model(KEYPOINTS)
    assert set(model) == {"H_i2c", "H_c2i"}
    for (u, v), idx in zip([(0, 0), (0, 1), (1, 1), (1, 0)], CORNER_ORDER):
        x, y = KEYPOINTS[2 * idx], KEYPOINTS[2 * idx + 1]
        np.testing.assert_allclose(homography_apply(model["H_i2c"], x, y), (u, v), atol=1e-6)
        np.testing.assert_allclose(homography_apply(model["H_c2i"], u, v), (x, y), atol=1e-4)
    assert build_ground_projection_model(None) is None
    assert homography_apply(np.zeros((3, 3)), 1.0, 1.0) is None


def test_last_valid_court_keypoints_needs_the_eight_geometry_points():
    corners_only = KEYPOINTS[:8]
    assert last_valid_court_keypoints([KEYPOINTS, corners_only, None]) is KEYPOINTS
    assert last_valid_court_keypoints([KEYPOINTS, KEYPOINTS[:MIN_KEYPOINT_VALUES]]) == KEYPOINTS[:16]
    assert last_valid_court_keypoints([None, corners_only, []]) is None
    assert last_valid_court_keypoints([corners_only], min_values=8) is corners_only
    assert last_valid_court_keypoints([]) is None


def test_build_court_polygon_takes_the_last_frame_with_valid_keypoints(capsys):
    poly, kps = build_court_polygon([None, KEYPOINTS, None])
    assert kps is KEYPOINTS and poly.shape == (4, 1, 2)
    assert "[court polygon] Ordered corners" in capsys.readouterr().out
    assert build_court_polygon([None, []]) == (None, None)
    # Fewer than the sixteen values the JSON export and pass 2 require: no polygon either.
    assert build_court_polygon([KEYPOINTS[:8]]) == (None, None)


def test_court_axis_spans_fall_back_to_the_frame():
    assert court_axis_spans(KEYPOINTS, 640, 360) == (480.0, 240.0)
    assert court_axis_spans(None, 640, 360) == (640.0, 360.0)
    tiny = list(KEYPOINTS)
    tiny[4:8] = [300.0, 95.0, 301.0, 95.0]       # 1 px near baseline, 5 px sideline
    assert court_axis_spans(tiny, 640, 360) == pytest.approx((0.35 * 640, 0.25 * 360))
