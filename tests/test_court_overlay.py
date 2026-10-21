"""Court overlay scenarios: bounce features, player contacts, rally legs and the minimap.

These were the court half of ``clean_tracker.py --self-test``; keep the fixtures
and expected values as they are.
"""
import numpy as np

from ball_in_play_selector import FrameResult
from tennis_tracker.court_minimap import _visible_completed_legs, draw_court_minimap
from tennis_tracker.court_overlay import (
    BouncePrediction,
    ContactEvent,
    _build_bounce_features,
    _cluster_bounce_candidates,
    _inside_player,
    _is_service_leg,
    assemble_rally_legs,
    infer_kinematic_bounces,
)
from tennis_tracker.config import Config
from tennis_tracker.court_detector import CourtDetector
from tennis_tracker.court_drawing import draw_court

HD_WIDTH, HD_HEIGHT = 1920, 1080
# A serve from the near baseline into the far service box, in court coordinates.
SERVE_START, SERVE_END = (0.80, 1.02), (0.25, 0.35)


def _minibar_leg():
    return {
        "start_frame": 0,
        "end_frame": 30,
        "start": SERVE_START,
        "end": SERVE_END,
        "phase": "serve",
        "outcome": "bounce",
        "quality": "confirmed",
    }


def _det(x, y):
    return FrameResult(cx=x, cy=y, conf=0.9, source="det")


def test_court_line_pairs_join_the_fourteen_keypoints():
    assert CourtDetector._LINE_PAIRS_14 == (
        (0, 4), (4, 6), (6, 1), (0, 2), (1, 3),
        (2, 5), (5, 7), (7, 3), (4, 8), (8, 10), (10, 5),
        (6, 9), (9, 11), (11, 7), (8, 12), (12, 9),
        (10, 13), (13, 11), (12, 13),
    )


def test_bounce_features_use_five_samples_at_the_native_30fps_cadence():
    # The bounce model consumes five samples at its native ~30 FPS cadence.
    bounce_probe = [None] * 9
    for frame, (x, y) in zip(range(0, 9, 2), ((10, 20), (13, 24), (17, 29), (22, 20), (28, 12))):
        bounce_probe[frame] = _det(x, y)
    features, feature_frames, stride = _build_bounce_features(bounce_probe, 60.0, 1280, 720)
    assert stride == 2 and feature_frames == [4]
    np.testing.assert_allclose(
        features[0],
        [4, 7, 5, 11, 0.8, 7 / 11, -5, -9, -9, -17, 5 / 9, 9 / 17],
        rtol=1e-5,
    )


def test_bounce_candidates_cluster_to_the_strongest_per_window():
    assert _cluster_bounce_candidates([4, 6, 8, 14], [0.4, 0.6, 0.8, 0.7], 2) == [
        (8, 0.8), (14, 0.7)
    ]


def test_padded_player_box_reaches_above_and_beside_but_not_below_the_feet():
    strike_box = {1: (100, 100, 140, 200)}
    assert _inside_player(70, 90, strike_box)
    assert not _inside_player(120, 210, strike_box)


def test_service_leg_must_land_in_the_service_box():
    assert _is_service_leg(SERVE_START, SERVE_END, True)
    assert not _is_service_leg(SERVE_START, (0.25, 0.80), True)


def test_kinematic_bounce_is_found_at_a_downward_turn_but_not_at_an_apex():
    bounce_motion = [_det(10 + frame, y) for frame, y in enumerate((0, 4, 10, 6, 4))]
    apex_motion = [_det(10 + frame, y) for frame, y in enumerate((10, 6, 0, 4, 10))]
    assert len(infer_kinematic_bounces(bounce_motion, [{}] * 5, 30.0, HD_WIDTH, HD_HEIGHT)) == 1
    assert not infer_kinematic_bounces(apex_motion, [{}] * 5, 30.0, HD_WIDTH, HD_HEIGHT)


def test_kinematic_bounce_is_found_across_a_detection_gap():
    gap_bounce = [
        _det(10, 0),
        _det(11, 4),
        None, None, None, None,
        _det(16, 4),
        _det(17, 2),
    ]
    assert len(infer_kinematic_bounces(
        gap_bounce, [{}] * len(gap_bounce), 30.0, HD_WIDTH, HD_HEIGHT
    )) == 1


def test_two_kinks_without_contacts_or_bounces_make_no_inferred_leg():
    test_kps = [10, 10, 110, 10, 10, 210, 110, 210] + [0] * 20
    inferred_leg = assemble_rally_legs(
        None, None, 30.0, 120, 220, [test_kps] * 80,
        [
            ContactEvent(0, 90, 210, 1, 10.0, "kink"),
            ContactEvent(50, 35, 10, 2, 10.0, "kink"),
        ],
        BouncePrediction([], []),
        [],
    )
    assert inferred_leg == []


def test_completed_leg_stays_visible_until_the_next_contact_or_timeout():
    minibar_leg = _minibar_leg()
    assert _visible_completed_legs([minibar_leg], [], 60, 60.0) == [minibar_leg]
    assert not _visible_completed_legs(
        [minibar_leg], [ContactEvent(61, 0, 0, 1, 1.0, "kink")], 61, 60.0
    )
    assert not _visible_completed_legs([minibar_leg], [], 91, 60.0)


def test_minimap_draws_only_in_the_top_right_corner():
    minibar_probe = np.zeros((360, 640, 3), dtype=np.uint8)
    draw_court_minimap(
        minibar_probe,
        {"H_i2c": np.eye(3)},
        {1: (0.4, 0.3, 0.6, 0.8)},
        [_minibar_leg()],
        30,
        60.0,
    )
    assert np.any(minibar_probe[:, 400:]) and not np.any(minibar_probe[:, :300])


def test_draw_court_draws_what_the_court_detector_draws():
    keypoints = [20.0 + 30.0 * i for i in range(28)]
    keypoints[4:6] = [0.0, 0.0]       # an undetected point is skipped
    for cfg in (Config(court_model_path=None),
                Config(court_model_path=None, court_draw_indices=True),
                Config(court_model_path=None, court_points_only=True)):
        detector = CourtDetector(cfg)
        detector.keypoints = list(keypoints)
        by_detector = detector.draw(np.zeros((900, 900, 3), np.uint8))
        by_function = np.zeros((900, 900, 3), np.uint8)
        assert draw_court(by_function, keypoints, cfg) is by_function
        assert by_function.any()
        np.testing.assert_array_equal(by_function, by_detector)
    untouched = np.zeros((8, 8, 3), np.uint8)
    assert draw_court(untouched, None, Config()) is untouched and not untouched.any()
