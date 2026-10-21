"""Court-event API contracts: named events, aligned stage signatures, cache key."""
import math

import numpy as np
import pytest

from ball_in_play_selector import FrameResult
from tennis_tracker.court_geometry import court_keypoints_key
from tennis_tracker.court_overlay import (
    BounceEvent,
    BouncePrediction,
    ContactEvent,
    assemble_rally_legs,
    detect_player_contacts,
    infer_kinematic_bounces,
    predict_bounce_events,
)

# Court corners TL, TR, BL, BR in a 120x220 image: unit court = (x-10)/100, (y-10)/200.
KEYPOINTS = [10, 10, 110, 10, 10, 210, 110, 210] + [0] * 20


def test_events_are_tuple_compatible_named_tuples():
    bounce = BounceEvent(12, 1.0, 2.0, 0.7)
    contact = ContactEvent(3, 4.0, 5.0, 1, 9.0, "kink")
    assert bounce == (12, 1.0, 2.0, 0.7) and bounce.frame == 12
    assert contact == (3, 4.0, 5.0, 1, 9.0, "kink") and contact.player_id == 1
    confirmed, candidates = BouncePrediction([bounce], [])
    assert confirmed == [bounce] and candidates == []


def test_kinematic_bounce_is_a_bounce_event():
    track = [
        FrameResult(cx=10 + frame, cy=y, conf=0.9, source="det")
        for frame, y in enumerate((0, 4, 10, 6, 4))
    ]
    (event,) = infer_kinematic_bounces(track, [{}] * len(track), 30.0, 1920, 1080)
    assert isinstance(event, BounceEvent) and event.frame == 2


def test_contacts_need_a_nearby_player():
    track = [
        FrameResult(cx=100.0 + 20 * frame, cy=300.0, conf=0.9, source="det")
        for frame in range(5)
    ]
    track += [
        FrameResult(cx=track[-1].cx - 20 * step, cy=300.0 - 2 * step, conf=0.9, source="det")
        for step in range(1, 6)
    ]
    near = [{1: (170.0, 250.0, 210.0, 360.0)}] * len(track)
    far = [{1: (900.0, 250.0, 940.0, 360.0)}] * len(track)
    contacts = detect_player_contacts(track, near, 30.0, 1920, 1080)
    assert contacts and all(isinstance(event, ContactEvent) for event in contacts)
    assert {event.player_id for event in contacts} == {1}
    assert contacts[0].y == 360.0  # anchored at the player's feet
    assert detect_player_contacts(track, far, 30.0, 1920, 1080) == []


def test_assemble_links_a_contact_to_its_confirmed_landing():
    contacts = [ContactEvent(0, 60.0, 205.0, 1, 10.0, "kink")]
    confirmed = [BounceEvent(30, 40.0, 60.0, 0.9)]
    keypoints = [KEYPOINTS] * 80
    legs = assemble_rally_legs(
        None, None, 30.0, 120, 220, keypoints, contacts,
        BouncePrediction(confirmed, []), [],
    )
    assert len(legs) == 1
    leg = legs[0]
    assert (leg["start_frame"], leg["end_frame"]) == (0, 30)
    assert leg["outcome"] == "bounce" and leg["quality"] == "confirmed"
    np.testing.assert_allclose(leg["end"], (0.30, 0.25), atol=1e-6)


def test_court_keypoints_key_rejects_short_and_non_finite_sets():
    assert court_keypoints_key(None) is None
    assert court_keypoints_key(KEYPOINTS[:15]) is None
    for bad in (math.nan, math.inf, None):
        keypoints = list(KEYPOINTS)
        keypoints[3] = bad
        assert court_keypoints_key(keypoints) is None
    assert court_keypoints_key(np.asarray(KEYPOINTS, dtype=np.float32) + 0.1) == tuple(
        int(round(float(value) * 4.0)) for value in np.asarray(KEYPOINTS[:16], np.float32) + 0.1
    )


def test_missing_bounce_model_fails_before_scoring(tmp_path):
    with pytest.raises(FileNotFoundError):
        predict_bounce_events([], [], 30.0, 1280, 720, str(tmp_path / "missing.cbm"))
