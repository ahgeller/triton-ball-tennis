"""Court-event API contracts: named events, the shared ``CourtScene``, the cache key, and the
``--court`` flow run end to end against a fake CatBoost model."""
import math
from typing import List

import numpy as np
import pytest

from ball_in_play_selector import FrameResult
from tennis_tracker.court.court_geometry import court_keypoints_key
from tests.test_pipeline_run import FULL_OVERRIDES, clip, load_golden, render_env, run_pipeline  # noqa: F401
from tennis_tracker.court import bounce_events
from tennis_tracker.court.bounce_events import infer_kinematic_bounces, predict_bounce_events
from tennis_tracker.court.contact_events import detect_player_contacts
from tennis_tracker.court.court_scene import BounceEvent, BouncePrediction, ContactEvent, CourtScene
from tennis_tracker.court.rally_legs import assemble_rally_legs

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
    (event,) = infer_kinematic_bounces(CourtScene(track, [{}] * len(track), 30.0, 1920, 1080))
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
    contacts = detect_player_contacts(CourtScene(track, near, 30.0, 1920, 1080))
    assert contacts and all(isinstance(event, ContactEvent) for event in contacts)
    assert {event.player_id for event in contacts} == {1}
    assert contacts[0].y == 360.0  # anchored at the player's feet
    assert detect_player_contacts(CourtScene(track, far, 30.0, 1920, 1080)) == []


def test_gap_contact_skips_a_player_without_a_box():
    """A short reversing gap where one tracked player has no box this frame: that player is
    skipped (as nearest_player and inside_player do) and the contact goes to the other one."""
    track = [FrameResult(cx=100.0 + 20 * frame, cy=100.0, conf=0.9, source="det") for frame in range(10)]
    track += [None] * 4                                                   # frames 10-13 hidden
    track += [FrameResult(cx=260.0 - 20 * step, cy=100.0, conf=0.9, source="det") for step in range(6)]
    boxes = [{7: None, 8: (270.0, 50.0, 330.0, 200.0)}] * len(track)
    contacts = detect_player_contacts(CourtScene(track, boxes, 30.0, 1920, 1080))
    assert contacts == [ContactEvent(11, 300.0, 200.0, 8, 1040.0, "gap")]


def test_assemble_links_a_contact_to_its_confirmed_landing():
    contacts = [ContactEvent(0, 60.0, 205.0, 1, 10.0, "kink")]
    confirmed = [BounceEvent(30, 40.0, 60.0, 0.9)]
    keypoints = [KEYPOINTS] * 80
    legs = assemble_rally_legs(
        CourtScene([None] * 80, [{}] * 80, 30.0, 120, 220), keypoints, contacts,
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
        predict_bounce_events(CourtScene([], [], 30.0, 1280, 720), str(tmp_path / "missing.cbm"))


class FakeBounceModel:
    """A CatBoost-like regressor: ``load_model`` records the path, ``predict`` scores every
    five-point window by how sharply the ball turns from descending to ascending."""

    loaded: List[str] = []

    def load_model(self, path: str) -> None:
        type(self).loaded.append(path)

    def predict(self, features: np.ndarray, thread_count: int = 1) -> np.ndarray:
        # Feature columns 6..9 are the y offsets of the two previous and two next samples
        # from the centre: a landing has both previous samples above (negative) and both
        # next samples above as well, so the centre is the lowest point.
        rows = np.asarray(features, dtype=np.float64)
        lowest = (rows[:, 6] < 0) & (rows[:, 7] < 0) & (rows[:, 8] < 0) & (rows[:, 9] < 0)
        return np.where(lowest, 0.95, 0.05)


@pytest.fixture
def bounce_model(monkeypatch, tmp_path):
    """The bounce-model file and the loader patched to the fake regressor."""
    FakeBounceModel.loaded = []
    monkeypatch.setattr(bounce_events, "_catboost_regressor", lambda: FakeBounceModel)
    path = tmp_path / "bounce.cbm"
    path.write_bytes(b"")
    return path


def _landing_track(frames: int = 30) -> List[FrameResult]:
    """A ball falling to y=200 at frame 15, then rising: one landing, no apex before it."""
    return [FrameResult(cx=20.0 + 3.0 * f, cy=200.0 - 6.0 * abs(f - 15), conf=0.9, source="det")
            for f in range(frames)]


def test_bounce_model_scores_windows_and_clusters_one_landing(bounce_model):
    track = _landing_track()
    scene = CourtScene(track, [{}] * len(track), 30.0, 120, 220)
    prediction = predict_bounce_events(scene, str(bounce_model))
    assert FakeBounceModel.loaded == [str(bounce_model.resolve())]
    assert [event.frame for event in prediction.confirmed] == [15]
    assert prediction.confirmed[0].x == track[15].cx and prediction.confirmed[0].y == 200.0
    assert prediction.candidates == prediction.confirmed            # nothing scored in between
    on_player = [{1: (track[15].cx - 10, 150.0, track[15].cx + 10, 205.0)}] * len(track)
    assert predict_bounce_events(CourtScene(track, on_player, 30.0, 120, 220), str(bounce_model)).confirmed == []


def test_court_event_stage_runs_the_model_contacts_and_legs(bounce_model, capsys):
    from types import SimpleNamespace

    from tennis_tracker import pipeline
    from tennis_tracker.config import Config
    from tennis_tracker.run_context import CourtEvents, VideoInfo

    track = _landing_track()
    # A far-side player (feet at court v=0.45) the ball leaves at frame 0 before landing on
    # the near side at frame 15: no visible stroke, so the leg's contact is rescued.
    player = {7: (10.0, 0.0, 40.0, 100.0)}
    ctx = SimpleNamespace(
        cfg=Config(draw_court_minimap=True, bounce_model_path=str(bounce_model)),
        video=VideoInfo(path=bounce_model, fps=30.0, width=120, height=220, frame_count_hint=len(track)),
    )
    pass1 = SimpleNamespace(player_boxes=[player] * len(track), court_keypoints=[KEYPOINTS] * len(track))
    selection = SimpleNamespace(per_frame=track)
    events = pipeline._court_events(ctx, pass1, selection)
    out = capsys.readouterr().out
    assert isinstance(events, CourtEvents)
    assert [bounce.frame for bounce in events.bounces] == [15] and events.contacts == []
    (leg,) = events.rally_legs
    assert (leg["start_frame"], leg["end_frame"], leg["phase"], leg["outcome"]) == (0, 15, "rally", "bounce")
    assert (leg["quality"], leg["start_quality"]) == ("confirmed", "inferred")
    assert out.startswith("[court minimap] 1 confirmed bounce(s), 0 player contact(s), 1 leg(s) (0 inferred, 0 volley)")
    # Off, the stage neither loads the model nor prints.
    FakeBounceModel.loaded = []
    ctx.cfg.draw_court_minimap = False
    assert pipeline._court_events(ctx, pass1, selection) == CourtEvents()
    assert FakeBounceModel.loaded == [] and capsys.readouterr().out == ""
    ctx.cfg.draw_court_minimap, ctx.cfg.bounce_model_path = True, None
    with pytest.raises(RuntimeError, match="bounce_model_path"):
        pipeline._court_events(ctx, pass1, selection)


def test_court_flag_draws_the_minimap_on_the_golden_clip(tmp_path, monkeypatch, fake_cuda, capsys, clip, bounce_model):
    """``--court`` on the golden run: the bounce model is loaded, the minimap is drawn on
    the tracking video and the tracking JSON is unchanged (events are not exported)."""
    plain = load_golden()
    result = run_pipeline(tmp_path, monkeypatch, clip, **FULL_OVERRIDES,
                          draw_court_minimap=True, bounce_model_path=str(bounce_model))
    out = capsys.readouterr().out
    assert FakeBounceModel.loaded == [str(bounce_model.resolve())]
    assert "[court minimap]" in out
    assert result["json"] == plain["json"]
    if plain.get("render_env") == render_env():
        assert result["streams"]["detector_input"] == plain["streams"]["detector_input"]
        assert result["streams"]["tracking"] != plain["streams"]["tracking"]
