"""Court-event API contracts: named events, the shared ``CourtScene``, the cache key, and the
``--court`` flow run end to end against a fake CatBoost model, its events exported."""
import copy
import json
import math
from typing import List

import numpy as np
import pytest

from ball_in_play_selector import FrameResult
from tennis_tracker.court.court_geometry import court_keypoints_key
from tests.test_pipeline_run import (  # noqa: F401
    FULL_OVERRIDES, JSON_ONLY_OVERRIDES, assert_json_close, clip, load_golden, render_env, run_pipeline,
)
from tennis_tracker.court import bounce_events
from tennis_tracker.court.bounce_events import infer_kinematic_bounces, predict_bounce_events
from tennis_tracker.court.contact_events import detect_player_contacts
from tennis_tracker.court.court_scene import (
    BallPoint, BounceEvent, BouncePrediction, ContactEvent, CourtScene, ball_point,
)
from tennis_tracker.court.rally_legs import assemble_rally_legs

# Court corners TL, TR, BL, BR in a 120x220 image: unit court = (x-10)/100, (y-10)/200.
KEYPOINTS = [10, 10, 110, 10, 10, 210, 110, 210] + [0] * 20


def _measured(cx: float, cy: float, moved_by: float = 0.0) -> FrameResult:
    """A detected frame whose measurement sits ``moved_by`` pixels left of the selected point."""
    return FrameResult(cx=cx, cy=cy, conf=0.9, source="det", measurement=(cx - moved_by, cy))


def test_events_are_tuple_compatible_named_tuples():
    bounce = BounceEvent(12, 1.0, 2.0, 0.7)
    contact = ContactEvent(3, 4.0, 5.0, 1, 9.0, "kink")
    assert bounce == (12, 1.0, 2.0, 0.7, "model", "filtered") and bounce.frame == 12
    assert contact == (3, 4.0, 5.0, 1, 9.0, "kink", None) and contact.player_id == 1
    gap = BounceEvent(12, 1.0, 2.0, 0.7, "gap", "extrapolated")
    assert (gap.origin, gap.position) == ("gap", "extrapolated")
    held = ContactEvent(3, 4.0, 5.0, 1, 9.0, "gap", BallPoint(6.0, 7.0, "extrapolated"))
    assert held.ball == (6.0, 7.0, "extrapolated") and held.ball.position == "extrapolated"
    confirmed, candidates = BouncePrediction([bounce], [])
    assert confirmed == [bounce] and candidates == []


def test_ball_point_reports_the_track_position_and_its_kind():
    assert ball_point(_measured(10.0, 20.0)) == BallPoint(10.0, 20.0, "measurement")
    assert ball_point(_measured(10.0, 20.0, moved_by=0.5)) == BallPoint(10.0, 20.0, "filtered")
    assert ball_point(FrameResult(cx=10.0, cy=20.0, source="interp")) == BallPoint(10.0, 20.0, "recovered")


def test_kinematic_bounce_is_a_bounce_event():
    track = [
        FrameResult(cx=10 + frame, cy=y, conf=0.9, source="det")
        for frame, y in enumerate((0, 4, 10, 6, 4))
    ]
    (event,) = infer_kinematic_bounces(CourtScene(track, [{}] * len(track), 30.0, 1920, 1080))
    assert isinstance(event, BounceEvent) and event.frame == 2


def test_kink_and_gap_bounces_say_where_their_position_came_from():
    kink_track = [_measured(10.0 + frame, y, moved_by=0.5) for frame, y in enumerate((0, 4, 10, 6, 4))]
    (kink,) = infer_kinematic_bounces(CourtScene(kink_track, [{}] * 5, 30.0, 1920, 1080))
    assert (kink.origin, kink.position) == ("kink", "filtered") and (kink.x, kink.y) == (12.0, 10.0)
    gap_track = _gap_bounce_track(60.0, vx=60.0, descent=240.0, rise=150.0)
    (gap,) = infer_kinematic_bounces(CourtScene(gap_track, [{}] * len(gap_track), 60.0, 1920, 1080))
    assert (gap.origin, gap.position) == ("gap", "extrapolated")
    assert gap_track[gap.frame] is None                              # the landing was hidden


def test_kinematic_bounces_are_clustered_in_frame_order(monkeypatch):
    # Gap bounces were appended after every visible kink; a gap bounce early in the clip then
    # sat "within" the last kink's merge radius (a negative frame difference) and the stronger
    # of two landings a quarter of a minute apart dropped the other.
    kink, gap = BounceEvent(500, 1.0, 2.0, 0.9), BounceEvent(8, 3.0, 4.0, 0.5)
    monkeypatch.setattr(bounce_events, "_kink_bounces", lambda scene: [kink])
    monkeypatch.setattr(bounce_events, "_gap_bounces", lambda scene: [gap])
    scene = CourtScene([], [], 30.0, 1920, 1080)
    assert infer_kinematic_bounces(scene) == [gap, kink]
    near = BounceEvent(503, 1.5, 2.5, 0.4)                     # within 0.15 s of the kink: one landing
    monkeypatch.setattr(bounce_events, "_gap_bounces", lambda scene: [near, gap])
    assert infer_kinematic_bounces(scene) == [gap, kink]


def _gap_bounce_track(fps: float, vx: float, descent: float, rise: float) -> List:
    """A ball crossing at `vx` px/s and descending `descent` px/s, hidden for 0.27 s around its
    landing, then rising `rise` px/s: the same physical path whatever `fps` samples it."""
    before, gap = round(0.33 * fps), round(0.27 * fps)
    landing_t = (before + (gap - 1) / 2) / fps
    landing_y = 100.0 + descent * landing_t
    track: List = [FrameResult(cx=100.0 + vx * f / fps, cy=100.0 + descent * f / fps, conf=0.9, source="det")
                   for f in range(before)]
    track += [None] * gap
    track += [FrameResult(cx=100.0 + vx * f / fps, cy=landing_y - rise * (f / fps - landing_t), conf=0.9, source="det")
              for f in range(before + gap, before + gap + round(0.33 * fps))]
    return track


@pytest.mark.parametrize("descent, found", [(240.0, 1), (90.0, 0)])
def test_a_hidden_landing_is_judged_the_same_at_30_and_60_fps(descent, found):
    # The gap fit's velocities are per source frame and its gates were tuned at 60 FPS, where a
    # 90 px/s descent (1.5 px/frame) is too slow for a landing. Unscaled, the same descent at
    # 30 FPS (3 px/frame) passed; now both frame rates give one answer per physical path.
    for fps in (30.0, 60.0):
        track = _gap_bounce_track(fps, vx=60.0, descent=descent, rise=150.0)
        scene = CourtScene(track, [{}] * len(track), fps, 1920, 1080)
        assert len(infer_kinematic_bounces(scene)) == found, f"{fps:.0f} FPS"


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
    assert contacts[0].ball == ball_point(track[contacts[0].frame])  # the ball sample it was judged at
    assert detect_player_contacts(CourtScene(track, far, 30.0, 1920, 1080)) == []


def test_gap_contact_skips_a_player_without_a_box():
    """A short reversing gap where one tracked player has no box this frame: that player is
    skipped (as nearest_player and inside_player do) and the contact goes to the other one."""
    track = [FrameResult(cx=100.0 + 20 * frame, cy=100.0, conf=0.9, source="det") for frame in range(10)]
    track += [None] * 4                                                   # frames 10-13 hidden
    track += [FrameResult(cx=260.0 - 20 * step, cy=100.0, conf=0.9, source="det") for step in range(6)]
    boxes = [{7: None, 8: (270.0, 50.0, 330.0, 200.0)}] * len(track)
    (contact,) = detect_player_contacts(CourtScene(track, boxes, 30.0, 1920, 1080))
    assert contact._replace(ball=None) == ContactEvent(11, 300.0, 200.0, 8, 1040.0, "gap")
    # The ball sample is the incoming path extrapolated into the gap.
    assert contact.ball.position == "extrapolated"
    np.testing.assert_allclose((contact.ball.x, contact.ball.y), (320.0, 100.0), atol=1e-6)


def test_assemble_links_a_contact_to_its_confirmed_landing():
    contacts = [ContactEvent(0, 60.0, 205.0, 1, 10.0, "kink")]
    confirmed = [BounceEvent(30, 40.0, 60.0, 0.9, "model", "measurement")]
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
    # The leg says how its contact was found and where its landing came from.
    assert (leg["start_quality"], leg["start_source"]) == ("confirmed", "kink")
    assert (leg["landing_origin"], leg["landing_position"]) == ("model", "measurement")


def test_a_volley_leg_has_no_landing_provenance():
    # A near-side stroke answered at the far side close to the net, with no landing between:
    # the leg ends in a volley, which is a contact and so has no landing origin or position.
    contacts = [ContactEvent(0, 60.0, 205.0, 1, 10.0, "kink"), ContactEvent(20, 40.0, 70.0, 2, 10.0, "gap")]
    (leg,) = assemble_rally_legs(
        CourtScene([None] * 80, [{}] * 80, 30.0, 120, 220), [KEYPOINTS] * 80, contacts,
        BouncePrediction([], []), [],
    )
    assert (leg["start_frame"], leg["end_frame"], leg["outcome"], leg["quality"]) == (0, 20, "volley", None)
    assert leg["start_source"] == "kink"
    assert leg["landing_origin"] is None and leg["landing_position"] is None


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
    # A model bounce reports its origin and the track position's kind: this track has no
    # measurements, so its points are "recovered"; a measured one says "measurement".
    assert (prediction.confirmed[0].origin, prediction.confirmed[0].position) == ("model", "recovered")
    measured = [_measured(point.cx, point.cy) for point in track]
    (landing,) = predict_bounce_events(CourtScene(measured, [{}] * len(track), 30.0, 120, 220), str(bounce_model)).confirmed
    assert (landing.origin, landing.position) == ("model", "measurement")
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
    assert isinstance(events, CourtEvents) and events.computed
    assert [bounce.frame for bounce in events.bounces] == [15] and events.contacts == []
    assert events.bounce_candidates == events.bounces                # nothing scored between 0.2 and 0.45
    assert [bounce.origin for bounce in events.kinematic_bounces] == ["kink"]   # the same landing, seen as a kink
    (leg,) = events.rally_legs
    assert (leg["start_frame"], leg["end_frame"], leg["phase"], leg["outcome"]) == (0, 15, "rally", "bounce")
    assert (leg["quality"], leg["start_quality"]) == ("confirmed", "inferred")
    assert out.startswith("[court events] 1 confirmed bounce(s) (1 candidate, 1 kinematic), "
                          "0 player contact(s), 1 leg(s) (0 inferred, 0 volley)")
    # Off, the stage neither loads the model nor prints; either flag alone turns it on.
    FakeBounceModel.loaded = []
    ctx.cfg.draw_court_minimap = False
    assert pipeline._court_events(ctx, pass1, selection) == CourtEvents()
    assert FakeBounceModel.loaded == [] and capsys.readouterr().out == ""
    ctx.cfg.court_events = True
    assert pipeline._court_events(ctx, pass1, selection) == events
    ctx.cfg.bounce_model_path = None
    with pytest.raises(RuntimeError, match="bounce_model_path"):
        pipeline._court_events(ctx, pass1, selection)


# What ``--court`` adds to the golden clip's tracking JSON: the stages run and find nothing.
EMPTY_COURT_EVENTS = {"bounces": [], "bounce_candidates": [], "kinematic_bounces": [], "contacts": [], "rally_legs": []}


def test_court_flag_adds_an_empty_court_events_block_and_changes_nothing_else(
        tmp_path, monkeypatch, fake_cuda, capsys, clip, bounce_model):
    """``--court`` on the golden run: the bounce model is loaded, the minimap is drawn on the
    tracking video, and the tracking JSON gains only the (empty) ``court_events`` block."""
    plain = load_golden()
    result = run_pipeline(tmp_path, monkeypatch, clip, **FULL_OVERRIDES,
                          draw_court_minimap=True, bounce_model_path=str(bounce_model))
    out = capsys.readouterr().out
    assert FakeBounceModel.loaded == [str(bounce_model.resolve())]
    assert ("[court events] 0 confirmed bounce(s) (0 candidate, 0 kinematic), 0 player contact(s), "
            "0 leg(s) (0 inferred, 0 volley)") in out
    assert result["json"]["court_events"] == EMPTY_COURT_EVENTS
    expected = copy.deepcopy(plain["json"])
    expected["court_events"] = EMPTY_COURT_EVENTS
    assert_json_close(result["json"], expected, "--court changed more than the court_events block")
    if plain.get("render_env") == render_env():
        assert result["streams"]["detector_input"] == plain["streams"]["detector_input"]
        assert result["streams"]["tracking"] != plain["streams"]["tracking"]


def test_court_events_are_computed_and_exported_without_any_video(
        tmp_path, monkeypatch, fake_cuda, capsys, clip, bounce_model):
    """``Config.court_events`` alone: pass 2 is skipped, the stages still run and the JSON
    carries the block."""
    result = run_pipeline(tmp_path, monkeypatch, clip, **JSON_ONLY_OVERRIDES,
                          court_events=True, bounce_model_path=str(bounce_model))
    out = capsys.readouterr().out
    assert "[pass 2] Skipped (no video outputs selected)" in out and "[court events] 0 confirmed" in out
    assert result["streams"] == {} and result["result"].court_events.computed
    expected = copy.deepcopy(load_golden()["json"])
    expected["summary"]["pass2_frames_rendered"] = 0
    expected["court_events"] = EMPTY_COURT_EVENTS
    assert_json_close(result["json"], expected, "the JSON-only --court run differs from the golden")


def test_a_model_bounce_is_exported_with_its_provenance_and_its_leg(tmp_path, monkeypatch, fake_cuda, clip, bounce_model):
    """One confirmed landing on the golden clip (the bounce model replaced): the JSON's bounce
    row carries the frame's position and kind, the leg that ends in it says where its landing
    and its (rescued) contact came from, and ``RunResult.court_events`` is what the file holds."""
    from tennis_tracker import pipeline
    from tennis_tracker.export import court_events_to_json

    def one_landing(scene, model_path):
        point = ball_point(scene.per_frame[35])
        return BouncePrediction([BounceEvent(35, point.x, point.y, 0.9, "model", point.position)], [])

    monkeypatch.setattr(pipeline, "predict_bounce_events", one_landing)
    result = run_pipeline(tmp_path, monkeypatch, clip, **JSON_ONLY_OVERRIDES,
                          court_events=True, bounce_model_path=str(bounce_model))
    block, row = result["raw_json"]["court_events"], result["raw_json"]["frames"][35]
    assert row["present"] and row["position_kind"] == "filtered"            # the refit moved every detected row
    assert block["bounces"] == [{"frame": 35, "x": row["x"], "y": row["y"], "score": 0.9,
                                 "origin": "model", "position": "filtered"}]
    assert block["bounce_candidates"] == [] and block["contacts"] == []
    (leg,) = block["rally_legs"]
    assert leg["end_frame"] == 35 and leg["outcome"] == "bounce" and leg["start_frame"] < 35
    assert (leg["quality"], leg["start_quality"], leg["start_source"]) == ("confirmed", "inferred", "inferred")
    assert (leg["landing_origin"], leg["landing_position"]) == ("model", "filtered")
    assert all(isinstance(leg[key], list) and len(leg[key]) == 2 for key in ("start", "end"))
    events = result["result"].court_events
    assert events.computed and [bounce.frame for bounce in events.bounces] == [35]
    assert json.loads(json.dumps(court_events_to_json(events))) == block
