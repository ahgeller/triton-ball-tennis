"""tennis_tracker.export: RunResult's frame rows are the tracking JSON's frames table, the
JSON keeps the measurement apart from the fitted position, and the court events are written
with their provenance."""
import json
from pathlib import Path
from types import SimpleNamespace
from typing import get_args

from ball_in_play_selector import FrameResult, models
from ball_in_play_selector.core import frame_result
from tennis_tracker import tracking_schema
from tennis_tracker.court import court_scene
from tennis_tracker.court.court_scene import BallPoint, BounceEvent, ContactEvent
from tennis_tracker.export import RunResult, court_events_to_json, frame_result_to_json, frame_row_to_result
from tennis_tracker.run_context import CourtEvents


def test_run_result_frame_rows_match_the_tracking_json_layout():
    hit = FrameResult(cx=10.0, cy=20.0, conf=0.9, source="det")
    result = RunResult(
        per_frame=[None, hit], chosen_track=None, all_tracks=[], frame_count=3,
        court_keypoints=[[1.0, 2.0], None], player_boxes=[{1: [0.0, 0.0, 4.0, 4.0]}, {}],
        last_valid_court_keypoints=None, elapsed_sec=1.5, timing={}, pass2_frames_rendered=0,
        outputs={"tracking_json": Path("out.json")},
    )
    rows = result.frame_rows()
    assert [row["frame"] for row in rows] == [0, 1, 2]           # one row per decoded frame
    assert rows[0]["present"] is False and rows[0]["court_keypoints"] == [1.0, 2.0]
    assert rows[0]["player_boxes"] == {"1": [0.0, 0.0, 4.0, 4.0]}   # JSON-safe keys
    assert rows[1]["present"] is True and (rows[1]["x"], rows[1]["y"], rows[1]["source"]) == (10.0, 20.0, "det")
    assert rows[1]["court_keypoints"] is None
    assert "court_keypoints" not in rows[2] and "player_boxes" not in rows[2]


def test_export_preserves_measurement_after_position_changes():
    detection = SimpleNamespace(cx=100., cy=103., conf=.9, x1=98, y1=101, x2=102, y2=105)
    result = frame_result(100, 103, "det", detection=detection)
    assert frame_result_to_json(0, result)["position_kind"] == "measurement"
    result.cy = 100.5
    row = frame_result_to_json(0, result)
    assert row["measurement"]["y"] == 103.
    assert row["y"] == 100.5
    assert row["position_kind"] == result.position_kind == "filtered"
    assert frame_result_to_json(1, frame_result(5, 6, "interp"))["position_kind"] == "recovered"


def test_frame_row_to_result_reverses_frame_result_to_json():
    """A scorer rebuilds the selection from the JSON's rows: every field the row holds comes
    back, and writing the rebuilt result gives the same row."""
    detection = SimpleNamespace(cx=100., cy=103., conf=.9, x1=98, y1=101, x2=102, y2=105)
    detected = frame_result(100.5, 103.25, "det", detection=detection, search_radius=12.0)
    moved = frame_result(5., 6., "motion", motion=SimpleNamespace(bbox=(1., 2., 3., 4.)), search_radius=30.0)
    filled = frame_result(7., 8., "interp")
    for result in (detected, moved, filled):
        row = frame_result_to_json(3, result)
        back = frame_row_to_result(row)
        assert back == result and back.position_kind == result.position_kind
        assert frame_result_to_json(3, back) == row
    assert frame_row_to_result({"frame": 4, "present": False}) is None


def test_the_schema_s_literals_are_the_selector_s_and_the_court_s():
    """``tracking_schema`` imports only ``typing``, so it spells the position kinds and the
    event origins and positions itself; each must name the same values as its source."""
    assert get_args(tracking_schema.PositionKind) == get_args(models.PositionKind)
    assert get_args(tracking_schema.EventOrigin) == get_args(court_scene.EventOrigin)
    assert get_args(tracking_schema.EventPosition) == get_args(court_scene.EventPosition)


def test_court_events_block_is_empty_until_computed_and_then_holds_every_list():
    assert court_events_to_json(CourtEvents()) == {}
    computed = CourtEvents(computed=True)
    assert court_events_to_json(computed) == {"bounces": [], "bounce_candidates": [], "kinematic_bounces": [],
                                              "contacts": [], "rally_legs": []}
    computed.bounces.append(BounceEvent(12, 1.0, 2.0, 0.7))
    computed.kinematic_bounces.append(BounceEvent(40, 3.0, 4.0, 5.0, "gap", "extrapolated"))
    computed.contacts.append(ContactEvent(3, 4.0, 5.0, 1, 9.0, "kink", BallPoint(6.0, 7.0, "measurement")))
    computed.contacts.append(ContactEvent(30, 4.0, 5.0, "p2", 1040.0, "gap"))
    computed.rally_legs.append({"start_frame": 3, "end_frame": 12, "start": (0.5, 0.9), "end": (0.3, 0.2),
                                "phase": "rally", "outcome": "bounce", "quality": "confirmed",
                                "start_quality": "confirmed", "start_source": "kink",
                                "landing_origin": "model", "landing_position": "filtered"})
    block = court_events_to_json(computed)
    assert block["bounces"] == [{"frame": 12, "x": 1.0, "y": 2.0, "score": 0.7, "origin": "model", "position": "filtered"}]
    assert block["kinematic_bounces"][0]["origin"] == "gap" and block["kinematic_bounces"][0]["position"] == "extrapolated"
    assert block["contacts"][0] == {"frame": 3, "x": 4.0, "y": 5.0, "player_id": 1, "score": 9.0, "source": "kink",
                                    "ball": {"x": 6.0, "y": 7.0, "position": "measurement"}}
    assert block["contacts"][1]["player_id"] == "p2" and block["contacts"][1]["ball"] is None
    assert block["rally_legs"][0]["start"] == [0.5, 0.9] and block["rally_legs"][0]["landing_origin"] == "model"
    assert json.loads(json.dumps(block)) == block                        # nothing the JSON writer cannot take
