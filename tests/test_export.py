"""tennis_tracker.export: RunResult's frame rows are the tracking JSON's frames table, and the
JSON keeps the measurement apart from the fitted position."""
from pathlib import Path
from types import SimpleNamespace
from typing import get_args

from ball_in_play_selector import FrameResult, models
from ball_in_play_selector.core import frame_result
from tennis_tracker import tracking_schema
from tennis_tracker.export import RunResult, frame_result_to_json


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


def test_the_schema_s_position_kinds_are_the_frame_result_s():
    """``tracking_schema`` imports only ``typing``, so it spells ``PositionKind`` itself; the
    two must name the same values."""
    assert get_args(tracking_schema.PositionKind) == get_args(models.PositionKind)
