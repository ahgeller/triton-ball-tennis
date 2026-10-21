"""tennis_tracker/run_context.py: Timing, RunContext.needs_pass2_outputs, Selection and RunResult."""
from pathlib import Path
from types import SimpleNamespace

from ball_in_play_selector import FrameResult

from tennis_tracker.config import Config
from tennis_tracker.run_context import RunContext, RunResult, Selection, Timing


def test_disabled_timing_is_a_no_op():
    timing = Timing(False)
    assert timing.start() == 0.0
    assert timing.add("k", 0.0) == 0.0
    timing.set("k", 3.0)
    assert timing.values == {}
    assert timing.get("k") == 0.0


def test_enabled_timing_accumulates_sets_and_reads():
    timing = Timing(True)
    t0 = timing.start()
    assert t0 > 0.0
    first = timing.add("k", t0)
    second = timing.add("k", t0)
    assert first > 0.0 and second >= first
    assert timing.values["k"] == first + second
    timing.set("k", 2.5)
    assert timing.get("k") == 2.5
    assert timing.get("unknown") == 0.0
    assert set(timing.values) == {"k"}


def _needs_pass2(**flags) -> bool:
    return RunContext.needs_pass2_outputs.fget(SimpleNamespace(cfg=Config(**flags)))


def test_pass2_outputs_are_needed_for_any_video_output():
    assert _needs_pass2(save_tracking_video=False, save_motion_debug=True, save_track_debug_video=False) is True
    assert _needs_pass2(save_tracking_video=True, save_motion_debug=False, save_track_debug_video=False) is True
    assert _needs_pass2(save_tracking_video=False, save_motion_debug=False, save_track_debug_video=True) is True
    assert _needs_pass2(save_tracking_video=False, save_motion_debug=False, save_track_debug_video=False) is False


def test_selection_counts_the_filled_frames():
    obj = SimpleNamespace()
    selection = Selection(per_frame=[None, obj, None, obj], chosen_track=None, all_tracks=[],
                          court_polygon=None, last_court_keypoints=None, display_path={})
    assert selection.filled_frames == 2
    assert Selection([], None, [], None, None, {}).filled_frames == 0


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
