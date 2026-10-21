"""tennis_tracker/run_context.py: Timing, RunContext.needs_pass2_outputs and Selection."""
from types import SimpleNamespace

from tennis_tracker.config import Config
from tennis_tracker.run_context import RunContext, Selection, Timing


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
