"""finetune/labeling/label_cursor.py: the OS pointer placer on a fake host, without user32 (the
Windows calls are patched out, so nothing on this machine moves)."""
import time
from types import SimpleNamespace
from unittest.mock import patch

from finetune.labeling import label_cursor, label_window
from finetune.labeling.label_cursor import WHEEL_QUIET, CursorPlacer


def placer():
    return CursorPlacer(SimpleNamespace(view_w=640, view_h=360, hud_h=58), lambda ms: None)


def test_reports_are_counted_and_a_rebuilt_window_forgets_its_map():
    mouse = placer()
    assert mouse.pointer is None and mouse.pointer_seq == 0 and mouse.map_stale
    mouse.report(10, 20)
    mouse.report(11, 20)
    assert mouse.pointer == (11, 20) and mouse.pointer_seq == 2
    mouse.cursor_map, mouse.map_stale = (2.0, 2.0, 0.0, 0.0), False
    mouse.forget_window()
    assert mouse.cursor_map is None and mouse.map_stale


def test_scrolling_holds_only_just_after_a_wheel_notch():
    mouse = placer()
    assert not mouse.scrolling()
    mouse.wheel_at = time.monotonic()
    assert mouse.scrolling()
    mouse.wheel_at = time.monotonic() - WHEEL_QUIET - 1.0
    assert not mouse.scrolling()


def test_screen_for_needs_both_a_window_and_a_measured_map():
    mouse = placer()
    with patch.object(label_window, "window_client", return_value=None):
        assert mouse.screen_for(10.0, 10.0) is None
    with patch.object(label_window, "window_client", return_value=(100, 200, 1280, 720)):
        assert mouse.screen_for(10.0, 10.0) is None                     # no map yet
        mouse.cursor_map = (2.0, 2.0, 1.0, 67.0)
        assert mouse.screen_for(10.0, 10.0) == (121, 287)


def test_await_pointer_returns_the_next_report_or_gives_up():
    mouse = placer()
    mouse.pump = lambda ms: mouse.report(5, 6)
    assert mouse.await_pointer(mouse.pointer_seq, 0.5) == (5, 6)
    mouse.pump = lambda ms: None
    assert mouse.await_pointer(mouse.pointer_seq, 0.01) is None


def test_calibrate_does_nothing_without_user32():
    mouse = placer()
    with patch.object(label_window, "_user32", None):
        mouse.calibrate()
    assert mouse.cursor_map is None and mouse.map_stale and not mouse.calibrating
    assert label_cursor.PLACEMENT_WAIT > 0 and label_cursor.SCALE_RANGE[0] < 1.0 < label_cursor.SCALE_RANGE[1]
