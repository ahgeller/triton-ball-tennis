"""finetune/labeling/label_mouse.py: the single-frame view's mouse over a synthetic tool - the
wheel's notches, which events are guarded, and what each button does."""
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from finetune.labeling.label_mouse import MouseHandler
from finetune.labeling.label_store import Row
from finetune.labeling.label_tool import Tool

WIDTH, HEIGHT = 640, 360


def source(clip, count=6):
    rows = [Row(i, 100.0 + 5 * i, 80.0 + 2 * i, True, "csv", 0.8) for i in range(count)]
    for row in rows:
        row.clip = clip
    capture = SimpleNamespace(release=lambda: None, isOpened=lambda: False)
    return SimpleNamespace(clip=clip, rows=rows, width=WIDTH, height=HEIGHT, frame_cache={}, dirty=False, cursor=0,
                           start=0, missing=set(), saved_cursor=0, capture=capture, position=0, release=lambda: None)


def shown_tool():
    """A tool whose view is up to date, so clicks are not held back."""
    tool = Tool([source("a")])
    tool.frame = lambda index: np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    tool.fit_view(1280, 720)
    tool.follow_ball()
    with patch.object(cv2, "imshow"):
        tool.show()
    return tool


def test_wheel_notches_come_from_the_packed_delta_with_a_fallback():
    assert MouseHandler.wheel_notches(120 << 16) == 1.0
    assert MouseHandler.wheel_notches(-60 << 16) == -0.5
    assert MouseHandler.wheel_notches(900 << 16) == 4.0                     # capped
    assert MouseHandler.wheel_notches(1) == 1.0 and MouseHandler.wheel_notches(0) == -1.0   # no packed delta: one notch


def test_guarded_is_a_pure_test_of_calibrating_the_overview_and_a_lagging_view():
    tool = shown_tool()
    pointer = tool.pointer
    assert not pointer.guarded(cv2.EVENT_LBUTTONDOWN) and not pointer.guarded(cv2.EVENT_MOUSEWHEEL)
    tool.mouse.calibrating = True
    assert pointer.guarded(cv2.EVENT_LBUTTONDOWN) and pointer.guarded(cv2.EVENT_MOUSEWHEEL)
    assert not pointer.guarded(cv2.EVENT_MOUSEMOVE)                          # moves are what calibration reads
    tool.mouse.calibrating = False
    tool.grid = True                                                         # the overview: no wheel, clicks are its own
    assert pointer.guarded(cv2.EVENT_MOUSEWHEEL) and not pointer.guarded(cv2.EVENT_LBUTTONDOWN)
    tool.grid = False
    tool.cursor = 3                                                          # the view lags the cursor
    assert pointer.guarded(cv2.EVENT_LBUTTONDOWN) and not pointer.guarded(cv2.EVENT_MOUSEMOVE)


def test_every_event_notes_the_pointer_and_an_unused_wheel_is_settled():
    tool = shown_tool()
    tool.on_mouse(cv2.EVENT_MOUSEMOVE, 7, 9, 0, None)
    assert tool.mouse.pointer == (7, 9)                                      # note_pointer, before any guard
    with patch.object(tool.viewport, "absorb_wheel", return_value=True), patch.object(tool.viewport, "repay_wheel") as repay:
        tool.on_mouse(cv2.EVENT_MOUSEWHEEL, 10, 10, 120 << 16, None)         # the tool's own wheel: swallowed
    assert tool.mouse.pointer == (7, 9) and not repay.called
    tool.mouse.calibrating = True
    with patch.object(tool.viewport, "repay_wheel") as repay:
        tool.on_mouse(cv2.EVENT_MOUSEWHEEL, 10, 10, 120 << 16, None)         # guarded: settled, not zoomed
    repay.assert_called_once_with(120)
    assert tool.mouse.pointer == (10, 10) and tool.zoom_target == 1.0


def test_a_left_click_places_the_ball_and_moves_on_a_right_click_recentres():
    tool = shown_tool()
    x, y = tool.to_view(150.0, 90.0)
    tool.on_mouse(cv2.EVENT_LBUTTONDOWN, x, y, 0, None)
    assert tool.rows[0].edited and abs(tool.rows[0].x - 150.0) < 1.0 and abs(tool.rows[0].y - 90.0) < 1.0
    assert tool.cursor == 1
    with patch.object(cv2, "imshow"):
        tool.show()
    x, y = tool.to_view(20.0, 30.0)
    tool.on_mouse(cv2.EVENT_RBUTTONDOWN, x, y, 0, None)
    assert abs(tool.center[0] - 20.0) < 1.0 and abs(tool.center[1] - 30.0) < 1.0 and tool.cursor == 1
    tool.on_mouse(cv2.EVENT_LBUTTONDOWN, -5000, 10, 0, None)                 # the black beyond the frame
    assert tool.cursor == 1


def test_a_click_on_the_bar_goes_to_the_chips_and_a_missing_frame_takes_no_label():
    tool = shown_tool()
    view = tool.frame_view
    tool.on_mouse(cv2.EVENT_LBUTTONDOWN, view.centre_button[0], view.centre_button[1], 0, None)
    assert not tool.follow and tool.cursor == 0
    tool.source.missing.add(tool.rows[0].frame)
    tool.pointer.place_click(10, 10)
    assert not tool.rows[0].edited and tool.cursor == 0
