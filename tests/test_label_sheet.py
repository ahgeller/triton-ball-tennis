"""finetune/labeling/label_sheet.py: the trajectory sheet as a state machine over a synthetic tool -
which rows it shows, what a click and a key do, and what each render records. Frames are blank
arrays; no workspace file is read."""
from types import SimpleNamespace

import cv2
import numpy as np

from finetune.labeling.label_store import Row
from finetune.labeling.label_tool import Tool
from finetune.labeling.label_sheet import SHEET_ROWS
from finetune.labeling.label_window import KEY_TAB

WIDTH, HEIGHT = 640, 360


def source(clip, count):
    rows = [Row(i, 100.0 + 5 * i, 80.0 + 2 * i, True, "csv", 0.0) for i in range(count)]
    for row in rows:
        row.clip = clip
    capture = SimpleNamespace(release=lambda: None, isOpened=lambda: False)
    return SimpleNamespace(clip=clip, rows=rows, width=WIDTH, height=HEIGHT, frame_cache={}, dirty=False, cursor=0,
                           start=0, missing=set(), saved_cursor=0, capture=capture, position=0, release=lambda: None)


def tool_over(*sources):
    tool = Tool(list(sources))
    tool.frame = lambda index: np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    tool.follow_ball()
    return tool


# --------------------------------------------------------------------------- the trajectory sheet

def test_sheet_centres_its_stretch_on_the_cursor_and_pages_within_the_clip():
    tool = tool_over(source("a", 150), source("b", 70))
    sheet = tool.sheet_view
    tool.cursor = 70
    sheet.toggle()
    assert sheet.active and sheet.start == 70 - SHEET_ROWS // 2
    sheet.page(1)
    assert sheet.start == 150 - SHEET_ROWS and tool.cursor == sheet.start
    sheet.page(1)
    assert sheet.start == 150 - SHEET_ROWS
    sheet.page(-1)
    assert sheet.start == 150 - 2 * SHEET_ROWS
    tool.cursor = 160                                        # clip b: 70 rows, so two stretches overlap
    sheet.toggle()
    sheet.toggle()
    assert sheet.start == 150
    sheet.page(1)
    assert sheet.start == 160                                # the last full stretch of clip b
    sheet.page(1)
    assert sheet.start == 160


def test_sheet_toggle_leaves_the_grid_and_the_grid_leaves_the_sheet():
    tool = tool_over(source("a", 100))
    tool.grid_view.toggle()
    tool.sheet_view.toggle()
    assert tool.sheet_view.active and not tool.grid_view.active
    tool.grid_view.toggle()
    assert tool.grid_view.active and not tool.sheet_view.active


def test_sheet_render_records_a_point_per_visible_label_and_caches():
    tool = tool_over(source("a", 100))
    sheet = tool.sheet_view
    tool.cursor = 30
    sheet.toggle()
    tool.rows[35].visible = False
    canvas = sheet.render()
    assert canvas.shape == (tool.view_h + tool.hud_h, tool.view_w, 3)
    indices = [p[0] for p in sheet.points]
    assert indices == [i for i in range(sheet.start, sheet.start + SHEET_ROWS) if i != 35]
    assert all(0 <= x < tool.view_w and 0 <= y < tool.view_h for _, x, y in sheet.points)
    assert sheet.render() is canvas and not tool.render_fresh
    tool.cursor = 31
    assert sheet.render() is not canvas


def test_sheet_click_opens_the_nearest_point_and_keys_only_page_or_leave():
    tool = tool_over(source("a", 100))
    sheet = tool.sheet_view
    tool.cursor = 30
    sheet.toggle()
    sheet.render()
    index, x, y = sheet.points[10]
    sheet.on_mouse(cv2.EVENT_LBUTTONDOWN, x + 1, y - 1)
    assert not sheet.active and tool.cursor == index and not tool.rows[index].settled
    sheet.toggle()
    sheet.render()
    sheet.on_mouse(cv2.EVENT_LBUTTONDOWN, tool.view_w - 1, tool.view_h - 1)
    assert sheet.active                                      # far from every point: nothing
    start = sheet.start
    sheet.handle_key(0, 1)
    assert sheet.start > start
    sheet.handle_key(ord("v"), 0)
    assert all(row.visible for row in tool.rows)             # the sheet edits nothing
    sheet.handle_key(KEY_TAB, 0)
    assert not sheet.active and tool.grid_view.active
    tool.grid_view.active = False
    sheet.toggle()
    sheet.handle_key(ord("w"), 0)
    assert not sheet.active and not tool.grid_view.active
