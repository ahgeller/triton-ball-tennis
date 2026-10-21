"""finetune/labeling/label_frame_view.py: the single-frame picture and HUD over a synthetic tool -
what the text lines say, what a render records for the mouse, and what a click on a chip or the
inset does. Frames are blank arrays."""
from types import SimpleNamespace

import numpy as np

from finetune.labeling.label_frame_view import INSET_SHARE, THUMBNAILS_KEPT
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


def tool_over(*sources):
    tool = Tool(list(sources or [source("a")]))
    tool.frame = lambda index: np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    tool.follow_ball()
    return tool


def test_hud_place_names_the_clip_the_run_and_the_walk():
    tool = tool_over(source("a"), source("b"))
    view = tool.frame_view
    tool.cursor = 7
    assert view.hud_place() == "b [2/6]  clip 2/2  run 8/12"
    tool.walk, tool.walk_pos = [1, 7, 9], 1
    assert view.hud_place().endswith("UNCERTAIN ONLY 2/3 (3 left)")
    assert tool_over(source("a")).frame_view.hud_place() == "a [1/6]"


def test_hud_state_reads_the_label_and_the_suggestion():
    tool = tool_over()
    view = tool.frame_view
    row = tool.rows[0]
    assert view.hud_state(row, None, None) == "(100.0, 80.0)"
    offer = {"x": 103.0, "y": 84.0, "trusted": True}
    assert view.hud_state(row, offer, offer) == "HINT (103.0, 84.0) sure  grey draft 15.0px off"
    row.visible = False
    assert view.hud_state(row, offer, {**offer, "trusted": False}) == "HINT (103.0, 84.0) UNSURE, look closely  draft: no ball"
    assert view.hud_state(row, offer, None) == "NOT VISIBLE  hint: the video shows a ball here (y takes it)"
    row.visible = True
    assert view.hud_state(row, {"x": 100.0, "y": 80.0, "trusted": False}, None) == "(100.0, 80.0)  hint: here (unsure)"


def test_hud_lines_and_counts_follow_the_rows():
    tool = tool_over()
    view = tool.frame_view
    tool.rows[0].reviewed = True
    tool.rows[1].suspect = True
    line1, line2 = view.hud_lines(tool.rows[0], None, None)
    assert line1.startswith("a [1/6]  frame 0  (100.0, 80.0)  from csv 0.80  ok  zoom x1.0")
    assert line2.startswith("reviewed 1/6  uncertain left 1  flagged left 1   SPACE=good  y=take hint")   # a suspect row is uncertain
    tool.dirty = True
    assert "*unsaved*" in view.hud_lines(tool.rows[0], None, None)[0]
    assert "SPACE=take hint  x=keep draft" in view.hud_lines(tool.rows[0], None, {"x": 1.0, "y": 1.0, "trusted": True})[1]


def test_render_records_the_chips_and_the_inset_for_the_mouse():
    tool = tool_over()
    tool.fit_view(1280, 720)
    tool.zoom = tool.zoom_target = 1.0
    tool.follow_ball()
    view = tool.frame_view
    canvas = tool.render()
    assert canvas.shape == (tool.view_h + tool.hud_h, tool.view_w, 3)
    assert view.centre_button and view.filter_button and view.inset_button and len(view.step_buttons) == 5
    assert all(box[1] >= tool.view_h for box, _ in view.step_buttons)        # on the bar, under the picture
    assert view.inset_box is None                                            # no inset at x1
    tool.zoom = tool.zoom_target = 4.0
    tool.follow_ball()
    tool.render()
    left, top, right, bottom = view.inset_box
    assert right - left + 1 == int(tool.view_w * INSET_SHARE) and bottom < tool.view_h


def test_clicks_on_the_chips_and_the_inset_do_what_they_say():
    tool = tool_over()
    tool.fit_view(1280, 720)
    tool.zoom = tool.zoom_target = 4.0
    tool.follow_ball()
    tool.render()
    view = tool.frame_view
    box, jump = view.step_buttons[-1]                                       # x1, the leftmost chip
    view.click_hud(box[0], box[1])
    assert tool.step == jump
    view.click_hud(view.centre_button[0], view.centre_button[1])
    assert not tool.follow
    view.click_hud(view.inset_button[0], view.inset_button[1])
    assert not tool.inset
    tool.inset = True
    left, top, right, bottom = view.inset_box
    assert view.inset_hit(left, top) and not view.inset_hit(left - 1, top)
    view.click_inset(right, bottom)
    assert tool.center[0] > WIDTH * 0.9 and tool.center[1] > HEIGHT * 0.9   # the inset's far corner is the frame's


def test_thumbnails_are_cached_by_frame_and_width_and_bounded():
    tool = tool_over()
    view = tool.frame_view
    image = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    first = view.thumbnail(0, image, 64, 36)
    assert first.shape == (36, 64, 3) and view.thumbnail(0, image, 64, 36) is first
    for frame in range(1, THUMBNAILS_KEPT + 1):
        view.thumbnail(frame, image, 64, 36)
    assert len(view.thumbs) == THUMBNAILS_KEPT and ("a", 0, 64) not in view.thumbs   # the oldest went first
