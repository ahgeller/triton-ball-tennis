"""finetune/labeling/label_grid.py: the grid as a state machine over a synthetic tool - which rows a
page holds, where the selection goes, what a click and a key do, and what each render records.
Frames are blank arrays; no workspace file is read."""
import cv2
import pytest

from finetune.labeling.label_grid import GRID_SIZES
from finetune.labeling.label_window import KEYS_CONFIRM
from tests._helpers import fake_source as source, fake_tool as tool_over


# --------------------------------------------------------------------------- the grid's rows

def test_grid_fills_a_page_from_the_cursor_inside_one_clip():
    tool = tool_over(source("a", 12), source("b", 12))
    grid = tool.grid_view
    tool.session.cursor = 9
    grid.toggle()
    assert grid.active and grid.indices == [9, 10, 11]       # clip a ends at 11
    assert grid.centers[9] == (tool.session.rows[9].x, tool.session.rows[9].y) and tool.session.cursor == 9
    grid.page(1)
    assert grid.indices == list(range(12, 20))               # the next page starts clip b
    grid.page(-1)
    assert grid.indices == [4, 5, 6, 7, 8, 9, 10, 11]        # a page back, inside clip a
    grid.page(-1)
    assert grid.indices == list(range(8))
    grid.page(-1)
    assert grid.indices == list(range(8))                    # the start: unchanged


def test_grid_follows_the_uncertain_walk_when_there_is_one():
    tool = tool_over(source("a", 30))
    tool.session.walk, tool.session.walk_pos = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29], 0
    grid = tool.grid_view
    grid.prepare(5)
    assert grid.indices == [5, 8, 11, 14, 17, 20, 23, 26] and tool.session.walk_pos == 1
    grid.page(1)
    assert grid.indices == [29] and tool.session.cursor == 29 and tool.session.walk_pos == 9
    grid.page(1)
    assert grid.indices == [29]                              # nothing past the walk's end


def test_grid_select_and_step_turn_pages_at_the_ends():
    tool = tool_over(source("a", 20))
    grid = tool.grid_view
    grid.prepare(0)
    grid.select(3)
    assert tool.session.cursor == 3
    grid.select(99)
    assert tool.session.cursor == 7                                  # clamped to the page
    grid.step(1)
    assert grid.indices == list(range(8, 16)) and tool.session.cursor == 8
    grid.step(-1)
    assert grid.indices == list(range(8)) and tool.session.cursor == 7
    grid.select(-5)
    assert tool.session.cursor == 0
    grid.step(-1)
    assert grid.indices == list(range(8)) and tool.session.cursor == 0


def test_grid_size_toggles_and_keeps_the_selection():
    tool = tool_over(source("a", 40))
    grid = tool.grid_view
    grid.toggle()
    grid.select(6)
    grid.toggle_size()
    assert grid.size == GRID_SIZES[1] and len(grid.indices) == 16 and tool.session.cursor == 6
    grid.toggle_size()
    assert grid.size == GRID_SIZES[0]
    grid.size = 5                                            # a stray setting: back to the first size
    grid.toggle_size()
    assert grid.size == GRID_SIZES[0]


def test_grid_confirms_only_a_page_it_rendered():
    tool = tool_over(source("a", 20))
    grid = tool.grid_view
    grid.toggle()
    grid.confirm_page()
    assert not any(row.reviewed for row in tool.session.rows)        # nothing drawn yet
    grid.render()
    grid.confirm_page()
    assert all(tool.session.rows[i].reviewed for i in range(8)) and not tool.session.rows[8].reviewed
    assert grid.indices == list(range(8, 16))                # and the page turned


# --------------------------------------------------------------------------- the grid's picture and hand

def test_grid_render_lays_out_tiles_and_caches_until_something_changes():
    tool = tool_over(source("a", 20))
    grid = tool.grid_view
    grid.toggle()
    canvas = grid.render()
    assert canvas.shape == (tool.session.view_h + tool.session.hud_h, tool.session.view_w, 3)
    assert [tile[0] for tile in grid.tiles] == list(range(8))
    boxes = [box for _, box, *_ in grid.tiles]
    assert len({box[1] for box in boxes}) == 2 and all(box[3] < tool.session.view_h for box in boxes)
    assert grid.render() is canvas
    tool.session.rows[3].reviewed = True
    assert grid.render() is not canvas
    grid.invalidate()
    assert grid.render() is not grid.cache_key


def test_grid_click_selects_a_tile_places_a_ball_or_leaves_the_grid():
    tool = tool_over(source("a", 20))
    grid = tool.grid_view
    grid.toggle()
    grid.render()
    index, box, ox, oy, scale = grid.tiles[5]
    grid.on_mouse(cv2.EVENT_MOUSEMOVE, box[0] + 10, box[1] + 10)
    assert tool.session.cursor == 0                                  # a move is not a click
    grid.on_mouse(cv2.EVENT_LBUTTONDOWN, box[0] + 10, box[1] + 10)
    assert tool.session.cursor == index and tool.session.rows[index].edited
    fx, fy = ox + (10 + 0.5) / scale, oy + (10 + 0.5) / scale
    assert (tool.session.rows[index].x, tool.session.rows[index].y) == (pytest.approx(fx), pytest.approx(fy))
    tool.session.rows[7].visible = False
    grid.render()
    _, box7, *_ = grid.tiles[7]                              # the whole frame, letterboxed: click its middle
    grid.on_mouse(cv2.EVENT_LBUTTONDBLCLK, (box7[0] + box7[2]) // 2, (box7[1] + box7[3]) // 2)
    assert tool.session.rows[7].visible and grid.centers[7] == (tool.session.rows[7].x, tool.session.rows[7].y)   # zoomed on the new ball
    grid.on_mouse(cv2.EVENT_RBUTTONDOWN, box[0] + 10, box[1] + 10)
    assert not grid.active and tool.session.cursor == index


def test_grid_keys_select_page_and_zoom():
    tool = tool_over(source("a", 40))
    grid = tool.grid_view
    grid.toggle()
    assert grid.handle_key(ord("3"), 0) and tool.session.cursor == 2
    assert grid.handle_key(ord("."), 0) and tool.session.cursor == 3
    assert grid.handle_key(ord(","), 0) and tool.session.cursor == 2
    assert grid.handle_key(0, 1) and grid.indices == list(range(8, 16))
    assert grid.handle_key(ord("+"), 0) and grid.zoom > 1.0
    assert grid.handle_key(ord("-"), 0) and grid.zoom == pytest.approx(1.0)
    assert grid.handle_key(ord("v"), 0) and not tool.session.rows[tool.session.cursor].visible
    assert grid.handle_key(ord("r"), 0)                      # swallowed: no batch accept off-screen
    assert not grid.handle_key(ord("t"), 0)                  # shared with the single frame
    grid.render()
    assert grid.handle_key(KEYS_CONFIRM[0], 0) and grid.indices == list(range(16, 24))
