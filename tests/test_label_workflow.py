"""Headless label-tool tests on synthetic frames; never open workspace labels."""
import io
import json
from contextlib import ExitStack, contextmanager, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import DEFAULT, Mock, call, patch

import cv2
import numpy as np
import pytest

from finetune.labeling import label_window as win
from finetune.labeling.label_draw import ZOOM_STEP
from finetune.labeling.label_grid import GridView
from finetune.labeling.label_sheet import SheetView
from finetune.labeling.label_store import Row, apply_flags, apply_review, load_rows, save_flags, save_review
from finetune.labeling.label_tool import Tool
from finetune.labeling.label_window import KEYS_NEXT, KEYS_PREV
from tests._helpers import ball_frames, fake_source, fake_tool, write_frames


def source(clip, count=10, width=1280, height=720):
    """This file's clips: 720p by default, the balls 11 px apart along x (see fake_source)."""
    return fake_source(clip, count, width, height, origin=(30.25, 24.75), step=(11.0, 3.0))


def make_tool(*sources):
    """A headless Tool over `sources` (one 10-row clip by default), centred on its first ball."""
    return fake_tool(*(sources or [source("test")]))


# --------------------------------------------------------------------------- the single-frame view

def test_zoomed_next_frame_discards_old_anchor_and_cursor_report():
    tool = make_tool()
    tool.session.zoom, tool.session.zoom_target = 5., 9.
    tool.session.anchor = ((50., 80.), (120., 250.))
    tool.mouse.aim = ("old cursor feedback",)
    tool.move(1)
    assert tool.session.anchor is None
    assert tool.mouse.aim is None
    for _ in range(10):
        tool.session.step_zoom()
        x, y = tool.session.to_view(tool.session.rows[1].x, tool.session.rows[1].y)
        assert (x, y) == (round(tool.session.view_w/2), round(tool.session.view_h/2))
    assert tool.session.zoom == 9.


def test_queued_click_cannot_edit_an_unshown_next_frame():
    tool = make_tool()
    tool.session.zoom = tool.session.zoom_target = 8.
    tool.follow_ball()
    with patch.object(cv2, "imshow"):
        tool.show()
    tool.pointer.on_mouse(cv2.EVENT_LBUTTONDOWN, tool.session.view_w//2, tool.session.view_h//2, 0, None)
    assert tool.session.cursor == 1
    tool.pointer.on_mouse(cv2.EVENT_LBUTTONDBLCLK, tool.session.view_w//2, tool.session.view_h//2, 0, None)
    assert tool.session.cursor == 1
    assert not tool.session.rows[1].reviewed


def test_crop_click_maps_to_subpixel_source_coordinates():
    tool = make_tool()
    tool.grid_view.toggle()
    tool.grid_view.render()
    index, box, ox, oy, scale = tool.grid_view.tiles[0]
    # Pick a valid point nearer the middle when the crop runs over an edge.
    x, y = (box[0]+box[2])//2 + 3, (box[1]+box[3])//2 + 2
    expected = (ox+(x+.5-box[0])/scale, oy+(y+.5-box[1])/scale)
    tool.grid_view.on_mouse(cv2.EVENT_LBUTTONDOWN, x, y)
    assert tool.session.rows[index].x == pytest.approx(expected[0], abs=5e-8)
    assert tool.session.rows[index].y == pytest.approx(expected[1], abs=5e-8)
    assert tool.grid_view.indices == list(range(8))
    assert tool.session.rows[0].edited


def test_resize_drops_stale_zoom_anchor_and_keeps_target_centered():
    tool = make_tool()
    tool.session.zoom = tool.session.zoom_target = 10.
    tool.follow_ball()
    tool.session.anchor = ((1000, 600), (10, 10))
    tool.mouse.aim = ("stale",)
    tool.fit_view(1000, 700)
    assert tool.session.anchor is None
    assert tool.mouse.aim is None
    assert tool.session.to_view(tool.session.rows[0].x, tool.session.rows[0].y) == (round(tool.session.view_w/2), round(tool.session.view_h/2))


def test_stale_wheel_still_cancels_native_window_zoom():
    tool = make_tool()
    with patch.object(tool.viewport, "repay_wheel") as repay:
        tool.pointer.on_mouse(cv2.EVENT_MOUSEWHEEL, 100, 100, 120 << 16, None)
    repay.assert_called_once_with(120)


def test_confirm_grid_requires_displayed_page_and_preserves_missing():
    tool = make_tool()
    tool.grid_view.toggle()
    tool.grid_view.confirm_page()
    assert not any(r.reviewed for r in tool.session.rows)
    tool.session.source.missing.add(2)
    tool.grid_view.render()
    tool.grid_view.confirm_page()
    assert all(r.reviewed for i, r in enumerate(tool.session.rows[:8]) if i != 2)
    assert not tool.session.rows[2].reviewed
    assert not any(r.reviewed for r in tool.session.rows[8:])
    tool.grid_view.confirm_page()  # next page has not been rendered yet
    assert not any(r.reviewed for r in tool.session.rows[8:])


def test_mixed_resolution_pages_keep_source_identity():
    tool = make_tool(source("small", 2, 640, 360), source("large", 2, 1920, 1080))
    tool.grid_view.toggle()
    assert tool.grid_view.indices == [0, 1]
    tool.grid_view.page(1)
    assert (tool.session.clip, tool.session.width, tool.session.height) == ("large", 1920, 1080)
    assert tool.grid_view.indices == [2, 3]


def test_grid_tiles_read_their_own_clip_without_moving_the_cursor():
    tool = make_tool(source("small", 3, 640, 360), source("large", 3, 1920, 1080))
    read = []
    tool.session.frame = lambda index: read.append((tool.session.source.clip, index)) or np.zeros(
        (tool.session.source.height, tool.session.source.width, 3), np.uint8)
    tool.grid_view.toggle()
    tool.grid_view.page(1)                                  # the page of clip "large": rows 3..5
    tool.grid_view.select(1)
    tool.grid_view.render()
    assert sorted(read) == [("large", 0), ("large", 1), ("large", 2)]
    assert tool.session.cursor == 4                            # rendering leaves the selection alone
    assert [tile[0] for tile in tool.grid_view.tiles] == [3, 4, 5]


def test_undo_restores_review_flags():
    tool = make_tool()
    row = tool.session.rows[0]
    row.suspect = True
    tool.set_point(90, 100)
    row.undo()
    assert not row.reviewed
    assert not row.edited
    assert row.suspect
    assert (row.x, row.y) == (30.25, 24.75)


def test_evidence_survives_resume_without_inventing_confidence(tmp_path):
    csv = tmp_path / "clip_ball.csv"
    review = csv.with_suffix(".review.json")
    csv.write_text("frame,ball_x,ball_y\nframe_001,20,30\n")
    rows = load_rows(csv, 640, 360, "csv")
    assert rows[0].conf == 0.
    assert rows[0].uncertain
    rows[0].source, rows[0].conf = "det", .43
    save_review(review, rows, 0, 4.)
    resumed = load_rows(csv, 640, 360, "csv")
    apply_review(review, resumed)
    assert (resumed[0].source, resumed[0].conf) == ("det", .43)


def test_an_edited_frame_is_a_reviewed_frame_for_the_tool_as_for_every_reader(tmp_path):
    """One vocabulary: a frame a person confirmed or edited is reviewed. An older sidecar that
    lists a frame under "edited" only reads back as reviewed too, as labels.reviewed_frames has it."""
    from finetune.data.labels import reviewed_frames
    review = tmp_path / "clip_ball.review.json"
    review.write_text(json.dumps({"reviewed": [1], "edited": [2]}))
    rows = source("clip", 4).rows
    apply_review(review, rows)
    assert [row.reviewed for row in rows] == [False, True, True, False]
    assert [row.edited for row in rows] == [False, False, True, False]
    assert reviewed_frames("clip", tmp_path) == {1, 2}
    tool = make_tool(source("clip", 4))
    tool.set_point(50.0, 40.0)                                   # an edit in the tool reviews the frame too
    assert tool.session.rows[0].reviewed and tool.session.rows[0].edited
    save_review(review, tool.session.rows, 0, 2.0)
    assert json.loads(review.read_text())["reviewed"] == [0]


def test_review_evidence_is_applied_per_frame_and_bad_evidence_is_warned_about(tmp_path):
    review = tmp_path / "clip_ball.review.json"
    review.write_text(json.dumps({"reviewed": [1], "evidence": {
        "1": {"source": "det", "conf": 0.4}, "2": {"source": "det", "conf": "bad"}}}))
    rows = source("clip", 4).rows
    out = io.StringIO()
    with redirect_stdout(out):
        state = apply_review(review, rows)
    assert state["reviewed"] == [1]
    assert (rows[1].source, rows[1].conf, rows[1].reviewed) == ("det", 0.4, True)
    assert (rows[2].source, rows[2].conf, rows[2].reviewed) == ("csv", 0., False)
    assert "[warn]" in out.getvalue()
    assert review.name in out.getvalue()
    assert "frame 2" in out.getvalue()


def test_apply_flags_counts_a_legacy_frames_list_as_check_flags(tmp_path):
    flags = tmp_path / "clip_ball.suspect.json"
    flags.write_text(json.dumps({"unplaceable": [1], "frames": [3]}))
    rows = source("clip", 5).rows
    assert apply_flags(flags, rows) == 2
    assert [row.frame for row in rows if row.suspect] == [1, 3]
    assert apply_flags(flags.with_name("absent.json"), rows) == 0
    assert not any(row.suspect for row in rows)


def test_accept_run_marks_each_clip_dirty_including_last_row():
    a, b = source("a", 2), source("b", 2)
    tool = make_tool(a, b)
    for row in tool.session.rows:
        row.conf = .9
    assert tool.accept_run() == 4
    assert a.dirty and b.dirty
    assert all(row.reviewed for row in tool.session.rows)


def test_drawn_picture_and_stored_click_agree_to_the_pixel():
    tool = make_tool()
    blank = np.zeros((720, 1280, 3), np.uint8)
    image = blank.copy()
    image[50, 100] = 255                       # frame pixel (100, 50): centre at (100.5, 50.5)
    tool.session.rows[0].x, tool.session.rows[0].y = 100.5, 50.5
    tool.session.zoom = tool.session.zoom_target = 6.
    tool.follow_ball()
    tool.session.frame = lambda index: blank           # the same markers over nothing, to subtract
    markers = tool.render()[:tool.session.view_h].astype(int)
    tool.session.frame = lambda index: image
    drawn = tool.render()[:tool.session.view_h].astype(int)
    lit = np.argwhere(np.abs(drawn - markers).max(2) > 127)
    (top, left), (bottom, right) = lit.min(0), lit.max(0)
    click = ((left + right) // 2, (top + bottom) // 2)
    stored = tool.session.to_frame(click[0] + .5, click[1] + .5)
    # Clicking the middle of the drawn pixel stores its centre, to within half a view pixel;
    # the warp used to draw everything 0.5 - 0.5/scale frame px off from where clicks went.
    slack = .5 / tool.session.scale + 1e-9
    assert abs(stored[0] - 100.5) <= slack, stored
    assert abs(stored[1] - 50.5) <= slack, stored
    assert left <= tool.session.to_view(100.5, 50.5)[0] <= right
    for u, v in ((0, 0), (37, 91), (tool.session.view_w - 1, tool.session.view_h - 1)):
        assert tool.session.to_view(*tool.session.to_frame(u + .5, v + .5)) == (u, v)


def test_idle_render_is_cached_and_every_change_redraws():
    tool = make_tool()
    frames = {}
    tool.session.frame = lambda index: frames.setdefault(index, np.zeros((720, 1280, 3), np.uint8))
    tool.follow_ball()
    first = tool.render()
    with patch.object(cv2, "warpAffine", wraps=cv2.warpAffine) as warp:
        assert tool.render() is first                # nothing changed: nothing drawn
        assert warp.call_count == 0
    changes = {
        "label moved": lambda t: setattr(t.session.rows[0], "x", t.session.rows[0].x + 1),
        "another row reviewed (counter)": lambda t: setattr(t.session.rows[7], "reviewed", True),
        "flagged": lambda t: setattr(t.session.rows[0], "suspect", True),
        "filter": lambda t: setattr(t.session, "filter", 1),
        "inset": lambda t: setattr(t.session, "inset", not t.session.inset),
        "trail": lambda t: setattr(t.session, "trail", not t.session.trail),
        "follow": lambda t: setattr(t.session, "follow", not t.session.follow),
        "step": lambda t: setattr(t.session, "step", 4),
        "unsaved": lambda t: setattr(t.session, "dirty", not t.session.dirty),
        "zoom": lambda t: setattr(t.session, "zoom", t.session.zoom * 1.5),
        "pan": lambda t: setattr(t.session, "center", (t.session.center[0] + 7, t.session.center[1])),
        "walk": lambda t: t.session.only_uncertain(),
        "ghost": lambda t: setattr(t.session, "ghost", (400., 300.)),
        "next frame": lambda t: t.move(1),
        "frame re-decoded": lambda t: frames.update({t.session.rows[t.session.cursor].frame: np.ones((720, 1280, 3), np.uint8)}),
        "window resized": lambda t: t.fit_view(900, 600),
    }
    for name, change in changes.items():
        before = tool.render()
        change(tool)
        assert tool.render() is not before, name


def test_show_resends_only_when_needed():
    tool = make_tool()
    with patch.object(cv2, "imshow") as imshow:
        tool.show()
        tool.show()
        assert imshow.call_count == 1                # the same picture is not re-sent
        tool.shown = None                            # what reset_window does for a new window
        tool.show()
        assert imshow.call_count == 2
        tool.grid_view.toggle()
        tool.show()
        tool.grid_view.toggle()
        tool.show()                                  # back from the grid: the frame view again
        assert imshow.call_count == 4


def test_stepping_back_near_the_start_hits_the_cache():
    frames = [np.full((36, 64, 3), i, np.uint8) for i in range(80)]

    class Capture:
        opened = 0

        def __init__(self):
            Capture.opened += 1
            self.at = 0

        def isOpened(self):
            return True

        def grab(self):
            self.at += 1
            return self.at <= len(frames)

        def retrieve(self):
            return True, frames[self.at - 1].copy()

        def read(self):
            return (self.grab() and True), (frames[self.at - 1].copy() if self.at <= len(frames) else None)

        def set(self, *args):
            return False

        def release(self):
            pass

    from finetune.labeling.label_source import Source
    rows = [Row(i, 10., 10., True, "csv", 0.) for i in range(1, 60, 2)]   # odd phase, as 60 FPS clips
    with patch.object(cv2, "VideoCapture", lambda path: Capture()):
        clip = Source("c", Path("c.mp4"), rows, 64, 36, Path("c_ball.csv"), Path("c_ball.review.json"))
        clip.cache_limit = 64
        tool = Tool([clip])
        tool.session.cursor = len(rows) - 1
        tool.session.frame(rows[-1].frame)
        Capture.opened = 0
        for i in range(len(rows) - 1, -1, -1):       # `a` all the way back to the start
            tool.session.cursor = i
            assert int(tool.session.frame(rows[i].frame)[0, 0, 0]) == rows[i].frame
    assert Capture.opened <= 2                       # was one reopen per press


def test_point_in_the_no_ball_corner_is_refused():
    tool = make_tool()
    before = (tool.session.rows[0].x, tool.session.rows[0].y)
    assert not tool.set_point(1270, 10)  # would read back as "no ball"
    assert (tool.session.rows[0].x, tool.session.rows[0].y) == before
    assert not tool.session.rows[0].reviewed
    assert tool.set_point(600, 300)


def test_click_beyond_the_frame_is_not_a_label():
    tool = make_tool()
    with patch.object(cv2, "imshow"):
        tool.show()
    tool.pointer.on_mouse(cv2.EVENT_LBUTTONDOWN, 0, 0, 0, None)   # top-left of a view centred near (30, 25)
    assert tool.session.cursor == 0
    assert not tool.session.rows[0].reviewed


def test_wheel_in_grid_or_while_calibrating_still_cancels_window_zoom():
    tool = make_tool()
    tool.grid_view.toggle()
    with patch.object(tool.viewport, "repay_wheel") as repay:
        tool.pointer.on_mouse(cv2.EVENT_MOUSEWHEEL, 100, 100, 120 << 16, None)
    repay.assert_called_once_with(120)
    tool.grid_view.toggle()
    tool.mouse.calibrating = True
    with patch.object(tool.viewport, "repay_wheel") as repay:
        tool.pointer.on_mouse(cv2.EVENT_MOUSEWHEEL, 100, 100, 120 << 16, None)
    repay.assert_called_once_with(120)


def test_window_rebuilds_stop_when_the_check_keeps_failing():
    from finetune.labeling import label_window
    tool = make_tool()
    with patch.object(label_window, "_user32", None), patch.object(cv2, "destroyWindow"), \
            patch.object(cv2, "waitKey"), patch.object(tool, "create_window"), patch.object(tool, "show"):
        for _ in range(6):
            tool.viewport.reset_needed = True
            tool.reset_window()
    assert tool.viewport.window_resets == 3
    assert tool.viewport.probe_at == float("inf")
    assert not tool.viewport.reset_needed


def test_function_keys_do_not_pass_for_letters():
    from finetune.labeling import label_window
    held = {0x71}                                # F2, which reads as "q"
    fake = SimpleNamespace(GetKeyState=lambda vk: 0x8000 if vk in held else 0)
    with patch.object(label_window, "_user32", fake):
        assert label_window.non_character_key(ord("q"))
        held.clear()                             # the q key itself: VK_F2 is not down
        assert not label_window.non_character_key(ord("q"))
        assert not label_window.non_character_key(ord("d"))


def test_save_survives_a_locked_file_and_keeps_the_edits(tmp_path):
    import finetune.labeling.label_tool as label_tool
    a, b = source("a", 2), source("b", 2)
    for clip in (a, b):
        clip.out_csv = tmp_path / f"{clip.clip}_ball.csv"
        clip.review = tmp_path / f"{clip.clip}_ball.review.json"
        clip.suspects = tmp_path / f"{clip.clip}_ball.suspect.json"
    tool = make_tool(a, b)
    a.dirty = b.dirty = tool.session.dirty = True
    real = label_tool.save_rows

    def locked(path, *rest):
        if path == a.out_csv:
            raise PermissionError(13, "in use by another program")
        real(path, *rest)
    with patch.object(label_tool, "save_rows", side_effect=locked):
        tool.save(quiet=True)
    assert b.out_csv.is_file()     # one stuck clip does not stop the others
    assert a.dirty and tool.session.dirty  # ... and its edits are kept for the next try
    assert not b.dirty
    tool.rescue_unsaved()
    assert (tmp_path / "a_ball.unsaved.csv").is_file()


def test_legacy_flag_list_survives_a_save(tmp_path):
    flags = tmp_path / "clip_ball.suspect.json"
    flags.write_text(json.dumps({"frames": [1, 3]}))
    rows = source("clip", 5).rows
    assert apply_flags(flags, rows) == 2
    save_flags(flags, rows)               # nothing settled: nothing may be dropped
    assert apply_flags(flags, rows) == 2
    rows[1].suspect = False                  # answered (SPACE or an edit in the tool)
    save_flags(flags, rows)
    assert json.loads(flags.read_text()) == {"unplaceable": [], "check": [3], "answered": {"1": [41.25, 27.75]}}


def test_new_audit_flag_reopens_a_reviewed_frame_until_answered():
    tool = make_tool()
    for done in tool.session.rows[:8]:
        done.reviewed = True                     # the start of the clip is reviewed ...
    row = tool.session.rows[4]
    row.suspect = True                           # ... and an audit after that flagged frame 4
    assert row.uncertain
    tool.session.cursor = 0
    tool.jump(1)
    assert tool.session.cursor == 4                      # f goes to it even though it was reviewed
    assert (tool.session.only_uncertain() and 4 in tool.session.walk) is True
    tool.session.walk = []
    tool.session.cursor = 4
    tool.confirm()                               # SPACE: it is right where it is
    assert not row.suspect
    row.undo()
    assert row.suspect                           # undo puts the question back


def test_a_later_audit_does_not_reask_an_answered_frame(tmp_path):
    from finetune.data import label_audit
    labels = tmp_path
    (labels / "clip_ball.csv").write_text("frame,ball_x,ball_y\nframe_001,41.25,27.75\nframe_003,50,30\n")
    (labels / "clip_ball.suspect.json").write_text(json.dumps(
        {"unplaceable": [], "check": [], "answered": {"1": [41.25, 27.75], "3": [10, 10]}}))
    with patch.multiple("finetune.data.workspace", LABELS=labels, MANIFEST=labels / "clips.csv"):
        label_audit.flag_frames("clip", "check", [1, 3], "audit")
        assert label_audit.open_flags("clip")["check"] == [3]   # 3's label moved since
        assert label_audit.outstanding_flags("clip")["check"] == [3]


def test_damaged_review_file_is_an_error_that_names_it_and_is_never_replaced(tmp_path):
    from finetune.data.labels import LabelFormatError
    from finetune.labeling.label_source import build_source
    review = tmp_path / "clip_ball.review.json"
    rows = source("clip", 3).rows
    for odd in ('{"reviewed": null, "edited": null}',             # parseable: tolerated field by field
                '{"evidence": {"0": {"source": "det"}}}', '{"evidence": [1]}'):
        review.write_text(odd)
        apply_review(review, rows)
    for broken in ("[1, 2]", "{not json"):                     # not a review file: an error naming it
        review.write_text(broken)
        with pytest.raises(LabelFormatError) as caught:
            apply_review(review, rows)
        assert review.name in str(caught.value)
    # The tool skips such a clip rather than open it on a fresh review file.
    labels = tmp_path
    (labels / "clip_ball.csv").write_text("frame,ball_x,ball_y\nframe_001,20,30\n")
    with patch("finetune.labeling.label_source.find_video", return_value=labels / "clip.mp4"), \
            patch("finetune.labeling.label_source.video_meta", return_value=(30.0, 640, 360, 10)):
        assert build_source("clip", labels, labels) is None
    assert review.read_text() == "{not json"


def test_ranges_include_clips_that_only_have_a_draft(tmp_path):
    from finetune.labeling.label_source import expand_clip_ranges
    for name in ("v1_ball.csv", "v2_ball.csv.draft", "v3_ball.csv", "v4_ball.unsaved.csv"):
        (tmp_path / name).write_text("frame,ball_x,ball_y\n")
    assert expand_clip_ranges(["v1..v3"], tmp_path) == ["v1", "v2", "v3"]


# --------------------------------------------------------------------------- suggestions

def hint(row, x=None, y=None, trusted=True, ball_here=True, seed="label"):
    """A refine.suggest() entry for `row`: at (x, y), measured from its current label."""
    seed = (row.x, row.y) if seed == "label" else seed
    return {"x": row.x if x is None else x, "y": row.y if y is None else y, "seed": seed, "from": "label",
            "move": 0.0, "path": 0.3, "blobs": 1, "ratio": 1.0, "contrast": 40.0,
            "trusted": trusted, "ball_here": ball_here}


def test_centroid_finds_the_ball_in_label_coordinates():
    from finetune.refiner.refine import motion_centroid
    previous, current, following = ball_frames([(40, 30), (50, 40), (60, 50)])
    x, y = motion_centroid(previous, current, following, (52.0, 38.0), 9)
    assert x == pytest.approx(50.5, abs=5e-5)    # pixel 50's centre is 50.5 in label files
    assert y == pytest.approx(40.5, abs=5e-5)
    still = ball_frames([(50, 40)])[0]
    assert motion_centroid(still, still, still, (50.5, 40.5), 9) is None


def test_suggest_reads_a_video_and_scores_trust(tmp_path):
    from finetune.refiner import refine
    path_of = [(100 + 12 * i, 200 - 18 * i + i * i) for i in range(12)]   # an exact parabola
    video = write_frames(tmp_path / "clip.avi", ball_frames(path_of))
    seeds = {i: (column + 0.9, row + 0.2) for i, (column, row) in enumerate(path_of)}
    found = refine.suggest(video, seeds, refine.PassOptions(1))
    assert set(found) == set(range(12))       # the ends diff against one side only
    for frame, found_hint in found.items():
        column, row = path_of[frame]              # the 3x3 ball's middle pixel: centre at k + 0.5
        assert found_hint["x"] == pytest.approx(column + .5, abs=5e-4)   # lossless clip: exact
        assert found_hint["y"] == pytest.approx(row + .5, abs=5e-4)
    assert found[5]["trusted"] and found[5]["ball_here"]
    assert found[5]["path"] < 1.0


def test_a_placer_moves_the_chosen_points_and_a_far_move_is_not_trusted(tmp_path):
    from finetune.refiner import cnn_refiner, refine
    path_of = [(100 + 12 * i, 200 - 18 * i + i * i) for i in range(12)]
    seen = []

    class Nudge:                      # stands in for the network: 0.5 px right, frame 5 far off
        cut_crop = staticmethod(cnn_refiner.cut_crop)
        sure = None                   # a version without the self-check

        def place(self, items, height):
            seen.extend(crop.shape for crop, _, _ in items)
            return [(start[0] + (8.0 if abs(start[0] - 160.5) < 2 else 0.5), start[1], self.sure)
                    for _, _, start in items]
    video = write_frames(tmp_path / "clip.avi", ball_frames(path_of))
    seeds = {i: (column + 0.9, row + 0.2) for i, (column, row) in enumerate(path_of)}
    plain = refine.suggest(video, seeds, refine.PassOptions(1))
    placed = refine.suggest(video, seeds, refine.PassOptions(1, placer=Nudge()))
    assert set(seen) == {(cnn_refiner.CROP, cnn_refiner.CROP, 9)}
    for frame, placed_hint in placed.items():
        assert placed_hint["centroid"][0] == pytest.approx(plain[frame]["x"], abs=5e-3)   # the measurement is kept
        moved = 8.0 if frame == 5 else 0.5
        assert placed_hint["x"] == pytest.approx(placed_hint["centroid"][0] + moved, abs=5e-3)
    assert plain[4]["trusted"] and placed[4]["trusted"]    # 1.5 px at 1080p: still trusted
    assert plain[5]["trusted"]
    assert not placed[5]["trusted"]
    # a self-check decides trust by itself
    video = write_frames(tmp_path / "clip2.avi", ball_frames(path_of))
    unsure, checked = Nudge(), Nudge()
    unsure.sure, checked.sure = False, True
    doubted = refine.suggest(video, seeds, refine.PassOptions(1, placer=unsure))
    vouched = refine.suggest(video, seeds, refine.PassOptions(1, placer=checked))
    assert not any(h["trusted"] for h in doubted.values())
    assert vouched[5]["trusted"]            # far from its measurement (8 px at 360p is 24 px at 1080p), but the network is sure


# --------------------------------------------------------------------------- cnn_refiner

def test_cut_centres_the_point_and_place_maps_window_pixels_back():
    import torch
    from finetune.refiner import cnn_refiner
    image = np.zeros((720, 1280, 3), np.uint8)
    image[300, 500] = 255
    crop = cnn_refiner.cut_crop([image] * 3, (500.5, 300.5), 720)     # pixel 500's centre, 720p -> 1080p scale
    assert crop.shape == (96, 96, 9)
    peak = np.unravel_index(np.argmax(crop[..., 4]), crop.shape[:2])
    assert 47 <= peak[0] <= 48 and 47 <= peak[1] <= 48
    placer = object.__new__(cnn_refiner.Placer)
    placer.device, placer.ROUNDS, placer.threshold = "cpu", 1, 0.5

    def net(x):                       # the label frame's bright blob, however the window is turned
        weight = x[:, 4].double() + 0.5                                  # to_tensor took 0.5 off
        total = weight.sum((1, 2))
        gx, gy = torch.arange(x.shape[3]).double(), torch.arange(x.shape[2]).double()
        return (torch.stack([(weight.sum(1) * gx).sum(1) / total, (weight.sum(2) * gy).sum(1) / total], 1),
                torch.full((x.shape[0],), 2.0))                           # ... and sure (0.88)
    placer.net = net
    (x, y, sure), = placer.place([(crop, (500.5, 300.5), (502.5, 301.5))], 720)   # start 2 px right, 1 down
    assert sure
    assert x == pytest.approx(500.5, abs=5e-4)
    assert y == pytest.approx(300.5, abs=5e-4)
    lo = (cnn_refiner.WINDOW - 1) / 2.0
    placer.net = lambda x: (torch.tensor([[lo + 3.0, lo]] * x.shape[0]),   # "3 px right" whatever it sees
                            torch.full((x.shape[0],), 2.0))
    (x, y, _), = placer.place([(crop, (500.5, 300.5), (502.5, 301.5))], 720)
    assert x == pytest.approx(502.5, abs=5e-5)                            # the mirrored look cancels it
    assert y == pytest.approx(301.5, abs=5e-5)


def test_sure_threshold_keeps_far_misses_rare_and_folds_keep_a_game_together(tmp_path):
    from finetune.refiner import refiner_training
    confs = np.linspace(1.0, 0.0, 3000)
    errors = np.where(np.arange(3000) >= 2000, 9.0, 0.5)       # the least confident third is wrong
    threshold = refiner_training.sure_threshold(errors, confs)
    sure = confs >= threshold
    assert np.mean(errors[sure] > 5) <= refiner_training.SURE_FAR_SHARE
    assert sure.sum() > 1990                                     # ... and nearly all the good ones are sure
    manifest = tmp_path / "clips.csv"
    manifest.write_text("clip,origin\n"
                        "video13,C:\\prof\\Dataset\\game2\\Clip1\n"
                        "video14,C:\\prof\\Dataset\\game2\\Clip2\n"
                        "video1,C:\\archive\\video1.mp4\n", encoding="utf-8")
    with patch("finetune.data.workspace.MANIFEST", manifest):
        fold_of = refiner_training.folds(["video13", "video14", "video1", "grid_match4"],
                                         {"video13": 300, "video14": 200, "video1": 400, "grid_match4": 600})
    assert fold_of["video13"] == fold_of["video14"]
    assert len(set(fold_of.values())) == 3


def test_a_version_from_before_the_self_check_still_loads(tmp_path):
    import torch
    from finetune.refiner import cnn_refiner
    path = tmp_path / "refiner-20261019-120547.pt"
    torch.save({"state": cnn_refiner.build_network(False).state_dict(), "trained": "then"}, path)
    placer = cnn_refiner.Placer(path)
    crop = np.zeros((cnn_refiner.CROP, cnn_refiner.CROP, 9), np.uint8)
    (x, y, sure), = placer.place([(crop, (100.0, 100.0), (100.0, 100.0))], 1080)
    assert placer.threshold is None
    assert sure is None                                          # refine.py's own rule decides


def test_versions_newest_is_current_until_one_is_chosen_and_old_ones_stay(workspace_dir):
    from finetune.refiner import cnn_refiner
    models = cnn_refiner.models_dir()                        # read from the workspace when asked, not at import
    assert models == workspace_dir / "models" / "refiner"
    models.mkdir(parents=True)
    assert cnn_refiner.current() is None
    for name in ("refiner-20261001-090000.pt", "refiner-20261019-110000.pt"):
        (models / name).write_bytes(b"")
    assert cnn_refiner.current().name == "refiner-20261019-110000.pt"
    cnn_refiner.make_current("20261001")
    assert cnn_refiner.current().name == "refiner-20261001-090000.pt"
    (models / "refiner-20261020-080000.pt").write_bytes(b"")
    assert cnn_refiner.current().name == "refiner-20261001-090000.pt"   # a choice sticks
    (models / "refiner-20261001-090000.pt").unlink()
    assert cnn_refiner.current().name == "refiner-20261020-080000.pt"   # ... unless it is gone
    assert len(cnn_refiner.versions()) == 2
    with pytest.raises(ValueError):
        cnn_refiner.make_current("2026")                 # ambiguous


def test_label_tool_reloads_the_placer_when_the_current_version_changes():
    tool = make_tool()
    first, second = Path("refiner-a.pt"), Path("refiner-b.pt")
    loaded = []
    with patch("finetune.labeling.label_hints.cnn_refiner.current", side_effect=[first, first, second, None]), \
            patch("finetune.labeling.label_hints.cnn_refiner.load_current",
                  side_effect=lambda: loaded.append(1) or SimpleNamespace(name=f"v{len(loaded)}")):
        assert tool.session.hints.current_placer().name == "v1"
        assert tool.session.hints.current_placer().name == "v1"       # unchanged: not loaded again
        assert tool.session.hints.current_placer().name == "v2"       # retrained: the new version
        assert tool.session.hints.current_placer() is None            # none left: the centroid alone
    assert len(loaded) == 2


# --------------------------------------------------------------------------- the grid from the keyboard

def grid_tool():
    return make_tool(source("test", 40))


def test_sixteen_tiles_page_and_select_from_the_keyboard():
    tool = grid_tool()
    tool.grid_view.size = 16
    tool.grid_view.toggle()
    tool.grid_view.render()
    assert len(tool.grid_view.tiles) == 16
    boxes = [box for _, box, *_ in tool.grid_view.tiles]
    assert len({box[1] for box in boxes}) == 4          # four rows of four
    assert all(box[3] < tool.session.view_h for box in boxes)   # none run into the bar
    tool.grid_view.select(5)
    assert tool.session.cursor == 5
    tool.grid_view.step(-1)
    assert tool.session.cursor == 4
    tool.grid_view.select(15)
    tool.grid_view.step(1)                                   # off the end: next page
    assert tool.grid_view.indices == list(range(16, 32))
    assert tool.session.cursor == 16
    tool.grid_view.step(-1)                                  # and back onto the last tile
    assert tool.grid_view.indices == list(range(16))
    assert tool.session.cursor == 15


def test_switching_size_keeps_the_selected_frame():
    tool = grid_tool()
    tool.grid_view.toggle()
    tool.grid_view.select(6)
    tool.grid_view.toggle_size()
    assert tool.grid_view.size == 16
    assert len(tool.grid_view.indices) == 16
    assert tool.session.cursor == 6
    tool.grid_view.toggle_size()
    assert tool.grid_view.size == 8


def test_y_in_the_grid_takes_the_selected_tiles_suggestion_and_moves_on():
    tool = grid_tool()
    tool.grid_view.toggle()
    tool.grid_view.select(2)
    row = tool.session.rows[2]
    tool.session.hints.found["test"] = {2: hint(row, x=row.x + 1)}
    assert tool.take_suggestion()
    tool.grid_view.step(1)
    assert (row.source, tool.session.cursor) == ("refined", 3)


# --------------------------------------------------------------------------- the trajectory sheet

def test_sheet_draws_a_stretch_and_a_click_opens_that_frame():
    tool = make_tool(source("test", 150))
    tool.session.cursor = 70
    tool.sheet_view.toggle()
    assert tool.sheet_view.active and not tool.grid_view.active
    assert tool.sheet_view.start == 40                        # the cursor's stretch, centred on it
    tool.render()
    assert [p[0] for p in tool.sheet_view.points] == list(range(40, 100))
    index, x, y = tool.sheet_view.points[25]
    tool.sheet_view.on_mouse(cv2.EVENT_LBUTTONDOWN, x + 2, y - 1)
    assert not tool.sheet_view.active
    assert tool.session.cursor == index
    assert not any(row.reviewed for row in tool.session.rows)     # finding a frame edits nothing


def test_sheet_pages_stay_inside_the_clip():
    tool = make_tool(source("a", 70), source("b", 70))
    tool.session.cursor = 5
    tool.sheet_view.toggle()
    assert tool.sheet_view.start == 0
    tool.sheet_view.page(1)
    assert tool.sheet_view.start == 10                        # the last full stretch of clip a
    tool.sheet_view.page(1)
    assert tool.sheet_view.start == 10
    tool.render()
    assert all(tool.session.rows[i].clip == "a" for i, *_ in tool.sheet_view.points)
    tool.sheet_view.page(-1)
    assert tool.sheet_view.start == 0


def test_a_click_far_from_every_point_does_nothing():
    tool = make_tool(source("test", 150))
    tool.sheet_view.toggle()
    tool.render()
    cursor = tool.session.cursor
    tool.sheet_view.on_mouse(cv2.EVENT_LBUTTONDOWN, tool.session.view_w - 1, tool.session.view_h - 1)
    assert tool.sheet_view.active
    assert tool.session.cursor == cursor
    tool.sheet_view.on_mouse(cv2.EVENT_RBUTTONDOWN, 0, 0)          # right click goes back
    assert not tool.sheet_view.active


# --------------------------------------------------------------------------- drafts

def test_drafts_use_the_unshifted_detector_point():
    from finetune.training import pretrack
    from finetune.data.workspace import VideoMeta
    from tennis_tracker import ball_detectors
    seen = {}

    class Detector:
        def __init__(self, weights, cfg):
            seen["offset"] = cfg.gridtracknet_y_offset_px
            seen["conf"] = cfg.conf

        def detect_clip(self, video, fps, width, height, total):
            seen["clip"] = (video, fps, width, height, total)
            return [[([10, 20, 14, 24], .9)], []]
    with patch.object(ball_detectors, "GridTrackNetBallDetector", Detector):
        rows = pretrack.raw_rows(Path("clip.mp4"), Path("w.npz"), .5, "cpu", VideoMeta(30, 1280, 720, 2))
    assert seen["offset"] == 0.0
    assert seen["conf"] == .5
    assert seen["clip"] == (Path("clip.mp4"), 30, 1280, 720, 2)
    assert rows == {0: (12.0, 22.0, "det", .9)}      # the box centre; frame 1 had nothing


# --------------------------------------------------------------------------- taking a suggestion

def hinted_tool():
    """make_tool() with a suggestion for its first row, 0.75 px right and up of the label."""
    tool = make_tool()
    tool.session.hints.found["test"] = {0: hint(tool.session.rows[0], 31.0, 25.5)}
    tool.session.hints.version += 1
    return tool


def test_y_takes_the_suggestion_as_refined_and_undo_restores_it():
    tool = hinted_tool()
    row = tool.session.rows[0]
    assert tool.take_suggestion()
    assert (row.x, row.y, row.source) == (31.0, 25.5, "refined")
    assert row.edited
    row.undo()
    assert (row.x, row.y, row.source) == (30.25, 24.75, "csv")
    tool.take_suggestion()
    tool.set_point(32, 26)                       # then corrected by hand
    assert row.source == "click"


def test_suggestion_is_dropped_once_the_label_moves_away_from_its_seed():
    tool = hinted_tool()
    assert tool.session.suggestion_for(0) is not None
    tool.session.rows[0].x += 40                         # far from where the suggestion was measured
    assert tool.session.suggestion_for(0) is None
    assert not tool.take_suggestion()
    assert tool.session.suggestion_for(1) is None        # no suggestion measured for this frame


def test_verdict_flags_labels_off_a_trusted_suggestion_and_e_walks_them():
    tool = make_tool()     # 1280x720: 1 frame px is 1.5 px at 1080p
    hints = {row.frame: hint(row, x=row.x + 1.0) for row in tool.session.rows}
    hints[3]["x"] = tool.session.rows[3].x + 2.5          # 3.75 px at 1080p: the label is off
    hints[6]["x"] = tool.session.rows[6].x + 2.5
    hints[8]["trusted"] = False                   # the suggestion itself is not trusted here
    hints[8]["x"] = tool.session.rows[8].x + 2.5
    tool.session.hints.found["test"] = hints
    tool.session.hints.version += 1
    assert tool.session.hint_verdict(0) == "agrees"
    assert tool.session.hint_verdict(3) == "differs"
    assert tool.session.hint_verdict(8) is None
    assert tool.session.disputed_count() == 2
    assert tool.jump_disputed(1)
    assert tool.session.cursor == 3
    assert tool.jump_disputed(1)
    assert tool.session.cursor == 6
    assert not tool.jump_disputed(1)
    assert tool.jump_disputed(-1)
    assert tool.session.cursor == 3
    tool.set_point(tool.session.rows[3].x + 2.5, tool.session.rows[3].y)   # fixed: now it agrees
    assert tool.session.disputed_count() == 1
    assert not any(row.reviewed for row in tool.session.rows if row.frame != 3)   # nothing auto-accepted


def test_no_ball_frame_shows_only_a_trusted_suggestion_and_y_places_it():
    tool = make_tool()
    row = tool.session.rows[2]
    row.visible = False
    tool.session.hints.found["test"] = {2: hint(row, 300.0, 200.0, trusted=False, seed=None)}
    assert tool.session.suggestion_for(2) is None         # a guess is not enough to put a ball there
    tool.session.hints.found["test"][2]["trusted"] = True
    assert tool.session.suggestion_for(2) is not None
    tool.session.cursor = 2
    assert tool.take_suggestion()
    assert row.visible and (row.x, row.y) == (300.0, 200.0) and row.source == "refined"
    tool.session.hints.found["test"][2]["ball_here"] = False
    assert tool.session.suggestion_for(2) is None         # nothing ball-like moved: never shown


def test_a_changed_label_is_remeasured_with_reviewed_labels_steering():
    tool = make_tool()
    calls = []
    with patch("finetune.labeling.label_hints.refine.suggest",
               side_effect=lambda video, seeds, options:
               calls.append((seeds, options.reviewed, options.only)) or {}), \
            patch("finetune.labeling.label_hints.cnn_refiner.current", return_value=None):
        tool.session.source.video = Path("clip.mp4")
        tool.session.hints.suggest_clip(tool.session.source)
        assert calls[-1][1] == []                             # nothing reviewed yet
        before = tool.session.hints.label_state(tool.session.source)
        tool.session.cursor = 1
        tool.set_point(80, 40)
        assert tool.session.hints.label_state(tool.session.source) != before
        tool.session.hints.suggest_clip(tool.session.source)
        seeds, reviewed, only = calls[-1]
        assert seeds[1] == (80.0, 40.0)
        assert reviewed == [1]
        assert only == set(range(10))                         # the stretch around frame 1 (10 rows)
        assert calls[0][2] is None                            # the first pass measures everything
        assert tool.session.hints.measured_state["test"] == tool.session.hints.label_state(tool.session.source)


def test_suggestion_is_drawn_and_its_arrival_redraws():
    tool = make_tool()
    tool.session.zoom = tool.session.zoom_target = 6.
    tool.follow_ball()
    plain = tool.render()
    orange = lambda image: int(np.sum((image[:, :, 0] == 0) & (image[:, :, 1] == 150) & (image[:, :, 2] == 255)))
    assert orange(plain) == 0
    tool.session.hints.found["test"] = {0: hint(tool.session.rows[0], 33.0, 27.0)}
    tool.session.hints.version += 1
    hinted = tool.render()
    assert hinted is not plain
    assert orange(hinted) > 0


# --------------------------------------------------------------------------- the offer
# On an unreviewed frame the suggestion is the mark: SPACE takes it, x keeps the draft.

def test_space_takes_the_offer_and_a_reviewed_label_stays_the_mark():
    tool = make_tool()
    row = tool.session.rows[0]
    tool.session.hints.found["test"] = {0: hint(row, row.x + 1.0, row.y - 0.5)}
    tool.session.hints.version += 1
    tool.follow_ball()
    assert tool.session.target_point() == (31.25, 24.25)       # the view and the mouse go to the offer
    tool.render()
    tool.confirm()
    assert (row.x, row.y, row.source) == (31.25, 24.25, "refined")
    assert row.reviewed and row.edited
    tool.session.hints.found["test"][0] = hint(row, row.x + 2.0, row.y)   # re-measured from the new label
    tool.session.hints.version += 1
    assert tool.session.offer_for(0) is None                   # reviewed: your label is the mark
    assert tool.session.suggestion_for(0) is not None          # the hint is still shown beside it
    tool.render()
    tool.confirm()
    assert (row.x, row.y) == (31.25, 24.25)


def test_an_offer_that_was_not_drawn_is_not_taken():
    tool = make_tool()
    row = tool.session.rows[0]
    tool.render()                                      # judged without a suggestion
    tool.session.hints.found["test"] = {0: hint(row, row.x + 1.0, row.y)}
    tool.session.hints.version += 1
    tool.confirm()                                     # SPACE lands after it arrived
    assert (row.x, row.reviewed, row.edited) == (30.25, True, False)
    tool.session.cursor = 1
    tool.session.hints.found["test"][1] = hint(tool.session.rows[1], tool.session.rows[1].x + 1.0, tool.session.rows[1].y)
    tool.session.hints.version += 1
    tool.render()
    tool.session.cursor = 2                                    # drawn for frame 1, pressed on 2
    tool.confirm()
    assert (tool.session.rows[1].reviewed, tool.session.rows[2].edited) == (False, False)
    assert tool.session.rows[2].reviewed


def test_x_keeps_the_draft_and_the_offer_rules():
    from finetune.labeling import label_hints
    tool = make_tool()
    row = tool.session.rows[0]
    tool.session.hints.found["test"] = {0: hint(row, row.x + 1.0, row.y, trusted=False)}
    tool.session.hints.version += 1
    with patch.object(label_hints, "OFFER_UNSURE", False):
        assert tool.session.offer_for(0) is None
    assert tool.session.offer_for(0) is not None               # unsure but near the draft: offered, dull
    tool.session.hints.found["test"][0]["x"] = row.x + 2.5     # 3.75 px at 1080p: too far to offer unsure
    tool.session.hints.version += 1
    assert tool.session.offer_for(0) is None
    assert tool.session.suggestion_for(0) is not None          # ... but still a hint beside the draft
    tool.session.hints.found["test"][0]["trusted"] = True      # sure: offered wherever it is
    tool.session.hints.version += 1
    assert tool.session.offer_for(0) is not None
    tool.session.hints.found["test"][0] = hint(row, row.x + 1.0, row.y, trusted=False)
    tool.session.hints.version += 1
    tool.render()
    tool.confirm(take_offer=False)                     # x: the draft was right
    assert (row.x, row.reviewed, row.edited) == (30.25, True, False)
    other = tool.session.rows[1]
    tool.session.hints.found["test"][1] = hint(other, other.x + 0.1, other.y)
    tool.session.hints.version += 1
    assert tool.session.offer_for(1) is None                   # nothing to choose between
    for row in tool.session.rows:
        row.conf = .9
    tool.session.hints.found["test"][2] = hint(tool.session.rows[2], tool.session.rows[2].x + 1.0, tool.session.rows[2].y)
    tool.session.hints.version += 1
    tool.session.cursor = 1
    tool.render()
    tool.accept_run()                                  # r confirms the drafts, unseen
    assert (tool.session.rows[2].x, tool.session.rows[2].edited) == (52.25, False)
    assert all(row.reviewed for row in tool.session.rows)


def test_confirm_grid_takes_the_offers_it_drew():
    tool = make_tool()
    rows = tool.session.rows
    tool.session.hints.found["test"] = {0: hint(rows[0], rows[0].x + 1.0, rows[0].y),
                                3: hint(rows[3], rows[3].x, rows[3].y + 1.0, trusted=False),
                                9: hint(rows[9], rows[9].x + 1.0, rows[9].y)}
    tool.session.hints.version += 1
    tool.grid_view.toggle()
    tool.grid_view.render()
    tool.grid_view.confirm_page()
    assert (rows[0].x, rows[0].source) == (31.25, "refined")
    assert (rows[3].y, rows[3].source) == (34.75, "refined")
    assert all(row.reviewed for row in rows[:8])
    assert not rows[1].edited
    assert not rows[9].reviewed                         # next page: not drawn, not taken
    assert rows[9].x == 129.25


def test_a_no_ball_draft_with_a_sure_hint_offers_it():
    tool = make_tool()
    row = tool.session.rows[2]
    row.visible = False
    tool.session.cursor = 2
    tool.session.hints.found["test"] = {2: hint(row, 300.0, 200.0, seed=None)}
    tool.session.hints.version += 1
    tool.follow_ball()
    assert tool.session.target_point() == (300.0, 200.0)
    tool.render()
    tool.confirm()
    assert row.visible
    assert (row.x, row.y, row.source) == (300.0, 200.0, "refined")


# --------------------------------------------------------------------------- the key table
# Every action a key can reach, replaced by a mock while a key is pressed, so a key that calls the
# wrong one, or one too many, fails the table below. The tool's actions go by their own names; the
# grid's and the sheet's are mocked on their views under the names the table has always used.
TOOL_HANDLERS = ("save", "confirm", "move", "jump", "jump_disputed", "take_suggestion", "mark_invisible", "accept_run",
                 "nudge", "zoom_key", "toggle_fullscreen", "remember_settings", "toggle_follow", "cycle_filter",
                 "toggle_inset", "follow_ball")
VIEW_HANDLERS = {GridView: {"toggle": "toggle_grid", "toggle_size": "toggle_grid_size", "confirm_page": "confirm_grid",
                            "page": "grid_page", "step": "grid_step", "select": "grid_select", "prepare": "prepare_grid"},
                 SheetView: {"toggle": "toggle_sheet", "page": "sheet_page"}}


@contextmanager
def every_handler_mocked():
    """Every action the keys can reach replaced by a mock, as {table name: mock}."""
    with ExitStack() as stack:
        mocks = stack.enter_context(patch.multiple(Tool, **{name: DEFAULT for name in TOOL_HANDLERS}))
        for view, names in VIEW_HANDLERS.items():
            view_mocks = stack.enter_context(patch.multiple(view, **{method: DEFAULT for method in names}))
            mocks.update({names[method]: mock for method, mock in view_mocks.items()})
        yield mocks


FORWARD = sorted({ord("n"), ord("d"), *KEYS_NEXT})
BACKWARD = sorted({ord("p"), ord("a"), *KEYS_PREV})


def keys(*names):
    return [ord(name) if isinstance(name, str) else name for name in names]


def nudges(step, letters="ijkl"):
    return [(ord(letter), [call.nudge(dx * step, dy * step)])
            for letter, (dx, dy) in zip(letters, ((0, -1), (-1, 0), (0, 1), (1, 0)))]


# (key, the calls it must make, in order) per mode. `move` is tested at step 1, the default.
PLAIN_BINDINGS = (
    [(key, [call.save()]) for key in keys("q", 27)]
    + [(ord("w"), [call.toggle_sheet()]), (9, [call.toggle_grid()])]
    + [(key, [call.confirm(), call.move(1)]) for key in keys(" ", 13, "c")]
    + [(ord("e"), [call.jump_disputed(1)]),
       (ord("y"), [call.take_suggestion(), call.move(1)]),
       (ord("x"), [call.confirm(take_offer=False), call.move(1)]),
       (ord("v"), [call.mark_invisible(), call.move(1)]),
       (ord("r"), [call.accept_run()])]
    + [(key, [call.move(1)]) for key in FORWARD] + [(key, [call.move(-1)]) for key in BACKWARD]
    + [(key, []) for key in keys("1", "2", "4", "6", "8")]       # the step, checked below
    + [(ord("f"), [call.jump(1)]), (ord("b"), [call.jump(-1)]), (ord("u"), [call.follow_ball()])]
    + nudges(1.0) + nudges(.25, "IJKL")
    + [(key, [call.zoom_key(1.0)]) for key in keys("+", "=")] + [(key, [call.zoom_key(-1.0)]) for key in keys("-", "_")]
    + [(ord("m"), [call.toggle_fullscreen()]), (ord("t"), [call.remember_settings()]), (ord("g"), [call.toggle_follow()]),
       (ord("h"), [call.cycle_filter()]), (ord("o"), [call.toggle_inset()]), (ord("s"), [call.save()]),
       (ord("z"), [call.toggle_grid_size()])]
)
GRID_BINDINGS = (
    [(key, [call.save()]) for key in keys("q", 27)]
    + [(ord("w"), [call.toggle_sheet()]), (9, [call.toggle_grid()])]
    + [(key, [call.confirm_grid()]) for key in keys(" ", 13, "c")]
    + [(key, [call.grid_page(1)]) for key in FORWARD] + [(key, [call.grid_page(-1)]) for key in BACKWARD]
    + [(ord(","), [call.grid_step(-1)]), (ord("."), [call.grid_step(1)])]
    + [(ord(digit), [call.grid_select(slot)]) for slot, digit in enumerate("123456789")]
    + [(ord("y"), [call.take_suggestion(), call.grid_step(1)]),
       (ord("x"), [call.confirm(take_offer=False), call.grid_step(1)]),
       (ord("z"), [call.toggle_grid_size()]),
       (ord("r"), []),
       (ord("e"), [call.jump_disputed(1), call.prepare_grid(0)]),
       (ord("v"), [call.mark_invisible()]),
       (ord("f"), [call.jump(1), call.prepare_grid(0)]), (ord("b"), [call.jump(-1), call.prepare_grid(0)]),
       (ord("u"), [call.follow_ball()])]
    + nudges(1.0) + nudges(.25, "IJKL")
    + [(key, []) for key in keys("+", "=", "-", "_")]            # the grid's own zoom, checked below
    + [(ord("m"), [call.toggle_fullscreen()]), (ord("t"), [call.remember_settings()]), (ord("g"), [call.toggle_follow()]),
       (ord("h"), [call.cycle_filter()]), (ord("o"), [call.toggle_inset()]), (ord("s"), [call.save()])]
)
SHEET_BINDINGS = (
    [(key, [call.save()]) for key in keys("q", 27)]
    + [(ord("w"), [call.toggle_sheet()]), (9, [call.toggle_sheet(), call.toggle_grid()])]
    + [(key, [call.sheet_page(1)]) for key in FORWARD] + [(key, [call.sheet_page(-1)]) for key in BACKWARD]
    + [(key, []) for key in keys(" ", 13, "c", "e", "y", "x", "v", "r", "1", "f", "b", "u", "i", "I", "+", "-",
                                 "m", "t", "g", "h", "o", "s", "z", ",", ".")]
)
BINDINGS = {"plain": PLAIN_BINDINGS, "grid": GRID_BINDINGS, "sheet": SHEET_BINDINGS}


def key_name(key: int) -> str:
    """The key as it reads in a test id: the character for a printable one, the code otherwise."""
    return chr(key) if 32 < key < 127 else str(key)


def press(mode, key, returns=None, shift=False, ready=True):
    """(the handler calls the key made, whether the tool wants to quit, the tool)."""
    tool = Tool([source("test")])
    tool.grid_view.active, tool.sheet_view.active = mode == "grid", mode == "sheet"
    calls = Mock()
    with every_handler_mocked() as mocks, \
            patch.object(Tool, "view_ready", return_value=ready), patch.object(Row, "undo"), \
            patch.object(win, "shift_held", return_value=shift), patch.object(win, "non_character_key", return_value=False):
        for name, mock in mocks.items():
            calls.attach_mock(mock, name)
        for name, value in {"take_suggestion": True, "jump_disputed": True, "accept_run": 0, **(returns or {})}.items():
            mocks[name].return_value = value          # what the real ones answer, so the chain reads them
        quit_ = tool.keys.handle(key)
    return calls.mock_calls, quit_, tool


@pytest.mark.parametrize("mode", BINDINGS)
def test_every_key_has_one_row_per_view(mode):
    bindings = BINDINGS[mode]
    assert len({key for key, _ in bindings}) == len(bindings), mode


@pytest.mark.parametrize("mode, key, expected",
                         [(mode, key, expected) for mode, bindings in BINDINGS.items() for key, expected in bindings],
                         ids=[f"{mode}-{key_name(key)}" for mode, bindings in BINDINGS.items() for key, _ in bindings])
def test_every_binding_in_every_view(mode, key, expected):
    """handle_key against the table: what each key does in each view, with every handler mocked."""
    assert press(mode, key)[0] == expected


def test_q_quits_unless_unsaved_labels_have_not_been_warned_about():
    for mode in BINDINGS:
        assert press(mode, ord("q"))[1]
        assert not press(mode, ord("w"))[1]
    tool = Tool([source("test")])
    tool.session.dirty = True
    with every_handler_mocked(), \
            patch.object(win, "non_character_key", return_value=False), patch("builtins.print"):
        assert not tool.keys.handle(ord("q"))            # once: the warning
        assert tool.quit_warned
        assert tool.keys.handle(27)                      # again: quits anyway


def test_y_moves_on_only_when_a_suggestion_was_taken():
    assert press("plain", ord("y"), returns={"take_suggestion": False})[0] == [call.take_suggestion()]
    assert press("grid", ord("y"), returns={"take_suggestion": False})[0] == [call.take_suggestion()]


def test_shift_reverses_e_and_makes_a_nudge_fine():
    assert press("plain", ord("e"), shift=True)[0] == [call.jump_disputed(-1)]
    assert press("plain", ord("i"), shift=True)[0] == [call.nudge(0.0, -0.25)]
    with patch("builtins.print") as printed:
        press("plain", ord("e"), returns={"jump_disputed": False})
    assert "no more frames" in printed.call_args[0][0]


def test_keys_that_change_state_rather_than_call_a_handler():
    for digit in "12468":
        assert press("plain", ord(digit))[2].session.step == int(digit)
    assert not press("plain", ord("t"))[2].session.trail
    tool = press("plain", ord("u"))[2]
    assert tool.session.dirty and tool.session.source.dirty
    for key, factor in ((ord("+"), ZOOM_STEP), (ord("="), ZOOM_STEP), (ord("-"), 1 / ZOOM_STEP), (ord("_"), 1 / ZOOM_STEP)):
        assert press("grid", key)[2].grid_view.zoom == pytest.approx(factor, abs=5e-8)


def test_guards_swallow_function_keys_and_keys_ahead_of_the_view():
    tool = Tool([source("test")])
    with every_handler_mocked() as mocks, \
            patch.object(win, "non_character_key", return_value=True):
        assert not tool.keys.handle(0x70)                  # F1 reads as p; it is not p
    assert not any(mock.called for mock in mocks.values())
    assert press("plain", ord(" "), ready=False)[0] == []    # the view lags: not SPACE
    assert press("plain", ord("q"), ready=False)[0] == [call.save()]   # but q still saves
    assert press("grid", ord(" "), ready=False)[0] == [call.confirm_grid()]   # no single frame to lag
