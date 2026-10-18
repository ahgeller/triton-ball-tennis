"""Headless annotation tests using synthetic frames; never open workspace labels."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from finetune.label_tool import Tool, Row, save_review, load_review, load_rows


def source(clip, count=10, width=1280, height=720):
    rows = [Row(i, 30.25 + i*11, 24.75+i*3, True, "csv", 0.) for i in range(count)]
    for row in rows:
        row.clip = clip
    capture = SimpleNamespace(release=lambda: None, isOpened=lambda: False)
    return SimpleNamespace(clip=clip, rows=rows, width=width, height=height, frame_cache={},
                           dirty=False, cursor=0, start=0, missing=set(), saved_cursor=0,
                           capture=capture, position=0)


class LabelWorkflowTests(unittest.TestCase):
    def tool(self, *sources):
        tool = Tool(list(sources or [source("test")]))
        frames = {}                   # one array per frame, as the real cache hands back
        tool.frame = lambda index: frames.setdefault(
            (tool.source.clip, index), np.zeros((tool.source.height, tool.source.width, 3), np.uint8))
        tool.follow_ball()
        return tool

    def test_zoomed_next_frame_discards_old_anchor_and_cursor_report(self):
        tool = self.tool()
        tool.zoom, tool.zoom_target = 5., 9.
        tool.anchor = ((50., 80.), (120., 250.))
        tool.aim = ("old cursor feedback",)
        tool.move(1)
        self.assertIsNone(tool.anchor)
        self.assertIsNone(tool.aim)
        for _ in range(10):
            tool.step_zoom()
            x, y = tool.to_view(tool.rows[1].x, tool.rows[1].y)
            self.assertEqual((x, y), (round(tool.view_w/2), round(tool.view_h/2)))
        self.assertEqual(tool.zoom, 9.)

    def test_queued_click_cannot_edit_an_unshown_next_frame(self):
        tool = self.tool()
        tool.zoom = tool.zoom_target = 8.
        tool.follow_ball()
        with patch.object(cv2, "imshow"):
            tool.show()
        tool.on_mouse(cv2.EVENT_LBUTTONDOWN, tool.view_w//2, tool.view_h//2, 0, None)
        self.assertEqual(tool.cursor, 1)
        tool.on_mouse(cv2.EVENT_LBUTTONDBLCLK, tool.view_w//2, tool.view_h//2, 0, None)
        self.assertEqual(tool.cursor, 1)
        self.assertFalse(tool.rows[1].settled)

    def test_crop_click_maps_to_subpixel_source_coordinates(self):
        tool = self.tool()
        tool.toggle_grid()
        tool.render_grid()
        index, box, ox, oy, scale = tool.grid_tiles[0]
        # Pick a valid point nearer the middle when the crop runs over an edge.
        x, y = (box[0]+box[2])//2 + 3, (box[1]+box[3])//2 + 2
        expected = (ox+(x+.5-box[0])/scale, oy+(y+.5-box[1])/scale)
        tool.grid_mouse(cv2.EVENT_LBUTTONDOWN, x, y)
        self.assertAlmostEqual(tool.rows[index].x, expected[0])
        self.assertAlmostEqual(tool.rows[index].y, expected[1])
        self.assertEqual(tool.grid_indices, list(range(8)))
        self.assertTrue(tool.rows[0].edited)

    def test_resize_drops_stale_zoom_anchor_and_keeps_target_centered(self):
        tool = self.tool()
        tool.zoom = tool.zoom_target = 10.
        tool.follow_ball()
        tool.anchor = ((1000, 600), (10, 10))
        tool.fit_view(1000, 700)
        self.assertIsNone(tool.anchor)
        self.assertEqual(tool.to_view(tool.rows[0].x, tool.rows[0].y),
                         (round(tool.view_w/2), round(tool.view_h/2)))

    def test_stale_wheel_still_cancels_native_window_zoom(self):
        tool = self.tool()
        with patch.object(tool, "repay_wheel") as repay:
            tool.on_mouse(cv2.EVENT_MOUSEWHEEL, 100, 100, 120 << 16, None)
        repay.assert_called_once_with(120)

    def test_confirm_grid_requires_displayed_page_and_preserves_missing(self):
        tool = self.tool()
        tool.toggle_grid()
        tool.confirm_grid()
        self.assertFalse(any(r.settled for r in tool.rows))
        tool.source.missing.add(2)
        tool.render_grid()
        tool.confirm_grid()
        self.assertTrue(all(r.settled for i, r in enumerate(tool.rows[:8]) if i != 2))
        self.assertFalse(tool.rows[2].settled)
        self.assertFalse(any(r.settled for r in tool.rows[8:]))
        tool.confirm_grid()  # next page has not been rendered yet
        self.assertFalse(any(r.settled for r in tool.rows[8:]))

    def test_mixed_resolution_pages_keep_source_identity(self):
        tool = self.tool(source("small", 2, 640, 360), source("large", 2, 1920, 1080))
        tool.toggle_grid()
        self.assertEqual(tool.grid_indices, [0, 1])
        tool.grid_page(1)
        self.assertEqual((tool.clip, tool.width, tool.height), ("large", 1920, 1080))
        self.assertEqual(tool.grid_indices, [2, 3])

    def test_undo_restores_review_flags(self):
        tool = self.tool()
        row = tool.rows[0]
        row.suspect = True
        tool.set_point(90, 100)
        row.undo()
        self.assertFalse(row.reviewed)
        self.assertFalse(row.edited)
        self.assertTrue(row.suspect)
        self.assertEqual((row.x, row.y), (30.25, 24.75))

    def test_evidence_survives_resume_without_inventing_confidence(self):
        with tempfile.TemporaryDirectory() as directory:
            csv = Path(directory)/"clip_ball.csv"
            review = csv.with_suffix(".review.json")
            csv.write_text("frame,ball_x,ball_y\nframe_001,20,30\n")
            rows = load_rows(csv, 640, 360, "csv")
            self.assertEqual(rows[0].conf, 0.)
            self.assertTrue(rows[0].uncertain)
            rows[0].source, rows[0].conf = "det", .43
            save_review(review, rows, 0, 4.)
            resumed = load_rows(csv, 640, 360, "csv")
            load_review(review, resumed)
            self.assertEqual((resumed[0].source, resumed[0].conf), ("det", .43))

    def test_accept_run_marks_each_clip_dirty_including_last_row(self):
        a, b = source("a", 2), source("b", 2)
        tool = self.tool(a, b)
        for row in tool.rows:
            row.conf = .9
        self.assertEqual(tool.accept_run(), 4)
        self.assertTrue(a.dirty and b.dirty)
        self.assertTrue(all(row.reviewed for row in tool.rows))

    def test_drawn_picture_and_stored_click_agree_to_the_pixel(self):
        tool = self.tool()
        blank = np.zeros((720, 1280, 3), np.uint8)
        image = blank.copy()
        image[50, 100] = 255                       # frame pixel (100, 50): centre at (100.5, 50.5)
        tool.rows[0].x, tool.rows[0].y = 100.5, 50.5
        tool.zoom = tool.zoom_target = 6.
        tool.follow_ball()
        tool.frame = lambda index: blank           # the same markers over nothing, to subtract
        markers = tool.render()[:tool.view_h].astype(int)
        tool.frame = lambda index: image
        drawn = tool.render()[:tool.view_h].astype(int)
        lit = np.argwhere(np.abs(drawn - markers).max(2) > 127)
        (top, left), (bottom, right) = lit.min(0), lit.max(0)
        click = ((left + right) // 2, (top + bottom) // 2)
        stored = tool.to_frame(click[0] + .5, click[1] + .5)
        # Clicking the middle of the drawn pixel stores its centre, to within half a view pixel;
        # the warp used to draw everything 0.5 - 0.5/scale frame px off from where clicks went.
        slack = .5 / tool.scale + 1e-9
        self.assertLessEqual(abs(stored[0] - 100.5), slack, stored)
        self.assertLessEqual(abs(stored[1] - 50.5), slack, stored)
        self.assertTrue(left <= tool.to_view(100.5, 50.5)[0] <= right)
        for u, v in ((0, 0), (37, 91), (tool.view_w - 1, tool.view_h - 1)):
            self.assertEqual(tool.to_view(*tool.to_frame(u + .5, v + .5)), (u, v))

    def test_idle_render_is_cached_and_every_change_redraws(self):
        tool = self.tool()
        frames = {}
        tool.frame = lambda index: frames.setdefault(index, np.zeros((720, 1280, 3), np.uint8))
        tool.follow_ball()
        first = tool.render()
        with patch.object(cv2, "warpAffine", wraps=cv2.warpAffine) as warp:
            self.assertIs(tool.render(), first)          # nothing changed: nothing drawn
            self.assertEqual(warp.call_count, 0)
        changes = {
            "label moved": lambda t: setattr(t.rows[0], "x", t.rows[0].x + 1),
            "another row reviewed (counter)": lambda t: setattr(t.rows[7], "reviewed", True),
            "flagged": lambda t: setattr(t.rows[0], "suspect", True),
            "filter": lambda t: setattr(t, "filter", 1),
            "inset": lambda t: setattr(t, "inset", not t.inset),
            "trail": lambda t: setattr(t, "trail", not t.trail),
            "follow": lambda t: setattr(t, "follow", not t.follow),
            "step": lambda t: setattr(t, "step", 4),
            "unsaved": lambda t: setattr(t, "dirty", not t.dirty),
            "zoom": lambda t: setattr(t, "zoom", t.zoom * 1.5),
            "pan": lambda t: setattr(t, "center", (t.center[0] + 7, t.center[1])),
            "walk": lambda t: t.only_uncertain(),
            "ghost": lambda t: setattr(t, "ghost", (400., 300.)),
            "next frame": lambda t: t.move(1),
            "frame re-decoded": lambda t: frames.update({t.rows[t.cursor].frame: np.ones((720, 1280, 3), np.uint8)}),
            "window resized": lambda t: t.fit_view(900, 600),
        }
        for name, change in changes.items():
            before = tool.render()
            change(tool)
            self.assertIsNot(tool.render(), before, name)

    def test_show_resends_only_when_needed(self):
        tool = self.tool()
        with patch.object(cv2, "imshow") as imshow:
            tool.show()
            tool.show()
            self.assertEqual(imshow.call_count, 1)       # the same picture is not re-sent
            tool.shown = None                            # what reset_window does for a new window
            tool.show()
            self.assertEqual(imshow.call_count, 2)
            tool.toggle_grid()
            tool.show()
            tool.toggle_grid()
            tool.show()                                  # back from the grid: the frame view again
            self.assertEqual(imshow.call_count, 4)

    def test_stepping_back_near_the_start_hits_the_cache(self):
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

        rows = [Row(i, 10., 10., True, "csv", 0.) for i in range(1, 60, 2)]   # odd phase, as 60 FPS clips
        clip = source("c")
        clip.rows, clip.width, clip.height = rows, 64, 36
        for row in rows:
            row.clip = "c"
        clip.label_frames, clip.sig_of, clip.end, clip.cache_limit = {r.frame for r in rows}, {}, None, 64
        clip.capture, clip.video = Capture(), Path("c.mp4")
        tool = Tool([clip])
        with patch.object(cv2, "VideoCapture", lambda path: Capture()):
            tool.cursor = len(rows) - 1
            tool.frame(rows[-1].frame)
            Capture.opened = 0
            for i in range(len(rows) - 1, -1, -1):       # `a` all the way back to the start
                tool.cursor = i
                self.assertEqual(int(tool.frame(rows[i].frame)[0, 0, 0]), rows[i].frame)
        self.assertLessEqual(Capture.opened, 2)          # was one reopen per press

    def test_point_in_the_no_ball_corner_is_refused(self):
        tool = self.tool()
        before = (tool.rows[0].x, tool.rows[0].y)
        self.assertFalse(tool.set_point(1270, 10))  # would read back as "no ball"
        self.assertEqual((tool.rows[0].x, tool.rows[0].y), before)
        self.assertFalse(tool.rows[0].settled)
        self.assertTrue(tool.set_point(600, 300))

    def test_click_beyond_the_frame_is_not_a_label(self):
        tool = self.tool()
        with patch.object(cv2, "imshow"):
            tool.show()
        tool.on_mouse(cv2.EVENT_LBUTTONDOWN, 0, 0, 0, None)   # top-left of a view centred near (30, 25)
        self.assertEqual(tool.cursor, 0)
        self.assertFalse(tool.rows[0].settled)

    def test_wheel_in_grid_or_while_calibrating_still_cancels_window_zoom(self):
        tool = self.tool()
        tool.toggle_grid()
        with patch.object(tool, "repay_wheel") as repay:
            tool.on_mouse(cv2.EVENT_MOUSEWHEEL, 100, 100, 120 << 16, None)
        repay.assert_called_once_with(120)
        tool.toggle_grid()
        tool.calibrating = True
        with patch.object(tool, "repay_wheel") as repay:
            tool.on_mouse(cv2.EVENT_MOUSEWHEEL, 100, 100, 120 << 16, None)
        repay.assert_called_once_with(120)

    def test_window_rebuilds_stop_when_the_check_keeps_failing(self):
        import finetune.label_tool as label_tool
        tool = self.tool()
        with patch.object(label_tool, "_user32", None), patch.object(cv2, "destroyWindow"), \
                patch.object(cv2, "waitKey"), patch.object(tool, "create_window"), patch.object(tool, "show"):
            for _ in range(6):
                tool.reset_window()
        self.assertEqual(tool.window_resets, 3)
        self.assertEqual(tool.probe_at, float("inf"))

    def test_function_keys_do_not_pass_for_letters(self):
        import finetune.label_tool as label_tool
        held = {0x71}                                # F2, which reads as "q"
        fake = SimpleNamespace(GetKeyState=lambda vk: 0x8000 if vk in held else 0)
        with patch.object(label_tool, "_user32", fake):
            self.assertTrue(label_tool.non_character_key(ord("q")))
            held.clear()                             # the q key itself: VK_F2 is not down
            self.assertFalse(label_tool.non_character_key(ord("q")))
            self.assertFalse(label_tool.non_character_key(ord("d")))

    def test_save_survives_a_locked_file_and_keeps_the_edits(self):
        import finetune.label_tool as label_tool
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            a, b = source("a", 2), source("b", 2)
            for clip in (a, b):
                clip.out_csv = directory / f"{clip.clip}_ball.csv"
                clip.review = directory / f"{clip.clip}_ball.review.json"
                clip.suspects = directory / f"{clip.clip}_ball.suspect.json"
            tool = self.tool(a, b)
            a.dirty = b.dirty = tool.dirty = True
            real = label_tool.save_rows

            def locked(path, *rest):
                if path == a.out_csv:
                    raise PermissionError(13, "in use by another program")
                real(path, *rest)
            with patch.object(label_tool, "save_rows", side_effect=locked):
                tool.save(quiet=True)
            self.assertTrue(b.out_csv.is_file())     # one stuck clip does not stop the others
            self.assertTrue(a.dirty and tool.dirty)  # ... and its edits are kept for the next try
            self.assertFalse(b.dirty)
            tool.rescue_unsaved()
            self.assertTrue((directory / "a_ball.unsaved.csv").is_file())

    def test_legacy_flag_list_survives_a_save(self):
        from finetune.label_tool import save_suspects, load_suspects
        with tempfile.TemporaryDirectory() as directory:
            flags = Path(directory) / "clip_ball.suspect.json"
            flags.write_text(json.dumps({"frames": [1, 3]}))
            rows = source("clip", 5).rows
            self.assertEqual(load_suspects(flags, rows), 2)
            save_suspects(flags, rows)               # nothing settled: nothing may be dropped
            self.assertEqual(load_suspects(flags, rows), 2)
            rows[1].suspect = False                  # answered (SPACE or an edit in the tool)
            save_suspects(flags, rows)
            self.assertEqual(json.loads(flags.read_text()),
                             {"unplaceable": [], "check": [3], "answered": {"1": [41.25, 27.75]}})

    def test_new_audit_flag_reopens_a_reviewed_frame_until_answered(self):
        tool = self.tool()
        for done in tool.rows[:8]:
            done.reviewed = True                     # the start of the clip is reviewed ...
        row = tool.rows[4]
        row.suspect = True                           # ... and an audit after that flagged frame 4
        self.assertTrue(row.uncertain)
        tool.cursor = 0
        tool.jump(1)
        self.assertEqual(tool.cursor, 4)             # f goes to it even though it was reviewed
        self.assertEqual(tool.only_uncertain() and 4 in tool.walk, True)
        tool.walk = []
        tool.cursor = 4
        tool.confirm()                               # SPACE: it is right where it is
        self.assertFalse(row.suspect)
        row.undo()
        self.assertTrue(row.suspect)                 # undo puts the question back

    def test_a_later_audit_does_not_reask_an_answered_frame(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "finetune"))
        import ft
        with tempfile.TemporaryDirectory() as directory:
            labels = Path(directory)
            (labels / "clip_ball.csv").write_text("frame,ball_x,ball_y\nframe_001,41.25,27.75\nframe_003,50,30\n")
            (labels / "clip_ball.suspect.json").write_text(json.dumps(
                {"unplaceable": [], "check": [], "answered": {"1": [41.25, 27.75], "3": [10, 10]}}))
            with patch.object(ft, "LABELS", labels), patch.object(ft, "read_manifest", lambda: {}):
                ft.write_suspects("clip", "check", [1, 3], "audit")
                self.assertEqual(ft.read_suspects("clip")["check"], [3])   # 3's label moved since
                self.assertEqual(ft.outstanding_suspects("clip")["check"], [3])

    def test_damaged_review_file_does_not_stop_the_tool(self):
        with tempfile.TemporaryDirectory() as directory:
            review = Path(directory) / "clip_ball.review.json"
            rows = source("clip", 3).rows
            for broken in ("[1, 2]", '{"reviewed": null, "edited": null}',
                           '{"evidence": {"0": {"source": "det"}}}', '{"evidence": [1]}'):
                review.write_text(broken)
                load_review(review, rows)            # must not raise

    def test_ranges_include_clips_that_only_have_a_draft(self):
        from finetune.label_tool import expand
        with tempfile.TemporaryDirectory() as directory:
            labels = Path(directory)
            for name in ("v1_ball.csv", "v2_ball.csv.draft", "v3_ball.csv", "v4_ball.unsaved.csv"):
                (labels / name).write_text("frame,ball_x,ball_y\n")
            self.assertEqual(expand(["v1..v3"], labels), ["v1", "v2", "v3"])


def hint(row, x=None, y=None, trusted=True, ball_here=True, seed="label"):
    """A refine.suggest() entry for `row`: at (x, y), measured from its current label."""
    seed = (row.x, row.y) if seed == "label" else seed
    return {"x": row.x if x is None else x, "y": row.y if y is None else y, "seed": seed, "from": "label",
            "move": 0.0, "path": 0.3, "blobs": 1, "ratio": 1.0, "contrast": 40.0,
            "trusted": trusted, "ball_here": ball_here}


def ball_frames(positions, size=(360, 640)):
    """Grey frames with a 3x3 bright ball whose middle pixel is at each (column, row)."""
    frames = []
    for column, row in positions:
        image = np.full(size + (3,), 90, np.uint8)
        image[row - 1:row + 2, column - 1:column + 2] = (60, 230, 230)
        frames.append(image)
    return frames


class SuggestionTests(unittest.TestCase):
    def test_centroid_finds_the_ball_in_label_coordinates(self):
        from finetune.refine import motion_centroid
        previous, current, following = ball_frames([(40, 30), (50, 40), (60, 50)])
        x, y = motion_centroid(previous, current, following, (52.0, 38.0), 9)
        self.assertAlmostEqual(x, 50.5, places=4)    # pixel 50's centre is 50.5 in label files
        self.assertAlmostEqual(y, 40.5, places=4)
        still = ball_frames([(50, 40)])[0]
        self.assertIsNone(motion_centroid(still, still, still, (50.5, 40.5), 9))

    def test_suggest_reads_a_video_and_scores_trust(self):
        from finetune import refine
        path_of = [(100 + 12 * i, 200 - 18 * i + i * i) for i in range(12)]   # an exact parabola
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "clip.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 30, (640, 360))
            for image in ball_frames(path_of):
                writer.write(image)
            writer.release()
            seeds = {i: (column + 0.9, row + 0.2) for i, (column, row) in enumerate(path_of)}
            found = refine.suggest(video, seeds, spacing=1)
        self.assertEqual(set(found), set(range(12)))       # the ends diff against one side only
        for frame, hint in found.items():
            column, row = path_of[frame]
            self.assertLess(abs(hint["x"] - (column + .5)), .35)   # MJPG is lossy, so not exact
            self.assertLess(abs(hint["y"] - (row + .5)), .35)
        self.assertTrue(found[5]["trusted"] and found[5]["ball_here"])
        self.assertLess(found[5]["path"], 1.0)


class GridKeyboardTests(unittest.TestCase):
    def tool(self):
        tool = Tool([source("test", 40)])
        tool.frame = lambda index: np.zeros((720, 1280, 3), np.uint8)
        tool.follow_ball()
        return tool

    def test_sixteen_tiles_page_and_select_from_the_keyboard(self):
        tool = self.tool()
        tool.grid_size = 16
        tool.toggle_grid()
        tool.render_grid()
        self.assertEqual(len(tool.grid_tiles), 16)
        boxes = [box for _, box, *_ in tool.grid_tiles]
        self.assertEqual(len({box[1] for box in boxes}), 4)          # four rows of four
        self.assertTrue(all(box[3] < tool.view_h for box in boxes))  # none run into the bar
        tool.grid_select(5)
        self.assertEqual(tool.cursor, 5)
        tool.grid_step(-1)
        self.assertEqual(tool.cursor, 4)
        tool.grid_select(15)
        tool.grid_step(1)                                            # off the end: next page
        self.assertEqual(tool.grid_indices, list(range(16, 32)))
        self.assertEqual(tool.cursor, 16)
        tool.grid_step(-1)                                           # and back onto the last tile
        self.assertEqual(tool.grid_indices, list(range(16)))
        self.assertEqual(tool.cursor, 15)

    def test_switching_size_keeps_the_selected_frame(self):
        tool = self.tool()
        tool.toggle_grid()
        tool.grid_select(6)
        tool.toggle_grid_size()
        self.assertEqual(tool.grid_size, 16)
        self.assertEqual(len(tool.grid_indices), 16)
        self.assertEqual(tool.cursor, 6)
        tool.toggle_grid_size()
        self.assertEqual(tool.grid_size, 8)

    def test_y_in_the_grid_takes_the_selected_tiles_suggestion_and_moves_on(self):
        tool = self.tool()
        tool.toggle_grid()
        tool.grid_select(2)
        row = tool.rows[2]
        tool.suggestions["test"] = {2: hint(row, x=row.x + 1)}
        self.assertTrue(tool.take_suggestion())
        tool.grid_step(1)
        self.assertEqual((row.source, tool.cursor), ("refined", 3))


class TrajectorySheetTests(unittest.TestCase):
    def tool(self, *sources):
        tool = Tool(list(sources or [source("test", 150)]))
        frames = {}
        tool.frame = lambda index: frames.setdefault((tool.clip, index),
                                                     np.zeros((tool.source.height, tool.source.width, 3), np.uint8))
        tool.follow_ball()
        return tool

    def test_sheet_draws_a_stretch_and_a_click_opens_that_frame(self):
        tool = self.tool()
        tool.cursor = 70
        tool.toggle_sheet()
        self.assertTrue(tool.sheet and not tool.grid)
        self.assertEqual(tool.sheet_start, 40)                 # the cursor's stretch, centred on it
        tool.render()
        self.assertEqual([p[0] for p in tool.sheet_points], list(range(40, 100)))
        index, x, y = tool.sheet_points[25]
        tool.sheet_mouse(cv2.EVENT_LBUTTONDOWN, x + 2, y - 1)
        self.assertFalse(tool.sheet)
        self.assertEqual(tool.cursor, index)
        self.assertFalse(any(row.settled for row in tool.rows))  # finding a frame edits nothing

    def test_sheet_pages_stay_inside_the_clip(self):
        tool = self.tool(source("a", 70), source("b", 70))
        tool.cursor = 5
        tool.toggle_sheet()
        self.assertEqual(tool.sheet_start, 0)
        tool.sheet_page(1)
        self.assertEqual(tool.sheet_start, 10)                 # the last full stretch of clip a
        tool.sheet_page(1)
        self.assertEqual(tool.sheet_start, 10)
        tool.render()
        self.assertTrue(all(tool.rows[i].clip == "a" for i, *_ in tool.sheet_points))
        tool.sheet_page(-1)
        self.assertEqual(tool.sheet_start, 0)

    def test_a_click_far_from_every_point_does_nothing(self):
        tool = self.tool()
        tool.toggle_sheet()
        tool.render()
        cursor = tool.cursor
        tool.sheet_mouse(cv2.EVENT_LBUTTONDOWN, tool.view_w - 1, tool.view_h - 1)
        self.assertTrue(tool.sheet)
        self.assertEqual(tool.cursor, cursor)
        tool.sheet_mouse(cv2.EVENT_RBUTTONDOWN, 0, 0)          # right click goes back
        self.assertFalse(tool.sheet)


class DraftTests(unittest.TestCase):
    def test_drafts_use_the_unshifted_detector_point(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "finetune"))
        import pretrack
        from tennis_tracker import detectors
        seen = {}

        class Detector:
            def __init__(self, weights, cfg):
                seen["offset"] = cfg.gridtracknet_y_offset_px
                self.precomputed = [[([10, 20, 14, 24], .9)]]

            def prepare_video(self, *args):
                pass
        with patch.object(detectors, "GridTrackNetBallDetector", Detector):
            rows = pretrack.raw_rows(Path("clip.mp4"), Path("w.npz"), .5, "cpu", 30, 1280, 720, 1)
        self.assertEqual(seen["offset"], 0.0)
        self.assertEqual(rows[0][:2], (12.0, 22.0))


class TakeSuggestionTests(LabelWorkflowTests.__base__):
    def tool(self):
        tool = LabelWorkflowTests.tool(self)
        tool.suggestions["test"] = {0: hint(tool.rows[0], 31.0, 25.5)}
        tool.suggest_version += 1
        return tool

    def test_y_takes_the_suggestion_as_refined_and_undo_restores_it(self):
        tool = self.tool()
        row = tool.rows[0]
        self.assertTrue(tool.take_suggestion())
        self.assertEqual((row.x, row.y, row.source), (31.0, 25.5, "refined"))
        self.assertTrue(row.edited)
        row.undo()
        self.assertEqual((row.x, row.y, row.source), (30.25, 24.75, "csv"))
        tool.take_suggestion()
        tool.set_point(32, 26)                       # then corrected by hand
        self.assertEqual(row.source, "click")

    def test_suggestion_is_dropped_once_the_label_moves_away_from_its_seed(self):
        tool = self.tool()
        self.assertIsNotNone(tool.suggestion_for(0))
        tool.rows[0].x += 40                         # far from where the suggestion was measured
        self.assertIsNone(tool.suggestion_for(0))
        self.assertFalse(tool.take_suggestion())
        self.assertIsNone(tool.suggestion_for(1))    # no suggestion measured for this frame

    def test_verdict_flags_labels_off_a_trusted_suggestion_and_e_walks_them(self):
        tool = LabelWorkflowTests.tool(self)     # 1280x720: 1 frame px is 1.5 px at 1080p
        hints = {row.frame: hint(row, x=row.x + 1.0) for row in tool.rows}
        hints[3]["x"] = tool.rows[3].x + 2.5          # 3.75 px at 1080p: the label is off
        hints[6]["x"] = tool.rows[6].x + 2.5
        hints[8]["trusted"] = False                   # the suggestion itself is not trusted here
        hints[8]["x"] = tool.rows[8].x + 2.5
        tool.suggestions["test"] = hints
        tool.suggest_version += 1
        self.assertEqual(tool.hint_verdict(0), "agrees")
        self.assertEqual(tool.hint_verdict(3), "differs")
        self.assertIsNone(tool.hint_verdict(8))
        self.assertEqual(tool.disputed_count(), 2)
        self.assertTrue(tool.jump_disputed(1))
        self.assertEqual(tool.cursor, 3)
        self.assertTrue(tool.jump_disputed(1))
        self.assertEqual(tool.cursor, 6)
        self.assertFalse(tool.jump_disputed(1))
        self.assertTrue(tool.jump_disputed(-1))
        self.assertEqual(tool.cursor, 3)
        tool.set_point(tool.rows[3].x + 2.5, tool.rows[3].y)   # fixed: now it agrees
        self.assertEqual(tool.disputed_count(), 1)
        self.assertFalse(any(row.reviewed for row in tool.rows if row.frame != 3))   # nothing auto-accepted

    def test_no_ball_frame_shows_only_a_trusted_suggestion_and_y_places_it(self):
        tool = LabelWorkflowTests.tool(self)
        row = tool.rows[2]
        row.visible = False
        tool.suggestions["test"] = {2: hint(row, 300.0, 200.0, trusted=False, seed=None)}
        self.assertIsNone(tool.suggestion_for(2))     # a guess is not enough to put a ball there
        tool.suggestions["test"][2]["trusted"] = True
        self.assertIsNotNone(tool.suggestion_for(2))
        tool.cursor = 2
        self.assertTrue(tool.take_suggestion())
        self.assertTrue(row.visible and (row.x, row.y) == (300.0, 200.0) and row.source == "refined")
        tool.suggestions["test"][2]["ball_here"] = False
        self.assertIsNone(tool.suggestion_for(2))     # nothing ball-like moved: never shown

    def test_a_changed_label_is_remeasured_with_checked_labels_steering(self):
        tool = LabelWorkflowTests.tool(self)
        calls = []
        with patch("finetune.label_tool.refine.suggest",
                   side_effect=lambda video, seeds, spacing, stop, confirmed, only: calls.append((seeds, confirmed, only)) or {}):
            tool.source.video = Path("clip.mp4")
            tool.suggest_clip(tool.source)
            self.assertEqual(calls[-1][1], [])                     # nothing checked yet
            before = tool.label_state(tool.source)
            tool.cursor = 1
            tool.set_point(80, 40)
            self.assertNotEqual(tool.label_state(tool.source), before)
            tool.suggest_clip(tool.source)
            seeds, confirmed, only = calls[-1]
            self.assertEqual(seeds[1], (80.0, 40.0))
            self.assertEqual(confirmed, [1])
            self.assertEqual(only, set(range(10)))                # the stretch around frame 1 (10 rows)
            self.assertIsNone(calls[0][2])                         # the first pass measures everything
            self.assertEqual(tool.suggested_state["test"], tool.label_state(tool.source))

    def test_suggestion_is_drawn_and_its_arrival_redraws(self):
        tool = LabelWorkflowTests.tool(self)
        tool.zoom = tool.zoom_target = 6.
        tool.follow_ball()
        plain = tool.render()
        orange = lambda image: int(np.sum((image[:, :, 0] == 0) & (image[:, :, 1] == 150) & (image[:, :, 2] == 255)))
        self.assertEqual(orange(plain), 0)
        tool.suggestions["test"] = {0: hint(tool.rows[0], 33.0, 27.0)}
        tool.suggest_version += 1
        hinted = tool.render()
        self.assertIsNot(hinted, plain)
        self.assertGreater(orange(hinted), 0)


if __name__ == "__main__":
    unittest.main()
