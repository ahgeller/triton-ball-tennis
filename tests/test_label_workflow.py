"""Headless label-tool tests on synthetic frames; never open workspace labels."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import DEFAULT, Mock, call, patch

import cv2
import numpy as np

from finetune.labeling import label_window as win
from finetune.labeling.label_store import Row, load_review, load_rows, load_suspects, save_review
from finetune.labeling.label_tool import ZOOM_STEP, Tool
from finetune.labeling.label_window import KEYS_NEXT, KEYS_PREV


def source(clip, count=10, width=1280, height=720):
    rows = [Row(i, 30.25 + i*11, 24.75+i*3, True, "csv", 0.) for i in range(count)]
    for row in rows:
        row.clip = clip
    capture = SimpleNamespace(release=lambda: None, isOpened=lambda: False)
    return SimpleNamespace(clip=clip, rows=rows, width=width, height=height, frame_cache={},
                           dirty=False, cursor=0, start=0, missing=set(), saved_cursor=0,
                           capture=capture, position=0, release=lambda: None)


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
        tool.mouse.aim = ("old cursor feedback",)
        tool.move(1)
        self.assertIsNone(tool.anchor)
        self.assertIsNone(tool.mouse.aim)
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
        tool.mouse.aim = ("stale",)
        tool.fit_view(1000, 700)
        self.assertIsNone(tool.anchor)
        self.assertIsNone(tool.mouse.aim)
        self.assertEqual(tool.to_view(tool.rows[0].x, tool.rows[0].y),
                         (round(tool.view_w/2), round(tool.view_h/2)))

    def test_stale_wheel_still_cancels_native_window_zoom(self):
        tool = self.tool()
        with patch.object(tool.viewport, "repay_wheel") as repay:
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

    def test_grid_tiles_read_their_own_clip_without_moving_the_cursor(self):
        tool = self.tool(source("small", 3, 640, 360), source("large", 3, 1920, 1080))
        read = []
        tool.frame = lambda index: read.append((tool.source.clip, index)) or np.zeros(
            (tool.source.height, tool.source.width, 3), np.uint8)
        tool.toggle_grid()
        tool.grid_page(1)                                  # the page of clip "large": rows 3..5
        tool.grid_select(1)
        tool.render_grid()
        self.assertEqual(sorted(read), [("large", 0), ("large", 1), ("large", 2)])
        self.assertEqual(tool.cursor, 4)                   # rendering leaves the selection alone
        self.assertEqual([tile[0] for tile in tool.grid_tiles], [3, 4, 5])

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

    def test_review_evidence_is_applied_per_frame_and_bad_evidence_is_warned_about(self):
        import io
        from contextlib import redirect_stdout
        with tempfile.TemporaryDirectory() as directory:
            review = Path(directory) / "clip_ball.review.json"
            review.write_text(json.dumps({"reviewed": [1], "evidence": {
                "1": {"source": "det", "conf": 0.4}, "2": {"source": "det", "conf": "bad"}}}))
            rows = source("clip", 4).rows
            out = io.StringIO()
            with redirect_stdout(out):
                state = load_review(review, rows)
            self.assertEqual(state["reviewed"], [1])
            self.assertEqual((rows[1].source, rows[1].conf, rows[1].reviewed), ("det", 0.4, True))
            self.assertEqual((rows[2].source, rows[2].conf, rows[2].reviewed), ("csv", 0., False))
            self.assertIn("[warn]", out.getvalue())
            self.assertIn(review.name, out.getvalue())
            self.assertIn("frame 2", out.getvalue())

    def test_load_suspects_counts_a_legacy_frames_list_as_check_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            flags = Path(directory) / "clip_ball.suspect.json"
            flags.write_text(json.dumps({"unplaceable": [1], "frames": [3]}))
            rows = source("clip", 5).rows
            self.assertEqual(load_suspects(flags, rows), 2)
            self.assertEqual([row.frame for row in rows if row.suspect], [1, 3])
            self.assertEqual(load_suspects(flags.with_name("absent.json"), rows), 0)
            self.assertFalse(any(row.suspect for row in rows))

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

        from finetune.labeling.label_source import Source
        rows = [Row(i, 10., 10., True, "csv", 0.) for i in range(1, 60, 2)]   # odd phase, as 60 FPS clips
        with patch.object(cv2, "VideoCapture", lambda path: Capture()):
            clip = Source("c", Path("c.mp4"), rows, 64, 36, Path("c_ball.csv"), Path("c_ball.review.json"))
            clip.cache_limit = 64
            tool = Tool([clip])
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
        with patch.object(tool.viewport, "repay_wheel") as repay:
            tool.on_mouse(cv2.EVENT_MOUSEWHEEL, 100, 100, 120 << 16, None)
        repay.assert_called_once_with(120)
        tool.toggle_grid()
        tool.mouse.calibrating = True
        with patch.object(tool.viewport, "repay_wheel") as repay:
            tool.on_mouse(cv2.EVENT_MOUSEWHEEL, 100, 100, 120 << 16, None)
        repay.assert_called_once_with(120)

    def test_window_rebuilds_stop_when_the_check_keeps_failing(self):
        from finetune.labeling import label_window
        tool = self.tool()
        with patch.object(label_window, "_user32", None), patch.object(cv2, "destroyWindow"), \
                patch.object(cv2, "waitKey"), patch.object(tool, "create_window"), patch.object(tool, "show"):
            for _ in range(6):
                tool.viewport.reset_needed = True
                tool.reset_window()
        self.assertEqual(tool.viewport.window_resets, 3)
        self.assertEqual(tool.viewport.probe_at, float("inf"))
        self.assertFalse(tool.viewport.reset_needed)

    def test_function_keys_do_not_pass_for_letters(self):
        from finetune.labeling import label_window
        held = {0x71}                                # F2, which reads as "q"
        fake = SimpleNamespace(GetKeyState=lambda vk: 0x8000 if vk in held else 0)
        with patch.object(label_window, "_user32", fake):
            self.assertTrue(label_window.non_character_key(ord("q")))
            held.clear()                             # the q key itself: VK_F2 is not down
            self.assertFalse(label_window.non_character_key(ord("q")))
            self.assertFalse(label_window.non_character_key(ord("d")))

    def test_save_survives_a_locked_file_and_keeps_the_edits(self):
        import finetune.labeling.label_tool as label_tool
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
        from finetune.labeling.label_tool import save_suspects, load_suspects
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
        from finetune.data import label_audit
        with tempfile.TemporaryDirectory() as directory:
            labels = Path(directory)
            (labels / "clip_ball.csv").write_text("frame,ball_x,ball_y\nframe_001,41.25,27.75\nframe_003,50,30\n")
            (labels / "clip_ball.suspect.json").write_text(json.dumps(
                {"unplaceable": [], "check": [], "answered": {"1": [41.25, 27.75], "3": [10, 10]}}))
            with patch.multiple("finetune.data.workspace", LABELS=labels, MANIFEST=labels / "clips.csv"):
                label_audit.flag_frames("clip", "check", [1, 3], "audit")
                self.assertEqual(label_audit.open_flags("clip")["check"], [3])   # 3's label moved since
                self.assertEqual(label_audit.outstanding_flags("clip")["check"], [3])

    def test_damaged_review_file_is_an_error_that_names_it_and_is_never_replaced(self):
        from finetune.labeling.label_tool import LabelFormatError, build_source
        with tempfile.TemporaryDirectory() as directory:
            review = Path(directory) / "clip_ball.review.json"
            rows = source("clip", 3).rows
            for odd in ('{"reviewed": null, "edited": null}',             # parseable: tolerated field by field
                        '{"evidence": {"0": {"source": "det"}}}', '{"evidence": [1]}'):
                review.write_text(odd)
                load_review(review, rows)
            for broken in ("[1, 2]", "{not json"):                     # not a review file: an error naming it
                review.write_text(broken)
                with self.assertRaises(LabelFormatError) as caught:
                    load_review(review, rows)
                self.assertIn(review.name, str(caught.exception))
            # The tool skips such a clip rather than open it on a fresh review file.
            labels = Path(directory)
            (labels / "clip_ball.csv").write_text("frame,ball_x,ball_y\nframe_001,20,30\n")
            with patch("finetune.labeling.label_source.find_video", return_value=labels / "clip.mp4"), \
                    patch("finetune.labeling.label_source.video_meta", return_value=(30.0, 640, 360, 10)):
                self.assertIsNone(build_source("clip", labels, labels))
            self.assertEqual(review.read_text(), "{not json")

    def test_ranges_include_clips_that_only_have_a_draft(self):
        from finetune.labeling.label_tool import expand
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


def write_clip(path, frames, fps=30):
    """Encode ``frames`` losslessly (FFV1 in AVI, as tests/test_pipeline_run.py does), so what
    refine.suggest measures is the synthetic picture itself and not an encoder's noise."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), fps, (frames[0].shape[1], frames[0].shape[0]))
    if not writer.isOpened():
        raise unittest.SkipTest("this OpenCV build cannot write FFV1; the test needs a lossless clip")
    for image in frames:
        writer.write(image)
    writer.release()
    return path


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
        from finetune.labeling.refine import motion_centroid
        previous, current, following = ball_frames([(40, 30), (50, 40), (60, 50)])
        x, y = motion_centroid(previous, current, following, (52.0, 38.0), 9)
        self.assertAlmostEqual(x, 50.5, places=4)    # pixel 50's centre is 50.5 in label files
        self.assertAlmostEqual(y, 40.5, places=4)
        still = ball_frames([(50, 40)])[0]
        self.assertIsNone(motion_centroid(still, still, still, (50.5, 40.5), 9))

    def test_suggest_reads_a_video_and_scores_trust(self):
        from finetune.labeling import refine
        path_of = [(100 + 12 * i, 200 - 18 * i + i * i) for i in range(12)]   # an exact parabola
        with tempfile.TemporaryDirectory() as directory:
            video = write_clip(Path(directory) / "clip.avi", ball_frames(path_of))
            seeds = {i: (column + 0.9, row + 0.2) for i, (column, row) in enumerate(path_of)}
            found = refine.suggest(video, seeds, spacing=1)
        self.assertEqual(set(found), set(range(12)))       # the ends diff against one side only
        for frame, hint in found.items():
            column, row = path_of[frame]              # the 3x3 ball's middle pixel: centre at k + 0.5
            self.assertAlmostEqual(hint["x"], column + .5, places=3)   # lossless clip: exact
            self.assertAlmostEqual(hint["y"], row + .5, places=3)
        self.assertTrue(found[5]["trusted"] and found[5]["ball_here"])
        self.assertLess(found[5]["path"], 1.0)

    def test_a_placer_moves_the_chosen_points_and_a_far_move_is_not_trusted(self):
        from finetune.labeling import cnn_refiner, refine
        path_of = [(100 + 12 * i, 200 - 18 * i + i * i) for i in range(12)]
        seen = []

        class Nudge:                      # stands in for the network: 0.5 px right, frame 5 far off
            cut_crop = staticmethod(cnn_refiner.cut_crop)
            sure = None                   # a version without the self-check

            def place(self, items, height):
                seen.extend(crop.shape for crop, _, _ in items)
                return [(start[0] + (8.0 if abs(start[0] - 160.5) < 2 else 0.5), start[1], self.sure)
                        for _, _, start in items]
        with tempfile.TemporaryDirectory() as directory:
            video = write_clip(Path(directory) / "clip.avi", ball_frames(path_of))
            seeds = {i: (column + 0.9, row + 0.2) for i, (column, row) in enumerate(path_of)}
            plain = refine.suggest(video, seeds, spacing=1)
            placed = refine.suggest(video, seeds, spacing=1, placer=Nudge())
        self.assertEqual(set(seen), {(cnn_refiner.CROP, cnn_refiner.CROP, 9)})
        for frame, hint in placed.items():
            self.assertAlmostEqual(hint["centroid"][0], plain[frame]["x"], places=2)   # the measurement is kept
            moved = 8.0 if frame == 5 else 0.5
            self.assertAlmostEqual(hint["x"], hint["centroid"][0] + moved, places=2)
        self.assertTrue(plain[4]["trusted"] and placed[4]["trusted"])    # 1.5 px at 1080p: still trusted
        self.assertTrue(plain[5]["trusted"])
        self.assertFalse(placed[5]["trusted"])
        with tempfile.TemporaryDirectory() as directory:     # a self-check decides trust by itself
            video = write_clip(Path(directory) / "clip.avi", ball_frames(path_of))
            unsure, checked = Nudge(), Nudge()
            unsure.sure, checked.sure = False, True
            doubted = refine.suggest(video, seeds, spacing=1, placer=unsure)
            vouched = refine.suggest(video, seeds, spacing=1, placer=checked)
        self.assertFalse(any(h["trusted"] for h in doubted.values()))
        self.assertTrue(vouched[5]["trusted"])            # far from its measurement, but the network is sure           # 8 px at 360p is 24 px at 1080p from its measurement


class CnnRefinerTests(unittest.TestCase):
    def test_cut_centres_the_point_and_place_maps_window_pixels_back(self):
        import torch
        from finetune.labeling import cnn_refiner
        image = np.zeros((720, 1280, 3), np.uint8)
        image[300, 500] = 255
        crop = cnn_refiner.cut_crop([image] * 3, (500.5, 300.5), 720)     # pixel 500's centre, 720p -> 1080p scale
        self.assertEqual(crop.shape, (96, 96, 9))
        peak = np.unravel_index(np.argmax(crop[..., 4]), crop.shape[:2])
        self.assertTrue(47 <= peak[0] <= 48 and 47 <= peak[1] <= 48)
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
        self.assertTrue(sure)
        self.assertAlmostEqual(x, 500.5, places=3)
        self.assertAlmostEqual(y, 300.5, places=3)
        lo = (cnn_refiner.WINDOW - 1) / 2.0
        placer.net = lambda x: (torch.tensor([[lo + 3.0, lo]] * x.shape[0]),   # "3 px right" whatever it sees
                                torch.full((x.shape[0],), 2.0))
        (x, y, _), = placer.place([(crop, (500.5, 300.5), (502.5, 301.5))], 720)
        self.assertAlmostEqual(x, 502.5, places=4)                            # the mirrored look cancels it
        self.assertAlmostEqual(y, 301.5, places=4)

    def test_sure_threshold_keeps_far_misses_rare_and_folds_keep_a_game_together(self):
        from finetune.labeling import cnn_refiner
        confs = np.linspace(1.0, 0.0, 3000)
        errors = np.where(np.arange(3000) >= 2000, 9.0, 0.5)       # the least confident third is wrong
        threshold = cnn_refiner.sure_threshold(errors, confs)
        sure = confs >= threshold
        self.assertLessEqual(np.mean(errors[sure] > 5), cnn_refiner.SURE_FAR_SHARE)
        self.assertGreater(sure.sum(), 1990)                         # ... and nearly all the good ones are sure
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "clips.csv"
            manifest.write_text("clip,origin\n"
                                "video13,C:\\prof\\Dataset\\game2\\Clip1\n"
                                "video14,C:\\prof\\Dataset\\game2\\Clip2\n"
                                "video1,C:\\archive\\video1.mp4\n", encoding="utf-8")
            with patch.object(cnn_refiner, "HERE", Path(directory)):
                fold_of = cnn_refiner.folds(["video13", "video14", "video1", "grid_match4"],
                                            {"video13": 300, "video14": 200, "video1": 400, "grid_match4": 600})
        self.assertEqual(fold_of["video13"], fold_of["video14"])
        self.assertEqual(len(set(fold_of.values())), 3)

    def test_a_version_from_before_the_self_check_still_loads(self):
        import torch
        from finetune.labeling import cnn_refiner
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "refiner-20261019-120547.pt"
            torch.save({"state": cnn_refiner.build_network(False).state_dict(), "trained": "then"}, path)
            placer = cnn_refiner.Placer(path)
            crop = np.zeros((cnn_refiner.CROP, cnn_refiner.CROP, 9), np.uint8)
            (x, y, sure), = placer.place([(crop, (100.0, 100.0), (100.0, 100.0))], 1080)
        self.assertIsNone(placer.threshold)
        self.assertIsNone(sure)                                      # refine.py's own rule decides

    def test_versions_newest_is_current_until_one_is_chosen_and_old_ones_stay(self):
        from finetune.labeling import cnn_refiner
        with tempfile.TemporaryDirectory() as directory:
            models = Path(directory)
            with patch.object(cnn_refiner, "MODELS", models):     # current.txt is looked up under it
                self.assertIsNone(cnn_refiner.current())
                for name in ("refiner-20261001-090000.pt", "refiner-20261019-110000.pt"):
                    (models / name).write_bytes(b"")
                self.assertEqual(cnn_refiner.current().name, "refiner-20261019-110000.pt")
                cnn_refiner.make_current("20261001")
                self.assertEqual(cnn_refiner.current().name, "refiner-20261001-090000.pt")
                (models / "refiner-20261020-080000.pt").write_bytes(b"")
                self.assertEqual(cnn_refiner.current().name, "refiner-20261001-090000.pt")   # a choice sticks
                (models / "refiner-20261001-090000.pt").unlink()
                self.assertEqual(cnn_refiner.current().name, "refiner-20261020-080000.pt")   # ... unless it is gone
                self.assertEqual(len(cnn_refiner.versions()), 2)
                with self.assertRaises(ValueError):
                    cnn_refiner.make_current("2026")                 # ambiguous

    def test_label_tool_reloads_the_placer_when_the_current_version_changes(self):
        tool = LabelWorkflowTests.tool(self)
        first, second = Path("refiner-a.pt"), Path("refiner-b.pt")
        loaded = []
        with patch("finetune.labeling.label_hints.cnn_refiner.current", side_effect=[first, first, second, None]), \
                patch("finetune.labeling.label_hints.cnn_refiner.load_current",
                      side_effect=lambda: loaded.append(1) or SimpleNamespace(name=f"v{len(loaded)}")):
            self.assertEqual(tool.hints.current_placer().name, "v1")
            self.assertEqual(tool.hints.current_placer().name, "v1")       # unchanged: not loaded again
            self.assertEqual(tool.hints.current_placer().name, "v2")       # retrained: the new version
            self.assertIsNone(tool.hints.current_placer())                 # none left: the centroid alone
        self.assertEqual(len(loaded), 2)


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
        tool.hints.found["test"] = {2: hint(row, x=row.x + 1)}
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
        self.assertEqual(seen["offset"], 0.0)
        self.assertEqual(seen["conf"], .5)
        self.assertEqual(seen["clip"], (Path("clip.mp4"), 30, 1280, 720, 2))
        self.assertEqual(rows, {0: (12.0, 22.0, "det", .9)})      # the box centre; frame 1 had nothing


class TakeSuggestionTests(LabelWorkflowTests.__base__):
    def tool(self):
        tool = LabelWorkflowTests.tool(self)
        tool.hints.found["test"] = {0: hint(tool.rows[0], 31.0, 25.5)}
        tool.hints.version += 1
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
        tool.hints.found["test"] = hints
        tool.hints.version += 1
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
        tool.hints.found["test"] = {2: hint(row, 300.0, 200.0, trusted=False, seed=None)}
        self.assertIsNone(tool.suggestion_for(2))     # a guess is not enough to put a ball there
        tool.hints.found["test"][2]["trusted"] = True
        self.assertIsNotNone(tool.suggestion_for(2))
        tool.cursor = 2
        self.assertTrue(tool.take_suggestion())
        self.assertTrue(row.visible and (row.x, row.y) == (300.0, 200.0) and row.source == "refined")
        tool.hints.found["test"][2]["ball_here"] = False
        self.assertIsNone(tool.suggestion_for(2))     # nothing ball-like moved: never shown

    def test_a_changed_label_is_remeasured_with_checked_labels_steering(self):
        tool = LabelWorkflowTests.tool(self)
        calls = []
        with patch("finetune.labeling.label_hints.refine.suggest",
                   side_effect=lambda video, seeds, spacing, stop, confirmed, only, placer:
                   calls.append((seeds, confirmed, only)) or {}), \
                patch("finetune.labeling.label_hints.cnn_refiner.current", return_value=None):
            tool.source.video = Path("clip.mp4")
            tool.hints.suggest_clip(tool.source)
            self.assertEqual(calls[-1][1], [])                     # nothing checked yet
            before = tool.hints.label_state(tool.source)
            tool.cursor = 1
            tool.set_point(80, 40)
            self.assertNotEqual(tool.hints.label_state(tool.source), before)
            tool.hints.suggest_clip(tool.source)
            seeds, confirmed, only = calls[-1]
            self.assertEqual(seeds[1], (80.0, 40.0))
            self.assertEqual(confirmed, [1])
            self.assertEqual(only, set(range(10)))                # the stretch around frame 1 (10 rows)
            self.assertIsNone(calls[0][2])                         # the first pass measures everything
            self.assertEqual(tool.hints.measured_state["test"], tool.hints.label_state(tool.source))

    def test_suggestion_is_drawn_and_its_arrival_redraws(self):
        tool = LabelWorkflowTests.tool(self)
        tool.zoom = tool.zoom_target = 6.
        tool.follow_ball()
        plain = tool.render()
        orange = lambda image: int(np.sum((image[:, :, 0] == 0) & (image[:, :, 1] == 150) & (image[:, :, 2] == 255)))
        self.assertEqual(orange(plain), 0)
        tool.hints.found["test"] = {0: hint(tool.rows[0], 33.0, 27.0)}
        tool.hints.version += 1
        hinted = tool.render()
        self.assertIsNot(hinted, plain)
        self.assertGreater(orange(hinted), 0)


class OfferTests(LabelWorkflowTests.__base__):
    """On an unreviewed frame the suggestion is the mark: SPACE takes it, x keeps the draft."""

    def test_space_takes_the_offer_and_a_reviewed_label_stays_the_mark(self):
        tool = LabelWorkflowTests.tool(self)
        row = tool.rows[0]
        tool.hints.found["test"] = {0: hint(row, row.x + 1.0, row.y - 0.5)}
        tool.hints.version += 1
        tool.follow_ball()
        self.assertEqual(tool.target_point(), (31.25, 24.25))    # the view and the mouse go to the offer
        tool.render()
        tool.confirm()
        self.assertEqual((row.x, row.y, row.source), (31.25, 24.25, "refined"))
        self.assertTrue(row.reviewed and row.edited)
        tool.hints.found["test"][0] = hint(row, row.x + 2.0, row.y)   # re-measured from the new label
        tool.hints.version += 1
        self.assertIsNone(tool.offer_for(0))                     # reviewed: your label is the mark
        self.assertIsNotNone(tool.suggestion_for(0))             # the hint is still shown beside it
        tool.render()
        tool.confirm()
        self.assertEqual((row.x, row.y), (31.25, 24.25))

    def test_an_offer_that_was_not_drawn_is_not_taken(self):
        tool = LabelWorkflowTests.tool(self)
        row = tool.rows[0]
        tool.render()                                            # judged without a suggestion
        tool.hints.found["test"] = {0: hint(row, row.x + 1.0, row.y)}
        tool.hints.version += 1
        tool.confirm()                                           # SPACE lands after it arrived
        self.assertEqual((row.x, row.reviewed, row.edited), (30.25, True, False))
        tool.cursor = 1
        tool.hints.found["test"][1] = hint(tool.rows[1], tool.rows[1].x + 1.0, tool.rows[1].y)
        tool.hints.version += 1
        tool.render()
        tool.cursor = 2                                          # drawn for frame 1, pressed on 2
        tool.confirm()
        self.assertEqual((tool.rows[1].reviewed, tool.rows[2].edited), (False, False))
        self.assertTrue(tool.rows[2].reviewed)

    def test_x_keeps_the_draft_and_the_offer_rules(self):
        from finetune.labeling import label_hints
        tool = LabelWorkflowTests.tool(self)
        row = tool.rows[0]
        tool.hints.found["test"] = {0: hint(row, row.x + 1.0, row.y, trusted=False)}
        tool.hints.version += 1
        with patch.object(label_hints, "OFFER_UNSURE", False):
            self.assertIsNone(tool.offer_for(0))
        self.assertIsNotNone(tool.offer_for(0))                  # unsure but near the draft: offered, dull
        tool.hints.found["test"][0]["x"] = row.x + 2.5           # 3.75 px at 1080p: too far to offer unsure
        tool.hints.version += 1
        self.assertIsNone(tool.offer_for(0))
        self.assertIsNotNone(tool.suggestion_for(0))             # ... but still a hint beside the draft
        tool.hints.found["test"][0]["trusted"] = True            # sure: offered wherever it is
        tool.hints.version += 1
        self.assertIsNotNone(tool.offer_for(0))
        tool.hints.found["test"][0] = hint(row, row.x + 1.0, row.y, trusted=False)
        tool.hints.version += 1
        tool.render()
        tool.confirm(take_offer=False)                           # x: the draft was right
        self.assertEqual((row.x, row.reviewed, row.edited), (30.25, True, False))
        other = tool.rows[1]
        tool.hints.found["test"][1] = hint(other, other.x + 0.1, other.y)
        tool.hints.version += 1
        self.assertIsNone(tool.offer_for(1))                     # nothing to choose between
        for row in tool.rows:
            row.conf = .9
        tool.hints.found["test"][2] = hint(tool.rows[2], tool.rows[2].x + 1.0, tool.rows[2].y)
        tool.hints.version += 1
        tool.cursor = 1
        tool.render()
        tool.accept_run()                                        # r confirms the drafts, unseen
        self.assertEqual((tool.rows[2].x, tool.rows[2].edited), (52.25, False))
        self.assertTrue(all(row.reviewed for row in tool.rows))

    def test_confirm_grid_takes_the_offers_it_drew(self):
        tool = LabelWorkflowTests.tool(self)
        rows = tool.rows
        tool.hints.found["test"] = {0: hint(rows[0], rows[0].x + 1.0, rows[0].y),
                                    3: hint(rows[3], rows[3].x, rows[3].y + 1.0, trusted=False),
                                    9: hint(rows[9], rows[9].x + 1.0, rows[9].y)}
        tool.hints.version += 1
        tool.toggle_grid()
        tool.render_grid()
        tool.confirm_grid()
        self.assertEqual((rows[0].x, rows[0].source), (31.25, "refined"))
        self.assertEqual((rows[3].y, rows[3].source), (34.75, "refined"))
        self.assertTrue(all(row.reviewed for row in rows[:8]))
        self.assertFalse(rows[1].edited)
        self.assertFalse(rows[9].settled)                        # next page: not drawn, not taken
        self.assertEqual(rows[9].x, 129.25)

    def test_a_no_ball_draft_with_a_sure_hint_offers_it(self):
        tool = LabelWorkflowTests.tool(self)
        row = tool.rows[2]
        row.visible = False
        tool.cursor = 2
        tool.hints.found["test"] = {2: hint(row, 300.0, 200.0, seed=None)}
        tool.hints.version += 1
        tool.follow_ball()
        self.assertEqual(tool.target_point(), (300.0, 200.0))
        tool.render()
        tool.confirm()
        self.assertTrue(row.visible)
        self.assertEqual((row.x, row.y, row.source), (300.0, 200.0, "refined"))


# Every Tool method a key can reach, replaced by a mock while a key is pressed, so a key that
# calls the wrong one, or one too many, fails the table below.
KEY_HANDLERS = ("save", "toggle_sheet", "toggle_grid", "toggle_grid_size", "confirm", "confirm_grid", "move",
                "jump", "jump_disputed", "take_suggestion", "mark_invisible", "accept_run", "nudge", "zoom_key",
                "toggle_fullscreen", "remember_settings", "toggle_follow", "cycle_filter", "toggle_inset",
                "follow_ball", "grid_page", "grid_step", "grid_select", "prepare_grid", "sheet_page")
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


class KeyBindingTests(unittest.TestCase):
    """handle_key against the table: what each key does in each view, with every handler mocked."""

    def press(self, mode, key, returns=None, shift=False, ready=True):
        """(the handler calls the key made, whether the tool wants to quit, the tool)."""
        tool = Tool([source("test")])
        tool.grid, tool.sheet = mode == "grid", mode == "sheet"
        calls = Mock()
        with patch.multiple(Tool, **{name: DEFAULT for name in KEY_HANDLERS}) as mocks, \
                patch.object(Tool, "view_ready", return_value=ready), patch.object(Row, "undo"), \
                patch.object(win, "shift_held", return_value=shift), patch.object(win, "non_character_key", return_value=False):
            for name, mock in mocks.items():
                calls.attach_mock(mock, name)
            for name, value in {"take_suggestion": True, "jump_disputed": True, "accept_run": 0, **(returns or {})}.items():
                mocks[name].return_value = value          # what the real ones answer, so the chain reads them
            quit_ = tool.handle_key(key)
        return calls.mock_calls, quit_, tool

    def test_every_binding_in_every_view(self):
        for mode, bindings in BINDINGS.items():
            self.assertEqual(len({key for key, _ in bindings}), len(bindings), mode)   # one row per key
            for key, expected in bindings:
                with self.subTest(mode=mode, key=key):
                    self.assertEqual(self.press(mode, key)[0], expected)

    def test_q_quits_unless_unsaved_labels_have_not_been_warned_about(self):
        for mode in BINDINGS:
            self.assertTrue(self.press(mode, ord("q"))[1])
            self.assertFalse(self.press(mode, ord("w"))[1])
        tool = Tool([source("test")])
        tool.dirty = True
        with patch.multiple(Tool, **{name: DEFAULT for name in KEY_HANDLERS}), \
                patch.object(win, "non_character_key", return_value=False), patch("builtins.print"):
            self.assertFalse(tool.handle_key(ord("q")))           # once: the warning
            self.assertTrue(tool.quit_warned)
            self.assertTrue(tool.handle_key(27))                  # again: quits anyway

    def test_y_moves_on_only_when_a_suggestion_was_taken(self):
        self.assertEqual(self.press("plain", ord("y"), returns={"take_suggestion": False})[0], [call.take_suggestion()])
        self.assertEqual(self.press("grid", ord("y"), returns={"take_suggestion": False})[0], [call.take_suggestion()])

    def test_shift_reverses_e_and_makes_a_nudge_fine(self):
        self.assertEqual(self.press("plain", ord("e"), shift=True)[0], [call.jump_disputed(-1)])
        self.assertEqual(self.press("plain", ord("i"), shift=True)[0], [call.nudge(0.0, -0.25)])
        with patch("builtins.print") as printed:
            self.press("plain", ord("e"), returns={"jump_disputed": False})
        self.assertIn("no more frames", printed.call_args[0][0])

    def test_keys_that_change_state_rather_than_call_a_handler(self):
        for digit in "12468":
            self.assertEqual(self.press("plain", ord(digit))[2].step, int(digit))
        self.assertFalse(self.press("plain", ord("t"))[2].trail)
        tool = self.press("plain", ord("u"))[2]
        self.assertTrue(tool.dirty and tool.source.dirty)
        for key, factor in ((ord("+"), ZOOM_STEP), (ord("="), ZOOM_STEP), (ord("-"), 1 / ZOOM_STEP), (ord("_"), 1 / ZOOM_STEP)):
            self.assertAlmostEqual(self.press("grid", key)[2].grid_zoom, factor)

    def test_guards_swallow_function_keys_and_keys_ahead_of_the_view(self):
        tool = Tool([source("test")])
        with patch.multiple(Tool, **{name: DEFAULT for name in KEY_HANDLERS}) as mocks, \
                patch.object(win, "non_character_key", return_value=True):
            self.assertFalse(tool.handle_key(0x70))                  # F1 reads as p; it is not p
        self.assertFalse(any(mock.called for mock in mocks.values()))
        self.assertEqual(self.press("plain", ord(" "), ready=False)[0], [])    # the view lags: not SPACE
        self.assertEqual(self.press("plain", ord("q"), ready=False)[0], [call.save()])   # but q still saves
        self.assertEqual(self.press("grid", ord(" "), ready=False)[0], [call.confirm_grid()])   # no single frame to lag


if __name__ == "__main__":
    unittest.main()
