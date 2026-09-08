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
    return SimpleNamespace(clip=clip, rows=rows, width=width, height=height, frame_cache={},
                           dirty=False, cursor=0, start=0, missing=set(), saved_cursor=0)


class LabelWorkflowTests(unittest.TestCase):
    def tool(self, *sources):
        tool = Tool(list(sources or [source("test")]))
        tool.frame = lambda index: np.zeros((tool.source.height, tool.source.width, 3), np.uint8)
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


if __name__ == "__main__":
    unittest.main()
