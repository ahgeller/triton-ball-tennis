"""finetune.label_audit on synthetic label files in a temporary workspace; no workspace file is read."""
import contextlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import pytest

from finetune import label_audit
from finetune.label_audit import (AUDIT_MIN_COVERAGE, AUDIT_MIN_FAST, ClipAudit, audit_labels, check_label_file,
                                  outstanding_flags, open_flags, shift_labels, flag_frames)
from finetune.labels import read_labels
from finetune.workspace import VideoMeta

WIDTH, HEIGHT = 1920, 1080


def write_csv(path: Path, points) -> Path:
    """A label file from (frame, x, y) rows; an empty list writes the header alone."""
    lines = ["frame,ball_x,ball_y"] + [f"frame_{frame:03d},{x},{y}" for frame, x, y in points]
    path.write_text("\n".join(lines) + "\n")
    return path


def moving_ball(frames, speed):
    """frame -> (x, y) of a ball crossing the frame at `speed` px/frame."""
    return {frame: (100.0 + speed * frame, 300.0) for frame in frames}


class Workspace(unittest.TestCase):
    """Every path the module touches is under the ``workspace_dir`` fixture's folder; ``self.folder``
    is its labels folder."""

    @pytest.fixture(autouse=True)
    def _workspace(self, workspace_dir):
        self.workspace = workspace_dir
        self.folder = workspace_dir / "labels"

    def label_file(self, clip, points):
        return write_csv(self.folder / f"{clip}_ball.csv", points)


class CheckLabelFileTests(Workspace):
    def check(self, points, fps=30.0, total=100):
        issues, rows = check_label_file(self.label_file("clip", points), fps, WIDTH, HEIGHT, total)
        return "; ".join(issues), rows

    def test_clean_30fps_file_has_no_issues(self):
        self.assertEqual(self.check([(f, 10 + f, 20) for f in range(8)]), ("", 8))

    def test_60fps_clip_labelled_on_every_frame(self):
        issues, rows = self.check([(f, 10 + f, 20) for f in range(8)], fps=60.0)
        self.assertIn("needs every 2", issues)
        self.assertEqual(rows, 8)

    def test_60fps_clip_labelled_on_both_phases(self):
        issues, _ = self.check([(f, 10 + f, 20) for f in (0, 2, 4, 7, 9)], fps=60.0)
        self.assertIn("labels mix even and odd", issues)
        self.assertNotIn("needs every", issues)

    def test_label_past_the_end_of_the_video(self):
        issues, _ = self.check([(f, 10, 20) for f in range(5)], total=3)
        self.assertIn("beyond the video's 3 frames", issues)

    def test_point_off_the_image(self):
        issues, rows = self.check([(0, 10, 20), (1, 2000, 10), (2, 10, 20)])
        self.assertIn("outside 1920x1080", issues)
        self.assertEqual(rows, 2)                  # reading stops at the bad point

    def test_unsupported_frame_rate(self):
        issues, _ = self.check([(f, 10, 20) for f in range(5)], fps=15.0)   # 24 counts as 30 (frame_rate)
        self.assertIn("neither 30 nor 60", issues)

    def test_empty_file(self):
        self.assertEqual(self.check([]), ("no label rows", 0))

    def test_duplicates_and_gaps(self):
        issues, _ = self.check([(0, 10, 20), (0, 11, 20), (20, 12, 20)])
        self.assertIn("duplicate frame rows", issues)
        self.assertIn("only 3 of 21 frames", issues)


class FlagSidecarTests(Workspace):
    def test_write_and_read_round_trip_keeps_the_other_kind(self):
        flag_frames("clip", "unplaceable", [5, 3, 3], "cut here")
        flag_frames("clip", "check", [8], "disputed")
        self.assertEqual(open_flags("clip"), {"unplaceable": [3, 5], "check": [8]})
        blob = json.loads((self.folder / "clip_ball.suspect.json").read_text())
        self.assertEqual(blob["notes"], {"unplaceable": "cut here", "check": "disputed"})
        flag_frames("clip", "unplaceable", [], "")
        flag_frames("clip", "check", [], "")
        self.assertFalse((self.folder / "clip_ball.suspect.json").exists())   # nothing left to ask

    def test_a_later_audit_does_not_reask_an_answered_frame(self):
        self.label_file("clip", [(1, 41.25, 27.75), (3, 50, 30)])
        (self.folder / "clip_ball.suspect.json").write_text(json.dumps(
            {"unplaceable": [], "check": [], "answered": {"1": [41.25, 27.75], "3": [10, 10]}}))
        flag_frames("clip", "check", [1, 3], "audit")
        self.assertEqual(open_flags("clip")["check"], [3])   # 3's label moved since
        self.assertEqual(outstanding_flags("clip")["check"], [3])

    def test_outstanding_drops_an_unplaceable_frame_with_no_ball(self):
        self.label_file("clip", [(1, WIDTH - 1, 0), (3, 50, 30)])     # frame 1 is the sentinel
        flag_frames("clip", "unplaceable", [1, 3], "cut")
        self.assertEqual(open_flags("clip")["unplaceable"], [1, 3])
        self.assertEqual(outstanding_flags("clip"), {"unplaceable": [3], "check": []})

    def test_nothing_flagged_reads_as_empty(self):
        self.assertEqual(outstanding_flags("absent"), {"unplaceable": [], "check": []})


class ClipAuditTests(unittest.TestCase):
    def test_agreeing_detector_is_ok(self):
        visible = moving_ball(range(30), 10)
        audit = ClipAudit(visible, dict(visible), 1, 1.0, 30)
        self.assertGreaterEqual(len(audit.fast), AUDIT_MIN_FAST)
        self.assertEqual(audit.agree0, 1.0)
        self.assertEqual(audit.best_shift, 0)
        self.assertEqual(audit.verdict(False), ("ok", False))
        self.assertEqual(audit.disputed_frames(), [])

    def test_one_contradicted_frame_is_disputed_the_clip_still_agrees(self):
        visible = moving_ball(range(30), 10)
        confident = dict(visible)
        confident[12] = (confident[12][0], confident[12][1] + 30)
        audit = ClipAudit(visible, confident, 1, 1.0, 30)
        self.assertEqual(audit.verdict(False)[1], False)
        self.assertEqual(audit.disputed_frames(), [12])

    def test_detector_two_label_frames_ahead_is_fixable(self):
        visible = moving_ball(range(40), 20)                     # fast enough that +1 and +3 miss
        confident = {frame + 2: point for frame, point in visible.items()}
        audit = ClipAudit(visible, confident, 1, 1.0, 40)
        self.assertGreaterEqual(len(audit.fast), AUDIT_MIN_FAST)
        self.assertEqual(audit.agree0, 0.0)
        self.assertEqual(audit.best_shift, 2)
        self.assertTrue(audit.fixable)
        verdict, suspect = audit.verdict(False)
        self.assertIn("+2 label frame(s) off", verdict)
        self.assertTrue(suspect)
        self.assertIn("-> fixed", audit.verdict(True)[0])

    def test_shift_counts_in_label_frames_on_a_60fps_clip(self):
        visible = moving_ball(range(0, 80, 2), 20)                 # the even phase
        confident = {frame + 4: point for frame, point in visible.items()}
        audit = ClipAudit(visible, confident, 2, 1.0, 80)
        self.assertEqual(audit.best_shift, 2)

    def test_detector_that_rarely_fires_cannot_judge(self):
        visible = moving_ball(range(40), 20)
        confident = {frame: (0.0, 0.0) for frame in range(5)}
        audit = ClipAudit(visible, confident, 1, 1.0, 40)
        self.assertLess(audit.coverage, AUDIT_MIN_COVERAGE)
        self.assertEqual(audit.verdict(False), ("detector rarely fires here; cannot judge", False))

    def test_hopeless_labels_disagree_and_are_excluded(self):
        visible = moving_ball(range(40), 20)
        confident = {frame: (1500.0, 900.0) for frame in range(40)}
        audit = ClipAudit(visible, confident, 1, 1.0, 40)
        self.assertFalse(audit.fixable)
        verdict, suspect = audit.verdict(False)
        self.assertTrue(suspect)
        self.assertIn("disagree with the detector", verdict)


class ShiftLabelsTests(Workspace):
    def test_refuses_when_review_metadata_exists(self):
        path = self.label_file("example", [(1, 20, 30)])
        original = path.read_text()
        path.with_suffix(".review.json").write_text(json.dumps({"reviewed": [1]}))
        with self.assertRaises(ValueError):
            shift_labels("example", 1, WIDTH, HEIGHT, 10)
        self.assertEqual(path.read_text(), original)

    def test_renumbers_backs_up_and_drops_rows_that_leave_the_video(self):
        path = self.label_file("example", [(0, 20, 30), (1, 21, 30), (2, 22, 30), (9, 29, 30)])
        original = path.read_text()
        with contextlib.redirect_stdout(io.StringIO()):
            shift_labels("example", 1, WIDTH, HEIGHT, 10)
        self.assertEqual(read_labels(path, WIDTH, HEIGHT), {1: (20.0, 30.0), 2: (21.0, 30.0), 3: (22.0, 30.0)})
        backups = list(self.folder.glob("example_ball.csv.before_shift_*"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_text(), original)

    def test_zero_shift_touches_nothing(self):
        path = self.label_file("example", [(0, 20, 30)])
        shift_labels("example", 0, WIDTH, HEIGHT, 10)
        self.assertEqual(list(self.folder.glob("example_ball.csv*")), [path])


class AuditLabelsTests(Workspace):
    """audit_labels with the detector replaced by a dictionary and no video on disk."""

    def run_audit(self, confident, **options):
        detector = {frame: (x, y, 0.9) for frame, (x, y) in confident.items()}
        output = io.StringIO()
        with patch.object(label_audit, "videos_in", lambda *_: {"clip": self.folder / "clip.mp4"}), \
                patch.object(label_audit, "video_meta", lambda *_: VideoMeta(30.0, WIDTH, HEIGHT, 40)), \
                patch("finetune.detect.ball_detector", lambda *_: object()), \
                patch("finetune.detect.detect_points", lambda *_: detector), \
                contextlib.redirect_stdout(output):
            suspects = audit_labels(None, None, **options)
        return suspects, output.getvalue()

    def test_mark_flags_the_disputed_frames_of_an_agreeing_clip(self):
        visible = moving_ball(range(30), 10)
        self.label_file("clip", [(f, x, y) for f, (x, y) in visible.items()])
        confident = dict(visible)
        confident[12] = (confident[12][0], confident[12][1] + 30)
        suspects, output = self.run_audit(confident, fix=False, mark=True)
        self.assertEqual(suspects, 0)
        self.assertIn("ok; 1 frame(s) to eyeball", output)
        self.assertEqual(open_flags("clip")["check"], [12])

    def test_fix_shifts_a_clip_the_detector_sees_two_frames_later(self):
        visible = moving_ball(range(30), 20)
        self.label_file("clip", [(f, x, y) for f, (x, y) in visible.items()])
        confident = {frame + 2: point for frame, point in visible.items()}
        suspects, output = self.run_audit(confident, fix=True)
        self.assertEqual(suspects, 1)
        self.assertIn("+2 label frame(s) off -> fixed", output)
        shifted = read_labels(self.folder / "clip_ball.csv", WIDTH, HEIGHT)
        self.assertEqual(shifted, {frame + 2: point for frame, point in visible.items() if frame + 2 < 40})
        self.assertFalse((self.workspace / "exclude.txt").exists())

    def test_fix_excludes_a_clip_nothing_lines_up(self):
        visible = moving_ball(range(30), 20)
        self.label_file("clip", [(f, x, y) for f, (x, y) in visible.items()])
        suspects, output = self.run_audit({frame: (1500.0, 900.0) for frame in range(30)}, fix=True)
        self.assertEqual(suspects, 1)
        self.assertIn("-> excluded", output)
        self.assertIn("clip  # audit", (self.workspace / "exclude.txt").read_text())

    def test_patterns_pick_clips_and_a_clip_without_a_video_is_skipped(self):
        visible = moving_ball(range(30), 10)
        for clip in ("clip", "other"):
            self.label_file(clip, [(f, x, y) for f, (x, y) in visible.items()])
        suspects, output = self.run_audit(dict(visible), fix=False)
        self.assertEqual(suspects, 0)
        self.assertIn("audit: 2 clips", output)
        self.assertNotIn("other ", output.split("verdict", 1)[1])   # no video named other: no row


if __name__ == "__main__":
    unittest.main()
