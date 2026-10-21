"""finetune.data.label_audit on synthetic label files in a temporary workspace; no workspace file is read.
Every path the module touches is under the ``workspace_dir`` fixture's folder."""
import contextlib
import io
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from finetune.data import label_audit
from finetune.data.label_audit import (AUDIT_MIN_COVERAGE, ClipAudit, audit_labels, check_label_file,
                                       outstanding_flags, open_flags, shift_labels, flag_frames)
from finetune.data.labels import read_labels
from finetune.data.scoring import MIN_FAST
from finetune.data.workspace import VideoMeta

WIDTH, HEIGHT = 1920, 1080


def write_csv(path: Path, points) -> Path:
    """A label file from (frame, x, y) rows; an empty list writes the header alone."""
    lines = ["frame,ball_x,ball_y"] + [f"frame_{frame:03d},{x},{y}" for frame, x, y in points]
    path.write_text("\n".join(lines) + "\n")
    return path


def moving_ball(frames, speed):
    """frame -> (x, y) of a ball crossing the frame at `speed` px/frame."""
    return {frame: (100.0 + speed * frame, 300.0) for frame in frames}


@pytest.fixture
def labels(workspace_dir):
    """The relocated workspace's labels folder."""
    return workspace_dir / "labels"


def label_file(labels: Path, clip, points):
    return write_csv(labels / f"{clip}_ball.csv", points)


# --------------------------------------------------------------------------- check_label_file

def check(labels, points, fps=30.0, total=100):
    issues, rows = check_label_file(label_file(labels, "clip", points), fps, WIDTH, HEIGHT, total)
    return "; ".join(issues), rows


def test_clean_30fps_file_has_no_issues(labels):
    assert check(labels, [(f, 10 + f, 20) for f in range(8)]) == ("", 8)


def test_60fps_clip_labelled_on_every_frame(labels):
    issues, rows = check(labels, [(f, 10 + f, 20) for f in range(8)], fps=60.0)
    assert "needs every 2" in issues
    assert rows == 8


def test_60fps_clip_labelled_on_both_phases(labels):
    issues, _ = check(labels, [(f, 10 + f, 20) for f in (0, 2, 4, 7, 9)], fps=60.0)
    assert "labels mix even and odd" in issues
    assert "needs every" not in issues


def test_label_past_the_end_of_the_video(labels):
    issues, _ = check(labels, [(f, 10, 20) for f in range(5)], total=3)
    assert "beyond the video's 3 frames" in issues


def test_point_off_the_image(labels):
    issues, rows = check(labels, [(0, 10, 20), (1, 2000, 10), (2, 10, 20)])
    assert "outside 1920x1080" in issues
    assert rows == 2                           # reading stops at the bad point


def test_unsupported_frame_rate(labels):
    issues, _ = check(labels, [(f, 10, 20) for f in range(5)], fps=15.0)   # 24 counts as 30 (frame_rate)
    assert "neither 30 nor 60" in issues


def test_empty_file(labels):
    assert check(labels, []) == ("no label rows", 0)


def test_duplicates_and_gaps(labels):
    issues, _ = check(labels, [(0, 10, 20), (0, 11, 20), (20, 12, 20)])
    assert "duplicate frame rows" in issues
    assert "only 3 of 21 frames" in issues


# --------------------------------------------------------------------------- the flag sidecar

def test_write_and_read_round_trip_keeps_the_other_kind(labels):
    flag_frames("clip", "unplaceable", [5, 3, 3], "cut here")
    flag_frames("clip", "check", [8], "disputed")
    assert open_flags("clip") == {"unplaceable": [3, 5], "check": [8]}
    blob = json.loads((labels / "clip_ball.suspect.json").read_text())
    assert blob["notes"] == {"unplaceable": "cut here", "check": "disputed"}
    flag_frames("clip", "unplaceable", [], "")
    flag_frames("clip", "check", [], "")
    assert not (labels / "clip_ball.suspect.json").exists()   # nothing left to ask


def test_a_later_audit_does_not_reask_an_answered_frame(labels):
    label_file(labels, "clip", [(1, 41.25, 27.75), (3, 50, 30)])
    (labels / "clip_ball.suspect.json").write_text(json.dumps(
        {"unplaceable": [], "check": [], "answered": {"1": [41.25, 27.75], "3": [10, 10]}}))
    flag_frames("clip", "check", [1, 3], "audit")
    assert open_flags("clip")["check"] == [3]   # 3's label moved since
    assert outstanding_flags("clip")["check"] == [3]


def test_outstanding_drops_an_unplaceable_frame_with_no_ball(labels):
    label_file(labels, "clip", [(1, WIDTH - 1, 0), (3, 50, 30)])     # frame 1 is the sentinel
    flag_frames("clip", "unplaceable", [1, 3], "cut")
    assert open_flags("clip")["unplaceable"] == [1, 3]
    assert outstanding_flags("clip") == {"unplaceable": [3], "check": []}


def test_nothing_flagged_reads_as_empty(labels):
    assert outstanding_flags("absent") == {"unplaceable": [], "check": []}


# --------------------------------------------------------------------------- ClipAudit

def test_agreeing_detector_is_ok():
    visible = moving_ball(range(30), 10)
    audit = ClipAudit(visible, dict(visible), 1, 1.0, 30)
    assert len(audit.fast) >= MIN_FAST
    assert audit.agree0 == 1.0
    assert audit.best_shift == 0
    assert audit.verdict(False) == ("ok", False)
    assert audit.disputed_frames() == []


def test_one_contradicted_frame_is_disputed_the_clip_still_agrees():
    visible = moving_ball(range(30), 10)
    confident = dict(visible)
    confident[12] = (confident[12][0], confident[12][1] + 30)
    audit = ClipAudit(visible, confident, 1, 1.0, 30)
    assert audit.verdict(False)[1] is False
    assert audit.disputed_frames() == [12]


def test_detector_two_label_frames_ahead_is_fixable():
    visible = moving_ball(range(40), 20)                     # fast enough that +1 and +3 miss
    confident = {frame + 2: point for frame, point in visible.items()}
    audit = ClipAudit(visible, confident, 1, 1.0, 40)
    assert len(audit.fast) >= MIN_FAST
    assert audit.agree0 == 0.0
    assert audit.best_shift == 2
    assert audit.fixable
    verdict, suspect = audit.verdict(False)
    assert "+2 label frame(s) off" in verdict
    assert suspect
    assert "-> fixed" in audit.verdict(True)[0]


def test_shift_counts_in_label_frames_on_a_60fps_clip():
    visible = moving_ball(range(0, 80, 2), 20)                 # the even phase
    confident = {frame + 4: point for frame, point in visible.items()}
    audit = ClipAudit(visible, confident, 2, 1.0, 80)
    assert audit.best_shift == 2


def test_detector_that_rarely_fires_cannot_judge():
    visible = moving_ball(range(40), 20)
    confident = {frame: (0.0, 0.0) for frame in range(5)}
    audit = ClipAudit(visible, confident, 1, 1.0, 40)
    assert audit.coverage < AUDIT_MIN_COVERAGE
    assert audit.verdict(False) == ("detector rarely fires here; cannot judge", False)


def test_hopeless_labels_disagree_and_are_excluded():
    visible = moving_ball(range(40), 20)
    confident = {frame: (1500.0, 900.0) for frame in range(40)}
    audit = ClipAudit(visible, confident, 1, 1.0, 40)
    assert not audit.fixable
    verdict, suspect = audit.verdict(False)
    assert suspect
    assert "disagree with the detector" in verdict


# --------------------------------------------------------------------------- shift_labels

def test_refuses_when_review_metadata_exists(labels):
    path = label_file(labels, "example", [(1, 20, 30)])
    original = path.read_text()
    path.with_suffix(".review.json").write_text(json.dumps({"reviewed": [1]}))
    with pytest.raises(ValueError):
        shift_labels("example", 1, WIDTH, HEIGHT, 10)
    assert path.read_text() == original


def test_renumbers_backs_up_and_drops_rows_that_leave_the_video(labels):
    path = label_file(labels, "example", [(0, 20, 30), (1, 21, 30), (2, 22, 30), (9, 29, 30)])
    original = path.read_text()
    with contextlib.redirect_stdout(io.StringIO()):
        shift_labels("example", 1, WIDTH, HEIGHT, 10)
    assert read_labels(path, WIDTH, HEIGHT) == {1: (20.0, 30.0), 2: (21.0, 30.0), 3: (22.0, 30.0)}
    backups = list(labels.glob("example_ball.csv.before_shift_*"))
    assert len(backups) == 1
    assert backups[0].read_text() == original


def test_zero_shift_touches_nothing(labels):
    path = label_file(labels, "example", [(0, 20, 30)])
    shift_labels("example", 0, WIDTH, HEIGHT, 10)
    assert list(labels.glob("example_ball.csv*")) == [path]


def test_shift_refuses_review_metadata_without_touching_labels(labels):
    path = labels / "example_ball.csv"
    original = "frame,ball_x,ball_y\nframe_001,20,30\n"
    path.write_text(original)
    path.with_suffix(".review.json").write_text(json.dumps({"reviewed": [1]}))
    with pytest.raises(ValueError):
        shift_labels("example", 1, 1920, 1080, 10)
    assert path.read_text() == original


# --------------------------------------------------------------------------- audit_labels
# audit_labels with the detector replaced by a dictionary and no video on disk.

def run_audit(labels, confident, **options):
    detector = {frame: (x, y, 0.9) for frame, (x, y) in confident.items()}
    output = io.StringIO()
    with patch.object(label_audit, "videos_in", lambda *_: {"clip": labels / "clip.mp4"}), \
            patch.object(label_audit, "video_meta", lambda *_: VideoMeta(30.0, WIDTH, HEIGHT, 40)), \
            patch("finetune.training.detect.ball_detector", lambda *_: object()), \
            patch("finetune.training.detect.detect_points", lambda *_: detector), \
            contextlib.redirect_stdout(output):
        suspects = audit_labels(None, None, **options)
    return suspects, output.getvalue()


def test_mark_flags_the_disputed_frames_of_an_agreeing_clip(labels):
    visible = moving_ball(range(30), 10)
    label_file(labels, "clip", [(f, x, y) for f, (x, y) in visible.items()])
    confident = dict(visible)
    confident[12] = (confident[12][0], confident[12][1] + 30)
    suspects, output = run_audit(labels, confident, fix=False, mark=True)
    assert suspects == 0
    assert "ok; 1 frame(s) to eyeball" in output
    assert open_flags("clip")["check"] == [12]


def test_fix_shifts_a_clip_the_detector_sees_two_frames_later(workspace_dir, labels):
    visible = moving_ball(range(30), 20)
    label_file(labels, "clip", [(f, x, y) for f, (x, y) in visible.items()])
    confident = {frame + 2: point for frame, point in visible.items()}
    suspects, output = run_audit(labels, confident, fix=True)
    assert suspects == 1
    assert "+2 label frame(s) off -> fixed" in output
    shifted = read_labels(labels / "clip_ball.csv", WIDTH, HEIGHT)
    assert shifted == {frame + 2: point for frame, point in visible.items() if frame + 2 < 40}
    assert not (workspace_dir / "exclude.txt").exists()


def test_fix_excludes_a_clip_nothing_lines_up(workspace_dir, labels):
    visible = moving_ball(range(30), 20)
    label_file(labels, "clip", [(f, x, y) for f, (x, y) in visible.items()])
    suspects, output = run_audit(labels, {frame: (1500.0, 900.0) for frame in range(30)}, fix=True)
    assert suspects == 1
    assert "-> excluded" in output
    assert "clip  # audit" in (workspace_dir / "exclude.txt").read_text()


def test_patterns_pick_clips_and_a_clip_without_a_video_is_skipped(labels):
    visible = moving_ball(range(30), 10)
    for clip in ("clip", "other"):
        label_file(labels, clip, [(f, x, y) for f, (x, y) in visible.items()])
    suspects, output = run_audit(labels, dict(visible), fix=False)
    assert suspects == 0
    assert "audit: 2 clips" in output
    assert "other " not in output.split("verdict", 1)[1]   # no video named other: no row
