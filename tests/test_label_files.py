"""The shared label-file format, its sidecars, video facts and the importer's row rules, on
synthetic files in temporary folders. No workspace label or video is ever read."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from finetune import data_policy, fileio, labels, workspace
from finetune.import_data import ReviewedLabelsError, convert_rows, protect_existing, read_tracknet_csv
from finetune.workspace import VideoMeta, video_meta


# --------------------------------------------------------------------------- cells and the sentinel

@pytest.mark.parametrize("cell, index", [("frame_007", 7), ("frame_0", 0), (" frame_1234 ", 1234)])
def test_frame_cell_names_the_zero_based_source_frame(cell, index):
    assert labels.frame_index(cell) == index
    assert labels.frame_name(index) == f"frame_{index:03d}"


@pytest.mark.parametrize("cell", ["7", "frame_7x", "frame_12_5", "frame_", "", "Frame_7"])
def test_a_cell_that_is_not_exactly_frame_index_is_an_error_not_a_guess(cell):
    with pytest.raises(labels.LabelFormatError):
        labels.frame_index(cell)


def test_the_top_right_corner_means_no_ball():
    assert labels.sentinel(1920) == (1919, 0)
    assert not labels.is_visible(1919, 0, 1920, 1080)
    assert not labels.is_visible(1824, 54, 1920, 1080)      # anywhere in the 5% corner
    assert labels.is_visible(1823, 54, 1920, 1080)
    assert labels.is_visible(1900, 55, 1920, 1080)
    assert labels.is_visible(0, 0, 1920, 1080)


# --------------------------------------------------------------------------- label files

def test_write_then_read_keeps_frames_points_and_absent_balls(tmp_path):
    path = tmp_path / "clip_ball.csv"
    rows, visible = labels.write_labels(path, {4: (10.123, 20.5), 2: None, 6: (-3.0, 2000.0)}, 640, 360)
    assert (rows, visible) == (3, 2)
    text = path.read_text(encoding="utf-8")
    assert text.splitlines() == ["frame,ball_x,ball_y", "frame_002,639,0", "frame_004,10.12,20.50", "frame_006,0.00,359.00"]
    assert not list(tmp_path.glob("*.tmp"))
    assert labels.read_labels(path, 640, 360) == {2: None, 4: (10.12, 20.5), 6: (0.0, 359.0)}


def test_read_label_rows_keeps_file_order_and_draft_columns(tmp_path):
    path = tmp_path / "clip_ball.csv.draft"
    path.write_text("frame,ball_x,ball_y,source,conf\nframe_003,5,6,det,0.75\nframe_001,639,0,none,0.00\n", encoding="utf-8")
    rows = labels.read_label_rows(path)
    assert [row.frame for row in rows] == [3, 1]
    assert rows[0] == labels.LabelRow(3, 5.0, 6.0, "det", 0.75)
    assert rows[1].source == "none" and rows[1].conf == 0.0
    plain = tmp_path / "plain_ball.csv"
    plain.write_text("frame,ball_x,ball_y\nframe_001,1,2\n", encoding="utf-8-sig")   # a BOM is fine
    assert labels.read_label_rows(plain) == [labels.LabelRow(1, 1.0, 2.0, None, None)]


@pytest.mark.parametrize("content", ["frame,x,y\nframe_001,1,2\n", "frame,ball_x,ball_y\n7,1,2\n",
                                     "frame,ball_x,ball_y\nframe_001,one,2\n",
                                     "frame,ball_x,ball_y,source,conf\nframe_001,1,2,det,nan?\n"])
def test_a_malformed_label_file_is_an_error_naming_the_file(tmp_path, content):
    path = tmp_path / "bad_ball.csv"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(labels.LabelFormatError) as caught:
        labels.read_label_rows(path)
    assert "bad_ball.csv" in str(caught.value)


# --------------------------------------------------------------------------- sidecars

def test_review_sidecar_missing_is_none_and_damaged_is_an_error(tmp_path):
    path = tmp_path / "clip_ball.review.json"
    assert labels.read_review(path) is None
    assert labels.reviewed_frames("clip", tmp_path) == set()
    path.write_text(json.dumps({"reviewed": [1, "3"], "edited": [5], "cursor": 2}), encoding="utf-8")
    assert labels.reviewed_frames("clip", tmp_path) == {1, 3, 5}
    for broken in ("[1, 2]", "{nope", ""):
        path.write_text(broken, encoding="utf-8")
        with pytest.raises(labels.LabelFormatError) as caught:
            labels.read_review(path)
        assert path.name in str(caught.value)
        assert path.read_text(encoding="utf-8") == broken      # never replaced


def test_review_frame_lists_are_validated_once_for_every_reader(tmp_path):
    """reviewed_frames and label_store.apply_review share frame_list, so a sidecar whose frame
    list is not one is the same LabelFormatError naming the file for both; null lists are empty."""
    from finetune.label_store import Row, apply_review

    path = tmp_path / "clip_ball.review.json"
    path.write_text(json.dumps({"reviewed": None, "edited": None}), encoding="utf-8")
    assert labels.reviewed_frames("clip", tmp_path) == set()
    rows = [Row(1, 1.0, 1.0, True, "csv", 0.0)]
    apply_review(path, rows)
    assert not rows[0].reviewed
    path.write_text(json.dumps({"reviewed": ["x"]}), encoding="utf-8")
    for read in (lambda: labels.reviewed_frames("clip", tmp_path), lambda: apply_review(path, rows)):
        with pytest.raises(labels.LabelFormatError) as caught:
            read()
        assert path.name in str(caught.value) and "'reviewed'" in str(caught.value)
    path.write_text(json.dumps({"reviewed": [1]}), encoding="utf-8")
    apply_review(path, rows)
    assert rows[0].reviewed and labels.reviewed_frames("clip", tmp_path) == {1}


def test_flag_sidecar_reads_the_legacy_frames_list_as_check(tmp_path):
    path = tmp_path / "clip_ball.suspect.json"
    assert labels.read_flags(path) == {"generated": "", "notes": {}, "unplaceable": [], "check": [], "answered": {}}
    path.write_text(json.dumps({"frames": [1, 3]}), encoding="utf-8")
    assert labels.read_flags(path)["check"] == [1, 3]
    path.write_text(json.dumps({"unplaceable": [2], "check": [4], "answered": {"4": [1.5, 2.5]}, "notes": {"check": "n"},
                                "generated": "2026-10-01"}), encoding="utf-8")
    assert labels.read_flags(path) == {"generated": "2026-10-01", "notes": {"check": "n"}, "unplaceable": [2], "check": [4],
                                       "answered": {"4": [1.5, 2.5]}}
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(labels.LabelFormatError):
        labels.read_flags(path)
    path.write_text(json.dumps({"check": "3"}), encoding="utf-8")
    with pytest.raises(labels.LabelFormatError) as caught:
        labels.read_flags(path)
    assert path.name in str(caught.value)


def test_flag_sidecar_has_one_writer_and_one_shape(tmp_path):
    """write_flags owns the on-disk shape: provenance only when there is some, frames sorted and
    unique, and what it wrote reads back as it was given."""
    path = tmp_path / "clip_ball.suspect.json"
    flags = {"generated": "", "notes": {}, "unplaceable": [5, 3, 3], "check": [], "answered": {"1": None}}
    labels.write_flags(path, flags)
    assert json.loads(path.read_text(encoding="utf-8")) == {"unplaceable": [3, 5], "check": [], "answered": {"1": None}}
    full = {"generated": "2026-10-21", "notes": {"check": "look"}, "unplaceable": [], "check": [2], "answered": {}}
    labels.write_flags(path, full)
    assert json.loads(path.read_text(encoding="utf-8")) == full
    assert labels.read_flags(path) == full
    assert labels.answered_point((1.234, 5.678)) == [1.23, 5.68] and labels.answered_point(None) is None
    assert not list(tmp_path.glob("*.tmp"))


# --------------------------------------------------------------------------- atomic writes

def test_a_failed_write_leaves_the_old_file_and_no_temporary(tmp_path):
    path = tmp_path / "settings.json"
    fileio.write_atomic(path, "old")
    with pytest.raises(RuntimeError):
        with fileio.atomic_text(path) as handle:
            handle.write("half")
            raise RuntimeError("power cut")
    assert path.read_text(encoding="utf-8") == "old"
    backup = fileio.backup_copy(path, "test")
    assert backup.name.startswith("settings.json.before_test_") and backup.read_text(encoding="utf-8") == "old"


def test_copy_atomic_lands_the_whole_file_or_nothing(tmp_path):
    source, target = tmp_path / "new.npz", tmp_path / "sub" / "weights.npz"
    source.write_bytes(b"\x00" * 1000 + b"end")
    fileio.copy_atomic(source, target)
    assert target.read_bytes() == source.read_bytes() and not list(tmp_path.rglob("*.tmp"))
    fileio.copy_atomic(source, target)                         # over an existing file too
    assert target.read_bytes() == source.read_bytes()
    with pytest.raises(FileNotFoundError):
        fileio.copy_atomic(tmp_path / "missing", target)
    assert target.read_bytes() == source.read_bytes()          # the failed copy left it alone


# --------------------------------------------------------------------------- video facts

def write_clip(path: Path, count: int, size=(64, 36), fps=30.0) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), fps, size)
    if not writer.isOpened():
        pytest.skip("this OpenCV build cannot write FFV1")
    for index in range(count):
        writer.write(np.full((size[1], size[0], 3), index, np.uint8))
    writer.release()
    return path


def test_video_meta_reads_a_clip_and_raises_on_anything_else(tmp_path):
    clip = write_clip(tmp_path / "clip.avi", 7)
    meta = video_meta(clip)
    assert isinstance(meta, VideoMeta)
    assert (round(meta.fps), meta.width, meta.height, meta.total) == (30, 64, 36, 7)
    with pytest.raises(RuntimeError):
        video_meta(tmp_path / "missing.mp4")
    (tmp_path / "text.mp4").write_text("not a video", encoding="utf-8")
    with pytest.raises(RuntimeError):
        video_meta(tmp_path / "text.mp4")


# --------------------------------------------------------------------------- the importer's rules

def test_tracknet_visibility_rules(tmp_path):
    path = tmp_path / "Label.csv"
    path.write_text("file name,visibility,x-coordinate,y-coordinate,status\n"
                    "0000.jpg,1,100,200,0\n0001.jpg,0,0,0,0\n0002.jpg,2,50,60,0\n0003.jpg,3,70,80,0\n0004.jpg,1,0,0,0\n",
                    encoding="utf-8")
    rows = read_tracknet_csv(path, keep_occluded=False)
    assert rows[0] == (1, 100.0, 200.0)
    assert rows[1][0] == 0
    assert rows[2][0] == 2                     # hard to see still counts as visible
    assert rows[3][0] == 0                     # occluded: dropped unless asked for
    assert rows[4][0] == 0                     # "visible" at (0, 0) is no ball
    assert read_tracknet_csv(path, keep_occluded=True)[3] == (3, 70.0, 80.0)
    (tmp_path / "odd.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_tracknet_csv(tmp_path / "odd.csv", keep_occluded=False)


def test_convert_rows_keeps_frame_identity_and_drops_rows_past_the_video():
    raw = {0: (1, 100.0, 50.0), 1: (0, 0.0, 0.0), 2: (1, 640.0, 360.0), 3: (1, 1.0, 1.0)}
    grid_60fps = lambda index: 2 * index + 1          # GridTrackNet's own set: label k is frame 2k + 1
    converted, dropped = convert_rows(raw, grid_60fps, (1.5, 1.5), total=6)
    assert converted == {1: (150.0, 75.0), 3: None, 5: (960.0, 540.0)}
    assert dropped == 1
    every_frame, dropped = convert_rows(raw, lambda index: index, (1.0, 1.0), total=100)
    assert (list(every_frame), dropped) == ([0, 1, 2, 3], 0)


def test_import_refuses_to_overwrite_hand_reviewed_labels_and_backs_up_the_rest(tmp_path):
    csv_path = tmp_path / "clip_ball.csv"
    labels.write_labels(csv_path, {0: (1.0, 2.0)}, 640, 360)
    protect_existing(tmp_path / "other_ball.csv")             # nothing there: nothing to do
    protect_existing(csv_path)                                # unreviewed: backed up
    backups = list(tmp_path.glob("clip_ball.csv.before_import_*"))
    assert len(backups) == 1 and backups[0].read_text(encoding="utf-8") == csv_path.read_text(encoding="utf-8")
    (tmp_path / "clip_ball.review.json").write_text(json.dumps({"reviewed": [0]}), encoding="utf-8")
    with pytest.raises(ReviewedLabelsError):
        protect_existing(csv_path)
    assert len(list(tmp_path.glob("clip_ball.csv.before_import_*"))) == 1


# --------------------------------------------------------------------------- data policy

def test_eligibility_policy(tmp_path):
    assert data_policy.read_review_status(tmp_path) is None
    (tmp_path / workspace.REVIEW_STATUS_NAME).write_text('{"verified_video_min": 13}', encoding="utf-8")
    assert data_policy.read_review_status(tmp_path) == {"verified_video_min": 13}
    status = {"verified_video_min": 13, "verified_clips": ["grid_match4"]}
    assert data_policy.is_trainable("video13", "prof", status)
    assert not data_policy.is_trainable("video13", "custom-uncorrected", status)
    assert not data_policy.is_trainable("video12", "custom", status)
    assert data_policy.is_trainable("video12", "custom", None)
    manifest = {"video1": {"source": "custom"}, "video13": {"source": "prof"}, "grid_match4": {"source": "grid"}}
    assert data_policy.evaluation_clips(manifest, status) == ["video13", "grid_match4"]
    assert data_policy.evaluation_clips(manifest, None) == ["video1"]       # no policy: own camera only
    assert data_policy.evaluation_clips({}, None) is None                   # no manifest: the whole folder
