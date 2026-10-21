"""finetune/data/import_data.py's importers on synthetic datasets under tmp_path, into the
``workspace_dir`` fixture's workspace: the frame-phase and pixel mapping of each source, and
the provenance that lands in clips.csv. No real dataset, label or video is read."""
import io
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from finetune.data import import_data
from finetune.data.import_data import (GRID_LABEL_CANVAS, ImportOptions, Manifest, Provenance, cadence_frame, canvas_scale,
                                       grid_frame, import_archive, import_grid, import_tracknetv2)
from finetune.data.labels import LabelFormatError, read_labels
from finetune.data.workspace import read_manifest
from tests._helpers import write_clip

SIZE = (64, 36)      # the synthetic videos' frame size; GridTrackNet's label canvas is 1280x720, 20x bigger


# --------------------------------------------------------------------------- the mappings themselves

@pytest.mark.parametrize("index, stride, frame", [(0, 2, 1), (1, 2, 3), (7, 2, 15), (0, 1, 0), (7, 1, 7)])
def test_grid_label_k_is_frame_2k_plus_1_at_60_fps_and_k_at_30(index, stride, frame):
    assert grid_frame(index, stride) == frame


@pytest.mark.parametrize("index, stride, frame", [(0, 2, 0), (1, 2, 2), (7, 1, 7)])
def test_a_cadence_label_k_is_frame_k_times_stride(index, stride, frame):
    assert cadence_frame(index, stride) == frame


def test_canvas_scale_takes_the_label_canvas_to_the_video():
    assert canvas_scale((1280, 720), 1920, 1080) == (1.5, 1.5)
    assert canvas_scale(GRID_LABEL_CANVAS, 64, 36) == (0.05, 0.05)
    assert canvas_scale(None, 1920, 1080) == (1.0, 1.0)


# --------------------------------------------------------------------------- the importers end to end

def tracknet_csv(path: Path, rows) -> Path:
    """A TrackNet-style Label.csv from (index, visibility, x, y) rows."""
    lines = ["Frame,Visibility,X,Y"] + [f"{i},{v},{x},{y}" for i, v, x, y in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run(importer, src: Path, options=ImportOptions()) -> tuple:
    manifest = Manifest()
    out = io.StringIO()
    with redirect_stdout(out):
        count = importer(src, manifest, options)
    manifest.save()
    return count, manifest, out.getvalue()


@pytest.mark.parametrize("fps, frame_of", [(60.0, lambda k: 2 * k + 1), (30.0, lambda k: k)])
def test_import_grid_puts_label_k_on_the_right_phase_and_scales_from_the_label_canvas(workspace_dir, tmp_path, fps, frame_of):
    src = tmp_path / "Match-Data"
    match = src / "match3"
    match.mkdir(parents=True)
    write_clip(match / "match3.avi", 12, SIZE, fps)
    tracknet_csv(match / "Labels.csv", [(0, 1, 640, 360), (1, 0, 0, 0), (2, 1, 1280, 720), (3, 2, 200, 100), (99, 1, 1, 1)])

    count, manifest, out = run(import_grid, src)
    assert count == 1
    labels = read_labels(workspace_dir / "labels" / "grid_match3_ball.csv", *SIZE)
    sx, sy = canvas_scale(GRID_LABEL_CANVAS, *SIZE)
    assert labels == {frame_of(0): (640 * sx, 360 * sy), frame_of(1): None, frame_of(2): (63.0, 35.0),
                      frame_of(3): (200 * sx, 100 * sy)}                         # (1280, 720) clamps onto the image
    assert "1 label rows beyond the video's 12 frames were dropped" in out      # index 99 maps past the clip
    assert (workspace_dir / "videos" / "grid_match3.avi").is_file()
    row = read_manifest()["grid_match3"]
    assert (row["source"], row["group"], row["stride"], row["label_frames"], row["visible"]) == (
        "grid", "grid_match3", str(2 if fps == 60.0 else 1), "4", "3")
    assert row["note"] == "labels scaled from 1280x720" and row["origin"].endswith("match3.avi")


def test_import_tracknetv2_labels_sit_at_model_cadence_from_frame_0_in_native_pixels(workspace_dir, tmp_path):
    src = tmp_path / "TrackNetV2"
    (src / "Test" / "match1" / "csv").mkdir(parents=True)
    (src / "Test" / "match1" / "video").mkdir()
    write_clip(src / "Test" / "match1" / "video" / "1_01.avi", 10, SIZE, 60.0)
    tracknet_csv(src / "Test" / "match1" / "csv" / "1_01_ball.csv", [(0, 1, 10, 20), (1, 1, 30, 31), (2, 3, 50, 30)])

    count, manifest, out = run(import_tracknetv2, src)
    assert count == 1
    clip = "tnv2_Test_match1_1_01"
    assert read_labels(workspace_dir / "labels" / f"{clip}_ball.csv", *SIZE) == {0: (10.0, 20.0), 2: (30.0, 31.0), 4: None}
    row = read_manifest()[clip]
    assert (row["source"], row["section"], row["group"]) == ("tracknetv2-badminton", "Test", "tnv2_Test_match1")
    assert "shuttlecock" in row["note"] and "visibility values converted" in out
    count, _, _ = run(import_tracknetv2, src, ImportOptions(keep_occluded=True, force=True))
    assert read_labels(workspace_dir / "labels" / f"{clip}_ball.csv", *SIZE)[4] == (50.0, 30.0)


def test_import_archive_copies_labels_as_they_are_and_refuses_a_malformed_file(workspace_dir, tmp_path):
    src = tmp_path / "archive"
    src.mkdir()
    write_clip(src / "video2.avi", 5, SIZE)
    (src / "video2_ball.csv").write_text("frame,ball_x,ball_y\nframe_001,10.123,20.456\nframe_003,63,0\n", encoding="utf-8")
    count, manifest, _ = run(import_archive, src)
    assert count == 1
    copied = workspace_dir / "labels" / "video2_ball.csv"
    assert copied.read_bytes() == (src / "video2_ball.csv").read_bytes()          # byte for byte, not reformatted
    row = read_manifest()["video2"]
    assert (row["source"], row["section"], row["group"], row["label_frames"], row["visible"]) == (
        "custom", "archive", "own-camera", "2", "1")
    assert not list(workspace_dir.glob("labels/*.tmp"))

    (src / "video2_ball.csv").write_text("frame,ball_x,ball_y\nframe_001,bad,20\n", encoding="utf-8")
    before = copied.read_bytes()
    with pytest.raises(LabelFormatError):
        run(import_archive, src, ImportOptions(force=True))
    assert copied.read_bytes() == before                                           # nothing written first
    assert not list(workspace_dir.glob("labels/video2_ball.csv.before_import_*"))


def test_dry_run_plans_without_writing_and_kept_clips_are_not_redone(workspace_dir, tmp_path):
    src = tmp_path / "archive"
    src.mkdir()
    write_clip(src / "video1.avi", 3, SIZE)
    (src / "video1_ball.csv").write_text("frame,ball_x,ball_y\nframe_000,1,2\n", encoding="utf-8")
    count, manifest, out = run(import_archive, src, ImportOptions(dry_run=True))
    assert count == 1 and "[plan] video1" in out
    assert not (workspace_dir / "labels" / "video1_ball.csv").exists() and manifest.rows == {}
    run(import_archive, src)
    _, _, out = run(import_archive, src)
    assert "[   kept] video1" in out and not list(workspace_dir.glob("labels/*.before_import_*"))
    _, _, out = run(import_archive, src, ImportOptions(force=True))
    assert "[ copied] video1" in out and len(list(workspace_dir.glob("labels/*.before_import_*"))) == 1


def test_provenance_and_options_are_plain_records():
    assert Provenance("grid") == Provenance("grid", "", "", "")
    assert ImportOptions() == ImportOptions(keep_occluded=False, force=False, link=False, dry_run=False)
    assert set(import_data.DEFAULT_SOURCES) == {"archive", "video11", "tracknetv2", "grid", "prof"}
