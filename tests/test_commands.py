"""finetune/commands.py, the ft.py commands that work in this process, on the ``workspace_dir``
fixture's empty workspace and synthetic clips: the workspace facts status prints, the paths
command and the camera measurement. promote, check and status themselves run through
ft.run_cli in test_ft.py."""
import math

from finetune import commands
from finetune.commands import (clip_counts, cmd_paths, label_frame_size, label_stats, measure_camera_motion,
                               print_own_camera_clips, print_unfinished_clips, review_progress, sha8, shown)
from finetune.data import workspace
from finetune.data.labels import write_labels
from tennis_tracker.paths import ROOT
from tests._helpers import write_clip

SIZE = (64, 36)


def test_label_stats_and_counts_prefer_the_manifest_row(workspace_dir, capsys):
    csv_path = workspace_dir / "labels" / "rally_ball.csv"
    write_labels(csv_path, {0: (1.0, 2.0), 1: None, 2: (3.0, 4.0)}, *SIZE)
    assert label_stats(csv_path, *SIZE) == (3, 2)
    assert clip_counts("rally", csv_path, {"label_frames": "9", "visible": "7"}, {}) == (9, 7)
    # No video to size the sentinel by: counted as 1080p, where a 64x36 clip's "no ball" corner is a point
    assert clip_counts("rally", csv_path, None, {}) == (3, 3)
    assert clip_counts("rally", csv_path, {"label_frames": "", "visible": ""}, {}) == (3, 3)   # added, not yet counted
    video = write_clip(workspace_dir / "videos" / "rally.avi", 3, SIZE)
    assert clip_counts("rally", csv_path, None, {"rally": video}) == (3, 2)
    assert clip_counts("rally", csv_path, None, {"rally": workspace_dir / "videos" / "missing.avi"}) == (3, 3)
    assert "counting rally_ball.csv as 1080p" in capsys.readouterr().out


def test_label_frame_size_falls_back_to_the_reference_frame(workspace_dir, capsys):
    csv_path = workspace_dir / "labels" / "rally_ball.csv"
    assert label_frame_size(None, csv_path) == (1920, 1080)
    assert label_frame_size(write_clip(workspace_dir / "videos" / "rally.avi", 2, SIZE), csv_path) == SIZE
    assert capsys.readouterr().out == ""
    assert label_frame_size(workspace_dir / "videos" / "missing.avi", csv_path) == (1920, 1080)
    assert "[warn]" in capsys.readouterr().out


def test_status_sections_list_own_clips_pending_videos_and_orphans(workspace_dir, capsys):
    finals = {"video10": workspace_dir / "labels" / "video10_ball.csv", "grid_match1": workspace_dir / "labels" / "g_ball.csv"}
    manifest = {"video10": {"source": "custom", "fps": "60", "width": "1920", "height": "1080", "label_frames": "40",
                            "visible": "30", "note": "baseline"},
                "grid_match1": {"source": "gridtracknet"}}
    print_own_camera_clips(finals, manifest)
    out = capsys.readouterr().out
    assert "own-camera clips" in out and "video10" in out and "1920x1080 @60" in out and "grid_match1" not in out
    print_own_camera_clips({"grid_match1": finals["grid_match1"]}, manifest)
    assert capsys.readouterr().out == ""                                       # nothing from an own camera: no section
    print_unfinished_clips({"video10": workspace_dir / "v.mp4", "video11": workspace_dir / "w.mp4"}, finals, {"video11"})
    out = capsys.readouterr().out
    assert "draft ready -> python finetune/ft.py label video11" in out
    assert "label files without a video: grid_match1" in out and "video10" not in out


def test_review_progress_reads_the_sidecar(workspace_dir, capsys):
    assert review_progress("rally", 4) == ""                                       # no review file yet
    review = workspace.review_json("rally")
    review.write_text('{"reviewed": [0, 2], "edited": [2]}', encoding="utf-8")
    assert review_progress("rally", 4) == "2/4 reviewed"
    assert review_progress("rally", 0) == "2 reviewed"
    review.write_text('{"reviewed": "x"}', encoding="utf-8")
    assert review_progress("rally", 4) == "review file unreadable"
    assert "[warn]" in capsys.readouterr().out


def test_sha8_and_shown(tmp_path):
    path = tmp_path / "w.npz"
    path.write_bytes(b"weights")
    assert len(sha8(path)) == 8 and sha8(path) == sha8(path)
    assert shown(ROOT / "models" / "x.npz") == str(ROOT.joinpath("models", "x.npz").relative_to(ROOT))
    assert shown(tmp_path / "elsewhere") == str(tmp_path / "elsewhere")


def test_paths_lists_the_interpreter_workspace_and_sources(workspace_dir, capsys):
    assert cmd_paths(None) is None                                           # a listing: no exit status
    out = capsys.readouterr().out
    assert f"workspace  {workspace.current().root}" in out and f"repo       {ROOT}" in out
    assert all(f"source {name:11s}" in out for name in workspace.DEFAULT_SOURCES)


def test_camera_motion_on_a_clip_with_nothing_to_correlate(workspace_dir):
    video = write_clip(workspace_dir / "videos" / "flat.avi", 40, SIZE)           # flat frames: no peak
    p75, peak, weak = measure_camera_motion(video)
    assert math.isnan(p75) and math.isnan(peak) and weak == 1.0
    assert commands.CAMERA_FIXED_PX < commands.CAMERA_MAX_PX
