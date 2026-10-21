"""finetune.data.workspace: the workspace layout, its two records (clips.csv, exclude.txt) and the
TENNIS_WORKSPACE override, on folders under tmp_path. No real workspace file is read."""
import os
import subprocess
import sys

import pytest

from finetune.data import workspace
from tennis_tracker.paths import ROOT
from finetune.data.workspace import (MANIFEST_COLUMNS, add_exclusion, clip_of, draft_csv, draft_files, find_video,
                                     label_csv, label_files, labelled_clips, natural_key, read_exclusions, read_manifest,
                                     require_video, review_json, suspect_json, unsaved_csv, videos_in, write_manifest)



def test_a_clip_names_its_files(tmp_path):
    assert label_csv("rally7", tmp_path) == tmp_path / "rally7_ball.csv"
    assert draft_csv("rally7", tmp_path) == tmp_path / "rally7_ball.csv.draft"
    assert review_json("rally7", tmp_path) == tmp_path / "rally7_ball.review.json"
    assert suspect_json("rally7", tmp_path) == tmp_path / "rally7_ball.suspect.json"
    assert unsaved_csv("rally7", tmp_path) == tmp_path / "rally7_ball.unsaved.csv"
    assert clip_of(label_csv("rally7", tmp_path)) == clip_of(draft_csv("rally7", tmp_path)) == "rally7"


def test_label_files_come_in_natural_clip_order(tmp_path):
    for name in ("video10_ball.csv", "video2_ball.csv", "video1_ball.csv.draft", "notes.txt"):
        (tmp_path / name).write_text("")
    assert [p.name for p in label_files(tmp_path)] == ["video2_ball.csv", "video10_ball.csv"]
    assert labelled_clips(tmp_path) == ["video2", "video10"]
    assert [p.name for p in draft_files(tmp_path)] == ["video1_ball.csv.draft"]
    assert label_files(tmp_path / "missing") == [] and draft_files(tmp_path / "missing") == []
    assert natural_key("video10") > natural_key("video2")


def test_videos_are_found_whatever_their_container(tmp_path):
    for name in ("a.mov", "b.MP4", "b.txt", "c.avi"):
        (tmp_path / name).write_bytes(b"")
    assert find_video("a", tmp_path) == tmp_path / "a.mov"
    assert find_video("b", tmp_path) == tmp_path / "b.MP4"
    assert find_video("zzz", tmp_path) is None
    assert videos_in(tmp_path) == {"a": tmp_path / "a.mov", "b": tmp_path / "b.MP4", "c": tmp_path / "c.avi"}
    assert videos_in(tmp_path / "missing") == {}
    assert require_video("c", tmp_path) == tmp_path / "c.avi"
    with pytest.raises(FileNotFoundError, match="zzz"):
        require_video("zzz", tmp_path)


def test_manifest_round_trip_fills_every_column_and_sorts_naturally(tmp_path):
    path = tmp_path / "clips.csv"
    assert read_manifest(path) == {}
    write_manifest({"video10": {"clip": "video10", "source": "prof"},
                    "video2": {"clip": "video2", "source": "custom", "note": "n"}}, path)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == ",".join(MANIFEST_COLUMNS)
    assert [line.split(",")[0] for line in lines[1:]] == ["video2", "video10"]
    rows = read_manifest(path)
    assert set(rows) == {"video2", "video10"}
    assert set(rows["video2"]) == set(MANIFEST_COLUMNS)
    assert rows["video2"]["note"] == "n" and rows["video2"]["camera"] == "" and rows["video10"]["source"] == "prof"
    assert not list(tmp_path.glob("*.tmp"))


def test_exclusions_are_listed_once_with_a_header_and_comments_ignored(tmp_path):
    assert read_exclusions(tmp_path) == []
    add_exclusion("clipA", "first reason", tmp_path)
    add_exclusion("clipA", "second reason", tmp_path)
    add_exclusion("clipB", "other", tmp_path)
    lines = (tmp_path / "exclude.txt").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3 and lines[0].startswith("#") and "first reason" in lines[1]
    assert read_exclusions(tmp_path) == ["clipA", "clipB"]
    (tmp_path / "exclude.txt").write_text("# header\nvideo7  # audit\n\ngrid_match9\n", encoding="utf-8")
    assert read_exclusions(tmp_path) == ["video7", "grid_match9"]


def test_the_default_paths_follow_the_patched_workspace(workspace_dir):
    """The module-level defaults are read when called, so one patch moves every path."""
    assert label_csv("x") == workspace_dir / "labels" / "x_ball.csv"
    assert find_video("x") is None
    assert read_manifest() == {} and read_exclusions() == []
    add_exclusion("x", "why")
    assert (workspace_dir / "exclude.txt").is_file() and read_exclusions() == ["x"]
    write_manifest({"x": {"clip": "x"}})
    assert read_manifest()["x"]["clip"] == "x" and workspace.MANIFEST.parent == workspace_dir


def test_tennis_workspace_relocates_the_workspace_without_touching_the_code_paths(tmp_path):
    env = dict(os.environ, TENNIS_WORKSPACE=str(tmp_path))
    code = ("from finetune.data import workspace as w; "
            "print(w.WORKSPACE); print(w.LABELS); print(w.MANIFEST); print(w.ROOT); print(w.REPO_WEIGHTS)")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    lines = out.stdout.splitlines()
    assert lines[:3] == [str(tmp_path.resolve()), str(tmp_path.resolve() / "labels"), str(tmp_path.resolve() / "clips.csv")]
    assert lines[3] == str(ROOT) and lines[4] == str(ROOT / "models" / "gridtracknet_weights_torch.npz")
    env.pop("TENNIS_WORKSPACE")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    assert out.stdout.splitlines()[0] == str(ROOT / "finetune")


def test_tennis_workspace_reaches_the_label_tool_and_pretrack(tmp_path):
    """Both scripts read the workspace's folders when they run, not when they are imported, so
    the relocated workspace is where the label tool looks and where pretrack drafts."""
    env = dict(os.environ, TENNIS_WORKSPACE=str(tmp_path))
    weights = tmp_path / "w.npz"
    weights.write_bytes(b"")
    code = ("from finetune.labeling import label_cli; from finetune.training import pretrack; "
            "args = label_cli.build_parser().parse_args(['clip']); print(args.videos); print(args.labels); "
            f"raise SystemExit(pretrack.main(['--clips', 'clip', '--weights', {str(weights)!r}]))")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True)
    assert out.returncode == 1, out.stderr                       # no video named clip there: nothing drafted
    videos, labels = tmp_path.resolve() / "videos", tmp_path.resolve() / "labels"
    assert out.stdout.splitlines() == [str(videos), str(labels), f"No videos to track in {videos}"]
    assert videos.is_dir() and labels.is_dir()                   # made under the relocated workspace


def test_the_layout_is_one_object_and_every_module_name_reads_it_when_asked(tmp_path, monkeypatch):
    """No path is stored at import: ``ws.LABELS`` and the rest answer from ``current()`` on every
    access, so replacing ``current`` (what ``workspace_dir`` does) moves all of them, the tracker's
    weights included, and the environment is read as it is now."""
    layout = workspace.Workspace(tmp_path, repo_weights=tmp_path / "tracker.npz")
    assert layout.videos == tmp_path / "videos" and layout.labels == tmp_path / "labels"
    assert layout.manifest == tmp_path / workspace.MANIFEST_NAME and layout.exclude_file == tmp_path / workspace.EXCLUDE_NAME
    assert layout.review_status == tmp_path / workspace.REVIEW_STATUS_NAME
    assert layout.best_weights == tmp_path / "models" / workspace.BEST_WEIGHTS_NAME
    assert set(workspace._LAYOUT_NAMES) == {"WORKSPACE", "VIDEOS", "LABELS", "MODELS", "RUNS", "CACHE", "MANIFEST",
                                            "EXCLUDE_FILE", "REVIEW_STATUS", "BEST_WEIGHTS", "REPO_WEIGHTS"}
    assert not any(name in vars(workspace) for name in workspace._LAYOUT_NAMES)
    monkeypatch.setattr(workspace, "current", lambda: layout)
    assert workspace.WORKSPACE == tmp_path and workspace.LABELS == layout.labels
    assert workspace.REPO_WEIGHTS == layout.repo_weights and workspace.BEST_WEIGHTS == layout.best_weights
    with pytest.raises(AttributeError):
        workspace.NOT_A_PATH
    monkeypatch.undo()
    monkeypatch.setenv(workspace.WORKSPACE_VARIABLE, str(tmp_path / "elsewhere"))
    assert workspace.current().root == (tmp_path / "elsewhere").resolve()
    assert workspace.current().repo_weights == ROOT / "models" / "gridtracknet_weights_torch.npz"
    monkeypatch.delenv(workspace.WORKSPACE_VARIABLE)
    assert workspace.current().root == ROOT / "finetune"

