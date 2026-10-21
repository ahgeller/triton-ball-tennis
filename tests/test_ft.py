"""finetune/ft.py, the workflow front door: its import-time invariant, the pure clip-choice rule,
weight promotion on a temporary models folder, the check/status commands on a synthetic clip, and
the dispatch and menu plumbing with the script launchers replaced. The real models/ folder and
workspace are never touched (``workspace_dir``)."""
import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from finetune import ft, menu
from finetune.data import workspace
from finetune.data.labels import write_labels
from tennis_tracker.paths import ROOT
from tests._helpers import write_clip

SIZE = (64, 36)

BLOCKED_IMPORT = """
import importlib.abc, sys
BLOCKED = ("cv2", "numpy", "torch")
class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError(f"{name} is not available before ft.py hands over to the training interpreter")
sys.meta_path.insert(0, Blocker())
import finetune.ft
assert not any(name.split(".")[0] in BLOCKED for name in sys.modules), sorted(sys.modules)
for light in ("finetune.data.label_audit", "finetune.data.labels", "finetune.data.workspace", "finetune.data.data_policy",
              "finetune.data.detector_metrics", "finetune.data.fileio"):
    assert light in sys.modules, light
print("light")
"""


def test_ft_imports_without_cv2_numpy_or_torch():
    """ensure_interpreter can only hand over to an interpreter that has them if ft.py (and every
    module it imports at the top) loads without them."""
    done = subprocess.run([sys.executable, "-c", BLOCKED_IMPORT], cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "light"


# --------------------------------------------------------------------------- the clip choice

CHOICES = ["grid_match1", "grid_match23", "video23", "clipA"]


@pytest.mark.parametrize("answer, expected", [
    ("1", "grid_match1"),          # number 1, and the only clip ending in 1 is the same clip
    ("3", "video23"),              # number 3; no clip ends in a bare 3 (video23's 3 follows a digit)
    ("clipa", "clipA"),            # exact, case-insensitive
    ("match1", "grid_match1"),     # one substring match
    ("zzz", "zzz"),                # nothing matches: passed through for the tool to complain about
])
def test_resolve_choice_takes_numbers_names_and_unique_substrings(answer, expected):
    assert menu._resolve_choice(answer, CHOICES) == expected


def test_resolve_choice_refuses_a_number_that_is_also_a_clip_name(capsys):
    assert menu._resolve_choice("2", ["a", "b", "video2"]) is None
    out = capsys.readouterr().out
    assert "2 is two things here" in out and "number 2 in the list is b" in out and "video2" in out


def test_resolve_choice_refuses_an_ambiguous_substring(capsys):
    assert menu._resolve_choice("23", CHOICES) is None       # 23 is past the list, and two clips contain it
    assert "'23' matches 2: grid_match23, video23" in capsys.readouterr().out


# --------------------------------------------------------------------------- promote

def test_promote_backs_up_the_tracker_weights_and_copies_the_winner_atomically(workspace_dir, capsys):
    workspace.MODELS.mkdir()
    assert ft.run_cli(["promote"]) == 1                               # nothing trained yet
    assert "No weights at" in capsys.readouterr().out
    workspace.BEST_WEIGHTS.write_bytes(b"new weights")
    assert ft.run_cli(["promote"]) == 0                               # no tracker weights yet: nothing to back up
    assert workspace.REPO_WEIGHTS.read_bytes() == b"new weights"
    assert not list(workspace.MODELS.glob("*.before_promote_*"))
    assert ft.run_cli(["promote"]) == 0
    assert "already is" in capsys.readouterr().out                    # identical: a no-op
    assert not list(workspace.MODELS.glob("*.before_promote_*"))
    workspace.BEST_WEIGHTS.write_bytes(b"newer weights")
    assert ft.run_cli(["promote"]) == 0
    backup, = workspace.MODELS.glob("tracker.npz.before_promote_*")
    assert backup.read_bytes() == b"new weights" and workspace.REPO_WEIGHTS.read_bytes() == b"newer weights"
    assert not list(workspace.MODELS.glob("*.tmp"))
    other = workspace_dir / "other.npz"
    other.write_bytes(b"explicit")
    assert ft.run_cli(["promote", "--weights", str(other)]) == 0
    assert workspace.REPO_WEIGHTS.read_bytes() == b"explicit" and len(list(workspace.MODELS.glob("*.before_promote_*"))) == 2


# --------------------------------------------------------------------------- check and status

def test_check_reports_each_label_file_against_its_video(workspace_dir, capsys):
    write_clip(workspace_dir / "videos" / "good.avi", 6, SIZE)
    write_labels(workspace_dir / "labels" / "good_ball.csv", {f: (10.0, 10.0) for f in range(6)}, *SIZE)
    assert ft.run_cli(["check", "-v"]) == 0
    out = capsys.readouterr().out
    assert "[ ok ] good: 6 labels, 64x36 @30" in out and "1 label files checked, 0 with problems" in out
    write_labels(workspace_dir / "labels" / "good_ball.csv", {f: (10.0, 10.0) for f in range(9)}, *SIZE)
    write_labels(workspace_dir / "labels" / "orphan_ball.csv", {0: None}, *SIZE)
    assert ft.run_cli(["check"]) == 1
    out = capsys.readouterr().out
    assert "[fail] good: label frame 8 beyond the video's 6 frames" in out
    assert "[fail] orphan: no video" in out and "2 label files checked, 2 with problems" in out
    assert ft.run_cli(["check", "--clips", "good"]) == 1 and "1 label files checked" in capsys.readouterr().out


def test_status_on_an_empty_and_a_small_workspace(workspace_dir, capsys):
    assert ft.run_cli(["status"]) == 0
    assert "is empty" in capsys.readouterr().out
    write_clip(workspace_dir / "videos" / "rally.avi", 3, SIZE)
    write_labels(workspace_dir / "labels" / "rally_ball.csv", {0: (1.0, 2.0), 1: None}, *SIZE)
    (workspace_dir / "videos" / "pending.avi").write_bytes(b"")
    workspace.add_exclusion("rally", "test")
    assert ft.run_cli(["status"]) == 0
    out = capsys.readouterr().out
    assert re.search(r"custom\s+1\s+2\s+1\n", out)                       # counted from the file: 2 rows, 1 visible
    assert "videos without a final label file" in out and "pending" in out
    assert "excluded from training by exclude.txt (1): rally" in out


# --------------------------------------------------------------------------- dispatch

def test_commands_hand_their_arguments_to_the_right_script(workspace_dir):
    launched = []
    with patch.object(ft, "run_script", lambda script, arguments: launched.append((script, arguments)) or 0), \
            patch.object(ft, "run_module", lambda module, arguments: launched.append((module, arguments)) or 0):
        assert ft.run_cli(["train", "--epochs", "3", "--lr", "0.3"]) == 0
        assert ft.run_cli(["import", "all"]) == 0
        assert ft.run_cli(["hints"]) == 0
        assert ft.run_cli(["prelabel", "--clips", "x"]) == 0
        assert ft.run_cli(["eval", "--clips", "video13", "--weights", "w.npz"]) == 0
        assert ft.run_cli(["label", "video10", "--zoom", "3", "video11"]) == 0      # two clips: one run
    assert launched[:4] == [("training/train_gridtracknet.py", ["--epochs", "3", "--lr", "0.3"]),
                            ("data/import_data.py", ["all"]), ("labeling/cnn_refiner.py", ["list"]),
                            ("training/pretrack.py", ["--clips", "x"])]
    module, arguments = launched[4]
    assert module == "evaluation.archive"
    assert arguments[:6] == ["--mode", "raw", "--models", "gridtracknet", "--archive", str(workspace.WORKSPACE)]
    assert arguments[6:] == ["--gridtracknet-weights", str(Path("w.npz").resolve()), "--clips", "video13"]
    assert launched[5] == ("labeling/label_tool.py", ["video10", "video11", "--zoom", "3.0"])


def test_eval_scores_the_clips_the_policy_admits_and_says_when_none_do(workspace_dir, capsys):
    launched = []
    with patch.object(ft, "run_module", lambda module, arguments: launched.append(arguments) or 0):
        assert ft.run_cli(["eval"]) == 1                                  # nothing labelled: nothing to score
        assert "no clips qualify" in capsys.readouterr().out
        for clip in ("video13", "video14", "video2"):
            write_labels(workspace_dir / "labels" / f"{clip}_ball.csv", {0: (1.0, 2.0)}, *SIZE)
        (workspace_dir / "review_status.json").write_text('{"verified_video_min": 13}', encoding="utf-8")
        workspace.add_exclusion("video14", "audit")
        assert ft.run_cli(["eval"]) == 0
        assert ft.run_cli(["eval", "--all"]) == 0
    assert launched[0][-2:] == ["--clips", "video13"]                     # video2 unreviewed, video14 excluded
    assert launched[1][-1] == "--include-excluded" and "--clips" not in launched[1]


def test_label_pre_labels_a_clip_that_has_no_labels_yet(workspace_dir, capsys):
    (workspace_dir / "videos" / "fresh.mp4").write_bytes(b"")
    launched = []
    with patch.object(ft, "run_script", lambda script, arguments: launched.append((script, arguments)) or 0):
        assert ft.run_cli(["label", "fresh"]) == 0
        assert ft.run_cli(["label", "missing"]) == 1
    assert launched == [("training/pretrack.py", ["--clips", "fresh"]), ("labeling/label_tool.py", ["fresh"])]
    assert "No video named missing.*" in capsys.readouterr().out


def test_commands_that_do_not_forward_arguments_refuse_leftovers(workspace_dir, capsys):
    """check video5 (no --clips) and camera --forse used to run as if nothing had been typed."""
    for argv in (["check", "video5"], ["camera", "--forse"], ["status", "extra"], ["promote", "--weights", "w", "x"],
                 ["paths", "--verbose"]):
        with pytest.raises(SystemExit) as stop:
            ft.run_cli(argv)
        assert stop.value.code == 2 and "unrecognized arguments" in capsys.readouterr().err
    launched = []
    with patch.object(ft, "run_script", lambda script, arguments: launched.append((script, arguments)) or 0):
        assert ft.run_cli(["train", "--anything", "goes"]) == 0            # a forwarding command passes it on
    assert launched == [("training/train_gridtracknet.py", ["--anything", "goes"])]


def test_the_menu_adds_a_video_through_add_videos_and_offers_the_label_tool(workspace_dir):
    source = write_clip(workspace_dir / "my rally!.avi", 3, SIZE)
    answers = iter([str(source), ""])
    ran = []
    with patch.object(ft, "ask", lambda prompt: next(answers)), \
            patch.object(ft, "run_script", lambda script, arguments: ran.append((script, arguments)) or 0), \
            patch.object(ft, "run_cli", lambda argv: ran.append(("cli", argv)) or 0):
        menu._menu_add()
    assert ft.clip_name(source) == "my_rally_"
    assert (workspace_dir / "videos" / "my_rally_.avi").is_file()
    assert ran == [("training/pretrack.py", ["--clips", "my_rally_"]), ("cli", ["label", "my_rally_"])]


def test_add_videos_names_copies_and_reports_what_it_could_not_add(workspace_dir, capsys):
    good = write_clip(workspace_dir / "good.avi", 3, SIZE)
    (workspace_dir / "bad.avi").write_bytes(b"not a video")
    with patch.object(ft, "run_script", lambda script, arguments: 7):
        assert ft.add_videos([good, workspace_dir / "bad.avi", workspace_dir / "missing.avi"], name="ignored") == (["good"], 7)
        assert ft.add_videos([good], name="renamed", prelabel=False) == (["renamed"], 0)
        assert ft.add_videos([good], name="renamed", prelabel=False) == (["renamed"], 0)   # kept, still in the workspace
    out = capsys.readouterr().out
    assert "[skip] not a file" in out and "[skip] Could not open" in out and "[keep] renamed.avi already exists" in out
    assert sorted(p.name for p in (workspace_dir / "videos").iterdir()) == ["good.avi", "renamed.avi"]


def test_scripts_run_from_the_package_folder_and_modules_from_the_root():
    ran = []
    with patch.object(ft, "wait_child", lambda command, **kwargs: ran.append((command, kwargs)) or 0):
        ft.run_script("training/pretrack.py", ["--clips", "a"])
        ft.run_module("evaluation.archive", ["--help"])
    assert ran[0][0] == [sys.executable, str(ft.SCRIPTS / "training" / "pretrack.py"), "--clips", "a"]
    assert ran[1][0] == [sys.executable, "-m", "evaluation.archive", "--help"]
    assert all(kwargs == {"cwd": str(ft.ROOT)} for _, kwargs in ran)
    assert ft.SCRIPTS == ROOT / "finetune"


def test_menu_dispatches_a_number_or_a_name_and_quits(workspace_dir, capsys):
    answers = iter(["1", "", "status", "q", "nope", "", None])
    ran = []
    with patch.object(ft, "ask", lambda prompt: next(answers)), \
            patch.object(ft, "run_cli", lambda argv: ran.append(argv) or 0):
        assert menu.interactive_menu() is None                # 1 -> status, Enter, 'status' -> status, q
        assert menu.interactive_menu() is None                # nope -> not an option, Enter -> ..., EOF quits
    assert ran == [["status"], ["status"]]
    out = capsys.readouterr().out
    assert "'nope' is not an option" in out and "the workspace is empty" in out
