"""The evaluation package and the root scripts that front it: the documented commands and the
imports other scripts make keep working, and the package's two scorers are the ones they name."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_root_scripts_front_the_package_modules():
    import check_parity
    import evaluate_archive
    import validate_tracking
    from evaluation import archive, parity, tracking

    assert validate_tracking.validate is tracking.validate and validate_tracking.main is tracking.main
    assert check_parity.check is parity.check and check_parity.BASELINE is parity.BASELINE
    assert evaluate_archive.main is archive.main
    assert parity.validate is tracking.validate              # check_parity gates on the tracking scorer


def test_the_documented_commands_still_answer_help():
    for script in ("evaluate_archive.py", "validate_tracking.py", "check_parity.py"):
        done = subprocess.run([sys.executable, str(ROOT / script), "--help"], cwd=ROOT, capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
        assert "usage:" in done.stdout
    done = subprocess.run([sys.executable, "-m", "evaluation.archive", "--help"], cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0 and "--archive" in done.stdout


def test_archive_scorer_uses_the_workspace_and_the_shared_model_paths():
    from evaluation import archive
    from finetune import workspace
    from tennis_tracker import paths

    assert archive.default_archive() == workspace.WORKSPACE or archive.ARCHIVE_VARIABLE in __import__("os").environ
    assert archive.MODELS["gridtracknet"] == paths.GRIDTRACKNET_WEIGHTS
