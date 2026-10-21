"""finetune/interpreter.py: finding a python with cv2, numpy and torch and handing ft.py over to
it, with the probes and the hand-over replaced so nothing is really launched."""
import os
import sys
from pathlib import Path
from unittest.mock import patch

from finetune import interpreter
from finetune.interpreter import (QT_NEEDED_EXIT, candidate_interpreters, conda_pythons, ensure_interpreter,
                                  probe_interpreter, python_names, remember_interpreter, training_python, venv_pythons,
                                  wait_child)


def test_candidates_end_with_this_interpreter_and_the_path():
    found = candidate_interpreters()
    assert Path(sys.executable) in found
    assert len(found) == len(set(found))
    assert all(name.endswith(".exe") for name in python_names()) == (os.name == "nt")


def test_venv_and_conda_layouts_are_searched_in_a_fixed_order(tmp_path):
    venvs = list(venv_pythons())
    assert venvs[0].parts[-3:-1] == (".venv", "Scripts") and all(p.name in python_names() for p in venvs)
    assert len(venvs) == 2 * 2 * 2 * len(python_names())            # (repo, parent) x (.venv, venv) x (Scripts, bin)
    assert list(conda_pythons(tmp_path)) == []                       # no conda roots: nothing to try
    (tmp_path / ".conda" / "envs" / "b").mkdir(parents=True)
    (tmp_path / "miniconda3" / "envs" / "a").mkdir(parents=True)
    found = list(conda_pythons(tmp_path))
    assert found[0].parent == tmp_path / ".conda" / "envs" / "b" and found[-1].parent == tmp_path / "miniconda3" / "envs" / "a" / "bin"


def test_remember_interpreter_reports_a_cache_it_could_not_write(tmp_path, capsys):
    with patch.object(interpreter, "PYTHON_CACHE", tmp_path / ".training-python"):
        assert remember_interpreter(tmp_path / "python") is True
        assert (tmp_path / ".training-python").read_text(encoding="utf-8") == str(tmp_path / "python")
    with patch.object(interpreter, "PYTHON_CACHE", tmp_path / "no-such-dir" / "x" / ".training-python"), \
            patch.object(interpreter, "write_atomic", side_effect=OSError("read-only")):
        assert remember_interpreter(tmp_path / "python") is False
    assert "could not remember" in capsys.readouterr().out


def test_probe_answers_none_for_a_python_that_is_not_there(tmp_path, capsys):
    assert probe_interpreter(tmp_path / "nope" / "python") is None
    assert "[python] trying" in capsys.readouterr().out
    with patch.object(interpreter.subprocess, "run") as run:
        run.return_value.returncode = QT_NEEDED_EXIT
        assert probe_interpreter(Path(sys.executable)) == QT_NEEDED_EXIT
        run.return_value.returncode = 1
        assert probe_interpreter(Path(sys.executable)) is None


def test_training_python_prefers_a_qt_build_and_remembers_it(tmp_path):
    cache = tmp_path / ".training-python"
    answers = {tmp_path / "a": None, tmp_path / "b": QT_NEEDED_EXIT, tmp_path / "c": 0}
    with patch.object(interpreter, "PYTHON_CACHE", cache), \
            patch.object(interpreter, "candidate_interpreters", return_value=list(answers)), \
            patch.object(interpreter, "probe_interpreter", side_effect=answers.get):
        assert training_python() == tmp_path / "c"
        assert cache.read_text(encoding="utf-8") == str(tmp_path / "c")
    with patch.object(interpreter, "PYTHON_CACHE", cache), \
            patch.object(interpreter, "candidate_interpreters", return_value=[tmp_path / "a", tmp_path / "b"]), \
            patch.object(interpreter, "probe_interpreter", side_effect=answers.get):
        assert training_python() == tmp_path / "b"                                 # usable, if not Qt
    with patch.object(interpreter, "PYTHON_CACHE", cache), \
            patch.object(interpreter, "candidate_interpreters", return_value=[tmp_path / "a"]), \
            patch.object(interpreter, "probe_interpreter", return_value=None):
        assert training_python() is None and not cache.exists()                    # the cached one is forgotten


def test_ensure_interpreter_stays_put_when_told_or_when_complete(tmp_path):
    with patch.dict(os.environ, {"TENNIS_FINETUNE_NO_REEXEC": "1"}):
        assert ensure_interpreter(tmp_path / "ft.py") is None
    with patch.dict(os.environ, {}, clear=False), patch.object(interpreter, "current_is_complete", return_value=True):
        os.environ.pop("TENNIS_FINETUNE_NO_REEXEC", None)
        assert ensure_interpreter(tmp_path / "ft.py") is None


def test_ensure_interpreter_hands_over_with_the_same_arguments(tmp_path, capsys):
    launched = []
    with patch.dict(os.environ, {}, clear=False), patch.object(interpreter, "current_is_complete", return_value=False), \
            patch.object(interpreter, "training_python", return_value=tmp_path / "venv" / "python"), \
            patch.object(interpreter, "wait_child", lambda command, **kwargs: launched.append((command, kwargs)) or 7), \
            patch.object(sys, "argv", ["ft.py", "status", "--x"]):
        os.environ.pop("TENNIS_FINETUNE_NO_REEXEC", None)
        assert ensure_interpreter(tmp_path / "ft.py") == 7
    command, kwargs = launched[0]
    assert command == [str(tmp_path / "venv" / "python"), str(tmp_path / "ft.py"), "status", "--x"]
    assert kwargs["env"]["TENNIS_FINETUNE_NO_REEXEC"] == "1"
    with patch.dict(os.environ, {}, clear=False), patch.object(interpreter, "current_is_complete", return_value=False), \
            patch.object(interpreter, "training_python", return_value=None):
        os.environ.pop("TENNIS_FINETUNE_NO_REEXEC", None)
        assert ensure_interpreter(tmp_path / "ft.py") == 2
    assert "No interpreter with cv2, numpy and torch" in capsys.readouterr().err


def test_wait_child_returns_the_exit_code():
    assert wait_child([sys.executable, "-c", "raise SystemExit(5)"]) == 5
