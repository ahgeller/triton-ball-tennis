"""Finding an interpreter that has cv2 (the Qt build), numpy and torch, and handing ``ft.py``
over to it: plain ``python finetune/ft.py`` then works from any shell. Only the standard library
is imported here, so the hand-over itself never needs those packages."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from finetune.data.fileio import write_atomic
from finetune.data.workspace import ROOT

# cheapest first: cv2 rules out most interpreters fast. Exit 3 means usable for training but with an
# OpenCV whose windows are not Qt's, which the label tool refuses (see label_tool.highgui_backend).
TRAINING_IMPORTS = ("import sys, cv2, numpy, torch; sys.exit(0 if any(line.strip().startswith('GUI:') "
                    "and 'QT' in line.upper() for line in cv2.getBuildInformation().splitlines()) else 3)")
PYTHON_CACHE = Path(__file__).resolve().parent / ".training-python"   # the interpreter that worked last time
PROBE_TIMEOUT_S = 120              # an interpreter that takes longer than this to import torch is not usable
QT_NEEDED_EXIT = 3                 # probe_interpreter's "usable, but its cv2 is not the Qt build"


def python_names() -> Tuple[str, ...]:
    return ("python.exe",) if os.name == "nt" else ("python3", "python")


def candidate_interpreters() -> List[Path]:
    """Places a torch + cv2 interpreter tends to live, best guess first."""
    found: List[Path] = []

    def add(path) -> None:
        if not path:
            return
        candidate = Path(path)
        if candidate.is_file() and candidate not in found:
            found.append(candidate)

    add(os.environ.get("TENNIS_FINETUNE_PYTHON"))
    if PYTHON_CACHE.is_file():                       # whatever worked last time
        add(PYTHON_CACHE.read_text(encoding="utf-8").strip())
    for root in (ROOT, ROOT.parent):                 # a venv in or beside the repo
        for venv in (".venv", "venv"):
            for name in python_names():
                add(root / venv / "Scripts" / name)
                add(root / venv / "bin" / name)
    home = Path.home()                               # conda / miniconda / anaconda / miniforge environments
    conda_roots = ("anaconda3", "miniconda3", "miniforge3", "mambaforge")
    for base in (home / ".conda" / "envs", *(home / root / "envs" for root in conda_roots)):
        if base.is_dir():
            for env in sorted(base.iterdir()):
                for name in python_names():
                    add(env / name)
                    add(env / "bin" / name)
    add(sys.executable)
    for name in python_names():                      # finally whatever is on PATH
        add(shutil.which(name))
    return found


def probe_interpreter(python: Path) -> Optional[int]:
    """0: has cv2 (Qt build), numpy and torch. 3: the same with a non-Qt cv2. None: not usable."""
    print(f"[python] trying {python}", flush=True)
    try:
        done = subprocess.run([str(python), "-c", TRAINING_IMPORTS], capture_output=True, timeout=PROBE_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.returncode if done.returncode in (0, QT_NEEDED_EXIT) else None


def training_python() -> Optional[Path]:
    """First interpreter that can import cv2, numpy and torch, preferring one whose cv2 is the Qt
    build the label tool needs; the winner is cached for next time."""
    fallback = None
    for candidate in candidate_interpreters():
        result = probe_interpreter(candidate)
        if result == 0:
            fallback = candidate
            break
        if result == QT_NEEDED_EXIT and fallback is None:
            fallback = candidate
    if fallback is None:
        PYTHON_CACHE.unlink(missing_ok=True)         # the cached one stopped working
        return None
    try:
        write_atomic(PYTHON_CACHE, str(fallback))
    except OSError as error:
        print(f"[python] could not remember {fallback} in {PYTHON_CACHE.name} ({error}); it will be probed again next time")
    return fallback


def current_is_complete() -> bool:
    """Does this very interpreter have cv2 (the Qt build), numpy and torch? Torch is only looked up,
    not imported: importing it costs seconds, and the menu itself never needs it."""
    try:
        import importlib.util
        import cv2
        import numpy
    except ImportError:
        return False
    qt = any(line.strip().startswith("GUI:") and "QT" in line.upper()
             for line in cv2.getBuildInformation().splitlines())
    return qt and bool(numpy.__version__) and importlib.util.find_spec("torch") is not None


def wait_child(command: List[str], **kwargs) -> int:
    """subprocess.call, minus its habit of killing the child 0.25 s after a Ctrl+C. The child got
    the same Ctrl+C and may be saving labels; let it finish and pass its exit code on."""
    process = subprocess.Popen(command, **kwargs)
    interrupted = False
    while True:
        try:
            return process.wait()
        except KeyboardInterrupt:
            if not interrupted:
                interrupted = True
                print("[wait] interrupted; letting the child finish saving", flush=True)


def ensure_interpreter(script: Path) -> Optional[int]:
    """Re-run `script` with this process's arguments under the venv that has torch + cv2 when the
    current python lacks them. Returns the exit code to finish with when the work was handed to
    that interpreter (or none was found), None when this interpreter should carry on."""
    if os.environ.get("TENNIS_FINETUNE_NO_REEXEC") or current_is_complete():
        return None
    python = training_python()
    if python is not None and python.resolve() == Path(sys.executable).resolve():
        return None                                  # the best there is; the label tool says if it is not Qt
    if python is None:
        print("No interpreter with cv2, numpy and torch was found; install them or point "
              "TENNIS_FINETUNE_PYTHON at a python that has them", file=sys.stderr)
        return 2
    env = dict(os.environ, TENNIS_FINETUNE_NO_REEXEC="1")
    return wait_child([str(python), str(script), *sys.argv[1:]], env=env)
