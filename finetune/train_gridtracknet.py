"""``python finetune/train_gridtracknet.py ...`` still runs the trainer, which lives in
``finetune/training/train_gridtracknet.py`` (``python -m finetune.training.train_gridtracknet``
is the same program). This file only hands over to it; it is not a module to import from."""
import runpy
import sys
from pathlib import Path

if __name__ != "__main__":
    raise ImportError("the trainer is finetune.training.train_gridtracknet; this file only runs it as a script")


def main() -> None:
    """Make the repository root importable, then run the trainer as ``__main__``."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    runpy.run_module("finetune.training.train_gridtracknet", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
