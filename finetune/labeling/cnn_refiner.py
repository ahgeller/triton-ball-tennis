"""The old path of the suggestion network's command line, kept only so ``ft.py hints`` (which
runs ``labeling/cnn_refiner.py`` by path) keeps working until it is pointed at
``finetune/refiner/hints.py``; delete this file with that edit. The network itself is
``finetune.refiner.cnn_refiner``."""
import sys
from pathlib import Path

if __name__ == "__main__":          # run as a script: make the repository root importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from finetune.refiner.hints import main
    raise SystemExit(main())
