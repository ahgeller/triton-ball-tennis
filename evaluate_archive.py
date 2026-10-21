"""Score ball detectors and the full pipeline against the hand-labelled clips (``evaluation.archive``).

    python evaluate_archive.py --archive finetune --mode raw --clips video13
"""
from evaluation.archive import main

if __name__ == "__main__":
    raise SystemExit(main())
