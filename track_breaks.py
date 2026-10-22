"""Where a pipeline run loses the ball, from its tracking JSONs and the clips' labels (``evaluation.track_breaks``).

    python track_breaks.py output/fix1_no_court/pipeline --compare output/baseline/track_breaks.log
"""
from evaluation.track_breaks import main

if __name__ == "__main__":
    raise SystemExit(main())
