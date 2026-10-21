"""The detector metric: how a set of (label, prediction) pairs is scored, and the thresholds at
which a label and a raw detection are said to agree.

Distances are measured in 1080p pixels (a 1920-wide frame as it is; a smaller frame scaled up),
so clips of different sizes pool. ``score`` is used by evaluate_archive.py and by
train_gridtracknet.py to pick the better of the trained and the starting weights; the agreement
thresholds by ``finetune.alignment`` (the label audit and realign.py).
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

REFERENCE_WIDTH = 1920.0    # the frame size pixel distances are quoted at ...
REFERENCE_HEIGHT = 1080.0   # ... (1080p); a 720p clip's pixel counts 1.5 times
HIT_PX = 10.0               # a prediction counts as correct within this many pixels
WRONG_PX = 30.0             # beyond this it is tracking something else

AGREE_PX = 15.0             # a label and a raw detection agree within this many 1080p pixels
FAST_PX = 6.0               # a label that moved this far (1080p px) since the previous row can tell one frame from the next
MIN_FAST = 20               # fewer fast labels than this and every label is judged instead
DETECTION_CONF = 0.5        # detections at or above this confidence count as the detector's answer

Point = Optional[Tuple[float, float]]


def scale_for_width(width: int) -> float:
    """Multiplier that takes this frame's pixels to 1080p pixels."""
    return REFERENCE_WIDTH / width


def scale_for_height(height: int) -> float:
    return REFERENCE_HEIGHT / height


def score(pairs: Sequence[Tuple[Point, Point]], scale: float) -> Dict[str, float]:
    """pairs: [(label or None, prediction or None)], distances multiplied by `scale` to 1080p."""
    hit = wrong = miss = false_alarm = quiet = 0
    errors = []
    for label, prediction in pairs:
        if label is None:
            if prediction is None:
                quiet += 1
            else:
                false_alarm += 1
            continue
        if prediction is None:
            miss += 1
            continue
        error = math.hypot(prediction[0] - label[0], prediction[1] - label[1]) * scale
        errors.append(error)
        if error <= HIT_PX:
            hit += 1
        elif error > WRONG_PX:
            wrong += 1
    visible = hit + wrong + miss + sum(1 for e in errors if HIT_PX < e <= WRONG_PX)
    errors.sort()
    return {
        "visible": visible,
        "recall": hit / max(visible, 1),
        "wrong_obj": wrong / max(visible, 1),
        "miss": miss / max(visible, 1),
        "invisible": false_alarm + quiet,
        "false_alarm": false_alarm / max(false_alarm + quiet, 1),
        "median_px": errors[len(errors) // 2] if errors else float("nan"),
        "p90_px": errors[int(0.9 * len(errors))] if errors else float("nan"),
    }
