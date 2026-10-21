"""Where the model assets live, relative to the repository root.

The tracker's defaults, the evaluation scripts and the fine-tuning workspace all point at
the same files; this is the one place their paths are spelled. Everything is absolute, so a
script started from another working directory still finds them.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
GRIDTRACKNET_WEIGHTS = MODELS / "gridtracknet_weights_torch.npz"   # the ball detector the tracker runs
PLAYER_ENGINE = MODELS / "player.engine"                           # TensorRT player detector
COURT_ENGINE = MODELS / "courtdetection.engine"                    # TensorRT court keypoint detector
BOUNCE_MODEL = MODELS / "ctb_regr_bounce.cbm"                      # CatBoost bounce model (--court)
