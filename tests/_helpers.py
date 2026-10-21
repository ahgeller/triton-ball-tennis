"""Synthetic fixtures shared by the regression tests. No match media or labels."""
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from unittest.mock import patch

import numpy as np
import torch

from finetune import train_gridtracknet as training

# A detector output the metric probe returns for every five-frame unit.
_PROBE_POINT = (100, 100)
_PROBE_CONFIDENCE = 0.9
_PROBE_FPS = 30
_PROBE_SIZE = (1920, 1080)


class Capture:
    """A ``cv2.VideoCapture`` stand-in whose frame ``i`` is filled with the value ``i``."""

    def __init__(self, count: int):
        self.count, self.index = count, 0

    def isOpened(self) -> bool:  # noqa: N802 - mirrors cv2.VideoCapture
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.index >= self.count:
            return False, None
        frame = np.full((2, 2, 3), self.index, np.uint8)
        self.index += 1
        return True, frame

    def release(self) -> None:
        """Nothing to release; present because callers always release captures."""


def metric_probe(count: int, labels: Dict[int, Optional[List[float]]]) -> Dict[str, float]:
    """``detector_metrics`` on a ``count``-frame synthetic clip whose detector always fires at (100, 100)."""
    width, height = _PROBE_SIZE
    with patch.object(training, "load_model", return_value=lambda _: None), \
         patch.object(training, "video_meta", return_value=(_PROBE_FPS, width, height, count)), \
         patch.object(training, "read_labels", return_value=labels), \
         patch.object(training.cv2, "VideoCapture", side_effect=lambda _: Capture(count)), \
         patch.object(training, "frame_tensor", return_value=torch.zeros(3, 1, 1)), \
         patch.object(training, "decode_predictions",
                      return_value=[(_PROBE_POINT, _PROBE_CONFIDENCE)] * 5):
        return training.detector_metrics(
            Path("unused"), [("probe", Path("unused"), Path("unused"), "custom")], torch.device("cpu")
        )
