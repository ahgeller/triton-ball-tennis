"""The detector metric a training run is judged by: the raw GridTrackNet prepass over whole
clips scored as ``evaluate_archive.py`` scores it (recall / wrong-object / false-alarm), and the
rule that says when a candidate beats the incumbent."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import torch

from tennis_tracker.gridtracknet import FRAMES_PER_UNIT, decode_predictions, frame_tensor, load_model
from tennis_tracker.frame_rate import require_phase_stride
from finetune.data.scoring import REFERENCE_HEIGHT, scale_for_width, score
from finetune.data.workspace import video_meta
from finetune.training.clips import Clip
from finetune.training.frame_cache import training_targets

METRIC_SLACK = 0.005       # a candidate may be worse by this on wrong/false-alarm rate: 1 frame in 200
WRONG_PENALTY = 2.0        # ... and each unit of extra wrong-object rate costs this much recall


def detector_metrics(weights: Path, clips: List[Clip], device: torch.device, threshold: float = 0.5) -> Dict[str, float]:
    """Raw-detector recall / wrong-object / false-alarm on whole clips (the evaluate_archive metric)."""
    model = load_model(weights, device)
    pairs = []
    with torch.inference_mode():
        for clip, video, csv_path, _ in clips:
            fps, width, height, total = video_meta(video)
            labels = training_targets(csv_path, width, height)
            scale = scale_for_width(width)
            predictions = predict_clip(model, video, require_phase_stride(fps), width, height, device, threshold)
            for frame, label in labels.items():
                point = predictions.get(frame)
                pairs.append((None if label is None else (label[0] * scale, label[1] * scale),
                              None if point is None else (point[0] * scale, point[1] * scale)))
    stats = score(pairs, 1.0)
    stats["wrong"] = stats.pop("wrong_obj")
    return stats


def predict_clip(model, video: Path, stride: int, width: int, height: int, device: torch.device,
                 threshold: float) -> Dict[int, Optional[Tuple[float, float]]]:
    """frame -> the model's point (None for a miss) over a whole clip, read in five-frame units per
    phase the way the tracker's prepass does, the tail of each phase inferred from its last five."""
    from tennis_tracker.config import Config
    capture = cv2.VideoCapture(str(video))
    units = [[] for _ in range(stride)]
    recent = [deque(maxlen=FRAMES_PER_UNIT) for _ in range(stride)]
    predictions: Dict[int, Optional[Tuple[float, float]]] = {}
    y_offset = Config().gridtracknet_y_offset_px * height / REFERENCE_HEIGHT

    def infer(unit):
        output = model(torch.cat([t for t, _ in unit]).unsqueeze(0))
        for (_, frame_index), (point, _) in zip(unit, decode_predictions(output, width, height, threshold)):
            # A finalized miss is immutable, just like a detection.
            if frame_index not in predictions:
                predictions[frame_index] = None if point is None else (point[0], point[1] + y_offset)
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            phase = index % stride
            unit = units[phase]
            item = (frame_tensor(frame, device), index)
            unit.append(item)
            recent[phase].append(item)
            if len(unit) == FRAMES_PER_UNIT:
                infer(unit)
                unit.clear()
            index += 1
        for phase in range(stride):
            if units[phase] and len(recent[phase]) == FRAMES_PER_UNIT:
                infer(recent[phase])
    finally:
        capture.release()
    return predictions


def better(candidate: Dict[str, float], incumbent: Dict[str, float]) -> bool:
    """Best = more correct ball with no more wrong-object outputs (within METRIC_SLACK, 1 frame in 200)."""
    if candidate["wrong"] > incumbent["wrong"] + METRIC_SLACK:
        return False
    if candidate.get("false_alarm", 0.0) > incumbent.get("false_alarm", 0.0) + METRIC_SLACK:
        return False
    return candidate["recall"] - WRONG_PENALTY * max(0.0, candidate["wrong"] - incumbent["wrong"]) > incumbent["recall"]
