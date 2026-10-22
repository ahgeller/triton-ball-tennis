"""The detector metric a training run is judged by: the raw GridTrackNet prepass over whole
clips scored as ``evaluate_archive.py`` scores it (``DetectorMetrics``: recall / wrong-object /
false-alarm), and the rule that says when a candidate beats the incumbent.

The prepass is the tracker's own (``GridTrackNetBallDetector.detect_clip`` through
``finetune.data.detect``, the path ``evaluate_archive.py`` takes too), so the winner is judged on
exactly what the tracker will do: the same five-frame units per phase, the same trailing window,
the same first-answer-wins rule and y offset. There is no second copy of that protocol here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Set

import torch

from finetune.data.detect import ball_detector, detect_points
from finetune.data.detector_metrics import DetectorMetrics, scale_for_width
from finetune.data.workspace import video_meta
from finetune.training.clips import Clip
from finetune.training.frame_cache import TrainingUnit, training_targets

METRIC_SLACK = 0.005       # a candidate may be worse by this on wrong/false-alarm rate: 1 frame in 200
WRONG_PENALTY = 2.0        # ... and each unit of extra wrong-object rate costs this much recall
FALSE_ALARM_CONF = 0.3     # a detection at or above this on a no-ball frame makes its unit a hard negative


def score_weights(weights: Path, clips: List[Clip], device: torch.device, threshold: float = 0.5) -> DetectorMetrics:
    """Raw-detector recall / wrong-object / false-alarm of ``weights`` on whole clips (the
    evaluate_archive metric). The detector runs on CUDA only; ``device`` names which card
    (``cuda`` alone is card 0)."""
    detector = ball_detector(weights, conf=threshold, device=str(device.index or 0))
    pairs = []
    for clip, video, csv_path, _ in clips:
        meta = video_meta(video)
        labels = training_targets(csv_path, meta.width, meta.height)
        scale = scale_for_width(meta.width)
        points = detect_points(detector, video, meta)
        for frame, label in labels.items():
            point = points.get(frame)
            pairs.append((None if label is None else (label[0] * scale, label[1] * scale),
                          None if point is None else (point[0] * scale, point[1] * scale)))
    return DetectorMetrics.of(pairs, 1.0)


def false_alarm_units(weights: Path, clips: List[Clip], samples: Sequence[TrainingUnit], device: torch.device,
                      threshold: float = FALSE_ALARM_CONF) -> Set[int]:
    """The indices into ``samples`` of the units in which the detector on ``weights`` fired, at or
    above ``threshold``, on a frame labelled as having no ball: one prepass over each clip the
    units come from, the hard negatives ``--oversample-false-alarms`` weights up."""
    detector = ball_detector(weights, conf=threshold, device=str(device.index or 0))
    videos = {clip: video for clip, video, _, _ in clips}
    fired: Dict[str, Set[int]] = {}
    for clip in sorted({sample["clip"] for sample in samples}):
        fired[clip] = set(detect_points(detector, videos[clip], video_meta(videos[clip])))
    return {index for index, sample in enumerate(samples)
            if any(point is None and frame in fired[sample["clip"]]
                   for frame, point in zip(sample["indices"], sample["points"]))}


def promotion_improves(candidate: DetectorMetrics, incumbent: DetectorMetrics) -> bool:
    """Does the candidate beat the incumbent: more correct ball with no more wrong-object or
    false-alarm outputs (within METRIC_SLACK, 1 frame in 200)? Asymmetric: a tie keeps the incumbent."""
    if candidate.wrong > incumbent.wrong + METRIC_SLACK:
        return False
    if candidate.false_alarm > incumbent.false_alarm + METRIC_SLACK:
        return False
    return candidate.recall - WRONG_PENALTY * max(0.0, candidate.wrong - incumbent.wrong) > incumbent.recall
