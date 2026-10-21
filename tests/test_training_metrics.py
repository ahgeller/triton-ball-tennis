"""finetune/training/metrics.py on its own: the winner rule, and the whole-clip measurement on a
synthetic clip through the tracker's own prepass with a stand-in decoder."""
from pathlib import Path
from unittest.mock import patch

import torch

from finetune.data.detector_metrics import DetectorMetrics
from finetune.data.workspace import VideoMeta
from finetune.training import metrics
from finetune.training.metrics import METRIC_SLACK, promotion_improves, score_weights
from tennis_tracker.engines import ball_detectors
from tests._helpers import Capture, cpu_detector, metric_probe


def rates(recall: float, wrong: float, false_alarm: float = 0.0) -> DetectorMetrics:
    return DetectorMetrics(100, recall, wrong, 0.0, 100, false_alarm, 1.0, 2.0)


def test_promotion_improves_wants_more_recall_without_more_wrong_or_false_alarms():
    assert promotion_improves(rates(0.90, 0.01), rates(0.85, 0.01))
    assert not promotion_improves(rates(0.85, 0.01), rates(0.85, 0.01))            # equal: the incumbent stays
    assert promotion_improves(rates(0.90, 0.01 + METRIC_SLACK), rates(0.85, 0.01))  # within the slack
    assert not promotion_improves(rates(0.95, 0.05), rates(0.85, 0.01))
    assert not promotion_improves(rates(0.95, 0.01, 0.5), rates(0.85, 0.01, 0.1))


def test_score_weights_runs_the_trackers_prepass_with_the_threshold_as_its_confidence():
    """The winner metric takes the detector's own whole-clip answer (first answer per frame, the
    trailing window inferred, the y offset applied): on a 7-frame clip the decoder, which answers
    (10 * frame + position in the unit, 20), is asked for frames 0-4 and then the tail 2-6, and
    frames 2-4 keep their first answer. Misses (frame 3, below threshold) are scored as misses."""
    built = {}

    def decode(output, width, height, threshold):
        decoded = []
        for unit in output.reshape(-1, 5):
            for position, value in enumerate(unit.tolist()):
                confidence = 0.2 if int(value) == 3 else 0.9
                decoded.append(((10.0 * value + position, 20.0) if confidence >= threshold else None, confidence))
        return decoded

    def build(weights, conf, device):
        built.update(weights=weights, conf=conf, device=device)
        return cpu_detector(decode, conf=conf)
    labels = {frame: [10.0 * frame + min(frame, 2), 20.0] for frame in range(7)}     # where the first answer lands
    labels[3] = [30.0, 20.0]
    with patch.object(metrics, "ball_detector", side_effect=build), \
            patch.object(metrics, "video_meta", return_value=VideoMeta(30.0, 1920, 1080, 7)), \
            patch.object(metrics, "training_targets", return_value=labels), \
            patch.object(ball_detectors.cv2, "VideoCapture", side_effect=lambda _: Capture(7)), \
            patch.object(torch.cuda, "synchronize"):
        stats = score_weights(Path("w.npz"), [("probe", Path("v"), Path("c"), "custom")], torch.device("cuda:1"), 0.5)
    assert built == {"weights": Path("w.npz"), "conf": 0.5, "device": "1"}
    assert stats.visible == 7 and stats.recall == 6 / 7 and stats.miss == 1 / 7 and stats.wrong == 0.0


def test_score_weights_names_card_zero_for_a_bare_cuda_device():
    with patch.object(metrics, "ball_detector", side_effect=RuntimeError("stop")) as build:
        try:
            score_weights(Path("w.npz"), [], torch.device("cuda"))
        except RuntimeError:
            pass
    assert build.call_args.kwargs == {"conf": 0.5, "device": "0"}


def test_score_weights_scores_a_clip_end_to_end():
    stats = metric_probe(5, {i: [100, 100] for i in range(5)})
    assert isinstance(stats, DetectorMetrics) and stats.recall == 1.0 and stats.wrong == 0.0
