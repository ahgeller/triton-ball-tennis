"""finetune/training/metrics.py on its own: the winner rule, and the whole-clip prediction on a
synthetic clip with a stand-in detector."""
from unittest.mock import patch

import torch

from finetune.training import metrics
from finetune.training.metrics import METRIC_SLACK, better, predict_clip
from tests._helpers import Capture, metric_probe


def test_better_wants_more_recall_without_more_wrong_or_false_alarms():
    assert better({"recall": 0.90, "wrong": 0.01}, {"recall": 0.85, "wrong": 0.01})
    assert not better({"recall": 0.85, "wrong": 0.01}, {"recall": 0.85, "wrong": 0.01})            # equal: the incumbent stays
    assert better({"recall": 0.90, "wrong": 0.01 + METRIC_SLACK}, {"recall": 0.85, "wrong": 0.01})  # within the slack
    assert not better({"recall": 0.95, "wrong": 0.05}, {"recall": 0.85, "wrong": 0.01})
    assert not better({"recall": 0.95, "wrong": 0.01, "false_alarm": 0.5}, {"recall": 0.85, "wrong": 0.01, "false_alarm": 0.1})


def test_predict_clip_keeps_the_first_answer_for_a_frame_and_infers_the_tail():
    seen = []

    def model(batch):
        seen.append(batch.shape[0])
        return "output"

    def decode(output, width, height, threshold):
        return [((10.0, 20.0), 0.9)] * 5

    with patch.object(metrics.cv2, "VideoCapture", side_effect=lambda _: Capture(7)), \
            patch.object(metrics, "frame_tensor", return_value=torch.zeros(3, 1, 1)), \
            patch.object(metrics, "decode_predictions", side_effect=decode):
        predictions = predict_clip(model, metrics.Path("unused"), 1, 640, 360, torch.device("cpu"), 0.5)
    assert sorted(predictions) == list(range(7)) and len(seen) == 2          # frames 0-4, then the tail 2-6
    assert predictions[6][0] == 10.0 and predictions[6][1] >= 20.0           # the y offset is added, never subtracted


def test_detector_metrics_scores_a_clip_end_to_end():
    stats = metric_probe(5, {i: [100, 100] for i in range(5)})
    assert set(stats) >= {"recall", "wrong", "false_alarm", "visible"} and stats["recall"] == 1.0
