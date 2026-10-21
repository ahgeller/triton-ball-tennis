"""tennis_tracker/engines/engine_boxes.py: the engine-output decode shared by the ball and player backends."""
import numpy as np
import pytest
import torch

from tennis_tracker.engines.engine_boxes import (
    OUTPUT_MODE_POSTPROCESSED,
    OUTPUT_MODE_RAW,
    decode_engine_boxes,
    engine_output_mode,
    nms_xyxy,
    xywh_to_xyxy,
)


def test_xywh_to_xyxy_converts_centre_boxes_and_keeps_extra_columns():
    boxes = np.array([[10.0, 20.0, 4.0, 6.0, 0.9]], dtype=np.float32)
    out = xywh_to_xyxy(boxes)
    np.testing.assert_allclose(out, [[8.0, 17.0, 12.0, 23.0, 0.9]])
    np.testing.assert_allclose(boxes, [[10.0, 20.0, 4.0, 6.0, 0.9]])  # the input is not modified


def test_nms_keeps_the_best_of_overlapping_boxes_in_score_order():
    boxes = np.array([
        [0.0, 0.0, 10.0, 10.0],
        [1.0, 1.0, 11.0, 11.0],   # overlaps the first, lower score
        [50.0, 50.0, 60.0, 60.0],
    ], dtype=np.float32)
    scores = np.array([0.8, 0.6, 0.9], dtype=np.float32)
    assert nms_xyxy(boxes, scores, 0.45).tolist() == [2, 0]
    assert nms_xyxy(boxes, scores, 0.95).tolist() == [2, 0, 1]
    assert nms_xyxy(np.empty((0, 4)), np.empty((0,)), 0.45).tolist() == []


def test_output_mode_is_decided_by_the_tensor_shape():
    assert engine_output_mode(torch.zeros((1, 3, 6))) == OUTPUT_MODE_POSTPROCESSED
    assert engine_output_mode(torch.zeros((1, 5, 8))) == OUTPUT_MODE_RAW
    assert engine_output_mode(torch.zeros((1, 8, 5))) == OUTPUT_MODE_RAW
    assert engine_output_mode(torch.zeros((3, 6))) == OUTPUT_MODE_RAW


def test_postprocessed_rows_are_thresholded_filtered_and_rescaled():
    rows = torch.tensor([[
        [10.0, 20.0, 30.0, 40.0, 0.9, 0.0],
        [1.0, 1.0, 2.0, 2.0, 0.2, 0.0],      # below threshold
        [5.0, 5.0, 6.0, 6.0, 0.8, 1.0],      # another class
    ]])
    out = decode_engine_boxes(rows, (2.0, 0.5), 0.5, 0.45, OUTPUT_MODE_POSTPROCESSED, cls_filter=[0])
    assert out.dtype == np.float64 and out.shape == (1, 6)
    assert out[0].tolist() == [20.0, 10.0, 60.0, 20.0, pytest.approx(0.9), 0.0]
    both = decode_engine_boxes(rows, (1.0, 1.0), 0.5, 0.45, OUTPUT_MODE_POSTPROCESSED)
    assert both[:, 5].tolist() == [0.0, 1.0]
    assert decode_engine_boxes(rows, (1.0, 1.0), 0.95, 0.45, OUTPUT_MODE_POSTPROCESSED).shape == (0, 6)
    assert decode_engine_boxes(torch.zeros((1, 0, 6)), (1.0, 1.0), 0.5, 0.45, OUTPUT_MODE_POSTPROCESSED).shape == (0, 6)


def _raw_head(anchors, channel_major=True):
    """``[1, 4 + classes, anchors]`` like a YOLO head, or its transpose.  Real
    heads have thousands of anchors, which is how the layout is told apart,
    so the given anchors are padded with empty ones."""
    width = len(anchors[0])
    padded = list(anchors) + [[0.0] * width for _ in range(width + 1)]
    p = torch.tensor(padded, dtype=torch.float32).transpose(0, 1).unsqueeze(0)
    return p if channel_major else p.transpose(1, 2)


@pytest.mark.parametrize("channel_major", [True, False])
def test_raw_head_rows_are_thresholded_suppressed_and_rescaled(channel_major):
    raw = _raw_head([
        [10.0, 10.0, 4.0, 4.0, 0.9],
        [10.5, 10.0, 4.0, 4.0, 0.7],   # overlaps the first at IoU 0.78: suppressed
        [50.0, 50.0, 2.0, 2.0, 0.3],   # below threshold
        [80.0, 20.0, 2.0, 4.0, 0.6],
    ], channel_major)
    out = decode_engine_boxes(raw, (2.0, 3.0), 0.5, 0.45, OUTPUT_MODE_RAW)
    assert out.shape == (2, 6)
    assert out[0].tolist() == [16.0, 24.0, 24.0, 36.0, pytest.approx(0.9), 0.0]   # best score first
    assert out[1].tolist() == [158.0, 54.0, 162.0, 66.0, pytest.approx(0.6), 0.0]
    assert decode_engine_boxes(raw, (1.0, 1.0), 0.5, 0.9, OUTPUT_MODE_RAW).shape == (3, 6)  # a looser NMS keeps it


def test_raw_head_takes_the_best_class_and_honours_the_class_filter():
    raw = _raw_head([
        [10.0, 10.0, 4.0, 4.0, 0.9, 0.1],   # class 0
        [40.0, 40.0, 4.0, 4.0, 0.2, 0.8],   # class 1
    ])
    out = decode_engine_boxes(raw, (1.0, 1.0), 0.5, 0.45, OUTPUT_MODE_RAW)
    assert out[:, 4].tolist() == [pytest.approx(0.9), pytest.approx(0.8)]
    assert out[:, 5].tolist() == [0.0, 1.0]
    only_people = decode_engine_boxes(raw, (1.0, 1.0), 0.5, 0.45, OUTPUT_MODE_RAW, cls_filter=[0])
    assert only_people.shape == (1, 6) and only_people[0, 5] == 0.0
    assert decode_engine_boxes(torch.zeros((5, 8)), (1.0, 1.0), 0.5, 0.45, OUTPUT_MODE_RAW).shape == (0, 6)
    assert decode_engine_boxes(torch.zeros((1, 4, 8)), (1.0, 1.0), 0.5, 0.45, OUTPUT_MODE_RAW).shape == (0, 6)
