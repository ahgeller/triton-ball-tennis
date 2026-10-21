"""Decoding Ultralytics detection-engine outputs into boxes.

Shared by the ball and player engine backends.  An engine exported with NMS
emits ``[x1, y1, x2, y2, conf, cls]`` rows (``OUTPUT_MODE_POSTPROCESSED``);
a raw detection head emits ``[cx, cy, w, h, class scores...]`` per anchor
(``OUTPUT_MODE_RAW``), which is thresholded and suppressed here.  Either way
``decode_engine_boxes`` returns ``[x1, y1, x2, y2, conf, cls]`` rows scaled
to source pixels.
"""
from __future__ import annotations

from typing import Literal, Optional, Sequence, Tuple

import numpy as np
import torch

POSTPROCESSED_COLUMNS = 6  # x1, y1, x2, y2, conf, cls rows from an NMS-exporting engine
# How an engine's output tensor is laid out: NMS'd rows, or a raw detection head.
OutputMode = Literal["postprocessed", "raw_nms"]
OUTPUT_MODE_POSTPROCESSED: OutputMode = "postprocessed"
OUTPUT_MODE_RAW: OutputMode = "raw_nms"


def engine_output_mode(pred: torch.Tensor) -> OutputMode:
    """``OUTPUT_MODE_POSTPROCESSED`` for engines that export NMS'd
    ``[x1, y1, x2, y2, conf, cls]`` rows, ``OUTPUT_MODE_RAW`` for raw heads."""
    if pred.dim() == 3 and pred.shape[-1] == POSTPROCESSED_COLUMNS:
        return OUTPUT_MODE_POSTPROCESSED
    return OUTPUT_MODE_RAW


def xywh_to_xyxy(boxes_xywh: np.ndarray) -> np.ndarray:
    """Centre/size boxes as corner boxes (rows of ``[x1, y1, x2, y2, ...]``)."""
    out = boxes_xywh.copy()
    out[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] * 0.5
    out[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] * 0.5
    out[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] * 0.5
    out[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] * 0.5
    return out


def nms_xyxy(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_thres: float) -> np.ndarray:
    """Indices of the boxes greedy non-maximum suppression keeps, best score first."""
    if boxes_xyxy.size == 0 or scores.size == 0:
        return np.empty((0,), dtype=np.int32)
    x1 = boxes_xyxy[:, 0]
    y1 = boxes_xyxy[:, 1]
    x2 = boxes_xyxy[:, 2]
    y2 = boxes_xyxy[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = np.argsort(scores)[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter + 1e-6
        iou = inter / union
        order = rest[iou <= float(iou_thres)]
    return np.asarray(keep, dtype=np.int32)


def _no_rows() -> np.ndarray:
    return np.empty((0, POSTPROCESSED_COLUMNS), dtype=np.float64)


def _postprocessed_rows(preds: torch.Tensor, conf_threshold: float) -> np.ndarray:
    """The engine's own NMS'd rows at or above ``conf_threshold``, as float64."""
    if preds.dim() == 3:
        preds = preds[0]
    rows = preds.detach().float().cpu().numpy()
    if rows.ndim != 2 or rows.size == 0 or rows.shape[1] < POSTPROCESSED_COLUMNS:
        return _no_rows()
    rows = rows[:, :POSTPROCESSED_COLUMNS].astype(np.float64)
    return rows[rows[:, 4] >= float(conf_threshold)]


def _raw_head_rows(preds: torch.Tensor, conf_threshold: float, iou_threshold: float) -> np.ndarray:
    """Threshold and suppress a raw ``[1, 4 + classes, anchors]`` (or
    transposed) head; the class is the best-scoring one."""
    if preds.dim() != 3:
        return _no_rows()
    p = preds[0]
    if p.shape[0] <= p.shape[1]:
        p = p.transpose(0, 1)
    p = p.detach().float().cpu().numpy()
    if p.size == 0 or p.shape[1] <= 4:
        return _no_rows()
    boxes = xywh_to_xyxy(p[:, :4].astype(np.float32))
    cls_scores = p[:, 4:].astype(np.float32)
    if cls_scores.shape[1] > 1:
        cls_ids = np.argmax(cls_scores, axis=1)
        scores = cls_scores[np.arange(cls_scores.shape[0]), cls_ids]
    else:
        cls_ids = np.zeros((cls_scores.shape[0],), dtype=np.int32)
        scores = cls_scores[:, 0]
    keep = scores >= float(conf_threshold)
    boxes, scores, cls_ids = boxes[keep], scores[keep], cls_ids[keep]
    if boxes.shape[0] == 0:
        return _no_rows()
    keep_idx = nms_xyxy(boxes, scores, float(iou_threshold))
    boxes, scores, cls_ids = boxes[keep_idx], scores[keep_idx], cls_ids[keep_idx]
    return np.column_stack((boxes.astype(np.float64), scores.astype(np.float64), cls_ids.astype(np.float64)))


def decode_engine_boxes(
    preds: torch.Tensor,
    scale: Tuple[float, float],
    conf_threshold: float,
    iou_threshold: float,
    output_mode: OutputMode,
    cls_filter: Optional[Sequence[int]] = None,
    frame_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """One engine output tensor as ``[x1, y1, x2, y2, conf, cls]`` rows in source pixels.

    ``scale`` is ``(sx, sy)`` from engine input pixels back to the source
    frame (what the session's preprocessing returns).  ``iou_threshold`` is
    the NMS threshold for raw heads; postprocessed engines already ran NMS.
    ``output_mode`` is what ``engine_output_mode`` said about the tensor.
    ``cls_filter`` keeps only those class ids (None keeps every class).
    ``frame_size`` is the source frame's ``(width, height)``; given, every box
    is clipped onto it, since a box can overhang the engine input on either
    path and the consumers (ByteTrack, the court-side player filter, the ROI
    tracker) expect frame pixels.
    """
    if output_mode == OUTPUT_MODE_POSTPROCESSED:
        rows = _postprocessed_rows(preds, conf_threshold)
    elif output_mode == OUTPUT_MODE_RAW:
        rows = _raw_head_rows(preds, conf_threshold, iou_threshold)
    else:
        raise ValueError(f"unknown engine output mode {output_mode!r}")
    if cls_filter is not None and rows.shape[0]:
        rows = rows[np.isin(rows[:, 5].astype(np.int64), list(cls_filter))]
    if rows.shape[0]:
        sx, sy = scale
        rows[:, [0, 2]] *= sx
        rows[:, [1, 3]] *= sy
        if frame_size is not None:
            width, height = frame_size
            rows[:, [0, 2]] = np.clip(rows[:, [0, 2]], 0, width - 1)
            rows[:, [1, 3]] = np.clip(rows[:, [1, 3]], 0, height - 1)
    return rows
