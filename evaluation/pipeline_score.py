"""Scoring one clip of a pipeline run against its labels: the detector metric over the run's
per-frame rows, where each wrong or false output landed (a player, a court line, elsewhere) and
the frames worth a look."""

from __future__ import annotations

import math
from typing import Dict, List, NamedTuple, Optional, Tuple

from finetune.data.detector_metrics import HIT_PX, WRONG_PX, DetectorMetrics, scale_for_width
from tennis_tracker.court.court_geometry import LINE_PAIRS_14, MIN_KEYPOINT_VALUES, last_valid_court_keypoints

LINE_PX = 10.0  # "on a court line" if within this many 1080p pixels of one


def _inside_player(x: float, y: float, boxes) -> bool:
    values = boxes.values() if isinstance(boxes, dict) else (boxes or [])
    return any(b and len(b) >= 4 and b[0] <= x <= b[2] and b[1] <= y <= b[3] for b in values)


def _segment_distance(x: float, y: float, ax: float, ay: float, bx: float, by: float) -> float:
    dx, dy = bx - ax, by - ay
    length2 = dx * dx + dy * dy
    t = 0.0 if length2 <= 0 else max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / length2))
    return math.hypot(x - (ax + t * dx), y - (ay + t * dy))


def _near_court_line(x: float, y: float, keypoints, scale: float) -> bool:
    """Within LINE_PX (1080p) of a court line the keypoints describe; a keypoint list is usable
    from MIN_KEYPOINT_VALUES values (the tracker's rule) and only the lines it holds are tried."""
    if not keypoints or len(keypoints) < MIN_KEYPOINT_VALUES:
        return False
    for a, b in LINE_PAIRS_14:
        if 2 * max(a, b) + 1 >= len(keypoints):
            continue
        ax, ay, bx, by = keypoints[2 * a], keypoints[2 * a + 1], keypoints[2 * b], keypoints[2 * b + 1]
        if (ax, ay) == (0.0, 0.0) or (bx, by) == (0.0, 0.0):
            continue
        if _segment_distance(x, y, ax, ay, bx, by) * scale <= LINE_PX:
            return True
    return False


def _classify(x: float, y: float, row: dict, fallback_keypoints: Optional[List[float]], scale: float) -> str:
    """What did a wrong/false output land on: player, line, or something else."""
    if _inside_player(x, y, row.get("player_boxes")):
        return "player"
    if _near_court_line(x, y, row.get("court_keypoints") or fallback_keypoints, scale):
        return "line"
    return "other"


class ClipScore(NamedTuple):
    """One clip's pipeline output scored against its labels."""
    stats: DetectorMetrics
    by_source: Dict[str, Dict[str, int]]
    pairs: List[Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]]    # 1080p (label, pred)
    landed: Dict[str, int]
    bad_frames: List[Tuple[int, str, Optional[str], str]]    # (frame, kind, source, where)


def score_pipeline_clip(labels: Dict[int, Optional[Tuple[float, float]]], payload: dict, width: int) -> ClipScore:
    """One clip's pipeline output against its labels: (stats, by_source, 1080p pairs, landed, bad frames)."""
    frames = {row["frame"]: row for row in payload["frames"]}
    # The court as last seen, by the tracker's own rule (export.py writes the same thing).
    fallback_keypoints = (payload.get("last_valid_court_keypoints")
                          or last_valid_court_keypoints([row.get("court_keypoints") for row in payload["frames"]]))
    scale = scale_for_width(width)
    pairs = []
    by_source: Dict[str, Dict[str, int]] = {}
    landed: Dict[str, int] = {}
    bad_frames = []
    for frame, label in sorted(labels.items()):
        row = frames.get(frame)
        present = bool(row and row.get("present"))
        pred = (row["x"], row["y"]) if present else None
        pairs.append((label, pred))
        if not present:
            continue
        bucket = by_source.setdefault(row.get("source", "?"), {"hit": 0, "localized_error": 0, "wrong": 0, "invisible": 0})
        bad = None
        if label is None:
            bucket["invisible"] += 1
            bad = "invisible"
        else:
            error = math.hypot(pred[0] - label[0], pred[1] - label[1]) * scale
            if error > WRONG_PX:
                bucket["wrong"] += 1
                bad = "wrong"
            else:
                bucket["hit" if error <= HIT_PX else "localized_error"] += 1
        if bad:
            where = _classify(pred[0], pred[1], row, fallback_keypoints, scale)
            landed[where] = landed.get(where, 0) + 1
            bad_frames.append((frame, bad, row.get("source"), where))
    normalized_pairs = [(None if label is None else (label[0] * scale, label[1] * scale),
                         None if pred is None else (pred[0] * scale, pred[1] * scale))
                        for label, pred in pairs]
    return ClipScore(DetectorMetrics.of(pairs, scale), by_source, normalized_pairs, landed, bad_frames)
