"""The landing labels: ``labels/<clip>_bounces.csv``, the frames where the ball hit the ground.

A file has the ball label files' columns, ``frame,ball_x,ball_y`` (``finetune.data.labels``):
``frame_###`` names the zero-based source frame of the landing, the coordinates are the landing
point in native image pixels. Both coordinate cells may be left blank: the point is then the
clip's ball label at that frame, or at the adjacent frame when the clip is labelled on the other
60 FPS phase (``resolve_points``). One coordinate without the other is an error. The file lists
every landing of every rally, serves and faults included; a frame within the scorer's tolerance
(``evaluation.court_events_score``, 0.1 s) of the true landing is close enough, so either 60 FPS
phase will do.

It is a file of its own, not a column of ``_ball.csv``: ``labels.write_labels`` rewrites that file
with exactly its columns. ``workspace.bounce_csv`` names it; the files are small and are versioned
with the scorer that reads them.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List, Mapping, NamedTuple, Optional, Tuple

from finetune.data.fileio import atomic_csv
from finetune.data.labels import COLUMNS, COORDINATE_DECIMALS, LabelFormatError, LabelPoint, frame_index, frame_name

# A landing labelled without a point reads the ball label at its frame, else at one of these
# offsets, in this order: the adjacent frame on the other 60 FPS phase.
_ADJACENT_PHASE_OFFSETS = (-1, 1)


class BounceLabel(NamedTuple):
    """One labelled landing: its source frame and its point, None when the row left it blank."""

    frame: int
    point: LabelPoint = None


def _point_cells(record: Mapping[str, str], path: Path) -> LabelPoint:
    x_cell, y_cell = (record.get("ball_x") or "").strip(), (record.get("ball_y") or "").strip()
    if not x_cell and not y_cell:
        return None
    if not x_cell or not y_cell:
        raise LabelFormatError(f"{path}: {record.get('frame')!r} gives one coordinate without the other")
    try:
        return float(x_cell), float(y_cell)
    except ValueError as error:
        raise LabelFormatError(f"{path}: {record.get('frame')!r}: {error}") from error


def read_bounce_labels(path: Path) -> List[BounceLabel]:
    """Every landing in the file, in frame order. A frame listed twice, a frame cell that is not
    ``frame_<index>``, a coordinate that does not parse or a lone coordinate is a
    ``LabelFormatError`` naming the file."""
    labels: List[BounceLabel] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not set(COLUMNS).issubset(reader.fieldnames or ()):
            raise LabelFormatError(f"{path}: columns are {reader.fieldnames}, expected {','.join(COLUMNS)}")
        for record in reader:
            if not any((value or "").strip() for value in record.values()):
                continue                                  # a blank line
            try:
                frame = frame_index(record.get("frame") or "")
            except LabelFormatError as error:
                raise LabelFormatError(f"{path}: {error}") from error
            labels.append(BounceLabel(frame, _point_cells(record, path)))
    seen = set()
    for label in labels:
        if label.frame in seen:
            raise LabelFormatError(f"{path}: frame {label.frame} is listed twice")
        seen.add(label.frame)
    return sorted(labels)


def write_bounce_labels(path: Path, labels: Iterable[BounceLabel]) -> int:
    """Write the landings, sorted by frame, atomically; a landing without a point gets blank
    coordinate cells. Returns the row count."""
    ordered = sorted(labels)
    with atomic_csv(path) as writer:
        writer.writerow(list(COLUMNS))
        for frame, point in ordered:
            cells = ("", "") if point is None else (f"{point[0]:.{COORDINATE_DECIMALS}f}", f"{point[1]:.{COORDINATE_DECIMALS}f}")
            writer.writerow([frame_name(frame), *cells])
    return len(ordered)


def resolve_points(labels: Iterable[BounceLabel], ball_labels: Mapping[int, LabelPoint]) -> List[BounceLabel]:
    """The landings with every blank point filled from the clip's ball labels (``labels.read_labels``):
    the label at the landing's frame, else at the adjacent frame on the other 60 FPS phase. A
    landing with no visible ball label there keeps ``None``; its timing is still scored."""
    resolved = []
    for label in labels:
        point = label.point
        if point is None:
            for offset in (0, *_ADJACENT_PHASE_OFFSETS):
                found = ball_labels.get(label.frame + offset)
                if found is not None:
                    point = (float(found[0]), float(found[1]))
                    break
        resolved.append(BounceLabel(label.frame, point))
    return resolved
