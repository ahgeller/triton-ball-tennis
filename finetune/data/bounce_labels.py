"""The landing labels: ``labels/<clip>_bounces.csv``, the frames where the ball hit the ground.

A file has the ball label files' columns, ``frame,ball_x,ball_y`` (``finetune.data.labels``):
``frame_###`` names the zero-based source frame of the landing, and ``frame_###.5`` a landing
between that frame and the next, for a ball that is in the air on both (``landing_frame`` reads
exactly those two forms, ``landing_name`` writes them); the coordinates are the landing point in
native image pixels. Both coordinate cells may be left blank: the point is then the clip's ball
label at that frame, or, when the frame has none (the ball hidden, or the clip labelled on the
other 60 FPS phase), the midpoint of the ball labels on the frames either side of it, or the one
of them there is; a landing between two frames takes the same midpoint of the two
(``resolve_points``). One coordinate without the other is an error. The file
lists every landing of every rally, serves and faults included; a frame within the scorer's
tolerance (``evaluation.court_events_score``, 0.1 s) of the true landing is close enough, so
either 60 FPS phase will do.

It is a file of its own, not a column of ``_ball.csv``: ``labels.write_labels`` rewrites that file
with exactly its columns. ``workspace.bounce_csv`` names it; the files are small and are versioned
with the scorer that reads them.
"""
from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Iterable, List, Mapping, NamedTuple, Optional, Tuple

from finetune.data.fileio import atomic_csv
from finetune.data.labels import COLUMNS, COORDINATE_DECIMALS, LabelFormatError, LabelPoint, frame_name

# A landing's frame cell: frame_<index>, or frame_<index>.5 for one between that frame and the next.
LANDING_FRAME_PATTERN = re.compile(r"frame_(\d+)(\.5)?")


class BounceLabel(NamedTuple):
    """One labelled landing: its source frame (an int, or a half frame ending .5 for a landing
    between two frames) and its point, None when the row left it blank."""

    frame: float
    point: LabelPoint = None


def landing_frame(cell: str) -> float:
    """The frame a landing's ``frame`` cell names: an int for ``frame_###``, the half frame for
    ``frame_###.5``. As strict as ``labels.frame_index``: a cell that is not exactly one of those
    two forms (``frame_12.4``, ``frame_12.50``, ``frame_.5``) is an error, not a landing moved to
    whatever frame could be fished out of it."""
    match = LANDING_FRAME_PATTERN.fullmatch((cell or "").strip())
    if match is None:
        raise LabelFormatError(f"bad frame cell {cell!r}; expected frame_<index> or frame_<index>.5")
    whole = int(match.group(1))
    return whole + 0.5 if match.group(2) else whole


def landing_name(frame: float) -> str:
    """A landing's ``frame`` cell: ``frame_###`` for a whole frame, ``frame_###.5`` for one between
    two. Any other fraction is a ValueError: the file has no way to say it."""
    whole = math.floor(frame)
    if frame == whole:
        return frame_name(int(whole))
    if frame - whole == 0.5:
        return frame_name(int(whole)) + ".5"
    raise ValueError(f"a landing sits on a frame or halfway between two, not at {frame!r}")


def landing_text(frame: float) -> str:
    """A landing's frame as the reports print it: ``412``, or ``412.5`` between two frames."""
    return str(int(frame)) if float(frame).is_integer() else f"{frame:.1f}"


def between_frames(first: int, second: int) -> float:
    """The frame of a landing halfway between two label frames: the half frame between consecutive
    frames (3.5), the whole frame between two a step apart (413 between 412 and 414 on a 60 FPS
    clip labelled every other frame), as an int when it is whole."""
    midpoint = (first + second) / 2.0
    return int(midpoint) if midpoint.is_integer() else midpoint


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
    ``frame_<index>`` or ``frame_<index>.5``, a coordinate that does not parse or a lone coordinate
    is a ``LabelFormatError`` naming the file."""
    labels: List[BounceLabel] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not set(COLUMNS).issubset(reader.fieldnames or ()):
            raise LabelFormatError(f"{path}: columns are {reader.fieldnames}, expected {','.join(COLUMNS)}")
        for record in reader:
            if not any((value or "").strip() for value in record.values()):
                continue                                  # a blank line
            try:
                frame = landing_frame(record.get("frame") or "")
            except LabelFormatError as error:
                raise LabelFormatError(f"{path}: {error}") from error
            labels.append(BounceLabel(frame, _point_cells(record, path)))
    seen = set()
    for label in labels:
        if label.frame in seen:
            raise LabelFormatError(f"{path}: frame {landing_text(label.frame)} is listed twice")
        seen.add(label.frame)
    return sorted(labels, key=lambda label: label.frame)


def write_bounce_labels(path: Path, labels: Iterable[BounceLabel]) -> int:
    """Write the landings, sorted by frame, atomically; a landing without a point gets blank
    coordinate cells. Returns the row count. A frame given twice is a ValueError before anything
    is written: ``read_bounce_labels`` would refuse the file."""
    ordered = sorted(labels, key=lambda label: label.frame)   # by frame alone: a point is not an order
    for earlier, later in zip(ordered, ordered[1:]):
        if earlier.frame == later.frame:
            raise ValueError(f"frame {landing_text(later.frame)} is given twice; a file holds one landing per frame")
    with atomic_csv(path) as writer:
        writer.writerow(list(COLUMNS))
        for frame, point in ordered:
            cells = ("", "") if point is None else (f"{point[0]:.{COORDINATE_DECIMALS}f}", f"{point[1]:.{COORDINATE_DECIMALS}f}")
            writer.writerow([landing_name(frame), *cells])
    return len(ordered)


def _point_from_ball_labels(frame: float, ball_labels: Mapping[int, LabelPoint]) -> LabelPoint:
    """Where the ball labels put a landing that has no point of its own (see ``resolve_points``):
    the label on its frame, else the midpoint of the labels on the frames either side of it (the
    two a half frame lies between), or the one of them there is."""
    whole = int(math.floor(frame))
    if frame == whole:
        found = ball_labels.get(whole)
        if found is not None:
            return float(found[0]), float(found[1])
        either_side = (ball_labels.get(whole - 1), ball_labels.get(whole + 1))
    else:
        either_side = (ball_labels.get(whole), ball_labels.get(whole + 1))
    labelled = [point for point in either_side if point is not None]
    if not labelled:
        return None
    return (sum(float(point[0]) for point in labelled) / len(labelled),
            sum(float(point[1]) for point in labelled) / len(labelled))


def resolve_points(labels: Iterable[BounceLabel], ball_labels: Mapping[int, LabelPoint]) -> List[BounceLabel]:
    """The landings with every blank point filled from the clip's ball labels (``labels.read_labels``):
    the label at the landing's frame, else the midpoint of the labels on the frame before and the
    frame after it (the ball hidden there, or the clip labelled on the other 60 FPS phase), or the
    one of them there is; a landing between two frames takes the same midpoint of the two. A landing
    with no visible ball label on any of those frames keeps ``None``; its timing is still scored."""
    return [BounceLabel(label.frame, _point_from_ball_labels(label.frame, ball_labels) if label.point is None else label.point)
            for label in labels]
