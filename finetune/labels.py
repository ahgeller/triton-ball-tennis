"""The label file format, read and written in one place.

``labels/<clip>_ball.csv`` has the columns ``frame,ball_x,ball_y``: one row per labelled frame,
``frame_###`` naming the zero-based source frame, the ball's centre in native image pixels
(pixel k's centre is at k + 0.5), and an invisible ball parked at the top-right corner,
``(width - 1, 0)``. Any point in that corner - the top 5% of the height and the right 5% of the
width - reads as "no ball". A label file is not a list of absent-ball frames: a frame with no
row has no label at all. On a 60 FPS clip the rows may sit on either the even or the odd
frames; the readers here never change that.

Two sidecars sit beside a label file. ``<clip>_ball.review.json`` records which frames a person
has confirmed (``reviewed``) or changed (``edited``) in the label tool, with a cursor to resume
from; ``<clip>_ball.suspect.json`` holds the frames an audit flagged (``unplaceable``,
``check``) and the answers already given. A sidecar that cannot be parsed is an error that
names the file: nothing here guesses at it or writes over it.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, NamedTuple, Optional, Set, Tuple

from finetune.fileio import atomic_csv

COLUMNS = ("frame", "ball_x", "ball_y")
FRAME_PATTERN = re.compile(r"frame_(\d+)")
SENTINEL_X_SHARE = 0.95      # x at or beyond this share of the width ...
SENTINEL_Y_SHARE = 0.05      # ... and y at or below this share of the height means "no ball"
COORDINATE_DECIMALS = 2      # how the writers format a visible point

LabelPoint = Optional[Tuple[float, float]]
Labels = Dict[int, LabelPoint]


class LabelFormatError(ValueError):
    """A label file or sidecar that does not follow the format; the message names the file."""


class LabelRow(NamedTuple):
    """One row of a label file, as the file has it (the sentinel included)."""
    frame: int
    x: float
    y: float
    source: Optional[str] = None   # pretrack drafts carry where a guess came from ...
    conf: Optional[float] = None   # ... and how confident the detector was


# --------------------------------------------------------------------------- cells

def frame_index(cell: str) -> int:
    """The source frame a ``frame_###`` cell names. Strict: a cell that is not exactly that is an
    error, not a number fished out of it, because a number fished out of a malformed cell is a
    label silently moved to the wrong frame."""
    match = FRAME_PATTERN.fullmatch((cell or "").strip())
    if match is None:
        raise LabelFormatError(f"bad frame cell {cell!r}; expected frame_<index>")
    return int(match.group(1))


def frame_name(index: int) -> str:
    return f"frame_{index:03d}"


def is_visible(x: float, y: float, width: int, height: int) -> bool:
    """False for a point in the corner the files reserve for "no ball"."""
    return not (x >= width * SENTINEL_X_SHARE and y <= height * SENTINEL_Y_SHARE)


def sentinel(width: int) -> Tuple[int, int]:
    """Where an invisible ball is parked: the top-right pixel."""
    return width - 1, 0


def format_point(point: LabelPoint, width: int, height: int) -> Tuple[str, str]:
    """The two coordinate cells for a point (clamped onto the image) or for no ball."""
    if point is None:
        x, y = sentinel(width)
        return str(x), str(y)
    x = min(max(float(point[0]), 0.0), width - 1.0)
    y = min(max(float(point[1]), 0.0), height - 1.0)
    return f"{x:.{COORDINATE_DECIMALS}f}", f"{y:.{COORDINATE_DECIMALS}f}"


# --------------------------------------------------------------------------- label files

def read_label_rows(path: Path) -> List[LabelRow]:
    """Every row of a label file (or a pretrack draft), in file order, sentinel rows included."""
    rows: List[LabelRow] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not set(COLUMNS).issubset(reader.fieldnames or ()):
            raise LabelFormatError(f"{path}: columns are {reader.fieldnames}, expected {','.join(COLUMNS)}")
        for record in reader:
            try:
                frame = frame_index(record["frame"])
                x, y = float(record["ball_x"]), float(record["ball_y"])
            except (LabelFormatError, TypeError, ValueError) as error:
                raise LabelFormatError(f"{path}: {error}") from error
            conf = record.get("conf")
            rows.append(LabelRow(frame, x, y, record.get("source") or None,
                                 float(conf) if conf not in (None, "") else None))
    return rows


def read_labels(path: Path, width: int, height: int) -> Labels:
    """frame -> (x, y) in native pixels, or None where the row says there is no ball."""
    return {row.frame: (row.x, row.y) if is_visible(row.x, row.y, width, height) else None
            for row in read_label_rows(path)}


def write_label_cells(path: Path, rows: Iterable[Tuple[int, str, str]]) -> int:
    """Write ``(frame, x cell, y cell)`` rows, sorted by frame, atomically. Returns the row count."""
    ordered = sorted(rows)
    with atomic_csv(path) as writer:
        writer.writerow(list(COLUMNS))
        for frame, x, y in ordered:
            writer.writerow([frame_name(frame), x, y])
    return len(ordered)


def write_labels(path: Path, labels: Mapping[int, LabelPoint], width: int, height: int) -> Tuple[int, int]:
    """Write frame -> point (None for no ball) as a label file. Returns (rows, visible)."""
    cells = [(frame, *format_point(point, width, height)) for frame, point in labels.items()]
    return write_label_cells(path, cells), sum(point is not None for point in labels.values())


# --------------------------------------------------------------------------- sidecars

def _read_json_object(path: Path, what: str) -> Optional[dict]:
    """The JSON object in a sidecar, None when the file is missing, LabelFormatError otherwise."""
    if not path.is_file():
        return None
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise LabelFormatError(f"{path} is not a {what} this workspace can read ({error})") from error
    if not isinstance(blob, dict):
        raise LabelFormatError(f"{path} is not a {what} this workspace can read (not a JSON object)")
    return blob


def _frame_list(blob: dict, key: str) -> List[int]:
    try:
        return [int(frame) for frame in (blob.get(key) or [])]
    except (TypeError, ValueError) as error:
        raise LabelFormatError(f"{key!r} is not a list of frame numbers ({error})") from error


def read_review(path: Path) -> Optional[dict]:
    """The review sidecar's content, or None when there is none yet. A damaged file is an error
    naming the file: it is somebody's record of which frames they checked, so it is never
    silently replaced by a fresh one."""
    return _read_json_object(path, "review file")


def settled_frames(state: Optional[dict]) -> Set[int]:
    """The frames a review sidecar says a person has confirmed or changed."""
    if state is None:
        return set()
    return set(_frame_list(state, "reviewed")) | set(_frame_list(state, "edited"))


def reviewed_frames(path: Path) -> Set[int]:
    """``settled_frames`` of the sidecar at `path` (empty when there is none)."""
    return settled_frames(read_review(path))


def read_suspects(path: Path) -> dict:
    """The flag sidecar, normalised: ``unplaceable`` and ``check`` frame lists (an older audit's
    ``frames`` list counts as ``check``), the ``answered`` map (frame as text -> [x, y] or None)
    and the ``notes`` map. All empty when there is no file."""
    blob = _read_json_object(path, "flag file")
    if blob is None:
        return {"unplaceable": [], "check": [], "answered": {}, "notes": {}}
    try:
        flags = {"unplaceable": _frame_list(blob, "unplaceable"),
                 "check": _frame_list(blob, "check" if "check" in blob else "frames")}
    except LabelFormatError as error:
        raise LabelFormatError(f"{path}: {error}") from error
    answered = blob.get("answered")
    notes = blob.get("notes")
    flags["answered"] = dict(answered) if isinstance(answered, dict) else {}
    flags["notes"] = dict(notes) if isinstance(notes, dict) else {}
    return flags
