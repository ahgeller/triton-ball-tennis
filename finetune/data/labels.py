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
from; together those are the clip's *reviewed frames* (``reviewed_frames``). The flag sidecar
``<clip>_ball.suspect.json`` holds the frames an audit flagged for a person to look at
(``unplaceable``, ``check``), the answers already given and a note per kind; ``read_flags`` and
``write_flags`` are its only reader and writer. A sidecar that cannot be parsed is an error
that names the file: nothing here guesses at it or writes over it.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Mapping, NamedTuple, Optional, Set, Tuple, TypedDict, get_args

from finetune.data.fileio import atomic_csv, write_atomic
from finetune.data.workspace import review_json

COLUMNS = ("frame", "ball_x", "ball_y")
FRAME_PATTERN = re.compile(r"frame_(\d+)")
SENTINEL_X_SHARE = 0.95      # x at or beyond this share of the width ...
SENTINEL_Y_SHARE = 0.05      # ... and y at or below this share of the height means "no ball"
COORDINATE_DECIMALS = 2      # how the writers format a visible point
FLAG_ANSWER_DECIMALS = 2     # how an answered flag's label is recorded; same_point compares at this precision

LabelPoint = Optional[Tuple[float, float]]
Labels = Dict[int, LabelPoint]
FlagKind = Literal["unplaceable", "check"]
FLAG_KINDS: Tuple[FlagKind, ...] = get_args(FlagKind)     # the two kinds of frame flag an audit raises, as values
DRAFT_COLUMNS = (*COLUMNS, "source", "conf")               # a pretrack draft: a label file plus where each guess came from


class Flags(TypedDict):
    """The flag sidecar as ``read_flags`` returns it and ``write_flags`` takes it."""
    generated: str                                  # the date of the audit that wrote it ("" if unknown)
    notes: Dict[str, str]                           # kind -> what the audit wants looked at
    unplaceable: List[int]
    check: List[int]
    answered: Dict[str, Optional[List[float]]]      # frame as text -> the label it was answered at


class LabelFormatError(ValueError):
    """A label file or sidecar that does not follow the format; the message names the file."""


class LabelRow(NamedTuple):
    """One row of a label file, as the file has it (the sentinel included)."""
    frame: int
    x: float
    y: float
    source: Optional[str] = None   # pretrack drafts carry where a guess came from ...
    conf: Optional[float] = None   # ... and how confident the detector was


class DraftRow(NamedTuple):
    """One row of a pretrack draft as ``write_draft`` takes it: the guess (None for no ball),
    where it came from and how confident the detector was."""
    point: LabelPoint
    source: str
    conf: float


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

def _label_row(record: Dict[str, str]) -> LabelRow:
    conf = record.get("conf")
    return LabelRow(frame_index(record["frame"]), float(record["ball_x"]), float(record["ball_y"]),
                    record.get("source") or None, float(conf) if conf not in (None, "") else None)


def read_label_rows(path: Path) -> List[LabelRow]:
    """Every row of a label file (or a pretrack draft), in file order, sentinel rows included.
    Any cell that does not parse - the frame, a coordinate or a draft's conf - is a
    LabelFormatError naming the file."""
    rows: List[LabelRow] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not set(COLUMNS).issubset(reader.fieldnames or ()):
            raise LabelFormatError(f"{path}: columns are {reader.fieldnames}, expected {','.join(COLUMNS)}")
        for record in reader:
            try:
                rows.append(_label_row(record))
            except (LabelFormatError, TypeError, ValueError) as error:
                raise LabelFormatError(f"{path}: {error}") from error
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


def write_draft(path: Path, rows: Mapping[int, DraftRow], width: int, height: int) -> Tuple[int, int]:
    """Write frame -> DraftRow as a pretrack draft (``DRAFT_COLUMNS``), sorted by frame,
    atomically; the point cells are a label file's (``format_point``), the ``conf`` cell two
    decimals. Returns (rows with a guess, rows whose guess sat in the corner reserved for "no
    ball" and so reads back as no ball: the caller says so, nothing here moves a guess)."""
    guessed = parked = 0
    with atomic_csv(path) as writer:
        writer.writerow(list(DRAFT_COLUMNS))
        for frame in sorted(rows):
            point, source, conf = rows[frame]
            writer.writerow([frame_name(frame), *format_point(point, width, height), source, f"{conf:.2f}"])
            if point is not None:
                guessed += 1
                parked += not is_visible(point[0], point[1], width, height)
    return guessed, parked


# --------------------------------------------------------------------------- sidecars

def read_json_object(path: Path, what: str) -> Optional[dict]:
    """The JSON object in a sidecar (or any workspace record kept as one), None when the file is
    missing, LabelFormatError naming the file otherwise."""
    if not path.is_file():
        return None
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise LabelFormatError(f"{path} is not a {what} this workspace can read ({error})") from error
    if not isinstance(blob, dict):
        raise LabelFormatError(f"{path} is not a {what} this workspace can read (not a JSON object)")
    return blob


def frame_list(state: Mapping, key: str, path: Path) -> List[int]:
    """The frame numbers under `key` in a sidecar's object (missing or null: none). Anything that
    is not a list of frame numbers is a LabelFormatError naming `path`: the one validation every
    reader of the review and flag sidecars applies."""
    value = state.get(key) or []
    if not isinstance(value, list):
        raise LabelFormatError(f"{path}: {key!r} is not a list of frame numbers (got {type(value).__name__})")
    try:
        return [int(frame) for frame in value]
    except (TypeError, ValueError) as error:
        raise LabelFormatError(f"{path}: {key!r} is not a list of frame numbers ({error})") from error


def read_review(path: Path) -> Optional[dict]:
    """The review sidecar's content, or None when there is none yet. A damaged file is an error
    naming the file: it is somebody's record of which frames they reviewed, so it is never
    silently replaced by a fresh one."""
    return read_json_object(path, "review file")


def reviewed_frames(clip: str, labels: Optional[Path] = None) -> Set[int]:
    """The frames of `clip` a person has dealt with in the label tool - confirmed (SPACE) or
    changed (a click), both count - from its review sidecar; empty when there is none."""
    path = review_json(clip, labels)
    state = read_review(path)
    if state is None:
        return set()
    return set(frame_list(state, "reviewed", path)) | set(frame_list(state, "edited", path))


def json_map(state: Mapping, key: str, path: Path) -> dict:
    """The object under `key` in a sidecar's object (missing or null: empty). Anything else is a
    LabelFormatError naming `path`, as ``frame_list`` does for the frame lists: the ``answered``
    map is the record of somebody's answers, never guessed at or written over."""
    value = state.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise LabelFormatError(f"{path}: {key!r} is not an object (got {type(value).__name__})")
    return dict(value)


def read_flags(path: Path) -> Flags:
    """The flag sidecar, normalised: the ``unplaceable`` and ``check`` frame lists (an older
    audit's ``frames`` list counts as ``check``), the ``answered`` map, the ``notes`` map and the
    audit date. All empty when there is no file; a field of the wrong shape is a LabelFormatError."""
    blob = read_json_object(path, "flag file")
    if blob is None:
        return {"generated": "", "notes": {}, "unplaceable": [], "check": [], "answered": {}}
    return {"generated": str(blob.get("generated") or ""),
            "notes": json_map(blob, "notes", path),
            "unplaceable": frame_list(blob, "unplaceable", path),
            "check": frame_list(blob, "check" if "check" in blob else "frames", path),
            "answered": json_map(blob, "answered", path)}


def write_flags(path: Path, flags: Flags) -> None:
    """Write the flag sidecar atomically in its one shape: ``generated`` and ``notes`` only when
    there is something to say, then ``unplaceable``, ``check`` and ``answered``."""
    blob: Dict[str, object] = {}
    if flags["generated"]:
        blob["generated"] = flags["generated"]
    if flags["notes"]:
        blob["notes"] = flags["notes"]
    blob.update(unplaceable=sorted(set(int(f) for f in flags["unplaceable"])),
                check=sorted(set(int(f) for f in flags["check"])), answered=flags["answered"])
    write_atomic(path, json.dumps(blob, indent=1))


def answered_point(point: LabelPoint) -> Optional[List[float]]:
    """How an answered flag records the label it was answered at."""
    return None if point is None else [round(point[0], FLAG_ANSWER_DECIMALS), round(point[1], FLAG_ANSWER_DECIMALS)]
