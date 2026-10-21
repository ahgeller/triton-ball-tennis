"""What the label tool reads and writes: a clip's rows, the review and flag sidecars, and the
settings file. The on-disk formats are those of ``finetune.labels``; this module turns them into
``Row`` objects and back. ``load_*`` functions read and return; ``apply_*`` functions write
what a sidecar says onto the rows they are given."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from finetune.fileio import write_atomic
from finetune.labels import (Flags, answered_point, format_point, frame_list, is_visible, read_flags, read_label_rows,
                             read_review, write_flags, write_label_cells)

CONFIDENT_CONF = 0.70       # a draft at or above this confidence is not "uncertain"
SETTINGS_NAME = "label_tool.settings.json"             # in the labels folder: filter, inset, trail


class Row:
    __slots__ = ("frame", "x", "y", "visible", "source", "conf", "reviewed", "edited", "suspect",
                 "history", "clip")
    revision = 0                      # bumped by every write to any row; Tool.render_state keys on it

    def __setattr__(self, name, value) -> None:
        object.__setattr__(self, name, value)
        Row.revision += 1

    def __init__(self, frame: int, x: float, y: float, visible: bool, source: str, conf: float):
        self.frame, self.x, self.y, self.visible = frame, x, y, visible
        self.clip = ""                # which clip this row belongs to, set by Source
        self.source, self.conf = source, conf
        self.reviewed = False
        self.edited = False
        self.suspect = False          # flagged by ft.py check --audit --mark; answered by SPACE or an edit
        self.history: List[tuple] = []

    @property
    def settled(self) -> bool:
        """You have looked at this frame and said something about it - either way it is dealt with."""
        return self.reviewed or self.edited

    @property
    def uncertain(self) -> bool:
        """Worth stopping on: no confident guess, or an audit flag you have not answered yet - which
        reopens a frame you had already reviewed, since catching those is what an audit is for."""
        return self.suspect or not (self.visible and self.conf >= CONFIDENT_CONF)

    def snapshot(self) -> None:
        self.history.append((self.x, self.y, self.visible, self.reviewed, self.edited, self.suspect,
                             self.source))

    def undo(self) -> None:
        if self.history:
            self.x, self.y, self.visible, self.reviewed, self.edited, self.suspect, self.source = \
                self.history.pop()


def load_rows(csv_path: Path, width: int, height: int, origin: str = "det") -> List[Row]:
    """origin names where a row with no source column of its own came from: "csv" for labels that
    arrived with the clip, "det" for ones pretrack.py guessed. Only pretrack writes a conf column,
    and the tool drops it on save, so without this a hand-labelled clip read back as "det 1.00"."""
    rows = []
    for record in read_label_rows(csv_path):
        visible = is_visible(record.x, record.y, width, height)
        rows.append(Row(record.frame, record.x if visible else width / 2.0, record.y if visible else height / 2.0,
                        visible, record.source or (origin if visible else "none"),
                        record.conf if record.conf is not None else 0.0))
    rows.sort(key=lambda row: row.frame)
    return rows


def save_rows(csv_path: Path, rows: List[Row], width: int, height: int) -> None:
    write_label_cells(csv_path, [(row.frame, *format_point((row.x, row.y) if row.visible else None, width, height))
                                 for row in rows])


def save_flags(path: Path, rows: List[Row]) -> None:
    """Drop the flags you have answered from the flag file, and remember each answer.

    A flag is a question, and confirming the label (SPACE) answers it just as well as changing it
    does. A flag on a frame you had reviewed before the audit stays open until you look at it
    again: catching mistakes in reviewed frames is what an audit is for. Answered frames go in an
    "answered" map with the label they were answered at, so a later audit does not raise the same
    question about the same label again (label_audit.flag_frames reads it). A damaged file is its
    LabelFormatError, naming the file; an older audit's ``frames`` list reads as ``check``.
    """
    if not path.is_file():
        return
    flags: Flags = read_flags(path)
    by_frame = {row.frame: row for row in rows}
    open_now = {row.frame for row in rows if row.suspect}
    kept = {kind: [f for f in flags[kind] if f in open_now or f not in by_frame] for kind in ("unplaceable", "check")}
    answered = dict(flags["answered"])
    for kind in ("unplaceable", "check"):
        for frame in flags[kind]:
            if frame in by_frame and frame not in open_now:
                row = by_frame[frame]
                answered[str(frame)] = answered_point((row.x, row.y) if row.visible else None)
    if kept["unplaceable"] == flags["unplaceable"] and kept["check"] == flags["check"] and answered == flags["answered"]:
        return
    write_flags(path, {**flags, "unplaceable": kept["unplaceable"], "check": kept["check"], "answered": answered})


def save_review(path: Path, rows: List[Row], cursor: int, zoom: float) -> None:
    write_atomic(path, json.dumps({
        "cursor": cursor,
        "frame": rows[cursor].frame if 0 <= cursor < len(rows) else None,   # resume by frame, not index
        "zoom": zoom,
        "reviewed": [row.frame for row in rows if row.reviewed],
        "edited": [row.frame for row in rows if row.edited],
        "evidence": {str(row.frame): {"source": row.source, "conf": row.conf} for row in rows},
    }))


def apply_flags(path: Path, rows: List[Row]) -> int:
    """Mark the frames a previous audit flagged on the rows; returns how many. They are the ones
    worth your time first. A flag file that cannot be read is an error naming it
    (LabelFormatError), not an empty answer."""
    flags: Flags = read_flags(path)
    frames = set(flags["unplaceable"]) | set(flags["check"])
    for row in rows:
        row.suspect = row.frame in frames
    return sum(row.suspect for row in rows)


def apply_review(path: Path, rows: List[Row]) -> Optional[dict]:
    """Mark the rows the review sidecar says are reviewed or edited (and restore their evidence),
    and return the sidecar; None when there is none yet. A damaged file, or a frame list that is
    not one, is an error naming it (LabelFormatError): the tool never starts afresh over
    somebody's record of what they checked. Odd evidence inside a parseable file is tolerated
    frame by frame."""
    state = read_review(path)
    if state is None:
        return None
    reviewed, edited = set(frame_list(state, "reviewed", path)), set(frame_list(state, "edited", path))
    evidence_of = state.get("evidence") if isinstance(state.get("evidence"), dict) else {}
    for row in rows:
        row.reviewed = row.frame in reviewed
        row.edited = row.frame in edited
        evidence = evidence_of.get(str(row.frame))
        if isinstance(evidence, dict) and "source" in evidence and "conf" in evidence:
            try:
                row.source, row.conf = str(evidence["source"]), float(evidence["conf"])
            except (TypeError, ValueError):
                print(f"[warn] {path.name}: frame {row.frame} has unreadable evidence {evidence!r}; ignored")
    return state


# The names label_tool.py and label_source.py still import; one release of grace, then gone.
load_review = apply_review
load_suspects = apply_flags
save_suspects = save_flags


def load_settings(path: Path) -> dict:
    """The toggles worth keeping between sessions. Missing or broken file: the defaults."""
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
        return blob if isinstance(blob, dict) else {}
    except (OSError, ValueError):
        return {}


def save_settings(path: Path, settings: dict) -> None:
    try:
        write_atomic(path, json.dumps(settings, indent=1))
    except OSError as error:
        print(f"[warn] could not save settings to {path}: {error}")
