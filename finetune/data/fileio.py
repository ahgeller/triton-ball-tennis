"""Writing workspace files so a crash, a power cut or a locked file never leaves a torn one.

Every label CSV, sidecar JSON, manifest and settings file in the workspace, and every weights
file the tracker runs, goes through here: the content is written to a temporary file beside the
target, flushed to disk, and then moved over the target, so the target is always either the old
file or the complete new one. Windows refuses the move while another program (Excel, an editor,
a virus scan) has the target open, so the move is retried for a moment before the error is
raised.
"""
from __future__ import annotations

import csv
import datetime as dt
import os
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

REPLACE_ATTEMPTS = 5       # tries before a locked target is given up on ...
REPLACE_PAUSE = 0.1        # ... and the seconds waited between them


def replace_file(temporary: Path, path: Path) -> None:
    """Move a finished temporary file over `path`, retrying a Windows sharing lock briefly."""
    for attempt in range(REPLACE_ATTEMPTS):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(REPLACE_PAUSE)


def temporary_for(path: Path) -> Path:
    return path.with_name(path.name + ".tmp")


@contextmanager
def atomic_text(path: Path, newline: str = "") -> Iterator:
    """A text handle whose content lands at `path` only once it is complete and on disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_for(path)
    with temporary.open("w", encoding="utf-8", newline=newline) as handle:
        yield handle
        handle.flush()
        os.fsync(handle.fileno())
    replace_file(temporary, path)


def write_atomic(path: Path, text: str) -> None:
    """Write `text` to `path` through a temporary file, flushed to disk first."""
    with atomic_text(path) as handle:
        handle.write(text)


@contextmanager
def atomic_csv(path: Path) -> Iterator[csv.writer]:
    """A csv.writer whose rows land at `path` only once every row is written and on disk."""
    with atomic_text(path, newline="") as handle:
        yield csv.writer(handle)


def copy_atomic(source: Path, path: Path) -> None:
    """Copy `source` over `path` through a temporary file beside it, flushed to disk first, so a
    reader of `path` sees the old file or the complete copy and nothing in between."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_for(path)
    with source.open("rb") as reader, temporary.open("wb") as writer:
        shutil.copyfileobj(reader, writer)
        writer.flush()
        os.fsync(writer.fileno())
    replace_file(temporary, path)


def backup_copy(path: Path, tag: str) -> Path:
    """Copy `path` to `<name>.before_<tag>_<timestamp>` beside it and return the copy's path. Every
    bulk write over a workspace file (an import, a shift, a realignment, a promotion) makes one."""
    stamp = f"{dt.datetime.now():%Y%m%d_%H%M%S_%f}"
    backup = path.with_name(f"{path.name}.before_{tag}_{stamp}")
    shutil.copyfile(path, backup)
    return backup
