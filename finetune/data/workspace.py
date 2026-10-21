"""Where the workspace keeps things, its records, and the facts about a clip's video.

The workspace is laid out as ``videos/<clip>.<ext>`` + ``labels/<clip>_ball.csv`` (plus the
sidecars described in ``finetune.data.labels``), with ``clips.csv`` listing every imported clip,
``exclude.txt`` the clips training skips and ``review_status.json`` the review policy. Every
script used to spell these paths out for itself; this is the one place they are defined, and
the one place ``clips.csv`` and ``exclude.txt`` are read and written.

The layout is a ``Workspace``: one root folder, every other path a property derived from it.
``current()`` builds it when asked, from ``TENNIS_WORKSPACE`` as the environment has it at that
moment (else the ``finetune/`` folder this package sits in), so nothing here is fixed at import.
The contract for every module that needs a path:

- read it when called, never at import: ``ws.current().labels`` (a function that needs several
  binds ``layout = ws.current()`` once and reads them from that);
- never bind a path to a module constant (``MODELS = ws.current().models / "x"`` snapshots the
  workspace of import time);
- a test points the whole workspace somewhere else by replacing ``current`` alone
  (``tests/conftest.py: workspace_dir``); there is no second thing to patch.

The repository itself is not a workspace path: ``tennis_tracker.paths.ROOT`` names it. ``REPO_WEIGHTS``
(the weights the tracker runs, in the repository's ``models/``) is not under the workspace either, but
promotion writes it, so ``Workspace.repo_weights`` carries it and a test redirects it with the rest.
"""
from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Union

from finetune.data.fileio import atomic_csv, write_atomic
from tennis_tracker.paths import GRIDTRACKNET_WEIGHTS

WORKSPACE_VARIABLE = "TENNIS_WORKSPACE"       # names another workspace folder, for tests and spare copies
DEFAULT_WORKSPACE = Path(__file__).resolve().parents[1]     # the finetune/ folder itself
MANIFEST_NAME = "clips.csv"
EXCLUDE_NAME = "exclude.txt"
REVIEW_STATUS_NAME = "review_status.json"
BEST_WEIGHTS_NAME = "gridtracknet_best.npz"   # the last training winner, under models/
VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".mkv", ".avi", ".m4v"})

LABEL_SUFFIX = "_ball.csv"
DRAFT_SUFFIX = "_ball.csv.draft"
REVIEW_SUFFIX = "_ball.review.json"
SUSPECT_SUFFIX = "_ball.suspect.json"
UNSAVED_SUFFIX = "_ball.unsaved.csv"      # the label tool's rescue copy when the label file cannot be written

MANIFEST_COLUMNS = ("clip", "source", "section", "group", "width", "height", "fps", "stride",
                    "label_frames", "visible", "camera", "motion_px", "origin", "note")
EXCLUDE_HEADER = "# clips the trainer skips by default (ft.py check --audit --fix writes these; edit freely)"

# Where import_data.py finds each dataset on the machine the workspace was built on.
DESKTOP = Path(r"C:\Users\Andrew\Desktop")
DEFAULT_SOURCES = {
    "archive": DESKTOP / "gridtracknet_finetuning" / "archive",
    "video11": DESKTOP / "V5Test" / "held_back_custom_768" / "sources" / "custom" / "video11",
    "tracknetv2": DESKTOP / "V5Test" / "archive" / "TrackNetV2" / "TrackNetV2",
    "grid": DESKTOP / "V5Test" / "archive" / "Match-Data-20260716T144245Z-1-001" / "Match-Data",
    "prof": DESKTOP / "prof" / "Dataset",
}


# --------------------------------------------------------------------------- the layout

@dataclass(frozen=True)
class Workspace:
    """One workspace folder and where it keeps things. Every path is derived from ``root`` when
    asked, so a ``Workspace`` is the whole layout in one value."""
    root: Path
    repo_weights: Path = GRIDTRACKNET_WEIGHTS    # the weights the tracker runs: not under root, promoted to from it

    @property
    def videos(self) -> Path:
        return self.root / "videos"

    @property
    def labels(self) -> Path:
        return self.root / "labels"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def cache(self) -> Path:
        return self.root / "cache"

    @property
    def manifest(self) -> Path:
        return self.root / MANIFEST_NAME

    @property
    def exclude_file(self) -> Path:
        return self.root / EXCLUDE_NAME

    @property
    def review_status(self) -> Path:
        return self.root / REVIEW_STATUS_NAME

    @property
    def best_weights(self) -> Path:
        return self.models / BEST_WEIGHTS_NAME


def current() -> Workspace:
    """The workspace as the environment names it now: ``TENNIS_WORKSPACE``, else ``finetune/``.
    Read on every call; tests replace this function to relocate everything at once."""
    override = os.environ.get(WORKSPACE_VARIABLE)
    return Workspace(Path(override).resolve() if override else DEFAULT_WORKSPACE)


class VideoMeta(NamedTuple):
    fps: float
    width: int
    height: int
    total: int


def video_meta(path: Path) -> VideoMeta:
    """Frame rate, size and frame count of a video. Raises RuntimeError when it cannot be read:
    a capture that failed to open answers zeros for everything, and a 0x0 frame would make every
    label read as the invisible-ball sentinel."""
    import cv2   # imported here so ft.py stays importable without cv2 (see finetune.interpreter.ensure_interpreter)
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open {path}")
        meta = VideoMeta(float(capture.get(cv2.CAP_PROP_FPS)), int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                         int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    finally:
        capture.release()
    if meta.width <= 0 or meta.height <= 0:
        raise RuntimeError(f"{path} could not be decoded (codec not supported, or the file is broken)")
    return meta


def natural_key(text: str) -> List[Union[int, str]]:
    """Sort key that puts video2 before video10."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


# --------------------------------------------------------------------------- a clip's files

def clip_of(label_csv: Path) -> str:
    """The clip stem of a ``<clip>_ball.csv`` (or ``.draft``) path."""
    return label_csv.name.split(LABEL_SUFFIX, 1)[0]


def _labels_dir(labels: Optional[Path]) -> Path:
    return current().labels if labels is None else labels


def _videos_dir(videos: Optional[Path]) -> Path:
    return current().videos if videos is None else videos


def label_csv(clip: str, labels: Optional[Path] = None) -> Path:
    return _labels_dir(labels) / f"{clip}{LABEL_SUFFIX}"


def draft_csv(clip: str, labels: Optional[Path] = None) -> Path:
    return _labels_dir(labels) / f"{clip}{DRAFT_SUFFIX}"


def review_json(clip: str, labels: Optional[Path] = None) -> Path:
    return _labels_dir(labels) / f"{clip}{REVIEW_SUFFIX}"


def suspect_json(clip: str, labels: Optional[Path] = None) -> Path:
    return _labels_dir(labels) / f"{clip}{SUSPECT_SUFFIX}"


def unsaved_csv(clip: str, labels: Optional[Path] = None) -> Path:
    return _labels_dir(labels) / f"{clip}{UNSAVED_SUFFIX}"


def _files_ending(suffix: str, labels: Optional[Path]) -> List[Path]:
    labels = _labels_dir(labels)
    if not labels.is_dir():
        return []
    return sorted(labels.glob(f"*{suffix}"), key=lambda path: natural_key(clip_of(path)))


def label_files(labels: Optional[Path] = None) -> List[Path]:
    """Every final label file, in natural clip order."""
    return _files_ending(LABEL_SUFFIX, labels)


def draft_files(labels: Optional[Path] = None) -> List[Path]:
    """Every draft (``pretrack.py``'s guesses, not yet corrected), in natural clip order."""
    return _files_ending(DRAFT_SUFFIX, labels)


def labelled_clips(labels: Optional[Path] = None) -> List[str]:
    return [clip_of(path) for path in label_files(labels)]


def find_video(clip: str, videos: Optional[Path] = None) -> Optional[Path]:
    """The clip's video in `videos`, whatever its container, or None."""
    videos = _videos_dir(videos)
    return next((p for p in sorted(videos.glob(f"{clip}.*")) if p.suffix.lower() in VIDEO_SUFFIXES), None)


def require_video(clip: str, videos: Optional[Path] = None) -> Path:
    """``find_video``, or a FileNotFoundError that says where it looked."""
    video = find_video(clip, videos)
    if video is None:
        raise FileNotFoundError(f"No video named {clip}.* in {_videos_dir(videos)}")
    return video


def videos_in(videos: Optional[Path] = None) -> Dict[str, Path]:
    """clip -> video for every video in the folder."""
    videos = _videos_dir(videos)
    if not videos.is_dir():
        return {}
    return {p.stem: p for p in sorted(videos.iterdir()) if p.suffix.lower() in VIDEO_SUFFIXES}


# --------------------------------------------------------------------------- clips.csv

def read_manifest(path: Optional[Path] = None) -> Dict[str, dict]:
    """clip -> clips.csv row (every MANIFEST_COLUMNS key, missing ones empty); empty when no
    manifest has been written."""
    path = current().manifest if path is None else path
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["clip"]: {key: row.get(key) or "" for key in MANIFEST_COLUMNS} for row in csv.DictReader(handle)}


_RECORDING_PART = re.compile(r"frames|Clip\d+", re.IGNORECASE)   # one recording's frame folder or clip folder


def recording_of(clip: str, manifest: Optional[Dict[str, dict]] = None) -> str:
    """The recording a clip was cut from, for keeping one camera's footage on one side of a split.

    Its clips.csv group when that names a shared one (an importer's default group is the clip's
    own name and says nothing); else its origin with a trailing ``frames`` or ``ClipN`` folder
    dropped, so the three rallies cut from one video (grid_match47, 49 and 50) and the clips of
    one TrackNet game share a recording; else the clip itself.
    """
    row = (read_manifest() if manifest is None else manifest).get(clip)
    if row is None:
        return clip
    group = row.get("group") or ""
    if group and group != clip:
        return group
    parts = [part for part in re.split(r"[\\/]", row.get("origin") or "") if part]
    if parts and _RECORDING_PART.fullmatch(parts[-1]):
        parts.pop()
    return "/".join(parts) or clip


def write_manifest(rows: Dict[str, dict], path: Optional[Path] = None) -> None:
    """Write clip -> row as clips.csv, atomically, in natural clip order."""
    with atomic_csv(current().manifest if path is None else path) as writer:
        writer.writerow(MANIFEST_COLUMNS)
        for clip in sorted(rows, key=natural_key):
            writer.writerow([rows[clip].get(key, "") for key in MANIFEST_COLUMNS])


# --------------------------------------------------------------------------- exclude.txt

def _exclude_file(workspace: Optional[Path]) -> Path:
    return current().exclude_file if workspace is None else workspace / EXCLUDE_NAME


def read_exclusions(workspace: Optional[Path] = None) -> List[str]:
    """Clip stems listed in the workspace's exclude.txt (a ``#`` starts a comment)."""
    path = _exclude_file(workspace)
    if not path.is_file():
        return []
    return [line.split("#", 1)[0].strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.split("#", 1)[0].strip()]


def add_exclusion(clip: str, reason: str, workspace: Optional[Path] = None) -> None:
    """Append `clip` to exclude.txt with `reason` as its comment; a clip already listed is left alone."""
    if clip in read_exclusions(workspace):
        return
    path = _exclude_file(workspace)
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    if not lines:
        lines = [EXCLUDE_HEADER]
    lines.append(f"{clip}  # {reason}")
    write_atomic(path, "\n".join(lines) + "\n")
