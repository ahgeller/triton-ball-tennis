"""Where the workspace keeps things, its records, and the facts about a clip's video.

The workspace is laid out as ``videos/<clip>.<ext>`` + ``labels/<clip>_ball.csv`` (plus the
sidecars described in ``finetune.labels``), with ``clips.csv`` listing every imported clip,
``exclude.txt`` the clips training skips and ``review_status.json`` the review policy. Every
script used to spell these paths out for itself; this is the one place they are defined, and
the one place ``clips.csv`` and ``exclude.txt`` are read and written.

The workspace is the ``finetune/`` folder itself unless ``TENNIS_WORKSPACE`` names another
folder. Modules that need the paths read them from this module when called
(``workspace.LABELS``), not at import, so a test can point the whole workspace at a
temporary folder by patching this module alone (``tests/conftest.py: workspace_dir``).
"""
from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Union

from finetune.fileio import atomic_csv, write_atomic
from tennis_tracker.paths import GRIDTRACKNET_WEIGHTS

WORKSPACE_VARIABLE = "TENNIS_WORKSPACE"       # names another workspace folder, for tests and spare copies
WORKSPACE = Path(os.environ.get(WORKSPACE_VARIABLE) or Path(__file__).resolve().parent).resolve()
ROOT = Path(__file__).resolve().parents[1]
VIDEOS = WORKSPACE / "videos"
LABELS = WORKSPACE / "labels"
MODELS = WORKSPACE / "models"
RUNS = WORKSPACE / "runs"
CACHE = WORKSPACE / "cache"
MANIFEST_NAME = "clips.csv"
EXCLUDE_NAME = "exclude.txt"
REVIEW_STATUS_NAME = "review_status.json"
MANIFEST = WORKSPACE / MANIFEST_NAME
EXCLUDE_FILE = WORKSPACE / EXCLUDE_NAME
REVIEW_STATUS = WORKSPACE / REVIEW_STATUS_NAME
REPO_WEIGHTS = GRIDTRACKNET_WEIGHTS                                  # the weights the tracker runs
BEST_WEIGHTS = MODELS / "gridtracknet_best.npz"                      # the last training winner
VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".mkv", ".avi", ".m4v"})

LABEL_SUFFIX = "_ball.csv"
DRAFT_SUFFIX = "_ball.csv.draft"
REVIEW_SUFFIX = "_ball.review.json"
SUSPECT_SUFFIX = "_ball.suspect.json"

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


class VideoMeta(NamedTuple):
    fps: float
    width: int
    height: int
    total: int


def video_meta(path: Path) -> VideoMeta:
    """Frame rate, size and frame count of a video. Raises RuntimeError when it cannot be read:
    a capture that failed to open answers zeros for everything, and a 0x0 frame would make every
    label read as the invisible-ball sentinel."""
    import cv2   # imported here so ft.py stays importable without cv2 (see ft.ensure_interpreter)
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


def label_csv(clip: str, labels: Optional[Path] = None) -> Path:
    return (LABELS if labels is None else labels) / f"{clip}{LABEL_SUFFIX}"


def draft_csv(clip: str, labels: Optional[Path] = None) -> Path:
    return (LABELS if labels is None else labels) / f"{clip}{DRAFT_SUFFIX}"


def review_json(clip: str, labels: Optional[Path] = None) -> Path:
    return (LABELS if labels is None else labels) / f"{clip}{REVIEW_SUFFIX}"


def suspect_json(clip: str, labels: Optional[Path] = None) -> Path:
    return (LABELS if labels is None else labels) / f"{clip}{SUSPECT_SUFFIX}"


def label_files(labels: Optional[Path] = None) -> List[Path]:
    """Every final label file, in natural clip order."""
    labels = LABELS if labels is None else labels
    if not labels.is_dir():
        return []
    return sorted(labels.glob(f"*{LABEL_SUFFIX}"), key=lambda path: natural_key(path.stem))


def labelled_clips(labels: Optional[Path] = None) -> List[str]:
    return [clip_of(path) for path in label_files(labels)]


def find_video(clip: str, videos: Optional[Path] = None) -> Optional[Path]:
    """The clip's video in `videos`, whatever its container, or None."""
    videos = VIDEOS if videos is None else videos
    return next((p for p in sorted(videos.glob(f"{clip}.*")) if p.suffix.lower() in VIDEO_SUFFIXES), None)


def require_video(clip: str, videos: Optional[Path] = None) -> Path:
    """``find_video``, or a FileNotFoundError that says where it looked."""
    video = find_video(clip, videos)
    if video is None:
        raise FileNotFoundError(f"No video named {clip}.* in {VIDEOS if videos is None else videos}")
    return video


def videos_in(videos: Optional[Path] = None) -> Dict[str, Path]:
    """clip -> video for every video in the folder."""
    videos = VIDEOS if videos is None else videos
    if not videos.is_dir():
        return {}
    return {p.stem: p for p in sorted(videos.iterdir()) if p.suffix.lower() in VIDEO_SUFFIXES}


# --------------------------------------------------------------------------- clips.csv

def read_manifest(path: Optional[Path] = None) -> Dict[str, dict]:
    """clip -> clips.csv row (every MANIFEST_COLUMNS key, missing ones empty); empty when no
    manifest has been written."""
    path = MANIFEST if path is None else path
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["clip"]: {key: row.get(key) or "" for key in MANIFEST_COLUMNS} for row in csv.DictReader(handle)}


def write_manifest(rows: Dict[str, dict], path: Optional[Path] = None) -> None:
    """Write clip -> row as clips.csv, atomically, in natural clip order."""
    with atomic_csv(MANIFEST if path is None else path) as writer:
        writer.writerow(MANIFEST_COLUMNS)
        for clip in sorted(rows, key=natural_key):
            writer.writerow([rows[clip].get(key, "") for key in MANIFEST_COLUMNS])


# --------------------------------------------------------------------------- exclude.txt

def read_exclusions(workspace: Optional[Path] = None) -> List[str]:
    """Clip stems listed in the workspace's exclude.txt (a ``#`` starts a comment)."""
    path = EXCLUDE_FILE if workspace is None else workspace / EXCLUDE_NAME
    if not path.is_file():
        return []
    return [line.split("#", 1)[0].strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.split("#", 1)[0].strip()]


def add_exclusion(clip: str, reason: str, workspace: Optional[Path] = None) -> None:
    """Append `clip` to exclude.txt with `reason` as its comment; a clip already listed is left alone."""
    if clip in read_exclusions(workspace):
        return
    path = EXCLUDE_FILE if workspace is None else workspace / EXCLUDE_NAME
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    if not lines:
        lines = [EXCLUDE_HEADER]
    lines.append(f"{clip}  # {reason}")
    write_atomic(path, "\n".join(lines) + "\n")
