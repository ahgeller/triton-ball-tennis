"""Where the workspace keeps things, and the facts about a clip's video.

``finetune/`` is laid out as ``videos/<clip>.<ext>`` + ``labels/<clip>_ball.csv`` (plus the
sidecars described in ``finetune.labels``), with ``clips.csv`` listing every imported clip,
``exclude.txt`` the clips training skips and ``review_status.json`` the review policy. Every
script used to spell these paths out for itself; this is the one place they are defined.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Union

WORKSPACE = Path(__file__).resolve().parent
ROOT = WORKSPACE.parent
VIDEOS = WORKSPACE / "videos"
LABELS = WORKSPACE / "labels"
MODELS = WORKSPACE / "models"
RUNS = WORKSPACE / "runs"
CACHE = WORKSPACE / "cache"
MANIFEST = WORKSPACE / "clips.csv"
EXCLUDE_FILE = WORKSPACE / "exclude.txt"
REPO_WEIGHTS = ROOT / "models" / "gridtracknet_weights_torch.npz"   # the weights the tracker runs
BEST_WEIGHTS = MODELS / "gridtracknet_best.npz"                      # the last training winner
VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".mkv", ".avi", ".m4v"})

LABEL_SUFFIX = "_ball.csv"
DRAFT_SUFFIX = "_ball.csv.draft"
REVIEW_SUFFIX = "_ball.review.json"
SUSPECT_SUFFIX = "_ball.suspect.json"


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


def clip_of(label_csv: Path) -> str:
    """The clip stem of a ``<clip>_ball.csv`` (or ``.draft``) path."""
    return label_csv.name.split(LABEL_SUFFIX, 1)[0]


def label_csv(clip: str, labels: Path = LABELS) -> Path:
    return labels / f"{clip}{LABEL_SUFFIX}"


def draft_csv(clip: str, labels: Path = LABELS) -> Path:
    return labels / f"{clip}{DRAFT_SUFFIX}"


def review_json(clip: str, labels: Path = LABELS) -> Path:
    return labels / f"{clip}{REVIEW_SUFFIX}"


def suspect_json(clip: str, labels: Path = LABELS) -> Path:
    return labels / f"{clip}{SUSPECT_SUFFIX}"


def label_files(labels: Path = LABELS) -> List[Path]:
    """Every final label file, in natural clip order."""
    if not labels.is_dir():
        return []
    return sorted(labels.glob(f"*{LABEL_SUFFIX}"), key=lambda path: natural_key(path.stem))


def labelled_clips(labels: Path = LABELS) -> List[str]:
    return [clip_of(path) for path in label_files(labels)]


def find_video(clip: str, videos: Path = VIDEOS) -> Optional[Path]:
    """The clip's video in `videos`, whatever its container, or None."""
    return next((p for p in sorted(videos.glob(f"{clip}.*")) if p.suffix.lower() in VIDEO_SUFFIXES), None)


def videos_in(videos: Path = VIDEOS) -> Dict[str, Path]:
    """clip -> video for every video in the folder."""
    if not videos.is_dir():
        return {}
    return {p.stem: p for p in sorted(videos.iterdir()) if p.suffix.lower() in VIDEO_SUFFIXES}


def read_manifest(path: Path = MANIFEST) -> Dict[str, dict]:
    """clip -> clips.csv row (source, camera, size, ...); empty when no manifest has been written."""
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["clip"]: row for row in csv.DictReader(handle)}
