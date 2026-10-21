"""The trainer's per-clip frame cache: the 768x432 frames a clip's training units need, stored
losslessly as PNGs through the tracker's own resize (``resize_to_model``), and the record
(``clips/<clip>.json``) that says which frames are on disk and lists the units with their targets.

A clip's cached frames are keyed on its video alone (``clip_signature``) and the record format
(``DATA_VERSION``): the units are derived from the labels on every run, and ``prepare_clip``
extracts only the frames the units need that are not on disk yet, so a label edit costs the
new frames and never a second decode of the clip. ``prepare_data`` builds every clip's entry and
splits the units into the training and validation sets.
"""

from __future__ import annotations

import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Set, TypedDict

import cv2

from tennis_tracker.gridtracknet import FRAMES_PER_UNIT, resize_to_model
from tennis_tracker.frame_rate import require_phase_stride
from finetune.data.fileio import write_atomic
from finetune.data.labels import LabelPoint, read_labels
from finetune.data.workspace import video_meta
from finetune.training.clips import Clip

DATA_VERSION = 5
PNG_COMPRESSION = 1        # the frame cache's encoding: lossless, and the fastest zlib level to write and read
FRAME_SUFFIX = ".png"

Target = Optional[List[float]]       # a training target: [x, y] on the image, or None for no ball


class TrainingUnit(TypedDict):
    """One training sample: five cached frames at model cadence and their targets."""
    clip: str
    source: str
    indices: List[int]      # the five source frame indices, at model cadence
    frames: List[str]       # cache-relative paths, frames/<clip>/<index>.png
    points: List[Target]    # one per frame, in the clip's native pixels
    size: List[int]         # [width, height] of the clip


class ClipRecord(TypedDict):
    """One clip's cache entry: ``prepare_clip``'s record, also stored as ``clips/<clip>.json``."""
    version: int
    signature: List[object]
    clip: str
    source: str
    fps: float
    stride: int
    size: List[int]
    frames: List[int]       # the source frame indices on disk under frames/<clip>/
    samples: List[TrainingUnit]
    visible: int


class DataManifest(TypedDict):
    """What ``prepare_data`` hands the trainer: the units of each split and the held-out clips."""
    train: List[TrainingUnit]
    val: List[TrainingUnit]
    validation_clips: List[str]


# --------------------------------------------------------------------------- labels to units

def training_targets(path: Path, width: int, height: int) -> Dict[int, Target]:
    """The label file as frame -> [x, y] (clamped onto the image) or None, as the targets want it."""
    def target(point: LabelPoint) -> Target:
        return None if point is None else [min(max(point[0], 0.0), width - 1.0), min(max(point[1], 0.0), height - 1.0)]
    return {frame: target(point) for frame, point in read_labels(path, width, height).items()}


def build_units(labels: Dict[int, Optional[List[float]]], stride: int) -> List[List[int]]:
    """5-frame windows at model cadence over the labelled frames, advancing two label frames per unit.

    Works for labels that start on frame 0 or 1 and for clips with gaps: a window is only
    emitted when all five of its frames are labelled.
    """
    indices = sorted(labels)
    present = set(indices)
    units = []
    for position in range(0, len(indices), 2):
        start = indices[position]
        window = [start + offset * stride for offset in range(FRAMES_PER_UNIT)]
        if all(index in present for index in window):
            units.append(window)
    return units


# --------------------------------------------------------------------------- the cache (per clip)

def frame_name(index: int) -> str:
    """The file a cached source frame is stored as, under frames/<clip>/."""
    return f"{index:06d}{FRAME_SUFFIX}"


def clip_signature(video: Path) -> List[object]:
    """What the cached frames of a clip were made from: the video file, by name, size and
    modification time. The labels are not part of it; they only say which frames are needed."""
    stat = video.stat()
    return [video.name, stat.st_size, stat.st_mtime_ns]


def cached_record(meta_path: Path, signature: List[object]) -> Optional[ClipRecord]:
    """The record at ``meta_path`` when it was built from this very video by this format version."""
    if not meta_path.is_file():
        return None
    try:
        cached = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        # The cache is a derived copy of the clip; a damaged entry is rebuilt by the caller from
        # the video and labels, so this is a note on the extra work, not a failure.
        print(f"[data] {meta_path.stem}: cache entry unreadable ({error}); rebuilding it")
        return None
    if cached.get("version") == DATA_VERSION and cached.get("signature") == signature:
        return cached
    return None


def extract_frames(video: Path, frame_dir: Path, needed: Set[int]) -> None:
    """Write the frames ``needed`` of ``video`` to ``frame_dir`` as lossless 768x432 PNGs through
    the tracker's resize, beside whatever is there; raises ValueError when the video ends before
    the last of them and OSError when a PNG cannot be written (so no cache record ever points at
    a frame that is not there)."""
    frame_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    remaining = set(needed)
    frame_index = 0
    while remaining:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index in remaining:
            target = frame_dir / frame_name(frame_index)
            if not cv2.imwrite(str(target), resize_to_model(frame), [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION]):
                capture.release()
                raise OSError(f"could not write frame {frame_index} of {video.name} to {target}")
            remaining.remove(frame_index)
        frame_index += 1
    capture.release()
    if remaining:
        raise ValueError(f"{video.name} ended before frames {sorted(remaining)[:5]}")


def prepare_clip(entry: Clip, data_dir: Path, rebuild: bool) -> ClipRecord:
    """Describe one clip's training units from its labels as they are now, and extract the
    768x432 frames they need that the cache does not hold yet (every frame with ``rebuild``)."""
    clip, video, csv_path, source = entry
    meta_path = data_dir / "clips" / f"{clip}.json"
    frame_dir = data_dir / "frames" / clip
    signature = clip_signature(video)
    cached = None if rebuild else cached_record(meta_path, signature)
    if rebuild and frame_dir.exists():
        shutil.rmtree(frame_dir)
    on_disk = {index for index in (cached or {"frames": []})["frames"] if (frame_dir / frame_name(index)).is_file()}
    fps, width, height, _ = video_meta(video)
    stride = require_phase_stride(fps)
    labels = training_targets(csv_path, width, height)
    samples: List[TrainingUnit] = []
    needed: Set[int] = set()
    for window in build_units(labels, stride):
        samples.append(TrainingUnit(clip=clip, source=source, indices=list(window),
                                    frames=[f"frames/{clip}/{frame_name(index)}" for index in window],
                                    points=[labels[index] for index in window], size=[width, height]))
        needed.update(window)
    missing = needed - on_disk
    if missing:
        extract_frames(video, frame_dir, missing)
    record = ClipRecord(version=DATA_VERSION, signature=signature, clip=clip, source=source, fps=fps, stride=stride,
                        size=[width, height], frames=sorted(on_disk | needed), samples=samples,
                        visible=sum(point is not None for sample in samples for point in sample["points"]))
    if cached != record:
        write_atomic(meta_path, json.dumps(record))
    return record


def prepare_data(clips: List[Clip], data_dir: Path, val_clips: List[str], rebuild: bool,
                 workers: int) -> DataManifest:
    """Every clip's cache entry, built or refreshed, with its units split into the training and
    validation sets."""
    data_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        records: List[ClipRecord] = list(pool.map(lambda entry: prepare_clip(entry, data_dir, rebuild), clips))
    manifest = DataManifest(train=[], val=[], validation_clips=val_clips)
    per_source: Dict[str, Dict[str, int]] = {}
    for record in records:
        split = "val" if record["clip"] in val_clips else "train"
        (manifest["val"] if split == "val" else manifest["train"]).extend(record["samples"])
        bucket = per_source.setdefault(record["source"], {"train": 0, "val": 0, "clips": 0})
        bucket[split] += len(record["samples"])
        bucket["clips"] += 1
    print(f"[data] {len(records)} clips ready in {time.perf_counter() - started:.0f}s "
          f"({len(manifest['train'])} train / {len(manifest['val'])} val units)")
    for source, bucket in sorted(per_source.items()):
        print(f"[data]   {source:20s} clips={bucket['clips']:4d} train units={bucket['train']:6d} val units={bucket['val']:6d}")
    if not manifest["train"] or not manifest["val"]:
        raise ValueError("Need labelled clips in both the training and validation split")
    return manifest
