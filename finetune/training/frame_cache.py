"""The trainer's per-clip frame cache: the 768x432 JPEGs of every frame a clip's training units
need, and the record (``clips/<clip>.json``) that lists those units with their targets.

A clip's cache entry is rebuilt when its video or label file changes (``clip_signature``) or the
record format does (``DATA_VERSION``); ``prepare_data`` builds every clip's entry and splits the
units into the training and validation sets.
"""

from __future__ import annotations

import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Set, TypedDict

import cv2

from tennis_tracker.gridtracknet import FRAMES_PER_UNIT, HEIGHT, WIDTH
from tennis_tracker.frame_rate import require_phase_stride
from finetune.data.fileio import write_atomic
from finetune.data.labels import LabelPoint, read_labels
from finetune.data.workspace import video_meta
from finetune.training.clips import Clip

DATA_VERSION = 4
JPEG_QUALITY = 92          # the frame cache's encoding

Target = Optional[List[float]]       # a training target: [x, y] on the image, or None for no ball


class TrainingUnit(TypedDict):
    """One training sample: five cached frames at model cadence and their targets."""
    clip: str
    source: str
    frames: List[str]       # cache-relative paths, frames/<clip>/<index>.jpg
    points: List[Target]    # one per frame, in the clip's native pixels
    size: List[int]         # [width, height] of the clip


class ClipRecord(TypedDict):
    """One clip's cache entry: ``prepare_clip``'s record, also stored as ``clips/<clip>.json``."""
    version: int
    signature: List[List[object]]
    clip: str
    source: str
    fps: float
    stride: int
    size: List[int]
    samples: List[TrainingUnit]
    visible: int


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

def clip_signature(video: Path, csv_path: Path) -> List[List[object]]:
    signature = []
    for path in (csv_path, video):
        stat = path.stat()
        signature.append([path.name, stat.st_size, stat.st_mtime_ns])
    return signature


def cached_record(meta_path: Path, signature: List[List[object]]) -> Optional[ClipRecord]:
    """The record at ``meta_path`` when it was built from these very files by this format version."""
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
    """Write the frames ``needed`` of ``video`` to ``frame_dir`` as 768x432 JPEGs, replacing what
    was there; raises ValueError when the video ends before the last of them."""
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True)
    capture = cv2.VideoCapture(str(video))
    remaining = set(needed)
    frame_index = 0
    while remaining:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index in remaining:
            resized = cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(frame_dir / f"{frame_index:06d}.jpg"), resized, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            remaining.remove(frame_index)
        frame_index += 1
    capture.release()
    if remaining:
        raise ValueError(f"{video.name} ended before frames {sorted(remaining)[:5]}")


def prepare_clip(entry: Clip, data_dir: Path, rebuild: bool) -> ClipRecord:
    """Extract the 768x432 frames one clip needs and describe its training units. Cached per clip."""
    clip, video, csv_path, source = entry
    meta_path = data_dir / "clips" / f"{clip}.json"
    signature = clip_signature(video, csv_path)
    if not rebuild:
        cached = cached_record(meta_path, signature)
        if cached is not None:
            cached["source"] = source
            return cached
    fps, width, height, _ = video_meta(video)
    stride = require_phase_stride(fps)
    labels = training_targets(csv_path, width, height)
    samples: List[TrainingUnit] = []
    needed = set()
    for window in build_units(labels, stride):
        samples.append(TrainingUnit(clip=clip, source=source, frames=[f"frames/{clip}/{index:06d}.jpg" for index in window],
                                    points=[labels[index] for index in window], size=[width, height]))
        needed.update(window)
    extract_frames(video, data_dir / "frames" / clip, needed)
    record = {"version": DATA_VERSION, "signature": signature, "clip": clip, "source": source,
              "fps": fps, "stride": stride, "size": [width, height], "samples": samples,
              "visible": sum(point is not None for sample in samples for point in sample["points"])}
    write_atomic(meta_path, json.dumps(record))
    return record


def prepare_data(clips: List[Clip], data_dir: Path, val_clips: List[str], rebuild: bool,
                 workers: int) -> Dict[str, list]:
    data_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        records: List[ClipRecord] = list(pool.map(lambda entry: prepare_clip(entry, data_dir, rebuild), clips))
    manifest = {"train": [], "val": [], "validation_clips": val_clips}
    per_source: Dict[str, Dict[str, int]] = {}
    for record in records:
        split = "val" if record["clip"] in val_clips else "train"
        manifest[split].extend(record["samples"])
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
