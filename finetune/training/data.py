"""The trainer's data: which labelled clips take part, the per-clip frame cache of 768x432
JPEGs with their five-frame training units, the targets, the augmentation and the sampler.

Data contract (same as the archive): ``videos/<clip>.mp4`` + ``labels/<clip>_ball.csv``
with ``frame,ball_x,ball_y`` rows at 30 FPS cadence (every 2nd frame of a 60 FPS clip, every
frame at 30 FPS; the first labelled frame may be odd) and invisible balls parked in the
top-right corner. ``clips.csv`` (written by import_data.py) tags every clip with a source so a
run can pick, hold out and oversample by source (``ClipSelection``, ``make_sampler``).
"""

from __future__ import annotations

import fnmatch
import json
import math
import random
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, TypedDict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from tennis_tracker.gridtracknet import FRAMES_PER_UNIT, GRID_COLS, GRID_ROWS, HEIGHT, WIDTH
from tennis_tracker.frame_rate import require_phase_stride
from finetune.data.data_policy import is_trainable, is_verified, read_review_status
from finetune.data.fileio import write_atomic
from finetune.data.labels import LabelPoint, read_labels
from finetune.data.workspace import (MANIFEST_NAME, clip_of, find_video, label_files, read_exclusions, read_manifest,
                                     video_meta)

DATA_VERSION = 4
DEFAULT_VAL_CLIP = "video10"
JPEG_QUALITY = 92          # the frame cache's encoding

Clip = Tuple[str, Path, Path, str]   # (clip, video, label csv, source)
Target = Optional[List[float]]       # a training target: [x, y] on the image, or None for no ball


class TrainingUnit(TypedDict):
    """One training sample: five cached frames at model cadence and their targets."""
    clip: str
    source: str
    frames: List[str]       # cache-relative paths, frames/<clip>/<index>.jpg
    points: List[Target]    # one per frame, in the clip's native pixels
    size: List[int]         # [width, height] of the clip


# --------------------------------------------------------------------------- labels

def training_targets(path: Path, width: int, height: int) -> Dict[int, Target]:
    """The label file as frame -> [x, y] (clamped onto the image) or None, as the targets want it."""
    def target(point: LabelPoint) -> Target:
        return None if point is None else [min(max(point[0], 0.0), width - 1.0), min(max(point[1], 0.0), height - 1.0)]
    return {frame: target(point) for frame, point in read_labels(path, width, height).items()}


def matches(clip: str, patterns: Optional[List[str]]) -> bool:
    return bool(patterns) and any(fnmatch.fnmatchcase(clip, pattern) for pattern in patterns)


@dataclass(frozen=True)
class ClipSelection:
    """Which labelled clips a run trains on, as the trainer's flags say it (the defaults: every
    clip the review policy allows, exclude.txt honoured)."""
    sources: Optional[List[str]] = None         # --sources: only clips with these clips.csv sources
    exclude_clips: Optional[List[str]] = None   # --exclude-clips: stems or globs left out entirely
    include_excluded: bool = False              # --include-excluded: keep excluded/unverified clips too
    cameras: Optional[List[str]] = None         # --cameras: only clips with these camera tags

    def admits(self, clip: str, row: Dict[str, str], source: str, review_status: dict, excluded: Set[str]) -> bool:
        """Whether a labelled clip with this manifest row and source is in the selection."""
        if not self.include_excluded and not is_trainable(clip, source, review_status):
            return False
        if self.sources and source not in self.sources:
            return False
        if self.cameras and (row.get("camera") or "untagged") not in self.cameras:
            return False
        return not (matches(clip, self.exclude_clips) or clip in excluded)


def _excluded_clips(selection: ClipSelection, labels: Path, manifest: Dict[str, Dict[str, str]]) -> Set[str]:
    """The clips exclude.txt keeps out of this selection (none with --include-excluded); also the
    place a --cameras request meets a manifest without camera tags."""
    excluded = set() if selection.include_excluded else set(read_exclusions(labels.parent))
    if excluded:
        print(f"[data] skipping {len(excluded)} clip(s) listed in exclude.txt (--include-excluded to keep them)")
    if selection.cameras and not any(row.get("camera") for row in manifest.values()):
        raise ValueError("--cameras needs camera tags in clips.csv: run  python finetune/ft.py camera  first")
    return excluded


def find_clips(videos: Path, labels: Path, selection: ClipSelection = ClipSelection()) -> List[Clip]:
    """Every (clip, video, label csv, source) the selection admits, in label-file order."""
    manifest = read_manifest(labels.parent / MANIFEST_NAME)
    review_status = read_review_status(labels.parent)
    excluded = _excluded_clips(selection, labels, manifest)
    clips: List[Clip] = []
    for csv_path in label_files(labels):
        clip = clip_of(csv_path)
        row = manifest.get(clip, {})
        source = row.get("source") or ("custom" if not manifest else "unknown")
        if not selection.admits(clip, row, source, review_status, excluded):
            continue
        video = find_video(clip, videos)
        if video is None:
            raise FileNotFoundError(f"No video for {csv_path.name} in {videos}")
        clips.append((clip, video, csv_path, source))
    if not clips:
        raise FileNotFoundError(f"No *_ball.csv files in {labels}"
                                + (f" for sources {selection.sources}" if selection.sources else ""))
    return clips


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


# --------------------------------------------------------------------------- data cache (per clip)

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


def clip_signature(video: Path, csv_path: Path) -> List[List[object]]:
    signature = []
    for path in (csv_path, video):
        stat = path.stat()
        signature.append([path.name, stat.st_size, stat.st_mtime_ns])
    return signature


def prepare_clip(entry: Clip, data_dir: Path, rebuild: bool) -> ClipRecord:
    """Extract the 768x432 frames one clip needs and describe its training units. Cached per clip."""
    clip, video, csv_path, source = entry
    meta_path = data_dir / "clips" / f"{clip}.json"
    signature = clip_signature(video, csv_path)
    if not rebuild and meta_path.is_file():
        try:
            cached = json.loads(meta_path.read_text(encoding="utf-8"))
            if cached.get("version") == DATA_VERSION and cached.get("signature") == signature:
                cached["source"] = source
                return cached
        except (OSError, ValueError) as error:
            # The cache is a derived copy of the clip; a damaged entry is rebuilt below from the
            # video and labels, so this is a note on the extra work, not a failure.
            print(f"[data] {clip}: cache entry unreadable ({error}); rebuilding it")
    fps, width, height, _ = video_meta(video)
    stride = require_phase_stride(fps)
    labels = training_targets(csv_path, width, height)
    samples: List[TrainingUnit] = []
    needed = set()
    for window in build_units(labels, stride):
        samples.append(TrainingUnit(clip=clip, source=source, frames=[f"frames/{clip}/{index:06d}.jpg" for index in window],
                                    points=[labels[index] for index in window], size=[width, height]))
        needed.update(window)
    frame_dir = data_dir / "frames" / clip
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


def make_target(points, width: int, height: int) -> torch.Tensor:
    target = np.zeros((FRAMES_PER_UNIT * 3, GRID_ROWS, GRID_COLS), dtype=np.float32)
    for index, point in enumerate(points):
        if point is None:
            continue
        grid_x = point[0] * GRID_COLS / width
        grid_y = point[1] * GRID_ROWS / height
        col = min(GRID_COLS - 1, int(grid_x))
        row = min(GRID_ROWS - 1, int(grid_y))
        target[index * 3, row, col] = 1
        target[index * 3 + 1, row, col] = grid_x - col
        target[index * 3 + 2, row, col] = grid_y - row
    return torch.from_numpy(target)


class Augmenter:
    """The random draws one data-loading process makes, seeded from torch's seed for that process.

    In a DataLoader worker ``torch.initial_seed()`` is ``base_seed + worker_id``, the very seed torch
    gives that worker's global ``random`` module, and in the main process it is the seed
    ``train`` set; so ``draws`` reproduces the sequence the module-level ``random.*`` calls used to
    make, in both cases. The numpy generator (sensor noise) was never seeded before, so it starts
    from the same seed now and is reproducible for the first time.
    """

    def __init__(self, seed: int) -> None:
        self.draws = random.Random(seed)
        self.noise = np.random.default_rng(seed)


def photometric_augment(images: np.ndarray, rng: Augmenter) -> np.ndarray:
    """Strong lighting/colour augmentation applied identically to all five frames of a window
    (so motion cues survive): gamma (dark night courts and washed-out daylight), contrast, brightness,
    per-channel colour cast, sensor noise and occasional softness. images: (15, H, W) float in [0, 1]."""
    draws = rng.draws
    gamma = math.exp(draws.uniform(math.log(0.45), math.log(2.2)))       # >1 darkens: night-court look
    contrast = draws.uniform(0.6, 1.4)
    brightness = draws.uniform(-0.12, 0.12)
    cast = np.array([draws.uniform(0.85, 1.15) for _ in range(3)], dtype=np.float32)
    out = np.power(np.clip(images, 0.0, 1.0), gamma)
    out = (out - 0.5) * contrast + 0.5 + brightness
    out = out.reshape(FRAMES_PER_UNIT, 3, *out.shape[1:]) * cast[None, :, None, None]
    out = out.reshape(-1, *out.shape[2:])
    if draws.random() < 0.5:
        out = out + rng.noise.normal(0.0, draws.uniform(0.005, 0.03), size=out.shape).astype(np.float32)
    if draws.random() < 0.2:
        sigma = draws.uniform(0.4, 1.0)
        out = np.stack([cv2.GaussianBlur(plane, (0, 0), sigma) for plane in out])
    return np.clip(out, 0.0, 1.0).astype(np.float32)


class UnitDataset(Dataset):
    def __init__(self, data_dir: Path, samples: Sequence[TrainingUnit], augment: bool, strength: str = "basic") -> None:
        self.data_dir, self.samples, self.augment, self.strength = data_dir, samples, augment, strength
        self.rng: Optional[Augmenter] = None     # made in the process that reads, see Augmenter

    def __len__(self) -> int:
        return len(self.samples)

    def augmenter(self) -> Augmenter:
        if self.rng is None:
            self.rng = Augmenter(torch.initial_seed())
        return self.rng

    def read_frames(self, sample: TrainingUnit, flip: bool) -> np.ndarray:
        """The unit's five cached frames as one (15, H, W) float array in [0, 1], RGB, mirrored when `flip`."""
        frames = []
        for relative in sample["frames"]:
            image = cv2.imread(str(self.data_dir / relative), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(self.data_dir / relative)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if flip:
                image = cv2.flip(image, 1)
            frames.append(np.moveaxis(image, -1, 0))
        return np.concatenate(frames).astype(np.float32) / 255.0

    def __getitem__(self, index: int):
        sample = self.samples[index]
        width, height = sample["size"]
        rng = self.augmenter()
        # The draws are made in this order on purpose: it is the sequence every run has made.
        flip = self.augment and rng.draws.random() < 0.5
        jitter = self.augment and self.strength == "basic" and rng.draws.random() < 0.5
        gain = rng.draws.uniform(0.85, 1.15) if jitter else 1.0
        images = self.read_frames(sample, flip)
        points = sample["points"]
        if flip:
            points = [None if point is None else [width - 1 - point[0], point[1]] for point in points]
        if jitter:
            images = np.clip(images * gain, 0.0, 1.0)
        elif self.augment and self.strength == "strong" and rng.draws.random() < 0.8:
            images = photometric_augment(images, rng)
        return torch.from_numpy(np.ascontiguousarray(images)), make_target(points, width, height)


def make_sampler(samples: Sequence[TrainingUnit], oversample: Dict[str, float],
                 epoch_units: Optional[int]) -> Optional[WeightedRandomSampler]:
    """Weighted sampling by source (weight 0 drops a source from training); None = plain shuffle."""
    weights = [float(oversample.get(sample["source"], 1.0)) for sample in samples]
    if epoch_units is None and all(weight == 1.0 for weight in weights):
        return None
    drawable = sum(1 for weight in weights if weight > 0)
    if drawable == 0:
        raise ValueError("Every training unit has sampling weight 0")
    return WeightedRandomSampler(weights, num_samples=epoch_units or drawable, replacement=True)


# --------------------------------------------------------------------------- the validation split

def resolve_val_clips(names: List[str], patterns: Optional[List[str]]) -> List[str]:
    if patterns:
        chosen = [clip for clip in names if matches(clip, patterns)]
        unmatched = [pattern for pattern in patterns if not any(fnmatch.fnmatchcase(clip, pattern) for clip in names)]
        if unmatched:
            raise ValueError(f"Validation pattern(s) match nothing: {unmatched}; have {names[:8]}...")
        return chosen
    return [DEFAULT_VAL_CLIP] if DEFAULT_VAL_CLIP in names else [names[-1]]


def validation_split(clips: List[Clip], labels: Path, patterns: Optional[List[str]]) -> List[str]:
    """The held-out clips: what --val-clips names plus every clip of the same recording group, all
    of which must have reviewed labels, and never the whole set."""
    names = [clip for clip, _, _, _ in clips]
    val_clips = resolve_val_clips(names, patterns)
    metadata = read_manifest(labels.parent / MANIFEST_NAME)
    val_groups = {metadata.get(name, {}).get("group") for name in val_clips} - {None, ""}
    val_clips = [name for name in names if name in val_clips or metadata.get(name, {}).get("group") in val_groups]
    review_status = read_review_status(labels.parent)
    unverified = [name for name in val_clips if not is_verified(name, review_status)]
    if unverified:
        raise ValueError(f"Validation requires reviewed labels, including group members: {unverified}")
    if len(val_clips) == len(names):
        raise ValueError("Group-safe validation leaves no training clips; choose an independent recording group")
    return val_clips
