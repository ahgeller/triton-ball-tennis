"""Fine-tune (or train from scratch) GridTrackNet on the finetune workspace labels.

Data contract (same as the archive): ``videos/<clip>.mp4`` + ``labels/<clip>_ball.csv``
with ``frame,ball_x,ball_y`` rows at 30 FPS cadence (every 2nd frame of a 60 FPS
clip, every frame at 30 FPS; the first labelled frame may be odd) and invisible
balls parked in the top-right corner. ``clips.csv`` (written by import_data.py)
tags every clip with a source so you can pick, hold out and oversample by source.

    python finetune/train_gridtracknet.py                              # fine-tune bundled weights, hold out video10
    python finetune/train_gridtracknet.py --val-clips video10 video11  # own-camera hold-out
    python finetune/train_gridtracknet.py --sources custom             # own footage only
    python finetune/train_gridtracknet.py --oversample custom=6 --epoch-units 20000
    python finetune/train_gridtracknet.py --from-scratch --epochs 30   # train a fresh network on everything
    python finetune/train_gridtracknet.py --self-test

After training, the best-validation-loss checkpoint and the starting weights are
both scored on the validation clips with the detector metric used by
``evaluate_archive.py`` (recall / wrong-object / false-alarm), and the winner is
copied to ``--save`` so the tracker always points at the best model.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import math
import random
import shutil
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, TypedDict

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

if __name__ == "__main__":          # run as a script: make the repository root importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tennis_tracker.gridtracknet import (
    FRAMES_PER_UNIT, GRID_COLS, GRID_ROWS, HEIGHT, WIDTH, GridTrackNet, decode_predictions, frame_tensor, load_model,
)
from tennis_tracker.frame_rate import require_phase_stride
from finetune.data_policy import is_trainable, is_verified, read_review_status
from finetune.fileio import copy_atomic, write_atomic
from finetune.labels import LabelPoint, read_labels
from finetune.scoring import REFERENCE_HEIGHT, scale_for_width, score
from finetune import workspace as ws
from finetune.workspace import MANIFEST_NAME, clip_of, find_video, label_files, read_exclusions, read_manifest
from finetune.workspace import video_meta   # noqa: F401 - looked up by name here (tests patch it)

DATA_VERSION = 4
DEFAULT_VAL_CLIP = "video10"
SEED = 7                   # every run draws the same augmentation and sampling sequence
JPEG_QUALITY = 92          # the frame cache's encoding
PROGRESS_EVERY = 200       # training steps between progress lines
LOSS_EPSILON = 1e-6        # keeps the confidence loss's logs finite
OFFSET_LOSS_WEIGHT = 0.001
POSITIVE_WEIGHT, NEGATIVE_WEIGHT = 0.75, 0.25    # the focal loss's class weights
METRIC_SLACK = 0.005       # a candidate may be worse by this on wrong/false-alarm rate: 1 frame in 200
WRONG_PENALTY = 2.0        # ... and each unit of extra wrong-object rate costs this much recall

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


def find_clips(videos: Path, labels: Path, sources: Optional[List[str]] = None,
               exclude_clips: Optional[List[str]] = None, include_excluded: bool = False,
               cameras: Optional[List[str]] = None) -> List[Clip]:
    manifest = read_manifest(labels.parent / MANIFEST_NAME)
    review_status = read_review_status(labels.parent)
    excluded = set() if include_excluded else set(read_exclusions(labels.parent))
    if excluded:
        print(f"[data] skipping {len(excluded)} clip(s) listed in exclude.txt (--include-excluded to keep them)")
    if cameras and not any(row.get("camera") for row in manifest.values()):
        raise ValueError("--cameras needs camera tags in clips.csv: run  python finetune/ft.py camera  first")
    clips: List[Clip] = []
    for csv_path in label_files(labels):
        clip = clip_of(csv_path)
        row = manifest.get(clip, {})
        source = row.get("source") or ("custom" if not manifest else "unknown")
        if not include_excluded and not is_trainable(clip, source, review_status):
            continue
        if sources and source not in sources:
            continue
        if cameras and (row.get("camera") or "untagged") not in cameras:
            continue
        if matches(clip, exclude_clips) or clip in excluded:
            continue
        video = find_video(clip, videos)
        if video is None:
            raise FileNotFoundError(f"No video for {csv_path.name} in {videos}")
        clips.append((clip, video, csv_path, source))
    if not clips:
        raise FileNotFoundError(f"No *_ball.csv files in {labels}" + (f" for sources {sources}" if sources else ""))
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

    def __getitem__(self, index: int):
        sample = self.samples[index]
        width, height = sample["size"]
        rng = self.augmenter()
        flip = self.augment and rng.draws.random() < 0.5
        jitter = self.augment and self.strength == "basic" and rng.draws.random() < 0.5
        gain = rng.draws.uniform(0.85, 1.15) if jitter else 1.0
        frames = []
        for relative in sample["frames"]:
            image = cv2.imread(str(self.data_dir / relative), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(self.data_dir / relative)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if flip:
                image = cv2.flip(image, 1)
            frames.append(np.moveaxis(image, -1, 0))
        points = sample["points"]
        if flip:
            points = [None if point is None else [width - 1 - point[0], point[1]] for point in points]
        images = np.concatenate(frames).astype(np.float32) / 255.0
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


# --------------------------------------------------------------------------- loss / train

def gridtracknet_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = prediction.float().reshape(-1, FRAMES_PER_UNIT, 3, GRID_ROWS, GRID_COLS)
    target = target.reshape_as(prediction)
    confidence_true = target[:, :, 0]
    confidence_pred = prediction[:, :, 0].clamp(LOSS_EPSILON, 1 - LOSS_EPSILON)
    positive = POSITIVE_WEIGHT * confidence_true * (1 - confidence_pred).square() * confidence_pred.log()
    negative = NEGATIVE_WEIGHT * (1 - confidence_true) * confidence_pred.square() * (1 - confidence_pred).log()
    confidence_loss = -(positive + negative).mean()
    offset_loss = ((prediction[:, :, 1:3] - target[:, :, 1:3]).abs() * confidence_true.unsqueeze(2)).sum(dim=(2, 3, 4)).mean()
    return confidence_loss + OFFSET_LOSS_WEIGHT * offset_loss


def save_weights(model: torch.nn.Module, template: Path, destination: Path) -> None:
    """Write the model in the .npz layout the tracker loads (template supplies BN stats)."""
    with np.load(template) as source:
        arrays = {name: source[name] for name in source.files}
    for index, conv in enumerate(model.convs):
        name = "conv2d" if index == 0 else f"conv2d_{index}"
        prefix = f"{name}/{name}"
        arrays[f"{prefix}/kernel:0"] = conv.weight.detach().float().cpu().numpy().transpose(2, 3, 1, 0)
        arrays[f"{prefix}/bias:0"] = conv.bias.detach().float().cpu().numpy()
    for index, layer in enumerate(model.batch_norms):
        name = "batch_normalization" if index == 0 else f"batch_normalization_{index}"
        prefix = f"{name}/{name}"
        arrays[f"{prefix}/gamma:0"] = layer.gamma.detach().float().cpu().numpy()
        arrays[f"{prefix}/beta:0"] = layer.beta.detach().float().cpu().numpy()
        arrays[f"{prefix}/moving_mean:0"] = layer.mean.detach().float().cpu().numpy()
        arrays[f"{prefix}/moving_variance:0"] = layer.variance.detach().float().cpu().numpy()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(destination)


def fresh_model(device: torch.device) -> GridTrackNet:
    model = GridTrackNet()
    for conv in model.convs:
        torch.nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
        torch.nn.init.zeros_(conv.bias)
    return model.to(device).float()


def evaluate_loss(model, loader, device, max_steps=None) -> float:
    model.eval()
    total, steps = 0.0, 0
    with torch.inference_mode():
        for images, target in loader:
            images, target = images.to(device, non_blocking=True), target.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = gridtracknet_loss(model(images), target)
            total += float(loss)
            steps += 1
            if max_steps and steps >= max_steps:
                break
    return total / max(steps, 1)


# --------------------------------------------------------------------------- detector metric

def detector_metrics(weights: Path, clips: List[Clip], device: torch.device, threshold: float = 0.5) -> Dict[str, float]:
    """Raw-detector recall / wrong-object / false-alarm on whole clips (the evaluate_archive metric)."""
    model = load_model(weights, device)
    pairs = []
    with torch.inference_mode():
        for clip, video, csv_path, _ in clips:
            fps, width, height, total = video_meta(video)
            labels = training_targets(csv_path, width, height)
            scale = scale_for_width(width)
            predictions = predict_clip(model, video, require_phase_stride(fps), width, height, device, threshold)
            for frame, label in labels.items():
                point = predictions.get(frame)
                pairs.append((None if label is None else (label[0] * scale, label[1] * scale),
                              None if point is None else (point[0] * scale, point[1] * scale)))
    stats = score(pairs, 1.0)
    stats["wrong"] = stats.pop("wrong_obj")
    return stats


def predict_clip(model, video: Path, stride: int, width: int, height: int, device: torch.device,
                 threshold: float) -> Dict[int, Optional[Tuple[float, float]]]:
    """frame -> the model's point (None for a miss) over a whole clip, read in five-frame units per
    phase the way the tracker's prepass does, the tail of each phase inferred from its last five."""
    from tennis_tracker.config import Config
    capture = cv2.VideoCapture(str(video))
    units = [[] for _ in range(stride)]
    recent = [deque(maxlen=FRAMES_PER_UNIT) for _ in range(stride)]
    predictions: Dict[int, Optional[Tuple[float, float]]] = {}
    y_offset = Config().gridtracknet_y_offset_px * height / REFERENCE_HEIGHT

    def infer(unit):
        output = model(torch.cat([t for t, _ in unit]).unsqueeze(0))
        for (_, frame_index), (point, _) in zip(unit, decode_predictions(output, width, height, threshold)):
            # A finalized miss is immutable, just like a detection.
            if frame_index not in predictions:
                predictions[frame_index] = None if point is None else (point[0], point[1] + y_offset)
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            phase = index % stride
            unit = units[phase]
            item = (frame_tensor(frame, device), index)
            unit.append(item)
            recent[phase].append(item)
            if len(unit) == FRAMES_PER_UNIT:
                infer(unit)
                unit.clear()
            index += 1
        for phase in range(stride):
            if units[phase] and len(recent[phase]) == FRAMES_PER_UNIT:
                infer(recent[phase])
    finally:
        capture.release()
    return predictions


def better(candidate: Dict[str, float], incumbent: Dict[str, float]) -> bool:
    """Best = more correct ball with no more wrong-object outputs (within METRIC_SLACK, 1 frame in 200)."""
    if candidate["wrong"] > incumbent["wrong"] + METRIC_SLACK:
        return False
    if candidate.get("false_alarm", 0.0) > incumbent.get("false_alarm", 0.0) + METRIC_SLACK:
        return False
    return candidate["recall"] - WRONG_PENALTY * max(0.0, candidate["wrong"] - incumbent["wrong"]) > incumbent["recall"]


def parse_oversample(items: Optional[List[str]]) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    for item in items or []:
        name, _, value = item.partition("=")
        if not value:
            raise ValueError(f"--oversample expects source=weight, got {item!r}")
        weights[name.strip()] = float(value)
    return weights


def resolve_val_clips(names: List[str], patterns: Optional[List[str]]) -> List[str]:
    if patterns:
        chosen = [clip for clip in names if matches(clip, patterns)]
        unmatched = [pattern for pattern in patterns if not any(fnmatch.fnmatchcase(clip, pattern) for clip in names)]
        if unmatched:
            raise ValueError(f"Validation pattern(s) match nothing: {unmatched}; have {names[:8]}...")
        return chosen
    return [DEFAULT_VAL_CLIP] if DEFAULT_VAL_CLIP in names else [names[-1]]


# --------------------------------------------------------------------------- main

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


def train_epoch(model, loader, optimizer, scaler, device: torch.device, epoch: int, epochs: int,
                max_steps: Optional[int]) -> float:
    """One pass over the training loader; returns the mean loss."""
    model.train()
    total, steps = 0.0, 0
    started = time.perf_counter()
    for images, target in loader:
        images, target = images.to(device, non_blocking=True), target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            loss = gridtracknet_loss(model(images), target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += float(loss)
        steps += 1
        if steps == 1 or steps % PROGRESS_EVERY == 0:
            rate = steps / max(time.perf_counter() - started, 1e-6)
            remaining = (len(loader) - steps) / max(rate, 1e-6)
            print(f"[train] epoch {epoch}/{epochs} step {steps}/{len(loader)} "
                  f"loss={total / steps:.6f} {rate:.1f} it/s eta {remaining / 60:.0f} min", flush=True)
        if max_steps and steps >= max_steps:
            break
    return total / max(steps, 1)


def pick_winner(args: argparse.Namespace, clips: List[Clip], val_clips: List[str], best_path: Path,
                device: torch.device) -> None:
    """Keep the best model by the metric that matters, not just the loss: the trained weights win
    only when they beat the starting weights on the held-out clips (see `better`)."""
    val_only = [entry for entry in clips if entry[0] in val_clips]
    trained = detector_metrics(best_path, val_only, device)
    print(f"[eval] trained best.npz on {val_clips}: {trained}")
    winner, winner_stats = best_path, trained
    if not args.from_scratch:
        baseline = detector_metrics(args.weights, val_only, device)
        print(f"[eval] starting weights on {val_clips}: {baseline}")
        if not better(trained, baseline):
            winner, winner_stats = args.weights, baseline
    copy_atomic(winner, args.save)
    write_atomic(args.run_dir / "winner.json", json.dumps({"winner": str(winner), "metrics": winner_stats,
                                                           "validation_clips": val_clips}, indent=2))
    print(f"[done] best model -> {args.save}  (from {winner.name}: {winner_stats})")
    if winner == best_path:
        print("[done] promote it with: python finetune/ft.py promote")
    else:
        print("[done] the starting weights stayed ahead on the held-out clips; nothing to promote")


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(SEED)       # the data loaders' draws derive from it too, see Augmenter
    clips = find_clips(args.videos, args.labels, args.sources, args.exclude_clips, args.include_excluded, args.cameras)
    names = [clip for clip, _, _, _ in clips]
    val_clips = validation_split(clips, args.labels, args.val_clips)
    oversample = parse_oversample(args.oversample)
    manifest = prepare_data(clips, args.data_dir, val_clips, args.rebuild_data, args.prep_workers)
    if args.prepare_only:
        print("[data] cache ready; --prepare-only stops here")
        return
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    sampler = make_sampler(manifest["train"], oversample, args.epoch_units)
    train_loader = DataLoader(UnitDataset(args.data_dir, manifest["train"], augment=True, strength=args.augment),
                              batch_size=args.batch_size,
                              shuffle=sampler is None, sampler=sampler, num_workers=args.workers,
                              pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    val_loader = DataLoader(UnitDataset(args.data_dir, manifest["val"], augment=False), batch_size=args.batch_size,
                            num_workers=args.workers, pin_memory=device.type == "cuda")
    if args.from_scratch:
        model = fresh_model(device)
        print("[train] fresh GridTrackNet (random init)")
    else:
        model = load_model(args.weights, device).float()
        print(f"[train] fine-tuning from {args.weights}")
    model.train()
    optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr, rho=0.95, eps=1e-7)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    results = args.run_dir / "results.csv"
    write_atomic(results, "epoch,train_loss,val_loss,seconds\n")
    write_atomic(args.run_dir / "config.json", json.dumps({
        "clips": names, "validation_clips": val_clips, "sources": args.sources, "cameras": args.cameras,
        "augment": args.augment, "oversample": oversample,
        "epoch_units": args.epoch_units, "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
        "from_scratch": args.from_scratch, "weights": str(args.weights)}, indent=2))
    best_path = args.run_dir / "best.npz"
    best = math.inf
    print(f"[train] device={device} epochs={args.epochs} batch={args.batch_size} "
          f"units/epoch={len(train_loader) * args.batch_size} val={val_clips}"
          + (f" oversample={oversample}" if oversample else ""))
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device, epoch, args.epochs, args.max_train_steps)
        val_loss = evaluate_loss(model, val_loader, device, args.max_val_steps)
        save_weights(model, args.weights, args.run_dir / "last.npz")
        if val_loss < best:
            best = val_loss
            save_weights(model, args.weights, best_path)
        seconds = time.perf_counter() - started
        with results.open("a", encoding="utf-8", newline="") as handle:    # one line per epoch, appended
            csv.writer(handle).writerow([epoch, f"{train_loss:.8f}", f"{val_loss:.8f}", f"{seconds:.0f}"])
        print(f"[train] epoch {epoch}: train={train_loss:.6f} val={val_loss:.6f} best={best:.6f} ({seconds / 60:.1f} min)",
              flush=True)
    pick_winner(args, clips, val_clips, best_path, device)


def expect(condition: bool, what: str) -> None:
    """A self-test check that survives ``python -O`` (an assert would not)."""
    if not condition:
        raise RuntimeError(f"train_gridtracknet self-test failed: {what}")


def self_test() -> None:
    from finetune.labels import is_visible
    expect(not is_visible(1910, 8, 1920, 1080) and is_visible(736, 508, 1920, 1080), "the invisible-ball corner")
    target = make_target([[960, 540], None, None, None, None], 1920, 1080)
    expect(bool(target[0, 13, 24] == 1), "the target cell of a centred ball")
    prediction = torch.full((1, 15, GRID_ROWS, GRID_COLS), 0.1, requires_grad=True)
    loss = gridtracknet_loss(prediction, target.unsqueeze(0))
    loss.backward()
    expect(bool(torch.isfinite(loss) and torch.isfinite(prediction.grad).all()), "a finite loss and gradient")
    expect(better({"recall": 0.90, "wrong": 0.01}, {"recall": 0.85, "wrong": 0.01}), "more recall at equal wrong wins")
    expect(not better({"recall": 0.95, "wrong": 0.05}, {"recall": 0.85, "wrong": 0.01}), "more wrong loses")
    even = {index: [1.0, 1.0] for index in range(0, 20, 2)}
    expect(build_units(even, 2)[:2] == [[0, 2, 4, 6, 8], [4, 6, 8, 10, 12]], "even-phase units")
    odd = {index: None for index in range(1, 20, 2)}
    expect(build_units(odd, 2)[:2] == [[1, 3, 5, 7, 9], [5, 7, 9, 11, 13]], "odd-phase units")
    gapped = {index: None for index in range(0, 12) if index != 5}
    expect(all(5 not in window for window in build_units(gapped, 1)), "no unit across a label gap")
    expect(parse_oversample(["custom=4", "grid=0.5"]) == {"custom": 4.0, "grid": 0.5}, "--oversample parsing")
    sampler = make_sampler([{"source": "a"}, {"source": "b"}], {"b": 0.0}, None)
    expect(sampler is not None and set(iter(sampler)) == {0}, "weight 0 drops a source")
    expect(make_sampler([{"source": "a"}], {}, None) is None, "plain shuffle without weights")
    expect(resolve_val_clips(["video1", "video10", "tnv2_Test_match1_1"], ["tnv2_Test_*"]) == ["tnv2_Test_match1_1"],
           "validation globs")
    expect(resolve_val_clips(["video1", "video10"], None) == ["video10"], "the default validation clip")
    augmented = photometric_augment(np.full((15, 8, 8), 0.5, dtype=np.float32), Augmenter(1))
    expect(augmented.shape == (15, 8, 8) and augmented.dtype == np.float32
           and 0.0 <= augmented.min() <= augmented.max() <= 1.0, "augmentation keeps shape and range")
    model = fresh_model(torch.device("cpu"))
    expect(model(torch.zeros(1, 15, HEIGHT, WIDTH)).shape == (1, 15, GRID_ROWS, GRID_COLS), "the fresh model's output")
    print("train_gridtracknet self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos", type=Path, default=ws.VIDEOS)
    parser.add_argument("--labels", type=Path, default=ws.LABELS)
    parser.add_argument("--weights", type=Path, default=ws.REPO_WEIGHTS,
                        help="Starting weights (also the .npz template for saving)")
    parser.add_argument("--save", type=Path, default=ws.BEST_WEIGHTS)
    parser.add_argument("--data-dir", type=Path, default=ws.CACHE / "data")
    parser.add_argument("--run-dir", type=Path, default=ws.RUNS / "train")
    parser.add_argument("--val-clips", nargs="*", help="Held-out clip stems or globs (default: video10 if present, else the last clip)")
    parser.add_argument("--sources", nargs="*", help="Only use clips whose clips.csv source is listed (e.g. custom grid)")
    parser.add_argument("--exclude-clips", nargs="*", help="Clip stems or globs to leave out entirely")
    parser.add_argument("--include-excluded", action="store_true", help="Explicitly include excluded/unverified clips for training (validation still requires verified labels)")
    parser.add_argument("--cameras", nargs="*", choices=("fixed", "moving", "unknown", "untagged"),
                        help="Only clips with these clips.csv camera tags (from  ft.py camera), e.g. --cameras fixed")
    parser.add_argument("--augment", choices=("basic", "strong"), default="basic",
                        help="basic = flip + small gain (as before); strong = + gamma/contrast/colour cast/noise/blur "
                             "for lighting diversity (dark courts)")
    parser.add_argument("--oversample", nargs="*", metavar="SOURCE=WEIGHT",
                        help="Relative sampling weight per source, e.g. custom=6 grid=1 tracknetv2=0.5 (0 drops it from training)")
    parser.add_argument("--epoch-units", type=int, help="Units drawn per epoch (default: every training unit once)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1.0, help="Adadelta lr (1.0 = upstream; 0.3 for gentle fine-tunes)")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--prep-workers", type=int, default=4, help="Parallel clips while extracting the frame cache")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--from-scratch", action="store_true", help="Train a new network instead of fine-tuning")
    parser.add_argument("--rebuild-data", action="store_true")
    parser.add_argument("--prepare-only", action="store_true", help="Build/refresh the frame cache and stop")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--max-train-steps", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--max-val-steps", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    train(args)


if __name__ == "__main__":
    main()
