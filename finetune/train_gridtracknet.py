"""Fine-tune (or train from scratch) GridTrackNet on the finetune workspace labels.

Data contract (same as the archive): ``videos/<clip>.mp4`` + ``labels/<clip>_ball.csv``
with ``frame,ball_x,ball_y`` rows at 30 FPS cadence (every 2nd frame of a 60 FPS
clip) and invisible balls parked in the top-right corner.

    python finetune/train_gridtracknet.py                       # fine-tune bundled weights, keep best
    python finetune/train_gridtracknet.py --from-scratch        # train a fresh network
    python finetune/train_gridtracknet.py --val-clips rally7    # choose the held-out clip(s)
    python finetune/train_gridtracknet.py --self-test

After training, the best-validation-loss checkpoint and the starting weights are
both scored on the validation clips with the detector metric used by
``evaluate_archive.py`` (recall / wrong-object / false-alarm), and the winner is
copied to ``--save`` so the tracker always points at the best model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tennis_tracker.gridtracknet import (  # noqa: E402
    FRAMES_PER_UNIT, GRID_COLS, GRID_ROWS, HEIGHT, WIDTH, GridTrackNet, decode_predictions, frame_tensor, load_model,
)

WORKSPACE = Path(__file__).resolve().parent
DATA_VERSION = 3
HIT_PX = 10.0
WRONG_PX = 30.0


# --------------------------------------------------------------------------- labels

def natural_key(path: Path) -> list:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.stem)]


def is_visible(x: float, y: float, width: int, height: int) -> bool:
    return not (x >= width * 0.95 and y <= height * 0.05)


def read_labels(path: Path, width: int, height: int) -> Dict[int, Optional[List[float]]]:
    labels: Dict[int, Optional[List[float]]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not {"frame", "ball_x", "ball_y"}.issubset(reader.fieldnames or ()):
            raise ValueError(f"{path} must contain frame, ball_x, and ball_y columns")
        for row in reader:
            match = re.fullmatch(r"frame_(\d+)", row["frame"].strip())
            if not match:
                raise ValueError(f"Bad frame value in {path}: {row['frame']!r}")
            index = int(match.group(1))
            x, y = float(row["ball_x"]), float(row["ball_y"])
            if is_visible(x, y, width, height):
                labels[index] = [min(max(x, 0.0), width - 1.0), min(max(y, 0.0), height - 1.0)]
            else:
                labels[index] = None
    return labels


def find_clips(videos: Path, labels: Path) -> List[Tuple[str, Path, Path]]:
    clips = []
    for csv_path in sorted(labels.glob("*_ball.csv"), key=natural_key):
        clip = csv_path.stem[: -len("_ball")]
        candidates = [p for p in videos.glob(f"{clip}.*") if p.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi", ".m4v"}]
        if not candidates:
            raise FileNotFoundError(f"No video for {csv_path.name} in {videos}")
        clips.append((clip, candidates[0], csv_path))
    if not clips:
        raise FileNotFoundError(f"No *_ball.csv files in {labels}")
    return clips


def video_meta(path: Path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {path}")
    meta = (
        float(capture.get(cv2.CAP_PROP_FPS)),
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    capture.release()
    return meta


def source_stride(fps: float) -> int:
    if 57 <= fps <= 62:
        return 2
    if 22 <= fps <= 32:
        return 1
    raise ValueError(f"{fps:.3f} FPS; expected 30 or 60 FPS")


# --------------------------------------------------------------------------- data cache

def prepare_data(clips, data_dir: Path, val_clips: List[str], rebuild: bool) -> dict:
    signature = []
    for clip, video, csv_path in clips:
        for path in (csv_path, video):
            stat = path.stat()
            signature.append([path.name, stat.st_size, stat.st_mtime_ns])
    manifest_path = data_dir / "manifest.json"
    if not rebuild and manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest.get("version") == DATA_VERSION
                and manifest.get("source") == signature
                and manifest.get("validation_clips") == val_clips
            ):
                print(f"[data] reusing {data_dir}")
                return manifest
        except (OSError, ValueError):
            pass

    building = data_dir.with_name(f"{data_dir.name}.building")
    if building.exists():
        shutil.rmtree(building)
    building.mkdir(parents=True)
    manifest = {"version": DATA_VERSION, "source": signature, "validation_clips": val_clips, "train": [], "val": []}
    for clip, video, csv_path in clips:
        fps, width, height, _ = video_meta(video)
        stride = source_stride(fps)
        labels = read_labels(csv_path, width, height)
        max_index = max(labels)
        samples, needed = [], set()
        for start in range(0, max_index - 4 * stride + 1, 2 * stride):
            indices = [start + offset * stride for offset in range(FRAMES_PER_UNIT)]
            if not all(index in labels for index in indices):
                continue
            samples.append({
                "clip": clip,
                "frames": [f"frames/{clip}/{index:06d}.jpg" for index in indices],
                "points": [labels[index] for index in indices],
                "size": [width, height],
            })
            needed.update(indices)
        frame_dir = building / "frames" / clip
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
                cv2.imwrite(str(frame_dir / f"{frame_index:06d}.jpg"), resized, [cv2.IMWRITE_JPEG_QUALITY, 92])
                remaining.remove(frame_index)
            frame_index += 1
        capture.release()
        if remaining:
            raise ValueError(f"{video.name} ended before frames {sorted(remaining)[:5]}")
        split = "val" if clip in val_clips else "train"
        manifest[split].extend(samples)
        visible = sum(point is not None for sample in samples for point in sample["points"])
        print(f"[data] {clip}: {len(samples)} {split} units, {visible}/{len(samples) * 5} visible labels")
    if not manifest["train"] or not manifest["val"]:
        raise ValueError("Need labelled clips in both the training and validation split")
    (building / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    if data_dir.exists():
        shutil.rmtree(data_dir)
    building.replace(data_dir)
    print(f"[data] prepared {len(manifest['train'])} train and {len(manifest['val'])} validation units")
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


class UnitDataset(Dataset):
    def __init__(self, data_dir: Path, samples: list, augment: bool) -> None:
        self.data_dir, self.samples, self.augment = data_dir, samples, augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        width, height = sample["size"]
        flip = self.augment and random.random() < 0.5
        jitter = self.augment and random.random() < 0.5
        gain = random.uniform(0.85, 1.15) if jitter else 1.0
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
        return torch.from_numpy(images), make_target(points, width, height)


# --------------------------------------------------------------------------- loss / train

def gridtracknet_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = prediction.float().reshape(-1, FRAMES_PER_UNIT, 3, GRID_ROWS, GRID_COLS)
    target = target.reshape_as(prediction)
    confidence_true = target[:, :, 0]
    confidence_pred = prediction[:, :, 0].clamp(1e-6, 1 - 1e-6)
    positive = 0.75 * confidence_true * (1 - confidence_pred).square() * confidence_pred.log()
    negative = 0.25 * (1 - confidence_true) * confidence_pred.square() * (1 - confidence_pred).log()
    confidence_loss = -(positive + negative).mean()
    offset_loss = ((prediction[:, :, 1:3] - target[:, :, 1:3]).abs() * confidence_true.unsqueeze(2)).sum(dim=(2, 3, 4)).mean()
    return confidence_loss + 0.001 * offset_loss


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

def detector_metrics(weights: Path, clips, device: torch.device, threshold: float = 0.5) -> Dict[str, float]:
    """Raw-detector recall / wrong-object / false-alarm on whole clips (the evaluate_archive metric)."""
    model = load_model(weights, device)
    hit = wrong = miss = false_alarm = quiet = 0
    with torch.inference_mode():
        for clip, video, csv_path in clips:
            fps, width, height, total = video_meta(video)
            stride = source_stride(fps)
            labels = read_labels(csv_path, width, height)
            scale = 1920.0 / width
            capture = cv2.VideoCapture(str(video))
            units = [[] for _ in range(stride)]
            predictions: Dict[int, Optional[Tuple[float, float]]] = {}
            index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                unit = units[index % stride]
                unit.append((frame_tensor(frame, device), index))
                if len(unit) == FRAMES_PER_UNIT:
                    output = model(torch.cat([t for t, _ in unit]).unsqueeze(0))
                    for (_, frame_index), (point, _) in zip(unit, decode_predictions(output, width, height, threshold)):
                        predictions[frame_index] = point
                    unit.clear()
                index += 1
            capture.release()
            for frame, label in labels.items():
                point = predictions.get(frame)
                if label is None:
                    if point is None:
                        quiet += 1
                    else:
                        false_alarm += 1
                elif point is None:
                    miss += 1
                else:
                    error = math.hypot(point[0] - label[0], point[1] - label[1]) * scale
                    if error <= HIT_PX:
                        hit += 1
                    elif error > WRONG_PX:
                        wrong += 1
    visible = hit + wrong + miss
    return {
        "recall": hit / max(visible, 1),
        "wrong": wrong / max(visible, 1),
        "false_alarm": false_alarm / max(false_alarm + quiet, 1),
        "visible": visible,
        "invisible": false_alarm + quiet,
    }


def better(candidate: Dict[str, float], incumbent: Dict[str, float]) -> bool:
    """Best = more correct ball with no more wrong-object outputs (within 1 frame in 200)."""
    if candidate["wrong"] > incumbent["wrong"] + 0.005:
        return False
    return candidate["recall"] - 2.0 * max(0.0, candidate["wrong"] - incumbent["wrong"]) > incumbent["recall"]


# --------------------------------------------------------------------------- main

def train(args: argparse.Namespace) -> None:
    random.seed(7)
    torch.manual_seed(7)
    clips = find_clips(args.videos, args.labels)
    names = [clip for clip, _, _ in clips]
    val_clips = args.val_clips or [names[-1]]
    unknown = [clip for clip in val_clips if clip not in names]
    if unknown:
        raise ValueError(f"Unknown validation clip(s): {unknown}; have {names}")
    manifest = prepare_data(clips, args.data_dir, val_clips, args.rebuild_data)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    train_loader = DataLoader(UnitDataset(args.data_dir, manifest["train"], augment=True), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers, pin_memory=device.type == "cuda")
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
    results.write_text("epoch,train_loss,val_loss\n", encoding="utf-8")
    best_path = args.run_dir / "best.npz"
    best = math.inf
    print(f"[train] device={device} epochs={args.epochs} batch={args.batch_size} val={val_clips}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, steps = 0.0, 0
        for images, target in train_loader:
            images, target = images.to(device, non_blocking=True), target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = gridtracknet_loss(model(images), target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += float(loss)
            steps += 1
            if steps == 1 or steps % 50 == 0:
                print(f"[train] epoch {epoch}/{args.epochs} step {steps}/{len(train_loader)} loss={total / steps:.6f}", flush=True)
            if args.max_train_steps and steps >= args.max_train_steps:
                break
        train_loss = total / max(steps, 1)
        val_loss = evaluate_loss(model, val_loader, device, args.max_val_steps)
        save_weights(model, args.weights, args.run_dir / "last.npz")
        if val_loss < best:
            best = val_loss
            save_weights(model, args.weights, best_path)
        with results.open("a", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow([epoch, f"{train_loss:.8f}", f"{val_loss:.8f}"])
        print(f"[train] epoch {epoch}: train={train_loss:.6f} val={val_loss:.6f} best={best:.6f}", flush=True)

    # Keep the best model by the metric that matters, not just the loss.
    val_only = [entry for entry in clips if entry[0] in val_clips]
    trained = detector_metrics(best_path, val_only, device)
    print(f"[eval] trained best.npz on {val_clips}: {trained}")
    winner, winner_stats = best_path, trained
    if not args.from_scratch:
        baseline = detector_metrics(args.weights, val_only, device)
        print(f"[eval] starting weights on {val_clips}: {baseline}")
        if not better(trained, baseline):
            winner, winner_stats = args.weights, baseline
    args.save.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(winner, args.save)
    (args.run_dir / "winner.json").write_text(json.dumps({"winner": str(winner), "metrics": winner_stats,
                                                          "validation_clips": val_clips}, indent=2), encoding="utf-8")
    print(f"[done] best model -> {args.save}  (from {winner.name}: {winner_stats})")


def self_test() -> None:
    assert not is_visible(1910, 8, 1920, 1080) and is_visible(736, 508, 1920, 1080)
    target = make_target([[960, 540], None, None, None, None], 1920, 1080)
    assert target[0, 13, 24] == 1
    prediction = torch.full((1, 15, GRID_ROWS, GRID_COLS), 0.1, requires_grad=True)
    loss = gridtracknet_loss(prediction, target.unsqueeze(0))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(prediction.grad).all()
    assert better({"recall": 0.90, "wrong": 0.01}, {"recall": 0.85, "wrong": 0.01})
    assert not better({"recall": 0.95, "wrong": 0.05}, {"recall": 0.85, "wrong": 0.01})
    model = fresh_model(torch.device("cpu"))
    assert model(torch.zeros(1, 15, HEIGHT, WIDTH)).shape == (1, 15, GRID_ROWS, GRID_COLS)
    print("train_gridtracknet self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos", type=Path, default=WORKSPACE / "videos")
    parser.add_argument("--labels", type=Path, default=WORKSPACE / "labels")
    parser.add_argument("--weights", type=Path, default=ROOT / "models" / "gridtracknet_weights_torch.npz",
                        help="Starting weights (also the .npz template for saving)")
    parser.add_argument("--save", type=Path, default=WORKSPACE / "models" / "gridtracknet_best.npz")
    parser.add_argument("--data-dir", type=Path, default=WORKSPACE / "cache" / "data")
    parser.add_argument("--run-dir", type=Path, default=WORKSPACE / "runs" / "train")
    parser.add_argument("--val-clips", nargs="*", help="Held-out clip stems (default: the last clip)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1.0, help="Adadelta lr (1.0 = upstream; 0.3 for gentle fine-tunes)")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--from-scratch", action="store_true", help="Train a new network instead of fine-tuning")
    parser.add_argument("--rebuild-data", action="store_true")
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
