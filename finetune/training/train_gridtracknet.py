"""Fine-tune (or train from scratch) GridTrackNet on the finetune workspace labels.

Data contract (same as the archive): ``videos/<clip>.mp4`` + ``labels/<clip>_ball.csv``
with ``frame,ball_x,ball_y`` rows at 30 FPS cadence (every 2nd frame of a 60 FPS
clip, every frame at 30 FPS; the first labelled frame may be odd) and invisible
balls parked in the top-right corner. ``clips.csv`` (written by import_data.py)
tags every clip with a source so you can pick, hold out and oversample by source.

    python finetune/training/train_gridtracknet.py                              # fine-tune bundled weights, hold out video10
    python finetune/training/train_gridtracknet.py --val-clips video10 video11  # own-camera hold-out
    python finetune/training/train_gridtracknet.py --sources custom             # own footage only
    python finetune/training/train_gridtracknet.py --oversample custom=6 --epoch-units 20000
    python finetune/training/train_gridtracknet.py --from-scratch --epochs 30   # train a fresh network on everything
    python finetune/training/train_gridtracknet.py --self-test

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
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

if __name__ == "__main__":          # run as a script: make the repository root importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tennis_tracker.gridtracknet import GRID_COLS, GRID_ROWS, HEIGHT, WIDTH, load_model
from finetune.data.detector_metrics import DetectorMetrics
from finetune.data.fileio import copy_atomic, write_atomic
from finetune.data import workspace as ws
from finetune.training.clips import Clip, ClipSelection, find_clips, resolve_val_clips, validation_split
from finetune.training.dataset import Augmenter, UnitDataset, make_sampler, make_target, photometric_augment
from finetune.training.frame_cache import build_units, prepare_data
from finetune.training.loop import evaluate_loss, fresh_model, gridtracknet_loss, save_weights, train_epoch
from finetune.training.metrics import promotion_improves, score_weights

SEED = 7                   # every run draws the same augmentation and sampling sequence

# --------------------------------------------------------------------------- the run

def parse_oversample(items: Optional[List[str]]) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    for item in items or []:
        name, _, value = item.partition("=")
        if not value:
            raise ValueError(f"--oversample expects source=weight, got {item!r}")
        weights[name.strip()] = float(value)
    return weights


def pick_winner(args: argparse.Namespace, clips: List[Clip], val_clips: List[str], best_path: Path,
                device: torch.device) -> None:
    """Keep the best model by the metric that matters, not just the loss: the trained weights win
    only when they beat the starting weights on the held-out clips (see `promotion_improves`)."""
    val_only = [entry for entry in clips if entry[0] in val_clips]
    trained = score_weights(best_path, val_only, device)
    print(f"[eval] trained best.npz on {val_clips}: {trained}")
    winner, winner_stats = best_path, trained
    if not args.from_scratch:
        baseline = score_weights(args.weights, val_only, device)
        print(f"[eval] starting weights on {val_clips}: {baseline}")
        if not promotion_improves(trained, baseline):
            winner, winner_stats = args.weights, baseline
    copy_atomic(winner, args.save)
    write_atomic(args.run_dir / "winner.json", json.dumps({"winner": str(winner), "metrics": winner_stats._asdict(),
                                                           "validation_clips": val_clips}, indent=2))
    print(f"[done] best model -> {args.save}  (from {winner.name}: {winner_stats})")
    if winner == best_path:
        print("[done] promote it with: python finetune/ft.py promote")
    else:
        print("[done] the starting weights stayed ahead on the held-out clips; nothing to promote")


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(SEED)       # the data loaders' draws derive from it too, see Augmenter
    clips = find_clips(args.videos, args.labels, ClipSelection(args.sources, args.exclude_clips, args.include_excluded, args.cameras))
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


def metrics_of(recall: float, wrong: float, false_alarm: float = 0.0) -> DetectorMetrics:
    """A DetectorMetrics with the three rates the winner rule reads, for the self-test."""
    return DetectorMetrics(visible=100, recall=recall, wrong=wrong, miss=0.0, invisible=100, false_alarm=false_alarm,
                           median_px=1.0, p90_px=2.0)


def self_test() -> None:
    from finetune.data.labels import is_visible
    expect(not is_visible(1910, 8, 1920, 1080) and is_visible(736, 508, 1920, 1080), "the invisible-ball corner")
    target = make_target([[960, 540], None, None, None, None], 1920, 1080)
    expect(bool(target[0, 13, 24] == 1), "the target cell of a centred ball")
    prediction = torch.full((1, 15, GRID_ROWS, GRID_COLS), 0.1, requires_grad=True)
    loss = gridtracknet_loss(prediction, target.unsqueeze(0))
    loss.backward()
    expect(bool(torch.isfinite(loss) and torch.isfinite(prediction.grad).all()), "a finite loss and gradient")
    expect(promotion_improves(metrics_of(0.90, 0.01), metrics_of(0.85, 0.01)), "more recall at equal wrong wins")
    expect(not promotion_improves(metrics_of(0.95, 0.05), metrics_of(0.85, 0.01)), "more wrong loses")
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
