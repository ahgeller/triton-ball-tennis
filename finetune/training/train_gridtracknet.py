"""Fine-tune (or train from scratch) GridTrackNet on the finetune workspace labels.

Data contract (same as the archive): ``videos/<clip>.mp4`` + ``labels/<clip>_ball.csv``
with ``frame,ball_x,ball_y`` rows at 30 FPS cadence (every 2nd frame of a 60 FPS
clip, every frame at 30 FPS; the first labelled frame may be odd) and invisible
balls parked in the top-right corner. ``clips.csv`` (written by import_data.py)
tags every clip with a source so you can pick, hold out and oversample by source.

    python finetune/training/train_gridtracknet.py                              # fine-tune bundled weights, hold out video10
    python finetune/training/train_gridtracknet.py --val-clips video10 video11  # own-camera hold-out
    python finetune/training/train_gridtracknet.py --holdout mixed_v1           # the fixed held-out set in holdouts.json
    python finetune/training/train_gridtracknet.py --sources custom             # own footage only
    python finetune/training/train_gridtracknet.py --oversample custom=6 --epoch-units 20000
    python finetune/training/train_gridtracknet.py --augment basic translate reverse --oversample-absent 3
    python finetune/training/train_gridtracknet.py --from-scratch --epochs 30   # train a fresh network on everything
    python finetune/training/train_gridtracknet.py --all-clips --selection-run finetune/runs/r5 --augment basic translate
    python finetune/training/train_gridtracknet.py --self-test

After every scored epoch (``--score-every``, the last epoch always) the weights are
scored on the held-out clips with the detector metric ``evaluate_archive.py`` reports
(recall / wrong-object / false-alarm), and ``best.npz`` is the epoch that metric
prefers (``promotion_improves``; a tie keeps the earlier epoch), not the one with the
lowest validation loss (kept as ``best_loss.npz`` for the record). The run ends by
setting that best epoch against the starting weights on the same clips, and the
winner is copied to ``--save`` so the tracker always points at the best model.

``--all-clips`` is the other kind of run: once a recipe (the flags and the epoch count)
has been chosen on a held-out set, train it on every clip with nothing held out and no
scoring, and take ``last.npz`` as the result; ``winner.json`` then says
``selected_by: "recipe"`` and copies in the selection run's details (``--selection-run``),
since the deployable weights were never measured on footage they did not see.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, NamedTuple, Optional, Sequence

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
from finetune.training.dataset import (AUGMENTATIONS, Augmenter, UnitDataset, make_sampler, make_target,
                                       photometric_augment, unit_weights)
from finetune.training.frame_cache import build_units, prepare_data
from finetune.training.loop import evaluate_loss, fresh_model, gridtracknet_loss, save_weights, train_epoch
from finetune.training.metrics import FALSE_ALARM_CONF, false_alarm_units, promotion_improves, score_weights

SEED = 7                   # every run draws the same augmentation and sampling sequence unless --seed says otherwise
DEFAULT_EPOCHS = 10
RESULTS_COLUMNS = ("epoch", "train_loss", "val_loss", "seconds", "score_seconds", *DetectorMetrics._fields)

# --------------------------------------------------------------------------- the run

def parse_oversample(items: Optional[List[str]]) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    for item in items or []:
        name, _, value = item.partition("=")
        if not value:
            raise ValueError(f"--oversample expects source=weight, got {item!r}")
        weights[name.strip()] = float(value)
    return weights


def run_config(args: argparse.Namespace, names: List[str], val_clips: List[str], oversample: Dict[str, float],
               flagged: int = 0) -> dict:
    """What a run was: the clips, the held-out ones and how they were chosen, and every option that
    shaped the training (``flagged``: how many training units ``--oversample-false-alarms`` weighted
    up), as ``config.json`` records it."""
    return {"clips": names, "validation_clips": val_clips, "holdout": args.holdout, "val_clips": args.val_clips,
            "all_clips": args.all_clips, "selection_run": str(args.selection_run) if args.selection_run else None,
            "sources": args.sources, "cameras": args.cameras, "augment": sorted(set(args.augment)),
            "oversample": oversample, "oversample_absent": args.oversample_absent,
            "oversample_false_alarms": args.oversample_false_alarms, "false_alarm_units": flagged,
            "epoch_units": args.epoch_units, "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
            "seed": args.seed, "score_every": args.score_every,
            "from_scratch": args.from_scratch, "weights": str(args.weights)}


class Selection(NamedTuple):
    """What the epochs left behind: the epoch the detector metric chose (``best.npz``) with its
    metrics, and the epoch of the lowest validation loss (``best_loss.npz``), for the record. A
    run with nothing held out (``--all-clips``) scores nothing: its result is ``last.npz``, the
    final epoch, and the metrics and the loss epoch are absent."""
    best_path: Path
    best_epoch: int
    best_metrics: Optional[DetectorMetrics]
    best_loss_epoch: int
    best_loss: float


def results_row(epoch: int, train_loss: float, val_loss: Optional[float], seconds: float,
                score_seconds: Optional[float], metrics: Optional[DetectorMetrics]) -> List[str]:
    """One ``results.csv`` row; the validation column is empty when nothing is held out and the
    scoring columns are empty on an epoch that was not scored."""
    row = [str(epoch), f"{train_loss:.8f}", "" if val_loss is None else f"{val_loss:.8f}", f"{seconds:.0f}"]
    if metrics is None:
        return row + [""] * (len(RESULTS_COLUMNS) - len(row))
    return row + [f"{score_seconds:.0f}"] + [f"{value:.6f}" if isinstance(value, float) else str(value) for value in metrics]


def fit(args: argparse.Namespace, model: torch.nn.Module, train_loader, val_loader, optimizer, scaler,
        device: torch.device, score: Optional[Callable[[Path], DetectorMetrics]]) -> Selection:
    """Run the epochs. After each one ``last.npz`` is written and ``results.csv`` gets its row; on a
    scored epoch (every ``--score-every``, and always the last) ``score(last.npz)`` measures the
    held-out detector metric and ``best.npz`` follows it by ``promotion_improves`` (a tie keeps
    the earlier epoch); ``best_loss.npz`` follows the validation loss. With nothing held out
    (``val_loader`` and ``score`` None) the epochs only run, and ``last.npz`` is the result."""
    results = args.run_dir / "results.csv"
    write_atomic(results, ",".join(RESULTS_COLUMNS) + "\n")
    last_path, best_path, best_loss_path = (args.run_dir / name for name in ("last.npz", "best.npz", "best_loss.npz"))
    best_epoch, best_metrics = 0, None
    best_loss_epoch, best_loss = 0, math.inf
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device, epoch, args.epochs, args.max_train_steps)
        val_loss = None if val_loader is None else evaluate_loss(model, val_loader, device, args.max_val_steps)
        save_weights(model, args.weights, last_path)
        if val_loss is not None and val_loss < best_loss:
            best_loss_epoch, best_loss = epoch, val_loss
            copy_atomic(last_path, best_loss_path)
        seconds = time.perf_counter() - started
        metrics = score_seconds = None
        if score is not None and (epoch % args.score_every == 0 or epoch == args.epochs):
            scoring = time.perf_counter()
            metrics = score(last_path)
            score_seconds = time.perf_counter() - scoring
            if best_metrics is None or promotion_improves(metrics, best_metrics):
                best_epoch, best_metrics = epoch, metrics
                copy_atomic(last_path, best_path)
        with results.open("a", encoding="utf-8", newline="") as handle:    # one line per epoch, appended
            csv.writer(handle).writerow(results_row(epoch, train_loss, val_loss, seconds, score_seconds, metrics))
        validated = "" if val_loss is None else f" val={val_loss:.6f}"
        scored = (f" | held-out recall={metrics.recall:.3f} wrong={metrics.wrong:.3f} falseAlarm={metrics.false_alarm:.3f}"
                  f" ({score_seconds / 60:.1f} min) | best epoch {best_epoch}" if metrics is not None else "")
        print(f"[train] epoch {epoch}: train={train_loss:.6f}{validated} ({seconds / 60:.1f} min){scored}", flush=True)
    if score is None:
        return Selection(last_path, args.epochs, None, best_loss_epoch, best_loss)
    return Selection(best_path, best_epoch, best_metrics, best_loss_epoch, best_loss)


def read_selection_run(run_dir: Path) -> dict:
    """What a held-out selection run decided, for the ``--all-clips`` run that follows its recipe:
    its ``winner.json`` (the best epoch and the held-out metrics) and the ``config.json`` fields
    that make up the recipe. A folder without the two files is a ValueError."""
    try:
        winner = json.loads((run_dir / "winner.json").read_text(encoding="utf-8"))
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"--selection-run {run_dir} has no readable winner.json and config.json ({error})") from error
    recipe = {key: config.get(key) for key in ("augment", "oversample", "oversample_absent", "oversample_false_alarms",
                                                "epoch_units", "batch_size", "lr", "seed", "holdout", "validation_clips")}
    return {"run": str(run_dir), "best_epoch": winner.get("best_epoch"), "selected_by": winner.get("selected_by"),
            "winner": winner.get("winner"), "metrics": winner.get("metrics"),
            "starting_weights": winner.get("starting_weights"), "recipe": recipe}


def record_recipe_result(args: argparse.Namespace, selection: Selection, selection_run: Optional[dict]) -> None:
    """The end of an ``--all-clips`` run: ``last.npz`` is copied to ``--save`` and ``winner.json``
    says the weights were chosen by recipe, with the selection run's details beside, not by a
    measurement of these weights, which saw every clip."""
    copy_atomic(selection.best_path, args.save)
    write_atomic(args.run_dir / "winner.json", json.dumps({
        "winner": str(selection.best_path), "selected_by": "recipe", "epochs": args.epochs, "metrics": None,
        "validation_clips": [], "selection_run": selection_run}, indent=2))
    print(f"[done] trained on every clip for {args.epochs} epochs -> {args.save}  (from {selection.best_path.name}; "
          "nothing was held out, so these weights carry no measurement of their own)")
    print("[done] promote it with: python finetune/ft.py promote")


def pick_winner(args: argparse.Namespace, clips: List[Clip], val_clips: List[str], selection: Selection,
                device: torch.device) -> None:
    """Keep the best model by the metric that matters, not just the loss: the best epoch's weights
    (already scored while training) win only when they beat the starting weights on the held-out
    clips (see `promotion_improves`)."""
    trained = selection.best_metrics
    print(f"[eval] best.npz (epoch {selection.best_epoch}) on {val_clips}: {trained}")
    winner, winner_stats = selection.best_path, trained
    record = {"selected_by": "detector_metric", "best_epoch": selection.best_epoch,
              "best_loss_epoch": selection.best_loss_epoch, "validation_clips": val_clips}
    if not args.from_scratch:
        val_only = [entry for entry in clips if entry[0] in val_clips]
        baseline = score_weights(args.weights, val_only, device)
        print(f"[eval] starting weights on {val_clips}: {baseline}")
        record["starting_weights"] = {"path": str(args.weights), "metrics": baseline._asdict()}
        if not promotion_improves(trained, baseline):
            winner, winner_stats = args.weights, baseline
    copy_atomic(winner, args.save)
    write_atomic(args.run_dir / "winner.json",
                 json.dumps({"winner": str(winner), "metrics": winner_stats._asdict(), **record}, indent=2))
    print(f"[done] best model -> {args.save}  (from {winner.name}: {winner_stats})")
    if winner == selection.best_path:
        print("[done] promote it with: python finetune/ft.py promote")
    else:
        print("[done] the starting weights stayed ahead on the held-out clips; nothing to promote")


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)       # the data loaders' draws derive from it too, see Augmenter
    selection_run = read_selection_run(args.selection_run) if args.selection_run else None
    if args.epochs is None:
        args.epochs = (selection_run or {}).get("best_epoch") or DEFAULT_EPOCHS
    clips = find_clips(args.videos, args.labels, ClipSelection(args.sources, args.exclude_clips, args.include_excluded, args.cameras))
    names = [clip for clip, _, _, _ in clips]
    val_clips = [] if args.all_clips else validation_split(clips, args.labels, args.val_clips, args.holdout)
    oversample = parse_oversample(args.oversample)
    manifest = prepare_data(clips, args.data_dir, val_clips, args.rebuild_data, args.prep_workers)
    if not args.all_clips and not manifest["val"]:
        raise ValueError("Need labelled clips in both the training and validation split")
    if args.prepare_only:
        print("[data] cache ready; --prepare-only stops here")
        return
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    flagged = set()
    if args.oversample_false_alarms != 1.0:
        flagged = false_alarm_units(args.weights, clips, manifest["train"], device)
        print(f"[data] {len(flagged)} of {len(manifest['train'])} training units hold a no-ball frame the starting "
              f"weights fired on at conf>={FALSE_ALARM_CONF:.2f}: "
              f"sampled x{args.oversample_false_alarms:g}")
    sampler = make_sampler(manifest["train"], oversample, args.epoch_units, args.oversample_absent,
                           args.oversample_false_alarms, flagged)
    train_loader = DataLoader(UnitDataset(args.data_dir, manifest["train"], augment=True, augmentations=args.augment),
                              batch_size=args.batch_size,
                              shuffle=sampler is None, sampler=sampler, num_workers=args.workers,
                              pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    val_loader = None if args.all_clips else DataLoader(
        UnitDataset(args.data_dir, manifest["val"], augment=False), batch_size=args.batch_size,
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
    write_atomic(args.run_dir / "config.json", json.dumps(run_config(args, names, val_clips, oversample, len(flagged)), indent=2))
    print(f"[train] device={device} epochs={args.epochs} batch={args.batch_size} seed={args.seed} "
          f"units/epoch={len(train_loader) * args.batch_size} augment={sorted(set(args.augment))} "
          + ("nothing held out (--all-clips)" if args.all_clips else f"val={val_clips} score every {args.score_every}")
          + (f" oversample={oversample}" if oversample else ""))
    if args.all_clips:
        record_recipe_result(args, fit(args, model, train_loader, None, optimizer, scaler, device, score=None), selection_run)
        return
    val_only = [entry for entry in clips if entry[0] in val_clips]
    selection = fit(args, model, train_loader, val_loader, optimizer, scaler, device,
                    score=lambda weights: score_weights(weights, val_only, device))
    pick_winner(args, clips, val_clips, selection, device)


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
    expect(not is_visible(1919, 0, 1920, 1080) and is_visible(1910, 8, 1920, 1080), "the invisible-ball sentinel")
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
    units = [{"source": "a", "points": [None]}, {"source": "b", "points": [[1.0, 1.0]]}]
    sampler = make_sampler(units, {"b": 0.0}, None)
    expect(sampler is not None and set(iter(sampler)) == {0}, "weight 0 drops a source")
    expect(make_sampler(units, {}, None) is None, "plain shuffle without weights")
    expect(unit_weights(units, {}, absent=3.0, false_alarms=2.0, flagged={1}) == [3.0, 2.0], "absent and flagged weights")
    expect(resolve_val_clips(["video1", "video10", "tnv2_Test_match1_1"], ["tnv2_Test_*"]) == ["tnv2_Test_match1_1"],
           "validation globs")
    expect(resolve_val_clips(["video1", "video10"], None) == ["video10"], "the default validation clip")
    augmented = photometric_augment(np.full((15, 8, 8), 0.5, dtype=np.float32), Augmenter(1))
    expect(augmented.shape == (15, 8, 8) and augmented.dtype == np.float32
           and 0.0 <= augmented.min() <= augmented.max() <= 1.0, "augmentation keeps shape and range")
    row = results_row(2, 0.5, 0.25, 90.0, 10.0, metrics_of(0.9, 0.01))
    expect(len(row) == len(RESULTS_COLUMNS) and row[5] == "100" and row[6] == "0.900000", "a scored results row")
    expect(results_row(1, 0.5, 0.25, 90.0, None, None)[4:] == [""] * (len(RESULTS_COLUMNS) - 4), "an unscored results row")
    expect(results_row(1, 0.5, None, 90.0, None, None)[2] == "", "a results row with nothing held out")
    model = fresh_model(torch.device("cpu"))
    expect(model(torch.zeros(1, 15, HEIGHT, WIDTH)).shape == (1, 15, GRID_ROWS, GRID_COLS), "the fresh model's output")
    print("train_gridtracknet self-test passed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    layout = ws.current()
    parser.add_argument("--videos", type=Path, default=layout.videos)
    parser.add_argument("--labels", type=Path, default=layout.labels)
    parser.add_argument("--weights", type=Path, default=layout.repo_weights,
                        help="Starting weights (also the .npz template for saving)")
    parser.add_argument("--save", type=Path, default=layout.best_weights)
    parser.add_argument("--data-dir", type=Path, default=layout.cache / "data")
    parser.add_argument("--run-dir", type=Path, default=layout.runs / "train")
    held_out = parser.add_mutually_exclusive_group()
    held_out.add_argument("--val-clips", nargs="*", help="Held-out clip stems or globs (default: video10 if present, else the last clip)")
    held_out.add_argument("--holdout", metavar="NAME",
                          help="Hold out a named set of whole recordings from the workspace's holdouts.json (e.g. mixed_v1)")
    held_out.add_argument("--all-clips", action="store_true",
                          help="Hold out nothing and score nothing: train the recipe on every clip and take last.npz "
                               "(for the deployable weights, once a recipe was chosen on a held-out set)")
    parser.add_argument("--selection-run", type=Path, metavar="DIR",
                        help="With --all-clips: the held-out run whose recipe this follows; its best epoch is the "
                             "default --epochs and its details are copied into winner.json")
    parser.add_argument("--sources", nargs="*", help="Only use clips whose clips.csv source is listed (e.g. custom grid)")
    parser.add_argument("--exclude-clips", nargs="*", help="Clip stems or globs to leave out entirely")
    parser.add_argument("--include-excluded", action="store_true", help="Explicitly include excluded/unverified clips for training (validation still requires verified labels)")
    parser.add_argument("--cameras", nargs="*", choices=("fixed", "moving", "unknown", "untagged"),
                        help="Only clips with these clips.csv camera tags (from  ft.py camera), e.g. --cameras fixed")
    parser.add_argument("--augment", nargs="+", choices=AUGMENTATIONS, default=["basic"], metavar="NAME",
                        help="The augmentations to draw, any of: basic (flip + small gain, the default), strong (flip + "
                             "gamma/contrast/colour cast/noise/blur instead of the gain), translate (one shift of up to "
                             "20 px at 768x432 for all five frames), reverse (time reversed), compress (JPEG re-encode "
                             "and occasional blur)")
    parser.add_argument("--oversample", nargs="*", metavar="SOURCE=WEIGHT",
                        help="Relative sampling weight per source, e.g. custom=6 grid=1 tracknetv2=0.5 (0 drops it from training)")
    parser.add_argument("--oversample-absent", type=float, default=1.0, metavar="K",
                        help="Sampling weight of a unit holding an absent-ball label, relative to the others (default 1)")
    parser.add_argument("--oversample-false-alarms", type=float, default=1.0, metavar="K",
                        help="Sampling weight of a unit in which the starting weights fired (conf>=0.3) on a no-ball frame, "
                             "found by one prepass over the training clips before the run (default 1: no prepass)")
    parser.add_argument("--epoch-units", type=int, help="Units drawn per epoch (default: every training unit once)")
    parser.add_argument("--epochs", type=int, default=None,
                        help=f"Default {DEFAULT_EPOCHS}, or the --selection-run's best epoch")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1.0, help="Adadelta lr (1.0 = upstream; 0.3 for gentle fine-tunes)")
    parser.add_argument("--seed", type=int, default=SEED, help=f"The augmentation and sampling seed (default {SEED})")
    parser.add_argument("--score-every", type=int, default=1, metavar="K",
                        help="Score the held-out clips with the detector metric every K epochs (the last epoch always); "
                             "best.npz is the scored epoch the metric prefers")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--prep-workers", type=int, default=4, help="Parallel clips while extracting the frame cache")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--from-scratch", action="store_true", help="Train a new network instead of fine-tuning")
    parser.add_argument("--rebuild-data", action="store_true")
    parser.add_argument("--prepare-only", action="store_true", help="Build/refresh the frame cache and stop")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--max-train-steps", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--max-val-steps", type=int, help=argparse.SUPPRESS)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """0 when the self-test or the run finished (whichever weights won), 1 when the clips, the
    labels, the frame cache or an option cannot make a run: the ValueError or OSError the
    training modules raise for those is reported in one line, not as a traceback."""
    args = build_parser().parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if args.score_every < 1:
        print("[error] --score-every must be at least 1", file=sys.stderr)
        return 1
    if args.selection_run and not args.all_clips:
        print("[error] --selection-run goes with --all-clips", file=sys.stderr)
        return 1
    try:
        train(args)
    except (ValueError, OSError) as error:
        print(f"[error] {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
