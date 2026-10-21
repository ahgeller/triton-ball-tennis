"""Training the network that places the label tool's suggestions (``cnn_refiner.Placer``).

``train`` first trains FOLDS networks that each leave a third of the clips out (the clips of one
recording together, ``workspace.recording_of``) and scores them on those: that sets how sure is sure enough (at most
SURE_FAR_SHARE of the "sure" frames more than FAR_PX off) and is the check a version must pass.
Then it trains on everything, saves the version in ``cnn_refiner.models_dir()`` and makes it
current when it passes. ``python finetune/refiner/hints.py train`` runs it. The crops come from
``refiner_crops``; fitting and scoring one network is ``refiner_fitting``."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

import numpy as np

from finetune.data import workspace as ws
from finetune.data.fileio import write_atomic
from finetune.data.workspace import read_manifest, recording_of
from finetune.refiner import cnn_refiner
from finetune.refiner.refiner_crops import gather_crops
from finetune.refiner.refiner_fitting import fold_errors, to_device, train_network

EPOCHS = 40
CHECK_MEDIAN_PX, CHECK_FAR_SHARE = 1.0, 0.05   # a new version must beat these on clips it never saw
SURE_FAR_SHARE = 0.001   # "sure" is set so at most this share of sure frames is over 5 px off
FOLDS = 3
FAR_PX = 5.0               # a placement this far off (1080p px) counts as a miss in the self-check


def recordings(clips: Sequence[str]) -> Dict[str, str]:
    """{clip: the recording it was cut from}, one manifest read for all of them."""
    manifest = read_manifest()
    return {clip: recording_of(clip, manifest) for clip in clips}


def folds(clips: Sequence[str], sizes: Dict[str, int]) -> Dict[str, int]:
    """{clip: fold}: whole recordings (one video's rallies, one TrackNet game's clips, one camera's
    own clips), balanced by frame count, the same every run. A fold's clips are what its test
    network never saw, so a court it trained on must not come back in another clip."""
    group_of = recordings(clips)
    totals: Dict[str, int] = {}
    for clip, group in group_of.items():
        totals[group] = totals.get(group, 0) + sizes[clip]
    load, fold_of_group = [0] * FOLDS, {}
    for group in sorted(totals, key=lambda g: (-totals[g], g)):
        k = int(np.argmin(load))
        fold_of_group[group] = k
        load[k] += totals[group]
    return {clip: fold_of_group[group] for clip, group in group_of.items()}


def sure_threshold(errors: np.ndarray, confs: np.ndarray) -> float:
    """The lowest confidence at which, counting every frame at least that confident, at most
    SURE_FAR_SHARE of them are more than FAR_PX off."""
    order = np.argsort(-confs)
    far = np.cumsum(errors[order] > FAR_PX) / np.arange(1, len(order) + 1)
    ok = np.where(far <= SURE_FAR_SHARE)[0]
    return float(confs[order][ok[-1]]) if len(ok) else 1.0


def train(epochs: int = EPOCHS, log: Optional[Callable[..., None]] = None, seed: int = 0) -> Optional[Path]:
    """Calibrate on folds of clips each test network never saw, train on every reviewed frame, save
    a new version and make it current if it passes."""
    import torch
    log = log or (lambda *parts: print(*parts, flush=True))
    log("gathering the frames you have reviewed ...")
    crops = gather_crops(log)
    total = sum(len(c) for c in crops.values())
    fold_of = folds(sorted(crops), {c: len(v) for c, v in crops.items()})
    # Fewer recordings than folds leaves a test network nothing it never saw.
    if total < 500 or len(set(fold_of.values())) < FOLDS:
        log(f"only {total} reviewed frames in {len(crops)} clips of {len(set(recordings(crops).values()))} "
            f"recordings - label more (from at least {FOLDS} recordings) before training")
        return None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    log(f"{total} frames from {len(crops)} clips ({device}): {FOLDS} test networks, then the final one")
    errors, confs = [], []
    for k in range(FOLDS):
        train_part = to_device([crops[c] for c in sorted(crops) if fold_of[c] != k], device)
        net = train_network(train_part, epochs, device, rng, log, f"test {k + 1}/{FOLDS}")
        del train_part
        e, c = fold_errors(net, to_device([crops[c] for c in sorted(crops) if fold_of[c] == k], device), device,
                     np.random.default_rng(k + 1))
        errors.append(e)
        confs.append(c)
    errors, confs = np.concatenate(errors), np.concatenate(confs)
    threshold = sure_threshold(errors, confs)
    sure = confs >= threshold
    median, far = float(np.median(errors)), float(np.mean(errors > FAR_PX))
    sure_far = float(np.mean(errors[sure] > FAR_PX)) if sure.any() else 0.0
    log(f"on clips each test network never saw: median {median:.2f} px, {far:.1%} over 5 px; "
        f"{sure.mean():.0%} sure, {sure_far:.2%} of those over 5 px")
    net = train_network(to_device(list(crops.values()), device), epochs, device, rng, log, "final")
    previous = cnn_refiner.current()
    models = cnn_refiner.models_dir()
    models.mkdir(parents=True, exist_ok=True)
    path = models / f"refiner-{time.strftime('%Y%m%d-%H%M%S')}.pt"
    torch.save({"state": net.state_dict(), "self_check": True, "threshold": threshold,
                "trained": time.strftime("%Y-%m-%d %H:%M:%S"), "clips": len(crops), "frames": int(total),
                "check": {"median_px": median, "over5": far, "sure": float(sure.mean()), "sure_over5": sure_far}},
               path)
    if median > CHECK_MEDIAN_PX or far > CHECK_FAR_SHARE:
        log(f"saved {path.name}, but it failed the check (needs median <= {CHECK_MEDIAN_PX} px and "
            f"<= {CHECK_FAR_SHARE:.0%} over 5 px), so the label tool keeps "
            f"{previous.stem if previous else 'the motion centroid'}")
        return path
    write_atomic(cnn_refiner.current_file(), path.name)
    log(f"saved {path.name}; the label tool uses it from its next start "
        f"(older versions stay in {models.relative_to(ws.current().root.parent)})")
    return path
