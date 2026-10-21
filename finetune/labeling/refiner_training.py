"""Training the network that places the label tool's suggestions (``cnn_refiner.Placer``).

``train`` first trains FOLDS networks that each leave a third of the clips out (TrackNet clips
of one game together) and scores them on those: that sets how sure is sure enough (at most
SURE_FAR_SHARE of the "sure" frames more than FAR_PX off) and is the check a version must pass.
Then it trains on everything, saves the version in ``cnn_refiner.MODELS`` and makes it current
when it passes. ``python finetune/labeling/cnn_refiner.py train`` runs it."""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from finetune.data.fileio import write_atomic
from finetune.data.labels import Labels, read_labels, reviewed_frames
from finetune.data.workspace import find_video, label_csv, labelled_clips, read_manifest, review_json, video_meta
from finetune.labeling import cnn_refiner, refine
from finetune.labeling.cnn_refiner import BACKWARDS, CROP, SURE_PX, WINDOW, build_network, cut_crop

if TYPE_CHECKING:
    import torch
    from torch import nn

EPOCHS = 40
CHECK_MEDIAN_PX, CHECK_FAR_SHARE = 1.0, 0.05   # a new version must beat these on clips it never saw
SURE_FAR_SHARE = 0.001   # "sure" is set so at most this share of sure frames is over 5 px off
FOLDS = 3
SCORE_BATCH = 1024         # windows scored per network call
FAR_PX = 5.0               # a placement this far off (1080p px) counts as a miss in the self-check


def clip_crops(clip: str) -> Optional[np.ndarray]:
    """Crops around every checked, visible label of a clip; cached until its labels or review change."""
    labels_csv, review = label_csv(clip), review_json(clip)
    video = find_video(clip)
    if video is None or not review.is_file():
        return None
    cached = cnn_refiner.CACHE / f"{clip}.npz"
    stamp = max(labels_csv.stat().st_mtime, review.stat().st_mtime, video.stat().st_mtime)
    if cached.is_file() and cached.stat().st_mtime >= stamp:
        crops = np.load(cached)["crops"]
        if crops.shape[1:] == (CROP, CROP, 9):      # not cut for an older crop size
            return crops
    meta = video_meta(video)
    labels: Labels = read_labels(labels_csv, meta.width, meta.height)
    step = refine.label_spacing(labels)
    points = {f: labels[f] for f in reviewed_frames(clip) if labels.get(f)}   # a damaged review file is an error
    if not points:
        return None
    crops = crops_from_video(video, points, step, meta.height)
    cnn_refiner.CACHE.mkdir(parents=True, exist_ok=True)
    out = np.stack(crops) if crops else np.zeros((0, CROP, CROP, 9), np.uint8)
    np.savez(cached, crops=out)
    return out


def frame_users(points: Dict[int, Tuple[float, float]], step: int) -> Dict[int, int]:
    """frame -> how many labels (the frame itself and the ones `step` either side) still need it."""
    users: Dict[int, int] = {}
    for f in points:
        for g in (f - step, f, f + step):
            users[g] = users.get(g, 0) + 1
    return users


def cut_label(f: int, kept: Dict[int, np.ndarray], users: Dict[int, int], point: Tuple[float, float], step: int,
              height: int) -> Optional[np.ndarray]:
    """The crop for label frame `f` from the frames kept so far (None when the video never showed
    it), letting go of every frame that no later label needs."""
    crop = cut_crop([kept.get(g, kept[f]) for g in (f - step, f, f + step)], point, height) if f in kept else None
    for g in (f - step, f, f + step):
        users[g] -= 1
        if users[g] == 0:
            kept.pop(g, None)
    return crop


def crops_from_video(video: Path, points: Dict[int, Tuple[float, float]], step: int, height: int) -> List[np.ndarray]:
    """One crop per labelled frame in `points`, in one pass over the video: each frame is kept
    only while a label `step` frames either side still needs it."""
    capture = cv2.VideoCapture(str(video))
    users = frame_users(points, step)
    pending, kept, crops, index = sorted(points), {}, [], 0
    while pending:
        ok, image = capture.read()
        if ok and index in users:
            kept[index] = image
        index += 1
        while pending and (not ok or index > pending[0] + step):
            f = pending.pop(0)
            crop = cut_label(f, kept, users, points[f], step, height)
            if crop is not None:
                crops.append(crop)
        if not ok:
            break
    capture.release()
    return crops


def gather_crops(log: Callable[..., None] = lambda *parts: print(*parts, flush=True)) -> Dict[str, np.ndarray]:
    """{clip: crops} for every clip with checked frames."""
    out = {}
    for clip in labelled_clips():
        crops = clip_crops(clip)
        if crops is not None and len(crops):
            out[clip] = crops
            log(f"  {clip}: {len(crops)} checked frames")
    return out


def folds(clips: Sequence[str], sizes: Dict[str, int]) -> Dict[str, int]:
    """{clip: fold}: whole groups (the TrackNet clips of one game together, other clips alone),
    balanced by frame count, the same every run."""
    origin = {clip: row["origin"] for clip, row in read_manifest().items()}
    group_of = {}
    for clip in clips:
        parts = re.split(r"[\\/]", origin.get(clip, ""))
        group_of[clip] = next((p for p in parts if re.fullmatch(r"game\d+", p)), clip)
    totals: Dict[str, int] = {}
    for clip, group in group_of.items():
        totals[group] = totals.get(group, 0) + sizes[clip]
    load, fold_of_group = [0] * FOLDS, {}
    for group in sorted(totals, key=lambda g: (-totals[g], g)):
        k = int(np.argmin(load))
        fold_of_group[group] = k
        load[k] += totals[group]
    return {clip: fold_of_group[group] for clip, group in group_of.items()}


def to_device(parts: Sequence[np.ndarray], device: str) -> "torch.Tensor":
    import torch
    out = torch.empty((sum(len(p) for p in parts), CROP, CROP, 9), dtype=torch.uint8, device=device)
    i = 0
    for p in parts:
        out[i:i + len(p)] = torch.from_numpy(p).to(device)
        i += len(p)
    return out


def sample_batch(data, device: str, rng: np.random.Generator, size: int, flip: bool = True,
                 in_order: bool = False) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """One batch of `size` random windows of random crops on the GPU: (inputs, targets in window
    pixels). `in_order` takes the first `size` crops in order instead of drawing them at random
    (scoring)."""
    import torch
    count = data.shape[0]
    span, mid = CROP - WINDOW, (CROP - 1) / 2.0
    ar = torch.arange(WINDOW, device=device)
    idx = torch.arange(size, device=device) if in_order else torch.from_numpy(rng.integers(0, count, size)).to(device)
    ox = torch.from_numpy(rng.integers(0, span + 1, size)).to(device)
    oy = torch.from_numpy(rng.integers(0, span + 1, size)).to(device)
    rows, cols = (oy[:, None] + ar)[:, :, None], (ox[:, None] + ar)[:, None, :]
    win = data[idx[:, None, None], rows, cols]                       # size x 64 x 64 x 9
    tx, ty = mid - ox.float(), mid - oy.float()
    if flip:
        mirror = torch.from_numpy(rng.random(size) < 0.5).to(device)
        win = torch.where(mirror[:, None, None, None], win.flip(2), win)
        tx = torch.where(mirror, WINDOW - 1 - tx, tx)
        back = torch.from_numpy(rng.random(size) < 0.5).to(device)   # the three frames played backwards
        win = torch.where(back[:, None, None, None], win[..., BACKWARDS], win)
    x = win.permute(0, 3, 1, 2).float().div_(255.0).sub_(0.5)
    if flip:
        x = x * torch.empty(size, 1, 1, 1, device=device).uniform_(0.7, 1.3)
    return x.contiguous(memory_format=torch.channels_last), torch.stack([tx, ty], 1)


def train_network(data, epochs: int, device: str, rng: np.random.Generator, log: Callable[..., None],
        label: str) -> "nn.Module":
    """One network trained on `data` (crops already on the device)."""
    import torch
    net = build_network().to(device).to(memory_format=torch.channels_last)
    batch = 256
    per_epoch = max(1, len(data) // batch)
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=epochs * per_epoch)
    bce = torch.nn.BCEWithLogitsLoss()
    start = time.time()
    for epoch in range(epochs):
        net.train()
        total = 0.0
        for _ in range(per_epoch):
            x, target = sample_batch(data, device, rng, batch)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                xy, logit = net(x)
            hit = ((xy.detach() - target).norm(dim=1) <= SURE_PX).float()
            loss = torch.nn.functional.smooth_l1_loss(xy, target, beta=0.5) + bce(logit.float(), hit)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
        if epoch % 10 == 9 or epoch == epochs - 1:
            log(f"  {label}: epoch {epoch + 1}/{epochs}  loss {total / per_epoch:.3f}  ({time.time() - start:.0f}s)")
    return net.eval()


def fold_errors(net, data, device: str, rng: np.random.Generator, repeats: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Errors (window px = 1080p px) and confidences of every frame, `repeats` times each from a
    random start up to 16 px off."""
    import torch
    errors, confs = [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        for _ in range(repeats):
            for i in range(0, len(data), SCORE_BATCH):
                part = data[i:i + SCORE_BATCH]
                x, target = sample_batch(part, device, rng, len(part), flip=False, in_order=True)
                xy, logit = net(x)
                errors.append((xy.float() - target).norm(dim=1).cpu().numpy())
                confs.append(torch.sigmoid(logit.float()).cpu().numpy())
    return np.concatenate(errors), np.concatenate(confs)


def sure_threshold(errors: np.ndarray, confs: np.ndarray) -> float:
    """The lowest confidence at which, counting every frame at least that confident, at most
    SURE_FAR_SHARE of them are more than FAR_PX off."""
    order = np.argsort(-confs)
    far = np.cumsum(errors[order] > FAR_PX) / np.arange(1, len(order) + 1)
    ok = np.where(far <= SURE_FAR_SHARE)[0]
    return float(confs[order][ok[-1]]) if len(ok) else 1.0


def train(epochs: int = EPOCHS, log: Optional[Callable[..., None]] = None, seed: int = 0) -> Optional[Path]:
    """Calibrate on folds of clips each test network never saw, train on every checked frame, save
    a new version and make it current if it passes."""
    import torch
    log = log or (lambda *parts: print(*parts, flush=True))
    log("gathering the frames you have checked ...")
    crops = gather_crops(log)
    total = sum(len(c) for c in crops.values())
    if total < 500 or len(crops) < FOLDS:
        log(f"only {total} checked frames in {len(crops)} clips - label more before training")
        return None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    fold_of = folds(sorted(crops), {c: len(v) for c, v in crops.items()})
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
    cnn_refiner.MODELS.mkdir(parents=True, exist_ok=True)
    path = cnn_refiner.MODELS / f"refiner-{time.strftime('%Y%m%d-%H%M%S')}.pt"
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
        f"(older versions stay in {cnn_refiner.MODELS.relative_to(cnn_refiner.ws.WORKSPACE.parent)})")
    return path
