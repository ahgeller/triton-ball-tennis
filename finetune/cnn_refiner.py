"""A small CNN that learns where you click the ball, used to place the label tool's suggestions.

refine.py still finds the ball (the motion blobs, the path through the neighbours) and says whether
a ball is there at all; this network then puts it where you would click, from the label frame and
the label frames either side cut 64x64 around that point, and says how sure it is (a second head
predicting whether its own answer is within 2 px). It learns only from frames you have checked in
the label tool (reviewed or edited), so retraining as you label keeps it current:

    python finetune/cnn_refiner.py train        # or "hint model" in the finetune menu
    python finetune/cnn_refiner.py list
    python finetune/cnn_refiner.py use <version>

Training first trains three networks that each leave a third of the clips out (TrackNet clips of
one game together) and scores them on those: that sets how sure is sure enough (at most 0.1% of the
"sure" frames more than 5 px off) and is the check a version must pass. Then it trains on everything.
Every version is kept in finetune/models/refiner/ and the new one becomes the one the label tool
uses (current.txt) once it passes; `use` switches back to any older version.

Measured on 66 reviewed clips, each scored by a network that never saw it: a median 0.66 px from
your clicks against 0.79 px for refine.py's centroid alone, and 64% of frames sure with 0.09% of
those over 5 px off (refine.py's own trust rule: 55% sure at the same 0.09%)."""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
LABELS, VIDEOS = HERE / "labels", HERE / "videos"
MODELS = HERE / "models" / "refiner"
CACHE = HERE / "cache" / "refiner"
CURRENT = MODELS / "current.txt"
CROP = 96            # cut around a point at 1080p scale ...
WINDOW = 64          # ... and the network sees this much of it, so the ball may sit 16 px off centre
EPOCHS = 40
CHECK_MEDIAN_PX, CHECK_FAR_SHARE = 1.0, 0.05   # a new version must beat these on clips it never saw
SURE_PX = 2.0            # the self-check learns whether its answer is within this
SURE_FAR_SHARE = 0.001   # ... and "sure" is set so at most this share of sure frames is over 5 px off
FOLDS = 3
VIDEO_SUFFIXES = (".mp4", ".mov", ".avi", ".mkv", ".m4v")


# ---------------------------------------------------------------------------------------- crops
def cut(images: Sequence[np.ndarray], point: Tuple[float, float], height: int) -> np.ndarray:
    """uint8 CROP x CROP x 9: the three frames (before, label frame, after) around `point`, scaled
    to 1080p. Crop pixel j sits at point + (j - (CROP - 1) / 2) * height / 1080 in the image."""
    side = int(round(CROP * height / 1080.0))
    out = []
    for image in images:
        patch = cv2.getRectSubPix(image, (side, side), (point[0] - 0.5, point[1] - 0.5))
        out.append(patch if side == CROP else cv2.resize(patch, (CROP, CROP), interpolation=cv2.INTER_LINEAR))
    return np.concatenate(out, axis=2)


def window(crop: np.ndarray, centre: Tuple[float, float]) -> np.ndarray:
    """WINDOW x WINDOW x 9 of a crop, centred (sub-pixel) at crop coordinates `centre`."""
    return np.concatenate([cv2.getRectSubPix(np.ascontiguousarray(crop[..., i:i + 3]), (WINDOW, WINDOW), centre)
                           for i in (0, 3, 6)], axis=2)


# -------------------------------------------------------------------------------------- network
def network(self_check: bool = True):
    """self_check=False is the first design (versions before the self-check), kept so they still load."""
    import torch.nn as nn
    import torch

    class Net(nn.Module):
        """Fully convolutional: a heatmap over the window, read out as its soft-argmax (sub-pixel),
        and a small head on what it looked at that predicts whether that answer is within SURE_PX."""

        def __init__(self, width: int = 32):
            super().__init__()
            layers, inp = [], 9
            for dilation in (1, 1, 2, 4, 8, 1):
                layers += [nn.Conv2d(inp, width, 3, padding=dilation, dilation=dilation),
                           nn.BatchNorm2d(width), nn.ReLU()]
                inp = width
            if not self_check:
                layers += [nn.Conv2d(width, 1, 1)]
            self.body = nn.Sequential(*layers)
            self.heat = nn.Conv2d(width, 1, 1) if self_check else None
            self.sure = nn.Sequential(nn.Linear(width + 3, 64), nn.ReLU(), nn.Linear(64, 1)) if self_check else None

        def forward(self, x):
            n, _, h, w = x.shape
            feat = self.body(x)
            logits = (feat if self.heat is None else self.heat(feat)).flatten(1).float()
            prob = logits.softmax(1)
            p2 = prob.view(n, h, w)
            gx = torch.arange(w, device=x.device, dtype=prob.dtype)
            gy = torch.arange(h, device=x.device, dtype=prob.dtype)
            px, py = (p2.sum(1) * gx).sum(1), (p2.sum(2) * gy).sum(1)
            xy = torch.stack([px, py], 1)
            if self.sure is None:
                return xy, None
            spread = ((p2.sum(1) * (gx - px[:, None]) ** 2).sum(1)
                      + (p2.sum(2) * (gy - py[:, None]) ** 2).sum(1)).sqrt()
            pooled = (feat.float().flatten(2) * prob[:, None]).sum(2)          # features where it looks
            stats = torch.stack([prob.max(1).values * h * w / 100.0, spread / 10.0, logits.std(1) / 10.0], 1)
            return xy, self.sure(torch.cat([pooled, stats], 1)).squeeze(1)

    return Net()


def to_tensor(windows: np.ndarray, device):
    import torch
    return torch.from_numpy(np.ascontiguousarray(windows)).to(device).permute(0, 3, 1, 2).float().div_(255.0).sub_(0.5)


class Placer:
    """Puts suggestions where you would click. `place` takes (crop, crop_point, start) per frame:
    a crop from `cut` around crop_point, and the point to start from (within about 16 px of it), and
    returns (x, y, sure) - sure is None for a version without the self-check, else whether its
    confidence reaches the version's threshold."""
    ROUNDS = 2
    cut = staticmethod(cut)

    def __init__(self, path: Path):
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        saved = torch.load(path, map_location=self.device)
        self.net = network(bool(saved.get("self_check"))).to(self.device)
        self.net.load_state_dict(saved["state"])
        self.net.eval()
        self.threshold = saved.get("threshold") if saved.get("self_check") else None
        self.name = path.stem

    def place(self, items: List[Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]], height: int):
        import torch
        if not items:
            return []
        scale = height / 1080.0
        mid, lo, hi = (CROP - 1) / 2.0, (WINDOW - 1) / 2.0, CROP - 1 - (WINDOW - 1) / 2.0
        # where each window is centred, in crop pixels (kept inside the crop)
        centres = [(min(hi, max(lo, mid + (s[0] - c[0]) / scale)), min(hi, max(lo, mid + (s[1] - c[1]) / scale)))
                   for _, c, s in items]
        sure = [None] * len(items)
        for _ in range(self.ROUNDS):
            moved = []
            with torch.no_grad():
                for i in range(0, len(items), 512):
                    batch = np.stack([window(items[j][0], centres[j]) for j in range(i, min(len(items), i + 512))])
                    xy, logit = self.net(to_tensor(batch, self.device))
                    for k, (px, py) in enumerate(xy.cpu().numpy()):
                        cx, cy = centres[i + k]
                        moved.append((min(hi, max(lo, cx + px - lo)), min(hi, max(lo, cy + py - lo))))
                    if logit is not None and self.threshold is not None:
                        for k, c in enumerate(torch.sigmoid(logit).cpu().numpy()):
                            sure[i + k] = bool(c >= self.threshold)       # the last round's answer counts
            centres = moved
        return [(c[0] + (cx - mid) * scale, c[1] + (cy - mid) * scale, ok)
                for (_, c, _), (cx, cy), ok in zip(items, centres, sure)]


# ------------------------------------------------------------------------------------- versions
def versions() -> List[Path]:
    return sorted(MODELS.glob("refiner-*.pt"))


def current() -> Optional[Path]:
    """The version the label tool uses: the one current.txt names, else the newest."""
    try:
        named = MODELS / CURRENT.read_text(encoding="utf-8").strip()
        if named.is_file():
            return named
    except OSError:
        pass
    found = versions()
    return found[-1] if found else None


def use(name: str) -> Path:
    match = [p for p in versions() if p.stem == name or p.name == name or name in p.stem]
    if len(match) != 1:
        raise SystemExit(f"no single version matches {name!r}; `list` shows them")
    CURRENT.write_text(match[0].name, encoding="utf-8")
    return match[0]


def load_current() -> Optional[Placer]:
    """The current version, or None when there is none or torch cannot load it (the tool then
    shows refine.py's own positions)."""
    path = current()
    if path is None:
        return None
    try:
        return Placer(path)
    except Exception as error:
        print(f"[hint] could not load {path.name} ({error}); using the motion centroid")
        return None


# ------------------------------------------------------------------------------------ training
def read_labels(path: Path, width: int, height: int) -> Dict[int, Optional[Tuple[float, float]]]:
    labels = {}
    for row in csv.DictReader(open(path, encoding="utf-8-sig")):
        frame = int(re.search(r"\d+", row["frame"]).group())
        x, y = float(row["ball_x"]), float(row["ball_y"])
        labels[frame] = None if (x >= width * .95 and y <= height * .05) else (x, y)
    return labels


def checked_frames(clip: str) -> set:
    """Frames you have reviewed or edited in the label tool."""
    try:
        state = json.loads((LABELS / f"{clip}_ball.review.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    return set(state.get("reviewed") or []) | set(state.get("edited") or [])


def video_for(clip: str) -> Optional[Path]:
    return next((p for p in VIDEOS.glob(f"{clip}.*") if p.suffix.lower() in VIDEO_SUFFIXES), None)


def spacing(frames) -> int:
    gaps = np.diff(sorted(frames))
    return int(np.median(gaps)) if len(gaps) else 1


def clip_crops(clip: str) -> Optional[np.ndarray]:
    """Crops around every checked, visible label of a clip; cached until its labels or review change."""
    labels_csv, review = LABELS / f"{clip}_ball.csv", LABELS / f"{clip}_ball.review.json"
    video = video_for(clip)
    if video is None or not review.is_file():
        return None
    cached = CACHE / f"{clip}.npz"
    stamp = max(labels_csv.stat().st_mtime, review.stat().st_mtime)
    if cached.is_file() and cached.stat().st_mtime >= stamp:
        return np.load(cached)["crops"]
    capture = cv2.VideoCapture(str(video))
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    labels = read_labels(labels_csv, width, height)
    step = spacing(labels)
    points = {f: labels[f] for f in checked_frames(clip) if labels.get(f)}
    if not points:
        capture.release()
        return None
    users: Dict[int, int] = {}
    for f in points:
        for g in (f - step, f, f + step):
            users[g] = users.get(g, 0) + 1
    pending, kept, crops, index = sorted(points), {}, [], 0
    while pending:
        ok, image = capture.read()
        if ok and index in users:
            kept[index] = image
        index += 1
        while pending and (not ok or index > pending[0] + step):
            f = pending.pop(0)
            if f in kept:
                crops.append(cut([kept.get(g, kept[f]) for g in (f - step, f, f + step)], points[f], height))
            for g in (f - step, f, f + step):
                users[g] -= 1
                if users[g] == 0:
                    kept.pop(g, None)
        if not ok:
            break
    capture.release()
    CACHE.mkdir(parents=True, exist_ok=True)
    out = np.stack(crops) if crops else np.zeros((0, CROP, CROP, 9), np.uint8)
    np.savez(cached, crops=out)
    return out


def gather(log=lambda *parts: print(*parts, flush=True)) -> Dict[str, np.ndarray]:
    """{clip: crops} for every clip with checked frames."""
    out = {}
    for labels_csv in sorted(LABELS.glob("*_ball.csv")):
        clip = labels_csv.name[:-len("_ball.csv")]
        crops = clip_crops(clip)
        if crops is not None and len(crops):
            out[clip] = crops
            log(f"  {clip}: {len(crops)} checked frames")
    return out


def folds(clips: Sequence[str], sizes: Dict[str, int]) -> Dict[str, int]:
    """{clip: fold}: whole groups (the TrackNet clips of one game together, other clips alone),
    balanced by frame count, the same every run."""
    try:
        with (HERE / "clips.csv").open(newline="", encoding="utf-8") as handle:
            origin = {r["clip"]: r.get("origin", "") for r in csv.DictReader(handle)}
    except OSError:
        origin = {}
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


def to_device(parts: Sequence[np.ndarray], device):
    import torch
    out = torch.empty((sum(len(p) for p in parts), CROP, CROP, 9), dtype=torch.uint8, device=device)
    i = 0
    for p in parts:
        out[i:i + len(p)] = torch.from_numpy(p).to(device)
        i += len(p)
    return out


def batches(data, device, rng, size, flip=True):
    """Random windows of random crops on the GPU: (inputs, targets in window pixels)."""
    import torch
    count = data.shape[0]
    span, mid = CROP - WINDOW, (CROP - 1) / 2.0
    ar = torch.arange(WINDOW, device=device)
    idx = torch.from_numpy(rng.integers(0, count, size)).to(device)
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
        win = torch.where(back[:, None, None, None], win[..., [6, 7, 8, 3, 4, 5, 0, 1, 2]], win)
    x = win.permute(0, 3, 1, 2).float().div_(255.0).sub_(0.5)
    if flip:
        x = x * torch.empty(size, 1, 1, 1, device=device).uniform_(0.7, 1.3)
    return x.contiguous(memory_format=torch.channels_last), torch.stack([tx, ty], 1)


def fit(data, epochs: int, device, rng, log, label: str):
    """One network trained on `data` (crops already on the device)."""
    import torch
    net = network().to(device).to(memory_format=torch.channels_last)
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
            x, target = batches(data, device, rng, batch)
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


def score(net, data, device, rng, repeats: int = 2):
    """Errors (window px = 1080p px) and confidences from random starts up to 16 px off."""
    import torch
    errors, confs = [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        for _ in range(repeats):
            for i in range(0, len(data), 1024):
                part = data[i:i + 1024]
                x, target = batches(part, device, rng, len(part), flip=False)
                xy, logit = net(x)
                errors.append((xy.float() - target).norm(dim=1).cpu().numpy())
                confs.append(torch.sigmoid(logit.float()).cpu().numpy())
    return np.concatenate(errors), np.concatenate(confs)


def sure_threshold(errors: np.ndarray, confs: np.ndarray) -> float:
    """The lowest confidence at which, counting every frame at least that confident, at most
    SURE_FAR_SHARE of them are more than 5 px off."""
    order = np.argsort(-confs)
    far = np.cumsum(errors[order] > 5) / np.arange(1, len(order) + 1)
    ok = np.where(far <= SURE_FAR_SHARE)[0]
    return float(confs[order][ok[-1]]) if len(ok) else 1.0


def train(epochs: int = EPOCHS, log=None, seed: int = 0) -> Optional[Path]:
    """Calibrate on folds of clips each test network never saw, train on every checked frame, save
    a new version and make it current if it passes."""
    import torch
    log = log or (lambda *parts: print(*parts, flush=True))
    log("gathering the frames you have checked ...")
    crops = gather(log)
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
        net = fit(train_part, epochs, device, rng, log, f"test {k + 1}/{FOLDS}")
        del train_part
        e, c = score(net, to_device([crops[c] for c in sorted(crops) if fold_of[c] == k], device), device,
                     np.random.default_rng(k + 1))
        errors.append(e)
        confs.append(c)
    errors, confs = np.concatenate(errors), np.concatenate(confs)
    threshold = sure_threshold(errors, confs)
    sure = confs >= threshold
    median, far = float(np.median(errors)), float(np.mean(errors > 5))
    sure_far = float(np.mean(errors[sure] > 5)) if sure.any() else 0.0
    log(f"on clips each test network never saw: median {median:.2f} px, {far:.1%} over 5 px; "
        f"{sure.mean():.0%} sure, {sure_far:.2%} of those over 5 px")
    net = fit(to_device(list(crops.values()), device), epochs, device, rng, log, "final")
    previous = current()
    MODELS.mkdir(parents=True, exist_ok=True)
    path = MODELS / f"refiner-{time.strftime('%Y%m%d-%H%M%S')}.pt"
    torch.save({"state": net.state_dict(), "self_check": True, "threshold": threshold,
                "trained": time.strftime("%Y-%m-%d %H:%M:%S"), "clips": len(crops), "frames": int(total),
                "check": {"median_px": median, "over5": far, "sure": float(sure.mean()), "sure_over5": sure_far}},
               path)
    if median > CHECK_MEDIAN_PX or far > CHECK_FAR_SHARE:
        log(f"saved {path.name}, but it failed the check (needs median <= {CHECK_MEDIAN_PX} px and "
            f"<= {CHECK_FAR_SHARE:.0%} over 5 px), so the label tool keeps "
            f"{previous.stem if previous else 'the motion centroid'}")
        return path
    CURRENT.write_text(path.name, encoding="utf-8")
    log(f"saved {path.name}; the label tool uses it from its next start "
        f"(older versions stay in {MODELS.relative_to(HERE.parent)})")
    return path


def describe() -> List[str]:
    import torch
    lines, active = [], current()
    for path in versions():
        try:
            meta = torch.load(path, map_location="cpu")
            check = meta.get("check", {})
            info = (f"trained {meta.get('trained', '?')}, {meta.get('frames', '?')} frames from "
                    f"{meta.get('clips', '?')} clips, median {check.get('median_px', float('nan')):.2f} px")
            if meta.get("self_check"):
                info += f" on clips it never saw, {check.get('sure', 0):.0%} sure"
            else:
                info += " on frames of its own clips (looks better than it is), no self-check"
        except Exception as error:
            info = f"unreadable ({error})"
        lines.append(f"{'*' if path == active else ' '} {path.stem}  {info}")
    return lines or ["  no versions yet - train one"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    trainer = sub.add_parser("train", help="train a new version on every frame you have checked")
    trainer.add_argument("--epochs", type=int, default=EPOCHS)
    sub.add_parser("list", help="list the versions; * is the one the label tool uses")
    chooser = sub.add_parser("use", help="make an older (or any) version the current one")
    chooser.add_argument("version")
    args = parser.parse_args(argv)
    if args.command == "train":
        return 0 if train(args.epochs) else 1
    if args.command == "list":
        print("\n".join(describe()))
        return 0
    print(f"the label tool now uses {use(args.version).stem}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
