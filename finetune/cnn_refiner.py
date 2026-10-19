"""A small CNN that learns where you click the ball, used to place the label tool's suggestions.

refine.py still finds the ball (the motion blobs, the path through the neighbours) and decides
whether a suggestion is trusted; this network then puts it where you would click, from the label
frame and the label frames either side cut 64x64 around that point. It learns only from frames you
have checked in the label tool (reviewed or edited), so retraining as you label keeps it current:

    python finetune/cnn_refiner.py train        # or "hint model" in the finetune menu
    python finetune/cnn_refiner.py list
    python finetune/cnn_refiner.py use <version>

Every training run is kept in finetune/models/refiner/ and the new one becomes the one the label
tool uses (current.txt) once it passes a sanity check; `use` switches back to any older version.
Measured on clips each test network never saw (66 reviewed clips, three folds), it put the ball a
median 0.66 px from your clicks against 0.79 px for refine.py's centroid alone."""
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
CHECK_MEDIAN_PX, CHECK_FAR_SHARE = 1.0, 0.05   # a new version must beat these on its own frames
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
def network():
    import torch.nn as nn
    import torch

    class Net(nn.Module):
        """Fully convolutional: a heatmap over the window, read out as its soft-argmax (sub-pixel)."""

        def __init__(self, width: int = 32):
            super().__init__()
            layers, inp = [], 9
            for dilation in (1, 1, 2, 4, 8, 1):
                layers += [nn.Conv2d(inp, width, 3, padding=dilation, dilation=dilation),
                           nn.BatchNorm2d(width), nn.ReLU()]
                inp = width
            layers += [nn.Conv2d(width, 1, 1)]
            self.body = nn.Sequential(*layers)

        def forward(self, x):
            prob = self.body(x).flatten(1).softmax(1).view(-1, x.shape[2], x.shape[3])
            grid = torch.arange(x.shape[3], device=x.device, dtype=x.dtype)
            return torch.stack([(prob.sum(1) * grid).sum(1), (prob.sum(2) * grid).sum(1)], 1)

    return Net()


def to_tensor(windows: np.ndarray, device):
    import torch
    return torch.from_numpy(np.ascontiguousarray(windows)).to(device).permute(0, 3, 1, 2).float().div_(255.0).sub_(0.5)


class Placer:
    """Puts suggestions where you would click. `place` takes (crop, crop_point, start) per frame:
    a crop from `cut` around crop_point, and the point to start from (within about 16 px of it)."""
    ROUNDS = 2
    cut = staticmethod(cut)

    def __init__(self, path: Path):
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        saved = torch.load(path, map_location=self.device)
        self.net = network().to(self.device)
        self.net.load_state_dict(saved["state"])
        self.net.eval()
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
        for _ in range(self.ROUNDS):
            moved = []
            with torch.no_grad():
                for i in range(0, len(items), 512):
                    batch = np.stack([window(items[j][0], centres[j]) for j in range(i, min(len(items), i + 512))])
                    for k, (px, py) in enumerate(self.net(to_tensor(batch, self.device)).cpu().numpy()):
                        cx, cy = centres[i + k]
                        moved.append((min(hi, max(lo, cx + px - lo)), min(hi, max(lo, cy + py - lo))))
            centres = moved
        return [(c[0] + (cx - mid) * scale, c[1] + (cy - mid) * scale) for (_, c, _), (cx, cy) in zip(items, centres)]


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


def gather(log=lambda *parts: print(*parts, flush=True)) -> Tuple[np.ndarray, int]:
    parts, clips = [], 0
    for labels_csv in sorted(LABELS.glob("*_ball.csv")):
        clip = labels_csv.name[:-len("_ball.csv")]
        crops = clip_crops(clip)
        if crops is not None and len(crops):
            parts.append(crops)
            clips += 1
            log(f"  {clip}: {len(crops)} checked frames")
    return (np.concatenate(parts) if parts else np.zeros((0, CROP, CROP, 9), np.uint8)), clips


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
    return x, torch.stack([tx, ty], 1)


def check(net, data, device, rng, count=2000):
    """Median error and share over 5 px, from random starts up to 16 px off, on held-back crops."""
    import torch
    with torch.no_grad():
        x, target = batches(data, device, rng, min(count, max(1, data.shape[0] * 4)), flip=False)
        error = (net(x) - target).norm(dim=1).cpu().numpy()
    return float(np.median(error)), float(np.mean(error > 5))


def train(epochs: int = EPOCHS, log=None, seed: int = 0) -> Optional[Path]:
    """Train on every checked frame, save a new version and make it current if it passes."""
    import torch
    log = log or (lambda *parts: print(*parts, flush=True))
    log("gathering the frames you have checked ...")
    data, clips = gather(log)
    if len(data) < 500:
        log(f"only {len(data)} checked frames - label more before training")
        return None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    order = rng.permutation(len(data))
    held = max(200, len(data) // 20)                      # 5% kept back for the check
    held_back = torch.from_numpy(data[order[:held]]).to(device)
    train_set = torch.from_numpy(data[order[held:]]).to(device)
    log(f"training on {len(train_set)} frames from {clips} clips ({device}), {epochs} epochs")
    net = network().to(device)
    batch = 256
    per_epoch = max(1, len(train_set) // batch)
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=epochs * per_epoch)
    start = time.time()
    for epoch in range(epochs):
        net.train()
        total = 0.0
        for _ in range(per_epoch):
            x, target = batches(train_set, device, rng, batch)
            loss = torch.nn.functional.smooth_l1_loss(net(x), target, beta=0.5)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
        if epoch % 10 == 9 or epoch == epochs - 1:
            log(f"  epoch {epoch + 1}/{epochs}  loss {total / per_epoch:.3f}  ({time.time() - start:.0f}s)")
    net.eval()
    median, far = check(net, held_back, device, np.random.default_rng(1))
    log(f"check on {held} frames it did not train on: median {median:.2f} px, {far:.1%} over 5 px")
    previous = current()
    if previous is not None:
        try:
            old = Placer(previous)
            pm, pf = check(old.net, held_back, device, np.random.default_rng(1))
            log(f"  the current version ({previous.stem}) on the same frames: median {pm:.2f} px, {pf:.1%} over 5 px")
        except Exception as error:
            log(f"  (could not score {previous.stem}: {error})")
    MODELS.mkdir(parents=True, exist_ok=True)
    path = MODELS / f"refiner-{time.strftime('%Y%m%d-%H%M%S')}.pt"
    torch.save({"state": net.state_dict(), "trained": time.strftime("%Y-%m-%d %H:%M:%S"), "clips": clips,
                "frames": int(len(train_set)), "check": {"median_px": median, "over5": far}}, path)
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
            info = (f"trained {meta.get('trained', '?')}, {meta.get('frames', '?')} frames from "
                    f"{meta.get('clips', '?')} clips, check median {meta.get('check', {}).get('median_px', float('nan')):.2f} px")
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
