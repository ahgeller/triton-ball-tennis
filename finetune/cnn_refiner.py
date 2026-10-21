"""A small CNN that learns where you click the ball, used to place the label tool's suggestions.

refine.py still finds the ball (the motion blobs, the path through the neighbours) and says whether
a ball is there at all; this network then puts it where you would click, from the label frame and
the label frames either side cut 64x64 around that point, and says how sure it is (a second head
predicting whether its own answer is within 2 px). It learns only from frames you have checked in
the label tool (reviewed or edited), so retraining as you label keeps it current:

    python finetune/cnn_refiner.py train        # or "hint model" in the finetune menu
    python finetune/cnn_refiner.py list
    python finetune/cnn_refiner.py use <version>
    python finetune/cnn_refiner.py sheet <clip> [<version> <version>|centroid]   # judge two by eye

Training first trains three networks that each leave a third of the clips out (TrackNet clips of
one game together) and scores them on those: that sets how sure is sure enough (at most 0.1% of the
"sure" frames more than 5 px off) and is the check a version must pass. Then it trains on everything.
Every version is kept in finetune/models/refiner/ and the new one becomes the one the label tool
uses (current.txt) once it passes; `use` switches back to any older version.

Measured on 66 reviewed clips, each scored by a network that never saw it: a median 0.66 px from
your clicks against 0.79 px for refine.py's centroid alone, and 64% of frames sure with 0.09% of
those over 5 px off (refine.py's own trust rule: 55% sure at the same 0.09%). Placing looks at each
window four ways (as is, mirrored, frames reversed, both) and averages: 3% closer again.

What did not help (measured on 13,562 clicked frames, 2026-10-19): fitting a parabola or cubic
through the hints of the label frames around each one and offering the fit, or a blend of it,
moves the marks further from the clicks in every variant (0.455 -> 0.485 px median on sure frames,
far worse at hits); the hints are no smoother along the path than the clicks themselves (both
0.84 px from a parabola through the neighbouring clicks). Giving the network five frames instead
of three places no closer (0.600 px median either way). Three networks averaged place 4% closer
and cut the frames over 5 px off from 3.4% to 2.6%, at three times the training time."""
from __future__ import annotations

import argparse
import math
import re
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np

if TYPE_CHECKING:
    import torch
    from torch import nn

if __name__ == "__main__":          # run as a script: make the repository root importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from finetune import refine
from finetune.fileio import write_atomic
from finetune.labels import Labels, read_labels, reviewed_frames
from finetune.scoring import REFERENCE_HEIGHT
from finetune.workspace import (MANIFEST, WORKSPACE as HERE, find_video, label_csv, labelled_clips, read_manifest,
                                review_json, video_meta)
from finetune.workspace import CACHE as WORKSPACE_CACHE, MODELS as WORKSPACE_MODELS

MODELS = WORKSPACE_MODELS / "refiner"
CACHE = WORKSPACE_CACHE / "refiner"
CURRENT_NAME = "current.txt"     # in MODELS: names the version the label tool uses
CROP = 96            # cut around a point at 1080p scale ...
WINDOW = 64          # ... and the network sees this much of it, so the ball may sit 16 px off centre
EPOCHS = 40
CHECK_MEDIAN_PX, CHECK_FAR_SHARE = 1.0, 0.05   # a new version must beat these on clips it never saw
SURE_PX = 2.0            # the self-check learns whether its answer is within this
SURE_FAR_SHARE = 0.001   # ... and "sure" is set so at most this share of sure frames is over 5 px off
FOLDS = 3
BACKWARDS = [6, 7, 8, 3, 4, 5, 0, 1, 2]   # the 9 channels with the three frames in reverse order
PLACE_BATCH = 512          # windows placed per network call ...
SCORE_BATCH = 1024         # ... and scored per call
FAR_PX = 5.0               # a placement this far off (1080p px) counts as a miss in the self-check


# ---------------------------------------------------------------------------------------- crops
def cut_crop(images: Sequence[np.ndarray], point: Tuple[float, float], height: int) -> np.ndarray:
    """uint8 CROP x CROP x 9: the three frames (before, label frame, after) around `point`, scaled
    to 1080p. Crop pixel j sits at point + (j - (CROP - 1) / 2) * height / 1080 in the image."""
    side = int(round(CROP * height / REFERENCE_HEIGHT))
    out = []
    for image in images:
        patch = cv2.getRectSubPix(image, (side, side), (point[0] - 0.5, point[1] - 0.5))
        out.append(patch if side == CROP else cv2.resize(patch, (CROP, CROP), interpolation=cv2.INTER_LINEAR))
    return np.concatenate(out, axis=2)


def cut_window(crop: np.ndarray, centre: Tuple[float, float]) -> np.ndarray:
    """WINDOW x WINDOW x 9 of a crop, centred (sub-pixel) at crop coordinates `centre`."""
    return np.concatenate([cv2.getRectSubPix(np.ascontiguousarray(crop[..., i:i + 3]), (WINDOW, WINDOW), centre)
                           for i in (0, 3, 6)], axis=2)


# -------------------------------------------------------------------------------------- network
def build_network(self_check: bool = True) -> "nn.Module":
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


def to_tensor(windows: np.ndarray, device: str) -> "torch.Tensor":
    import torch
    return torch.from_numpy(np.ascontiguousarray(windows)).to(device).permute(0, 3, 1, 2).float().div_(255.0).sub_(0.5)


class Placer:
    """Puts suggestions where you would click. `place` takes (crop, crop_point, start) per frame:
    a crop from `cut_crop` around crop_point, and the point to start from (within about 16 px of it), and
    returns (x, y, sure) - sure is None for a version without the self-check, else whether its
    confidence reaches the version's threshold."""
    ROUNDS = 2
    cut_crop = staticmethod(cut_crop)

    def __init__(self, path: Path):
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        saved = torch.load(path, map_location=self.device, weights_only=True)
        self.net = build_network(bool(saved.get("self_check"))).to(self.device)
        self.net.load_state_dict(saved["state"])
        self.net.eval()
        self.threshold = saved.get("threshold") if saved.get("self_check") else None
        self.name = path.stem

    def predict_averaged(self, x: "torch.Tensor") -> Tuple["torch.Tensor", Optional["torch.Tensor"]]:
        """The network's answer averaged over the window, its mirror image, its frames reversed and
        both (each mapped back): 3% closer to the click than one look (0.600 -> 0.582 px median on
        the folds). The self-check reads the plain look only, so its calibration stands."""
        xy, logit = self.net(x)
        xy = xy.float()
        for mirror, back in ((True, False), (False, True), (True, True)):
            v = x.flip(3) if mirror else x
            v = v[:, BACKWARDS] if back else v
            vxy, _ = self.net(v)
            vxy = vxy.float()
            if mirror:
                vxy[:, 0] = WINDOW - 1 - vxy[:, 0]
            xy = xy + vxy
        return xy / 4.0, logit

    def place(self, items: List[Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]],
              height: int) -> List[Tuple[float, float, Optional[bool]]]:
        import torch
        if not items:
            return []
        scale = height / REFERENCE_HEIGHT
        mid, lo, hi = (CROP - 1) / 2.0, (WINDOW - 1) / 2.0, CROP - 1 - (WINDOW - 1) / 2.0
        # where each window is centred, in crop pixels (kept inside the crop)
        centres = [(min(hi, max(lo, mid + (s[0] - c[0]) / scale)), min(hi, max(lo, mid + (s[1] - c[1]) / scale)))
                   for _, c, s in items]
        sure = [None] * len(items)
        for _ in range(self.ROUNDS):
            moved = []
            with torch.no_grad():
                for i in range(0, len(items), PLACE_BATCH):
                    batch = np.stack([cut_window(items[j][0], centres[j]) for j in range(i, min(len(items), i + PLACE_BATCH))])
                    xy, logit = self.predict_averaged(to_tensor(batch, self.device))
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


def current_file() -> Path:
    """current.txt, looked up when asked so a redirected MODELS (tests) is honoured."""
    return MODELS / CURRENT_NAME


def current() -> Optional[Path]:
    """The version the label tool uses: the one current.txt names, else the newest."""
    marker = current_file()
    if marker.is_file():
        named = MODELS / marker.read_text(encoding="utf-8").strip()
        if named.is_file():
            return named
    found = versions()
    return found[-1] if found else None


def version_named(name: str) -> Path:
    """The one version `name` picks out (its stem, file name or a unique part of the stem)."""
    match = [p for p in versions() if p.stem == name or p.name == name or name in p.stem]
    if len(match) != 1:
        raise ValueError(f"no single version matches {name!r}; `list` shows them")
    return match[0]


def make_current(name: str) -> Path:
    chosen = version_named(name)
    write_atomic(current_file(), chosen.name)
    return chosen


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
def checked_frames(clip: str) -> Set[int]:
    """Frames you have reviewed or edited in the label tool (a damaged review file is an error)."""
    return reviewed_frames(review_json(clip))


def clip_crops(clip: str) -> Optional[np.ndarray]:
    """Crops around every checked, visible label of a clip; cached until its labels or review change."""
    labels_csv, review = label_csv(clip), review_json(clip)
    video = find_video(clip)
    if video is None or not review.is_file():
        return None
    cached = CACHE / f"{clip}.npz"
    stamp = max(labels_csv.stat().st_mtime, review.stat().st_mtime, video.stat().st_mtime)
    if cached.is_file() and cached.stat().st_mtime >= stamp:
        crops = np.load(cached)["crops"]
        if crops.shape[1:] == (CROP, CROP, 9):      # not cut for an older crop size
            return crops
    meta = video_meta(video)
    labels: Labels = read_labels(labels_csv, meta.width, meta.height)
    step = refine.label_spacing(labels)
    points = {f: labels[f] for f in checked_frames(clip) if labels.get(f)}
    if not points:
        return None
    height = meta.height
    capture = cv2.VideoCapture(str(video))
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
                crops.append(cut_crop([kept.get(g, kept[f]) for g in (f - step, f, f + step)], points[f], height))
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
    origin = {clip: row["origin"] for clip, row in read_manifest(MANIFEST).items()}
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
    write_atomic(current_file(), path.name)
    log(f"saved {path.name}; the label tool uses it from its next start "
        f"(older versions stay in {MODELS.relative_to(HERE.parent)})")
    return path


# ---------------------------------------------------------------------------------------- sheet
SHEET_PATCH, SHEET_ZOOM, SHEET_COLUMNS = 32, 10, 6   # a sheet tile is PATCH px (1080p) around the marks, ZOOM x
SHEET_COLOURS = {"label": (255, 255, 255), "a": (0, 165, 255), "b": (255, 0, 255)}      # BGR: white, orange, magenta


def placer_for(name: str) -> Optional[Placer]:
    """A Placer for a version name (`version_named`'s matching), or None for "centroid" (refine.py alone)."""
    return None if name == "centroid" else Placer(version_named(name))


def draw_hollow_cross(canvas: np.ndarray, point: Tuple[float, float], origin: Tuple[int, int], zoom: float, colour) -> None:
    """A cross with a hole at `point` (frame coordinates) on a canvas that shows the frame from
    `origin` zoomed `zoom` times, so the pixel under the centre stays visible."""
    cx, cy = (point[0] - origin[0]) * zoom, (point[1] - origin[1]) * zoom
    near, far = 0.5 * zoom, 2.2 * zoom
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        a = (int(round((cx + dx * near) * 16)), int(round((cy + dy * near) * 16)))
        b = (int(round((cx + dx * far) * 16)), int(round((cy + dy * far) * 16)))
        cv2.line(canvas, a, b, (0, 0, 0), 4, cv2.LINE_AA, shift=4)
        cv2.line(canvas, a, b, colour, 2, cv2.LINE_AA, shift=4)


def sheet(clip: str, names: Sequence[str], count: int = 36, out: Optional[Path] = None) -> Optional[Path]:
    """A PNG of the label frames where two versions of the placer (or "centroid") put the ball
    furthest apart, each zoomed SHEET_ZOOM times with the pixel grid intact and three crosses: your
    label (white), the first version (orange) and the second (magenta). For judging by eye which one
    sits on the ball; the captions give each mark's distance from your label in 1080p px."""
    video, labels_csv = find_video(clip), label_csv(clip)
    if video is None or not labels_csv.is_file():
        print(f"{clip}: no video or labels under {HERE}")
        return None
    width, height = video_meta(video)[1:3]
    seeds = read_labels(labels_csv, width, height)
    if not seeds:
        print(f"{clip}: no labels")
        return None
    to_1080 = REFERENCE_HEIGHT / height
    found = []
    for name in names:
        placer = placer_for(name)
        hints = refine.suggest(video, seeds, refine.label_spacing(seeds), placer=placer)
        found.append({f: (h["x"], h["y"]) for f, h in hints.items() if h["ball_here"]})
        print(f"  {name}: {len(found[-1])} suggestions", flush=True)
    shared = sorted(f for f in found[0] if f in found[1] and seeds.get(f) is not None)
    apart = {f: math.hypot(found[0][f][0] - found[1][f][0], found[0][f][1] - found[1][f][1]) * to_1080 for f in shared}
    frames = sorted(sorted(shared, key=lambda f: -apart[f])[:count])
    if not frames:
        print(f"{clip}: no frame has a label and a suggestion from both")
        return None
    side = max(8, int(round(SHEET_PATCH * height / REFERENCE_HEIGHT)))
    tile, strip = SHEET_PATCH * SHEET_ZOOM, 18
    scale = tile / float(side)                # canvas px per frame px
    tiles = []
    capture = cv2.VideoCapture(str(video))
    index, wanted = 0, set(frames)
    while wanted:
        if index in wanted:
            ok, image = capture.read()
        else:
            ok, image = capture.grab(), None
        if not ok:
            break
        if image is not None:
            f = index
            wanted.discard(f)
            marks = [seeds[f], found[0][f], found[1][f]]
            mx, my = (sum(m[0] for m in marks) / 3.0), (sum(m[1] for m in marks) / 3.0)
            x0 = min(max(0, int(math.floor(mx)) - side // 2), max(0, width - side))
            y0 = min(max(0, int(math.floor(my)) - side // 2), max(0, height - side))
            patch = cv2.resize(image[y0:y0 + side, x0:x0 + side], (tile, tile), interpolation=cv2.INTER_NEAREST)
            for key, mark in reversed(list(zip(("label", "a", "b"), marks))):     # your label on top
                draw_hollow_cross(patch, mark, (x0, y0), scale, SHEET_COLOURS[key])
            canvas = np.zeros((tile + strip, tile, 3), np.uint8)
            canvas[strip:] = patch
            da, db = (math.hypot(m[0] - seeds[f][0], m[1] - seeds[f][1]) * to_1080 for m in marks[1:])
            cv2.putText(canvas, f"f{f}  {da:.1f} / {db:.1f} px", (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (230, 230, 230), 1, cv2.LINE_AA)
            tiles.append(canvas)
        index += 1
    capture.release()
    columns = min(SHEET_COLUMNS, len(tiles))
    rows = (len(tiles) + columns - 1) // columns
    header = 44
    board = np.zeros((header + rows * (tile + strip + 4), columns * (tile + 4), 3), np.uint8)
    cv2.putText(board, f"{clip}: the {len(tiles)} label frames where the two differ most, {SHEET_ZOOM}x, "
                       f"frame pixels as they are", (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    legend = (f"white = your label   orange = {names[0]}   magenta = {names[1]}   "
              f"caption: orange / magenta distance from your label")
    cv2.putText(board, legend, (6, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    for i, t in enumerate(tiles):
        r, c = divmod(i, columns)
        y, x = header + r * (tile + strip + 4), c * (tile + 4)
        board[y:y + tile + strip, x:x + tile] = t
    out = out or (CACHE.parent / "sheets" / f"{clip}_{names[0]}_vs_{names[1]}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), board)
    print(f"wrote {out}")
    return out


def version_lines() -> List[str]:
    import torch
    lines, active = [], current()
    for path in versions():
        try:
            meta = torch.load(path, map_location="cpu", weights_only=True)
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    trainer = sub.add_parser("train", help="train a new version on every frame you have checked")
    trainer.add_argument("--epochs", type=int, default=EPOCHS)
    sub.add_parser("list", help="list the versions; * is the one the label tool uses")
    chooser = sub.add_parser("use", help="make an older (or any) version the current one")
    chooser.add_argument("version")
    viewer = sub.add_parser("sheet", help="a PNG of one clip's frames where two versions disagree most, "
                                          "zoomed, to judge by eye which sits on the ball")
    viewer.add_argument("clip")
    viewer.add_argument("versions", nargs="*", help="two version names (or 'centroid'); default: the "
                                                   "current one and the one before it")
    viewer.add_argument("--count", type=int, default=36)
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            return 0 if train(args.epochs) else 1
        if args.command == "list":
            print("\n".join(version_lines()))
            return 0
        if args.command == "sheet":
            names = list(args.versions)
            if not names:
                active = current()
                older = [p for p in versions() if p != active]
                names = [active.stem if active else "centroid", older[-1].stem if older else "centroid"]
            if len(names) != 2:
                parser.error("sheet takes two names")
            return 0 if sheet(args.clip, names, args.count) else 1
        print(f"the label tool now uses {make_current(args.version).stem}")
    except ValueError as error:
        print(f"[error] {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
