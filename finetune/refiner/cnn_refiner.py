"""A small CNN that learns where you click the ball, used to place the label tool's suggestions.

refine.py still finds the ball (the motion blobs, the path through the neighbours) and says whether
a ball is there at all; this network then puts it where you would click, from the label frame and
the label frames either side cut 64x64 around that point, and says how sure it is (a second head
predicting whether its own answer is within 2 px). It learns only from frames you have reviewed in
the label tool (confirmed or edited), so retraining as you label keeps it current (``hints.py``:
``train``, ``list``, ``use``, ``sheet``; ``ft.py hints`` or "hint model" in the finetune menu).

Training (``refiner_training``) first trains three networks that each leave a third of the clips
out (TrackNet clips of one game together) and scores them on those: that sets how sure is sure
enough (at most 0.1% of the "sure" frames more than 5 px off) and is the check a version must pass.
Then it trains on everything. Every version is kept in finetune/models/refiner/ and the new one
becomes the one the label tool uses (current.txt) once it passes; `use` switches back to any older
version. This module holds the crops, the network, the ``Placer`` the label tool loads and the
version bookkeeping.

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

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

import cv2
import numpy as np

if TYPE_CHECKING:
    import torch
    from torch import nn

from finetune.data import workspace as ws
from finetune.data.fileio import write_atomic
from finetune.data.detector_metrics import REFERENCE_HEIGHT

CURRENT_NAME = "current.txt"     # in models_dir(): names the version the label tool uses
CROP = 96            # cut around a point at 1080p scale ...
WINDOW = 64          # ... and the network sees this much of it, so the ball may sit 16 px off centre
SURE_PX = 2.0            # the self-check learns whether its answer is within this
BACKWARDS = [6, 7, 8, 3, 4, 5, 0, 1, 2]   # the 9 channels with the three frames in reverse order
PLACE_BATCH = 512          # windows placed per network call


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

        def forward(self, x: "torch.Tensor") -> Tuple["torch.Tensor", Optional["torch.Tensor"]]:
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
# The folders are read from the workspace when asked, never at import, so a redirected workspace
# (TENNIS_WORKSPACE, the tests' workspace_dir fixture) is honoured.
def models_dir() -> Path:
    """Where the versions and current.txt live: the workspace's models folder, under refiner/."""
    return ws.MODELS / "refiner"


def cache_dir() -> Path:
    """Where the training crops are cached: the workspace's cache folder, under refiner/."""
    return ws.CACHE / "refiner"


def versions() -> List[Path]:
    return sorted(models_dir().glob("refiner-*.pt"))


def current_file() -> Path:
    return models_dir() / CURRENT_NAME


def current() -> Optional[Path]:
    """The version the label tool uses: the one current.txt names, else the newest."""
    marker = current_file()
    if marker.is_file():
        named = models_dir() / marker.read_text(encoding="utf-8").strip()
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
