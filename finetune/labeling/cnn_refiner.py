"""A small CNN that learns where you click the ball, used to place the label tool's suggestions.

refine.py still finds the ball (the motion blobs, the path through the neighbours) and says whether
a ball is there at all; this network then puts it where you would click, from the label frame and
the label frames either side cut 64x64 around that point, and says how sure it is (a second head
predicting whether its own answer is within 2 px). It learns only from frames you have checked in
the label tool (reviewed or edited), so retraining as you label keeps it current:

    python finetune/labeling/cnn_refiner.py train        # or "hint model" in the finetune menu
    python finetune/labeling/cnn_refiner.py list
    python finetune/labeling/cnn_refiner.py use <version>
    python finetune/labeling/cnn_refiner.py sheet <clip> [<version> <version>|centroid]   # judge two by eye

Training (``refiner_training``) first trains three networks that each leave a third of the clips
out (TrackNet clips of one game together) and scores them on those: that sets how sure is sure
enough (at most 0.1% of the "sure" frames more than 5 px off) and is the check a version must pass.
Then it trains on everything. Every version is kept in finetune/models/refiner/ and the new one
becomes the one the label tool uses (current.txt) once it passes; `use` switches back to any older
version. This module holds the crops, the network, the ``Placer`` the label tool loads, the version
bookkeeping, the comparison sheet and the command line.

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
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

if TYPE_CHECKING:
    import torch
    from torch import nn

if __name__ == "__main__":          # run as a script: make the repository root importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from finetune.data import workspace as ws
from finetune.data.fileio import write_atomic
from finetune.data.labels import read_labels
from finetune.data.scoring import REFERENCE_HEIGHT
from finetune.data.workspace import find_video, label_csv, video_meta
from finetune.labeling import refine

MODELS = ws.MODELS / "refiner"
CACHE = ws.CACHE / "refiner"
CURRENT_NAME = "current.txt"     # in MODELS: names the version the label tool uses
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
        print(f"{clip}: no video or labels under {ws.WORKSPACE}")
        return None
    width, height = video_meta(video)[1:3]
    seeds = read_labels(labels_csv, width, height)
    if not seeds:
        print(f"{clip}: no labels")
        return None
    found = [placed_by(name, video, seeds) for name in names]
    frames = frames_furthest_apart(found, seeds, REFERENCE_HEIGHT / height, count)
    if not frames:
        print(f"{clip}: no frame has a label and a suggestion from both")
        return None
    marks_of = {f: [seeds[f], found[0][f], found[1][f]] for f in frames}
    tiles = sheet_tiles(video, marks_of, (width, height))
    board = sheet_board(tiles, clip, names)
    out = out or (CACHE.parent / "sheets" / f"{clip}_{names[0]}_vs_{names[1]}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), board)
    print(f"wrote {out}")
    return out


def placed_by(name: str, video: Path, seeds: Dict[int, Optional[Tuple[float, float]]]) -> Dict[int, Tuple[float, float]]:
    """frame -> where the version called `name` (or "centroid") puts the ball on this clip."""
    hints = refine.suggest(video, seeds, refine.PassOptions(refine.label_spacing(seeds), placer=placer_for(name)))
    found = {f: (h["x"], h["y"]) for f, h in hints.items() if h["ball_here"]}
    print(f"  {name}: {len(found)} suggestions", flush=True)
    return found


def frames_furthest_apart(found: Sequence[Dict[int, Tuple[float, float]]],
                          seeds: Dict[int, Optional[Tuple[float, float]]], to_1080: float, count: int) -> List[int]:
    """The `count` labelled frames where the two versions' marks differ most, in frame order."""
    shared = sorted(f for f in found[0] if f in found[1] and seeds.get(f) is not None)
    apart = {f: math.hypot(found[0][f][0] - found[1][f][0], found[0][f][1] - found[1][f][1]) * to_1080 for f in shared}
    return sorted(sorted(shared, key=lambda f: -apart[f])[:count])


SHEET_STRIP = 18          # the caption strip above each tile, in canvas px
SHEET_HEADER = 44         # the title and legend above the board


def sheet_tile(image: np.ndarray, frame: int, marks: Sequence[Tuple[float, float]], size: Tuple[int, int]) -> np.ndarray:
    """One tile: the SHEET_PATCH px (1080p) around the three marks, zoomed SHEET_ZOOM times with
    the pixel grid intact, the crosses on top and a caption with each mark's distance from the label."""
    width, height = size
    to_1080 = REFERENCE_HEIGHT / height
    side = max(8, int(round(SHEET_PATCH * height / REFERENCE_HEIGHT)))
    tile = SHEET_PATCH * SHEET_ZOOM
    scale = tile / float(side)                # canvas px per frame px
    mx, my = (sum(m[0] for m in marks) / 3.0), (sum(m[1] for m in marks) / 3.0)
    x0 = min(max(0, int(math.floor(mx)) - side // 2), max(0, width - side))
    y0 = min(max(0, int(math.floor(my)) - side // 2), max(0, height - side))
    patch = cv2.resize(image[y0:y0 + side, x0:x0 + side], (tile, tile), interpolation=cv2.INTER_NEAREST)
    for key, mark in reversed(list(zip(("label", "a", "b"), marks))):     # your label on top
        draw_hollow_cross(patch, mark, (x0, y0), scale, SHEET_COLOURS[key])
    canvas = np.zeros((tile + SHEET_STRIP, tile, 3), np.uint8)
    canvas[SHEET_STRIP:] = patch
    da, db = (math.hypot(m[0] - marks[0][0], m[1] - marks[0][1]) * to_1080 for m in marks[1:])
    cv2.putText(canvas, f"f{frame}  {da:.1f} / {db:.1f} px", (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (230, 230, 230), 1, cv2.LINE_AA)
    return canvas


def sheet_tiles(video: Path, marks_of: Dict[int, List[Tuple[float, float]]], size: Tuple[int, int]) -> List[np.ndarray]:
    """A tile per frame of `marks_of` (frame -> [label, first version's mark, second's]), read in
    one pass over the video; frames the video does not reach are left out."""
    tiles = []
    capture = cv2.VideoCapture(str(video))
    index, wanted = 0, set(marks_of)
    while wanted:
        if index in wanted:
            ok, image = capture.read()
        else:
            ok, image = capture.grab(), None
        if not ok:
            break
        if image is not None:
            wanted.discard(index)
            tiles.append(sheet_tile(image, index, marks_of[index], size))
        index += 1
    capture.release()
    return tiles


def sheet_board(tiles: Sequence[np.ndarray], clip: str, names: Sequence[str]) -> np.ndarray:
    """The tiles laid out in SHEET_COLUMNS columns under a title and the colour legend."""
    tile = SHEET_PATCH * SHEET_ZOOM
    columns = min(SHEET_COLUMNS, len(tiles))
    rows = (len(tiles) + columns - 1) // columns
    board = np.zeros((SHEET_HEADER + rows * (tile + SHEET_STRIP + 4), columns * (tile + 4), 3), np.uint8)
    cv2.putText(board, f"{clip}: the {len(tiles)} label frames where the two differ most, {SHEET_ZOOM}x, "
                       f"frame pixels as they are", (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    legend = (f"white = your label   orange = {names[0]}   magenta = {names[1]}   "
              f"caption: orange / magenta distance from your label")
    cv2.putText(board, legend, (6, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    for i, t in enumerate(tiles):
        r, c = divmod(i, columns)
        y, x = SHEET_HEADER + r * (tile + SHEET_STRIP + 4), c * (tile + 4)
        board[y:y + tile + SHEET_STRIP, x:x + tile] = t
    return board


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
    # The trainer imports this module for the crops, the network and the version bookkeeping,
    # so the command line reaches it from here rather than at import.
    from finetune.labeling import refiner_training
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    trainer = sub.add_parser("train", help="train a new version on every frame you have checked")
    trainer.add_argument("--epochs", type=int, default=refiner_training.EPOCHS)
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
            return 0 if refiner_training.train(args.epochs) else 1
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
