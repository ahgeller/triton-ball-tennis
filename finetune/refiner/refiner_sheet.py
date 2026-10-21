"""``ft.py hints sheet``: a zoomed PNG of the label frames where two versions of the placer (or
the motion centroid alone) put the ball furthest apart, with the label and both marks crossed on
each tile, for judging by eye which one sits on the ball."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from finetune.data import workspace as ws
from finetune.data.labels import read_labels
from finetune.data.detector_metrics import REFERENCE_HEIGHT
from finetune.data.workspace import find_video, label_csv, video_meta
from finetune.refiner import refine
from finetune.refiner.cnn_refiner import Placer, version_named

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
        print(f"{clip}: no video or labels under {ws.current().root}")
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
    out = out or (ws.current().cache / "sheets" / f"{clip}_{names[0]}_vs_{names[1]}.png")
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
