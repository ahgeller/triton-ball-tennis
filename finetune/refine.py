"""Sub-pixel ball suggestions from the video itself, for the click tool to offer.

A tennis ball in a broadcast-style clip is a few pixels of motion blur, and what a careful click
lands on is the centre of that blur. The ball is the thing that moves between label frames, so the
change between a frame and its neighbours finds it: the per-pixel minimum of |I_t - I_prev| and
|I_t - I_next| keeps what is different from *both* neighbours - the ball where it is now - and drops
the ghosts of where it was and will be. Its weighted centroid in a small window around the current
label, iterated a few times, is the suggestion. CPU only, no model: nothing here was trained on
these clips, so it is as good on new footage as on old.

Measured against the hand-verified clips (video13 and up), seeded from the label being corrected,
the suggestion sits a median 0.8 px (at 1080p) from the hand label - the same as the scatter between
two careful clicks on the same frame.

A suggestion is a measurement of the picture, not a fitted path. The smoothness check below uses
a parabola through the neighbouring suggestions only to say how far to trust this one; the point
offered is always the centroid itself.

Frame coordinates follow the label files: pixel k's centre is at k + 0.5.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np

ITERATIONS = 3             # centroid -> recentre the window -> centroid ...
RADIUS_720P = 9            # search half-width at 720p; scaled with the clip height
CORE_SHARE = 0.5           # keep the part of the change map above this share of its peak
SMOOTH_PX_1080 = 1.5       # a suggestion this close (at 1080p) to the curve through its neighbours
MOVE_PX_1080 = 3.0         # ... and this close to where the label already was, "fits"


def radius_for(height: int) -> int:
    return max(4, int(round(RADIUS_720P * height / 720.0)))


def motion_centroid(previous: np.ndarray, current: np.ndarray, following: np.ndarray,
                    seed: Tuple[float, float], radius: int) -> Optional[Tuple[float, float]]:
    """The centre of what changed at `seed` relative to both neighbours, or None if nothing did.
    Images are the full frames (any dtype, same shape); `seed` and the result are in label
    coordinates."""
    height, width = current.shape[:2]
    x, y = seed
    for _ in range(ITERATIONS):
        cx, cy = int(math.floor(x)), int(math.floor(y))       # the pixel holding the point
        x0, x1 = max(0, cx - radius), min(width, cx + radius + 1)
        y0, y1 = max(0, cy - radius), min(height, cy + radius + 1)
        if x0 >= x1 or y0 >= y1:
            return None
        here = current[y0:y1, x0:x1].astype(np.int16)
        change = np.minimum(np.abs(here - previous[y0:y1, x0:x1]).sum(axis=2),
                            np.abs(here - following[y0:y1, x0:x1]).sum(axis=2)).astype(np.float32)
        peak = float(change.max())
        if peak <= 0.0:
            return None
        weight = np.maximum(change - CORE_SHARE * peak, 0.0)
        total = float(weight.sum())
        if total <= 0.0:
            return None
        ys, xs = np.mgrid[y0:y1, x0:x1]
        x = float((weight * xs).sum() / total) + 0.5            # index -> pixel centre
        y = float((weight * ys).sum() / total) + 0.5
    return x, y


def smoothness(points: Dict[int, Tuple[float, float]], frame: int, spacing: int) -> Optional[float]:
    """How far `points[frame]` is from a parabola through the suggestions two label frames either
    side of it, or None when any of those four is missing."""
    neighbours = [frame - 2 * spacing, frame - spacing, frame + spacing, frame + 2 * spacing]
    if frame not in points or any(n not in points for n in neighbours):
        return None
    t = np.array(neighbours, dtype=np.float64)
    xs = np.array([points[n][0] for n in neighbours])
    ys = np.array([points[n][1] for n in neighbours])
    px = float(np.polyval(np.polyfit(t, xs, 2), frame))
    py = float(np.polyval(np.polyfit(t, ys, 2), frame))
    return math.hypot(px - points[frame][0], py - points[frame][1])


def label_spacing(frames: Iterable[int]) -> int:
    """The usual gap between label frames: 1 on clips labelled every frame, 2 on 60 FPS clips."""
    ordered = sorted(set(frames))
    gaps = [b - a for a, b in zip(ordered, ordered[1:]) if b > a]
    if not gaps:
        return 1
    return max(1, int(np.median(gaps)))


def suggest(video, seeds: Dict[int, Tuple[float, float]], spacing: Optional[int] = None,
            stop=lambda: False) -> Dict[int, dict]:
    """A suggestion for every label frame that has a seed, reading the video once from the start.

    `seeds` maps a frame to the point to start from (normally the current label). Returns
    frame -> {"x", "y", "move", "smooth", "fits"}, where `move` is how far the suggestion is from the
    seed and `smooth` how far it is from the curve through its neighbours, both scaled to 1080p
    pixels; `fits` says both are small. Frames whose picture shows nothing moving are left out.
    `stop` is polled between frames so a caller can abandon the pass.
    """
    if not seeds:
        return {}
    spacing = spacing or label_spacing(seeds)
    capture = cv2.VideoCapture(str(video))
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            return {}
        radius = radius_for(height)
        to_1080 = 1080.0 / height
        wanted = set()
        for frame in seeds:
            wanted.update((frame - spacing, frame, frame + spacing))
        last = max(seeds) + spacing
        kept: Dict[int, np.ndarray] = {}
        points: Dict[int, Tuple[float, float]] = {}
        moved: Dict[int, float] = {}
        index = 0
        while index <= last and not stop():
            if index not in wanted:
                if not capture.grab():
                    break
                index += 1
                continue
            ok, image = capture.read()
            if not ok:
                break
            kept[index] = image
            centre = index - spacing                # every frame it needs is now in hand
            if centre in seeds and centre - spacing in kept:
                found = motion_centroid(kept[centre - spacing], kept[centre], image, seeds[centre], radius)
                if found is not None:
                    points[centre] = found
                    moved[centre] = math.hypot(found[0] - seeds[centre][0], found[1] - seeds[centre][1])
            for old in [k for k in kept if k < centre]:
                del kept[old]
            index += 1
    finally:
        capture.release()
    out = {}
    for frame, (x, y) in points.items():
        smooth = smoothness(points, frame, spacing)
        move = moved[frame] * to_1080
        smooth = None if smooth is None else smooth * to_1080
        out[frame] = {"x": round(x, 3), "y": round(y, 3), "move": round(move, 3),
                      "smooth": None if smooth is None else round(smooth, 3),
                      "fits": smooth is not None and smooth <= SMOOTH_PX_1080 and move <= MOVE_PX_1080}
    return out
