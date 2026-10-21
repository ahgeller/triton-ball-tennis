"""What moved around a point on one frame, for ``refine.suggest`` to choose between.

A tennis ball in a broadcast-style clip is a few pixels of motion blur, and what a careful click
lands on is the centre of that blur. The ball is the thing that moves between frames, so the change
between a frame and its neighbours finds it: the per-pixel minimum of |I_t - I_t+o| over the label
frames one and two either side keeps what differs from *all* of them - the ball where it is now -
and drops the ghosts of where it was and will be (``change_map``). A slight weight for yellow pixels
keeps the map on the ball rather than a white shirt. Around a starting point the map's weighted
centroid, iterated, is a measurement of the ball (``centroid``); every local peak nearby is a
candidate (``blobs``), and ``blob_features`` says how ball-like each is.
Frame coordinates follow the label files: pixel k's centre is at k + 0.5.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]            # (x0, y0, x1, y1) of a patch cut from a frame

ITERATIONS = 3             # centroid -> recentre the window -> centroid ...
RADIUS_720P = 7            # centroid half-width at 720p; scaled with the clip height. 9 drifted onto
                           # players and rackets next to the ball more often
SEARCH_720P = 12           # candidate blobs are looked for this far around a starting point
CORE_SHARE = 0.5           # keep the part of the change map above this share of its peak
YELLOW_WEIGHT = 0.02       # change *= 1 + this * max(0, (G + R) / 2 - B)


def centroid_radius_px(height: int) -> int:
    """Half-width of the centroid window, in frame pixels, for a clip this high."""
    return max(4, int(round(RADIUS_720P * height / 720.0)))


def search_radius_px(height: int) -> int:
    """How far around a starting point candidate blobs are looked for, in frame pixels."""
    return max(6, int(round(SEARCH_720P * height / 720.0)))



def change_map(current: np.ndarray, others: List[np.ndarray], box: Box) -> np.ndarray:
    """What differs from every neighbour, inside box = (x0, y0, x1, y1), yellow pixels weighted up."""
    x0, y0, x1, y1 = box
    here = current[y0:y1, x0:x1].astype(np.int16)
    change = np.minimum.reduce([np.abs(here - other[y0:y1, x0:x1]).sum(axis=2) for other in others])
    yellow = np.clip((here[..., 1] + here[..., 2]) / 2.0 - here[..., 0], 0, None)
    return change.astype(np.float32) * (1.0 + YELLOW_WEIGHT * yellow.astype(np.float32))


def centroid(change: np.ndarray, origin: Tuple[int, int], seed: Tuple[float, float], radius: int,
             iterations: int = ITERATIONS) -> Optional[Tuple[float, float]]:
    """Iterated weighted centroid of `change` (a patch whose top-left pixel is at `origin` in the
    frame) around `seed`, in label coordinates, or None if nothing in reach changed."""
    height, width = change.shape
    x, y = seed[0] - origin[0], seed[1] - origin[1]
    for _ in range(iterations):
        cx, cy = int(math.floor(x)), int(math.floor(y))
        x0, x1 = max(0, cx - radius), min(width, cx + radius + 1)
        y0, y1 = max(0, cy - radius), min(height, cy + radius + 1)
        if x0 >= x1 or y0 >= y1:
            return None
        window = change[y0:y1, x0:x1]
        peak = float(window.max())
        if peak <= 0.0:
            return None
        weight = np.maximum(window - CORE_SHARE * peak, 0.0)
        total = float(weight.sum())
        if total <= 0.0:
            return None
        ys, xs = np.mgrid[y0:y1, x0:x1]
        x = float((weight * xs).sum() / total) + 0.5            # index -> pixel centre
        y = float((weight * ys).sum() / total) + 0.5
    return x + origin[0], y + origin[1]


def motion_centroid(previous: np.ndarray, current: np.ndarray, following: np.ndarray,
                    seed: Tuple[float, float], radius: int) -> Optional[Tuple[float, float]]:
    """The centre of what changed at `seed` relative to both neighbours, or None if nothing did."""
    height, width = current.shape[:2]
    pad = radius * (ITERATIONS + 1) + 2
    cx, cy = int(math.floor(seed[0])), int(math.floor(seed[1]))
    box = (max(0, cx - pad), max(0, cy - pad), min(width, cx + pad + 1), min(height, cy + pad + 1))
    return centroid(change_map(current, [previous, following], box), box[:2], seed, radius)


def blobs(change: np.ndarray, origin: Tuple[int, int], centre: Tuple[float, float], reach: int) -> List[Tuple[float, float, float]]:
    """(x, y, peak) of every distinct local peak of `change` within `reach` of `centre`, each
    measured by a small centroid of its own."""
    height, width = change.shape
    cx, cy = int(centre[0] - origin[0]), int(centre[1] - origin[1])
    x0, x1 = max(0, cx - reach), min(width, cx + reach + 1)
    y0, y1 = max(0, cy - reach), min(height, cy + reach + 1)
    if x0 >= x1 or y0 >= y1:
        return []
    smooth = cv2.GaussianBlur(change, (0, 0), 1.0)
    sub = smooth[y0:y1, x0:x1]
    peaks = (sub >= cv2.dilate(sub, np.ones((7, 7), np.uint8))) & (sub >= max(30.0, 0.2 * float(sub.max())))
    found: List[Tuple[float, float, float]] = []
    for y, x in zip(*np.nonzero(peaks)):
        point = centroid(change, origin, (x0 + x + 0.5 + origin[0], y0 + y + 0.5 + origin[1]), 4, 2)
        if point is None or any(math.hypot(point[0] - o[0], point[1] - o[1]) < 2.0 for o in found):
            continue
        found.append((point[0], point[1], float(sub[y, x])))
    return found


def blob_features(change: np.ndarray, origin: Tuple[int, int], point: Tuple[float, float], reach: int,
                  others: List[Tuple[float, float, float]]) -> Dict[str, float]:
    """How ball-like the change at `point` is: its own peak against the window's, against the
    window's background, and how many other blobs compete with it."""
    height, width = change.shape
    px, py = int(point[0] - origin[0]), int(point[1] - origin[1])
    window = change[max(0, py - reach):py + reach + 1, max(0, px - reach):px + reach + 1]
    local = change[max(0, py - 2):py + 3, max(0, px - 2):px + 3]
    if window.size == 0 or local.size == 0:
        return {"ratio": 0.0, "contrast": 0.0, "blobs": 0}
    peak, own = float(window.max()), float(local.max())
    rivals = sum(1 for o in others if math.hypot(o[0] - point[0], o[1] - point[1]) >= 3.0)
    return {"ratio": own / max(peak, 1.0), "contrast": own / (float(np.median(window)) + 5.0),
            "blobs": 1 + rivals}
