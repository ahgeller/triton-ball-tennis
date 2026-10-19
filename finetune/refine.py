"""Sub-pixel ball suggestions from the video itself, for the click tool to offer.

A tennis ball in a broadcast-style clip is a few pixels of motion blur, and what a careful click
lands on is the centre of that blur. The ball is the thing that moves between frames, so the change
between a frame and its neighbours finds it: the per-pixel minimum of |I_t - I_t+o| over the label
frames one and two either side keeps what differs from *all* of them - the ball where it is now -
and drops the ghosts of where it was and will be; at the ends of a clip one side is enough. A
slight weight for yellow pixels keeps the map on the ball rather than a white shirt.

Around each starting point the map's weighted centroid, iterated, is a measurement of the ball. The
starting points are the frame's label and, separately, where the labels either side say the ball
should be: a label on the wrong thing still gets a measurement at the right place, and a frame with
no label at all gets one wherever its neighbours put the ball. Every local peak of the map near a
starting point is a candidate; the candidate the trajectory through the neighbouring suggestions
points at is chosen. The trajectory only *chooses* between blobs - the point offered is always a
blob's centroid, a measurement, never the fitted path itself.

Each suggestion carries how far to trust it: `trusted` when it sits within 2 px (1080p) of the
path through its neighbours' suggestions, it is the only blob around, and its peak is the window's
peak; and `ball_here`, which is false where nothing ball-like moved (an invisible ball). Measured on
the hand-verified clips video13-53, seeded from the detector's drafts: median 0.79 px from the hand
label; the trusted ones (61%) are more than 3 px off 2% of the time and more than 5 px off 0.1%.

With a `placer` (finetune/cnn_refiner.py, a small network trained on your own clicks) the chosen
blob is then placed where you would click it; everything else - finding the ball, choosing between
blobs, trusting - stays this model-free measurement. Without one this runs on the CPU, no model.
Frame coordinates follow the label files: pixel k's centre is at k + 0.5.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

ITERATIONS = 3             # centroid -> recentre the window -> centroid ...
RADIUS_720P = 7            # centroid half-width at 720p; scaled with the clip height. 9 drifted onto
                           # players and rackets next to the ball more often
SEARCH_720P = 12           # candidate blobs are looked for this far around a starting point
CORE_SHARE = 0.5           # keep the part of the change map above this share of its peak
YELLOW_WEIGHT = 0.02       # change *= 1 + this * max(0, (G + R) / 2 - B)
OFFSETS = (1, 2)           # label frames either side whose difference finds the ball. On 60 FPS
                           # clips labelled every second frame, the unlabelled frames in between
                           # were measured too and made it worse (median 0.66 vs 0.58 px): 16 ms
                           # apart, a slow ball's blob overlaps its own ghost
TRAJECTORY_PX_1080 = 12.0  # a candidate further than this from the predicted path is not chosen for it
TRUST_PATH_PX_1080 = 2.0   # "trusted": this close to the path through the neighbours' suggestions
TRUST_PEAK_SHARE = 0.9     # ... its own peak this share of the brightest change in the window
BALL_CONTRAST = 20.0       # peak over the window's median: below this nothing ball-like moved
PLACE_REACH_1080 = 16.0    # the placer sees this far around the point it was cut at
TRUST_PLACE_PX_1080 = 2.0  # a placed suggestion further than this from its measurement is not trusted:
                           # keeps 97% of the trusted frames, and those over 5 px off 0.19% -> 0.08%
AGREE_PX_1080 = 3.0        # a label this close to a trusted suggestion agrees with it


def radius_for(height: int) -> int:
    return max(4, int(round(RADIUS_720P * height / 720.0)))


def search_for(height: int) -> int:
    return max(6, int(round(SEARCH_720P * height / 720.0)))


def change_map(current: np.ndarray, others: List[np.ndarray], box) -> np.ndarray:
    """What differs from every neighbour, inside box = (x0, y0, x1, y1), yellow pixels weighted up."""
    x0, y0, x1, y1 = box
    here = current[y0:y1, x0:x1].astype(np.int16)
    change = np.minimum.reduce([np.abs(here - other[y0:y1, x0:x1]).sum(axis=2) for other in others])
    yellow = np.clip((here[..., 1] + here[..., 2]) / 2.0 - here[..., 0], 0, None)
    return change.astype(np.float32) * (1.0 + YELLOW_WEIGHT * yellow.astype(np.float32))


def centroid(change: np.ndarray, origin, seed, radius: int, iterations: int = ITERATIONS):
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


def blobs(change: np.ndarray, origin, centre, reach: int) -> List[Tuple[float, float, float]]:
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


def describe(change: np.ndarray, origin, point, reach: int, others) -> dict:
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


def predict(points: Dict[int, Tuple[float, float]], frame: int, order: List[int], span: int = 3,
            reach: int = 8):
    """Where the path through the nearest `span` known points either side - within `reach` label
    frames - puts the ball at `frame` (a robust parabola, or a line through three), or None. Only
    ever a starting point or a chooser between measured blobs - never offered as a measurement."""
    if frame not in order:
        return None
    at = order.index(frame)
    before = [g for g in reversed(order[max(0, at - reach):at]) if points.get(g) is not None][:span]
    after = [g for g in order[at + 1:at + 1 + reach] if points.get(g) is not None][:span]
    near = sorted(before + after)
    if len(near) < 3:
        return None
    t = np.array(near, dtype=np.float64)
    xy = np.array([points[g] for g in near], dtype=np.float64)
    degree = 2 if len(near) >= 4 else 1
    for _ in range(2):
        fx, fy = np.polyfit(t, xy[:, 0], degree), np.polyfit(t, xy[:, 1], degree)
        residual = np.hypot(np.polyval(fx, t) - xy[:, 0], np.polyval(fy, t) - xy[:, 1])
        if residual.max() <= 2.0 or len(t) <= degree + 2:
            break
        keep = residual < residual.max()
        t, xy = t[keep], xy[keep]
    return float(np.polyval(fx, frame)), float(np.polyval(fy, frame))


def label_spacing(frames: Iterable[int]) -> int:
    """The usual gap between label frames: 1 on clips labelled every frame, 2 on 60 FPS clips."""
    ordered = sorted(set(frames))
    gaps = [b - a for a, b in zip(ordered, ordered[1:]) if b > a]
    if not gaps:
        return 1
    return max(1, int(np.median(gaps)))


def suggest(video, seeds: Dict[int, Optional[Tuple[float, float]]], spacing: Optional[int] = None,
            stop=lambda: False, confirmed: Optional[Iterable[int]] = None,
            only: Optional[Iterable[int]] = None, placer=None) -> Dict[int, dict]:
    """A suggestion for every label frame, reading the video once from the start.

    `seeds` maps every label frame to its current label, or None where the label says there is no
    ball. Returns frame -> {"x", "y", "seed", "from", "move", "path", "blobs", "ratio", "contrast",
    "trusted", "ball_here"}:
      seed      the label the suggestion was measured from (None for a no-ball label)
      from      "label", or "path" when it was found where the neighbours put the ball
      move      distance from the label, 1080p px (None without one)
      path      distance from the path through the neighbours' suggestions, 1080p px (None if too
                few neighbours)
    `confirmed` names the frames whose labels a person has checked; where enough of them are near,
    only those decide where the path puts the ball, so a run of wrong drafts cannot vouch for
    itself (elsewhere every label does). `only` limits the measuring to those label frames - after
    an edit, a stretch around it - while every label still informs the path. Frames where
    nothing moved are left out. `stop` is polled so a caller can abandon the pass. A `placer`
    (cnn_refiner.Placer) moves each chosen point to where you would click it; "centroid" keeps
    where the measurement itself put it.
    """
    order = sorted(seeds)
    if not order:
        return {}
    capture = cv2.VideoCapture(str(video))
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            return {}
        radius, reach = radius_for(height), search_for(height)
        to_1080 = 1080.0 / height
        labels = {f: p for f, p in seeds.items() if p is not None}
        checked = labels if confirmed is None else {f: labels[f] for f in confirmed if f in labels}
        # Where to look on each frame: its label, and where the labels either side put the ball
        # when that is somewhere else (or the frame has no label).
        centres: Dict[int, List[Tuple[str, Tuple[float, float]]]] = {}
        measured = order if only is None else sorted(set(only) & set(order))
        for frame in measured:
            spots = [("label", labels[frame])] if frame in labels else []
            guess = predict(checked, frame, order) or predict(labels, frame, order)
            if guess is not None and 0 <= guess[0] < width and 0 <= guess[1] < height and (
                    frame not in labels or math.hypot(guess[0] - labels[frame][0], guess[1] - labels[frame][1]) > 2.0):
                spots.append(("path", guess))
            if spots:
                centres[frame] = spots
        if not centres:
            return {}
        step = spacing or label_spacing(order)
        offsets = sorted({sign * o * step for sign in (-1, 1) for o in OFFSETS})
        wanted = {frame + o for frame in centres for o in [0] + offsets}
        pad = reach + radius * (ITERATIONS + 1) + 4
        options: Dict[int, List[dict]] = {}
        crops: Dict[int, List[tuple]] = {}           # frame -> [(centre, crop for the placer)]
        base: Dict[int, Tuple[float, float]] = {}
        kept: Dict[int, np.ndarray] = {}
        pending = sorted(centres)
        last = max(pending) + max(offsets)
        index = 0

        def measure(frame: int) -> None:
            others = [kept[frame + o] for o in offsets if frame + o in kept]
            if len(others) < 2:
                return
            found = []
            for kind, centre in centres[frame]:
                cx, cy = int(math.floor(centre[0])), int(math.floor(centre[1]))
                box = (max(0, cx - pad), max(0, cy - pad), min(width, cx + pad + 1), min(height, cy + pad + 1))
                change = change_map(kept[frame], others, box)
                near = blobs(change, box[:2], centre, reach)
                point = centroid(change, box[:2], centre, radius)
                points = ([(point[0], point[1])] if point is not None else []) + [(b[0], b[1]) for b in near]
                for i, p in enumerate(points):
                    entry = {"x": p[0], "y": p[1], "from": kind, **describe(change, box[:2], p, reach, near)}
                    found.append(entry)
                    if i == 0 and point is not None and frame not in base:
                        base[frame] = p          # the label's own centroid first, else the path's
            if found:
                options[frame] = found
                if placer is not None:
                    images = [kept.get(frame - step, kept[frame]), kept[frame], kept.get(frame + step, kept[frame])]
                    crops[frame] = [(centre, placer.cut(images, centre, height)) for _, centre in centres[frame]]

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
            while pending and pending[0] + max(offsets) <= index:
                measure(pending.pop(0))
            for old in [k for k in kept if pending and k < pending[0] + min(offsets)]:
                del kept[old]
            index += 1
        if not stop():
            for frame in pending:             # the clip ended before their last neighbours
                measure(frame)
    finally:
        capture.release()

    # Choose, per frame, the blob the path through the neighbours' choices points at. Twice, so a
    # choice that moved can pull its neighbours along.
    chosen = {f: p for f, p in base.items()}
    anchors = None if confirmed is None else set(checked)
    for _ in range(2):
        picked = {}
        # With checked labels known, only their measurements steer the choice (as for the starting
        # points); otherwise every frame's current choice does.
        steer = chosen if anchors is None else {f: p for f, p in chosen.items() if f in anchors}
        for frame, found in options.items():
            guess = predict(steer, frame, order) or predict(chosen, frame, order)
            fallback = chosen.get(frame)
            if guess is None:
                picked[frame] = fallback
                continue
            best = min(found, key=lambda o: math.hypot(o["x"] - guess[0], o["y"] - guess[1]))
            near = math.hypot(best["x"] - guess[0], best["y"] - guess[1]) * to_1080 <= TRAJECTORY_PX_1080
            picked[frame] = (best["x"], best["y"]) if near else fallback
        chosen = {f: p for f, p in picked.items() if p is not None}

    placed = {}
    if placer is not None and chosen:
        reach_px = PLACE_REACH_1080 * height / 1080.0
        items, frames = [], []
        for frame, point in chosen.items():
            centre, crop = min(crops.get(frame, []), key=lambda c: math.hypot(c[0][0] - point[0], c[0][1] - point[1]),
                               default=(None, None))
            if centre is not None and math.hypot(centre[0] - point[0], centre[1] - point[1]) <= reach_px:
                items.append((crop, centre, point))
                frames.append(frame)
        placed = dict(zip(frames, placer.place(items, height)))
    out = {}
    for frame, point in chosen.items():
        centroid_point = point
        point = placed.get(frame, point)
        entry = min(options[frame], key=lambda o: math.hypot(o["x"] - point[0], o["y"] - point[1]))
        guess = (predict(chosen if anchors is None else {f: p for f, p in chosen.items() if f in anchors},
                         frame, order) or predict(chosen, frame, order))
        path = None if guess is None else math.hypot(point[0] - guess[0], point[1] - guess[1]) * to_1080
        seed = labels.get(frame)
        move = None if seed is None else math.hypot(point[0] - seed[0], point[1] - seed[1]) * to_1080
        ball_here = entry["contrast"] >= BALL_CONTRAST
        trusted = (ball_here and path is not None and path <= TRUST_PATH_PX_1080 and entry["blobs"] <= 1
                   and entry["ratio"] >= TRUST_PEAK_SHARE
                   and math.hypot(point[0] - centroid_point[0], point[1] - centroid_point[1]) * to_1080
                   <= TRUST_PLACE_PX_1080)
        out[frame] = {"x": round(point[0], 3), "y": round(point[1], 3), "seed": seed, "from": entry["from"],
                      "move": None if move is None else round(move, 3),
                      "path": None if path is None else round(path, 3),
                      "blobs": entry["blobs"], "ratio": round(entry["ratio"], 3),
                      "contrast": round(entry["contrast"], 2), "trusted": bool(trusted),
                      "ball_here": bool(ball_here),
                      "centroid": (round(centroid_point[0], 3), round(centroid_point[1], 3))}
    return out
