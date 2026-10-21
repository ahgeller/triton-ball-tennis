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
blob is then placed where you would click it, and a placer with a self-check decides `trusted` by
its own confidence (where a ball is here at all); finding the ball and choosing between blobs stay
this model-free measurement. Without one this runs on the CPU, no model.
Frame coordinates follow the label files: pixel k's centre is at k + 0.5.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, NamedTuple, Optional, Set, Tuple, TypedDict

import cv2
import numpy as np

from finetune.scoring import REFERENCE_HEIGHT

Point = Tuple[float, float]

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
TRUST_PLACE_PX_1080 = 2.0  # without a self-check, a placed suggestion further than this from its
                           # measurement is not trusted: keeps 97% of the trusted frames, and those
                           # over 5 px off 0.19% -> 0.08%
AGREE_PX_1080 = 3.0        # a label this close to a trusted suggestion agrees with it


# One label frame's suggestion, as ``suggest`` returns it (the functional form, because "from" is a
# keyword). Positions are in label coordinates (pixel k's centre at k + 0.5); distances in 1080p px.
Suggestion = TypedDict("Suggestion", {
    "x": float,
    "y": float,
    "seed": Optional[Point],      # the label it was measured from; None for a no-ball label
    "from": str,                  # "label", or "path" when found where the neighbours put the ball
    "move": Optional[float],      # distance from the label; None without one
    "path": Optional[float],      # distance from the path through the neighbours' suggestions;
                                  # None with too few neighbours
    "blobs": int,                 # distinct moving blobs around it, itself included
    "ratio": float,               # its own peak as a share of the brightest change in the window
    "contrast": float,            # its peak over the window's background
    "trusted": bool,              # close to the path, the only blob and the window's peak - or the
                                  # placer's self-check says so
    "ball_here": bool,            # False where nothing ball-like moved (an invisible ball)
    "centroid": Point,            # where the measurement itself put it, before a placer moved it
})


def centroid_radius_px(height: int) -> int:
    """Half-width of the centroid window, in frame pixels, for a clip this high."""
    return max(4, int(round(RADIUS_720P * height / 720.0)))


def search_radius_px(height: int) -> int:
    """How far around a starting point candidate blobs are looked for, in frame pixels."""
    return max(6, int(round(SEARCH_720P * height / 720.0)))


def change_map(current: np.ndarray, others: List[np.ndarray], box) -> np.ndarray:
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


def predict(points: Dict[int, Tuple[float, float]], frame: int, order: List[int], span: int = 3,
            reach: int = 8) -> Optional[Tuple[float, float]]:
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


class Candidate(NamedTuple):
    """One measured blob on a frame: where it is, which starting point found it, how ball-like."""
    x: float
    y: float
    kind: str            # "label" or "path": the starting point it was found around
    ratio: float         # its own peak as a share of the brightest change in the window
    contrast: float      # its peak over the window's background
    blobs: int           # distinct blobs in the window, itself included


@dataclass
class Measurement:
    """What one label frame's change map showed."""
    candidates: List[Candidate]                 # every distinct moving blob near a starting point
    base: Optional[Point]                       # the label's own centroid, else the path's: the first choice
    crops: List[Tuple[Point, np.ndarray]] = field(default_factory=list)   # (starting point, crop) for a placer


@dataclass(frozen=True)
class ClipGeometry:
    """A clip's size and the search sizes that follow from it."""
    width: int
    height: int
    step: int                  # video frames between label frames
    radius: int                # centroid half-width, frame px
    reach: int                 # candidate blobs are looked for this far around a starting point
    pad: int                   # the change map is cut this far around a starting point
    offsets: Tuple[int, ...]   # the neighbours a frame is compared against, relative to it
    to_1080: float             # frame px -> 1080p px

    @classmethod
    def of(cls, width: int, height: int, step: int) -> "ClipGeometry":
        radius, reach = centroid_radius_px(height), search_radius_px(height)
        offsets = tuple(sorted({sign * o * step for sign in (-1, 1) for o in OFFSETS}))
        return cls(width, height, step, radius, reach, reach + radius * (ITERATIONS + 1) + 4, offsets,
                   REFERENCE_HEIGHT / height)

    def box(self, centre: Point) -> Tuple[int, int, int, int]:
        """(x0, y0, x1, y1) of the change map around `centre`, kept inside the frame."""
        cx, cy = int(math.floor(centre[0])), int(math.floor(centre[1]))
        return (max(0, cx - self.pad), max(0, cy - self.pad),
                min(self.width, cx + self.pad + 1), min(self.height, cy + self.pad + 1))


def suggest(video, seeds: Dict[int, Optional[Tuple[float, float]]], spacing: Optional[int] = None,
            stop=lambda: False, confirmed: Optional[Iterable[int]] = None,
            only: Optional[Iterable[int]] = None, placer=None) -> Dict[int, Suggestion]:
    """A suggestion for every label frame, reading the video once from the start.

    `seeds` maps every label frame to its current label, or None where the label says there is no
    ball. Returns frame -> Suggestion (its fields are documented there). `confirmed` names the
    frames whose labels a person has checked; where enough of them are near, only those decide
    where the path puts the ball, so a run of wrong drafts cannot vouch for itself (elsewhere every
    label does). `only` limits the measuring to those label frames - after an edit, a stretch
    around it - while every label still informs the path. Frames where nothing moved are left
    out. `stop` is polled so a caller can abandon the pass. A `placer` (cnn_refiner.Placer) moves
    each chosen point to where you would click it; "centroid" keeps where the measurement itself
    put it.

    The stages: which frames and labels take part (plan), where to look on each frame
    (search_centres), one streamed pass measuring the candidates there (measure_clip), the choice
    between them by the path through the neighbours (choose), the placer (place), and the trust
    rule (is_trusted) on the way out (to_suggestion).
    """
    if not seeds:
        return {}
    plan = plan_pass(seeds, confirmed, only)
    capture = cv2.VideoCapture(str(video))
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            return {}
        geometry = ClipGeometry.of(width, height, spacing or label_spacing(plan.order))
        centres = search_centres(plan, width, height)
        if not centres:
            return {}
        measured = measure_clip(capture, centres, geometry, placer, stop)
    finally:
        capture.release()
    chosen = choose(measured, plan, geometry.to_1080)
    placed = place(chosen, measured, placer, height)
    steer = steering(chosen, plan.anchors)
    out: Dict[int, Suggestion] = {}
    for frame, point in chosen.items():
        guess = path_guess(steer, chosen, frame, plan.order)
        out[frame] = to_suggestion(frame, measured[frame], point, placed, guess, plan.labels.get(frame), geometry.to_1080)
    return out


class Plan(NamedTuple):
    """Which frames a pass measures and which labels steer the path through them."""
    order: List[int]               # every label frame
    labels: Dict[int, Point]       # the frames with a ball
    checked: Dict[int, Point]      # the labels a person has checked (every label when unknown)
    anchors: Optional[Set[int]]    # frames whose choices alone steer the path; None: every frame's
    frames: List[int]              # the label frames to measure


def plan_pass(seeds: Dict[int, Optional[Point]], confirmed: Optional[Iterable[int]],
              only: Optional[Iterable[int]]) -> Plan:
    order = sorted(seeds)
    labels = {f: p for f, p in seeds.items() if p is not None}
    checked = labels if confirmed is None else {f: labels[f] for f in confirmed if f in labels}
    anchors = None if confirmed is None else set(checked)
    frames = order if only is None else sorted(set(only) & set(order))
    return Plan(order, labels, checked, anchors, frames)


def search_centres(plan: Plan, width: int, height: int) -> Dict[int, List[Tuple[str, Point]]]:
    """Where to look on each frame: its label, and where the labels either side put the ball when
    that is somewhere else (or the frame has no label). frame -> [("label" | "path", point)]."""
    labels, centres = plan.labels, {}
    for frame in plan.frames:
        spots = [("label", labels[frame])] if frame in labels else []
        guess = predict(plan.checked, frame, plan.order) or predict(labels, frame, plan.order)
        if guess is not None and 0 <= guess[0] < width and 0 <= guess[1] < height and (
                frame not in labels or math.hypot(guess[0] - labels[frame][0], guess[1] - labels[frame][1]) > 2.0):
            spots.append(("path", guess))
        if spots:
            centres[frame] = spots
    return centres


def read_wanted(capture, wanted: Set[int], last: int, stop) -> Iterator[Tuple[int, np.ndarray]]:
    """(index, image) for each wanted frame up to `last`, in order, grabbing past the rest; ends
    early when the clip does or `stop` says so."""
    index = 0
    while index <= last and not stop():
        if index in wanted:
            ok, image = capture.read()
            if not ok:
                return
            yield index, image
        elif not capture.grab():
            return
        index += 1


def measure_clip(capture, centres: Dict[int, List[Tuple[str, Point]]], geometry: ClipGeometry, placer,
                 stop) -> Dict[int, Measurement]:
    """One pass over the video, keeping only the frames still needed; each label frame is measured
    once the last neighbour it is compared against has been read."""
    offsets = geometry.offsets
    wanted = {frame + o for frame in centres for o in (0,) + offsets}
    pending = sorted(centres)
    kept: Dict[int, np.ndarray] = {}
    measured: Dict[int, Measurement] = {}
    for index, image in read_wanted(capture, wanted, max(pending) + max(offsets), stop):
        kept[index] = image
        measured.update(measure_due(pending, index, kept, centres, geometry, placer))
        for old in [k for k in kept if pending and k < pending[0] + min(offsets)]:
            del kept[old]
    if not stop():
        measured.update(measure_due(pending, math.inf, kept, centres, geometry, placer))   # the clip ended first
    return measured


def measure_due(pending: List[int], upto: float, kept: Dict[int, np.ndarray], centres, geometry: ClipGeometry,
                placer) -> Dict[int, Measurement]:
    """Measure, and take off `pending`, every frame whose last neighbour has been read by `upto`."""
    out: Dict[int, Measurement] = {}
    while pending and pending[0] + max(geometry.offsets) <= upto:
        frame = pending.pop(0)
        others = [kept[frame + o] for o in geometry.offsets if frame + o in kept]
        if len(others) < 2:
            continue
        found = measure_frame(kept[frame], others, centres[frame], geometry)
        if found is None:
            continue
        if placer is not None:
            step = geometry.step
            images = [kept.get(frame - step, kept[frame]), kept[frame], kept.get(frame + step, kept[frame])]
            found.crops = [(centre, placer.cut(images, centre, geometry.height)) for _, centre in centres[frame]]
        out[frame] = found
    return out


def measure_frame(image: np.ndarray, others: List[np.ndarray], spots: List[Tuple[str, Point]],
                  geometry: ClipGeometry) -> Optional[Measurement]:
    """Every candidate on one frame: around each starting point, the centroid of what differs from
    the neighbours and each distinct peak nearby; None where nothing moved."""
    candidates: List[Candidate] = []
    base = None
    for kind, centre in spots:
        box = geometry.box(centre)
        change = change_map(image, others, box)
        near = blobs(change, box[:2], centre, geometry.reach)
        point = centroid(change, box[:2], centre, geometry.radius)
        points = ([(point[0], point[1])] if point is not None else []) + [(b[0], b[1]) for b in near]
        for p in points:
            candidates.append(Candidate(p[0], p[1], kind, **blob_features(change, box[:2], p, geometry.reach, near)))
        if point is not None and base is None:
            base = (point[0], point[1])          # the label's own centroid first, else the path's
    return Measurement(candidates, base) if candidates else None


def steering(chosen: Dict[int, Point], anchors: Optional[Set[int]]) -> Dict[int, Point]:
    """The points that steer the path: with checked labels known, only their measurements (as for
    the starting points); otherwise every frame's current choice."""
    return chosen if anchors is None else {f: p for f, p in chosen.items() if f in anchors}


def path_guess(steer: Dict[int, Point], chosen: Dict[int, Point], frame: int, order: List[int]) -> Optional[Point]:
    """Where the path through the steering points puts the ball at `frame`, falling back to the
    path through every choice when the steering points are too few there."""
    return predict(steer, frame, order) or predict(chosen, frame, order)


def choose(measured: Dict[int, Measurement], plan: Plan, to_1080: float) -> Dict[int, Point]:
    """Per frame, the candidate the path through the neighbours' choices points at, starting from
    each frame's own centroid. Twice, so a choice that moved can pull its neighbours along."""
    chosen = {f: m.base for f, m in measured.items() if m.base is not None}
    for _ in range(2):
        steer = steering(chosen, plan.anchors)
        picked: Dict[int, Optional[Point]] = {}
        for frame, found in measured.items():
            guess = path_guess(steer, chosen, frame, plan.order)
            picked[frame] = nearest_candidate(found.candidates, guess, chosen.get(frame), to_1080)
        chosen = {f: p for f, p in picked.items() if p is not None}
    return chosen


def nearest_candidate(candidates: List[Candidate], guess: Optional[Point], fallback: Optional[Point],
                      to_1080: float) -> Optional[Point]:
    """The candidate nearest `guess` when it is within TRAJECTORY_PX_1080 of it, else `fallback`."""
    if guess is None:
        return fallback
    best = min(candidates, key=lambda c: math.hypot(c.x - guess[0], c.y - guess[1]))
    near = math.hypot(best.x - guess[0], best.y - guess[1]) * to_1080 <= TRAJECTORY_PX_1080
    return (best.x, best.y) if near else fallback


def place(chosen: Dict[int, Point], measured: Dict[int, Measurement], placer,
          height: int) -> Dict[int, Tuple[float, float, Optional[bool]]]:
    """Each chosen point moved by the placer, from the crop cut nearest to it: frame -> (x, y,
    sure). A frame whose crops were all cut too far from its choice is left out; nothing is placed
    without a placer."""
    if placer is None or not chosen:
        return {}
    reach_px = PLACE_REACH_1080 * height / REFERENCE_HEIGHT
    items, frames = [], []
    for frame, point in chosen.items():
        centre, crop = min(measured[frame].crops, key=lambda c: math.hypot(c[0][0] - point[0], c[0][1] - point[1]),
                           default=(None, None))
        if centre is not None and math.hypot(centre[0] - point[0], centre[1] - point[1]) <= reach_px:
            items.append((crop, centre, point))
            frames.append(frame)
    return dict(zip(frames, placer.place(items, height)))


def is_trusted(ball_here: bool, sure: Optional[bool], path: Optional[float], entry: Candidate,
               moved_1080: float) -> bool:
    """A placer's self-check decides (64% sure vs 55% by the rule below, at the same 0.09% of them
    over 5 px off). Without one: close to the path, the only blob, the window's peak, and placed
    within TRUST_PLACE_PX_1080 of its measurement."""
    if sure is not None:
        return bool(ball_here and sure)
    return bool(ball_here and path is not None and path <= TRUST_PATH_PX_1080 and entry.blobs <= 1
                and entry.ratio >= TRUST_PEAK_SHARE and moved_1080 <= TRUST_PLACE_PX_1080)


def to_suggestion(frame: int, measurement: Measurement, centroid_point: Point,
               placed: Dict[int, Tuple[float, float, Optional[bool]]], guess: Optional[Point],
               label: Optional[Point], to_1080: float) -> Suggestion:
    """The record for one frame: the chosen point (placed, when a placer was), its distances from
    the label and the path, the features of the candidate it stands on, and the trust rule."""
    x, y, sure = placed.get(frame, (centroid_point[0], centroid_point[1], None))
    point = (x, y)
    entry = min(measurement.candidates, key=lambda c: math.hypot(c.x - point[0], c.y - point[1]))
    path = None if guess is None else math.hypot(point[0] - guess[0], point[1] - guess[1]) * to_1080
    move = None if label is None else math.hypot(point[0] - label[0], point[1] - label[1]) * to_1080
    ball_here = entry.contrast >= BALL_CONTRAST
    moved = math.hypot(point[0] - centroid_point[0], point[1] - centroid_point[1]) * to_1080
    return {"x": round(point[0], 3), "y": round(point[1], 3), "seed": label, "from": entry.kind,
            "move": None if move is None else round(move, 3),
            "path": None if path is None else round(path, 3),
            "blobs": entry.blobs, "ratio": round(entry.ratio, 3), "contrast": round(entry.contrast, 2),
            "trusted": is_trusted(ball_here, sure, path, entry, moved), "ball_here": bool(ball_here),
            "centroid": (round(centroid_point[0], 3), round(centroid_point[1], 3))}
