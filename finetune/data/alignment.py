"""A clip's labels against the raw detector at a temporal shift.

Both the label audit (``ft.py check --audit``) and ``realign.py`` ask the same question - at
this offset between label frames and video frames, how many labels does the detector agree
with? - so they ask it here, with one set of thresholds (``finetune.data.scoring``). The labels are a
``Tracks`` bundle (frame, position, and whether the ball moved enough since the previous row to
tell one offset from the next); what the detector saw is an ``Evidence`` bundle of per-frame
lookup tables. Only a moving ball can separate one offset from the next - a slow ball stays
within AGREE_PX for several frames either way - so the fast rows are judged when the detector
saw at least MIN_FAST of them, and every row otherwise.
"""
from __future__ import annotations

from typing import Mapping, NamedTuple, Sequence, Tuple, Union

import numpy as np

from finetune.data.scoring import AGREE_PX, FAST_PX, MIN_FAST

Offsets = Union[int, np.ndarray]     # one offset for every row, or an array with one per row


class Evidence(NamedTuple):
    """What the detector saw, as lookup tables by video frame (NaN where it said nothing)."""
    lut_x: np.ndarray
    lut_y: np.ndarray
    total: int        # frames in the video
    scale: float      # the video's pixels to 1080p pixels

    @classmethod
    def of(cls, points: Mapping[int, Tuple[float, float]], total: int, scale: float) -> "Evidence":
        """From frame -> (x, y) of the detections worth trusting; frames past the video are ignored."""
        frames = np.array(sorted(f for f in points if 0 <= f < total), np.int64)
        lut_x = np.full(total + 2, np.nan, np.float32)
        lut_y = np.full(total + 2, np.nan, np.float32)
        lut_x[frames] = np.array([points[f][0] for f in frames], np.float32)
        lut_y[frames] = np.array([points[f][1] for f in frames], np.float32)
        return cls(lut_x, lut_y, int(total), float(scale))


class Tracks(NamedTuple):
    """The visible labels of a clip, in frame order."""
    frames: np.ndarray    # int64: the label's frame (or the video frame it maps to at a rate)
    xs: np.ndarray        # float32, in the video's pixels
    ys: np.ndarray
    fast: np.ndarray      # bool: moved FAST_PX or more (1080p) since the previous row

    @classmethod
    def of(cls, frames: Sequence[int], xs: Sequence[float], ys: Sequence[float], scale: float) -> "Tracks":
        xs_ = np.array(xs, np.float32)
        ys_ = np.array(ys, np.float32)
        if len(xs_):
            step = np.hypot(np.diff(xs_, prepend=xs_[0]), np.diff(ys_, prepend=ys_[0])) * scale
        else:
            step = np.zeros(0, np.float32)
        return cls(np.array(frames, np.int64), xs_, ys_, step >= FAST_PX)

    def at_rate(self, rate: float) -> "Tracks":
        """The same rows with their frames scaled (a 60 FPS label file against a 30 FPS cut)."""
        return self._replace(frames=np.rint(self.frames * rate).astype(np.int64))

    def rows(self, lo: int, hi: int) -> "Tracks":
        return Tracks(self.frames[lo:hi], self.xs[lo:hi], self.ys[lo:hi], self.fast[lo:hi])


def errors_at(tracks: Tracks, evidence: Evidence, offsets: Offsets) -> Tuple[np.ndarray, np.ndarray]:
    """Per row at ``frame + offset``: (did the detector see the ball there, its distance from the
    label in 1080p pixels - NaN where it did not)."""
    index = tracks.frames + offsets
    ok = (index >= 0) & (index < evidence.total)
    dx = np.full(len(index), np.nan, np.float32)
    dy = np.full(len(index), np.nan, np.float32)
    dx[ok] = evidence.lut_x[index[ok]]
    dy[ok] = evidence.lut_y[index[ok]]
    seen = ~np.isnan(dx)
    error = np.full(len(index), np.nan, np.float32)
    error[seen] = np.hypot(dx[seen] - tracks.xs[seen], dy[seen] - tracks.ys[seen]) * evidence.scale
    return seen, error


def judges_fast(tracks: Tracks, seen: np.ndarray, min_fast: int = MIN_FAST) -> bool:
    """Are the fast rows enough to judge on: did the detector see at least `min_fast` of them?"""
    return bool((seen & tracks.fast).sum() >= min_fast)


def judged_rows(tracks: Tracks, seen: np.ndarray, min_fast: int = MIN_FAST) -> np.ndarray:
    """Which seen rows the agreement is measured on: the fast ones when the detector saw at least
    `min_fast` of them, every seen row otherwise."""
    return seen & tracks.fast if judges_fast(tracks, seen, min_fast) else seen


def agreement(tracks: Tracks, evidence: Evidence, offsets: Offsets, min_fast: int = MIN_FAST) -> Tuple[float, int, float]:
    """(share of judged rows within AGREE_PX of a detection, rows judged, their median error) at
    `offsets`; (0, 0, nan) when the detector saw none of them."""
    seen, error = errors_at(tracks, evidence, offsets)
    judged = error[judged_rows(tracks, seen, min_fast)]
    if judged.size == 0:
        return 0.0, 0, float("nan")
    return float((judged <= AGREE_PX).mean()), int(judged.size), float(np.median(judged))
