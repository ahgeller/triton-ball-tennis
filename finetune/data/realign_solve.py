"""The realignment solve: which constant frame offsets the detector's evidence nominates for the
labels, the dynamic programme that picks one per row (paying for every change of offset), and
the rows no offset can place. Pure numpy over ``finetune.data.alignment``'s tracks and evidence;
``realign.py`` reads and writes the files around it."""

from __future__ import annotations

from typing import FrozenSet, List, NamedTuple, Optional, Set, Tuple

import numpy as np

from finetune.data.alignment import Evidence, Tracks, agreement, errors_at, judged_rows
from finetune.data.detector_metrics import AGREE_PX

COST_CAP = 60.0          # a label this far out tells us nothing beyond "wrong"
NO_DETECTION = 15.0      # cost where the detector said nothing: indifferent, not evidence
SWITCH_PENALTY = 400.0   # a cut must pay for itself in error saved before the DP will take it
RATES = (1.0, 0.5, 2.0)  # label frame -> video frame scaling worth trying
CHUNK_MIN_ROWS = 20      # a chunk of fewer rows than this is not searched ...
CHUNK_MIN_IN_VIDEO = 15  # ... nor an offset that leaves fewer rows than this inside the video ...
CHUNK_MIN_SEEN = 10      # ... or fewer than this the detector saw; fast rows are judged from this many
CHUNK_MIN_AGREE = 0.5    # a chunk's best offset must agree this well to nominate candidates
CANDIDATE_SPREAD = 5     # ... which are the offsets within this many frames of it


class Alignment(NamedTuple):
    """One answer of the solve: how well it agrees, the rate, the label frames at that rate
    (``bases``), the offset chosen per visible row, and the rows dropped as unplaceable."""
    result: Tuple[float, int, float]      # (agreement, rows judged, median px)
    rate: float
    bases: np.ndarray
    chosen: np.ndarray
    dropped: FrozenSet[int] = frozenset()   # indices into the visible rows


def best_offset(tracks: Tracks, evidence: Evidence, low: int, high: int) -> Tuple[float, Optional[int]]:
    """The constant offset in low..high at which the detector agrees with these rows best."""
    best = (0.0, None)
    for offset in range(low, high + 1):
        inside = ((tracks.frames + offset >= 0) & (tracks.frames + offset < evidence.total)).sum()
        if inside < CHUNK_MIN_IN_VIDEO:
            continue
        seen, error = errors_at(tracks, evidence, offset)
        if seen.sum() < CHUNK_MIN_SEEN:
            continue
        judged = error[judged_rows(tracks, seen, CHUNK_MIN_SEEN)]
        agree = float((judged <= AGREE_PX).mean())
        if agree > best[0]:
            best = (agree, offset)
    return best


def candidates(tracks: Tracks, evidence: Evidence, chunks: int) -> np.ndarray:
    """Offsets worth considering, from a coarse chunk-by-chunk best-offset search."""
    edges = np.linspace(0, len(tracks.frames), chunks + 1).astype(int)
    winners = set()
    for lo, hi in zip(edges, edges[1:]):
        if hi - lo < CHUNK_MIN_ROWS:
            continue
        agree, offset = best_offset(tracks.rows(lo, hi), evidence,
                                    -int(tracks.frames[-1]) - 1, evidence.total - int(tracks.frames[lo]))
        if offset is not None and agree >= CHUNK_MIN_AGREE:
            winners.update(range(offset - CANDIDATE_SPREAD, offset + CANDIDATE_SPREAD + 1))
    return np.array(sorted(winners), np.int64)


def _offset_costs(tracks: Tracks, evidence: Evidence, offsets: np.ndarray) -> np.ndarray:
    """cost[row, j]: the label's distance from the detection at offsets[j], capped; NO_DETECTION
    where the detector said nothing; COST_CAP outside the video."""
    n = len(tracks.frames)
    cost = np.empty((n, len(offsets)), np.float64)
    for j, offset in enumerate(offsets):
        index = tracks.frames + offset
        ok = (index >= 0) & (index < evidence.total)
        column = np.full(n, COST_CAP)
        dx = np.full(n, np.nan)
        dy = np.full(n, np.nan)
        dx[ok] = evidence.lut_x[index[ok]]
        dy[ok] = evidence.lut_y[index[ok]]
        seen = ~np.isnan(dx)
        column[seen] = np.minimum(np.hypot(dx[seen] - tracks.xs[seen], dy[seen] - tracks.ys[seen]) * evidence.scale, COST_CAP)
        column[ok & ~seen] = NO_DETECTION
        cost[:, j] = column
    return cost


def solve(tracks: Tracks, evidence: Evidence, offsets: np.ndarray, pinned: Optional[np.ndarray] = None) -> np.ndarray:
    """One offset per visible label, non-decreasing, via a DP that pays to change.

    pinned marks rows that must keep the frame number they already have (offset 0) because a human
    put them there.
    """
    n, k = len(tracks.frames), len(offsets)
    cost = _offset_costs(tracks, evidence, offsets)
    if pinned is not None and pinned.any():
        if 0 not in offsets:
            raise ValueError("Pinned rows require offset zero")
        zero = int(np.flatnonzero(offsets == 0)[0])
        cost[pinned, :] = np.inf
        cost[pinned, zero] = 0.0
    best = cost[0].copy()
    back = np.zeros((n, k), np.int32)
    for i in range(1, n):
        running, argmin = np.inf, 0
        switch = np.empty(k)
        source = np.empty(k, np.int32)
        for j in range(k):                    # prefix minimum: only smaller offsets may lead here
            if j > 0 and best[j - 1] < running:
                running, argmin = best[j - 1], j - 1
            switch[j] = running + SWITCH_PENALTY
            source[j] = argmin
        take = switch < best
        back[i] = np.where(take, source, np.arange(k))
        best = cost[i] + np.where(take, switch, best)
    path = np.empty(n, np.int32)
    if not np.isfinite(best).any():
        raise ValueError("No alignment can preserve all pinned rows")
    path[-1] = int(np.argmin(best))
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return offsets[path]


def unplaceable_at_cuts(tracks: Tracks, evidence: Evidence, chosen: np.ndarray) -> Set[int]:
    """Indices of visible labels sitting inside a cut the detector cannot pin down.

    Where the video jumps, only frames the detector actually saw fix which row it jumps on. Inside
    a run it said nothing about, the cut could be anywhere, so those rows get dropped.
    """
    seen, error = errors_at(tracks, evidence, chosen)
    confirmed = seen & (error <= AGREE_PX)
    drop = set()
    for i in range(1, len(chosen)):
        if chosen[i] == chosen[i - 1]:
            continue
        before = i - 1
        while before >= 0 and not confirmed[before]:
            before -= 1
        after = i
        while after < len(chosen) and not confirmed[after]:
            after += 1
        drop.update(range(before + 1, min(after, len(chosen))))
    return drop


def solve_alignment(tracks: Tracks, evidence: Evidence, reviewed: Set[int], pinned: np.ndarray,
                    chunks: int) -> Optional[Alignment]:
    """The best alignment over RATES, reporting each rate tried; None when no rate puts the labels
    on the ball. Reviewed rows allow rate 1 only and keep offset 0."""
    best = None
    for rate in RATES:
        if reviewed and rate != 1.0:
            continue  # a reviewed source-frame index is an exact temporal anchor
        based = tracks.at_rate(rate)
        offsets = candidates(based, evidence, chunks)
        if pinned.any() and rate == 1.0:
            offsets = np.array(sorted(set(offsets.tolist()) | {0}), np.int64)
        if offsets.size == 0:
            continue
        chosen = solve(based, evidence, offsets, pinned if rate == 1.0 else None)
        # Invisible reviewed rows are not in the visual solve, but are equally
        # authoritative. Reject any candidate that would shift their segment.
        if any(int(chosen[max(0, int(np.searchsorted(tracks.frames, f, side="right")) - 1)]) != 0
               for f in reviewed):
            continue
        result = agreement(based, evidence, chosen)
        print(f"  rate {rate:g}: {offsets.size} candidate offsets in {offsets.min():+d}..{offsets.max():+d}"
              f"  -> agree {result[0]:.2f}, median {result[2]:.1f} px")
        if best is None or result[0] > best.result[0]:
            best = Alignment(result, rate, based.frames, chosen)
    return best


def segments(frames: np.ndarray, chosen: np.ndarray) -> List[List[int]]:
    """``[first label frame, last label frame, offset]`` for each run of one offset along the clip."""
    runs: List[List[int]] = []
    for frame, offset in zip(frames, chosen):
        if not runs or runs[-1][2] != offset:
            runs.append([frame, frame, offset])
        else:
            runs[-1][1] = frame
    return runs
