"""Where the selected ball track turns, in image space: the five-sample windows and the short
occlusion gaps the bounce and contact stages score.

``det_windows`` yields 30 FPS-spaced sample windows with enough detections; ``gap_turns`` yields
short selector gaps whose flanking samples fit a turn.  Both read a ``court_scene.CourtScene``.
Every velocity, fit and extrapolation here reads each frame at ``court_scene.sample_xy``: the
detector's measurement when there is one, else the selected position.
"""

from __future__ import annotations

from typing import Iterable, Iterator, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from ball_in_play_selector import FrameResult

from .court_scene import BallTrack, CourtScene, sample_xy, usable_point


_MIN_DET_IN_WINDOW = 3  # detections required in a five-sample window
_MIN_COS_DENOMINATOR = 1e-6
_SHORT_GAP_MIN_S = 0.05  # shortest selector gap that may hide a turn
_SHORT_GAP_MAX_S = 0.30  # longest selector gap that may hide a turn


_Sample = Tuple[int, FrameResult]


class _GapTurn(NamedTuple):
    start: int
    end: int
    before: List[_Sample]  # two samples before the gap, oldest first
    after: List[_Sample]  # two samples after the gap, oldest first
    history: List[_Sample]  # up to 6 model samples before the gap, oldest first
    incoming: np.ndarray
    outgoing: np.ndarray

    @property
    def midpoint(self) -> int:
        return (self.start + self.end) // 2


def det_windows(
    per_frame: BallTrack, stride: int
) -> Iterator[Tuple[int, List[FrameResult]]]:
    """Yield (centre, five samples at +/-2*stride) windows with enough detections.

    Centre on every frame but keep the 30 FPS sample spacing: stepping by
    stride would only ever centre on even frames at 60 FPS, so an event on an
    odd frame could never be scored.
    """
    for frame in range(2 * stride, len(per_frame) - 2 * stride):
        points = [usable_point(per_frame, frame + offset * stride)
                  for offset in (-2, -1, 0, 1, 2)]
        if any(point is None for point in points):
            continue
        # Do not let a mostly synthetic window manufacture an event.
        if sum(point.source == "det" for point in points) < _MIN_DET_IN_WINDOW:
            continue
        yield frame, points


def window_velocities(points: Sequence[FrameResult]) -> Tuple[np.ndarray, np.ndarray]:
    """Per-sample velocity into and out of the window centre."""
    first, centre, last = (np.asarray(sample_xy(points[index])) for index in (0, 2, 4))
    return (centre - first) / 2.0, (last - centre) / 2.0


def turn_metrics(incoming: np.ndarray, outgoing: np.ndarray) -> Tuple[float, float, float]:
    """Return (incoming speed, outgoing speed, cosine of the turn)."""
    in_speed = float(np.linalg.norm(incoming))
    out_speed = float(np.linalg.norm(outgoing))
    cosine = float(np.dot(incoming, outgoing) / max(in_speed * out_speed, _MIN_COS_DENOMINATOR))
    return in_speed, out_speed, cosine


def _fit_gap_turn(
    before: Sequence[_Sample], after: Sequence[_Sample]
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Per-frame velocities into and out of a short gap, or None when the two
    branches run parallel and so fit no turn."""
    (frame_a, point_a), (frame_b, point_b) = before
    (frame_c, point_c), (frame_d, point_d) = after
    incoming = (np.asarray(sample_xy(point_b)) - np.asarray(sample_xy(point_a))) / max(1, frame_b - frame_a)
    outgoing = (np.asarray(sample_xy(point_d)) - np.asarray(sample_xy(point_c))) / max(1, frame_d - frame_c)
    slope = incoming - outgoing
    # A descending-then-ascending pair always meets on the height axis; any
    # other pair must differ in velocity to meet at all.
    reverses = incoming[1] > 0.0 > outgoing[1] and abs(slope[1]) >= 1e-8
    if not reverses and float(np.dot(slope, slope)) < 1e-8:
        return None
    return incoming, outgoing


def extrapolate_incoming(samples: Sequence[_Sample], frame: int) -> Optional[np.ndarray]:
    if len(samples) < 2:
        return None
    anchor = float(samples[-1][0])
    times = np.asarray([float(item[0]) - anchor for item in samples])
    degree = min(2, len(samples) - 1)
    target = float(frame) - anchor
    xs, ys = zip(*(sample_xy(item[1]) for item in samples))
    x_coeff = np.polyfit(times, xs, degree)
    y_coeff = np.polyfit(times, ys, degree)
    return np.array([
        float(np.polyval(x_coeff, target)),
        float(np.polyval(y_coeff, target)),
    ])


def _usable_samples(
    per_frame: BallTrack, frames: Iterable[int], limit: Optional[int] = None
) -> List[_Sample]:
    samples = []
    for index in frames:
        point = usable_point(per_frame, index)
        if point is not None:
            samples.append((index, point))
            if limit is not None and len(samples) == limit:
                break
    return samples


def gap_turns(scene: CourtScene) -> Iterator[_GapTurn]:
    """Yield short selector gaps whose flanking samples fit a turn."""
    per_frame, stride = scene.per_frame, scene.stride
    min_gap = max(2, scene.frames(_SHORT_GAP_MIN_S))
    max_gap = max(min_gap, scene.frames(_SHORT_GAP_MAX_S))
    count = len(per_frame)
    frame = 0
    while frame < count:
        if usable_point(per_frame, frame) is not None:
            frame += 1
            continue
        start = frame
        while frame < count and usable_point(per_frame, frame) is None:
            frame += 1
        end = frame - 1
        if not (min_gap <= end - start + 1 <= max_gap):
            continue
        before = _usable_samples(
            per_frame, range(start - 1, max(-1, start - 4 * stride), -1), limit=2
        )
        after = _usable_samples(
            per_frame, range(end + 1, min(count, end + 4 * stride + 1)), limit=2
        )
        if len(before) < 2 or len(after) < 2:
            continue
        before.reverse()
        fit = _fit_gap_turn(before, after)
        if fit is None:
            continue
        history = _usable_samples(
            per_frame, range(start - 1, max(-1, start - 6 * stride - 1), -1)
        )
        history.reverse()
        yield _GapTurn(start, end, before, after, history, *fit)
