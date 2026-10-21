"""finetune.alignment: labels against the detector at a shift, on synthetic arrays."""
import math

import numpy as np

from finetune.alignment import Evidence, Tracks, agreement, errors_at, judged_rows, judges_fast
from finetune.scoring import AGREE_PX, FAST_PX, MIN_FAST


def straight_line(count: int, speed: float, start: int = 0):
    return {start + i: (100.0 + speed * i, 300.0) for i in range(count)}


def tracks_of(points, scale=1.0) -> Tracks:
    frames = sorted(points)
    return Tracks.of(frames, [points[f][0] for f in frames], [points[f][1] for f in frames], scale)


def test_evidence_is_a_lookup_by_frame_that_ignores_frames_past_the_video():
    seen = Evidence.of({2: (10.0, 20.0), 5: (30.0, 40.0), 99: (1.0, 1.0)}, total=10, scale=1.5)
    assert (seen.total, seen.scale) == (10, 1.5) and len(seen.lut_x) == 12
    assert (seen.lut_x[2], seen.lut_y[2], seen.lut_x[5]) == (10.0, 20.0, 30.0)
    assert math.isnan(seen.lut_x[3]) and math.isnan(seen.lut_x[11])
    assert seen.lut_x.dtype == np.float32


def test_tracks_mark_the_rows_that_moved_since_the_previous_one():
    tracks = tracks_of(straight_line(5, 4.0), scale=1.5)       # 6 px at 1080p: exactly FAST_PX
    assert tracks.fast.tolist() == [False, True, True, True, True]
    assert tracks_of(straight_line(5, 3.9), scale=1.5).fast.sum() == 0
    assert tracks.at_rate(0.5).frames.tolist() == [0, 0, 1, 2, 2]      # rint: halves go to even
    assert tracks.rows(1, 3).frames.tolist() == [1, 2] and tracks.rows(1, 3).fast.tolist() == [True, True]
    empty = Tracks.of([], [], [], 1.0)
    assert len(empty.frames) == 0 and empty.fast.dtype == bool


def test_errors_at_an_offset_are_nan_where_the_detector_said_nothing_or_outside_the_video():
    labels = straight_line(4, 10.0)
    seen = Evidence.of({f + 1: p for f, p in labels.items() if f != 2}, total=5, scale=2.0)
    tracks = tracks_of(labels)
    saw, error = errors_at(tracks, seen, 1)
    assert saw.tolist() == [True, True, False, True]
    assert error[saw].tolist() == [0.0, 0.0, 0.0] and math.isnan(error[2])
    saw, error = errors_at(tracks, seen, np.array([1, 1, 1, 5]))     # per-row offsets; the last leaves the video
    assert saw.tolist() == [True, True, False, False]
    saw, error = errors_at(tracks, seen, 0)                           # one frame early: 10 px * 2.0
    assert saw.tolist() == [False, True, True, False] and error[1:3].tolist() == [20.0, 20.0]


def test_fast_rows_are_judged_only_when_the_detector_saw_enough_of_them():
    labels = straight_line(MIN_FAST + 5, 20.0)
    tracks = tracks_of(labels)
    everything = Evidence.of(labels, total=100, scale=1.0)
    saw, _ = errors_at(tracks, everything, 0)
    assert judges_fast(tracks, saw) and judged_rows(tracks, saw).tolist() == tracks.fast.tolist()
    few = Evidence.of({f: p for f, p in labels.items() if f < MIN_FAST}, total=100, scale=1.0)
    saw, _ = errors_at(tracks, few, 0)
    assert not judges_fast(tracks, saw) and judged_rows(tracks, saw).tolist() == saw.tolist()


def test_agreement_counts_hits_within_agree_px_among_the_judged_rows():
    labels = straight_line(30, 10.0)
    tracks = tracks_of(labels)
    exact = Evidence.of(labels, total=40, scale=1.0)
    assert agreement(tracks, exact, 0) == (1.0, 29, 0.0)          # the first row is never fast
    off = Evidence.of({f: (x + AGREE_PX, y) for f, (x, y) in labels.items()}, total=40, scale=1.0)
    assert agreement(tracks, off, 0)[0] == 1.0                     # exactly AGREE_PX still agrees
    assert agreement(tracks, off, 0, min_fast=10)[1] == 29
    later = Evidence.of({f + 2: p for f, p in labels.items()}, total=40, scale=1.0)
    share, judged, median = agreement(tracks, later, 0)
    assert share == 0.0 and judged == 28 and median == 20.0         # frames 0 and 1 have no detection yet
    assert agreement(tracks, later, 2)[0] == 1.0
    none = agreement(tracks, Evidence.of({}, total=40, scale=1.0), 0)
    assert none[:2] == (0.0, 0) and math.isnan(none[2])
    assert FAST_PX == 6.0 and AGREE_PX == 15.0
