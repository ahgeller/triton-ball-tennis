"""tennis_tracker.render.motion_trails: raw-motion trails fall back to straight segments only when scipy
cannot fit a spline; a bug in the drawing itself is not swallowed."""
import numpy as np
import pytest

from tennis_tracker.render import motion_trails
from tennis_tracker.render.motion_trails import RawMotionTrails, _fit_spline_trail

scipy = pytest.importorskip("scipy")


def _samples(points):
    return [(frame, float(x), float(y), 5.0) for frame, (x, y) in enumerate(points)]


def test_spline_fit_returns_none_for_a_degenerate_sample_set():
    assert _fit_spline_trail(3, _samples([(10, 10)] * 4)) is None              # no curve through one point
    x_fine, y_fine, ages = _fit_spline_trail(3, _samples([(10, 10), (20, 14), (30, 20), (40, 28)]))
    assert len(x_fine) == len(y_fine) == len(ages) >= 100
    assert ages[0] == pytest.approx(3.0) and ages[-1] == pytest.approx(0.0)


def test_trail_falls_back_to_straight_segments_when_the_fit_fails(monkeypatch):
    trails = RawMotionTrails(frame_w=640)
    vis = np.zeros((64, 64, 3), np.uint8)
    straight = []
    monkeypatch.setattr(motion_trails, "_draw_straight_trail", lambda v, f, pts: straight.append(len(pts)))
    monkeypatch.setattr(motion_trails, "_fit_spline_trail", lambda frame, pts: None)
    trails._draw_trail(vis, 3, _samples([(10, 10), (20, 14), (30, 20), (40, 28)]))
    assert straight == [4]


def test_a_drawing_error_is_not_swallowed(monkeypatch):
    trails = RawMotionTrails(frame_w=640)
    vis = np.zeros((64, 64, 3), np.uint8)

    def broken(v, spline):
        raise IndexError("a bug in the drawing")

    monkeypatch.setattr(motion_trails, "_draw_spline_trail", broken)
    with pytest.raises(IndexError, match="a bug in the drawing"):
        trails._draw_trail(vis, 3, _samples([(10, 10), (20, 14), (30, 20), (40, 28)]))


def test_trails_draw_a_moving_blob_as_a_curve_and_its_head():
    trails = RawMotionTrails(frame_w=640)
    for frame in range(6):
        mask = np.zeros((64, 64), np.uint8)
        x = 8 + 8 * frame
        mask[20:24, x:x + 4] = 255
        trails.advance(frame, [(0, 0, 64, 64)], mask)
    vis = np.zeros((64, 64, 3), np.uint8)
    trails.draw(vis, 5)
    assert len(trails.trails) == 1 and len(trails.trails[0]) == 6
    assert vis.any() and vis[24, 50].any()                 # the head dot at the last blob
