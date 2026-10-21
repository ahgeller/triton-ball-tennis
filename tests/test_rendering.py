"""tennis_tracker.render.rendering: the marker sizes, the search-centre smoothing, and what each drawing
primitive puts where.  These assert structure (which pixels a marker covers, which frame gets a
call, what the HUD says) and do not depend on how this OpenCV build rasterises, unlike the
byte-exact frame hashes in tests/test_pipeline_run.py."""
from typing import List, Tuple

import cv2
import numpy as np
import pytest
from ball_in_play_selector import FrameResult

from tennis_tracker.render import rendering
from tennis_tracker.config import Config
from tennis_tracker.render.rendering import (
    COLOR_DET,
    COLOR_INTERP,
    COLOR_MOTION,
    COLOR_RAW,
    COLOR_SEARCH,
    COLOR_WHITE,
    _SEARCH_CENTRE_MAX_STEP_PX,
    Canvas,
    MarkerSizer,
    SearchCentreSmoother,
    draw_hud,
    draw_raw_detections,
    draw_selected_result,
)
from tennis_tracker.render.trails import BallTrail

WIDTH, HEIGHT = 320, 180


def blank() -> np.ndarray:
    return np.zeros((HEIGHT, WIDTH, 3), np.uint8)


def result(x: float, y: float, source: str, **fields) -> FrameResult:
    return FrameResult(cx=x, cy=y, conf=0.9, source=source, **fields)


@pytest.fixture
def recorder(monkeypatch):
    """Record every cv2 text and line call: (function, id of the target image, the label or
    the two end points)."""
    calls: List[Tuple[str, int, object]] = []
    real_text, real_line = cv2.putText, cv2.line

    def put_text(img, label, origin, *args, **kwargs):
        calls.append(("text", id(img), label))
        return real_text(img, label, origin, *args, **kwargs)

    def line(img, p0, p1, *args, **kwargs):
        calls.append(("line", id(img), (p0, p1)))
        return real_line(img, p0, p1, *args, **kwargs)

    monkeypatch.setattr(cv2, "putText", put_text)
    monkeypatch.setattr(cv2, "line", line)
    return calls


# --------------------------------------------------------------------------- MarkerSizer

def test_marker_sizer_follows_detector_boxes_within_its_limits():
    cfg = Config(ball_marker_min_radius=3, ball_marker_max_radius=12, ball_marker_box_scale=0.5)
    sizer = MarkerSizer(cfg)
    assert sizer.radius == 4.0 and sizer.estimate_radius == 4
    assert sizer.observe_box(10.0, 10.0) == 5                     # round(10 * 0.5), inside the limits
    assert sizer.radius == pytest.approx(0.8 * 4.0 + 0.2 * 5.0)   # the EMA of the box radius
    assert sizer.observe_box(2.0, 2.0) == 3                       # clipped to the minimum
    assert sizer.observe_box(100.0, 100.0) == 12                  # ... and the maximum
    for _ in range(100):
        sizer.observe_box(400.0, 400.0)
    assert sizer.radius == 36.0 and sizer.estimate_radius == 12   # radius capped at 3x max, estimate at 12


# --------------------------------------------------------------------------- SearchCentreSmoother

def test_search_centre_smoother_caps_a_jump_only_between_consecutive_frames():
    smoother = SearchCentreSmoother()
    assert smoother.smooth(10, 100, 100) == (100, 100)
    scx, scy = smoother.smooth(11, 100 + 300, 100)                 # a 300 px jump on the next frame
    assert (scx, scy) == (100 + int(_SEARCH_CENTRE_MAX_STEP_PX), 100)
    assert smoother.smooth(13, 400, 100) == (400, 100)             # a skipped frame: no cap
    smoother.reset()
    assert smoother.smooth(14, 10, 10) == (10, 10)


# --------------------------------------------------------------------------- draw_selected_result

def test_detection_marker_and_box_land_on_both_frames():
    main, debug = blank(), blank()
    canvas = Canvas(main, debug)
    res = result(100.0, 80.0, "det", bbox=(94.0, 74.0, 106.0, 86.0))
    draw_selected_result(canvas, res, 5, MarkerSizer(Config()), SearchCentreSmoother(), draw_search_regions=False)
    for frame in (main, debug):
        assert tuple(frame[80, 100]) == COLOR_DET                  # the filled marker at the centre
        assert tuple(frame[73, 94]) == COLOR_DET                   # the box's top-left corner
        assert tuple(frame[87, 100]) == COLOR_WHITE                # the emphasised bottom edge
        assert not frame[150:, :].any()                            # nothing far from the ball


def test_motion_marker_draws_its_search_region_on_the_debug_frame_only(recorder):
    main, debug = blank(), blank()
    canvas = Canvas(main, debug)
    res = result(100.0, 80.0, "motion", search_cx=120.0, search_cy=90.0, search_radius=20.0)
    draw_selected_result(canvas, res, 5, MarkerSizer(Config()), SearchCentreSmoother(), draw_search_regions=False)
    assert tuple(main[80, 100]) == COLOR_MOTION and tuple(debug[82, 98]) == COLOR_MOTION
    assert tuple(debug[80, 100]) == COLOR_SEARCH                   # the search line ends on the marker
    assert tuple(debug[90, 140]) == COLOR_SEARCH                   # the search circle's right edge
    assert not any(tuple(main[90, x]) == COLOR_SEARCH for x in range(WIDTH))
    search_lines = [call for call in recorder if call[0] == "line" and call[2] == ((120, 90), (100, 80))]
    assert [target for _, target, _ in search_lines] == [id(debug)]
    # With draw_search_regions the main frame gets the region too.
    main, debug = blank(), blank()
    draw_selected_result(Canvas(main, debug), res, 6, MarkerSizer(Config()), SearchCentreSmoother(), True)
    assert tuple(main[90, 140]) == COLOR_SEARCH


def test_interpolated_marker_is_the_interp_colour_with_its_label(recorder):
    main = blank()
    draw_selected_result(Canvas(main, None), result(50.0, 50.0, "interp"), 1, MarkerSizer(Config()),
                         SearchCentreSmoother(), False)
    assert tuple(main[50, 50]) == COLOR_INTERP
    assert [label for kind, _, label in recorder if kind == "text"] == ["PHY"]


# --------------------------------------------------------------------------- HUD and raw detections

def test_hud_names_the_frame_source_and_confidence_and_the_track_on_the_debug_frame(recorder):
    main, debug = blank(), blank()
    canvas = Canvas(main, debug)
    track = type("Track", (), {"track_id": 7})()
    draw_hud(canvas, 3, result(1.0, 1.0, "det"), track)
    texts = [(target, label) for kind, target, label in recorder if kind == "text"]
    assert (id(main), "F3") in texts and (id(main), "ID:7 DET c=0.90") in texts
    assert (id(debug), "track=7") in texts and (id(main), "track=7") not in texts
    assert main[:40, :200].any() and not main[40:, :].any()        # the text sits in the top-left HUD strip
    recorder.clear()
    draw_hud(Canvas(blank(), None), 4, None, None)
    assert [label for _, _, label in recorder] == ["F4"]           # no result: the frame number alone


def test_raw_detections_are_boxes_on_the_debug_frame_only():
    main, debug = blank(), blank()
    canvas = Canvas(main, debug)
    draw_raw_detections(canvas.debug_only, [([20.0, 30.0, 40.0, 50.0], 0.7)])
    assert tuple(debug[30, 30]) == COLOR_RAW and tuple(debug[40, 20]) == COLOR_RAW
    assert not main.any()


# --------------------------------------------------------------------------- trail segments

def test_ball_trail_draws_one_outlined_segment_per_consecutive_pair(recorder):
    trail = BallTrail(Config(), 30.0, WIDTH, HEIGHT)
    for i in range(5):
        trail.advance(result(50.0 + 10 * i, 60.0, "det"), None)
    trail.advance(None, None)                                       # a break
    trail.advance(result(200.0, 60.0, "det"), None)
    trail.advance(result(210.0, 60.0, "det"), None)
    main, debug = blank(), blank()
    trail.draw(Canvas(main, debug), draw_main=True)
    lines = [call for call in recorder if call[0] == "line"]
    # Four segments before the break and one after, each an outline plus a coloured line,
    # on both frames.
    assert len([call for call in lines if call[1] == id(debug)]) == 2 * 5
    assert len([call for call in lines if call[1] == id(main)]) == 2 * 5
    assert not any(call[2] == ((90, 60), (200, 60)) for call in lines)   # nothing bridges the break
