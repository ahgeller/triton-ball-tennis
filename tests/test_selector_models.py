"""ball_in_play_selector.models: Track.update (the Kalman filter starts with the first
observation, and a track whose observations were assigned directly cannot be updated) and
FrameResult.position_kind."""
import pytest

from ball_in_play_selector.config import SelectorConfig
from ball_in_play_selector.models import Detection, FrameResult, Track


def test_position_kind_follows_the_measurement_as_the_position_moves():
    """Read when asked, not stored: the refit moves ``cx, cy`` after the row is built."""
    filled = FrameResult(cx=10.0, cy=20.0, source="interp")
    assert filled.position_kind == "recovered"
    detected = FrameResult(cx=10.0, cy=20.0, conf=0.9, source="det", measurement=(10.0, 20.0))
    assert detected.position_kind == "measurement"
    detected.cx = 10.25
    assert detected.position_kind == "filtered"


def _cfg() -> SelectorConfig:
    return SelectorConfig(fps=30.0, width=1920, height=1080).auto_scale()


def _det(frame: int, cx: float, cy: float) -> Detection:
    return Detection(frame, cx, cy, cx - 2, cy - 2, cx + 2, cy + 2, 0.9, 16.0, True)


def test_update_starts_the_filter_on_the_first_observation_and_steps_it_after():
    track = Track(track_id=1, cfg=_cfg())
    track.update(_det(0, 100.0, 50.0))
    assert track.kf is not None and track.last_vel == (0.0, 0.0)
    track.update(_det(2, 110.0, 50.0))
    assert track.num_obs == 2 and track.last_pos == (110.0, 50.0)
    assert track.last_vel[0] > 0.0                       # bootstrapped from the first two observations


def test_update_refuses_a_track_given_its_observations_directly():
    """scoring stitches timeline chains this way (observations assigned, no filter): growing one
    is a caller error, reported as such rather than an AttributeError on None."""
    stitched = Track(track_id=7, cfg=_cfg())
    stitched.observations = [_det(0, 100.0, 50.0), _det(1, 104.0, 50.0)]
    with pytest.raises(ValueError, match="track 7 has 2 observations but no Kalman filter"):
        stitched.update(_det(2, 108.0, 50.0))
    assert stitched.num_obs == 2                          # nothing appended
