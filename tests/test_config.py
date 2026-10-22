"""tennis_tracker.config: the knobs that were removed stay removed; the selector config's
derived fields follow the clip."""
import pytest

from ball_in_play_selector import SelectorConfig
from tennis_tracker.config import Config


def test_manual_scaling_and_unused_filters_are_removed():
    for field in ("court_depth", "court_side", "motion_flicker_suppress",
                  "motion_raw_ball_color_gate", "motion_raw_component_filter"):
        assert field not in Config.__dataclass_fields__


@pytest.mark.parametrize("fps, stride", [
    (24.0, 1), (29.97, 1), (30.0, 1), (59.94, 2), (60.0, 2),
])
def test_detector_phase_stride_counts_the_30fps_streams_in_a_clip(fps, stride):
    assert SelectorConfig(fps=fps).detector_phase_stride == stride
