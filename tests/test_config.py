"""tennis_tracker.config: the knobs that were removed stay removed."""
from tennis_tracker.config import Config


def test_manual_scaling_and_unused_filters_are_removed():
    for field in ("court_depth", "court_side", "motion_flicker_suppress",
                  "motion_raw_ball_color_gate", "motion_raw_component_filter"):
        assert field not in Config.__dataclass_fields__
