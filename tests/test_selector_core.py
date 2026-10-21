"""ball_in_play_selector.core helpers on synthetic masks (the scenarios are in test_selector_scenarios.py)."""
import numpy as np

from ball_in_play_selector.core import motion_near


def test_small_motion_component_uses_pixel_area():
    mask = np.zeros((100, 100), np.uint8)
    mask[49:52, 49:52] = 255
    result = motion_near(mask, 50, 50, 175., 20., 4.)
    assert (result["x"], result["y"], result["area"]) == (50., 50., 9.)
    assert motion_near(mask, 0, 0, 175., 20., 4.) is None
