"""ball_in_play_selector.core helpers on synthetic masks (the scenarios are in test_selector_scenarios.py)."""
import cv2
import numpy as np

from ball_in_play_selector.core import motion_near


def test_small_motion_component_uses_pixel_area():
    mask = np.zeros((100, 100), np.uint8)
    mask[49:52, 49:52] = 255
    result = motion_near(mask, 50, 50, 175., 20., 4.)
    assert (result.x, result.y, result.area) == (50., 50., 9.)
    assert motion_near(mask, 0, 0, 175., 20., 4.) is None


def test_motion_near_on_a_packed_mask_matches_the_unpacked_frame():
    """The packed rectangle holds every set pixel, so the components found in it are the frame's,
    offset back by its corner: the same blob, centre, bbox and area as on the unpacked mask."""
    from ball_in_play_selector import PackedMask
    mask = np.zeros((1080, 1920), dtype=np.uint8)
    cv2.circle(mask, (900, 600), 4, 255, -1)          # the ball-sized blob
    mask[590:620, 700:760] = 255                        # a large one, outside the area limit
    mask[0:3, 0:3] = 255                                # a far one
    for roi in (None, (680, 560, 940, 640), (0, 0, 1920, 1080)):
        packed = PackedMask.pack(mask, roi=roi)
        from_packed = motion_near(packed, 905., 603., 40., 25., 4.)
        from_frame = motion_near(packed.unpack(), 905., 603., 40., 25., 4.)
        assert from_packed is not None and from_packed == from_frame
        assert from_packed.bbox == (896, 596, 905, 605)
    assert motion_near(PackedMask.pack(mask), 100., 100., 40., 25., 4.) is None
