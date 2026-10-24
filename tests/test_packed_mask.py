"""PackedMask: the one packed-mask format the tracker stores and the selector reads."""
import numpy as np

from ball_in_play_selector import PackedMask
from ball_in_play_selector.utils import ensure_mask_u8


def _mask(h=20, w=30):
    mask = np.zeros((h, w), np.uint8)
    mask[5:9, 10:16] = 255
    mask[12, 20] = 255
    return mask


def test_pack_defaults_to_the_bounding_box_of_the_set_pixels():
    packed = PackedMask.pack(_mask())
    assert packed.roi == (10, 5, 21, 13) and (packed.height, packed.width) == (20, 30)
    np.testing.assert_array_equal(packed.unpack(), _mask())


def test_pack_with_an_roi_keeps_that_rectangle_and_clips_it_to_the_frame():
    packed = PackedMask.pack(_mask(), roi=(8, 4, 40, 15))
    assert packed.roi == (8, 4, 30, 15)
    np.testing.assert_array_equal(packed.unpack(), _mask())
    # An ROI that clips to nothing falls back to the bounding box.
    assert PackedMask.pack(_mask(), roi=(35, 4, 40, 15)).roi == (10, 5, 21, 13)


def test_empty_masks_are_never_packed():
    assert PackedMask.pack(None) is None
    assert PackedMask.pack(np.zeros((4, 4), np.uint8)) is None


def test_repack_after_erasing_keeps_the_rectangle_and_drops_emptied_masks():
    packed = PackedMask.pack(_mask(), roi=(0, 0, 30, 20))
    mask = packed.unpack()
    mask[5:9, 10:16] = 0
    repacked = PackedMask.pack(mask, roi=packed.roi)
    assert repacked.roi == packed.roi and repacked.unpack()[12, 20] == 255
    mask[12, 20] = 0
    assert PackedMask.pack(mask, roi=packed.roi) is None


def test_ensure_mask_u8_accepts_packed_and_raw_masks():
    raw = _mask()
    assert ensure_mask_u8(None) is None
    assert ensure_mask_u8(raw) is raw
    np.testing.assert_array_equal(ensure_mask_u8(PackedMask.pack(raw)), raw)


def test_unpack_rect_is_the_packed_rectangle_of_the_full_mask():
    mask = np.zeros((40, 60), dtype=np.uint8)
    mask[10:14, 20:25] = 255
    mask[30, 50] = 255
    packed = PackedMask.pack(mask, roi=(18, 8, 55, 33))
    rect = packed.unpack_rect()
    assert rect.shape == (25, 37) and rect.dtype == np.uint8
    np.testing.assert_array_equal(rect, packed.unpack()[8:33, 18:55])
    np.testing.assert_array_equal(rect, mask[8:33, 18:55])
