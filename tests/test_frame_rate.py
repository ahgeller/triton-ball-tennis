"""tennis_tracker/frame_rate.py: the 30/60 FPS bands and the stride they map to."""
import pytest

from tennis_tracker.frame_rate import SIXTY_FPS_RANGE, THIRTY_FPS_RANGE, phase_stride, require_phase_stride


@pytest.mark.parametrize("fps, stride", [
    (22.0, 1), (32.0, 1), (29.97, 1), (30, 1),
    (57.0, 2), (62.0, 2), (59.94, 2), (60, 2),
    (32.01, None), (56.99, None), (21.99, None), (20.0, None), (120.0, None), (0.0, None),
])
def test_phase_stride_bands(fps, stride):
    assert phase_stride(fps) == stride


def test_band_edges_are_inclusive():
    assert phase_stride(THIRTY_FPS_RANGE[0]) == 1 and phase_stride(THIRTY_FPS_RANGE[1]) == 1
    assert phase_stride(SIXTY_FPS_RANGE[0]) == 2 and phase_stride(SIXTY_FPS_RANGE[1]) == 2


def test_require_phase_stride_names_the_unsupported_rate():
    assert require_phase_stride(60) == 2
    assert require_phase_stride(29.97) == 1
    assert require_phase_stride(24.0) == 1      # 24 FPS sits inside the 30 FPS band
    with pytest.raises(ValueError) as caught:
        require_phase_stride(20.0)
    assert "20.000 FPS" in str(caught.value)
