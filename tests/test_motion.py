"""tennis_tracker.motion on synthetic frames, on the CPU."""
from types import SimpleNamespace

import numpy as np
import torch

from tennis_tracker.config import Config
from tennis_tracker.frame_window import FrameWindow
from tennis_tracker.motion import BackgroundModel, filter_boost_mask, preprocess_frame_cuda, refine_raw_motion_temporal_cpu


def test_empty_motion_clears_reused_boost_buffer():
    buffers = SimpleNamespace(boost_mask_u8=np.full((10, 10), 255, np.uint8))
    result = filter_boost_mask(np.zeros((10, 10), np.uint8), 0, 600, Config(), buffers=buffers)
    assert not np.any(result)


def test_motion_retains_non_yellow_evidence():
    frame = np.zeros((32, 32, 3), np.uint8)
    frame[10:20, 10:20, 2] = 255
    zeros = torch.zeros(32, 32)
    cfg = Config(motion_raw_temporal_gate=False, motion_raw_close_size=0,
                 motion_thresh=1., motion_v_min=0.)
    window = FrameWindow(torch.device("cpu"), True, frame, None)
    background = BackgroundModel(cfg, torch.device("cpu"))
    background.mean_v = background.mean_s = background.var_v = background.var_s = zeros
    raw = preprocess_frame_cuda(
        window, background, cfg, protect_mask=zeros.bool(), skip_dim=True,
        need_cpu_frame=False, need_detector_boost=False).raw_motion
    assert np.count_nonzero(raw) == 100
    expected = refine_raw_motion_temporal_cpu(raw, None, frame, None, cfg)
    np.testing.assert_array_equal(expected, raw)


def test_court_side_protect_mask_covers_the_outside_of_each_sideline():
    """A trapezoid court seen from the baseline: the guides run one alley width outside each
    sideline, and the mask keeps what lies beyond them (and nothing when there is no court)."""
    from tennis_tracker.motion import build_court_side_protect_mask
    keypoints = [0.0] * 28
    for index, (x, y) in {0: (120.0, 20.0), 1: (200.0, 20.0), 2: (60.0, 170.0), 3: (260.0, 170.0),
                          5: (80.0, 170.0), 7: (240.0, 170.0)}.items():
        keypoints[2 * index], keypoints[2 * index + 1] = x, y
    mask = build_court_side_protect_mask(180, 320, keypoints)
    assert mask is not None and mask.shape == (180, 320) and mask.dtype == bool
    assert mask[170, 5] and mask[170, 315]                 # the outer edges near the baseline
    assert not mask[170, 160] and not mask[100, 160]       # the court itself
    assert not mask[170, 70] and not mask[170, 250]        # inside the guides' alley offset
    assert build_court_side_protect_mask(180, 320, None) is None
    assert build_court_side_protect_mask(180, 320, [0.0] * 28) is None
