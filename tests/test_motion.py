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
