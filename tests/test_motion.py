"""tennis_tracker.motion on synthetic frames, on the CPU."""
from types import SimpleNamespace

import numpy as np
import torch

from tennis_tracker.config import Config
from tennis_tracker.frame_window import FrameWindow
from tennis_tracker.motion import (_MaskReader, BackgroundModel, _fullframe_boost_masks, filter_boost_mask,
                                   preprocess_frame_cuda, refine_raw_motion_temporal_cpu)


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


def test_mask_reader_downloads_as_u8_and_charges_timed_steps_to_perf():
    perf = {}
    reader = _MaskReader(Config(boost_min_blob_area=1, boost_max_blob_area=600, motion_raw_close_size=0), perf)
    mask = torch.zeros(12, 12, dtype=torch.bool)
    mask[4:8, 4:8] = True
    raw_u8 = reader.download(mask)
    assert raw_u8.dtype == np.uint8 and int(raw_u8[5, 5]) == 255 and int(raw_u8[0, 0]) == 0
    assert "pre_cuda_raw_d2h" in perf and "pre_cuda_cc_filter" not in perf
    reader.components(raw_u8, timed=False)
    assert "pre_cuda_cc_filter" not in perf
    assert reader.components(raw_u8, timed=True).any() and "pre_cuda_cc_filter" in perf
    assert _MaskReader(Config(), None).download(mask).shape == (12, 12)          # no timing: nothing recorded


def test_fullframe_boost_masks_share_the_narrow_mask_unless_a_wide_source_differs():
    cfg = Config(boost_min_blob_area=1, boost_max_blob_area=600, motion_raw_close_size=0)
    narrow_src = torch.zeros(12, 12, dtype=torch.bool)
    narrow_src[4:8, 4:8] = True
    wide_src = narrow_src.clone()
    wide_src[0:3, 0:3] = True
    raw_u8, narrow, wide = _fullframe_boost_masks(narrow_src, None, cfg, _MaskReader(cfg, None))
    assert raw_u8.shape == (12, 12) and narrow is not None and wide is narrow
    _, narrow, wide = _fullframe_boost_masks(narrow_src, wide_src, cfg, _MaskReader(cfg, None))
    assert wide is not narrow and wide.any() and int(wide[1, 1]) == 255 and int(narrow[1, 1]) == 0
    _, narrow, wide = _fullframe_boost_masks(torch.zeros(12, 12, dtype=torch.bool), None, cfg, _MaskReader(cfg, None))
    assert narrow is None and wide is None
