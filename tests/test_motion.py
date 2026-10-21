"""tennis_tracker.motion on synthetic frames, on the CPU."""
from types import SimpleNamespace

import numpy as np
import torch

from tennis_tracker.config import Config
from tennis_tracker.frame_window import FrameWindow
from tennis_tracker.motion import (
    MASKS_ONLY, _MaskReader, BackgroundModel, DetectorInputRequest, _fullframe_boost_masks, _raw_motion_cuda,
    filter_boost_mask, preprocess_frame_cuda,
)
from tests._helpers import refine_raw_motion_temporal_cpu


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
    raw = preprocess_frame_cuda(window, background, cfg, request=MASKS_ONLY, protect_mask=zeros.bool()).raw_motion
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


def _window_frames():
    """Three 48x64 frames: a ball that moves its own width each frame, a static blob and a blob
    that is only in the current frame (a flicker: it moves against both neighbours)."""
    frames = []
    for index in range(3):
        frame = np.zeros((48, 64, 3), np.uint8)
        x = 6 + 10 * index
        frame[10:16, x:x + 6] = (70, 235, 240)        # the moving ball
        frame[30:36, 40:46] = (200, 200, 200)          # static: foreground, never moving
        if index == 1:
            frame[30:36, 10:16] = (255, 255, 255)      # flicker: appears for one frame only
        frames.append(frame)
    return frames


def test_temporal_gate_matches_the_cpu_reference_with_the_gate_on():
    """With the gate on, the CUDA path keeps the moving ball and the one-frame flicker and drops
    the static blob, exactly as the CPU reference in tests/_helpers.py does."""
    cfg = Config(motion_raw_temporal_gate=True, motion_raw_close_size=0, motion_thresh=1., motion_v_min=0.)
    prev, curr, nxt = _window_frames()
    window = FrameWindow(torch.device("cpu"), True, prev, curr)
    window.slide(nxt)                                   # prev / curr / next all present
    assert window.curr is curr and window.prev_gpu is not None and window.next_gpu is not None
    background = BackgroundModel(cfg, torch.device("cpu"))
    zeros = torch.zeros(48, 64)
    background.mean_v = background.mean_s = background.var_v = background.var_s = zeros
    gated, ungated = _raw_motion_cuda(window, background, cfg, None)
    ungated_u8 = ungated.numpy().astype(np.uint8) * 255
    assert ungated_u8[12, 18] and ungated_u8[32, 42] and ungated_u8[32, 12]   # all three are foreground
    cuda_raw = preprocess_frame_cuda(window, background, cfg, request=MASKS_ONLY).raw_motion
    np.testing.assert_array_equal(cuda_raw, refine_raw_motion_temporal_cpu(ungated_u8, prev, curr, nxt, cfg))
    assert cuda_raw[12, 18] and cuda_raw[32, 12] and not cuda_raw[32, 42]
    assert np.count_nonzero(gated.numpy()) < np.count_nonzero(ungated.numpy())

    # At the end of the clip there is no next frame: the gate is the previous delta alone.
    window.slide(None)
    assert window.curr is nxt and window.next_gpu is None
    gated, ungated = _raw_motion_cuda(window, background, cfg, None)
    ungated_u8 = ungated.numpy().astype(np.uint8) * 255
    cuda_raw = preprocess_frame_cuda(window, background, cfg, request=MASKS_ONLY).raw_motion
    np.testing.assert_array_equal(cuda_raw, refine_raw_motion_temporal_cpu(ungated_u8, curr, nxt, None, cfg))
    assert cuda_raw[12, 28] and not cuda_raw[32, 42]


def test_detector_input_request_decides_what_preprocessing_materialises():
    frame = np.zeros((32, 32, 3), np.uint8)
    frame[10:20, 10:20, 2] = 255
    cfg = Config(motion_raw_temporal_gate=False, motion_raw_close_size=0, motion_thresh=1., motion_v_min=0.)
    window = FrameWindow(torch.device("cpu"), True, frame, None)
    background = BackgroundModel(cfg, torch.device("cpu"))
    zeros = torch.zeros(32, 32)
    background.mean_v = background.mean_s = background.var_v = background.var_s = zeros
    masks_only = preprocess_frame_cuda(window, background, cfg, request=MASKS_ONLY)
    assert masks_only.detector_frame is None and masks_only.detector_frame_cuda is None
    assert np.count_nonzero(masks_only.raw_motion) == 100
    both = preprocess_frame_cuda(window, background, cfg, request=DetectorInputRequest(cpu=True, cuda=True))
    assert both.detector_frame is not None and both.detector_frame.shape == (32, 32, 3)
    assert both.detector_frame_cuda is not None and tuple(both.detector_frame_cuda.shape) == (32, 32, 3)
    np.testing.assert_array_equal(both.detector_frame, both.detector_frame_cuda.numpy())
    cpu_only = preprocess_frame_cuda(window, background, cfg, request=DetectorInputRequest(cpu=True))
    assert cpu_only.detector_frame_cuda is None and cpu_only.detector_frame is not None
    np.testing.assert_array_equal(cpu_only.raw_motion, masks_only.raw_motion)
