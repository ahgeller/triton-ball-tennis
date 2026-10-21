"""FrameWindow: the prev/curr/next window pass 1 slides over the clip, and the background model."""
import numpy as np
import torch

from tennis_tracker.config import Config
from tennis_tracker.frame_window import FrameWindow
from tennis_tracker.motion import BackgroundModel

CPU = torch.device("cpu")


def _frames(n, value_step=10):
    return [np.full((4, 6, 3), i * value_step, np.uint8) for i in range(n)]


def test_slide_moves_curr_to_prev_and_next_to_curr_with_uploads():
    f = _frames(3)
    window = FrameWindow(CPU, True, f[0], f[1])
    assert window.curr is f[0] and window.next is f[1]
    assert window.prev_gpu is None
    assert torch.allclose(window.curr_gpu.chw, torch.zeros(3, 4, 6))
    assert torch.allclose(window.next_gpu.v, torch.full((4, 6), 10 / 255))
    window.slide(f[2])
    assert window.curr is f[1] and window.next is f[2]
    assert torch.allclose(window.prev_gpu.chw, torch.zeros(3, 4, 6))
    assert torch.allclose(window.curr_gpu.v, torch.full((4, 6), 10 / 255))
    assert torch.allclose(window.next_gpu.v, torch.full((4, 6), 20 / 255))
    assert not window.exhausted
    window.slide(None)
    assert window.curr is f[2] and window.next is None and window.next_gpu is None
    window.slide(None)
    assert window.exhausted and window.curr is None


def test_without_upload_the_window_keeps_frames_but_no_tensors():
    f = _frames(2)
    window = FrameWindow(CPU, False, f[0], f[1])
    assert window.curr_gpu is None and window.next_gpu is None
    window.slide(None)
    assert window.curr is f[1] and window.prev_gpu is None and window.curr_gpu is None


def test_background_model_initialises_then_adapts_slower_at_motion_pixels():
    cfg = Config(background_alpha=0.5, motion_freeze_alpha=0.0, motion_thresh=25.5)
    model = BackgroundModel(cfg, CPU)
    assert not model.ready
    model.update(torch.zeros(2, 2), torch.zeros(2, 2))
    assert model.ready and torch.allclose(model.var_v, torch.full((2, 2), 0.1 ** 2))
    motion = np.zeros((2, 2), np.uint8)
    motion[0, 0] = 255
    model.remember_motion(motion)
    model.update(torch.ones(2, 2), torch.ones(2, 2))
    # The motion pixel keeps the old mean (alpha 0); the rest moved halfway.
    assert model.mean_v[0, 0] == 0.0 and model.mean_v[1, 1] == 0.5
    model.remember_motion(None)
    model.update(torch.ones(2, 2), torch.ones(2, 2))
    assert model.mean_v[0, 0] == 0.5
