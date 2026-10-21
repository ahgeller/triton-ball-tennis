"""The frame window pass 1 slides over the clip.

Pass 1 looks at three consecutive source frames at once: the current frame
is preprocessed, and the previous and next frames feed the temporal motion
gate.  On the CPU the window keeps the current frame and the one read ahead;
with preprocessing enabled it also keeps each frame's CUDA tensors so a frame
is uploaded exactly once.
"""
from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np
import torch

from .video_io import PinnedFrameUploader, cuda_frame_to_chw_f32, cuda_vs_tensors


class GpuFrame(NamedTuple):
    """One frame on CUDA: CHW float32 in [0, 1] plus its HSV value and saturation planes."""

    chw: torch.Tensor
    v: torch.Tensor
    s: torch.Tensor


class FrameWindow:
    """``prev`` / ``curr`` / ``next`` source frames and their CUDA tensors.

    ``curr`` and ``next`` are the decoded BGR frames (``next`` is None at the
    end of the clip).  ``prev_gpu`` / ``curr_gpu`` / ``next_gpu`` are their
    CUDA tensors when ``upload`` is on, else None; the previous frame is only
    kept on the GPU, which is where the temporal gate reads it.  ``slide``
    moves the window one frame along.
    """

    def __init__(self, device: torch.device, upload: bool,
                 first: np.ndarray, second: Optional[np.ndarray]) -> None:
        self.device = device
        self.upload = bool(upload)
        self.uploader: Optional[PinnedFrameUploader] = None
        if self.upload:
            try:
                self.uploader = PinnedFrameUploader(first.shape[0], first.shape[1], device)
            except Exception:
                self.uploader = None      # no CUDA driver / pinned memory: plain uploads
        self.curr: np.ndarray = first
        self.next: Optional[np.ndarray] = second
        self.prev_gpu: Optional[GpuFrame] = None
        self.curr_gpu: Optional[GpuFrame] = self._to_gpu(first)
        self.next_gpu: Optional[GpuFrame] = self._to_gpu(second)

    def _to_gpu(self, frame: Optional[np.ndarray]) -> Optional[GpuFrame]:
        if frame is None or not self.upload:
            return None
        chw = cuda_frame_to_chw_f32(frame, self.device, uploader=self.uploader)
        v, s = cuda_vs_tensors(None, self.device, gpu_tensor=chw)
        return GpuFrame(chw, v, s)

    def slide(self, incoming: Optional[np.ndarray]) -> None:
        """Make ``next`` the current frame and ``incoming`` the next one (None at the end)."""
        self.prev_gpu = self.curr_gpu
        self.curr_gpu = self.next_gpu
        self.curr = self.next
        self.next = incoming
        self.next_gpu = self._to_gpu(incoming)

    @property
    def exhausted(self) -> bool:
        """True once ``slide`` has moved past the last frame."""
        return self.curr is None
