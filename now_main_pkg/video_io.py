# Imports
import argparse
import copy
import glob
import json
import math
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from collections import OrderedDict, namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import scipy.interpolate
from ball_in_play_selector import select_ball_in_play, FrameResult, _predict_projectile, SelectorConfig
HAS_NMS = False
_nms = None
try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except Exception:
    torch = None
    F = None
    HAS_TORCH = False

try:
    from boxmot import ByteTrack
except ImportError:
    print("[warning] boxmot not found. Player tracking will be disabled. Run 'pip install boxmot'")
    ByteTrack = None

from .config import Config
from .utils import find_ffmpeg


class _PinnedFrameUploader:
    """Pinned host staging for faster/non-blocking H2D frame uploads."""

    def __init__(self, h: int, w: int, device):
        self.h = int(h)
        self.w = int(w)
        self.device = device
        self._pinned_u8 = torch.empty((self.h, self.w, 3), dtype=torch.uint8, pin_memory=True)
        self._pinned_np = self._pinned_u8.numpy()

    @torch.no_grad()
    def upload_chw_f32(self, frame_bgr: np.ndarray):
        if frame_bgr.shape[0] != self.h or frame_bgr.shape[1] != self.w or frame_bgr.shape[2] != 3:
            return torch.from_numpy(frame_bgr).to(
                device=self.device, dtype=torch.float32
            ).permute(2, 0, 1).contiguous() / 255.0
        np.copyto(self._pinned_np, frame_bgr)
        t = self._pinned_u8.to(device=self.device, dtype=torch.float32, non_blocking=True)
        return t.permute(2, 0, 1).contiguous() / 255.0

@torch.no_grad()
def _cuda_frame_to_chw_f32(frame_bgr: np.ndarray, device, uploader: Optional[_PinnedFrameUploader] = None):
    """Upload one BGR frame once as CHW float32 in [0,1] on CUDA."""
    if uploader is not None:
        return uploader.upload_chw_f32(frame_bgr)
    return torch.from_numpy(frame_bgr).to(
        device=device, dtype=torch.float32
    ).permute(2, 0, 1).contiguous() / 255.0

@torch.no_grad()
def _cuda_vs_tensors(frame_bgr: Optional[np.ndarray], device, gpu_tensor=None):
    """Extract V and S channels as CUDA tensors (no grad tracking).
    If gpu_tensor is provided, reuse it instead of uploading frame again."""
    if gpu_tensor is not None:
        t = gpu_tensor
    else:
        if frame_bgr is None:
            raise ValueError("frame_bgr is required when gpu_tensor is None")
        t = _cuda_frame_to_chw_f32(frame_bgr, device)
    maxc = torch.max(t, dim=0).values
    minc = torch.min(t, dim=0).values
    s = torch.where(maxc > 1e-6, (maxc - minc) / (maxc + 1e-6), torch.zeros_like(maxc))
    return maxc, s

class VideoWriter:
    def __init__(self, path, fps, w, h, cfg):
        self._proc = None
        self._cv = None
        self._encoder = "opencv"
        self._thread_error = None
        self._path = str(path)
        ffmpeg = find_ffmpeg()

        if cfg.use_nvenc and ffmpeg:
            try:
                cmd = [ffmpeg, "-y", "-loglevel", "error",
                       "-f", "rawvideo", "-pix_fmt", "bgr24",
                       "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
                       "-c:v", "h264_nvenc", "-preset", cfg.nvenc_preset,
                       "-b:v", cfg.nvenc_bitrate, "-pix_fmt", "yuv420p", path]
                self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
                self._proc.stdin.write(np.zeros((h, w, 3), dtype=np.uint8).tobytes())
                self._proc.stdin.flush()
                if self._proc.poll() is None:
                    self._encoder = "nvenc"
                else:
                    err = self._read_proc_stderr()
                    print(f"[writer] NVENC init failed; falling back ({err or 'ffmpeg exited'})")
                    self._kill_proc()
            except Exception:
                self._kill_proc()

        if self._proc is None and ffmpeg and (getattr(cfg, '_use_libx264', False) or cfg.use_nvenc):
            try:
                cmd = [ffmpeg, "-y", "-loglevel", "error",
                       "-f", "rawvideo", "-pix_fmt", "bgr24",
                       "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
                       "-c:v", "libx264", "-preset", "fast",
                       "-crf", "18", "-pix_fmt", "yuv420p", path]
                self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
                if self._proc.poll() is None:
                    self._encoder = "libx264"
                else:
                    err = self._read_proc_stderr()
                    print(f"[writer] libx264 init failed; falling back ({err or 'ffmpeg exited'})")
                    self._kill_proc()
            except Exception:
                self._kill_proc()

        if self._proc is None:
            self._cv = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        if cfg.use_async_writer:
            self._q = queue.Queue(maxsize=cfg.async_queue)
            self._thread = threading.Thread(target=self._drain, daemon=True)
            self._thread.start()
        else:
            self._q = None

    def _kill_proc(self):
        if self._proc:
            try:
                if self._proc.stdin:
                    self._proc.stdin.close()
                self._proc.kill()
            except Exception:
                pass
        self._proc = None

    def _read_proc_stderr(self) -> str:
        if not self._proc or not self._proc.stderr:
            return ""
        try:
            data = self._proc.stderr.read()
        except Exception:
            return ""
        if not data:
            return ""
        try:
            return data.decode("utf-8", errors="replace").strip()
        except Exception:
            return str(data)

    def _raise_proc_write_error(self, prefix: str):
        rc = None
        try:
            rc = self._proc.poll() if self._proc else None
        except Exception:
            rc = None
        err = self._read_proc_stderr()
        details = f"{prefix}. ffmpeg encoder='{self._encoder}'"
        if rc is not None:
            details += f", returncode={rc}"
        if err:
            details += f", stderr={err}"
        raise RuntimeError(details)

    def _raw_write(self, frame):
        if self._proc:
            if self._proc.poll() is not None:
                self._raise_proc_write_error("FFmpeg process exited before frame write")
            try:
                self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())
            except OSError:
                self._raise_proc_write_error("Failed writing frame bytes to FFmpeg")
        else:
            self._cv.write(frame)

    def _drain(self):
        try:
            while True:
                item = self._q.get()
                if item is None:
                    break
                self._raw_write(item)
        except Exception as e:
            self._thread_error = e

    def write(self, frame):
        if self._thread_error is not None:
            raise RuntimeError(f"Async video writer failed: {self._thread_error}")
        if self._q:
            self._q.put(frame)
        else:
            self._raw_write(frame)

    def close(self):
        if self._q:
            self._q.put(None)
            self._thread.join()
            if self._thread_error is not None:
                raise RuntimeError(f"Async video writer failed: {self._thread_error}")
        if self._proc:
            try:
                if self._proc.stdin:
                    self._proc.stdin.close()
            except Exception:
                pass
            self._proc.wait()
            if self._proc.returncode not in (0, None):
                self._raise_proc_write_error("FFmpeg exited with a non-zero status")
        elif self._cv:
            self._cv.release()

class ThreadedFrameReader:
    """Prefetch frames in a background thread to overlap decode with GPU work."""

    def __init__(self, cap: cv2.VideoCapture, prefetch: int = 4):
        self._cap = cap
        self._q = queue.Queue(maxsize=prefetch)
        self._done = False
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        while True:
            ret, frame = self._cap.read()
            if not ret:
                self._q.put(None)
                break
            self._q.put(frame)

    def read(self):
        if self._done:
            return None
        frame = self._q.get()
        if frame is None:
            self._done = True
        return frame

    def release(self):
        self._done = True
        self._cap.release()

