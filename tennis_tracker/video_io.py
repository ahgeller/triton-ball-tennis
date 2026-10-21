from __future__ import annotations

import contextlib
import queue
import subprocess
import threading
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

from .config import Config
from .utils import (
    ENCODER_LIBX264,
    ENCODER_NVENC,
    ENCODER_OPENCV,
    FFMPEG_ENCODER_NAMES,
    EncoderChoice,
    choose_encoder,
)

LIBX264_PRESET = "fast"
LIBX264_CRF = "18"
_OPENCV_FOURCC = "mp4v"
# How long close() lets FFmpeg flush its last frames, and how long a kill may take.
_FFMPEG_CLOSE_TIMEOUT_S = 30.0
_FFMPEG_KILL_TIMEOUT_S = 5.0
_QUEUE_POLL_S = 0.1


class PinnedFrameUploader:
    """Pinned host staging for faster/non-blocking H2D frame uploads."""

    def __init__(self, h: int, w: int, device: torch.device):
        self.h = int(h)
        self.w = int(w)
        self.device = device
        self._slot = 0
        self._pinned_u8 = [
            torch.empty((self.h, self.w, 3), dtype=torch.uint8, pin_memory=True)
            for _ in range(2)
        ]
        self._pinned_np = [t.numpy() for t in self._pinned_u8]
        self._copy_done: List[Optional[torch.cuda.Event]] = [None, None]

    @torch.no_grad()
    def upload_chw_f32(self, frame_bgr: np.ndarray) -> torch.Tensor:
        if frame_bgr.shape[0] != self.h or frame_bgr.shape[1] != self.w or frame_bgr.shape[2] != 3:
            return torch.from_numpy(frame_bgr).to(
                device=self.device, dtype=torch.float32
            ).permute(2, 0, 1).contiguous() / 255.0
        slot = self._slot
        self._slot = (self._slot + 1) % len(self._pinned_u8)
        if self._copy_done[slot] is not None:
            self._copy_done[slot].synchronize()
        np.copyto(self._pinned_np[slot], frame_bgr)
        t = self._pinned_u8[slot].to(device=self.device, dtype=torch.float32, non_blocking=True)
        done = torch.cuda.Event(blocking=False)
        done.record(torch.cuda.current_stream(device=self.device))
        self._copy_done[slot] = done
        return t.permute(2, 0, 1).contiguous() / 255.0


@torch.no_grad()
def cuda_frame_to_chw_f32(frame_bgr: np.ndarray, device: torch.device,
                           uploader: Optional[PinnedFrameUploader] = None) -> torch.Tensor:
    """Upload one BGR frame once as CHW float32 in [0,1] on CUDA."""
    if uploader is not None:
        return uploader.upload_chw_f32(frame_bgr)
    return torch.from_numpy(frame_bgr).to(
        device=device, dtype=torch.float32
    ).permute(2, 0, 1).contiguous() / 255.0


@torch.no_grad()
def cuda_vs_tensors(frame_bgr: Optional[np.ndarray], device: torch.device,
                     gpu_tensor: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract V and S channels as CUDA tensors (no grad tracking).
    If gpu_tensor is provided, reuse it instead of uploading frame again."""
    if gpu_tensor is not None:
        t = gpu_tensor
    else:
        if frame_bgr is None:
            raise ValueError("frame_bgr is required when gpu_tensor is None")
        t = cuda_frame_to_chw_f32(frame_bgr, device)
    maxc = torch.max(t, dim=0).values
    minc = torch.min(t, dim=0).values
    s = torch.where(maxc > 1e-6, (maxc - minc) / (maxc + 1e-6), torch.zeros_like(maxc))
    return maxc, s


def ffmpeg_command(ffmpeg: str, encoder: str, w: int, h: int, fps: float, path: str, cfg: Config) -> List[str]:
    """The FFmpeg invocation that encodes raw BGR frames from stdin into ``path``.

    ``encoder`` is ``ENCODER_NVENC`` or ``ENCODER_LIBX264``; the codec options
    are the only part that differs.
    """
    if encoder == ENCODER_NVENC:
        codec = ["-c:v", FFMPEG_ENCODER_NAMES[ENCODER_NVENC], "-preset", str(cfg.nvenc_preset),
                 "-b:v", str(cfg.nvenc_bitrate)]
    elif encoder == ENCODER_LIBX264:
        codec = ["-c:v", FFMPEG_ENCODER_NAMES[ENCODER_LIBX264], "-preset", LIBX264_PRESET, "-crf", LIBX264_CRF]
    else:
        raise ValueError(f"not an FFmpeg encoder: {encoder!r}")
    return [
        str(ffmpeg), "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{int(w)}x{int(h)}", "-r", str(fps), "-i", "-",
        *codec,
        "-pix_fmt", "yuv420p", str(path),
    ]


class VideoWriter:
    """Encode BGR frames to ``path`` through FFmpeg (NVENC, then libx264) or OpenCV.

    The encoder order comes from an ``EncoderChoice`` (see
    ``utils.resolve_runtime``); when none is passed it is derived from
    ``cfg`` in the same way.  Each FFmpeg encoder that fails to start is
    reported and the next one is tried; OpenCV is the last resort and raising
    when it cannot open the output.  With ``cfg.use_async_writer`` frames are
    queued to a drain thread; an encoder failure there surfaces on the next
    ``write`` or on ``close``.
    """

    def __init__(self, path: str, fps: float, w: int, h: int, cfg: Config,
                 encoder: Optional[EncoderChoice] = None):
        self._proc: Optional[subprocess.Popen] = None
        self._cv: Optional[cv2.VideoWriter] = None
        self._encoder = ENCODER_OPENCV
        self._thread_error: Optional[BaseException] = None
        self._path = str(path)
        choice = encoder if encoder is not None else choose_encoder(cfg)

        for name in choice.ffmpeg_encoders:
            proc = self._start_ffmpeg(name, ffmpeg_command(choice.ffmpeg, name, w, h, fps, self._path, cfg))
            if proc is not None:
                self._proc, self._encoder = proc, name
                break

        if self._proc is None:
            self._cv = cv2.VideoWriter(self._path, cv2.VideoWriter_fourcc(*_OPENCV_FOURCC), fps, (int(w), int(h)))
            if not self._cv.isOpened():
                self._cv.release()
                raise RuntimeError(f"Could not open video output: {path}")

        self._q: Optional[queue.Queue] = None
        if cfg.use_async_writer:
            self._q = queue.Queue(maxsize=cfg.async_queue)
            self._thread = threading.Thread(target=self._drain, daemon=True)
            self._thread.start()

    @property
    def encoder(self) -> str:
        return self._encoder

    def _start_ffmpeg(self, name: str, cmd: List[str]) -> Optional[subprocess.Popen]:
        """Launch one FFmpeg encoder; None (after a message) when it cannot start."""
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
        except OSError as error:
            print(f"[writer] {name} init failed; falling back ({error})")
            return None
        if proc.poll() is None:
            return proc
        err = _read_stderr(proc)
        print(f"[writer] {name} init failed; falling back ({err or 'ffmpeg exited'})")
        _close_pipes(proc)
        return None

    def _kill_proc(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        if proc.poll() is None:
            proc.kill()
            try:
                proc.wait(timeout=_FFMPEG_KILL_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                print(f"[writer] ffmpeg ({self._encoder}) did not exit after kill")
        _close_pipes(proc)

    def _raise_proc_write_error(self, prefix: str) -> None:
        rc = self._proc.poll() if self._proc else None
        err = _read_stderr(self._proc) if self._proc else ""
        details = f"{prefix}. ffmpeg encoder='{self._encoder}'"
        if rc is not None:
            details += f", returncode={rc}"
        if err:
            details += f", stderr={err}"
        raise RuntimeError(details)

    def _raw_write(self, frame: np.ndarray) -> None:
        if self._proc:
            if self._proc.poll() is not None:
                self._raise_proc_write_error("FFmpeg process exited before frame write")
            try:
                self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())
            except OSError:
                self._raise_proc_write_error("Failed writing frame bytes to FFmpeg")
        else:
            self._cv.write(frame)

    def _drain(self) -> None:
        try:
            while True:
                item = self._q.get()
                if item is None:
                    break
                self._raw_write(item)
        except Exception as e:  # noqa: BLE001 - any failure must reach the producer thread
            self._thread_error = e

    def _enqueue(self, item: Optional[np.ndarray]) -> None:
        while True:
            if self._thread_error is not None:
                raise RuntimeError(f"Async video writer failed: {self._thread_error}") from self._thread_error
            if not self._thread.is_alive():
                raise RuntimeError("Async video writer stopped before accepting output")
            try:
                self._q.put(item, timeout=_QUEUE_POLL_S)
                return
            except queue.Full:
                continue

    def write(self, frame: np.ndarray) -> None:
        if self._q:
            self._enqueue(frame)
        else:
            self._raw_write(frame)

    def close(self) -> None:
        try:
            if self._q:
                self._enqueue(None)
                self._thread.join()
                if self._thread_error is not None:
                    raise RuntimeError(f"Async video writer failed: {self._thread_error}") from self._thread_error
            if self._proc:
                if self._proc.stdin:
                    try:
                        self._proc.stdin.close()
                    except OSError:
                        self._raise_proc_write_error("Failed flushing the last frames to FFmpeg")
                try:
                    self._proc.wait(timeout=_FFMPEG_CLOSE_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    self._raise_proc_write_error(f"FFmpeg did not finish within {_FFMPEG_CLOSE_TIMEOUT_S:.0f}s")
                if self._proc.returncode not in (0, None):
                    self._raise_proc_write_error("FFmpeg exited with a non-zero status")
        finally:
            if self._proc:
                if self._proc.poll() is None:
                    self._kill_proc()
                else:
                    _close_pipes(self._proc)
                    self._proc = None
            if self._cv:
                self._cv.release()
                self._cv = None


def _read_stderr(proc: subprocess.Popen) -> str:
    if not proc.stderr:
        return ""
    try:
        data = proc.stderr.read()
    except (OSError, ValueError):  # pipe broken or already closed
        return ""
    return data.decode("utf-8", errors="replace").strip() if data else ""


def _close_pipes(proc: subprocess.Popen) -> None:
    for pipe in (proc.stdin, proc.stderr):
        if pipe is None:
            continue
        # Closing stdin flushes to a process that may already be gone.
        with contextlib.suppress(OSError):
            pipe.close()


class ThreadedFrameReader:
    """Prefetch frames in a background thread to overlap decode with GPU work."""

    def __init__(self, cap: cv2.VideoCapture, prefetch: int = 4):
        self._cap = cap
        self._q: queue.Queue = queue.Queue(maxsize=prefetch)
        self._done = False
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self) -> None:
        while True:
            ret, frame = self._cap.read()
            if not ret:
                self._q.put(None)
                break
            self._q.put(frame)

    def read(self) -> Optional[np.ndarray]:
        if self._done:
            return None
        frame = self._q.get()
        if frame is None:
            self._done = True
        return frame

    def release(self) -> None:
        self._done = True
        self._cap.release()
