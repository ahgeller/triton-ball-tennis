"""Ball detector backends: GridTrackNet and the single-frame TensorRT engine.

Both sit behind ``BallDetectorBackend``: GridTrackNet (the default, a
five-frame temporal model run as a whole-clip prepass; ``_prepass`` keeps
source-frame indices and temporal order) and the single-frame Ultralytics
TensorRT engine (``Config.model_path`` ending in ``.engine``), driven through
``TensorRTRuntimeSession`` from ``engine_session``.
"""
from __future__ import annotations

import contextlib
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

from .config import Config
from .engine_boxes import decode_engine_boxes, engine_output_mode
from .engine_session import TensorRTRuntimeSession
from .frame_rate import require_phase_stride
from .gridtracknet import HEIGHT, WIDTH, decode_predictions, frame_tensor, load_model

# One ball detection: an xyxy box in source pixels and its confidence.
Detection = Tuple[List[float], float]
# A BGR HWC uint8 frame, either a host array or a CUDA tensor of the same layout.
FrameInput = Union[np.ndarray, torch.Tensor]

_BALL_NMS_IOU = 0.45
# GridTrackNet gives a point; the box handed to the selector is this fraction
# of the frame diagonal around it, and its vertical offset is set at 1080p.
_BOX_RADIUS_FRAC = 0.003
_MIN_BOX_RADIUS_PX = 2.0
_REFERENCE_HEIGHT_PX = 1080.0
_UNITS_PER_INFERENCE = 4
_DEFAULT_NAMES: Dict[int, str] = {0: "ball"}


class BallDetectorBackend:
    """Swappable ball detector interface.

    ``detect`` returns the frame's detections as ``(xyxy_box, confidence)``
    pairs in source pixels.  ``detect_async_start`` / ``detect_async_finish``
    split that into launch and collect so the pipeline can overlap the next
    frame's preprocessing with this frame's inference; the pending token is a
    backend-specific dataclass (``CachedDetections`` or ``TensorRTBallPending``)
    that only the same backend's ``detect_async_finish`` understands.
    ``frame_index`` is the zero-based source frame of ``frame_bgr``; a
    backend that caches a whole clip answers for that frame.

    ``prepare_video`` is called once per clip before the first frame; a
    per-frame backend has nothing to prepare.
    """

    uses_raw_frames = False
    use_async = False
    async_slots = 1

    def prepare_video(self, video_path: Path, fps: float, width: int, height: int, total: int) -> None:
        """Hook for whole-clip backends; ``total`` is the container's frame count hint."""

    def detect(self, frame_bgr: FrameInput, frame_index: int) -> List[Detection]:
        raise NotImplementedError

    def detect_async_start(self, frame_bgr: FrameInput, frame_index: int) -> Any:
        raise NotImplementedError

    def detect_async_finish(self, pending: Any) -> List[Detection]:
        raise NotImplementedError

    def supports_cuda_frame(self) -> bool:
        return False


@dataclass
class CachedDetections:
    """GridTrackNet's pending token: the detections are final when it is issued."""

    frame_index: int
    detections: List[Detection]


class GridTrackNetBallDetector(BallDetectorBackend):
    """Run the pretrained model at its native five-frame, 30-FPS cadence.

    ``prepare_video`` decodes the whole clip once (by default on a worker
    thread, see ``Config.gridtracknet_prepass_background``) and caches one
    detection per decoded source frame; ``detections_at(i)`` returns frame
    ``i``'s result, waiting for the prepass if it is behind, and raises once
    the prepass has finished without ever decoding frame ``i``.

    ``detect`` / ``detect_async_start`` never read the frame they are given:
    the prepass already decoded the clip.  They answer for ``frame_index``,
    so skipping frames is safe and call order never matters.
    """

    uses_raw_frames = True
    use_async = False
    async_slots = 1

    def __init__(self, weights_path: str, cfg: Config):
        if not torch.cuda.is_available():
            raise RuntimeError("GridTrackNet requires CUDA-enabled torch.")
        self.cfg = cfg
        self.device = torch.device(f"cuda:{int(cfg.device)}")
        self._decode = decode_predictions
        self._frame_tensor = frame_tensor
        self.model = load_model(Path(weights_path), self.device)
        # One entry per decoded source frame, appended by the prepass.
        self.precomputed: Optional[List[List[Detection]]] = None
        # Progressive prepass state: frames < ready_upto are final; ``_done``
        # once the prepass has released its capture (with or without an error).
        self._ready_upto = 0
        self._done = False
        self._ready = threading.Condition()
        self._worker: Optional[threading.Thread] = None
        self._worker_error: Optional[BaseException] = None
        with torch.inference_mode():
            self.model(torch.zeros((1, 15, HEIGHT, WIDTH), device=self.device, dtype=torch.float16))
        torch.cuda.synchronize(self.device)
        print(f"[detector] GridTrackNet FP16 (confidence={float(cfg.conf):.2f}, device={self.device})")

    def source_stride_for(self, fps: float) -> int:
        override = self.cfg.gridtracknet_source_stride
        if override:
            return max(1, int(override))
        try:
            return require_phase_stride(fps)
        except ValueError as error:
            raise RuntimeError(str(error)) from error

    def prepare_video(self, video_path: Path, fps: float, width: int, height: int, total: int,
                      background: Optional[bool] = None) -> None:
        """Decode the video once and cache one detection per decoded frame.

        ``total`` is the container's frame count, only a hint: the cache
        grows with the frames actually decoded, and a mismatch is reported.
        ``background`` (default ``cfg.gridtracknet_prepass_background``) runs
        the prepass on a worker thread so pass 1 can start on frame 0 while
        the detector is still ahead of it; ``detect()`` then blocks only if
        pass 1 catches up.  Results are identical to the synchronous pass,
        whose failure raises from here.
        """
        source_stride = self.source_stride_for(fps)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open input video for GridTrackNet: {video_path}")
        self.precomputed = []
        self._ready_upto = 0
        self._done = False
        self._worker_error = None
        if background is None:
            background = bool(self.cfg.gridtracknet_prepass_background)
        args = (capture, source_stride, width, height, int(total))
        if background:
            self._worker = threading.Thread(target=self._prepass, args=args, daemon=True, name="gridtracknet-prepass")
            self._worker.start()
        else:
            self._worker = None
            self._prepass(*args)
            self._raise_if_failed()

    def _publish(self, upto: int) -> None:
        with self._ready:
            self._ready_upto = max(self._ready_upto, upto)
            self._ready.notify_all()

    def _finish(self) -> None:
        with self._ready:
            self._done = True
            self._ready.notify_all()

    def _raise_if_failed(self) -> None:
        error = self._worker_error
        if error is None:
            return
        if not isinstance(error, Exception):   # KeyboardInterrupt, SystemExit: pass through
            raise error
        raise RuntimeError("GridTrackNet prepass failed") from error

    def _prepass(self, capture: cv2.VideoCapture, source_stride: int, width: int, height: int,
                 total_hint: int) -> None:
        cached = self.precomputed
        finalized = set()
        # One independent five-frame unit per phase. The model wants 30 FPS
        # spacing, so at 60 FPS the odd frames form a second, equally valid
        # unit stream; running only phase 0 leaves every odd frame unobserved.
        units = [[] for _ in range(source_stride)]
        recent = [deque(maxlen=5) for _ in range(source_stride)]
        batches = []
        frame_index = inference_calls = detected = 0
        started = time.perf_counter()
        y_offset = float(self.cfg.gridtracknet_y_offset_px) * height / _REFERENCE_HEIGHT_PX
        radius = max(_MIN_BOX_RADIUS_PX, _BOX_RADIUS_FRAC * np.hypot(width, height))

        def flush() -> None:
            nonlocal inference_calls, detected
            if not batches:
                return
            with torch.inference_mode():
                output = self.model(torch.stack([item[0] for item in batches]))
                decoded = self._decode(output, width, height, float(self.cfg.conf))
            for batch_index, (_, indices) in enumerate(batches):
                start = batch_index * 5
                for source_index, (point, confidence) in zip(indices, decoded[start:start + 5]):
                    if source_index in finalized:
                        continue
                    finalized.add(source_index)
                    if point is None:
                        continue
                    x, y = point[0], point[1] + y_offset
                    cached[source_index] = [([x - radius, y - radius, x + radius, y + radius], confidence)]
                    detected += 1
            inference_calls += 1
            # Every frame below the oldest unit still being assembled is final.
            pending_min = min((item[1] for unit in units for item in unit), default=frame_index + 1)
            batches.clear()
            self._publish(min(pending_min, frame_index + 1))

        try:
            # All of the worker's CUDA work lives on its own stream so it overlaps
            # pass 1 on the default stream instead of serialising behind it.
            stream = torch.cuda.Stream(device=self.device) if self._worker is not None else None
            stream_scope = torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()
            with stream_scope:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    cached.append([])
                    phase = frame_index % source_stride
                    unit = units[phase]
                    item = (self._frame_tensor(frame, self.device), frame_index)
                    recent[phase].append(item)
                    unit.append(item)
                    if len(unit) == 5:
                        batches.append((torch.cat([value for value, _ in unit]), [index for _, index in unit]))
                        unit.clear()
                        if len(batches) == _UNITS_PER_INFERENCE:
                            flush()
                    frame_index += 1
                # Cover each phase's trailing frames with one overlapping window.
                for phase in range(source_stride):
                    if units[phase] and len(recent[phase]) == 5:
                        batches.append((
                            torch.cat([value for value, _ in recent[phase]]),
                            [index for _, index in recent[phase]],
                        ))
                flush()
                if stream is not None:
                    stream.synchronize()
                else:
                    torch.cuda.synchronize(self.device)
        except BaseException as error:  # surfaced to the consumer by _raise_if_failed
            self._worker_error = error
        finally:
            capture.release()
            self._publish(len(cached))
            self._finish()

        elapsed = time.perf_counter() - started
        print(
            f"[detector] GridTrackNet prepass: {frame_index} source frames, {detected} detections, "
            f"{inference_calls} batched calls across {source_stride} phase(s) "
            f"in {elapsed:.1f}s ({frame_index / max(elapsed, 1e-9):.1f} source fps)"
            + (" [overlapped with pass 1]" if self._worker is not None else "")
        )
        if self._worker_error is None and frame_index != total_hint:
            print(f"[detector] GridTrackNet prepass: the container reported {total_hint} frames, "
                  f"{frame_index} decoded; the decoded count is used")

    def wait_until_ready(self, index: int) -> None:
        """Block until frame ``index`` is final or the prepass has finished; raises if it failed."""
        with self._ready:
            while self._ready_upto <= index and not self._done:
                self._ready.wait(timeout=0.5)
        self._raise_if_failed()

    def detect_clip(self, video_path: Path, fps: float, width: int, height: int,
                    total: int) -> List[List[Detection]]:
        """Run the prepass in the foreground and return each decoded frame's detections.

        Index ``i`` of the result holds frame ``i``'s detections: an empty list
        when the ball was not found, else one ``(xyxy_box, confidence)`` pair.
        A prepass that fails raises instead of returning partial results.
        """
        self.prepare_video(video_path, fps, width, height, total, background=False)
        return list(self.precomputed)

    def detections_at(self, index: int) -> List[Detection]:
        """Detections for source frame ``index``, waiting for the prepass if it
        is behind.  Raises ``IndexError`` once the prepass has finished and
        decoded no such frame."""
        if self.precomputed is None:
            raise RuntimeError("GridTrackNet.prepare_video() must run before detection.")
        self.wait_until_ready(index)
        if index >= len(self.precomputed):
            raise IndexError(
                f"GridTrackNet prepass decoded {len(self.precomputed)} frames; source frame {index} does not exist"
            )
        return self.precomputed[index]

    def detect(self, frame_bgr: FrameInput, frame_index: int) -> List[Detection]:
        """Detections for source frame ``frame_index``.  ``frame_bgr`` is never
        read: the prepass already decoded the clip."""
        return self.detect_async_start(frame_bgr, frame_index).detections

    def detect_async_start(self, frame_bgr: FrameInput, frame_index: int) -> CachedDetections:
        if frame_index is None:
            raise ValueError("GridTrackNet answers per source frame: frame_index is required")
        index = int(frame_index)
        return CachedDetections(index, self.detections_at(index))

    def detect_async_finish(self, pending: CachedDetections) -> List[Detection]:
        return pending.detections


def decode_ball_predictions(
    preds: torch.Tensor,
    scale: Tuple[float, float],
    conf_threshold: float,
    cls_filter: Optional[List[int]],
    output_mode: str,
) -> List[Detection]:
    """Turn one engine output tensor into ``(xyxy_box, confidence)`` pairs in source pixels."""
    rows = decode_engine_boxes(preds, scale, conf_threshold, _BALL_NMS_IOU, output_mode, cls_filter)
    return [([float(row[0]), float(row[1]), float(row[2]), float(row[3])], float(row[4])) for row in rows]


@dataclass
class TensorRTBallPending:
    """An inference in flight on the ball engine's stream."""

    outputs: List[torch.Tensor]
    scale: Tuple[float, float]
    event: Optional[torch.cuda.Event]
    # Keeps the engine input buffer alive until the inference reading it is done.
    input_tensor: torch.Tensor


class TensorRTRuntimeBallDetector(BallDetectorBackend):
    """Single-frame ball detector for an Ultralytics-exported TensorRT engine.

    Selected when ``Config.model_path`` ends in ``.engine`` (``build_trt_engines.py``
    builds one from ``models/ball.pt``); GridTrackNet is the default backend.
    Each frame is one inference, launched on the session's stream with
    ``cfg.trt_async_slots`` output buffers when ``cfg.trt_async_execute`` is
    set, so the next frame's preprocessing overlaps this frame's inference.
    The caller resolved the engine path, so a missing or broken engine raises.
    ``device`` defaults to ``cfg.device`` (CUDA 0 when that is not an index).
    """

    def __init__(
        self,
        engine_path: str,
        cfg: Config,
        ball_cls_id: Optional[int],
        names: Optional[Dict[int, str]] = None,
        device: Optional[torch.device] = None,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT runtime backend requires CUDA.")
        self.cfg = cfg
        self.names = names or dict(_DEFAULT_NAMES)
        self.ball_cls_id = ball_cls_id
        self.cls_filter = [ball_cls_id] if ball_cls_id is not None else None
        if device is None:
            device = torch.device(f"cuda:{cfg.device}") if str(cfg.device).isdigit() else torch.device("cuda:0")
        self.session = TensorRTRuntimeSession(
            engine_path,
            device=device,
            async_execute=bool(cfg.trt_async_execute),
            output_slots=max(1, int(cfg.trt_async_slots)),
        )
        self.device = self.session.device
        self.use_async = self.session.use_async
        self.async_slots = self.session.output_slot_count

        probe = torch.zeros(
            (1, 3, self.session.input_h, self.session.input_w),
            device=self.device,
            dtype=torch.float16 if self.session.fp16 else torch.float32,
        )
        outputs = self.session.forward_tensor(probe, out_slot=0)
        self.session.wait()
        self.output_mode = engine_output_mode(outputs[0])
        print(
            f"[detector] TensorRT ball engine, {self.output_mode} "
            f"(fp16={self.session.fp16}, async={self.use_async}, slots={self.async_slots}, "
            f"device={self.device}, preprocess=gpu_resize_norm)"
        )

    def supports_cuda_frame(self) -> bool:
        return self.device.type == "cuda"

    def detect(self, frame_bgr: FrameInput, frame_index: int) -> List[Detection]:
        return self.detect_async_finish(self.detect_async_start(frame_bgr, frame_index))

    def detect_async_start(self, frame_bgr: FrameInput, frame_index: int) -> TensorRTBallPending:
        """Launch inference on ``frame_bgr``; ``frame_index`` is part of the
        interface and not needed here (every call runs its own frame)."""
        out_slot = self.session.next_output_slot()
        t = None
        if isinstance(frame_bgr, torch.Tensor):
            t, scale = self.session.preprocess_cuda_hwc_frame(frame_bgr)
            if t is None:
                frame_bgr = frame_bgr.detach().cpu().numpy()
        if t is None:
            t, scale = self.session.preprocess_frame(frame_bgr)
        outputs = self.session.forward_tensor(t, out_slot=out_slot)
        return TensorRTBallPending(outputs, scale, self.session.completion_event(), t)

    def detect_async_finish(self, pending: TensorRTBallPending) -> List[Detection]:
        self.session.wait(pending.event)
        return decode_ball_predictions(
            pending.outputs[0], pending.scale, float(self.cfg.conf), self.cls_filter, self.output_mode
        )

