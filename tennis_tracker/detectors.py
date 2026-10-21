"""Ball, player and court detector backends.

Ball detection has two backends behind ``BallDetectorBackend``: GridTrackNet
(the default, a five-frame temporal model run as a whole-clip prepass) and the
single-frame Ultralytics TensorRT engine (``Config.model_path`` ending in
``.engine``).  Player and court detection are TensorRT engines driven through
``_TensorRTRuntimeSession``, which the ball engine backend shares.

Engine policy (player and court): an unset model path skips that detector
with one message; a configured path whose file is missing or whose runtime
fails to initialise raises, so a broken install does not silently produce a
run without players or court.

The runtime requires CUDA torch, so ``torch`` is imported unconditionally and a
broken install fails here, at import.
"""
from __future__ import annotations

import contextlib
import json
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .config import Config
from .frame_rate import require_phase_stride
from .gridtracknet import HEIGHT, WIDTH, decode_predictions, frame_tensor, load_model
from .motion import _nms_xyxy_np, _xywh_to_xyxy_np
from .utils import _resolve_engine_path, engine_is_trt10, iter_engine_io, read_engine_header

# One ball detection: an xyxy box in source pixels and its confidence.
Detection = Tuple[List[float], float]
# A BGR HWC uint8 frame, either a host array or a CUDA tensor of the same layout.
FrameInput = Union[np.ndarray, torch.Tensor]

_BALL_NMS_IOU = 0.45
_POSTPROCESSED_COLUMNS = 6  # x1, y1, x2, y2, conf, cls rows from an NMS-exporting engine
_OUTPUT_MODE_POSTPROCESSED = "postprocessed"
_OUTPUT_MODE_RAW = "raw_nms"
# GridTrackNet gives a point; the box handed to the selector is this fraction
# of the frame diagonal around it, and its vertical offset is set at 1080p.
_BOX_RADIUS_FRAC = 0.003
_MIN_BOX_RADIUS_PX = 2.0
_REFERENCE_HEIGHT_PX = 1080.0
_UNITS_PER_INFERENCE = 4
_NEVER = -10**9
_DEFAULT_NAMES: Dict[int, str] = {0: "ball"}


def _load_bytetrack() -> Tuple[Optional[type], Optional[str]]:
    """boxmot is optional; player tracking is disabled without it.  Returns
    ``(ByteTrack, None)`` or ``(None, reason)``."""
    try:
        from boxmot import ByteTrack
    except ImportError:
        try:
            from boxmot.trackers.bbox.bytetrack import ByteTrack
        except ImportError as error:
            return None, str(error)
    return ByteTrack, None


def _import_tensorrt() -> Any:
    """tensorrt is only needed by the engine backends, so it is imported here."""
    try:
        import tensorrt as trt  # type: ignore
    except ImportError as error:
        raise RuntimeError(f"TensorRT package not available: {error}") from error
    return trt


def _torch_dtype(np_dtype: Any) -> torch.dtype:
    return torch.from_numpy(np.empty((0,), dtype=np_dtype)).dtype


def _deserialize_engine(trt: Any, engine_path: str) -> Tuple[Any, Any]:
    """Load ``engine_path`` (with or without the Ultralytics metadata header)
    and create its execution context."""
    _, raw = read_engine_header(Path(engine_path))
    logger = trt.Logger(trt.Logger.INFO)
    with trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(raw)
    if engine is None:
        raise RuntimeError(f"Failed to load TensorRT engine: {engine_path}")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError(f"Failed to create a TensorRT execution context for {engine_path}")
    return engine, context


class _Binding(NamedTuple):
    name: str
    index: int
    dtype: Any  # numpy dtype
    shape: Tuple[int, ...]
    data: torch.Tensor
    ptr: int


class _TensorRTRuntimeSession:
    """A deserialized TensorRT engine with device buffers and GPU preprocessing.

    Works with the TensorRT 10 tensor API and the TensorRT 8 binding API.
    Inference runs on its own CUDA stream when ``async_execute`` is set and the
    context supports it; ``output_slots`` > 1 keeps that many output buffer
    sets so a caller can launch frame N+1 before reading frame N.
    """

    def __init__(
        self,
        engine_path: str,
        device: Optional[torch.device] = None,
        async_execute: bool = True,
        output_slots: int = 1,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT session requires CUDA-enabled torch.")
        trt = _import_tensorrt()
        self.device = device if device is not None else torch.device("cuda:0")
        self.engine, self.context = _deserialize_engine(trt, engine_path)
        self.is_trt10 = engine_is_trt10(self.engine)

        self._has_async_v3 = hasattr(self.context, "execute_async_v3")
        self._has_async_v2 = hasattr(self.context, "execute_async_v2")
        self.use_async = bool(async_execute) and (self._has_async_v3 or self._has_async_v2)
        self.stream = torch.cuda.Stream(device=self.device) if self.use_async else None

        self.bindings: "OrderedDict[str, _Binding]" = OrderedDict()
        self.binding_order: List[str] = []
        self.output_names: List[str] = []
        self.input_name: Optional[str] = None
        self.dynamic = False
        self.fp16 = False
        for io in iter_engine_io(self.engine, trt):
            np_dtype = trt.nptype(io.dtype)
            self.binding_order.append(io.name)
            if io.is_input:
                self.input_name = io.name
                if io.dynamic:
                    self.dynamic = True
                    self._set_input_shape(io.name, io.index, io.profile_shape)
                if np_dtype == np.float16:
                    self.fp16 = True
            else:
                self.output_names.append(io.name)
            shape_now = self._current_shape(io.name, io.index)
            tensor = torch.empty(shape_now, dtype=_torch_dtype(np_dtype), device=self.device)
            self.bindings[io.name] = _Binding(io.name, io.index, np_dtype, shape_now, tensor, int(tensor.data_ptr()))

        if self.input_name is None:
            raise RuntimeError("TensorRT engine has no input tensor.")
        if not self.output_names:
            raise RuntimeError("TensorRT engine has no output tensors.")
        inp_shape = tuple(self.bindings[self.input_name].shape)
        if len(inp_shape) != 4:
            raise RuntimeError(f"Unsupported TensorRT input shape: {inp_shape}")
        self.input_h = int(inp_shape[2])
        self.input_w = int(inp_shape[3])

        # Slot 0 is the binding buffers themselves; extra slots are private copies.
        slots = max(1, int(output_slots)) if self.use_async else 1
        self.output_slots: List[Dict[str, torch.Tensor]] = [
            {name: self.bindings[name].data for name in self.output_names}
        ]
        for _ in range(1, slots):
            self.output_slots.append({name: torch.empty_like(self.bindings[name].data) for name in self.output_names})
        self._slot_cursor = 0

    @property
    def output_slot_count(self) -> int:
        return len(self.output_slots)

    def next_output_slot(self) -> int:
        """Round-robin over the output buffer sets."""
        slot = self._slot_cursor
        self._slot_cursor = (slot + 1) % len(self.output_slots)
        return slot

    # --- shapes -----------------------------------------------------------

    def _current_shape(self, name: str, index: int) -> Tuple[int, ...]:
        if self.is_trt10:
            return tuple(self.context.get_tensor_shape(name))
        return tuple(self.context.get_binding_shape(index))

    def _set_input_shape(self, name: str, index: int, shape: Tuple[int, ...]) -> None:
        if self.is_trt10:
            self.context.set_input_shape(name, shape)
        else:
            self.context.set_binding_shape(index, shape)

    def _reshape_for(self, t: torch.Tensor) -> None:
        """Dynamic engines: adopt ``t``'s shape and reallocate outputs that grew."""
        if not self.dynamic or tuple(t.shape) == tuple(self.bindings[self.input_name].shape):
            return
        inp = self.bindings[self.input_name]
        self._set_input_shape(inp.name, inp.index, tuple(t.shape))
        self.bindings[self.input_name] = inp._replace(shape=tuple(t.shape))
        for name in self.output_names:
            binding = self.bindings[name]
            shape_new = self._current_shape(name, binding.index)
            for slot in self.output_slots:
                if tuple(slot[name].shape) != shape_new:
                    slot[name] = torch.empty(shape_new, dtype=slot[name].dtype, device=self.device)
            data = self.output_slots[0][name]
            self.bindings[name] = binding._replace(shape=shape_new, data=data, ptr=int(data.data_ptr()))

    # --- preprocessing ----------------------------------------------------

    def _finish_preprocess(self, t: torch.Tensor, h0: int, w0: int) -> Tuple[torch.Tensor, Tuple[float, float]]:
        if h0 != self.input_h or w0 != self.input_w:
            t = F.interpolate(t, size=(self.input_h, self.input_w), mode="bilinear", align_corners=False)
        if self.fp16:
            t = t.half()
        scale = (w0 / float(self.input_w), h0 / float(self.input_h))
        return t, scale

    def preprocess_frame(self, frame_bgr: np.ndarray) -> Tuple[torch.Tensor, Tuple[float, float]]:
        """Host HWC BGR uint8 frame -> NCHW RGB [0, 1] engine input plus (sx, sy) back to source pixels."""
        h0, w0 = frame_bgr.shape[:2]
        t = torch.from_numpy(frame_bgr).to(device=self.device, dtype=torch.float32)
        t = t.permute(2, 0, 1).unsqueeze(0).contiguous()
        t = t[:, [2, 1, 0], :, :] / 255.0
        return self._finish_preprocess(t, h0, w0)

    def preprocess_cuda_chw_frame(self, frame_bgr_chw: Optional[torch.Tensor]):
        """Preprocess an already-uploaded CHW BGR float tensor in [0, 1]."""
        if frame_bgr_chw is None:
            return None, None
        if frame_bgr_chw.dim() == 3:
            t = frame_bgr_chw.unsqueeze(0)
        elif frame_bgr_chw.dim() == 4:
            t = frame_bgr_chw
        else:
            return None, None
        h0, w0 = int(t.shape[-2]), int(t.shape[-1])
        t = t[:, [2, 1, 0], :, :].contiguous()
        return self._finish_preprocess(t, h0, w0)

    def preprocess_cuda_hwc_frame(self, frame_bgr: torch.Tensor):
        """Preprocess a CUDA HWC BGR frame (uint8 or float); ``(None, None)`` when
        it is not on this session's device or not HWC with three channels."""
        if frame_bgr.device != self.device or frame_bgr.dim() != 3 or frame_bgr.shape[-1] != 3:
            return None, None
        h0, w0 = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
        if h0 <= 0 or w0 <= 0:
            return None, None
        t = frame_bgr
        if t.dtype == torch.uint8:
            t = t.to(dtype=torch.float32).mul_(1.0 / 255.0)
        else:
            if t.dtype != torch.float32:
                t = t.float()
            if t.max() > 1.5:
                t = t / 255.0
        t = t.permute(2, 0, 1).unsqueeze(0).contiguous()
        t = t[:, [2, 1, 0], :, :]
        return self._finish_preprocess(t, h0, w0)

    # --- execution --------------------------------------------------------

    def forward_tensor(self, t: torch.Tensor, out_slot: int = 0) -> List[torch.Tensor]:
        """Run the engine on ``t``; returns the output tensors (sorted by name) of
        ``out_slot``.  Asynchronous sessions return before the GPU is done: call
        ``wait`` (or ``wait(completion_event())``) before reading them."""
        self._reshape_for(t)
        expected = tuple(self.bindings[self.input_name].shape)
        if tuple(t.shape) != expected:
            raise RuntimeError(f"TensorRT input shape mismatch: got {tuple(t.shape)}, expected {expected}")

        slot = self.output_slots[int(out_slot) % len(self.output_slots)]
        addrs = {name: binding.ptr for name, binding in self.bindings.items()}
        addrs[self.input_name] = int(t.data_ptr())
        for name in self.output_names:
            addrs[name] = int(slot[name].data_ptr())
        ptrs = [addrs[name] for name in self.binding_order]

        if self.is_trt10 and hasattr(self.context, "set_tensor_address"):
            for name in [self.input_name, *self.output_names]:
                self.context.set_tensor_address(name, addrs[name])
        if self.use_async:
            self.stream.wait_stream(torch.cuda.current_stream(device=self.device))
            if self.is_trt10 and self._has_async_v3:
                ok = self.context.execute_async_v3(stream_handle=int(self.stream.cuda_stream))
            else:
                ok = self.context.execute_async_v2(ptrs, int(self.stream.cuda_stream))
            if ok is False:
                raise RuntimeError("TensorRT async execution failed.")
        else:
            ok = self.context.execute_v2(ptrs)
            if ok is False:
                raise RuntimeError("TensorRT execution failed.")
        return [slot[name] for name in sorted(self.output_names)]

    def completion_event(self) -> Optional[torch.cuda.Event]:
        """An event marking the end of the work launched so far on the inference
        stream, or None for synchronous sessions."""
        if not self.use_async:
            return None
        event = torch.cuda.Event(blocking=False)
        event.record(self.stream)
        return event

    def wait(self, event: Optional[torch.cuda.Event] = None) -> None:
        """Make the current stream wait for ``event`` or, without one, for every
        inference launched so far."""
        if event is not None:
            torch.cuda.current_stream(device=self.device).wait_event(event)
        elif self.use_async:
            torch.cuda.current_stream(device=self.device).wait_stream(self.stream)


def _open_engine_session(label: str, model_path: Optional[str], cfg: Config) -> Optional[_TensorRTRuntimeSession]:
    """Apply the engine policy for the player and court detectors.

    No configured path: print one message and return None.  A configured
    path whose ``.engine`` file is missing raises ``FileNotFoundError``; a
    runtime failure raises ``RuntimeError`` from the session.
    """
    if not model_path:
        print(f"[{label}] No engine configured, {label} detection disabled")
        return None
    engine_path = _resolve_engine_path(model_path)
    if engine_path is None:
        raise FileNotFoundError(f"[{label}] TensorRT engine not found for '{model_path}'")
    session = _TensorRTRuntimeSession(
        str(engine_path),
        device=torch.device(f"cuda:{int(cfg.device)}"),
        async_execute=bool(cfg.trt_async_execute),
    )
    print(f"[{label}] TensorRT runtime initialized: {engine_path}")
    return session


class BallDetectorBackend:
    """Swappable ball detector interface.

    ``detect`` returns the frame's detections as ``(xyxy_box, confidence)``
    pairs in source pixels.  ``detect_async_start`` / ``detect_async_finish``
    split that into launch and collect so the pipeline can overlap the next
    frame's preprocessing with this frame's inference; the pending token is a
    backend-specific dataclass (``CachedDetections`` or ``TensorRTBallPending``)
    that only the same backend's ``detect_async_finish`` understands.
    ``frame_index`` is the zero-based source frame; backends that cache a
    whole clip use it to look the frame up.
    """

    uses_raw_frames = False
    use_async = False
    async_slots = 1

    def detect(self, frame_bgr: FrameInput, frame_index: Optional[int] = None) -> List[Detection]:
        raise NotImplementedError

    def detect_async_start(self, frame_bgr: FrameInput, frame_index: Optional[int] = None) -> Any:
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
    thread) and caches one detection per source frame; ``detections_at(i)``
    returns frame ``i``'s result, waiting for the prepass if it is behind.

    ``detect`` / ``detect_async_start`` without ``frame_index`` hand out the
    cached results in call order, one call per source frame: they never read
    the frame they are given.  That contract breaks as soon as a caller skips
    a frame, which is why the constructor refuses ``cfg.skip_frame_detect > 1``
    (every later frame would receive an earlier frame's detections with no
    error).  Pass ``frame_index`` to be independent of call order.
    """

    uses_raw_frames = True
    use_async = False
    async_slots = 1

    def __init__(self, weights_path: str, cfg: Config):
        if not torch.cuda.is_available():
            raise RuntimeError("GridTrackNet requires CUDA-enabled torch.")
        if int(cfg.skip_frame_detect) > 1:
            raise ValueError(
                "GridTrackNet returns one cached result per source frame in call order; "
                f"skip_frame_detect={cfg.skip_frame_detect} would skip calls and shift every later "
                "frame's detections. Use skip_frame_detect=1 with this backend."
            )
        self.cfg = cfg
        self.device = torch.device(f"cuda:{int(cfg.device)}")
        self._decode = decode_predictions
        self._frame_tensor = frame_tensor
        torch.backends.cudnn.benchmark = True
        self.model = load_model(Path(weights_path), self.device)
        self.precomputed: Optional[List[List[Detection]]] = None
        self.precomputed_index = 0
        # Progressive prepass state: frames < ready_upto are final.
        self._ready_upto = 0
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
        """Decode the video once and cache one detection per frame.

        By default this runs on a worker thread so pass 1 can start on frame
        0 while the detector is still ahead of it; ``detect()`` blocks only if
        pass 1 catches up.  Results are identical to the synchronous pass.
        """
        source_stride = self.source_stride_for(fps)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open input video for GridTrackNet: {video_path}")
        self.precomputed = [[] for _ in range(max(0, int(total)))]
        self.precomputed_index = 0
        self._ready_upto = 0
        self._worker_error = None
        if background is None:
            background = bool(self.cfg.gridtracknet_prepass_background)
        if background:
            self._worker = threading.Thread(
                target=self._prepass, args=(capture, source_stride, width, height), daemon=True, name="gridtracknet-prepass"
            )
            self._worker.start()
        else:
            self._worker = None
            self._prepass(capture, source_stride, width, height)

    def _publish(self, upto: int) -> None:
        with self._ready:
            self._ready_upto = max(self._ready_upto, upto)
            self._ready.notify_all()

    def _prepass(self, capture: cv2.VideoCapture, source_stride: int, width: int, height: int) -> None:
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
        stream = torch.cuda.Stream(device=self.device) if self._worker is not None else None

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

        # All of the worker's CUDA work lives on its own stream so it overlaps
        # pass 1 on the default stream instead of serialising behind it.
        stream_scope = torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()
        try:
            with stream_scope:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    if frame_index < len(cached):
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
        except BaseException as error:  # surfaced to the consumer in detect()
            self._worker_error = error
        finally:
            capture.release()
            self._publish(len(cached))

        elapsed = time.perf_counter() - started
        print(
            f"[detector] GridTrackNet prepass: {frame_index} source frames, {detected} detections, "
            f"{inference_calls} batched calls across {source_stride} phase(s) "
            f"in {elapsed:.1f}s ({frame_index / max(elapsed, 1e-9):.1f} source fps)"
            + (" [overlapped with pass 1]" if self._worker is not None else "")
        )

    def wait_until_ready(self, index: int) -> None:
        with self._ready:
            while self._ready_upto <= index and self._worker_error is None:
                if self._worker is not None and not self._worker.is_alive() and self._ready_upto <= index:
                    break
                self._ready.wait(timeout=0.5)
        if self._worker_error is not None:
            raise RuntimeError("GridTrackNet prepass failed") from self._worker_error

    def detect_clip(self, video_path: Path, fps: float, width: int, height: int,
                    total: int) -> List[List[Detection]]:
        """Run the prepass in the foreground and return each source frame's detections.

        Index ``i`` of the result holds frame ``i``'s detections: an empty list
        when the ball was not found, else one ``(xyxy_box, confidence)`` pair.
        """
        self.prepare_video(video_path, fps, width, height, total, background=False)
        return list(self.precomputed)

    def detections_at(self, index: int) -> List[Detection]:
        """Detections for source frame ``index``, waiting for the prepass if it is behind."""
        if self.precomputed is None:
            raise RuntimeError("GridTrackNet.prepare_video() must run before detection.")
        if index >= len(self.precomputed):
            return []
        self.wait_until_ready(index)
        return self.precomputed[index]

    def next_detections(self) -> Tuple[int, List[Detection]]:
        """The next source frame in decode order and its detections.

        This is the call-order contract: the n-th call answers frame n.
        """
        index = self.precomputed_index
        self.precomputed_index += 1
        return index, self.detections_at(index)

    def detect(self, frame_bgr: FrameInput, frame_index: Optional[int] = None) -> List[Detection]:
        """Detections for ``frame_index``, or for the next frame in call order
        when it is None.  ``frame_bgr`` is never read: the prepass already
        decoded the clip."""
        return self.detect_async_start(frame_bgr, frame_index).detections

    def detect_async_start(self, frame_bgr: FrameInput, frame_index: Optional[int] = None) -> CachedDetections:
        if frame_index is None:
            index, detections = self.next_detections()
        else:
            index = int(frame_index)
            detections = self.detections_at(index)
            self.precomputed_index = index + 1  # keep a later call-order caller in step
        return CachedDetections(index, detections)

    def detect_async_finish(self, pending: CachedDetections) -> List[Detection]:
        return pending.detections


def _rescale_box(det: Any, scale: Tuple[float, float]) -> List[float]:
    sx, sy = scale
    return [float(det[0]) * sx, float(det[1]) * sy, float(det[2]) * sx, float(det[3]) * sy]


def ball_engine_output_mode(pred: torch.Tensor) -> str:
    """``postprocessed`` for engines that export NMS'd ``[x1, y1, x2, y2, conf, cls]``
    rows, ``raw_nms`` for raw ``[xywh, class scores...]`` heads."""
    if pred.dim() == 3 and pred.shape[-1] == _POSTPROCESSED_COLUMNS:
        return _OUTPUT_MODE_POSTPROCESSED
    return _OUTPUT_MODE_RAW


def decode_ball_predictions(
    preds: torch.Tensor,
    scale: Tuple[float, float],
    conf_threshold: float,
    cls_filter: Optional[List[int]],
    output_mode: str,
) -> List[Detection]:
    """Turn one engine output tensor into ``(xyxy_box, confidence)`` pairs in source pixels."""
    dets: List[Detection] = []
    if output_mode == _OUTPUT_MODE_POSTPROCESSED:
        if preds.dim() == 3:
            preds = preds[0]
        rows = preds.detach().float().cpu().numpy()
        if rows.size == 0 or rows.shape[1] < _POSTPROCESSED_COLUMNS:
            return dets
        for det in rows:
            conf = float(det[4])
            if conf < conf_threshold:
                continue
            cls_id = int(det[5])
            if cls_filter is not None and cls_id not in cls_filter:
                continue
            dets.append((_rescale_box(det, scale), conf))
        return dets

    if preds.dim() != 3:
        return dets
    p = preds[0]
    if p.shape[0] <= p.shape[1]:
        p = p.transpose(0, 1)
    p = p.detach().float().cpu().numpy()
    if p.size == 0 or p.shape[1] <= 4:
        return dets
    boxes = _xywh_to_xyxy_np(p[:, :4].astype(np.float32))
    cls_scores = p[:, 4:].astype(np.float32)
    if cls_scores.shape[1] > 1:
        cls_ids = np.argmax(cls_scores, axis=1)
        scores = cls_scores[np.arange(cls_scores.shape[0]), cls_ids]
    else:
        cls_ids = np.zeros((cls_scores.shape[0],), dtype=np.int32)
        scores = cls_scores[:, 0]
    keep = scores >= float(conf_threshold)
    boxes, scores, cls_ids = boxes[keep], scores[keep], cls_ids[keep]
    if boxes.shape[0] == 0:
        return dets
    keep_idx = _nms_xyxy_np(boxes, scores, _BALL_NMS_IOU)
    boxes, scores, cls_ids = boxes[keep_idx], scores[keep_idx], cls_ids[keep_idx]
    for i in range(boxes.shape[0]):
        cls_id = int(cls_ids[i])
        if cls_filter is not None and cls_id not in cls_filter:
            continue
        dets.append((_rescale_box(boxes[i], scale), float(scores[i])))
    return dets


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
        self.session = _TensorRTRuntimeSession(
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
        self.output_mode = ball_engine_output_mode(outputs[0])
        print(
            f"[detector] TensorRT ball engine, {self.output_mode} "
            f"(fp16={self.session.fp16}, async={self.use_async}, slots={self.async_slots}, "
            f"device={self.device}, preprocess=gpu_resize_norm)"
        )

    def supports_cuda_frame(self) -> bool:
        return self.device.type == "cuda"

    def detect(self, frame_bgr: FrameInput, frame_index: Optional[int] = None) -> List[Detection]:
        return self.detect_async_finish(self.detect_async_start(frame_bgr, frame_index))

    def detect_async_start(self, frame_bgr: FrameInput, frame_index: Optional[int] = None) -> TensorRTBallPending:
        """Launch inference on ``frame_bgr``; ``frame_index`` is accepted for
        interface parity and not needed (every call runs its own frame)."""
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


@dataclass
class PlayerPending:
    """A player detection launched for one frame.

    ``outputs`` is None on frames between detector runs (``cfg.player_detect_interval``):
    the tracker is then updated with no detections and coasts its boxes.
    """

    frame: np.ndarray
    frame_idx: int
    width: int
    height: int
    court_kps: Optional[List[float]]
    outputs: Optional[List[torch.Tensor]] = None
    input_tensor: Optional[torch.Tensor] = None  # keeps the engine input alive until wait()


class PlayerDetector:
    """TensorRT player detection with ByteTrack association (no ReID).

    Engine policy: see ``_open_engine_session``.  Without boxmot the detector
    is disabled with one message, since the boxes are only useful tracked.
    ``detect_async_start`` returns None while disabled and
    ``detect_async_finish`` then returns no boxes.
    """

    def __init__(self, cfg: Config, court_keypoints: Optional[List[float]] = None):
        self.cfg = cfg
        self.court_kps = court_keypoints
        self.cached_boxes: List[List[float]] = []
        self._frame_diag = 1.0
        self.session: Optional[_TensorRTRuntimeSession] = None
        self.tracker: Optional[Any] = None
        # Detection cadence is controlled by the CLI/config, while ByteTrack
        # still receives an update every frame and coasts on empty detections.
        self._current_interval = max(1, int(cfg.player_detect_interval))
        # ByteTrack's live boxes keyed by track id.
        self.slots: Dict[int, List[float]] = {}

        if not cfg.player_model_path:
            print("[player] No engine configured, player tracking disabled")
            return
        bytetrack, reason = _load_bytetrack()
        if bytetrack is None:
            print(f"[player] boxmot not found ({reason}); player tracking disabled. Run 'pip install boxmot'")
            return
        self.session = _open_engine_session("player", cfg.player_model_path, cfg)
        self.tracker = bytetrack(
            min_conf=0.1,
            track_thresh=0.3,
            track_buffer=60,
            match_thresh=0.8,
        )
        print("[player] ByteTrack tracking initialized")

    @property
    def enabled(self) -> bool:
        return self.session is not None and self.tracker is not None

    @staticmethod
    def _center(b: List[float]) -> Tuple[float, float]:
        return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)

    def set_court_keypoints(self, court_keypoints: Optional[List[float]]) -> None:
        self.court_kps = court_keypoints

    def _decode_player_boxes(self, pred_t: torch.Tensor, frame_w: int, frame_h: int) -> np.ndarray:
        """Engine output -> ``[x1, y1, x2, y2, conf, cls]`` rows in frame pixels."""
        pred = pred_t.detach().float()
        if pred.dim() == 3 and pred.shape[-1] == _POSTPROCESSED_COLUMNS:
            rows = pred[0].cpu().numpy()
            if rows.size == 0:
                return np.empty((0, 6))
            conf = rows[:, 4]
            keep = conf >= float(self.cfg.player_conf)
            rows = rows[keep]
            return rows

        if pred.dim() == 3:
            p = pred[0]
            if p.shape[0] <= p.shape[1]:
                p = p.transpose(0, 1)
            p = p.cpu().numpy()
        else:
            return np.empty((0, 6))

        if p.shape[1] < 5:
            return np.empty((0, 6))

        boxes_xyxy = _xywh_to_xyxy_np(p[:, :4].astype(np.float32))
        scores = p[:, 4].astype(np.float32)
        keep = scores >= float(self.cfg.player_conf)
        boxes_xyxy = boxes_xyxy[keep]
        scores = scores[keep]

        if boxes_xyxy.shape[0] == 0:
            return np.empty((0, 6))

        keep_idx = _nms_xyxy_np(boxes_xyxy, scores, float(self.cfg.player_iou))
        boxes_xyxy = boxes_xyxy[keep_idx]
        scores = scores[keep_idx]

        sx = frame_w / float(max(self.session.input_w, 1))
        sy = frame_h / float(max(self.session.input_h, 1))
        boxes_xyxy[:, [0, 2]] *= sx
        boxes_xyxy[:, [1, 3]] *= sy
        boxes_xyxy[:, 0] = np.clip(boxes_xyxy[:, 0], 0, frame_w - 1)
        boxes_xyxy[:, 2] = np.clip(boxes_xyxy[:, 2], 0, frame_w - 1)
        boxes_xyxy[:, 1] = np.clip(boxes_xyxy[:, 1], 0, frame_h - 1)
        boxes_xyxy[:, 3] = np.clip(boxes_xyxy[:, 3], 0, frame_h - 1)

        # Format for boxmot: [x1, y1, x2, y2, conf, class_id]
        class_ids = np.zeros_like(scores)  # Usually 0 for 'person'
        return np.column_stack((boxes_xyxy, scores, class_ids))

    def detect_async_start(self, frame: np.ndarray, frame_idx: int,
                           frame_gpu_t: Optional[torch.Tensor] = None) -> Optional[PlayerPending]:
        if not self.enabled:
            return None

        h, w = frame.shape[:2]
        self._frame_diag = (w ** 2 + h ** 2) ** 0.5
        pending = PlayerPending(frame, int(frame_idx), int(w), int(h), self.court_kps)

        # The engine runs every N frames; the tracker is still finished every
        # frame, with empty detections in between.
        if frame_idx % self._current_interval != 0:
            return pending

        t = None
        if frame_gpu_t is not None:
            t, _ = self.session.preprocess_cuda_chw_frame(frame_gpu_t)
        if t is None:
            t, _ = self.session.preprocess_frame(frame)
        pending.outputs = self.session.forward_tensor(t)
        pending.input_tensor = t
        return pending

    def _keep_nearest_to_court(self, dets: np.ndarray, court_kps: List[float]) -> np.ndarray:
        """When the engine finds more people than players, keep the ones closest to a court keypoint."""
        scored = []
        for row in dets:
            bx1, by1, bx2, by2 = row[:4]
            bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
            dist = min(((bcx - court_kps[ki]) ** 2 + (bcy - court_kps[ki + 1]) ** 2) ** 0.5
                       for ki in range(0, len(court_kps), 2))
            scored.append((dist, row))
        scored.sort(key=lambda x: x[0])
        return np.array([r for _, r in scored[:self.cfg.num_players]])

    def detect_async_finish(self, pending: Optional[PlayerPending]) -> List[List[float]]:
        if pending is None:
            return []

        if pending.outputs is None:
            # Pass empty detections to tell the tracker to coast the boxes forward.
            dets = np.empty((0, 6))
        else:
            self.session.wait()
            dets = self._decode_player_boxes(pending.outputs[0], pending.width, pending.height)
            # Drop spectators before the tracker learns them.
            if pending.court_kps and len(dets) > self.cfg.num_players:
                dets = self._keep_nearest_to_court(dets, pending.court_kps)
            if len(dets) == 0:
                dets = np.empty((0, 6))

        # tracked_objects rows: [x1, y1, x2, y2, track_id, conf, cls, ind]
        tracked_objects = self.tracker.update(dets, pending.frame)

        boxes: List[List[float]] = []
        self.slots.clear()
        for track in tracked_objects:
            x1, y1, x2, y2, track_id, _conf, _cls, _ = track
            box = [float(x1), float(y1), float(x2), float(y2)]
            boxes.append(box)
            self.slots[int(track_id)] = box

        self.cached_boxes = boxes
        return boxes

    def detect(self, frame: np.ndarray, frame_idx: int) -> List[List[float]]:
        return self.detect_async_finish(self.detect_async_start(frame, frame_idx))

    def get_player_dict(self) -> Dict[int, List[float]]:
        return dict(self.slots)


class CourtDetector:
    """Runs the court keypoint TensorRT engine every N frames and caches the last good result.

    Engine policy: see ``_open_engine_session``; while disabled ``detect``
    returns None and ``draw`` leaves the frame alone.
    """
    _MODEL_TO_SEMANTIC_14 = (0, 4, 6, 1, 2, 5, 7, 3, 8, 12, 9, 11, 13, 10)
    _LINE_PAIRS_14 = (
        (0, 4), (4, 6), (6, 1),
        (0, 2), (1, 3),
        (2, 5), (5, 7), (7, 3),
        (4, 8), (8, 10), (10, 5),
        (6, 9), (9, 11), (11, 7),
        (8, 12), (12, 9),
        (10, 13), (13, 11),
        (12, 13),
    )
    # Rows are [x1, y1, x2, y2, conf, cls, kx, ky, kconf, ...].
    _KEYPOINT_COLUMN = 6
    _MIN_KEYPOINTS = 4

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.keypoints: Optional[List[float]] = None
        self.last_run_frame = _NEVER
        self.session = _open_engine_session("court", cfg.court_model_path, cfg)
        if self.session is not None:
            remap = "enabled" if cfg.court_remap_semantic_14 else "disabled (raw model order)"
            print(f"[court] 14-point semantic remap: {remap}")

    @classmethod
    def _remap_kpt_indices(cls, xy: np.ndarray) -> np.ndarray:
        pts = np.asarray(xy, dtype=np.float32)
        if pts.shape[0] != len(cls._MODEL_TO_SEMANTIC_14):
            return pts
        out = np.zeros_like(pts)
        for model_i, semantic_i in enumerate(cls._MODEL_TO_SEMANTIC_14):
            out[semantic_i] = pts[model_i]
        return out

    def _print_raw_result(self, rows: np.ndarray, frame_idx: int) -> None:
        best_idx = int(np.argmax(rows[:, 4])) if rows.size else -1
        print(f"[court][raw] frame={frame_idx} rows={rows.shape[0]} best={best_idx}")
        if best_idx >= 0:
            r = rows[best_idx]
            print(json.dumps({
                "x1": float(r[0]), "y1": float(r[1]), "x2": float(r[2]), "y2": float(r[3]),
                "conf": float(r[4]), "cls": float(r[5]),
            }))

    def _report_miss(self, frame_idx: int) -> Optional[List[float]]:
        if self.cfg.print_court_raw:
            msg = "miss, using cached court" if self.keypoints else "no court detected"
            print(f"[court] frame={frame_idx}: {msg}")
        return self.keypoints

    def detect(self, frame: np.ndarray, frame_idx: Optional[int] = None) -> Optional[List[float]]:
        if self.session is None:
            return self.keypoints
        interval = max(1, int(self.cfg.court_detect_interval))
        if frame_idx is None:
            frame_idx = self.last_run_frame + interval
        if self.keypoints is not None and frame_idx != 0 and \
                (frame_idx - self.last_run_frame) < interval:
            return self.keypoints

        self.last_run_frame = frame_idx
        t, scale = self.session.preprocess_frame(frame)
        outputs = self.session.forward_tensor(t)
        self.session.wait()
        out = outputs[0].detach().float()
        rows = out[0].cpu().numpy() if out.dim() == 3 else np.empty((0, 0), dtype=np.float32)
        if rows.size == 0:
            return self.keypoints
        if self.cfg.print_court_raw:
            self._print_raw_result(rows, frame_idx)

        conf = rows[:, 4] if rows.shape[1] > 4 else np.zeros((rows.shape[0],), dtype=np.float32)
        best_i = int(np.argmax(conf))
        if float(conf[best_i]) < float(self.cfg.court_conf):
            return self._report_miss(frame_idx)

        row = rows[best_i]
        kpt_raw = row[self._KEYPOINT_COLUMN:]
        nk = int(kpt_raw.shape[0] // 3)
        if nk <= 0:
            return self.keypoints
        k = kpt_raw[: nk * 3].reshape(nk, 3)
        xy = k[:, :2].astype(np.float32)
        sx, sy = float(scale[0]), float(scale[1])
        xy[:, 0] *= sx
        xy[:, 1] *= sy
        if xy.shape[0] == 14 and self.cfg.court_remap_semantic_14:
            xy = self._remap_kpt_indices(xy)

        flat = np.asarray(xy, dtype=np.float32).reshape(-1).astype(float).tolist()
        if len(flat) >= 2 * self._MIN_KEYPOINTS:
            self.keypoints = flat
            return self.keypoints
        return self._report_miss(frame_idx)

    def draw(self, frame: np.ndarray) -> np.ndarray:
        if self.keypoints is None:
            return frame
        kps = self.keypoints
        n = len(kps) // 2

        for ki in range(0, len(kps) - 1, 2):
            x, y = int(kps[ki]), int(kps[ki + 1])
            if x > 0 or y > 0:
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                if self.cfg.court_draw_indices:
                    cv2.putText(frame, str(ki // 2), (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        if self.cfg.court_points_only:
            return frame

        line_pairs: Any = []
        if n == 14:
            line_pairs = self._LINE_PAIRS_14
        elif n >= self._MIN_KEYPOINTS:
            line_pairs = [(j, j + 1) for j in range(n - 1)]

        for a, b in line_pairs:
            if a < n and b < n:
                ax, ay = int(kps[a * 2]), int(kps[a * 2 + 1])
                bx, by = int(kps[b * 2]), int(kps[b * 2 + 1])
                if (ax > 0 or ay > 0) and (bx > 0 or by > 0):
                    cv2.line(frame, (ax, ay), (bx, by), (0, 255, 0), 2)
        return frame
