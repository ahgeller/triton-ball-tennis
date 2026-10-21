"""The TensorRT runtime session shared by the ball, player and court engine backends.

Engine policy (player and court): an unset model path skips that detector
with one message; a configured path whose file is missing or whose runtime
fails to initialise raises, so a broken install does not silently produce a
run without players or court.

The runtime requires CUDA torch, so ``torch`` is imported unconditionally and a
broken install fails here, at import.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .config import Config
from .engine_files import iter_engine_io, read_engine_header, resolve_engine_path


def import_tensorrt() -> Any:
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


class TensorRTRuntimeSession:
    """A deserialized TensorRT engine with device buffers and GPU preprocessing.

    Uses the TensorRT 10 tensor API (``iter_engine_io`` rejects a TensorRT 8
    engine).  Inference runs on its own CUDA stream when ``async_execute`` is
    set; ``output_slots`` > 1 keeps that many output buffer sets so a caller
    can launch frame N+1 before reading frame N.
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
        trt = import_tensorrt()
        self.device = device if device is not None else torch.device("cuda:0")
        self.engine, self.context = _deserialize_engine(trt, engine_path)

        self.use_async = bool(async_execute) and hasattr(self.context, "execute_async_v3")
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
                    self.context.set_input_shape(io.name, io.profile_shape)
                if np_dtype == np.float16:
                    self.fp16 = True
            else:
                self.output_names.append(io.name)
            shape_now = tuple(self.context.get_tensor_shape(io.name))
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

    def _reshape_for(self, t: torch.Tensor) -> None:
        """Dynamic engines: adopt ``t``'s shape and reallocate outputs that grew."""
        if not self.dynamic or tuple(t.shape) == tuple(self.bindings[self.input_name].shape):
            return
        inp = self.bindings[self.input_name]
        self.context.set_input_shape(inp.name, tuple(t.shape))
        self.bindings[self.input_name] = inp._replace(shape=tuple(t.shape))
        for name in self.output_names:
            binding = self.bindings[name]
            shape_new = tuple(self.context.get_tensor_shape(name))
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

    def preprocess_cuda_chw_frame(self, frame_bgr_chw: Optional[torch.Tensor]
                                  ) -> Tuple[Optional[torch.Tensor], Optional[Tuple[float, float]]]:
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

    def preprocess_cuda_hwc_frame(self, frame_bgr: torch.Tensor
                                  ) -> Tuple[Optional[torch.Tensor], Optional[Tuple[float, float]]]:
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

        for name in [self.input_name, *self.output_names]:
            self.context.set_tensor_address(name, addrs[name])
        if self.use_async:
            self.stream.wait_stream(torch.cuda.current_stream(device=self.device))
            ok = self.context.execute_async_v3(stream_handle=int(self.stream.cuda_stream))
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


def open_engine_session(label: str, model_path: Optional[str], cfg: Config) -> Optional[TensorRTRuntimeSession]:
    """Apply the engine policy for the player and court detectors.

    No configured path: print one message and return None.  A configured
    path whose ``.engine`` file is missing raises ``FileNotFoundError``; a
    runtime failure raises ``RuntimeError`` from the session.
    """
    if not model_path:
        print(f"[{label}] No engine configured, {label} detection disabled")
        return None
    engine_path = resolve_engine_path(model_path)
    if engine_path is None:
        raise FileNotFoundError(f"[{label}] TensorRT engine not found for '{model_path}'")
    session = TensorRTRuntimeSession(
        str(engine_path),
        device=torch.device(f"cuda:{int(cfg.device)}"),
        async_execute=bool(cfg.trt_async_execute),
    )
    print(f"[{label}] TensorRT runtime initialized: {engine_path}")
    return session
