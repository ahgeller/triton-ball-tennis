from __future__ import annotations

import contextlib
import json
import threading
import time
from collections import OrderedDict, deque, namedtuple
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
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
    try:
        from boxmot.trackers.bbox.bytetrack import ByteTrack
    except ImportError:
        print("[warning] boxmot not found. Player tracking will be disabled. Run 'pip install boxmot'")
        ByteTrack = None

from .config import Config
from .utils import _resolve_engine_path
from .motion import _xywh_to_xyxy_np, _nms_xyxy_np


class _TensorRTRuntimeSession:
    """Minimal TensorRT runtime session with GPU resize/normalize preprocessing."""

    def __init__(
        self,
        engine_path: str,
        device: Optional[torch.device] = None,
        async_execute: bool = True,
    ):
        if not HAS_TORCH or not torch.cuda.is_available():
            raise RuntimeError("TensorRT session requires CUDA-enabled torch.")
        try:
            import tensorrt as trt  # type: ignore
        except Exception as e:
            raise RuntimeError(f"TensorRT package not available: {e}")

        self.device = device if device is not None else torch.device("cuda:0")
        Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))

        raw = Path(engine_path).read_bytes()
        try:
            meta_len = int.from_bytes(raw[:4], byteorder="little")
            json.loads(raw[4 : 4 + meta_len].decode("utf-8"))
            raw = raw[4 + meta_len :]
        except Exception:
            pass

        logger = trt.Logger(trt.Logger.INFO)
        with trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(raw)
        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context.")

        self.is_trt10 = not hasattr(self.engine, "num_bindings")
        self.bindings: OrderedDict[str, Any] = OrderedDict()
        self.binding_addrs: OrderedDict[str, int] = OrderedDict()
        self.output_names: List[str] = []
        self.input_name: Optional[str] = None
        self.dynamic = False
        self.fp16 = False
        self.use_async = bool(async_execute)
        self.stream = torch.cuda.Stream(device=self.device) if self.use_async else None
        self._has_async_v3 = hasattr(self.context, "execute_async_v3")
        self._has_async_v2 = hasattr(self.context, "execute_async_v2")
        if self.use_async and not (self._has_async_v3 or self._has_async_v2):
            self.use_async = False
            self.stream = None

        num = range(self.engine.num_io_tensors) if self.is_trt10 else range(self.engine.num_bindings)
        for i in num:
            if self.is_trt10:
                name = self.engine.get_tensor_name(i)
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                shape = tuple(self.engine.get_tensor_shape(name))
                profile_shape = tuple(self.engine.get_tensor_profile_shape(name, 0)[2]) if is_input else None
            else:
                name = self.engine.get_binding_name(i)
                dtype = trt.nptype(self.engine.get_binding_dtype(i))
                is_input = self.engine.binding_is_input(i)
                shape = tuple(self.engine.get_binding_shape(i))
                profile_shape = tuple(self.engine.get_profile_shape(0, i)[1]) if is_input else None

            if is_input:
                self.input_name = name
                if -1 in shape:
                    self.dynamic = True
                    if self.is_trt10:
                        self.context.set_input_shape(name, profile_shape)
                    else:
                        self.context.set_binding_shape(i, profile_shape)
                if dtype == np.float16:
                    self.fp16 = True
            else:
                self.output_names.append(name)

            shape_now = tuple(self.context.get_tensor_shape(name)) if self.is_trt10 else tuple(self.context.get_binding_shape(i))
            tensor = torch.from_numpy(np.empty(shape_now, dtype=dtype)).to(self.device)
            self.bindings[name] = Binding(name, dtype, shape_now, tensor, int(tensor.data_ptr()))
            self.binding_addrs[name] = int(tensor.data_ptr())

        if self.input_name is None:
            raise RuntimeError("TensorRT engine has no input tensor.")
        if not self.output_names:
            raise RuntimeError("TensorRT engine has no output tensors.")
        inp_shape = tuple(self.bindings[self.input_name].shape)
        if len(inp_shape) != 4:
            raise RuntimeError(f"Unsupported TensorRT input shape: {inp_shape}")
        self.input_h = int(inp_shape[2])
        self.input_w = int(inp_shape[3])

    def preprocess_frame(self, frame_bgr: np.ndarray):
        h0, w0 = frame_bgr.shape[:2]
        t = torch.from_numpy(frame_bgr).to(device=self.device, dtype=torch.float32)
        t = t.permute(2, 0, 1).unsqueeze(0).contiguous()
        t = t[:, [2, 1, 0], :, :] / 255.0
        if h0 != self.input_h or w0 != self.input_w:
            t = F.interpolate(t, size=(self.input_h, self.input_w), mode="bilinear", align_corners=False)
        if self.fp16:
            t = t.half()
        scale = (w0 / float(self.input_w), h0 / float(self.input_h))
        return t, scale

    def preprocess_cuda_chw_frame(self, frame_bgr_chw):
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
        if h0 != self.input_h or w0 != self.input_w:
            t = F.interpolate(t, size=(self.input_h, self.input_w), mode="bilinear", align_corners=False)
        if self.fp16:
            t = t.half()
        scale = (w0 / float(self.input_w), h0 / float(self.input_h))
        return t, scale

    def forward_tensor(self, t):
        if self.dynamic and tuple(t.shape) != tuple(self.bindings[self.input_name].shape):
            if self.is_trt10:
                self.context.set_input_shape(self.input_name, tuple(t.shape))
                self.bindings[self.input_name] = self.bindings[self.input_name]._replace(shape=tuple(t.shape))
                for name in self.output_names:
                    shape_new = tuple(self.context.get_tensor_shape(name))
                    self.bindings[name].data.resize_(shape_new)
                    self.bindings[name] = self.bindings[name]._replace(
                        shape=shape_new,
                        ptr=int(self.bindings[name].data.data_ptr()),
                    )
                    self.binding_addrs[name] = int(self.bindings[name].data.data_ptr())
            else:
                i = self.engine.get_binding_index(self.input_name)
                self.context.set_binding_shape(i, tuple(t.shape))
                self.bindings[self.input_name] = self.bindings[self.input_name]._replace(shape=tuple(t.shape))
                for name in self.output_names:
                    j = self.engine.get_binding_index(name)
                    shape_new = tuple(self.context.get_binding_shape(j))
                    self.bindings[name].data.resize_(shape_new)
                    self.bindings[name] = self.bindings[name]._replace(
                        shape=shape_new,
                        ptr=int(self.bindings[name].data.data_ptr()),
                    )
                    self.binding_addrs[name] = int(self.bindings[name].data.data_ptr())
        self.binding_addrs[self.input_name] = int(t.data_ptr())
        if self.use_async and self.stream is not None:
            self.stream.wait_stream(torch.cuda.current_stream(device=self.device))
            ok = True
            if self.is_trt10 and self._has_async_v3:
                self.context.set_tensor_address(self.input_name, self.binding_addrs[self.input_name])
                for name in self.output_names:
                    self.context.set_tensor_address(name, self.binding_addrs[name])
                ok = self.context.execute_async_v3(stream_handle=int(self.stream.cuda_stream))
            elif self._has_async_v2:
                ok = self.context.execute_async_v2(
                    list(self.binding_addrs.values()),
                    int(self.stream.cuda_stream),
                )
            else:
                ok = self.context.execute_v2(list(self.binding_addrs.values()))
            if ok is False:
                raise RuntimeError("TensorRT async execution failed.")
        else:
            ok = self.context.execute_v2(list(self.binding_addrs.values()))
            if ok is False:
                raise RuntimeError("TensorRT execution failed.")
        return [self.bindings[x].data for x in sorted(self.output_names)]

    def wait(self):
        if self.use_async and self.stream is not None:
            torch.cuda.current_stream(device=self.device).wait_stream(self.stream)


def _null_context():
    return contextlib.nullcontext()


class BallDetectorBackend:
    """Swappable detector interface for ball detection backends."""

    uses_raw_frames = False

    def detect(self, frame_bgr: np.ndarray) -> List[Tuple[list, float]]:
        raise NotImplementedError

    def detect_async_start(self, frame_bgr: np.ndarray):
        return frame_bgr

    def detect_async_finish(self, pending) -> List[Tuple[list, float]]:
        return self.detect(pending)

    def supports_cuda_frame(self) -> bool:
        return False


class GridTrackNetBallDetector(BallDetectorBackend):
    """Run the pretrained model at its native five-frame, 30-FPS cadence."""

    uses_raw_frames = True
    use_async = False
    async_slots = 1

    def __init__(self, weights_path: str, cfg: Config):
        if not HAS_TORCH or not torch.cuda.is_available():
            raise RuntimeError("GridTrackNet requires CUDA-enabled torch.")
        from .gridtracknet import HEIGHT, WIDTH, decode_predictions, frame_tensor, load_model

        self.cfg = cfg
        self.device = torch.device(f"cuda:{int(cfg.device)}")
        self._decode = decode_predictions
        self._frame_tensor = frame_tensor
        torch.backends.cudnn.benchmark = True
        self.model = load_model(Path(weights_path), self.device)
        self.precomputed: Optional[List[List[Tuple[list, float]]]] = None
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
        override = getattr(self.cfg, "gridtracknet_source_stride", None)
        if override:
            return max(1, int(override))
        if 57.0 <= float(fps) <= 62.0:
            return 2
        if 22.0 <= float(fps) <= 32.0:
            return 1
        raise RuntimeError(f"GridTrackNet expects a 30 or 60 FPS video, got {fps:.3f} FPS")

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
            background = bool(getattr(self.cfg, "gridtracknet_prepass_background", True))
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

    def _prepass(self, capture, source_stride: int, width: int, height: int) -> None:
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
        y_offset = float(self.cfg.gridtracknet_y_offset_px) * height / 1080.0
        radius = max(2.0, 0.003 * np.hypot(width, height))
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
        stream_scope = torch.cuda.stream(stream) if stream is not None else _null_context()
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
                            if len(batches) == 4:
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

    def detect(self, frame_bgr: np.ndarray) -> List[Tuple[list, float]]:
        if self.precomputed is None:
            raise RuntimeError("GridTrackNet.prepare_video() must run before detection.")
        index = self.precomputed_index
        self.precomputed_index += 1
        if index >= len(self.precomputed):
            return []
        self.wait_until_ready(index)
        return self.precomputed[index]

    def detect_async_start(self, frame_bgr: np.ndarray):
        return self.detect(frame_bgr)

    def detect_async_finish(self, pending) -> List[Tuple[list, float]]:
        return pending


class TensorRTRuntimeBallDetector(BallDetectorBackend):
    """Direct TensorRT runtime backend for ball detection."""

    def __init__(
        self,
        engine_path: str,
        cfg: Config,
        ball_cls_id: Optional[int],
        names: Optional[Dict[int, str]] = None,
    ):
        if not HAS_TORCH:
            raise RuntimeError("TensorRT runtime backend requires torch.")
        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT runtime backend requires CUDA.")
        try:
            import tensorrt as trt  # type: ignore
        except Exception as e:
            raise RuntimeError(f"TensorRT package not available: {e}")

        self.cfg = cfg
        self.names = names or {0: "ball"}
        self.ball_cls_id = ball_cls_id
        self.cls_filter = [ball_cls_id] if ball_cls_id is not None else None
        self.device = torch.device(f"cuda:{cfg.device}") if cfg.device not in ("auto", "cpu", "mps") else torch.device("cuda:0")

        self._trt = trt
        Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
        self._Binding = Binding

        logger = trt.Logger(trt.Logger.INFO)
        with open(engine_path, "rb") as f:
            raw = f.read()
        # Ultralytics .engine files may prepend a metadata JSON blob.
        try:
            meta_len = int.from_bytes(raw[:4], byteorder="little")
            json.loads(raw[4:4 + meta_len].decode("utf-8"))
            raw = raw[4 + meta_len:]
        except Exception:
            pass
        with trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(raw)
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed creating TensorRT execution context.")

        self.is_trt10 = not hasattr(self.engine, "num_bindings")
        self.bindings: OrderedDict[str, Any] = OrderedDict()
        self.binding_addrs: OrderedDict[str, int] = OrderedDict()
        self.binding_order: List[str] = []
        self.output_names: List[str] = []
        self.input_name: Optional[str] = None
        self.dynamic = False
        self.half = False
        self.input_h = 0
        self.input_w = 0
        self.use_async = bool(getattr(cfg, "trt_async_execute", True))
        self.async_slots = max(1, int(getattr(cfg, "trt_async_slots", 2))) if self.use_async else 1
        self._slot_cursor = 0
        self.stream = torch.cuda.Stream(device=self.device) if self.use_async else None
        self._has_async_v3 = hasattr(self.context, "execute_async_v3")
        self._has_async_v2 = hasattr(self.context, "execute_async_v2")
        if self.use_async and not (self._has_async_v3 or self._has_async_v2):
            self.use_async = False
            self.stream = None
            self.async_slots = 1

        num = range(self.engine.num_io_tensors) if self.is_trt10 else range(self.engine.num_bindings)
        for i in num:
            if self.is_trt10:
                name = self.engine.get_tensor_name(i)
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                shape = tuple(self.engine.get_tensor_shape(name))
                profile_shape = tuple(self.engine.get_tensor_profile_shape(name, 0)[2]) if is_input else None
            else:
                name = self.engine.get_binding_name(i)
                dtype = trt.nptype(self.engine.get_binding_dtype(i))
                is_input = self.engine.binding_is_input(i)
                shape = tuple(self.engine.get_binding_shape(i))
                profile_shape = tuple(self.engine.get_profile_shape(0, i)[1]) if is_input else None

            self.binding_order.append(name)
            if is_input:
                self.input_name = name
                if -1 in shape:
                    self.dynamic = True
                    if self.is_trt10:
                        self.context.set_input_shape(name, profile_shape)
                    else:
                        self.context.set_binding_shape(i, profile_shape)
                if dtype == np.float16:
                    self.half = True
            else:
                self.output_names.append(name)

            shape_now = tuple(self.context.get_tensor_shape(name)) if self.is_trt10 else tuple(self.context.get_binding_shape(i))
            tensor = torch.from_numpy(np.empty(shape_now, dtype=dtype)).to(self.device)
            self.bindings[name] = Binding(name, dtype, shape_now, tensor, int(tensor.data_ptr()))
            self.binding_addrs[name] = int(tensor.data_ptr())

        self.output_slot_tensors: List[Dict[str, torch.Tensor]] = []
        for s in range(self.async_slots):
            slot = {}
            for name in self.output_names:
                if s == 0:
                    slot[name] = self.bindings[name].data
                else:
                    slot[name] = torch.empty_like(self.bindings[name].data, device=self.device)
            self.output_slot_tensors.append(slot)

        if self.input_name is None:
            raise RuntimeError("TensorRT engine has no input binding.")
        if not self.output_names:
            raise RuntimeError("TensorRT engine has no output bindings.")
        in_shape = tuple(self.bindings[self.input_name].shape)
        if len(in_shape) != 4:
            raise RuntimeError(f"Unsupported TensorRT input shape: {in_shape}")
        self.input_h = int(in_shape[2])
        self.input_w = int(in_shape[3])

        dummy = torch.zeros(
            1,
            3,
            self.input_h,
            self.input_w,
            device=self.device,
            dtype=torch.float16 if self.half else torch.float32,
        )
        test_out, _ = self._forward_tensor(dummy, out_slot=0)
        test_pred = test_out[0] if isinstance(test_out, (list, tuple)) else test_out
        if hasattr(test_pred, "dim") and test_pred.dim() == 3 and test_pred.shape[-1] == 6:
            self.output_mode = "postprocessed"
        else:
            self.output_mode = "raw_nms"
        print(
            f"[detector] TensorRT runtime backend, {self.output_mode} "
            f"(half={self.half}, async={self.use_async}, slots={self.async_slots}, "
            f"device={self.device}, preprocess=gpu_resize_norm)"
        )

    @staticmethod
    def _rescale_box(det, scale):
        sx, sy = scale
        return [
            float(det[0]) * sx,
            float(det[1]) * sy,
            float(det[2]) * sx,
            float(det[3]) * sy,
        ]

    def _preprocess_frame(self, frame_bgr: np.ndarray):
        h0, w0 = frame_bgr.shape[:2]
        t = torch.from_numpy(frame_bgr).to(device=self.device, dtype=torch.float32)
        t = t.permute(2, 0, 1).unsqueeze(0).contiguous()
        t = t[:, [2, 1, 0], :, :] / 255.0
        if h0 != self.input_h or w0 != self.input_w:
            t = F.interpolate(t, size=(self.input_h, self.input_w), mode="bilinear", align_corners=False)
        if self.half:
            t = t.half()
        scale = (w0 / float(max(self.input_w, 1)), h0 / float(max(self.input_h, 1)))
        return t, scale

    def _preprocess_cuda_frame(self, frame_bgr_cuda):
        if not HAS_TORCH or not isinstance(frame_bgr_cuda, torch.Tensor):
            return None, None
        if frame_bgr_cuda.device != self.device:
            return None, None
        if frame_bgr_cuda.dim() != 3 or frame_bgr_cuda.shape[-1] != 3:
            return None, None

        h0, w0 = int(frame_bgr_cuda.shape[0]), int(frame_bgr_cuda.shape[1])
        if h0 <= 0 or w0 <= 0:
            return None, None

        t = frame_bgr_cuda
        if t.dtype == torch.uint8:
            t = t.to(dtype=torch.float32).mul_(1.0 / 255.0)
        else:
            if t.dtype != torch.float32:
                t = t.float()
            if t.max() > 1.5:
                t = t / 255.0
        t = t.permute(2, 0, 1).unsqueeze(0).contiguous()
        t = t[:, [2, 1, 0], :, :]
        if h0 != self.input_h or w0 != self.input_w:
            t = F.interpolate(t, size=(self.input_h, self.input_w), mode="bilinear", align_corners=False)

        if self.half:
            t = t.half()
        scale = (w0 / float(max(self.input_w, 1)), h0 / float(max(self.input_h, 1)))
        return t, scale

    def _forward_tensor(self, t, out_slot: int = 0):
        if self.dynamic and tuple(t.shape) != tuple(self.bindings[self.input_name].shape):
            if self.is_trt10:
                self.context.set_input_shape(self.input_name, tuple(t.shape))
                self.bindings[self.input_name] = self.bindings[self.input_name]._replace(shape=tuple(t.shape))
                for name in self.output_names:
                    shape_new = tuple(self.context.get_tensor_shape(name))
                    self.bindings[name] = self.bindings[name]._replace(shape=shape_new)
                    for s in range(self.async_slots):
                        cur = self.output_slot_tensors[s][name]
                        if tuple(cur.shape) != shape_new:
                            cur = torch.empty(shape_new, device=self.device, dtype=cur.dtype)
                        self.output_slot_tensors[s][name] = cur
                    self.bindings[name] = self.bindings[name]._replace(
                        data=self.output_slot_tensors[0][name],
                        ptr=int(self.output_slot_tensors[0][name].data_ptr()),
                    )
                    self.binding_addrs[name] = int(self.output_slot_tensors[0][name].data_ptr())
            else:
                i = self.engine.get_binding_index(self.input_name)
                self.context.set_binding_shape(i, tuple(t.shape))
                self.bindings[self.input_name] = self.bindings[self.input_name]._replace(shape=tuple(t.shape))
                for name in self.output_names:
                    j = self.engine.get_binding_index(name)
                    shape_new = tuple(self.context.get_binding_shape(j))
                    self.bindings[name] = self.bindings[name]._replace(shape=shape_new)
                    for s in range(self.async_slots):
                        cur = self.output_slot_tensors[s][name]
                        if tuple(cur.shape) != shape_new:
                            cur = torch.empty(shape_new, device=self.device, dtype=cur.dtype)
                        self.output_slot_tensors[s][name] = cur
                    self.bindings[name] = self.bindings[name]._replace(
                        data=self.output_slot_tensors[0][name],
                        ptr=int(self.output_slot_tensors[0][name].data_ptr()),
                    )
                    self.binding_addrs[name] = int(self.output_slot_tensors[0][name].data_ptr())

        s = tuple(self.bindings[self.input_name].shape)
        if tuple(t.shape) != s:
            raise RuntimeError(f"TensorRT input shape mismatch: got {tuple(t.shape)}, expected {s}")

        out_slot = int(out_slot) % max(self.async_slots, 1)
        slot_tensors = self.output_slot_tensors[out_slot]
        addr_map = dict(self.binding_addrs)
        addr_map[self.input_name] = int(t.data_ptr())
        for name in self.output_names:
            addr_map[name] = int(slot_tensors[name].data_ptr())

        pending_event = None
        if self.use_async and self.stream is not None:
            self.stream.wait_stream(torch.cuda.current_stream(device=self.device))
            ok = True
            if self.is_trt10 and self._has_async_v3:
                self.context.set_tensor_address(self.input_name, addr_map[self.input_name])
                for name in self.output_names:
                    self.context.set_tensor_address(name, addr_map[name])
                ok = self.context.execute_async_v3(stream_handle=int(self.stream.cuda_stream))
            elif self._has_async_v2:
                ptrs = [int(addr_map[name]) for name in self.binding_order]
                ok = self.context.execute_async_v2(
                    ptrs,
                    int(self.stream.cuda_stream),
                )
            else:
                ptrs = [int(addr_map[name]) for name in self.binding_order]
                ok = self.context.execute_v2(ptrs)
            if ok is False:
                raise RuntimeError("TensorRT async execution failed.")
            pending_event = torch.cuda.Event(blocking=False)
            pending_event.record(self.stream)
        else:
            if self.is_trt10 and hasattr(self.context, "set_tensor_address"):
                self.context.set_tensor_address(self.input_name, addr_map[self.input_name])
                for name in self.output_names:
                    self.context.set_tensor_address(name, addr_map[name])
                ok = self.context.execute_v2([int(addr_map[name]) for name in self.binding_order])
            else:
                ptrs = [int(addr_map[name]) for name in self.binding_order]
                ok = self.context.execute_v2(ptrs)
            if ok is False:
                raise RuntimeError("TensorRT execution failed.")
        return [slot_tensors[x] for x in sorted(self.output_names)], pending_event

    def _decode_preds(self, preds, scale, pending_event=None):
        if pending_event is not None:
            torch.cuda.current_stream(device=self.device).wait_event(pending_event)
        elif self.use_async and self.stream is not None:
            torch.cuda.current_stream(device=self.device).wait_stream(self.stream)
        if isinstance(preds, (list, tuple)):
            preds = preds[0]

        dets = []
        if self.output_mode == "postprocessed":
            if preds.dim() == 3:
                preds = preds[0]
            rows = preds.detach().float().cpu().numpy()
            if rows.size == 0 or rows.shape[1] < 6:
                return dets
            for det in rows:
                conf = float(det[4])
                if conf < self.cfg.conf:
                    continue
                cls_id = int(det[5])
                if self.cls_filter is not None and cls_id not in self.cls_filter:
                    continue
                dets.append((self._rescale_box(det, scale), conf))
            return dets

        p = preds
        if p.dim() == 3:
            p = p[0]
            if p.shape[0] <= p.shape[1]:
                p = p.transpose(0, 1)
        else:
            return dets
        p = p.detach().float().cpu().numpy()
        if p.size == 0 or p.shape[1] <= 4:
            return dets
        boxes = _xywh_to_xyxy_np(p[:, :4].astype(np.float32))
        cls_scores = p[:, 4:].astype(np.float32)
        cls_ids = np.argmax(cls_scores, axis=1) if cls_scores.shape[1] > 1 else np.zeros((cls_scores.shape[0],), dtype=np.int32)
        scores = cls_scores[np.arange(cls_scores.shape[0]), cls_ids] if cls_scores.shape[1] > 1 else cls_scores[:, 0]
        keep = scores >= float(self.cfg.conf)
        boxes = boxes[keep]
        scores = scores[keep]
        cls_ids = cls_ids[keep]
        if boxes.shape[0] == 0:
            return dets
        keep_idx = _nms_xyxy_np(boxes, scores, 0.45)
        boxes = boxes[keep_idx]
        scores = scores[keep_idx]
        cls_ids = cls_ids[keep_idx]
        for i in range(boxes.shape[0]):
            cls_id = int(cls_ids[i])
            if self.cls_filter is not None and cls_id not in self.cls_filter:
                continue
            dets.append((self._rescale_box(boxes[i], scale), float(scores[i])))
        return dets

    def supports_cuda_frame(self) -> bool:
        return bool(HAS_TORCH and self.device.type == "cuda")

    def detect(self, frame_bgr: np.ndarray) -> List[Tuple[list, float]]:
        pending = self.detect_async_start(frame_bgr)
        return self.detect_async_finish(pending)

    def detect_async_start(self, frame_bgr: np.ndarray):
        out_slot = 0
        if self.use_async:
            out_slot = self._slot_cursor
            self._slot_cursor = (self._slot_cursor + 1) % max(self.async_slots, 1)
        if HAS_TORCH and isinstance(frame_bgr, torch.Tensor):
            if self.supports_cuda_frame():
                t, scale = self._preprocess_cuda_frame(frame_bgr)
                if t is not None:
                    preds, ev = self._forward_tensor(t, out_slot=out_slot)
                    return {"preds": preds, "scale": scale, "event": ev, "input_tensor": t}
            frame_bgr = frame_bgr.detach().cpu().numpy()

        t, scale = self._preprocess_frame(frame_bgr)
        preds, ev = self._forward_tensor(t, out_slot=out_slot)
        return {"preds": preds, "scale": scale, "event": ev, "input_tensor": t}

    def detect_async_finish(self, pending) -> List[Tuple[list, float]]:
        if isinstance(pending, dict):
            return self._decode_preds(
                pending.get("preds"),
                pending.get("scale"),
                pending_event=pending.get("event"),
            )
        preds, scale = pending
        return self._decode_preds(preds, scale)

class PlayerDetector:
    """Runs BoTSORT (StrongSORT successor) player tracking with ReID."""

    def __init__(self, cfg: Config, court_keypoints=None):
        self.cfg = cfg
        self.court_kps = court_keypoints
        self.cached_boxes = []
        self._frame_diag = 1.0
        self.session: Optional[_TensorRTRuntimeSession] = None
        
        # Detection cadence is controlled by the CLI/config, while ByteTrack
        # still receives an update every frame and coasts on empty detections.
        self._current_interval = max(1, int(getattr(cfg, "player_detect_interval", 1)))
        
        # BoTSORT maintains its own robust internal dictionary of {ID: BoundingBox}
        self.slots = {} 

        ep = _resolve_engine_path(cfg.player_model_path)
        if ep is None:
            print(f"[player] Engine not found at {cfg.player_model_path}, skipping")
            return
            
        try:
            self.session = _TensorRTRuntimeSession(
                str(ep),
                device=torch.device(f"cuda:{int(cfg.device)}"),
                async_execute=bool(getattr(cfg, "trt_async_execute", True)),
            )
            print(f"[player] TensorRT runtime initialized: {ep}")
        except Exception as e:
            self.session = None
            print(f"[player] TensorRT runtime init failed ({e}), skipping")
            
        if ByteTrack is None:
            self.tracker = None
        else:
            self.tracker = ByteTrack(
                min_conf=0.1,
                track_thresh=0.3,
                track_buffer=60,
                match_thresh=0.8,
            )
            print("[player] ByteTrack tracking initialized")

    @staticmethod
    def _center(b):
        return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)

    def set_court_keypoints(self, court_keypoints):
        self.court_kps = court_keypoints

    def _decode_player_boxes(self, pred_t, frame_w: int, frame_h: int) -> np.ndarray:
        pred = pred_t.detach().float()
        if pred.dim() == 3 and pred.shape[-1] == 6:
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
        class_ids = np.zeros_like(scores) # Usually 0 for 'person'
        dets = np.column_stack((boxes_xyxy, scores, class_ids))
        return dets

    def detect_async_start(self, frame, frame_idx, frame_gpu_t=None):
        if self.session is None or self.tracker is None:
            return {"disabled": True}
            
        h, w = frame.shape[:2]
        self._frame_diag = (w**2 + h**2)**0.5
        
        pending = {
            "disabled": False,
            "frame": frame,
            "frame_idx": int(frame_idx),
            "h": int(h),
            "w": int(w),
            "court_kps": self.court_kps,
            "outputs": None,
        }

        # We only run the heavy YOLO TensorRT model every N frames.
        # The tracker itself is still finished every frame with empty detections.
        if frame_idx % self._current_interval != 0:
            return pending

        t = None
        if frame_gpu_t is not None and HAS_TORCH and isinstance(frame_gpu_t, torch.Tensor):
            t, _ = self.session.preprocess_cuda_chw_frame(frame_gpu_t)
        if t is None:
            t, _ = self.session.preprocess_frame(frame)
        pending["outputs"] = self.session.forward_tensor(t)
        pending["input_tensor"] = t
        return pending

    def detect_async_finish(self, pending):
        if not pending or pending.get("disabled", False):
            return []

        frame = pending["frame"]
        w = int(pending["w"])
        h = int(pending["h"])
        court_kps = pending.get("court_kps")

        if pending.get("outputs") is None:
            # Pass empty detections to tell the tracker to coast the boxes forward.
            dets = np.empty((0, 6))
        else:
            self.session.wait()

            # Raw YOLO detections [x1, y1, x2, y2, conf, cls]
            dets = self._decode_player_boxes(pending["outputs"][0], w, h)

            # Filter raw YOLO detections by court keypoint distance if necessary
            # before passing to the tracker so it doesn't learn noise.
            if court_kps and len(dets) > self.cfg.num_players:
                scored_dets = []
                for row in dets:
                    bx1, by1, bx2, by2 = row[:4]
                    bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                    dist = min(((bcx - court_kps[ki])**2 + (bcy - court_kps[ki+1])**2)**0.5
                               for ki in range(0, len(court_kps), 2))
                    scored_dets.append((dist, row))
                scored_dets.sort(key=lambda x: x[0])
                dets = np.array([r for _, r in scored_dets[:self.cfg.num_players]])

            if len(dets) == 0:
                dets = np.empty((0, 6))
            
        # Update tracker with actual image pixels
        # tracked_objects format: [x1, y1, x2, y2, track_id, conf, cls, ind]
        tracked_objects = self.tracker.update(dets, frame)
        
        boxes = []
        self.slots.clear()
        
        if len(tracked_objects) > 0:
            for track in tracked_objects:
                x1, y1, x2, y2, track_id, conf, cls, _ = track
                box = [float(x1), float(y1), float(x2), float(y2)]
                boxes.append(box)
                self.slots[int(track_id)] = box
                
        self.cached_boxes = boxes
        return boxes

    def detect(self, frame, frame_idx):
        return self.detect_async_finish(self.detect_async_start(frame, frame_idx))

    def get_player_dict(self):
        return dict(self.slots)

class CourtDetector:
    """Runs court keypoint TensorRT model every N frames and caches the last good result."""
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

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.keypoints = None
        self.last_run_frame = -10**9
        self.session: Optional[_TensorRTRuntimeSession] = None
        ep = _resolve_engine_path(cfg.court_model_path)
        if ep is None:
            print(f"[court] Engine not found at {cfg.court_model_path}, skipping")
            return
        try:
            self.session = _TensorRTRuntimeSession(
                str(ep),
                device=torch.device(f"cuda:{int(cfg.device)}"),
                async_execute=bool(getattr(cfg, "trt_async_execute", True)),
            )
            print(f"[court] TensorRT runtime initialized: {ep}")
            remap = "enabled" if cfg.court_remap_semantic_14 else "disabled (raw YOLO order)"
            print(f"[court] 14-point semantic remap: {remap}")
        except Exception as e:
            self.session = None
            print(f"[court] TensorRT runtime init failed ({e}), skipping")

    @classmethod
    def _remap_kpt_indices(cls, xy):
        pts = np.asarray(xy, dtype=np.float32)
        if pts.shape[0] != len(cls._MODEL_TO_SEMANTIC_14):
            return pts
        out = np.zeros_like(pts)
        for model_i, semantic_i in enumerate(cls._MODEL_TO_SEMANTIC_14):
            out[semantic_i] = pts[model_i]
        return out

    def _print_raw_result(self, rows: np.ndarray, frame_idx: int):
        best_idx = int(np.argmax(rows[:, 4])) if rows.size else -1
        print(f"[court][raw] frame={frame_idx} rows={rows.shape[0]} best={best_idx}")
        if best_idx >= 0:
            r = rows[best_idx]
            print(json.dumps({
                "x1": float(r[0]), "y1": float(r[1]), "x2": float(r[2]), "y2": float(r[3]),
                "conf": float(r[4]), "cls": float(r[5]),
            }))

    def detect(self, frame, frame_idx=None):
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
            if self.cfg.print_court_raw:
                msg = "miss, using cached court" if self.keypoints else "no court detected"
                print(f"[court] frame={frame_idx}: {msg}")
            return self.keypoints

        row = rows[best_i]
        if row.shape[0] < 9:
            return self.keypoints
        kpt_raw = row[6:]
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
        if len(flat) >= 8:
            self.keypoints = flat
            return self.keypoints

        if self.cfg.print_court_raw:
            msg = "miss, using cached court" if self.keypoints else "no court detected"
            print(f"[court] frame={frame_idx}: {msg}")
        return self.keypoints

    def draw(self, frame):
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

        line_pairs = []
        if n == 14:
            line_pairs = self._LINE_PAIRS_14
        elif n >= 4:
            line_pairs = [(j, j+1) for j in range(n-1)]

        for a, b in line_pairs:
            if a < n and b < n:
                ax, ay = int(kps[a*2]), int(kps[a*2+1])
                bx, by = int(kps[b*2]), int(kps[b*2+1])
                if (ax > 0 or ay > 0) and (bx > 0 or by > 0):
                    cv2.line(frame, (ax, ay), (bx, by), (0, 255, 0), 2)
        return frame
