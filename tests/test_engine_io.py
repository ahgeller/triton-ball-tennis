"""TensorRT engine files and the runtime session, exercised with fake engines on CPU."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tennis_tracker.engines import engine_session
from tennis_tracker.config import Config
from tennis_tracker.engines.ball_detectors import TensorRTBallPending, TensorRTRuntimeBallDetector, decode_ball_predictions
from tennis_tracker.engines.engine_boxes import engine_output_mode
from tennis_tracker.engines.engine_files import (
    EngineIO,
    engine_class_names,
    find_ball_class_id_from_names,
    iter_engine_io,
    read_engine_class_names,
    read_engine_header,
    split_engine_header,
)
from tennis_tracker.engines.engine_session import TensorRTRuntimeSession

PAYLOAD = b"\x00TRT engine bytes\xff\xfe"


def _with_header(metadata: dict, payload: bytes = PAYLOAD) -> bytes:
    body = json.dumps(metadata).encode("utf-8")
    return len(body).to_bytes(4, "little") + body + payload


# --- header -----------------------------------------------------------------

def test_split_engine_header_separates_metadata_from_engine():
    metadata, raw = split_engine_header(_with_header({"names": ["ball", "racket"]}))
    assert metadata == {"names": ["ball", "racket"]}
    assert raw == PAYLOAD


def test_split_engine_header_leaves_plain_engines_alone():
    plain = b"\x10\x00\x00\x00" + b"\x7f" * 40  # length says 16 but the bytes are not JSON
    assert split_engine_header(plain) == (None, plain)
    assert split_engine_header(b"") == (None, b"")
    too_long = b"\xff\xff\xff\x7f" + PAYLOAD  # length past the end of the file
    assert split_engine_header(too_long) == (None, too_long)


def test_split_engine_header_requires_a_json_object():
    body = b"[1, 2]"
    raw = len(body).to_bytes(4, "little") + body + PAYLOAD
    assert split_engine_header(raw) == (None, raw)


def test_read_engine_header_and_class_names(tmp_path):
    engine = tmp_path / "ball.engine"
    engine.write_bytes(_with_header({"names": {"0": "ball", "1": "racket"}}))
    metadata, raw = read_engine_header(engine)
    assert raw == PAYLOAD
    assert engine_class_names(metadata) == {0: "ball", 1: "racket"}
    assert read_engine_class_names(engine) == {0: "ball", 1: "racket"}
    assert read_engine_class_names(tmp_path / "missing.engine") == {0: "ball"}
    assert engine_class_names(None) == {0: "ball"}
    with pytest.raises(FileNotFoundError):
        read_engine_header(tmp_path / "missing.engine")


def test_find_ball_class_id_from_names():
    assert find_ball_class_id_from_names({0: "ball"}) == (0, "ball")
    assert find_ball_class_id_from_names(["person", "sports ball"]) == (1, "sports ball")
    assert find_ball_class_id_from_names({0: "person", 3: "Tennis_Ball"}, override="tennis_ball") == (3, "Tennis_Ball")
    assert find_ball_class_id_from_names({0: "person", 1: "racket"}) == (None, None)
    assert find_ball_class_id_from_names(None) == (0, "ball")


# --- fake TensorRT ----------------------------------------------------------

class FakeContext10:
    """TensorRT 10 execution context: tensor API, synchronous execute only."""

    def __init__(self, engine):
        self.engine = engine
        self.shapes = {t.name: t.shape for t in engine.tensors}
        self.addresses = {}
        self.executions = []

    def get_tensor_shape(self, name):
        return self.shapes[name]

    def set_input_shape(self, name, shape):
        self.shapes[name] = tuple(shape)
        batch = shape[0]
        for t in self.engine.tensors:
            if not t.is_input:
                self.shapes[t.name] = (batch,) + tuple(t.shape[1:])

    def set_tensor_address(self, name, ptr):
        self.addresses[name] = ptr

    def execute_v2(self, ptrs):
        self.executions.append(list(ptrs))
        return True


class FakeEngine10:
    def __init__(self, tensors):
        self.tensors = tensors
        self.num_io_tensors = len(tensors)

    def _tensor(self, name):
        return next(t for t in self.tensors if t.name == name)

    def get_tensor_name(self, index):
        return self.tensors[index].name

    def get_tensor_dtype(self, name):
        return self._tensor(name).dtype

    def get_tensor_mode(self, name):
        return "INPUT" if self._tensor(name).is_input else "OUTPUT"

    def get_tensor_shape(self, name):
        return self._tensor(name).shape

    def get_tensor_profile_shape(self, name, profile_index):
        """(min, opt, max) of the optimisation profile, as TensorRT returns them."""
        t = self._tensor(name)
        return [t.profile_min, t.profile_opt, t.profile_max]

    def create_execution_context(self):
        return FakeContext10(self)


class FakeEngine8:
    """A TensorRT 8 engine exposes the binding API and no I/O tensor table."""

    def __init__(self, tensors):
        self.tensors = tensors
        self.num_bindings = len(tensors)


def _tensor(name, dtype, is_input, shape, profile_max=None):
    """A fake I/O tensor; a dynamic input's profile is (1, ..., max) with ``opt`` between."""
    profile_min = profile_opt = None
    if profile_max is not None:
        profile_min = (1,) + tuple(profile_max[1:])
        profile_opt = (max(1, profile_max[0] // 2),) + tuple(profile_max[1:])
    return SimpleNamespace(name=name, dtype=dtype, is_input=is_input, shape=tuple(shape),
                           profile_min=profile_min, profile_opt=profile_opt, profile_max=profile_max)


def _fake_trt(engine):
    class Logger:
        INFO = 1
        ERROR = 2

        def __init__(self, level=None):
            self.level = level

    class Runtime:
        def __init__(self, logger):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def deserialize_cuda_engine(self, raw):
            engine.raw = raw
            return engine

    return SimpleNamespace(
        Logger=Logger,
        Runtime=Runtime,
        TensorIOMode=SimpleNamespace(INPUT="INPUT", OUTPUT="OUTPUT"),
        nptype=lambda dtype: {"FLOAT": np.float32, "HALF": np.float16}[dtype],
    )


def test_iter_engine_io_reads_the_tensor_api_and_the_profile_max():
    tensors = [
        _tensor("images", "HALF", True, (-1, 3, 8, 8), profile_max=(4, 3, 8, 8)),
        _tensor("output0", "FLOAT", False, (-1, 4, 6)),
    ]
    trt = _fake_trt(None)
    engine = FakeEngine10(tensors)
    assert engine.get_tensor_profile_shape("images", 0) == [(1, 3, 8, 8), (2, 3, 8, 8), (4, 3, 8, 8)]
    table = list(iter_engine_io(engine, trt))
    assert table == [
        EngineIO(0, "images", "HALF", True, (-1, 3, 8, 8), (4, 3, 8, 8)),   # max, not min or opt
        EngineIO(1, "output0", "FLOAT", False, (-1, 4, 6), None),
    ]
    assert table[0].dynamic and table[1].dynamic
    static = list(iter_engine_io(FakeEngine10([_tensor("images", "FLOAT", True, (1, 3, 8, 8))]), trt))
    assert static == [EngineIO(0, "images", "FLOAT", True, (1, 3, 8, 8), None)] and not static[0].dynamic


def test_iter_engine_io_rejects_a_tensorrt8_engine():
    with pytest.raises(RuntimeError, match="TensorRT 8 binding API.*TensorRT 10"):
        list(iter_engine_io(FakeEngine8([_tensor("images", "FLOAT", True, (1, 3, 8, 8))]), _fake_trt(None)))


@pytest.fixture
def cuda_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)


def _session_for(monkeypatch, engine, tmp_path, **kwargs):
    engine_file = tmp_path / "fake.engine"
    engine_file.write_bytes(_with_header({"names": ["ball"]}))
    monkeypatch.setattr(engine_session, "import_tensorrt", lambda: _fake_trt(engine))
    return TensorRTRuntimeSession(str(engine_file), device=torch.device("cpu"), async_execute=False, **kwargs)


def test_session_binds_tensors_and_runs_with_slot_buffers(monkeypatch, tmp_path, cuda_available):
    engine = FakeEngine10([
        _tensor("output0", "FLOAT", False, (1, 4, 6)),
        _tensor("images", "HALF", True, (1, 3, 8, 8)),
    ])
    session = _session_for(monkeypatch, engine, tmp_path, output_slots=3)

    assert engine.raw == PAYLOAD  # the metadata header never reaches TensorRT
    assert session.input_name == "images" and session.output_names == ["output0"]
    assert (session.input_h, session.input_w) == (8, 8)
    assert session.fp16 and not session.dynamic and not session.use_async
    assert session.binding_order == ["output0", "images"]
    assert session.bindings["images"].data.dtype == torch.float16
    assert session.bindings["output0"].data.shape == (1, 4, 6)
    assert session.output_slot_count == 1  # extra slots need asynchronous execution

    frame = np.zeros((16, 32, 3), dtype=np.uint8)
    t, scale = session.preprocess_frame(frame)
    assert t.shape == (1, 3, 8, 8) and t.dtype == torch.float16
    assert scale == (4.0, 2.0)

    outputs = session.forward_tensor(t)
    assert outputs[0] is session.output_slots[0]["output0"]
    assert session.context.executions == [[outputs[0].data_ptr(), t.data_ptr()]]
    assert session.context.addresses == {"images": t.data_ptr(), "output0": outputs[0].data_ptr()}
    with pytest.raises(RuntimeError):
        session.forward_tensor(torch.zeros((1, 3, 4, 4), dtype=torch.float16))


def test_session_rejects_a_tensorrt8_engine(monkeypatch, tmp_path, cuda_available):
    engine = FakeEngine8([_tensor("images", "FLOAT", True, (1, 3, 8, 8))])
    engine.create_execution_context = lambda: SimpleNamespace()
    with pytest.raises(RuntimeError, match="TensorRT 10"):
        _session_for(monkeypatch, engine, tmp_path)


def test_session_handles_dynamic_shapes(monkeypatch, tmp_path, cuda_available):
    engine = FakeEngine10([
        _tensor("images", "FLOAT", True, (-1, 3, 8, 8), profile_max=(4, 3, 8, 8)),
        _tensor("output0", "FLOAT", False, (-1, 4, 6)),
    ])
    session = _session_for(monkeypatch, engine, tmp_path)
    assert session.dynamic
    assert session.bindings["images"].shape == (4, 3, 8, 8)  # the profile's max shape, not its opt (2)
    assert session.bindings["output0"].data.shape == (4, 4, 6)

    outputs = session.forward_tensor(torch.zeros((1, 3, 8, 8)))
    assert outputs[0].shape == (1, 4, 6)
    assert session.bindings["output0"].data is outputs[0]
    assert session.context.addresses == {"images": session.context.executions[0][0], "output0": outputs[0].data_ptr()}


def test_session_preprocesses_cuda_layouts(monkeypatch, tmp_path, cuda_available):
    engine = FakeEngine10([
        _tensor("images", "FLOAT", True, (1, 3, 8, 8)),
        _tensor("output0", "FLOAT", False, (1, 4, 6)),
    ])
    session = _session_for(monkeypatch, engine, tmp_path)
    hwc = torch.zeros((16, 16, 3), dtype=torch.uint8)
    hwc[..., 0] = 255  # blue
    t, scale = session.preprocess_cuda_hwc_frame(hwc)
    assert t.shape == (1, 3, 8, 8) and scale == (2.0, 2.0)
    assert t[0, 2].mean().item() == pytest.approx(1.0) and t[0, 0].mean().item() == 0.0  # RGB order
    assert session.preprocess_cuda_hwc_frame(torch.zeros((3, 16, 16))) == (None, None)
    chw = torch.zeros((3, 8, 8))
    chw[0] = 1.0  # blue in CHW BGR
    t, _ = session.preprocess_cuda_chw_frame(chw)
    assert t[0, 2].mean().item() == pytest.approx(1.0)


# --- ball engine backend ----------------------------------------------------

def test_ball_output_mode_and_decoding():
    rows = torch.tensor([[[10.0, 20.0, 30.0, 40.0, 0.9, 0.0], [1.0, 1.0, 2.0, 2.0, 0.2, 0.0], [5.0, 5.0, 6.0, 6.0, 0.8, 1.0]]])
    assert engine_output_mode(rows) == "postprocessed"
    assert decode_ball_predictions(rows, (2.0, 0.5), 0.5, "postprocessed", [0]) == [([20.0, 10.0, 60.0, 20.0], pytest.approx(0.9))]
    assert len(decode_ball_predictions(rows, (1.0, 1.0), 0.5, "postprocessed")) == 2

    raw = torch.zeros((1, 5, 8))  # [cx, cy, w, h, score] x 8 anchors, channel-major like a YOLO head
    raw[0, :, 0] = torch.tensor([10.0, 10.0, 4.0, 4.0, 0.9])
    raw[0, :, 1] = torch.tensor([10.0, 10.0, 4.0, 4.0, 0.7])  # suppressed duplicate
    raw[0, :, 2] = torch.tensor([50.0, 50.0, 2.0, 2.0, 0.3])  # below threshold
    assert engine_output_mode(raw) == "raw_nms"
    assert decode_ball_predictions(raw, (1.0, 1.0), 0.5, "raw_nms") == [([8.0, 8.0, 12.0, 12.0], pytest.approx(0.9))]
    with pytest.raises(ValueError, match="output mode"):
        decode_ball_predictions(raw, (1.0, 1.0), 0.5, "postprocess")


def test_ball_engine_backend_runs_through_the_shared_session(monkeypatch, tmp_path, cuda_available):
    engine = FakeEngine10([
        _tensor("images", "FLOAT", True, (1, 3, 8, 8)),
        _tensor("output0", "FLOAT", False, (1, 2, 6)),
    ])
    engine_file = tmp_path / "ball.engine"
    engine_file.write_bytes(_with_header({"names": ["ball"]}))
    monkeypatch.setattr(engine_session, "import_tensorrt", lambda: _fake_trt(engine))
    cfg = Config(conf=0.5, trt_async_execute=False)
    detector = TensorRTRuntimeBallDetector(str(engine_file), cfg, ball_cls_id=0, device=torch.device("cpu"))
    assert detector.output_mode == "postprocessed"
    assert detector.async_slots == 1 and not detector.use_async

    # The fake engine never writes outputs, so the buffer holds the "result".
    detector.session.output_slots[0]["output0"].copy_(
        torch.tensor([[[1.0, 2.0, 3.0, 4.0, 0.9, 0.0], [0.0, 0.0, 1.0, 1.0, 0.1, 0.0]]])
    )
    pending = detector.detect_async_start(np.zeros((16, 16, 3), dtype=np.uint8), frame_index=7)
    assert isinstance(pending, TensorRTBallPending) and pending.event is None
    assert detector.detect_async_finish(pending) == [([2.0, 4.0, 6.0, 8.0], pytest.approx(0.9))]
    assert detector.detect(torch.zeros((16, 16, 3), dtype=torch.uint8), 8) == [([2.0, 4.0, 6.0, 8.0], pytest.approx(0.9))]


def test_session_cuda_graph_needs_the_asynchronous_stream(monkeypatch, tmp_path, cuda_available):
    """``cuda_graph`` is a property of the asynchronous path: a synchronous session ignores it and
    launches the engine directly, as before."""
    engine = FakeEngine10([
        _tensor("images", "HALF", True, (1, 3, 8, 8)),
        _tensor("output0", "FLOAT", False, (1, 4, 6)),
    ])
    session = _session_for(monkeypatch, engine, tmp_path, cuda_graph=True)
    assert not session.use_cuda_graph
    t = torch.zeros((1, 3, 8, 8), dtype=torch.float16)
    outputs = session.forward_tensor(t)
    assert session.context.executions == [[t.data_ptr(), outputs[0].data_ptr()]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA and the shipped player engine")
def test_player_engine_graph_replay_matches_a_direct_launch():
    """On the real player engine, the launch replayed from the CUDA graph writes the same output as
    the direct enqueue, for several random inputs in a row."""
    from tennis_tracker.paths import PLAYER_ENGINE
    if not PLAYER_ENGINE.is_file():
        pytest.skip("player.engine is not present")
    device = torch.device("cuda:0")
    direct = TensorRTRuntimeSession(str(PLAYER_ENGINE), device=device, async_execute=True)
    graphed = TensorRTRuntimeSession(str(PLAYER_ENGINE), device=device, async_execute=True, cuda_graph=True)
    assert graphed.use_cuda_graph
    shape = direct.bindings[direct.input_name].shape
    for seed in range(3):
        frame = torch.rand(shape, device=device, generator=torch.Generator(device=device).manual_seed(seed)).half()
        expected = direct.forward_tensor(frame)[0]
        direct.wait(); torch.cuda.synchronize()
        expected = expected.clone()
        got = graphed.forward_tensor(frame)[0]
        graphed.wait(); torch.cuda.synchronize()
        assert torch.equal(got, expected)
