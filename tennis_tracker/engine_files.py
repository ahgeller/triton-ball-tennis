"""TensorRT engine files on disk: locating them, the Ultralytics metadata
header, class names and the I/O tensor table of a deserialized engine.

Nothing here touches CUDA; ``engine_session`` owns the runtime.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

# Ultralytics .engine files start with a little-endian uint32 length followed
# by that many bytes of JSON metadata, then the serialized TensorRT engine.
_ENGINE_HEADER_LEN_BYTES = 4
_DEFAULT_ENGINE_NAMES: Dict[int, str] = {0: "ball"}
_BALL_CLASS_ALIASES = ("ball", "sports ball", "tennis ball", "sports_ball", "tennis_ball")


def find_ball_class_id_from_names(names: Any, override: Optional[str] = None) -> Tuple[Optional[int], Optional[str]]:
    """Pick the ball class of a detection engine from its class-name table.

    A single-class table is taken as is; otherwise ``override`` wins, then the
    usual ball aliases.  Returns ``(None, None)`` when nothing matches, which
    the caller treats as "use every class".
    """
    if isinstance(names, list):
        names = dict(enumerate(names))
    if not isinstance(names, dict) or not names:
        names = dict(_DEFAULT_ENGINE_NAMES)
    names = {int(k): str(v) for k, v in names.items()}
    if len(names) == 1:
        cid = next(iter(names))
        return cid, names[cid]
    if override:
        for cid, name in names.items():
            if name.lower() == override.lower():
                return cid, name
        print(f"[WARN] class_name='{override}' not in {names}")
    for candidate in _BALL_CLASS_ALIASES:
        for cid, name in names.items():
            if name.lower() == candidate:
                print(f"[info] Auto-detected ball class: {cid} -> '{name}'")
                return cid, name
    print(f"[WARN] Could not find ball class in {names}. All classes used!")
    return None, None


def resolve_engine_path(model_path: Optional[str]) -> Optional[Path]:
    """The ``.engine`` file for a model path (a ``.pt``/``.onnx`` sibling or the
    path itself); None when unset or missing on disk."""
    if not model_path:
        return None
    base = Path(model_path)
    if base.suffix.lower() == ".engine":
        return base if base.exists() else None
    engine = base.with_suffix(".engine")
    return engine if engine.exists() else None


def split_engine_header(raw: bytes) -> Tuple[Optional[Dict[str, Any]], bytes]:
    """Separate the Ultralytics metadata header from a serialized engine.

    Returns ``(metadata, engine_bytes)``.  A file without the header (a plain
    TensorRT plan) comes back as ``(None, raw)``; only a header that does not
    parse as JSON is treated that way, any other error propagates.
    """
    if len(raw) < _ENGINE_HEADER_LEN_BYTES:
        return None, raw
    meta_len = int.from_bytes(raw[:_ENGINE_HEADER_LEN_BYTES], byteorder="little")
    end = _ENGINE_HEADER_LEN_BYTES + meta_len
    if meta_len <= 0 or end > len(raw):
        return None, raw
    try:
        metadata = json.loads(raw[_ENGINE_HEADER_LEN_BYTES:end].decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None, raw
    if not isinstance(metadata, dict):
        return None, raw
    return metadata, raw[end:]


def read_engine_header(path: Path) -> Tuple[Optional[Dict[str, Any]], bytes]:
    """``split_engine_header`` of the file at ``path`` (raises if it is missing)."""
    return split_engine_header(Path(path).read_bytes())


def engine_class_names(metadata: Optional[Dict[str, Any]]) -> Dict[int, str]:
    """The ``names`` table of an engine header as ``{class_id: name}``; the
    single-class ball table when the header carries none."""
    names = (metadata or {}).get("names")
    if isinstance(names, list):
        return {i: str(n) for i, n in enumerate(names)}
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return dict(_DEFAULT_ENGINE_NAMES)


def read_engine_class_names(engine_path: Optional[Path]) -> Dict[int, str]:
    """Class names of the engine at ``engine_path``; the ball default when the
    path is unset, the file is missing or it has no metadata header."""
    if engine_path is None or not Path(engine_path).exists():
        return dict(_DEFAULT_ENGINE_NAMES)
    metadata, _ = read_engine_header(Path(engine_path))
    return engine_class_names(metadata)


@dataclass(frozen=True)
class EngineIO:
    """One input or output tensor of a deserialized TensorRT engine.

    ``index`` is the TensorRT 8 binding index (also the I/O tensor index on
    TensorRT 10); ``dtype`` is the raw ``trt.DataType``; ``profile_shape`` is
    the optimisation profile's max shape, filled only for dynamic inputs.
    """

    index: int
    name: str
    dtype: Any
    is_input: bool
    shape: Tuple[int, ...]
    profile_shape: Optional[Tuple[int, ...]]

    @property
    def dynamic(self) -> bool:
        return -1 in self.shape


def engine_is_trt10(engine: Any) -> bool:
    """TensorRT 10 dropped the binding API; TensorRT 8 engines still carry it."""
    return not hasattr(engine, "num_bindings")


def iter_engine_io(engine: Any, trt: Any) -> Iterator[EngineIO]:
    """Walk an engine's I/O tensors with the TensorRT 10 tensor API or the
    TensorRT 8 binding API, whichever the engine exposes."""
    if engine_is_trt10(engine):
        for index in range(engine.num_io_tensors):
            name = engine.get_tensor_name(index)
            is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            shape = tuple(engine.get_tensor_shape(name))
            profile_shape = None
            if is_input and -1 in shape:
                profile_shape = tuple(engine.get_tensor_profile_shape(name, 0)[2])
            yield EngineIO(index, name, engine.get_tensor_dtype(name), is_input, shape, profile_shape)
        return
    for index in range(engine.num_bindings):
        name = engine.get_binding_name(index)
        is_input = bool(engine.binding_is_input(index))
        shape = tuple(engine.get_binding_shape(index))
        profile_shape = None
        if is_input and -1 in shape:
            profile_shape = tuple(engine.get_profile_shape(0, index)[1])
        yield EngineIO(index, name, engine.get_binding_dtype(index), is_input, shape, profile_shape)
