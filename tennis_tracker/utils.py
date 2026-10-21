"""Host-side helpers: FFmpeg discovery, encoder choice, device probing and
TensorRT engine files (Ultralytics metadata header, class names, I/O table).

Nothing here touches CUDA; ``torch`` is imported only to ask whether a CUDA
device exists.
"""
from __future__ import annotations

import copy
import glob
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch

from .config import Config

ENCODER_NVENC = "nvenc"
ENCODER_LIBX264 = "libx264"
ENCODER_OPENCV = "opencv"
FFMPEG_ENCODER_NAMES = {ENCODER_NVENC: "h264_nvenc", ENCODER_LIBX264: "libx264"}

# ``ffmpeg -encoders`` only lists capabilities; it never hangs for long.
_FFMPEG_PROBE_TIMEOUT_S = 5.0
_FFMPEG_ENCODER_CACHE: Dict[Tuple[str, str], bool] = {}

# Ultralytics .engine files start with a little-endian uint32 length followed
# by that many bytes of JSON metadata, then the serialized TensorRT engine.
_ENGINE_HEADER_LEN_BYTES = 4
_DEFAULT_ENGINE_NAMES: Dict[int, str] = {0: "ball"}
_BALL_CLASS_ALIASES = ("ball", "sports ball", "tennis ball", "sports_ball", "tennis_ball")


def ffmpeg_has_encoder(ffmpeg: Optional[str], encoder: str) -> bool:
    """Return whether the discovered FFmpeg build exposes a specific encoder."""
    if not ffmpeg:
        return False
    key = (str(ffmpeg), str(encoder))
    cached = _FFMPEG_ENCODER_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        proc = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=_FFMPEG_PROBE_TIMEOUT_S,
            shell=False,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        print(f"[info] Could not probe FFmpeg encoders ({error}); assuming '{encoder}' is unavailable")
        ok = False
    else:
        output = (proc.stdout or "") + (proc.stderr or "")
        ok = proc.returncode == 0 and any(
            len(line.split()) >= 2 and line.split()[1] == str(encoder)
            for line in output.splitlines()
        )
    _FFMPEG_ENCODER_CACHE[key] = bool(ok)
    return bool(ok)


def detect_device(requested: str) -> Tuple[str, str]:
    """Resolve ``Config.device``: an explicit value is returned as given, ``auto``
    picks CUDA device 0 when there is one.  The tracking runtime rejects
    anything that is not a CUDA index, so there is no CPU/MPS fallback."""
    if requested != "auto":
        return requested, f"user-specified ({requested})"
    if torch.cuda.is_available():
        return "0", f"CUDA ({torch.cuda.get_device_name(0)})"
    return "cpu", "no CUDA device"


@dataclass(frozen=True)
class EncoderChoice:
    """Which encoders ``VideoWriter`` may use, in the order it tries them.

    ``ffmpeg`` is the FFmpeg binary (None when none was found); ``chain`` holds
    ``ENCODER_NVENC`` / ``ENCODER_LIBX264`` / ``ENCODER_OPENCV`` and always ends
    with OpenCV, which needs no external process.
    """

    ffmpeg: Optional[str]
    chain: Tuple[str, ...]

    @property
    def preferred(self) -> str:
        return self.chain[0]

    @property
    def ffmpeg_encoders(self) -> Tuple[str, ...]:
        return tuple(name for name in self.chain if name != ENCODER_OPENCV)


def choose_encoder(cfg: Config, is_cuda: bool = True) -> EncoderChoice:
    """Decide the encoder fallback order from ``cfg.use_nvenc`` and the FFmpeg build.

    NVENC needs a CUDA device and an FFmpeg with ``h264_nvenc``; libx264 needs
    an FFmpeg with ``libx264``; OpenCV is always last.  ``use_nvenc=False``
    means "no FFmpeg at all", as it always has.
    """
    if not cfg.use_nvenc:
        return EncoderChoice(None, (ENCODER_OPENCV,))
    ffmpeg = find_ffmpeg()
    chain: List[str] = []
    if ffmpeg:
        if is_cuda and ffmpeg_has_encoder(ffmpeg, FFMPEG_ENCODER_NAMES[ENCODER_NVENC]):
            chain.append(ENCODER_NVENC)
        if ffmpeg_has_encoder(ffmpeg, FFMPEG_ENCODER_NAMES[ENCODER_LIBX264]):
            chain.append(ENCODER_LIBX264)
    chain.append(ENCODER_OPENCV)
    return EncoderChoice(ffmpeg, tuple(chain))


def check_capabilities(cfg: Config, device_str: str) -> Tuple[Config, EncoderChoice]:
    """Report the platform and return the config copy plus the encoder choice.

    The returned ``Config`` is a copy of ``cfg``; nothing undeclared is attached
    to it.  Pass the ``EncoderChoice`` to ``VideoWriter(..., encoder=...)`` so
    the writer does not have to probe FFmpeg again.
    """
    cfg = copy.copy(cfg)
    is_cuda = str(device_str).isdigit()
    encoder = choose_encoder(cfg, is_cuda=is_cuda)
    if cfg.use_nvenc and encoder.preferred != ENCODER_NVENC:
        fallback = "libx264" if encoder.preferred == ENCODER_LIBX264 else "OpenCV"
        if encoder.ffmpeg is None:
            reason = "ffmpeg not found"
        elif not is_cuda:
            reason = "requires NVIDIA GPU"
        else:
            reason = "ffmpeg lacks h264_nvenc"
        print(f"[info] NVENC disabled ({reason}), using {fallback}")
    accel = ["TensorRT"]
    accel.append({
        ENCODER_NVENC: "NVENC encoding",
        ENCODER_LIBX264: "libx264 encoding",
    }.get(encoder.preferred, "OpenCV encoding"))
    accel.append("CUDA preprocessing" if is_cuda and cfg.enable_preprocess
                 else "CPU preprocessing" if cfg.enable_preprocess else "no preprocessing")
    print(f"[platform] Device: {device_str}")
    print(f"[platform] Accelerations: {', '.join(accel)}")
    return cfg, encoder


def _ffmpeg_candidates() -> List[str]:
    candidates: List[str] = []

    def add(path: Optional[str]) -> None:
        if not path:
            return
        try:
            resolved = str(Path(path).resolve())
        except OSError:
            resolved = str(path)
        if os.path.exists(resolved) and resolved not in candidates:
            candidates.append(resolved)

    add(os.environ.get("TRITON_FFMPEG"))
    add(os.environ.get("FFMPEG_BINARY"))
    add(shutil.which("ffmpeg"))

    conda_ffmpeg = Path(sys.prefix) / "Library" / "bin" / "ffmpeg.exe"
    add(str(conda_ffmpeg))

    local = os.environ.get("LOCALAPPDATA", "")
    if local:
        for pat in [
            os.path.join(local, "Microsoft", "WinGet", "Packages",
                         "Gyan.FFmpeg_*", "ffmpeg-*", "bin", "ffmpeg.exe"),
            os.path.join(local, "Programs", "ffmpeg", "bin", "ffmpeg.exe"),
        ]:
            matches = glob.glob(pat)
            if matches:
                for match in sorted(matches, reverse=True):
                    add(match)
    return candidates


def find_ffmpeg(prefer_encoder: Optional[str] = "h264_nvenc") -> Optional[str]:
    candidates = _ffmpeg_candidates()
    if prefer_encoder:
        for path in candidates:
            if ffmpeg_has_encoder(path, prefer_encoder):
                return path
    return candidates[0] if candidates else None


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
