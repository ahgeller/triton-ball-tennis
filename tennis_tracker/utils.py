import copy
import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
try:
    import torch
    HAS_TORCH = True
except Exception:
    torch = None
    HAS_TORCH = False

from .config import Config

_FFMPEG_ENCODER_CACHE: Dict[Tuple[str, str], bool] = {}


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
            timeout=5,
        )
        output = (proc.stdout or "") + (proc.stderr or "")
        ok = proc.returncode == 0 and any(
            len(line.split()) >= 2 and line.split()[1] == str(encoder)
            for line in output.splitlines()
        )
    except Exception:
        ok = False
    _FFMPEG_ENCODER_CACHE[key] = bool(ok)
    return bool(ok)


def _detect_device(requested: str) -> Tuple[str, str]:
    HAS_CUDA = HAS_TORCH and torch.cuda.is_available()
    HAS_MPS = HAS_TORCH and hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if requested != "auto":
        return requested, f"user-specified ({requested})"
    if HAS_CUDA:
        return "0", f"CUDA ({torch.cuda.get_device_name(0)})"
    if HAS_MPS:
        return "mps", "Apple MPS"
    return "cpu", "CPU"

def _check_capabilities(cfg: Config, device_str: str) -> Config:
    cfg = copy.copy(cfg)
    is_cuda = device_str not in ("cpu", "mps")

    if cfg.use_tensorrt and not is_cuda:
        cfg.use_tensorrt = False
        print("[info] TensorRT disabled (requires NVIDIA CUDA GPU)")
    if cfg.use_tensorrt and is_cuda:
        try:
            import tensorrt  # noqa: F401
        except ImportError:
            cfg.use_tensorrt = False
            print("[info] TensorRT disabled (tensorrt package not found)")

    if cfg.use_nvenc:
        ffmpeg = find_ffmpeg()
        if not ffmpeg:
            cfg.use_nvenc = False
            cfg._use_libx264 = False
            print("[info] Hardware encoding disabled (ffmpeg not found)")
        elif not is_cuda:
            cfg.use_nvenc = False
            cfg._use_libx264 = ffmpeg_has_encoder(ffmpeg, "libx264")
            msg = "using libx264" if cfg._use_libx264 else "using OpenCV"
            print(f"[info] NVENC disabled (requires NVIDIA GPU), {msg}")
        elif not ffmpeg_has_encoder(ffmpeg, "h264_nvenc"):
            cfg.use_nvenc = False
            cfg._use_libx264 = ffmpeg_has_encoder(ffmpeg, "libx264")
            msg = "using libx264" if cfg._use_libx264 else "using OpenCV"
            print(f"[info] NVENC disabled (ffmpeg lacks h264_nvenc), {msg}")
    if not hasattr(cfg, "_use_libx264"):
        cfg._use_libx264 = False
    if cfg.tensorrt_half and not is_cuda:
        cfg.tensorrt_half = False

    accel = []
    if cfg.use_tensorrt:
        accel.append("TensorRT FP16")
    if cfg.use_nvenc:
        accel.append("NVENC encoding")
    elif cfg._use_libx264:
        accel.append("libx264 encoding")
    else:
        accel.append("OpenCV encoding")
    accel.append("CUDA preprocessing" if is_cuda and cfg.enable_preprocess
                 else "CPU preprocessing" if cfg.enable_preprocess else "no preprocessing")
    print(f"[platform] Device: {device_str}")
    print(f"[platform] Accelerations: {', '.join(accel)}")
    return cfg

def _ffmpeg_candidates() -> List[str]:
    candidates: List[str] = []

    def add(path: Optional[str]) -> None:
        if not path:
            return
        try:
            resolved = str(Path(path).resolve())
        except Exception:
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

def _normalize_class_name(name: Optional[str]) -> str:
    if name is None:
        return ""
    return str(name).strip().lower().replace("_", " ").replace("-", " ")

def _get_name_from_id(names, cls_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(cls_id, ""))
    if isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
        return str(names[cls_id])
    return ""

def find_ball_class_id_from_names(names, override: Optional[str] = None):
    if isinstance(names, list):
        names = dict(enumerate(names))
    if not isinstance(names, dict) or not names:
        names = {0: "ball"}
    names = {int(k): str(v) for k, v in names.items()}
    if len(names) == 1:
        cid = list(names.keys())[0]
        return cid, names[cid]
    if override:
        for cid, name in names.items():
            if name.lower() == override.lower():
                return cid, name
        print(f"[WARN] class_name='{override}' not in {names}")
    for candidate in ["ball", "sports ball", "tennis ball", "sports_ball", "tennis_ball"]:
        for cid, name in names.items():
            if name.lower() == candidate:
                print(f"[info] Auto-detected ball class: {cid} -> '{name}'")
                return cid, name
    print(f"[WARN] Could not find ball class in {names}. All classes used!")
    return None, None

def _resolve_engine_path_for_ball(cfg: Config) -> Optional[Path]:
    base = Path(cfg.model_path)
    if base.suffix.lower() == ".engine":
        return base if base.exists() else None
    engine = base.with_suffix(".engine")
    return engine if engine.exists() else None

def _resolve_engine_path(model_path: Optional[str]) -> Optional[Path]:
    if not model_path:
        return None
    base = Path(model_path)
    if base.suffix.lower() == ".engine":
        return base if base.exists() else None
    engine = base.with_suffix(".engine")
    return engine if engine.exists() else None

def _read_engine_names(engine_path: Path) -> Dict[int, str]:
    if engine_path is None or not engine_path.exists():
        return {0: "ball"}
    try:
        with open(engine_path, "rb") as f:
            meta_len = int.from_bytes(f.read(4), byteorder="little")
            meta = json.loads(f.read(meta_len).decode("utf-8"))
        names = meta.get("names")
        if isinstance(names, list):
            return {i: str(n) for i, n in enumerate(names)}
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
    except Exception:
        pass
    return {0: "ball"}

