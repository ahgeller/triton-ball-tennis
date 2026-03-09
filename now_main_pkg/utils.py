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
            cfg._use_libx264 = True
            print("[info] NVENC disabled (requires NVIDIA GPU), using libx264")
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

def find_ffmpeg() -> Optional[str]:
    path = shutil.which("ffmpeg")
    if path:
        return path
    local = os.environ.get("LOCALAPPDATA", "")
    if local:
        for pat in [
            os.path.join(local, "Microsoft", "WinGet", "Packages",
                         "Gyan.FFmpeg_*", "ffmpeg-*", "bin", "ffmpeg.exe"),
            os.path.join(local, "Programs", "ffmpeg", "bin", "ffmpeg.exe"),
        ]:
            matches = glob.glob(pat)
            if matches:
                return matches[0]
    return None

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

