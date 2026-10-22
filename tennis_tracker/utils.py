"""Host-side platform probing: FFmpeg discovery, encoder choice and the CUDA device.

Nothing here touches CUDA; ``torch`` is imported only to ask whether a CUDA
device exists.  TensorRT engine files are ``engine_files``'s business.
"""
from __future__ import annotations

import copy
import glob
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Tuple

import torch

from .config import Config

ENCODER_NVENC = "nvenc"
ENCODER_LIBX264 = "libx264"
ENCODER_OPENCV = "opencv"
FFMPEG_ENCODER_NAMES = {ENCODER_NVENC: "h264_nvenc", ENCODER_LIBX264: "libx264"}

# ``ffmpeg -encoders`` only lists capabilities; it never hangs for long.
_FFMPEG_PROBE_TIMEOUT_S = 5.0
_FFMPEG_ENCODER_CACHE: Dict[Tuple[str, str], bool] = {}
# An encoder that is listed may still fail to open: NVENC needs a driver FFmpeg can load, a
# capable device and a free session, none of which ``-encoders`` checks and all of which FFmpeg
# only finds out at the first frame. One frame of a test pattern through the encoder (about a
# second, once per process) settles it before any run starts.
_FFMPEG_DRY_RUN_TIMEOUT_S = 30.0
_FFMPEG_DRY_RUN_SIZE = "320x180"      # NVENC rejects frames below its minimum size
_FFMPEG_DRY_RUN_CACHE: Dict[Tuple[str, str], Optional[str]] = {}


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


def ffmpeg_encoder_failure(ffmpeg: str, encoder: str) -> Optional[str]:
    """Why ``encoder`` cannot encode on this machine right now, or None when it can: one frame of
    a test pattern is pushed through it to nowhere. FFmpeg opens an encoder at its first frame, so
    this is the only way to learn before a run that NVENC will refuse (no loadable CUDA driver, no
    capable device, every session taken); the answer is kept for the process."""
    key = (str(ffmpeg), str(encoder))
    if key in _FFMPEG_DRY_RUN_CACHE:
        return _FFMPEG_DRY_RUN_CACHE[key]
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi",
               "-i", f"color=size={_FFMPEG_DRY_RUN_SIZE}:rate=30", "-frames:v", "1",
               "-c:v", str(encoder), "-f", "null", "-"]
    try:
        proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                              timeout=_FFMPEG_DRY_RUN_TIMEOUT_S, shell=False, check=False)
    except (OSError, subprocess.SubprocessError) as error:
        failure: Optional[str] = f"could not run ffmpeg ({error})"
    else:
        failure = None if proc.returncode == 0 else ((proc.stderr or "").strip().splitlines() or ["ffmpeg exited"])[-1]
    _FFMPEG_DRY_RUN_CACHE[key] = failure
    return failure


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
    with OpenCV, which needs no external process.  ``nvenc_failure`` is what the
    NVENC dry run reported when FFmpeg lists ``h264_nvenc`` but cannot encode
    with it here, for the platform report.
    """

    ffmpeg: Optional[str]
    chain: Tuple[str, ...]
    nvenc_failure: Optional[str] = None

    @property
    def preferred(self) -> str:
        return self.chain[0]

    @property
    def ffmpeg_encoders(self) -> Tuple[str, ...]:
        return tuple(name for name in self.chain if name != ENCODER_OPENCV)


class EncoderSettings(Protocol):
    """The slice of ``Config`` the encoder choice reads."""

    use_nvenc: bool


def choose_encoder(cfg: EncoderSettings, is_cuda: bool = True) -> EncoderChoice:
    """Decide the encoder fallback order from ``cfg.use_nvenc`` and the FFmpeg build.

    NVENC needs a CUDA device, an FFmpeg with ``h264_nvenc`` and a dry run that
    encodes (``ffmpeg_encoder_failure``: the driver, device and session problems
    FFmpeg only reports at the first frame); libx264 needs an FFmpeg with
    ``libx264``; OpenCV is always last.  ``use_nvenc=False`` means "no FFmpeg at
    all", as it always has.
    """
    if not cfg.use_nvenc:
        return EncoderChoice(None, (ENCODER_OPENCV,))
    ffmpeg = find_ffmpeg()
    chain: List[str] = []
    nvenc_failure: Optional[str] = None
    if ffmpeg:
        if is_cuda and ffmpeg_has_encoder(ffmpeg, FFMPEG_ENCODER_NAMES[ENCODER_NVENC]):
            nvenc_failure = ffmpeg_encoder_failure(ffmpeg, FFMPEG_ENCODER_NAMES[ENCODER_NVENC])
            if nvenc_failure is None:
                chain.append(ENCODER_NVENC)
        if ffmpeg_has_encoder(ffmpeg, FFMPEG_ENCODER_NAMES[ENCODER_LIBX264]):
            chain.append(ENCODER_LIBX264)
    chain.append(ENCODER_OPENCV)
    return EncoderChoice(ffmpeg, tuple(chain), nvenc_failure)


def resolve_runtime(cfg: Config, device: str, *, writes_video: bool) -> Tuple[Config, Optional[EncoderChoice]]:
    """A copy of ``cfg`` with ``device`` set, and the encoder choice for it.

    ``device`` is the resolved device (see ``detect_device``); the caller's
    ``cfg`` is left untouched and nothing undeclared is attached to the copy.
    Pass the ``EncoderChoice`` to ``VideoWriter(..., encoder=...)`` so the
    writer does not have to probe FFmpeg again.  ``writes_video`` says whether
    the run writes any video at all; when it does not, the choice is None and
    FFmpeg is neither looked for nor dry-run, so a JSON-only run never pays the
    second the NVENC dry run costs.
    """
    cfg = copy.copy(cfg)
    cfg.device = str(device)
    encoder = choose_encoder(cfg, is_cuda=str(device).isdigit()) if writes_video else None
    return cfg, encoder


def print_platform_report(cfg: Config, encoder: Optional[EncoderChoice]) -> None:
    """The ``[platform]`` lines for a resolved config: device, why NVENC is
    off when it is, and the accelerations in use (``encoder`` is None when
    the run writes no video, which the report then says in the encoder's place)."""
    is_cuda = str(cfg.device).isdigit()
    if encoder is not None and cfg.use_nvenc and encoder.preferred != ENCODER_NVENC:
        fallback = "libx264" if encoder.preferred == ENCODER_LIBX264 else "OpenCV"
        if encoder.ffmpeg is None:
            reason = "ffmpeg not found"
        elif not is_cuda:
            reason = "requires NVIDIA GPU"
        elif encoder.nvenc_failure is not None:
            reason = f"h264_nvenc cannot encode here: {encoder.nvenc_failure}"
        else:
            reason = "ffmpeg lacks h264_nvenc"
        print(f"[info] NVENC disabled ({reason}), using {fallback}")
    accel = ["TensorRT"]
    if encoder is None:
        accel.append("no video written")
    else:
        accel.append({
            ENCODER_NVENC: "NVENC encoding",
            ENCODER_LIBX264: "libx264 encoding",
        }.get(encoder.preferred, "OpenCV encoding"))
    # Preprocessing has no CPU path; a non-CUDA device fails at detector setup.
    accel.append("CUDA preprocessing" if cfg.enable_preprocess else "no preprocessing")
    print(f"[platform] Device: {cfg.device}")
    print(f"[platform] Accelerations: {', '.join(accel)}")


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
