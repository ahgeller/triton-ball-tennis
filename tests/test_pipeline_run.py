"""Golden regression harness for ``tennis_tracker.pipeline.run``.

``run`` needs a CUDA device, a GridTrackNet weight file and TensorRT engines.
Here it runs on the CPU against a synthetic clip: the ``fake_cuda`` fixture
makes ``torch.device("cuda:0")`` the CPU, and the three model backends (in
``tennis_tracker.pipeline``) plus the video writer (in ``pass1``/``pass2``) are
replaced by scripted fakes with the same surface.  Everything else -- decoding, the CUDA motion
path (on CPU tensors), ROI tracking, the selector, pass-2 rendering and the
JSON export -- is the real code.

The golden fixture (``tests/fixtures/pipeline_run_golden.json``) holds the
tracking JSON, minus timing and machine-specific paths, plus a hash of every
frame handed to every writer.  Any behavioural change in ``run`` therefore
fails these tests.  To accept a change deliberately, regenerate the fixture:

    UPDATE_PIPELINE_GOLDEN=1 python3 -m pytest -q tests/test_pipeline_run.py

and commit the new fixture together with the change that caused it.

The frame hashes are byte-exact, so they also depend on how OpenCV rasterises
anti-aliased lines and text; a different OpenCV (or numpy) build can change
them without any tracking behaviour moving.  The fixture therefore records the
versions it was made with (``render_env``), and ``test_rendered_frames_match_golden``
is skipped, with a message, when the running versions differ.  The JSON
comparison (``test_run_matches_golden``) does not depend on that and always
runs.  Regenerate the fixture on the pinned environment to re-arm the frame check.

What the fakes cannot cover: the real GridTrackNet prepass and its worker
thread, the TensorRT player/court engines, real CUDA streams and pinned
uploads, and the FFmpeg/NVENC encoders.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytest

from tennis_tracker import pass1, pass2, pipeline
from tennis_tracker.config import Config
from tennis_tracker.court_detector import CourtDetector

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "pipeline_run_golden.json"
UPDATE_ENV = "UPDATE_PIPELINE_GOLDEN"

WIDTH, HEIGHT, FPS, FRAMES = 640, 360, 30.0, 40
SEED = 20261021

# 14 court keypoints in the detector's semantic order (see CourtDetector._LINE_PAIRS_14):
# 0/1 far baseline corners, 2/3 near baseline corners, 4/6 far singles, 5/7 near
# singles, 8/9 far service-line ends, 10/11 near service-line ends, 12/13 centre marks.
COURT_KEYPOINTS: List[float] = [
    180.0, 90.0,      # 0 far-left corner
    460.0, 90.0,      # 1 far-right corner
    80.0, 330.0,      # 2 near-left corner
    560.0, 330.0,     # 3 near-right corner
    210.0, 90.0,      # 4 far-left singles
    120.0, 330.0,     # 5 near-left singles
    430.0, 90.0,      # 6 far-right singles
    520.0, 330.0,     # 7 near-right singles
    187.5, 150.0,     # 8 far service line, left
    452.5, 150.0,     # 9 far service line, right
    146.25, 260.0,    # 10 near service line, left
    493.75, 260.0,    # 11 near service line, right
    320.0, 150.0,     # 12 far centre mark
    320.0, 260.0,     # 13 near centre mark
]
PLAYER_BOXES: List[List[float]] = [
    [118.0, 228.0, 166.0, 336.0],   # near-side player, bottom left
    [428.0, 66.0, 456.0, 134.0],    # far-side player, top right
]
STATIC_DISTRACTOR = (322, 48)        # bright blob above the far baseline; never moves
BALL_RADIUS_PX = 4.0
DET_BOX_RADIUS_PX = 6.0
# Detector dropouts: frames 9/10/19 still show the ball, so the selector can
# recover them from the motion mask; frames 27-29 also hide it in the video
# (an occlusion), so the selector has to coast through them.
DROPOUT_FRAMES = {9, 10, 19, 27, 28, 29}
OCCLUDED_FRAMES = {27, 28, 29}
FALSE_POSITIVE = {14: (452.0, 150.0, 0.57)}   # frame -> (x, y, conf) added beside the real ball


def ball_path(frame_index: int) -> Tuple[float, float]:
    """A parabolic arc across the court, left to right, apex near mid-court."""
    u = frame_index / float(FRAMES - 1)
    x = 100.0 + 440.0 * u
    y = 300.0 - 4.0 * 170.0 * u * (1.0 - u)
    return x, y


def scripted_detections() -> List[List[Tuple[List[float], float]]]:
    """The detections the fake GridTrackNet hands out, one list per source frame."""
    out: List[List[Tuple[List[float], float]]] = []
    r = DET_BOX_RADIUS_PX
    for i in range(FRAMES):
        dets: List[Tuple[List[float], float]] = []
        if i not in DROPOUT_FRAMES:
            x, y = ball_path(i)
            conf = round(0.62 + 0.3 * abs(math.sin(0.7 * i)), 4)
            dets.append(([x - r, y - r, x + r, y + r], conf))
        if i in FALSE_POSITIVE:
            fx, fy, fconf = FALSE_POSITIVE[i]
            dets.append(([fx - r, fy - r, fx + r, fy + r], fconf))
        out.append(dets)
    return out


def _court_background(rng: np.random.RandomState) -> np.ndarray:
    bg = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
    bg[:] = (38, 108, 48)                                   # dark green, BGR
    bg = bg.astype(np.int16) + rng.randint(-6, 7, size=bg.shape, dtype=np.int16)
    bg = np.clip(bg, 0, 255).astype(np.uint8)
    k = COURT_KEYPOINTS
    for a, b in CourtDetector._LINE_PAIRS_14:
        pa = (int(round(k[2 * a])), int(round(k[2 * a + 1])))
        pb = (int(round(k[2 * b])), int(round(k[2 * b + 1])))
        cv2.line(bg, pa, pb, (225, 225, 225), 2, cv2.LINE_AA)
    for x1, y1, x2, y2 in PLAYER_BOXES:                       # static "players"
        cv2.rectangle(bg, (int(x1), int(y1)), (int(x2), int(y2)), (70, 60, 150), -1)
    cv2.circle(bg, STATIC_DISTRACTOR, 5, (235, 235, 235), -1, cv2.LINE_AA)
    return bg


def synthetic_frames() -> List[np.ndarray]:
    """The clip: a static court plus a moving ball and a little sensor noise."""
    rng = np.random.RandomState(SEED)
    background = _court_background(rng)
    frames = []
    for i in range(FRAMES):
        frame = background.astype(np.int16) + rng.randint(-2, 3, size=background.shape, dtype=np.int16)
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        x, y = ball_path(i)
        shift = 4
        centre = (int(round(x * (1 << shift))), int(round(y * (1 << shift))))
        if i not in OCCLUDED_FRAMES:
            cv2.circle(frame, centre, int(round(BALL_RADIUS_PX * (1 << shift))), (70, 235, 240), -1, cv2.LINE_AA, shift)
        frames.append(frame)
    return frames


def write_lossless_clip(path: Path, frames: List[np.ndarray], fps: float = FPS) -> None:
    """Write ``frames`` losslessly (FFV1 in AVI) and check they decode byte-exact."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), fps, (WIDTH, HEIGHT))
    if not writer.isOpened():
        pytest.skip("this OpenCV build cannot write FFV1; the harness needs a lossless clip")
    for frame in frames:
        writer.write(frame)
    writer.release()
    cap = cv2.VideoCapture(str(path))
    try:
        assert cap.isOpened()
        for i, expected in enumerate(frames):
            ok, decoded = cap.read()
            assert ok, f"frame {i} did not decode"
            if not np.array_equal(decoded, expected):
                pytest.skip("this OpenCV build does not decode FFV1 losslessly; golden would not be portable")
        ok, _ = cap.read()
        assert not ok, "clip has more frames than were written"
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Fakes for the three model backends and the video writer
# ---------------------------------------------------------------------------

@dataclass
class _Pending:
    frame_index: int
    detections: List[Tuple[List[float], float]]


class FakeGridTrackNet:
    """``GridTrackNetBallDetector`` surface with scripted, cached detections."""

    uses_raw_frames = True
    use_async = False
    async_slots = 1
    script: List[List[Tuple[List[float], float]]] = []
    instances: List["FakeGridTrackNet"] = []

    def __init__(self, weights_path: str, cfg: Config):
        self.weights_path = str(weights_path)
        self.cfg = cfg
        self.prepared: Optional[Dict[str, Any]] = None
        self.precomputed: Optional[List[List[Tuple[List[float], float]]]] = None
        self.inputs_seen: List[Tuple[int, Any]] = []
        type(self).instances.append(self)

    def supports_cuda_frame(self) -> bool:
        return False

    def prepare_video(self, video_path: Path, fps: float, width: int, height: int, total: int,
                      background: Optional[bool] = None) -> None:
        self.prepared = {"video_path": str(video_path), "fps": float(fps), "width": int(width),
                         "height": int(height), "total": int(total)}
        self.precomputed = [list(self.script[i]) if i < len(self.script) else [] for i in range(max(0, int(total)))]

    def detections_at(self, index: int) -> List[Tuple[List[float], float]]:
        if self.precomputed is None:
            raise RuntimeError("prepare_video() must run before detection.")
        if index >= len(self.precomputed):
            return []
        return [(list(box), float(conf)) for box, conf in self.precomputed[index]]

    def detect_async_start(self, frame_bgr, frame_index: int) -> _Pending:
        index = int(frame_index)
        self.inputs_seen.append((index, type(frame_bgr).__name__))
        return _Pending(index, self.detections_at(index))

    def detect_async_finish(self, pending: _Pending) -> List[Tuple[List[float], float]]:
        return pending.detections

    def detect(self, frame_bgr, frame_index: int):
        return self.detect_async_finish(self.detect_async_start(frame_bgr, frame_index))


class FakeCourtDetector(CourtDetector):
    """A court detector that always sees the same 14 keypoints.  ``draw`` is the real one."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.keypoints: Optional[List[float]] = None
        self.last_run_frame = -1
        self.session = None
        self.detect_calls: List[int] = []

    def detect(self, frame: np.ndarray, frame_idx: Optional[int] = None) -> Optional[List[float]]:
        self.detect_calls.append(-1 if frame_idx is None else int(frame_idx))
        self.keypoints = list(COURT_KEYPOINTS)
        return self.keypoints


@dataclass
class _PlayerPending:
    frame_idx: int


class FakePlayerDetector:
    """``PlayerDetector`` surface returning two fixed tracked boxes."""

    def __init__(self, cfg: Config, court_keypoints: Optional[List[float]] = None):
        self.cfg = cfg
        self.court_kps = court_keypoints
        self.cached_boxes: List[List[float]] = []
        self.slots: Dict[int, List[float]] = {}
        self.start_calls: List[int] = []

    @property
    def enabled(self) -> bool:
        return True

    def set_court_keypoints(self, court_keypoints: Optional[List[float]]) -> None:
        self.court_kps = court_keypoints

    def detect_async_start(self, frame: np.ndarray, frame_idx: int, frame_gpu_t=None) -> _PlayerPending:
        assert frame.shape == (HEIGHT, WIDTH, 3)
        self.start_calls.append(int(frame_idx))
        return _PlayerPending(int(frame_idx))

    def detect_async_finish(self, pending: Optional[_PlayerPending]) -> List[List[float]]:
        if pending is None:
            return []
        boxes = [list(b) for b in PLAYER_BOXES]
        self.slots = {i + 1: box for i, box in enumerate(boxes)}
        self.cached_boxes = boxes
        return boxes

    def detect(self, frame: np.ndarray, frame_idx: int) -> List[List[float]]:
        return self.detect_async_finish(self.detect_async_start(frame, frame_idx))

    def get_player_dict(self) -> Dict[int, List[float]]:
        return dict(self.slots)


def frame_digest(frame: np.ndarray) -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update(f"{frame.shape}:{frame.dtype}".encode())
    h.update(np.ascontiguousarray(frame).tobytes())
    return h.hexdigest()


class CapturingVideoWriter:
    """``VideoWriter`` surface that keeps a hash of every frame instead of encoding."""

    streams: Dict[str, "CapturingVideoWriter"] = {}

    def __init__(self, path: str, fps: float, w: int, h: int, cfg: Config, encoder=None):
        self._path = str(path)
        self.fps, self.w, self.h = float(fps), int(w), int(h)
        self.encoder = "capture"
        self.hashes: List[str] = []
        self.closed = False
        type(self).streams[self._path] = self

    def write(self, frame: np.ndarray) -> None:
        assert not self.closed, "write after close"
        assert isinstance(frame, np.ndarray) and frame.dtype == np.uint8, "writer needs uint8 frames"
        assert frame.shape == (self.h, self.w, 3), f"frame shape {frame.shape} != {(self.h, self.w, 3)}"
        self.hashes.append(frame_digest(frame))

    def close(self) -> None:
        self.closed = True


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

STREAM_ROLES = {
    "tracking": "output_video",
    "motion_debug": "output_debug_path",
    "detector_input": "output_detector_input_debug_path",
    "track_debug": "output_track_debug_path",
}

FULL_OVERRIDES = dict(
    conf=0.5,
    save_tracking_video=True,
    save_motion_debug=True,
    save_detector_input_debug=True,
    save_track_debug_video=True,
    print_selector_tracks=False,
    use_nvenc=False,
    use_async_writer=False,
    pass2_cache_max_mb=64,
    info_timing=True,
    motion_diagnostics=True,
)

JSON_ONLY_OVERRIDES = dict(
    FULL_OVERRIDES,
    save_tracking_video=False,
    save_motion_debug=False,
    save_detector_input_debug=False,
    save_track_debug_video=False,
)


def make_config(tmp_path: Path, clip: Path, **overrides) -> Config:
    for name in overrides:
        assert hasattr(Config(), name), f"Config has no field {name!r}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    weights = tmp_path / "x.npz"
    if not weights.exists():
        np.savez(weights, placeholder=np.zeros(1))
    out = tmp_path / "out"
    cfg = Config(
        input_video=str(clip),
        output_video=str(out / "tracking.mp4"),
        output_debug_path=str(out / "motion_debug.mp4"),
        output_detector_input_debug_path=str(out / "detector_input.mp4"),
        output_track_debug_path=str(out / "track_debug.mp4"),
        tracking_json=str(out / "tracking.json"),
        model_path=str(weights),
        device="0",
        player_model_path=None,
        court_model_path=None,
    )
    for name, value in overrides.items():
        setattr(cfg, name, value)
    return cfg


def install_fakes(monkeypatch, script=None):
    FakeGridTrackNet.script = scripted_detections() if script is None else script
    FakeGridTrackNet.instances = []
    CapturingVideoWriter.streams = {}
    monkeypatch.setattr(pipeline, "GridTrackNetBallDetector", FakeGridTrackNet)
    monkeypatch.setattr(pipeline, "CourtDetector", FakeCourtDetector)
    monkeypatch.setattr(pipeline, "PlayerDetector", FakePlayerDetector)
    monkeypatch.setattr(pass1, "VideoWriter", CapturingVideoWriter)     # detector-input writer
    monkeypatch.setattr(pass2, "VideoWriter", CapturingVideoWriter)     # tracking and debug writers


def comparable_json(payload: Dict[str, Any]) -> Dict[str, Any]:
    """The tracking JSON minus timing and machine-specific paths."""
    out = copy.deepcopy(payload)
    out.pop("config", None)
    out.pop("timing", None)
    for key in ("input", "output"):
        out.get("video", {}).pop(key, None)
    for key in ("elapsed_sec", "effective_fps"):
        out.get("summary", {}).pop(key, None)
    return out


def run_pipeline(tmp_path: Path, monkeypatch, clip: Path, **overrides) -> Dict[str, Any]:
    install_fakes(monkeypatch)
    cfg = make_config(tmp_path, clip, **overrides)
    result = pipeline.run(cfg)
    assert result is None
    with open(cfg.tracking_json, encoding="utf-8") as f:
        payload = json.load(f)
    streams = {}
    for role, attr in STREAM_ROLES.items():
        writer = CapturingVideoWriter.streams.get(getattr(cfg, attr))
        if writer is not None:
            assert writer.closed, f"{role} writer was not closed"
            streams[role] = list(writer.hashes)
    return {"json": comparable_json(payload), "raw_json": payload, "streams": streams, "cfg": cfg}


def render_env() -> Dict[str, str]:
    """The library versions the rendered-frame hashes depend on."""
    return {"opencv": cv2.__version__, "numpy": np.__version__}


def load_golden() -> Dict[str, Any]:
    if not FIXTURE_PATH.exists():
        pytest.fail(f"missing golden fixture {FIXTURE_PATH}; run with {UPDATE_ENV}=1 to create it")
    with FIXTURE_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def maybe_update_golden(section: Dict[str, Any]) -> None:
    """With UPDATE_PIPELINE_GOLDEN=1, replace these sections of the fixture and keep the rest."""
    if os.environ.get(UPDATE_ENV) == "1":
        golden = dict(load_golden()) if FIXTURE_PATH.exists() else {}
        golden.update(section)
        FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with FIXTURE_PATH.open("w", encoding="utf-8") as f:
            json.dump(golden, f, indent=1, sort_keys=True)
            f.write("\n")


@pytest.fixture(scope="module")
def clip(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("clip") / "synthetic_rally.avi"
    write_lossless_clip(path, synthetic_frames())
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_pinned_uploader_falls_back_without_a_driver(fake_cuda):
    """``FrameWindow`` wraps ``PinnedFrameUploader`` in try/except; on a box without a
    CUDA driver the pinned allocation must raise so that fallback is what the
    golden captured.  On a real CUDA machine it may succeed; the values are the
    same either way, only the upload route differs."""
    import torch

    from tennis_tracker.video_io import PinnedFrameUploader

    try:
        uploader = PinnedFrameUploader(HEIGHT, WIDTH, torch.device("cuda:0"))
    except RuntimeError:
        return
    frame = synthetic_frames()[0]
    direct = torch.from_numpy(frame).to(dtype=torch.float32).permute(2, 0, 1).contiguous() / 255.0
    assert torch.equal(uploader.upload_chw_f32(frame), direct)


def test_run_matches_golden(tmp_path, monkeypatch, fake_cuda, capsys, clip):
    result = run_pipeline(tmp_path, monkeypatch, clip, **FULL_OVERRIDES)
    out = capsys.readouterr().out
    golden = {"json": result["json"], "streams": result["streams"], "render_env": render_env()}
    maybe_update_golden(golden)
    expected = load_golden()

    # Sanity checks on the scenario itself, so a golden regenerated by mistake
    # on a degenerate run is noticed.
    frames = result["json"]["frames"]
    assert len(frames) == FRAMES
    sources = {row.get("source") for row in frames if row.get("present")}
    assert "det" in sources
    assert sources & {"motion", "interp"}, f"no filled gap sources in {sources}"
    assert result["json"]["chosen_track"] is not None
    assert result["json"]["last_valid_court_keypoints"] == COURT_KEYPOINTS
    assert result["json"]["summary"]["pass2_frames_rendered"] == FRAMES
    assert result["json"]["motion_diagnostics"]["summary"]["frames_analyzed"] == FRAMES
    assert set(result["streams"]) == {"tracking", "motion_debug", "detector_input", "track_debug"}
    assert all(len(hashes) == FRAMES for hashes in result["streams"].values())
    assert "[pass 1] RAM frame cache enabled" in out
    assert "[pass 2] Using RAM frame cache" in out
    detector = FakeGridTrackNet.instances[-1]
    assert detector.prepared == {"video_path": str(Path(result["cfg"].input_video).resolve()), "fps": FPS,
                                 "width": WIDTH, "height": HEIGHT, "total": FRAMES}
    assert [index for index, _ in detector.inputs_seen] == list(range(FRAMES))
    assert "timing" in result["raw_json"] and result["raw_json"]["timing"].get("pass1_total", 0.0) > 0.0

    assert result["json"] == expected["json"], "tracking JSON changed; see module docstring to update the golden"


def test_rendered_frames_match_golden(tmp_path, monkeypatch, fake_cuda, clip):
    """Byte-exact hashes of every rendered frame, checked only on the fixture's OpenCV/numpy."""
    expected = load_golden()
    if expected.get("render_env") != render_env():
        pytest.skip(
            f"golden frames were rendered with {expected.get('render_env')}, this run has "
            f"{render_env()}; OpenCV rasterisation differs between builds, so the frame hashes "
            f"are not comparable.  Regenerate the fixture here ({UPDATE_ENV}=1) to check them."
        )
    result = run_pipeline(tmp_path, monkeypatch, clip, **FULL_OVERRIDES)
    assert set(result["streams"]) == set(expected["streams"])
    for role, hashes in expected["streams"].items():
        changed = [i for i, (a, b) in enumerate(zip(result["streams"][role], hashes)) if a != b]
        assert result["streams"][role] == hashes, f"{role} video frames changed at {changed[:10]}"


def test_json_only_run_matches_golden_without_rendering(tmp_path, monkeypatch, fake_cuda, capsys, clip):
    result = run_pipeline(tmp_path, monkeypatch, clip, **JSON_ONLY_OVERRIDES)
    out = capsys.readouterr().out
    expected = copy.deepcopy(load_golden()["json"])
    expected["summary"]["pass2_frames_rendered"] = 0
    assert result["streams"] == {}
    assert "[pass 2] Skipped (no video outputs selected)" in out
    assert not Path(result["cfg"].output_video).exists()
    assert result["json"] == expected


def test_disabling_preprocess_runs_on_detections_alone(tmp_path, monkeypatch, fake_cuda, capsys, clip):
    """``enable_preprocess=False`` skips the motion masks and ROI tracking;
    the selector then works from detections only (no 'motion' fills).  The
    JSON is the third golden section, ``no_preprocess_json``."""
    result = run_pipeline(tmp_path, monkeypatch, clip, **JSON_ONLY_OVERRIDES, enable_preprocess=False)
    out = capsys.readouterr().out
    assert "[preprocess] disabled" in out
    frames = result["json"]["frames"]
    assert len(frames) == FRAMES
    sources = {row.get("source") for row in frames if row.get("present")}
    assert "det" in sources and "motion" not in sources
    assert result["json"]["motion_diagnostics"]["summary"]["mask_source_counts"] == {"none": FRAMES}
    maybe_update_golden({"no_preprocess_json": result["json"]})
    assert result["json"] == load_golden()["no_preprocess_json"]


def test_zero_frame_video_raises_and_opens_no_writer(tmp_path, monkeypatch, fake_cuda, capsys):
    """An openable clip with no frames is an error like an unopenable one, and
    no output writer is opened (so no encoder process is leaked)."""
    empty = tmp_path / "empty.avi"
    writer = cv2.VideoWriter(str(empty), cv2.VideoWriter_fourcc(*"FFV1"), FPS, (WIDTH, HEIGHT))
    assert writer.isOpened()
    writer.release()
    cap = cv2.VideoCapture(str(empty))
    opened, frames = cap.isOpened(), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert opened and frames == 0

    install_fakes(monkeypatch)
    cfg = make_config(tmp_path, empty, **FULL_OVERRIDES)
    with pytest.raises(RuntimeError, match="No frames in video"):
        pipeline.run(cfg)
    out = capsys.readouterr().out
    assert "[pass 1] Collecting detections" not in out
    assert not Path(cfg.tracking_json).exists()
    assert FakeGridTrackNet.instances[-1].prepared["total"] == 0
    assert CapturingVideoWriter.streams == {}


def test_skip_frame_detection_hands_the_detector_the_true_frame_index(tmp_path, monkeypatch, fake_cuda, clip):
    """With ``skip_frame_detect=2`` the detector is asked about every other
    frame; because run() passes ``frame_index``, the frames it is asked about
    are the even source frames, not a call-order sequence shifted by the skips."""
    result = run_pipeline(tmp_path, monkeypatch, clip, **JSON_ONLY_OVERRIDES,
                          skip_frame_detect=2, skip_frame_require_roi=False)
    detector = FakeGridTrackNet.instances[-1]
    assert [index for index, _ in detector.inputs_seen] == list(range(0, FRAMES, 2))
    det_frames = {row["frame"] for row in result["json"]["frames"] if row.get("source") == "det"}
    assert det_frames and all(frame % 2 == 0 for frame in det_frames)


def test_run_is_deterministic_in_process(tmp_path, monkeypatch, fake_cuda, clip):
    first = run_pipeline(tmp_path / "a", monkeypatch, clip, **FULL_OVERRIDES)
    second = run_pipeline(tmp_path / "b", monkeypatch, clip, **FULL_OVERRIDES)
    assert first["json"] == second["json"]
    assert first["streams"] == second["streams"]
