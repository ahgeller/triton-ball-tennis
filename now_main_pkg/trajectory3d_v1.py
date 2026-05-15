"""Physics-first 3D trajectory and event reconstruction sidecar.

This module is intentionally conservative.  It treats 2D bounce/hit cues as
candidate events, then fits simple 3D ballistic arcs and reports confidence
instead of turning a noisy V-shaped 2D trail into a hard event.

Research anchors for the design:
- LosslessCut project files are JSON5-like .llc files:
  https://github.com/mifi/lossless-cut/blob/master/docs/index.md
- TT3D optimizes physics trajectories against reprojection error:
  https://arxiv.org/abs/2504.10035
- MonoTrack uses court/player/trajectory context for hit segmentation:
  https://arxiv.org/pdf/2204.01899
- OpenCV solvePnPRansac is used for robust camera pose from court points:
  https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import json5  # type: ignore
except Exception:  # pragma: no cover - exercised by CLI error path.
    json5 = None

try:
    from scipy.optimize import least_squares
except Exception:  # pragma: no cover - scipy is optional for importability.
    least_squares = None


COURT_WIDTH_M = 10.97
COURT_LENGTH_M = 23.77
BALL_RADIUS_M = 0.0335
GRAVITY_MPS2 = 9.81

SOURCE_WEIGHTS = {
    "det": 1.0,
    "motion": 0.55,
    "guide": 0.25,
    "carry": 0.20,
    "interp": 0.10,
}
SOFT_SOURCES = {"carry", "interp", "guide"}


def _import_cv2():
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on active env.
        raise RuntimeError(
            "OpenCV is required for 3D calibration/rendering. Activate the "
            "project environment or install opencv-python==4.9.0.80."
        ) from exc
    return cv2


def _finite_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


@dataclass
class LLCutSegment:
    source_start_sec: float
    source_end_sec: float
    name: str = ""
    selected: bool = True

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.source_end_sec - self.source_start_sec)


@dataclass
class SegmentWindow:
    segment_id: int
    source_start_sec: float
    source_end_sec: float
    merged_start_sec: float
    merged_end_sec: float
    start_frame: int
    end_frame: int
    mapping_confidence: float
    timebase: str
    warnings: List[str] = field(default_factory=list)

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.merged_end_sec - self.merged_start_sec)


@dataclass
class Observation:
    frame: int
    t_sec: float
    x: float
    y: float
    source: str
    source_weight: float
    conf: float
    weight: float
    interpolated: bool = False
    player_boxes: List[List[float]] = field(default_factory=list)
    court_keypoints: Optional[List[float]] = None


@dataclass
class CameraModel:
    K: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray
    reprojection_error_px: float
    inlier_count: int
    image_point_count: int
    court_keypoints: List[float]

    def to_json(self) -> Dict[str, Any]:
        return {
            "K": self.K.tolist(),
            "rvec": self.rvec.reshape(-1).tolist(),
            "tvec": self.tvec.reshape(-1).tolist(),
            "reprojection_error_px": float(self.reprojection_error_px),
            "inlier_count": int(self.inlier_count),
            "image_point_count": int(self.image_point_count),
            "court_keypoints": list(self.court_keypoints),
        }


@dataclass
class ArcFit:
    start_frame: int
    end_frame: int
    observation_count: int
    params: Optional[np.ndarray]
    mean_reprojection_px: Optional[float]
    max_reprojection_px: Optional[float]
    success: bool
    message: str = ""
    visibility_penalty_frames: int = 0

    def to_json(self) -> Dict[str, Any]:
        return {
            "start_frame": int(self.start_frame),
            "end_frame": int(self.end_frame),
            "observation_count": int(self.observation_count),
            "params": None if self.params is None else self.params.tolist(),
            "mean_reprojection_px": self.mean_reprojection_px,
            "max_reprojection_px": self.max_reprojection_px,
            "success": bool(self.success),
            "message": self.message,
            "visibility_penalty_frames": int(self.visibility_penalty_frames),
        }


def load_llc_cuts(path: str | Path) -> Tuple[List[LLCutSegment], Dict[str, Any]]:
    """Load selected LosslessCut .llc JSON5 cut segments."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if json5 is not None:
        doc = json5.loads(text)
    else:
        doc = _parse_losslesscut_json5_subset(text)
    raw_segments = doc.get("cutSegments") or []
    cuts: List[LLCutSegment] = []
    for raw in raw_segments:
        selected = bool(raw.get("selected", True))
        start = _finite_float(raw.get("start"))
        end = _finite_float(raw.get("end"))
        if start is None or end is None or end <= start:
            continue
        cuts.append(
            LLCutSegment(
                source_start_sec=float(start),
                source_end_sec=float(end),
                name=str(raw.get("name", "") or ""),
                selected=selected,
            )
        )
    metadata = {
        "version": doc.get("version"),
        "mediaFileName": doc.get("mediaFileName"),
        "segment_count": len(raw_segments),
        "selected_count": sum(1 for c in cuts if c.selected),
    }
    return [c for c in cuts if c.selected], metadata


def _parse_losslesscut_json5_subset(text: str) -> Dict[str, Any]:
    """Parse the LosslessCut .llc subset when the json5 package is unavailable.

    This is not a general JSON5 parser; it handles the structure LosslessCut
    writes for project files: object keys without quotes, single-quoted strings,
    booleans/null, and trailing commas.  The proper parser is still json5 when
    installed, but this keeps the sidecar runnable in older project envs.
    """
    s = re.sub(r"//.*?$", "", text, flags=re.MULTILINE)
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.DOTALL)
    s = re.sub(r"([{\[,]\s*)([A-Za-z_$][A-Za-z0-9_$]*)\s*:", r'\1"\2":', s)
    s = re.sub(r",\s*([}\]])", r"\1", s)

    def _single_to_double(match: re.Match) -> str:
        inner = match.group(1)
        inner = inner.replace("\\'", "'")
        return json.dumps(inner)

    s = re.sub(r"'((?:\\.|[^'\\])*)'", _single_to_double, s)
    try:
        doc = json.loads(s)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Could not parse .llc file. Install json5 for full JSON5 support, "
            "or check that the LosslessCut project file is valid."
        ) from exc
    if not isinstance(doc, dict):
        raise RuntimeError("LosslessCut .llc root must be an object.")
    return doc


def cut_duration_tolerance(video_duration_sec: float) -> float:
    """Tolerance for keyframe/export drift in merged LosslessCut videos."""
    return min(max(1.0, 0.02 * max(0.0, float(video_duration_sec))), 5.0)


def map_cuts_to_video_windows(
    cuts: Sequence[LLCutSegment],
    fps: float,
    total_frames: int,
    video_duration_sec: Optional[float] = None,
    timebase: str = "auto",
) -> Tuple[List[SegmentWindow], Dict[str, Any]]:
    """Map source-timeline cuts to frame windows in either original or merged video."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    total_frames = max(0, int(total_frames))
    if video_duration_sec is None:
        video_duration_sec = float(total_frames) / float(fps) if total_frames else 0.0
    selected = [c for c in cuts if c.selected and c.duration_sec > 0]
    if not selected:
        end_sec = max(0.0, float(video_duration_sec))
        end_frame = max(0, total_frames - 1)
        return [
            SegmentWindow(
                segment_id=0,
                source_start_sec=0.0,
                source_end_sec=end_sec,
                merged_start_sec=0.0,
                merged_end_sec=end_sec,
                start_frame=0,
                end_frame=end_frame,
                mapping_confidence=1.0,
                timebase="none",
            )
        ], {"mode": "none", "warnings": ["No selected .llc cuts; using full tracking video."]}

    cut_sum = sum(c.duration_sec for c in selected)
    max_source_end = max(c.source_end_sec for c in selected)
    tol = cut_duration_tolerance(float(video_duration_sec))
    warnings: List[str] = []

    if timebase not in {"auto", "merged", "original"}:
        raise ValueError("timebase must be auto, merged, or original")

    mode = timebase
    if timebase == "auto":
        if abs(cut_sum - float(video_duration_sec)) <= tol:
            mode = "merged"
        elif max_source_end <= float(video_duration_sec) + tol:
            mode = "original"
        else:
            mode = "merged"
            warnings.append(
                "Could not prove timebase from duration; using merged cut order because "
                "source cut end exceeds tracking video duration."
            )

    drift = float(video_duration_sec) - cut_sum
    mapping_confidence = 1.0
    if mode == "merged" and abs(drift) > tol:
        mapping_confidence = 0.45
        warnings.append(
            f"Merged video duration differs from selected cuts by {drift:.3f}s "
            f"(tolerance {tol:.3f}s). Segment mapping is low confidence."
        )
    elif mode == "merged" and abs(drift) > 1e-3:
        mapping_confidence = 0.85
        warnings.append(
            f"Merged video duration differs from selected cuts by {drift:.3f}s; "
            "continuing and clamping the final segment."
        )

    windows: List[SegmentWindow] = []
    cursor = 0.0
    for idx, cut in enumerate(selected):
        if mode == "original":
            start_sec = cut.source_start_sec
            end_sec = cut.source_end_sec
        else:
            start_sec = cursor
            end_sec = cursor + cut.duration_sec
            cursor = end_sec
            if idx == len(selected) - 1 and abs(float(video_duration_sec) - end_sec) <= tol:
                end_sec = float(video_duration_sec)

        start_frame = int(max(0, round(start_sec * fps)))
        end_frame = int(max(start_frame, math.ceil(end_sec * fps) - 1))
        if total_frames > 0:
            start_frame = min(start_frame, total_frames - 1)
            end_frame = min(end_frame, total_frames - 1)
        seg_warnings = []
        if end_frame <= start_frame:
            seg_warnings.append("Segment collapsed to <=1 frame after clamping.")
        windows.append(
            SegmentWindow(
                segment_id=idx,
                source_start_sec=cut.source_start_sec,
                source_end_sec=cut.source_end_sec,
                merged_start_sec=float(start_sec),
                merged_end_sec=float(end_sec),
                start_frame=int(start_frame),
                end_frame=int(end_frame),
                mapping_confidence=float(mapping_confidence),
                timebase=mode,
                warnings=list(seg_warnings),
            )
        )

    mapping = {
        "mode": mode,
        "video_duration_sec": float(video_duration_sec),
        "selected_cut_duration_sec": float(cut_sum),
        "duration_drift_sec": float(drift),
        "duration_tolerance_sec": float(tol),
        "mapping_confidence": float(mapping_confidence),
        "warnings": warnings,
    }
    return windows, mapping


def load_tracking_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_player_boxes(value: Any) -> List[List[float]]:
    if value is None:
        return []
    if isinstance(value, dict):
        raw_boxes = list(value.values())
    else:
        raw_boxes = value
    boxes: List[List[float]] = []
    for box in raw_boxes or []:
        if box is None or len(box) < 4:
            continue
        vals = [_finite_float(v) for v in box[:4]]
        if any(v is None for v in vals):
            continue
        boxes.append([float(v) for v in vals if v is not None])
    return boxes


def _valid_keypoints(kps: Any) -> Optional[List[float]]:
    if not isinstance(kps, (list, tuple)) or len(kps) < 16:
        return None
    vals: List[float] = []
    for v in kps:
        fv = _finite_float(v)
        if fv is None:
            return None
        vals.append(float(fv))
    valid_pairs = 0
    for i in range(0, min(len(vals), 16), 2):
        if abs(vals[i]) > 1e-6 or abs(vals[i + 1]) > 1e-6:
            valid_pairs += 1
    return vals if valid_pairs >= 4 else None


def _frame_court_keypoints(frame: Dict[str, Any]) -> Optional[List[float]]:
    return _valid_keypoints(
        frame.get("court_keypoints")
        or frame.get("court_kps")
        or frame.get("courtKeypoints")
    )


def choose_segment_court_keypoints(
    frames: Sequence[Dict[str, Any]],
    start_frame: int,
    end_frame: int,
    fallback: Optional[Sequence[float]] = None,
) -> Optional[List[float]]:
    """Pick a valid court keypoint set inside this segment, without crossing cuts."""
    for idx in range(max(0, start_frame), min(len(frames), end_frame + 1)):
        kps = _frame_court_keypoints(frames[idx])
        if kps is not None:
            return kps
    return _valid_keypoints(fallback)


def extract_observations(
    frames: Sequence[Dict[str, Any]],
    start_frame: int,
    end_frame: int,
    fps: float,
) -> List[Observation]:
    obs: List[Observation] = []
    lo = max(0, int(start_frame))
    hi = min(len(frames) - 1, int(end_frame))
    for idx in range(lo, hi + 1):
        row = frames[idx] or {}
        if not row.get("present", False):
            continue
        x = _finite_float(row.get("x"))
        y = _finite_float(row.get("y"))
        if x is None or y is None:
            continue
        source = str(row.get("source", "") or "")
        source_weight = float(SOURCE_WEIGHTS.get(source, 0.35))
        conf = _finite_float(row.get("conf"), 0.5) or 0.5
        conf = _clamp(conf, 0.05, 1.0)
        weight = source_weight * max(0.25, conf)
        obs.append(
            Observation(
                frame=int(row.get("frame", idx)),
                t_sec=float(idx) / float(fps),
                x=float(x),
                y=float(y),
                source=source,
                source_weight=source_weight,
                conf=float(conf),
                weight=float(weight),
                interpolated=bool(row.get("interpolated", False)),
                player_boxes=_normalize_player_boxes(row.get("player_boxes")),
                court_keypoints=_frame_court_keypoints(row),
            )
        )
    return obs


def calibrate_camera_from_court(
    court_keypoints: Optional[Sequence[float]],
    width: int,
    height: int,
    focal_px: Optional[float] = None,
) -> Optional[CameraModel]:
    """Estimate camera pose from detected court corners using solvePnPRansac."""
    kps = _valid_keypoints(court_keypoints)
    if kps is None:
        return None
    cv2 = _import_cv2()
    # Existing code uses keypoints 0,3,4,7 as TL, TR, BL, BR.
    index_to_world = {
        0: (0.0, 0.0, 0.0),
        3: (COURT_WIDTH_M, 0.0, 0.0),
        4: (0.0, COURT_LENGTH_M, 0.0),
        7: (COURT_WIDTH_M, COURT_LENGTH_M, 0.0),
    }
    object_points: List[Tuple[float, float, float]] = []
    image_points: List[Tuple[float, float]] = []
    for idx, world in index_to_world.items():
        bi = idx * 2
        if bi + 1 >= len(kps):
            continue
        x, y = float(kps[bi]), float(kps[bi + 1])
        if abs(x) <= 1e-6 and abs(y) <= 1e-6:
            continue
        object_points.append(world)
        image_points.append((x, y))
    if len(object_points) < 4:
        return None

    obj = np.asarray(object_points, dtype=np.float32)
    img = np.asarray(image_points, dtype=np.float32)
    f = float(focal_px) if focal_px else 1.2 * float(max(width, height))
    K = np.asarray(
        [[f, 0.0, float(width) / 2.0], [0.0, f, float(height) / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist = np.zeros((4, 1), dtype=np.float64)
    try:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj,
            img,
            K,
            dist,
            iterationsCount=100,
            reprojectionError=8.0,
            confidence=0.99,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    except Exception:
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        inliers = None
    if not ok:
        return None
    projected, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    projected = projected.reshape(-1, 2)
    errors = np.linalg.norm(projected - img, axis=1)
    inlier_count = int(len(object_points) if inliers is None else len(inliers))
    return CameraModel(
        K=K,
        rvec=np.asarray(rvec, dtype=np.float64).reshape(3, 1),
        tvec=np.asarray(tvec, dtype=np.float64).reshape(3, 1),
        reprojection_error_px=float(np.mean(errors)) if len(errors) else float("inf"),
        inlier_count=inlier_count,
        image_point_count=len(object_points),
        court_keypoints=list(kps),
    )


def _rotation_matrix(camera: CameraModel) -> np.ndarray:
    cv2 = _import_cv2()
    R, _ = cv2.Rodrigues(camera.rvec)
    return np.asarray(R, dtype=np.float64)


def project_points(camera: CameraModel, points_3d: np.ndarray) -> np.ndarray:
    cv2 = _import_cv2()
    points = np.asarray(points_3d, dtype=np.float64).reshape(-1, 3)
    projected, _ = cv2.projectPoints(
        points,
        camera.rvec,
        camera.tvec,
        camera.K,
        np.zeros((4, 1), dtype=np.float64),
    )
    return projected.reshape(-1, 2)


def _ray_intersect_z(camera: CameraModel, x: float, y: float, z_plane: float) -> Optional[np.ndarray]:
    R = _rotation_matrix(camera)
    inv_K = np.linalg.inv(camera.K)
    pixel = np.asarray([float(x), float(y), 1.0], dtype=np.float64)
    ray_cam = inv_K @ pixel
    ray_world = R.T @ ray_cam
    norm = np.linalg.norm(ray_world)
    if norm <= 1e-9:
        return None
    ray_world = ray_world / norm
    cam_center = -(R.T @ camera.tvec.reshape(3))
    if abs(ray_world[2]) <= 1e-9:
        return None
    lam = (float(z_plane) - float(cam_center[2])) / float(ray_world[2])
    if lam <= 0:
        return None
    return cam_center + lam * ray_world


def _ballistic_points(params: Sequence[float], rel_t: np.ndarray, gravity: float = GRAVITY_MPS2) -> np.ndarray:
    p = np.asarray(params, dtype=np.float64)
    t = np.asarray(rel_t, dtype=np.float64)
    pts = np.empty((len(t), 3), dtype=np.float64)
    pts[:, 0] = p[0] + p[3] * t
    pts[:, 1] = p[1] + p[4] * t
    pts[:, 2] = p[2] + p[5] * t - 0.5 * gravity * t * t
    return pts


def fit_ballistic_arc(
    observations: Sequence[Observation],
    camera: Optional[CameraModel],
    width: int,
    height: int,
    start_frame: int,
    end_frame: int,
) -> ArcFit:
    """Fit one ballistic 3D arc to observations by normalized reprojection loss."""
    if camera is None:
        return ArcFit(start_frame, end_frame, len(observations), None, None, None, False, "no_camera")
    if least_squares is None:
        return ArcFit(start_frame, end_frame, len(observations), None, None, None, False, "no_scipy")
    obs = [o for o in observations if start_frame <= o.frame <= end_frame]
    if len(obs) < 3:
        return ArcFit(start_frame, end_frame, len(obs), None, None, None, False, "too_few_observations")

    diag = math.hypot(float(width), float(height))
    t0 = obs[0].t_sec
    rel_t = np.asarray([o.t_sec - t0 for o in obs], dtype=np.float64)
    xy_obs = np.asarray([[o.x, o.y] for o in obs], dtype=np.float64)
    weights = np.asarray([max(0.05, o.weight) for o in obs], dtype=np.float64)

    p0 = _ray_intersect_z(camera, obs[0].x, obs[0].y, 1.0)
    p1 = _ray_intersect_z(camera, obs[-1].x, obs[-1].y, 1.0)
    if p0 is None:
        p0 = np.asarray([COURT_WIDTH_M / 2.0, COURT_LENGTH_M / 2.0, 1.0], dtype=np.float64)
    if p1 is None:
        p1 = p0.copy()
    dt = max(float(rel_t[-1]), 1.0 / 60.0)
    v0 = (p1 - p0) / dt
    v0[2] = (p1[2] - p0[2] + 0.5 * GRAVITY_MPS2 * dt * dt) / dt
    x0 = np.asarray([p0[0], p0[1], max(BALL_RADIUS_M, p0[2]), v0[0], v0[1], v0[2]], dtype=np.float64)

    lower = np.asarray([-COURT_WIDTH_M, -COURT_LENGTH_M, BALL_RADIUS_M, -70.0, -70.0, -45.0])
    upper = np.asarray([2 * COURT_WIDTH_M, 2 * COURT_LENGTH_M, 12.0, 70.0, 70.0, 45.0])
    x0 = np.minimum(np.maximum(x0, lower + 1e-6), upper - 1e-6)

    def residuals(params: np.ndarray) -> np.ndarray:
        pts = _ballistic_points(params, rel_t)
        proj = project_points(camera, pts)
        diff = (proj - xy_obs) / max(diag, 1.0)
        diff *= np.sqrt(weights)[:, None]
        # Softly discourage below-ground interiors without forbidding spin/drag-like arcs.
        below = np.minimum(0.0, pts[:, 2] - BALL_RADIUS_M)
        below_res = below * 6.0
        return np.concatenate([diff.reshape(-1), below_res])

    try:
        res = least_squares(
            residuals,
            x0,
            bounds=(lower, upper),
            loss="huber",
            f_scale=0.003,
            max_nfev=160,
        )
    except Exception as exc:
        return ArcFit(start_frame, end_frame, len(obs), None, None, None, False, f"fit_error:{exc}")

    pts = _ballistic_points(res.x, rel_t)
    proj = project_points(camera, pts)
    errors = np.linalg.norm(proj - xy_obs, axis=1)
    return ArcFit(
        start_frame=int(start_frame),
        end_frame=int(end_frame),
        observation_count=len(obs),
        params=np.asarray(res.x, dtype=np.float64),
        mean_reprojection_px=float(np.mean(errors)) if len(errors) else None,
        max_reprojection_px=float(np.max(errors)) if len(errors) else None,
        success=bool(res.success),
        message=str(res.message),
    )


def _near_any_player(x: float, y: float, boxes: Sequence[Sequence[float]], margin: float = 90.0) -> Tuple[bool, float]:
    best = float("inf")
    hit = False
    for box in boxes or []:
        if len(box) < 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in box[:4]]
        dx = max(x1 - x, 0.0, x - x2)
        dy = max(y1 - y, 0.0, y - y2)
        dist = math.hypot(dx, dy)
        best = min(best, dist)
        if x1 - margin <= x <= x2 + margin and y1 - margin <= y <= y2 + margin:
            hit = True
    return hit, best


def _local_sources(observations: Sequence[Observation], center_idx: int, radius: int = 1) -> List[str]:
    lo = max(0, center_idx - radius)
    hi = min(len(observations), center_idx + radius + 1)
    return [observations[i].source for i in range(lo, hi)]


def generate_event_candidates(
    observations: Sequence[Observation],
    segment: SegmentWindow,
    width: int,
    height: int,
    fps: float,
) -> List[Dict[str, Any]]:
    """Generate conservative bounce/hit/serve candidates from weighted 2D evidence."""
    if len(observations) < 3:
        return []
    diag = math.hypot(float(width), float(height))
    min_vel = max(80.0, 0.0015 * diag * float(fps))
    edge_frames = max(8, int(round(0.25 * fps)))
    events: List[Dict[str, Any]] = []

    for i in range(1, len(observations) - 1):
        a, b, c = observations[i - 1], observations[i], observations[i + 1]
        dt1 = b.t_sec - a.t_sec
        dt2 = c.t_sec - b.t_sec
        if dt1 <= 0 or dt2 <= 0 or dt1 > 0.5 or dt2 > 0.5:
            continue
        vx1, vy1 = (b.x - a.x) / dt1, (b.y - a.y) / dt1
        vx2, vy2 = (c.x - b.x) / dt2, (c.y - b.y) / dt2
        speed1, speed2 = math.hypot(vx1, vy1), math.hypot(vx2, vy2)
        dot = vx1 * vx2 + vy1 * vy2
        denom = max(speed1 * speed2, 1e-6)
        cos_turn = _clamp(dot / denom, -1.0, 1.0)
        edge_event = (
            b.frame - segment.start_frame <= edge_frames
            or segment.end_frame - b.frame <= edge_frames
            or i < 2
            or len(observations) - i <= 2
        )
        local_sources = _local_sources(observations, i, radius=1)
        carry_only = all(src in SOFT_SOURCES for src in local_sources)

        if vy1 > min_vel * 0.55 and vy2 < -min_vel * 0.35 and speed1 > min_vel:
            turn_strength = _clamp((abs(vy1) + abs(vy2)) / max(2.0 * min_vel, 1.0), 0.0, 1.0)
            source_support = max(SOURCE_WEIGHTS.get(src, 0.2) for src in local_sources)
            confidence = 0.25 + 0.45 * turn_strength + 0.20 * source_support
            if carry_only:
                confidence = min(confidence, 0.44)
            if edge_event:
                confidence = min(confidence, 0.58)
            ev_type = "bounce" if confidence >= 0.65 and not carry_only and not edge_event else "bounce_candidate"
            events.append(
                {
                    "type": ev_type,
                    "frame": int(b.frame),
                    "time_sec": float(b.t_sec),
                    "x": float(b.x),
                    "y": float(b.y),
                    "confidence": float(_clamp(confidence, 0.0, 0.95)),
                    "edge_event": bool(edge_event),
                    "carry_only": bool(carry_only),
                    "evidence": (
                        "carry_only_v_shape"
                        if carry_only
                        else ("partial_pre_or_post_arc" if edge_event else "down_up_direction_change")
                    ),
                    "metrics": {
                        "vy_before_px_s": float(vy1),
                        "vy_after_px_s": float(vy2),
                        "cos_turn": float(cos_turn),
                    },
                }
            )

        near_player, player_dist = _near_any_player(b.x, b.y, b.player_boxes, margin=100.0)
        speed_delta = abs(speed2 - speed1)
        if near_player and (cos_turn < -0.25 or speed_delta > min_vel * 0.9):
            direction_score = _clamp((-cos_turn + 0.25) / 1.25, 0.0, 1.0)
            speed_score = _clamp(speed_delta / max(min_vel * 2.0, 1.0), 0.0, 1.0)
            confidence = 0.30 + 0.30 * direction_score + 0.25 * speed_score
            if edge_event:
                confidence = min(confidence, 0.60)
            ev_type = "hit" if confidence >= 0.65 and not edge_event else "hit_candidate"
            events.append(
                {
                    "type": ev_type,
                    "frame": int(b.frame),
                    "time_sec": float(b.t_sec),
                    "x": float(b.x),
                    "y": float(b.y),
                    "confidence": float(_clamp(confidence, 0.0, 0.95)),
                    "edge_event": bool(edge_event),
                    "evidence": "velocity_reset_near_player",
                    "metrics": {
                        "speed_delta_px_s": float(speed_delta),
                        "cos_turn": float(cos_turn),
                        "player_distance_px": float(player_dist),
                    },
                }
            )

        near_segment_start = (b.frame - segment.start_frame) <= int(round(1.5 * fps))
        if near_segment_start and vy1 < -min_vel * 0.35 and vy2 > min_vel * 0.35:
            confidence = 0.35 + 0.30 * _clamp((abs(vy1) + abs(vy2)) / (2.0 * min_vel), 0.0, 1.0)
            events.append(
                {
                    "type": "serve_toss_apex",
                    "frame": int(b.frame),
                    "time_sec": float(b.t_sec),
                    "x": float(b.x),
                    "y": float(b.y),
                    "confidence": float(_clamp(confidence, 0.0, 0.80)),
                    "edge_event": bool(edge_event),
                    "evidence": "near_segment_start_up_to_down_motion",
                    "metrics": {
                        "vy_before_px_s": float(vy1),
                        "vy_after_px_s": float(vy2),
                    },
                }
            )

    # Deduplicate events that are nearly the same frame/type.
    events.sort(key=lambda e: (int(e["frame"]), -float(e.get("confidence", 0.0))))
    deduped: List[Dict[str, Any]] = []
    for ev in events:
        if deduped:
            prev = deduped[-1]
            same_family = str(prev["type"]).split("_")[0] == str(ev["type"]).split("_")[0]
            if same_family and abs(int(ev["frame"]) - int(prev["frame"])) <= 3:
                if float(ev.get("confidence", 0.0)) > float(prev.get("confidence", 0.0)):
                    deduped[-1] = ev
                continue
        deduped.append(ev)
    return deduped


def _event_split_frames(events: Sequence[Dict[str, Any]], segment: SegmentWindow) -> List[int]:
    split_types = {"bounce", "bounce_candidate", "hit", "hit_candidate"}
    out = []
    for ev in events:
        if ev.get("type") not in split_types:
            continue
        if bool(ev.get("edge_event", False)):
            continue
        if float(ev.get("confidence", 0.0)) < 0.45:
            continue
        frame = int(ev.get("frame", -1))
        if segment.start_frame + 2 <= frame <= segment.end_frame - 2:
            out.append(frame)
    return sorted(set(out))


def _arc_state_for_frame(
    arc: ArcFit,
    camera: Optional[CameraModel],
    frame: int,
    fps: float,
    width: int,
    height: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    if not arc.success or arc.params is None or camera is None:
        return None, None, "no_fit"
    rel_t = np.asarray([(frame - arc.start_frame) / float(fps)], dtype=np.float64)
    point = _ballistic_points(arc.params, rel_t)[0]
    try:
        proj = project_points(camera, point.reshape(1, 3))[0]
    except Exception:
        return point, None, "projection_failed"
    margin = 6.0
    outside = (
        proj[0] < -margin
        or proj[0] > float(width) + margin
        or proj[1] < -margin
        or proj[1] > float(height) + margin
    )
    return point, proj, "outside_frame" if outside else "inside_frame"


def _observation_map(observations: Sequence[Observation]) -> Dict[int, Observation]:
    best: Dict[int, Observation] = {}
    for obs in observations:
        prev = best.get(obs.frame)
        if prev is None or obs.weight > prev.weight:
            best[obs.frame] = obs
    return best


def classify_gaps_and_build_frames(
    segment: SegmentWindow,
    observations: Sequence[Observation],
    arcs: Sequence[ArcFit],
    camera: Optional[CameraModel],
    fps: float,
    width: int,
    height: int,
    max_hidden_gap_sec: float = 1.5,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Build frame rows and gap events from fitted arcs and observations."""
    obs_by_frame = _observation_map(observations)
    arc_by_frame: Dict[int, ArcFit] = {}
    for arc in arcs:
        for frame in range(max(segment.start_frame, arc.start_frame), min(segment.end_frame, arc.end_frame) + 1):
            arc_by_frame.setdefault(frame, arc)

    gap_events: List[Dict[str, Any]] = []
    sorted_obs = sorted(observations, key=lambda o: o.frame)
    max_hidden_frames = int(round(max_hidden_gap_sec * fps))
    for a, b in zip(sorted_obs, sorted_obs[1:]):
        gap = b.frame - a.frame - 1
        if gap <= 0:
            continue
        gap_start, gap_end = a.frame + 1, b.frame - 1
        sample_frames = range(gap_start, gap_end + 1)
        outside = 0
        inside = 0
        for frame in sample_frames:
            arc = arc_by_frame.get(frame)
            _, _, state = _arc_state_for_frame(arc, camera, frame, fps, width, height) if arc else (None, None, "no_fit")
            if state == "outside_frame":
                outside += 1
            elif state == "inside_frame":
                inside += 1
        if gap <= max_hidden_frames and outside >= max(1, inside):
            gap_events.append(
                {
                    "type": "inferred_gap",
                    "start_frame": int(gap_start),
                    "end_frame": int(gap_end),
                    "confidence": float(_clamp(0.35 + 0.35 * outside / max(1, gap), 0.0, 0.75)),
                    "evidence": "projected_arc_outside_frame",
                    "metrics": {"gap_frames": int(gap), "outside_frames": int(outside), "inside_frames": int(inside)},
                }
            )
        elif gap > max_hidden_frames:
            gap_events.append(
                {
                    "type": "ball_lost",
                    "start_frame": int(gap_start),
                    "end_frame": int(gap_end),
                    "confidence": 0.65,
                    "evidence": "gap_exceeds_hidden_bridge_window",
                    "metrics": {"gap_frames": int(gap), "max_hidden_frames": int(max_hidden_frames)},
                }
            )
        elif inside > outside and gap >= max(3, int(round(0.15 * fps))):
            gap_events.append(
                {
                    "type": "visible_missing_gap",
                    "start_frame": int(gap_start),
                    "end_frame": int(gap_end),
                    "confidence": 0.40,
                    "evidence": "projected_arc_visible_without_observation",
                    "metrics": {"gap_frames": int(gap), "outside_frames": int(outside), "inside_frames": int(inside)},
                }
            )

    frames_out: List[Dict[str, Any]] = []
    for frame in range(segment.start_frame, segment.end_frame + 1):
        obs = obs_by_frame.get(frame)
        arc = arc_by_frame.get(frame)
        point, proj, state = _arc_state_for_frame(arc, camera, frame, fps, width, height) if arc else (None, None, "no_fit")
        if obs is not None:
            row_state = "observed"
        elif state == "outside_frame":
            row_state = "inferred_gap"
        elif state == "inside_frame":
            row_state = "fitted_gap"
        else:
            row_state = "no_fit"

        residual_px = None
        if obs is not None and proj is not None:
            residual_px = float(math.hypot(float(proj[0]) - obs.x, float(proj[1]) - obs.y))
        shadow = None
        if point is not None:
            shadow = [float(point[0]), float(point[1]), 0.0]
        frames_out.append(
            {
                "frame": int(frame),
                "state": row_state,
                "observed": None
                if obs is None
                else {
                    "x": float(obs.x),
                    "y": float(obs.y),
                    "source": obs.source,
                    "weight": float(obs.weight),
                    "conf": float(obs.conf),
                },
                "xyz_m": None if point is None else [float(v) for v in point],
                "projected_2d": None if proj is None else [float(proj[0]), float(proj[1])],
                "court_shadow_m": shadow,
                "residual_px": residual_px,
            }
        )
    return frames_out, gap_events


def reconstruct_tracking(
    tracking: Dict[str, Any],
    cuts: Optional[Sequence[LLCutSegment]] = None,
    cuts_metadata: Optional[Dict[str, Any]] = None,
    timebase: str = "auto",
    max_hidden_gap_sec: float = 1.5,
) -> Dict[str, Any]:
    """Run 3D/event reconstruction over a loaded tracking JSON payload."""
    video = tracking.get("video") or {}
    fps = float(video.get("fps") or 0.0)
    width = int(video.get("width") or 0)
    height = int(video.get("height") or 0)
    total_frames = int(video.get("total_frames") or len(tracking.get("frames") or []))
    if fps <= 0 or width <= 0 or height <= 0:
        raise ValueError("tracking JSON must include video.fps, video.width, and video.height")
    frames = tracking.get("frames") or []
    video_duration = total_frames / fps if total_frames else 0.0

    if cuts is None:
        cuts = []
    windows, mapping = map_cuts_to_video_windows(cuts, fps, total_frames, video_duration, timebase=timebase)
    fallback_kps = tracking.get("last_valid_court_keypoints")

    out_segments: List[Dict[str, Any]] = []
    for segment in windows:
        observations = extract_observations(frames, segment.start_frame, segment.end_frame, fps)
        court_kps = choose_segment_court_keypoints(frames, segment.start_frame, segment.end_frame, fallback=fallback_kps)
        camera = calibrate_camera_from_court(court_kps, width, height) if court_kps is not None else None
        events = generate_event_candidates(observations, segment, width, height, fps)

        split_frames = _event_split_frames(events, segment)
        spans: List[Tuple[int, int]] = []
        cursor = segment.start_frame
        for split in split_frames:
            spans.append((cursor, split))
            cursor = split
        spans.append((cursor, segment.end_frame))

        arcs: List[ArcFit] = []
        for start, end in spans:
            span_obs = [o for o in observations if start <= o.frame <= end]
            if len(span_obs) < 3:
                arcs.append(ArcFit(start, end, len(span_obs), None, None, None, False, "too_few_observations"))
                continue
            arcs.append(fit_ballistic_arc(span_obs, camera, width, height, start, end))

        frame_rows, gap_events = classify_gaps_and_build_frames(
            segment,
            observations,
            arcs,
            camera,
            fps,
            width,
            height,
            max_hidden_gap_sec=max_hidden_gap_sec,
        )
        events.extend(gap_events)
        events.sort(key=lambda e: int(e.get("frame", e.get("start_frame", segment.start_frame))))

        fit_errors = [a.mean_reprojection_px for a in arcs if a.mean_reprojection_px is not None]
        out_segments.append(
            {
                "segment_id": int(segment.segment_id),
                "source_start_sec": float(segment.source_start_sec),
                "source_end_sec": float(segment.source_end_sec),
                "merged_start_sec": float(segment.merged_start_sec),
                "merged_end_sec": float(segment.merged_end_sec),
                "start_frame": int(segment.start_frame),
                "end_frame": int(segment.end_frame),
                "mapping_confidence": float(segment.mapping_confidence),
                "timebase": segment.timebase,
                "warnings": list(segment.warnings),
                "observation_count": int(len(observations)),
                "calibration": None if camera is None else camera.to_json(),
                "fit_summary": {
                    "arc_count": int(len(arcs)),
                    "successful_arc_count": int(sum(1 for a in arcs if a.success)),
                    "mean_reprojection_px": None if not fit_errors else float(np.mean(fit_errors)),
                    "max_mean_reprojection_px": None if not fit_errors else float(np.max(fit_errors)),
                    "arcs": [a.to_json() for a in arcs],
                },
                "events": _json_safe(events),
                "frames": _json_safe(frame_rows),
            }
        )

    all_events = []
    for segment in out_segments:
        for event in segment["events"]:
            merged = dict(event)
            merged["segment_id"] = segment["segment_id"]
            all_events.append(merged)

    return {
        "schema_version": "trajectory3d_v1",
        "research_basis": [
            "TT3D physics + reprojection fitting",
            "MonoTrack player/court hit context",
            "OpenCV solvePnPRansac court pose",
            "LosslessCut JSON5 .llc cut segments",
        ],
        "video": _json_safe(video),
        "cut_metadata": _json_safe(cuts_metadata or {}),
        "cut_mapping": _json_safe(mapping),
        "parameters": {
            "court_width_m": COURT_WIDTH_M,
            "court_length_m": COURT_LENGTH_M,
            "ball_radius_m": BALL_RADIUS_M,
            "gravity_mps2": GRAVITY_MPS2,
            "source_weights": dict(SOURCE_WEIGHTS),
            "max_hidden_gap_sec": float(max_hidden_gap_sec),
        },
        "summary": {
            "segment_count": int(len(out_segments)),
            "event_count": int(len(all_events)),
            "events_by_type": {
                typ: sum(1 for e in all_events if e.get("type") == typ)
                for typ in sorted({str(e.get("type")) for e in all_events})
            },
        },
        "events": _json_safe(all_events),
        "segments": _json_safe(out_segments),
    }


def reconstruct_from_files(
    tracking_json: str | Path,
    cuts_path: Optional[str | Path],
    output_json: str | Path,
    timebase: str = "auto",
    max_hidden_gap_sec: float = 1.5,
) -> Dict[str, Any]:
    tracking = load_tracking_json(tracking_json)
    cuts: List[LLCutSegment] = []
    cuts_metadata: Dict[str, Any] = {}
    if cuts_path:
        cuts, cuts_metadata = load_llc_cuts(cuts_path)
    result = reconstruct_tracking(
        tracking,
        cuts=cuts,
        cuts_metadata=cuts_metadata,
        timebase=timebase,
        max_hidden_gap_sec=max_hidden_gap_sec,
    )
    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def _draw_mini_court_overlay(
    frame,
    ball_xy_m: Optional[Tuple[float, float]],
    ball_z_m: Optional[float],
    trail_xy_m: List[Tuple[float, float]],
    court_w: float,
    court_l: float,
    z_max: float = 5.0,
):
    """Overlay a top-down mini court + vertical height bar in the top-right corner.

    World coords: X in [0, court_w] (sideline to sideline), Y in [0, court_l] (baseline to
    baseline, net at Y = court_l/2). Z is height in meters.
    """
    cv2 = _import_cv2()
    H, W = frame.shape[:2]

    panel_w = 200
    panel_h = 360
    margin = 18
    pad = 14
    bar_w = 18
    bar_gap = 10

    panel_x1 = W - panel_w - margin
    panel_y1 = margin
    panel_x2 = panel_x1 + panel_w
    panel_y2 = panel_y1 + panel_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x1, panel_y1), (panel_x2, panel_y2), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.rectangle(frame, (panel_x1, panel_y1), (panel_x2, panel_y2), (200, 200, 200), 1)

    court_x1 = panel_x1 + pad
    court_y1 = panel_y1 + pad
    court_x2 = panel_x2 - pad - bar_gap - bar_w - pad
    court_y2 = panel_y2 - pad - 30  # leave room for z text

    avail_w = court_x2 - court_x1
    avail_h = court_y2 - court_y1
    scale = min(avail_w / court_w, avail_h / court_l)
    draw_w = court_w * scale
    draw_h = court_l * scale
    cx = (court_x1 + court_x2) / 2.0
    cy = (court_y1 + court_y2) / 2.0
    ox = cx - draw_w / 2.0
    oy = cy - draw_h / 2.0

    def world_to_panel(xm: float, ym: float) -> Tuple[int, int]:
        px = ox + xm * scale
        py = oy + ym * scale
        return int(round(px)), int(round(py))

    # Court outlines
    p_tl = world_to_panel(0.0, 0.0)
    p_tr = world_to_panel(court_w, 0.0)
    p_bl = world_to_panel(0.0, court_l)
    p_br = world_to_panel(court_w, court_l)
    cv2.rectangle(frame, p_tl, p_br, (255, 255, 255), 1)

    # Singles sidelines (doubles court is 10.97m wide, singles is 8.23m → 1.37m inset each side)
    if court_w > 8.23 + 0.1:
        inset = (court_w - 8.23) / 2.0
        s_tl = world_to_panel(inset, 0.0)
        s_bl = world_to_panel(inset, court_l)
        s_tr = world_to_panel(court_w - inset, 0.0)
        s_br = world_to_panel(court_w - inset, court_l)
        cv2.line(frame, s_tl, s_bl, (200, 200, 200), 1)
        cv2.line(frame, s_tr, s_br, (200, 200, 200), 1)

    # Net
    net_l = world_to_panel(0.0, court_l / 2.0)
    net_r = world_to_panel(court_w, court_l / 2.0)
    cv2.line(frame, net_l, net_r, (0, 200, 255), 1)

    # Service line: 6.4m from net on each side
    svc_top = court_l / 2.0 - 6.4
    svc_bot = court_l / 2.0 + 6.4
    if svc_top > 0:
        cv2.line(frame, world_to_panel(0.0, svc_top), world_to_panel(court_w, svc_top), (160, 160, 160), 1)
    if svc_bot < court_l:
        cv2.line(frame, world_to_panel(0.0, svc_bot), world_to_panel(court_w, svc_bot), (160, 160, 160), 1)
    # Center service line between service lines
    cv2.line(frame, world_to_panel(court_w / 2.0, svc_top), world_to_panel(court_w / 2.0, svc_bot), (160, 160, 160), 1)

    # Trail
    pts = [world_to_panel(x, y) for x, y in trail_xy_m]
    for p0, p1 in zip(pts, pts[1:]):
        cv2.line(frame, p0, p1, (80, 180, 255), 1, cv2.LINE_AA)

    # Current ball position
    if ball_xy_m is not None:
        bx, by = world_to_panel(ball_xy_m[0], ball_xy_m[1])
        cv2.circle(frame, (bx, by), 4, (0, 0, 255), -1)
        cv2.circle(frame, (bx, by), 5, (255, 255, 255), 1)

    # Height bar
    bar_x1 = court_x2 + bar_gap
    bar_y1 = court_y1
    bar_x2 = bar_x1 + bar_w
    bar_y2 = court_y2
    cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (180, 180, 180), 1)
    # Tick marks at 1m intervals
    bar_h = bar_y2 - bar_y1
    for zt in range(0, int(z_max) + 1):
        ty = bar_y2 - int(round(bar_h * (zt / z_max)))
        cv2.line(frame, (bar_x1 - 3, ty), (bar_x1, ty), (180, 180, 180), 1)
        cv2.putText(frame, str(zt), (bar_x1 - 14, ty + 4), cv2.FONT_HERSHEY_PLAIN, 0.7, (180, 180, 180), 1)
    # Net height reference (0.914m at center) — dashed-ish
    net_z_y = bar_y2 - int(round(bar_h * (0.914 / z_max)))
    cv2.line(frame, (bar_x1, net_z_y), (bar_x2, net_z_y), (0, 200, 255), 1)

    if ball_z_m is not None:
        z_clamped = max(0.0, min(z_max, float(ball_z_m)))
        fill_y = bar_y2 - int(round(bar_h * (z_clamped / z_max)))
        cv2.rectangle(frame, (bar_x1 + 1, fill_y), (bar_x2 - 1, bar_y2 - 1), (0, 180, 0), -1)
        cv2.putText(
            frame,
            f"z = {ball_z_m:0.2f} m",
            (court_x1, panel_y2 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            frame,
            "z = --",
            (court_x1, panel_y2 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (160, 160, 160),
            1,
            cv2.LINE_AA,
        )


def render_debug_video(
    tracking_json: str | Path,
    trajectory_json: str | Path,
    output_video: str | Path,
    input_video: Optional[str | Path] = None,
) -> None:
    """Render the input video with a top-down mini-court + height bar overlay.

    Reads `xyz_m` (3D world meters) from the trajectory JSON and draws:
      - the observed ball position on the main frame (small green dot),
      - a top-down mini court in the top-right with the ball's ground position + recent trail,
      - a vertical height bar showing the ball's z (height) in meters.
    """
    cv2 = _import_cv2()
    tracking = load_tracking_json(tracking_json)
    with Path(trajectory_json).open("r", encoding="utf-8") as f:
        traj = json.load(f)

    video_info = tracking.get("video") or {}
    source = Path(input_video) if input_video else Path(str(video_info.get("input", "")))
    if not source.exists():
        raise FileNotFoundError(f"Input video for rendering not found: {source}")
    fps = float(video_info.get("fps") or 30.0)
    width = int(video_info.get("width") or 0)
    height = int(video_info.get("height") or 0)

    params = traj.get("parameters") or {}
    court_w = float(params.get("court_width_m") or COURT_WIDTH_M)
    court_l = float(params.get("court_length_m") or COURT_LENGTH_M)

    frame_rows: Dict[int, Dict[str, Any]] = {}
    for segment in traj.get("segments", []):
        for row in segment.get("frames", []):
            frame_rows[int(row["frame"])] = row

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {source}")
    if width <= 0:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    if height <= 0:
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = Path(output_video)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    frame_idx = 0
    trail_xy_m: List[Tuple[float, float]] = []
    trail_len = max(8, int(round(fps * 1.5)))  # ~1.5 sec of ground trail
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ball_xy_m: Optional[Tuple[float, float]] = None
        ball_z_m: Optional[float] = None
        row = frame_rows.get(frame_idx)
        if row:
            obs = row.get("observed")
            if obs:
                cv2.circle(frame, (int(round(obs["x"])), int(round(obs["y"]))), 4, (0, 255, 0), -1)
            xyz = row.get("xyz_m")
            if xyz and len(xyz) >= 3:
                ball_xy_m = (float(xyz[0]), float(xyz[1]))
                ball_z_m = float(xyz[2])
                trail_xy_m.append(ball_xy_m)
                if len(trail_xy_m) > trail_len:
                    trail_xy_m = trail_xy_m[-trail_len:]
        _draw_mini_court_overlay(
            frame,
            ball_xy_m,
            ball_z_m,
            trail_xy_m,
            court_w=court_w,
            court_l=court_l,
        )
        writer.write(frame)
        frame_idx += 1
    cap.release()
    writer.release()
