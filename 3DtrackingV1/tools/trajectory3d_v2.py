"""Hybrid single-camera 3D tennis-ball reconstruction sidecar.

This is the replacement path for the old v1 3D sidecar.  It keeps the useful
file parsing and debug-video compatibility from v1, but changes the estimator:

- calibrate camera pose from the full 14-point court template when available;
- expose calibrated ray features for every observation;
- fit competing toss and lateral-flight hypotheses instead of one generic arc;
- report ambiguity and QC flags instead of pretending short monocular arcs are
  fully observable.

Coordinate convention in this repo:
    x = court width, left doubles sideline to right doubles sideline, meters
    y = court length, far baseline to near baseline, meters
    z = height above court, meters
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.optimize import least_squares
except Exception:  # pragma: no cover - exercised by CLI error path.
    least_squares = None

from trajectory3d_v1 import (
    BALL_RADIUS_M,
    COURT_LENGTH_M,
    COURT_WIDTH_M,
    GRAVITY_MPS2,
    LLCutSegment,
    Observation,
    SegmentWindow,
    _clamp,
    _finite_float,
    _import_cv2,
    _json_safe,
    choose_segment_court_keypoints,
    extract_observations,
    generate_event_candidates,
    load_llc_cuts,
    load_tracking_json,
    map_cuts_to_video_windows,
)


SINGLES_WIDTH_M = 8.23
SINGLES_INSET_M = (COURT_WIDTH_M - SINGLES_WIDTH_M) / 2.0
NET_Y_M = COURT_LENGTH_M / 2.0
SERVICE_FROM_NET_M = 6.40
FAR_SERVICE_Y_M = NET_Y_M - SERVICE_FROM_NET_M
NEAR_SERVICE_Y_M = NET_Y_M + SERVICE_FROM_NET_M

SOURCE_SIGMA_PX = {
    "det": 6.0,
    "motion": 10.0,
    "guide": 18.0,
    "carry": 24.0,
    "interp": 30.0,
}


COURT_TEMPLATE_14_M: Dict[int, Tuple[float, float, float]] = {
    # Far/top baseline: doubles-L, singles-L, singles-R, doubles-R
    0: (0.0, 0.0, 0.0),
    4: (SINGLES_INSET_M, 0.0, 0.0),
    6: (COURT_WIDTH_M - SINGLES_INSET_M, 0.0, 0.0),
    1: (COURT_WIDTH_M, 0.0, 0.0),
    # Near/bottom baseline
    2: (0.0, COURT_LENGTH_M, 0.0),
    5: (SINGLES_INSET_M, COURT_LENGTH_M, 0.0),
    7: (COURT_WIDTH_M - SINGLES_INSET_M, COURT_LENGTH_M, 0.0),
    3: (COURT_WIDTH_M, COURT_LENGTH_M, 0.0),
    # Far service line: singles-L, center T, singles-R
    8: (SINGLES_INSET_M, FAR_SERVICE_Y_M, 0.0),
    12: (COURT_WIDTH_M / 2.0, FAR_SERVICE_Y_M, 0.0),
    9: (COURT_WIDTH_M - SINGLES_INSET_M, FAR_SERVICE_Y_M, 0.0),
    # Near service line
    10: (SINGLES_INSET_M, NEAR_SERVICE_Y_M, 0.0),
    13: (COURT_WIDTH_M / 2.0, NEAR_SERVICE_Y_M, 0.0),
    11: (COURT_WIDTH_M - SINGLES_INSET_M, NEAR_SERVICE_Y_M, 0.0),
}

COURT_LINE_PAIRS_14 = [
    (0, 4), (4, 6), (6, 1),
    (2, 5), (5, 7), (7, 3),
    (0, 2), (1, 3),
    (4, 8), (8, 10), (10, 5),
    (6, 9), (9, 11), (11, 7),
    (8, 12), (12, 9),
    (10, 13), (13, 11),
    (12, 13),
]


@dataclass
class CameraCalibrationV2:
    K: np.ndarray
    dist: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray
    reprojection_error_px: float
    max_reprojection_error_px: float
    inlier_count: int
    image_point_count: int
    template_point_count: int
    quality: str
    warnings: List[str] = field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        return {
            "K": self.K.tolist(),
            "dist": self.dist.reshape(-1).tolist(),
            "rvec": self.rvec.reshape(-1).tolist(),
            "tvec": self.tvec.reshape(-1).tolist(),
            "reprojection_error_px": float(self.reprojection_error_px),
            "max_reprojection_error_px": float(self.max_reprojection_error_px),
            "inlier_count": int(self.inlier_count),
            "image_point_count": int(self.image_point_count),
            "template_point_count": int(self.template_point_count),
            "quality": self.quality,
            "warnings": list(self.warnings),
        }


@dataclass
class RayFeature:
    frame: int
    pixel_xy: Tuple[float, float]
    ground_xy_m: Optional[Tuple[float, float]]
    z18_xy_m: Optional[Tuple[float, float]]
    net_plane_xyz_m: Optional[Tuple[float, float, float]]
    vertical_jacobian_px_per_m: Optional[Tuple[float, float]]

    def to_json(self) -> Dict[str, Any]:
        return {
            "frame": int(self.frame),
            "pixel_xy": [float(self.pixel_xy[0]), float(self.pixel_xy[1])],
            "ground_xy_m": None if self.ground_xy_m is None else [float(v) for v in self.ground_xy_m],
            "z18_xy_m": None if self.z18_xy_m is None else [float(v) for v in self.z18_xy_m],
            "net_plane_xyz_m": None if self.net_plane_xyz_m is None else [float(v) for v in self.net_plane_xyz_m],
            "vertical_jacobian_px_per_m": (
                None
                if self.vertical_jacobian_px_per_m is None
                else [float(v) for v in self.vertical_jacobian_px_per_m]
            ),
        }


@dataclass
class HypothesisFit:
    name: str
    start_frame: int
    end_frame: int
    observation_count: int
    params: Optional[np.ndarray]
    score: float
    image_rms_sigma: Optional[float]
    mean_reprojection_px: Optional[float]
    max_reprojection_px: Optional[float]
    success: bool
    message: str
    qc_flags: List[str] = field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "start_frame": int(self.start_frame),
            "end_frame": int(self.end_frame),
            "observation_count": int(self.observation_count),
            "params": None if self.params is None else self.params.tolist(),
            "score": None if not math.isfinite(self.score) else float(self.score),
            "image_rms_sigma": self.image_rms_sigma,
            "mean_reprojection_px": self.mean_reprojection_px,
            "max_reprojection_px": self.max_reprojection_px,
            "success": bool(self.success),
            "message": self.message,
            "qc_flags": list(self.qc_flags),
        }


@dataclass
class SpanReconstruction:
    start_frame: int
    end_frame: int
    selected_name: Optional[str]
    ambiguity: str
    score_gap: Optional[float]
    confidence: float
    fits: List[HypothesisFit]
    context: Dict[str, Any]
    qc_flags: List[str] = field(default_factory=list)

    def selected_fit(self) -> Optional[HypothesisFit]:
        for fit in self.fits:
            if fit.name == self.selected_name:
                return fit
        return None

    def to_json(self) -> Dict[str, Any]:
        return {
            "start_frame": int(self.start_frame),
            "end_frame": int(self.end_frame),
            "selected_hypothesis": self.selected_name,
            "ambiguity": self.ambiguity,
            "score_gap": self.score_gap,
            "confidence": float(self.confidence),
            "context": _json_safe(self.context),
            "qc_flags": list(self.qc_flags),
            "fits": [f.to_json() for f in self.fits],
        }


def _valid_keypoints(kps: Any) -> Optional[List[float]]:
    if not isinstance(kps, (list, tuple)) or len(kps) < 8:
        return None
    vals: List[float] = []
    for v in kps:
        fv = _finite_float(v)
        if fv is None:
            return None
        vals.append(float(fv))
    valid_pairs = 0
    for i in range(0, len(vals) - 1, 2):
        if abs(vals[i]) > 1e-6 or abs(vals[i + 1]) > 1e-6:
            valid_pairs += 1
    return vals if valid_pairs >= 4 else None


def court_object_image_points(
    court_keypoints: Optional[Sequence[float]],
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    kps = _valid_keypoints(court_keypoints)
    object_points: List[Tuple[float, float, float]] = []
    image_points: List[Tuple[float, float]] = []
    indices: List[int] = []
    if kps is None:
        return (
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 2), dtype=np.float64),
            indices,
        )
    for idx, world in sorted(COURT_TEMPLATE_14_M.items()):
        bi = idx * 2
        if bi + 1 >= len(kps):
            continue
        x = float(kps[bi])
        y = float(kps[bi + 1])
        if abs(x) <= 1e-6 and abs(y) <= 1e-6:
            continue
        object_points.append(world)
        image_points.append((x, y))
        indices.append(idx)
    return (
        np.asarray(object_points, dtype=np.float64),
        np.asarray(image_points, dtype=np.float64),
        indices,
    )


def load_camera_config(camera_json: Optional[str | Path]) -> Dict[str, Any]:
    if not camera_json:
        return {}
    path = Path(camera_json)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("--camera-json must contain a JSON object")
    return data


def _camera_matrix_from_config(
    width: int,
    height: int,
    focal_px: Optional[float] = None,
    camera_config: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    warnings: List[str] = []
    cfg = camera_config or {}
    if "K" in cfg:
        K = np.asarray(cfg["K"], dtype=np.float64).reshape(3, 3)
    else:
        f = _finite_float(cfg.get("focal_px"), None)
        if f is None:
            f = float(focal_px) if focal_px else 1.2 * float(max(width, height))
            warnings.append(
                "No calibrated intrinsics supplied; using a guessed focal length. "
                "Metric height/depth will be provisional."
            )
        cx = _finite_float(cfg.get("cx"), float(width) / 2.0)
        cy = _finite_float(cfg.get("cy"), float(height) / 2.0)
        K = np.asarray([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist_raw = cfg.get("dist") or cfg.get("dist_coeffs") or cfg.get("distCoeffs")
    if dist_raw is None:
        dist = np.zeros((5, 1), dtype=np.float64)
    else:
        dist = np.asarray(dist_raw, dtype=np.float64).reshape(-1, 1)
    return K, dist, warnings


def project_points(camera: CameraCalibrationV2, points_3d: np.ndarray) -> np.ndarray:
    cv2 = _import_cv2()
    points = np.asarray(points_3d, dtype=np.float64).reshape(-1, 3)
    projected, _ = cv2.projectPoints(points, camera.rvec, camera.tvec, camera.K, camera.dist)
    return projected.reshape(-1, 2)


def calibrate_camera_from_court_v2(
    court_keypoints: Optional[Sequence[float]],
    width: int,
    height: int,
    focal_px: Optional[float] = None,
    camera_config: Optional[Dict[str, Any]] = None,
) -> Optional[CameraCalibrationV2]:
    """Estimate camera pose from all visible semantic court keypoints."""
    cv2 = _import_cv2()
    obj, img, indices = court_object_image_points(court_keypoints)
    if len(obj) < 4:
        return None

    K, dist, warnings = _camera_matrix_from_config(width, height, focal_px, camera_config)
    obj32 = obj.astype(np.float32)
    img32 = img.astype(np.float32)
    candidates: List[Tuple[float, np.ndarray, np.ndarray, Optional[np.ndarray], str]] = []

    flags = []
    if hasattr(cv2, "SOLVEPNP_IPPE"):
        flags.append(("ippe", cv2.SOLVEPNP_IPPE))
    flags.append(("iterative", cv2.SOLVEPNP_ITERATIVE))

    for name, flag in flags:
        try:
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                obj32,
                img32,
                K,
                dist,
                iterationsCount=240,
                reprojectionError=6.0,
                confidence=0.995,
                flags=flag,
            )
        except Exception:
            ok = False
            rvec = None
            tvec = None
            inliers = None
        if not ok:
            try:
                ok, rvec, tvec = cv2.solvePnP(obj32, img32, K, dist, flags=flag)
                inliers = None
            except Exception:
                continue
        try:
            if hasattr(cv2, "solvePnPRefineLM") and len(obj) >= 4:
                rvec, tvec = cv2.solvePnPRefineLM(obj32, img32, K, dist, rvec, tvec)
        except Exception:
            pass
        projected, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        projected = projected.reshape(-1, 2)
        errors = np.linalg.norm(projected - img, axis=1)
        candidates.append((float(np.mean(errors)), np.asarray(rvec), np.asarray(tvec), inliers, name))

    if not candidates:
        return None

    candidates.sort(key=lambda c: c[0])
    mean_error, rvec, tvec, inliers, method = candidates[0]
    projected, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    projected = projected.reshape(-1, 2)
    errors = np.linalg.norm(projected - img, axis=1)
    inlier_count = int(len(obj) if inliers is None else len(inliers))

    quality = "good"
    if len(obj) < 8:
        quality = "weak"
        warnings.append("Court calibration used fewer than 8 semantic points.")
    if mean_error > 10.0:
        quality = "weak"
        warnings.append("Court reprojection error is high; 3D output should be treated as low confidence.")
    if mean_error > 18.0 or inlier_count < 4:
        quality = "bad"
        warnings.append("Court calibration failed quality gates; use v2 output for diagnostics only.")
    if method == "iterative" and len(obj) == 4:
        warnings.append("Planar PnP used only four points; focal/depth ambiguity remains large.")

    return CameraCalibrationV2(
        K=K,
        dist=dist,
        rvec=np.asarray(rvec, dtype=np.float64).reshape(3, 1),
        tvec=np.asarray(tvec, dtype=np.float64).reshape(3, 1),
        reprojection_error_px=float(mean_error),
        max_reprojection_error_px=float(np.max(errors)) if len(errors) else float("inf"),
        inlier_count=inlier_count,
        image_point_count=len(obj),
        template_point_count=len(COURT_TEMPLATE_14_M),
        quality=quality,
        warnings=warnings,
    )


def _rotation_matrix(camera: CameraCalibrationV2) -> np.ndarray:
    cv2 = _import_cv2()
    R, _ = cv2.Rodrigues(camera.rvec)
    return np.asarray(R, dtype=np.float64)


def camera_center_world(camera: CameraCalibrationV2) -> np.ndarray:
    R = _rotation_matrix(camera)
    return -(R.T @ camera.tvec.reshape(3))


def image_ray_world(camera: CameraCalibrationV2, x: float, y: float) -> Tuple[np.ndarray, np.ndarray]:
    cv2 = _import_cv2()
    pixel = np.asarray([[[float(x), float(y)]]], dtype=np.float64)
    undist = cv2.undistortPoints(pixel, camera.K, camera.dist)
    ray_cam = np.asarray([undist[0, 0, 0], undist[0, 0, 1], 1.0], dtype=np.float64)
    R = _rotation_matrix(camera)
    direction = R.T @ ray_cam
    norm = np.linalg.norm(direction)
    if norm <= 1e-12:
        direction = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        direction = direction / norm
    return camera_center_world(camera), direction


def ray_intersect_z(
    camera: CameraCalibrationV2,
    x: float,
    y: float,
    z_plane: float,
) -> Optional[np.ndarray]:
    origin, direction = image_ray_world(camera, x, y)
    if abs(direction[2]) <= 1e-10:
        return None
    lam = (float(z_plane) - float(origin[2])) / float(direction[2])
    if lam <= 0:
        return None
    return origin + lam * direction


def ray_intersect_y(
    camera: CameraCalibrationV2,
    x: float,
    y: float,
    y_plane: float,
) -> Optional[np.ndarray]:
    origin, direction = image_ray_world(camera, x, y)
    if abs(direction[1]) <= 1e-10:
        return None
    lam = (float(y_plane) - float(origin[1])) / float(direction[1])
    if lam <= 0:
        return None
    return origin + lam * direction


def ray_features_for_observation(camera: CameraCalibrationV2, obs: Observation) -> RayFeature:
    ground = ray_intersect_z(camera, obs.x, obs.y, 0.0)
    z18 = ray_intersect_z(camera, obs.x, obs.y, 1.8)
    net = ray_intersect_y(camera, obs.x, obs.y, NET_Y_M)
    vertical_j = None
    anchor = z18 if z18 is not None else ground
    if anchor is not None:
        try:
            base = project_points(camera, anchor.reshape(1, 3))[0]
            up = anchor.copy()
            up[2] += 1.0
            lifted = project_points(camera, up.reshape(1, 3))[0]
            vertical_j = (float(lifted[0] - base[0]), float(lifted[1] - base[1]))
        except Exception:
            vertical_j = None
    return RayFeature(
        frame=obs.frame,
        pixel_xy=(obs.x, obs.y),
        ground_xy_m=None if ground is None else (float(ground[0]), float(ground[1])),
        z18_xy_m=None if z18 is None else (float(z18[0]), float(z18[1])),
        net_plane_xyz_m=None if net is None else (float(net[0]), float(net[1]), float(net[2])),
        vertical_jacobian_px_per_m=vertical_j,
    )


def _ballistic_points(params: Sequence[float], rel_t: np.ndarray) -> np.ndarray:
    p = np.asarray(params, dtype=np.float64)
    t = np.asarray(rel_t, dtype=np.float64)
    pts = np.empty((len(t), 3), dtype=np.float64)
    pts[:, 0] = p[0] + p[3] * t
    pts[:, 1] = p[1] + p[4] * t
    pts[:, 2] = p[2] + p[5] * t - 0.5 * GRAVITY_MPS2 * t * t
    return pts


def _soft_lower(value: float, lower: float, sigma: float) -> float:
    return min(0.0, float(value) - float(lower)) / max(float(sigma), 1e-9)


def _soft_upper(value: float, upper: float, sigma: float) -> float:
    return max(0.0, float(value) - float(upper)) / max(float(sigma), 1e-9)


def _obs_sigmas(observations: Sequence[Observation]) -> np.ndarray:
    sigmas = []
    for obs in observations:
        base = SOURCE_SIGMA_PX.get(obs.source, 14.0)
        conf = _clamp(float(obs.conf), 0.05, 1.0)
        sigmas.append(base / math.sqrt(conf))
    return np.asarray(sigmas, dtype=np.float64)


def _initial_guesses(
    observations: Sequence[Observation],
    camera: CameraCalibrationV2,
    rel_t: np.ndarray,
    hypothesis: str,
) -> List[np.ndarray]:
    first = observations[0]
    last = observations[-1]
    dt = max(float(rel_t[-1]), 1.0 / 60.0)
    guesses: List[np.ndarray] = []

    if hypothesis == "toss":
        start_heights = [1.2, 1.7, 2.2]
        start_vz = [3.0, 5.5, 7.5, -2.0]
        for z0 in start_heights:
            p0 = ray_intersect_z(camera, first.x, first.y, z0)
            if p0 is None:
                continue
            for vz in start_vz:
                guesses.append(np.asarray([p0[0], p0[1], p0[2], 0.0, 0.0, vz], dtype=np.float64))
    else:
        heights = [BALL_RADIUS_M, 0.8, 1.5, 2.5]
        for z0 in heights:
            p0 = ray_intersect_z(camera, first.x, first.y, z0)
            if p0 is None:
                continue
            for z1 in heights:
                p1 = ray_intersect_z(camera, last.x, last.y, z1)
                if p1 is None:
                    continue
                v = (p1 - p0) / dt
                v[2] = (p1[2] - p0[2] + 0.5 * GRAVITY_MPS2 * dt * dt) / dt
                guesses.append(np.asarray([p0[0], p0[1], p0[2], v[0], v[1], v[2]], dtype=np.float64))

    if not guesses:
        fallback = ray_intersect_z(camera, first.x, first.y, 1.5)
        if fallback is None:
            fallback = np.asarray([COURT_WIDTH_M / 2.0, COURT_LENGTH_M / 2.0, 1.5], dtype=np.float64)
        guesses.append(np.asarray([fallback[0], fallback[1], fallback[2], 0.0, 0.0, 2.0], dtype=np.float64))
    return guesses


def _fit_qc_flags(params: np.ndarray, points: np.ndarray, mean_reproj: float) -> List[str]:
    flags: List[str] = []
    if np.any(points[:, 2] < -0.03):
        flags.append("below_court")
    if float(np.max(points[:, 2])) > 8.0:
        flags.append("very_high_arc")
    if np.linalg.norm(params[3:6]) > 65.0:
        flags.append("implausible_speed")
    if mean_reproj > 25.0:
        flags.append("large_reprojection_error")
    if np.max(points[:, 0]) < -1.5 or np.min(points[:, 0]) > COURT_WIDTH_M + 1.5:
        flags.append("outside_court_width")
    if np.max(points[:, 1]) < -3.0 or np.min(points[:, 1]) > COURT_LENGTH_M + 3.0:
        flags.append("outside_court_length")
    return flags


def fit_hypothesis(
    observations: Sequence[Observation],
    camera: Optional[CameraCalibrationV2],
    width: int,
    height: int,
    start_frame: int,
    end_frame: int,
    hypothesis: str,
    context: Optional[Dict[str, Any]] = None,
) -> HypothesisFit:
    if camera is None:
        return HypothesisFit(hypothesis, start_frame, end_frame, 0, None, float("inf"), None, None, None, False, "no_camera")
    if least_squares is None:
        return HypothesisFit(hypothesis, start_frame, end_frame, 0, None, float("inf"), None, None, None, False, "no_scipy")

    obs = [o for o in observations if start_frame <= o.frame <= end_frame]
    if len(obs) < 4:
        return HypothesisFit(hypothesis, start_frame, end_frame, len(obs), None, float("inf"), None, None, None, False, "too_few_observations")

    t0 = obs[0].t_sec
    rel_t = np.asarray([o.t_sec - t0 for o in obs], dtype=np.float64)
    xy_obs = np.asarray([[o.x, o.y] for o in obs], dtype=np.float64)
    sigmas = _obs_sigmas(obs)
    ctx = context or {}
    near_segment_start = bool(ctx.get("near_segment_start", False))

    lower = np.asarray([-COURT_WIDTH_M, -4.0, BALL_RADIUS_M, -55.0, -70.0, -35.0], dtype=np.float64)
    upper = np.asarray([2.0 * COURT_WIDTH_M, COURT_LENGTH_M + 4.0, 12.0, 55.0, 70.0, 35.0], dtype=np.float64)

    def residuals(params: np.ndarray) -> np.ndarray:
        pts = _ballistic_points(params, rel_t)
        try:
            proj = project_points(camera, pts)
        except Exception:
            return np.ones(2 * len(obs) + 16, dtype=np.float64) * 1e3
        img_res = ((proj - xy_obs) / sigmas[:, None]).reshape(-1)

        prior_res: List[float] = []
        below = np.minimum(0.0, pts[:, 2] - BALL_RADIUS_M) / 0.08
        prior_res.extend(below.tolist())
        prior_res.append(_soft_upper(float(np.max(pts[:, 2])), 9.0, 0.6))

        speed = float(np.linalg.norm(params[3:6]))
        prior_res.append(_soft_upper(speed, 62.0, 8.0))

        if hypothesis == "toss":
            vxy = math.hypot(float(params[3]), float(params[4]))
            displacement_xy = float(np.linalg.norm(pts[-1, :2] - pts[0, :2]))
            prior_res.append(vxy / 1.8)
            prior_res.append(displacement_xy / 1.4)
            prior_res.append(_soft_lower(float(params[2]), 0.9, 0.35))
            prior_res.append(_soft_upper(float(params[2]), 2.8, 0.45))
            prior_res.append(_soft_lower(float(np.max(pts[:, 2])), 1.8, 0.5))
            prior_res.append(_soft_upper(float(np.max(pts[:, 2])), 5.5, 0.8))
            if near_segment_start:
                prior_res.append(min(0.0, float(params[5])) / 2.5)
        elif hypothesis == "lateral_flight":
            vxy = math.hypot(float(params[3]), float(params[4]))
            displacement_xy = float(np.linalg.norm(pts[-1, :2] - pts[0, :2]))
            prior_res.append(_soft_lower(vxy, 2.5, 2.0))
            prior_res.append(_soft_lower(displacement_xy, 0.7, 0.7))
            prior_res.append(_soft_upper(float(params[2]), 4.0, 1.5))

        return np.concatenate([img_res, np.asarray(prior_res, dtype=np.float64)])

    best = None
    best_cost = float("inf")
    best_msg = ""
    for guess in _initial_guesses(obs, camera, rel_t, hypothesis):
        x0 = np.minimum(np.maximum(guess, lower + 1e-6), upper - 1e-6)
        try:
            res = least_squares(
                residuals,
                x0,
                bounds=(lower, upper),
                loss="soft_l1",
                f_scale=1.0,
                max_nfev=260,
            )
        except Exception as exc:
            best_msg = f"fit_error:{exc}"
            continue
        cost = float(2.0 * res.cost / max(1, len(obs)))
        if cost < best_cost:
            best = res
            best_cost = cost
            best_msg = str(res.message)

    if best is None:
        return HypothesisFit(hypothesis, start_frame, end_frame, len(obs), None, float("inf"), None, None, None, False, best_msg or "fit_failed")

    pts = _ballistic_points(best.x, rel_t)
    proj = project_points(camera, pts)
    px_errors = np.linalg.norm(proj - xy_obs, axis=1)
    image_norm = ((proj - xy_obs) / sigmas[:, None]).reshape(-1)
    image_rms = math.sqrt(float(np.mean(image_norm * image_norm))) if len(image_norm) else None
    mean_reproj = float(np.mean(px_errors)) if len(px_errors) else float("inf")
    qc = _fit_qc_flags(np.asarray(best.x), pts, mean_reproj)
    return HypothesisFit(
        name=hypothesis,
        start_frame=int(start_frame),
        end_frame=int(end_frame),
        observation_count=len(obs),
        params=np.asarray(best.x, dtype=np.float64),
        score=float(best_cost),
        image_rms_sigma=image_rms,
        mean_reprojection_px=mean_reproj,
        max_reprojection_px=float(np.max(px_errors)) if len(px_errors) else None,
        success=bool(best.success),
        message=best_msg,
        qc_flags=qc,
    )


def _near_any_player(x: float, y: float, observations: Sequence[Observation], margin: float = 110.0) -> bool:
    for obs in observations:
        for box in obs.player_boxes or []:
            if len(box) < 4:
                continue
            x1, y1, x2, y2 = [float(v) for v in box[:4]]
            if x1 - margin <= x <= x2 + margin and y1 - margin <= y <= y2 + margin:
                return True
    return False


def span_context(
    observations: Sequence[Observation],
    segment: SegmentWindow,
    start_frame: int,
    end_frame: int,
    events: Sequence[Dict[str, Any]],
    fps: float,
) -> Dict[str, Any]:
    obs = [o for o in observations if start_frame <= o.frame <= end_frame]
    near_segment_start = bool(start_frame - segment.start_frame <= max(6, int(round(1.2 * fps))))
    has_toss_apex = any(
        e.get("type") == "serve_toss_apex" and start_frame <= int(e.get("frame", -1)) <= end_frame
        for e in events
    )
    starts_near_player = False
    if obs:
        starts_near_player = _near_any_player(obs[0].x, obs[0].y, obs[: min(5, len(obs))])
    return {
        "near_segment_start": near_segment_start,
        "has_toss_apex_candidate": bool(has_toss_apex),
        "starts_near_player_box": bool(starts_near_player),
        "duration_sec": 0.0 if not obs else float(obs[-1].t_sec - obs[0].t_sec),
        "observation_count": int(len(obs)),
    }


def choose_hypothesis(fits: Sequence[HypothesisFit], context: Dict[str, Any]) -> Tuple[Optional[str], str, Optional[float], float, List[str]]:
    usable = [f for f in fits if f.success and f.params is not None and math.isfinite(f.score)]
    if not usable:
        return None, "none", None, 0.0, ["no_successful_hypothesis"]
    usable.sort(key=lambda f: f.score)
    best = usable[0]
    runner = usable[1] if len(usable) > 1 else None
    gap = None if runner is None else float(runner.score - best.score)
    ambiguity = "low"
    if gap is None:
        ambiguity = "medium"
    elif gap < 0.35:
        ambiguity = "high"
    elif gap < 0.9:
        ambiguity = "medium"

    serve_context = bool(
        context.get("near_segment_start")
        and (context.get("has_toss_apex_candidate") or context.get("starts_near_player_box"))
    )
    selected = best.name
    qc_flags: List[str] = []
    if best.name == "toss" and not serve_context:
        lateral = next((f for f in usable if f.name == "lateral_flight"), None)
        if lateral is not None and lateral.score <= best.score + 0.75:
            selected = "lateral_flight"
            ambiguity = "high" if ambiguity != "high" else ambiguity
            qc_flags.append("toss_was_best_without_serve_context")

    selected_fit = next((f for f in usable if f.name == selected), best)
    reproj = selected_fit.mean_reprojection_px or 40.0
    conf = 1.0
    conf -= _clamp(reproj / 45.0, 0.0, 0.7)
    if ambiguity == "high":
        conf -= 0.25
    elif ambiguity == "medium":
        conf -= 0.12
    if selected_fit.qc_flags:
        conf -= min(0.25, 0.08 * len(selected_fit.qc_flags))
    return selected, ambiguity, gap, _clamp(conf, 0.0, 1.0), qc_flags


def _event_split_frames(events: Sequence[Dict[str, Any]], segment: SegmentWindow) -> List[int]:
    split_types = {"bounce", "bounce_candidate", "hit", "hit_candidate"}
    out: List[int] = []
    for ev in events:
        if ev.get("type") not in split_types:
            continue
        if bool(ev.get("edge_event", False)):
            continue
        if float(ev.get("confidence", 0.0)) < 0.45:
            continue
        frame = int(ev.get("frame", -1))
        if segment.start_frame + 3 <= frame <= segment.end_frame - 3:
            out.append(frame)
    return sorted(set(out))


def reconstruct_span(
    observations: Sequence[Observation],
    camera: Optional[CameraCalibrationV2],
    width: int,
    height: int,
    segment: SegmentWindow,
    start_frame: int,
    end_frame: int,
    events: Sequence[Dict[str, Any]],
    fps: float,
) -> SpanReconstruction:
    context = span_context(observations, segment, start_frame, end_frame, events, fps)
    fits = [
        fit_hypothesis(observations, camera, width, height, start_frame, end_frame, "toss", context),
        fit_hypothesis(observations, camera, width, height, start_frame, end_frame, "lateral_flight", context),
    ]
    selected, ambiguity, gap, confidence, qc = choose_hypothesis(fits, context)
    return SpanReconstruction(
        start_frame=int(start_frame),
        end_frame=int(end_frame),
        selected_name=selected,
        ambiguity=ambiguity,
        score_gap=gap,
        confidence=confidence,
        fits=fits,
        context=context,
        qc_flags=qc,
    )


def _observation_map(observations: Sequence[Observation]) -> Dict[int, Observation]:
    best: Dict[int, Observation] = {}
    for obs in observations:
        prev = best.get(obs.frame)
        if prev is None or obs.weight > prev.weight:
            best[obs.frame] = obs
    return best


def _span_for_frame(spans: Sequence[SpanReconstruction], frame: int) -> Optional[SpanReconstruction]:
    for span in spans:
        if span.start_frame <= frame <= span.end_frame:
            return span
    return None


def build_frame_rows(
    segment: SegmentWindow,
    observations: Sequence[Observation],
    spans: Sequence[SpanReconstruction],
    camera: Optional[CameraCalibrationV2],
    fps: float,
) -> List[Dict[str, Any]]:
    obs_by_frame = _observation_map(observations)
    rows: List[Dict[str, Any]] = []
    for frame in range(segment.start_frame, segment.end_frame + 1):
        obs = obs_by_frame.get(frame)
        span = _span_for_frame(spans, frame)
        fit = span.selected_fit() if span is not None else None
        xyz = None
        proj = None
        residual = None
        if fit is not None and fit.params is not None and camera is not None:
            rel_t = np.asarray([(frame - fit.start_frame) / float(fps)], dtype=np.float64)
            point = _ballistic_points(fit.params, rel_t)[0]
            xyz = [float(v) for v in point]
            try:
                p2 = project_points(camera, point.reshape(1, 3))[0]
                proj = [float(p2[0]), float(p2[1])]
                if obs is not None:
                    residual = float(math.hypot(p2[0] - obs.x, p2[1] - obs.y))
            except Exception:
                proj = None
        state = "observed" if obs is not None else ("fitted_gap" if xyz is not None else "no_fit")
        rows.append(
            {
                "frame": int(frame),
                "state": state,
                "observed": None
                if obs is None
                else {
                    "x": float(obs.x),
                    "y": float(obs.y),
                    "source": obs.source,
                    "weight": float(obs.weight),
                    "conf": float(obs.conf),
                },
                "xyz_m": xyz,
                "projected_2d": proj,
                "court_shadow_m": None if xyz is None else [float(xyz[0]), float(xyz[1]), 0.0],
                "residual_px": residual,
                "selected_hypothesis": None if span is None else span.selected_name,
                "ambiguity": None if span is None else span.ambiguity,
                "confidence": None if span is None else float(span.confidence),
            }
        )
    return rows


def reconstruct_tracking_v2(
    tracking: Dict[str, Any],
    cuts: Optional[Sequence[LLCutSegment]] = None,
    cuts_metadata: Optional[Dict[str, Any]] = None,
    timebase: str = "auto",
    focal_px: Optional[float] = None,
    camera_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    video = tracking.get("video") or {}
    fps = float(video.get("fps") or 0.0)
    width = int(video.get("width") or 0)
    height = int(video.get("height") or 0)
    total_frames = int(video.get("total_frames") or len(tracking.get("frames") or []))
    if fps <= 0 or width <= 0 or height <= 0:
        raise ValueError("tracking JSON must include video.fps, video.width, and video.height")

    frames = tracking.get("frames") or []
    video_duration = total_frames / fps if total_frames else 0.0
    cuts = [] if cuts is None else list(cuts)
    windows, mapping = map_cuts_to_video_windows(cuts, fps, total_frames, video_duration, timebase=timebase)
    fallback_kps = tracking.get("last_valid_court_keypoints")

    out_segments: List[Dict[str, Any]] = []
    all_qc_flags: List[str] = []
    for segment in windows:
        observations = extract_observations(frames, segment.start_frame, segment.end_frame, fps)
        court_kps = choose_segment_court_keypoints(frames, segment.start_frame, segment.end_frame, fallback=fallback_kps)
        camera = calibrate_camera_from_court_v2(court_kps, width, height, focal_px=focal_px, camera_config=camera_config)
        if camera is None:
            all_qc_flags.append("no_camera_calibration")
        elif camera.quality != "good":
            all_qc_flags.append(f"camera_calibration_{camera.quality}")

        events = generate_event_candidates(observations, segment, width, height, fps)
        split_frames = _event_split_frames(events, segment)
        spans_bounds: List[Tuple[int, int]] = []
        cursor = segment.start_frame
        for split in split_frames:
            spans_bounds.append((cursor, split))
            cursor = split
        spans_bounds.append((cursor, segment.end_frame))

        spans: List[SpanReconstruction] = []
        for start, end in spans_bounds:
            spans.append(reconstruct_span(observations, camera, width, height, segment, start, end, events, fps))

        ray_features = []
        if camera is not None:
            ray_features = [ray_features_for_observation(camera, obs).to_json() for obs in observations]

        frame_rows = build_frame_rows(segment, observations, spans, camera, fps)
        span_errors = [
            fit.mean_reprojection_px
            for span in spans
            for fit in span.fits
            if fit.name == span.selected_name and fit.mean_reprojection_px is not None
        ]

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
                "events": _json_safe(events),
                "ray_features": ray_features,
                "fit_summary": {
                    "span_count": int(len(spans)),
                    "successful_span_count": int(sum(1 for s in spans if s.selected_name is not None)),
                    "mean_selected_reprojection_px": None if not span_errors else float(np.mean(span_errors)),
                    "spans": [s.to_json() for s in spans],
                },
                "frames": _json_safe(frame_rows),
            }
        )

    all_events = []
    for segment in out_segments:
        for event in segment["events"]:
            merged = dict(event)
            merged["segment_id"] = segment["segment_id"]
            all_events.append(merged)

    ambiguity_counts: Dict[str, int] = {}
    hypothesis_counts: Dict[str, int] = {}
    for segment in out_segments:
        for span in segment["fit_summary"]["spans"]:
            ambiguity = str(span.get("ambiguity"))
            ambiguity_counts[ambiguity] = ambiguity_counts.get(ambiguity, 0) + 1
            hyp = str(span.get("selected_hypothesis"))
            hypothesis_counts[hyp] = hypothesis_counts.get(hyp, 0) + 1

    return {
        "schema_version": "trajectory3d_v2",
        "coordinate_system": {
            "x": "court width in meters, left doubles sideline to right doubles sideline",
            "y": "court length in meters, far baseline to near baseline",
            "z": "height above court in meters",
        },
        "video": _json_safe(video),
        "cut_metadata": _json_safe(cuts_metadata or {}),
        "cut_mapping": _json_safe(mapping),
        "parameters": {
            "court_width_m": COURT_WIDTH_M,
            "court_length_m": COURT_LENGTH_M,
            "ball_radius_m": BALL_RADIUS_M,
            "gravity_mps2": GRAVITY_MPS2,
            "singles_inset_m": SINGLES_INSET_M,
            "net_y_m": NET_Y_M,
            "source_sigma_px": dict(SOURCE_SIGMA_PX),
        },
        "summary": {
            "segment_count": int(len(out_segments)),
            "event_count": int(len(all_events)),
            "events_by_type": {
                typ: sum(1 for e in all_events if e.get("type") == typ)
                for typ in sorted({str(e.get("type")) for e in all_events})
            },
            "ambiguity_counts": dict(sorted(ambiguity_counts.items())),
            "selected_hypothesis_counts": dict(sorted(hypothesis_counts.items())),
            "qc_flags": sorted(set(all_qc_flags)),
        },
        "events": _json_safe(all_events),
        "segments": _json_safe(out_segments),
        "next_required_work": [
            "Supply calibrated intrinsics/distortion in --camera-json; guessed focal length is not enough for metric 3D.",
            "Add line-segment court refinement after keypoint PnP; v2 currently uses point reprojection only.",
            "Label toss/hit/bounce events and train a small temporal event model; heuristics still only propose candidates.",
            "Export pose keypoints or a racket proxy, not just player boxes, to make toss release/contact priors real.",
            "Validate against synchronized multi-camera or mocap ground truth before trusting absolute centimeter errors.",
        ],
    }


def reconstruct_from_files_v2(
    tracking_json: str | Path,
    cuts_path: Optional[str | Path],
    output_json: str | Path,
    timebase: str = "auto",
    focal_px: Optional[float] = None,
    camera_json: Optional[str | Path] = None,
) -> Dict[str, Any]:
    tracking = load_tracking_json(tracking_json)
    cuts: List[LLCutSegment] = []
    cuts_metadata: Dict[str, Any] = {}
    if cuts_path:
        cuts, cuts_metadata = load_llc_cuts(cuts_path)
    camera_config = load_camera_config(camera_json)
    result = reconstruct_tracking_v2(
        tracking,
        cuts=cuts,
        cuts_metadata=cuts_metadata,
        timebase=timebase,
        focal_px=focal_px,
        camera_config=camera_config,
    )
    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result
