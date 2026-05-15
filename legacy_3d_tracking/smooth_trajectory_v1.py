#!/usr/bin/env python3
"""Physics-constrained piecewise ballistic smoother for trajectory3d_v1 output.

Takes a trajectory3d_v1.json (the per-frame 3D ball estimate from
tools/reconstruct_3d_v1.py) and produces:

  1. <out>.json — per-frame smoothed ball state (x, y, z, vx, vy, vz, speed, status)
                  with a conservative contact event list.
  2. <out>.mp4  — optional re-render of the input video with a mini-court /
                  height-bar overlay driven by the smoothed positions.

Algorithm (PC-PBS — physics-constrained piecewise ballistic smoothing):

  Step 1. Robust outlier prefilter.
      Per-frame xyz_m is collected from segments with state in {observed, fitted_gap}
      (fitted_gap carries higher measurement noise). For each frame we compute the
      median-absolute-deviation of (x, y, z) over a 9-frame local window and reject
      points farther than k * 1.4826 * MAD (k = 5). This kills the multi-thousand-
      m/s^2 single-frame spikes the upstream reconstruction can emit while keeping
      the underlying ballistic structure intact.

  Step 2. Track identification.
      Contiguous runs with observation density above threshold form tracks. Gaps
      shorter than max_short_gap_sec (default 0.35 s) are tolerated within a track;
      anything longer splits the track. Frames outside any track are marked LOST
      and never interpolated through.

  Step 3. RTS smoother per track.
      State x_k = [px, py, pz, vx, vy, vz]^T.
      Linear dynamics with gravity as a deterministic control offset:
          F = [[I_3, dt I_3]; [0,   I_3   ]]
          g_vec = [0, 0, -0.5 g dt^2, 0, 0, -g dt]^T
          x_{k+1} = F x_k + g_vec + w_k
      Process noise via the standard white-noise-acceleration form:
          Q_axis = sigma_a^2 [[dt^3/3, dt^2/2]; [dt^2/2, dt]]
      Measurement model: H picks (x, y, z); R = diag(sigma_m^2) with sigma_m
      depending on the source state.
      Forward pass: standard Kalman filter with a Mahalanobis chi^2 outlier gate
      (3 d.o.f., 99.93%) so any residual outliers get skipped.
      Backward pass: Rauch-Tung-Striebel (1965) smoother.

  Step 4. Conservative contact detection.
      Over the smoothed velocity series we compute
          delta = (v_{k+1} - v_{k-1}) - (0, 0, -2 g dt)
      i.e. the velocity change beyond what gravity alone would explain over a
      2-frame window. A contact is registered when |delta| > 4 m/s AND the previous
      contact is at least 120 ms earlier. Bounces vs hits are distinguished after
      the fact via the z-velocity sign flip near the ground.

  Step 5. Per-sub-segment re-smoothing.
      The track is split at detected contact frames and the RTS smoother is rerun
      independently on each sub-segment so the velocity discontinuities at hits
      and bounces are preserved rather than smoothed out.

References:
  Rauch, Tung & Striebel (1965), Maximum likelihood estimates of linear
    dynamic systems. AIAA Journal 3(8).
  Sarkka (2013), Bayesian Filtering and Smoothing. Cambridge UP, chapter 8.
  Gossard et al., TT3D (CVPRW 2025) — physics-constrained 3D ball recovery.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

GRAVITY_MPS2 = 9.81
COURT_WIDTH_M = 10.97
COURT_LENGTH_M = 23.77

# Typical assumed-fixed camera intrinsics for the Pomona clip (and similar
# fixed-camera tennis setups). The focal length is approximate but the 3D fit
# is robust to small K errors — what matters is consistency across frames.
DEFAULT_FOCAL_FACTOR = 1.2


# ---------- BLAS-free linear-algebra helpers --------------------------------
#
# On this Windows host the installed numpy build crashes inside MKL whenever
# `@` / `np.dot` / `np.linalg.inv` / `np.linalg.solve` get called on matrices
# larger than 2x2 (status 0xc06d007e, STATUS_PROCEDURE_NOT_FOUND). The crashes
# are reproducible from a one-line script. `np.einsum` and plain elementwise
# operations route through a different path and work fine, so we use them
# instead. The math is identical — these are just hand-coded substitutes.


def _mm(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Matrix-matrix product without touching BLAS."""
    return np.einsum("ij,jk->ik", A, B)


def _mv(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Matrix-vector product without touching BLAS."""
    return np.einsum("ij,j->i", A, x)


def _abat(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """A @ B @ A.T without touching BLAS."""
    return np.einsum("ij,jk,lk->il", A, B, A)


def _inv3(M: np.ndarray) -> np.ndarray:
    """Closed-form 3x3 inverse via cofactor expansion."""
    a = float(M[0, 0]); b = float(M[0, 1]); c = float(M[0, 2])
    d = float(M[1, 0]); e = float(M[1, 1]); f = float(M[1, 2])
    g = float(M[2, 0]); h = float(M[2, 1]); i = float(M[2, 2])
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if abs(det) < 1e-20:
        raise np.linalg.LinAlgError("singular 3x3 matrix")
    inv_det = 1.0 / det
    out = np.empty((3, 3), dtype=np.float64)
    out[0, 0] = (e * i - f * h) * inv_det
    out[0, 1] = (c * h - b * i) * inv_det
    out[0, 2] = (b * f - c * e) * inv_det
    out[1, 0] = (f * g - d * i) * inv_det
    out[1, 1] = (a * i - c * g) * inv_det
    out[1, 2] = (c * d - a * f) * inv_det
    out[2, 0] = (d * h - e * g) * inv_det
    out[2, 1] = (b * g - a * h) * inv_det
    out[2, 2] = (a * e - b * d) * inv_det
    return out


def _inv_gj(M: np.ndarray) -> np.ndarray:
    """General N x N inverse via Gauss-Jordan elimination with partial pivoting.
    Only uses elementwise numpy ops, so it never calls BLAS/LAPACK."""
    n = M.shape[0]
    A = np.concatenate([M.astype(np.float64, copy=True), np.eye(n, dtype=np.float64)], axis=1)
    for i in range(n):
        pivot = i
        max_val = abs(A[i, i])
        for r in range(i + 1, n):
            v = abs(A[r, i])
            if v > max_val:
                max_val = v
                pivot = r
        if max_val < 1e-20:
            raise np.linalg.LinAlgError("singular matrix")
        if pivot != i:
            tmp = A[i].copy()
            A[i] = A[pivot]
            A[pivot] = tmp
        piv_val = float(A[i, i])
        A[i] = A[i] / piv_val
        for r in range(n):
            if r == i:
                continue
            factor = float(A[r, i])
            if factor != 0.0:
                A[r] = A[r] - factor * A[i]
    return A[:, n:].copy()


def _vec_norm(v: np.ndarray) -> float:
    """Vector norm without BLAS."""
    s = 0.0
    for x in v.ravel():
        s += float(x) * float(x)
    return math.sqrt(s)


def _cross3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """3D cross product without BLAS."""
    return np.array([
        float(a[1]) * float(b[2]) - float(a[2]) * float(b[1]),
        float(a[2]) * float(b[0]) - float(a[0]) * float(b[2]),
        float(a[0]) * float(b[1]) - float(a[1]) * float(b[0]),
    ])


# ---------- Camera pose from ground-plane homography ------------------------
#
# Given a homography H mapping image pixels to court ground plane (z=0) and
# an assumed pinhole intrinsics K, we can recover the camera pose:
#   H = K * [r1, r2, t] * scale
# where r1, r2 are the first two columns of the camera rotation matrix
# (world -> camera frame), t is the translation, and scale is fixed by the
# orthonormality of R. This is the Faugeras-Zhang plane-homography
# decomposition (see Hartley & Zisserman, "Multiple View Geometry" §13).
# K is assumed (we don't have calibration); errors in K mostly translate
# into a depth-scaling bias which the 3D-fit residual absorbs as long as
# K is consistent across frames.

def _camera_pose_from_court_keypoints(
    keypoints: Sequence[float], image_w: int, image_h: int
) -> Optional[Dict[str, Any]]:
    """Recover (K, R, t, cam_center) using cv2.solvePnP on the 4 court
    corners. This is the standard tennis camera-pose pipeline and is far
    more numerically stable than decomposing the homography by hand.

    Assumed intrinsics: principal point at image center, focal = 1.2 * H,
    zero distortion. K errors translate mostly into a depth-scaling bias
    that the trajectory fit absorbs.
    """
    try:
        import cv2  # type: ignore
    except ImportError:
        return None
    if keypoints is None or len(keypoints) < 8:
        return None
    obj_pts: List[Tuple[float, float, float]] = []
    img_pts: List[Tuple[float, float]] = []
    for idx, world in COURT_CORNER_INDICES:
        bi = idx * 2
        if bi + 1 >= len(keypoints):
            continue
        u, v = float(keypoints[bi]), float(keypoints[bi + 1])
        if not (math.isfinite(u) and math.isfinite(v)):
            continue
        if abs(u) <= 1e-6 and abs(v) <= 1e-6:
            continue
        obj_pts.append((float(world[0]), float(world[1]), 0.0))
        img_pts.append((u, v))
    if len(obj_pts) < 4:
        return None
    obj = np.asarray(obj_pts, dtype=np.float64)
    img = np.asarray(img_pts, dtype=np.float64)
    f = DEFAULT_FOCAL_FACTOR * float(image_h)
    cx = float(image_w) / 2.0
    cy = float(image_h) / 2.0
    K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.zeros((4, 1), dtype=np.float64)
    try:
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    except Exception:
        return None
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    R = np.asarray(R, dtype=np.float64)
    t_vec = np.asarray(tvec, dtype=np.float64).reshape(3)
    # Camera center in world coordinates: C = -R^T * t. We can't trust the
    # blas-routed R.T @ t (could MKL-crash), so compute via _mv.
    cam_center = -_mv(R.T, t_vec)
    return {"K": K, "R": R, "t": t_vec, "cam_center": cam_center}


def _project_world_to_image(
    P_world: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray
) -> Optional[Tuple[float, float]]:
    """Pinhole projection of a 3D world point to image pixel."""
    P_cam = _mv(R, P_world) + t
    if P_cam[2] < 0.001:
        return None
    u = float(K[0, 0]) * float(P_cam[0]) / float(P_cam[2]) + float(K[0, 2])
    v = float(K[1, 1]) * float(P_cam[1]) / float(P_cam[2]) + float(K[1, 2])
    return u, v


def _fit_3d_trajectory(
    detections: List[Tuple[int, float, float]],
    contact_frames: List[int],
    pose: Dict[str, Any],
    dt_per_frame: float,
    track_start_frame: int,
    initial_xy: Tuple[float, float, float, float],
) -> Optional[Tuple[float, float, float, float, float, float, float]]:
    """Levenberg-Marquardt fit of a 3D parabolic ball trajectory to the
    pixel detections in a track, with hard Z=0 constraints at every detected
    contact frame.

    State: (X0, Y0, Z0, Vx, Vy, Vz). In this coordinate system positive Z is
    INTO the ground (solvePnP's convention given our court keypoint layout
    on the z=0 plane), so the gravity term in Z(t) is +0.5 g t^2:
        X(t) = X0 + Vx t
        Y(t) = Y0 + Vy t
        Z(t) = Z0 + Vz t + 0.5 g t^2
    "Above ground" means Z<0; gravity pulls Z from negative back up to zero.

    initial_xy = (X0_init, Y0_init, Vx_init, Vy_init) — derived upstream from
    the linear-interp anchors so the optimizer doesn't have to find the X,Y
    basin from scratch.

    Returns (X0, Y0, Z0, Vx, Vy, Vz, final_cost) or None on failure.
    """
    if not detections:
        return None
    try:
        from scipy.optimize import least_squares  # type: ignore
    except ImportError:
        return None

    K = pose["K"]
    R = pose["R"]
    t_vec = pose["t"]
    g = GRAVITY_MPS2

    X0i, Y0i, Vxi, Vyi = initial_xy
    # If there's a contact at the start, the ball is on the ground (Z0=0).
    # Otherwise assume the ball was already at a typical racket-strike height
    # of ~1.4 m above the ground (Z0 = -1.4 in this convention).
    Z0i = 0.0 if (contact_frames and contact_frames[0] == track_start_frame) else -1.4
    # If we have any contact, derive Vz so that Z(t_contact) = 0:
    #   0 = Z0 + Vz*tc + 0.5 g tc^2  ->  Vz = -(Z0 + 0.5 g tc^2) / tc
    Vzi = 0.0
    if contact_frames:
        tc = (contact_frames[0] - track_start_frame) * dt_per_frame
        if tc > 0.05:
            Vzi = -(Z0i + 0.5 * g * tc * tc) / tc

    contact_weight = 200.0

    def residuals(params: np.ndarray) -> np.ndarray:
        X0, Y0, Z0, Vx, Vy, Vz = (float(x) for x in params)
        res: List[float] = []
        for frame, u_obs, v_obs in detections:
            t_rel = (frame - track_start_frame) * dt_per_frame
            X = X0 + Vx * t_rel
            Y = Y0 + Vy * t_rel
            Z = Z0 + Vz * t_rel + 0.5 * g * t_rel * t_rel
            P = np.array([X, Y, Z])
            proj = _project_world_to_image(P, K, R, t_vec)
            if proj is None:
                res.append(2000.0)
                res.append(2000.0)
                continue
            res.append(u_obs - proj[0])
            res.append(v_obs - proj[1])
        for cf in contact_frames:
            t_rel = (cf - track_start_frame) * dt_per_frame
            Z = Z0 + Vz * t_rel + 0.5 * g * t_rel * t_rel
            res.append(contact_weight * Z)
        return np.asarray(res, dtype=np.float64)

    x0 = np.array([X0i, Y0i, Z0i, Vxi, Vyi, Vzi], dtype=np.float64)
    try:
        result = least_squares(
            residuals, x0, method="lm", max_nfev=200,
        )
    except Exception:
        return None
    if not result.success:
        return None
    X0, Y0, Z0, Vx, Vy, Vz = (float(p) for p in result.x)
    return X0, Y0, Z0, Vx, Vy, Vz, float(result.cost)


# Court keypoint indices for the four doubles-court corners, matching the
# upstream now_main_pkg/trajectory3d_v1.py convention:
#   0 = (0, 0),  3 = (W, 0),  4 = (0, L),  7 = (W, L)
COURT_CORNER_INDICES: Tuple[Tuple[int, Tuple[float, float]], ...] = (
    (0, (0.0, 0.0)),
    (3, (COURT_WIDTH_M, 0.0)),
    (4, (0.0, COURT_LENGTH_M)),
    (7, (COURT_WIDTH_M, COURT_LENGTH_M)),
)


@dataclass
class SmoothConfig:
    fps: float
    # The smoothed state is 4D (u, v, du, dv) in PIXEL coordinates. We smooth
    # in pixel space — not court space — because the ball's image is close to
    # a constant-velocity 2D trajectory while in flight, whereas its homography
    # ground projection bunches up at the arc apex (the projected speed drops
    # to ~zero as the ball rises and then snaps back as it falls). Pixel-space
    # smoothing also escapes per-frame court-keypoint jitter: that jitter only
    # affects the final projection, not the smoothing itself. The court-plane
    # (X, Y) is recomputed at output time by applying the per-frame homography
    # to the smoothed pixel position.
    sigma_accel_pxps2: float = 600.0
    # sigma_meas accounts for detection noise (~2–3 px) PLUS model error from
    # the constant-velocity approximation during pixel-space flight curvature.
    sigma_meas_pix: float = 8.0
    max_short_gap_sec: float = 0.35
    min_track_obs: int = 8
    # Permissive chi^2 gate. Real hits cause innovations in the tens of pixels
    # after gaps; we want those incorporated, not rejected. The MAD prefilter
    # already removes hard outliers — this gate only catches the truly absurd.
    outlier_chi2_2d: float = 200.0
    # Contact detection in pixel space: real hits jump >~8 px/frame between
    # consecutive frames at 60 fps (>~500 px/s), which is well above the
    # natural per-frame change during flight (~50–150 px/s of curvature).
    contact_speed_change_pxps: float = 500.0
    contact_min_sep_sec: float = 0.18
    mad_window_frames: int = 15
    mad_reject_k: float = 3.5


# ---------- Step 1: ingest from tracking.json --------------------------------

def _homography_image_to_court(keypoints: Sequence[float]) -> Optional[np.ndarray]:
    """Fit the 2D homography mapping image pixels to court-plane meters.

    Uses the four doubles-court corners (kp indices 0, 3, 4, 7 in the upstream
    schema) as the four point correspondences. Requires at least 4 finite corners.
    Returns the 3x3 H or None if the keypoints are degenerate.
    """
    if keypoints is None:
        return None
    kps = list(keypoints)
    if len(kps) < 8:
        return None
    image_pts: List[Tuple[float, float]] = []
    world_pts: List[Tuple[float, float]] = []
    for idx, world in COURT_CORNER_INDICES:
        bi = idx * 2
        if bi + 1 >= len(kps):
            continue
        u, v = float(kps[bi]), float(kps[bi + 1])
        if not (math.isfinite(u) and math.isfinite(v)):
            continue
        if abs(u) <= 1e-6 and abs(v) <= 1e-6:
            continue
        image_pts.append((u, v))
        world_pts.append(world)
    if len(image_pts) < 4:
        return None
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("OpenCV (cv2) is required for homography computation") from exc
    src = np.asarray(image_pts, dtype=np.float64)
    dst = np.asarray(world_pts, dtype=np.float64)
    H, _ = cv2.findHomography(src, dst, method=0)
    if H is None or not np.all(np.isfinite(H)):
        return None
    return H


def _project_image_to_court(H: np.ndarray, u: float, v: float) -> Optional[Tuple[float, float]]:
    """Apply a 3x3 homography to a single (u, v) pixel; return court-plane (X, Y) meters."""
    p = _mv(H, np.asarray([u, v, 1.0], dtype=np.float64))
    if abs(p[2]) < 1e-9:
        return None
    return float(p[0] / p[2]), float(p[1] / p[2])


def _player_foot_court(
    bbox: Sequence[float], H: Optional[np.ndarray]
) -> Optional[Tuple[float, float]]:
    """Foot position (bottom-center of bbox) projected to court meters."""
    if bbox is None or len(bbox) < 4 or H is None:
        return None
    try:
        x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    except (TypeError, ValueError):
        return None
    foot_u = (x1 + x2) / 2.0
    foot_v = y2
    return _project_image_to_court(H, foot_u, foot_v)


def _gather_observations_from_tracking(
    tracking: Dict[str, Any],
    cfg: SmoothConfig,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    List[Optional[np.ndarray]],
    np.ndarray, np.ndarray,
    List[Dict[str, Tuple[float, float]]],
]:
    """Build per-frame pixel-space observations + per-frame homographies + raw
    detections (including ones that fail the out-of-court sanity gate, for
    diagnostic overlay) + per-frame player foot positions in court meters.

    Returns:
      obs (N, 2): pixel (u, v) detections to feed the smoother (NaN where dropped).
      sigma (N, 2): per-axis pixel measurement std.
      mask (N,): True where a usable detection is present.
      orig_state (N,): 'detected' / 'no_ball' / 'no_homography' / 'out_of_court'.
      homographies (N): list of 3x3 image->court H (or None) per frame.
      raw_pix (N, 2): RAW per-frame ball pixel detection (NaN if no detection),
                      regardless of out-of-court / homography status. For diagnostic.
      player_court (N): list per frame of {player_id: (X_m, Y_m)} foot positions.
    """
    video = tracking.get("video") or {}
    total = int(video.get("total_frames") or 0)
    frames = tracking.get("frames") or []
    if total <= 0:
        total = max((int(f.get("frame", 0)) for f in frames), default=-1) + 1
    if total <= 0:
        return (
            np.empty((0, 2)), np.empty((0, 2)),
            np.zeros(0, bool), np.empty(0, dtype=object), [],
            np.empty((0, 2)), [],
        )

    fallback_kps = tracking.get("last_valid_court_keypoints")
    obs = np.full((total, 2), np.nan, dtype=np.float64)
    sigma = np.full((total, 2), np.inf, dtype=np.float64)
    mask = np.zeros((total,), dtype=bool)
    orig_state = np.array(["no_ball"] * total, dtype=object)
    homographies: List[Optional[np.ndarray]] = [None] * total
    raw_pix = np.full((total, 2), np.nan, dtype=np.float64)
    player_court: List[Dict[str, Tuple[float, float]]] = [dict() for _ in range(total)]

    last_kps_id: Optional[int] = None
    last_H: Optional[np.ndarray] = None

    # Wide tolerance: an airborne ball at ~3m above the court projects up to
    # 6–8m past the baseline in the ground-plane homography. We rely on MAD
    # outlier rejection downstream to drop genuine spurious detections.
    img_margin_m = 10.0
    x_min, x_max = -img_margin_m, COURT_WIDTH_M + img_margin_m
    y_min, y_max = -img_margin_m, COURT_LENGTH_M + img_margin_m

    for fr in frames:
        i = int(fr.get("frame", -1))
        if i < 0 or i >= total:
            continue
        kps = fr.get("court_keypoints") or fallback_kps
        if kps is not None:
            kps_id = id(kps)
            if kps_id != last_kps_id:
                last_H = _homography_image_to_court(kps)
                last_kps_id = kps_id
            homographies[i] = last_H
        pbox = fr.get("player_boxes") or {}
        if isinstance(pbox, dict) and homographies[i] is not None:
            for pid, bbox in pbox.items():
                pos = _player_foot_court(bbox, homographies[i])
                if pos is not None and math.isfinite(pos[0]) and math.isfinite(pos[1]):
                    player_court[i][str(pid)] = pos
        if not fr.get("present"):
            continue
        try:
            u = float(fr["x"])
            v = float(fr["y"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (math.isfinite(u) and math.isfinite(v)):
            continue
        # Record the raw pixel detection regardless of court gate.
        raw_pix[i] = (u, v)
        H = homographies[i]
        if H is None:
            orig_state[i] = "no_homography"
            continue
        proj = _project_image_to_court(H, u, v)
        if proj is None:
            continue
        X, Y = proj
        if not (math.isfinite(X) and math.isfinite(Y)):
            continue
        if X < x_min or X > x_max or Y < y_min or Y > y_max:
            orig_state[i] = "out_of_court"
            continue
        s = cfg.sigma_meas_pix
        obs[i] = (u, v)
        sigma[i] = (s, s)
        mask[i] = True
        orig_state[i] = "detected"
    return obs, sigma, mask, orig_state, homographies, raw_pix, player_court


def _mad_outlier_filter(obs: np.ndarray, mask: np.ndarray, window: int, k: float) -> np.ndarray:
    """Reject points whose per-axis deviation from the local median exceeds k * 1.4826 * MAD."""
    new_mask = mask.copy()
    N = obs.shape[0]
    if N == 0:
        return new_mask
    half = max(1, window // 2)
    for i in range(N):
        if not mask[i]:
            continue
        lo = max(0, i - half)
        hi = min(N, i + half + 1)
        local = obs[lo:hi]
        local_mask = mask[lo:hi]
        valid = local[local_mask]
        if valid.shape[0] < 5:
            continue
        med = np.median(valid, axis=0)
        mad = np.median(np.abs(valid - med), axis=0) * 1.4826 + 1e-6
        if np.any(np.abs(obs[i] - med) > k * mad):
            new_mask[i] = False
    return new_mask


# ---------- Step 2: track identification ------------------------------------

def _identify_tracks(mask: np.ndarray, max_gap: int, min_obs: int) -> List[Tuple[int, int]]:
    """Return (start, end) inclusive frame indices of tracks."""
    tracks: List[Tuple[int, int]] = []
    N = mask.shape[0]
    i = 0
    while i < N:
        if not mask[i]:
            i += 1
            continue
        last_obs = i
        j = i + 1
        while j < N:
            if mask[j]:
                last_obs = j
                j += 1
            elif j - last_obs > max_gap:
                break
            else:
                j += 1
        n_obs = int(mask[i:last_obs + 1].sum())
        if n_obs >= min_obs:
            tracks.append((i, last_obs))
        i = last_obs + 1
    return tracks


# ---------- Step 3: Kalman + RTS smoother -----------------------------------

def _build_F_Q(dt: float, sigma_a: float) -> Tuple[np.ndarray, np.ndarray]:
    """4D constant-velocity dynamics: state = [u, v, du, dv] in pixels. No gravity."""
    F = np.eye(4)
    F[0, 2] = dt
    F[1, 3] = dt
    Q = np.zeros((4, 4))
    sa2 = sigma_a * sigma_a
    q11 = (dt ** 3) / 3.0
    q12 = (dt ** 2) / 2.0
    q22 = dt
    for axis in range(2):
        Q[axis, axis] = sa2 * q11
        Q[axis, axis + 2] = sa2 * q12
        Q[axis + 2, axis] = sa2 * q12
        Q[axis + 2, axis + 2] = sa2 * q22
    return F, Q


def _inv2(M: np.ndarray) -> np.ndarray:
    """Closed-form 2x2 inverse."""
    a = float(M[0, 0]); b = float(M[0, 1])
    c = float(M[1, 0]); d = float(M[1, 1])
    det = a * d - b * c
    if abs(det) < 1e-20:
        raise np.linalg.LinAlgError("singular 2x2 matrix")
    inv_det = 1.0 / det
    out = np.empty((2, 2), dtype=np.float64)
    out[0, 0] = d * inv_det
    out[0, 1] = -b * inv_det
    out[1, 0] = -c * inv_det
    out[1, 1] = a * inv_det
    return out


def _smooth_segment(
    obs: np.ndarray,
    mask: np.ndarray,
    sigma: np.ndarray,
    dt: float,
    cfg: SmoothConfig,
) -> np.ndarray:
    """Forward Kalman + backward RTS over one contiguous span. Returns (N, 4) smoothed state."""
    N = obs.shape[0]
    F, Q = _build_F_Q(dt, cfg.sigma_accel_pxps2)
    H = np.zeros((2, 4))
    H[0, 0] = H[1, 1] = 1.0
    I4 = np.eye(4)
    out = np.full((N, 4), np.nan)
    if N == 0:
        return out

    first = next((i for i in range(N) if mask[i]), None)
    last = next((i for i in range(N - 1, -1, -1) if mask[i]), None)
    if first is None or last is None or last <= first:
        return out

    init_idx: List[int] = []
    for j in range(first, min(first + 20, N)):
        if mask[j]:
            init_idx.append(j)
        if len(init_idx) >= 8:
            break
    x_init = np.zeros(4)
    x_init[:2] = obs[first]
    if len(init_idx) >= 2:
        t_arr = np.asarray([(j - first) * dt for j in init_idx], dtype=np.float64)
        pos_arr = obs[init_idx]
        t_sum = float(t_arr.sum())
        tt_sum = float((t_arr * t_arr).sum())
        n_pts = float(len(init_idx))
        det = n_pts * tt_sum - t_sum * t_sum
        for axis in range(2):
            if abs(det) < 1e-12:
                x_init[axis + 2] = 0.0
                continue
            y_arr = pos_arr[:, axis]
            y_sum = float(y_arr.sum())
            ty_sum = float((t_arr * y_arr).sum())
            slope = (n_pts * ty_sum - t_sum * y_sum) / det
            intercept = (tt_sum * y_sum - t_sum * ty_sum) / det
            x_init[axis] = intercept
            x_init[axis + 2] = slope
    # Pixel-space init uncertainties: ~5 px of position uncertainty so the
    # measurement gate breathes around real motion; very generous velocity
    # uncertainty so the first few updates can chase the true motion
    # regardless of the linear-fit init.
    P_init = np.diag([25.0, 25.0, 1.0e6, 1.0e6])

    span = last - first + 1
    x_pred = np.zeros((span, 4))
    P_pred = np.zeros((span, 4, 4))
    x_filt = np.zeros((span, 4))
    P_filt = np.zeros((span, 4, 4))

    x_pred[0] = x_init
    P_pred[0] = P_init
    R0 = np.diag(sigma[first] ** 2)
    S0 = _abat(H, P_pred[0]) + R0
    S0_inv = _inv2(S0)
    PHT0 = _mm(P_pred[0], H.T)
    K0 = _mm(PHT0, S0_inv)
    innov0 = obs[first] - _mv(H, x_pred[0])
    x_filt[0] = x_pred[0] + _mv(K0, innov0)
    P_filt[0] = _mm(I4 - _mm(K0, H), P_pred[0])

    for s in range(1, span):
        k = first + s
        x_pred[s] = _mv(F, x_filt[s - 1])
        P_pred[s] = _abat(F, P_filt[s - 1]) + Q
        if mask[k]:
            R = np.diag(sigma[k] ** 2)
            S = _abat(H, P_pred[s]) + R
            innov = obs[k] - _mv(H, x_pred[s])
            try:
                S_inv = _inv2(S)
            except np.linalg.LinAlgError:
                x_filt[s] = x_pred[s]
                P_filt[s] = P_pred[s]
                continue
            chi2 = float(np.einsum("i,i->", innov, _mv(S_inv, innov)))
            if chi2 <= cfg.outlier_chi2_2d:
                PHT = _mm(P_pred[s], H.T)
                K = _mm(PHT, S_inv)
                x_filt[s] = x_pred[s] + _mv(K, innov)
                P_filt[s] = _mm(I4 - _mm(K, H), P_pred[s])
                continue
        x_filt[s] = x_pred[s]
        P_filt[s] = P_pred[s]

    x_sm = np.zeros((span, 4))
    P_sm = np.zeros((span, 4, 4))
    x_sm[-1] = x_filt[-1]
    P_sm[-1] = P_filt[-1]
    for s in range(span - 2, -1, -1):
        try:
            P_pred_inv = _inv_gj(P_pred[s + 1])
            PFT = _mm(P_filt[s], F.T)
            G = _mm(PFT, P_pred_inv)
        except np.linalg.LinAlgError:
            G = np.zeros((4, 4))
        x_sm[s] = x_filt[s] + _mv(G, x_sm[s + 1] - x_pred[s + 1])
        P_sm[s] = P_filt[s] + _abat(G, P_sm[s + 1] - P_pred[s + 1])

    out[first:last + 1] = x_sm
    return out


# ---------- Step 4: contact detection ---------------------------------------

def _detect_contacts(
    smoothed: np.ndarray, mask: np.ndarray, dt: float, cfg: SmoothConfig
) -> List[int]:
    """Detect contacts from the smoothed pixel-velocity series.

    The smoothed state holds [u, v, du, dv] in pixels and pixels/sec. A contact
    is a discontinuity in pixel velocity. Free-flight pixel motion is close to
    constant velocity; near a contact |v_next - v_prev| spikes far above the
    natural per-frame curvature change.

    Returns frames where the velocity increment is a local maximum, exceeds
    contact_speed_change_pxps, and is at least contact_min_sep_sec from the
    previous accepted contact.
    """
    N = smoothed.shape[0]
    contacts: List[int] = []
    if N < 3:
        return contacts
    incr = np.full(N, np.nan)
    for i in range(1, N - 1):
        v_prev = smoothed[i - 1, 2:]
        v_next = smoothed[i + 1, 2:]
        if np.any(np.isnan(v_prev)) or np.any(np.isnan(v_next)):
            continue
        dv = v_next - v_prev
        incr[i] = float(math.sqrt(float(dv[0] * dv[0] + dv[1] * dv[1])))

    thresh = cfg.contact_speed_change_pxps
    min_sep = max(1, int(round(cfg.contact_min_sep_sec / dt)))
    last = -10 ** 9
    for i in range(1, N - 1):
        r = incr[i]
        if not np.isfinite(r) or r < thresh:
            continue
        rp = incr[i - 1] if np.isfinite(incr[i - 1]) else -np.inf
        rn = incr[i + 1] if np.isfinite(incr[i + 1]) else -np.inf
        if r >= rp and r >= rn and (i - last) >= min_sep:
            contacts.append(i)
            last = i
    return contacts


# ---------- Step 5: drive everything ----------------------------------------

def _smooth_track_with_contacts(
    obs: np.ndarray,
    mask: np.ndarray,
    sigma: np.ndarray,
    dt: float,
    cfg: SmoothConfig,
) -> Tuple[np.ndarray, List[int]]:
    """RTS-smooth a track; detect contacts from the smoothed velocity series;
    re-smooth each sub-segment so velocity discontinuities at contacts survive."""
    first_pass = _smooth_segment(obs, mask, sigma, dt, cfg)
    contacts = _detect_contacts(first_pass, mask, dt, cfg)
    if not contacts:
        return first_pass, []
    N = obs.shape[0]
    boundaries = [0] + contacts + [N]
    final = np.full((N, 4), np.nan)
    for s, e in zip(boundaries[:-1], boundaries[1:]):
        if int(mask[s:e].sum()) >= 4:
            final[s:e] = _smooth_segment(obs[s:e], mask[s:e], sigma[s:e], dt, cfg)
    return final, contacts


def _identify_hitter(
    contact_xy: Tuple[float, float],
    players: Dict[str, Tuple[float, float]],
) -> Optional[Tuple[str, Tuple[float, float]]]:
    """Closest player to a court (X, Y) — used to anchor track start/end."""
    if not players:
        return None
    best_id: Optional[str] = None
    best_d = float("inf")
    for pid, pxy in players.items():
        if pxy is None:
            continue
        dx = pxy[0] - contact_xy[0]
        dy = pxy[1] - contact_xy[1]
        d = dx * dx + dy * dy
        if d < best_d:
            best_d = d
            best_id = pid
    if best_id is None:
        return None
    return best_id, players[best_id]


def _identify_hitter_via_backprojection(
    raw_pix: np.ndarray,
    track_start: int,
    N: int,
    tracking_frames_by_idx: Dict[int, Dict[str, Any]],
    homographies: List[Optional[np.ndarray]],
    forward: bool = False,
) -> Optional[Tuple[str, Tuple[float, float]]]:
    """Find the player who initiated (or received) the track by back- (or
    forward-) extrapolating the ball's PIXEL trajectory to a player's foot
    pixel height.

    Why this works: when a track starts, the ball is already a few frames past
    the racket strike, so its position projects past the player in court coords.
    But in PIXEL space the ball was at the player's racket height a few frames
    earlier. We use the ball's first measurable pixel velocity to extrapolate
    back to the pixel-v of each player's foot; the player whose foot the
    extrapolated pixel-u lands closest to is the most likely hitter.

    Returns (player_id, foot_xy_court_m) or None if the trajectory is too
    short / static to back-extrapolate confidently.
    """
    if not (0 <= track_start < N):
        return None
    # Collect the first 6 valid raw detections relative to track_start
    # (for back-extrap) or from end of track (for forward-extrap).
    sample_frames: List[int] = []
    if not forward:
        for j in range(track_start, min(track_start + 8, N)):
            if not np.any(np.isnan(raw_pix[j])):
                sample_frames.append(j)
            if len(sample_frames) >= 5:
                break
    else:
        for j in range(track_start, max(track_start - 8, -1), -1):
            if not np.any(np.isnan(raw_pix[j])):
                sample_frames.append(j)
            if len(sample_frames) >= 5:
                break
        sample_frames.reverse()
    if len(sample_frames) < 2:
        return None

    # Average pixel velocity per frame.
    diffs = []
    for a, b in zip(sample_frames[:-1], sample_frames[1:]):
        gap = b - a
        if gap <= 0:
            continue
        diffs.append((raw_pix[b] - raw_pix[a]) / gap)
    if not diffs:
        return None
    avg_v = np.mean(diffs, axis=0)
    if abs(avg_v[1]) < 1.0:  # too slow in v to extrapolate reliably
        return None

    anchor_idx = sample_frames[0] if not forward else sample_frames[-1]
    ball_at_anchor = raw_pix[anchor_idx]
    track_frame_data = tracking_frames_by_idx.get(track_start, {})
    pboxes = track_frame_data.get("player_boxes") or {}
    if not pboxes:
        # Try the anchor frame instead.
        track_frame_data = tracking_frames_by_idx.get(anchor_idx, {})
        pboxes = track_frame_data.get("player_boxes") or {}
    if not pboxes:
        return None

    H = homographies[anchor_idx] if 0 <= anchor_idx < len(homographies) else None
    best_id: Optional[str] = None
    best_dist = float("inf")
    best_foot_court: Optional[Tuple[float, float]] = None
    for pid, bbox in pboxes.items():
        if bbox is None or len(bbox) < 4:
            continue
        try:
            foot_u = (float(bbox[0]) + float(bbox[2])) / 2.0
            foot_v = float(bbox[3])
        except (TypeError, ValueError):
            continue
        # Solve for the "time" delta_t such that the ball's pixel-v aligns
        # with the player's foot pixel-v. delta_t > 0 means "in the past"
        # for back-extrapolation, "in the future" for forward-extrapolation.
        if abs(avg_v[1]) < 0.5:
            continue
        delta_t = (ball_at_anchor[1] - foot_v) / avg_v[1]
        # For back-extrap we want past times -> delta_t > 0 means we went
        # backward and the player was there. For forward-extrap reverse sign.
        if not forward:
            if delta_t <= 0:
                continue  # this player is in the *future* direction of motion
        else:
            if delta_t >= 0:
                continue
        # Pixel-u where the ball would have been at that foot-v level.
        extrapolated_u = ball_at_anchor[0] - avg_v[0] * delta_t
        dist = abs(foot_u - extrapolated_u)
        # Penalize implausible extrapolation distances (>~1.5 s back/forward).
        if abs(delta_t) > 90.0:
            dist += abs(delta_t) - 90.0
        if dist < best_dist:
            best_dist = dist
            best_id = pid
            # Project the player's foot to court coords.
            if H is not None:
                proj = _project_image_to_court(H, foot_u, foot_v)
                if proj is not None:
                    best_foot_court = (float(proj[0]), float(proj[1]))
    if best_id is None or best_foot_court is None:
        return None
    return best_id, best_foot_court


def _detect_contacts_raw(
    raw_pix: np.ndarray,
    track_start: int,
    track_end: int,
    cfg: SmoothConfig,
    min_pixel_step: float = 4.0,
    min_angle_deg: float = 75.0,
) -> List[int]:
    """Find contacts by looking at sharp pixel-trajectory direction reversals
    in RAW per-frame detections.

    For each frame i in the track that has raw detections at i-2, i-1, i, i+1,
    i+2, we compute the angle between (pix[i] - pix[i-2]) and (pix[i+2] - pix[i]).
    Angles above min_angle_deg with both legs ≥ min_pixel_step pixels are
    treated as direction reversals (a hit or a bounce). The smoother's
    velocity-based detector misses these when its sigma_accel rounds the
    corner; raw pixel data preserves the sharp transition."""
    contacts: List[int] = []
    min_sep = max(2, int(round(cfg.contact_min_sep_sec * cfg.fps)))
    cos_thresh = math.cos(math.radians(min_angle_deg))
    last_emit = -10 ** 9
    i = track_start + 2
    while i <= track_end - 2:
        # Need detections at i-2, i, i+2.
        if (np.any(np.isnan(raw_pix[i - 2])) or np.any(np.isnan(raw_pix[i]))
                or np.any(np.isnan(raw_pix[i + 2]))):
            i += 1
            continue
        v_prev = raw_pix[i] - raw_pix[i - 2]
        v_next = raw_pix[i + 2] - raw_pix[i]
        np_p = math.sqrt(float(v_prev[0] * v_prev[0] + v_prev[1] * v_prev[1]))
        np_n = math.sqrt(float(v_next[0] * v_next[0] + v_next[1] * v_next[1]))
        if np_p < min_pixel_step or np_n < min_pixel_step:
            i += 1
            continue
        dot = float(v_prev[0] * v_next[0] + v_prev[1] * v_next[1])
        cos_ang = dot / (np_p * np_n)
        if cos_ang < cos_thresh and (i - last_emit) >= min_sep:
            contacts.append(i)
            last_emit = i
        i += 1
    return contacts


def _build_anchors_and_interpolate(
    N: int,
    tracks: List[Tuple[int, int]],
    contacts_global: List[int],
    pos_m_proj: np.ndarray,
    raw_pix: np.ndarray,
    homographies: List[Optional[np.ndarray]],
    player_court: List[Dict[str, Tuple[float, float]]],  # noqa: ARG001 (kept for future use)
    tracking_frames_by_idx: Optional[Dict[int, Dict[str, Any]]] = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Replace the airborne ground projection with linear interp between
    detected contacts. Contacts are the only anchors we trust:
      - At a contact the ball is on the ground, so the homography projection
        is honest (no Z bias).
      - We use the RAW per-frame detection at the contact frame (not the
        smoothed pixel) so the bounce sits at the actual sharp event.

    Player-based anchors are intentionally NOT used. They were misleading:
      - Serve tosses: ball stays near the server in 3D but its airborne
        projection lands at the far baseline, so "closest player" picks the
        wrong side.
      - Track-start moments are already a few frames past the racket strike,
        so the ball is airborne and the projection is biased — closest-player
        anchoring inherits that bias.

    Frames outside any contact-to-contact bracket are left at their raw
    projection. The trail will still show the airborne-bunching for those
    portions, but it won't lie about which player initiated motion.
    """
    pos_m_corr = pos_m_proj.copy()
    anchors_log: List[Dict[str, Any]] = []
    contacts_in_track_map: Dict[Tuple[int, int], List[int]] = {
        (s, e): [c for c in contacts_global if s <= c <= e] for s, e in tracks
    }

    for s, e in tracks:
        contacts_here = contacts_in_track_map[(s, e)]
        anchors: List[Tuple[int, Tuple[float, float], str]] = []

        # Hitter at track start, by back-extrapolating ball pixel trajectory.
        if tracking_frames_by_idx is not None:
            hitter = _identify_hitter_via_backprojection(
                raw_pix, s, N, tracking_frames_by_idx, homographies, forward=False
            )
            if hitter is not None:
                anchors.append((s, hitter[1], f"hitter_start_{hitter[0]}"))

        for c in contacts_here:
            # Prefer the RAW pixel detection at the contact frame: that frame
            # is when the ball is on the ground and our detector saw it
            # sharply, so the homography projection of the raw pixel is the
            # most accurate court-plane bounce location we can produce.
            H = homographies[c]
            if H is not None and not np.any(np.isnan(raw_pix[c])):
                proj = _project_image_to_court(H, float(raw_pix[c, 0]), float(raw_pix[c, 1]))
                if proj is not None and math.isfinite(proj[0]) and math.isfinite(proj[1]):
                    anchors.append((c, (float(proj[0]), float(proj[1])), "contact"))
                    continue
            if not np.any(np.isnan(pos_m_proj[c])):
                anchors.append((c, (float(pos_m_proj[c, 0]), float(pos_m_proj[c, 1])), "contact_smoothed"))

        # Receiver / next-hitter at track end, by forward-extrapolating.
        if tracking_frames_by_idx is not None:
            receiver = _identify_hitter_via_backprojection(
                raw_pix, e, N, tracking_frames_by_idx, homographies, forward=True
            )
            if receiver is not None:
                # Only add this as an end anchor if it's a different player
                # from the start anchor and meaningfully far away in court
                # coords. Otherwise it's likely the same player and would
                # collapse the trail.
                start_player = anchors[0][2].split("_")[-1] if anchors else None
                if receiver[0] != start_player:
                    anchors.append((e, receiver[1], f"hitter_end_{receiver[0]}"))

        anchors_log.append({
            "track": [int(s), int(e)],
            "applied": bool(len(anchors) >= 1),
            "anchors": [
                {"frame": int(fr), "xy_m": [float(xy[0]), float(xy[1])], "kind": kind}
                for fr, xy, kind in anchors
            ],
        })

        # Drop duplicate-frame anchors (keep the first).
        seen_frames = set()
        deduped: List[Tuple[int, Tuple[float, float], str]] = []
        for fr, xy, kind in anchors:
            if fr in seen_frames:
                continue
            seen_frames.add(fr)
            deduped.append((fr, xy, kind))
        anchors = deduped

        # Pin the exact position at each anchor.
        for fr, xy, _ in anchors:
            pos_m_corr[fr, 0] = xy[0]
            pos_m_corr[fr, 1] = xy[1]

        # If we have two or more anchors, linearly interpolate between them.
        if len(anchors) >= 2:
            for a, b in zip(anchors[:-1], anchors[1:]):
                fa, xya, _ = a
                fb, xyb, _ = b
                if fb <= fa:
                    continue
                for k in range(fa, fb + 1):
                    t = (k - fa) / (fb - fa)
                    pos_m_corr[k, 0] = xya[0] + (xyb[0] - xya[0]) * t
                    pos_m_corr[k, 1] = xya[1] + (xyb[1] - xya[1]) * t

    return pos_m_corr, anchors_log


def smooth_trajectory(tracking: Dict[str, Any], cfg: SmoothConfig) -> Dict[str, Any]:
    obs, sigma, mask, orig_state, homographies, raw_pix, player_court = (
        _gather_observations_from_tracking(tracking, cfg)
    )
    n_pre = int(mask.sum())
    mask = _mad_outlier_filter(obs, mask, cfg.mad_window_frames, cfg.mad_reject_k)
    n_post = int(mask.sum())

    max_short_gap = max(1, int(round(cfg.max_short_gap_sec * cfg.fps)))
    tracks = _identify_tracks(mask, max_short_gap, cfg.min_track_obs)

    N = obs.shape[0]
    pos_px = np.full((N, 2), np.nan)
    vel_px = np.full((N, 2), np.nan)
    pos_m = np.full((N, 2), np.nan)
    status = np.array(["lost"] * N, dtype=object)
    contacts_global: List[int] = []
    dt = 1.0 / cfg.fps

    for s, e in tracks:
        sub_smooth, sub_contacts = _smooth_track_with_contacts(
            obs[s:e + 1], mask[s:e + 1], sigma[s:e + 1], dt, cfg
        )
        for k in range(e - s + 1):
            if np.any(np.isnan(sub_smooth[k])):
                continue
            pos_px[s + k] = sub_smooth[k, :2]
            vel_px[s + k] = sub_smooth[k, 2:]
            status[s + k] = "observed_smoothed" if mask[s + k] else "interp"
        contacts_global.extend(s + c for c in sub_contacts)
        # Also detect contacts via raw-pixel direction reversals — catches
        # bounces the smoothed-velocity detector missed because RTS rounded
        # the corner below the velocity-change threshold.
        raw_contacts = _detect_contacts_raw(raw_pix, s, e, cfg)
        # Merge with smoothed-velocity contacts, deduping near-duplicates.
        min_sep = max(2, int(round(cfg.contact_min_sep_sec * cfg.fps)))
        existing = {c for c in contacts_global if s <= c <= e}
        for rc in raw_contacts:
            if all(abs(rc - x) >= min_sep for x in existing):
                contacts_global.append(rc)
                existing.add(rc)
    contacts_global = sorted(set(contacts_global))

    # Project smoothed pixels to court meters using each frame's homography.
    # This is the raw projection — kept for diagnostic comparison.
    pos_m_proj = np.full((N, 2), np.nan)
    for i in range(N):
        if np.any(np.isnan(pos_px[i])):
            continue
        H = homographies[i]
        if H is None:
            continue
        proj = _project_image_to_court(H, float(pos_px[i, 0]), float(pos_px[i, 1]))
        if proj is None:
            continue
        pos_m_proj[i] = proj

    # Replace the in-flight projection with anchor-based linear interpolation
    # between the hitter's foot at track start, each detected contact (on the
    # ground), and the next hitter's foot at track end. Bounces stay where the
    # projection put them (they're on the ground); flight portions become clean
    # straight lines in court coords, which is geometrically correct because
    # the ball's (X, Y) component is approximately linear in time during flight.
    tracking_frames_by_idx = {int(f.get("frame", -1)): f for f in (tracking.get("frames") or [])}
    pos_m, anchors_log = _build_anchors_and_interpolate(
        N, tracks, contacts_global, pos_m_proj, raw_pix, homographies, player_court,
        tracking_frames_by_idx,
    )

    # Physics-based override: for every track that has at least one detected
    # contact AND a usable per-frame homography, fit a 3D parabolic trajectory
    # in court coords to the raw pixel detections. The fit gives us the true
    # (X, Y) court position during flight by exploiting the ball's known 3D
    # parabolic motion plus the Z=0 anchors at contacts. When the fit
    # succeeds and is reasonable, overwrite the linear-interp positions with
    # the projected (X, Y) from the 3D fit.
    video_info = tracking.get("video") or {}
    image_w = int(video_info.get("width") or 1920)
    image_h = int(video_info.get("height") or 1080)
    fit_log: List[Dict[str, Any]] = []
    for s, e in tracks:
        contacts_here = [c for c in contacts_global if s <= c <= e]
        if not contacts_here:
            continue
        ref_kps = tracking_frames_by_idx.get(contacts_here[0], {}).get("court_keypoints")
        if ref_kps is None:
            ref_kps = tracking.get("last_valid_court_keypoints")
        if ref_kps is None:
            continue
        pose = _camera_pose_from_court_keypoints(ref_kps, image_w, image_h)
        if pose is None:
            continue

        # Fit a separate parabola for EACH segment between consecutive
        # contacts. The endpoints of each segment are z=0 anchors, so the
        # fit becomes a well-constrained physics problem with most of the
        # parameters pinned. Start/end portions outside the contact-bracketed
        # range keep their projected position (already corrected by anchor
        # linear interp earlier).
        segment_bounds: List[Tuple[int, int]] = []
        for a, b in zip(contacts_here[:-1], contacts_here[1:]):
            segment_bounds.append((a, b))
        if not segment_bounds:
            fit_log.append({
                "track": [int(s), int(e)],
                "applied": False,
                "reason": "only one contact in track — need ≥2 for segmented fit",
            })
            continue

        track_fits: List[Dict[str, Any]] = []
        for seg_start, seg_end in segment_bounds:
            seg_detections: List[Tuple[int, float, float]] = []
            for k in range(seg_start, seg_end + 1):
                if not np.any(np.isnan(raw_pix[k])):
                    seg_detections.append((k, float(raw_pix[k, 0]), float(raw_pix[k, 1])))
            if len(seg_detections) < 4:
                continue
            if np.any(np.isnan(pos_m[seg_start])) or np.any(np.isnan(pos_m[seg_end])):
                continue
            dt_total = max(1e-6, (seg_end - seg_start) * dt)
            Vx_init = (pos_m[seg_end, 0] - pos_m[seg_start, 0]) / dt_total
            Vy_init = (pos_m[seg_end, 1] - pos_m[seg_start, 1]) / dt_total
            initial = (
                float(pos_m[seg_start, 0]), float(pos_m[seg_start, 1]),
                float(Vx_init), float(Vy_init),
            )
            fit = _fit_3d_trajectory(
                seg_detections, [seg_start, seg_end], pose, dt, seg_start, initial,
            )
            if fit is None:
                continue
            X0, Y0, Z0, Vx, Vy, Vz, cost = fit
            avg_resid = math.sqrt(2.0 * cost / max(1, len(seg_detections) * 2))
            if avg_resid > 60.0:
                track_fits.append({"seg": [seg_start, seg_end], "applied": False,
                                  "reason": f"resid {avg_resid:.1f}px"})
                continue
            z_samples = []
            for k in range(seg_start, seg_end + 1, max(1, (seg_end - seg_start) // 6)):
                tr = (k - seg_start) * dt
                z_samples.append(Z0 + Vz * tr + 0.5 * GRAVITY_MPS2 * tr * tr)
            peak_h = -min(z_samples)
            if peak_h > 15.0 or peak_h < -1.0:
                track_fits.append({"seg": [seg_start, seg_end], "applied": False,
                                  "reason": f"peak_h {peak_h:.1f}m"})
                continue
            for k in range(seg_start, seg_end + 1):
                tr = (k - seg_start) * dt
                pos_m[k, 0] = X0 + Vx * tr
                pos_m[k, 1] = Y0 + Vy * tr
            track_fits.append({
                "seg": [seg_start, seg_end], "applied": True,
                "avg_residual_px": float(avg_resid),
                "peak_height_m": float(peak_h),
                "n_detections": len(seg_detections),
            })

        fit_log.append({
            "track": [int(s), int(e)],
            "segments": track_fits,
        })

    # Court-space velocity is the finite-difference of the corrected positions.
    vel_m = np.full((N, 2), np.nan)
    for i in range(N):
        prev_i = i - 1
        next_i = i + 1
        p = pos_m[prev_i] if prev_i >= 0 else None
        n = pos_m[next_i] if next_i < N else None
        if p is not None and not np.any(np.isnan(p)) and n is not None and not np.any(np.isnan(n)):
            vel_m[i] = (n - p) / (2.0 * dt)
        elif not np.any(np.isnan(pos_m[i])):
            if n is not None and not np.any(np.isnan(n)):
                vel_m[i] = (n - pos_m[i]) / dt
            elif p is not None and not np.any(np.isnan(p)):
                vel_m[i] = (pos_m[i] - p) / dt

    # Classify contacts as 'hit' (>~100 degrees of court-space velocity
    # redirection) vs 'contact' (smaller change).
    events: List[Dict[str, Any]] = []
    for k in contacts_global:
        v_before = vel_m[k - 1] if k - 1 >= 0 and not np.any(np.isnan(vel_m[k - 1])) else None
        v_after = vel_m[k + 1] if k + 1 < N and not np.any(np.isnan(vel_m[k + 1])) else None
        kind = "contact"
        if v_before is not None and v_after is not None:
            nb = math.sqrt(float(v_before[0] * v_before[0] + v_before[1] * v_before[1]))
            na = math.sqrt(float(v_after[0] * v_after[0] + v_after[1] * v_after[1]))
            if nb > 0.5 and na > 0.5:
                cosang = float(v_before[0] * v_after[0] + v_before[1] * v_after[1]) / (nb * na)
                if cosang < -0.2:
                    kind = "hit"
        xy = None if np.any(np.isnan(pos_m[k])) else [float(pos_m[k, 0]), float(pos_m[k, 1])]
        events.append({
            "frame": int(k),
            "type": kind,
            "xy_m": xy,
            "v_before_mps": None if v_before is None else [float(v_before[0]), float(v_before[1])],
            "v_after_mps": None if v_after is None else [float(v_after[0]), float(v_after[1])],
            "speed_change_mps": (
                None if (v_before is None or v_after is None)
                else math.sqrt(float((v_after[0] - v_before[0]) ** 2 + (v_after[1] - v_before[1]) ** 2))
            ),
        })

    contact_set = set(contacts_global)
    frames_out: List[Dict[str, Any]] = []
    for i in range(N):
        row: Dict[str, Any] = {
            "frame": int(i),
            "status": str(status[i]),
            "original_state": str(orig_state[i]),
        }
        if not np.any(np.isnan(pos_m[i])):
            row["xy_m"] = [float(pos_m[i, 0]), float(pos_m[i, 1])]
            row["uv_pix"] = [float(pos_px[i, 0]), float(pos_px[i, 1])]
            if not np.any(np.isnan(pos_m_proj[i])):
                row["xy_m_raw_projection"] = [float(pos_m_proj[i, 0]), float(pos_m_proj[i, 1])]
            if not np.any(np.isnan(vel_m[i])):
                row["vxy_mps"] = [float(vel_m[i, 0]), float(vel_m[i, 1])]
                row["speed_mps"] = math.sqrt(float(vel_m[i, 0] ** 2 + vel_m[i, 1] ** 2))
            else:
                row["vxy_mps"] = None
                row["speed_mps"] = None
        else:
            row["xy_m"] = None
            row["uv_pix"] = None
            row["vxy_mps"] = None
            row["speed_mps"] = None
        if not np.any(np.isnan(raw_pix[i])):
            row["raw_uv_pix"] = [float(raw_pix[i, 0]), float(raw_pix[i, 1])]
        if i in contact_set:
            row["contact"] = True
        frames_out.append(row)

    counts = {
        "observed_smoothed": int(np.sum(status == "observed_smoothed")),
        "interp": int(np.sum(status == "interp")),
        "lost": int(np.sum(status == "lost")),
    }
    by_type: Dict[str, int] = {}
    for e in events:
        by_type[e["type"]] = by_type.get(e["type"], 0) + 1

    return {
        "schema_version": "smoothed_v5",
        "algorithm": "pixel-space RTS smoother + per-frame homography + anchor interp + 3D parabolic fit",
        "anchors": anchors_log,
        "trajectory_3d_fits": fit_log,
        "research_basis": [
            "Rauch-Tung-Striebel (1965) smoother",
            "Sarkka, Bayesian Filtering and Smoothing (2013), ch. 8",
            "per-frame image->court homography (cv2.findHomography)",
        ],
        "note": (
            "Smoothing happens in pixel coordinates (u, v) because the ball's "
            "image is near-constant-velocity during flight, while its ground "
            "projection bunches at the arc apex. The per-frame H is applied "
            "to the smoothed pixel position to produce xy_m. xy_m is therefore "
            "still the ground projection of a possibly-airborne ball — it "
            "sits past the true bounce point during flight — but the trail "
            "now progresses smoothly across the court."
        ),
        "video": tracking.get("video"),
        "parameters": {
            "fps": cfg.fps,
            "sigma_accel_pxps2": cfg.sigma_accel_pxps2,
            "sigma_meas_pix": cfg.sigma_meas_pix,
            "max_short_gap_sec": cfg.max_short_gap_sec,
            "min_track_obs": cfg.min_track_obs,
            "outlier_chi2_2d": cfg.outlier_chi2_2d,
            "contact_speed_change_pxps": cfg.contact_speed_change_pxps,
            "contact_min_sep_sec": cfg.contact_min_sep_sec,
            "mad_window_frames": cfg.mad_window_frames,
            "mad_reject_k": cfg.mad_reject_k,
        },
        "summary": {
            "frame_count": int(N),
            "track_count": int(len(tracks)),
            "tracks": [{"start": int(s), "end": int(e)} for s, e in tracks],
            "observations_before_mad": int(n_pre),
            "observations_after_mad": int(n_post),
            "mad_rejected": int(n_pre - n_post),
            "status_counts": counts,
            "contact_count": int(len(events)),
            "contacts_by_type": by_type,
        },
        "events": events,
        "frames": frames_out,
    }


# ---------- Rendering ------------------------------------------------------

def _draw_overlay(
    frame,
    pos: Optional[Tuple[float, float]],
    vel: Optional[Tuple[float, float]],
    status: str,
    contact: bool,
    trail_xy_m: List[Tuple[float, float]],
    court_w: float,
    court_l: float,
    cv2,
) -> None:
    H, W = frame.shape[:2]
    panel_w = 210
    panel_h = 370
    margin = 18
    pad = 14
    bar_w = 18
    bar_gap = 10
    panel_x1 = W - panel_w - margin
    panel_y1 = margin
    panel_x2 = panel_x1 + panel_w
    panel_y2 = panel_y1 + panel_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x1, panel_y1), (panel_x2, panel_y2), (15, 25, 25), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.rectangle(frame, (panel_x1, panel_y1), (panel_x2, panel_y2), (220, 220, 220), 2)

    court_x1 = panel_x1 + pad
    court_y1 = panel_y1 + pad + 14  # leave a row for status label
    court_x2 = panel_x2 - pad
    court_y2 = panel_y2 - pad - 30
    avail_w = court_x2 - court_x1
    avail_h = court_y2 - court_y1
    scale = min(avail_w / court_w, avail_h / court_l)
    draw_w = court_w * scale
    draw_h = court_l * scale
    cx = (court_x1 + court_x2) / 2.0
    cy = (court_y1 + court_y2) / 2.0
    ox = cx - draw_w / 2.0
    oy = cy - draw_h / 2.0

    def w2p(xm: float, ym: float) -> Tuple[int, int]:
        return int(round(ox + xm * scale)), int(round(oy + ym * scale))

    cv2.rectangle(frame, w2p(0, 0), w2p(court_w, court_l), (255, 255, 255), 2)
    if court_w > 8.23 + 0.1:
        inset = (court_w - 8.23) / 2.0
        cv2.line(frame, w2p(inset, 0), w2p(inset, court_l), (220, 220, 220), 1)
        cv2.line(frame, w2p(court_w - inset, 0), w2p(court_w - inset, court_l), (220, 220, 220), 1)
    cv2.line(frame, w2p(0, court_l / 2.0), w2p(court_w, court_l / 2.0), (0, 220, 255), 2)
    svc_top = court_l / 2.0 - 6.4
    svc_bot = court_l / 2.0 + 6.4
    if svc_top > 0:
        cv2.line(frame, w2p(0, svc_top), w2p(court_w, svc_top), (180, 180, 180), 1)
    if svc_bot < court_l:
        cv2.line(frame, w2p(0, svc_bot), w2p(court_w, svc_bot), (180, 180, 180), 1)
    cv2.line(frame, w2p(court_w / 2.0, svc_top), w2p(court_w / 2.0, svc_bot), (180, 180, 180), 1)

    # Trail with fading alpha — newest segments brightest, oldest faded — so
    # the eye reads the recent ball motion rather than the cumulative loop.
    pts = [w2p(x, y) for x, y in trail_xy_m]
    n_segs = max(1, len(pts) - 1)
    for idx, (p0, p1) in enumerate(zip(pts, pts[1:])):
        age = idx / n_segs  # 0 = oldest, ~1 = newest
        intensity = int(80 + 175 * age)
        color = (intensity, max(160, intensity), 255)
        thickness = 1 if age < 0.5 else 2
        cv2.line(frame, p0, p1, color, thickness, cv2.LINE_AA)

    if pos is not None:
        bx, by = w2p(pos[0], pos[1])
        if status == "interp":
            cv2.circle(frame, (bx, by), 5, (0, 165, 255), -1)
        else:
            cv2.circle(frame, (bx, by), 5, (0, 0, 255), -1)
        cv2.circle(frame, (bx, by), 6, (255, 255, 255), 1)
        if contact:
            cv2.circle(frame, (bx, by), 11, (0, 255, 255), 2)

    cv2.putText(
        frame, "ground projection",
        (court_x1, panel_y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
        (170, 170, 170), 1, cv2.LINE_AA,
    )

    label_color = {
        "observed_smoothed": (0, 220, 0),
        "interp": (0, 165, 255),
        "lost": (120, 120, 120),
    }.get(status, (200, 200, 200))
    label = {
        "observed_smoothed": "TRACKING",
        "interp": "INTERP",
        "lost": "LOST",
    }.get(status, status.upper())
    cv2.putText(frame, label, (court_x1, panel_y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                label_color, 1, cv2.LINE_AA)
    if contact:
        cv2.putText(frame, "CONTACT", (court_x1 + 105, panel_y1 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)


def _draw_source_frame_overlay(
    frame,
    raw_trail_px: List[Tuple[float, float]],
    smooth_trail_px: List[Tuple[float, float]],
    raw_now: Optional[Tuple[float, float]],
    smooth_now: Optional[Tuple[float, float]],
    contact: bool,
    cv2,
) -> None:
    """Draw the per-frame raw detection trail (lime) and smoothed pixel trail
    (magenta) directly on the source video frame, plus a yellow ring at the
    current frame when it's a detected contact. Both trails fade with age."""

    def fade_polyline(pts, near_color, far_color, base_thickness):
        n = max(1, len(pts) - 1)
        for idx, (p0, p1) in enumerate(zip(pts, pts[1:])):
            age = idx / n  # 0 = oldest, 1 = newest
            color = (
                int(far_color[0] + (near_color[0] - far_color[0]) * age),
                int(far_color[1] + (near_color[1] - far_color[1]) * age),
                int(far_color[2] + (near_color[2] - far_color[2]) * age),
            )
            thick = base_thickness if age > 0.5 else max(1, base_thickness - 1)
            cv2.line(frame, (int(round(p0[0])), int(round(p0[1]))),
                     (int(round(p1[0])), int(round(p1[1]))),
                     color, thick, cv2.LINE_AA)

    # Lime = raw detections (BGR (0, 255, 0) is lime green).
    fade_polyline(raw_trail_px, (0, 255, 0), (0, 100, 0), 3)
    # Magenta = smoothed pixel positions.
    fade_polyline(smooth_trail_px, (255, 0, 220), (100, 0, 100), 3)

    # Per-frame raw dots (so even isolated detections are visible).
    for x, y in raw_trail_px[-12:]:
        cv2.circle(frame, (int(round(x)), int(round(y))), 2, (0, 255, 80), -1, cv2.LINE_AA)

    if raw_now is not None:
        cv2.circle(frame, (int(round(raw_now[0])), int(round(raw_now[1]))), 7,
                   (0, 255, 0), 2, cv2.LINE_AA)
    if smooth_now is not None:
        cv2.circle(frame, (int(round(smooth_now[0])), int(round(smooth_now[1]))), 9,
                   (255, 0, 220), 2, cv2.LINE_AA)
    if contact and smooth_now is not None:
        cv2.circle(frame, (int(round(smooth_now[0])), int(round(smooth_now[1]))), 18,
                   (0, 255, 255), 3, cv2.LINE_AA)

    # Legend in the bottom-left.
    pad = 10
    bx, by = pad, frame.shape[0] - 70
    cv2.rectangle(frame, (bx - 4, by - 18), (bx + 220, by + 50),
                  (20, 20, 20), -1)
    cv2.line(frame, (bx, by - 6), (bx + 28, by - 6), (0, 255, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, "raw detection", (bx + 36, by - 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (180, 255, 180), 1, cv2.LINE_AA)
    cv2.line(frame, (bx, by + 14), (bx + 28, by + 14), (255, 0, 220), 3, cv2.LINE_AA)
    cv2.putText(frame, "smoothed pixel", (bx + 36, by + 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 200, 255), 1, cv2.LINE_AA)
    cv2.circle(frame, (bx + 14, by + 34), 7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "contact", (bx + 36, by + 38), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (180, 240, 240), 1, cv2.LINE_AA)


def render_smoothed_video(
    smoothed: Dict[str, Any],
    input_video: Path,
    output_video: Path,
    court_w: float,
    court_l: float,
) -> None:
    import cv2  # type: ignore

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or float(smoothed["parameters"]["fps"])

    frames = {int(r["frame"]): r for r in smoothed["frames"]}
    trail_len = max(8, int(round(fps * 0.7)))
    # Per-trail buffers for the mini-court (court coords) and the source-frame
    # overlay (pixel coords). Cleared on lost/no-track frames.
    court_trail: List[Tuple[float, float]] = []
    raw_pix_trail: List[Tuple[float, float]] = []
    smooth_pix_trail: List[Tuple[float, float]] = []

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        row = frames.get(idx)
        pos = None
        vel = None
        status = "lost"
        contact = False
        raw_pix_now: Optional[Tuple[float, float]] = None
        smooth_pix_now: Optional[Tuple[float, float]] = None
        if row is not None:
            status = str(row.get("status", "lost"))
            contact = bool(row.get("contact", False))
            xy = row.get("xy_m")
            v = row.get("vxy_mps")
            uv_smooth = row.get("uv_pix")
            uv_raw = row.get("raw_uv_pix")
            if xy is not None:
                pos = (float(xy[0]), float(xy[1]))
                court_trail.append((pos[0], pos[1]))
                if len(court_trail) > trail_len:
                    court_trail = court_trail[-trail_len:]
            else:
                court_trail = []
            if v is not None:
                vel = (float(v[0]), float(v[1]))
            if uv_raw is not None and math.isfinite(uv_raw[0]) and math.isfinite(uv_raw[1]):
                raw_pix_now = (float(uv_raw[0]), float(uv_raw[1]))
                raw_pix_trail.append(raw_pix_now)
                if len(raw_pix_trail) > trail_len:
                    raw_pix_trail = raw_pix_trail[-trail_len:]
            if uv_smooth is not None and math.isfinite(uv_smooth[0]) and math.isfinite(uv_smooth[1]):
                smooth_pix_now = (float(uv_smooth[0]), float(uv_smooth[1]))
                smooth_pix_trail.append(smooth_pix_now)
                if len(smooth_pix_trail) > trail_len:
                    smooth_pix_trail = smooth_pix_trail[-trail_len:]
            if status == "lost":
                raw_pix_trail = []
                smooth_pix_trail = []
        else:
            court_trail = []
            raw_pix_trail = []
            smooth_pix_trail = []

        _draw_source_frame_overlay(
            frame, raw_pix_trail, smooth_pix_trail,
            raw_pix_now, smooth_pix_now, contact, cv2,
        )
        _draw_overlay(frame, pos, vel, status, contact, court_trail,
                      court_w, court_l, cv2)
        writer.write(frame)
        idx += 1
    cap.release()
    writer.release()


# ---------- CLI -------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Physics-constrained piecewise ballistic smoother. Reads tracking.json "
                    "(2D ball detections + per-frame court keypoints), projects detections to "
                    "the court ground plane via per-frame homography, then smooths with an RTS "
                    "filter and detects hits via gravity-constrained ballistic residual peaks."
    )
    parser.add_argument("--tracking-json", required=True, help="Input tracking.json from now_main.")
    parser.add_argument("--output-json", required=True, help="Output smoothed JSON path.")
    parser.add_argument("--input-video", default=None, help="Optional input video for re-rendering.")
    parser.add_argument("--render-video", default=None, help="Optional output MP4 with mini-court overlay.")
    parser.add_argument("--fps", type=float, default=None, help="Override fps (default: read from tracking JSON video.fps).")
    parser.add_argument("--sigma-accel-pxps2", type=float, default=600.0, help="Process noise (unmodeled accel) in pixels/sec^2.")
    parser.add_argument("--contact-speed-change-pxps", type=float, default=500.0, help="Velocity-discontinuity threshold (pixels/sec).")
    parser.add_argument("--max-short-gap-sec", type=float, default=0.35)
    args = parser.parse_args()

    tracking_path = Path(args.tracking_json)
    with tracking_path.open("r", encoding="utf-8") as f:
        tracking = json.load(f)
    video_info = tracking.get("video") or {}
    fps = float(args.fps if args.fps is not None else (video_info.get("fps") or 0.0))
    if fps <= 0:
        raise SystemExit("Could not determine fps; pass --fps explicitly.")
    cfg = SmoothConfig(
        fps=fps,
        sigma_accel_pxps2=args.sigma_accel_pxps2,
        contact_speed_change_pxps=args.contact_speed_change_pxps,
        max_short_gap_sec=args.max_short_gap_sec,
    )

    smoothed = smooth_trajectory(tracking, cfg)
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(smoothed, f, indent=2)
    summary = smoothed["summary"]
    print(
        f"[smooth_v1] tracks={summary['track_count']} "
        f"obs={summary['observations_after_mad']}/{summary['observations_before_mad']} "
        f"(rejected {summary['mad_rejected']}) "
        f"frames(obs/interp/lost)={summary['status_counts']['observed_smoothed']}/"
        f"{summary['status_counts']['interp']}/{summary['status_counts']['lost']} "
        f"contacts={summary['contact_count']} by_type={summary['contacts_by_type']}"
    )
    print(f"[smooth_v1] Output JSON: {out_path}")

    if args.render_video:
        if not args.input_video:
            raise SystemExit("--render-video requires --input-video")
        render_smoothed_video(
            smoothed, Path(args.input_video), Path(args.render_video),
            court_w=COURT_WIDTH_M, court_l=COURT_LENGTH_M,
        )
        print(f"[smooth_v1] Debug video: {args.render_video}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
