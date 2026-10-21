import math
import numpy as np
from typing import Optional, Tuple

from .config import SelectorConfig
from .utils import clamp01, pxpf_at_fps


def kinematic_motion_frac(
    avg_speed: float,
    peak_speed: float,
    extent_px: float,
    near_player_frac: float,
    cfg: SelectorConfig,
    diag: float
) -> float:
    """Estimate motion evidence from kinematics when mask overlap is weak/noisy."""
    weak = pxpf_at_fps(1.6, cfg.fps, clamp=True)
    strong = pxpf_at_fps(4.6, cfg.fps, clamp=True)
    peak_ref = pxpf_at_fps(8.0, cfg.fps, clamp=True)
    extent_ref = max(0.08 * diag, 1.0)

    avg_term = clamp01((avg_speed - weak) / max(strong - weak, 1e-6))
    peak_term = clamp01((peak_speed - (1.6 * weak)) / max(peak_ref - (1.6 * weak), 1e-6))
    extent_term = clamp01(extent_px / extent_ref)

    kin = max(avg_term, 0.80 * peak_term, 0.60 * extent_term)
    if near_player_frac >= 0.10 and avg_speed >= 0.85 * weak:
        kin = max(kin, 0.30)
    return clamp01(kin)

def xy_dist(ax: float, ay: float, bx: float, by: float) -> float:
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)

def _projectile_matrices(cfg: SelectorConfig, dt: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Exact linear-drag transition in source-frame units; drag is a 30 FPS reference."""
    drag = cfg.gravity_drag_factor if cfg.gravity_enabled else 1.0
    if not 0.0 < drag <= 1.0:
        raise ValueError("gravity_drag_factor must be in (0, 1]")
    rate = -math.log(drag) * 30.0 / max(cfg.fps, 1.0)
    if rate < 1e-8:
        decay, travel, fall = 1.0, float(dt), 0.5 * dt * dt
    else:
        decay = math.exp(-rate * dt)
        travel = -math.expm1(-rate * dt) / rate
        fall = (dt - travel) / rate
    F = np.array([[1., 0., travel, 0.], [0., 1., 0., travel],
                  [0., 0., decay, 0.], [0., 0., 0., decay]])
    B = np.array([[0.], [fall], [0.], [travel]])
    return F, B

def predict_projectile(
    pos: Tuple[float, float],
    vel: Tuple[float, float],
    dt: int,
    cfg: SelectorConfig,
) -> Tuple[float, float]:
    """Predict ball position using projectile physics (gravity + drag).

    pos: (x, y) - last known position (pixel coords, y increases downward)
    vel: (vx, vy) - velocity in px/frame
    dt: frames ahead to predict
    Returns (pred_x, pred_y).
    """
    if not cfg.gravity_enabled or dt <= 0:
        return (pos[0] + vel[0] * dt, pos[1] + vel[1] * dt)

    F, B = _projectile_matrices(cfg, dt)
    state = F @ np.array([*pos, *vel]) + B[:, 0] * cfg.gravity_px_per_frame2
    return float(state[0]), float(state[1])

def predict_projectile_velocity(
    vel: Tuple[float, float],
    dt: int,
    cfg: SelectorConfig,
) -> Tuple[float, float]:
    """Predict velocity after dt frames under projectile physics."""
    if not cfg.gravity_enabled or dt <= 0:
        return vel
    F, B = _projectile_matrices(cfg, dt)
    state = F @ np.array([0., 0., *vel]) + B[:, 0] * cfg.gravity_px_per_frame2
    return float(state[2]), float(state[3])

class LinearKalman:
    """A linear Kalman filter over a column-vector state, in plain numpy.

    ``x`` is the ``(dim_x, 1)`` state, ``P`` its covariance; ``F``/``B`` the
    transition and control matrices, ``Q`` the process noise, ``H`` the
    measurement matrix and ``R`` the measurement noise.  The caller may set
    any of them between steps (the ball filter swaps ``F``/``B`` for the
    projectile transition and ``R`` per detection confidence).

    The equations, their operation order and the column-vector shapes are
    exactly those of filterpy 1.4.5's ``KalmanFilter.predict``/``update``,
    which this replaced, so the two produce bit-identical states.
    """

    def __init__(self, x: np.ndarray, P: np.ndarray, F: np.ndarray, B: np.ndarray,
                 Q: np.ndarray, H: np.ndarray, R: np.ndarray) -> None:
        self.x = np.asarray(x, dtype=float)
        self.P = np.asarray(P, dtype=float)
        self.F = np.asarray(F, dtype=float)
        self.B = np.asarray(B, dtype=float)
        self.Q = np.asarray(Q, dtype=float)
        self.H = np.asarray(H, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self._identity = np.eye(self.x.shape[0])

    def predict(self, u: np.ndarray) -> None:
        """x = F x + B u;  P = F P F' + Q."""
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, z: np.ndarray) -> None:
        """Fold in the ``(dim_z, 1)`` measurement ``z`` with the Joseph-form covariance update."""
        y = z - np.dot(self.H, self.x)
        PHT = np.dot(self.P, self.H.T)
        S = np.dot(self.H, PHT) + self.R
        K = np.dot(PHT, np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I_KH = self._identity - np.dot(K, self.H)
        self.P = np.dot(np.dot(I_KH, self.P), I_KH.T) + np.dot(np.dot(K, self.R), K.T)


class BallKalmanFilter:
    """Constant-velocity (x, y, vx, vy) filter with the projectile transition and adaptive gating."""

    def __init__(self, x0: float, y0: float, cfg: SelectorConfig) -> None:
        self.cfg = cfg
        self.kf = LinearKalman(
            # State [x, y, vx, vy] as a column vector.
            x=np.array([[x0], [y0], [0.0], [0.0]]),
            # High uncertainty for the unobserved initial velocity.
            P=np.diag([10.0, 10.0, 100.0, 100.0]),
            # Constant-velocity transition; predict() swaps in the projectile one.
            F=np.array([[1., 0., 1., 0.],
                        [0., 1., 0., 1.],
                        [0., 0., 1., 0.],
                        [0., 0., 0., 1.]]),
            # Control input (gravity acts on vy); predict() swaps in the projectile one.
            B=np.array([[0.], [0.], [0.], [1.]]),
            Q=np.diag([0.1, 0.1, 2.0, 2.0]),
            # Only x and y are observed.
            H=np.array([[1., 0., 0., 0.],
                        [0., 1., 0., 0.]]),
            # Measurement noise is set per update from the detection confidence;
            # this default is ~2 px std.
            R=np.diag([4.0, 4.0]),
        )

    def _meas_sigma_px(self, conf: Optional[float]) -> float:
        """Map detection confidence -> measurement std-dev in pixels."""
        if conf is None:
            return 2.5
        c = float(conf)
        if not math.isfinite(c):
            return 3.5
        c = max(0.0, min(1.0, c))
        # High-confidence detections are precise; low-confidence boxes are noisy.
        # 0.0 -> ~7px, 1.0 -> ~2px
        return 2.0 + (1.0 - c) * 5.0

    def _mahalanobis_d2(self, z_x: float, z_y: float, R: np.ndarray) -> float:
        """Compute squared Mahalanobis distance for measurement residual."""
        x = self._x()
        H = self.kf.H
        z = np.array([z_x, z_y], dtype=float)
        hx = np.dot(H, x).astype(float)
        y = (z - hx).reshape(2, 1)
        S = (H @ self.kf.P @ H.T) + R
        try:
            Sinv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            Sinv = np.linalg.pinv(S)
        d2 = float((y.T @ Sinv @ y).item())
        return d2


    def _x(self) -> np.ndarray:
        """The (4, 1) column state flattened to a 1D copy."""
        return self.kf.x.flatten()


    def predict(self) -> Tuple[float, float]:
        """Use the same transition for ROI, filter state and covariance."""
        gravity = self.cfg.gravity_px_per_frame2 if self.cfg.gravity_enabled else 0.0
        self.kf.F, self.kf.B = _projectile_matrices(self.cfg)
        self.kf.predict(np.array([[gravity]]))
        x = self._x()
        return float(x[0]), float(x[1])

    def predict_dt(self, dt: int) -> Tuple[float, float]:
        """Predict the state an arbitrary number of frames ahead WITHOUT changing the true state."""
        x_pred = self._x().copy()
        gravity = self.cfg.gravity_px_per_frame2 if self.cfg.gravity_enabled else 0.0
        F, B = _projectile_matrices(self.cfg)
        for _ in range(dt):
            x_pred = F @ x_pred + B[:, 0] * gravity
                
        return float(x_pred[0]), float(x_pred[1])

    def update(self, z_x: float, z_y: float, conf: Optional[float] = None) -> float:
        """Update filter with a measurement.

        Returns squared Mahalanobis distance (innovation) used for gating/diagnostics.
        """
        sigma = self._meas_sigma_px(conf)
        R = np.diag([sigma * sigma, sigma * sigma]).astype(float)
        self.kf.R = R

        # Uncertainty-aware gating: if the measurement is an outlier, inflate R so the
        # filter doesn't snap hard, but still allows strong corrections when uncertainty is high.
        d2 = self._mahalanobis_d2(z_x, z_y, R)

        # Chi-square thresholds for 2 DoF:
        # 0.99 => 9.21, 0.999 => 13.82
        gate_soft = 9.21
        gate_hard = 13.82
        if d2 > gate_hard:
            # Big outlier: trust measurement less + allow velocity to change more.
            scale = min(12.0, max(1.5, math.sqrt(d2 / gate_hard)))
            self.kf.R = np.diag([(sigma * scale) ** 2, (sigma * scale) ** 2]).astype(float)
            self.kf.P[2, 2] += 120.0 * scale
            self.kf.P[3, 3] += 120.0 * scale
        elif d2 > gate_soft:
            # Mild outlier: soften update a bit.
            scale = min(4.0, max(1.2, math.sqrt(d2 / gate_soft)))
            self.kf.R = np.diag([(sigma * scale) ** 2, (sigma * scale) ** 2]).astype(float)

        z = np.array([[z_x], [z_y]], dtype=float)
        x = self._x()

        # Bounce heuristic (only if descending and measurement is significantly above prediction).
        # This keeps bounces plausible without forbidding racket hits (which can look similar).
        hx = np.dot(self.kf.H, x).astype(float)
        residual = np.array([z_x, z_y], dtype=float) - hx
        pred_vy = float(x[3])
        if pred_vy > 2.0 and float(residual[1]) < -10.0:
            self.kf.x[3, 0] = -abs(pred_vy) * self.cfg.bounce_restitution
            self.kf.P[3, 3] += 80.0

        self.kf.update(z)
        return d2

    def get_velocity(self) -> Tuple[float, float]:
        x = self._x()
        return float(x[2]), float(x[3])
