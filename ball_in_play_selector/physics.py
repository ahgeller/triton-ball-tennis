import math
import numpy as np
from typing import Optional, Tuple
try:
    from filterpy.kalman import KalmanFilter
except ImportError:
    KalmanFilter = None

from .config import SelectorConfig
from .utils import _fps_norm_pxpf, _clamp01, court_px_per_meter


def _kinematic_motion_frac(
    avg_speed: float,
    peak_speed: float,
    extent_px: float,
    near_player_frac: float,
    cfg: SelectorConfig,
    diag: float
) -> float:
    """Estimate motion evidence from kinematics when mask overlap is weak/noisy."""
    weak = _fps_norm_pxpf(1.6, cfg)
    strong = _fps_norm_pxpf(4.6, cfg)
    peak_ref = _fps_norm_pxpf(8.0, cfg)
    extent_ref = max(0.08 * diag, 1.0)

    avg_term = _clamp01((avg_speed - weak) / max(strong - weak, 1e-6))
    peak_term = _clamp01((peak_speed - (1.6 * weak)) / max(peak_ref - (1.6 * weak), 1e-6))
    extent_term = _clamp01(extent_px / extent_ref)

    kin = max(avg_term, 0.80 * peak_term, 0.60 * extent_term)
    if near_player_frac >= 0.10 and avg_speed >= 0.85 * weak:
        kin = max(kin, 0.30)
    return _clamp01(kin)

def _xy_dist(ax: float, ay: float, bx: float, by: float) -> float:
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)

def _predict_projectile(
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

    g = cfg.gravity_px_per_frame2
    drag = cfg.gravity_drag_factor
    x, y = pos
    vx, vy = vel

    for _ in range(dt):
        x += vx
        y += vy
        vx *= drag
        vy = vy * drag + g
    return (x, y)

def _predict_projectile_vel(
    vel: Tuple[float, float],
    dt: int,
    cfg: SelectorConfig,
) -> Tuple[float, float]:
    """Predict velocity after dt frames under projectile physics."""
    if not cfg.gravity_enabled or dt <= 0:
        return vel
    g = cfg.gravity_px_per_frame2
    drag = cfg.gravity_drag_factor
    vx, vy = vel
    for _ in range(dt):
        vx *= drag
        vy = vy * drag + g
    return (vx, vy)

class BallKalmanFilter:
    def __init__(self, x0: float, y0: float, cfg: SelectorConfig):
        if KalmanFilter is None:
            raise RuntimeError("filterpy is required for BallKalmanFilter. Install filterpy==1.4.5.")
        self.cfg = cfg
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        
        # State: [x, y, vx, vy] as column vector (shape 4,1) as filterpy expects
        self.kf.x = np.array([[x0], [y0], [0.0], [0.0]])
        
        # Transition Matrix
        self.kf.F = np.array([[1., 0., 1., 0.],
                              [0., 1., 0., 1.],
                              [0., 0., 1., 0.],
                              [0., 0., 0., 1.]])
                              
        # Measurement Matrix (we only observe x and y)
        self.kf.H = np.array([[1., 0., 0., 0.],
                              [0., 1., 0., 0.]])
                              
        # Covariance Matrix
        # High uncertainty for unobserved initial velocity
        self.kf.P = np.diag([10.0, 10.0, 100.0, 100.0])
        
        # Measurement Noise
        # Measurement noise is adapted per-update (based on det confidence).
        # Initialize with a reasonable default (~2px std).
        self.kf.R = np.diag([4.0, 4.0])
        
        # Process Noise
        self.kf.Q = np.diag([0.1, 0.1, 2.0, 2.0])
        
        # Control Input Matrix (for gravity and drag)
        self.kf.B = np.array([[0.], [0.], [0.], [1.]])
        # Court homography for depth-aware gravity (set after construction)
        self._court_H: Optional[np.ndarray] = None
        self._court_H_inv: Optional[np.ndarray] = None
        self._court_w_m: float = 10.97
        # Cached reference px/m scale (updated each predict call)
        self._ref_px_per_m: Optional[float] = None

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

    def set_homography(
        self,
        H: np.ndarray,
        H_inv: np.ndarray,
        court_w_m: float = 10.97,
    ) -> None:
        """Attach a court homography for depth-aware gravity scaling."""
        self._court_H = H
        self._court_H_inv = H_inv
        self._court_w_m = court_w_m

    def _x(self) -> np.ndarray:
        """Always return a 1D view of the state (handles both (4,) and (4,1) shapes)."""
        return self.kf.x.flatten()

    def predict(self) -> Tuple[float, float]:
        """Predict the next state, scaling gravity by local pixel depth."""
        x = self._x()
        # Depth-aware gravity: compute local px/meter at current ball position,
        # then scale gravity so 1 real-world g maps correctly regardless of depth.
        base_g = float(self.cfg.gravity_px_per_frame2) if self.cfg.gravity_enabled else 0.0
        if base_g != 0.0 and self._court_H is not None:
            local_scale = court_px_per_meter(
                float(x[0]), float(x[1]),
                self._court_H, self._court_H_inv, self._court_w_m
            )
            if local_scale is not None and self._ref_px_per_m is not None and self._ref_px_per_m > 0:
                # Scale gravity so that near/far balls both feel the same real g
                depth_scale = local_scale / self._ref_px_per_m
                base_g *= depth_scale
            elif local_scale is not None:
                # First valid measurement: store as reference
                self._ref_px_per_m = local_scale
        u = np.array([[base_g]])

        # Apply drag to velocity before prediction
        if self.cfg.gravity_enabled:
            self.kf.x[2, 0] = x[2] * self.cfg.gravity_drag_factor
            self.kf.x[3, 0] = x[3] * self.cfg.gravity_drag_factor

        self.kf.predict(u=u)
        x = self._x()
        return float(x[0]), float(x[1])

    def predict_dt(self, dt: int) -> Tuple[float, float]:
        """Predict the state an arbitrary number of frames ahead WITHOUT changing the true state."""
        x_pred = self._x().copy()
        
        for _ in range(dt):
            # Apply drag to predictions
            if self.cfg.gravity_enabled:
                x_pred[2] *= self.cfg.gravity_drag_factor
                x_pred[3] *= self.cfg.gravity_drag_factor
                
            x_pred = np.dot(self.kf.F, x_pred)
            x_pred = np.asarray(x_pred).flatten()
            
            # Apply gravity to predictions
            if self.cfg.gravity_enabled:
                x_pred[3] += self.cfg.gravity_px_per_frame2
                
        return float(x_pred[0]), float(x_pred[1])

    def get_search_radius(self, scale_mult: float = 3.0) -> float:
        """Extract search radius based on position uncertainty."""
        # Max of variance in X or Y
        var_pos = max(self.kf.P[0, 0], self.kf.P[1, 1])
        std_pos = math.sqrt(var_pos)
        
        # Ensure it doesn't get ridiculously small or large
        radius = np.clip(std_pos * scale_mult, 10.0, 100.0)
        return float(radius)

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
            self.kf.x[3, 0] = -abs(pred_vy) * max(0.0, float(self.cfg.bounce_restitution))
            self.kf.P[3, 3] += 80.0

        self.kf.update(z)
        return d2

    def get_velocity(self) -> Tuple[float, float]:
        x = self._x()
        return float(x[2]), float(x[3])
