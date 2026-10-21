"""The selector's Kalman filter: one predict/update step against hand-computed values.

``LinearKalman`` replaced filterpy's ``KalmanFilter``; the equations and their
operation order are the same, so when filterpy happens to be installed the
second test checks the two stay bit-identical through the adaptive gates.
"""
import numpy as np
import pytest

from ball_in_play_selector.config import SelectorConfig
from ball_in_play_selector.physics import BallKalmanFilter


def _constant_velocity_cfg() -> SelectorConfig:
    cfg = SelectorConfig(fps=30.0, width=1920, height=1080)
    cfg.gravity_enabled = False      # F is then the plain constant-velocity matrix, u = 0
    return cfg


def test_one_predict_update_step_matches_hand_computed_values():
    kf = BallKalmanFilter(10.0, 20.0, _constant_velocity_cfg())

    assert kf.predict() == (10.0, 20.0)       # zero initial velocity
    # P = F P0 F' + Q with P0 = diag(10, 10, 100, 100), Q = diag(0.1, 0.1, 2, 2).
    expected_prior = np.array([
        [110.1, 0.0, 100.0, 0.0],
        [0.0, 110.1, 0.0, 100.0],
        [100.0, 0.0, 102.0, 0.0],
        [0.0, 100.0, 0.0, 102.0],
    ])
    np.testing.assert_allclose(kf.kf.P, expected_prior, rtol=0, atol=1e-12)

    # conf=None -> sigma 2.5 px -> R = diag(6.25); S = 110.1 + 6.25 per axis.
    d2 = kf.update(12.0, 22.0, conf=None)
    s = 116.35
    assert d2 == pytest.approx((2.0 * 2.0 + 2.0 * 2.0) / s, rel=1e-12)
    k_pos, k_vel = 110.1 / s, 100.0 / s
    np.testing.assert_allclose(
        kf.kf.x[:, 0],
        [10.0 + 2.0 * k_pos, 20.0 + 2.0 * k_pos, 2.0 * k_vel, 2.0 * k_vel],
        rtol=1e-12,
    )
    assert kf.get_velocity() == pytest.approx((2.0 * k_vel, 2.0 * k_vel), rel=1e-12)
    # Posterior covariance of the (x, vx) block, (I - KH) P in closed form.
    gain_complement = 6.25 / s
    np.testing.assert_allclose(kf.kf.P[0, 0], 110.1 * gain_complement, rtol=1e-12)
    np.testing.assert_allclose(kf.kf.P[0, 2], 100.0 * gain_complement, rtol=1e-12)
    np.testing.assert_allclose(kf.kf.P[2, 0], 100.0 * gain_complement, rtol=1e-12)
    np.testing.assert_allclose(kf.kf.P[2, 2], 102.0 - 100.0 * k_vel, rtol=1e-12)
    np.testing.assert_allclose(kf.kf.P, kf.kf.P.T, rtol=0, atol=1e-12)


def test_matches_filterpy_bit_for_bit_when_it_is_installed():
    filterpy_kalman = pytest.importorskip("filterpy.kalman")
    rng = np.random.RandomState(3)
    cfg = SelectorConfig(fps=60.0, width=1280, height=720)
    ball = BallKalmanFilter(300.0, 200.0, cfg)
    ball.predict()                            # installs the projectile F and B
    ours = ball.kf
    theirs = filterpy_kalman.KalmanFilter(dim_x=4, dim_z=2)
    theirs.x, theirs.P, theirs.F, theirs.B, theirs.Q, theirs.H = (
        ours.x.copy(), ours.P.copy(), ours.F.copy(), ours.B.copy(), ours.Q.copy(), ours.H.copy())
    u = np.array([[cfg.gravity_px_per_frame2]])
    for _ in range(60):
        # The ball filter rescales R per detection; vary it here the same way.
        sigma = rng.uniform(2.0, 30.0)
        ours.R = np.diag([sigma * sigma, sigma * sigma])
        theirs.R = ours.R.copy()
        ours.predict(u)
        theirs.predict(u=u)
        z = np.array([[rng.uniform(0.0, 1280.0)], [rng.uniform(0.0, 720.0)]])
        ours.update(z)
        theirs.update(z)
        assert np.array_equal(ours.x, theirs.x)
        assert np.array_equal(ours.P, theirs.P)
