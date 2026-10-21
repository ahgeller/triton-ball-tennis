"""ball_in_play_selector.physics: the projectile prediction and the filter agree with each other
at either frame rate."""
import numpy as np

from ball_in_play_selector.config import SelectorConfig
from ball_in_play_selector.physics import BallKalmanFilter, predict_projectile, predict_projectile_velocity


def test_projectile_equal_timestamps_and_filter_consistency():
    cfg30 = SelectorConfig(fps=30).auto_scale()
    cfg60 = SelectorConfig(fps=60).auto_scale()
    np.testing.assert_allclose(predict_projectile((100, 100), (30, -10), 30, cfg30),
                               predict_projectile((100, 100), (15, -5), 60, cfg60), rtol=1e-10)
    np.testing.assert_allclose(predict_projectile_velocity((30, -10), 30, cfg30),
                               np.array(predict_projectile_velocity((15, -5), 60, cfg60)) * 2, rtol=1e-10)
    kf = BallKalmanFilter(100, 100, cfg30)
    kf.kf.x[2:, 0] = [30, -10]
    predicted = kf.predict_dt(5)
    np.testing.assert_allclose(predicted, predict_projectile((100, 100), (30, -10), 5, cfg30))
    for _ in range(5):
        actual = kf.predict()
    np.testing.assert_allclose(predicted, actual)
    assert np.all(np.linalg.eigvalsh(kf.kf.P) >= 0)
