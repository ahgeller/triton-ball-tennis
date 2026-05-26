import importlib.util
import math
import sys
import unittest
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from trajectory3d_v1 import Observation, SegmentWindow
import trajectory3d_v2 as trajectory3d_v2
from trajectory3d_v2 import (
    CameraCalibrationV2,
    COURT_TEMPLATE_14_M,
    calibrate_camera_from_court_v2,
    fit_hypothesis,
    project_points,
    reconstruct_span,
)


def _look_at_camera(eye, target, focal_px=1200.0, width=1920, height=1080):
    cv2 = __import__("cv2")
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    down = down / np.linalg.norm(down)
    R = np.vstack([right, down, forward])
    rvec, _ = cv2.Rodrigues(R)
    tvec = -R @ eye.reshape(3, 1)
    K = np.asarray(
        [[focal_px, 0.0, width / 2.0], [0.0, focal_px, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return CameraCalibrationV2(
        K=K,
        dist=np.zeros((5, 1), dtype=np.float64),
        rvec=rvec.reshape(3, 1),
        tvec=tvec.reshape(3, 1),
        reprojection_error_px=0.0,
        max_reprojection_error_px=0.0,
        inlier_count=14,
        image_point_count=14,
        template_point_count=14,
        quality="good",
    )


class Trajectory3DV2Tests(unittest.TestCase):
    def test_court_template_has_expected_semantic_layout(self):
        self.assertEqual(COURT_TEMPLATE_14_M[0], (0.0, 0.0, 0.0))
        self.assertEqual(COURT_TEMPLATE_14_M[3], (trajectory3d_v2.COURT_WIDTH_M, trajectory3d_v2.COURT_LENGTH_M, 0.0))
        self.assertAlmostEqual(COURT_TEMPLATE_14_M[4][0], trajectory3d_v2.SINGLES_INSET_M)
        self.assertAlmostEqual(COURT_TEMPLATE_14_M[12][0], trajectory3d_v2.COURT_WIDTH_M / 2.0)
        self.assertAlmostEqual(COURT_TEMPLATE_14_M[12][1], trajectory3d_v2.FAR_SERVICE_Y_M)
        self.assertAlmostEqual(COURT_TEMPLATE_14_M[13][1], trajectory3d_v2.NEAR_SERVICE_Y_M)

    @unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV not available in this Python env")
    def test_full_court_calibration_uses_all_visible_points(self):
        camera = _look_at_camera(
            eye=[trajectory3d_v2.COURT_WIDTH_M / 2.0, -16.0, 7.0],
            target=[trajectory3d_v2.COURT_WIDTH_M / 2.0, trajectory3d_v2.NET_Y_M, 1.0],
        )
        points = np.asarray([COURT_TEMPLATE_14_M[i] for i in range(14)], dtype=np.float64)
        projected = project_points(camera, points)
        kps = []
        for x, y in projected:
            kps.extend([float(x), float(y)])

        calibrated = calibrate_camera_from_court_v2(kps, 1920, 1080, focal_px=1200.0)

        self.assertIsNotNone(calibrated)
        assert calibrated is not None
        self.assertEqual(calibrated.image_point_count, 14)
        self.assertLess(calibrated.reprojection_error_px, 1.0)
        self.assertEqual(calibrated.quality, "good")

    @unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV not available in this Python env")
    @unittest.skipUnless(importlib.util.find_spec("scipy"), "SciPy not available in this Python env")
    def test_vertical_toss_prefers_toss_hypothesis(self):
        camera = _look_at_camera(
            eye=[trajectory3d_v2.COURT_WIDTH_M / 2.0, -15.0, 6.5],
            target=[trajectory3d_v2.COURT_WIDTH_M / 2.0, trajectory3d_v2.NET_Y_M, 1.2],
        )
        fps = 60.0
        obs = []
        for frame in range(0, 31, 3):
            t = frame / fps
            z = 1.55 + 5.9 * t - 0.5 * trajectory3d_v2.GRAVITY_MPS2 * t * t
            point = np.asarray([[trajectory3d_v2.COURT_WIDTH_M / 2.0, 19.0, z]], dtype=np.float64)
            x, y = project_points(camera, point)[0]
            obs.append(Observation(frame, t, float(x), float(y), "det", 1.0, 0.95, 0.95))

        context = {"near_segment_start": True, "has_toss_apex_candidate": True, "starts_near_player_box": False}
        toss = fit_hypothesis(obs, camera, 1920, 1080, 0, 30, "toss", context)
        lateral = fit_hypothesis(obs, camera, 1920, 1080, 0, 30, "lateral_flight", context)

        self.assertTrue(toss.success, toss.message)
        self.assertTrue(lateral.success, lateral.message)
        self.assertLess(toss.score, lateral.score)

    @unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV not available in this Python env")
    @unittest.skipUnless(importlib.util.find_spec("scipy"), "SciPy not available in this Python env")
    def test_reconstruct_span_selects_lateral_without_serve_context(self):
        camera = _look_at_camera(
            eye=[trajectory3d_v2.COURT_WIDTH_M / 2.0, -15.0, 6.5],
            target=[trajectory3d_v2.COURT_WIDTH_M / 2.0, trajectory3d_v2.NET_Y_M, 1.2],
        )
        fps = 60.0
        obs = []
        for frame in range(0, 31, 3):
            t = frame / fps
            point = np.asarray([[3.0 + 7.5 * t, 14.0 + 5.5 * t, 1.4 + 2.0 * t - 0.5 * trajectory3d_v2.GRAVITY_MPS2 * t * t]], dtype=np.float64)
            x, y = project_points(camera, point)[0]
            obs.append(Observation(frame, t, float(x), float(y), "det", 1.0, 0.95, 0.95))

        segment = SegmentWindow(0, 0.0, 2.0, 0.0, 2.0, -120, 120, 1.0, "merged")
        span = reconstruct_span(obs, camera, 1920, 1080, segment, 0, 30, [], fps)

        self.assertEqual(span.selected_name, "lateral_flight")
        self.assertGreater(span.confidence, 0.3)


if __name__ == "__main__":
    unittest.main()
