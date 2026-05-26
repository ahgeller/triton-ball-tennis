import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from calibrate_camera_intrinsics import (
    build_checkerboard_object_points,
    expand_image_patterns,
)


class CalibrateCameraIntrinsicsTests(unittest.TestCase):
    def test_checkerboard_object_points_use_inner_corner_grid(self):
        points = build_checkerboard_object_points(cols=3, rows=2, square_size_m=0.05)

        self.assertEqual(points.shape, (6, 3))
        np.testing.assert_allclose(points[0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(points[1], [0.05, 0.0, 0.0])
        np.testing.assert_allclose(points[2], [0.10, 0.0, 0.0])
        np.testing.assert_allclose(points[3], [0.0, 0.05, 0.0])
        np.testing.assert_allclose(points[-1], [0.10, 0.05, 0.0])

    def test_expand_image_patterns_returns_existing_files(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            a = root / "a.jpg"
            b = root / "b.jpg"
            a.write_bytes(b"x")
            b.write_bytes(b"x")

            paths = expand_image_patterns([str(root / "*.jpg")])

        self.assertEqual(len(paths), 2)
        self.assertTrue(all(p.suffix == ".jpg" for p in paths))


if __name__ == "__main__":
    unittest.main()
