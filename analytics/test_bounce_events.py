"""Synthetic-trajectory tests for analytics/bounce_events.py.

Run: python -m unittest discover -s analytics
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bounce_events  # noqa: E402

FPS = 30.0
W, H = 1920, 1080

# Court trapezoid (image px): far side narrow, near side wide.
TL, TR = (710.0, 320.0), (1210.0, 320.0)
BL, BR = (360.0, 950.0), (1560.0, 950.0)


def _court_keypoints_raw() -> list:
    """Raw-order keypoints: corners at indices 0=TL, 3=TR, 4=BL, 7=BR."""
    kps = [0.0] * 16
    for idx, (x, y) in ((0, TL), (3, TR), (4, BL), (7, BR)):
        kps[idx * 2] = x
        kps[idx * 2 + 1] = y
    # Fill remaining slots with plausible in-court points so they're non-zero.
    for idx, (x, y) in ((1, (875.0, 320.0)), (2, (1040.0, 320.0)),
                        (5, (760.0, 950.0)), (6, (1160.0, 950.0))):
        kps[idx * 2] = x
        kps[idx * 2 + 1] = y
    return kps


def _arc_y(frame: int, f0: int, f1: int, y_bounce: float, y_apex: float) -> float:
    """Parabolic image-y between two ground contacts (max y at both ends)."""
    mid = 0.5 * (f0 + f1)
    half = max(1.0, 0.5 * (f1 - f0))
    t = (frame - mid) / half  # -1..1
    return y_apex + (y_bounce - y_apex) * t * t


def build_synthetic_tracking(bounce_frames=(30, 70), total=110) -> dict:
    frames = []
    x0, x1 = 600.0, 1300.0
    y_bounce, y_apex = 820.0, 500.0
    contacts = [0, *bounce_frames, total - 1]
    player_box = [1250.0, 520.0, 1400.0, 860.0]  # near the trajectory end (hit)
    for f in range(total):
        seg = 0
        while seg + 1 < len(contacts) - 1 and f > contacts[seg + 1]:
            seg += 1
        y = _arc_y(f, contacts[seg], contacts[seg + 1], y_bounce, y_apex)
        x = x0 + (x1 - x0) * f / (total - 1)
        frames.append({
            "frame": f, "present": True, "x": x, "y": y,
            "conf": 0.8, "source": "det", "interpolated": False,
            "player_boxes": {"p1": player_box},
        })
    return {
        "video": {"input": "synthetic.mp4", "fps": FPS, "width": W, "height": H,
                  "total_frames": total},
        "last_valid_court_keypoints": _court_keypoints_raw(),
        "frames": frames,
    }


class BounceDetectionTest(unittest.TestCase):
    def test_detects_bounces_at_known_frames(self):
        tracking = build_synthetic_tracking()
        fps = FPS
        homog = bounce_events.build_court_homography(
            tracking["last_valid_court_keypoints"], W, H)
        self.assertIsNotNone(homog)
        points = bounce_events._extract_points(tracking, fps)
        events = bounce_events.detect_events(points, fps, W, H, homography=homog)
        bounces = [e for e in events if e["type"] == "bounce"]
        self.assertGreaterEqual(len(bounces), 2)
        found = sorted(e["frame"] for e in bounces)
        for target in (30, 70):
            self.assertTrue(any(abs(f - target) <= 2 for f in found),
                            f"no bounce within 2 frames of {target}: {found}")

    def test_bounce_court_position_inside(self):
        tracking = build_synthetic_tracking()
        homog = bounce_events.build_court_homography(
            tracking["last_valid_court_keypoints"], W, H)
        points = bounce_events._extract_points(tracking, FPS)
        events = bounce_events.detect_events(points, FPS, W, H, homography=homog)
        bounces = [e for e in events if e["type"] == "bounce" and e.get("court")]
        self.assertTrue(bounces)
        for ev in bounces:
            court = ev["court"]
            self.assertTrue(court["in_doubles_court"],
                            f"bounce at frame {ev['frame']} mapped outside: {court}")
            self.assertTrue(0.0 <= court["u"] <= 1.0)
            self.assertTrue(0.0 <= court["v"] <= 1.0)

    def test_homography_corner_roundtrip(self):
        homog = bounce_events.build_court_homography(_court_keypoints_raw(), W, H)
        # TL corner -> (0,0); BR corner -> (1,1)
        for (px, py), (eu, ev_) in ((TL, (0.0, 0.0)), (BR, (1.0, 1.0))):
            u, v = bounce_events._apply_h(homog["H_i2c"], px, py)
            self.assertAlmostEqual(u, eu, places=4)
            self.assertAlmostEqual(v, ev_, places=4)

    def test_annotation_scoring(self):
        tracking = build_synthetic_tracking()
        with tempfile.TemporaryDirectory() as td:
            tr_path = Path(td) / "tracking.json"
            out_path = Path(td) / "bounces.json"
            ann_path = Path(td) / "ann.json"
            tr_path.write_text(json.dumps(tracking), encoding="utf-8")
            ann_path.write_text(json.dumps({
                "video": "synthetic.mp4",
                "ball": [],
                "events": [{"frame": 30, "type": "bounce"},
                           {"frame": 70, "type": "bounce"}],
            }), encoding="utf-8")
            result = bounce_events.run(str(tr_path), str(out_path),
                                       annotations_json=str(ann_path))
            score = result["annotation_score"]["by_type"]["bounce"]
            self.assertEqual(score["labeled"], 2)
            self.assertGreaterEqual(score["recall"], 0.99)

    def test_no_events_on_straight_line(self):
        frames = [{"frame": f, "present": True, "x": 500.0 + 8.0 * f,
                   "y": 400.0 + 2.0 * f, "conf": 0.8, "source": "det",
                   "player_boxes": {}} for f in range(80)]
        tracking = {"video": {"fps": FPS, "width": W, "height": H, "total_frames": 80},
                    "last_valid_court_keypoints": _court_keypoints_raw(),
                    "frames": frames}
        points = bounce_events._extract_points(tracking, FPS)
        events = bounce_events.detect_events(points, FPS, W, H)
        self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()
