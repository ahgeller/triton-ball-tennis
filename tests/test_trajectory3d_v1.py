import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import now_main_pkg.trajectory3d_v1 as trajectory3d_v1
from now_main_pkg.trajectory3d_v1 import (
    LLCutSegment,
    Observation,
    SegmentWindow,
    generate_event_candidates,
    load_llc_cuts,
    map_cuts_to_video_windows,
)


class Trajectory3DV1Tests(unittest.TestCase):
    def test_parse_losslesscut_json5_llc(self):
        text = """
        {
          version: 2,
          mediaFileName: 'input.mp4',
          cutSegments: [
            { start: 10.5, end: 12.0, name: '', selected: true, },
            { start: 20.0, end: 22.0, name: 'skip', selected: false, },
          ],
        }
        """
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test.llc"
            path.write_text(text, encoding="utf-8")
            cuts, meta = load_llc_cuts(path)
        self.assertEqual(meta["version"], 2)
        self.assertEqual(meta["mediaFileName"], "input.mp4")
        self.assertEqual(len(cuts), 1)
        self.assertAlmostEqual(cuts[0].duration_sec, 1.5)

    def test_parse_losslesscut_json5_without_json5_package(self):
        text = """
        {
          version: 2,
          mediaFileName: 'input.mp4',
          cutSegments: [
            { start: 10.5, end: 12.0, name: '', selected: true, },
          ],
        }
        """
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test.llc"
            path.write_text(text, encoding="utf-8")
            with mock.patch.object(trajectory3d_v1, "json5", None):
                cuts, meta = load_llc_cuts(path)
        self.assertEqual(meta["version"], 2)
        self.assertEqual(len(cuts), 1)
        self.assertAlmostEqual(cuts[0].source_start_sec, 10.5)

    def test_merged_cut_mapping_accepts_small_duration_drift(self):
        cuts = [
            LLCutSegment(324.163884, 339.276295),
            LLCutSegment(357.042945, 376.728229),
            LLCutSegment(792.567477, 818.206360),
        ]
        windows, mapping = map_cuts_to_video_windows(
            cuts,
            fps=60.0,
            total_frames=3674,
            video_duration_sec=61.263033,
            timebase="auto",
        )
        self.assertEqual(mapping["mode"], "merged")
        self.assertGreater(mapping["mapping_confidence"], 0.8)
        self.assertEqual(len(windows), 3)
        self.assertEqual(windows[0].start_frame, 0)
        self.assertEqual(windows[-1].end_frame, 3673)

    def test_carry_only_v_shape_stays_low_confidence_candidate(self):
        segment = SegmentWindow(0, 0.0, 1.0, 0.0, 1.0, 0, 60, 1.0, "merged")
        obs = [
            Observation(10, 10 / 60, 100.0, 100.0, "carry", 0.2, 0.2, 0.2),
            Observation(11, 11 / 60, 105.0, 124.0, "carry", 0.2, 0.2, 0.2),
            Observation(12, 12 / 60, 110.0, 100.0, "carry", 0.2, 0.2, 0.2),
        ]
        events = generate_event_candidates(obs, segment, 1920, 1080, 60.0)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "bounce_candidate")
        self.assertTrue(events[0]["carry_only"])
        self.assertLessEqual(events[0]["confidence"], 0.44)

    def test_clip_edge_v_shape_is_marked_edge_event(self):
        segment = SegmentWindow(0, 0.0, 1.0, 0.0, 1.0, 0, 60, 1.0, "merged")
        obs = [
            Observation(1, 1 / 60, 100.0, 100.0, "det", 1.0, 0.9, 0.9),
            Observation(2, 2 / 60, 105.0, 124.0, "det", 1.0, 0.9, 0.9),
            Observation(3, 3 / 60, 110.0, 100.0, "det", 1.0, 0.9, 0.9),
        ]
        events = generate_event_candidates(obs, segment, 1920, 1080, 60.0)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "bounce_candidate")
        self.assertTrue(events[0]["edge_event"])
        self.assertLessEqual(events[0]["confidence"], 0.58)

    @unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV not available in this Python env")
    def test_cv2_available_for_future_calibration_checks(self):
        self.assertIsNotNone(importlib.util.find_spec("cv2"))


if __name__ == "__main__":
    unittest.main()
