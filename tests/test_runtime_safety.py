import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from ball_in_play_selector.core import _find_motion_blob
from tennis_tracker.detectors import TensorRTRuntimeBallDetector
from tennis_tracker.motion import _pack_mask_u8, _unpack_mask_u8
from tennis_tracker.pipeline import _detector_can_overlap, _validate_io_paths
from tennis_tracker.video_io import _PinnedFrameUploader


def _config(input_path, output_path):
    return SimpleNamespace(
        input_video=str(input_path),
        output_video=str(output_path),
        save_tracking_video=True,
        save_motion_debug=False,
        save_yolo_input_debug=False,
        save_guide_video=False,
        save_motion_tracks_video=False,
        tracking_json=None,
    )


class RuntimeSafetyTests(unittest.TestCase):
    def test_output_cannot_overwrite_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "input.mp4"
            video.write_bytes(b"video")
            with self.assertRaisesRegex(ValueError, "must not overwrite"):
                _validate_io_paths(_config(video, video))

    def test_sparse_mask_round_trip_and_empty_elision(self):
        mask = np.zeros((20, 30), dtype=np.uint8)
        self.assertIsNone(_pack_mask_u8(mask))
        mask[4:8, 9:15] = 255
        packed = _pack_mask_u8(mask)
        self.assertEqual(packed[0], "roi")
        np.testing.assert_array_equal(_unpack_mask_u8(packed), mask)

    def test_motion_blob_uses_centroid_coordinates(self):
        mask = np.zeros((30, 30), dtype=np.uint8)
        cv2.circle(mask, (10, 10), 3, 255, -1)
        blob = _find_motion_blob(mask, 10, 10, 8, 10, 10, (0, 0), 1, 100)
        self.assertIsNotNone(blob)
        self.assertAlmostEqual(blob[0], 10.0)
        self.assertAlmostEqual(blob[1], 10.0)

    def test_ball_pending_owns_its_input_tensor(self):
        detector = object.__new__(TensorRTRuntimeBallDetector)
        detector.use_async = False
        detector.async_slots = 1
        detector._preprocess_frame = lambda frame: ("input-tensor", "scale")
        detector._forward_tensor = lambda tensor, out_slot=0: (["prediction"], "event")
        pending = detector.detect_async_start(np.zeros((2, 2, 3), dtype=np.uint8))
        self.assertEqual(pending["input_tensor"], "input-tensor")

    def test_overlap_requires_async_and_two_slots(self):
        self.assertFalse(_detector_can_overlap(SimpleNamespace(use_async=False, async_slots=3)))
        self.assertFalse(_detector_can_overlap(SimpleNamespace(use_async=True, async_slots=1)))
        self.assertTrue(_detector_can_overlap(SimpleNamespace(use_async=True, async_slots=2)))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_pinned_uploader_keeps_consecutive_frames_distinct(self):
        uploader = _PinnedFrameUploader(2, 2, torch.device("cuda:0"))
        first = uploader.upload_chw_f32(np.zeros((2, 2, 3), dtype=np.uint8))
        second = uploader.upload_chw_f32(np.full((2, 2, 3), 255, dtype=np.uint8))
        torch.cuda.synchronize()
        self.assertEqual(float(first.max()), 0.0)
        self.assertEqual(float(second.min()), 1.0)


if __name__ == "__main__":
    unittest.main()
