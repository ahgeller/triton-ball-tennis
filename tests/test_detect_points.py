"""finetune/training/detect.py: detect_points keeps each frame's best detection as a centre and confidence.

ball_detector itself needs the CUDA backend and is not built here.
"""
from pathlib import Path

from finetune.training.detect import detect_points
from finetune.data.workspace import VideoMeta


class StubDetector:
    def __init__(self, found):
        self.found = found
        self.calls = []

    def detect_clip(self, video, fps, width, height, total):
        self.calls.append((video, fps, width, height, total))
        return self.found


def test_detect_points_keeps_the_first_detection_of_each_frame_with_one():
    detector = StubDetector([[([10, 20, 14, 24], 0.9)], [], [([0, 0, 2, 2], 0.3), ([5, 5, 7, 7], 0.1)]])
    meta = VideoMeta(30, 1280, 720, 3)
    video = Path("clip.mp4")
    points = detect_points(detector, video, meta)
    assert points == {0: (12.0, 22.0, 0.9), 2: (1.0, 1.0, 0.3)}
    assert all(isinstance(value, float) for point in points.values() for value in point)
    assert detector.calls == [(video, 30, 1280, 720, 3)]


def test_detect_points_is_empty_when_nothing_was_found():
    assert detect_points(StubDetector([[], []]), Path("clip.mp4"), VideoMeta(60, 1920, 1080, 2)) == {}
