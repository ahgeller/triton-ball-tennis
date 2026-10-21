"""Running the raw GridTrackNet prepass over a whole clip, for the scripts that draft, audit,
realign or score labels against it."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Tuple

from finetune.workspace import VideoMeta

if TYPE_CHECKING:
    from tennis_tracker.ball_detectors import GridTrackNetBallDetector

BallPoint = Tuple[float, float, float]     # x, y (native pixels) and confidence


def ball_detector(weights: Path, conf: float = 0.0, device: str = "0",
                  source_stride: Optional[int] = None) -> "GridTrackNetBallDetector":
    """A GridTrackNet detector on `weights`, keeping every candidate at or above `conf`. It sees
    the same five RGB frames the tracker feeds the model; `detect_clip` runs the prepass in the
    foreground, so these scripts get every frame's answer at once. `source_stride` forces the
    frame spacing fed to the model (None: the training cadence, 2 at 60 FPS and 1 at 30)."""
    import torch
    from tennis_tracker.config import Config
    from tennis_tracker import ball_detectors as backends
    # These scripts own their process: let cuDNN autotune the fixed-shape convolutions.
    torch.backends.cudnn.benchmark = True
    cfg = Config(conf=conf, device=device, gridtracknet_source_stride=source_stride)
    return backends.GridTrackNetBallDetector(str(weights), cfg)


def detect_points(detector, video: Path, meta: VideoMeta) -> Dict[int, BallPoint]:
    """frame -> the best detection's centre and confidence, for every frame that had one."""
    found = detector.detect_clip(video, meta.fps, meta.width, meta.height, meta.total)
    points: Dict[int, BallPoint] = {}
    for frame, detections in enumerate(found):
        if detections:
            box, confidence = detections[0]
            points[frame] = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0, float(confidence))
    return points
