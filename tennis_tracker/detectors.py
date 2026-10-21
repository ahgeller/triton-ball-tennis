"""Compatibility facade over the detector backend modules.

The backends live in ``engine_session`` (the TensorRT runtime session the
engine backends share), ``ball_detectors`` (GridTrackNet and the single-frame
TensorRT ball engine behind ``BallDetectorBackend``), ``player_detector``
(TensorRT players with ByteTrack association) and ``court_detector`` (the
court keypoint engine).  This module re-exports their public names so
``pipeline``, ``run_context``, ``pass1`` and ``finetune`` import unchanged;
private helpers are not re-exported.
"""
from __future__ import annotations

from .ball_detectors import (  # noqa: F401
    BallDetectorBackend,
    CachedDetections,
    Detection,
    FrameInput,
    GridTrackNetBallDetector,
    TensorRTBallPending,
    TensorRTRuntimeBallDetector,
    ball_engine_output_mode,
    decode_ball_predictions,
)
from .court_detector import CourtDetector  # noqa: F401
from .player_detector import PlayerDetector, PlayerPending  # noqa: F401
