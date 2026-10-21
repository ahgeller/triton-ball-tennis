"""The player detector without an engine or boxmot: its disabled state and pure helpers."""
import sys
from unittest.mock import patch

import numpy as np

from tennis_tracker import detectors
from tennis_tracker.config import Config
from tennis_tracker.player_detector import PlayerDetector, PlayerPending, _load_bytetrack


def test_load_bytetrack_reports_why_boxmot_is_missing():
    with patch.dict(sys.modules, {"boxmot": None, "boxmot.trackers.bbox.bytetrack": None}):
        tracker, reason = _load_bytetrack()
    assert tracker is None
    assert isinstance(reason, str) and reason


def test_detector_without_an_engine_is_disabled_and_returns_no_boxes(capsys):
    detector = PlayerDetector(Config(player_model_path=None))
    assert detector.enabled is False
    assert detector.session is None and detector.tracker is None
    assert "player tracking disabled" in capsys.readouterr().out
    frame = np.zeros((8, 8, 3), np.uint8)
    assert detector.detect_async_start(frame, 0) is None
    assert detector.detect_async_finish(None) == []
    assert detector.detect(frame, 0) == []
    assert detector.get_player_dict() == {}


def test_keep_nearest_to_court_keeps_num_players_closest_boxes():
    detector = PlayerDetector(Config(player_model_path=None, num_players=2))
    rows = np.array([
        [500.0, 500.0, 520.0, 540.0, 0.9, 0.0],   # far from every keypoint
        [10.0, 10.0, 20.0, 30.0, 0.8, 0.0],       # on the first keypoint
        [100.0, 10.0, 110.0, 30.0, 0.7, 0.0],     # on the second keypoint
    ])
    kept = detector._keep_nearest_to_court(rows, [15.0, 20.0, 105.0, 20.0])
    assert kept.shape == (2, 6)
    assert kept[:, 4].tolist() == [0.8, 0.7]


def test_pending_defaults_to_a_coasting_frame():
    pending = PlayerPending(np.zeros((2, 2, 3), np.uint8), 3, 2, 2, None)
    assert pending.outputs is None and pending.input_tensor is None


def test_detectors_facade_re_exports_the_player_names():
    assert detectors.PlayerDetector is PlayerDetector
    assert detectors.PlayerPending is PlayerPending
