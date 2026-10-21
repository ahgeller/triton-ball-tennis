"""The player detector without an engine or boxmot: its disabled state, the engine
output decode and the pure helpers."""
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from tennis_tracker.config import Config
from tennis_tracker.engines.player_detector import PlayerDetector, PlayerPending, _load_bytetrack


def test_load_bytetrack_reports_why_boxmot_is_missing():
    with patch.dict(sys.modules, {"boxmot": None}):
        tracker, reason = _load_bytetrack()
    assert tracker is None
    assert isinstance(reason, str) and reason


def test_configured_engine_without_boxmot_raises_like_a_missing_engine(tmp_path):
    with patch.dict(sys.modules, {"boxmot": None}):
        with pytest.raises(RuntimeError, match="requires boxmot.*pip install boxmot"):
            PlayerDetector(Config(player_model_path=str(tmp_path / "player.engine")))


def test_detector_without_an_engine_is_disabled_and_returns_no_boxes(capsys):
    detector = PlayerDetector(Config(player_model_path=None))
    assert detector.enabled is False
    assert detector.session is None and detector.tracker is None
    assert "player tracking disabled" in capsys.readouterr().out
    frame = np.zeros((8, 8, 3), np.uint8)
    assert detector.detect_async_start(frame, 0) is None
    assert detector.detect_async_finish(None) == []
    assert detector.detect(frame, 0) == []
    assert detector.player_boxes_by_slot() == {}


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


def _decoder(input_w=8, input_h=4, player_conf=0.5, player_iou=0.45) -> PlayerDetector:
    """A disabled detector given a fake session, enough for ``_decode_player_boxes``."""
    detector = PlayerDetector(Config(player_model_path=None, player_conf=player_conf, player_iou=player_iou))
    detector.session = SimpleNamespace(input_w=input_w, input_h=input_h)
    return detector


def test_decode_rescales_postprocessed_rows_to_the_frame_and_clips_them():
    rows = torch.tensor([[
        [1.0, 1.0, 3.0, 2.0, 0.9, 0.0],
        [6.0, 2.0, 9.0, 5.0, 0.8, 0.0],    # runs past the engine input, so past the frame
        [2.0, 2.0, 3.0, 3.0, 0.3, 0.0],    # below player_conf
        [2.0, 2.0, 3.0, 3.0, 0.7, 1.0],    # not the player class
    ]])
    out = _decoder()._decode_player_boxes(rows, frame_w=80, frame_h=40)   # 10x the engine input
    assert out.shape == (2, 6)
    assert out[0].tolist() == [10.0, 10.0, 30.0, 20.0, pytest.approx(0.9), 0.0]
    assert out[1].tolist() == [60.0, 20.0, 79.0, 39.0, pytest.approx(0.8), 0.0]


@pytest.mark.parametrize("channel_major", [True, False])
def test_decode_thresholds_and_suppresses_a_raw_head_in_either_layout(channel_major):
    anchors = [
        [4.0, 2.0, 2.0, 2.0, 0.9],   # cx, cy, w, h, person score
        [4.0, 2.0, 2.0, 2.0, 0.6],   # the same box again: suppressed by NMS
        [7.0, 3.0, 2.0, 2.0, 0.7],   # overhangs the input: clipped after rescaling
        [1.0, 1.0, 1.0, 1.0, 0.2],   # below player_conf
    ] + [[0.0] * 5 for _ in range(6)]  # real heads have far more anchors than channels
    head = torch.tensor(anchors).transpose(0, 1).unsqueeze(0)   # [1, 5, anchors]
    if not channel_major:
        head = head.transpose(1, 2)
    out = _decoder()._decode_player_boxes(head, frame_w=80, frame_h=40)
    assert out.shape == (2, 6)
    assert out[0].tolist() == [30.0, 10.0, 50.0, 30.0, pytest.approx(0.9), 0.0]
    assert out[1].tolist() == [60.0, 20.0, 79.0, 39.0, pytest.approx(0.7), 0.0]
    assert _decoder(player_conf=0.95)._decode_player_boxes(head, 80, 40).shape == (0, 6)


def test_pending_defaults_to_a_coasting_frame():
    pending = PlayerPending(np.zeros((2, 2, 3), np.uint8), 3, 2, 2, None)
    assert pending.outputs is None and pending.input_tensor is None
