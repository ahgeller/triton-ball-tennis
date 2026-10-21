"""The court detector without an engine: keypoint remapping, line pairs and the disabled state."""
import numpy as np

from tennis_tracker import court_geometry
from tennis_tracker.config import Config
from tennis_tracker.court_detector import CourtDetector


def test_line_pairs_are_the_shared_court_geometry_constant():
    assert CourtDetector._LINE_PAIRS_14 is court_geometry.LINE_PAIRS_14
    pairs = CourtDetector._LINE_PAIRS_14
    assert len(pairs) == 19 and len(set(pairs)) == 19
    assert {index for pair in pairs for index in pair} == set(range(14))


def test_remap_writes_each_model_point_to_its_semantic_slot():
    pts = np.array([[i, i] for i in range(14)], np.float32)
    out = CourtDetector._remap_kpt_indices(pts)
    assert out.shape == pts.shape and out.dtype == np.float32
    for model_i, semantic_i in enumerate(CourtDetector._MODEL_TO_SEMANTIC_14):
        assert out[semantic_i].tolist() == [model_i, model_i]
    assert sorted(CourtDetector._MODEL_TO_SEMANTIC_14) == list(range(14))  # a permutation


def test_remap_leaves_other_keypoint_counts_alone():
    pts = np.array([[i, i] for i in range(13)], np.float32)
    out = CourtDetector._remap_kpt_indices(pts)
    np.testing.assert_array_equal(out, pts)


def test_detector_without_an_engine_is_disabled(capsys):
    detector = CourtDetector(Config(court_model_path=None))
    assert detector.session is None and detector.keypoints is None
    assert "court detection disabled" in capsys.readouterr().out
    frame = np.zeros((8, 8, 3), np.uint8)
    assert detector.detect(frame, 0) is None
    drawn = detector.draw(frame)
    assert drawn is frame and not frame.any()
