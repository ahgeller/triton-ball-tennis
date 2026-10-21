"""The court detector without an engine: keypoint remapping, line pairs and the disabled state."""
import numpy as np

from tennis_tracker.court import court_geometry
from tennis_tracker.config import Config
from tennis_tracker.engines.court_detector import CourtDetector


def test_line_pairs_join_the_fourteen_semantic_keypoints():
    pairs = court_geometry.LINE_PAIRS_14
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


def test_every_miss_path_reports_and_returns_the_cached_court(capsys):
    """An empty output, a low-confidence row, a row without keypoints and a row with too few
    keypoints all return the cached keypoints through the same report."""
    from types import SimpleNamespace

    import torch

    cfg = Config(court_model_path=None, print_court_raw=True, court_conf=0.5, court_detect_interval=1)
    detector = CourtDetector(cfg)
    outputs = {}
    detector.session = SimpleNamespace(
        preprocess_frame=lambda frame: (None, (1.0, 1.0)),
        forward_tensor=lambda t: [outputs["rows"]],
        wait=lambda: None,
    )
    detector.keypoints = [1.0, 2.0] * 14
    good = [0.0, 0.0, 1.0, 1.0, 0.9, 0.0] + [float(v) for pair in range(14) for v in (pair, pair, 1.0)]
    cases = {
        "empty output": torch.zeros((1, 0, 6)),
        "low confidence": torch.tensor([good[:4] + [0.2, 0.0] + good[6:]]).unsqueeze(0),
        "no keypoints": torch.tensor([good[:6]]).unsqueeze(0),
        "too few keypoints": torch.tensor([good[:6 + 3 * 3]]).unsqueeze(0),
    }
    for index, (name, rows) in enumerate(cases.items()):
        outputs["rows"] = rows
        capsys.readouterr()
        assert detector.detect(np.zeros((4, 4, 3), np.uint8), frame_index=index) == [1.0, 2.0] * 14, name
        assert f"[court] frame={index}: miss, using cached court" in capsys.readouterr().out, name
    outputs["rows"] = torch.tensor([good]).unsqueeze(0)
    assert detector.detect(np.zeros((4, 4, 3), np.uint8), frame_index=9)[:4] == [0.0, 0.0, 1.0, 1.0]
