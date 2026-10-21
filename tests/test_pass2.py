"""tennis_tracker/pass2.py: the renderer's per-frame court model, on the golden clip with the
golden harness's fakes."""
from tennis_tracker import pass2
from tests.test_pipeline_run import FULL_OVERRIDES, FakeCourtDetector, clip, run_pipeline  # noqa: F401


def test_frames_before_the_first_court_detection_are_drawn_without_a_court_model(
        tmp_path, monkeypatch, fake_cuda, clip):
    # The ground model used to start from the clip's last keypoints, so the opening frames of a
    # panning clip drew the net line and the minimap with the end-of-clip homography.
    has_model = []
    draw_court_layer = pass2.draw_court_layer

    def recording(targets, cfg, court_kps, ground_model, court_polygon):
        has_model.append(ground_model is not None)
        draw_court_layer(targets, cfg, court_kps, ground_model, court_polygon)

    monkeypatch.setattr(pass2, "draw_court_layer", recording)
    monkeypatch.setattr(FakeCourtDetector, "first_detection", 5)
    result = run_pipeline(tmp_path, monkeypatch, clip, **FULL_OVERRIDES)
    assert len(has_model) == result["raw_json"]["video"]["total_frames"]
    assert has_model[:5] == [False] * 5 and all(has_model[5:])
