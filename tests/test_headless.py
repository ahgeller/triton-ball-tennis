"""tennis_tracker/headless.py: the one Config every programmatic caller of the tracker builds,
and the tracking-JSON shape taken from a RunResult; the pipeline itself is replaced."""
from pathlib import Path
from unittest.mock import patch

from tennis_tracker import headless
from tennis_tracker.config import Config
from tennis_tracker.paths import BOUNCE_MODEL, COURT_ENGINE, PLAYER_ENGINE
from tennis_tracker.export import RunResult


def test_headless_config_writes_json_and_no_video_with_the_bundled_engines():
    cfg = headless.headless_config(Path("v.mp4"), model_path=Path("w.npz"), conf=0.3, device="1", tracking_json=Path("out/v.json"))
    assert isinstance(cfg, Config)
    assert (cfg.input_video, cfg.model_path, cfg.conf, cfg.device) == ("v.mp4", "w.npz", 0.3, "1")
    assert cfg.tracking_json == str(Path("out/v.json")) and cfg.output_video == str(Path("out/v.mp4"))
    assert cfg.player_model_path == str(PLAYER_ENGINE) and cfg.court_model_path == str(COURT_ENGINE)
    assert cfg.save_tracking_video is False and cfg.print_selector_tracks is False
    assert cfg.court_events is False and cfg.draw_court_minimap is False
    assert cfg.bounce_model_path == str(BOUNCE_MODEL)          # named even when the events are off
    named = headless.headless_config(Path("v.mp4"), model_path=Path("w.npz"), conf=0.5, device="0",
                                     tracking_json=Path("v.json"), output_video=Path("x/v_grid.mp4"))
    assert named.output_video == str(Path("x/v_grid.mp4"))
    with_events = headless.headless_config(Path("v.mp4"), model_path=Path("w.npz"), conf=0.5, device="0",
                                           tracking_json=Path("v.json"), court_events=True)
    assert with_events.court_events is True and with_events.draw_court_minimap is False   # computed, not drawn


def _result(frame_count: int) -> RunResult:
    return RunResult(per_frame=[None] * frame_count, chosen_track=None, all_tracks=[], frame_count=frame_count,
                     court_keypoints=[], player_boxes=[], last_valid_court_keypoints=[1.0, 2.0], elapsed_sec=0.0,
                     timing={}, pass2_frames_rendered=0, outputs={})


def test_run_headless_runs_the_pipeline_on_that_config_and_the_payload_mirrors_the_json(monkeypatch):
    seen = {}

    def run(cfg):
        seen["cfg"] = cfg
        return _result(2)
    with patch("tennis_tracker.pipeline.run", run):
        result = headless.run_headless(Path("v.mp4"), model_path=Path("w.npz"), conf=0.5, device="0", tracking_json=Path("v.json"))
    assert seen["cfg"] == headless.headless_config(Path("v.mp4"), model_path=Path("w.npz"), conf=0.5, device="0",
                                                   tracking_json=Path("v.json"))
    payload = headless.tracking_payload(result)
    assert set(payload) == {"frames", "last_valid_court_keypoints"} and payload["last_valid_court_keypoints"] == [1.0, 2.0]
    assert [row["frame"] for row in payload["frames"]] == [0, 1] and not any(row["present"] for row in payload["frames"])
    with patch("tennis_tracker.pipeline.run", run):
        headless.run_headless(Path("v.mp4"), model_path=Path("w.npz"), conf=0.5, device="0", tracking_json=Path("v.json"),
                              court_events=True)
    assert seen["cfg"] == headless.headless_config(Path("v.mp4"), model_path=Path("w.npz"), conf=0.5, device="0",
                                                   tracking_json=Path("v.json"), court_events=True)
