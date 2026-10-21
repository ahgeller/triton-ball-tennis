"""clean_tracker.main: the command line becomes a Config; run() and the self-test are patched out."""
from unittest.mock import Mock, patch

import clean_tracker


def _run_main(tmp_path, argv):
    """main(argv) with run() replaced by a mock and OUTPUT pointed at tmp_path/out; (status, Config)."""
    run = Mock()
    with patch.object(clean_tracker, "run", run), patch.object(clean_tracker, "OUTPUT", tmp_path / "out"):
        status = clean_tracker.main(argv)
    assert run.call_count == 1
    return status, run.call_args.args[0]


FLAGS = ["--no-video", "--conf", "0.3", "--court", "--info"]


def test_cli_paths_reach_the_config_and_the_output_folder_is_made(tmp_path):
    status, cfg = _run_main(tmp_path, ["-i", str(tmp_path / "in.mp4"),
                                       "--tracking-json", str(tmp_path / "t.json"), *FLAGS])
    assert status == 0                                             # no --annotations: nothing to validate
    assert cfg.input_video == str((tmp_path / "in.mp4").resolve())
    assert cfg.tracking_json == str((tmp_path / "t.json").resolve())
    assert cfg.model_path.endswith("gridtracknet_weights_torch.npz")
    assert (tmp_path / "out").is_dir()


def test_cli_flags_reach_the_config(tmp_path):
    _, cfg = _run_main(tmp_path, FLAGS)
    assert cfg.save_tracking_video is False
    assert cfg.conf == 0.3
    assert cfg.draw_court_minimap is True and cfg.court_events is True     # --court computes and draws
    assert cfg.bounce_model_path.endswith("ctb_regr_bounce.cbm")
    assert cfg.info_timing is True
    assert cfg.print_selector_tracks is False
    assert cfg.motion_diagnostics is False


def test_cli_defaults(tmp_path):
    run = Mock()
    with patch.object(clean_tracker, "run", run), patch.object(clean_tracker, "OUTPUT", tmp_path):
        assert clean_tracker.main([]) == 0
    cfg = run.call_args.args[0]
    assert cfg.save_tracking_video is True
    assert cfg.conf == 0.5
    assert cfg.draw_court_minimap is False and cfg.court_events is False and cfg.info_timing is False
    assert cfg.input_video.endswith("pomona.mp4")


def test_self_test_flag_runs_the_suite_and_never_the_tracker(capsys):
    run = Mock()
    with patch.object(clean_tracker, "run", run), patch.object(clean_tracker, "_self_test", return_value=0):
        assert clean_tracker.main(["--self-test"]) == 0
    assert "clean_version self-test passed" in capsys.readouterr().out
    run.assert_not_called()


def test_failed_self_test_reports_its_status(capsys):
    with patch.object(clean_tracker, "_self_test", return_value=1):
        assert clean_tracker.main(["--self-test"]) == 1
    assert "passed" not in capsys.readouterr().out
