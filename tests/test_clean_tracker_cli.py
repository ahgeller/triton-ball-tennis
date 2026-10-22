"""clean_tracker.main: the command line becomes a Config; run() and the self-test are patched out."""
import os
from unittest.mock import Mock, patch

import pytest

import clean_tracker
from clean_tracker import resolve_output_paths


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
    assert cfg.draw_court_minimap is True
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
    assert cfg.draw_court_minimap is False and cfg.info_timing is False
    assert cfg.input_video.endswith("pomona.mp4")
    assert cfg.tracking_json == str(tmp_path / "tracking.json")


def test_output_paths_name_a_file_as_given_and_default_the_json_to_the_output_folder(tmp_path):
    with patch.object(clean_tracker, "OUTPUT", tmp_path / "out"):
        video, tracking = resolve_output_paths("clips/video47.mp4", str(tmp_path / "run.mp4"), None)
    assert video == (tmp_path / "run.mp4").resolve()
    assert tracking == tmp_path / "out" / "tracking.json"
    _, given = resolve_output_paths("clips/video47.mp4", str(tmp_path / "run.mp4"), str(tmp_path / "t.json"))
    assert given == (tmp_path / "t.json").resolve()


def test_output_paths_in_an_existing_folder_are_named_after_the_input(tmp_path):
    # The folder exists, so it is a folder even without a trailing separator.
    folder = tmp_path / "output"
    folder.mkdir()
    video, tracking = resolve_output_paths("clips/video47.mp4", str(folder), None)
    assert video == folder / "video47_tracked.mp4"
    assert tracking == folder / "video47_tracking.json"
    _, given = resolve_output_paths("clips/video47.mp4", str(folder), str(tmp_path / "t.json"))
    assert given == (tmp_path / "t.json").resolve()


@pytest.mark.parametrize("separator", sorted({os.sep, os.altsep or os.sep}))
def test_output_paths_treat_a_trailing_separator_as_a_folder_to_create(tmp_path, separator):
    folder = tmp_path / "later"
    assert not folder.exists()
    video, tracking = resolve_output_paths("clips/video47.mp4", str(folder) + separator, None)
    assert video == folder / "video47_tracked.mp4"
    assert tracking == folder / "video47_tracking.json"


def test_cli_output_folder_reaches_the_config_and_is_announced(tmp_path, capsys):
    # The user's case: ``--output <folder>`` used to send FFmpeg a path with no
    # container suffix, which it refused after pass 1 and the selector.
    folder = tmp_path / "output"
    folder.mkdir()
    _, cfg = _run_main(tmp_path, ["-i", str(tmp_path / "video47.mp4"), "-o", str(folder)])
    assert cfg.output_video == str(folder / "video47_tracked.mp4")
    assert cfg.tracking_json == str(folder / "video47_tracking.json")
    out = capsys.readouterr().out
    assert f"[init] Tracking video: {folder / 'video47_tracked.mp4'}" in out
    assert f"[init] Tracking JSON: {folder / 'video47_tracking.json'}" in out


def test_cli_no_video_announces_only_the_json(tmp_path, capsys):
    _, cfg = _run_main(tmp_path, ["-i", str(tmp_path / "video47.mp4"), "-o", str(tmp_path / "output") + os.sep,
                                  "--no-video"])
    assert cfg.output_video == str(tmp_path / "output" / "video47_tracked.mp4")
    out = capsys.readouterr().out
    assert "[init] Tracking video" not in out and "[init] Tracking JSON" in out


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
