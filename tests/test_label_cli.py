"""finetune/labeling/label_cli.py: the label tool's command line, on a synthetic tool and sources;
nothing is opened and no window is made."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from finetune.labeling import label_cli
from finetune.labeling.label_cli import announce, apply_settings, build_parser, open_sources, qt_backend, starting_cursor
from finetune.labeling.label_draw import FILTERS
from tests._helpers import fake_source, fake_tool


def source(clip, count=4, cursor=0, flagged=0):
    """A clip as the command line sees it: with a video name, its audit's flag count and a zoom."""
    return fake_source(clip, count, 64, 36, origin=(30.0, 20.0), step=(1.0, 1.0), conf=0.9, cursor=cursor,
                       video=SimpleNamespace(name=f"{clip}.mp4"), flagged=flagged, zoom=2.0)


def tool_over(*sources):
    return fake_tool(*sources, follow=False)


def test_parser_defaults_to_the_workspace_folders_and_takes_a_range():
    args = build_parser().parse_args(["video22..video25", "--zoom", "3"])
    assert args.clips == ["video22..video25"] and args.zoom == 3.0 and not args.grid and not args.uncertain
    assert args.videos.name == "videos" and args.labels.name == "labels"
    with pytest.raises(SystemExit):
        build_parser().parse_args([])                              # a clip is required


def test_qt_backend_says_what_to_install_when_the_windows_are_not_qts(capsys):
    with patch.object(label_cli.win, "highgui_backend", return_value="QT6"):
        assert qt_backend()
    with patch.object(label_cli.win, "highgui_backend", return_value="WIN32UI"):
        assert not qt_backend()
    assert "conda install -c conda-forge opencv" in capsys.readouterr().err


def test_open_sources_keeps_the_names_and_drops_clips_that_are_not_ready(tmp_path):
    ready = source("b")
    with patch.object(label_cli, "expand_clip_ranges", return_value=["a", "b"]), \
            patch.object(label_cli, "build_source", side_effect=lambda name, videos, labels: ready if name == "b" else None):
        names, sources = open_sources(build_parser().parse_args(["a..b"]))
    assert names == ["a", "b"] and sources == [ready]


def test_a_single_clip_resumes_where_it_was_and_a_run_at_the_first_unfinished_frame():
    one = source("a", cursor=3)
    assert starting_cursor(tool_over(one).session, [one]) == 3
    first, second = source("a"), source("b")
    for row in first.rows:
        row.reviewed = True
    tool = tool_over(first, second)
    assert starting_cursor(tool.session, [first, second]) == 4                       # clip b's first row


def test_settings_apply_only_where_they_still_make_sense():
    tool = tool_over(source("a"))
    apply_settings(tool, {"filter": FILTERS[1], "inset": False, "trail": False, "grid_size": 16})
    assert tool.session.filter == 1 and not tool.session.inset and not tool.session.trail and tool.grid_view.size == 16
    apply_settings(tool, {"filter": "no such filter", "grid_size": 5})
    assert tool.session.filter == 1 and tool.grid_view.size == 16                         # unknown values: unchanged


def test_announce_describes_one_clip_a_run_and_the_walk(capsys):
    one = source("a", flagged=2)
    announce(tool_over(one).session, [one], ["a"], uncertain=False)
    out = capsys.readouterr().out
    assert "[open] a.mp4: 4 label frames" in out and "[flag] 2 frame(s)" in out
    first, second = source("a"), source("b")
    tool = tool_over(first, second)
    tool.session.walk = [1, 5]
    announce(tool.session, [first, second], ["a", "b"], uncertain=True)
    out = capsys.readouterr().out
    assert "[open] 2 clips as one run: a .. b, 8 label frames" in out
    assert "[only ] stepping through 2 uncertain frame(s) of 8" in out and "[flag]" not in out


def test_main_stops_before_opening_anything_without_a_qt_build():
    with patch.object(label_cli, "qt_backend", return_value=False), patch.object(label_cli, "build_parser") as parser:
        assert label_cli.main() == 2
    assert not parser.called
