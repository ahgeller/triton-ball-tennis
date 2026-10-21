"""finetune/refiner/hints.py, the ``ft.py hints`` command line: each subcommand reaches the
function it names with the parsed arguments, the sheet's default pair of versions, and the usage
and ValueError paths. The training, the version store and the sheet are replaced, so nothing is
trained and no workspace or models folder is read."""
from pathlib import Path
from typing import List, Optional, Tuple

import pytest

from finetune.refiner import hints, refiner_training


@pytest.fixture
def calls(monkeypatch) -> List[Tuple]:
    """Record every call main makes into the refiner; each fake answers as a success."""
    made: List[Tuple] = []

    def train(epochs):
        made.append(("train", epochs))
        return Path("v9.pt")

    def make_current(name):
        made.append(("use", name))
        return Path("models") / f"{name}.pt"

    def sheet(clip, names, count):
        made.append(("sheet", clip, names, count))
        return Path("sheet.png")

    monkeypatch.setattr(refiner_training, "train", train)
    monkeypatch.setattr(hints, "make_current", make_current)
    monkeypatch.setattr(hints, "sheet", sheet)
    monkeypatch.setattr(hints, "version_lines", lambda: ["  v1", "* v2"])
    monkeypatch.setattr(hints, "current", lambda: Path("v2.pt"))
    monkeypatch.setattr(hints, "versions", lambda: [Path("v1.pt"), Path("v2.pt")])
    return made


@pytest.mark.parametrize("argv, expected", [
    (["train"], ("train", refiner_training.EPOCHS)),
    (["train", "--epochs", "3"], ("train", 3)),
    (["use", "v1"], ("use", "v1")),
    (["sheet", "rally", "v1", "centroid", "--count", "5"], ("sheet", "rally", ["v1", "centroid"], 5)),
    (["sheet", "rally"], ("sheet", "rally", ["v2", "v1"], 36)),      # default: the current one and the one before
])
def test_each_subcommand_dispatches_with_its_parsed_arguments(calls, argv, expected):
    assert hints.main(argv) == 0
    assert calls == [expected]


def test_list_prints_the_version_lines(calls, capsys):
    assert hints.main(["list"]) == 0
    assert capsys.readouterr().out == "  v1\n* v2\n"
    assert calls == []


def test_use_reports_the_version_the_label_tool_now_uses(calls, capsys):
    assert hints.main(["use", "v1"]) == 0
    assert capsys.readouterr().out == "the label tool now uses v1\n"


@pytest.mark.parametrize("command", [["train"], ["sheet", "rally", "v1", "v2"]])
def test_a_run_that_produces_nothing_exits_1(monkeypatch, calls, command):
    monkeypatch.setattr(refiner_training, "train", lambda epochs: None)
    monkeypatch.setattr(hints, "sheet", lambda clip, names, count: None)
    assert hints.main(command) == 1


@pytest.mark.parametrize("current, versions, expected", [
    ("v3", ["v1", "v2", "v3"], ["v3", "v2"]),
    (None, ["v1"], ["centroid", "v1"]),                 # versions, none chosen: compared with refine.py alone
    ("v1", ["v1"], ["v1", "centroid"]),                 # one version: against the centroid
    (None, [], ["centroid", "centroid"]),
])
def test_sheet_names_defaults_to_the_current_version_and_the_one_before(monkeypatch, current, versions, expected):
    monkeypatch.setattr(hints, "current", lambda: Path(f"{current}.pt") if current else None)
    monkeypatch.setattr(hints, "versions", lambda: [Path(f"{name}.pt") for name in versions])
    assert hints.sheet_names([]) == expected
    assert hints.sheet_names(["a", "b"]) == ["a", "b"]


@pytest.mark.parametrize("argv", [["sheet", "rally", "v1"], ["sheet", "rally", "v1", "v2", "v3"]])
def test_a_sheet_naming_one_or_three_versions_is_an_error(calls, capsys, argv):
    assert hints.main(argv) == 1
    assert capsys.readouterr().err == "[error] sheet takes two version names\n"
    assert calls == []


def test_an_unknown_version_is_reported_not_raised(monkeypatch, calls, capsys):
    def refuse(name: str) -> Optional[Path]:
        raise ValueError(f"no version named {name}")
    monkeypatch.setattr(hints, "make_current", refuse)
    assert hints.main(["use", "v7"]) == 1
    captured = capsys.readouterr()
    assert captured.err == "[error] no version named v7\n" and "now uses" not in captured.out


@pytest.mark.parametrize("argv", [[], ["fit"], ["use"], ["train", "--epochs", "x"], ["sheet"], ["list", "extra"]])
def test_usage_errors_exit_2_before_anything_runs(calls, capsys, argv):
    with pytest.raises(SystemExit) as stop:
        hints.main(argv)
    assert stop.value.code == 2
    assert "usage:" in capsys.readouterr().err
    assert calls == []
