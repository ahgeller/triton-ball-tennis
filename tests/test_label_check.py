"""finetune.data.label_check: one label file against its video's frame rate, size and length, on
synthetic label files under the ``workspace_dir`` fixture's folder."""
from pathlib import Path

import pytest

from finetune.data.label_check import check_label_file, matches_any

WIDTH, HEIGHT = 1920, 1080


def write_csv(path: Path, points) -> Path:
    """A label file from (frame, x, y) rows; an empty list writes the header alone."""
    lines = ["frame,ball_x,ball_y"] + [f"frame_{frame:03d},{x},{y}" for frame, x, y in points]
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def labels(workspace_dir):
    """The relocated workspace's labels folder."""
    return workspace_dir / "labels"


def label_file(labels: Path, clip, points):
    return write_csv(labels / f"{clip}_ball.csv", points)


def test_matches_any_takes_globs_and_everything_without_patterns():
    assert matches_any("video13", None) and matches_any("video13", [])
    assert matches_any("video13", ["video1*"]) and not matches_any("video13", ["grid*", "video2"])
    assert matches_any("video13", ["grid*", "video13"])


# --------------------------------------------------------------------------- check_label_file

def check(labels, points, fps=30.0, total=100):
    issues, rows = check_label_file(label_file(labels, "clip", points), fps, WIDTH, HEIGHT, total)
    return "; ".join(issues), rows


def test_clean_30fps_file_has_no_issues(labels):
    assert check(labels, [(f, 10 + f, 20) for f in range(8)]) == ("", 8)


def test_60fps_clip_labelled_on_every_frame(labels):
    issues, rows = check(labels, [(f, 10 + f, 20) for f in range(8)], fps=60.0)
    assert "needs every 2" in issues
    assert rows == 8


def test_60fps_clip_labelled_on_both_phases(labels):
    issues, _ = check(labels, [(f, 10 + f, 20) for f in (0, 2, 4, 7, 9)], fps=60.0)
    assert "labels mix even and odd" in issues
    assert "needs every" not in issues


def test_label_past_the_end_of_the_video(labels):
    issues, _ = check(labels, [(f, 10, 20) for f in range(5)], total=3)
    assert "beyond the video's 3 frames" in issues


def test_point_off_the_image(labels):
    issues, rows = check(labels, [(0, 10, 20), (1, 2000, 10), (2, 10, 20)])
    assert "outside 1920x1080" in issues
    assert rows == 2                           # reading stops at the bad point


def test_unsupported_frame_rate(labels):
    issues, _ = check(labels, [(f, 10, 20) for f in range(5)], fps=15.0)   # 24 counts as 30 (frame_rate)
    assert "neither 30 nor 60" in issues


def test_empty_file(labels):
    assert check(labels, []) == ("no label rows", 0)


def test_duplicates_and_gaps(labels):
    issues, _ = check(labels, [(0, 10, 20), (0, 11, 20), (20, 12, 20)])
    assert "duplicate frame rows" in issues
    assert "only 3 of 21 frames" in issues
