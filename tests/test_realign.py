"""finetune/realign.py on synthetic detections and labels in a temporary folder.

The end-to-end tests are an oracle of what ``main()`` prints, writes and returns for one fixture:
a 240-row label file whose ball the (stubbed) detector sees three video frames later. They were
recorded from the script before its main() was split into pieces, so they are the regression that
the split changed no output. Every path the script reads or writes is patched into ``tmp_path``:
no workspace label, review file or video is ever touched.
"""
import io
import json
import sys
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from typing import Iterator, List, Optional, Tuple
from unittest.mock import patch

import numpy as np
import pytest

from finetune import labels, realign

CLIP = "synthetic_clip"
ROWS = 240                   # candidates() needs chunks of 20+ rows and main() asks for 8 chunks
SHIFT = 3                    # the video shows each labelled position SHIFT frames later
WIDTH, HEIGHT, TOTAL = 1280, 720, 300
SENTINEL_ROWS = (17, 150)    # rows that say "no ball"
PINNED_FRAME = 10


def ball_x(frame: int) -> int:
    """A triangle wave, 12 px per frame between x=100 and x=1180 (18 px at 1080p: a fast row)."""
    leg = (frame * 12) % 2160
    return 100 + leg if leg <= 1080 else 100 + (2160 - leg)


def label_rows() -> List[Tuple[int, str, str]]:
    sentinel = tuple(str(v) for v in labels.sentinel(WIDTH))
    return [(f, *sentinel) if f in SENTINEL_ROWS else (f, f"{ball_x(f)}", "360") for f in range(ROWS)]


def shifted_detections() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """What ``realign.detections`` would answer for a video running SHIFT frames behind the labels."""
    frames = [f + SHIFT for f in range(ROWS) if f not in SENTINEL_ROWS]
    return (np.array(frames, np.int32), np.array([ball_x(f - SHIFT) for f in frames], np.float32),
            np.array([360.0] * len(frames), np.float32), np.array([WIDTH, HEIGHT, TOTAL], np.int32))


def write_fixture(root: Path, reviewed: Optional[List[int]] = None) -> Path:
    """labels/<CLIP>_ball.csv (and the review sidecar when `reviewed` is given) under `root`."""
    folder = root / "labels"
    folder.mkdir()
    csv = folder / f"{CLIP}_ball.csv"
    csv.write_text("\n".join(["frame,ball_x,ball_y"] + [",".join([labels.frame_name(f), x, y])
                                                         for f, x, y in label_rows()]) + "\n")
    if reviewed is not None:
        (folder / f"{CLIP}_ball.review.json").write_text(json.dumps({"reviewed": reviewed}))
    return csv


@contextmanager
def bound_to(root: Path) -> Iterator[None]:
    """Point every workspace path realign reads or writes into `root` and stub the detector."""
    detections = shifted_detections()
    with patch.object(realign, "detections", lambda clip: detections), \
         patch.object(realign, "frame_size", lambda clip: (WIDTH, HEIGHT)), \
         patch.object(realign, "label_csv", lambda clip: root / "labels" / f"{clip}_ball.csv"), \
         patch("finetune.workspace.LABELS", root / "labels"), \
         patch.object(realign, "WS", root):
        yield


def run_main(root: Path, *args: str, out: Optional[io.StringIO] = None) -> Tuple[int, str]:
    """``main()`` as the script runs it, on ``sys.argv``; returns (exit code, stdout). Pass `out`
    to keep what was printed when main() raises."""
    out = io.StringIO() if out is None else out
    with bound_to(root), patch.object(sys, "argv", ["realign.py", *args]), redirect_stdout(out):
        code = realign.main()
    return code, out.getvalue()


def listing(root: Path) -> List[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*"))


# --------------------------------------------------------------------------- recorded output

HEADER = (f"{CLIP}: 240 label rows (238 visible) against {CLIP} (300 frames, 1280x720)\n"
          "  as it stands:  agree 0.00 on 233 judged, median 54.0 px\n")
SOLVED = ("  rate 1: 22 candidate offsets in -182..+8  -> agree 1.00, median 0.0 px\n"
          "  best: rate 1, agree 1.00 on 237 judged, median 0.0 px\n"
          "  1 segment(s):\n"
          "      label frames     0-239   -> video frames     3-242    (offset +3)\n")
DRY_RUN = "\n  (dry run - pass --apply to write it)\n"
WRITTEN = (f"\n  [write] {CLIP}_ball.csv: 240 of 240 rows kept\n"
           f"  next:  python finetune/ft.py check --audit --mark {CLIP}\n")
PINNED = ("  1 frame(s) you corrected by hand are pinned and will not move\n"
          "  rate 1: 22 candidate offsets in -182..+8  -> agree 0.97, median 0.0 px\n"
          "  best: rate 1, agree 0.97 on 235 judged, median 0.0 px\n"
          "  2 segment(s):\n"
          "      label frames     0-10    -> video frames     0-10     (offset +0)\n"
          "      label frames    11-239   -> video frames    14-242    (offset +3)\n"
          "  11 visible row(s) sit inside a cut the detector cannot place - dropped, not guessed\n")


# --------------------------------------------------------------------------- main() end to end

def test_fewer_than_two_clips_prints_the_doc_and_returns_2(tmp_path):
    write_fixture(tmp_path)
    code, out = run_main(tmp_path, CLIP)
    assert code == 2
    assert out == realign.__doc__ + "\n"


def test_dry_run_reports_the_offset_and_writes_nothing(tmp_path):
    csv = write_fixture(tmp_path)
    original = csv.read_bytes()
    before = listing(tmp_path)
    code, out = run_main(tmp_path, CLIP, CLIP)
    assert code == 0
    assert out == HEADER + SOLVED + DRY_RUN
    assert csv.read_bytes() == original
    assert listing(tmp_path) == before


def test_apply_backs_the_file_up_once_then_retimes_every_row(tmp_path):
    csv = write_fixture(tmp_path)
    original = csv.read_bytes()
    code, out = run_main(tmp_path, CLIP, CLIP, "--apply")
    assert code == 0
    assert out == HEADER + SOLVED + WRITTEN
    backup = csv.with_suffix(".csv.premisaligned")
    assert backup.read_bytes() == original
    assert listing(tmp_path) == ["labels", f"labels/{CLIP}_ball.csv", f"labels/{CLIP}_ball.csv.premisaligned"]
    rewritten = [(row.frame, row.x, row.y) for row in labels.read_label_rows(csv)]
    assert rewritten == [(f + SHIFT, float(x), float(y)) for f, x, y in label_rows()]
    assert all(not labels.is_visible(x, y, WIDTH, HEIGHT) for f, x, y in rewritten if f - SHIFT in SENTINEL_ROWS)


def test_reviewed_frame_is_pinned_and_the_alignment_steps_past_it(tmp_path):
    csv = write_fixture(tmp_path, reviewed=[PINNED_FRAME])
    original = csv.read_bytes()
    code, out = run_main(tmp_path, CLIP, CLIP)
    assert code == 0
    assert out == HEADER + PINNED + DRY_RUN
    assert csv.read_bytes() == original


def test_apply_refuses_to_drop_the_pinned_row_and_changes_nothing(tmp_path):
    csv = write_fixture(tmp_path, reviewed=[PINNED_FRAME])
    original = csv.read_bytes()
    before = listing(tmp_path)
    out = io.StringIO()
    with pytest.raises(ValueError, match="would drop a reviewed row; nothing changed"):
        run_main(tmp_path, CLIP, CLIP, "--apply", out=out)
    assert out.getvalue() == HEADER + PINNED
    assert csv.read_bytes() == original
    assert listing(tmp_path) == before                       # no .premisaligned backup either
    rows = {row.frame: (row.x, row.y) for row in labels.read_label_rows(csv)}
    assert rows[PINNED_FRAME] == (float(ball_x(PINNED_FRAME)), 360.0)


def test_no_visible_labels_is_an_early_exit(tmp_path):
    csv = write_fixture(tmp_path)
    sentinel = ",".join(str(v) for v in labels.sentinel(WIDTH))
    csv.write_text("frame,ball_x,ball_y\n" + "".join(f"{labels.frame_name(f)},{sentinel}\n" for f in range(ROWS)))
    code, out = run_main(tmp_path, CLIP, CLIP)
    assert code == 1
    assert out == "No visible labels provide alignment evidence; nothing changed\n"


def test_reviewed_labels_cannot_have_their_video_replaced(tmp_path):
    write_fixture(tmp_path, reviewed=[PINNED_FRAME])
    before = listing(tmp_path)
    code, out = run_main(tmp_path, CLIP, "other_clip", "--apply")
    assert code == 1
    assert out.endswith("Cannot replace the video under reviewed labels; resolve the source mapping manually first\n")
    assert listing(tmp_path) == before


# --------------------------------------------------------------------------- the pieces main() calls

def evidence() -> Tuple[np.ndarray, ...]:
    """(frames, xs, ys, fast, lut_x, lut_y) for the fixture, built the way main() builds them."""
    det_frames, det_x, det_y, _ = shifted_detections()
    lut_x, lut_y = realign.lookup_tables(det_frames, det_x, det_y, TOTAL)
    visible = [(f, float(x), float(y)) for f, x, y in label_rows() if labels.is_visible(float(x), float(y), WIDTH, HEIGHT)]
    frames = np.array([f for f, _, _ in visible], np.int64)
    xs = np.array([x for _, x, _ in visible], np.float32)
    ys = np.array([y for _, _, y in visible], np.float32)
    step = np.hypot(np.diff(xs, prepend=xs[0]), np.diff(ys, prepend=ys[0])) * realign.scale_for_width(WIDTH)
    return frames, xs, ys, step >= realign.FAST_PX, lut_x, lut_y


def test_parse_args_takes_two_clips_and_the_two_flags():
    assert realign.parse_args(["realign.py", "a", "b", "--preview"]) == ("a", "b", True, False)
    assert realign.parse_args(["realign.py", "a", "b", "--apply"]) == ("a", "b", False, True)
    assert realign.parse_args(["realign.py", "x"]) is None
    assert realign.parse_args(["realign.py"]) is None


def test_solve_alignment_finds_the_constant_plus_three_offset_at_rate_1(capsys):
    frames, xs, ys, fast, lut_x, lut_y = evidence()
    assert len(frames) == 238 and realign.scale_for_width(WIDTH) == 1.5 and fast[1:].all() and not fast[0]
    pinned = np.zeros(len(frames), bool)
    best = realign.solve_alignment(frames, xs, ys, fast, lut_x, lut_y, TOTAL, 1.5, set(), pinned, chunks=8)
    assert best is not None
    result, rate, bases, chosen = best
    assert rate == 1.0
    assert (chosen == SHIFT).all()
    np.testing.assert_array_equal(bases, frames)
    assert result[0] == 1.0 and result[2] == 0.0
    assert capsys.readouterr().out == "  rate 1: 22 candidate offsets in -182..+8  -> agree 1.00, median 0.0 px\n"


def test_solve_alignment_is_none_when_no_rate_has_a_candidate_offset(capsys):
    frames, xs, ys, fast, lut_x, lut_y = evidence()
    pinned = np.zeros(len(frames), bool)
    with patch.object(realign, "candidates", lambda *args: np.array([], np.int64)):
        best = realign.solve_alignment(frames, xs, ys, fast, lut_x, lut_y, TOTAL, 1.5, set(), pinned, chunks=8)
    assert best is None
    assert capsys.readouterr().out == ""


def test_solve_alignment_keeps_a_pinned_row_at_offset_zero():
    frames, xs, ys, fast, lut_x, lut_y = evidence()
    fixed = {PINNED_FRAME}
    pinned = np.array([f in fixed for f in frames.tolist()], bool)
    best = realign.solve_alignment(frames, xs, ys, fast, lut_x, lut_y, TOTAL, 1.5, fixed, pinned, chunks=8)
    assert best is not None
    _, rate, _, chosen = best
    assert rate == 1.0
    assert chosen[frames <= PINNED_FRAME].tolist() == [0] * int((frames <= PINNED_FRAME).sum())
    assert (chosen[frames > PINNED_FRAME] == SHIFT).all()


def test_segments_groups_runs_of_one_offset():
    assert realign.segments(np.array([0, 1, 2, 3, 4]), np.array([0, 0, 2, 2, 2])) == [[0, 1, 0], [2, 4, 2]]
    assert realign.segments(np.array([5, 9]), np.array([3, 3])) == [[5, 9, 3]]
    assert realign.segments(np.array([], np.int64), np.array([], np.int64)) == []


def test_apply_alignment_backs_up_once_and_retimes_every_row(tmp_path):
    csv = write_fixture(tmp_path)
    original = csv.read_bytes()
    before = listing(tmp_path)
    entries = sorted(labels.read_label_rows(csv))
    frames = evidence()[0]
    chosen = np.full(len(frames), SHIFT, np.int64)
    with bound_to(tmp_path):
        code = realign.apply_alignment(entries, frames, chosen, set(), set(), 1.0, TOTAL,
                                       (WIDTH, HEIGHT), (WIDTH, HEIGHT), CLIP, CLIP)
    assert code == 0
    backup = csv.with_suffix(".csv.premisaligned")
    assert backup.read_bytes() == original
    assert listing(tmp_path) == before + [f"labels/{CLIP}_ball.csv.premisaligned"]
    rewritten = [(row.frame, row.x, row.y) for row in labels.read_label_rows(csv)]
    assert rewritten == [(f + SHIFT, float(x), float(y)) for f, x, y in label_rows()]
    assert [f for f, x, y in rewritten if not labels.is_visible(x, y, WIDTH, HEIGHT)] == [f + SHIFT for f in SENTINEL_ROWS]

    # a second apply re-times the already re-timed rows but leaves the first backup alone
    entries = sorted(labels.read_label_rows(csv))
    with bound_to(tmp_path):
        realign.apply_alignment(entries, frames + SHIFT, chosen, set(), set(), 1.0, TOTAL,
                                (WIDTH, HEIGHT), (WIDTH, HEIGHT), CLIP, CLIP)
    assert backup.read_bytes() == original
    assert listing(tmp_path) == before + [f"labels/{CLIP}_ball.csv.premisaligned"]
    assert [row.frame for row in labels.read_label_rows(csv)] == [f + 2 * SHIFT for f in range(ROWS)]


def test_apply_alignment_writes_nothing_when_a_reviewed_row_would_move(tmp_path):
    csv = write_fixture(tmp_path, reviewed=[PINNED_FRAME])
    original = csv.read_bytes()
    before = listing(tmp_path)
    entries = sorted(labels.read_label_rows(csv))
    frames = evidence()[0]
    chosen = np.full(len(frames), SHIFT, np.int64)
    with bound_to(tmp_path), pytest.raises(ValueError, match="would move a reviewed row; nothing changed"):
        realign.apply_alignment(entries, frames, chosen, set(), {PINNED_FRAME}, 1.0, TOTAL,
                                (WIDTH, HEIGHT), (WIDTH, HEIGHT), CLIP, CLIP)
    assert csv.read_bytes() == original
    assert listing(tmp_path) == before
