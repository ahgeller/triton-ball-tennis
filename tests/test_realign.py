"""finetune/data/realign.py on synthetic detections and labels in a temporary folder.

The end-to-end tests are an oracle of what ``main()`` prints, writes and returns for one fixture:
a 240-row label file whose ball the (stubbed) detector sees three video frames later. They were
recorded from the script before its main() was split into pieces, so they are the regression that
the split (and the move onto the Tracks/Evidence bundles) changed no output. Every path the
script reads or writes is patched into ``tmp_path``: no workspace label, review file or video is
ever touched.
"""
import io
import json
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from typing import Iterator, List, Optional, Tuple
from unittest.mock import patch

import numpy as np
import pytest

from finetune.data import labels, realign
from finetune.data.alignment import Evidence, Tracks
from finetune.data.workspace import VideoMeta

CLIP = "synthetic_clip"
ROWS = 240                   # candidates() needs chunks of 20+ rows and main() asks for 8 chunks
SHIFT = 3                    # the video shows each labelled position SHIFT frames later
WIDTH, HEIGHT, TOTAL = 1280, 720, 300
META = VideoMeta(30.0, WIDTH, HEIGHT, TOTAL)
SENTINEL_ROWS = (17, 150)    # rows that say "no ball"
PINNED_FRAME = 10


def ball_x(frame: int) -> int:
    """A triangle wave, 12 px per frame between x=100 and x=1180 (18 px at 1080p: a fast row)."""
    leg = (frame * 12) % 2160
    return 100 + leg if leg <= 1080 else 100 + (2160 - leg)


def label_rows() -> List[Tuple[int, str, str]]:
    sentinel = tuple(str(v) for v in labels.sentinel(WIDTH))
    return [(f, *sentinel) if f in SENTINEL_ROWS else (f, f"{ball_x(f)}", "360") for f in range(ROWS)]


def shifted_detections() -> Tuple[Evidence, VideoMeta]:
    """What ``realign.detections`` would answer for a video running SHIFT frames behind the labels."""
    points = {f + SHIFT: (float(ball_x(f)), 360.0) for f in range(ROWS) if f not in SENTINEL_ROWS}
    return Evidence.of(points, TOTAL, realign.scale_for_width(WIDTH)), META


def write_fixture(root: Path, reviewed: Optional[List[int]] = None) -> Path:
    """labels/<CLIP>_ball.csv (and the review sidecar when `reviewed` is given) under `root`."""
    folder = root / "labels"
    folder.mkdir(exist_ok=True)
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
         patch.object(realign, "require_video", lambda clip: root / "videos" / f"{clip}.mp4"), \
         patch.object(realign, "video_meta", lambda path: META), \
         patch.multiple("finetune.data.workspace", WORKSPACE=root, LABELS=root / "labels", VIDEOS=root / "videos"):
        yield


def run_main(root: Path, *args: str, out: Optional[io.StringIO] = None) -> Tuple[int, str]:
    """``main()`` on `args`; returns (exit code, stdout). Pass `out` to keep what was printed when
    main() raises."""
    out = io.StringIO() if out is None else out
    with bound_to(root), redirect_stdout(out):
        code = realign.main(list(args))
    return code, out.getvalue()


def listing(root: Path) -> List[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*"))


def backups(csv: Path) -> List[Path]:
    return sorted(csv.parent.glob(f"{csv.name}.before_realign_*"))


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

def test_fewer_than_two_clips_is_a_usage_error(tmp_path, capsys):
    write_fixture(tmp_path)
    with pytest.raises(SystemExit) as caught:
        run_main(tmp_path, CLIP)
    assert caught.value.code == 2
    assert "usage: realign.py" in capsys.readouterr().err


def test_dry_run_reports_the_offset_and_writes_nothing(tmp_path):
    csv = write_fixture(tmp_path)
    original = csv.read_bytes()
    before = listing(tmp_path)
    code, out = run_main(tmp_path, CLIP, CLIP)
    assert code == 0
    assert out == HEADER + SOLVED + DRY_RUN
    assert csv.read_bytes() == original
    assert listing(tmp_path) == before


def test_apply_backs_the_file_up_then_retimes_every_row(tmp_path):
    csv = write_fixture(tmp_path)
    original = csv.read_bytes()
    code, out = run_main(tmp_path, CLIP, CLIP, "--apply")
    assert code == 0
    assert out == HEADER + SOLVED + WRITTEN
    backup, = backups(csv)
    assert backup.read_bytes() == original
    assert listing(tmp_path) == ["labels", f"labels/{CLIP}_ball.csv", f"labels/{backup.name}"]
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
    assert listing(tmp_path) == before                       # no backup either
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


def test_the_video_the_labels_belong_to_is_swapped_in_whatever_its_container(tmp_path):
    """A clip whose labels belong to another clip's video: apply copies that video in under the
    labels' name (keeping its container) and keeps the wrong one as a .bak."""
    write_fixture(tmp_path)
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / f"{CLIP}.mov").write_bytes(b"wrong")
    (videos / "other_clip.mkv").write_bytes(b"right")
    with patch.multiple("finetune.data.workspace", WORKSPACE=tmp_path, LABELS=tmp_path / "labels", VIDEOS=videos), \
            redirect_stdout(io.StringIO()):
        realign.swap_video(CLIP, "other_clip")
    assert (videos / f"{CLIP}.mkv").read_bytes() == b"right"
    assert (videos / f"{CLIP}_wrong_video.mov.bak").read_bytes() == b"wrong"
    assert not (videos / f"{CLIP}.mov").exists()


# --------------------------------------------------------------------------- the pieces main() calls

def evidence() -> Tuple[Tracks, Evidence]:
    """(the visible rows, what the detector saw) for the fixture, built the way main() builds them."""
    entries = sorted(labels.LabelRow(f, float(x), float(y)) for f, x, y in label_rows())
    return realign.visible_tracks(entries, (WIDTH, HEIGHT), META), shifted_detections()[0]


def test_parse_args_takes_two_clips_and_the_two_flags():
    args = realign.parse_args(["a", "b", "--preview"])
    assert (args.label_clip, args.video_clip, args.preview, args.apply) == ("a", "b", True, False)
    assert realign.parse_args(["a", "b", "--apply"]).apply is True
    for bad in (["x"], [], ["a", "b", "--aply"]):
        with pytest.raises(SystemExit):
            realign.parse_args(bad)


def test_visible_tracks_scales_onto_the_video_and_marks_the_fast_rows():
    tracks, _ = evidence()
    assert len(tracks.frames) == 238 and realign.scale_for_width(WIDTH) == 1.5
    assert tracks.fast[1:].all() and not tracks.fast[0]
    entries = [labels.LabelRow(0, 100.0, 50.0), labels.LabelRow(1, 1279.0, 0.0), labels.LabelRow(2, 200.0, 100.0)]
    small = realign.visible_tracks(entries, (640, 360), META)       # labels made on a 640x360 copy
    assert small.frames.tolist() == [0, 2] and small.xs.tolist() == [200.0, 400.0] and small.ys.tolist() == [100.0, 200.0]


def test_solve_alignment_finds_the_constant_plus_three_offset_at_rate_1(capsys):
    tracks, seen = evidence()
    pinned = np.zeros(len(tracks.frames), bool)
    best = realign.solve_alignment(tracks, seen, set(), pinned, chunks=8)
    assert best is not None
    assert best.rate == 1.0
    assert (best.chosen == SHIFT).all()
    np.testing.assert_array_equal(best.bases, tracks.frames)
    assert best.result[0] == 1.0 and best.result[2] == 0.0 and best.dropped == frozenset()
    assert capsys.readouterr().out == "  rate 1: 22 candidate offsets in -182..+8  -> agree 1.00, median 0.0 px\n"


def test_solve_alignment_is_none_when_no_rate_has_a_candidate_offset(capsys):
    tracks, seen = evidence()
    pinned = np.zeros(len(tracks.frames), bool)
    with patch.object(realign, "candidates", lambda *args: np.array([], np.int64)):
        best = realign.solve_alignment(tracks, seen, set(), pinned, chunks=8)
    assert best is None
    assert capsys.readouterr().out == ""


def test_solve_alignment_keeps_a_pinned_row_at_offset_zero():
    tracks, seen = evidence()
    fixed = {PINNED_FRAME}
    pinned = np.array([f in fixed for f in tracks.frames.tolist()], bool)
    best = realign.solve_alignment(tracks, seen, fixed, pinned, chunks=8)
    assert best is not None
    assert best.rate == 1.0
    frames = tracks.frames
    assert best.chosen[frames <= PINNED_FRAME].tolist() == [0] * int((frames <= PINNED_FRAME).sum())
    assert (best.chosen[frames > PINNED_FRAME] == SHIFT).all()


def test_human_pin_is_a_hard_constraint():
    n = 220
    bases = np.arange(n, dtype=np.int64) * 2
    points = {int(frame): (1000.0, 0.0) for frame in bases}
    seen = Evidence.of(points, n * 2 + 3, 1.0)
    tracks = Tracks(bases, np.zeros(n, np.float32), np.zeros(n, np.float32), np.zeros(n, bool))
    pinned = np.zeros(n, bool)
    pinned[n // 2] = True
    offsets = realign.solve(tracks, seen, np.array([0, 1]), pinned)
    assert offsets[n // 2] == 0


def test_segments_groups_runs_of_one_offset():
    assert realign.segments(np.array([0, 1, 2, 3, 4]), np.array([0, 0, 2, 2, 2])) == [[0, 1, 0], [2, 4, 2]]
    assert realign.segments(np.array([5, 9]), np.array([3, 3])) == [[5, 9, 3]]
    assert realign.segments(np.array([], np.int64), np.array([], np.int64)) == []


def aligned_by(shift: int, frames: np.ndarray) -> realign.Alignment:
    return realign.Alignment((1.0, len(frames), 0.0), 1.0, frames, np.full(len(frames), shift, np.int64))


def test_apply_alignment_backs_up_every_time_and_retimes_every_row(tmp_path):
    csv = write_fixture(tmp_path)
    original = csv.read_bytes()
    before = listing(tmp_path)
    entries = sorted(labels.read_label_rows(csv))
    frames = evidence()[0].frames
    with bound_to(tmp_path), redirect_stdout(io.StringIO()):
        code = realign.apply_alignment(entries, frames, aligned_by(SHIFT, frames), set(), META, (WIDTH, HEIGHT), CLIP, CLIP)
    assert code == 0
    backup, = backups(csv)
    assert backup.read_bytes() == original
    assert listing(tmp_path) == before + [f"labels/{backup.name}"]
    rewritten = [(row.frame, row.x, row.y) for row in labels.read_label_rows(csv)]
    assert rewritten == [(f + SHIFT, float(x), float(y)) for f, x, y in label_rows()]
    assert [f for f, x, y in rewritten if not labels.is_visible(x, y, WIDTH, HEIGHT)] == [f + SHIFT for f in SENTINEL_ROWS]

    # a second apply re-times the already re-timed rows and backs the file up again, like every
    # other bulk write over a label file (fileio.backup_copy)
    entries = sorted(labels.read_label_rows(csv))
    retimed = frames + SHIFT
    with bound_to(tmp_path), redirect_stdout(io.StringIO()):
        realign.apply_alignment(entries, retimed, aligned_by(SHIFT, retimed), set(), META, (WIDTH, HEIGHT), CLIP, CLIP)
    first, second = backups(csv)
    assert first == backup and first.read_bytes() == original
    assert [row.frame for row in labels.read_label_rows(second)] == [f + SHIFT for f in range(ROWS)]
    assert [row.frame for row in labels.read_label_rows(csv)] == [f + 2 * SHIFT for f in range(ROWS)]


def test_apply_alignment_writes_nothing_when_a_reviewed_row_would_move(tmp_path):
    csv = write_fixture(tmp_path, reviewed=[PINNED_FRAME])
    original = csv.read_bytes()
    before = listing(tmp_path)
    entries = sorted(labels.read_label_rows(csv))
    frames = evidence()[0].frames
    with bound_to(tmp_path), pytest.raises(ValueError, match="would move a reviewed row; nothing changed"):
        realign.apply_alignment(entries, frames, aligned_by(SHIFT, frames), {PINNED_FRAME}, META, (WIDTH, HEIGHT), CLIP, CLIP)
    assert csv.read_bytes() == original
    assert listing(tmp_path) == before
