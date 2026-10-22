"""evaluation/track_breaks.py: where a run lost the ball, on hand-built tracking payloads and labels."""
import json
from collections import Counter

from finetune.data.labels import write_labels
from finetune.data.workspace import label_csv
from evaluation.track_breaks import (ClipBreaks, clip_breaks, compare, fmt, pool, read_scores, report, start_lines,
                                     track_starts)


def _payload(present, chosen_frames, other_frames, width=1000, height=500, fps=30.0):
    frames = [{"frame": f, "present": False} for f in range(10)]
    for f, (x, y) in present.items():
        frames[f] = {"frame": f, "present": True, "x": x, "y": y, "source": "det"}
    observations = lambda fs: [{"frame": f, "x": 1.0, "y": 1.0} for f in fs]   # noqa: E731
    return {
        "video": {"fps": fps, "width": width, "height": height, "total_frames": 10},
        "chosen_track": {"track_id": 0, "observations": observations(chosen_frames)},
        "tracks": [{"track_id": 0, "observations": observations(chosen_frames)},
                   {"track_id": 1, "observations": observations(other_frames)}],
        "frames": frames,
    }


LABELS = {f: (10.0, 10.0) for f in range(10)}
LABELS[5] = None     # the one frame with no ball


def test_losses_are_bucketed_by_what_the_run_had_on_the_frame():
    # Output on 0, 1, 2, 6, 7; the chosen track also saw 3 (pruned); track 1 alone saw 8; 4 and 9 nobody.
    present = {0: (10.0, 10.0), 1: (12.0, 10.0), 2: (14.0, 10.0), 6: (20.0, 10.0), 7: (900.0, 400.0)}
    breaks = clip_breaks(_payload(present, [0, 1, 2, 3, 6, 7], [8]), LABELS)
    assert breaks == ClipBreaks(visible=9, hit=5, no_track=2, other_track=1, chosen_pruned=1, switches=1, longest_gap=3)
    assert breaks.missed == 4


def test_a_jump_counts_as_a_switch_only_over_the_diagonal_share():
    present = {0: (10.0, 10.0), 1: (40.0, 10.0)}     # 30 px of a 1118 px diagonal
    assert clip_breaks(_payload(present, [0, 1], []), LABELS, switch_frac=0.03).switches == 0
    assert clip_breaks(_payload(present, [0, 1], []), LABELS, switch_frac=0.02).switches == 1


def test_pooling_sums_the_counts_and_keeps_the_longest_gap():
    pooled = pool([ClipBreaks(9, 5, 2, 1, 1, 1, 3), ClipBreaks(4, 4, 0, 0, 0, 0, 7)])
    assert pooled == ClipBreaks(13, 9, 2, 1, 1, 1, 7)


def test_lines_round_trip_through_the_log_and_compare():
    before = "\n".join([fmt("video13", ClipBreaks(9, 5, 2, 1, 1, 1, 3)), fmt("fps=30", ClipBreaks(9, 5, 2, 1, 1, 1, 3)),
                        fmt("ALL", ClipBreaks(9, 5, 2, 1, 1, 1, 3))])
    after = "\n".join([fmt("video13", ClipBreaks(9, 7, 1, 1, 0, 0, 2)), fmt("ALL", ClipBreaks(9, 7, 1, 1, 0, 0, 2))])
    assert read_scores(before)["video13"] == ClipBreaks(9, 5, 2, 1, 1, 1, 3)
    text = compare(read_scores(before), read_scores(after))
    row = next(line for line in text.splitlines() if line.startswith("video13"))
    assert "7 (  +2)" in row and "0 (  -1)" in row and "2 (  -1)" in row
    assert "only in the first log: fps=30" in text


def test_report_reads_a_run_folder_against_the_workspace_labels(workspace_dir, tmp_path):
    out_dir = tmp_path / "pipeline"
    out_dir.mkdir()
    present = {0: (10.0, 10.0), 1: (12.0, 10.0)}
    (out_dir / "video13_gridtracknet.json").write_text(json.dumps(_payload(present, [0, 1, 2], [], fps=59.94)),
                                                       encoding="utf-8")
    write_labels(label_csv("video13"), {0: (10.0, 10.0), 1: (12.0, 10.0), 2: (14.0, 10.0)}, 1000, 500)
    lines = report(out_dir, "gridtracknet")
    assert lines[0].startswith("video13") and read_scores(lines[0])["video13"] == ClipBreaks(3, 2, 0, 0, 1, 0, 1)
    names = [line.split()[0] for line in lines if line]
    assert names == ["video13", "fps=60", "source=custom", "ALL"]      # no clips.csv: the custom default


def test_track_starts_are_counted_by_gate_when_the_run_recorded_them():
    payload = _payload({}, [], [])
    assert track_starts(payload) == Counter()                      # an older JSON: nothing to count
    payload["tracks"][0]["start"] = {"reason": "first", "rival_track": None}
    payload["tracks"][1]["start"] = {"reason": "gate", "rival_track": 0}
    assert track_starts(payload) == Counter({"first": 1, "gate": 1})
    lines = start_lines({"fps=60": Counter({"gate": 3, "first": 1}), "ALL": Counter({"gate": 3, "first": 1})})
    assert lines[1].startswith("  ALL        tracks=    4: gate=3, first=1")
