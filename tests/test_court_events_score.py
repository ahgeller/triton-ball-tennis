"""evaluation.court_events_score (score_court_events.py): the landing tiers, the matcher, the
scores and their lines, the recompute from a tracking JSON, and the command end to end on a
temporary archive. No GPU, no real archive."""
import json
from pathlib import Path

import pytest

from ball_in_play_selector.core import frame_result
from evaluation import court_events_score as scorer
from evaluation.court_events_score import (
    ALL, compare, format_line, landings_at_tier, match_landings, pooled, read_scores, scene_from_payload, score_clip,
)
from finetune.data.bounce_labels import BounceLabel
from tennis_tracker.court.court_scene import BounceEvent, BouncePrediction
from tennis_tracker.court.rally_legs import inferred_landings
from tennis_tracker.export import frame_result_to_json


def _row(frame, score=0.9, origin="model", position="filtered", x=100.0, y=200.0):
    return {"frame": frame, "x": x, "y": y, "score": score, "origin": origin, "position": position}


# --------------------------------------------------------------------------- tiers

def test_inferred_landings_leaves_out_those_a_confirmed_landing_covers():
    confirmed = [BounceEvent(100, 1.0, 2.0, 0.9)]
    candidates = [BounceEvent(100, 1.0, 2.0, 0.3), BounceEvent(104, 1.0, 2.0, 0.25), BounceEvent(400, 1.0, 2.0, 0.3)]
    kinematic = [BounceEvent(96, 1.0, 2.0, 5.0, "kink", "measurement"), BounceEvent(250, 1.0, 2.0, 3.0, "gap", "extrapolated")]
    # EVENT_MERGE_S at 30 FPS is 4 frames: 96, 100 and 104 are the confirmed landing seen again.
    assert inferred_landings(confirmed, candidates, kinematic, 30.0) == [candidates[2], kinematic[1]]
    assert inferred_landings([], candidates, kinematic, 30.0) == candidates + kinematic


def test_landings_at_each_tier():
    block = {
        "bounces": [_row(100)],
        "bounce_candidates": [_row(100, 0.3), _row(103, 0.25), _row(400, 0.3)],
        "kinematic_bounces": [_row(102, 5.0, "kink", "measurement"), _row(250, 3.0, "gap", "extrapolated"), _row(402, 7.0, "kink")],
    }
    frames = lambda tier: [event.frame for event in landings_at_tier(block, tier, 30.0)]
    assert frames("confirmed") == [100]
    assert frames("placed") == [100, 250, 400, 402]           # the leg assembly's rule: 400 and 402 both stand
    assert frames("all") == [100, 250, 402]                   # 400 is 402 (the stronger kink) seen again
    (landing,) = landings_at_tier(block, "confirmed", 30.0)
    assert landing == BounceEvent(100, 100.0, 200.0, 0.9, "model", "filtered")
    with pytest.raises(ValueError, match="unknown tier"):
        landings_at_tier(block, "some", 30.0)
    assert landings_at_tier({}, "all", 30.0) == []


# --------------------------------------------------------------------------- matching and scoring

def test_match_landings_is_greedy_one_to_one_closest_pair_first():
    assert match_landings([10, 20], [14, 16, 60], 6) == [(0, 0), (1, 1)]
    assert match_landings([10, 20], [13], 6) == [(0, 0)]              # one landing serves one label
    assert match_landings([10, 20], [13], 2) == []
    assert match_landings([10], [9, 11], 3) == [(0, 0)]               # a tie goes to the earlier landing
    assert match_landings([], [1, 2], 3) == [] and match_landings([1], [], 3) == []


def test_a_label_between_two_frames_matches_a_landing_on_either_and_is_printed_as_a_half_frame():
    assert match_landings([412.5], [412], 0.5) == [(0, 0)] and match_landings([412.5], [413], 0.5) == [(0, 0)]
    assert match_landings([412.5], [414], 1.4) == [] and match_landings([412.5], [414], 1.5) == [(0, 0)]
    assert match_landings([412.5], [412, 413], 3.0) == [(0, 0)]        # the tie goes to the earlier landing
    labels = [BounceLabel(412.5, (10.0, 10.0)), BounceLabel(600.5, None)]
    score = score_clip("video6", labels, [BounceEvent(413, 13.0, 14.0, 0.9)], fps=30.0, tolerance_s=0.1, px_scale=1.0)
    assert (score.labels, score.found, score.hits) == (2, 1, 1)
    assert score.timing_errors_s == pytest.approx([0.5 / 30]) and score.px_errors == [5.0]
    assert format_line(score).startswith("video6         labels=   2 found=   1 hits=   1 precision=1.000 recall=0.500 | dt med=  17ms p90=  17ms | px med=  5.0px")
    assert scorer.unmatched_lines(score, 30.0) == ["    unmatched labels:   600.5 (20.02s)"]
    # The tolerance is a fraction of a frame, not rounded: 0.1 s at 25 FPS is 2.5 frames, so a
    # landing 3 frames from a whole-frame label misses and one 2.5 frames from a half-frame label hits.
    assert score_clip("v", [BounceLabel(100)], [BounceEvent(103, 0.0, 0.0, 1.0)], 25.0, 0.1, 1.0).hits == 0
    assert score_clip("v", [BounceLabel(100.5)], [BounceEvent(103, 0.0, 0.0, 1.0)], 25.0, 0.1, 1.0).hits == 1


def test_score_clip_counts_hits_timing_and_pixels_and_lists_the_rest():
    labels = [BounceLabel(100, (100.0, 200.0)), BounceLabel(200, None), BounceLabel(300, (0.0, 0.0))]
    events = [BounceEvent(102, 103.0, 204.0, 0.9), BounceEvent(199, 1.0, 1.0, 0.8), BounceEvent(500, 1.0, 1.0, 0.7, "kink")]
    score = score_clip("video6", labels, events, fps=30.0, tolerance_s=0.1, px_scale=2.0)   # 3 frames
    assert (score.labels, score.found, score.hits) == (3, 3, 2)
    assert score.precision == pytest.approx(2 / 3) and score.recall == pytest.approx(2 / 3)
    assert score.timing_errors_s == pytest.approx([2 / 30, -1 / 30])
    assert score.px_errors == [10.0]                                   # hypot(3, 4) at 1080p scale; no point, no error
    assert [label.frame for label in score.unmatched_labels] == [300]
    assert [event.frame for event in score.unmatched_events] == [500]

    line = format_line(score)
    assert line.startswith("video6         labels=   3 found=   3 hits=   2 precision=0.667 recall=0.667 | dt med=  50ms p90=  67ms | px med= 10.0px")
    parsed = read_scores(line)["video6"]
    assert (parsed.labels, parsed.found, parsed.hits) == (3, 3, 2)
    assert (parsed.precision, parsed.recall, parsed.dt_med_ms, parsed.dt_p90_ms, parsed.px_med) == (0.667, 0.667, 50.0, 67.0, 10.0)
    assert scorer.unmatched_lines(score, 30.0) == [
        "    unmatched labels:   300 (10.00s)",
        "    unmatched landings: 500 (16.67s, kink, score 0.70)",
    ]

    together = pooled([score, score])
    assert (together.clip, together.labels, together.hits) == (ALL, 6, 4) and together.unmatched_labels == []
    assert read_scores(format_line(together))[ALL].px_med == 10.0
    pointless = score_clip("v", [BounceLabel(5)], [BounceEvent(5, 0.0, 0.0, 1.0)], 30.0, 0.1, 1.0)
    assert format_line(pointless).endswith("px med=  n/a") and read_scores(format_line(pointless))["v"].px_med is None
    assert format_line(score_clip("v", [], [], 30.0, 0.1, 1.0)).startswith("v              labels=   0 found=   0 hits=   0 precision=0.000 recall=0.000")


def test_compare_prints_deltas_per_clip_and_pooled():
    before = read_scores("video6         labels=   3 found=   3 hits=   2 precision=0.667 recall=0.667 | dt med=  50ms p90=  67ms | px med= 10.0px\n"
                         "video8         labels=   2 found=   1 hits=   1 precision=1.000 recall=0.500 | dt med=  33ms p90=  33ms | px med=  n/a\n"
                         "ALL            labels=   5 found=   4 hits=   3 precision=0.750 recall=0.600 | dt med=  33ms p90=  67ms | px med= 10.0px\n")
    after = read_scores("video6         labels=   3 found=   4 hits=   3 precision=0.750 recall=1.000 | dt med=  40ms p90=  67ms | px med=  9.0px\n"
                        "video9         labels=   1 found=   1 hits=   1 precision=1.000 recall=1.000 | dt med=   0ms p90=   0ms | px med=  n/a\n"
                        "ALL            labels=   4 found=   5 hits=   4 precision=0.800 recall=1.000 | dt med=  33ms p90=  67ms | px med=  9.0px\n")
    text = compare(before, after)
    (row,) = [line for line in text.splitlines() if line.startswith("video6")]
    assert "4 (+1)" in row and "3 (+1)" in row and "0.750 (+0.083)" in row and "1.000 (+0.333)" in row and "40 (-10)" in row and "9.0 (-1.0)" in row
    assert "video8" not in text.split("only in the first log")[0] and "only in the first log: video8" in text
    assert "only in the second log: video9" in text
    lines = text.splitlines()
    (pooled_at,) = [index for index, line in enumerate(lines) if line.startswith(ALL)]
    assert lines[pooled_at - 1] == "" and lines[pooled_at - 2].startswith("video6")   # the pooled line last, after a blank


# --------------------------------------------------------------------------- recompute

def _payload(frames, fps=30.0, width=1920, height=1080, block=None):
    return {"video": {"fps": fps, "width": width, "height": height, "total_frames": len(frames)},
            "court_events": {} if block is None else block, "frames": frames}


def test_scene_from_payload_rebuilds_the_selection_the_stages_read():
    detected = frame_result(100.5, 103.25, "det", detection=type("D", (), dict(cx=100., cy=103., conf=.9, x1=98, y1=101, x2=102, y2=105))())
    filled = frame_result(7.0, 8.0, "interp")
    rows = [frame_result_to_json(0, None), frame_result_to_json(1, detected), frame_result_to_json(2, filled)]
    rows[1]["player_boxes"] = {"1": [0.0, 0.0, 4.0, 4.0]}
    scene = scene_from_payload(_payload(rows, fps=60.0, width=1280, height=720))
    assert scene.per_frame == [None, detected, filled]
    assert scene.player_boxes_by_frame == [None, {"1": [0.0, 0.0, 4.0, 4.0]}, None]
    assert (scene.fps, scene.width, scene.height, scene.stride) == (60.0, 1280, 720, 2)


# --------------------------------------------------------------------------- the command

def _archive(tmp_path: Path) -> Path:
    archive = tmp_path / "archive"
    (archive / "labels").mkdir(parents=True)
    (archive / "videos").mkdir()
    (archive / "videos" / "video1.mp4").write_bytes(b"")                # only the name is looked at
    (archive / "videos" / "video2.mp4").write_bytes(b"")
    (archive / "labels" / "video1_ball.csv").write_text("frame,ball_x,ball_y\nframe_100,100,200\nframe_300,10,10\n", encoding="utf-8")
    (archive / "labels" / "video1_bounces.csv").write_text("frame,ball_x,ball_y\nframe_100,,\nframe_300.5,,\n", encoding="utf-8")
    (archive / "labels" / "video2_ball.csv").write_text("frame,ball_x,ball_y\n", encoding="utf-8")   # no landing labels
    return archive


def _write_json(out: Path, payload) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / "video1_gridtracknet.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_main_scores_the_clips_with_landing_labels(tmp_path, capsys):
    archive = _archive(tmp_path)
    out = tmp_path / "run"
    _write_json(out, _payload([], block={"bounces": [_row(101, x=103.0, y=204.0)], "bounce_candidates": [], "kinematic_bounces": []}))
    assert scorer.main(["--archive", str(archive), "--out-dir", str(out)]) == 0
    text = capsys.readouterr().out
    assert "(1 clips with landing labels: video1)" in text                # video2 has none, and is not an error
    assert "=== landings [gridtracknet] tier=confirmed tolerance=0.10s ===" in text
    assert "video1         labels=   2 found=   1 hits=   1 precision=1.000 recall=0.500 | dt med=  33ms p90=  33ms | px med=  5.0px" in text
    assert "    unmatched labels:   300.5 (10.02s)" in text               # a landing between frames 300 and 301
    assert "ALL            labels=   2 found=   1 hits=   1 precision=1.000 recall=0.500" in text


def test_main_refuses_a_run_without_court_events_or_without_the_json(tmp_path, capsys):
    archive = _archive(tmp_path)
    out = tmp_path / "run"
    assert scorer.main(["--archive", str(archive), "--out-dir", str(out)]) == 1
    assert "video1_gridtracknet.json is missing" in capsys.readouterr().err
    _write_json(out, _payload([]))                                         # written without --court
    assert scorer.main(["--archive", str(archive), "--out-dir", str(out)]) == 1
    assert "has no court events" in capsys.readouterr().err
    assert scorer.main(["--archive", str(archive), "--out-dir", str(out), "--clips", "video2"]) == 1
    assert "No clip in" in capsys.readouterr().out


def test_main_compares_against_an_earlier_log(tmp_path, capsys):
    archive = _archive(tmp_path)
    out = tmp_path / "run"
    _write_json(out, _payload([], block={"bounces": [_row(101)], "bounce_candidates": [], "kinematic_bounces": []}))
    assert scorer.main(["--archive", str(archive), "--out-dir", str(out)]) == 0
    before = tmp_path / "before.log"
    before.write_text(capsys.readouterr().out, encoding="utf-8")
    _write_json(out, _payload([], block={"bounces": [_row(101), _row(299)], "bounce_candidates": [], "kinematic_bounces": []}))
    assert scorer.main(["--archive", str(archive), "--out-dir", str(out), "--compare", str(before)]) == 0
    text = capsys.readouterr().out
    assert f"=== against {before} ===" in text
    (row,) = [line for line in text.splitlines() if line.startswith("video1 ") and "(+1)" in line]
    assert "2 (+1)" in row and "1.000 (+0.500)" in row
    (tmp_path / "empty.log").write_text("nothing here\n", encoding="utf-8")
    assert scorer.main(["--archive", str(archive), "--out-dir", str(out), "--compare", str(tmp_path / "empty.log")]) == 1
    assert "no landing metric lines" in capsys.readouterr().err


def test_main_recompute_reruns_the_stages_on_the_json_frames(tmp_path, capsys, monkeypatch):
    archive = _archive(tmp_path)
    out = tmp_path / "run"
    rows = [frame_result_to_json(frame, frame_result(100.0 + frame, 200.0, "det")) for frame in range(120)]
    _write_json(out, _payload(rows, block={"bounces": [_row(50)], "bounce_candidates": [], "kinematic_bounces": []}))
    seen = {}

    def one_landing(scene, model_path):
        seen["frames"], seen["model"] = len(scene.per_frame), model_path
        point = scene.per_frame[99]
        return BouncePrediction([BounceEvent(99, point.cx, point.cy, 0.9, "model", point.position_kind)], [])

    monkeypatch.setattr(scorer, "predict_bounce_events", one_landing)
    assert scorer.main(["--archive", str(archive), "--out-dir", str(out), "--recompute", "--bounce-model", "x.cbm"]) == 0
    text = capsys.readouterr().out
    assert seen == {"frames": 120, "model": "x.cbm"}
    assert "(events recomputed from the frames)" in text
    assert "video1         labels=   2 found=   1 hits=   1 precision=1.000 recall=0.500 | dt med=  33ms" in text   # 99 vs 100, not the file's 50
