"""evaluation.archive (evaluate_archive.py) helpers on synthetic inputs: no GPU, no real archive, never finetune/labels."""
import json
from pathlib import Path

import pytest

from evaluation.archive import list_clips
from evaluation.pipeline_score import (ClipScore, LINE_PX, _inside_player, _near_court_line, _segment_distance,
                                       score_pipeline_clip)


def _archive(tmp_path: Path) -> Path:
    (tmp_path / "labels").mkdir()
    (tmp_path / "videos").mkdir()
    for clip in ("video1", "video2", "video3"):
        (tmp_path / "labels" / f"{clip}_ball.csv").write_text("frame,ball_x,ball_y\n")
    (tmp_path / "videos" / "video1.mp4").write_bytes(b"")      # only the name is checked
    (tmp_path / "videos" / "video2.mp4").write_bytes(b"")
    (tmp_path / "review_status.json").write_text(json.dumps({"verified_clips": ["video1", "video3"]}))
    return tmp_path


def test_list_clips_leaves_out_clips_in_exclude_txt_unless_asked(tmp_path, capsys):
    archive = _archive(tmp_path)
    (archive / "exclude.txt").write_text("video1  # audit: labels belong to another cut\n", encoding="utf-8")
    assert list_clips(None, archive) == []
    assert "[skip] video1: listed in exclude.txt" in capsys.readouterr().out
    assert [video.stem for video, _ in list_clips(None, archive, include_excluded=True)] == ["video1", "video2"]


def test_list_clips_keeps_verified_clips_that_have_a_video(tmp_path, capsys):
    archive = _archive(tmp_path)
    assert list_clips(None, archive) == [(archive / "videos" / "video1.mp4", archive / "labels" / "video1_ball.csv")]
    out = capsys.readouterr().out
    assert "[skip] video2: labels are still under review" in out
    assert "[skip] no video for video3_ball.csv" in out


def test_list_clips_filters_by_name_pattern(tmp_path):
    archive = _archive(tmp_path)
    assert list_clips(["video9"], archive) == []
    assert list_clips(["video*"], archive) == list_clips(None, archive)


def test_list_clips_narrows_to_a_named_held_out_set(tmp_path, capsys):
    archive = _archive(tmp_path)
    (archive / "review_status.json").write_text(json.dumps({"verified_clips": ["video1", "video2", "video3"]}))
    (archive / "clips.csv").write_text("clip,source,group\nvideo1,custom,own\nvideo2,custom,own\nvideo3,prof,g\n")
    (archive / "holdouts.json").write_text(json.dumps({"own": {"groups": ["own"]}, "g": {"groups": ["g"]}}))
    assert [video.stem for video, _ in list_clips(None, archive, holdout="own")] == ["video1", "video2"]
    assert [video.stem for video, _ in list_clips(["video2"], archive, holdout="own")] == ["video2"]     # both narrow
    assert list_clips(None, archive, holdout="g") == []                        # video3 has no video
    assert "[skip] no video for video3_ball.csv" in capsys.readouterr().out
    with pytest.raises(ValueError, match="no held-out set 'nope'"):
        list_clips(None, archive, holdout="nope")


def test_list_clips_trusts_an_archive_without_a_review_policy(tmp_path):
    archive = _archive(tmp_path)
    (archive / "review_status.json").unlink()
    assert [video.stem for video, _ in list_clips(None, archive)] == ["video1", "video2"]


def test_segment_distance():
    assert _segment_distance(0, 5, 0, 0, 10, 0) == 5
    assert _segment_distance(15, 0, 0, 0, 10, 0) == 5       # beyond the end: to the endpoint
    assert _segment_distance(3, 4, 0, 0, 0, 0) == 5         # zero-length segment: to the point
    assert _segment_distance(5, 0, 0, 0, 10, 0) == 0


def test_inside_player_accepts_dict_and_list_boxes():
    assert _inside_player(5, 5, {1: [0, 0, 10, 10]}) is True
    assert _inside_player(5, 5, [[0, 0, 10, 10]]) is True
    assert _inside_player(15, 5, [[0, 0, 10, 10]]) is False
    assert _inside_player(5, 5, [None, [0, 0, 10, 10]]) is True
    assert _inside_player(5, 5, None) is False
    assert _inside_player(5, 5, []) is False


def _keypoints(**points):
    """28 values, (0, 0) everywhere but the semantic indices given."""
    values = [0.0] * 28
    for index, (x, y) in points.items():
        values[2 * int(index[1:])] = x
        values[2 * int(index[1:]) + 1] = y
    return values


def test_near_court_line():
    live = _keypoints(k0=(100.0, 100.0), k4=(200.0, 100.0))     # pair (0, 4) is a court line
    assert _near_court_line(150, 105, live, 1.0) is True
    assert _near_court_line(150, 100 + LINE_PX, live, 1.0) is True
    assert _near_court_line(150, 111, live, 1.0) is False
    assert _near_court_line(150, 106, live, 2.0) is False            # 12 px at 1080p
    assert _near_court_line(150, 105, live[:15], 1.0) is False       # too few values (MIN_KEYPOINT_VALUES)
    assert _near_court_line(150, 105, live[:16], 1.0) is True        # the tracker's rule: 16 values are usable
    assert _near_court_line(150, 105, None, 1.0) is False
    undetected = _keypoints(k4=(200.0, 0.0))                         # its partner sits at (0, 0)
    assert _near_court_line(100, 0, undetected, 1.0) is False


def test_score_pipeline_clip_on_a_synthetic_payload():
    payload = {"frames": [{"frame": 0, "present": True, "x": 10, "y": 10, "source": "det", "player_boxes": None},
                          {"frame": 1, "present": True, "x": 500, "y": 500, "source": "det"},
                          {"frame": 2, "present": False}],
               "last_valid_court_keypoints": None}
    labels = {0: (10, 10), 1: (10, 10), 2: (50, 50), 3: None}
    result = score_pipeline_clip(labels, payload, 1920)
    assert isinstance(result, ClipScore)
    assert result.by_source == {"det": {"hit": 1, "localized_error": 0, "wrong": 1, "invisible": 0}}
    assert result.landed == {"other": 1}
    assert result.bad_frames == [(1, "wrong", "det", "other")]
    assert result.pairs == [((10, 10), (10, 10)), ((10, 10), (500, 500)), ((50, 50), None), (None, None)]
    assert result.stats.visible == 3
    assert result.stats.recall == pytest.approx(1 / 3)
    assert result.stats.wrong == pytest.approx(1 / 3)
    assert result.stats.miss == pytest.approx(1 / 3)
    assert result.stats.invisible == 1 and result.stats.false_alarm == 0.0


def test_score_pipeline_clip_scales_to_1080p_and_classifies_where_bad_outputs_land():
    payload = {"frames": [{"frame": 0, "present": True, "x": 100, "y": 100, "source": "det",
                           "player_boxes": [[90, 90, 110, 110]]},
                          {"frame": 1, "present": True, "x": 20, "y": 20, "source": "interp"}],
               "last_valid_court_keypoints": _keypoints(k0=(0.0, 20.0), k4=(100.0, 20.0))}
    result = score_pipeline_clip({0: None, 1: (10, 20)}, payload, 1280)
    assert result.pairs == [(None, (150.0, 150.0)), ((15.0, 30.0), (30.0, 30.0))]
    assert result.by_source == {"det": {"hit": 0, "localized_error": 0, "wrong": 0, "invisible": 1},
                                "interp": {"hit": 0, "localized_error": 1, "wrong": 0, "invisible": 0}}
    assert result.landed == {"player": 1}
    assert result.bad_frames == [(0, "invisible", "det", "player")]


def test_score_pipeline_clip_takes_the_court_as_last_seen_when_the_payload_does_not_say():
    """Without the exporter's last_valid_court_keypoints, the latest frame with enough keypoint
    values (the tracker's own rule) classifies the bad outputs of the frames that have none."""
    line = _keypoints(k0=(0.0, 20.0), k4=(100.0, 20.0))
    payload = {"frames": [{"frame": 0, "present": False, "court_keypoints": line},
                          {"frame": 1, "present": False, "court_keypoints": [1.0] * 15},      # too few: not the court
                          {"frame": 2, "present": True, "x": 20, "y": 20, "source": "det"}]}
    result = score_pipeline_clip({2: None}, payload, 1920)
    assert result.landed == {"line": 1} and result.bad_frames == [(2, "invisible", "det", "line")]


def test_report_raw_pools_pairs_at_1080p_and_measures_the_hit_offset(capsys):
    from evaluation.archive import PER_CLIP_CONF, LabelledDetection, RawClip, _hit_offset, _pooled_pairs, report_raw
    from finetune.data.detector_metrics import HIT_PX
    rows = [LabelledDetection((100.0, 100.0), (101.0, 102.0), 0.9),          # a hit, 1 right and 2 down
            LabelledDetection((100.0, 100.0), (100.0, 100.0), 0.3),          # too faint at the per-clip threshold
            LabelledDetection(None, (50.0, 50.0), 0.8),                      # a false alarm
            LabelledDetection((200.0, 200.0), None, 0.0)]                    # a miss
    per_clip = [RawClip("clip", 960, 540, 30.0, 1.0, rows)]
    pairs = _pooled_pairs(per_clip, 0.5)
    assert pairs[0] == ((200.0, 200.0), (202.0, 204.0)) and pairs[1] == ((200.0, 200.0), None)
    assert pairs[2] == (None, (100.0, 100.0)) and pairs[3] == ((400.0, 400.0), None)
    assert len(_pooled_pairs(per_clip, 0.2)) == 4 and _pooled_pairs(per_clip, 0.2)[1][1] == (200.0, 200.0)
    dx, dy, count = _hit_offset(per_clip)
    assert (dx, dy, count) == (1.0, 2.0, 1) and PER_CLIP_CONF > 0.3 and HIT_PX >= 3
    report_raw("probe", per_clip, [0.5])
    out = capsys.readouterr().out
    assert "probe: threshold sweep" in out and "conf>=0.50" in out and "clip" in out
    assert "mean offset on hits dx=+1.00px dy=+2.00px (n=1)" in out
    assert "clip           vis=   3 recall=0.333 wrong=0.000 miss=0.667 | invis=  1 falseAlarm=1.000 (1/1) |" in out
    assert "source=" not in out                                               # no manifest given: no per-source lines


def _raw_clip(name, width, height, hits, misses, invisible, alarms, offset=(0.0, 0.0)):
    """A RawClip of `hits` labels the detector hit (at `offset` from the label, in native px),
    `misses` it missed, and `invisible` labels of which it fired on `alarms`, laid along a
    diagonal so every label sits in a different grid cell position."""
    from evaluation.archive import LabelledDetection, RawClip
    rows = []
    for index in range(hits + misses):
        label = (20.0 + index * 7.3, 15.0 + index * 5.1)
        hit = index < hits
        rows.append(LabelledDetection(label, (label[0] + offset[0], label[1] + offset[1]) if hit else None, 0.9 if hit else 0.0))
    for index in range(invisible):
        rows.append(LabelledDetection(None, (5.0, 5.0) if index < alarms else None, 0.9 if index < alarms else 0.0))
    return RawClip(name, width, height, 30.0, 1.0, rows)


def test_report_raw_adds_a_macro_line_and_one_line_per_source(capsys):
    """The pooled line lets a long clip outvote a short one and a source with few invisible labels
    hide a false-alarm problem; the macro line weighs every clip the same, the source lines pool
    within one source, and every false-alarm rate carries the count it is made of."""
    from evaluation.archive import macro_metrics, report_raw
    from evaluation.compare_eval_logs import read_scores
    long_prof = _raw_clip("video13", 1280, 720, hits=90, misses=10, invisible=20, alarms=1)        # recall 0.9, fa 0.05
    short_own = _raw_clip("video1", 1920, 1080, hits=5, misses=5, invisible=0, alarms=0)           # recall 0.5, no invisible
    grid = _raw_clip("grid_match8", 1920, 1080, hits=20, misses=0, invisible=4, alarms=2)          # recall 1.0, fa 0.5
    report_raw("probe", [long_prof, short_own, grid], [0.5],
               sources={"video13": "prof", "video1": "custom", "grid_match8": "grid"})
    out = capsys.readouterr().out
    scores = read_scores(out)
    assert scores["conf>=0.50"].recall == pytest.approx(115 / 130, abs=5e-4) and scores["ALL" if "ALL" in scores else "conf>=0.50"].visible == 130
    assert scores["macro"].recall == pytest.approx((0.9 + 0.5 + 1.0) / 3, abs=5e-4) and scores["macro"].visible == 130
    assert scores["macro"].false_alarm == pytest.approx((0.05 + 0.5) / 2, abs=5e-4)       # over the clips with invisible labels
    assert scores["macro"].invisible == 24
    assert scores["source=prof"].recall == pytest.approx(0.9) and scores["source=prof"].visible == 100
    assert scores["source=custom"].recall == pytest.approx(0.5) and scores["source=grid"].false_alarm == pytest.approx(0.5)
    lines = {line.split()[0]: line for line in out.splitlines() if " vis=" in line}
    assert "falseAlarm=0.050 (1/20) |" in lines["video13"] and "falseAlarm=0.000 (0/0) |" in lines["video1"]
    assert "falseAlarm=0.500 (2/4) |" in lines["grid_match8"] and "falseAlarm=0.125 (3/24) |" in lines["conf>=0.50"]
    assert "falseAlarm=0.275 (3/24) |" in lines["macro"]                       # the mean rate, beside the pooled count
    assert list(lines)[-4:] == ["macro", "source=custom", "source=grid", "source=prof"]
    stats = macro_metrics([])
    assert stats.visible == 0 and stats.invisible == 0 and stats.recall != stats.recall      # nan: no clip to average


def test_cell_histogram_bins_recall_by_position_within_the_grid_cell(capsys):
    """A label's place within its 48x27 cell decides the bin: at 1920x1080 a cell is 40 px wide,
    so x=20 sits mid-cell (fraction 0.5) and x=41 just inside the next one (0.025)."""
    from evaluation.archive import CELL_BINS, LabelledDetection, RawClip, cell_histogram, report_raw
    rows = [LabelledDetection((20.0, 20.0), (20.0, 20.0), 0.9),             # x 0.5, y 0.5: both middle bins, a hit
            LabelledDetection((41.0, 20.0), None, 0.0),                     # x 0.025: first bin, a miss
            LabelledDetection((79.0, 61.0), (79.0, 61.0), 0.9),             # x 0.975: last bin; y 1.525 -> 0.525
            LabelledDetection(None, (5.0, 5.0), 0.9)]                       # no label: not binned
    per_clip = [RawClip("c", 1920, 1080, 30.0, 1.0, rows)]
    bins = cell_histogram(per_clip, 0.5)
    assert set(bins) == {"x", "y"} and all(len(per_bin) == CELL_BINS for per_bin in bins.values())
    assert [stats.visible for stats in bins["x"]] == [1, 0, 1, 0, 1] and [stats.visible for stats in bins["y"]] == [0, 0, 3, 0, 0]
    assert bins["x"][0].recall == 0.0 and bins["x"][2].recall == 1.0 and bins["x"][4].recall == 1.0
    assert bins["y"][2].recall == pytest.approx(2 / 3)
    report_raw("probe", per_clip, [0.5], histogram=True)
    out = capsys.readouterr().out
    assert "recall by position within the grid cell" in out
    assert "x 0.0-0.2  n=    1 recall=0.000 wrong=0.000 miss=1.000" in out
    assert "y 0.4-0.6  n=    3 recall=0.667" in out and "x spread: 1.000" in out and "y spread: 0.000" in out
    report_raw("probe", per_clip, [0.5])
    assert "recall by position" not in capsys.readouterr().out
