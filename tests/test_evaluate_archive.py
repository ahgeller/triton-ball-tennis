"""evaluation.archive (evaluate_archive.py) helpers on synthetic inputs: no GPU, no real archive, never finetune/labels."""
import json
from pathlib import Path

import pytest

from evaluation import archive as evaluate_archive
from evaluation.archive import (
    LINE_PX, _inside_player, _near_court_line, _segment_distance, list_clips, score_pipeline_clip,
)


def _archive(tmp_path: Path) -> Path:
    (tmp_path / "labels").mkdir()
    (tmp_path / "videos").mkdir()
    for clip in ("video1", "video2", "video3"):
        (tmp_path / "labels" / f"{clip}_ball.csv").write_text("frame,ball_x,ball_y\n")
    (tmp_path / "videos" / "video1.mp4").write_bytes(b"")      # only the name is checked
    (tmp_path / "videos" / "video2.mp4").write_bytes(b"")
    (tmp_path / "review_status.json").write_text(json.dumps({"verified_clips": ["video1", "video3"]}))
    return tmp_path


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
    assert isinstance(result, evaluate_archive.ClipScore)
    assert result.by_source == {"det": {"hit": 1, "localized_error": 0, "wrong": 1, "invisible": 0}}
    assert result.landed == {"other": 1}
    assert result.bad_frames == [(1, "wrong", "det", "other")]
    assert result.pairs == [((10, 10), (10, 10)), ((10, 10), (500, 500)), ((50, 50), None), (None, None)]
    assert result.stats["visible"] == 3
    assert result.stats["recall"] == pytest.approx(1 / 3)
    assert result.stats["wrong_obj"] == pytest.approx(1 / 3)
    assert result.stats["miss"] == pytest.approx(1 / 3)
    assert result.stats["invisible"] == 1 and result.stats["false_alarm"] == 0.0


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
