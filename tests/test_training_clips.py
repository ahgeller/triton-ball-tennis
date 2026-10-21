"""finetune/training/clips.py on its own: the clip selection rule and the validation split.
Synthetic inputs only; find_clips is exercised end to end in test_train_gridtracknet.py."""
import json

import pytest

from finetune.data.labels import write_labels
from finetune.training.clips import DEFAULT_VAL_CLIP, ClipSelection, matches, resolve_val_clips, validation_split

SIZE = (64, 36)


# --------------------------------------------------------------------------- which clips

def test_selection_defaults_follow_the_review_policy_and_the_exclusions():
    status = {"verified_video_min": 13, "verified_clips": []}
    plain = ClipSelection()
    assert plain.admits("video13", {}, "prof", status, set())
    assert not plain.admits("video1", {}, "custom", status, set())                 # unverified
    assert not plain.admits("video13", {}, "prof", status, {"video13"})           # exclude.txt
    assert ClipSelection(include_excluded=True).admits("video1", {}, "custom", status, set())


def test_selection_filters_by_source_camera_and_glob():
    status = {"verified_video_min": 1, "verified_clips": ["grid_match4"]}
    assert not ClipSelection(sources=["grid"]).admits("video13", {}, "prof", status, set())
    assert ClipSelection(sources=["grid"]).admits("grid_match4", {}, "grid", status, set())
    assert ClipSelection(cameras=["fixed"]).admits("video13", {"camera": "fixed"}, "prof", status, set())
    assert not ClipSelection(cameras=["fixed"]).admits("video13", {}, "prof", status, set())
    assert ClipSelection(cameras=["untagged"]).admits("video13", {}, "prof", status, set())
    assert not ClipSelection(exclude_clips=["video1*"]).admits("video13", {}, "prof", status, set())
    assert matches("video13", ["video1*"]) and not matches("video13", None) and not matches("video13", ["grid*"])


# --------------------------------------------------------------------------- the validation split

def test_validation_clips_come_from_globs_or_the_default():
    assert resolve_val_clips(["video1", "video10", "tnv2_Test_match1_1"], ["tnv2_Test_*"]) == ["tnv2_Test_match1_1"]
    assert resolve_val_clips(["video1", DEFAULT_VAL_CLIP], None) == [DEFAULT_VAL_CLIP]
    assert resolve_val_clips(["video1", "video2"], None) == ["video2"]
    with pytest.raises(ValueError, match="match nothing"):
        resolve_val_clips(["video1"], ["nope*"])


def clips_in(root, names):
    (root / "labels").mkdir(exist_ok=True)
    (root / "videos").mkdir(exist_ok=True)
    out = []
    for name in names:
        csv_path = root / "labels" / f"{name}_ball.csv"
        write_labels(csv_path, {0: (1.0, 2.0)}, *SIZE)
        out.append((name, root / "videos" / f"{name}.mp4", csv_path, "custom"))
    return out


def test_validation_split_takes_whole_recording_groups_and_needs_reviewed_labels(tmp_path):
    clips = clips_in(tmp_path, ["video13", "video14", "video15"])
    (tmp_path / "clips.csv").write_text("clip,source,group\nvideo13,prof,g1\nvideo14,prof,g1\nvideo15,prof,g2\n",
                                        encoding="utf-8")
    (tmp_path / "review_status.json").write_text(json.dumps({"verified_video_min": 13, "verified_clips": []}),
                                                 encoding="utf-8")
    assert validation_split(clips, tmp_path / "labels", ["video13"]) == ["video13", "video14"]
    with pytest.raises(ValueError, match="no training clips"):
        validation_split(clips, tmp_path / "labels", ["video13", "video15"])
    (tmp_path / "review_status.json").write_text(json.dumps({"verified_video_min": 15, "verified_clips": []}),
                                                 encoding="utf-8")
    with pytest.raises(ValueError, match="reviewed labels"):
        validation_split(clips, tmp_path / "labels", ["video13"])


def test_validation_split_keeps_the_rallies_of_one_recording_together(tmp_path):
    """grid_match47, 49 and 50 are three rallies cut from one video: their clips.csv groups are
    their own names, so only the origin says they are one recording (``recording_of``). Holding
    out one of them used to leave the other two in training."""
    clips = clips_in(tmp_path, ["grid_match47", "grid_match49", "grid_match50", "grid_match53"])
    (tmp_path / "clips.csv").write_text(
        "clip,source,group,origin\n"
        "grid_match47,grid,grid_match47,C:\\data\\match47\\gravel amature.mp4\n"
        "grid_match49,grid,grid_match49,C:\\data\\match47\\gravel amature.mp4\n"
        "grid_match50,grid,grid_match50,C:\\data\\match47\\gravel amature.mp4\n"
        "grid_match53,grid,grid_match53,C:\\data\\match53\\frames\n", encoding="utf-8")
    (tmp_path / "review_status.json").write_text(json.dumps({"verified_clips": ["grid_match47", "grid_match49", "grid_match50", "grid_match53"]}),
                                                 encoding="utf-8")
    assert validation_split(clips, tmp_path / "labels", ["grid_match49"]) == ["grid_match47", "grid_match49", "grid_match50"]
    assert validation_split(clips, tmp_path / "labels", ["grid_match53"]) == ["grid_match53"]


def test_validation_split_takes_a_named_held_out_set(tmp_path):
    clips = clips_in(tmp_path, ["video1", "video2", "video13", "video14", "grid_match8"])
    (tmp_path / "clips.csv").write_text("clip,source,group\nvideo1,custom,own-camera\nvideo2,custom,own-camera\n"
                                        "video13,prof,prof_game2\nvideo14,prof,prof_game2\ngrid_match8,grid,grid_match8\n",
                                        encoding="utf-8")
    (tmp_path / "review_status.json").write_text(json.dumps({"verified_video_min": 1, "verified_clips": ["grid_match8"]}),
                                                 encoding="utf-8")
    (tmp_path / "holdouts.json").write_text(json.dumps({"mixed": {"groups": ["own-camera"], "clips": ["grid_match8"]},
                                                        "none_here": {"clips": ["video1"]}}), encoding="utf-8")
    assert validation_split(clips, tmp_path / "labels", None, "mixed") == ["video1", "video2", "grid_match8"]
    assert validation_split(clips[2:], tmp_path / "labels", None, "mixed") == ["grid_match8"]    # a run on fewer clips
    with pytest.raises(ValueError, match="covers none of this run's clips"):
        validation_split(clips[2:4], tmp_path / "labels", None, "none_here")
    with pytest.raises(ValueError, match="alternatives"):
        validation_split(clips, tmp_path / "labels", ["video13"], "mixed")
    with pytest.raises(ValueError, match="no held-out set 'typo'"):
        validation_split(clips, tmp_path / "labels", None, "typo")
