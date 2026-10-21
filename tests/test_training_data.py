"""finetune/training/data.py on its own: the clip selection rule, the five-frame units, the grid
targets, the sampler, the validation split and the augmentation. Synthetic inputs only; the frame
cache and find_clips are exercised end to end in test_train_gridtracknet.py."""
import json

import numpy as np
import pytest
import torch

from finetune.data.labels import write_labels
from finetune.training.data import (DEFAULT_VAL_CLIP, Augmenter, ClipSelection, build_units, make_sampler, make_target,
                                    matches, photometric_augment, resolve_val_clips, validation_split)
from tennis_tracker.gridtracknet import FRAMES_PER_UNIT, GRID_COLS, GRID_ROWS

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


# --------------------------------------------------------------------------- units and targets

def test_units_advance_two_label_frames_and_need_every_frame_labelled():
    even = {index: [1.0, 1.0] for index in range(0, 20, 2)}
    assert build_units(even, 2)[:2] == [[0, 2, 4, 6, 8], [4, 6, 8, 10, 12]]
    odd = {index: None for index in range(1, 20, 2)}
    assert build_units(odd, 2)[:2] == [[1, 3, 5, 7, 9], [5, 7, 9, 11, 13]]
    gapped = {index: None for index in range(0, 12) if index != 5}
    assert all(5 not in window for window in build_units(gapped, 1))
    assert build_units({}, 1) == []


def test_target_marks_the_cell_and_its_offsets_and_nothing_for_an_absent_ball():
    target = make_target([[960.0, 540.0], None, [0.0, 0.0], [1919.0, 1079.0], None], 1920, 1080)
    assert target.shape == (FRAMES_PER_UNIT * 3, GRID_ROWS, GRID_COLS)
    assert float(target[0, 13, 24]) == 1.0 and float(target[0].sum()) == 1.0
    assert float(target[3].sum()) == 0.0                                            # frame 1: no ball
    assert float(target[6, 0, 0]) == 1.0 and float(target[7, 0, 0]) == 0.0 == float(target[8, 0, 0])
    assert float(target[9, GRID_ROWS - 1, GRID_COLS - 1]) == 1.0                    # clamped into the last cell
    assert 0.0 <= float(target[10, GRID_ROWS - 1, GRID_COLS - 1]) < 1.0


# --------------------------------------------------------------------------- the sampler

def test_sampler_weights_by_source_and_is_absent_without_weights():
    units = [{"source": "a"}, {"source": "b"}, {"source": "a"}]
    assert make_sampler(units, {}, None) is None
    only_a = make_sampler(units, {"b": 0.0}, None)
    assert only_a is not None and len(only_a) == 2 and set(iter(only_a)) <= {0, 2}   # drawn with replacement
    assert len(make_sampler(units, {}, 7)) == 7                                     # --epoch-units draws that many
    with pytest.raises(ValueError, match="weight 0"):
        make_sampler(units, {"a": 0.0, "b": 0.0}, None)


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


# --------------------------------------------------------------------------- augmentation

def test_augmenter_is_seeded_and_photometric_augment_keeps_shape_and_range():
    first, second = Augmenter(3), Augmenter(3)
    assert first.draws.random() == second.draws.random()
    assert float(first.noise.normal()) == float(second.noise.normal())
    images = np.full((15, 8, 8), 0.5, dtype=np.float32)
    augmented = photometric_augment(images, Augmenter(1))
    assert augmented.shape == images.shape and augmented.dtype == np.float32
    assert 0.0 <= float(augmented.min()) <= float(augmented.max()) <= 1.0
    assert not np.array_equal(augmented, images)
    assert np.array_equal(photometric_augment(images, Augmenter(1)), augmented)   # the same seed, the same draw
    assert isinstance(make_target([None] * 5, *SIZE), torch.Tensor)
