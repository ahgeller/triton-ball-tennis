"""finetune/refiner/refiner_training.py on synthetic clips and arrays: the one-pass crop walk, the
training batches, the self-check threshold and which clips it gathers. No real workspace file
(``workspace_dir`` is a temporary one), no GPU."""
import json

import numpy as np
import pytest
import torch

from finetune.data.labels import write_labels
from finetune.data.workspace import add_exclusion, label_csv, review_json
from finetune.refiner.cnn_refiner import CROP, WINDOW
from finetune.refiner.refiner_training import (FAR_PX, SURE_FAR_SHARE, crops_from_video, cut_label, frame_users,
                                                gather_crops, sample_batch, sure_threshold, to_device)
from tests._helpers import write_clip

SIZE = (160, 90)


def test_frame_users_counts_every_label_that_needs_a_frame():
    users = frame_users({4: (1.0, 1.0), 6: (2.0, 2.0)}, 2)
    assert users == {2: 1, 4: 2, 6: 2, 8: 1}


def test_cut_label_lets_go_of_frames_no_later_label_needs():
    kept = {i: np.full((SIZE[1], SIZE[0], 3), i, np.uint8) for i in (2, 4, 6)}
    users = frame_users({4: (80.0, 45.0), 6: (80.0, 45.0)}, 2)
    crop = cut_label(4, kept, users, (80.0, 45.0), 2, SIZE[1])
    assert crop.shape == (CROP, CROP, 9) and crop[CROP // 2, CROP // 2, 0] == 2 and crop[CROP // 2, CROP // 2, 3] == 4
    assert 2 not in kept and 4 in kept and 6 in kept                    # frame 4 and 6 still serve label 6
    assert cut_label(9, kept, {7: 1, 9: 1, 11: 1}, (1.0, 1.0), 2, SIZE[1]) is None   # never decoded


def test_crops_from_video_reads_the_clip_once_and_skips_labels_past_its_end(tmp_path):
    video = write_clip(tmp_path / "clip.avi", 8, SIZE)
    crops = crops_from_video(video, {2: (80.0, 45.0), 5: (80.0, 45.0), 20: (80.0, 45.0)}, 1, SIZE[1])
    assert len(crops) == 2
    assert [int(c[CROP // 2, CROP // 2, 3]) for c in crops] == [2, 5]       # the label frame is the middle one
    assert [int(c[CROP // 2, CROP // 2, 0]) for c in crops] == [1, 4]       # with the frame before ...
    assert [int(c[CROP // 2, CROP // 2, 6]) for c in crops] == [3, 6]       # ... and the frame after


def test_to_device_stacks_the_parts_and_sample_batch_cuts_windows_with_targets():
    parts = [np.full((3, CROP, CROP, 9), 7, np.uint8), np.full((2, CROP, CROP, 9), 9, np.uint8)]
    data = to_device(parts, "cpu")
    assert data.shape == (5, CROP, CROP, 9) and int(data[3, 0, 0, 0]) == 9
    rng = np.random.default_rng(0)
    x, target = sample_batch(data, "cpu", rng, 4, flip=False, in_order=True)
    assert x.shape == (4, 9, WINDOW, WINDOW) and target.shape == (4, 2)
    assert torch.all((0 <= target) & (target <= WINDOW - 1))
    assert float(x.min()) == pytest.approx(7 / 255.0 - 0.5, abs=1e-6)
    flipped, _ = sample_batch(data, "cpu", np.random.default_rng(0), 4)
    assert flipped.shape == x.shape


def test_sure_threshold_keeps_far_misses_rare():
    confs = np.linspace(1.0, 0.0, 3000)
    errors = np.where(np.arange(3000) >= 2000, FAR_PX + 4.0, 0.5)
    threshold = sure_threshold(errors, confs)
    sure = confs >= threshold
    assert np.mean(errors[sure] > FAR_PX) <= SURE_FAR_SHARE and sure.sum() > 1990
    assert sure_threshold(np.array([9.0, 9.0]), np.array([0.9, 0.8])) == 1.0    # nothing is ever sure


def test_gather_crops_skips_clips_listed_in_exclude_txt(workspace_dir):
    for clip in ("clip_a", "clip_b"):
        write_clip(workspace_dir / "videos" / f"{clip}.avi", 8, SIZE)
        write_labels(label_csv(clip), {f: (80.0, 45.0) for f in range(8)}, *SIZE)
        review_json(clip).write_text(json.dumps({"reviewed": [2, 3, 4, 5]}), encoding="utf-8")
    add_exclusion("clip_b", "labels belong to another video")
    lines = []
    crops = gather_crops(lambda *parts: lines.append(" ".join(map(str, parts))))
    assert sorted(crops) == ["clip_a"] and len(crops["clip_a"]) == 4
    assert any("clip_b" in line and "exclude.txt" in line for line in lines)
