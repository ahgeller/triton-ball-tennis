"""finetune/training/train_gridtracknet.py and its data module: the script's own self-test, and the data
path - which clips train, the per-clip frame cache, and the flip augmentation's label transform - on
synthetic clips under tmp_path. No GPU, no real label or video."""
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from finetune.training import train_gridtracknet as training
from finetune.data.labels import write_labels
from finetune.training.data import (DATA_VERSION, Augmenter, ClipSelection, TrainingUnit, UnitDataset, find_clips,
                                    make_target, prepare_clip, training_targets)
from finetune.training.train_gridtracknet import better
from tests._helpers import metric_probe, write_clip

SIZE = (64, 36)


def test_training_self_test_passes():
    training.self_test()


# --------------------------------------------------------------------------- which clips train

def workspace(root: Path, clips, manifest=None, exclude=(), review_status=None) -> tuple:
    """labels/ and videos/ under `root` with a label file and an (empty) video per clip, plus the
    three records when given. Returns (videos, labels)."""
    (root / "labels").mkdir(exist_ok=True)
    (root / "videos").mkdir(exist_ok=True)
    for clip in clips:
        write_labels(root / "labels" / f"{clip}_ball.csv", {0: (1.0, 2.0)}, *SIZE)
        (root / "videos" / f"{clip}.mp4").write_bytes(b"")
    if manifest is not None:
        lines = ["clip,source,camera"] + [f"{clip},{source},{camera}" for clip, source, camera in manifest]
        (root / "clips.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if exclude:
        (root / "exclude.txt").write_text("\n".join(exclude) + "\n", encoding="utf-8")
    if review_status is not None:
        (root / "review_status.json").write_text(json.dumps(review_status), encoding="utf-8")
    return root / "videos", root / "labels"


def names(clips) -> list:
    return [clip for clip, _, _, _ in clips]


def test_find_clips_applies_exclusions_review_policy_sources_and_cameras(tmp_path, capsys):
    videos, labels = workspace(
        tmp_path, ["video1", "video13", "video14", "grid_match4", "tnv2_a"],
        manifest=[("video1", "custom", "fixed"), ("video13", "prof", "moving"), ("video14", "custom-uncorrected", ""),
                  ("grid_match4", "grid", "fixed")],
        exclude=["grid_match4  # audit"], review_status={"verified_video_min": 13, "verified_clips": []})
    assert names(find_clips(videos, labels)) == ["video13"]        # video1 unverified, video14 untrusted, grid excluded, tnv2 unknown
    assert "skipping 1 clip(s) listed in exclude.txt" in capsys.readouterr().out
    everything = ["grid_match4", "tnv2_a", "video1", "video13", "video14"]      # natural clip order
    assert names(find_clips(videos, labels, ClipSelection(include_excluded=True))) == everything
    assert names(find_clips(videos, labels, ClipSelection(include_excluded=True, sources=["custom", "grid"]))) == ["grid_match4", "video1"]
    assert names(find_clips(videos, labels, ClipSelection(include_excluded=True, cameras=["fixed"]))) == ["grid_match4", "video1"]
    assert names(find_clips(videos, labels, ClipSelection(include_excluded=True, cameras=["untagged"]))) == ["tnv2_a", "video14"]
    assert names(find_clips(videos, labels, ClipSelection(include_excluded=True, exclude_clips=["video1*", "tnv2_a"]))) == ["grid_match4"]
    for clip, video, csv_path, source in find_clips(videos, labels, ClipSelection(include_excluded=True)):
        assert video == videos / f"{clip}.mp4" and csv_path == labels / f"{clip}_ball.csv"
    assert [s for _, _, _, s in find_clips(videos, labels, ClipSelection(include_excluded=True))] == [
        "grid", "unknown", "custom", "prof", "custom-uncorrected"]


def test_find_clips_without_records_trusts_every_clip_as_custom(tmp_path):
    videos, labels = workspace(tmp_path, ["a", "b"])
    assert [(clip, source) for clip, _, _, source in find_clips(videos, labels)] == [("a", "custom"), ("b", "custom")]
    with pytest.raises(ValueError, match="--cameras needs camera tags"):
        find_clips(videos, labels, ClipSelection(cameras=["fixed"]))
    (videos / "b.mp4").unlink()
    with pytest.raises(FileNotFoundError, match="No video for b_ball.csv"):
        find_clips(videos, labels)
    with pytest.raises(FileNotFoundError, match="No .*_ball.csv files"):
        find_clips(videos, tmp_path / "empty")


# --------------------------------------------------------------------------- the frame cache

def test_training_targets_clamp_onto_the_image(tmp_path):
    csv_path = tmp_path / "c_ball.csv"
    csv_path.write_text("frame,ball_x,ball_y\nframe_000,-3,2000\nframe_001,10.5,20.5\nframe_002,63,0\n", encoding="utf-8")
    assert training_targets(csv_path, *SIZE) == {0: [0.0, 35.0], 1: [10.5, 20.5], 2: None}


def test_prepare_clip_caches_the_needed_frames_and_describes_the_units(tmp_path):
    video = write_clip(tmp_path / "c.avi", 14, SIZE, 30.0)
    csv_path = tmp_path / "c_ball.csv"
    write_labels(csv_path, {f: (10.0 + f, 20.0) if f != 3 else None for f in range(12)}, *SIZE)
    data = tmp_path / "data"
    record = prepare_clip(("c", video, csv_path, "custom"), data, rebuild=False)
    assert record["version"] == DATA_VERSION and record["clip"] == "c" and record["source"] == "custom"
    assert (record["stride"], record["size"]) == (1, list(SIZE))
    units = record["samples"]
    assert [unit["frames"] for unit in units] == [[f"frames/c/{i:06d}.jpg" for i in range(start, start + 5)]
                                                  for start in (0, 2, 4, 6)]
    assert units[0]["points"] == [[10.0, 20.0], [11.0, 20.0], [12.0, 20.0], None, [14.0, 20.0]]
    assert units[0] == TrainingUnit(clip="c", source="custom", frames=units[0]["frames"], points=units[0]["points"], size=list(SIZE))
    assert record["visible"] == sum(point is not None for unit in units for point in unit["points"])
    cached = sorted(p.name for p in (data / "frames" / "c").iterdir())
    assert cached == [f"{i:06d}.jpg" for i in range(11)]            # the last unit is 6..10; frame 11 is in none
    image = cv2.imread(str(data / "frames" / "c" / "000007.jpg"))
    assert image.shape == (training.HEIGHT, training.WIDTH, 3) and abs(int(image.mean()) - 7) <= 1
    assert json.loads((data / "clips" / "c.json").read_text(encoding="utf-8"))["signature"] == record["signature"]

    # a second call is served from the cache: the frames are not re-extracted ...
    stamp = (data / "frames" / "c" / "000000.jpg").stat().st_mtime_ns
    again = prepare_clip(("c", video, csv_path, "elsewhere"), data, rebuild=False)
    assert again["samples"] == units and again["source"] == "elsewhere"     # the source follows the manifest
    assert (data / "frames" / "c" / "000000.jpg").stat().st_mtime_ns == stamp
    # ... until the label file changes, which changes the signature
    write_labels(csv_path, {f: (10.0 + f, 20.0) for f in range(7)}, *SIZE)
    os.utime(csv_path, ns=(stamp + 10 ** 9, stamp + 10 ** 9))
    rebuilt = prepare_clip(("c", video, csv_path, "custom"), data, rebuild=False)
    assert len(rebuilt["samples"]) == 2 and rebuilt["signature"] != record["signature"]
    assert sorted(p.name for p in (data / "frames" / "c").iterdir()) == [f"{i:06d}.jpg" for i in range(7)]


def test_prepare_clip_refuses_a_video_that_ends_before_its_labels(tmp_path):
    video = write_clip(tmp_path / "short.avi", 4, SIZE, 30.0)
    csv_path = tmp_path / "short_ball.csv"
    write_labels(csv_path, {f: (1.0, 1.0) for f in range(8)}, *SIZE)
    with pytest.raises(ValueError, match="ended before frames"):
        prepare_clip(("short", video, csv_path, "custom"), tmp_path / "data", rebuild=True)


# --------------------------------------------------------------------------- augmentation

def unit_on_disk(data: Path) -> TrainingUnit:
    """One unit whose five cached frames carry a mark on their left side."""
    (data / "frames" / "u").mkdir(parents=True)
    for index in range(5):
        image = np.zeros((training.HEIGHT, training.WIDTH, 3), np.uint8)
        image[:, :10] = (0, 0, 255)                               # red (BGR) stripe at the left
        cv2.imwrite(str(data / "frames" / "u" / f"{index:06d}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 100])
    return TrainingUnit(clip="u", source="custom", frames=[f"frames/u/{i:06d}.jpg" for i in range(5)],
                        points=[[100.0, 200.0], None, [1919.0, 0.0], [0.0, 1079.0], [500.5, 500.5]], size=[1920, 1080])


def forced(dataset: UnitDataset, flip: float) -> None:
    """Make the next draws answer `flip` to the coin tosses (0.0: flip and jitter; 1.0: neither)."""
    dataset.rng = Augmenter(0)
    dataset.rng.draws = random.Random(0)
    dataset.rng.draws.random = lambda: flip


def test_flip_mirrors_the_frames_and_the_labels_about_the_image_width(tmp_path):
    unit = unit_on_disk(tmp_path)
    plain = UnitDataset(tmp_path, [unit], augment=False)
    images, target = plain[0]
    assert images.shape == (15, training.HEIGHT, training.WIDTH) and images.dtype == torch.float32
    assert float(images[0, 0, 0]) > 0.9 and float(images[0, 0, -1]) < 0.1            # red channel first (RGB); JPEG-close
    assert torch.equal(target, make_target(unit["points"], 1920, 1080))

    flipped = UnitDataset(tmp_path, [unit], augment=True)
    forced(flipped, 0.0)
    images, target = flipped[0]
    assert float(images[0, 0, -1]) > 0.5 and float(images[0, 0, 0]) < 0.5           # the stripe is on the right now
    mirrored = [None if p is None else [1920 - 1 - p[0], p[1]] for p in unit["points"]]
    assert torch.equal(target, make_target(mirrored, 1920, 1080))
    assert mirrored[2] == [0.0, 0.0] and mirrored[3] == [1919.0, 1079.0]           # pixel 0 <-> pixel width-1

    unflipped = UnitDataset(tmp_path, [unit], augment=True)
    forced(unflipped, 1.0)
    images, target = unflipped[0]
    assert float(images[0, 0, 0]) > 0.9 and torch.equal(target, make_target(unit["points"], 1920, 1080))


# --------------------------------------------------------------------------- the metric and the winner

def test_metric_counts_localization_errors():
    stats = metric_probe(5, {i: [100, 100] if i == 0 else [120, 100] for i in range(5)})
    assert stats["visible"] == 5
    assert stats["recall"] == .2


def test_metric_infers_tail():
    stats = metric_probe(7, {i: [100, 100] for i in range(7)})
    assert stats["visible"] == 7
    assert stats["recall"] == 1.


def test_false_alarms_block_promotion():
    assert not better({"recall": .91, "wrong": 0., "false_alarm": 1.},
                      {"recall": .90, "wrong": 0., "false_alarm": 0.})
