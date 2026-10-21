"""finetune/training/train_gridtracknet.py and the modules under it: the script's own self-test, and the data
path - which clips train, the per-clip frame cache, and the flip augmentation's label transform - on
synthetic clips under tmp_path. No GPU, no real label or video."""
import csv
import json
import os
import random
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import torch

from finetune.training import train_gridtracknet as training
from finetune.data.detector_metrics import DetectorMetrics
from finetune.data.labels import write_labels
from finetune.training.clips import ClipSelection, find_clips
from finetune.training.dataset import (TRANSLATE_PX, Augmenter, UnitDataset, compress, make_target, reverse, translate,
                                       translate_limits)
from finetune.training.frame_cache import DATA_VERSION, TrainingUnit, prepare_clip, training_targets
from finetune.training.loop import fresh_model
from finetune.training.metrics import promotion_improves
from tennis_tracker.gridtracknet import load_model
from tests._helpers import blank_template, metric_probe, write_clip

SIZE = (64, 36)


def test_training_self_test_passes():
    training.self_test()


def test_main_reports_a_run_that_cannot_start_as_status_1(tmp_path, capsys):
    """No clips, or an option the clips cannot satisfy: one line on stderr and exit status 1, the
    same as the other scripts' setup errors, before any cache or run folder is written."""
    videos, labels = workspace(tmp_path, ["a", "b"])
    folders = ["--data-dir", str(tmp_path / "cache"), "--run-dir", str(tmp_path / "run")]
    assert training.main(["--videos", str(videos), "--labels", str(tmp_path / "empty"), *folders]) == 1
    assert "[error] No *_ball.csv files" in capsys.readouterr().err
    assert training.main(["--videos", str(videos), "--labels", str(labels), "--cameras", "fixed", *folders]) == 1
    assert "[error] --cameras needs camera tags" in capsys.readouterr().err
    assert not (tmp_path / "cache").exists() and not (tmp_path / "run").exists()


def test_holdout_and_val_clips_are_alternatives_and_the_config_records_the_choice(capsys):
    parser = training.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--holdout", "mixed_v1", "--val-clips", "video10"])
    assert "not allowed with argument" in capsys.readouterr().err
    args = parser.parse_args(["--holdout", "mixed_v1", "--epochs", "3"])
    config = training.run_config(args, ["video1", "video13"], ["video1"], {"custom": 2.0})
    assert config["holdout"] == "mixed_v1" and config["val_clips"] is None and config["validation_clips"] == ["video1"]
    assert config["clips"] == ["video1", "video13"] and config["epochs"] == 3 and config["oversample"] == {"custom": 2.0}
    args = parser.parse_args(["--val-clips", "video10", "grid_match*"])
    config = training.run_config(args, ["video10"], ["video10"], {})
    assert config["holdout"] is None and config["val_clips"] == ["video10", "grid_match*"]


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
    assert [line for line in capsys.readouterr().out.splitlines() if line.startswith("[skip]")] == [
        "[skip] grid_match4: listed in exclude.txt (--include-excluded keeps it)",
        "[skip] tnv2_a: labels are still under review (--include-excluded keeps it)",     # unreviewed before unlisted
        "[skip] video1: labels are still under review (--include-excluded keeps it)",
        "[skip] video14: source 'custom-uncorrected' is machine labels nobody checked (--include-excluded keeps it)",
    ]
    assert names(find_clips(videos, labels, ClipSelection(include_excluded=True, sources=["prof"]))) == ["video13"]
    assert "[skip]" not in capsys.readouterr().out                 # a run's own choices are not announced
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
    assert [unit["frames"] for unit in units] == [[f"frames/c/{i:06d}.png" for i in range(start, start + 5)]
                                                  for start in (0, 2, 4, 6)]
    assert units[0]["points"] == [[10.0, 20.0], [11.0, 20.0], [12.0, 20.0], None, [14.0, 20.0]]
    assert units[0] == TrainingUnit(clip="c", source="custom", indices=[0, 1, 2, 3, 4], frames=units[0]["frames"],
                                    points=units[0]["points"], size=list(SIZE))
    assert record["visible"] == sum(point is not None for unit in units for point in unit["points"])
    cached = sorted(p.name for p in (data / "frames" / "c").iterdir())
    assert cached == [f"{i:06d}.png" for i in range(11)]            # the last unit is 6..10; frame 11 is in none
    assert record["frames"] == list(range(11))
    image = cv2.imread(str(data / "frames" / "c" / "000007.png"))
    assert image.shape == (training.HEIGHT, training.WIDTH, 3) and int(image.mean()) == 7      # lossless
    assert json.loads((data / "clips" / "c.json").read_text(encoding="utf-8")) == record

    # a second call is served from the cache: the frames are not re-extracted ...
    stamp = (data / "frames" / "c" / "000000.png").stat().st_mtime_ns
    again = prepare_clip(("c", video, csv_path, "elsewhere"), data, rebuild=False)
    assert [unit["frames"] for unit in again["samples"]] == [unit["frames"] for unit in units]
    assert again["source"] == "elsewhere" and {unit["source"] for unit in again["samples"]} == {"elsewhere"}   # the source follows the manifest
    assert (data / "frames" / "c" / "000000.png").stat().st_mtime_ns == stamp
    # ... and a label edit only changes the units: the signature is the video's, the frames stay
    write_labels(csv_path, {f: (10.0 + f, 20.0) for f in range(7)}, *SIZE)
    os.utime(csv_path, ns=(stamp + 10 ** 9, stamp + 10 ** 9))
    edited = prepare_clip(("c", video, csv_path, "custom"), data, rebuild=False)
    assert len(edited["samples"]) == 2 and edited["signature"] == record["signature"]
    assert (data / "frames" / "c" / "000000.png").stat().st_mtime_ns == stamp
    assert sorted(p.name for p in (data / "frames" / "c").iterdir()) == cached
    # --rebuild-data extracts everything the units need afresh and drops the rest
    rebuilt = prepare_clip(("c", video, csv_path, "custom"), data, rebuild=True)
    assert rebuilt["frames"] == list(range(7)) and (data / "frames" / "c" / "000000.png").stat().st_mtime_ns > stamp
    assert sorted(p.name for p in (data / "frames" / "c").iterdir()) == [f"{i:06d}.png" for i in range(7)]


def test_prepare_clip_refuses_a_video_that_ends_before_its_labels(tmp_path):
    video = write_clip(tmp_path / "short.avi", 4, SIZE, 30.0)
    csv_path = tmp_path / "short_ball.csv"
    write_labels(csv_path, {f: (1.0, 1.0) for f in range(8)}, *SIZE)
    with pytest.raises(ValueError, match="ended before frames"):
        prepare_clip(("short", video, csv_path, "custom"), tmp_path / "data", rebuild=True)


# --------------------------------------------------------------------------- augmentation

def unit_on_disk(data: Path, stripe=(0, 10)) -> TrainingUnit:
    """One unit whose five cached frames carry a red stripe over the cache columns `stripe`
    (the left edge by default)."""
    (data / "frames" / "u").mkdir(parents=True, exist_ok=True)
    for index in range(5):
        image = np.zeros((training.HEIGHT, training.WIDTH, 3), np.uint8)
        image[:, stripe[0]:stripe[1]] = (0, 0, 255)               # red (BGR) stripe
        cv2.imwrite(str(data / "frames" / "u" / f"{index:06d}.png"), image)
    return TrainingUnit(clip="u", source="custom", indices=list(range(5)), frames=[f"frames/u/{i:06d}.png" for i in range(5)],
                        points=[[100.0, 200.0], None, [1919.0, 0.0], [0.0, 1079.0], [500.5, 500.5]], size=[1920, 1080])


def forced(dataset: UnitDataset, flip: float) -> None:
    """Make the next draws answer `flip` to the coin tosses (0.0: flip and jitter; 1.0: neither)."""
    dataset.rng = Augmenter(0)
    dataset.rng.draws = random.Random(0)
    dataset.rng.draws.random = lambda: flip


def scripted(dataset: UnitDataset, tosses, integers=()) -> None:
    """Make the coin tosses answer `tosses` in turn and the integer draws `integers`."""
    dataset.rng = Augmenter(0)
    dataset.rng.draws = random.Random(0)
    dataset.rng.draws.random = iter(tosses).__next__
    dataset.rng.draws.randint = lambda low, high: next(remaining)
    remaining = iter(integers)


def stripe_centre(images: torch.Tensor) -> float:
    """Where the red stripe's mass sits, in the unit's 1920-wide frame coordinates."""
    red = images[0].numpy().sum(axis=0)
    columns = np.arange(training.WIDTH) + 0.5
    return float((red * columns).sum() / red.sum()) * 1920 / training.WIDTH


def decoded_x(target: torch.Tensor, frame: int = 0) -> float:
    row, col = np.argwhere(target[frame * 3].numpy() == 1)[0]
    return (col + float(target[frame * 3 + 1, row, col])) * 1920 / training.GRID_COLS


def test_flip_mirrors_the_frames_and_the_labels_about_the_image_width(tmp_path):
    unit = unit_on_disk(tmp_path)
    plain = UnitDataset(tmp_path, [unit], augment=False)
    images, target = plain[0]
    assert images.shape == (15, training.HEIGHT, training.WIDTH) and images.dtype == torch.float32
    assert float(images[0, 0, 0]) == 1.0 and float(images[0, 0, -1]) == 0.0           # red channel first (RGB); lossless
    assert torch.equal(target, make_target(unit["points"], 1920, 1080))

    flipped = UnitDataset(tmp_path, [unit], augment=True)
    forced(flipped, 0.0)
    images, target = flipped[0]
    assert float(images[0, 0, -1]) > 0.5 and float(images[0, 0, 0]) < 0.5           # the stripe is on the right now
    mirrored = [None if p is None else [1920 - p[0], p[1]] for p in unit["points"]]
    assert torch.equal(target, make_target(mirrored, 1920, 1080))
    assert mirrored[2] == [1.0, 0.0] and mirrored[3] == [1920.0, 1079.0]   # the left edge of pixel 0 <-> the right edge of pixel 1919

    unflipped = UnitDataset(tmp_path, [unit], augment=True)
    forced(unflipped, 1.0)
    images, target = unflipped[0]
    assert float(images[0, 0, 0]) > 0.9 and torch.equal(target, make_target(unit["points"], 1920, 1080))


def test_flipped_label_lands_on_the_flipped_pixels(tmp_path):
    # The stripe covers pixels 0-9 of the 768-wide cache (0-24 at the unit's 1920), so its
    # centre is 12.5 in frame coordinates (pixel k's centre is k + 0.5). After the flip the
    # pixels sit at 758-767 and the label must follow them, to 1907.5: mirroring pixel
    # indices (1920 - 1 - 12.5) would put it a pixel to the left.
    unit = unit_on_disk(tmp_path)
    unit["points"] = [[12.5, 200.0], None, None, None, None]
    dataset = UnitDataset(tmp_path, [unit], augment=True)
    forced(dataset, 0.0)
    images, target = dataset[0]
    red = images[0].numpy().sum(axis=0)                                            # red mass per cached column
    columns = np.arange(training.WIDTH) + 0.5
    stripe_centre = float((red * columns).sum() / red.sum()) * 1920 / training.WIDTH
    row, col = np.argwhere(target[0].numpy() == 1)[0]
    decoded_x = (col + float(target[1, row, col])) * 1920 / training.GRID_COLS
    assert abs(stripe_centre - 1907.5) < 0.05
    assert decoded_x == pytest.approx(1907.5) and decoded_x == pytest.approx(stripe_centre, abs=0.05)


def test_translate_moves_the_stripe_and_the_label_by_the_same_shift(tmp_path):
    # A stripe on cache pixels 40-49 has its centre at 45 cache pixels, 112.5 in the unit's
    # 1920-wide frame coordinates; a shift of 16 cache pixels is 40 native ones, so stripe and
    # label land at 152.5. (The stripe sits away from the edge because the border is replicated,
    # which would stretch an edge stripe over the gap the shift opens.)
    unit = unit_on_disk(tmp_path, stripe=(40, 50))
    unit["points"] = [[112.5, 200.0], None, None, None, None]
    dataset = UnitDataset(tmp_path, [unit], augment=True, augmentations={"translate"})
    scripted(dataset, tosses=[1.0, 0.0], integers=[16, 0])               # no flip; translate, dx=16, dy=0
    images, target = dataset[0]
    assert abs(stripe_centre(images) - 152.5) < 0.05
    assert decoded_x(target) == pytest.approx(152.5) and decoded_x(target) == pytest.approx(stripe_centre(images), abs=0.05)
    for frame in range(1, 5):                                             # every frame shifted alike
        assert torch.equal(images[frame * 3:frame * 3 + 3], images[:3])
    assert float(target[3:].sum()) == 0.0                                 # no ball stays no ball
    edge = unit_on_disk(tmp_path, stripe=(0, 10))
    edge["points"] = [[12.5, 200.0], None, None, None, None]
    dataset = UnitDataset(tmp_path, [edge], augment=True, augmentations={"translate"})
    scripted(dataset, tosses=[1.0, 0.0], integers=[16, 0])
    images, target = dataset[0]
    assert float(images[0, :, :26].mean()) > 0.99 and float(images[0, :, 26:].mean()) < 0.01     # replicated: red fills the gap
    assert decoded_x(target) == pytest.approx(52.5)


def test_translate_is_held_so_that_no_label_leaves_the_image(tmp_path):
    unit = unit_on_disk(tmp_path)                                         # labels at x=0 and x=1919, y=0 and y=1079
    assert translate_limits(unit["points"], 1920, 1080, training.WIDTH, training.HEIGHT) == (0, 0, 0, 0)
    assert translate_limits([[12.5, 200.0], None], 1920, 1080, training.WIDTH, training.HEIGHT) == (-5, 20, -20, 20)
    assert translate_limits([[1910.0, 1070.0]], 1920, 1080, training.WIDTH, training.HEIGHT) == (-20, 3, -20, 3)
    assert translate_limits([None] * 5, 1920, 1080, training.WIDTH, training.HEIGHT) == (-TRANSLATE_PX,) * 1 + (TRANSLATE_PX, -TRANSLATE_PX, TRANSLATE_PX)
    images = np.zeros((15, training.HEIGHT, training.WIDTH), np.float32)
    images[:, :, :10] = 1.0
    shifted, points = translate(images, [[1919.0, 1079.0], None, [0.0, 0.0]], 0, 0, 1920, 1080)
    assert np.array_equal(shifted, images) and points == [[1919.0, 1079.0], None, [0.0, 0.0]]
    shifted, points = translate(images, [[100.0, 100.0]], -4, 8, 1920, 1080)
    assert points == [[90.0, 120.0]] and shifted.shape == images.shape
    assert float(shifted[0, 0, :6].sum()) == 6.0 and float(shifted[0, 0, 6]) == 0.0      # the stripe moved 4 left
    assert float(shifted[0, 7].sum()) == 6.0 == float(shifted[0, 8].sum())                # rows moved 8 down, the top replicated
    dataset = UnitDataset(tmp_path, [unit], augment=True, augmentations={"translate"})
    scripted(dataset, tosses=[1.0, 0.0], integers=[0, 0])                 # the clamped draw: randint(0, 0)
    images, target = dataset[0]
    assert torch.equal(target, make_target(unit["points"], 1920, 1080))


def test_reverse_reverses_the_frames_and_the_targets(tmp_path):
    unit = unit_on_disk(tmp_path)
    plain, forward = UnitDataset(tmp_path, [unit], augment=False)[0]
    dataset = UnitDataset(tmp_path, [unit], augment=True, augmentations={"reverse"})
    scripted(dataset, tosses=[1.0, 0.0])                                  # no flip; reverse
    images, target = dataset[0]
    assert torch.equal(target, make_target(unit["points"][::-1], 1920, 1080))
    assert torch.equal(images.reshape(5, 3, training.HEIGHT, training.WIDTH).flip(0).reshape(15, training.HEIGHT, training.WIDTH), plain)
    frames = np.arange(15, dtype=np.float32).reshape(15, 1, 1) * np.ones((15, 2, 2), np.float32)
    reversed_frames, points = reverse(frames, [[1.0, 1.0], None, [3.0, 3.0], [4.0, 4.0], [5.0, 5.0]])
    assert reversed_frames[:, 0, 0].tolist() == [12, 13, 14, 9, 10, 11, 6, 7, 8, 3, 4, 5, 0, 1, 2]
    assert points == [[5.0, 5.0], [4.0, 4.0], [3.0, 3.0], None, [1.0, 1.0]]


def test_compress_keeps_the_shape_and_the_labels_and_changes_only_the_pixels(tmp_path):
    unit = unit_on_disk(tmp_path)
    plain, forward = UnitDataset(tmp_path, [unit], augment=False)[0]
    dataset = UnitDataset(tmp_path, [unit], augment=True, augmentations={"compress"})
    scripted(dataset, tosses=[1.0, 0.0, 1.0], integers=[35])             # no flip; compress at quality 35; no blur
    images, target = dataset[0]
    assert images.shape == plain.shape and images.dtype == torch.float32 and torch.equal(target, forward)
    assert 0.0 <= float(images.min()) <= float(images.max()) <= 1.0
    assert not torch.equal(images, plain) and float((images - plain).abs().mean()) < 0.05    # artefacts, not another picture
    assert float(images[0, 100, :5].mean()) > 0.8 and float(images[0, 100, 100:].mean()) < 0.1
    rough = compress(plain.numpy(), 35)
    assert np.array_equal(rough, images.numpy())                          # the dataset is the function, no more
    assert float(np.abs(compress(plain.numpy(), 95) - plain.numpy()).mean()) < float(np.abs(rough - plain.numpy()).mean())


def test_new_augmentations_draw_after_the_old_ones_so_old_runs_reproduce(tmp_path):
    """A run with --augment basic makes exactly the draws it always made: the newer augmentations
    draw nothing unless named, so the seed still reproduces an old run."""
    unit = unit_on_disk(tmp_path)
    before, after = UnitDataset(tmp_path, [unit], augment=True), UnitDataset(tmp_path, [unit], augment=True, augmentations={"basic"})
    before.rng, after.rng = Augmenter(5), Augmenter(5)
    for _ in range(3):
        images, target = before[0]
        again, target_again = after[0]
        assert torch.equal(images, again) and torch.equal(target, target_again)
    assert before.rng.draws.random() == after.rng.draws.random()         # the streams are in step
    everything = UnitDataset(tmp_path, [unit], augment=True, augmentations=set(training.AUGMENTATIONS))
    everything.rng = Augmenter(5)
    images, target = everything[0]
    assert images.shape == (15, training.HEIGHT, training.WIDTH) and target.shape == (15, training.GRID_ROWS, training.GRID_COLS)
    with pytest.raises(ValueError, match="unknown augmentation"):
        UnitDataset(tmp_path, [unit], augment=True, augmentations={"basic", "rotate"})
    assert UnitDataset(tmp_path, [unit], augment=False, augmentations={"translate"}).augmentations == set()


# --------------------------------------------------------------------------- the metric and the winner

def scripted_run(tmp_path, argv, train_losses, val_losses, scored):
    """``fit`` on a fresh CPU model with the epoch losses scripted and the held-out metric answered
    from `scored` (recall per scored call, wrong 0); returns (args, selection, the metric calls)."""
    model = fresh_model(torch.device("cpu"))
    with torch.no_grad():
        model.convs[0].bias.zero_()
    template = blank_template(model, tmp_path / "template.npz")
    args = training.build_parser().parse_args(["--run-dir", str(tmp_path / "run"), "--weights", str(template), *argv])
    args.run_dir.mkdir()
    calls = []

    def score(weights):
        calls.append(weights)
        return training.metrics_of(scored[len(calls) - 1], 0.0)

    def train_epoch(model, *_):
        with torch.no_grad():
            model.convs[0].bias += 1.0          # "training": the first bias counts the epochs, so each saved file is its epoch
        return train_losses[int(model.convs[0].bias[0]) - 1]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    with patch.object(training, "train_epoch", side_effect=train_epoch), \
            patch.object(training, "evaluate_loss", side_effect=val_losses):
        selection = training.fit(args, model, None, None, optimizer, scaler, torch.device("cpu"), score)
    return args, selection, calls


def results_of(run_dir):
    with (run_dir / "results.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_best_npz_is_the_epoch_the_detector_metric_prefers_not_the_lowest_loss(tmp_path):
    """Three epochs: the validation loss keeps falling, the held-out recall peaks at epoch 2 and
    epoch 3 only ties it. best.npz is epoch 2 (a tie keeps the earlier epoch), best_loss.npz is
    epoch 3, and results.csv carries both the losses and the metric of every scored epoch."""
    args, selection, calls = scripted_run(tmp_path, ["--epochs", "3"], [0.6, 0.4, 0.3], [0.5, 0.3, 0.2], [0.80, 0.90, 0.90])
    assert calls == [args.run_dir / "last.npz"] * 3
    assert (selection.best_epoch, selection.best_loss_epoch) == (2, 3) and selection.best_loss == 0.2
    assert selection.best_metrics.recall == 0.90 and selection.best_path == args.run_dir / "best.npz"
    rows = results_of(args.run_dir)
    assert [row["epoch"] for row in rows] == ["1", "2", "3"] and list(rows[0]) == list(training.RESULTS_COLUMNS)
    assert [float(row["val_loss"]) for row in rows] == [0.5, 0.3, 0.2]
    assert [float(row["recall"]) for row in rows] == [0.80, 0.90, 0.90] and rows[1]["visible"] == "100"
    assert all(row["score_seconds"] != "" and row["seconds"] != "" for row in rows)
    # the files are the epochs they claim (the scripted epoch counts in the first bias)
    assert epoch_of(args.run_dir / "best.npz") == 2 and epoch_of(args.run_dir / "best_loss.npz") == 3
    assert epoch_of(args.run_dir / "last.npz") == 3


def epoch_of(weights: Path) -> int:
    return int(load_model(weights, torch.device("cpu")).convs[0].bias[0])


def test_score_every_leaves_unscored_epochs_blank_and_always_scores_the_last(tmp_path):
    args, selection, calls = scripted_run(tmp_path, ["--epochs", "3", "--score-every", "2"], [0.6, 0.4, 0.3],
                                          [0.5, 0.3, 0.2], [0.95, 0.90])
    assert len(calls) == 2 and selection.best_epoch == 2                 # epochs 2 and 3 scored; 2 preferred
    assert epoch_of(args.run_dir / "best.npz") == 2 and epoch_of(args.run_dir / "last.npz") == 3
    rows = results_of(args.run_dir)
    assert rows[0]["recall"] == "" and rows[0]["score_seconds"] == "" and rows[0]["val_loss"] != ""
    assert rows[1]["recall"] == "0.950000" and rows[2]["recall"] == "0.900000"
    assert training.main(["--score-every", "0", "--self-test"]) == 0        # the self-test needs no run
    assert training.main(["--score-every", "0"]) == 1


def test_pick_winner_reuses_the_best_epochs_metrics_and_records_how_it_chose(tmp_path, capsys):
    args, selection, _ = scripted_run(tmp_path, ["--epochs", "2", "--save", str(tmp_path / "save.npz")],
                                      [0.6, 0.4], [0.5, 0.3], [0.90, 0.85])
    clips = [("video1", Path("v1"), Path("c1"), "custom"), ("video10", Path("v10"), Path("c10"), "custom")]
    with patch.object(training, "score_weights", return_value=training.metrics_of(0.80, 0.0)) as baseline:
        training.pick_winner(args, clips, ["video10"], selection, torch.device("cpu"))
    assert baseline.call_args.args[:2] == (args.weights, clips[1:])      # the starting weights, on the held-out clips only
    winner = json.loads((args.run_dir / "winner.json").read_text(encoding="utf-8"))
    assert Path(winner["winner"]) == args.run_dir / "best.npz" and winner["selected_by"] == "detector_metric"
    assert winner["best_epoch"] == 1 and winner["best_loss_epoch"] == 2 and winner["validation_clips"] == ["video10"]
    assert winner["metrics"]["recall"] == 0.90 and winner["starting_weights"]["metrics"]["recall"] == 0.80
    assert epoch_of(tmp_path / "save.npz") == 1
    assert "promote it with" in capsys.readouterr().out

    with patch.object(training, "score_weights", return_value=training.metrics_of(0.95, 0.0)):
        training.pick_winner(args, clips, ["video10"], selection, torch.device("cpu"))
    winner = json.loads((args.run_dir / "winner.json").read_text(encoding="utf-8"))
    assert Path(winner["winner"]) == args.weights and winner["metrics"]["recall"] == 0.95
    assert (tmp_path / "save.npz").read_bytes() == args.weights.read_bytes()
    assert "stayed ahead" in capsys.readouterr().out
    args.from_scratch = True
    with patch.object(training, "score_weights") as never:
        training.pick_winner(args, clips, ["video10"], selection, torch.device("cpu"))
    assert not never.called and "starting_weights" not in json.loads((args.run_dir / "winner.json").read_text(encoding="utf-8"))


def test_the_seed_is_recorded_and_defaults_to_the_fixed_one():
    parser = training.build_parser()
    assert parser.parse_args([]).seed == training.SEED == 7
    args = parser.parse_args(["--seed", "11", "--score-every", "2"])
    config = training.run_config(args, [], [], {})
    assert config["seed"] == 11 and config["score_every"] == 2


def test_augment_is_a_set_of_names_and_the_hard_negative_weights_are_recorded(capsys):
    parser = training.build_parser()
    assert parser.parse_args([]).augment == ["basic"]
    args = parser.parse_args(["--augment", "translate", "basic", "reverse", "--oversample-absent", "3",
                              "--oversample-false-alarms", "2.5"])
    config = training.run_config(args, [], [], {}, flagged=42)
    assert config["augment"] == ["basic", "reverse", "translate"] and config["oversample_absent"] == 3.0
    assert config["oversample_false_alarms"] == 2.5 and config["false_alarm_units"] == 42
    assert training.run_config(parser.parse_args([]), [], [], {})["false_alarm_units"] == 0
    with pytest.raises(SystemExit):
        parser.parse_args(["--augment", "rotate"])
    assert "invalid choice" in capsys.readouterr().err


def test_metric_counts_localization_errors():
    stats = metric_probe(5, {i: [100, 100] if i == 0 else [120, 100] for i in range(5)})
    assert stats.visible == 5
    assert stats.recall == .2


def test_metric_infers_tail():
    stats = metric_probe(7, {i: [100, 100] for i in range(7)})
    assert stats.visible == 7
    assert stats.recall == 1.


def test_false_alarms_block_promotion():
    assert not promotion_improves(training.metrics_of(.91, 0., false_alarm=1.), training.metrics_of(.90, 0., false_alarm=0.))
