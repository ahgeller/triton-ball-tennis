"""The trainer's torch side of the frame cache: the grid targets, the augmentation, the
``Dataset`` that reads cached units and the sampler that weights them by source."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Dict, Optional, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from tennis_tracker.gridtracknet import FRAMES_PER_UNIT, GRID_COLS, GRID_ROWS
from finetune.training.frame_cache import TrainingUnit


def make_target(points, width: int, height: int) -> torch.Tensor:
    target = np.zeros((FRAMES_PER_UNIT * 3, GRID_ROWS, GRID_COLS), dtype=np.float32)
    for index, point in enumerate(points):
        if point is None:
            continue
        grid_x = point[0] * GRID_COLS / width
        grid_y = point[1] * GRID_ROWS / height
        col = min(GRID_COLS - 1, int(grid_x))
        row = min(GRID_ROWS - 1, int(grid_y))
        target[index * 3, row, col] = 1
        target[index * 3 + 1, row, col] = grid_x - col
        target[index * 3 + 2, row, col] = grid_y - row
    return torch.from_numpy(target)


class Augmenter:
    """The random draws one data-loading process makes, seeded from torch's seed for that process.

    In a DataLoader worker ``torch.initial_seed()`` is ``base_seed + worker_id``, the very seed torch
    gives that worker's global ``random`` module, and in the main process it is the seed
    ``train`` set; so ``draws`` reproduces the sequence the module-level ``random.*`` calls used to
    make, in both cases. The numpy generator (sensor noise) was never seeded before, so it starts
    from the same seed now and is reproducible for the first time.
    """

    def __init__(self, seed: int) -> None:
        self.draws = random.Random(seed)
        self.noise = np.random.default_rng(seed)


def photometric_augment(images: np.ndarray, rng: Augmenter) -> np.ndarray:
    """Strong lighting/colour augmentation applied identically to all five frames of a window
    (so motion cues survive): gamma (dark night courts and washed-out daylight), contrast, brightness,
    per-channel colour cast, sensor noise and occasional softness. images: (15, H, W) float in [0, 1]."""
    draws = rng.draws
    gamma = math.exp(draws.uniform(math.log(0.45), math.log(2.2)))       # >1 darkens: night-court look
    contrast = draws.uniform(0.6, 1.4)
    brightness = draws.uniform(-0.12, 0.12)
    cast = np.array([draws.uniform(0.85, 1.15) for _ in range(3)], dtype=np.float32)
    out = np.power(np.clip(images, 0.0, 1.0), gamma)
    out = (out - 0.5) * contrast + 0.5 + brightness
    out = out.reshape(FRAMES_PER_UNIT, 3, *out.shape[1:]) * cast[None, :, None, None]
    out = out.reshape(-1, *out.shape[2:])
    if draws.random() < 0.5:
        out = out + rng.noise.normal(0.0, draws.uniform(0.005, 0.03), size=out.shape).astype(np.float32)
    if draws.random() < 0.2:
        sigma = draws.uniform(0.4, 1.0)
        out = np.stack([cv2.GaussianBlur(plane, (0, 0), sigma) for plane in out])
    return np.clip(out, 0.0, 1.0).astype(np.float32)


class UnitDataset(Dataset):
    def __init__(self, data_dir: Path, samples: Sequence[TrainingUnit], augment: bool, strength: str = "basic") -> None:
        self.data_dir, self.samples, self.augment, self.strength = data_dir, samples, augment, strength
        self.rng: Optional[Augmenter] = None     # made in the process that reads, see Augmenter

    def __len__(self) -> int:
        return len(self.samples)

    def augmenter(self) -> Augmenter:
        if self.rng is None:
            self.rng = Augmenter(torch.initial_seed())
        return self.rng

    def read_frames(self, sample: TrainingUnit, flip: bool) -> np.ndarray:
        """The unit's five cached frames as one (15, H, W) float array in [0, 1], RGB, mirrored when `flip`."""
        frames = []
        for relative in sample["frames"]:
            image = cv2.imread(str(self.data_dir / relative), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(self.data_dir / relative)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if flip:
                image = cv2.flip(image, 1)
            frames.append(np.moveaxis(image, -1, 0))
        return np.concatenate(frames).astype(np.float32) / 255.0

    def __getitem__(self, index: int):
        sample = self.samples[index]
        width, height = sample["size"]
        rng = self.augmenter()
        # The draws are made in this order on purpose: it is the sequence every run has made.
        flip = self.augment and rng.draws.random() < 0.5
        jitter = self.augment and self.strength == "basic" and rng.draws.random() < 0.5
        gain = rng.draws.uniform(0.85, 1.15) if jitter else 1.0
        images = self.read_frames(sample, flip)
        points = sample["points"]
        if flip:
            points = [None if point is None else [width - 1 - point[0], point[1]] for point in points]
        if jitter:
            images = np.clip(images * gain, 0.0, 1.0)
        elif self.augment and self.strength == "strong" and rng.draws.random() < 0.8:
            images = photometric_augment(images, rng)
        return torch.from_numpy(np.ascontiguousarray(images)), make_target(points, width, height)


def make_sampler(samples: Sequence[TrainingUnit], oversample: Dict[str, float],
                 epoch_units: Optional[int]) -> Optional[WeightedRandomSampler]:
    """Weighted sampling by source (weight 0 drops a source from training); None = plain shuffle."""
    weights = [float(oversample.get(sample["source"], 1.0)) for sample in samples]
    if epoch_units is None and all(weight == 1.0 for weight in weights):
        return None
    drawable = sum(1 for weight in weights if weight > 0)
    if drawable == 0:
        raise ValueError("Every training unit has sampling weight 0")
    return WeightedRandomSampler(weights, num_samples=epoch_units or drawable, replacement=True)
