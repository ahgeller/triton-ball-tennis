"""finetune/training/dataset.py on its own: the grid targets, the sampler and the augmentation,
on synthetic inputs. UnitDataset's reads and draws run on a cached unit in test_train_gridtracknet.py."""
import numpy as np
import pytest
import torch

from finetune.training.dataset import Augmenter, make_sampler, make_target, photometric_augment
from tennis_tracker.gridtracknet import FRAMES_PER_UNIT, GRID_COLS, GRID_ROWS

SIZE = (64, 36)


# --------------------------------------------------------------------------- units and targets

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
