"""finetune/training/loop.py on its own: the loss, a fresh network, an epoch of optimisation, the
validation loss and the weights written back in the tracker's .npz layout. Tiny synthetic units."""
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from finetune.training.loop import evaluate_loss, fresh_model, gridtracknet_loss, save_weights, train_epoch
from tennis_tracker.gridtracknet import FRAMES_PER_UNIT, GRID_COLS, GRID_ROWS, HEIGHT, WIDTH, load_model
from tests._helpers import blank_template


def one_unit_loader():
    images = torch.zeros(1, 3 * FRAMES_PER_UNIT, HEIGHT, WIDTH)
    target = torch.zeros(1, 3 * FRAMES_PER_UNIT, GRID_ROWS, GRID_COLS)
    target[0, 0, 13, 24] = 1.0
    return DataLoader(TensorDataset(images, target), batch_size=1)


def test_loss_is_finite_and_rewards_the_target_cell():
    target = torch.zeros(1, 3 * FRAMES_PER_UNIT, GRID_ROWS, GRID_COLS)
    target[0, 0, 13, 24] = 1.0
    flat = torch.full_like(target, 0.1)
    right = flat.clone()
    right[0, 0, 13, 24] = 0.9
    assert torch.isfinite(gridtracknet_loss(flat, target)) and gridtracknet_loss(right, target) < gridtracknet_loss(flat, target)
    assert torch.isfinite(gridtracknet_loss(torch.ones_like(target), target))     # clamped: no log(0)


@pytest.mark.filterwarnings("ignore:Converting a tensor with requires_grad")   # train_epoch reads the loss as a float
def test_a_fresh_model_trains_evaluates_and_round_trips_through_the_npz_layout(tmp_path):
    device = torch.device("cpu")
    model = fresh_model(device)
    assert model(torch.zeros(1, 3 * FRAMES_PER_UNIT, HEIGHT, WIDTH)).shape == (1, 3 * FRAMES_PER_UNIT, GRID_ROWS, GRID_COLS)
    loader = one_unit_loader()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    loss = train_epoch(model, loader, optimizer, scaler, device, epoch=1, epochs=1, max_steps=1)
    assert np.isfinite(loss) and np.isfinite(evaluate_loss(model, loader, device, max_steps=1))
    template = tmp_path / "template.npz"
    save_weights(model, blank_template(model, template), tmp_path / "out" / "weights.npz")
    reloaded = load_model(tmp_path / "out" / "weights.npz", device).float()
    with torch.inference_mode():
        probe = torch.rand(1, 3 * FRAMES_PER_UNIT, HEIGHT, WIDTH)
        assert torch.allclose(reloaded(probe), model.eval()(probe), atol=5e-3)   # the tracker loads in float16
