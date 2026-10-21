"""Fitting one suggestion network (``cnn_refiner.build_network``) to crops on the device, and
scoring it: the random training windows (``sample_batch``), the training loop (``train_network``)
and the per-frame errors and confidences on held-out crops (``fold_errors``). The run that
calls them, its folds and its self-check are in ``refiner_training``."""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Callable, Sequence, Tuple

import numpy as np

from finetune.refiner.cnn_refiner import BACKWARDS, CROP, SURE_PX, WINDOW, build_network

if TYPE_CHECKING:
    import torch
    from torch import nn

SCORE_BATCH = 1024         # windows scored per network call


def to_device(parts: Sequence[np.ndarray], device: str) -> "torch.Tensor":
    import torch
    out = torch.empty((sum(len(p) for p in parts), CROP, CROP, 9), dtype=torch.uint8, device=device)
    i = 0
    for p in parts:
        out[i:i + len(p)] = torch.from_numpy(p).to(device)
        i += len(p)
    return out


def sample_batch(data, device: str, rng: np.random.Generator, size: int, flip: bool = True,
                 in_order: bool = False) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """One batch of `size` random windows of random crops on the GPU: (inputs, targets in window
    pixels). `in_order` takes the first `size` crops in order instead of drawing them at random
    (scoring)."""
    import torch
    count = data.shape[0]
    span, mid = CROP - WINDOW, (CROP - 1) / 2.0
    ar = torch.arange(WINDOW, device=device)
    idx = torch.arange(size, device=device) if in_order else torch.from_numpy(rng.integers(0, count, size)).to(device)
    ox = torch.from_numpy(rng.integers(0, span + 1, size)).to(device)
    oy = torch.from_numpy(rng.integers(0, span + 1, size)).to(device)
    rows, cols = (oy[:, None] + ar)[:, :, None], (ox[:, None] + ar)[:, None, :]
    win = data[idx[:, None, None], rows, cols]                       # size x 64 x 64 x 9
    tx, ty = mid - ox.float(), mid - oy.float()
    if flip:
        mirror = torch.from_numpy(rng.random(size) < 0.5).to(device)
        win = torch.where(mirror[:, None, None, None], win.flip(2), win)
        tx = torch.where(mirror, WINDOW - 1 - tx, tx)
        back = torch.from_numpy(rng.random(size) < 0.5).to(device)   # the three frames played backwards
        win = torch.where(back[:, None, None, None], win[..., BACKWARDS], win)
    x = win.permute(0, 3, 1, 2).float().div_(255.0).sub_(0.5)
    if flip:
        x = x * torch.empty(size, 1, 1, 1, device=device).uniform_(0.7, 1.3)
    return x.contiguous(memory_format=torch.channels_last), torch.stack([tx, ty], 1)


def train_network(data, epochs: int, device: str, rng: np.random.Generator, log: Callable[..., None],
        label: str) -> "nn.Module":
    """One network trained on `data` (crops already on the device)."""
    import torch
    net = build_network().to(device).to(memory_format=torch.channels_last)
    batch = 256
    per_epoch = max(1, len(data) // batch)
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=epochs * per_epoch)
    bce = torch.nn.BCEWithLogitsLoss()
    start = time.time()
    for epoch in range(epochs):
        net.train()
        total = 0.0
        for _ in range(per_epoch):
            x, target = sample_batch(data, device, rng, batch)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                xy, logit = net(x)
            hit = ((xy.detach() - target).norm(dim=1) <= SURE_PX).float()
            loss = torch.nn.functional.smooth_l1_loss(xy, target, beta=0.5) + bce(logit.float(), hit)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
        if epoch % 10 == 9 or epoch == epochs - 1:
            log(f"  {label}: epoch {epoch + 1}/{epochs}  loss {total / per_epoch:.3f}  ({time.time() - start:.0f}s)")
    return net.eval()


def fold_errors(net, data, device: str, rng: np.random.Generator, repeats: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Errors (window px = 1080p px) and confidences of every frame, `repeats` times each from a
    random start up to 16 px off."""
    import torch
    errors, confs = [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        for _ in range(repeats):
            for i in range(0, len(data), SCORE_BATCH):
                part = data[i:i + SCORE_BATCH]
                x, target = sample_batch(part, device, rng, len(part), flip=False, in_order=True)
                xy, logit = net(x)
                errors.append((xy.float() - target).norm(dim=1).cpu().numpy())
                confs.append(torch.sigmoid(logit.float()).cpu().numpy())
    return np.concatenate(errors), np.concatenate(confs)
