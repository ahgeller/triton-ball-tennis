"""The network that places the label tool's suggestions (``cnn_refiner`` builds it, loads its
versions and runs it; ``refiner_training`` trains it).

It lives in its own module because its class derives from ``torch.nn.Module``: ``cnn_refiner``
imports this module only when a network is built, so the label tool and ``hints.py list`` load
without torch. A saved version holds only the ``state_dict`` (``body``, ``heat``, ``sure``), so
the class may move without old versions failing to load."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn


class RefinerNet(nn.Module):
    """Fully convolutional: a heatmap over the window, read out as its soft-argmax (sub-pixel),
    and a small head on what it looked at that predicts whether that answer is within
    ``cnn_refiner.SURE_PX``. self_check=False is the first design (versions before the
    self-check), kept so they still load."""

    def __init__(self, self_check: bool = True, width: int = 32):
        super().__init__()
        layers, inp = [], 9
        for dilation in (1, 1, 2, 4, 8, 1):
            layers += [nn.Conv2d(inp, width, 3, padding=dilation, dilation=dilation),
                       nn.BatchNorm2d(width), nn.ReLU()]
            inp = width
        if not self_check:
            layers += [nn.Conv2d(width, 1, 1)]
        self.body = nn.Sequential(*layers)
        self.heat = nn.Conv2d(width, 1, 1) if self_check else None
        self.sure = nn.Sequential(nn.Linear(width + 3, 64), nn.ReLU(), nn.Linear(64, 1)) if self_check else None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        n, _, h, w = x.shape
        feat = self.body(x)
        logits = (feat if self.heat is None else self.heat(feat)).flatten(1).float()
        prob = logits.softmax(1)
        p2 = prob.view(n, h, w)
        gx = torch.arange(w, device=x.device, dtype=prob.dtype)
        gy = torch.arange(h, device=x.device, dtype=prob.dtype)
        px, py = (p2.sum(1) * gx).sum(1), (p2.sum(2) * gy).sum(1)
        xy = torch.stack([px, py], 1)
        if self.sure is None:
            return xy, None
        spread = ((p2.sum(1) * (gx - px[:, None]) ** 2).sum(1)
                  + (p2.sum(2) * (gy - py[:, None]) ** 2).sum(1)).sqrt()
        pooled = (feat.float().flatten(2) * prob[:, None]).sum(2)          # features where it looks
        stats = torch.stack([prob.max(1).values * h * w / 100.0, spread / 10.0, logits.std(1) / 10.0], 1)
        return xy, self.sure(torch.cat([pooled, stats], 1)).squeeze(1)
