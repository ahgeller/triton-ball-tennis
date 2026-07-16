from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn


WIDTH = 768
HEIGHT = 432
FRAMES_PER_UNIT = 5
GRID_COLS = 48
GRID_ROWS = 27
POOL_AFTER = {1, 3, 5, 8}


class WidthBatchNorm(nn.Module):
    """Reproduce GridTrackNet's BatchNormalization(axis=-1) in NCHW."""

    def __init__(self, width: int):
        super().__init__()
        self.register_buffer("gamma", torch.ones(width))
        self.register_buffer("beta", torch.zeros(width))
        self.register_buffer("mean", torch.zeros(width))
        self.register_buffer("variance", torch.ones(width))

    def forward(self, value):
        shape = (1, 1, 1, -1)
        return (
            (value - self.mean.view(shape))
            * torch.rsqrt(self.variance.view(shape) + 0.001)
            * self.gamma.view(shape)
            + self.beta.view(shape)
        )


class GridTrackNet(nn.Module):
    def __init__(self):
        super().__init__()
        channels = [15, 64, 64, 128, 128, 256, 256, 256, 256, 256, 512, 512, 512, 15]
        widths = [768, 768, 384, 384, 192, 192, 96, 96, 96, 48, 48, 48]
        self.convs = nn.ModuleList(
            nn.Conv2d(channels[index], channels[index + 1], 3, padding=1)
            for index in range(len(channels) - 1)
        )
        self.batch_norms = nn.ModuleList(WidthBatchNorm(width) for width in widths)

    def forward(self, value):
        for index, conv in enumerate(self.convs):
            value = conv(value)
            if index == len(self.convs) - 1:
                return torch.sigmoid(value)
            value = self.batch_norms[index](torch.relu(value))
            if index in POOL_AFTER:
                value = torch.nn.functional.max_pool2d(value, 2)
        raise AssertionError("unreachable")


def load_model(weights: Path, device: torch.device) -> GridTrackNet:
    model = GridTrackNet()
    with np.load(weights) as source, torch.no_grad():
        for index, conv in enumerate(model.convs):
            name = "conv2d" if index == 0 else f"conv2d_{index}"
            prefix = f"{name}/{name}"
            conv.weight.copy_(torch.from_numpy(source[f"{prefix}/kernel:0"].transpose(3, 2, 0, 1)))
            conv.bias.copy_(torch.from_numpy(source[f"{prefix}/bias:0"]))
        for index, layer in enumerate(model.batch_norms):
            name = "batch_normalization" if index == 0 else f"batch_normalization_{index}"
            prefix = f"{name}/{name}"
            layer.gamma.copy_(torch.from_numpy(source[f"{prefix}/gamma:0"]))
            layer.beta.copy_(torch.from_numpy(source[f"{prefix}/beta:0"]))
            layer.mean.copy_(torch.from_numpy(source[f"{prefix}/moving_mean:0"]))
            layer.variance.copy_(torch.from_numpy(source[f"{prefix}/moving_variance:0"]))
    return model.eval().to(device).half()


def frame_tensor(frame, device: torch.device):
    rgb = cv2.cvtColor(cv2.resize(frame, (WIDTH, HEIGHT)), cv2.COLOR_BGR2RGB)
    chw = np.moveaxis(rgb, -1, 0).astype(np.float32) / 255.0
    return torch.from_numpy(chw).to(device, dtype=torch.float16)


def decode_predictions(output, output_width: int, output_height: int, threshold: float):
    units = output.float().cpu().numpy().reshape(-1, FRAMES_PER_UNIT, 3, GRID_ROWS, GRID_COLS)
    decoded = []
    for unit in units:
        for confidence_grid, x_grid, y_grid in unit:
            row, col = np.unravel_index(np.argmax(confidence_grid), confidence_grid.shape)
            confidence = float(confidence_grid[row, col])
            point = None
            if confidence >= threshold:
                x = (float(x_grid[row, col]) + col) * output_width / GRID_COLS
                y = (float(y_grid[row, col]) + row) * output_height / GRID_ROWS
                point = (x, y)
            decoded.append((point, confidence))
    return decoded


def self_test() -> None:
    output = torch.zeros((1, 15, GRID_ROWS, GRID_COLS))
    output[0, 0, 4, 6] = 0.9
    output[0, 1, 4, 6] = 0.5
    output[0, 2, 4, 6] = 0.25
    assert decode_predictions(output, 1920, 1080, 0.5)[0][0] == (260.0, 170.0)
