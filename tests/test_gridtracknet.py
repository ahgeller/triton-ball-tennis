"""The GridTrackNet input contract and NPZ weight mapping, on CPU with synthetic data."""
import numpy as np
import pytest
import torch

import cv2

from tennis_tracker import gridtracknet
from tennis_tracker.gridtracknet import (
    BN_EPSILON,
    GRID_COLS,
    GRID_ROWS,
    HEIGHT,
    RESIZE_INTERPOLATION,
    WIDTH,
    GridTrackNet,
    WidthBatchNorm,
    decode_predictions,
    frame_tensor,
    load_model,
    resize_to_model,
)

CPU = torch.device("cpu")


def test_frame_tensor_is_rgb_chw_in_unit_range():
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame[..., 0] = 255  # OpenCV frames are BGR: this is pure blue
    frame[..., 1] = 51   # green at 0.2
    tensor = frame_tensor(frame, CPU)
    assert tensor.shape == (3, HEIGHT, WIDTH)
    assert tensor.dtype == torch.float16
    channel_means = tensor.float().mean(dim=(1, 2)).tolist()
    assert channel_means == pytest.approx([0.0, 0.2, 1.0], abs=1e-3)  # R, G, B


def test_frame_tensor_resizes_any_frame_to_model_size():
    frame = np.full((1080, 1920, 3), 128, dtype=np.uint8)
    tensor = frame_tensor(frame, CPU)
    assert tensor.shape == (3, HEIGHT, WIDTH)
    assert tensor.float().min().item() == pytest.approx(128 / 255, abs=1e-3)
    assert tensor.float().max().item() == pytest.approx(128 / 255, abs=1e-3)


def test_frame_tensor_is_resize_to_model_in_rgb_over_255():
    """The tracker's input is the one resize rule (``resize_to_model``, INTER_LINEAR) and nothing
    else: the training cache stores that very picture, so what the network trains on is what it
    is run on."""
    assert RESIZE_INTERPOLATION == cv2.INTER_LINEAR
    frame = np.random.default_rng(0).integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
    resized = resize_to_model(frame)
    assert resized.shape == (HEIGHT, WIDTH, 3) and resized.dtype == np.uint8
    np.testing.assert_array_equal(resized, cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR))
    expected = np.moveaxis(resized[..., ::-1], -1, 0).astype(np.float32) / 255.0
    np.testing.assert_allclose(frame_tensor(frame, CPU).float().numpy(), expected, atol=1 / 2048)   # fp16 rounding
    model_sized = np.random.default_rng(1).integers(0, 256, (HEIGHT, WIDTH, 3), dtype=np.uint8)
    np.testing.assert_array_equal(resize_to_model(model_sized), model_sized)


def _synthetic_npz(path):
    """An NPZ with the upstream layer names, HWIO kernels and values that
    identify the kernel position, input channel and output channel."""
    model = GridTrackNet()
    arrays = {}
    for index, conv in enumerate(model.convs):
        name = "conv2d" if index == 0 else f"conv2d_{index}"
        out_ch, in_ch, kh, kw = conv.weight.shape
        ys, xs, cin, cout = np.meshgrid(np.arange(kh), np.arange(kw), np.arange(in_ch), np.arange(out_ch), indexing="ij")
        kernel = (ys * kw + xs) / 16 + (cin % 8) / 128 + (cout % 4) / 1024  # exact in fp16
        arrays[f"{name}/{name}/kernel:0"] = kernel.astype(np.float32)
        arrays[f"{name}/{name}/bias:0"] = np.full((out_ch,), index + 0.5, dtype=np.float32)
    for index, layer in enumerate(model.batch_norms):
        name = "batch_normalization" if index == 0 else f"batch_normalization_{index}"
        width = layer.gamma.shape[0]
        base = (np.arange(width) % 16).astype(np.float32) / 16  # exact in fp16
        arrays[f"{name}/{name}/gamma:0"] = base + 1.0
        arrays[f"{name}/{name}/beta:0"] = base + 2.0
        arrays[f"{name}/{name}/moving_mean:0"] = base + 3.0
        arrays[f"{name}/{name}/moving_variance:0"] = base + 4.0
    np.savez(path, **arrays)
    return arrays


def test_load_model_maps_npz_layers_and_kernel_orientation(tmp_path):
    weights = tmp_path / "weights.npz"
    arrays = _synthetic_npz(weights)

    model = load_model(weights, CPU)

    assert not model.training
    for index, conv in enumerate(model.convs):
        name = "conv2d" if index == 0 else f"conv2d_{index}"
        assert conv.weight.dtype == torch.float16
        expected = arrays[f"{name}/{name}/kernel:0"].transpose(3, 2, 0, 1)  # HWIO -> OIHW
        np.testing.assert_array_equal(conv.weight.detach().float().numpy(), expected)
        np.testing.assert_array_equal(conv.bias.detach().float().numpy(), arrays[f"{name}/{name}/bias:0"])
    for index, layer in enumerate(model.batch_norms):
        name = "batch_normalization" if index == 0 else f"batch_normalization_{index}"
        np.testing.assert_array_equal(layer.gamma.float().numpy(), arrays[f"{name}/{name}/gamma:0"])
        np.testing.assert_array_equal(layer.beta.float().numpy(), arrays[f"{name}/{name}/beta:0"])
        np.testing.assert_array_equal(layer.mean.float().numpy(), arrays[f"{name}/{name}/moving_mean:0"])
        np.testing.assert_array_equal(layer.variance.float().numpy(), arrays[f"{name}/{name}/moving_variance:0"])


def test_load_model_rejects_npz_missing_a_layer(tmp_path):
    weights = tmp_path / "weights.npz"
    arrays = _synthetic_npz(weights)
    del arrays["conv2d_3/conv2d_3/bias:0"]
    np.savez(weights, **arrays)
    with pytest.raises(KeyError):
        load_model(weights, CPU)


def test_width_batchnorm_normalizes_each_column_separately():
    layer = WidthBatchNorm(4)
    layer.mean.copy_(torch.tensor([0.0, 1.0, 2.0, 3.0]))
    layer.gamma.fill_(2.0)
    layer.beta.fill_(1.0)
    value = torch.ones((1, 1, 2, 4))
    out = layer(value)
    expected = (1.0 - torch.tensor([0.0, 1.0, 2.0, 3.0])) / (1.0 + BN_EPSILON) ** 0.5 * 2.0 + 1.0
    assert out.shape == value.shape
    torch.testing.assert_close(out[0, 0, 0], expected)
    torch.testing.assert_close(out[0, 0, 1], expected)


def test_decode_predictions_places_cell_offsets_in_output_pixels():
    output = torch.zeros((1, 15, GRID_ROWS, GRID_COLS))
    output[0, 0, 4, 6] = 0.9   # frame 0 confidence at row 4, col 6
    output[0, 1, 4, 6] = 0.5   # x offset within the cell
    output[0, 2, 4, 6] = 0.25  # y offset within the cell
    decoded = decode_predictions(output, 1920, 1080, 0.5)
    assert len(decoded) == 5
    assert decoded[0] == ((260.0, 170.0), pytest.approx(0.9))
    assert decoded[1] == (None, 0.0)


def test_decode_predictions_applies_threshold():
    output = torch.zeros((1, 15, GRID_ROWS, GRID_COLS))
    output[0, 0, 4, 6] = 0.4
    point, confidence = decode_predictions(output, 1920, 1080, 0.5)[0]
    assert point is None
    assert confidence == pytest.approx(0.4)


def test_self_test_passes():
    gridtracknet.self_test()
