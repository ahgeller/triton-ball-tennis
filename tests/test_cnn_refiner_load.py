"""Checkpoints written by refiner_training.train still load under weights_only=True.

Both shapes the trainer has ever written are rebuilt in tmp_path with the exact
writer keys; the workspace models under finetune/models/refiner/ are never read.
"""
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from finetune.refiner import cnn_refiner  # noqa: E402


def _write_checkpoints(tmp_path: Path):
    path_a = tmp_path / "refiner-a.pt"
    path_b = tmp_path / "refiner-b.pt"
    torch.save({"state": cnn_refiner.build_network(True).state_dict(), "self_check": True, "threshold": 0.5,
                "trained": "x", "clips": 1, "frames": 1,
                "check": {"median_px": 0.5, "over5": 0.0, "sure": 0.5, "sure_over5": 0.0}},
               path_a)
    torch.save({"state": cnn_refiner.build_network(False).state_dict(), "self_check": False}, path_b)
    return path_a, path_b


def test_placer_loads_both_checkpoint_shapes(tmp_path):
    path_a, path_b = _write_checkpoints(tmp_path)
    placer_a = cnn_refiner.Placer(path_a)
    assert placer_a.threshold == 0.5
    assert placer_a.name == "refiner-a"
    placer_b = cnn_refiner.Placer(path_b)
    assert placer_b.threshold is None
    assert placer_b.name == "refiner-b"


def test_describe_reads_both_checkpoint_shapes(tmp_path, monkeypatch):
    path_a, path_b = _write_checkpoints(tmp_path)
    monkeypatch.setattr(cnn_refiner, "versions", lambda: [path_a, path_b])
    monkeypatch.setattr(cnn_refiner, "current", lambda: path_a)
    lines = cnn_refiner.version_lines()
    assert len(lines) == 2
    assert lines[0].startswith("*")
    assert lines[1].startswith(" ")
    assert "unreadable" not in lines[0]
    assert "unreadable" not in lines[1]
