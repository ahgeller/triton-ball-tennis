"""Run the training script's own self-test under pytest (``train_gridtracknet.py --self-test``)."""
from finetune import train_gridtracknet


def test_training_self_test_passes():
    train_gridtracknet.self_test()
