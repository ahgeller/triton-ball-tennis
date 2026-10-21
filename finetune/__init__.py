"""The labelling and GridTrackNet fine-tuning workspace.

Modules import each other as ``finetune.<module>``. The scripts among them (``ft.py``,
``label_tool.py``, ``pretrack.py``, ``realign.py``, ``import_data.py``,
``train_gridtracknet.py``, ``cnn_refiner.py``) can also be run as ``python finetune/<script>.py``
from the repository root: each one puts the repository root on ``sys.path`` only when it is the
program being run, never as a side effect of being imported.
"""
