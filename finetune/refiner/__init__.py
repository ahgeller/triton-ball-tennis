"""The label tool's suggestion model: ``refine`` measures where the ball is from the video itself
(the motion centroid, the candidates a trajectory chooses between, the trust rule), ``cnn_refiner``
is the small network that places those measurements where a person would click and the versions
of it, ``refiner_training`` trains a new version on the frames a person reviewed, ``refiner_sheet``
draws where two versions disagree, and ``hints`` is the command line over them (``ft.py hints``).
``finetune.labeling.label_hints`` is what reads the suggestions into the label tool.
"""
