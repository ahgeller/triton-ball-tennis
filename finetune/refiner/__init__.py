"""The label tool's suggestion model: ``refine`` measures where the ball is from the video itself
(the candidates a trajectory chooses between, the trust rule; the motion centroid and the blobs on
one frame are ``motion_blobs``), ``cnn_refiner`` is the small network that places those
measurements where a person would click and the versions of it (the network's torch class is
``refiner_network``, imported only when one is built), ``refiner_training`` trains a new version
on the frames a person reviewed (from the crops ``refiner_crops`` gathers, fitting each network with
``refiner_fitting``), ``refiner_sheet`` draws where two versions disagree, and ``hints`` is the
command line over them (``ft.py hints``). ``finetune.labeling.label_hints`` is what reads the
suggestions into the label tool.
"""
