"""Training and the detector's own view of a clip: ``train_gridtracknet`` (the trainer's run, run as
a script or ``python -m finetune.training.train_gridtracknet``), ``loop`` (the loss, an epoch, the
validation loss and saving weights), ``metrics`` (the detector metric a run is judged by and the
winner rule), ``clips`` (which clips it trains
on and holds out), ``frame_cache`` (the per-clip cache of frames and training units), ``dataset``
(targets, augmentation, the Dataset and the sampler), ``detect`` (the raw GridTrackNet prepass over
one clip) and ``pretrack`` (drafts for the label tool from that prepass).
"""
