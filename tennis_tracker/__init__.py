"""Production tennis tracking runtime.

The modules are imported by name (``tennis_tracker.pipeline.run``, ``tennis_tracker.config.Config``):
the pipeline imports OpenCV and TensorRT when it is imported, while ``Config``, ``paths`` and
the other light modules stay importable for CLI help and tests without initialising a detector.
"""
