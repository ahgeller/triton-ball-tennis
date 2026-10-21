"""finetune.labeling.label_source: decoding keeps sequential frame identity (the rest of the label tool's
tests live in test_label_workflow.py)."""
from types import SimpleNamespace

import numpy as np

from finetune.labeling.label_source import Source
from tests._helpers import Capture


def test_duplicate_video_frames_keep_sequential_identity():
    class DuplicateCapture(Capture):
        def read(self):
            ok, frame = super().read()
            return ok, np.full_like(frame, 42) if ok else None
    source = SimpleNamespace(position=0, capture=DuplicateCapture(3), sig_of={}, end=None,
                             frame_cache={}, label_frames=set())
    for index in range(3):
        assert Source.decode_at(source, index) is not None
    assert source.position == 3
