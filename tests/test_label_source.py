"""finetune.labeling.label_source: decoding keeps sequential frame identity (the rest of the label tool's
tests live in test_label_workflow.py)."""
from types import SimpleNamespace

import numpy as np

from finetune.labeling.label_source import Source
from tests._helpers import Capture, fake_source


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


# --------------------------------------------------------------------------- the prefetcher

def prefetching_tool(count=20, step=1, direction=1, walk=None, cursor=5):
    from finetune.labeling.label_tool import Tool

    clip = fake_source("a", count, 64, 36, origin=(10.0, 10.0), step=(0.0, 0.0), conf=0.9)
    tool = Tool([clip])
    decoded = []
    clip.frame = lambda index: decoded.append(index) or clip.frame_cache.setdefault(index, np.zeros((36, 64, 3), np.uint8))
    tool.session.cursor, tool.session.step, tool.session.direction = cursor, step, direction
    if walk is not None:
        tool.session.walk, tool.session.walk_pos = walk, walk.index(cursor)
    return tool, clip, decoded


def test_upcoming_rows_run_ahead_the_way_the_last_move_went_at_the_step():
    tool, _, _ = prefetching_tool()
    assert list(tool.prefetcher.upcoming_rows()) == [6, 7, 8, 9, 10, 11]
    tool.session.step = 4
    assert list(tool.prefetcher.upcoming_rows()) == [9, 13, 17]
    tool.session.direction, tool.session.step = -1, 2
    assert list(tool.prefetcher.upcoming_rows()) == [3, 1]
    tool, _, _ = prefetching_tool(walk=[1, 5, 9, 13, 17], cursor=5)
    assert list(tool.prefetcher.upcoming_rows()) == [9, 13, 17]
    tool.session.direction = -1
    assert list(tool.prefetcher.upcoming_rows()) == [1]
    tool.session.walk_pos = 0
    assert list(tool.prefetcher.upcoming_rows()) == []


def test_prefetch_decodes_what_is_not_cached_and_the_frame_before_for_the_motion_view():
    tool, clip, decoded = prefetching_tool()
    clip.frame_cache[7] = np.zeros((36, 64, 3), np.uint8)
    clip.missing.add(8)
    assert tool.prefetcher.run() and decoded[:2] == [6, 9]
    tool, clip, decoded = prefetching_tool()
    tool.session.filter = tool.session.rows and __import__("finetune.labeling.label_draw", fromlist=["FILTERS"]).FILTERS.index("motion")
    tool.prefetcher.warm_frame_before(6)
    assert decoded == [5]                                                    # the row before, ahead of the capture
    clip.position = 9
    tool.prefetcher.warm_frame_before(6)
    assert decoded == [5]                                                    # behind the capture: a seek, not worth it
    for frame in list(clip.frame_cache):
        clip.frame_cache.pop(frame)
    assert not tool.prefetcher.run() or decoded[1] >= 5
