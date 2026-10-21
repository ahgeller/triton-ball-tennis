"""finetune/training/frame_cache.py on its own: the five-frame units over a label file, the cache
record check and the frame extraction, on synthetic clips. prepare_clip end to end is in
test_train_gridtracknet.py."""
import json

import cv2
import pytest

from finetune.training.frame_cache import DATA_VERSION, build_units, cached_record, extract_frames
from tennis_tracker.gridtracknet import HEIGHT, WIDTH
from tests._helpers import write_clip

SIZE = (64, 36)


def test_units_advance_two_label_frames_and_need_every_frame_labelled():
    even = {index: [1.0, 1.0] for index in range(0, 20, 2)}
    assert build_units(even, 2)[:2] == [[0, 2, 4, 6, 8], [4, 6, 8, 10, 12]]
    odd = {index: None for index in range(1, 20, 2)}
    assert build_units(odd, 2)[:2] == [[1, 3, 5, 7, 9], [5, 7, 9, 11, 13]]
    gapped = {index: None for index in range(0, 12) if index != 5}
    assert all(5 not in window for window in build_units(gapped, 1))
    assert build_units({}, 1) == []




def test_cached_record_accepts_only_this_version_built_from_these_files(tmp_path, capsys):
    meta = tmp_path / "clips" / "c.json"
    signature = [["c_ball.csv", 1, 2], ["c.mp4", 3, 4]]
    assert cached_record(meta, signature) is None                              # nothing cached yet
    meta.parent.mkdir()
    meta.write_text(json.dumps({"version": DATA_VERSION, "signature": signature, "clip": "c"}), encoding="utf-8")
    assert cached_record(meta, signature)["clip"] == "c"
    assert cached_record(meta, [["c_ball.csv", 1, 9], ["c.mp4", 3, 4]]) is None   # the labels changed
    meta.write_text(json.dumps({"version": DATA_VERSION - 1, "signature": signature}), encoding="utf-8")
    assert cached_record(meta, signature) is None                              # an older record format
    meta.write_text("{not json", encoding="utf-8")
    assert cached_record(meta, signature) is None
    assert "cache entry unreadable" in capsys.readouterr().out


def test_extract_frames_writes_the_needed_frames_at_model_size_and_replaces_old_ones(tmp_path):
    video = write_clip(tmp_path / "c.avi", 8, SIZE)
    frame_dir = tmp_path / "frames" / "c"
    frame_dir.mkdir(parents=True)
    (frame_dir / "stale.jpg").write_bytes(b"x")
    extract_frames(video, frame_dir, {1, 6})
    assert sorted(p.name for p in frame_dir.iterdir()) == ["000001.jpg", "000006.jpg"]
    assert cv2.imread(str(frame_dir / "000006.jpg")).shape == (HEIGHT, WIDTH, 3)
    with pytest.raises(ValueError, match="ended before frames"):
        extract_frames(video, frame_dir, {2, 40})
