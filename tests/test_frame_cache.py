"""finetune/training/frame_cache.py on its own: the five-frame units over a label file, the cache
record check and the frame extraction, on synthetic clips. prepare_clip end to end is in
test_train_gridtracknet.py."""
import json
import os
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from finetune.data.labels import write_labels
from finetune.training import frame_cache
from finetune.training.frame_cache import (DATA_VERSION, build_units, cached_record, clip_signature, extract_frames,
                                           prepare_clip)
from tennis_tracker.gridtracknet import HEIGHT, WIDTH, resize_to_model
from tests._helpers import write_clip, write_frames

SIZE = (64, 36)


def test_units_advance_two_label_frames_and_need_every_frame_labelled():
    even = {index: [1.0, 1.0] for index in range(0, 20, 2)}
    assert build_units(even, 2)[:2] == [[0, 2, 4, 6, 8], [4, 6, 8, 10, 12]]
    odd = {index: None for index in range(1, 20, 2)}
    assert build_units(odd, 2)[:2] == [[1, 3, 5, 7, 9], [5, 7, 9, 11, 13]]
    gapped = {index: None for index in range(0, 12) if index != 5}
    assert all(5 not in window for window in build_units(gapped, 1))
    assert build_units({}, 1) == []




def test_the_signature_is_the_videos_alone(tmp_path):
    """The cached frames depend on the video and nothing else: a label edit changes which frames
    are needed, not what any frame looks like, so it must not invalidate them."""
    video = write_clip(tmp_path / "c.avi", 4, SIZE)
    signature = clip_signature(video)
    assert signature == ["c.avi", video.stat().st_size, video.stat().st_mtime_ns]
    csv_path = tmp_path / "c_ball.csv"
    write_labels(csv_path, {0: (1.0, 1.0)}, *SIZE)
    assert clip_signature(video) == signature
    stamp = video.stat().st_mtime_ns + 10 ** 9
    os.utime(video, ns=(stamp, stamp))
    assert clip_signature(video) != signature


def test_cached_record_accepts_only_this_version_built_from_this_video(tmp_path, capsys):
    meta = tmp_path / "clips" / "c.json"
    signature = ["c.mp4", 3, 4]
    assert cached_record(meta, signature) is None                              # nothing cached yet
    meta.parent.mkdir()
    meta.write_text(json.dumps({"version": DATA_VERSION, "signature": signature, "clip": "c"}), encoding="utf-8")
    assert cached_record(meta, signature)["clip"] == "c"
    assert cached_record(meta, ["c.mp4", 3, 9]) is None                        # the video changed
    meta.write_text(json.dumps({"version": DATA_VERSION - 1, "signature": signature}), encoding="utf-8")
    assert cached_record(meta, signature) is None                              # an older record format
    meta.write_text("{not json", encoding="utf-8")
    assert cached_record(meta, signature) is None
    assert "cache entry unreadable" in capsys.readouterr().out


def test_extract_frames_writes_the_needed_frames_at_model_size_beside_the_ones_there(tmp_path):
    video = write_clip(tmp_path / "c.avi", 8, SIZE)
    frame_dir = tmp_path / "frames" / "c"
    frame_dir.mkdir(parents=True)
    (frame_dir / "000003.png").write_bytes(b"x")
    extract_frames(video, frame_dir, {1, 6})
    assert sorted(p.name for p in frame_dir.iterdir()) == ["000001.png", "000003.png", "000006.png"]
    assert cv2.imread(str(frame_dir / "000006.png")).shape == (HEIGHT, WIDTH, 3)
    assert (frame_dir / "000003.png").read_bytes() == b"x"                     # what was there is left alone
    with pytest.raises(ValueError, match="ended before frames"):
        extract_frames(video, frame_dir, {2, 40})


def test_a_cached_frame_is_the_trackers_resize_of_the_decoded_frame_pixel_for_pixel(tmp_path):
    """Lossless: the PNG the trainer reads is exactly ``resize_to_model`` of the source frame, the
    picture the tracker's prepass feeds the network. (The JPEG cache this replaced was not, and it
    was resized with INTER_AREA while the tracker resizes with INTER_LINEAR.)"""
    rng = np.random.default_rng(5)
    frames = [rng.integers(0, 256, (90, 160, 3), dtype=np.uint8) for _ in range(3)]
    video = write_frames(tmp_path / "c.avi", frames)
    frame_dir = tmp_path / "frames" / "c"
    extract_frames(video, frame_dir, {2})
    cached = cv2.imread(str(frame_dir / "000002.png"), cv2.IMREAD_COLOR)
    capture = cv2.VideoCapture(str(video))
    decoded = [capture.read()[1] for _ in range(3)][2]
    capture.release()
    np.testing.assert_array_equal(decoded, frames[2])                           # FFV1 is lossless
    np.testing.assert_array_equal(cached, resize_to_model(decoded))


def test_a_label_edit_extracts_only_the_frames_it_newly_needs(tmp_path):
    video = write_clip(tmp_path / "c.avi", 14, SIZE, 30.0)
    csv_path = tmp_path / "c_ball.csv"
    write_labels(csv_path, {f: (1.0, 1.0) for f in range(7)}, *SIZE)
    data = tmp_path / "data"
    first = prepare_clip(("c", video, csv_path, "custom"), data, rebuild=False)
    frame_dir = data / "frames" / "c"
    assert first["frames"] == list(range(7)) and len(first["samples"]) == 2
    stamps = {path.name: path.stat().st_mtime_ns for path in frame_dir.iterdir()}
    assert sorted(stamps) == [f"{i:06d}.png" for i in range(7)]

    write_labels(csv_path, {f: (1.0, 1.0) for f in range(11)}, *SIZE)         # four more labelled frames
    with patch.object(frame_cache, "extract_frames", wraps=frame_cache.extract_frames) as extract:
        second = prepare_clip(("c", video, csv_path, "custom"), data, rebuild=False)
    assert extract.call_args.args[2] == {7, 8, 9, 10} and second["signature"] == first["signature"]
    assert second["frames"] == list(range(11)) and len(second["samples"]) == 4
    assert all(path.stat().st_mtime_ns == stamps[path.name] for path in frame_dir.iterdir() if path.name in stamps)
    assert json.loads((data / "clips" / "c.json").read_text(encoding="utf-8")) == second

    (frame_dir / "000004.png").unlink()                                         # a frame gone from disk is extracted again
    with patch.object(frame_cache, "extract_frames", wraps=frame_cache.extract_frames) as extract:
        prepare_clip(("c", video, csv_path, "custom"), data, rebuild=False)
    assert extract.call_args.args[2] == {4}
    with patch.object(frame_cache, "extract_frames", wraps=frame_cache.extract_frames) as extract:
        prepare_clip(("c", video, csv_path, "custom"), data, rebuild=False)    # nothing missing: no decode at all
    assert not extract.called


def test_frames_no_record_vouches_for_are_dropped_before_extracting(tmp_path):
    """A record that no longer matches (the video changed, the format version moved on, the
    JSON is damaged) says nothing about what is on disk: those frames go, the needed ones are
    extracted afresh, and a stray file of another format does not stay for good."""
    video = write_clip(tmp_path / "c.avi", 8, SIZE)
    csv_path = tmp_path / "c_ball.csv"
    write_labels(csv_path, {f: (1.0, 1.0) for f in range(5)}, *SIZE)
    data = tmp_path / "data"
    first = prepare_clip(("c", video, csv_path, "custom"), data, rebuild=False)
    frame_dir = data / "frames" / "c"
    (frame_dir / "000000.jpg").write_bytes(b"x")                              # the old cache's format
    stamp = video.stat().st_mtime_ns + 10 ** 9
    os.utime(video, ns=(stamp, stamp))                                        # the video changed
    with patch.object(frame_cache, "extract_frames", wraps=frame_cache.extract_frames) as extract:
        second = prepare_clip(("c", video, csv_path, "custom"), data, rebuild=False)
    assert extract.call_args.args[2] == set(range(5)) and second["signature"] != first["signature"]
    assert sorted(p.name for p in frame_dir.iterdir()) == [f"{i:06d}.png" for i in range(5)]
    (data / "clips" / "c.json").write_text("{not json", encoding="utf-8")   # a damaged record
    (frame_dir / "stray.png").write_bytes(b"x")
    prepare_clip(("c", video, csv_path, "custom"), data, rebuild=False)
    assert sorted(p.name for p in frame_dir.iterdir()) == [f"{i:06d}.png" for i in range(5)]


def test_a_frame_that_cannot_be_written_fails_the_clip_before_any_record_is_cached(tmp_path):
    """cv2.imwrite answers False instead of raising; left unchecked, prepare_clip cached a record
    pointing at PNGs that were never written, which only surfaced as a FileNotFoundError during
    training and again on every rerun."""
    video = write_clip(tmp_path / "c.avi", 8, SIZE)
    csv_path = tmp_path / "c_ball.csv"
    write_labels(csv_path, {f: (1.0, 1.0) for f in range(6)}, *SIZE)
    data = tmp_path / "data"
    with patch.object(frame_cache.cv2, "imwrite", return_value=False):
        with pytest.raises(OSError, match=r"could not write frame 0 of c.avi to .*000000.png"):
            prepare_clip(("c", video, csv_path, "custom"), data, rebuild=True)
    assert not (data / "clips").exists()
