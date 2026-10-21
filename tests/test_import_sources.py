"""finetune/data/import_sources.py: the TrackNet CSV reader's column and row rules, the frame and
pixel mapping, media copies, frame encodes and the manifest, on synthetic files under tmp_path."""
import re
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest

from finetune.data import import_sources
from finetune.data.import_sources import (Manifest, cadence_frame, canvas_scale, convert_rows, copy_media, frames_to_video,
                                          grid_frame, read_tracknet_csv, tracknet_columns, tracknet_row)
from finetune.data.workspace import MANIFEST_COLUMNS, read_manifest


def test_columns_accept_each_spelling_and_refuse_a_foreign_header(tmp_path):
    assert tracknet_columns(tmp_path / "a.csv", ["Frame", "Visibility", "X", "Y", "status"]) == ("Frame", "Visibility", "X", "Y")
    assert tracknet_columns(tmp_path / "a.csv", ["file name", "visibility", "x-coordinate", "y-coordinate"]) == \
        ("file name", "visibility", "x-coordinate", "y-coordinate")
    with pytest.raises(ValueError, match="unrecognised TrackNet label columns"):
        tracknet_columns(tmp_path / "a.csv", ["frame", "ball_x", "ball_y"])


def test_rows_map_visibility_and_drop_rows_without_a_frame_number():
    columns = ("Frame", "Visibility", "X", "Y")
    assert tracknet_row({"Frame": "0007.jpg", "Visibility": "1", "X": "10", "Y": "20"}, columns, False) == (7, (1, 10.0, 20.0))
    assert tracknet_row({"Frame": "7", "Visibility": "3", "X": "10", "Y": "20"}, columns, False) == (7, (0, 10.0, 20.0))
    assert tracknet_row({"Frame": "7", "Visibility": "3", "X": "10", "Y": "20"}, columns, True) == (7, (3, 10.0, 20.0))
    assert tracknet_row({"Frame": "7", "Visibility": "1", "X": "0", "Y": "0"}, columns, False) == (7, (0, 0.0, 0.0))
    assert tracknet_row({"Frame": "", "Visibility": "1", "X": "0", "Y": "0"}, columns, False) is None
    assert tracknet_row({"Frame": "2", "Visibility": "", "X": "", "Y": ""}, columns, False) == (2, (0, 0.0, 0.0))


def test_read_tracknet_csv_keeps_the_last_row_per_frame(tmp_path):
    path = tmp_path / "Label.csv"
    path.write_text("﻿file name,visibility,x-coordinate,y-coordinate\n"
                    "0001.jpg,1,5,6\nnoise,1,1,1\n0001.jpg,2,7,8\n0003.jpg,0,0,0\n", encoding="utf-8")
    assert read_tracknet_csv(path, keep_occluded=False) == {1: (2, 7.0, 8.0), 3: (0, 0.0, 0.0)}


def test_frame_and_pixel_mapping():
    assert [grid_frame(k, 2) for k in range(3)] == [1, 3, 5] and grid_frame(4, 1) == 4
    assert [cadence_frame(k, 2) for k in range(3)] == [0, 2, 4]
    assert canvas_scale((1280, 720), 1920, 1080) == (1.5, 1.5) and canvas_scale(None, 640, 360) == (1.0, 1.0)
    labels, dropped = convert_rows({0: (1, 10.0, 20.0), 1: (0, 0.0, 0.0), 2: (1, 30.0, 40.0)},
                                   lambda k: 2 * k + 1, (2.0, 0.5), total=4)
    assert labels == {1: (20.0, 10.0), 3: None} and dropped == 1


def test_copy_media_keeps_links_or_copies(tmp_path):
    src = tmp_path / "src.mp4"
    src.write_bytes(b"video")
    dst = tmp_path / "out" / "clip.mp4"
    assert copy_media(src, dst, link=False) == "copied" and dst.read_bytes() == b"video"
    assert copy_media(src, dst, link=False) == "kept"
    dst.write_bytes(b"stale and longer")
    with patch.object(import_sources.os, "link", side_effect=OSError("cross-device")):
        assert copy_media(src, dst, link=True) == "copied"            # a link that cannot be made is a copy
    assert dst.read_bytes() == b"video"
    dst.unlink()
    assert copy_media(src, dst, link=True) in ("linked", "copied") and dst.read_bytes() == b"video"


def test_frames_to_video_falls_back_to_opencv_without_ffmpeg(tmp_path):
    frames = []
    for index in range(3):
        frame = tmp_path / f"{index:03d}.png"
        cv2.imwrite(str(frame), np.full((36, 64, 3), 40 * index, np.uint8))
        frames.append(frame)
    out = tmp_path / "out" / "clip.mp4"
    with patch.object(import_sources.shutil, "which", return_value=None):
        encoder = frames_to_video(frames, out, 30.0)
    assert encoder.startswith("cv2") and out.is_file()
    capture = cv2.VideoCapture(str(out))
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
    capture.release()


def test_the_opencv_encode_refuses_a_frame_it_cannot_read_or_that_is_the_wrong_size(tmp_path):
    """OpenCV writes nothing for an unreadable or odd-sized frame and says nothing; the labels
    are then mapped onto a video one frame short, so each such frame is an error naming it."""
    good = tmp_path / "000.png"
    cv2.imwrite(str(good), np.full((36, 64, 3), 10, np.uint8))
    broken = tmp_path / "001.png"
    broken.write_bytes(b"not an image")
    odd = tmp_path / "002.png"
    cv2.imwrite(str(odd), np.zeros((18, 64, 3), np.uint8))
    out = tmp_path / "clip.mp4"
    with patch.object(import_sources.shutil, "which", return_value=None):
        with pytest.raises(RuntimeError, match=re.escape(f"could not read {broken}")):
            frames_to_video([good, broken], out, 30.0)
        with pytest.raises(RuntimeError, match=re.escape(f"{odd} is 64x18, not 64x36 like 000.png")):
            frames_to_video([good, odd], out, 30.0)
        with pytest.raises(RuntimeError, match=re.escape(f"could not read {broken}")):
            frames_to_video([broken, good], out, 30.0)                          # the first frame sizes the video
        unopenable = Mock()
        unopenable.isOpened.return_value = False
        with patch.object(import_sources.cv2, "VideoWriter", return_value=unopenable), \
                pytest.raises(RuntimeError, match="could not open .*clip.mp4 for writing"):
            frames_to_video([good], out, 30.0)
        unopenable.write.assert_not_called()


def test_manifest_fills_every_column_and_round_trips(tmp_path):
    path = tmp_path / "clips.csv"
    manifest = Manifest(path)
    manifest.put("rally", source="custom", width=64, height=36, fps="30.000", note=None)
    assert manifest.rows["rally"]["clip"] == "rally" and manifest.rows["rally"]["note"] == ""
    assert set(manifest.rows["rally"]) == set(MANIFEST_COLUMNS)
    manifest.put("rally", note="again")
    manifest.save()
    assert read_manifest(path)["rally"]["source"] == "custom" and read_manifest(path)["rally"]["note"] == "again"
    assert Manifest(path).rows["rally"]["width"] == "64"
