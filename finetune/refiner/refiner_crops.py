"""The crops the label tool's suggestion network (``cnn_refiner.Placer``) is trained and checked on:
one CROP x CROP crop of the label frame and its neighbours around every reviewed, visible label,
cut in one pass over each clip's video and cached per clip in ``cnn_refiner.cache_dir()`` until its
labels or review change. ``refiner_training.train`` gathers them with ``gather_crops``."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from finetune.data import workspace as ws
from finetune.data.data_policy import is_in_progress, read_review_status
from finetune.data.labels import Labels, read_labels, reviewed_frames
from finetune.data.workspace import find_video, label_csv, labelled_clips, read_exclusions, review_json, video_meta
from finetune.refiner import cnn_refiner, refine
from finetune.refiner.cnn_refiner import CROP, cut_crop


def clip_crops(clip: str) -> Optional[np.ndarray]:
    """Crops around every reviewed, visible label of a clip; cached until its labels or review change."""
    labels_csv, review = label_csv(clip), review_json(clip)
    video = find_video(clip)
    if video is None or not review.is_file():
        return None
    cached = cnn_refiner.cache_dir() / f"{clip}.npz"
    stamp = max(labels_csv.stat().st_mtime, review.stat().st_mtime, video.stat().st_mtime)
    if cached.is_file() and cached.stat().st_mtime >= stamp:
        crops = np.load(cached)["crops"]
        if crops.shape[1:] == (CROP, CROP, 9):      # not cut for an older crop size
            return crops
    meta = video_meta(video)
    labels: Labels = read_labels(labels_csv, meta.width, meta.height)
    step = refine.label_spacing(labels)
    points = {f: labels[f] for f in reviewed_frames(clip) if labels.get(f)}   # a damaged review file is an error
    if not points:
        return None
    crops = crops_from_video(video, points, step, meta.height)
    cached.parent.mkdir(parents=True, exist_ok=True)
    out = np.stack(crops) if crops else np.zeros((0, CROP, CROP, 9), np.uint8)
    np.savez(cached, crops=out)
    return out


def frame_users(points: Dict[int, Tuple[float, float]], step: int) -> Dict[int, int]:
    """frame -> how many labels (the frame itself and the ones `step` either side) still need it."""
    users: Dict[int, int] = {}
    for f in points:
        for g in (f - step, f, f + step):
            users[g] = users.get(g, 0) + 1
    return users


def cut_label(f: int, kept: Dict[int, np.ndarray], users: Dict[int, int], point: Tuple[float, float], step: int,
              height: int) -> Optional[np.ndarray]:
    """The crop for label frame `f` from the frames kept so far (None when the video never showed
    it), letting go of every frame that no later label needs."""
    crop = cut_crop([kept.get(g, kept[f]) for g in (f - step, f, f + step)], point, height) if f in kept else None
    for g in (f - step, f, f + step):
        users[g] -= 1
        if users[g] == 0:
            kept.pop(g, None)
    return crop


def crops_from_video(video: Path, points: Dict[int, Tuple[float, float]], step: int, height: int) -> List[np.ndarray]:
    """One crop per labelled frame in `points`, in one pass over the video: each frame is kept
    only while a label `step` frames either side still needs it."""
    capture = cv2.VideoCapture(str(video))
    users = frame_users(points, step)
    pending, kept, crops, index = sorted(points), {}, [], 0
    while pending:
        ok, image = capture.read()
        if ok and index in users:
            kept[index] = image
        index += 1
        while pending and (not ok or index > pending[0] + step):
            f = pending.pop(0)
            crop = cut_label(f, kept, users, points[f], step, height)
            if crop is not None:
                crops.append(crop)
        if not ok:
            break
    capture.release()
    return crops


def gather_crops(log: Callable[..., None] = lambda *parts: print(*parts, flush=True)) -> Dict[str, np.ndarray]:
    """{clip: crops} for every clip with reviewed frames that the workspace's exclude.txt does not
    list and its review_status.json does not list as in progress (``in_progress_clips``: a person is
    labelling it now, so its partly reviewed frames are not settled). This is the clip set the folds,
    the "sure" threshold, the self-check and the final network all use. The trust signal is the
    per-frame review sidecar on top of those two lists; the trainable/scorable rules of
    ``finetune.data.data_policy`` are not consulted, since nothing here selects clips by source."""
    excluded = set(read_exclusions())
    status = read_review_status(ws.current().root)
    out = {}
    for clip in labelled_clips():
        if clip in excluded:
            log(f"  {clip}: skipped (listed in exclude.txt)")
            continue
        if is_in_progress(clip, status):
            log(f"  {clip}: skipped (in progress in review_status.json)")
            continue
        crops = clip_crops(clip)
        if crops is not None and len(crops):
            out[clip] = crops
            log(f"  {clip}: {len(crops)} reviewed frames")
    return out
