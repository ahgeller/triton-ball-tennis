"""The frames an audit flagged for a person to look at: the flag sidecar (``<clip>_ball.suspect.json``,
its format in ``finetune.data.labels``) as the audit writes it and the label tool and the menu read it.

Two kinds of flag: ``unplaceable`` (which video frame this label belongs to could not be worked
out) and ``check`` (the detector disagrees with the label). ``flag_frames`` records one kind,
``open_flags`` reads what is in the file and ``outstanding_flags`` is what is still worth a look.
A frame answered in the label tool (SPACE or a fix) is not asked about again while its label is
still the one it was answered at. Nothing here needs the detector or anything beyond the standard
library, so ``ft.py`` and the menu read the flags under any interpreter.
"""
from __future__ import annotations

import datetime as dt
from typing import Dict, List, Optional, Tuple

from finetune.data.detector_metrics import REFERENCE_HEIGHT, REFERENCE_WIDTH
from finetune.data.labels import FLAG_ANSWER_DECIMALS, FlagKind, Flags, Labels, read_flags, read_labels, write_flags
from finetune.data.workspace import find_video, label_csv, read_manifest, suspect_json, video_meta

SAME_POINT_PX = 10.0 ** -FLAG_ANSWER_DECIMALS   # an answered flag's label and the current one agree within this


def label_frame_size(clip: str, row: Dict[str, str]) -> Tuple[float, float]:
    """The frame size a clip's "no ball" sentinel is read at: its clips.csv row's, else its video's,
    else the 1080p reference. The guess comes last because a 720p sentinel read at 1080p is a
    visible ball at (1279, 0)."""
    if row.get("width") and row.get("height"):
        return float(row["width"]), float(row["height"])
    video = find_video(clip)
    if video is not None:
        try:
            _, width, height = video_meta(video)[:3]
            return float(width), float(height)
        except RuntimeError:
            pass
    return float(REFERENCE_WIDTH), float(REFERENCE_HEIGHT)


def open_flags(clip: str) -> Dict[str, List[int]]:
    """Frames of this clip somebody still has to look at, in two kinds.

    unplaceable  we could not work out which video frame this label belongs to - the clip is only
                 partly repaired, and these are the doubtful part of it
    check        the detector disagrees with the label here. Usually that is the detector losing a
                 ball at the racket rather than a bad label, so it is a prompt, not a defect
    """
    flags = read_flags(suspect_json(clip))
    return {"unplaceable": flags["unplaceable"], "check": flags["check"]}


def flag_frames(clip: str, kind: FlagKind, frames: List[int], note: str) -> None:
    """Record (or clear) one kind of flag, leaving the other kind alone."""
    path = suspect_json(clip)
    flags: Flags = read_flags(path)
    flags[kind] = sorted(set(int(f) for f in frames))
    flags["notes"][kind] = note
    flags["generated"] = str(dt.date.today())
    answered = flags["answered"]
    if answered:
        # A frame you already answered (SPACE or a fix in the label tool) is not asked about again
        # while its label is still the one you answered at; move the label and it can be flagged anew.
        labels = clip_labels(clip)
        flags[kind] = [f for f in flags[kind]
                       if not (str(f) in answered and same_point(answered[str(f)], labels.get(f, "missing")))]
    if not (flags["unplaceable"] or flags["check"] or answered):
        path.unlink(missing_ok=True)
        return
    write_flags(path, flags)


def clip_labels(clip: str) -> Labels:
    """frame -> (x, y), or None where the label says no ball, from the clip's label file (the
    frame size from clips.csv, else the video, else 1080p: ``label_frame_size``)."""
    path = label_csv(clip)
    if not path.is_file():
        return {}
    width, height = label_frame_size(clip, read_manifest().get(clip, {}))
    return read_labels(path, int(width), int(height))


def same_point(answered: Optional[List[float]], label) -> bool:
    if answered is None or label is None:
        return answered is None and label is None
    return label != "missing" and abs(answered[0] - label[0]) < SAME_POINT_PX and abs(answered[1] - label[1]) < SAME_POINT_PX


def outstanding_flags(clip: str) -> Dict[str, List[int]]:
    """The flagged frames still to deal with.

    The label tool takes a flag out of the file once you answer it - SPACE answers it as fully as
    a fix does - so what is in the file is still open, even on a frame you had reviewed before the
    audit raised it. A frame with no ball on it is dropped from 'unplaceable' too - there is no
    position to put in the wrong place, so which side of a cut it lands on cannot hurt anything.
    """
    flagged = open_flags(clip)
    if not (flagged["unplaceable"] or flagged["check"]):
        return flagged
    blind = {frame for frame, point in clip_labels(clip).items() if point is None} if flagged["unplaceable"] else set()
    return {"unplaceable": [f for f in flagged["unplaceable"] if f not in blind],
            "check": list(flagged["check"])}
