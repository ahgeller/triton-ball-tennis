"""Which clips may be trained on, validated on or scored: read from explicit records, never
inferred from a clip's source or file name alone. The trainer (``finetune.training.clips``),
the evaluator (``evaluation.archive``, ``ft.py eval``) and the status listings all ask here, so
one clip is admitted or kept out by the same rule everywhere.

Three records say different things. ``review_status.json`` (``verified_clips`` and
``verified_video_min``) names the clips whose labels a person has checked; without it an
external archive is taken at its word. ``exclude.txt`` lists clips whose labels do not belong
to their video (``ft.py check --audit --fix`` writes it; ``finetune.data.workspace`` reads it);
such labels are ground truth for nothing, so training and evaluation both leave them out.
``clips.csv`` tags each clip with a source (``clip_source`` is the one reading of it); two
sources are never trusted for training because nobody has checked their labels.

The label tool's suggestion network (``finetune.refiner.refiner_training``) is the one trainer
that does not ask here. Its trust signal is per frame: the frames a person confirmed or edited,
from each clip's review sidecar, on top of the workspace exclusion list (``exclude.txt``, read
through ``finetune.data.workspace``). It never selects clips by source, so the trainable and
scorable clip rules below do not apply to it.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Collection, Dict, List, Optional, Sequence

from finetune.data.labels import read_json_object
from finetune.data.workspace import REVIEW_STATUS_NAME

OWN_SOURCES = ("custom", "custom-uncorrected")                     # the user's own camera
UNTRUSTED_SOURCES = frozenset({"custom-uncorrected", "unknown"})   # machine labels nobody checked
UNMANIFESTED_SOURCE = "custom"     # every clip of a workspace with no clips.csv at all: nothing was imported, so the user's own
UNLISTED_SOURCE = "unknown"        # a clip a clips.csv exists but does not list: untrusted until it is imported or tagged


def read_review_status(workspace: Path) -> Optional[dict]:
    """The review policy (``verified_clips``, ``verified_video_min``), or None when the workspace
    has none. A damaged or non-object file is a ``LabelFormatError`` naming it, like the sidecars."""
    return read_json_object(workspace / REVIEW_STATUS_NAME, "review policy")


def clip_source(clip: str, manifest: Dict[str, dict]) -> str:
    """The source a clip is treated as, everywhere: its clips.csv row's; ``custom`` when the
    workspace has no clips.csv at all; ``unknown`` (untrusted) when clips.csv exists but does not
    list it. One default for the trainer, the status listing and the menu."""
    source = manifest.get(clip, {}).get("source")
    if source:
        return source
    return UNMANIFESTED_SOURCE if not manifest else UNLISTED_SOURCE


def is_verified(clip: str, status: Optional[dict]) -> bool:
    if status is None:
        return True  # external archives without a review policy retain their contract
    if clip in status.get("verified_clips", []):
        return True
    match = re.fullmatch(r"video(\d+)", clip)
    minimum = status.get("verified_video_min")
    return bool(match and minimum is not None and int(match.group(1)) >= int(minimum))


def is_scorable(clip: str, status: Optional[dict], excluded: Collection[str]) -> bool:
    """May a clip's labels stand as ground truth for evaluation: reviewed, and not in exclude.txt."""
    return is_verified(clip, status) and clip not in excluded


def is_trainable(clip: str, source: str, status: Optional[dict], excluded: Collection[str] = ()) -> bool:
    """May the trainer learn from this clip: labels it could be scored on, from a trusted source."""
    return is_scorable(clip, status, excluded) and source not in UNTRUSTED_SOURCES


def evaluation_clips(labelled: Sequence[str], manifest: Dict[str, dict], status: Optional[dict],
                     excluded: Collection[str]) -> List[str]:
    """The clips ``ft.py eval`` scores by default: every labelled clip whose labels may be scored,
    and when there is no review policy only the own-camera ones. Empty when nothing qualifies;
    it never stands for "every clip"."""
    clips = [clip for clip in labelled if is_scorable(clip, status, excluded)]
    if status is None:
        clips = [clip for clip in clips if clip_source(clip, manifest) in OWN_SOURCES]
    return clips
