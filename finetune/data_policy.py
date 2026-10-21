"""Which clips may be trained on, validated on or scored: read from explicit records, never
inferred from a clip's source or file name alone.

Three records say different things. ``review_status.json`` (``verified_clips`` and
``verified_video_min``) names the clips whose labels a person has checked; without it an
external archive is taken at its word. ``exclude.txt`` lists clips whose labels do not belong
to their video (``ft.py check --audit --fix`` writes it; ``finetune.workspace`` reads it).
``clips.csv`` tags each clip with a source; two sources are never trusted for training because
nobody has checked their labels.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from finetune.workspace import REVIEW_STATUS_NAME

OWN_SOURCES = ("custom", "custom-uncorrected")                     # the user's own camera
UNTRUSTED_SOURCES = frozenset({"custom-uncorrected", "unknown"})   # machine labels nobody checked


def read_review_status(workspace: Path) -> Optional[dict]:
    path = workspace / REVIEW_STATUS_NAME
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def is_verified(clip: str, status: Optional[dict]) -> bool:
    if status is None:
        return True  # external archives without a review policy retain their contract
    if clip in status.get("verified_clips", []):
        return True
    match = re.fullmatch(r"video(\d+)", clip)
    minimum = status.get("verified_video_min")
    return bool(match and minimum is not None and int(match.group(1)) >= int(minimum))


def is_trainable(clip: str, source: str, status: Optional[dict]) -> bool:
    """May the trainer learn from this clip: reviewed labels from a source that is trusted."""
    return is_verified(clip, status) and source not in UNTRUSTED_SOURCES


def evaluation_clips(manifest: Dict[str, dict], status: Optional[dict]) -> Optional[List[str]]:
    """The clips ``ft.py eval`` scores by default: every verified clip when a review policy
    exists, else the own-camera clips. None means "every clip in the folder" (no manifest, or
    nothing qualifies)."""
    if not manifest:
        return None
    if status is not None:
        return [clip for clip in manifest if is_verified(clip, status)] or None
    return [clip for clip, row in manifest.items() if row.get("source") in OWN_SOURCES] or None
