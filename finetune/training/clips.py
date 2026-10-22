"""Which labelled clips a training run uses, and which of them it holds out.

Data contract (same as the archive): ``videos/<clip>.mp4`` + ``labels/<clip>_ball.csv``
with ``frame,ball_x,ball_y`` rows at 30 FPS cadence (every 2nd frame of a 60 FPS clip, every
frame at 30 FPS; the first labelled frame may be odd) and invisible balls parked in the
top-right corner. ``clips.csv`` (written by import_data.py) tags every clip with a source and a
recording group, so a run can pick and hold out by them (``ClipSelection``, ``validation_split``);
``holdouts.json`` names fixed held-out sets of whole recordings (``--holdout``). Clips named by
hand (``--val-clips``) take the rest of their recording with them; a named set is held out as
listed, the same clips the evaluator's ``--holdout`` scores.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from finetune.data.data_policy import clip_source, is_trainable, is_verified, read_review_status, untrainable_reason
from finetune.data.holdouts import holdout_clips
from finetune.data.workspace import (HOLDOUTS_NAME, MANIFEST_NAME, clip_of, find_video, label_files, read_exclusions,
                                     read_manifest, recording_of)

DEFAULT_VAL_CLIP = "video10"

Clip = Tuple[str, Path, Path, str]   # (clip, video, label csv, source)


def matches(clip: str, patterns: Optional[List[str]]) -> bool:
    return bool(patterns) and any(fnmatch.fnmatchcase(clip, pattern) for pattern in patterns)


@dataclass(frozen=True)
class ClipSelection:
    """Which labelled clips a run trains on, as the trainer's flags say it (the defaults: every
    clip the review policy allows, exclude.txt honoured)."""
    sources: Optional[List[str]] = None         # --sources: only clips with these clips.csv sources
    exclude_clips: Optional[List[str]] = None   # --exclude-clips: stems or globs left out entirely
    include_excluded: bool = False              # --include-excluded: keep excluded/unverified clips too
    cameras: Optional[List[str]] = None         # --cameras: only clips with these camera tags

    def admits(self, clip: str, row: Dict[str, str], source: str, review_status: dict, excluded: Set[str]) -> bool:
        """Whether a labelled clip with this manifest row and source is in the selection."""
        if not self.include_excluded and not is_trainable(clip, source, review_status, excluded):
            return False
        if self.sources and source not in self.sources:
            return False
        if self.cameras and (row.get("camera") or "untagged") not in self.cameras:
            return False
        return not matches(clip, self.exclude_clips)


def _excluded_clips(selection: ClipSelection, labels: Path, manifest: Dict[str, Dict[str, str]]) -> Set[str]:
    """The clips exclude.txt keeps out of this selection (none with --include-excluded); also the
    place a --cameras request meets a manifest without camera tags."""
    excluded = set() if selection.include_excluded else set(read_exclusions(labels.parent))
    if selection.cameras and not any(row.get("camera") for row in manifest.values()):
        raise ValueError("--cameras needs camera tags in clips.csv: run  python finetune/ft.py camera  first")
    return excluded


def find_clips(videos: Path, labels: Path, selection: ClipSelection = ClipSelection()) -> List[Clip]:
    """Every (clip, video, label csv, source) the selection admits, in label-file order. A labelled
    clip the records keep out (exclude.txt, the review policy, an untrusted source) is announced
    with a ``[skip]`` line, as the evaluator does, so a clip missing from a run is never a mystery;
    the run's own --sources / --cameras / --exclude-clips choices are not."""
    manifest = read_manifest(labels.parent / MANIFEST_NAME)
    review_status = read_review_status(labels.parent)
    excluded = _excluded_clips(selection, labels, manifest)
    clips: List[Clip] = []
    for csv_path in label_files(labels):
        clip = clip_of(csv_path)
        row = manifest.get(clip, {})
        source = clip_source(clip, manifest)
        if not selection.admits(clip, row, source, review_status, excluded):
            reason = None if selection.include_excluded else untrainable_reason(clip, source, review_status, excluded)
            if reason:
                print(f"[skip] {clip}: {reason} (--include-excluded keeps it)")
            continue
        video = find_video(clip, videos)
        if video is None:
            raise FileNotFoundError(f"No video for {csv_path.name} in {videos}")
        clips.append((clip, video, csv_path, source))
    if not clips:
        raise FileNotFoundError(f"No *_ball.csv files in {labels}"
                                + (f" for sources {selection.sources}" if selection.sources else ""))
    return clips


# --------------------------------------------------------------------------- the validation split

def resolve_val_clips(names: List[str], patterns: Optional[List[str]]) -> List[str]:
    if patterns:
        chosen = [clip for clip in names if matches(clip, patterns)]
        unmatched = [pattern for pattern in patterns if not any(fnmatch.fnmatchcase(clip, pattern) for clip in names)]
        if unmatched:
            raise ValueError(f"Validation pattern(s) match nothing: {unmatched}; have {names[:8]}...")
        return chosen
    return [DEFAULT_VAL_CLIP] if DEFAULT_VAL_CLIP in names else [names[-1]]


def validation_split(clips: List[Clip], labels: Path, patterns: Optional[List[str]],
                     holdout: Optional[str] = None) -> List[str]:
    """The held-out clips, all of which must have reviewed labels, and never the whole set.

    ``--val-clips`` names clips by hand, so the split closes over the recording each was cut
    from (``recording_of``: the clips.csv group, else the origin video) and no rally of a
    held-out video trains. A ``--holdout`` set is explicit: exactly its members among the run's
    clips (``holdout_clips``), with no closure, so the trainer holds out the same clips
    ``evaluate_archive.py --holdout`` scores. Its author already chose whole recordings, and
    names the clips of a recording one by one when some of them must stay out of the set.
    """
    names = [clip for clip, _, _, _ in clips]
    metadata = read_manifest(labels.parent / MANIFEST_NAME)
    if holdout:
        if patterns:
            raise ValueError("--holdout and --val-clips are alternatives; give one")
        val_clips = holdout_clips(holdout, names, metadata, labels.parent / HOLDOUTS_NAME)
        if not val_clips:
            raise ValueError(f"held-out set {holdout!r} covers none of this run's clips")
    else:
        recordings = {recording_of(name, metadata) for name in resolve_val_clips(names, patterns)}
        val_clips = [name for name in names if recording_of(name, metadata) in recordings]
    review_status = read_review_status(labels.parent)
    unverified = [name for name in val_clips if not is_verified(name, review_status)]
    if unverified:
        raise ValueError(f"Validation requires reviewed labels, including group members: {unverified}")
    if len(val_clips) == len(names):
        raise ValueError("Group-safe validation leaves no training clips; choose an independent recording group")
    return val_clips
