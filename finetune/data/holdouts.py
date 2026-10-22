"""Named held-out sets of whole recordings (``holdouts.json``), so a detector is measured on
footage it never trained on, and every run that names the same set holds out and scores the
same clips.

A set names recording groups (the ``group`` column of ``clips.csv``: ``prof_game4``) and single
clips. ``holdout_clips`` expands a set to the labelled clips it covers, and to those exactly:
a set is explicit, so a clip of the same recording as a member joins it only when the set names
it (or its group) too, unlike the trainer's ``--val-clips``, which closes over the recording
(``workspace.recording_of``). The trainer's ``--holdout`` and the evaluator's both read the set
this way, so a run holds out the clips its scores are measured on. Whether each of those may be
trained on, held out or scored is the caller's policy (``data_policy``), the same one it applies
to every other clip. A name the record does not know, or an entry ``clips.csv`` has no clip for,
is an error, not an empty set: a typo in a held-out set would otherwise silently train on the
footage it meant to hold out.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, NamedTuple, Sequence, Tuple

from finetune.data.labels import LabelFormatError, read_json_object


class HoldoutSet(NamedTuple):
    """One named held-out set as ``holdouts.json`` records it."""
    name: str
    groups: Tuple[str, ...]     # clips.csv recording groups, every labelled member held out
    clips: Tuple[str, ...]      # single clips
    note: str                   # why these, in the record's own words


def _names(entry: dict, key: str, path: Path, name: str) -> Tuple[str, ...]:
    value = entry.get(key) or []
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise LabelFormatError(f"{path}: set {name!r}: {key!r} is not a list of names")
    return tuple(value)


def read_holdouts(path: Path) -> Dict[str, HoldoutSet]:
    """Every held-out set in the record at ``path``, by name; none when there is no record. A
    record that is not an object of sets is a ``LabelFormatError`` naming the file."""
    blob = read_json_object(path, "held-out sets record")
    if blob is None:
        return {}
    sets: Dict[str, HoldoutSet] = {}
    for name, entry in blob.items():
        if not isinstance(entry, dict):
            raise LabelFormatError(f"{path}: set {name!r} is not an object")
        groups, clips = _names(entry, "groups", path, name), _names(entry, "clips", path, name)
        if not groups and not clips:
            raise LabelFormatError(f"{path}: set {name!r} names no groups and no clips")
        sets[name] = HoldoutSet(name, groups, clips, str(entry.get("note") or ""))
    return sets


def holdout_set(name: str, path: Path) -> HoldoutSet:
    """The set called ``name`` in the record at ``path``; a ValueError naming the known sets otherwise."""
    sets = read_holdouts(path)
    if name not in sets:
        known = ", ".join(sorted(sets)) or "none"
        raise ValueError(f"no held-out set {name!r} in {path} (known: {known})")
    return sets[name]


def holdout_clips(name: str, labelled: Sequence[str], manifest: Dict[str, dict], path: Path) -> List[str]:
    """The clips among ``labelled`` that the set ``name`` covers, in ``labelled``'s order: every
    clip it names and every clip whose clips.csv group it names. A named clip or group that
    ``manifest`` (clips.csv; the labelled clips when there is none) has no clip for is a
    ValueError; a set that covers none of ``labelled`` is simply empty."""
    chosen = holdout_set(name, path)
    known = manifest if manifest else {clip: {} for clip in labelled}
    groups = {row.get("group") or "" for row in known.values()}
    unknown = [clip for clip in chosen.clips if clip not in known] + [group for group in chosen.groups if group not in groups]
    if unknown:
        raise ValueError(f"held-out set {name!r} names {unknown}, which clips.csv has no clip for")
    group_of = {clip: (manifest.get(clip) or {}).get("group") or "" for clip in labelled}
    return [clip for clip in labelled if clip in chosen.clips or group_of[clip] in chosen.groups]
