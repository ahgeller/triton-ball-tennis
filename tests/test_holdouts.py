"""finetune/data/holdouts.py: the named held-out sets record and its expansion to clips, on
records under tmp_path; plus the shape of the tracked finetune/holdouts.json."""
import json

import pytest

from finetune.data.holdouts import HoldoutSet, holdout_clips, holdout_set, read_holdouts
from finetune.data.labels import LabelFormatError
from finetune.data.workspace import read_manifest
from tennis_tracker.paths import ROOT

MANIFEST = {"video1": {"group": "own-camera"}, "video2": {"group": "own-camera"}, "video13": {"group": "prof_game2"},
            "grid_match8": {"group": "grid_match8"}, "grid_match16": {"group": "grid_match16"}}


def record(tmp_path, sets):
    path = tmp_path / "holdouts.json"
    path.write_text(json.dumps(sets), encoding="utf-8")
    return path


def test_read_holdouts_reads_every_set_and_refuses_a_malformed_one(tmp_path):
    assert read_holdouts(tmp_path / "missing.json") == {}
    path = record(tmp_path, {"a": {"groups": ["own-camera"], "clips": ["grid_match8"], "note": "why"},
                             "b": {"clips": ["video13"]}})
    assert read_holdouts(path) == {"a": HoldoutSet("a", ("own-camera",), ("grid_match8",), "why"),
                                   "b": HoldoutSet("b", (), ("video13",), "")}
    with pytest.raises(LabelFormatError, match="names no groups and no clips"):
        read_holdouts(record(tmp_path, {"empty": {"note": "nothing"}}))
    with pytest.raises(LabelFormatError, match="'clips' is not a list of names"):
        read_holdouts(record(tmp_path, {"bad": {"clips": "video13"}}))
    with pytest.raises(LabelFormatError, match="is not an object"):
        read_holdouts(record(tmp_path, {"bad": ["video13"]}))
    with pytest.raises(LabelFormatError):
        read_holdouts(record(tmp_path, ["not", "an", "object"]))


def test_holdout_set_names_the_known_sets_when_asked_for_an_unknown_one(tmp_path):
    path = record(tmp_path, {"a": {"clips": ["video13"]}, "b": {"clips": ["video1"]}})
    assert holdout_set("a", path).clips == ("video13",)
    with pytest.raises(ValueError, match=r"no held-out set 'c' in .*holdouts.json \(known: a, b\)"):
        holdout_set("c", path)
    with pytest.raises(ValueError, match=r"\(known: none\)"):
        holdout_set("a", tmp_path / "missing.json")


def test_holdout_clips_expands_groups_and_clips_in_labelled_order(tmp_path):
    path = record(tmp_path, {"mixed": {"groups": ["own-camera"], "clips": ["grid_match16"]}})
    labelled = ["grid_match8", "grid_match16", "video1", "video2", "video13", "video99"]     # video99: not in clips.csv
    assert holdout_clips("mixed", labelled, MANIFEST, path) == ["grid_match16", "video1", "video2"]
    assert holdout_clips("mixed", ["video2", "video1"], MANIFEST, path) == ["video2", "video1"]


def test_holdout_clips_takes_a_set_as_listed_with_no_recording_closure(tmp_path):
    """video1 and video2 share the own-camera recording, but a set naming video1 alone holds
    out video1 alone: a set is explicit (the trainer and the evaluator both read it this way),
    and only the trainer's --val-clips closes over the recording. Naming the group takes both."""
    path = record(tmp_path, {"one": {"clips": ["video1"]}, "whole": {"groups": ["own-camera"]}})
    labelled = ["video1", "video2", "video13"]
    assert holdout_clips("one", labelled, MANIFEST, path) == ["video1"]
    assert holdout_clips("whole", labelled, MANIFEST, path) == ["video1", "video2"]


def test_holdout_clips_refuses_an_entry_the_manifest_has_no_clip_for(tmp_path):
    """A typo in the record would otherwise hold out nothing and train on the footage it meant to
    keep. The guard reads clips.csv, not the run's clips: a run on one source may still name a
    set that mostly lives in another, and then holds out the members it has."""
    path = record(tmp_path, {"typo": {"groups": ["own-camera", "prof_game9"], "clips": ["grid_match88"]}})
    with pytest.raises(ValueError, match=r"'typo' names \['grid_match88', 'prof_game9'\], which clips.csv has no clip for"):
        holdout_clips("typo", ["video1", "grid_match8"], MANIFEST, path)
    elsewhere = record(tmp_path, {"own": {"groups": ["own-camera"]}})
    assert holdout_clips("own", ["video13", "grid_match8"], MANIFEST, elsewhere) == []
    with pytest.raises(ValueError, match="clips.csv has no clip for"):
        holdout_clips("own", ["video1"], {}, elsewhere)                       # no manifest: the labelled clips stand in
    assert holdout_clips("own", ["video1"], {"video1": {"group": "own-camera"}}, elsewhere) == ["video1"]


def test_the_tracked_record_names_mixed_v1_over_groups_and_clips_the_manifest_has():
    sets = read_holdouts(ROOT / "finetune" / "holdouts.json")
    mixed = sets["mixed_v1"]
    assert mixed.groups == ("prof_game4", "prof_game7")
    assert mixed.clips == tuple(f"video{n}" for n in range(1, 11)) + ("grid_match8", "grid_match16", "grid_match22")
    manifest = read_manifest(ROOT / "finetune" / "clips.csv")
    members = holdout_clips("mixed_v1", list(manifest), manifest, ROOT / "finetune" / "holdouts.json")
    assert len(members) == 24 and "video11" not in members and "video12" not in members
    assert {manifest[clip]["source"] for clip in members} == {"custom", "prof", "grid"}
