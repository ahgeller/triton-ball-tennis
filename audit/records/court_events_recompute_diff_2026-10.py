"""Count the court events a run's tracking JSONs hold against the events the current code
recomputes from the same frames (evaluation.court_events_score.recompute_block), so a change
to the event stages is judged on identical selections without a GPU run.

    set PYTHONPATH=. && python audit/records/court_events_recompute_diff_2026-10.py <json or folder of <clip>_gridtracknet.json> [more ...]

Per clip: the stored and the recomputed "[court events]" counts (confirmed, candidate,
kinematic, contacts, legs, inferred legs, volleys), then the totals and how the confirmed
landings compare: matched one to one within the scorer's 0.1 s tolerance (the same landing,
possibly on another frame), and the rest, which only one side found. Run from the repository
root with the conda interpreter.
"""
import json
import sys
from pathlib import Path

from evaluation.court_events_score import TOLERANCE_S, match_landings, recompute_block
from tennis_tracker.court.court_scene import seconds_to_frames
from tennis_tracker.paths import BOUNCE_MODEL

FIELDS = ("confirmed", "candidate", "kinematic", "contacts", "legs", "inferred", "volley")


def counts(block):
    legs = block.get("rally_legs", [])
    return dict(zip(FIELDS, (
        len(block.get("bounces", [])), len(block.get("bounce_candidates", [])), len(block.get("kinematic_bounces", [])),
        len(block.get("contacts", [])), len(legs),
        sum(leg["quality"] == "inferred" for leg in legs), sum(leg["outcome"] == "volley" for leg in legs),
    )))


def line(values):
    return (f"{values['confirmed']:3d} confirmed ({values['candidate']:3d} candidate, {values['kinematic']:3d} kinematic), "
            f"{values['contacts']:3d} contacts, {values['legs']:3d} legs ({values['inferred']} inferred, {values['volley']} volley)")


def main(paths):
    files = []
    for arg in paths:
        path = Path(arg)
        files.extend(sorted(path.glob("*_gridtracknet.json")) if path.is_dir() else [path])
    totals = {"stored": dict.fromkeys(FIELDS, 0), "recomputed": dict.fromkeys(FIELDS, 0)}
    same_frame = shifted = only_stored = only_fresh = moved_positions = 0
    clips_differing = 0
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        stored = payload["court_events"]
        if not stored:
            print(f"{path.name}: no court events stored")
            continue
        fresh = recompute_block(payload, BOUNCE_MODEL)
        before, after = counts(stored), counts(fresh)
        for key in FIELDS:
            totals["stored"][key] += before[key]
            totals["recomputed"][key] += after[key]
        fps = float(payload["video"]["fps"])
        old, new = stored["bounces"], fresh["bounces"]
        pairs = match_landings([e["frame"] for e in old], [e["frame"] for e in new], seconds_to_frames(TOLERANCE_S, fps))
        for oi, ni in pairs:
            if old[oi]["frame"] == new[ni]["frame"]:
                same_frame += 1
                if (old[oi]["x"], old[oi]["y"]) != (new[ni]["x"], new[ni]["y"]):
                    moved_positions += 1
            else:
                shifted += 1
        only_stored += len(old) - len(pairs)
        only_fresh += len(new) - len(pairs)
        mark = "" if before == after else "   <-- counts differ"
        clips_differing += before != after
        print(f"{path.stem.replace('_gridtracknet', ''):52s} stored     {line(before)}")
        print(f"{'':52s} recomputed {line(after)}{mark}")
    print()
    print(f"{'TOTAL':52s} stored     {line(totals['stored'])}")
    print(f"{'':52s} recomputed {line(totals['recomputed'])}   ({clips_differing} of {len(files)} clips differ)")
    print(f"confirmed landings matched within {TOLERANCE_S:.1f}s: {same_frame} on the same frame "
          f"({moved_positions} at another position), {shifted} on another frame; "
          f"only in the stored run: {only_stored}; only recomputed: {only_fresh}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
