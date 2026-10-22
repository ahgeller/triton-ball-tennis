"""Where a pipeline run loses the ball, from its tracking JSONs and the clips' labels.

    python evaluation/track_breaks.py output/fix1_no_court/pipeline [--compare before.log]

For every ``<clip>_<model>.json`` that ``evaluate_archive.py --mode pipeline`` wrote, the
labelled-visible frames with no output are bucketed by what the run had on that frame:
``no_track`` (no selector track observed it: the detector or the association never had it),
``other_track`` (only a track the timeline did not choose observed it) or ``chosen_pruned`` (the
chosen track observed it and the fills, the refit or the implausible-fill pruning let it go).
Beside them: ``switches``, present rows that jump more than a share of the diagonal from the
previous present row, and ``longest_gap``, the longest run of absent output between the clip's
first and last visible label. Pooled lines follow per frame rate (``fps=30``, ``fps=60``), per
clips.csv source and ``ALL``, so a 60 FPS clip's losses read against a 30 FPS clip's. Two logs
compare with ``--compare``; positive deltas in the miss buckets, switches and gap are worse.

When the JSONs record why each track started (schema 4), a second table counts, per frame
rate and over all clips, the tracks that started because a live track's gate turned their first
detection away, by gate (or because the track that accepted it took another detection,
``outbid``): that is where the association fragments, and which gate did it.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Set

from finetune.data.data_policy import clip_source
from finetune.data.labels import Labels, read_labels
from finetune.data.workspace import label_csv, natural_key, read_manifest
from tennis_tracker.tracking_schema import TrackingJson

SWITCH_DIAG_FRAC = 0.03     # a present row this far from the previous present row is a switch
LINE = re.compile(
    r"^(?P<name>\S+)\s+vis=\s*(?P<visible>\d+) hit=\s*(?P<hit>\d+) \| miss: no_track=\s*(?P<no_track>\d+) "
    r"other_track=\s*(?P<other_track>\d+) chosen_pruned=\s*(?P<chosen_pruned>\d+) \| "
    r"switches=\s*(?P<switches>\d+) longest_gap=\s*(?P<longest_gap>\d+)"
)


class ClipBreaks(NamedTuple):
    visible: int
    hit: int
    no_track: int
    other_track: int
    chosen_pruned: int
    switches: int
    longest_gap: int

    @property
    def missed(self) -> int:
        return self.visible - self.hit


def clip_breaks(payload: TrackingJson, labels: Labels, switch_frac: float = SWITCH_DIAG_FRAC) -> ClipBreaks:
    """Bucket a run's losses on one clip against its labels (frame -> point, or None for no ball)."""
    frames = payload["frames"]
    present = {int(row["frame"]): (float(row["x"]), float(row["y"])) for row in frames if row.get("present")}
    observed: Set[int] = set()
    for track in payload.get("tracks") or ():
        observed.update(int(item["frame"]) for item in track.get("observations") or ())
    chosen = payload.get("chosen_track") or {}
    chosen_frames = {int(item["frame"]) for item in chosen.get("observations") or ()}

    visible = sorted(frame for frame, point in labels.items() if point is not None)
    hit = no_track = other_track = chosen_pruned = 0
    for frame in visible:
        if frame in present:
            hit += 1
        elif frame in chosen_frames:
            chosen_pruned += 1
        elif frame in observed:
            other_track += 1
        else:
            no_track += 1

    video = payload["video"]
    switch_px = switch_frac * math.hypot(float(video["width"]), float(video["height"]))
    switches = 0
    previous = None
    for frame in sorted(present):
        point = present[frame]
        if previous is not None and math.hypot(point[0] - previous[0], point[1] - previous[1]) > switch_px:
            switches += 1
        previous = point

    longest_gap = run = 0
    if visible:
        for frame in range(visible[0], visible[-1] + 1):
            run = 0 if frame in present else run + 1
            longest_gap = max(longest_gap, run)
    return ClipBreaks(len(visible), hit, no_track, other_track, chosen_pruned, switches, longest_gap)


def pool(items: Iterable[ClipBreaks]) -> ClipBreaks:
    """The counts summed over clips; the longest gap is the longest of any clip."""
    items = list(items)
    return ClipBreaks(
        sum(item.visible for item in items), sum(item.hit for item in items),
        sum(item.no_track for item in items), sum(item.other_track for item in items),
        sum(item.chosen_pruned for item in items), sum(item.switches for item in items),
        max((item.longest_gap for item in items), default=0),
    )


def fmt(name: str, breaks: ClipBreaks) -> str:
    return (f"{name:14s} vis={breaks.visible:5d} hit={breaks.hit:5d} | miss: no_track={breaks.no_track:4d} "
            f"other_track={breaks.other_track:4d} chosen_pruned={breaks.chosen_pruned:4d} | "
            f"switches={breaks.switches:4d} longest_gap={breaks.longest_gap:4d}")


def read_scores(text: str) -> Dict[str, ClipBreaks]:
    """{clip or pooled-line name: its counts} for every line ``fmt`` printed into a log."""
    scores: Dict[str, ClipBreaks] = {}
    for line in text.splitlines():
        match = LINE.match(line.strip())
        if match:
            scores[match["name"]] = ClipBreaks(*(int(match[field]) for field in ClipBreaks._fields))
    return scores


def compare(before: Dict[str, ClipBreaks], after: Dict[str, ClipBreaks], only_changed: bool = False) -> str:
    """One row per name in both logs with the change in every count; pooled lines last."""
    shared = [name for name in before if name in after]
    pooled = [name for name in shared if name == "ALL" or "=" in name]
    clips = [name for name in shared if name not in pooled]
    header = (f"{'clip':14s} {'vis':>5s} {'hit':>12s} {'no_track':>12s} {'other_track':>12s} "
              f"{'chosen_pruned':>14s} {'switches':>12s} {'longest_gap':>12s}")
    rows = [header, "-" * len(header)]
    for name in clips + pooled:
        old, new = before[name], after[name]
        if only_changed and old == new:
            continue
        if pooled and name == pooled[0]:
            rows.append("")
        cells = " ".join(f"{getattr(new, field):5d} ({getattr(new, field) - getattr(old, field):+4d})"
                         for field in ClipBreaks._fields[1:])
        rows.append(f"{name[:14]:14s} {new.visible:5d} {cells}")
    missing_after = [name for name in before if name not in after]
    missing_before = [name for name in after if name not in before]
    if missing_after:
        rows.append(f"\nonly in the first log: {', '.join(missing_after)}")
    if missing_before:
        rows.append(f"only in the second log: {', '.join(missing_before)}")
    return "\n".join(rows)


def _rate_name(fps: float) -> str:
    return f"fps={int(round(fps / 30.0) * 30)}"


def track_starts(payload: TrackingJson) -> Counter[str]:
    """{start reason: tracks} for the run's tracks (empty when the JSON predates the record)."""
    return Counter(track["start"]["reason"] for track in payload.get("tracks") or () if "start" in track)


def start_lines(starts_by_group: Dict[str, Counter[str]]) -> List[str]:
    """One line per group listing its track starts by reason, most common first."""
    lines = ["track starts by the gate that turned the first detection away "
             "(first = no live track, outbid = the track that accepted it took another detection):"]
    for name, counter in sorted(starts_by_group.items()):
        total = sum(counter.values())
        parts = ", ".join(f"{reason}={count}" for reason, count in counter.most_common())
        lines.append(f"  {name:10s} tracks={total:5d}: {parts}")
    return lines


class _ClipReport(NamedTuple):
    """One clip's part of the report: its counts, the pooled lines it belongs to (``fps=<rate>``
    and ``source=<name>``) and its track starts by reason."""

    clip: str
    rate: str
    source: str
    breaks: ClipBreaks
    starts: Counter[str]


def report(out_dir: Path, model: str, switch_frac: float = SWITCH_DIAG_FRAC) -> List[str]:
    """The per-clip lines, then the pooled lines, for the run in ``out_dir``."""
    manifest = read_manifest()
    suffix = f"_{model}.json"
    per_clip: List[_ClipReport] = []
    for path in sorted(out_dir.glob(f"*{suffix}"), key=lambda item: natural_key(item.name)):
        clip = path.name[:-len(suffix)]
        payload: TrackingJson = json.loads(path.read_text(encoding="utf-8"))
        labels_path = label_csv(clip)
        if not labels_path.is_file():
            print(f"[skip] {clip}: no label file", file=sys.stderr)
            continue
        labels = read_labels(labels_path, int(payload["video"]["width"]), int(payload["video"]["height"]))
        per_clip.append(_ClipReport(clip, _rate_name(float(payload["video"]["fps"])),
                                    f"source={clip_source(clip, manifest)}",
                                    clip_breaks(payload, labels, switch_frac), track_starts(payload)))
    lines = [fmt(item.clip, item.breaks) for item in per_clip]
    if per_clip:
        lines.append("")
        groups: Dict[str, List[ClipBreaks]] = {}
        for item in per_clip:
            groups.setdefault(item.rate, []).append(item.breaks)
            groups.setdefault(item.source, []).append(item.breaks)
        for name in sorted(groups):
            lines.append(fmt(name, pool(groups[name])))
        lines.append(fmt("ALL", pool(item.breaks for item in per_clip)))
        starts_by_group: Dict[str, Counter[str]] = {}
        for item in per_clip:
            if item.starts:
                starts_by_group.setdefault(item.rate, Counter()).update(item.starts)
                starts_by_group.setdefault("ALL", Counter()).update(item.starts)
        if starts_by_group:
            lines.append("")
            lines.extend(start_lines(starts_by_group))
    return lines


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir", type=Path, help="A pipeline-mode evaluator output folder")
    parser.add_argument("--model", default="gridtracknet", help="The JSON suffix's model name")
    parser.add_argument("--switch-frac", type=float, default=SWITCH_DIAG_FRAC,
                        help="A jump between present rows over this share of the diagonal is a switch")
    parser.add_argument("--compare", type=Path, default=None,
                        help="A log this script printed earlier; print the change against it instead")
    args = parser.parse_args(argv)
    lines = report(args.out_dir, args.model, args.switch_frac)
    if args.compare is None:
        print("\n".join(lines))
    else:
        print(compare(read_scores(args.compare.read_text(encoding="utf-8")), read_scores("\n".join(lines))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
