"""Score a pipeline run's landings against the hand-labelled ones.

    python score_court_events.py --archive finetune --out-dir output/<run>/pipeline [--tier confirmed]
        [--tolerance-s 0.1] [--compare before.log] [--recompute]

For every clip that may be scored (``evaluation.archive.list_clips``: reviewed, not excluded) and has
a landing label file (``labels/<clip>_bounces.csv``, ``finetune.data.bounce_labels``), the run's tracking
JSON ``<out-dir>/<clip>_<model>.json`` is read; it must have been written with ``--court``, so that its
``court_events`` block is filled. The block's landings at the chosen tier are matched one to one to the
labelled landings within the tolerance (greedy, closest pair first). A label may sit halfway between
two frames (``frame_412.5``, a ball that landed between them): the tolerance is kept as a fractional
number of frames (``tolerance_s * fps``, never rounded), so a landing on either frame is half a frame
from it, and the timing error is that fraction. Per clip and pooled (``ALL``):
labels, landings found, hits, precision, recall, the median and p90 timing error of the hits, and the
median pixel error at 1080p scale where the label has a point (``resolve_points`` takes it from the ball
labels when the row left it blank). The unmatched labels and landings are listed under each clip.

Tiers: ``confirmed`` (the bounce model's confirmed landings, the default), ``placed`` (those plus the
candidates and kinematic bounces not within ``EVENT_MERGE_S`` of a confirmed one: what the leg
assembly places, before its court-plane test; ``rally_legs.inferred_landings``) and ``all`` (every
landing of every list, the confirmed ones first, those within ``EVENT_MERGE_S`` of one already kept
dropped as the same landing).

``--recompute`` rebuilds each clip's ``CourtScene`` from the JSON's frames (``export.frame_row_to_result``)
and player boxes and reruns the event stages with the code as it is now, so a change to them is judged
on the same selection without a new GPU run. ``--compare before.log`` sets this run's lines beside an
earlier log of this script, clip by clip, in the ``compare_eval_logs.py`` style.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

from evaluation.archive import archive_folders, default_archive, list_clips
from finetune.data import workspace as ws
from finetune.data.bounce_labels import BounceLabel, landing_text, read_bounce_labels, resolve_points
from finetune.data.detector_metrics import scale_for_width
from finetune.data.labels import read_labels
from tennis_tracker.court.bounce_events import infer_kinematic_bounces, predict_bounce_events
from tennis_tracker.court.contact_events import detect_player_contacts
from tennis_tracker.court.court_scene import EVENT_MERGE_S, BounceEvent, CourtScene, seconds_to_frames
from tennis_tracker.court.rally_legs import assemble_rally_legs, inferred_landings
from tennis_tracker.export import court_events_to_json, frame_row_to_result
from tennis_tracker.paths import BOUNCE_MODEL
from tennis_tracker.run_context import CourtEvents
from tennis_tracker.tracking_schema import BounceEventJson, CourtEventsJson, TrackingJson

TOLERANCE_S = 0.1          # a landing this close in time to a label is that label's landing
TIERS = ("confirmed", "placed", "all")
ALL = "ALL"                # the pooled line's name

LINE = re.compile(
    r"^(?P<name>\S+)\s+labels=\s*(?P<labels>\d+) found=\s*(?P<found>\d+) hits=\s*(?P<hits>\d+) "
    r"precision=(?P<precision>[\d.]+) recall=(?P<recall>[\d.]+) \| "
    r"dt med=\s*(?P<dt_med>[\d.]+)ms p90=\s*(?P<dt_p90>[\d.]+)ms \| px med=\s*(?P<px_med>[\d.]+px|n/a)"
)


# --------------------------------------------------------------------------- the landings of a JSON

def bounce_from_json(row: BounceEventJson) -> BounceEvent:
    return BounceEvent(int(row["frame"]), float(row["x"]), float(row["y"]), float(row["score"]),
                       row["origin"], row["position"])


def _without_repeats(events: Sequence[BounceEvent], fps: float) -> List[BounceEvent]:
    """``events`` in the order given, dropping each one within ``EVENT_MERGE_S`` of one already kept."""
    merge = seconds_to_frames(EVENT_MERGE_S, fps)
    kept: List[BounceEvent] = []
    for event in events:
        if not any(abs(event.frame - other.frame) <= merge for other in kept):
            kept.append(event)
    return kept


def landings_at_tier(block: CourtEventsJson, tier: str, fps: float) -> List[BounceEvent]:
    """The block's landings the tier admits, in frame order (see the module docstring)."""
    confirmed = [bounce_from_json(row) for row in block.get("bounces", [])]
    candidates = [bounce_from_json(row) for row in block.get("bounce_candidates", [])]
    kinematic = [bounce_from_json(row) for row in block.get("kinematic_bounces", [])]
    if tier == "confirmed":
        landings = list(confirmed)
    elif tier == "placed":
        landings = confirmed + inferred_landings(confirmed, candidates, kinematic, fps)
    elif tier == "all":
        inferred = sorted(candidates + kinematic, key=lambda event: -event.score)
        landings = _without_repeats(confirmed + inferred, fps)
    else:
        raise ValueError(f"unknown tier {tier!r}; expected one of {', '.join(TIERS)}")
    return sorted(landings, key=lambda event: event.frame)


# --------------------------------------------------------------------------- matching and scoring

def match_landings(label_frames: Sequence[float], event_frames: Sequence[float], tolerance: float) -> List[Tuple[int, int]]:
    """Greedy one-to-one matching: (label index, event index) pairs no further apart than
    ``tolerance`` frames, the closest pair taken first (ties by label, then event, order). Frames
    and the tolerance may be fractional: a label between two frames, a tolerance in seconds."""
    pairs = sorted(
        (abs(label - event), li, ei)
        for li, label in enumerate(label_frames)
        for ei, event in enumerate(event_frames)
        if abs(label - event) <= tolerance
    )
    matched: List[Tuple[int, int]] = []
    used_labels, used_events = set(), set()
    for _, li, ei in pairs:
        if li in used_labels or ei in used_events:
            continue
        matched.append((li, ei))
        used_labels.add(li)
        used_events.add(ei)
    return sorted(matched)


class ClipLandingScore(NamedTuple):
    """One clip's landings against its labels. ``timing_errors_s`` are signed (landing minus
    label) per hit; ``px_errors`` the hits' distances at 1080p scale where the label has a point."""

    clip: str
    labels: int
    found: int
    hits: int
    timing_errors_s: List[float]
    px_errors: List[float]
    unmatched_labels: List[BounceLabel]
    unmatched_events: List[BounceEvent]

    @property
    def precision(self) -> float:
        return self.hits / self.found if self.found else 0.0

    @property
    def recall(self) -> float:
        return self.hits / self.labels if self.labels else 0.0


def score_clip(clip: str, labels: Sequence[BounceLabel], events: Sequence[BounceEvent], fps: float,
               tolerance_s: float, px_scale: float) -> ClipLandingScore:
    tolerance = max(0.0, tolerance_s * fps)      # fractional: a label between two frames is half a frame from either
    pairs = match_landings([label.frame for label in labels], [event.frame for event in events], tolerance)
    timing, px = [], []
    for li, ei in pairs:
        label, event = labels[li], events[ei]
        timing.append((event.frame - label.frame) / fps)
        if label.point is not None:
            px.append(math.hypot(event.x - label.point[0], event.y - label.point[1]) * px_scale)
    matched_labels = {li for li, _ in pairs}
    matched_events = {ei for _, ei in pairs}
    return ClipLandingScore(
        clip, len(labels), len(events), len(pairs), timing, px,
        [label for li, label in enumerate(labels) if li not in matched_labels],
        [event for ei, event in enumerate(events) if ei not in matched_events],
    )


def pooled(scores: Sequence[ClipLandingScore]) -> ClipLandingScore:
    return ClipLandingScore(
        ALL,
        sum(score.labels for score in scores),
        sum(score.found for score in scores),
        sum(score.hits for score in scores),
        [value for score in scores for value in score.timing_errors_s],
        [value for score in scores for value in score.px_errors],
        [], [],
    )


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(math.ceil(fraction * len(ordered))) - 1)] if ordered else 0.0


def format_line(score: ClipLandingScore) -> str:
    """The metric line ``read_scores`` parses back."""
    absolute_ms = [abs(value) * 1000.0 for value in score.timing_errors_s]
    dt_med = statistics.median(absolute_ms) if absolute_ms else 0.0
    px_med = f"{statistics.median(score.px_errors):5.1f}px" if score.px_errors else "  n/a"
    return (
        f"{score.clip:14s} labels={score.labels:4d} found={score.found:4d} hits={score.hits:4d} "
        f"precision={score.precision:.3f} recall={score.recall:.3f} | "
        f"dt med={dt_med:4.0f}ms p90={_percentile(absolute_ms, 0.9):4.0f}ms | px med={px_med}"
    )


def unmatched_lines(score: ClipLandingScore, fps: float) -> List[str]:
    """What the clip's line does not show: the labels no landing matched and the landings no
    label matched, each with its frame (and time)."""
    lines = []
    if score.unmatched_labels:
        lines.append("    unmatched labels:   " + ", ".join(
            f"{landing_text(label.frame)} ({label.frame / fps:.2f}s)" for label in score.unmatched_labels))
    if score.unmatched_events:
        lines.append("    unmatched landings: " + ", ".join(
            f"{event.frame} ({event.frame / fps:.2f}s, {event.origin}, score {event.score:.2f})"
            for event in score.unmatched_events))
    return lines


# --------------------------------------------------------------------------- comparing two logs

class LandingScore(NamedTuple):
    labels: int
    found: int
    hits: int
    precision: float
    recall: float
    dt_med_ms: float
    dt_p90_ms: float
    px_med: Optional[float]


def read_scores(text: str) -> Dict[str, LandingScore]:
    """{clip or ALL: its scores} for every metric line in a log of this script."""
    scores: Dict[str, LandingScore] = {}
    for line in text.splitlines():
        match = LINE.match(line.strip())
        if match:
            px = match["px_med"]
            scores[match["name"]] = LandingScore(
                int(match["labels"]), int(match["found"]), int(match["hits"]),
                float(match["precision"]), float(match["recall"]), float(match["dt_med"]), float(match["dt_p90"]),
                None if px == "n/a" else float(px[:-2]),
            )
    return scores


def _delta(value: float, places: int = 3) -> str:
    return f"{value:+.{places}f}"


def compare(before: Dict[str, LandingScore], after: Dict[str, LandingScore]) -> str:
    """One row per clip in both logs, the pooled line last; positive precision and recall
    deltas and negative timing and pixel deltas are improvements."""
    shared = [name for name in before if name in after]
    clips = [name for name in shared if name != ALL]
    header = (f"{'clip':30s} {'labels':>6s} {'found':>12s} {'hits':>12s} {'precision':>16s} {'recall':>16s} "
              f"{'dt med ms':>14s} {'px med':>14s}")
    rows = [header, "-" * len(header)]
    for name in clips + ([ALL] if ALL in shared else []):
        old, new = before[name], after[name]
        if name == ALL:
            rows.append("")
        px = ("n/a" if new.px_med is None or old.px_med is None
              else f"{new.px_med:5.1f} ({_delta(new.px_med - old.px_med, 1)})")
        rows.append(
            f"{name[:30]:30s} {new.labels:6d} "
            f"{new.found:5d} ({new.found - old.found:+d}) "
            f"{new.hits:5d} ({new.hits - old.hits:+d}) "
            f"{new.precision:.3f} ({_delta(new.precision - old.precision)}) "
            f"{new.recall:.3f} ({_delta(new.recall - old.recall)}) "
            f"{new.dt_med_ms:5.0f} ({new.dt_med_ms - old.dt_med_ms:+.0f}) "
            f"{px}"
        )
    missing_after = [name for name in before if name not in after]
    missing_before = [name for name in after if name not in before]
    if missing_after:
        rows.append(f"\nonly in the first log: {', '.join(missing_after)}")
    if missing_before:
        rows.append(f"only in the second log: {', '.join(missing_before)}")
    return "\n".join(rows)


# --------------------------------------------------------------------------- recomputing from a JSON

def scene_from_payload(payload: TrackingJson) -> CourtScene:
    """The ``CourtScene`` the run's court-event stages read, rebuilt from its tracking JSON."""
    rows = payload["frames"]
    video = payload["video"]
    return CourtScene(
        [frame_row_to_result(row) for row in rows],
        [row.get("player_boxes") for row in rows],
        float(video["fps"]), int(video["width"]), int(video["height"]),
    )


def recompute_block(payload: TrackingJson, bounce_model: Path) -> CourtEventsJson:
    """The ``court_events`` block the stages produce now from the JSON's selection."""
    scene = scene_from_payload(payload)
    bounces = predict_bounce_events(scene, str(bounce_model))
    contacts = detect_player_contacts(scene)
    kinematic = infer_kinematic_bounces(scene)
    legs = assemble_rally_legs(scene, [row.get("court_keypoints") for row in payload["frames"]],
                               contacts, bounces, kinematic)
    return court_events_to_json(CourtEvents(True, list(bounces.confirmed), list(bounces.candidates),
                                            list(kinematic), list(contacts), list(legs)))


# --------------------------------------------------------------------------- the command

def _labelled_clips(names: Optional[List[str]], archive: Path, include_excluded: bool) -> List[Tuple[str, Path, Path]]:
    """(clip, ball label csv, landing label csv) for every scorable clip that has landing labels."""
    label_dir, _ = archive_folders(archive)
    found = []
    for video, csv_path in list_clips(names, archive, include_excluded):
        bounce_path = ws.bounce_csv(video.stem, label_dir)
        if bounce_path.is_file():
            found.append((video.stem, csv_path, bounce_path))
    return found


def score_run(clips: Sequence[Tuple[str, Path, Path]], out_dir: Path, model: str, tier: str, tolerance_s: float,
              recompute: Optional[Path]) -> List[str]:
    """The report lines for one model's run; raises FileNotFoundError or ValueError naming the
    JSON that is missing or holds no court events."""
    lines = [f"=== landings [{model}] tier={tier} tolerance={tolerance_s:.2f}s"
             f"{' (events recomputed from the frames)' if recompute else ''} ==="]
    scores = []
    for clip, csv_path, bounce_path in clips:
        json_path = out_dir / f"{clip}_{model}.json"
        if not json_path.is_file():
            raise FileNotFoundError(f"{json_path} is missing: run evaluate_archive.py --mode pipeline --court "
                                    f"--out-dir {out_dir} on the clip first")
        payload: TrackingJson = json.loads(json_path.read_text(encoding="utf-8"))
        block = recompute_block(payload, recompute) if recompute else payload.get("court_events") or {}
        if not block:
            raise ValueError(f"{json_path} has no court events: it was written without --court "
                             "(evaluate_archive.py --mode pipeline --court)")
        video = payload["video"]
        fps, width, height = float(video["fps"]), int(video["width"]), int(video["height"])
        labels = resolve_points(read_bounce_labels(bounce_path), read_labels(csv_path, width, height))
        events = landings_at_tier(block, tier, fps)
        score = score_clip(clip, labels, events, fps, tolerance_s, scale_for_width(width))
        scores.append(score)
        lines.append(format_line(score))
        lines.extend(unmatched_lines(score, fps))
    lines.append(format_line(pooled(scores)))
    return lines


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="the pipeline run's folder of <clip>_<model>.json files (evaluate_archive.py --out-dir)")
    parser.add_argument("--models", nargs="+", default=["gridtracknet"])
    parser.add_argument("--clips", nargs="*", default=None, help="Clip stems or globs, e.g. video6 'grid_match*'")
    parser.add_argument("--archive", default=None,
                        help="Label set: a finetune-style folder with videos/ and labels/ (default: $TENNIS_ARCHIVE, else finetune/)")
    parser.add_argument("--include-excluded", action="store_true",
                        help="Score every clip with landing labels, including those the review policy holds back")
    parser.add_argument("--tier", choices=TIERS, default="confirmed")
    parser.add_argument("--tolerance-s", type=float, default=TOLERANCE_S,
                        help=f"a landing within this many seconds of a label matches it (default {TOLERANCE_S})")
    parser.add_argument("--compare", type=Path, default=None, help="an earlier log of this script to print deltas against")
    parser.add_argument("--recompute", action="store_true",
                        help="rerun the court-event stages on each JSON's frames with the current code instead of "
                             "reading its court_events block")
    parser.add_argument("--bounce-model", type=Path, default=BOUNCE_MODEL, help="the CatBoost bounce model --recompute runs")
    args = parser.parse_args(argv)

    archive = Path(args.archive) if args.archive else default_archive()
    clips = _labelled_clips(args.clips, archive, args.include_excluded)
    if not clips:
        print(f"No clip in {archive} has a landing label file (labels/<clip>_bounces.csv)")
        return 1
    print(f"[archive] {archive}  ({len(clips)} clips with landing labels: {', '.join(clip for clip, _, _ in clips)})")
    lines: List[str] = []
    try:
        for model in args.models:
            lines.extend(score_run(clips, args.out_dir, model, args.tier, args.tolerance_s,
                                   args.bounce_model if args.recompute else None))
    except (FileNotFoundError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    print("\n".join(lines))
    if args.compare is not None:
        before = read_scores(args.compare.read_text(encoding="utf-8"))
        if not before:
            print(f"no landing metric lines found in {args.compare}", file=sys.stderr)
            return 1
        print(f"\n=== against {args.compare} ===")
        print(compare(before, read_scores("\n".join(lines))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
