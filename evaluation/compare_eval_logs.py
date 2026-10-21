"""Set two ``evaluate_archive.py`` logs side by side, clip by clip.

    python evaluation/compare_eval_logs.py output/baseline/pipeline.log output/fix/pipeline.log

Both logs may be raw or pipeline mode; each ``name vis=... recall=... | ... | med=...px p90=...px``
line the evaluator prints is read, and for every clip in both logs the change in recall, wrong,
false alarm and median/p90 pixels is printed, pooled ``ALL`` (or threshold) lines included.
Clips in only one log are listed at the end. Positive recall deltas and negative wrong / false
alarm / pixel deltas are improvements.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, NamedTuple, Optional, Sequence

LINE = re.compile(
    r"^(?P<name>\S+)\s+vis=\s*(?P<vis>\d+) recall=(?P<recall>[\d.]+) wrong=(?P<wrong>[\d.]+) "
    r"miss=(?P<miss>[\d.]+) \| invis=\s*(?P<invis>\d+) falseAlarm=(?P<false_alarm>[\d.]+) \| "
    r"med=(?P<med>[\d.]+)px p90=(?P<p90>[\d.]+)px"
)


class Score(NamedTuple):
    visible: int
    recall: float
    wrong: float
    invisible: int
    false_alarm: float
    median_px: float
    p90_px: float


def read_scores(text: str) -> Dict[str, Score]:
    """{clip or pooled-line name: its scores} for every metric line in an evaluator log; a name
    printed twice (raw mode prints the sweep, then the per-clip table) keeps its last line."""
    scores: Dict[str, Score] = {}
    for line in text.splitlines():
        match = LINE.match(line.strip())
        if match:
            scores[match["name"]] = Score(int(match["vis"]), float(match["recall"]), float(match["wrong"]),
                                          int(match["invis"]), float(match["false_alarm"]),
                                          float(match["med"]), float(match["p90"]))
    return scores


def _delta(value: float, places: int = 3) -> str:
    return f"{value:+.{places}f}"


def compare(before: Dict[str, Score], after: Dict[str, Score], only_changed: bool = False) -> str:
    """The comparison table as text: one row per name in both logs, pooled lines last."""
    shared = [name for name in before if name in after]
    pooled = [name for name in shared if name == "ALL" or name.startswith("conf>=")]
    clips = [name for name in shared if name not in pooled]
    header = f"{'clip':30s} {'vis':>5s} {'recall':>14s} {'wrong':>14s} {'falseAlarm':>14s} {'med px':>12s} {'p90 px':>12s}"
    rows = [header, "-" * len(header)]
    for name in clips + pooled:
        old, new = before[name], after[name]
        if only_changed and old == new:
            continue
        if pooled and name == pooled[0]:
            rows.append("")
        rows.append(
            f"{name[:30]:30s} {new.visible:5d} "
            f"{new.recall:.3f} ({_delta(new.recall - old.recall)}) "
            f"{new.wrong:.3f} ({_delta(new.wrong - old.wrong)}) "
            f"{new.false_alarm:.3f} ({_delta(new.false_alarm - old.false_alarm)}) "
            f"{new.median_px:5.1f} ({_delta(new.median_px - old.median_px, 1)}) "
            f"{new.p90_px:5.1f} ({_delta(new.p90_px - old.p90_px, 1)})"
        )
    missing_after = [name for name in before if name not in after]
    missing_before = [name for name in after if name not in before]
    if missing_after:
        rows.append(f"\nonly in the first log: {', '.join(missing_after)}")
    if missing_before:
        rows.append(f"only in the second log: {', '.join(missing_before)}")
    return "\n".join(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("before", type=Path, help="evaluator log of the reference run")
    parser.add_argument("after", type=Path, help="evaluator log of the run to judge")
    parser.add_argument("--only-changed", action="store_true", help="leave out clips whose numbers did not move")
    args = parser.parse_args(argv)
    before = read_scores(args.before.read_text(encoding="utf-8"))
    after = read_scores(args.after.read_text(encoding="utf-8"))
    if not before or not after:
        print("no evaluator metric lines found in", args.before if not before else args.after, file=sys.stderr)
        return 1
    print(compare(before, after, only_changed=args.only_changed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
