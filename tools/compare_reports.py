#!/usr/bin/env python3
"""Diff two validation reports side by side.

Usage:
  python tools/compare_reports.py --before <a>.json --after <b>.json

Prints recall, error stats, fill ratio, large jumps, FP count, accuracy buckets,
and source-mix deltas. Arrows show whether the change moved each metric in the
better direction (better metric defined per-row).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


# (key, label, "higher" or "lower" is better, format spec)
METRIC_ROWS: Tuple[Tuple[str, str, str, str], ...] = (
    ("recall",                      "recall",              "higher", "{:.4f}"),
    ("prediction_filled_ratio",     "fill_ratio",          "higher", "{:.4f}"),
    ("mean_error_px",               "mean_err_px",         "lower",  "{:.2f}"),
    ("median_error_px",             "median_err_px",       "lower",  "{:.2f}"),
    ("p90_error_px",                "p90_err_px",          "lower",  "{:.2f}"),
    ("max_error_px",                "max_err_px",          "lower",  "{:.2f}"),
    ("within_5px",                  "within_5px",          "higher", "{:d}"),
    ("within_10px",                 "within_10px",         "higher", "{:d}"),
    ("within_20px",                 "within_20px",         "higher", "{:d}"),
    ("large_jump_count",            "large_jumps",         "lower",  "{:d}"),
    ("false_positive_absent_frames","FP_on_absent",        "lower",  "{:d}"),
    ("matched_visible_frames",      "matched_labels",      "higher", "{:d}"),
    ("missed_visible_frames",       "missed_labels",       "lower",  "{:d}"),
)


def _load(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _summary(report: Dict[str, Any]) -> Dict[str, Any]:
    s = report.get("summary")
    if not isinstance(s, dict):
        raise ValueError("report has no 'summary' object — is this a validation report?")
    return s


def _fmt(value: Any, spec: str) -> str:
    if value is None:
        return "n/a"
    try:
        if spec.endswith("d"):
            return spec.format(int(value))
        return spec.format(float(value))
    except (ValueError, TypeError):
        return str(value)


def _delta_marker(before: Any, after: Any, direction: str) -> Tuple[str, str]:
    if before is None or after is None:
        return "n/a", " "
    try:
        b = float(before)
        a = float(after)
    except (ValueError, TypeError):
        return "n/a", " "
    diff = a - b
    if diff == 0.0:
        return f"{diff:+.4g}", "="
    if direction == "higher":
        return f"{diff:+.4g}", ("v" if diff > 0 else "^")
    return f"{diff:+.4g}", ("v" if diff < 0 else "^")


def _print_metric_table(b: Dict[str, Any], a: Dict[str, Any]) -> int:
    name_w = max(len(label) for _, label, _, _ in METRIC_ROWS)
    print(f"{'metric'.ljust(name_w)}   {'before':>14}   {'after':>14}   {'delta':>12}  dir  better?")
    print("-" * (name_w + 60))
    regressions = 0
    for key, label, direction, spec in METRIC_ROWS:
        bv = b.get(key)
        av = a.get(key)
        delta_str, arrow = _delta_marker(bv, av, direction)
        better = " "
        if arrow == "v":
            better = "WORSE"
            regressions += 1
        elif arrow == "^":
            better = "better"
        elif arrow == "=":
            better = "same"
        print(
            f"{label.ljust(name_w)}   "
            f"{_fmt(bv, spec):>14}   "
            f"{_fmt(av, spec):>14}   "
            f"{delta_str:>12}  "
            f"{direction[:3]:>3}  {better}"
        )
    return regressions


def _print_source_counts(b: Dict[str, Any], a: Dict[str, Any]) -> None:
    b_src = b.get("source_counts") or {}
    a_src = a.get("source_counts") or {}
    keys = sorted(set(list(b_src.keys()) + list(a_src.keys())))
    if not keys:
        return
    print()
    print("source mix (counts):")
    name_w = max(len(k) for k in keys)
    for k in keys:
        bv = int(b_src.get(k, 0))
        av = int(a_src.get(k, 0))
        diff = av - bv
        sign = "+" if diff > 0 else ""
        print(f"  {k.ljust(name_w)}   before={bv:>5}   after={av:>5}   delta={sign}{diff}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Diff two validation reports.")
    parser.add_argument("--before", required=True, help="Baseline report JSON")
    parser.add_argument("--after", required=True, help="Candidate report JSON")
    parser.add_argument("--fail-on-regression", action="store_true",
                        help="Exit non-zero if any tracked metric regressed.")
    args = parser.parse_args()

    before_path = Path(args.before)
    after_path = Path(args.after)
    for label, p in (("before", before_path), ("after", after_path)):
        if not p.exists():
            print(f"[compare] {label} report not found: {p}", file=sys.stderr)
            return 2

    before = _summary(_load(before_path))
    after = _summary(_load(after_path))

    print(f"before: {before_path}")
    print(f"after:  {after_path}")
    print()
    regressions = _print_metric_table(before, after)
    _print_source_counts(before, after)
    print()
    if regressions:
        print(f"[compare] {regressions} metric(s) regressed.")
    else:
        print("[compare] no tracked-metric regressions.")

    if args.fail_on_regression and regressions:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
