#!/usr/bin/env python3
"""Validate exported tennis tracking JSON against hand-labeled frames."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _percentile(values: List[float], pct: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * max(0.0, min(100.0, pct)) / 100.0
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return ordered[lo]
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _is_ignored(frame: int, ranges: Iterable[Dict[str, Any]]) -> bool:
    for item in ranges:
        start = int(item.get("start", item.get("from", -1)))
        end = int(item.get("end", item.get("to", start)))
        if start <= frame <= end:
            return True
    return False


def _prediction_frames(payload: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    frames = payload.get("frames", [])
    out: Dict[int, Dict[str, Any]] = {}
    if isinstance(frames, list):
        for idx, row in enumerate(frames):
            if row is None:
                out[idx] = {"frame": idx, "present": False}
                continue
            if isinstance(row, dict):
                frame = int(row.get("frame", idx))
                out[frame] = row
    elif isinstance(frames, dict):
        for key, row in frames.items():
            frame = int(key)
            out[frame] = row if isinstance(row, dict) else {"frame": frame, "present": False}
    return out


def _prediction_present(row: Optional[Dict[str, Any]]) -> bool:
    if not row:
        return False
    return bool(row.get("present", False)) and row.get("x") is not None and row.get("y") is not None


def _annotation_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = payload.get("ball", payload.get("frames", []))
    out: List[Dict[str, Any]] = []
    if isinstance(rows, dict):
        for frame_s, value in rows.items():
            row = dict(value or {})
            row["frame"] = int(frame_s)
            out.append(row)
    elif isinstance(rows, list):
        out.extend(row for row in rows if isinstance(row, dict))
    else:
        raise ValueError("Annotation JSON must contain a 'ball' or 'frames' list/dict")
    return out


def _source_counts(pred_frames: Dict[int, Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in pred_frames.values():
        if not _prediction_present(row):
            continue
        source = str(row.get("source", "unknown") or "unknown")
        counts[source] = counts.get(source, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def _large_jumps(
    pred_frames: Dict[int, Dict[str, Any]],
    jump_px: float,
    max_gap: int,
) -> List[Dict[str, Any]]:
    jumps: List[Dict[str, Any]] = []
    last: Optional[Tuple[int, float, float]] = None
    for frame in sorted(pred_frames):
        row = pred_frames[frame]
        if not _prediction_present(row):
            last = None
            continue
        x = float(row["x"])
        y = float(row["y"])
        if last is not None:
            prev_frame, prev_x, prev_y = last
            gap = frame - prev_frame
            if 0 < gap <= max_gap:
                dist = math.hypot(x - prev_x, y - prev_y)
                if dist > jump_px:
                    jumps.append({
                        "from_frame": prev_frame,
                        "to_frame": frame,
                        "gap": gap,
                        "distance_px": dist,
                    })
        last = (frame, x, y)
    return jumps


def validate(
    predictions: Dict[str, Any],
    annotations: Dict[str, Any],
    jump_px: float,
    jump_max_gap: int,
) -> Dict[str, Any]:
    pred_frames = _prediction_frames(predictions)
    ignore_ranges = annotations.get("ignore_ranges", [])
    visible_labels = []
    absent_labels = []
    for row in _annotation_rows(annotations):
        frame = int(row["frame"])
        if _is_ignored(frame, ignore_ranges):
            continue
        visible = bool(row.get("visible", True))
        has_xy = row.get("x") is not None and row.get("y") is not None
        if visible and has_xy:
            visible_labels.append(row)
        elif not visible:
            absent_labels.append(row)

    errors: List[float] = []
    misses: List[int] = []
    matches: List[Dict[str, Any]] = []
    for label in visible_labels:
        frame = int(label["frame"])
        pred = pred_frames.get(frame)
        if not _prediction_present(pred):
            misses.append(frame)
            continue
        dx = float(pred["x"]) - float(label["x"])
        dy = float(pred["y"]) - float(label["y"])
        err = math.hypot(dx, dy)
        errors.append(err)
        matches.append({
            "frame": frame,
            "error_px": err,
            "pred": {"x": float(pred["x"]), "y": float(pred["y"]), "source": pred.get("source")},
            "label": {"x": float(label["x"]), "y": float(label["y"])},
        })

    false_positive_frames = []
    for label in absent_labels:
        frame = int(label["frame"])
        if _prediction_present(pred_frames.get(frame)):
            false_positive_frames.append(frame)

    pred_present = sum(1 for row in pred_frames.values() if _prediction_present(row))
    total_prediction_frames = len(pred_frames)
    recall = len(errors) / max(len(visible_labels), 1)
    precision_on_labeled_absence = None
    if absent_labels:
        precision_on_labeled_absence = 1.0 - (
            len(false_positive_frames) / max(len(absent_labels), 1)
        )

    large_jumps = _large_jumps(pred_frames, jump_px=jump_px, max_gap=jump_max_gap)
    summary = {
        "prediction_file_summary": predictions.get("summary", {}),
        "labeled_visible_frames": len(visible_labels),
        "matched_visible_frames": len(errors),
        "missed_visible_frames": len(misses),
        "recall": recall,
        "labeled_absent_frames": len(absent_labels),
        "false_positive_absent_frames": len(false_positive_frames),
        "absence_precision": precision_on_labeled_absence,
        "prediction_present_frames": pred_present,
        "prediction_total_frames": total_prediction_frames,
        "prediction_filled_ratio": pred_present / max(total_prediction_frames, 1),
        "mean_error_px": statistics.fmean(errors) if errors else None,
        "median_error_px": statistics.median(errors) if errors else None,
        "p90_error_px": _percentile(errors, 90.0),
        "max_error_px": max(errors) if errors else None,
        "within_5px": sum(1 for e in errors if e <= 5.0),
        "within_10px": sum(1 for e in errors if e <= 10.0),
        "within_20px": sum(1 for e in errors if e <= 20.0),
        "source_counts": _source_counts(pred_frames),
        "large_jump_count": len(large_jumps),
    }
    return {
        "summary": summary,
        "miss_frames": misses,
        "false_positive_frames": false_positive_frames,
        "large_jumps": large_jumps,
        "matches": matches,
    }


def _fmt_optional(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate tracking JSON against labels.")
    parser.add_argument("--predictions", required=True, help="Path from now_main.py --tracking-json")
    parser.add_argument("--annotations", required=True, help="Hand-labeled annotation JSON")
    parser.add_argument("--report-json", default=None, help="Optional path for full validation report")
    parser.add_argument("--jump-px", type=float, default=120.0, help="Large jump threshold in pixels")
    parser.add_argument("--jump-max-gap", type=int, default=1, help="Only count jumps across gaps up to this many frames")
    parser.add_argument("--max-mean-error", type=float, default=None, help="Fail if mean error exceeds this")
    parser.add_argument("--max-p90-error", type=float, default=None, help="Fail if p90 error exceeds this")
    parser.add_argument("--min-recall", type=float, default=None, help="Fail if recall is below this 0..1 value")
    args = parser.parse_args()

    predictions = _load_json(Path(args.predictions))
    annotations = _load_json(Path(args.annotations))
    report = validate(
        predictions,
        annotations,
        jump_px=float(args.jump_px),
        jump_max_gap=max(1, int(args.jump_max_gap)),
    )

    if args.report_json:
        out_path = Path(args.report_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    summary = report["summary"]
    print("[validation] labels matched/missed: "
          f"{summary['matched_visible_frames']}/{summary['missed_visible_frames']}")
    print("[validation] recall: "
          f"{summary['recall']:.3f}")
    print("[validation] error px: "
          f"mean={_fmt_optional(summary['mean_error_px'])} "
          f"median={_fmt_optional(summary['median_error_px'])} "
          f"p90={_fmt_optional(summary['p90_error_px'])} "
          f"max={_fmt_optional(summary['max_error_px'])}")
    print("[validation] prediction filled ratio: "
          f"{summary['prediction_filled_ratio']:.3f}")
    print("[validation] sources: "
          f"{json.dumps(summary['source_counts'], sort_keys=True)}")
    print("[validation] large jumps: "
          f"{summary['large_jump_count']}")
    if summary["labeled_absent_frames"]:
        print("[validation] absence false positives: "
              f"{summary['false_positive_absent_frames']}/{summary['labeled_absent_frames']}")

    failed = False
    if args.max_mean_error is not None:
        mean_error = summary["mean_error_px"]
        failed = failed or mean_error is None or mean_error > float(args.max_mean_error)
    if args.max_p90_error is not None:
        p90_error = summary["p90_error_px"]
        failed = failed or p90_error is None or p90_error > float(args.max_p90_error)
    if args.min_recall is not None:
        failed = failed or float(summary["recall"]) < float(args.min_recall)

    return 2 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
