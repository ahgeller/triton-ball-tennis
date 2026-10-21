"""Validate exported tennis tracking JSON against hand-labeled frames.

Errors are native pixels and the summary's numbers are what ``check_parity`` freezes, so this
scorer stays as it is; the detector metric lives in ``finetune.data.detector_metrics``.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, TypedDict, TypeGuard, Union

if TYPE_CHECKING:
    # The tracking JSON's shape is defined where it is written.  Imported for the type checker
    # only, so this scorer stays importable without the tracker's CUDA and TensorRT stack.
    from tennis_tracker.export import FrameRow, SummaryJson, TrackingJson


class _LabelAlways(TypedDict):
    frame: int


class LabelRow(_LabelAlways, total=False):
    """One hand label: ``frame`` always, ``x``/``y`` when the ball was marked, ``visible`` False
    for a labelled absence (True when the key is missing)."""

    x: Optional[float]
    y: Optional[float]
    visible: bool


class _VisibleLabelAlways(_LabelAlways):
    x: float
    y: float


class VisibleLabel(_VisibleLabelAlways, total=False):
    """A ``LabelRow`` that marks the ball: ``_has_position`` tells them apart."""

    visible: bool


# A frame range the scorer skips, as ``start``/``end`` or ``from``/``to`` (inclusive).
IgnoreRange = TypedDict("IgnoreRange", {"start": int, "end": int, "from": int, "to": int}, total=False)
LabelRows = Union[List[LabelRow], Dict[str, LabelRow]]     # a list, or keyed by frame number


class AnnotationJson(TypedDict, total=False):
    """The annotation file: the labels under ``ball`` (or ``frames``) and the ranges to skip."""

    ball: LabelRows
    frames: LabelRows
    ignore_ranges: List[IgnoreRange]


class LargeJump(TypedDict):
    from_frame: int
    to_frame: int
    gap: int
    distance_px: float


class MatchPrediction(TypedDict):
    x: float
    y: float
    source: Optional[str]


class MatchLabel(TypedDict):
    x: float
    y: float


class Match(TypedDict):
    """A visible label with a prediction on its frame, and how far apart they are."""

    frame: int
    error_px: float
    pred: MatchPrediction
    label: MatchLabel


class ErrorStatistics(TypedDict):
    """The summary's error figures: None where there was no match to measure."""

    mean_error_px: Optional[float]
    median_error_px: Optional[float]
    p90_error_px: Optional[float]
    max_error_px: Optional[float]
    within_5px: int
    within_10px: int
    within_20px: int


class ValidationSummary(ErrorStatistics):
    """The figures ``check_parity`` freezes (it adds a ``smoothness`` block of its own)."""

    prediction_file_summary: SummaryJson
    labeled_visible_frames: int
    matched_visible_frames: int
    missed_visible_frames: int
    recall: float
    presence_recall: float
    localization_recall_5px: float
    localization_recall_10px: float
    localization_recall_20px: float
    labeled_absent_frames: int
    false_positive_absent_frames: int
    absence_precision: Optional[float]
    absence_specificity: Optional[float]
    prediction_present_frames: int
    prediction_total_frames: int
    prediction_filled_ratio: float
    source_counts: Dict[str, int]
    large_jump_count: int


class ValidationReport(TypedDict):
    summary: ValidationSummary
    miss_frames: List[int]
    false_positive_frames: List[int]
    large_jumps: List[LargeJump]
    matches: List[Match]


def _load_json(path: Path) -> Any:
    """The file's JSON as it is on disk; ``validate`` says what shape it expects."""
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


def _is_ignored(frame: int, ranges: Iterable[IgnoreRange]) -> bool:
    for item in ranges:
        start = int(item.get("start", item.get("from", -1)))
        end = int(item.get("end", item.get("to", start)))
        if start <= frame <= end:
            return True
    return False


def _prediction_frames(payload: TrackingJson) -> Dict[int, FrameRow]:
    frames = payload.get("frames", [])
    out: Dict[int, FrameRow] = {}
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


def _prediction_present(row: Optional[FrameRow]) -> TypeGuard[FrameRow]:
    """Whether the row holds a ball position (``present`` with ``x`` and ``y``)."""
    if not row:
        return False
    return bool(row.get("present", False)) and row.get("x") is not None and row.get("y") is not None


def _has_position(row: LabelRow) -> TypeGuard[VisibleLabel]:
    return row.get("x") is not None and row.get("y") is not None


def _annotation_rows(payload: AnnotationJson) -> List[LabelRow]:
    rows = payload.get("ball", payload.get("frames", []))
    out: List[LabelRow] = []
    if isinstance(rows, dict):
        for frame_s, value in rows.items():
            row: LabelRow = {**value, "frame": int(frame_s)} if value else {"frame": int(frame_s)}
            out.append(row)
    elif isinstance(rows, list):
        out.extend(row for row in rows if isinstance(row, dict))
    else:
        raise ValueError("Annotation JSON must contain a 'ball' or 'frames' list/dict")
    return out


def _source_counts(pred_frames: Dict[int, FrameRow]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in pred_frames.values():
        if not _prediction_present(row):
            continue
        source = str(row.get("source", "unknown") or "unknown")
        counts[source] = counts.get(source, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def _large_jumps(
    pred_frames: Dict[int, FrameRow],
    jump_px: float,
    max_gap: int,
) -> List[LargeJump]:
    jumps: List[LargeJump] = []
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


def _label_rows(annotations: AnnotationJson) -> Tuple[List[VisibleLabel], List[LabelRow]]:
    """(the visible labels with a position, the absent labels), ignored ranges left out."""
    ignore_ranges = annotations.get("ignore_ranges", [])
    visible_labels: List[VisibleLabel] = []
    absent_labels: List[LabelRow] = []
    for row in _annotation_rows(annotations):
        frame = int(row["frame"])
        if _is_ignored(frame, ignore_ranges):
            continue
        visible = bool(row.get("visible", True))
        if visible and _has_position(row):
            visible_labels.append(row)
        elif not visible:
            absent_labels.append(row)
    return visible_labels, absent_labels


def _match_labels(pred_frames: Dict[int, FrameRow],
                  visible_labels: List[VisibleLabel]) -> Tuple[List[float], List[int], List[Match]]:
    """(the error of every visible label with a prediction, the frames without one, the matches)."""
    errors: List[float] = []
    misses: List[int] = []
    matches: List[Match] = []
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
    return errors, misses, matches


def _error_statistics(errors: List[float]) -> ErrorStatistics:
    return {
        "mean_error_px": statistics.fmean(errors) if errors else None,
        "median_error_px": statistics.median(errors) if errors else None,
        "p90_error_px": _percentile(errors, 90.0),
        "max_error_px": max(errors) if errors else None,
        "within_5px": sum(1 for e in errors if e <= 5.0),
        "within_10px": sum(1 for e in errors if e <= 10.0),
        "within_20px": sum(1 for e in errors if e <= 20.0),
    }


def validate(
    predictions: TrackingJson,
    annotations: AnnotationJson,
    jump_px: float,
    jump_max_gap: int,
) -> ValidationReport:
    pred_frames = _prediction_frames(predictions)
    visible_labels, absent_labels = _label_rows(annotations)
    errors, misses, matches = _match_labels(pred_frames, visible_labels)
    false_positive_frames = [int(label["frame"]) for label in absent_labels
                             if _prediction_present(pred_frames.get(int(label["frame"])))]
    large_jumps = _large_jumps(pred_frames, jump_px=jump_px, max_gap=jump_max_gap)

    pred_present = sum(1 for row in pred_frames.values() if _prediction_present(row))
    recall = len(errors) / max(len(visible_labels), 1)
    precision_on_labeled_absence = None
    if absent_labels:
        precision_on_labeled_absence = 1.0 - (
            len(false_positive_frames) / max(len(absent_labels), 1)
        )
    summary: ValidationSummary = {
        "prediction_file_summary": predictions.get("summary", {}),
        "labeled_visible_frames": len(visible_labels),
        "matched_visible_frames": len(errors),
        "missed_visible_frames": len(misses),
        "recall": recall,
        "presence_recall": recall,
        "localization_recall_5px": sum(e <= 5.0 for e in errors) / max(len(visible_labels), 1),
        "localization_recall_10px": sum(e <= 10.0 for e in errors) / max(len(visible_labels), 1),
        "localization_recall_20px": sum(e <= 20.0 for e in errors) / max(len(visible_labels), 1),
        "labeled_absent_frames": len(absent_labels),
        "false_positive_absent_frames": len(false_positive_frames),
        "absence_precision": precision_on_labeled_absence,
        "absence_specificity": precision_on_labeled_absence,
        "prediction_present_frames": pred_present,
        "prediction_total_frames": len(pred_frames),
        "prediction_filled_ratio": pred_present / max(len(pred_frames), 1),
        **_error_statistics(errors),
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


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate tracking JSON against labels.")
    parser.add_argument("--predictions", required=True, help="Path from now_main.py --tracking-json")
    parser.add_argument("--annotations", required=True, help="Hand-labeled annotation JSON")
    parser.add_argument("--report-json", default=None, help="Optional path for full validation report")
    parser.add_argument("--jump-px", type=float, default=120.0, help="Large jump threshold in pixels")
    parser.add_argument("--jump-max-gap", type=int, default=1, help="Only count jumps across gaps up to this many frames")
    parser.add_argument("--max-mean-error", type=float, default=None, help="Fail if mean error exceeds this")
    parser.add_argument("--max-p90-error", type=float, default=None, help="Fail if p90 error exceeds this")
    parser.add_argument("--min-recall", type=float, default=None, help="Fail if localization recall within 10px is below this 0..1 value")
    parser.add_argument("--min-presence-recall", type=float, default=None, help="Optional presence-only gate (does not check localization)")
    args = parser.parse_args(argv)

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
    for line in summary_lines(summary):
        print(line)
    return 2 if _gates_failed(args, summary) else 0


def summary_lines(summary: ValidationSummary) -> List[str]:
    """The console report of a validation summary, one ``[validation]`` line per figure."""
    lines = [
        "[validation] labels matched/missed: "
        f"{summary['matched_visible_frames']}/{summary['missed_visible_frames']}",
        "[validation] recall: "
        f"{summary['recall']:.3f}",
        "[validation] error px: "
        f"mean={_fmt_optional(summary['mean_error_px'])} "
        f"median={_fmt_optional(summary['median_error_px'])} "
        f"p90={_fmt_optional(summary['p90_error_px'])} "
        f"max={_fmt_optional(summary['max_error_px'])}",
        "[validation] prediction filled ratio: "
        f"{summary['prediction_filled_ratio']:.3f}",
        "[validation] sources: "
        f"{json.dumps(summary['source_counts'], sort_keys=True)}",
        "[validation] large jumps: "
        f"{summary['large_jump_count']}",
    ]
    if summary["labeled_absent_frames"]:
        lines.append("[validation] absence false positives: "
                     f"{summary['false_positive_absent_frames']}/{summary['labeled_absent_frames']}")
    return lines


def _gates_failed(args: argparse.Namespace, summary: ValidationSummary) -> bool:
    """Whether any threshold given on the command line is missed (a missing number fails it)."""
    failed = False
    if args.max_mean_error is not None:
        mean_error = summary["mean_error_px"]
        failed = failed or mean_error is None or mean_error > float(args.max_mean_error)
    if args.max_p90_error is not None:
        p90_error = summary["p90_error_px"]
        failed = failed or p90_error is None or p90_error > float(args.max_p90_error)
    if args.min_recall is not None:
        failed = failed or float(summary["localization_recall_10px"]) < float(args.min_recall)
    if args.min_presence_recall is not None:
        failed = failed or float(summary["presence_recall"]) < float(args.min_presence_recall)
    return failed


if __name__ == "__main__":
    raise SystemExit(main())
