#!/usr/bin/env python3
"""Multi-clip tracking benchmark: the referee for every "faster" or "better" claim.

Runs the full pipeline -> validate loop for every enabled clip in eval/clips.json
(reusing 3DtrackingV1/archived_tools/run_validation.py per clip, so path/report
conventions stay identical), then prints one scoreboard and writes an aggregate
result JSON tagged with the git SHA.

Usage:
    python eval/benchmark.py                          # run everything
    python eval/benchmark.py --clips pomona_baseline  # subset
    python eval/benchmark.py --skip-pipeline          # re-score existing tracking JSONs
    python eval/benchmark.py --compare eval/results/benchmark__abc1234.json
    python eval/benchmark.py --compare-latest --fail-on-regression

Exit codes: 0 ok, 1 threshold/regression failure, 2 setup error.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "eval" / "results"
RUN_VALIDATION = REPO_ROOT / "3DtrackingV1" / "archived_tools" / "run_validation.py"

# (report key, short label, format, direction: higher/lower is better)
SCORE_COLUMNS = [
    ("recall", "recall", "{:.3f}", "higher"),
    ("mean_error_px", "mean_px", "{:.1f}", "lower"),
    ("p90_error_px", "p90_px", "{:.1f}", "lower"),
    ("within_10px", "<=10px", "{:d}", "higher"),
    ("false_positive_absent_frames", "fp_abs", "{:d}", "lower"),
    ("large_jump_count", "jumps", "{:d}", "lower"),
    ("prediction_filled_ratio", "fill", "{:.2f}", "neutral"),
    ("effective_fps", "fps", "{:.1f}", "higher"),
]


def _git_short_sha() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip() or "nogit"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"
    try:
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        dirty = ""
    return f"{sha}-dirty" if dirty else sha


def _load_registry(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    clips = data.get("clips") or []
    return [c for c in clips if c.get("enabled", True)]


def _flat_summary(report: Dict[str, Any]) -> Dict[str, Any]:
    """Pull the metrics we score from a validate_tracking report."""
    summary = dict(report.get("summary") or {})
    pred = summary.get("prediction_file_summary") or {}
    summary["effective_fps"] = pred.get("effective_fps")
    return summary


def _run_clip(clip: Dict[str, Any], skip_pipeline: bool) -> Dict[str, Any]:
    name = str(clip["name"])
    cmd = [sys.executable, str(RUN_VALIDATION), "--clip", name]
    if skip_pipeline:
        cmd.append("--skip-pipeline")
    thresholds = clip.get("thresholds") or {}
    for arg, key in (
        ("--min-recall", "min_recall"),
        ("--max-mean-error", "max_mean_error"),
        ("--max-p90-error", "max_p90_error"),
    ):
        if thresholds.get(key) is not None:
            cmd.extend([arg, str(thresholds[key])])
    extra = str(clip.get("extra_now_args") or "").strip()
    if extra and not skip_pipeline:
        # '=' form: argparse rejects values that start with '-' as separate tokens.
        cmd.append(f"--extra-now-args={extra}")

    print(f"\n[benchmark] === {name} ===", flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=REPO_ROOT).returncode
    wall = time.time() - t0

    sha = _git_short_sha()
    report_path = REPO_ROOT / "validation" / "reports" / f"{name}__{sha}.json"
    row: Dict[str, Any] = {
        "clip": name,
        "returncode": int(rc),
        "wall_sec": float(wall),
        "report_path": str(report_path.relative_to(REPO_ROOT)),
        "thresholds": thresholds,
        "summary": None,
    }
    if report_path.exists():
        with report_path.open("r", encoding="utf-8") as f:
            row["summary"] = _flat_summary(json.load(f))
    else:
        print(f"[benchmark] WARNING: report missing for {name}: {report_path}")
    return row


def _fmt_cell(value: Any, spec: str) -> str:
    if value is None:
        return "n/a"
    try:
        if "d" in spec:
            return spec.format(int(value))
        return spec.format(float(value))
    except (TypeError, ValueError):
        return str(value)


def _print_scoreboard(rows: List[Dict[str, Any]]) -> None:
    name_w = max([len(r["clip"]) for r in rows] + [len("clip")])
    header = "clip".ljust(name_w) + "  pass  " + "  ".join(
        f"{label:>8}" for _, label, _, _ in SCORE_COLUMNS
    )
    print("\n" + header)
    print("-" * len(header))
    for row in rows:
        summary = row.get("summary") or {}
        status = "OK" if row["returncode"] == 0 else "FAIL"
        cells = "  ".join(
            f"{_fmt_cell(summary.get(key), spec):>8}"
            for key, _, spec, _ in SCORE_COLUMNS
        )
        print(f"{row['clip'].ljust(name_w)}  {status:>4}  {cells}")


def _print_comparison(before: Dict[str, Any], after_rows: List[Dict[str, Any]]) -> int:
    before_by_clip = {r["clip"]: r for r in before.get("clips") or []}
    regressions = 0
    for row in after_rows:
        prev = before_by_clip.get(row["clip"])
        if prev is None or not prev.get("summary") or not row.get("summary"):
            continue
        b, a = prev["summary"], row["summary"]
        print(f"\n[compare] {row['clip']} (vs {before.get('git_sha', '?')})")
        for key, label, spec, direction in SCORE_COLUMNS:
            bv, av = b.get(key), a.get(key)
            if bv is None or av is None:
                continue
            delta = float(av) - float(bv)
            marker = "same"
            if abs(delta) > 1e-9 and direction != "neutral":
                improved = (delta > 0) == (direction == "higher")
                marker = "better" if improved else "WORSE"
                if not improved:
                    regressions += 1
            print(f"  {label:>8}: {_fmt_cell(bv, spec):>8} -> {_fmt_cell(av, spec):>8}  {marker}")
    return regressions


def _latest_result(exclude: Path) -> Optional[Path]:
    if not RESULTS_DIR.exists():
        return None
    candidates = sorted(
        (p for p in RESULTS_DIR.glob("benchmark__*.json") if p != exclude),
        key=lambda p: p.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the multi-clip tracking benchmark.")
    parser.add_argument("--registry", default=str(REPO_ROOT / "eval" / "clips.json"))
    parser.add_argument("--clips", default=None,
                        help="Comma-separated subset of clip names to run.")
    parser.add_argument("--skip-pipeline", action="store_true",
                        help="Re-validate existing tracking JSONs without re-running now_main.py.")
    parser.add_argument("--compare", default=None,
                        help="Path to a previous eval/results/benchmark__*.json to diff against.")
    parser.add_argument("--compare-latest", action="store_true",
                        help="Diff against the most recent previous benchmark result.")
    parser.add_argument("--fail-on-regression", action="store_true",
                        help="Exit 1 if the comparison shows any metric getting worse.")
    args = parser.parse_args()

    registry_path = Path(args.registry)
    if not registry_path.exists():
        print(f"[benchmark] registry not found: {registry_path}", file=sys.stderr)
        return 2
    clips = _load_registry(registry_path)
    if args.clips:
        wanted = {c.strip() for c in args.clips.split(",") if c.strip()}
        clips = [c for c in clips if c["name"] in wanted]
        missing = wanted - {c["name"] for c in clips}
        if missing:
            print(f"[benchmark] unknown/disabled clips: {sorted(missing)}", file=sys.stderr)
            return 2
    if not clips:
        print("[benchmark] no enabled clips in registry.", file=sys.stderr)
        return 2

    rows = [_run_clip(clip, args.skip_pipeline) for clip in clips]
    _print_scoreboard(rows)

    sha = _git_short_sha()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    generated_at = time.gmtime()
    generated_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", generated_at)
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", generated_at)
    out_path = RESULTS_DIR / f"benchmark__{sha}__{timestamp}.json"
    payload = {
        "schema_version": 1,
        "git_sha": sha,
        "generated_utc": generated_utc,
        "skip_pipeline": bool(args.skip_pipeline),
        "clips": rows,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[benchmark] wrote {out_path}")

    regressions = 0
    compare_path: Optional[Path] = Path(args.compare) if args.compare else None
    if args.compare_latest and compare_path is None:
        compare_path = _latest_result(exclude=out_path)
        if compare_path is None:
            print("[benchmark] --compare-latest: no previous result found; skipping diff.")
    if compare_path is not None:
        if not compare_path.exists():
            print(f"[benchmark] compare file not found: {compare_path}", file=sys.stderr)
            return 2
        with compare_path.open("r", encoding="utf-8") as f:
            regressions = _print_comparison(json.load(f), rows)
        if regressions:
            print(f"\n[benchmark] {regressions} metric(s) regressed.")

    threshold_failures = sum(1 for r in rows if r["returncode"] != 0)
    if threshold_failures:
        print(f"[benchmark] {threshold_failures} clip(s) failed thresholds.")
        return 1
    if args.fail_on_regression and regressions:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
