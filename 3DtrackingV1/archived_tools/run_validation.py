#!/usr/bin/env python3
"""One-command validation loop: pipeline -> tracking.json -> validator -> report.

Resolves paths by convention from --clip <name>:
  annotations: validation/annotations/<clip>.json
  video:       read from the "video" field in the annotation JSON
  tracking:    output_videos/<clip>__<git_sha>_tracking.json
  report:      validation/reports/<clip>__<git_sha>.json

Re-uses the archived validate_tracking.py via subprocess; no duplicate metric logic.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[2]


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


def _load_annotation(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _run(cmd: List[str], label: str) -> int:
    print(f"[run_validation] {label}: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    return int(proc.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full validation loop for a clip.")
    parser.add_argument("--clip", required=True,
                        help="Clip name. Expects validation/annotations/<clip>.json.")
    parser.add_argument("--skip-pipeline", action="store_true",
                        help="Re-validate an existing tracking JSON without re-running now_main.py.")
    parser.add_argument("--max-mean-error", type=float, default=None,
                        help="Forward to validate_tracking.py")
    parser.add_argument("--max-p90-error", type=float, default=None,
                        help="Forward to validate_tracking.py")
    parser.add_argument("--min-recall", type=float, default=None,
                        help="Forward to validate_tracking.py")
    parser.add_argument("--jump-px", type=float, default=120.0,
                        help="Forward to validate_tracking.py")
    parser.add_argument("--extra-now-args", default="",
                        help="Space-separated extra args appended to now_main.py invocation.")
    args = parser.parse_args()

    ann_path = REPO_ROOT / "validation" / "annotations" / f"{args.clip}.json"
    if not ann_path.exists():
        print(f"[run_validation] missing annotation file: {ann_path}", file=sys.stderr)
        print(f"[run_validation] copy validation/annotations/pomona_baseline.template.json"
              f" -> validation/annotations/{args.clip}.json and label it first.", file=sys.stderr)
        return 2

    annotation = _load_annotation(ann_path)
    video_rel = annotation.get("video")
    if not video_rel:
        print(f"[run_validation] annotation {ann_path} has no 'video' field.", file=sys.stderr)
        return 2
    video_path = (REPO_ROOT / video_rel).resolve()
    if not video_path.exists():
        print(f"[run_validation] video not found: {video_path}", file=sys.stderr)
        return 2

    sha = _git_short_sha()
    tag = f"{args.clip}__{sha}"
    out_json = REPO_ROOT / "output_videos" / f"{tag}_tracking.json"
    out_report = REPO_ROOT / "validation" / "reports" / f"{tag}.json"

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)

    if not args.skip_pipeline:
        now_cmd = [
            sys.executable, "now_main.py",
            "--input", str(video_path),
            "--outputs", "none",
            "--tracking-json", str(out_json),
        ]
        if args.extra_now_args.strip():
            now_cmd.extend(args.extra_now_args.split())
        rc = _run(now_cmd, "pipeline")
        if rc != 0:
            print(f"[run_validation] now_main.py exited with {rc}", file=sys.stderr)
            return rc
    else:
        if not out_json.exists():
            print(f"[run_validation] --skip-pipeline but tracking JSON missing: {out_json}",
                  file=sys.stderr)
            return 2

    val_cmd: List[str] = [
        sys.executable, "3DtrackingV1/archived_tools/validate_tracking.py",
        "--predictions", str(out_json),
        "--annotations", str(ann_path),
        "--report-json", str(out_report),
        "--jump-px", str(args.jump_px),
    ]
    if args.max_mean_error is not None:
        val_cmd.extend(["--max-mean-error", str(args.max_mean_error)])
    if args.max_p90_error is not None:
        val_cmd.extend(["--max-p90-error", str(args.max_p90_error)])
    if args.min_recall is not None:
        val_cmd.extend(["--min-recall", str(args.min_recall)])

    rc = _run(val_cmd, "validate")
    print(f"[run_validation] report: {out_report}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
