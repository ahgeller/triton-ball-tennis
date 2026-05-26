"""Run physics-first 3D trajectory/event reconstruction from tracking JSON."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from trajectory3d_v1 import reconstruct_from_files, render_debug_video


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reconstruct v1 3D ball trajectory/events from tracking.json + LosslessCut .llc cuts."
    )
    parser.add_argument("--tracking-json", required=True, help="Path to now_main.py --tracking-json output")
    parser.add_argument("--cuts", default=None, help="Path to LosslessCut .llc JSON5 project file")
    parser.add_argument("--output-json", required=True, help="Path to write trajectory3d_v1 JSON")
    parser.add_argument(
        "--render-video",
        default=None,
        help="Optional debug MP4 path with projected 3D trajectory overlay",
    )
    parser.add_argument(
        "--input-video",
        default=None,
        help="Optional input video override for --render-video; defaults to tracking JSON video.input",
    )
    parser.add_argument(
        "--timebase",
        choices=("auto", "merged", "original"),
        default="auto",
        help="How to interpret .llc times relative to tracking video (default: auto)",
    )
    parser.add_argument(
        "--max-hidden-gap-sec",
        type=float,
        default=1.5,
        help="Max gap to bridge as inferred hidden/out-of-frame arc (default: 1.5)",
    )
    args = parser.parse_args()

    result = reconstruct_from_files(
        args.tracking_json,
        args.cuts,
        args.output_json,
        timebase=args.timebase,
        max_hidden_gap_sec=args.max_hidden_gap_sec,
    )
    summary = result.get("summary", {})
    mapping = result.get("cut_mapping", {})
    print(
        "[trajectory3d_v1] "
        f"segments={summary.get('segment_count')} events={summary.get('event_count')} "
        f"timebase={mapping.get('mode')} drift={mapping.get('duration_drift_sec')}"
    )
    for warning in mapping.get("warnings", []) or []:
        print(f"[trajectory3d_v1][warn] {warning}")

    if args.render_video:
        render_debug_video(
            args.tracking_json,
            args.output_json,
            args.render_video,
            input_video=args.input_video,
        )
        print(f"[trajectory3d_v1] Debug video: {args.render_video}")
    print(f"[trajectory3d_v1] Output JSON: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
