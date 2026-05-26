"""Run hybrid v2 3D trajectory/event reconstruction from tracking JSON."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from trajectory3d_v1 import render_debug_video
from trajectory3d_v2 import reconstruct_from_files_v2


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reconstruct v2 calibrated/hypothesis-tested 3D ball trajectory from tracking JSON."
    )
    parser.add_argument("--tracking-json", required=True, help="Path to now_main.py --tracking-json output")
    parser.add_argument("--cuts", default=None, help="Optional LosslessCut .llc JSON5 project file")
    parser.add_argument("--output-json", required=True, help="Path to write trajectory3d_v2 JSON")
    parser.add_argument(
        "--camera-json",
        default=None,
        help=(
            "Optional calibrated camera intrinsics JSON. Supports keys K and dist, "
            "or focal_px/cx/cy/dist."
        ),
    )
    parser.add_argument(
        "--focal-px",
        type=float,
        default=None,
        help="Fallback focal length in pixels when --camera-json does not provide K.",
    )
    parser.add_argument(
        "--timebase",
        choices=("auto", "merged", "original"),
        default="auto",
        help="How to interpret .llc times relative to tracking video (default: auto)",
    )
    parser.add_argument(
        "--render-video",
        default=None,
        help="Optional debug MP4 path with selected v2 trajectory overlay",
    )
    parser.add_argument(
        "--input-video",
        default=None,
        help="Optional input video override for --render-video; defaults to tracking JSON video.input",
    )
    args = parser.parse_args()

    result = reconstruct_from_files_v2(
        args.tracking_json,
        args.cuts,
        args.output_json,
        timebase=args.timebase,
        focal_px=args.focal_px,
        camera_json=args.camera_json,
    )
    summary = result.get("summary", {})
    mapping = result.get("cut_mapping", {})
    print(
        "[trajectory3d_v2] "
        f"segments={summary.get('segment_count')} events={summary.get('event_count')} "
        f"timebase={mapping.get('mode')} drift={mapping.get('duration_drift_sec')}"
    )
    print(f"[trajectory3d_v2] selected={summary.get('selected_hypothesis_counts')}")
    print(f"[trajectory3d_v2] ambiguity={summary.get('ambiguity_counts')}")
    for flag in summary.get("qc_flags", []) or []:
        print(f"[trajectory3d_v2][qc] {flag}")
    for warning in mapping.get("warnings", []) or []:
        print(f"[trajectory3d_v2][warn] {warning}")

    if args.render_video:
        render_debug_video(
            args.tracking_json,
            args.output_json,
            args.render_video,
            input_video=args.input_video,
        )
        print(f"[trajectory3d_v2] Debug video: {args.render_video}")
    print(f"[trajectory3d_v2] Output JSON: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
