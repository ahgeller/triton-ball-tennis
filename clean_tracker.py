"""Run the isolated clean tennis tracker."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple

from tennis_tracker.config import Config
from tennis_tracker.paths import BOUNCE_MODEL, COURT_ENGINE, GRIDTRACKNET_WEIGHTS, PLAYER_ENGINE, ROOT
from tennis_tracker.pipeline import run
from evaluation.validate import main as validate_tracking


OUTPUT = ROOT / "output"
# What a run in an output folder is named after its input: ``<stem>_tracked.mp4``
# and ``<stem>_tracking.json``.
TRACKED_VIDEO_SUFFIX = "_tracked.mp4"
TRACKING_JSON_SUFFIX = "_tracking.json"
_SEPARATORS = tuple(sep for sep in (os.sep, os.altsep) if sep)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Isolated clean tennis tracker")
    parser.add_argument("-i", "--input", default=str(ROOT / "sample" / "pomona.mp4"))
    parser.add_argument("-o", "--output", default=str(OUTPUT / "tracking.mp4"),
                        help="The tracking video, or a folder (existing, or ending with a separator) "
                             f"for <input stem>{TRACKED_VIDEO_SUFFIX}")
    parser.add_argument("--tracking-json",
                        help=f"Default: <input stem>{TRACKING_JSON_SUFFIX} beside a folder's video, "
                             f"else {OUTPUT / 'tracking.json'}")
    parser.add_argument("--conf", type=float, default=0.50)
    parser.add_argument("--device", default="0")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--court", action="store_true", help="Draw the top-right 2D court minimap")
    parser.add_argument("--info", action="store_true")
    parser.add_argument("--motion-diagnostics", action="store_true",
                        help="Add per-frame motion diagnostics to the tracking JSON "
                             "(costs ~1.7 ms/frame)")
    parser.add_argument("--annotations", help="Validate against this annotation JSON")
    parser.add_argument("--report-json", default=str(OUTPUT / "validation.json"))
    parser.add_argument("--self-test", action="store_true")
    return parser


# What ``--self-test`` runs: the selector, court-event, ROI-tracker and
# GridTrackNet suites. ``python -m pytest -q`` from the root runs every test.
SELF_TEST_MODULES = (
    "tests/test_selector_scenarios.py",
    "tests/test_court_scenarios.py",
    "tests/test_roi_tracker.py",
    "tests/test_gridtracknet.py",
)


def _self_test() -> int:
    """Run the tracker's synthetic regression tests under pytest; returns pytest's exit code."""
    try:
        import pytest  # optional: only the self-test needs it
    except ImportError:
        print("--self-test runs the tests under pytest, which is not installed "
              "(pip install pytest)", file=sys.stderr)
        return 2
    return int(pytest.main(["-q", "-p", "no:cacheprovider", "--rootdir", str(ROOT),
                            *(str(ROOT / module) for module in SELF_TEST_MODULES)]))


def resolve_output_paths(input_path: str, output: str, tracking_json: Optional[str]) -> Tuple[Path, Path]:
    """Where the tracking video and the tracking JSON go, as absolute paths.

    ``output`` names the video file, unless it is an existing directory or ends
    with a path separator: then it is a folder and the video is
    ``<input stem>_tracked.mp4`` in it.  ``tracking_json`` is used as given;
    when it is None the JSON goes beside a folder's video as
    ``<input stem>_tracking.json``, otherwise to ``OUTPUT / "tracking.json"``.
    """
    stem = Path(input_path).stem
    if output.endswith(_SEPARATORS) or Path(output).is_dir():
        folder = Path(output).resolve()
        video = folder / f"{stem}{TRACKED_VIDEO_SUFFIX}"
        default_json = folder / f"{stem}{TRACKING_JSON_SUFFIX}"
    else:
        video = Path(output).resolve()
        default_json = OUTPUT / "tracking.json"
    json_path = Path(tracking_json).resolve() if tracking_json else default_json
    return video, json_path


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.self_test:
        status = _self_test()
        if status == 0:
            print("clean_version self-test passed")
        return status

    OUTPUT.mkdir(parents=True, exist_ok=True)
    output_video, tracking_json = resolve_output_paths(args.input, args.output, args.tracking_json)
    if not args.no_video:
        print(f"[init] Tracking video: {output_video}")
    print(f"[init] Tracking JSON: {tracking_json}")
    cfg = Config(
        input_video=str(Path(args.input).resolve()),
        output_video=str(output_video),
        tracking_json=str(tracking_json),
        model_path=str(GRIDTRACKNET_WEIGHTS),
        player_model_path=str(PLAYER_ENGINE),
        court_model_path=str(COURT_ENGINE),
        conf=float(args.conf),
        device=str(args.device),
        save_tracking_video=not args.no_video,
        draw_court_minimap=bool(args.court),
        bounce_model_path=str(BOUNCE_MODEL),
        print_selector_tracks=False,
        info_timing=bool(args.info),
        motion_diagnostics=bool(args.motion_diagnostics),
    )
    run(cfg)
    if not args.annotations:
        return 0
    return validate_tracking([
        "--predictions", cfg.tracking_json,
        "--annotations", str(Path(args.annotations).resolve()),
        "--report-json", str(Path(args.report_json).resolve()),
    ])


if __name__ == "__main__":
    raise SystemExit(main())
