"""Run the isolated clean tennis tracker."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional

from tennis_tracker.config import Config
from tennis_tracker.paths import BOUNCE_MODEL, COURT_ENGINE, GRIDTRACKNET_WEIGHTS, PLAYER_ENGINE, ROOT
from tennis_tracker.pipeline import run
from evaluation.validate import main as validate_tracking


OUTPUT = ROOT / "output"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Isolated clean tennis tracker")
    parser.add_argument("-i", "--input", default=str(ROOT / "sample" / "pomona.mp4"))
    parser.add_argument("-o", "--output", default=str(OUTPUT / "tracking.mp4"))
    parser.add_argument("--tracking-json", default=str(OUTPUT / "tracking.json"))
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


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.self_test:
        status = _self_test()
        if status == 0:
            print("clean_version self-test passed")
        return status

    OUTPUT.mkdir(parents=True, exist_ok=True)
    cfg = Config(
        input_video=str(Path(args.input).resolve()),
        output_video=str(Path(args.output).resolve()),
        tracking_json=str(Path(args.tracking_json).resolve()),
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
