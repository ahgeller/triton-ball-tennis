"""The command line over the label tool's suggestion network (``ft.py hints`` runs it):

    python finetune/refiner/hints.py train        # or "hint model" in the finetune menu
    python finetune/refiner/hints.py list
    python finetune/refiner/hints.py use <version>
    python finetune/refiner/hints.py sheet <clip> [<version> <version>|centroid]   # judge two by eye

``cnn_refiner`` holds the network and its versions, ``refiner_training`` the training and
``refiner_sheet`` the comparison sheet; this module only reads the arguments and calls them.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

if __name__ == "__main__":          # run as a script: make the repository root importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from finetune.refiner import refiner_training
from finetune.refiner.cnn_refiner import current, make_current, version_lines, versions
from finetune.refiner.refiner_sheet import sheet


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    trainer = sub.add_parser("train", help="train a new version on every frame you have reviewed")
    trainer.add_argument("--epochs", type=int, default=refiner_training.EPOCHS)
    sub.add_parser("list", help="list the versions; * is the one the label tool uses")
    chooser = sub.add_parser("use", help="make an older (or any) version the current one")
    chooser.add_argument("version")
    viewer = sub.add_parser("sheet", help="a PNG of one clip's frames where two versions disagree most, "
                                          "zoomed, to judge by eye which sits on the ball")
    viewer.add_argument("clip")
    viewer.add_argument("versions", nargs="*", help="two version names (or 'centroid'); default: the "
                                                   "current one and the one before it")
    viewer.add_argument("--count", type=int, default=36)
    return parser


def sheet_names(named: Sequence[str]) -> List[str]:
    """The two versions a sheet compares: the ones named, else the current one and the one before
    it ("centroid", refine.py alone, standing in for a version that does not exist). A ValueError
    when the count is wrong."""
    names = list(named)
    if not names:
        active = current()
        older = [p for p in versions() if p != active]
        names = [active.stem if active else "centroid", older[-1].stem if older else "centroid"]
    if len(names) != 2:
        raise ValueError("sheet takes two version names")
    return names


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "train":
            return 0 if refiner_training.train(args.epochs) else 1
        if args.command == "list":
            print("\n".join(version_lines()))
            return 0
        if args.command == "sheet":
            return 0 if sheet(args.clip, sheet_names(args.versions), args.count) else 1
        print(f"the label tool now uses {make_current(args.version).stem}")
    except ValueError as error:
        print(f"[error] {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
