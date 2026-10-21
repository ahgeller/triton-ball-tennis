"""The label tool's command line: the arguments, the OpenCV build check, opening the clips and
starting the tool where the last session left off. ``finetune/labeling/label_tool.py`` is the
script people run; it hands over to ``main`` here."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

from finetune.labeling import label_tool
from finetune.labeling import label_window as win
from finetune.labeling.label_draw import FILTERS
from finetune.labeling.label_grid import GRID_SIZES
from finetune.labeling.label_session import ZOOM_MAX, LabelSession
from finetune.labeling.label_source import Source, build_source, expand_clip_ranges
from finetune.labeling.label_store import SETTINGS_NAME, Settings, load_settings
from finetune.labeling.label_tool import Tool
from finetune.data import workspace as ws


def build_parser() -> argparse.ArgumentParser:
    """The command line; the folder defaults are the workspace's as it is now (``TENNIS_WORKSPACE``)."""
    parser = argparse.ArgumentParser(description=label_tool.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clips", nargs="+", help="Clip stem(s), or a range like video22..video53")
    parser.add_argument("--videos", type=Path, default=ws.VIDEOS)
    parser.add_argument("--labels", type=Path, default=ws.LABELS)
    parser.add_argument("--zoom", type=float, default=None, help="Starting zoom (default: last used, else 2)")
    parser.add_argument("--grid", action="store_true", help="Review a page of enlarged ball crops together (z: 8 or 16); TAB toggles this view")
    parser.add_argument("--uncertain", action="store_true",
                        help="Only stop on frames that want attention: no confident guess, or flagged "
                             "by ft.py check --audit --mark. Every other label is left alone.")
    return parser


def qt_backend() -> bool:
    """Whether this OpenCV draws its windows with Qt; the message when it does not."""
    backend = win.highgui_backend()
    if backend.upper().startswith("QT"):
        return True
    # The view, wheel and key handling are built around Qt's window: on another backend the
    # zoom check never passes and the window is rebuilt over and over.
    print(f"[error] this OpenCV draws its windows with {backend}, but the label tool needs the Qt "
          f"build. Install conda-forge's instead of pip's:\n"
          f"        pip uninstall opencv-python && conda install -c conda-forge opencv",
          file=sys.stderr)
    return False


def open_sources(args: argparse.Namespace) -> Tuple[List[str], List[Source]]:
    """The clip names the command line expands to and the sources that are ready among them; a
    bad range is a ValueError naming it."""
    names = expand_clip_ranges(args.clips, args.labels)
    sources = [source for source in (build_source(name, args.videos, args.labels) for name in names)
               if source is not None]
    return names, sources


def main() -> int:
    if not qt_backend():
        return 2
    args = build_parser().parse_args()
    try:
        names, sources = open_sources(args)
    except ValueError as error:
        print(f"[error] {error}", file=sys.stderr)
        return 2
    if not sources:
        return 1
    tool = Tool(sources)
    session = tool.session
    if args.uncertain and not session.only_uncertain():
        print("[done] nothing uncertain left in " + (names[0] if len(names) == 1 else
                                                     f"{names[0]}..{names[-1]}") +
              " - every frame here has either a confident guess or your tick on it.")
        return 0
    if not args.uncertain:
        session.set_cursor(starting_cursor(session, sources))
    session.zoom = min(max(args.zoom or sources[0].zoom, 1.0), ZOOM_MAX)
    session.zoom_target = session.zoom
    tool.settings_path = args.labels / SETTINGS_NAME
    apply_settings(tool, load_settings(tool.settings_path))
    announce(session, sources, names, args.uncertain)
    if args.grid:
        tool.grid_view.toggle()
    tool.run()
    return 0


def starting_cursor(session: LabelSession, sources: List[Source]) -> int:
    """One clip resumes exactly where you left off; a run carries on at the first frame not done."""
    if len(sources) == 1:
        return sources[0].cursor
    return next((i for i, row in enumerate(session.rows) if not row.reviewed), 0)


def apply_settings(tool: Tool, settings: Settings) -> None:
    """The toggles the last session left in the settings file, where they still make sense."""
    session = tool.session
    if settings.get("filter") in FILTERS:
        session.filter = FILTERS.index(settings["filter"])
    session.inset = bool(settings.get("inset", session.inset))
    session.trail = bool(settings.get("trail", session.trail))
    if settings.get("grid_size") in GRID_SIZES:
        tool.grid_view.size = settings["grid_size"]


def announce(session: LabelSession, sources: List[Source], names: List[str], uncertain: bool) -> None:
    """The opening lines: what is open, where it starts and what the walk stops on."""
    total = sum(len(source.rows) for source in sources)
    flagged = sum(source.flagged for source in sources)
    if len(sources) == 1:
        source = sources[0]
        print(f"[open] {source.video.name}: {len(source.rows)} label frames, "
              f"{sum(r.uncertain for r in source.rows)} uncertain")
    else:
        done = sum(1 for row in session.rows if row.reviewed)
        print(f"[open] {len(sources)} clips as one run: {names[0]} .. {names[-1]}, {total} label frames, "
              f"{done} already done. Starting at {session.rows[session.cursor].clip} "
              f"frame {session.rows[session.cursor].frame}.")
        print("       n/p and SPACE carry straight on into the next clip; each clip saves to its own file.")
    if uncertain:
        print(f"[only ] stepping through {len(session.walk)} uncertain frame(s) of {total}; "
              f"n/p/SPACE move between those, everything else is left exactly as it is.")
    elif flagged:
        print(f"[flag] {flagged} frame(s) were marked doubtful by the last audit - press f to walk them; "
              f"they are drawn in magenta. Fixing or confirming one clears it.")
