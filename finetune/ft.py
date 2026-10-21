"""One front door for the fine-tuning workspace.

Run it with no arguments (double-click ``finetune.bat``, or ``.\\finetune.ps1``) for an
interactive menu that walks through everything below. The commands remain for scripts:

    python finetune/ft.py status                     what is in the workspace, what is labelled, which model is current
    python finetune/ft.py import all                 pull every labelled tennis clip on this PC in (archive, video11, Grid)
    python finetune/ft.py add C:\\clips\\rally7.mp4     copy a new video in and pre-label it
    python finetune/ft.py label rally7               correct the labels (pre-labels first if needed)
    python finetune/ft.py check                      validate every label file against its video
    python finetune/ft.py eval                       score the current detector on the reviewed clips
    python finetune/ft.py train --val-clips video10  fine-tune; keeps the better of new vs start
    python finetune/ft.py promote                    make the winner the tracker's model (backs up the old one)
    python finetune/ft.py hints train                retrain the network that places the label tool's suggestions
    python finetune/ft.py hints list | use <version> its versions (all kept) and which one the tool uses
    python finetune/ft.py hints sheet <clip> [a b]   a zoomed PNG of where two versions disagree, to judge by eye

Runs itself under the training venv automatically (``finetune/interpreter.py``), so plain
``python`` works from any shell (``.\\finetune.ps1 <command>`` does the same from PowerShell).
The commands that run another script - import, add, prelabel, label, eval, train and hints -
pass the arguments they do not declare straight through to it (``ft.py train --epochs 3 --lr
0.3``); every other command refuses an argument it does not know, so a mistyped option or a
clip name given without ``--clips`` is an error, not silently ignored. The commands that work
in this process (status, check, promote, camera, paths) live in ``finetune/commands.py``; the
menu, the launchers and the script commands are here.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple

if __name__ == "__main__":          # run as a script: make the repository root importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Only stdlib-backed finetune modules at the top: ft.py must import without cv2, numpy or torch so
# ensure_interpreter can hand over to an interpreter that has them.
from finetune.commands import cmd_camera, cmd_check, cmd_paths, cmd_promote, cmd_status
from finetune.data.data_policy import evaluation_clips, read_review_status
from finetune.data.label_audit import open_flags, outstanding_flags
from finetune.data.labels import reviewed_frames
from finetune.data import workspace as ws
from finetune.data.workspace import ROOT, draft_csv, label_csv, labelled_clips, read_exclusions, read_manifest, video_meta
from finetune.interpreter import ensure_interpreter, wait_child
from tennis_tracker.frame_rate import phase_stride

SCRIPTS = Path(__file__).resolve().parent           # the package folder the scripts live under (not the data workspace)


# --------------------------------------------------------------------------- launching the scripts

def run_command(command: List[str]) -> int:
    print("$", " ".join(f'"{part}"' if " " in part else part for part in command), flush=True)
    return wait_child(command, cwd=str(ROOT))


def run_script(script: str, arguments: List[str]) -> int:
    """Run one of the finetune scripts (its path under the package folder, e.g. ``training/pretrack.py``)
    under this interpreter."""
    return run_command([sys.executable, str(SCRIPTS / script), *arguments])


def run_module(module: str, arguments: List[str]) -> int:
    """Run a repository module (``python -m``) under this interpreter."""
    return run_command([sys.executable, "-m", module, *arguments])


def ask(prompt: str) -> Optional[str]:
    """One line of input; None means the user backed out (Ctrl+C / Ctrl+Z / end of input)."""
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None


# --------------------------------------------------------------------------- commands

def cmd_import(args: argparse.Namespace) -> int:
    return run_script("data/import_data.py", args.rest)


def clip_name(source: Path) -> str:
    """The clip stem a video file gets in the workspace: its file name, every run of characters
    outside ``A-Za-z0-9_.-`` replaced by one underscore."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", source.stem)


def _copy_video_in(source: Path, name: str, force: bool) -> bool:
    """Copy one video into the workspace as `name`; False when it is not a usable clip (a missing
    file, an unreadable video, a frame rate that is neither 30 nor 60). An existing copy is kept
    unless `force`."""
    if not source.is_file():
        print(f"[skip] not a file: {source}")
        return False
    try:
        fps, width, height, total = video_meta(source)
    except RuntimeError as error:
        print(f"[skip] {error}")
        return False
    if phase_stride(fps) is None:
        print(f"[skip] {source.name}: {fps:.2f} FPS is neither 30 nor 60. Re-encode first, e.g.\n"
              f"       ffmpeg -i \"{source}\" -vf fps=30 -c:v libx264 -crf 12 \"{ws.VIDEOS / (name + '.mp4')}\"")
        return False
    destination = ws.VIDEOS / f"{name}{source.suffix.lower()}"
    if destination.exists() and not force:
        print(f"[keep] {destination.name} already exists (use --force to overwrite)")
    else:
        shutil.copyfile(source, destination)
        print(f"[add ] {destination.name}: {width}x{height} @{fps:.2f} FPS, {total} frames")
    return True


def add_videos(sources: List[Path], name: Optional[str] = None, force: bool = False, prelabel: bool = True,
               pretrack_arguments: Sequence[str] = ()) -> Tuple[List[str], int]:
    """Copy videos into ``videos/`` (``clip_name`` names them; `name` names a single one) and,
    with `prelabel`, draft their labels with pretrack.py. Returns the clips now in the workspace
    and the exit status (pretrack's when it ran, else 0). The menu and ``ft.py add`` both do this."""
    ws.VIDEOS.mkdir(exist_ok=True)
    added = []
    for source in sources:
        source = Path(source)
        clip = name if (name and len(sources) == 1) else clip_name(source)
        if _copy_video_in(source, clip, force):
            added.append(clip)
    if added and prelabel:
        return added, run_script("training/pretrack.py", ["--clips", *added, *pretrack_arguments])
    return added, 0


def cmd_add(args: argparse.Namespace) -> int:
    return add_videos(args.videos, args.name, args.force, not args.no_prelabel, args.rest)[1]


def cmd_prelabel(args: argparse.Namespace) -> int:
    return run_script("training/pretrack.py", args.rest)


def cmd_label(args: argparse.Namespace) -> int:
    clips = args.clip if isinstance(args.clip, list) else [args.clip]
    # Every label_tool option is declared on this subcommand, so a bare word left over is a clip
    # that came after an option (label video10 --zoom 3 video11), not an option's value.
    clips += [word for word in args.rest if not word.startswith("-")]
    args.rest = [word for word in args.rest if word.startswith("-")]
    options = ((["--zoom", str(args.zoom)] if args.zoom is not None else [])
               + (["--grid"] if args.grid else []) + (["--uncertain"] if args.uncertain else []))
    args.rest = [*options, *args.rest]
    if len(clips) > 1 or ".." in clips[0]:
        # a run across several clips: label_tool expands the range and keeps each clip's files apart
        return run_script("labeling/label_tool.py", [*clips, *args.rest])
    clip = clips[0]
    if clip in read_exclusions():
        print(f"[warn] {clip} is in exclude.txt: its labels belong to a different cut of the video,"
              " so every frame looks wrong. There is nothing here worth correcting by hand.")
    if not any(ws.VIDEOS.glob(f"{clip}.*")):
        print(f"No video named {clip}.* in {ws.VIDEOS}. Add one with:  python finetune/ft.py add <path> --name {clip}")
        return 1
    if not label_csv(clip).is_file() and not draft_csv(clip).is_file():
        print(f"[label] no labels for {clip} yet; pre-labelling first")
        code = run_script("training/pretrack.py", ["--clips", clip])
        if code:
            return code
    before = reviewed_frames(clip)       # a damaged review file is an error naming it, never an empty answer
    code = run_script("labeling/label_tool.py", [clip, *args.rest])
    offer_realign(clip, before)
    return code


def offer_realign(clip: str, reviewed_before: Set[int]) -> None:
    """A clip with unplaceable frames gets better every time you review one of them.

    Each frame you deal with pins that row, which tells realign.py which side of the cut it is on,
    so the frames you have not reached can be re-timed to agree. Only worth raising when you have
    actually reviewed something inside a doubtful stretch.
    """
    unplaceable = set(outstanding_flags(clip)["unplaceable"]) | set(open_flags(clip)["unplaceable"])
    if not unplaceable:
        return
    reviewed_now = reviewed_frames(clip)
    fresh = (reviewed_now - reviewed_before) & unplaceable
    if not fresh:
        return
    left = len(unplaceable - reviewed_now)
    print(f"\n[auto] you reviewed {len(fresh)} of the {len(unplaceable)} frames that could not be placed "
          f"in {clip}; {left} still unplaced.")
    print(f"       realign.py can now re-time those {left} to agree with what you decided - your own "
          f"frames stay exactly where you put them.")
    answer = ask("       re-time them now? it re-runs the detector on this clip [y/N] ")
    if answer and answer.lower().startswith("y"):
        if run_script("data/realign.py", [clip, clip, "--apply"]) == 0:
            run_cli(["check", "--audit", "--mark", "--clips", clip])


def cmd_eval(args: argparse.Namespace) -> int:
    """Score the detector on the clips whose labels may stand as ground truth (``data_policy``):
    the reviewed clips not in exclude.txt, own-camera ones only when there is no review policy.
    ``--clips`` narrows that; ``--all`` lifts the gate (every labelled clip)."""
    command = ["--mode", "raw", "--models", "gridtracknet", "--archive", str(ws.WORKSPACE)]
    if args.weights:
        command += ["--gridtracknet-weights", str(Path(args.weights).resolve())]
    clips = args.clips
    if args.all:
        command.append("--include-excluded")
    elif not clips:
        clips = evaluation_clips(labelled_clips(), read_manifest(), read_review_status(ws.WORKSPACE), read_exclusions())
        if not clips:
            print("no clips qualify for the default evaluation (reviewed labels not in exclude.txt); "
                  "name some with --clips, or --all for every labelled clip")
            return 1
    if clips:
        command += ["--clips", *clips]
    return run_module("evaluation.archive", command + args.rest)


def cmd_train(args: argparse.Namespace) -> int:
    return run_script("training/train_gridtracknet.py", args.rest)


def cmd_hints(args: argparse.Namespace) -> int:
    return run_script("labeling/cnn_refiner.py", args.rest or ["list"])


# --------------------------------------------------------------------------- main

def run_cli(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("status", help="what is in the workspace").set_defaults(func=cmd_status)
    p = subparsers.add_parser("import", help="import labelled clips (see import_data.py --help)")
    p.set_defaults(func=cmd_import, forwards=True)
    p = subparsers.add_parser("add", help="copy new video(s) into videos/ and pre-label them (pretrack.py options pass through)")
    p.add_argument("videos", nargs="+")
    p.add_argument("--name", help="clip stem for a single video (default: the file name)")
    p.add_argument("--force", action="store_true")
    p.add_argument("--no-prelabel", action="store_true")
    p.set_defaults(func=cmd_add, forwards=True)
    p = subparsers.add_parser("prelabel", help="run the detector to draft labels (pretrack.py; its options pass through)")
    p.set_defaults(func=cmd_prelabel, forwards=True)
    p = subparsers.add_parser("label", help="open the click tool on a clip, or a run like video22..video53 "
                                            "(label_tool.py options pass through)")
    p.add_argument("clip", nargs="+")
    # Declared here, not left to pass through: unknown options mixed in with the clip names were
    # split wrongly (label --zoom 3 video10 read "3" as a clip).
    p.add_argument("--zoom", type=float, help="starting zoom (label_tool.py --zoom)")
    p.add_argument("--grid", action="store_true", help="open in the eight-crop review view")
    p.add_argument("--uncertain", action="store_true", help="only stop on frames that want attention")
    p.set_defaults(func=cmd_label, forwards=True)
    p = subparsers.add_parser("check", help="validate every label file against its video")
    p.add_argument("--clips", nargs="*", help="clip stems or globs")
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--audit", action="store_true",
                   help="also cross-check labels against the detector (finds clips labelled on a different cut)")
    p.add_argument("--fix", action="store_true", help="with --audit: shift fixable clips, list hopeless ones in exclude.txt")
    p.add_argument("--weights", help="with --audit: detector weights to audit with")
    p.add_argument("--mark", action="store_true",
                   help="with --audit: record the frames the detector disagrees with, so the click tool "
                        "can take you straight to them (labels/<clip>_ball.suspect.json)")
    p.set_defaults(func=cmd_check)
    p = subparsers.add_parser("eval", help="raw detector metrics on the reviewed clips not in exclude.txt "
                                           "(own-camera clips only when there is no review policy)")
    p.add_argument("--weights", help=".npz to score instead of the tracker's current weights")
    p.add_argument("--clips", nargs="*", help="clip stems or globs, among those")
    p.add_argument("--all", action="store_true",
                   help="every labelled clip instead, including those still under review or in exclude.txt "
                        "(not for choosing a model)")
    p.set_defaults(func=cmd_eval, forwards=True)
    p = subparsers.add_parser("train", help="train_gridtracknet.py (all its options pass through)")
    p.set_defaults(func=cmd_train, forwards=True)
    p = subparsers.add_parser("hints", help="cnn_refiner.py: train | list | use <version> (its arguments pass through)")
    p.set_defaults(func=cmd_hints, forwards=True)
    p = subparsers.add_parser("promote", help="copy the winning weights over the tracker's model (with backup)")
    p.add_argument("--weights", help="which .npz to promote (default: finetune/models/gridtracknet_best.npz)")
    p.set_defaults(func=cmd_promote)
    p = subparsers.add_parser("camera", help="measure camera motion per clip and tag clips.csv fixed/moving")
    p.add_argument("--clips", nargs="*", help="clip stems or globs")
    p.add_argument("--force", action="store_true", help="re-measure clips that already have a tag")
    p.set_defaults(func=cmd_camera)
    subparsers.add_parser("paths", help="show the interpreter, workspace and data source paths").set_defaults(func=cmd_paths)

    args, rest = parser.parse_known_args(argv)
    if rest and not getattr(args, "forwards", False):     # only the script commands take what they did not declare
        parser.error(f"{args.command}: unrecognized arguments: {' '.join(rest)}")
    args.rest = rest
    status = args.func(args)
    return 0 if status is None else status        # listings (status, paths) report no exit status


def main() -> int:
    handed_over = ensure_interpreter(Path(__file__).resolve())
    if handed_over is not None:
        return handed_over
    if len(sys.argv) == 1:
        # The menu runs this module's commands, so it imports ft: imported here, when the menu is
        # wanted, rather than at load, or the two would import each other half-made.
        from finetune.menu import interactive_menu
        interactive_menu()
        return 0
    return run_cli(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
