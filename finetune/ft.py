"""One front door for the fine-tuning workspace.

Run it with no arguments (double-click ``finetune.bat``, or ``.\\finetune.ps1``) for an
interactive menu that walks through everything below. The commands remain for scripts:

    python finetune/ft.py status                     what is in the workspace, what is labelled, which model is current
    python finetune/ft.py import all                 pull every labelled tennis clip on this PC in (archive, video11, Grid)
    python finetune/ft.py add C:\\clips\\rally7.mp4     copy a new video in and pre-label it
    python finetune/ft.py label rally7               correct the labels (pre-labels first if needed)
    python finetune/ft.py check                      validate every label file against its video
    python finetune/ft.py eval                       score the current detector on the own-camera clips
    python finetune/ft.py train --val-clips video10  fine-tune; keeps the better of new vs start
    python finetune/ft.py promote                    make the winner the tracker's model (backs up the old one)
    python finetune/ft.py hints train                retrain the network that places the label tool's suggestions
    python finetune/ft.py hints list | use <version> its versions (all kept) and which one the tool uses
    python finetune/ft.py hints sheet <clip> [a b]   a zoomed PNG of where two versions disagree, to judge by eye

Runs itself under the training venv automatically (``finetune/interpreter.py``), so plain
``python`` works from any shell (``.\\finetune.ps1 <command>`` does the same from PowerShell).
Sub-commands pass unknown arguments straight through to the underlying script (``ft.py train
--epochs 3 --lr 0.3``). The commands that work in this process (status, check, promote, camera,
paths) live in ``finetune/commands.py``; the menu, the launchers and the script commands are here.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

if __name__ == "__main__":          # run as a script: make the repository root importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Only stdlib-backed finetune modules at the top: ft.py must import without cv2, numpy or torch so
# ensure_interpreter can hand over to an interpreter that has them.
from finetune.commands import cmd_camera, cmd_check, cmd_paths, cmd_promote, cmd_status, drafted_clips, sha8
from finetune.data.data_policy import OWN_SOURCES, evaluation_clips, read_review_status
from finetune.data.label_audit import open_flags, outstanding_flags
from finetune.data.labels import reviewed_frames
from finetune.data import workspace as ws
from finetune.data.workspace import ROOT, clip_of, draft_csv, label_csv, label_files, natural_key, read_exclusions, read_manifest, video_meta, videos_in
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


# --------------------------------------------------------------------------- commands

def cmd_import(args: argparse.Namespace) -> int:
    return run_script("data/import_data.py", args.rest)


def cmd_add(args: argparse.Namespace) -> int:
    ws.VIDEOS.mkdir(exist_ok=True)
    added = []
    for source in args.videos:
        source = Path(source)
        if not source.is_file():
            print(f"[skip] not a file: {source}")
            continue
        name = args.name if (args.name and len(args.videos) == 1) else re.sub(r"[^A-Za-z0-9_.-]+", "_", source.stem)
        try:
            fps, width, height, total = video_meta(source)
        except RuntimeError as error:
            print(f"[skip] {error}")
            continue
        if phase_stride(fps) is None:
            print(f"[skip] {source.name}: {fps:.2f} FPS is neither 30 nor 60. Re-encode first, e.g.\n"
                  f"       ffmpeg -i \"{source}\" -vf fps=30 -c:v libx264 -crf 12 \"{ws.VIDEOS / (name + '.mp4')}\"")
            continue
        destination = ws.VIDEOS / f"{name}{source.suffix.lower()}"
        if destination.exists() and not args.force:
            print(f"[keep] {destination.name} already exists (use --force to overwrite)")
        else:
            shutil.copyfile(source, destination)
            print(f"[add ] {destination.name}: {width}x{height} @{fps:.2f} FPS, {total} frames")
        added.append(name)
    if added and not args.no_prelabel:
        return run_script("training/pretrack.py", ["--clips", *added, *args.rest])
    return 0


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
    answer = _ask("       re-time them now? it re-runs the detector on this clip [y/N] ")
    if answer and answer.lower().startswith("y"):
        if run_script("data/realign.py", [clip, clip, "--apply"]) == 0:
            run_cli(["check", "--audit", "--mark", "--clips", clip])


def cmd_eval(args: argparse.Namespace) -> int:
    clips = args.clips
    if not clips and not args.all:
        clips = evaluation_clips(read_manifest(), read_review_status(ws.WORKSPACE))
    command = ["--mode", "raw", "--models", "gridtracknet", "--archive", str(ws.WORKSPACE)]
    if args.weights:
        command += ["--gridtracknet-weights", str(Path(args.weights).resolve())]
    if clips:
        command += ["--clips", *clips]
    return run_module("evaluation.archive", command + args.rest)


def cmd_train(args: argparse.Namespace) -> int:
    return run_script("training/train_gridtracknet.py", args.rest)


def cmd_hints(args: argparse.Namespace) -> int:
    return run_script("labeling/cnn_refiner.py", args.rest or ["list"])


# --------------------------------------------------------------------------- interactive menu

def _ask(prompt: str) -> Optional[str]:
    """One line of input; None means the user backed out (Ctrl+C / Ctrl+Z / end of input)."""
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None


def _final_labels() -> Set[str]:
    return {clip_of(p) for p in label_files()}


def _summary_lines() -> List[str]:
    manifest = read_manifest()
    videos = videos_in()
    finals = _final_labels()
    if not videos and not finals:
        return ["the workspace is empty - pick 8 to import every labelled clip on this PC, or 2 to add a video"]
    lines = []
    own = [c for c in finals if (manifest.get(c, {}).get("source") or "custom") in OWN_SOURCES]
    excluded = [c for c in read_exclusions() if c in finals]
    line = f"{len(finals)} labelled clips ready for training ({len(own)} from your own camera)"
    if excluded:
        line += f", {len(excluded)} excluded"
    lines.append(line)
    pending = sorted(set(videos) - finals, key=natural_key)
    if pending:
        lines.append(f"{len(pending)} video(s) still need labels: " + ", ".join(pending[:6])
                     + (" ..." if len(pending) > 6 else "") + "   -> pick 3")
    if ws.REPO_WEIGHTS.is_file():
        lines.append(f"tracker model: {ws.REPO_WEIGHTS.name} "
                     f"({dt.datetime.fromtimestamp(ws.REPO_WEIGHTS.stat().st_mtime):%Y-%m-%d %H:%M})")
        if ws.BEST_WEIGHTS.is_file() and sha8(ws.BEST_WEIGHTS) != sha8(ws.REPO_WEIGHTS):
            lines.append("a training winner is waiting - pick 7 to make it the tracker's model")
    return lines


def _menu_add() -> None:
    print("tip: drag the video file onto this window instead of typing the path")
    answer = _ask("video file to add (Enter to go back): ")
    if not answer:
        return
    path = Path(answer.strip('"').strip("'"))
    run_cli(["add", str(path)])
    clip = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem)
    if any(ws.VIDEOS.glob(f"{clip}.*")):
        follow = _ask(f"open the click tool on {clip} now? [Y/n] ")
        if follow is not None and follow.lower() in ("", "y", "yes"):
            run_cli(["label", clip])


def _label_origin(clip: str, manifest: Dict[str, dict], drafts: Set[str], finals: Set[str]) -> str:
    """Where a clip's labels came from: a dataset's own CSV, or our detector's pre-labelling pass.

    Imported clips carry an origin in clips.csv - somebody hand-labelled those and the numbers are
    ground truth. Anything else got its labels from pretrack.py, which is the detector guessing,
    and needs your eyes on every frame. A source tagged '-uncorrected' counts as the second kind:
    the labels came from a machine and nobody has been over them (video11 is the TrackNetV5 prep).
    """
    row = manifest.get(clip)
    if row and row.get("origin") and not (row.get("source") or "").endswith("-uncorrected"):
        return "csv"
    return "det" if clip in drafts or clip in finals else "-"


def _columns(cells: List[str], width: int, gap: int = 2) -> List[str]:
    """Lay cells out down-then-across, like ls, so a long list stays one screen instead of one row each."""
    if not cells:
        return []
    cell_width = max(len(cell) for cell in cells)
    columns = max(1, (width + gap) // (cell_width + gap))
    rows = -(-len(cells) // columns)                        # ceil: each column is filled top to bottom
    lines = []
    for row in range(rows):
        parts = [cells[index] for index in range(row, len(cells), rows)]
        lines.append((" " * gap).join(part.ljust(cell_width) for part in parts).rstrip())
    return lines


def _resolve_choice(answer: str, choices: List[str]) -> Optional[str]:
    """Turn what was typed into a clip. None means it was ambiguous and has been explained.

    A list number and a number *in a clip's name* are easy to mix up - typing 23 for grid_match23
    opens whatever happens to be 23rd, which is a different clip with different labels. When both
    readings exist, say so rather than silently picking one.
    """
    if answer.isdigit() and 1 <= int(answer) <= len(choices):
        numbered = choices[int(answer) - 1]
        named = [c for c in choices if re.search(rf"(?<![0-9]){re.escape(answer)}$", c)]
        if named and named[0] != numbered:
            print(f"  {answer} is two things here: number {answer} in the list is {numbered}, "
                  f"and there is a clip called {named[0]}.")
            print(f"  Type the name you want - {numbered} or {named[0]} - rather than the number.")
            return None
        return numbered
    matches = [c for c in choices if c.lower() == answer.lower()] or \
              [c for c in choices if answer.lower() in c.lower()]
    if len(matches) > 1:
        print(f"  '{answer}' matches {len(matches)}: {', '.join(matches[:8])}{' ...' if len(matches) > 8 else ''}")
        return None
    return matches[0] if matches else answer


def _label_menu_rows(manifest: Dict[str, dict], videos: Dict[str, Path], finals: Set[str], drafts: Set[str],
                     excluded: Set[str]) -> Tuple[List[str], List[str], List[str], Dict[str, str], Dict[str, Dict[str, List[int]]]]:
    """The clips the label menu offers, in order: (choices, the pending ones, the labelled ones,
    clip -> tag, clip -> outstanding flags). Videos without labels come first."""
    pending = sorted(set(videos) - finals, key=natural_key)
    labelled = sorted((c for c in finals if c in videos), key=natural_key)
    choices = pending + labelled
    suspects = {clip: outstanding_flags(clip) for clip in choices}
    tags = {clip: "BAD" if clip in excluded else "BAD*" if suspects[clip]["unplaceable"]
            else _label_origin(clip, manifest, drafts, finals) for clip in choices}
    return choices, pending, labelled, tags, suspects


def _print_label_notes(counts: Dict[str, int], suspects: Dict[str, Dict[str, List[int]]]) -> None:
    """The legend: what each tag means, and which clips have frames an audit left open."""
    print(f"  csv = labels came with the clip, already ground truth ({counts['csv']})   "
          f"det = machine guesses nobody has checked, yours to correct ({counts['det']})")
    if counts["BAD"]:
        print(f"  BAD = labels do not match this video at all, excluded from training ({counts['BAD']}) - "
              f"see exclude.txt; there is nothing to correct in these, they belong to another cut")
    if counts["BAD*"]:
        worst = sorted(((len(v["unplaceable"]), c) for c, v in suspects.items() if v["unplaceable"]), reverse=True)
        print(f"  BAD* = mostly repaired, but some frames could not be placed ({counts['BAD*']}) - open it and "
              f"press f to walk them: " + ", ".join(f"{c} {n}" for n, c in worst[:4]))
    to_eyeball = sorted(((len(v["check"]), c) for c, v in suspects.items() if v["check"]), reverse=True)
    if to_eyeball:
        print(f"  {sum(n for n, _ in to_eyeball)} frame(s) across {len(to_eyeball)} clip(s) are worth an eyeball "
              f"(the detector disputes them; usually it is wrong): "
              + ", ".join(f"{c} {n}" for n, c in to_eyeball[:4]))


def _print_label_menu(manifest: Dict[str, dict], pending: List[str], labelled: List[str], tags: Dict[str, str],
                      suspects: Dict[str, Dict[str, List[int]]]) -> None:
    """The numbered clip list: the pending ones one per line, the labelled ones in columns."""
    choices = pending + labelled
    width = max(40, shutil.get_terminal_size((100, 30)).columns - 4)
    name_width = min(max(len(clip) for clip in choices), 18)   # one odd long name must not cost every column
    counts = {tag: sum(1 for t in tags.values() if t == tag) for tag in ("csv", "det", "BAD", "BAD*")}
    print(f"which clip? {len(choices)} across every source - custom, grid, prof, ...")
    _print_label_notes(counts, suspects)
    if pending:
        print(f"  needs labels ({len(pending)})")
        for index, clip in enumerate(pending, 1):
            row = manifest.get(clip, {})
            size = f"{row.get('width', '?')}x{row.get('height', '?')}@{float(row['fps']):.0f}" if row.get("fps") else ""
            print(f"    {index:3d} {clip}  {tags[clip]}  {size}")
        print(f"  labelled - open to review or correct ({len(labelled)})")
    cells = [f"{index:3d} {clip if len(clip) <= name_width else clip[:name_width - 1] + '~':<{name_width}} {tags[clip]:4s}"
             for index, clip in enumerate(labelled, len(pending) + 1)]
    for line in _columns(cells, width - 4):
        print("    " + line)
    print("    a range works too - type e.g. video22..video53 to work through them as one run;")
    print("    add 'u' on the end (video22..video53 u) to stop only on the uncertain frames")


def _menu_label() -> None:
    manifest = read_manifest()
    choices, pending, labelled, tags, suspects = _label_menu_rows(
        manifest, videos_in(), _final_labels(), drafted_clips(), set(read_exclusions()))
    if not choices:
        print("nothing to label yet: no videos in finetune/videos (pick 2 to add one, or 8 to import)")
        return
    _print_label_menu(manifest, pending, labelled, tags, suspects)
    answer = _ask("clip number or name (Enter to go back): ")
    if not answer:
        return
    parts = answer.split()
    only_uncertain = len(parts) > 1 and parts[-1].lower() in ("u", "uncertain")
    if only_uncertain:
        parts = parts[:-1]
    extra = ["--uncertain"] if only_uncertain else []
    if any(".." in part for part in parts):                  # let the tool expand and check it
        run_cli(["label", *parts, *extra])
        return
    clip = _resolve_choice(" ".join(parts), choices)
    if clip is None:
        return
    run_cli(["label", clip, *extra])


def _menu_check() -> None:
    audit = _ask("also cross-check the labels against the detector? slower, needs the GPU [y/N] ")
    if audit is None:
        return
    run_cli(["check", "--audit"] if audit.lower().startswith("y") else ["check"])


def _menu_train() -> None:
    print("training fine-tunes the tracker's current weights on every labelled clip,")
    print("holds out video10 to judge the result, and keeps the better of new vs current.")
    print("Enter uses those defaults; anything you type is passed through, e.g.")
    print("  --sources custom grid --oversample custom=6 --val-clips video10 video11")
    answer = _ask("options: ")
    if answer is None:
        return
    confirm = _ask("start training now? this ties up the GPU for a while [y/N] ")
    if confirm and confirm.lower().startswith("y"):
        run_cli(["train", *answer.split()])


def _menu_promote() -> None:
    if not ws.BEST_WEIGHTS.is_file():
        print("no training winner yet - train first (option 6)")
        return
    confirm = _ask(f"make {ws.BEST_WEIGHTS.name} the tracker's model? the current one is backed up first [y/N] ")
    if confirm and confirm.lower().startswith("y"):
        run_cli(["promote"])


def _menu_hints() -> None:
    print("the label tool's suggestions (what y places) are put where you would click by a small")
    print("network trained on the frames you have checked, which also decides which ones are sure")
    print("(bright orange). Retrain it after labelling more; every version is kept and the new one")
    print("is used from the label tool's next start.")
    print()
    run_script("labeling/cnn_refiner.py", ["list"])
    print()
    print("  t) train a new version now (about 10 minutes on the GPU)")
    print("  u <version>) go back to an older version, e.g. u refiner-20261019-120547 (a date alone")
    print("               is ambiguous once two versions share it)")
    print("  s <clip>) a PNG of the clip's frames where the current and the previous version disagree")
    print("            most, zoomed, to judge by eye which sits on the ball (hints sheet <clip> a b")
    print("            compares any two; 'centroid' is refine.py alone)")
    answer = _ask("pick, or Enter for the menu: ")
    if not answer:
        return
    if answer.lower() in ("t", "train"):
        run_script("labeling/cnn_refiner.py", ["train"])
    elif answer.lower().startswith("u "):
        run_script("labeling/cnn_refiner.py", ["use", answer[2:].strip()])
    elif answer.lower().startswith("s "):
        run_script("labeling/cnn_refiner.py", ["sheet"] + answer[2:].split())
    else:
        print(f"  '{answer}' is not an option")


def _menu_import() -> None:
    confirm = _ask("copy every labelled tennis clip on this PC into finetune/ (~85 clips)? [y/N] ")
    if confirm and confirm.lower().startswith("y"):
        run_cli(["import", "all"])


MENU = [
    ("1", "status", "everything in the workspace: clips, labels, models, recent runs", lambda: run_cli(["status"])),
    ("2", "add video", "copy a new video in and let the detector draft its labels", _menu_add),
    ("3", "label", "open the click tool to correct a clip's labels", _menu_label),
    ("4", "check labels", "validate every label file against its video", _menu_check),
    ("5", "evaluate", "score the current detector on your own-camera clips", lambda: run_cli(["eval"])),
    ("6", "train", "fine-tune the model on the labelled clips", _menu_train),
    ("7", "promote", "make the last training winner the tracker's model", _menu_promote),
    ("8", "import", "pull every labelled clip on this PC into the workspace", _menu_import),
    ("9", "camera tags", "mark each clip fixed/moving camera (train --cameras fixed uses it)", lambda: run_cli(["camera"])),
    ("10", "hint model", "retrain what places the label tool's suggestions; switch versions", _menu_hints),
]


def interactive_menu() -> int:
    print("Tennis ball fine-tuning workspace")
    while True:
        print()
        for line in _summary_lines():
            print(f"  {line}")
        print()
        for key, name, description, _ in MENU:
            print(f"  {key}) {name:12s} {description}")
        print(f"  q) {'quit':12s} (commands still work too: finetune status, finetune label video10, ...)")
        choice = _ask("\npick an option: ")
        if choice is None or choice.lower() in ("q", "quit", "exit"):
            return 0
        for key, name, _, handler in MENU:
            if choice == key or choice.lower() == name.split()[0]:
                print()
                try:
                    handler()
                except KeyboardInterrupt:
                    print("\n[interrupted]")
                if _ask("\npress Enter for the menu (q to quit): ") in (None, "q"):
                    return 0
                break
        else:
            print(f"  '{choice}' is not an option - type one of the numbers, or q to quit")


# --------------------------------------------------------------------------- main

def run_cli(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("status", help="what is in the workspace").set_defaults(func=cmd_status)
    p = subparsers.add_parser("import", help="import labelled clips (see import_data.py --help)")
    p.set_defaults(func=cmd_import)
    p = subparsers.add_parser("add", help="copy new video(s) into videos/ and pre-label them")
    p.add_argument("videos", nargs="+")
    p.add_argument("--name", help="clip stem for a single video (default: the file name)")
    p.add_argument("--force", action="store_true")
    p.add_argument("--no-prelabel", action="store_true")
    p.set_defaults(func=cmd_add)
    subparsers.add_parser("prelabel", help="run the detector to draft labels (pretrack.py)").set_defaults(func=cmd_prelabel)
    p = subparsers.add_parser("label", help="open the click tool on a clip, or a run like video22..video53")
    p.add_argument("clip", nargs="+")
    # Declared here, not left to pass through: unknown options mixed in with the clip names were
    # split wrongly (label --zoom 3 video10 read "3" as a clip).
    p.add_argument("--zoom", type=float, help="starting zoom (label_tool.py --zoom)")
    p.add_argument("--grid", action="store_true", help="open in the eight-crop review view")
    p.add_argument("--uncertain", action="store_true", help="only stop on frames that want attention")
    p.set_defaults(func=cmd_label)
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
    p = subparsers.add_parser("eval", help="raw detector metrics (default: own-camera clips only)")
    p.add_argument("--weights", help=".npz to score instead of the tracker's current weights")
    p.add_argument("--clips", nargs="*", help="clip stems or globs")
    p.add_argument("--all", action="store_true", help="every clip in the workspace, not just own footage")
    p.set_defaults(func=cmd_eval)
    subparsers.add_parser("train", help="train_gridtracknet.py (all its options pass through)").set_defaults(func=cmd_train)
    subparsers.add_parser("hints", help="cnn_refiner.py: train | list | use <version>").set_defaults(func=cmd_hints)
    p = subparsers.add_parser("promote", help="copy the winning weights over the tracker's model (with backup)")
    p.add_argument("--weights", help="which .npz to promote (default: finetune/models/gridtracknet_best.npz)")
    p.set_defaults(func=cmd_promote)
    p = subparsers.add_parser("camera", help="measure camera motion per clip and tag clips.csv fixed/moving")
    p.add_argument("--clips", nargs="*", help="clip stems or globs")
    p.add_argument("--force", action="store_true", help="re-measure clips that already have a tag")
    p.set_defaults(func=cmd_camera)
    subparsers.add_parser("paths", help="show the interpreter, workspace and data source paths").set_defaults(func=cmd_paths)

    args, rest = parser.parse_known_args(argv)
    args.rest = rest
    status = args.func(args)
    return 0 if status is None else status        # listings (status, paths) report no exit status


def main() -> int:
    handed_over = ensure_interpreter(Path(__file__).resolve())
    if handed_over is not None:
        return handed_over
    if len(sys.argv) == 1:
        return interactive_menu()
    return run_cli(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
