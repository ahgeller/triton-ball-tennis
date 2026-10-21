"""The interactive menu ``python finetune/ft.py`` shows with no arguments: the workspace summary,
the numbered options and the questions each one asks before it runs the matching ``ft.py``
command or script. Like ``ft.py`` it imports nothing beyond the standard library and the light
``finetune.data`` modules."""

from __future__ import annotations

import datetime as dt
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from finetune import ft
from finetune.commands import drafted_clips, sha8
from finetune.data.data_policy import OWN_SOURCES
from finetune.data.label_audit import outstanding_flags
from finetune.data import workspace as ws
from finetune.data.workspace import clip_of, label_files, natural_key, read_exclusions, read_manifest, videos_in

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
    answer = ft._ask("video file to add (Enter to go back): ")
    if not answer:
        return
    path = Path(answer.strip('"').strip("'"))
    ft.run_cli(["add", str(path)])
    clip = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem)
    if any(ws.VIDEOS.glob(f"{clip}.*")):
        follow = ft._ask(f"open the click tool on {clip} now? [Y/n] ")
        if follow is not None and follow.lower() in ("", "y", "yes"):
            ft.run_cli(["label", clip])


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
    answer = ft._ask("clip number or name (Enter to go back): ")
    if not answer:
        return
    parts = answer.split()
    only_uncertain = len(parts) > 1 and parts[-1].lower() in ("u", "uncertain")
    if only_uncertain:
        parts = parts[:-1]
    extra = ["--uncertain"] if only_uncertain else []
    if any(".." in part for part in parts):                  # let the tool expand and check it
        ft.run_cli(["label", *parts, *extra])
        return
    clip = _resolve_choice(" ".join(parts), choices)
    if clip is None:
        return
    ft.run_cli(["label", clip, *extra])


def _menu_check() -> None:
    audit = ft._ask("also cross-check the labels against the detector? slower, needs the GPU [y/N] ")
    if audit is None:
        return
    ft.run_cli(["check", "--audit"] if audit.lower().startswith("y") else ["check"])


def _menu_train() -> None:
    print("training fine-tunes the tracker's current weights on every labelled clip,")
    print("holds out video10 to judge the result, and keeps the better of new vs current.")
    print("Enter uses those defaults; anything you type is passed through, e.g.")
    print("  --sources custom grid --oversample custom=6 --val-clips video10 video11")
    answer = ft._ask("options: ")
    if answer is None:
        return
    confirm = ft._ask("start training now? this ties up the GPU for a while [y/N] ")
    if confirm and confirm.lower().startswith("y"):
        ft.run_cli(["train", *answer.split()])


def _menu_promote() -> None:
    if not ws.BEST_WEIGHTS.is_file():
        print("no training winner yet - train first (option 6)")
        return
    confirm = ft._ask(f"make {ws.BEST_WEIGHTS.name} the tracker's model? the current one is backed up first [y/N] ")
    if confirm and confirm.lower().startswith("y"):
        ft.run_cli(["promote"])


def _menu_hints() -> None:
    print("the label tool's suggestions (what y places) are put where you would click by a small")
    print("network trained on the frames you have checked, which also decides which ones are sure")
    print("(bright orange). Retrain it after labelling more; every version is kept and the new one")
    print("is used from the label tool's next start.")
    print()
    ft.run_script("labeling/cnn_refiner.py", ["list"])
    print()
    print("  t) train a new version now (about 10 minutes on the GPU)")
    print("  u <version>) go back to an older version, e.g. u refiner-20261019-120547 (a date alone")
    print("               is ambiguous once two versions share it)")
    print("  s <clip>) a PNG of the clip's frames where the current and the previous version disagree")
    print("            most, zoomed, to judge by eye which sits on the ball (hints sheet <clip> a b")
    print("            compares any two; 'centroid' is refine.py alone)")
    answer = ft._ask("pick, or Enter for the menu: ")
    if not answer:
        return
    if answer.lower() in ("t", "train"):
        ft.run_script("labeling/cnn_refiner.py", ["train"])
    elif answer.lower().startswith("u "):
        ft.run_script("labeling/cnn_refiner.py", ["use", answer[2:].strip()])
    elif answer.lower().startswith("s "):
        ft.run_script("labeling/cnn_refiner.py", ["sheet"] + answer[2:].split())
    else:
        print(f"  '{answer}' is not an option")


def _menu_import() -> None:
    confirm = ft._ask("copy every labelled tennis clip on this PC into finetune/ (~85 clips)? [y/N] ")
    if confirm and confirm.lower().startswith("y"):
        ft.run_cli(["import", "all"])


MENU = [
    ("1", "status", "everything in the workspace: clips, labels, models, recent runs", lambda: ft.run_cli(["status"])),
    ("2", "add video", "copy a new video in and let the detector draft its labels", _menu_add),
    ("3", "label", "open the click tool to correct a clip's labels", _menu_label),
    ("4", "check labels", "validate every label file against its video", _menu_check),
    ("5", "evaluate", "score the current detector on your own-camera clips", lambda: ft.run_cli(["eval"])),
    ("6", "train", "fine-tune the model on the labelled clips", _menu_train),
    ("7", "promote", "make the last training winner the tracker's model", _menu_promote),
    ("8", "import", "pull every labelled clip on this PC into the workspace", _menu_import),
    ("9", "camera tags", "mark each clip fixed/moving camera (train --cameras fixed uses it)", lambda: ft.run_cli(["camera"])),
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
        choice = ft._ask("\npick an option: ")
        if choice is None or choice.lower() in ("q", "quit", "exit"):
            return 0
        for key, name, _, handler in MENU:
            if choice == key or choice.lower() == name.split()[0]:
                print()
                try:
                    handler()
                except KeyboardInterrupt:
                    print("\n[interrupted]")
                if ft._ask("\npress Enter for the menu (q to quit): ") in (None, "q"):
                    return 0
                break
        else:
            print(f"  '{choice}' is not an option - type one of the numbers, or q to quit")


