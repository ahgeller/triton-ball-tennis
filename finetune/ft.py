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

Runs itself under the training venv automatically, so plain ``python`` works from any shell
(``.\\finetune.ps1 <command>`` does the same from PowerShell). Sub-commands pass unknown
arguments straight through to the underlying script (``ft.py train --epochs 3 --lr 0.3``).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

WORKSPACE = Path(__file__).resolve().parent
ROOT = WORKSPACE.parent
VIDEOS = WORKSPACE / "videos"
LABELS = WORKSPACE / "labels"
MODELS = WORKSPACE / "models"
RUNS = WORKSPACE / "runs"
MANIFEST = WORKSPACE / "clips.csv"
REPO_WEIGHTS = ROOT / "models" / "gridtracknet_weights_torch.npz"
BEST_WEIGHTS = MODELS / "gridtracknet_best.npz"
VENV_PYTHON = Path(r"C:\Users\Andrew\Desktop\gridtracknet_finetuning\.venv\Scripts\python.exe")
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}
OWN_SOURCES = ("custom", "custom-uncorrected")


# --------------------------------------------------------------------------- interpreter

def training_python() -> Optional[Path]:
    for candidate in (os.environ.get("TENNIS_FINETUNE_PYTHON"), VENV_PYTHON, ROOT / ".venv" / "Scripts" / "python.exe"):
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    return None


def ensure_interpreter() -> None:
    """Re-run under the venv that has torch + cv2 when the current python lacks them."""
    if os.environ.get("TENNIS_FINETUNE_NO_REEXEC"):
        return
    try:
        import cv2  # noqa: F401
        import numpy  # noqa: F401
        return
    except ImportError:
        pass
    python = training_python()
    if python is None or python.resolve() == Path(sys.executable).resolve():
        print("This python has no cv2/numpy and no training venv was found; set TENNIS_FINETUNE_PYTHON", file=sys.stderr)
        sys.exit(2)
    env = dict(os.environ, TENNIS_FINETUNE_NO_REEXEC="1")
    sys.exit(subprocess.call([str(python), str(Path(__file__).resolve()), *sys.argv[1:]], env=env))


def run_script(script: str, arguments: List[str]) -> int:
    command = [sys.executable, str(WORKSPACE / script if not script.startswith("..") else ROOT / script[3:]), *arguments]
    print("$", " ".join(f'"{part}"' if " " in part else part for part in command), flush=True)
    return subprocess.call(command, cwd=str(ROOT))


# --------------------------------------------------------------------------- workspace facts

def read_manifest() -> Dict[str, dict]:
    if not MANIFEST.is_file():
        return {}
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        return {row["clip"]: row for row in csv.DictReader(handle)}


def label_stats(csv_path: Path, width: int, height: int):
    rows = visible = 0
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            rows += 1
            x, y = float(row["ball_x"]), float(row["ball_y"])
            if not (x >= width * 0.95 and y <= height * 0.05):
                visible += 1
    return rows, visible


def review_progress(clip: str, rows: int) -> str:
    path = LABELS / f"{clip}_ball.review.json"
    if not path.is_file():
        return ""
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    reviewed = len(state.get("reviewed", []))
    return f"{reviewed}/{rows} reviewed" if rows else f"{reviewed} reviewed"


def video_meta(path: Path):
    import cv2
    capture = cv2.VideoCapture(str(path))
    meta = (float(capture.get(cv2.CAP_PROP_FPS)), int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    capture.release()
    return meta


def sha8(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:8]


def videos_in_workspace() -> Dict[str, Path]:
    if not VIDEOS.is_dir():
        return {}
    return {p.stem: p for p in sorted(VIDEOS.iterdir()) if p.suffix.lower() in VIDEO_SUFFIXES}


# --------------------------------------------------------------------------- commands

def cmd_status(args: argparse.Namespace) -> int:
    manifest = read_manifest()
    videos = videos_in_workspace()
    finals = {p.stem[: -len("_ball")]: p for p in LABELS.glob("*_ball.csv")} if LABELS.is_dir() else {}
    drafts = {p.name[: -len("_ball.csv.draft")] for p in LABELS.glob("*_ball.csv.draft")} if LABELS.is_dir() else set()
    if not videos and not finals:
        print(f"Workspace {WORKSPACE} is empty. Start with:  python finetune/ft.py import all")
        return 0

    print(f"workspace  {WORKSPACE}")
    per_source: Dict[str, Dict[str, int]] = {}
    for clip, csv_path in finals.items():
        row = manifest.get(clip)
        source = row["source"] if row else "custom"
        if row:
            rows, visible = int(row["label_frames"]), int(row["visible"])
        else:
            _, width, height, _ = video_meta(videos[clip]) if clip in videos else (0, 1920, 1080, 0)
            rows, visible = label_stats(csv_path, width, height)
        bucket = per_source.setdefault(source, {"clips": 0, "frames": 0, "visible": 0})
        bucket["clips"] += 1
        bucket["frames"] += rows
        bucket["visible"] += visible
    print("\nlabelled clips by source")
    print(f"  {'source':22s} {'clips':>5s} {'label frames':>13s} {'visible':>8s}")
    for source, bucket in sorted(per_source.items()):
        print(f"  {source:22s} {bucket['clips']:5d} {bucket['frames']:13d} {bucket['visible']:8d}")
    total = {key: sum(b[key] for b in per_source.values()) for key in ("clips", "frames", "visible")}
    print(f"  {'total':22s} {total['clips']:5d} {total['frames']:13d} {total['visible']:8d}")
    cameras: Dict[str, Dict[str, int]] = {}
    for clip in finals:
        row = manifest.get(clip)
        if row:
            bucket = cameras.setdefault(row.get("camera") or "untagged", {"clips": 0, "frames": 0})
            bucket["clips"] += 1
            bucket["frames"] += int(row.get("label_frames") or 0)
    if cameras:
        print("camera: " + ", ".join(f"{name} {b['clips']} clips / {b['frames']} frames" for name, b in sorted(cameras.items()))
              + ("   -> python finetune/ft.py camera" if "untagged" in cameras else ""))

    own = [clip for clip in sorted(finals, key=natural_key) if (manifest.get(clip, {}).get("source") or "custom") in OWN_SOURCES]
    if own:
        print("\nown-camera clips")
        for clip in own:
            row = manifest.get(clip, {})
            size = f"{row.get('width', '?')}x{row.get('height', '?')} @{float(row['fps']):.0f}" if row.get("fps") else ""
            progress = review_progress(clip, int(row.get("label_frames", 0) or 0))
            note = row.get("note", "")
            print(f"  {clip:12s} {size:16s} labels={row.get('label_frames', '?'):>4s} visible={row.get('visible', '?'):>4s}"
                  f"  {progress:18s} {note[:60]}")

    pending = sorted(set(videos) - set(finals), key=natural_key)
    if pending:
        print("\nvideos without a final label file")
        for clip in pending:
            state = "draft ready -> python finetune/ft.py label " + clip if clip in drafts else "not pre-labelled -> python finetune/ft.py label " + clip
            print(f"  {clip:30s} {state}")
    orphans = sorted(set(finals) - set(videos), key=natural_key)
    if orphans:
        print(f"\nlabel files without a video: {', '.join(orphans[:10])}{' ...' if len(orphans) > 10 else ''}")
    excluded = read_exclusions()
    if excluded:
        print(f"\nexcluded from training by exclude.txt ({len(excluded)}): {', '.join(excluded[:12])}{' ...' if len(excluded) > 12 else ''}")

    print("\nmodels")
    if REPO_WEIGHTS.is_file():
        print(f"  tracker uses   {REPO_WEIGHTS.relative_to(ROOT)}  sha {sha8(REPO_WEIGHTS)}  "
              f"{dt.datetime.fromtimestamp(REPO_WEIGHTS.stat().st_mtime):%Y-%m-%d %H:%M}")
    if BEST_WEIGHTS.is_file():
        same = sha8(BEST_WEIGHTS) == sha8(REPO_WEIGHTS) if REPO_WEIGHTS.is_file() else False
        print(f"  last winner    {BEST_WEIGHTS.relative_to(ROOT)}  sha {sha8(BEST_WEIGHTS)}  "
              f"{'(already promoted)' if same else '-> python finetune/ft.py promote'}")
    for winner in sorted(RUNS.glob("*/winner.json"), key=lambda p: p.stat().st_mtime)[-3:]:
        try:
            payload = json.loads(winner.read_text(encoding="utf-8"))
            metrics = payload.get("metrics", {})
            print(f"  run {winner.parent.name:12s} val={payload.get('validation_clips')} winner={Path(payload.get('winner', '')).name} "
                  f"recall={metrics.get('recall', 0):.3f} wrong={metrics.get('wrong', 0):.3f} false_alarm={metrics.get('false_alarm', 0):.3f}")
        except (OSError, ValueError):
            continue
    return 0


def natural_key(text: str) -> list:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def cmd_import(args: argparse.Namespace) -> int:
    return run_script("import_data.py", args.rest)


def cmd_add(args: argparse.Namespace) -> int:
    VIDEOS.mkdir(exist_ok=True)
    added = []
    for source in args.videos:
        source = Path(source)
        if not source.is_file():
            print(f"[skip] not a file: {source}")
            continue
        name = args.name if (args.name and len(args.videos) == 1) else re.sub(r"[^A-Za-z0-9_.-]+", "_", source.stem)
        fps, width, height, total = video_meta(source)
        if not (22 <= fps <= 32 or 57 <= fps <= 62):
            print(f"[skip] {source.name}: {fps:.2f} FPS is neither 30 nor 60. Re-encode first, e.g.\n"
                  f"       ffmpeg -i \"{source}\" -vf fps=30 -c:v libx264 -crf 12 \"{VIDEOS / (name + '.mp4')}\"")
            continue
        destination = VIDEOS / f"{name}{source.suffix.lower()}"
        if destination.exists() and not args.force:
            print(f"[keep] {destination.name} already exists (use --force to overwrite)")
        else:
            shutil.copyfile(source, destination)
            print(f"[add ] {destination.name}: {width}x{height} @{fps:.2f} FPS, {total} frames")
        added.append(name)
    if added and not args.no_prelabel:
        return run_script("pretrack.py", ["--clips", *added, *args.rest])
    return 0


def cmd_prelabel(args: argparse.Namespace) -> int:
    return run_script("pretrack.py", args.rest)


def cmd_label(args: argparse.Namespace) -> int:
    clip = args.clip
    if not any(VIDEOS.glob(f"{clip}.*")):
        print(f"No video named {clip}.* in {VIDEOS}. Add one with:  python finetune/ft.py add <path> --name {clip}")
        return 1
    if not (LABELS / f"{clip}_ball.csv").is_file() and not (LABELS / f"{clip}_ball.csv.draft").is_file():
        print(f"[label] no labels for {clip} yet; pre-labelling first")
        code = run_script("pretrack.py", ["--clips", clip])
        if code:
            return code
    return run_script("label_tool.py", [clip, *args.rest])


def cmd_check(args: argparse.Namespace) -> int:
    videos = videos_in_workspace()
    manifest = read_manifest()
    problems = 0
    checked = 0
    for csv_path in sorted(LABELS.glob("*_ball.csv"), key=lambda p: natural_key(p.stem)):
        clip = csv_path.stem[: -len("_ball")]
        if args.clips and not any(re.fullmatch(pattern.replace("*", ".*"), clip) for pattern in args.clips):
            continue
        checked += 1
        video = videos.get(clip)
        if video is None:
            print(f"[fail] {clip}: no video in {VIDEOS}")
            problems += 1
            continue
        fps, width, height, total = video_meta(video)
        stride = 2 if 57 <= fps <= 62 else 1 if 22 <= fps <= 32 else 0
        issues = []
        if stride == 0:
            issues.append(f"{fps:.2f} FPS is neither 30 nor 60")
        frames = []
        with csv_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not {"frame", "ball_x", "ball_y"}.issubset(reader.fieldnames or ()):
                issues.append(f"columns are {reader.fieldnames}, expected frame,ball_x,ball_y")
            else:
                for row in reader:
                    match = re.fullmatch(r"frame_(\d+)", row["frame"].strip())
                    if not match:
                        issues.append(f"bad frame value {row['frame']!r}")
                        break
                    index = int(match.group(1))
                    frames.append(index)
                    x, y = float(row["ball_x"]), float(row["ball_y"])
                    if not (0 <= x <= width - 1 and 0 <= y <= height - 1):
                        issues.append(f"frame {index}: ({x:.0f}, {y:.0f}) outside {width}x{height}")
                        break
        if frames:
            if max(frames) >= total:
                issues.append(f"label frame {max(frames)} beyond the video's {total} frames")
            if stride:
                steps = {b - a for a, b in zip(frames, frames[1:])}
                if steps and min(steps) < stride:
                    issues.append(f"labels every {min(steps)} frame(s) but a {fps:.0f} FPS clip needs every {stride}")
                if len(set(frame % stride for frame in frames)) > 1:
                    issues.append("labels mix even and odd frames of a 60 FPS clip")
            if len(frames) != len(set(frames)):
                issues.append("duplicate frame rows")
            expected = len(range(min(frames), max(frames) + 1, max(stride, 1)))
            if len(frames) < expected * 0.9:
                issues.append(f"only {len(frames)} of {expected} frames between first and last label are labelled")
        else:
            issues.append("no label rows")
        row = manifest.get(clip)
        if row and (int(row["width"]) != width or int(row["height"]) != height):
            issues.append(f"clips.csv says {row['width']}x{row['height']} but the video is {width}x{height}")
        if issues:
            problems += 1
            print(f"[fail] {clip}: " + "; ".join(issues))
        elif args.verbose:
            print(f"[ ok ] {clip}: {len(frames)} labels, {width}x{height} @{fps:.0f}")
    print(f"{checked} label files checked, {problems} with problems")
    if args.audit:
        problems += audit_labels(args.clips, args.weights, args.fix)
    return 1 if problems else 0


EXCLUDE_FILE = WORKSPACE / "exclude.txt"
AUDIT_AGREE_PX = 15.0        # detector and label agree when within this many 1080p pixels
AUDIT_MAX_SHIFT = 3000        # temporal shifts (label frames) tried when a clip disagrees: the whole clip
AUDIT_SUSPECT = 0.60         # below this agreement (with enough detector coverage) a clip is flagged
AUDIT_FIXABLE = 0.85         # a shifted alignment this good replaces the labels with --fix
AUDIT_FAST_PX = 6.0          # 1080p px/frame above which a label can tell neighbouring frames apart
AUDIT_MIN_FAST = 20          # fewer fast labels than this and the whole clip is judged instead


def read_exclusions() -> List[str]:
    if not EXCLUDE_FILE.is_file():
        return []
    return [line.split("#", 1)[0].strip() for line in EXCLUDE_FILE.read_text(encoding="utf-8").splitlines()
            if line.split("#", 1)[0].strip()]


def add_exclusion(clip: str, reason: str) -> None:
    lines = EXCLUDE_FILE.read_text(encoding="utf-8").splitlines() if EXCLUDE_FILE.is_file() else []
    if any(line.split("#", 1)[0].strip() == clip for line in lines):
        return
    if not lines:
        lines = ["# clips the trainer skips by default (ft.py check --audit --fix writes these; edit freely)"]
    lines.append(f"{clip}  # {reason}")
    EXCLUDE_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def audit_labels(patterns: Optional[List[str]], weights: Optional[str], fix: bool) -> int:
    """Cross-check every label file against the raw detector: agreement at offset 0, and the best temporal
    shift. Catches public clips whose labels were made on a different cut of the video."""
    import math
    sys.path.insert(0, str(ROOT))
    from tennis_tracker.config import Config
    from tennis_tracker import detectors as backends

    manifest = read_manifest()
    videos = videos_in_workspace()
    clips = [p.stem[: -len("_ball")] for p in sorted(LABELS.glob("*_ball.csv"), key=lambda p: natural_key(p.stem))]
    if patterns:
        clips = [c for c in clips if any(re.fullmatch(pattern.replace("*", ".*"), c) for pattern in patterns)]
    else:  # tennis clips only by default: the tennis detector cannot audit shuttlecock labels
        clips = [c for c in clips if not (manifest.get(c, {}).get("source") or "").endswith("badminton")]
    cfg = Config(conf=0.0, device="0", gridtracknet_prepass_background=False)
    detector = backends.GridTrackNetBallDetector(str(Path(weights).resolve() if weights else REPO_WEIGHTS), cfg)
    print(f"\naudit: {len(clips)} clips against {Path(weights).name if weights else REPO_WEIGHTS.name} "
          f"(agree = detector within {AUDIT_AGREE_PX:.0f} px of the label, measured on the fast labels; "
          f"shift = label frames)")
    print(f"  {'clip':36s} {'visible':>7s} {'fast':>5s} {'covered':>7s} {'agree':>6s} {'best':>6s} {'shift':>5s} {'med px':>7s}  verdict")
    suspects = 0
    for clip in clips:
        video = videos.get(clip)
        if video is None:
            continue
        fps, width, height, total = video_meta(video)
        stride = 2 if 57 <= fps <= 62 else 1
        labels = {}
        with (LABELS / f"{clip}_ball.csv").open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                index = int(re.search(r"(\d+)", row["frame"]).group(1))
                x, y = float(row["ball_x"]), float(row["ball_y"])
                labels[index] = None if (x >= width * 0.95 and y <= height * 0.05) else (x, y)
        visible = {f: p for f, p in labels.items() if p is not None}
        detector.prepare_video(video, fps, width, height, total)
        scale = 1920.0 / width
        confident = {}
        for f, dets in enumerate(detector.precomputed):
            if dets and dets[0][1] >= 0.5:
                box = dets[0][0]
                confident[f] = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

        def agreement(shift: int, frames):
            hits, n, errors = 0, 0, []
            for f in frames:
                x, y = visible[f]
                point = confident.get(f + shift * stride)
                if point is None:
                    continue
                n += 1
                error = math.hypot(point[0] - x, point[1] - y) * scale
                errors.append(error)
                hits += error <= AUDIT_AGREE_PX
            errors.sort()
            return hits / max(n, 1), n, (errors[len(errors) // 2] if errors else float("nan"))

        # Only a fast-moving ball can separate one temporal offset from the next: a slow ball stays
        # inside AUDIT_AGREE_PX for several frames either way, so judging every label equally lets a
        # shifted clip score "ok" on the strength of its slow frames (grid_match92, +2 frames, found
        # 2026-09-02 at 0.73 overall but 0.28 on its fast labels). Judge on the fast ones when there
        # are enough of them, and fall back to all of them when there are not.
        fast = {f for f in visible
                if any(visible.get(f + step) and
                       math.hypot(visible[f + step][0] - visible[f][0], visible[f + step][1] - visible[f][1])
                       * scale / stride >= AUDIT_FAST_PX for step in (stride, -stride))}
        judged = fast if len(fast) >= AUDIT_MIN_FAST else set(visible)

        best = agreement(0, judged)
        agree0, n0 = best[0], best[1]
        best_shift = 0
        max_shift = min(AUDIT_MAX_SHIFT, total // stride)
        if agree0 < AUDIT_SUSPECT:   # only search when needed: labels may belong to another segment of the video
            for shift in range(-max_shift, max_shift + 1):
                candidate = agreement(shift, judged)
                if candidate[0] > best[0] + 1e-9 and candidate[1] >= 0.5 * max(n0, 1):
                    best_shift, best = shift, candidate
        coverage = n0 / max(len(judged), 1)
        at_edge = abs(best_shift) >= max_shift
        if coverage < 0.3:
            verdict = "detector rarely fires here; cannot judge"
        elif agree0 >= AUDIT_SUSPECT:
            verdict = "ok"
        elif best_shift != 0 and best[0] >= AUDIT_FIXABLE and not at_edge:
            verdict = f"labels are {best_shift:+d} label frame(s) off" + (" -> fixed" if fix else " (use --fix)")
            suspects += 1
        else:
            verdict = "labels disagree with the detector" + (" -> excluded" if fix else " (use --fix to exclude)")
            if at_edge:
                verdict += f" (best shift at the {best_shift:+d} search limit)"
            suspects += 1
        print(f"  {clip:36s} {len(visible):7d} {len(fast):5d} {coverage:7.2f} {agree0:6.2f} {best[0]:6.2f} "
              f"{best_shift:+5d} {best[2]:7.1f}  {verdict}", flush=True)
        if fix and coverage >= 0.3 and agree0 < AUDIT_SUSPECT:
            if best_shift != 0 and best[0] >= AUDIT_FIXABLE and not at_edge:
                shift_labels(clip, best_shift * stride, width, height, total)
            else:
                add_exclusion(clip, f"audit {dt.date.today()}: detector agrees on {agree0:.0%} of frames (best shift {best_shift:+d}: {best[0]:.0%})")
    if suspects:
        print(f"audit: {suspects} clip(s) flagged" + ("" if fix else "; re-run with --fix to shift or exclude them"))
    else:
        print("audit: every clip agrees with the detector")
    return suspects


def shift_labels(clip: str, delta_frames: int, width: int, height: int, total: int) -> None:
    """Renumber a clip's label rows by delta_frames (dropping rows that leave the video) and record it."""
    path = LABELS / f"{clip}_ball.csv"
    rows = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            index = int(re.search(r"(\d+)", row["frame"]).group(1)) + delta_frames
            if 0 <= index < total:
                rows.append((index, row["ball_x"], row["ball_y"]))
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "ball_x", "ball_y"])
        for index, x, y in sorted(rows):
            writer.writerow([f"frame_{index:03d}", x, y])
    temporary.replace(path)
    manifest = read_manifest()
    if clip in manifest and MANIFEST.is_file():
        manifest[clip]["label_frames"] = str(len(rows))
        note = manifest[clip].get("note", "")
        manifest[clip]["note"] = (note + "; " if note else "") + f"audit {dt.date.today()}: shifted {delta_frames:+d} frames"
        write_manifest(manifest)
    print(f"  [fix] {clip}: labels shifted by {delta_frames:+d} frames ({len(rows)} rows kept)")


def cmd_eval(args: argparse.Namespace) -> int:
    manifest = read_manifest()
    clips = args.clips
    if not clips and not args.all:
        clips = [clip for clip, row in manifest.items() if row.get("source") in OWN_SOURCES] or None
        if clips is None and not manifest:
            clips = None   # no manifest: everything in the folder is own footage
    command = ["--mode", "raw", "--models", "gridtracknet", "--archive", str(WORKSPACE)]
    if args.weights:
        command += ["--gridtracknet-weights", str(Path(args.weights).resolve())]
    if clips:
        command += ["--clips", *clips]
    return run_script("../evaluate_archive.py", command + args.rest)


def cmd_train(args: argparse.Namespace) -> int:
    return run_script("train_gridtracknet.py", args.rest)


def cmd_promote(args: argparse.Namespace) -> int:
    source = Path(args.weights) if args.weights else BEST_WEIGHTS
    if not source.is_file():
        print(f"No weights at {source}; train first (python finetune/ft.py train)")
        return 1
    if REPO_WEIGHTS.is_file():
        if sha8(source) == sha8(REPO_WEIGHTS):
            print(f"{REPO_WEIGHTS.name} already is {source.name}; nothing to do")
            return 0
        backup = REPO_WEIGHTS.with_name(f"{REPO_WEIGHTS.stem}_prev_{dt.datetime.now():%Y%m%d_%H%M%S}.npz")
        shutil.copyfile(REPO_WEIGHTS, backup)
        print(f"[backup ] {REPO_WEIGHTS.name} -> {backup.name}")
    shutil.copyfile(source, REPO_WEIGHTS)
    print(f"[promote] {source} -> {REPO_WEIGHTS}")
    print("next: python finetune/ft.py eval      (own-camera clips)\n"
          "      python check_parity.py          (the Pomona gate; a new model may legitimately move its numbers)")
    return 0


CAMERA_FIXED_PX = 2.5    # 75th-percentile shift between frames ~1 s apart (1080p px) at or below this = fixed
CAMERA_MAX_PX = 8.0      # ... and no sampled pair moved more than this (a single pan breaks a background model)
CAMERA_SAMPLES = 24      # frame pairs measured per clip


def measure_camera_motion(video: Path):
    """Translation between frames ~1 s apart via phase correlation on a downscaled grey frame.

    Players cover a small part of the frame, so the dominant peak is the static background; a fixed
    camera gives well under a pixel of shift, pans and hand-held phones give several. Returns
    (p75_shift_px, max_shift_px at 1080p, fraction of pairs whose correlation response was weak = cuts/zooms).
    """
    import cv2
    import numpy as np
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    step = max(1, int(round(fps)))
    starts = np.linspace(0, max(0, total - step - 1), num=min(CAMERA_SAMPLES, max(1, total // step)), dtype=int)
    shifts, weak = [], 0
    window = cv2.createHanningWindow((320, 180), cv2.CV_32F)
    for start in starts:
        frames = []
        for index in (int(start), int(start) + step):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                break
            grey = cv2.cvtColor(cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            frames.append(grey.astype(np.float32) / 255.0)
        if len(frames) < 2:
            continue
        (dx, dy), response = cv2.phaseCorrelate(frames[0], frames[1], window)
        if response < 0.15:
            weak += 1
            continue
        shifts.append(float(np.hypot(dx, dy)) * width / 320.0)
    capture.release()
    if not shifts:
        return float("nan"), float("nan"), 1.0
    shifts.sort()
    return shifts[min(len(shifts) - 1, (3 * len(shifts)) // 4)], shifts[-1], weak / max(len(starts), 1)


def write_manifest(rows: Dict[str, dict]) -> None:
    from import_data import MANIFEST_COLUMNS
    temporary = MANIFEST.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for name in sorted(rows, key=natural_key):
            writer.writerow({key: rows[name].get(key, "") for key in MANIFEST_COLUMNS})
    temporary.replace(MANIFEST)


def cmd_camera(args: argparse.Namespace) -> int:
    """Tag every clip in clips.csv as a fixed or moving camera (needed by the background channel)."""
    manifest = read_manifest()
    if not manifest:
        print("No clips.csv yet; run  python finetune/ft.py import all  first")
        return 1
    videos = videos_in_workspace()
    clips = sorted(manifest, key=natural_key)
    if args.clips:
        clips = [c for c in clips if any(re.fullmatch(pattern.replace("*", ".*"), c) for pattern in args.clips)]
    counts: Dict[str, int] = {}
    print(f"  {'clip':36s} {'p75 px':>7s} {'max px':>7s} {'cuts':>5s}  camera")
    for clip in clips:
        row = manifest[clip]
        if row.get("camera") and not args.force:
            counts[row["camera"]] = counts.get(row["camera"], 0) + 1
            continue
        video = videos.get(clip)
        if video is None:
            continue
        p75, peak, weak = measure_camera_motion(video)
        if p75 != p75:
            camera = "unknown"
        elif p75 <= CAMERA_FIXED_PX and peak <= CAMERA_MAX_PX and weak <= 0.25:
            camera = "fixed"
        else:
            camera = "moving"
        row["camera"], row["motion_px"] = camera, "" if p75 != p75 else f"{p75:.1f}"
        counts[camera] = counts.get(camera, 0) + 1
        print(f"  {clip:36s} {p75:7.1f} {peak:7.1f} {weak:5.2f}  {camera}", flush=True)
    write_manifest(manifest)
    print("camera tags: " + ", ".join(f"{name}={count}" for name, count in sorted(counts.items())) +
          f"  (fixed = 75th-pct shift <= {CAMERA_FIXED_PX:g} px and max <= {CAMERA_MAX_PX:g} px between frames 1 s apart;"
          " edit clips.csv to override)")
    return 0


def cmd_paths(args: argparse.Namespace) -> int:
    from import_data import DEFAULT_SOURCES
    print(f"python     {sys.executable}")
    print(f"workspace  {WORKSPACE}")
    print(f"repo       {ROOT}")
    print(f"weights    {REPO_WEIGHTS}")
    for name, path in DEFAULT_SOURCES.items():
        print(f"source {name:11s} {'ok ' if path.exists() else 'MISSING'} {path}")
    return 0


# --------------------------------------------------------------------------- interactive menu

def _ask(prompt: str) -> Optional[str]:
    """One line of input; None means the user backed out (Ctrl+C / Ctrl+Z / end of input)."""
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None


def _final_labels() -> set:
    return {p.stem[: -len("_ball")] for p in LABELS.glob("*_ball.csv")} if LABELS.is_dir() else set()


def _summary_lines() -> List[str]:
    manifest = read_manifest()
    videos = videos_in_workspace()
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
    if REPO_WEIGHTS.is_file():
        lines.append(f"tracker model: {REPO_WEIGHTS.name} "
                     f"({dt.datetime.fromtimestamp(REPO_WEIGHTS.stat().st_mtime):%Y-%m-%d %H:%M})")
        if BEST_WEIGHTS.is_file() and sha8(BEST_WEIGHTS) != sha8(REPO_WEIGHTS):
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
    if any(VIDEOS.glob(f"{clip}.*")):
        follow = _ask(f"open the click tool on {clip} now? [Y/n] ")
        if follow is not None and follow.lower() in ("", "y", "yes"):
            run_cli(["label", clip])


def _menu_label() -> None:
    manifest = read_manifest()
    videos = videos_in_workspace()
    finals = _final_labels()
    pending = sorted(set(videos) - finals, key=natural_key)
    labelled = sorted((c for c in finals if c in videos), key=natural_key)
    choices = [(clip, "needs labels") for clip in pending] + \
              [(clip, "labelled - open to review or correct") for clip in labelled]
    if not choices:
        print("nothing to label yet: no videos in finetune/videos (pick 2 to add one, or 8 to import)")
        return
    print(f"which clip? ({len(pending)} pending, {len(labelled)} labelled across every source - custom, grid, etc.)")
    for index, (clip, note) in enumerate(choices, 1):
        row = manifest.get(clip, {})
        source = row.get("source") or "custom"
        size = f"{row.get('width', '?')}x{row.get('height', '?')}@{float(row['fps']):.0f}" if row.get("fps") else ""
        counts = f"labels={row.get('label_frames', '?')} visible={row.get('visible', '?')}" if row.get("label_frames") else ""
        detail = "  ".join(part for part in (source, size, counts) if part)
        print(f"  {index:2d}) {clip:30s} {note:34s} {detail}")
        if row.get("note"):
            print(f"        {row['note'][:100]}")
    answer = _ask("clip number or name (Enter to go back): ")
    if not answer:
        return
    clip = choices[int(answer) - 1][0] if answer.isdigit() and 1 <= int(answer) <= len(choices) else answer
    run_cli(["label", clip])


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
    if not BEST_WEIGHTS.is_file():
        print("no training winner yet - train first (option 6)")
        return
    confirm = _ask(f"make {BEST_WEIGHTS.name} the tracker's model? the current one is backed up first [y/N] ")
    if confirm and confirm.lower().startswith("y"):
        run_cli(["promote"])


def _menu_import() -> None:
    confirm = _ask("copy every labelled tennis clip on this PC into finetune/ (~44 clips)? [y/N] ")
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
    p = subparsers.add_parser("label", help="open the click tool on a clip (pre-labels first if needed)")
    p.add_argument("clip")
    p.set_defaults(func=cmd_label)
    p = subparsers.add_parser("check", help="validate every label file against its video")
    p.add_argument("--clips", nargs="*", help="clip stems or globs")
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--audit", action="store_true",
                   help="also cross-check labels against the detector (finds clips labelled on a different cut)")
    p.add_argument("--fix", action="store_true", help="with --audit: shift fixable clips, list hopeless ones in exclude.txt")
    p.add_argument("--weights", help="with --audit: detector weights to audit with")
    p.set_defaults(func=cmd_check)
    p = subparsers.add_parser("eval", help="raw detector metrics (default: own-camera clips only)")
    p.add_argument("--weights", help=".npz to score instead of the tracker's current weights")
    p.add_argument("--clips", nargs="*", help="clip stems or globs")
    p.add_argument("--all", action="store_true", help="every clip in the workspace, not just own footage")
    p.set_defaults(func=cmd_eval)
    subparsers.add_parser("train", help="train_gridtracknet.py (all its options pass through)").set_defaults(func=cmd_train)
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
    return args.func(args)


def main() -> int:
    ensure_interpreter()
    if len(sys.argv) == 1:
        return interactive_menu()
    return run_cli(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
