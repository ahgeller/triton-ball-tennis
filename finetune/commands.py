"""The ``ft.py`` commands that do their work in this process: status, check, promote, camera
tags and paths. The ones that launch another script (label, train, eval, ...) stay in ``ft.py``
with the launchers. Like ``ft.py``, this imports nothing beyond the standard library and the
light ``finetune.data`` modules, so it loads under any interpreter."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

from finetune.data.data_policy import OWN_SOURCES
from finetune.data.fileio import backup_copy, copy_atomic
from finetune.data.label_audit import audit_labels, check_label_file, matches_any
from finetune.data.labels import LabelFormatError, read_labels, reviewed_frames
from finetune.data.scoring import REFERENCE_HEIGHT, REFERENCE_WIDTH
from finetune.data import workspace as ws
from finetune.data.workspace import (DEFAULT_SOURCES, ROOT, clip_of, draft_files, label_files, natural_key, read_exclusions,
                                     read_manifest, review_json, video_meta, videos_in, write_manifest)

LISTED_CLIPS = 12                  # how many clip names status prints before "..."


# --------------------------------------------------------------------------- workspace facts

def label_stats(csv_path: Path, width: int, height: int) -> Tuple[int, int]:
    """(rows, rows with a visible ball) of a label file."""
    labels = read_labels(csv_path, width, height)
    return len(labels), sum(point is not None for point in labels.values())


def review_progress(clip: str, rows: int) -> str:
    """How many of a clip's frames a person has reviewed (confirmed or corrected), for status."""
    try:
        reviewed = len(reviewed_frames(clip))
    except LabelFormatError as error:
        print(f"[warn] {error}")
        return "review file unreadable"
    if not reviewed and not review_json(clip).is_file():
        return ""
    return f"{reviewed}/{rows} reviewed" if rows else f"{reviewed} reviewed"


def sha8(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:8]


def drafted_clips() -> Set[str]:
    return {clip_of(p) for p in draft_files()}


# --------------------------------------------------------------------------- status

def clip_counts(clip: str, csv_path: Path, row: Optional[Dict[str, str]], videos: Dict[str, Path]) -> Tuple[int, int]:
    """(label rows, visible balls): from clips.csv when the clip is listed, else counted from the file."""
    if row:
        return int(row["label_frames"]), int(row["visible"])
    width, height = int(REFERENCE_WIDTH), int(REFERENCE_HEIGHT)
    if clip in videos:
        try:
            width, height = video_meta(videos[clip])[1:3]
        except RuntimeError as error:
            print(f"[warn] {error}; counting {csv_path.name} as 1080p")
    return label_stats(csv_path, width, height)


def print_source_table(finals: Dict[str, Path], manifest: Dict[str, dict], videos: Dict[str, Path]) -> None:
    per_source: Dict[str, Dict[str, int]] = {}
    for clip, csv_path in finals.items():
        row = manifest.get(clip)
        rows, visible = clip_counts(clip, csv_path, row, videos)
        bucket = per_source.setdefault(row["source"] if row else "custom", {"clips": 0, "frames": 0, "visible": 0})
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


def shown(path: Path) -> str:
    """A path as status prints it: relative to the repository when it is inside it."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def print_models() -> None:
    print("\nmodels")
    if ws.REPO_WEIGHTS.is_file():
        print(f"  tracker uses   {shown(ws.REPO_WEIGHTS)}  sha {sha8(ws.REPO_WEIGHTS)}  "
              f"{dt.datetime.fromtimestamp(ws.REPO_WEIGHTS.stat().st_mtime):%Y-%m-%d %H:%M}")
    if ws.BEST_WEIGHTS.is_file():
        same = sha8(ws.BEST_WEIGHTS) == sha8(ws.REPO_WEIGHTS) if ws.REPO_WEIGHTS.is_file() else False
        print(f"  last winner    {shown(ws.BEST_WEIGHTS)}  sha {sha8(ws.BEST_WEIGHTS)}  "
              f"{'(already promoted)' if same else '-> python finetune/ft.py promote'}")
    for winner in sorted(ws.RUNS.glob("*/winner.json"), key=lambda p: p.stat().st_mtime)[-3:]:
        try:
            payload = json.loads(winner.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            print(f"  run {winner.parent.name:12s} winner.json unreadable ({error})")
            continue
        metrics = payload.get("metrics", {})
        print(f"  run {winner.parent.name:12s} val={payload.get('validation_clips')} winner={Path(payload.get('winner', '')).name} "
              f"recall={metrics.get('recall', 0):.3f} wrong={metrics.get('wrong', 0):.3f} false_alarm={metrics.get('false_alarm', 0):.3f}")


def cmd_status(args: argparse.Namespace) -> int:
    manifest = read_manifest()
    videos = videos_in()
    finals = {clip_of(p): p for p in label_files()}
    drafts = drafted_clips()
    if not videos and not finals:
        print(f"Workspace {ws.WORKSPACE} is empty. Start with:  python finetune/ft.py import all")
        return 0

    print(f"workspace  {ws.WORKSPACE}")
    print_source_table(finals, manifest, videos)
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
        print(f"\nlabel files without a video: {', '.join(orphans[:LISTED_CLIPS])}{' ...' if len(orphans) > LISTED_CLIPS else ''}")
    excluded = read_exclusions()
    if excluded:
        print(f"\nexcluded from training by exclude.txt ({len(excluded)}): {', '.join(excluded[:LISTED_CLIPS])}"
              f"{' ...' if len(excluded) > LISTED_CLIPS else ''}")
    print_models()
    return 0


# --------------------------------------------------------------------------- check

def cmd_check(args: argparse.Namespace) -> int:
    videos = videos_in()
    manifest = read_manifest()
    problems = 0
    checked = 0
    for csv_path in label_files():
        clip = clip_of(csv_path)
        if not matches_any(clip, args.clips):
            continue
        checked += 1
        video = videos.get(clip)
        if video is None:
            print(f"[fail] {clip}: no video in {ws.VIDEOS}")
            problems += 1
            continue
        try:
            fps, width, height, total = video_meta(video)
        except RuntimeError as error:
            print(f"[fail] {clip}: {error}")
            problems += 1
            continue
        issues, rows = check_label_file(csv_path, fps, width, height, total)
        row = manifest.get(clip)
        if row and (int(row["width"]) != width or int(row["height"]) != height):
            issues.append(f"clips.csv says {row['width']}x{row['height']} but the video is {width}x{height}")
        if issues:
            problems += 1
            print(f"[fail] {clip}: " + "; ".join(issues))
        elif args.verbose:
            print(f"[ ok ] {clip}: {rows} labels, {width}x{height} @{fps:.0f}")
    print(f"{checked} label files checked, {problems} with problems")
    if args.audit:
        problems += audit_labels(args.clips, args.weights, args.fix, args.mark)
    return 1 if problems else 0


# --------------------------------------------------------------------------- promote

def cmd_promote(args: argparse.Namespace) -> int:
    source = Path(args.weights) if args.weights else ws.BEST_WEIGHTS
    if not source.is_file():
        print(f"No weights at {source}; train first (python finetune/ft.py train)")
        return 1
    if ws.REPO_WEIGHTS.is_file():
        if sha8(source) == sha8(ws.REPO_WEIGHTS):
            print(f"{ws.REPO_WEIGHTS.name} already is {source.name}; nothing to do")
            return 0
        backup = backup_copy(ws.REPO_WEIGHTS, "promote")
        print(f"[backup ] {ws.REPO_WEIGHTS.name} -> {backup.name}")
    copy_atomic(source, ws.REPO_WEIGHTS)
    print(f"[promote] {source} -> {ws.REPO_WEIGHTS}")
    print("next: python finetune/ft.py eval      (own-camera clips)\n"
          "      python check_parity.py          (the Pomona gate; a new model may legitimately move its numbers)")
    return 0


# --------------------------------------------------------------------------- camera tags

CAMERA_FIXED_PX = 2.5    # 75th-percentile shift between frames ~1 s apart (1080p px) at or below this = fixed
CAMERA_MAX_PX = 8.0      # ... and no sampled pair moved more than this (a single pan breaks a background model)
CAMERA_SAMPLES = 24      # frame pairs measured per clip
CAMERA_GRID = (320, 180)     # the frame is measured at this size ...
CAMERA_WEAK_RESPONSE = 0.15  # ... and a correlation peak under this is a cut or a zoom, not a shift
CAMERA_MAX_WEAK = 0.25       # more weak pairs than this share and the camera is not "fixed"


def measure_camera_motion(video: Path) -> Tuple[float, float, float]:
    """Translation between frames ~1 s apart via phase correlation on a downscaled grey frame.

    Players cover a small part of the frame, so the dominant peak is the static background; a fixed
    camera gives well under a pixel of shift, pans and hand-held phones give several. Returns
    (p75_shift_px, max_shift_px at 1080p, fraction of pairs whose correlation response was weak = cuts/zooms).
    """
    import cv2
    import numpy as np
    fps, width, _, total = video_meta(video)
    step = max(1, int(round(fps)))
    starts = np.linspace(0, max(0, total - step - 1), num=min(CAMERA_SAMPLES, max(1, total // step)), dtype=int)
    shifts, weak = [], 0
    window = cv2.createHanningWindow(CAMERA_GRID, cv2.CV_32F)
    capture = cv2.VideoCapture(str(video))
    for start in starts:
        frames = []
        for index in (int(start), int(start) + step):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                break
            grey = cv2.cvtColor(cv2.resize(frame, CAMERA_GRID, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            frames.append(grey.astype(np.float32) / 255.0)
        if len(frames) < 2:
            continue
        (dx, dy), response = cv2.phaseCorrelate(frames[0], frames[1], window)
        if response < CAMERA_WEAK_RESPONSE:
            weak += 1
            continue
        shifts.append(float(np.hypot(dx, dy)) * width / CAMERA_GRID[0])
    capture.release()
    if not shifts:
        return float("nan"), float("nan"), 1.0
    shifts.sort()
    return shifts[min(len(shifts) - 1, (3 * len(shifts)) // 4)], shifts[-1], weak / max(len(starts), 1)


def cmd_camera(args: argparse.Namespace) -> int:
    """Tag every clip in clips.csv as a fixed or moving camera (``train --cameras fixed`` selects by it)."""
    manifest = read_manifest()
    if not manifest:
        print("No clips.csv yet; run  python finetune/ft.py import all  first")
        return 1
    videos = videos_in()
    clips = [c for c in sorted(manifest, key=natural_key) if matches_any(c, args.clips)]
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
        if math.isnan(p75):                       # no pair correlated: nothing measured
            camera = "unknown"
        elif p75 <= CAMERA_FIXED_PX and peak <= CAMERA_MAX_PX and weak <= CAMERA_MAX_WEAK:
            camera = "fixed"
        else:
            camera = "moving"
        row["camera"], row["motion_px"] = camera, "" if math.isnan(p75) else f"{p75:.1f}"
        counts[camera] = counts.get(camera, 0) + 1
        print(f"  {clip:36s} {p75:7.1f} {peak:7.1f} {weak:5.2f}  {camera}", flush=True)
    write_manifest(manifest)
    print("camera tags: " + ", ".join(f"{name}={count}" for name, count in sorted(counts.items())) +
          f"  (fixed = 75th-pct shift <= {CAMERA_FIXED_PX:g} px and max <= {CAMERA_MAX_PX:g} px between frames 1 s apart;"
          " edit clips.csv to override)")
    return 0


# --------------------------------------------------------------------------- paths

def cmd_paths(args: argparse.Namespace) -> int:
    print(f"python     {sys.executable}")
    print(f"workspace  {ws.WORKSPACE}")
    print(f"repo       {ROOT}")
    print(f"weights    {ws.REPO_WEIGHTS}")
    for name, path in DEFAULT_SOURCES.items():
        print(f"source {name:11s} {'ok ' if path.exists() else 'MISSING'} {path}")
    return 0
