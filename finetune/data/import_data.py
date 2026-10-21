"""Bring every labelled clip we own into the finetune workspace, in one label format.

Nothing here downloads anything: every source below is already on this PC.

  archive     gridtracknet_finetuning\\archive            own camera, 11 clips (video1-10, 12), 1080p, click-labelled
  video11     V5Test\\held_back_custom_768\\...\\video11    own camera, 300 frames as 768x432 PNGs, labels never corrected
  tracknetv2  V5Test\\archive\\TrackNetV2\\TrackNetV2      TrackNetV2 *badminton* set (V5Test called it tennis): 201 clips @ 1280x720/30
              (not part of 'all' - removed from the workspace 2026-08-31 as unneeded; import explicitly to bring it back)
  grid        V5Test\\archive\\Match-Data-...\\Match-Data  GridTrackNet's public set: 53 matches, 32 with usable media
  prof        Desktop\\prof\\Dataset                       public TrackNet tennis set: 41 clips @ 1280x720/30,
              imported as video13..video53 so they carry on the own-camera video1-12 numbering
  tracknet    any folder in a public TrackNet layout (see --src): Label.csv + frames, or csv/ + video/ pairs

Everything lands in the workspace contract that pretrack.py, label_tool.py,
train_gridtracknet.py and evaluate_archive.py already read:

  videos/<clip>.mp4                 30 or 60 FPS
  labels/<clip>_ball.csv            frame,ball_x,ball_y  (native pixels; invisible parked top-right)
  clips.csv                         one row per clip: source, group, size, fps, stride, counts

Label rows are at model cadence (every 2nd frame of a 60 FPS clip, every frame at 30 FPS) and
may start on an odd frame: GridTrackNet's own set was extracted from frames 1, 3, 5, ... of
its 60 FPS videos, so label k of a 60 FPS grid match is video frame 2k+1 (verified by matching
the shipped PNGs against the videos).

    python finetune/data/import_data.py all
    python finetune/data/import_data.py archive tracknetv2
    python finetune/data/import_data.py grid --dry-run
    python finetune/data/import_data.py tracknet --src D:\\downloads\\TrackNet_tennis
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

if __name__ == "__main__":          # run as a script: make the repository root importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from finetune.data.fileio import copy_atomic
from finetune.data.labels import read_label_rows, write_labels
from finetune.data import workspace as ws
from finetune.data.workspace import DEFAULT_SOURCES, VIDEO_SUFFIXES, clip_of, label_csv, natural_key, video_meta
from finetune.data.import_sources import (Canvas, ImportOptions, Manifest, Provenance, ReviewedLabelsError, TrackNetRow,
                                          cadence_frame, canvas_scale, convert_rows, copy_media, grid_frame,
                                          import_tracknet_frames, labels_file_in, protect_existing, read_tracknet_csv,
                                          record, report, source_video)
from tennis_tracker.frame_rate import require_phase_stride

PROF_FIRST = 13          # video1..video12 are the own-camera clips; the prof set carries the numbering on
PROF_FPS = 30.0          # the set ships one JPEG per frame of a 30 FPS broadcast
GRID_LABEL_CANVAS = (1280, 720)   # GridTrackNet's FrameGenerator wrote 1280x720 PNGs; Labels.csv is in that space
V5_CUSTOM_CANVAS = (1280, 720)    # the TrackNetV5 conversion put every custom label on a 1280x720 canvas


def import_archive(src: Path, manifest: Manifest, options: ImportOptions) -> int:
    """<root>/<clip>_ball.csv + <root>/<clip>.<ext>: the own-camera clips, labels copied as they are."""
    count = 0
    for csv_path in sorted(src.glob("*_ball.csv"), key=lambda p: natural_key(p.stem)):
        clip = clip_of(csv_path)
        video = source_video(src, clip)
        if video is None:
            print(f"[skip] {csv_path.name}: no video next to it")
            continue
        dst_video = ws.VIDEOS / f"{clip}{video.suffix.lower()}"
        dst_csv = label_csv(clip)
        if options.dry_run:
            print(f"[plan] {clip}: copy {video.name} + {csv_path.name}")
            count += 1
            continue
        if dst_csv.is_file() and dst_video.is_file() and not options.force:
            action = "kept"
        else:
            read_label_rows(csv_path)          # a malformed file is a LabelFormatError before anything is written
            protect_existing(dst_csv)
            copy_media(video, dst_video, options.link)
            copy_atomic(csv_path, dst_csv)
            action = "copied"
        report(clip, action, record(manifest, clip, dst_video, Provenance("custom", "archive", "own-camera"), str(video)))
        count += 1
    return count


def import_video11(src: Path, manifest: Manifest, options: ImportOptions) -> int:
    provenance = Provenance("custom-uncorrected", "held-back", "own-camera",
                            "768x432 PNGs from the TrackNetV5 prep; labels never click-corrected; eval only")
    return int(import_tracknet_frames(src, "video11", manifest, provenance, replace(options, keep_occluded=False),
                                      canvas=V5_CUSTOM_CANVAS, fps=30.0))


def _convert_tracknetv2_clip(clip: str, csv_path: Path, video: Path, dst_video: Path, dst_csv: Path,
                             options: ImportOptions, visibility_seen: Dict[int, int]) -> None:
    """Copy one rally's video in and write its labels at model cadence, counting the visibility
    values met into `visibility_seen`."""
    fps, width, height, total = video_meta(video)
    stride = require_phase_stride(fps)
    raw = read_tracknet_csv(csv_path, options.keep_occluded)
    for visibility, _, _ in raw.values():
        visibility_seen[visibility] = visibility_seen.get(visibility, 0) + 1
    labels, dropped = convert_rows(raw, lambda index: cadence_frame(index, stride), (1.0, 1.0), total)
    if dropped:
        print(f"[warn] {clip}: {dropped} label rows beyond the video's {total} frames were dropped")
    protect_existing(dst_csv)
    copy_media(video, dst_video, options.link)
    write_labels(dst_csv, labels, width, height)


def _tracknetv2_rallies(src: Path) -> List[Tuple[Path, Path, str, str, str]]:
    """(label csv, video, section, match, rally) of every rally under a TrackNetV2 layout that has
    its video; the ones without are reported and left out."""
    found = []
    for csv_path in sorted(src.rglob("*_ball.csv"), key=lambda p: natural_key(str(p))):
        if csv_path.parent.name.lower() != "csv":
            continue
        match_dir = csv_path.parent.parent
        rally = clip_of(csv_path)
        video = source_video(match_dir / "video", rally)
        if video is None:
            print(f"[skip] {csv_path}: no video/{rally}.mp4")
            continue
        relative = match_dir.relative_to(src).parts
        found.append((csv_path, video, relative[0] if len(relative) > 1 else "", relative[-1], rally))
    return found


def import_tracknetv2(src: Path, manifest: Manifest, options: ImportOptions, prefix: str = "tnv2",
                      source: str = "tracknetv2-badminton") -> int:
    """<root>/<section>/<match>/csv/<rally>_ball.csv + <root>/<section>/<match>/video/<rally>.mp4,
    labels at model cadence from frame 0 (``cadence_frame``) in the video's own pixels."""
    count = 0
    visibility_seen: Dict[int, int] = {}
    for csv_path, video, section, match, rally in _tracknetv2_rallies(src):
        clip = "_".join(part for part in (prefix, section, match, rally) if part)
        dst_video = ws.VIDEOS / f"{clip}{video.suffix.lower()}"
        dst_csv = label_csv(clip)
        if options.dry_run:
            print(f"[plan] {clip}: copy {video.name} + convert {csv_path.name}")
            count += 1
            continue
        if dst_csv.is_file() and dst_video.is_file() and not options.force:
            action = "kept"
        else:
            _convert_tracknetv2_clip(clip, csv_path, video, dst_video, dst_csv, options, visibility_seen)
            action = "copied"
        group = "_".join(part for part in (prefix, section, match) if part)
        note = "shuttlecock, not a tennis ball; cross-sport pre-training only" if source == "tracknetv2-badminton" else ""
        report(clip, action, record(manifest, clip, dst_video, Provenance(source, section, group, note), str(video)))
        count += 1
    if visibility_seen:
        print(f"[info] {source} visibility values converted: {dict(sorted(visibility_seen.items()))}")
    return count


def _import_grid_video(clip: str, video: Path, raw: Dict[int, TrackNetRow], manifest: Manifest,
                       options: ImportOptions) -> bool:
    """A GridTrackNet match that ships its video: copy it and map Labels.csv onto it (``grid_frame``,
    scaled from GRID_LABEL_CANVAS). False when no label lands on the video."""
    dst_video = ws.VIDEOS / f"{clip}{video.suffix.lower()}"
    dst_csv = label_csv(clip)
    if options.dry_run:
        print(f"[plan] {clip}: copy '{video.name}' + convert {len(raw)} labels")
        return True
    if dst_csv.is_file() and dst_video.is_file() and not options.force:
        action = "kept"
    else:
        fps, width, height, total = video_meta(video)
        stride = require_phase_stride(fps)
        labels, dropped = convert_rows(raw, lambda index: grid_frame(index, stride),
                                       canvas_scale(GRID_LABEL_CANVAS, width, height), total)
        if dropped:
            print(f"[warn] {clip}: {dropped} label rows beyond the video's {total} frames were dropped")
        if not labels:
            print(f"[skip] {clip}: no labels map onto the video")
            return False
        protect_existing(dst_csv)
        copy_media(video, dst_video, options.link)
        write_labels(dst_csv, labels, width, height)
        action = "copied"
    provenance = Provenance("grid", "", clip, f"labels scaled from {GRID_LABEL_CANVAS[0]}x{GRID_LABEL_CANVAS[1]}")
    report(clip, action, record(manifest, clip, dst_video, provenance, str(video)))
    return True


def import_grid(src: Path, manifest: Manifest, options: ImportOptions) -> int:
    """<root>/match*/Labels.csv + the match's video, or its frames/ folder when the video is missing."""
    count = 0
    for match_dir in sorted((p for p in src.iterdir() if p.is_dir() and p.name.lower().startswith("match")),
                            key=lambda p: natural_key(p.name)):
        labels_file = next((p for p in match_dir.iterdir() if p.name.lower() == "labels.csv"), None)
        if labels_file is None:
            print(f"[skip] {match_dir.name}: no Labels.csv")
            continue
        clip = f"grid_{match_dir.name}"
        video = next((p for p in sorted(match_dir.iterdir())
                      if p.suffix.lower() in VIDEO_SUFFIXES and not p.name.startswith("._")), None)
        raw = read_tracknet_csv(labels_file, options.keep_occluded)
        if video is not None:
            count += _import_grid_video(clip, video, raw, manifest, options)
            continue
        frames_dir = match_dir / "frames"
        if frames_dir.is_dir() and import_tracknet_frames(
                frames_dir, clip, manifest, Provenance("grid", "", clip, "rebuilt from the shipped 1280x720 PNGs"),
                options, fps=30.0, labels_file=labels_file):
            count += 1
        else:
            print(f"[skip] {clip}: no video and no complete frame set for its {len(raw)} labels")
    return count


def import_prof(src: Path, manifest: Manifest, options: ImportOptions, start: int = PROF_FIRST) -> int:
    """<root>/game*/Clip*/Label.csv + one 1280x720 JPEG per frame -> videoNN.mp4 + workspace CSV.

    Named videoNN rather than prof_gameN_ClipN on purpose: the clips then sit directly under
    video12 wherever ft.py lists a workspace by name. The number a clip gets comes from its
    position in the natural game/clip order, so re-running gives every clip the same name.
    The source tag stays 'prof' - these are broadcast clips, not the own camera.
    """
    folders = []
    for game in sorted((p for p in src.iterdir() if p.is_dir()), key=lambda p: natural_key(p.name)):
        for clip_dir in sorted((p for p in game.iterdir() if p.is_dir()), key=lambda p: natural_key(p.name)):
            if labels_file_in(clip_dir) is not None:
                folders.append((game.name, clip_dir))
    count = 0
    for offset, (game, clip_dir) in enumerate(folders):
        clip = f"video{start + offset}"
        provenance = Provenance("prof", game, f"prof_{game}", f"public TrackNet tennis set, {game}/{clip_dir.name}")
        if import_tracknet_frames(clip_dir, clip, manifest, provenance, options, fps=PROF_FPS):
            count += 1
    return count


def import_tracknet(src: Path, manifest: Manifest, options: ImportOptions, prefix: str, fps: float,
                    canvas: Optional[Canvas]) -> int:
    """Generic public layouts: <root>/**/Label.csv next to frames (TrackNet tennis: game1/Clip1/), or
    <root>/**/csv/*_ball.csv + video/*.mp4 (TrackNetV2 / WASB layout)."""
    count = import_tracknetv2(src, manifest, options, prefix=prefix, source=prefix)
    for labels_file in sorted(src.rglob("*"), key=lambda p: natural_key(str(p))):
        if labels_file.name.lower() not in ("label.csv", "labels.csv"):
            continue
        folder = labels_file.parent
        relative = folder.relative_to(src).parts
        clip = "_".join((prefix, *relative)) if relative else f"{prefix}_{folder.name}"
        clip = re.sub(r"[^A-Za-z0-9_.-]+", "_", clip)
        group = "_".join((prefix, *relative[:-1])) if len(relative) > 1 else clip
        provenance = Provenance(prefix, relative[0] if relative else "", group)
        if import_tracknet_frames(folder, clip, manifest, provenance, options, canvas=canvas, fps=fps):
            count += 1
    return count


# --------------------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sources", nargs="+",
                        choices=("all", "archive", "video11", "tracknetv2", "grid", "prof", "tracknet"),
                        help="what to import ('all' = archive video11 grid prof; the badminton set only with an explicit 'tracknetv2')")
    parser.add_argument("--src", type=Path, help="Override the source folder (required for 'tracknet')")
    parser.add_argument("--prefix", default="tracknet", help="Clip-name prefix / source tag for 'tracknet' imports")
    parser.add_argument("--fps", type=float, default=30.0, help="Frame rate for clips rebuilt from image folders")
    parser.add_argument("--canvas", type=int, nargs=2, metavar=("W", "H"),
                        help="Label coordinate canvas for 'tracknet' image folders when it differs from the image size")
    parser.add_argument("--keep-occluded", action="store_true",
                        help="Treat TrackNet visibility 3 (occluded, estimated position) as a visible ball")
    parser.add_argument("--link", action="store_true", help="Hard-link videos instead of copying (same drive only)")
    parser.add_argument("--force", action="store_true", help="Re-convert clips that are already in the workspace")
    parser.add_argument("--dry-run", action="store_true", help="List what would be imported and stop")
    args = parser.parse_args()

    wanted = ["archive", "video11", "grid", "prof"] if "all" in args.sources else list(dict.fromkeys(args.sources))
    if "tracknet" in wanted and args.src is None:
        parser.error("'tracknet' needs --src <folder>")
    ws.VIDEOS.mkdir(exist_ok=True)
    ws.LABELS.mkdir(exist_ok=True)
    manifest = Manifest()
    options = ImportOptions(keep_occluded=args.keep_occluded, force=args.force, link=args.link, dry_run=args.dry_run)
    importers: Dict[str, Callable[[Path], int]] = {
        "archive": lambda src: import_archive(src, manifest, options),
        "video11": lambda src: import_video11(src, manifest, options),
        "tracknetv2": lambda src: import_tracknetv2(src, manifest, options),
        "grid": lambda src: import_grid(src, manifest, options),
        "prof": lambda src: import_prof(src, manifest, options),
        "tracknet": lambda src: import_tracknet(src, manifest, options, args.prefix, args.fps,
                                                tuple(args.canvas) if args.canvas else None),
    }
    total = run_importers(wanted, importers, manifest, args.src, args.dry_run)
    if total is None:
        return 1
    if args.dry_run:
        print(f"\n{total} clips would be imported")
    else:
        rows = manifest.rows.values()
        print(f"\n{total} clips processed; clips.csv now lists {len(rows)} clips, "
              f"{sum(int(r['label_frames']) for r in rows)} label frames, "
              f"{sum(int(r['visible']) for r in rows)} visible balls -> python finetune/ft.py status")
    return 0


def run_importers(wanted: List[str], importers: Dict[str, Callable[[Path], int]], manifest: Manifest,
                  src: Optional[Path], dry_run: bool) -> Optional[int]:
    """Run each wanted importer on its source folder (`src` when one source was named, else the
    default), saving clips.csv after each; the number of clips, or None when hand-reviewed labels
    stopped the run."""
    total = 0
    for name in wanted:
        folder = src if (src and len(wanted) == 1) else DEFAULT_SOURCES.get(name)
        if folder is None or not folder.exists():
            print(f"[skip] {name}: source folder not found: {folder}")
            continue
        print(f"\n=== {name}  <-  {folder}", flush=True)
        try:
            total += importers[name](folder)
        except ReviewedLabelsError as error:
            print(f"[stop] {error}")
            return None
        if not dry_run:
            manifest.save()
    return total


if __name__ == "__main__":
    raise SystemExit(main())
