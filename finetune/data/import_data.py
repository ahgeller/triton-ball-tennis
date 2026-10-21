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
import csv
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2

if __name__ == "__main__":          # run as a script: make the repository root importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from finetune.data.fileio import backup_copy, copy_atomic
from finetune.data.labels import LabelPoint, read_label_rows, read_labels, reviewed_frames, write_labels
from finetune.data import workspace as ws
from finetune.data.workspace import (DEFAULT_SOURCES, MANIFEST_COLUMNS, VIDEO_SUFFIXES, clip_of, label_csv, natural_key,
                                     read_manifest, video_meta, write_manifest)
from tennis_tracker.frame_rate import require_phase_stride

ENCODE_TIMEOUT_S = 3600          # an ffmpeg encode of one clip that takes longer than this is stuck
PROF_FIRST = 13          # video1..video12 are the own-camera clips; the prof set carries the numbering on
PROF_FPS = 30.0          # the set ships one JPEG per frame of a 30 FPS broadcast
GRID_LABEL_CANVAS = (1280, 720)   # GridTrackNet's FrameGenerator wrote 1280x720 PNGs; Labels.csv is in that space
V5_CUSTOM_CANVAS = (1280, 720)    # the TrackNetV5 conversion put every custom label on a 1280x720 canvas

Point = LabelPoint
TrackNetRow = Tuple[int, float, float]       # visibility, x, y as the source CSV has them
Canvas = Tuple[int, int]                     # the frame size a source's label coordinates are in


@dataclass(frozen=True)
class ImportOptions:
    """The flags one import run applies to every clip."""
    keep_occluded: bool = False      # TrackNet visibility 3 (occluded, estimated) counts as a visible ball
    force: bool = False              # re-convert clips already in the workspace
    link: bool = False               # hard-link videos instead of copying them
    dry_run: bool = False            # list what would be imported and stop


@dataclass(frozen=True)
class Provenance:
    """Where a clip's labels came from, as clips.csv records it."""
    source: str                      # the dataset tag training and evaluation select by
    section: str = ""                # the dataset's own subdivision (a game, a split folder)
    group: str = ""                  # clips of one recording, held out together
    note: str = ""


# --------------------------------------------------------------------------- frame and pixel mapping

def grid_frame(index: int, stride: int) -> int:
    """GridTrackNet's own set: its FrameGenerator.py skipped a frame first at 60 FPS, so label k
    is video frame 2k + 1 there and frame k on a 30 FPS match (verified by matching the shipped
    PNGs against the videos)."""
    return 2 * index + 1 if stride == 2 else index


def cadence_frame(index: int, stride: int) -> int:
    """A label set written at model cadence from frame 0: label k is video frame k * stride."""
    return index * stride


def canvas_scale(canvas: Optional[Canvas], width: int, height: int) -> Tuple[float, float]:
    """The multipliers that take a source's label coordinates onto the video's pixels."""
    return (width / canvas[0], height / canvas[1]) if canvas else (1.0, 1.0)


# --------------------------------------------------------------------------- small helpers

class ReviewedLabelsError(RuntimeError):
    """The workspace label file about to be replaced has rows a person settled by hand."""


def protect_existing(dst_csv: Path) -> None:
    """Called before a workspace label file is written over. A file whose review sidecar says a
    person has confirmed or corrected rows is refused outright - an import is a machine's or a
    dataset's labels, never better than hand work - and any other existing file is backed up
    beside itself first, as ft.py and realign.py do."""
    if not dst_csv.is_file():
        return
    reviewed = reviewed_frames(clip_of(dst_csv), dst_csv.parent)
    if reviewed:
        raise ReviewedLabelsError(f"{dst_csv.name} has {len(reviewed)} hand-reviewed row(s); not overwriting it "
                                  f"(move its review sidecar aside first if you really mean to)")
    backup = backup_copy(dst_csv, "import")
    print(f"[backup] {dst_csv.name} -> {backup.name}")


def convert_rows(raw: Dict[int, TrackNetRow], frame_of: Callable[[int], int], scale: Tuple[float, float],
                 total: int) -> Tuple[Dict[int, Point], int]:
    """TrackNet rows -> workspace labels: `frame_of` maps a source row index to a video frame, `scale`
    takes the source canvas to the video's pixels. Returns the labels and how many rows fell beyond
    the video's `total` frames."""
    labels: Dict[int, Point] = {}
    dropped = 0
    for index in sorted(raw):
        frame = frame_of(index)
        if frame >= total:
            dropped += 1
            continue
        visibility, x, y = raw[index]
        labels[frame] = (x * scale[0], y * scale[1]) if visibility > 0 else None
    return labels, dropped


def read_tracknet_csv(path: Path, keep_occluded: bool) -> Dict[int, TrackNetRow]:
    """TrackNet-family CSVs: (Frame|file name), (Visibility|visibility), (X|x-coordinate), (Y|y-coordinate)[, status].

    Returns frame index -> (visibility, x, y). Visibility 0 = no ball, 1 = visible, 2 = hard to
    see, 3 = occluded with an estimated position (dropped unless keep_occluded).
    """
    out: Dict[int, TrackNetRow] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = {name.strip().lower(): name for name in reader.fieldnames or []}
        frame_key = fields.get("frame") or fields.get("file name") or fields.get("filename")
        vis_key = fields.get("visibility")
        x_key = fields.get("x") or fields.get("x-coordinate")
        y_key = fields.get("y") or fields.get("y-coordinate")
        if not (frame_key and vis_key and x_key and y_key):
            raise ValueError(f"{path}: unrecognised TrackNet label columns {reader.fieldnames}")
        for row in reader:
            match = re.search(r"(\d+)", row[frame_key])
            if not match:
                continue
            frame = int(match.group(1))
            visibility = int(float(row[vis_key] or 0))
            x, y = float(row[x_key] or 0.0), float(row[y_key] or 0.0)
            if visibility == 3 and not keep_occluded:
                visibility = 0
            if visibility > 0 and x <= 0.0 and y <= 0.0:
                visibility = 0
            out[frame] = (visibility, x, y)
    return out


def copy_media(src: Path, dst: Path, link: bool) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return "kept"
    if dst.exists():
        dst.unlink()
    if link:
        try:
            os.link(src, dst)
            return "linked"
        except OSError as error:
            # A hard link is only an optimisation (same drive, no second copy); across drives or on
            # a filesystem without links the copy below is the right answer, so say so and go on.
            print(f"[info] {dst.name}: cannot hard-link ({error}); copying instead")
    shutil.copyfile(src, dst)
    return "copied"


def frames_to_video(frames: List[Path], dst: Path, fps: float) -> str:
    """Encode an ordered list of same-size images into dst. ffmpeg (near-lossless x264) if present."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        with tempfile.TemporaryDirectory(prefix="frames_", dir=str(dst.parent)) as temp:
            suffix = frames[0].suffix.lower()
            for index, frame in enumerate(frames):
                target = Path(temp) / f"{index:06d}{suffix}"
                try:
                    os.link(frame, target)
                except OSError:
                    shutil.copyfile(frame, target)
            command = [ffmpeg, "-y", "-loglevel", "error", "-framerate", f"{fps:g}",
                       "-i", str(Path(temp) / f"%06d{suffix}"),
                       "-c:v", "libx264", "-preset", "slow", "-crf", "10", "-pix_fmt", "yuv420p",
                       "-movflags", "+faststart", str(dst)]
            subprocess.run(command, check=True, timeout=ENCODE_TIMEOUT_S)
        return "ffmpeg"
    first = cv2.imread(str(frames[0]))
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for frame in frames:
        writer.write(cv2.imread(str(frame)))
    writer.release()
    return "cv2 (install ffmpeg for a near-lossless encode)"


# --------------------------------------------------------------------------- manifest

class Manifest:
    """clips.csv as the importers fill it: ``put`` a clip's row, ``save`` when a source is done."""

    def __init__(self, path: Optional[Path] = None):
        self.path = ws.MANIFEST if path is None else path
        self.rows: Dict[str, dict] = read_manifest(self.path)

    def put(self, clip: str, **fields) -> None:
        row = {key: "" for key in MANIFEST_COLUMNS}
        row.update(self.rows.get(clip, {}))
        row.update({key: ("" if value is None else value) for key, value in fields.items()})
        row["clip"] = clip
        self.rows[clip] = row

    def save(self) -> None:
        write_manifest(self.rows, self.path)


def record(manifest: Manifest, clip: str, video: Path, provenance: Provenance, origin: str) -> Dict[str, str]:
    fps, width, height, _ = video_meta(video)
    labels = read_labels(label_csv(clip), width, height)
    manifest.put(clip, source=provenance.source, section=provenance.section, group=provenance.group,
                 width=width, height=height, fps=f"{fps:.3f}", stride=require_phase_stride(fps),
                 label_frames=len(labels), visible=sum(point is not None for point in labels.values()),
                 origin=origin, note=provenance.note)
    return manifest.rows[clip]


def report(clip: str, action: str, row: Dict[str, str]) -> None:
    print(f"[{action:>7}] {clip:34s} {row['width']}x{row['height']} @{float(row['fps']):.0f} "
          f"labels={row['label_frames']} visible={row['visible']}", flush=True)


def source_video(folder: Path, stem: str) -> Optional[Path]:
    """The video called `stem` in a source folder, whatever its container (macOS resource forks skipped)."""
    return next((p for p in sorted(folder.glob(f"{stem}.*"))
                 if p.suffix.lower() in VIDEO_SUFFIXES and not p.name.startswith("._")), None)


# --------------------------------------------------------------------------- importers

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


def labels_file_in(folder: Path) -> Optional[Path]:
    return next((p for p in folder.iterdir() if p.name.lower() in ("label.csv", "labels.csv")), None)


def import_tracknet_frames(folder: Path, clip: str, manifest: Manifest, provenance: Provenance, options: ImportOptions,
                           canvas: Optional[Canvas] = None, fps: float = 30.0, labels_file: Optional[Path] = None) -> bool:
    """A folder with Label.csv (TrackNet columns) + one image per labelled frame -> mp4 + workspace
    CSV. Label k is the k-th image in index order, so it is video frame k."""
    labels_file = labels_file_in(folder) if labels_file is None else labels_file
    if labels_file is None:
        return False
    images = sorted((p for p in folder.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")),
                    key=lambda p: natural_key(p.stem))
    if not images:
        print(f"[skip] {clip}: {labels_file.name} but no frames")
        return False
    raw = read_tracknet_csv(labels_file, options.keep_occluded)
    by_index = {int(re.search(r"(\d+)", p.stem).group(1)): p for p in images if re.search(r"(\d+)", p.stem)}
    missing = sorted(set(raw) - set(by_index))
    if missing:
        print(f"[skip] {clip}: {len(missing)} labelled frames have no image (first {missing[:3]})")
        return False
    ordered = sorted(raw)
    dst_video = ws.VIDEOS / f"{clip}.mp4"
    dst_csv = label_csv(clip)
    if options.dry_run:
        print(f"[plan] {clip}: encode {len(ordered)} frames from {folder}")
        return True
    if not (dst_video.is_file() and dst_csv.is_file()) or options.force:
        protect_existing(dst_csv)
        encoder = frames_to_video([by_index[i] for i in ordered], dst_video, fps)
        _, width, height, _ = video_meta(dst_video)
        position_of = {index: position for position, index in enumerate(ordered)}   # one video frame per image
        labels, _ = convert_rows(raw, position_of.__getitem__, canvas_scale(canvas, width, height), len(ordered))
        write_labels(dst_csv, labels, width, height)
        action = "encoded"
        provenance = replace(provenance, note=(provenance.note + "; " if provenance.note else "") + f"frames->{encoder}")
    else:
        action = "kept"
    report(clip, action, record(manifest, clip, dst_video, provenance, str(folder)))
    return True


def import_video11(src: Path, manifest: Manifest, options: ImportOptions) -> int:
    provenance = Provenance("custom-uncorrected", "held-back", "own-camera",
                            "768x432 PNGs from the TrackNetV5 prep; labels never click-corrected; eval only")
    return int(import_tracknet_frames(src, "video11", manifest, provenance, replace(options, keep_occluded=False),
                                      canvas=V5_CUSTOM_CANVAS, fps=30.0))


def import_tracknetv2(src: Path, manifest: Manifest, options: ImportOptions, prefix: str = "tnv2",
                      source: str = "tracknetv2-badminton") -> int:
    """<root>/<section>/<match>/csv/<rally>_ball.csv + <root>/<section>/<match>/video/<rally>.mp4,
    labels at model cadence from frame 0 (``cadence_frame``) in the video's own pixels."""
    count = 0
    visibility_seen: Dict[int, int] = {}
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
        section = relative[0] if len(relative) > 1 else ""
        match = relative[-1]
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
    importers = {
        "archive": lambda src: import_archive(src, manifest, options),
        "video11": lambda src: import_video11(src, manifest, options),
        "tracknetv2": lambda src: import_tracknetv2(src, manifest, options),
        "grid": lambda src: import_grid(src, manifest, options),
        "prof": lambda src: import_prof(src, manifest, options),
        "tracknet": lambda src: import_tracknet(src, manifest, options, args.prefix, args.fps,
                                                tuple(args.canvas) if args.canvas else None),
    }
    total = 0
    for name in wanted:
        src = args.src if (args.src and len(wanted) == 1) else DEFAULT_SOURCES.get(name)
        if src is None or not src.exists():
            print(f"[skip] {name}: source folder not found: {src}")
            continue
        print(f"\n=== {name}  <-  {src}", flush=True)
        try:
            total += importers[name](src)
        except ReviewedLabelsError as error:
            print(f"[stop] {error}")
            return 1
        if not args.dry_run:
            manifest.save()
    if args.dry_run:
        print(f"\n{total} clips would be imported")
    else:
        rows = manifest.rows.values()
        print(f"\n{total} clips processed; clips.csv now lists {len(rows)} clips, "
              f"{sum(int(r['label_frames']) for r in rows)} label frames, "
              f"{sum(int(r['visible']) for r in rows)} visible balls -> python finetune/ft.py status")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
