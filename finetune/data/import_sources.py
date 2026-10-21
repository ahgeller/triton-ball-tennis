"""What every dataset importer is built from: the options and provenance of a run, the frame
and pixel mapping of a source's labels, the TrackNet-family CSV reader, media copies and frame
encodes, the clips.csv manifest, and the importer for a folder of labelled frames that the
video11, GridTrackNet, prof and generic TrackNet sources share. ``import_data`` holds the
importers themselves and the command line."""

from __future__ import annotations

import csv
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2

from finetune.data.fileio import backup_copy
from finetune.data.labels import LabelPoint, read_labels, reviewed_frames, write_labels
from finetune.data import workspace as ws
from finetune.data.workspace import MANIFEST_COLUMNS, VIDEO_SUFFIXES, clip_of, label_csv, natural_key, read_manifest, video_meta, write_manifest
from tennis_tracker.frame_rate import require_phase_stride

ENCODE_TIMEOUT_S = 3600          # an ffmpeg encode of one clip that takes longer than this is stuck

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


TrackNetColumns = Tuple[str, str, str, str]     # the frame, visibility, x and y column names of one CSV


def tracknet_columns(path: Path, fieldnames: Optional[List[str]]) -> TrackNetColumns:
    """The four columns a TrackNet-family CSV must have, by whichever of their spellings it uses."""
    fields = {name.strip().lower(): name for name in fieldnames or []}
    frame_key = fields.get("frame") or fields.get("file name") or fields.get("filename")
    vis_key = fields.get("visibility")
    x_key = fields.get("x") or fields.get("x-coordinate")
    y_key = fields.get("y") or fields.get("y-coordinate")
    if not (frame_key and vis_key and x_key and y_key):
        raise ValueError(f"{path}: unrecognised TrackNet label columns {fieldnames}")
    return frame_key, vis_key, x_key, y_key


def tracknet_row(row: Dict[str, str], columns: TrackNetColumns, keep_occluded: bool) -> Optional[Tuple[int, TrackNetRow]]:
    """One CSV row as (frame, (visibility, x, y)), or None for a row whose frame cell has no number."""
    frame_key, vis_key, x_key, y_key = columns
    match = re.search(r"(\d+)", row[frame_key])
    if not match:
        return None
    visibility = int(float(row[vis_key] or 0))
    x, y = float(row[x_key] or 0.0), float(row[y_key] or 0.0)
    if visibility == 3 and not keep_occluded:
        visibility = 0
    if visibility > 0 and x <= 0.0 and y <= 0.0:
        visibility = 0
    return int(match.group(1)), (visibility, x, y)


def read_tracknet_csv(path: Path, keep_occluded: bool) -> Dict[int, TrackNetRow]:
    """TrackNet-family CSVs: (Frame|file name), (Visibility|visibility), (X|x-coordinate), (Y|y-coordinate)[, status].

    Returns frame index -> (visibility, x, y). Visibility 0 = no ball, 1 = visible, 2 = hard to
    see, 3 = occluded with an estimated position (dropped unless keep_occluded).
    """
    out: Dict[int, TrackNetRow] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = tracknet_columns(path, reader.fieldnames)
        for row in reader:
            parsed = tracknet_row(row, columns, keep_occluded)
            if parsed is not None:
                out[parsed[0]] = parsed[1]
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


def _read_image(path: Path):
    """The image at `path`; an unreadable one is a RuntimeError naming it, never None."""
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"could not read {path}")
    return image


def _encode_with_opencv(frames: List[Path], dst: Path, fps: float) -> None:
    """The no-ffmpeg encode. Every frame is checked before it is written: OpenCV drops a frame it
    cannot read or that is the wrong size without a word, and a dropped frame moves every label
    after it onto the wrong frame, so each one is an error naming the file instead."""
    first = _read_image(frames[0])
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"could not open {dst} for writing")
    try:
        for frame in frames:
            image = first if frame == frames[0] else _read_image(frame)
            if image.shape[:2] != (height, width):
                raise RuntimeError(f"{frame} is {image.shape[1]}x{image.shape[0]}, not {width}x{height} like {frames[0].name}")
            writer.write(image)
    finally:
        writer.release()


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
    _encode_with_opencv(frames, dst, fps)
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
