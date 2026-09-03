"""Bring every labelled clip we own into the finetune workspace, in one label format.

Nothing here downloads anything: every source below is already on this PC.

  archive     gridtracknet_finetuning\\archive            own camera, 11 clips (video1-10, 12), 1080p, click-labelled
  video11     V5Test\\held_back_custom_768\\...\\video11    own camera, 300 frames as 768x432 PNGs, labels never corrected
  tracknetv2  V5Test\\archive\\TrackNetV2\\TrackNetV2      TrackNetV2 *badminton* set (V5Test called it tennis): 201 clips @ 1280x720/30
              (not part of 'all' - removed from the workspace 2026-08-31 as unneeded; import explicitly to bring it back)
  grid        V5Test\\archive\\Match-Data-...\\Match-Data  GridTrackNet's public set: 53 matches, 32 with usable media
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

    python finetune/import_data.py all
    python finetune/import_data.py archive tracknetv2
    python finetune/import_data.py grid --dry-run
    python finetune/import_data.py tracknet --src D:\\downloads\\TrackNet_tennis
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
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2

WORKSPACE = Path(__file__).resolve().parent
VIDEOS = WORKSPACE / "videos"
LABELS = WORKSPACE / "labels"
MANIFEST = WORKSPACE / "clips.csv"
MANIFEST_COLUMNS = ["clip", "source", "section", "group", "width", "height", "fps", "stride",
                    "label_frames", "visible", "camera", "motion_px", "origin", "note"]
VIDEO_SUFFIXES = (".mp4", ".mov", ".mkv", ".avi", ".m4v")

DESKTOP = Path(r"C:\Users\Andrew\Desktop")
DEFAULT_SOURCES = {
    "archive": DESKTOP / "gridtracknet_finetuning" / "archive",
    "video11": DESKTOP / "V5Test" / "held_back_custom_768" / "sources" / "custom" / "video11",
    "tracknetv2": DESKTOP / "V5Test" / "archive" / "TrackNetV2" / "TrackNetV2",
    "grid": DESKTOP / "V5Test" / "archive" / "Match-Data-20260716T144245Z-1-001" / "Match-Data",
}
GRID_LABEL_CANVAS = (1280, 720)   # GridTrackNet's FrameGenerator wrote 1280x720 PNGs; Labels.csv is in that space
V5_CUSTOM_CANVAS = (1280, 720)    # the TrackNetV5 conversion put every custom label on a 1280x720 canvas

Point = Optional[Tuple[float, float]]


# --------------------------------------------------------------------------- small helpers

def natural_key(text: str) -> list:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def video_meta(path: Path) -> Tuple[float, int, int, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {path}")
    meta = (
        float(capture.get(cv2.CAP_PROP_FPS)),
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    capture.release()
    return meta


def stride_for(fps: float) -> int:
    if 57.0 <= fps <= 62.0:
        return 2
    if 22.0 <= fps <= 32.0:
        return 1
    raise ValueError(f"{fps:.2f} FPS is neither 30 nor 60")


def is_visible(x: float, y: float, width: int, height: int) -> bool:
    return not (x >= width * 0.95 and y <= height * 0.05)


def write_labels(path: Path, labels: Dict[int, Point], width: int, height: int) -> Tuple[int, int]:
    """labels: frame index -> (x, y) in native pixels, or None. Returns (rows, visible)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    visible = 0
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "ball_x", "ball_y"])
        for frame in sorted(labels):
            point = labels[frame]
            if point is None:
                writer.writerow([f"frame_{frame:03d}", width - 1, 0])
            else:
                x = min(max(point[0], 0.0), width - 1.0)
                y = min(max(point[1], 0.0), height - 1.0)
                writer.writerow([f"frame_{frame:03d}", f"{x:.2f}", f"{y:.2f}"])
                visible += 1
    temporary.replace(path)
    return len(labels), visible


def read_workspace_labels(path: Path, width: int, height: int) -> Dict[int, Point]:
    labels: Dict[int, Point] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            frame = int(re.fullmatch(r"frame_(\d+)", row["frame"].strip()).group(1))
            x, y = float(row["ball_x"]), float(row["ball_y"])
            labels[frame] = (x, y) if is_visible(x, y, width, height) else None
    return labels


def read_tracknet_csv(path: Path, keep_occluded: bool) -> Dict[int, Tuple[int, float, float]]:
    """TrackNet-family CSVs: (Frame|file name), (Visibility|visibility), (X|x-coordinate), (Y|y-coordinate)[, status].

    Returns frame index -> (visibility, x, y). Visibility 0 = no ball, 1 = visible, 2 = hard to
    see, 3 = occluded with an estimated position (dropped unless keep_occluded).
    """
    out: Dict[int, Tuple[int, float, float]] = {}
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
        except OSError:
            pass
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
            subprocess.run(command, check=True)
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
    def __init__(self, path: Path = MANIFEST):
        self.path = path
        self.rows: Dict[str, dict] = {}
        if path.is_file():
            with path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    self.rows[row["clip"]] = {key: row.get(key, "") for key in MANIFEST_COLUMNS}

    def put(self, clip: str, **fields) -> None:
        row = {key: "" for key in MANIFEST_COLUMNS}
        row.update(self.rows.get(clip, {}))
        row.update({key: ("" if value is None else value) for key, value in fields.items()})
        row["clip"] = clip
        self.rows[clip] = row

    def save(self) -> None:
        temporary = self.path.with_suffix(".tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()
            for clip in sorted(self.rows, key=natural_key):
                writer.writerow(self.rows[clip])
        temporary.replace(self.path)


def load_manifest(path: Path = MANIFEST) -> Dict[str, dict]:
    """clip -> manifest row (empty when no clips.csv has been written yet)."""
    return Manifest(path).rows


def record(manifest: Manifest, clip: str, video: Path, source: str, section: str, group: str,
           origin: str, note: str = "") -> dict:
    fps, width, height, _ = video_meta(video)
    labels = read_workspace_labels(LABELS / f"{clip}_ball.csv", width, height)
    manifest.put(clip, source=source, section=section, group=group, width=width, height=height,
                 fps=f"{fps:.3f}", stride=stride_for(fps), label_frames=len(labels),
                 visible=sum(point is not None for point in labels.values()), origin=origin, note=note)
    return manifest.rows[clip]


def report(clip: str, action: str, row: dict) -> None:
    print(f"[{action:>7}] {clip:34s} {row['width']}x{row['height']} @{float(row['fps']):.0f} "
          f"labels={row['label_frames']} visible={row['visible']}", flush=True)


# --------------------------------------------------------------------------- importers

def import_archive(src: Path, manifest: Manifest, force: bool, link: bool, dry_run: bool) -> int:
    count = 0
    for csv_path in sorted(src.glob("*_ball.csv"), key=lambda p: natural_key(p.stem)):
        clip = csv_path.stem[: -len("_ball")]
        videos = [p for p in src.glob(f"{clip}.*") if p.suffix.lower() in VIDEO_SUFFIXES]
        if not videos:
            print(f"[skip] {csv_path.name}: no video next to it")
            continue
        video = videos[0]
        dst_video = VIDEOS / f"{clip}{video.suffix.lower()}"
        dst_csv = LABELS / f"{clip}_ball.csv"
        if dry_run:
            print(f"[plan] {clip}: copy {video.name} + {csv_path.name}")
            count += 1
            continue
        if dst_csv.is_file() and dst_video.is_file() and not force:
            action = "kept"
        else:
            copy_media(video, dst_video, link)
            shutil.copyfile(csv_path, dst_csv)
            action = "copied"
        report(clip, action, record(manifest, clip, dst_video, "custom", "archive", "own-camera", str(video)))
        count += 1
    return count


def import_tracknet_frames(folder: Path, clip: str, manifest: Manifest, source: str, section: str, group: str,
                           canvas: Optional[Tuple[int, int]], fps: float, keep_occluded: bool, force: bool,
                           dry_run: bool, note: str = "", label_csv: Optional[Path] = None) -> bool:
    """A folder with Label.csv (TrackNet columns) + one image per labelled frame -> mp4 + workspace CSV."""
    if label_csv is None:
        label_csv = next((p for p in folder.iterdir() if p.name.lower() in ("label.csv", "labels.csv")), None)
    if label_csv is None:
        return False
    images = sorted((p for p in folder.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")),
                    key=lambda p: natural_key(p.stem))
    if not images:
        print(f"[skip] {clip}: {label_csv.name} but no frames")
        return False
    raw = read_tracknet_csv(label_csv, keep_occluded)
    by_index = {int(re.search(r"(\d+)", p.stem).group(1)): p for p in images if re.search(r"(\d+)", p.stem)}
    missing = sorted(set(raw) - set(by_index))
    if missing:
        print(f"[skip] {clip}: {len(missing)} labelled frames have no image (first {missing[:3]})")
        return False
    ordered = sorted(raw)
    dst_video = VIDEOS / f"{clip}.mp4"
    dst_csv = LABELS / f"{clip}_ball.csv"
    if dry_run:
        print(f"[plan] {clip}: encode {len(ordered)} frames from {folder}")
        return True
    if not (dst_video.is_file() and dst_csv.is_file()) or force:
        encoder = frames_to_video([by_index[i] for i in ordered], dst_video, fps)
        _, width, height, _ = video_meta(dst_video)
        sx = width / canvas[0] if canvas else 1.0
        sy = height / canvas[1] if canvas else 1.0
        labels: Dict[int, Point] = {}
        for position, index in enumerate(ordered):
            visibility, x, y = raw[index]
            labels[position] = (x * sx, y * sy) if visibility > 0 else None
        write_labels(dst_csv, labels, width, height)
        action = "encoded"
        note = (note + "; " if note else "") + f"frames->{encoder}"
    else:
        action = "kept"
    report(clip, action, record(manifest, clip, dst_video, source, section, group, str(folder), note))
    return True


def import_video11(src: Path, manifest: Manifest, force: bool, link: bool, dry_run: bool) -> int:
    ok = import_tracknet_frames(src, "video11", manifest, "custom-uncorrected", "held-back", "own-camera",
                                V5_CUSTOM_CANVAS, 30.0, keep_occluded=False, force=force, dry_run=dry_run,
                                note="768x432 PNGs from the TrackNetV5 prep; labels never click-corrected; eval only")
    return int(ok)


def import_tracknetv2(src: Path, manifest: Manifest, force: bool, link: bool, dry_run: bool,
                      keep_occluded: bool, prefix: str = "tnv2", source: str = "tracknetv2-badminton") -> int:
    """<root>/<section>/<match>/csv/<rally>_ball.csv + <root>/<section>/<match>/video/<rally>.mp4."""
    count = 0
    visibility_seen: Dict[int, int] = {}
    for csv_path in sorted(src.rglob("*_ball.csv"), key=lambda p: natural_key(str(p))):
        if csv_path.parent.name.lower() != "csv":
            continue
        match_dir = csv_path.parent.parent
        rally = csv_path.stem[: -len("_ball")]
        videos = [p for p in (match_dir / "video").glob(f"{rally}.*") if p.suffix.lower() in VIDEO_SUFFIXES]
        if not videos:
            print(f"[skip] {csv_path}: no video/{rally}.mp4")
            continue
        video = videos[0]
        relative = match_dir.relative_to(src).parts
        section = relative[0] if len(relative) > 1 else ""
        match = relative[-1]
        clip = "_".join(part for part in (prefix, section, match, rally) if part)
        dst_video = VIDEOS / f"{clip}{video.suffix.lower()}"
        dst_csv = LABELS / f"{clip}_ball.csv"
        if dry_run:
            print(f"[plan] {clip}: copy {video.name} + convert {csv_path.name}")
            count += 1
            continue
        if dst_csv.is_file() and dst_video.is_file() and not force:
            action = "kept"
        else:
            fps, width, height, total = video_meta(video)
            stride = stride_for(fps)
            raw = read_tracknet_csv(csv_path, keep_occluded)
            for visibility, _, _ in raw.values():
                visibility_seen[visibility] = visibility_seen.get(visibility, 0) + 1
            labels: Dict[int, Point] = {}
            dropped = 0
            for index in sorted(raw):
                frame = index * stride if stride == 2 else index
                if frame >= total:
                    dropped += 1
                    continue
                visibility, x, y = raw[index]
                labels[frame] = (x, y) if visibility > 0 else None
            if dropped:
                print(f"[warn] {clip}: {dropped} label rows beyond the video's {total} frames were dropped")
            copy_media(video, dst_video, link)
            write_labels(dst_csv, labels, width, height)
            action = "copied"
        group = "_".join(part for part in (prefix, section, match) if part)
        note = "shuttlecock, not a tennis ball; cross-sport pre-training only" if source == "tracknetv2-badminton" else ""
        report(clip, action, record(manifest, clip, dst_video, source, section, group, str(video), note))
        count += 1
    if visibility_seen:
        print(f"[info] {source} visibility values converted: {dict(sorted(visibility_seen.items()))}")
    return count


def import_grid(src: Path, manifest: Manifest, force: bool, link: bool, dry_run: bool, keep_occluded: bool) -> int:
    count = 0
    for match_dir in sorted((p for p in src.iterdir() if p.is_dir() and p.name.lower().startswith("match")),
                            key=lambda p: natural_key(p.name)):
        label_csv = next((p for p in match_dir.iterdir() if p.name.lower() == "labels.csv"), None)
        if label_csv is None:
            print(f"[skip] {match_dir.name}: no Labels.csv")
            continue
        clip = f"grid_{match_dir.name}"
        videos = [p for p in match_dir.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES and not p.name.startswith("._")]
        raw = read_tracknet_csv(label_csv, keep_occluded)
        dst_csv = LABELS / f"{clip}_ball.csv"
        if videos:
            video = videos[0]
            dst_video = VIDEOS / f"{clip}{video.suffix.lower()}"
            if dry_run:
                print(f"[plan] {clip}: copy '{video.name}' + convert {len(raw)} labels")
                count += 1
                continue
            if dst_csv.is_file() and dst_video.is_file() and not force:
                action = "kept"
            else:
                fps, width, height, total = video_meta(video)
                stride = stride_for(fps)
                sx, sy = width / GRID_LABEL_CANVAS[0], height / GRID_LABEL_CANVAS[1]
                labels: Dict[int, Point] = {}
                dropped = 0
                for index in sorted(raw):
                    frame = 2 * index + 1 if stride == 2 else index   # FrameGenerator.py skips a frame first at 60 FPS
                    if frame >= total:
                        dropped += 1
                        continue
                    visibility, x, y = raw[index]
                    labels[frame] = (x * sx, y * sy) if visibility > 0 else None
                if dropped:
                    print(f"[warn] {clip}: {dropped} label rows beyond the video's {total} frames were dropped")
                if not labels:
                    print(f"[skip] {clip}: no labels map onto the video")
                    continue
                copy_media(video, dst_video, link)
                write_labels(dst_csv, labels, width, height)
                action = "copied"
            report(clip, action, record(manifest, clip, dst_video, "grid", "", clip, str(video),
                                        note=f"labels scaled from {GRID_LABEL_CANVAS[0]}x{GRID_LABEL_CANVAS[1]}"))
            count += 1
        else:
            frames_dir = match_dir / "frames"
            if frames_dir.is_dir() and import_tracknet_frames(
                    frames_dir, clip, manifest, "grid", "", clip, None, 30.0, keep_occluded, force, dry_run,
                    note="rebuilt from the shipped 1280x720 PNGs", label_csv=label_csv):
                count += 1
            else:
                print(f"[skip] {clip}: no video and no complete frame set for its {len(raw)} labels")
    return count


def import_tracknet(src: Path, manifest: Manifest, force: bool, link: bool, dry_run: bool, keep_occluded: bool,
                    prefix: str, fps: float, canvas: Optional[Tuple[int, int]]) -> int:
    """Generic public layouts: <root>/**/Label.csv next to frames (TrackNet tennis: game1/Clip1/), or
    <root>/**/csv/*_ball.csv + video/*.mp4 (TrackNetV2 / WASB layout)."""
    count = import_tracknetv2(src, manifest, force, link, dry_run, keep_occluded, prefix=prefix, source=prefix)
    for label_csv in sorted(src.rglob("*"), key=lambda p: natural_key(str(p))):
        if label_csv.name.lower() not in ("label.csv", "labels.csv"):
            continue
        folder = label_csv.parent
        relative = folder.relative_to(src).parts
        clip = "_".join((prefix, *relative)) if relative else f"{prefix}_{folder.name}"
        clip = re.sub(r"[^A-Za-z0-9_.-]+", "_", clip)
        group = "_".join((prefix, *relative[:-1])) if len(relative) > 1 else clip
        if import_tracknet_frames(folder, clip, manifest, prefix, relative[0] if relative else "", group,
                                  canvas, fps, keep_occluded, force, dry_run):
            count += 1
    return count


# --------------------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sources", nargs="+", choices=("all", "archive", "video11", "tracknetv2", "grid", "tracknet"),
                        help="what to import ('all' = archive video11 grid; the badminton set only with an explicit 'tracknetv2')")
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

    wanted = ["archive", "video11", "grid"] if "all" in args.sources else list(dict.fromkeys(args.sources))
    if "tracknet" in wanted and args.src is None:
        parser.error("'tracknet' needs --src <folder>")
    VIDEOS.mkdir(exist_ok=True)
    LABELS.mkdir(exist_ok=True)
    manifest = Manifest()
    total = 0
    for name in wanted:
        src = args.src if (args.src and len(wanted) == 1) else DEFAULT_SOURCES.get(name)
        if src is None or not src.exists():
            print(f"[skip] {name}: source folder not found: {src}")
            continue
        print(f"\n=== {name}  <-  {src}", flush=True)
        if name == "archive":
            total += import_archive(src, manifest, args.force, args.link, args.dry_run)
        elif name == "video11":
            total += import_video11(src, manifest, args.force, args.link, args.dry_run)
        elif name == "tracknetv2":
            total += import_tracknetv2(src, manifest, args.force, args.link, args.dry_run, args.keep_occluded)
        elif name == "grid":
            total += import_grid(src, manifest, args.force, args.link, args.dry_run, args.keep_occluded)
        elif name == "tracknet":
            total += import_tracknet(src, manifest, args.force, args.link, args.dry_run, args.keep_occluded,
                                     args.prefix, args.fps, tuple(args.canvas) if args.canvas else None)
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
    sys.exit(main())
