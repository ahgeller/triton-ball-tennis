"""Put a clip's labels back in time with its video.

    python finetune/data/realign.py <label clip> <video clip> [--preview] [--apply]

Run this when ``ft.py check --audit`` says a clip's labels do not line up. It handles the two ways
that happens, both of which turned up in the GridTrackNet set:

  * the labels belong to a *different* video - match47, match49 and match50 all ship a file called
    'gravel amature.mp4' because they are three rallies cut from one recording, and the importer
    paired each label set with a different copy of it. That shows up as one constant offset onto
    the other video, so name that clip as <video clip> and its media is copied in for you.
  * our copy of the video is missing frames relative to the one the labels were made on, which
    shows up as an offset that steps upward part way through (match26 once, match55 seven times).

Both are solved the same way: decide, per label row, which video frame it belongs to. The model is

    video frame = round(rate * label frame) + offset(row)

- rate covers a label file written at a different cadence to the video (GridTrackNet numbered its
  60 FPS labels 1, 3, 5, ... so mapping those onto a 30 FPS cut of the same footage wants 0.5), and
  offset covers where in the video the labelled part starts and any frames lost since. Candidate
  offsets come from a coarse chunk-by-chunk search; a DP over the rows then picks one per row, kept
  non-decreasing (frames go missing, they do not come back) and charged a penalty per switch, so
  the answer is a few clean steps rather than per-row noise.

Rows inside a cut that the detector cannot place are dropped rather than guessed: a label on the
wrong side of a cut is worse than no label. Because the model reads the labels' *current* frame
numbers, running this on a clip that is already right is a no-op - it finds rate 1, offset 0.

--preview writes a PNG to check by eye. Nothing is written without --apply, and what is written is
backed up first: labels/<clip>_ball.csv.before_realign_<timestamp> (every time, like every other
bulk write over a label file), and any video that turned out not to belong to the labels as
videos/<clip>_wrong_video.<ext>.bak.

Afterwards run ``ft.py check --audit --mark <clip>`` to confirm it, and to flag any frame still in
doubt so the click tool can take you to them.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple

import numpy as np

if __name__ == "__main__":          # run as a script: make the repository root importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from finetune.data import workspace as ws
from finetune.data.alignment import Evidence, Tracks, agreement
from finetune.data.detect import ball_detector, detect_points
from finetune.data.fileio import backup_copy
from finetune.data.realign_solve import Alignment, segments, solve_alignment, unplaceable_at_cuts
from finetune.data.labels import LabelRow, format_point, is_visible, read_label_rows, reviewed_frames, write_label_cells
from finetune.data.detector_metrics import DETECTION_CONF, scale_for_width
from finetune.data.workspace import VideoMeta, find_video, label_csv, require_video, video_meta

PREVIEW_TILES = 8            # frames shown on the --preview sheet ...
PREVIEW_CROP = 70            # ... each a crop this many pixels either side of the label ...
PREVIEW_TILE_PX = 220        # ... drawn this big



# --------------------------------------------------------------------------- the two sides

def detections(clip: str) -> Tuple[Evidence, VideoMeta]:
    """What the detector is confident about in the clip's video, and the video's facts."""
    video = require_video(clip)
    meta = video_meta(video)
    points = detect_points(ball_detector(ws.REPO_WEIGHTS), video, meta)
    confident = {frame: (x, y) for frame, (x, y, conf) in points.items() if conf >= DETECTION_CONF}
    return Evidence.of(confident, meta.total, scale_for_width(meta.width)), meta


def label_rows_of(clip: str) -> List[LabelRow]:
    """The clip's rows sorted by frame, exactly as the file has them (the sentinel included)."""
    return sorted(read_label_rows(label_csv(clip)))


def visible_tracks(entries: List[LabelRow], own_size: Tuple[int, int], meta: VideoMeta) -> Tracks:
    """The visible rows, scaled from the labels' own frame size onto the video's."""
    own_w, own_h = own_size
    visible = [entry for entry in entries if is_visible(entry.x, entry.y, own_w, own_h)]
    return Tracks.of([entry.frame for entry in visible], [entry.x * meta.width / own_w for entry in visible],
                     [entry.y * meta.height / own_h for entry in visible], scale_for_width(meta.width))


# --------------------------------------------------------------------------- writing it

def remapped_rows(entries: List[LabelRow], frames: np.ndarray, alignment: Alignment, reviewed: Set[int],
                  total: int, own_size: Tuple[int, int], video_size: Tuple[int, int]) -> List[Tuple[int, str, str]]:
    """The label file's rows re-timed onto the video: ``(video frame, x cell, y cell)`` for every row
    that lands inside the video and outside a cut, scaled from the labels' own frame size
    (``frames`` are the visible rows' label frames) to the video's. Raises ValueError rather than
    touch a row a person has reviewed."""
    own_w, own_h = own_size
    width, height = video_size
    per_frame = dict(zip(frames.tolist(), alignment.chosen.tolist()))
    drop_frames = {int(frames[i]) for i in alignment.dropped}
    if drop_frames & reviewed:
        raise ValueError("Alignment would drop a reviewed row; nothing changed")
    current = int(alignment.chosen[0])
    written = []
    for entry in entries:
        current = per_frame.get(entry.frame, current)
        if entry.frame in drop_frames:
            continue
        index = int(round(entry.frame * alignment.rate)) + current
        if entry.frame in reviewed and (index != entry.frame or not 0 <= index < total):
            raise ValueError("Alignment would move a reviewed row; nothing changed")
        if not 0 <= index < total:
            continue
        point = None if not is_visible(entry.x, entry.y, own_w, own_h) else (entry.x * width / own_w, entry.y * height / own_h)
        written.append((index, *format_point(point, width, height)))
    written.sort()
    if any(a[0] >= b[0] for a, b in zip(written, written[1:])):
        raise ValueError("Alignment maps two label rows onto one video frame; nothing changed")
    return written


def write_preview(video: Path, tracks: Tracks, alignment: Alignment, out: Path) -> Path:
    """A PNG of PREVIEW_TILES frames spread along the clip, each cropped around the label at the
    video frame the alignment maps it to, for checking the alignment by eye."""
    import cv2
    capture = cv2.VideoCapture(str(video))
    tiles = []
    for k in [int(j * (len(tracks.frames) - 1) / (PREVIEW_TILES - 1)) for j in range(PREVIEW_TILES)]:
        index = int(alignment.bases[k] + alignment.chosen[k])
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            continue
        x, y = int(float(tracks.xs[k])), int(float(tracks.ys[k]))
        crop = frame[max(0, y - PREVIEW_CROP):y + PREVIEW_CROP, max(0, x - PREVIEW_CROP):x + PREVIEW_CROP].copy()
        if crop.size == 0:
            crop = np.zeros((2 * PREVIEW_CROP, 2 * PREVIEW_CROP, 3), np.uint8)
        crop = cv2.resize(crop, (PREVIEW_TILE_PX, PREVIEW_TILE_PX), interpolation=cv2.INTER_NEAREST)
        cv2.circle(crop, (PREVIEW_TILE_PX // 2, PREVIEW_TILE_PX // 2), 22, (0, 255, 255), 2)
        cv2.putText(crop, f"label {tracks.frames[k]}->f{index}", (4, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(crop)
    capture.release()
    while len(tiles) < PREVIEW_TILES:
        tiles.append(np.zeros((PREVIEW_TILE_PX, PREVIEW_TILE_PX, 3), np.uint8))
    half = PREVIEW_TILES // 2
    cv2.imwrite(str(out), np.vstack([np.hstack(tiles[:half]), np.hstack(tiles[half:PREVIEW_TILES])]))
    return out


def swap_video(label_clip: str, video_clip: str) -> None:
    """Put the video the labels belong to under them, keeping the one that did not as a .bak."""
    source = require_video(video_clip)
    media = ws.VIDEOS / f"{label_clip}{source.suffix.lower()}"
    wrong = find_video(label_clip)
    if wrong is not None:
        orphan = wrong.with_name(f"{label_clip}_wrong_video{wrong.suffix}.bak")
        if not orphan.exists():
            wrong.rename(orphan)
            print(f"  [keep ] the video these labels did NOT belong to -> {orphan.name}")
    shutil.copyfile(source, media)
    print(f"  [media] {media.name} is now a copy of {source.name}")


def apply_alignment(entries: List[LabelRow], frames: np.ndarray, alignment: Alignment, reviewed: Set[int],
                    meta: VideoMeta, own_size: Tuple[int, int], label_clip: str, video_clip: str) -> int:
    """Rewrite the label file (backed up first) and, when the labels belong to a different video,
    swap that video in under them. Nothing is written if a reviewed row would move."""
    written = remapped_rows(entries, frames, alignment, reviewed, meta.total, own_size, (meta.width, meta.height))
    path = label_csv(label_clip)
    backup_copy(path, "realign")
    write_label_cells(path, written)
    print(f"\n  [write] {path.name}: {len(written)} of {len(entries)} rows kept")
    if video_clip != label_clip:
        swap_video(label_clip, video_clip)
    print("  next:  python finetune/ft.py check --audit --mark " + label_clip)
    return 0


# --------------------------------------------------------------------------- main

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="realign.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("label_clip", help="the clip whose labels are re-timed")
    parser.add_argument("video_clip", help="the clip whose video the labels belong to (the same clip, usually)")
    parser.add_argument("--preview", action="store_true", help="write finetune/realign_<clip>.png to check by eye")
    parser.add_argument("--apply", action="store_true", help="rewrite the label file (backed up first)")
    return parser.parse_args(argv)


def print_segments(runs: List[List[int]], rate: float) -> None:
    print(f"  {len(runs)} segment(s):")
    for lo, hi, offset in runs:
        print(f"      label frames {lo:5d}-{hi:<5d} -> video frames "
              f"{int(round(lo * rate)) + offset:5d}-{int(round(hi * rate)) + offset:<5d}  (offset {offset:+d})")


def own_frame_size(label_clip: str, video_clip: str, meta: VideoMeta) -> Tuple[int, int]:
    """The frame size the labels were made at: their own clip's video, or the target's."""
    if label_clip == video_clip:
        return meta.width, meta.height
    own = video_meta(require_video(label_clip))
    return own.width, own.height


def pinned_rows(reviewed: Set[int], tracks: Tracks) -> np.ndarray:
    """Which visible rows a person has reviewed in the label tool (``reviewed_frames``).

    Clicking the ball says "it is here on this frame" and SPACE says "it is right where it is",
    so either way that row stays put and the solve works around it."""
    pinned = np.array([f in reviewed for f in tracks.frames.tolist()], bool)
    if pinned.any():
        print(f"  {int(pinned.sum())} frame(s) you corrected by hand are pinned and will not move")
    return pinned


def report_alignment(best: Alignment, now: Tuple[float, int, float], runs: List[List[int]]) -> bool:
    """Print the solve's answer; True when it is no better than leaving the labels as they are."""
    print(f"  best: rate {best.rate:g}, agree {best.result[0]:.2f} on {best.result[1]} judged, median {best.result[2]:.1f} px")
    print_segments(runs, best.rate)
    return best.result[0] <= now[0] + 0.01 and len(runs) == 1 and best.rate == 1.0 and runs[0][2] == 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    label_clip, video_clip = args.label_clip, args.video_clip
    evidence, meta = detections(video_clip)
    own_size = own_frame_size(label_clip, video_clip, meta)
    entries = label_rows_of(label_clip)
    tracks = visible_tracks(entries, own_size, meta)
    if not len(tracks.frames):
        print("No visible labels provide alignment evidence; nothing changed")
        return 1
    print(f"{label_clip}: {len(entries)} label rows ({len(tracks.frames)} visible) against {video_clip} "
          f"({meta.total} frames, {meta.width}x{meta.height})")
    now = agreement(tracks, evidence, 0)
    print(f"  as it stands:  agree {now[0]:.2f} on {now[1]} judged, median {now[2]:.1f} px")

    reviewed = reviewed_frames(label_clip)          # a damaged review file is an error: no alignment
    if reviewed and label_clip != video_clip:
        print("Cannot replace the video under reviewed labels; resolve the source mapping manually first")
        return 1
    pinned = pinned_rows(reviewed, tracks)
    chunks = int(min(60, max(8, len(entries) // 60)))
    best = solve_alignment(tracks, evidence, reviewed, pinned, chunks)
    if best is None:
        print("  nothing in this video puts the labels on the ball - not salvageable this way")
        return 1
    runs = segments(tracks.frames, best.chosen)
    if report_alignment(best, now, runs):
        print("\n  already aligned - nothing to do")
        return 0

    if len(runs) > 1:
        best = best._replace(dropped=frozenset(unplaceable_at_cuts(tracks.at_rate(best.rate), evidence, best.chosen)))
    if best.dropped:
        print(f"  {len(best.dropped)} visible row(s) sit inside a cut the detector cannot place - dropped, not guessed")
    if args.preview:
        out = write_preview(require_video(video_clip), tracks, best, ws.WORKSPACE / f"realign_{label_clip}.png")
        print(f"  preview -> {out}")
    if not args.apply:
        print("\n  (dry run - pass --apply to write it)")
        return 0
    return apply_alignment(entries, tracks.frames, best, reviewed, meta, own_size, label_clip, video_clip)


if __name__ == "__main__":
    raise SystemExit(main())
