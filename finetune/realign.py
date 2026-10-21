"""Put a clip's labels back in time with its video.

    python finetune/realign.py <label clip> <video clip> [--preview] [--apply]

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
from typing import FrozenSet, List, NamedTuple, Optional, Sequence, Set, Tuple

import numpy as np

if __name__ == "__main__":          # run as a script: make the repository root importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from finetune import workspace as ws
from finetune.alignment import Evidence, Tracks, agreement, errors_at, judged_rows
from finetune.detect import ball_detector, detect_points
from finetune.fileio import backup_copy
from finetune.labels import LabelRow, format_point, is_visible, read_label_rows, reviewed_frames, write_label_cells
from finetune.scoring import AGREE_PX, DETECTION_CONF, scale_for_width
from finetune.workspace import VideoMeta, find_video, label_csv, require_video, video_meta

PREVIEW_TILES = 8            # frames shown on the --preview sheet ...
PREVIEW_CROP = 70            # ... each a crop this many pixels either side of the label ...
PREVIEW_TILE_PX = 220        # ... drawn this big

COST_CAP = 60.0          # a label this far out tells us nothing beyond "wrong"
NO_DETECTION = 15.0      # cost where the detector said nothing: indifferent, not evidence
SWITCH_PENALTY = 400.0   # a cut must pay for itself in error saved before the DP will take it
RATES = (1.0, 0.5, 2.0)  # label frame -> video frame scaling worth trying
CHUNK_MIN_ROWS = 20      # a chunk of fewer rows than this is not searched ...
CHUNK_MIN_IN_VIDEO = 15  # ... nor an offset that leaves fewer rows than this inside the video ...
CHUNK_MIN_SEEN = 10      # ... or fewer than this the detector saw; fast rows are judged from this many
CHUNK_MIN_AGREE = 0.5    # a chunk's best offset must agree this well to nominate candidates
CANDIDATE_SPREAD = 5     # ... which are the offsets within this many frames of it


class Alignment(NamedTuple):
    """One answer of the solve: how well it agrees, the rate, the label frames at that rate
    (``bases``), the offset chosen per visible row, and the rows dropped as unplaceable."""
    result: Tuple[float, int, float]      # (agreement, rows judged, median px)
    rate: float
    bases: np.ndarray
    chosen: np.ndarray
    dropped: FrozenSet[int] = frozenset()   # indices into the visible rows


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


# --------------------------------------------------------------------------- the solve

def best_offset(tracks: Tracks, evidence: Evidence, low: int, high: int) -> Tuple[float, Optional[int]]:
    """The constant offset in low..high at which the detector agrees with these rows best."""
    best = (0.0, None)
    for offset in range(low, high + 1):
        inside = ((tracks.frames + offset >= 0) & (tracks.frames + offset < evidence.total)).sum()
        if inside < CHUNK_MIN_IN_VIDEO:
            continue
        seen, error = errors_at(tracks, evidence, offset)
        if seen.sum() < CHUNK_MIN_SEEN:
            continue
        judged = error[judged_rows(tracks, seen, CHUNK_MIN_SEEN)]
        agree = float((judged <= AGREE_PX).mean())
        if agree > best[0]:
            best = (agree, offset)
    return best


def candidates(tracks: Tracks, evidence: Evidence, chunks: int) -> np.ndarray:
    """Offsets worth considering, from a coarse chunk-by-chunk best-offset search."""
    edges = np.linspace(0, len(tracks.frames), chunks + 1).astype(int)
    winners = set()
    for lo, hi in zip(edges, edges[1:]):
        if hi - lo < CHUNK_MIN_ROWS:
            continue
        agree, offset = best_offset(tracks.rows(lo, hi), evidence,
                                    -int(tracks.frames[-1]) - 1, evidence.total - int(tracks.frames[lo]))
        if offset is not None and agree >= CHUNK_MIN_AGREE:
            winners.update(range(offset - CANDIDATE_SPREAD, offset + CANDIDATE_SPREAD + 1))
    return np.array(sorted(winners), np.int64)


def _offset_costs(tracks: Tracks, evidence: Evidence, offsets: np.ndarray) -> np.ndarray:
    """cost[row, j]: the label's distance from the detection at offsets[j], capped; NO_DETECTION
    where the detector said nothing; COST_CAP outside the video."""
    n = len(tracks.frames)
    cost = np.empty((n, len(offsets)), np.float64)
    for j, offset in enumerate(offsets):
        index = tracks.frames + offset
        ok = (index >= 0) & (index < evidence.total)
        column = np.full(n, COST_CAP)
        dx = np.full(n, np.nan)
        dy = np.full(n, np.nan)
        dx[ok] = evidence.lut_x[index[ok]]
        dy[ok] = evidence.lut_y[index[ok]]
        seen = ~np.isnan(dx)
        column[seen] = np.minimum(np.hypot(dx[seen] - tracks.xs[seen], dy[seen] - tracks.ys[seen]) * evidence.scale, COST_CAP)
        column[ok & ~seen] = NO_DETECTION
        cost[:, j] = column
    return cost


def solve(tracks: Tracks, evidence: Evidence, offsets: np.ndarray, pinned: Optional[np.ndarray] = None) -> np.ndarray:
    """One offset per visible label, non-decreasing, via a DP that pays to change.

    pinned marks rows that must keep the frame number they already have (offset 0) because a human
    put them there.
    """
    n, k = len(tracks.frames), len(offsets)
    cost = _offset_costs(tracks, evidence, offsets)
    if pinned is not None and pinned.any():
        if 0 not in offsets:
            raise ValueError("Pinned rows require offset zero")
        zero = int(np.flatnonzero(offsets == 0)[0])
        cost[pinned, :] = np.inf
        cost[pinned, zero] = 0.0
    best = cost[0].copy()
    back = np.zeros((n, k), np.int32)
    for i in range(1, n):
        running, argmin = np.inf, 0
        switch = np.empty(k)
        source = np.empty(k, np.int32)
        for j in range(k):                    # prefix minimum: only smaller offsets may lead here
            if j > 0 and best[j - 1] < running:
                running, argmin = best[j - 1], j - 1
            switch[j] = running + SWITCH_PENALTY
            source[j] = argmin
        take = switch < best
        back[i] = np.where(take, source, np.arange(k))
        best = cost[i] + np.where(take, switch, best)
    path = np.empty(n, np.int32)
    if not np.isfinite(best).any():
        raise ValueError("No alignment can preserve all pinned rows")
    path[-1] = int(np.argmin(best))
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return offsets[path]


def unplaceable_at_cuts(tracks: Tracks, evidence: Evidence, chosen: np.ndarray) -> Set[int]:
    """Indices of visible labels sitting inside a cut the detector cannot pin down.

    Where the video jumps, only frames the detector actually saw fix which row it jumps on. Inside
    a run it said nothing about, the cut could be anywhere, so those rows get dropped.
    """
    seen, error = errors_at(tracks, evidence, chosen)
    confirmed = seen & (error <= AGREE_PX)
    drop = set()
    for i in range(1, len(chosen)):
        if chosen[i] == chosen[i - 1]:
            continue
        before = i - 1
        while before >= 0 and not confirmed[before]:
            before -= 1
        after = i
        while after < len(chosen) and not confirmed[after]:
            after += 1
        drop.update(range(before + 1, min(after, len(chosen))))
    return drop


def solve_alignment(tracks: Tracks, evidence: Evidence, fixed_by_hand: Set[int], pinned: np.ndarray,
                    chunks: int) -> Optional[Alignment]:
    """The best alignment over RATES, reporting each rate tried; None when no rate puts the labels
    on the ball. Reviewed rows allow rate 1 only and keep offset 0."""
    best = None
    for rate in RATES:
        if fixed_by_hand and rate != 1.0:
            continue  # a reviewed source-frame index is an exact temporal anchor
        based = tracks.at_rate(rate)
        offsets = candidates(based, evidence, chunks)
        if pinned.any() and rate == 1.0:
            offsets = np.array(sorted(set(offsets.tolist()) | {0}), np.int64)
        if offsets.size == 0:
            continue
        chosen = solve(based, evidence, offsets, pinned if rate == 1.0 else None)
        # Invisible reviewed rows are not in the visual solve, but are equally
        # authoritative. Reject any candidate that would shift their segment.
        if any(int(chosen[max(0, int(np.searchsorted(tracks.frames, f, side="right")) - 1)]) != 0
               for f in fixed_by_hand):
            continue
        result = agreement(based, evidence, chosen)
        print(f"  rate {rate:g}: {offsets.size} candidate offsets in {offsets.min():+d}..{offsets.max():+d}"
              f"  -> agree {result[0]:.2f}, median {result[2]:.1f} px")
        if best is None or result[0] > best.result[0]:
            best = Alignment(result, rate, based.frames, chosen)
    return best


def segments(frames: np.ndarray, chosen: np.ndarray) -> List[List[int]]:
    """``[first label frame, last label frame, offset]`` for each run of one offset along the clip."""
    runs: List[List[int]] = []
    for frame, offset in zip(frames, chosen):
        if not runs or runs[-1][2] != offset:
            runs.append([frame, frame, offset])
        else:
            runs[-1][1] = frame
    return runs


# --------------------------------------------------------------------------- writing it

def remapped_rows(entries: List[LabelRow], frames: np.ndarray, alignment: Alignment, fixed_by_hand: Set[int],
                  total: int, own_size: Tuple[int, int], video_size: Tuple[int, int]) -> List[Tuple[int, str, str]]:
    """The label file's rows re-timed onto the video: ``(video frame, x cell, y cell)`` for every row
    that lands inside the video and outside a cut, scaled from the labels' own frame size
    (``frames`` are the visible rows' label frames) to the video's. Raises ValueError rather than
    touch a row a person has reviewed."""
    own_w, own_h = own_size
    width, height = video_size
    per_frame = dict(zip(frames.tolist(), alignment.chosen.tolist()))
    drop_frames = {int(frames[i]) for i in alignment.dropped}
    if drop_frames & fixed_by_hand:
        raise ValueError("Alignment would drop a reviewed row; nothing changed")
    current = int(alignment.chosen[0])
    written = []
    for entry in entries:
        current = per_frame.get(entry.frame, current)
        if entry.frame in drop_frames:
            continue
        index = int(round(entry.frame * alignment.rate)) + current
        if entry.frame in fixed_by_hand and (index != entry.frame or not 0 <= index < total):
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


def apply_alignment(entries: List[LabelRow], frames: np.ndarray, alignment: Alignment, fixed_by_hand: Set[int],
                    meta: VideoMeta, own_size: Tuple[int, int], label_clip: str, video_clip: str) -> int:
    """Rewrite the label file (backed up first) and, when the labels belong to a different video,
    swap that video in under them. Nothing is written if a reviewed row would move."""
    written = remapped_rows(entries, frames, alignment, fixed_by_hand, meta.total, own_size, (meta.width, meta.height))
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    label_clip, video_clip = args.label_clip, args.video_clip
    evidence, meta = detections(video_clip)
    own_size = (meta.width, meta.height)
    if label_clip != video_clip:
        own = video_meta(require_video(label_clip))
        own_size = (own.width, own.height)

    entries = label_rows_of(label_clip)
    tracks = visible_tracks(entries, own_size, meta)
    if not len(tracks.frames):
        print("No visible labels provide alignment evidence; nothing changed")
        return 1
    print(f"{label_clip}: {len(entries)} label rows ({len(tracks.frames)} visible) against {video_clip} "
          f"({meta.total} frames, {meta.width}x{meta.height})")
    now = agreement(tracks, evidence, 0)
    print(f"  as it stands:  agree {now[0]:.2f} on {now[1]} judged, median {now[2]:.1f} px")

    # Frames a person has reviewed in the label tool are pinned: clicking the ball says "it is here
    # on this frame" and SPACE says "it is right where it is", so either way that row stays put and
    # the solve works around it. A damaged review file is an error: no alignment.
    fixed_by_hand = reviewed_frames(label_clip)
    if fixed_by_hand and label_clip != video_clip:
        print("Cannot replace the video under reviewed labels; resolve the source mapping manually first")
        return 1
    pinned = np.array([f in fixed_by_hand for f in tracks.frames.tolist()], bool)
    if pinned.any():
        print(f"  {int(pinned.sum())} frame(s) you corrected by hand are pinned and will not move")

    chunks = int(min(60, max(8, len(entries) // 60)))
    best = solve_alignment(tracks, evidence, fixed_by_hand, pinned, chunks)
    if best is None:
        print("  nothing in this video puts the labels on the ball - not salvageable this way")
        return 1
    runs = segments(tracks.frames, best.chosen)
    print(f"  best: rate {best.rate:g}, agree {best.result[0]:.2f} on {best.result[1]} judged, median {best.result[2]:.1f} px")
    print_segments(runs, best.rate)
    if best.result[0] <= now[0] + 0.01 and len(runs) == 1 and best.rate == 1.0 and runs[0][2] == 0:
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
    return apply_alignment(entries, tracks.frames, best, fixed_by_hand, meta, own_size, label_clip, video_clip)


if __name__ == "__main__":
    raise SystemExit(main())
