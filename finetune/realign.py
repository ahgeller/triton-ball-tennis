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
backed up first: labels/<clip>_ball.csv.premisaligned, and any video that turned out not to belong
to the labels as videos/<clip>_wrong_video.mp4.bak.

Afterwards run ``ft.py check --audit --mark <clip>`` to confirm it, and to flag any frame still in
doubt so the click tool can take you to them.
"""
from __future__ import annotations

import csv
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parent
ROOT = WS.parent
sys.path.insert(0, str(ROOT))

AGREE_PX = 15.0          # detector and label agree within this many 1080p pixels
FAST_PX = 6.0            # only a moving ball can tell one offset from the next
MIN_CONF = 0.5
COST_CAP = 60.0          # a label this far out tells us nothing beyond "wrong"
NO_DETECTION = 15.0      # cost where the detector said nothing: indifferent, not evidence
SWITCH_PENALTY = 400.0   # a cut must pay for itself in error saved before the DP will take it
RATES = (1.0, 0.5, 2.0)  # label frame -> video frame scaling worth trying


def detections(clip: str):
    """(frames, x, y, [width, height, total]) for every frame the detector is confident about."""
    import cv2
    from tennis_tracker.config import Config
    from tennis_tracker import detectors as backends
    video = WS / "videos" / f"{clip}.mp4"
    capture = cv2.VideoCapture(str(video))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    cfg = Config(conf=0.0, device="0", gridtracknet_prepass_background=False)
    detector = backends.GridTrackNetBallDetector(str(ROOT / "models" / "gridtracknet_weights_torch.npz"), cfg)
    detector.prepare_video(video, fps, width, height, total)
    frames, xs, ys = [], [], []
    for frame, found in enumerate(detector.precomputed):
        if found and found[0][1] >= MIN_CONF:
            box = found[0][0]
            frames.append(frame)
            xs.append((box[0] + box[2]) / 2.0)
            ys.append((box[1] + box[3]) / 2.0)
    return (np.array(frames, np.int32), np.array(xs, np.float32), np.array(ys, np.float32),
            np.array([width, height, total], np.int32))


def frame_size(clip: str):
    import cv2
    capture = cv2.VideoCapture(str(WS / "videos" / f"{clip}.mp4"))
    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    capture.release()
    return size


def anchors(clip: str):
    """Frames you have already corrected by hand, from the click tool's review file.

    Both kinds count. Clicking the ball says "it is here on this frame"; pressing SPACE says "the
    label is already right on this frame". Either way you have looked at that video frame and
    settled it, so the row is pinned where it is and the solve works around it: every frame you
    deal with pushes the cut past itself, and the ones you have not reached follow.
    """
    path = WS / "labels" / f"{clip}_ball.review.json"
    if not path.is_file():
        return set()
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot read review anchors from {path}; refusing alignment") from error
    return {int(f) for f in state.get("edited", [])} | {int(f) for f in state.get("reviewed", [])}


def read_labels(clip: str):
    """[(frame, x, y)] sorted by frame, exactly as the file has them."""
    with (WS / "labels" / f"{clip}_ball.csv").open(newline="", encoding="utf-8-sig") as handle:
        entries = [(int(re.search(r"(\d+)", row["frame"]).group(1)),
                    float(row["ball_x"]), float(row["ball_y"])) for row in csv.DictReader(handle)]
    entries.sort()
    return entries


def lookup_tables(det_frames, det_x, det_y, total):
    lut_x = np.full(total + 2, np.nan, np.float32)
    lut_y = np.full(total + 2, np.nan, np.float32)
    keep = det_frames < total
    lut_x[det_frames[keep]] = det_x[keep]
    lut_y[det_frames[keep]] = det_y[keep]
    return lut_x, lut_y


def agreement(bases, xs, ys, offsets, lut_x, lut_y, total, scale, fast):
    index = bases + offsets
    ok = (index >= 0) & (index < total)
    if ok.sum() == 0:
        return 0.0, 0, float("nan")
    dx, dy = lut_x[index[ok]], lut_y[index[ok]]
    seen = ~np.isnan(dx)
    if seen.sum() == 0:
        return 0.0, 0, float("nan")
    error = np.hypot(dx[seen] - xs[ok][seen], dy[seen] - ys[ok][seen]) * scale
    quick = fast[ok][seen]
    judged = error[quick] if quick.sum() >= 20 else error
    return float((judged <= AGREE_PX).mean()), int(judged.size), float(np.median(judged))


def best_offset(bases, xs, ys, fast, lut_x, lut_y, total, scale, low, high):
    best = (0.0, None)
    for offset in range(low, high + 1):
        index = bases + offset
        ok = (index >= 0) & (index < total)
        if ok.sum() < 15:
            continue
        dx, dy = lut_x[index[ok]], lut_y[index[ok]]
        seen = ~np.isnan(dx)
        if seen.sum() < 10:
            continue
        error = np.hypot(dx[seen] - xs[ok][seen], dy[seen] - ys[ok][seen]) * scale
        quick = fast[ok][seen]
        judged = error[quick] if quick.sum() >= 10 else error
        agree = float((judged <= AGREE_PX).mean())
        if agree > best[0]:
            best = (agree, offset)
    return best


def candidates(bases, xs, ys, fast, lut_x, lut_y, total, scale, chunks):
    """Offsets worth considering, from a coarse chunk-by-chunk best-offset search."""
    edges = np.linspace(0, len(bases), chunks + 1).astype(int)
    winners = set()
    for lo, hi in zip(edges, edges[1:]):
        if hi - lo < 20:
            continue
        agree, offset = best_offset(bases[lo:hi], xs[lo:hi], ys[lo:hi], fast[lo:hi],
                                    lut_x, lut_y, total, scale,
                                    -int(bases[-1]) - 1, total - int(bases[lo]))
        if offset is not None and agree >= 0.5:
            winners.update(range(offset - 5, offset + 6))
    return np.array(sorted(winners), np.int64)


def solve(bases, xs, ys, lut_x, lut_y, total, scale, offsets, pinned=None):
    """One offset per visible label, non-decreasing, via a DP that pays to change.

    pinned marks rows that must keep the frame number they already have (offset 0) because a human
    put them there.
    """
    n, k = len(bases), len(offsets)
    cost = np.empty((n, k), np.float64)
    for j, offset in enumerate(offsets):
        index = bases + offset
        ok = (index >= 0) & (index < total)
        column = np.full(n, COST_CAP)
        dx = np.full(n, np.nan)
        dy = np.full(n, np.nan)
        dx[ok] = lut_x[index[ok]]
        dy[ok] = lut_y[index[ok]]
        seen = ~np.isnan(dx)
        column[seen] = np.minimum(np.hypot(dx[seen] - xs[seen], dy[seen] - ys[seen]) * scale, COST_CAP)
        column[ok & ~seen] = NO_DETECTION
        cost[:, j] = column
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


def unplaceable_at_cuts(bases, chosen, xs, ys, lut_x, lut_y, total, scale):
    """Indices of visible labels sitting inside a cut the detector cannot pin down.

    Where the video jumps, only frames the detector actually saw fix which row it jumps on. Inside
    a run it said nothing about, the cut could be anywhere, so those rows get dropped.
    """
    index = np.clip(bases + chosen, 0, total - 1)
    inside = (bases + chosen >= 0) & (bases + chosen < total)
    dx, dy = lut_x[index], lut_y[index]
    error = np.hypot(dx - xs, dy - ys) * scale
    confirmed = inside & ~np.isnan(dx) & (error <= AGREE_PX)
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


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    label_clip, video_clip = sys.argv[1], sys.argv[2]
    det_frames, det_x, det_y, meta = detections(video_clip)
    width, height, total = int(meta[0]), int(meta[1]), int(meta[2])
    own_w, own_h = frame_size(label_clip) if label_clip != video_clip else (width, height)
    lut_x, lut_y = lookup_tables(det_frames, det_x, det_y, total)

    entries = read_labels(label_clip)
    rows = [(frame, x * width / own_w, y * height / own_h) for frame, x, y in entries]
    keep = [i for i, (_, x, y) in enumerate(rows)
            if not (entries[i][1] >= own_w * 0.95 and entries[i][2] <= own_h * 0.05)]
    frames = np.array([rows[i][0] for i in keep], np.int64)
    if not len(frames):
        print("No visible labels provide alignment evidence; nothing changed")
        return 1
    xs = np.array([rows[i][1] for i in keep], np.float32)
    ys = np.array([rows[i][2] for i in keep], np.float32)
    scale = 1920.0 / width
    step = np.hypot(np.diff(xs, prepend=xs[0]), np.diff(ys, prepend=ys[0])) * scale
    fast = step >= FAST_PX

    print(f"{label_clip}: {len(entries)} label rows ({len(frames)} visible) against {video_clip} "
          f"({total} frames, {width}x{height})")
    now = agreement(frames, xs, ys, np.zeros_like(frames), lut_x, lut_y, total, scale, fast)
    print(f"  as it stands:  agree {now[0]:.2f} on {now[1]} judged, median {now[2]:.1f} px")

    fixed_by_hand = anchors(label_clip)
    if fixed_by_hand and label_clip != video_clip:
        print("Cannot replace the video under reviewed labels; resolve the source mapping manually first")
        return 1
    pinned = np.array([f in fixed_by_hand for f in frames.tolist()], bool)
    if pinned.any():
        print(f"  {int(pinned.sum())} frame(s) you corrected by hand are pinned and will not move")

    chunks = int(min(60, max(8, len(entries) // 60)))
    best = None
    for rate in RATES:
        if fixed_by_hand and rate != 1.0:
            continue  # a reviewed source-frame index is an exact temporal anchor
        bases = np.rint(frames * rate).astype(np.int64)
        offsets = candidates(bases, xs, ys, fast, lut_x, lut_y, total, scale, chunks)
        if pinned.any() and rate == 1.0:
            offsets = np.array(sorted(set(offsets.tolist()) | {0}), np.int64)
        if offsets.size == 0:
            continue
        chosen = solve(bases, xs, ys, lut_x, lut_y, total, scale, offsets,
                       pinned if rate == 1.0 else None)
        # Invisible reviewed rows are not in the visual solve, but are equally
        # authoritative. Reject any candidate that would shift their segment.
        if any(int(chosen[max(0, int(np.searchsorted(frames, f, side="right")) - 1)]) != 0
               for f in fixed_by_hand):
            continue
        result = agreement(bases, xs, ys, chosen, lut_x, lut_y, total, scale, fast)
        print(f"  rate {rate:g}: {offsets.size} candidate offsets in {offsets.min():+d}..{offsets.max():+d}"
              f"  -> agree {result[0]:.2f}, median {result[2]:.1f} px")
        if best is None or result[0] > best[0][0]:
            best = (result, rate, bases, chosen)
    if best is None:
        print("  nothing in this video puts the labels on the ball - not salvageable this way")
        return 1
    result, rate, bases, chosen = best

    runs = []
    for frame, offset in zip(frames, chosen):
        if not runs or runs[-1][2] != offset:
            runs.append([frame, frame, offset])
        else:
            runs[-1][1] = frame
    print(f"  best: rate {rate:g}, agree {result[0]:.2f} on {result[1]} judged, median {result[2]:.1f} px")
    print(f"  {len(runs)} segment(s):")
    for lo, hi, offset in runs:
        print(f"      label frames {lo:5d}-{hi:<5d} -> video frames "
              f"{int(round(lo * rate)) + offset:5d}-{int(round(hi * rate)) + offset:<5d}  (offset {offset:+d})")
    if result[0] <= now[0] + 0.01 and len(runs) == 1 and rate == 1.0 and runs[0][2] == 0:
        print("\n  already aligned - nothing to do")
        return 0

    dropped = unplaceable_at_cuts(bases, chosen, xs, ys, lut_x, lut_y, total, scale) if len(runs) > 1 else set()
    if dropped:
        print(f"  {len(dropped)} visible row(s) sit inside a cut the detector cannot place - dropped, not guessed")

    if "--preview" in sys.argv:
        import cv2
        capture = cv2.VideoCapture(str(WS / "videos" / f"{video_clip}.mp4"))
        tiles = []
        for k in [int(j * (len(frames) - 1) / 7) for j in range(8)]:
            index = int(bases[k] + chosen[k])
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                continue
            x, y = float(xs[k]), float(ys[k])
            crop = frame[max(0, int(y) - 70):int(y) + 70, max(0, int(x) - 70):int(x) + 70].copy()
            if crop.size == 0:
                crop = np.zeros((140, 140, 3), np.uint8)
            crop = cv2.resize(crop, (220, 220), interpolation=cv2.INTER_NEAREST)
            cv2.circle(crop, (110, 110), 22, (0, 255, 255), 2)
            cv2.putText(crop, f"label {frames[k]}->f{index}", (4, 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (0, 255, 255), 1, cv2.LINE_AA)
            tiles.append(crop)
        capture.release()
        while len(tiles) < 8:
            tiles.append(np.zeros((220, 220, 3), np.uint8))
        out = WS / f"realign_{label_clip}.png"
        cv2.imwrite(str(out), np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:8])]))
        print(f"  preview -> {out}")

    if "--apply" not in sys.argv:
        print("\n  (dry run - pass --apply to write it)")
        return 0

    per_frame = dict(zip(frames.tolist(), chosen.tolist()))
    drop_frames = {int(frames[i]) for i in dropped}
    if drop_frames & fixed_by_hand:
        raise ValueError("Alignment would drop a reviewed row; nothing changed")
    current = int(chosen[0])
    written = []
    for position, (frame, x, y) in enumerate(entries):
        current = per_frame.get(frame, current)
        if frame in drop_frames:
            continue
        index = int(round(frame * rate)) + current
        if frame in fixed_by_hand and (index != frame or not 0 <= index < total):
            raise ValueError("Alignment would move a reviewed row; nothing changed")
        if not 0 <= index < total:
            continue
        invisible = x >= own_w * 0.95 and y <= own_h * 0.05
        written.append((index, width - 1, 0) if invisible
                       else (index, f"{x * width / own_w:.2f}", f"{y * height / own_h:.2f}"))
    written.sort()
    assert all(a[0] < b[0] for a, b in zip(written, written[1:])), "frame numbers must stay increasing"

    path = WS / "labels" / f"{label_clip}_ball.csv"
    backup = path.with_suffix(".csv.premisaligned")
    if not backup.exists():
        shutil.copyfile(path, backup)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "ball_x", "ball_y"])
        for index, x, y in written:
            writer.writerow([f"frame_{index:03d}", x, y])
    temporary.replace(path)
    print(f"\n  [write] {path.name}: {len(written)} of {len(entries)} rows kept")

    if video_clip != label_clip:
        media = WS / "videos" / f"{label_clip}.mp4"
        orphan = WS / "videos" / f"{label_clip}_wrong_video.mp4.bak"
        if media.exists() and not orphan.exists():
            media.rename(orphan)
            print(f"  [keep ] the video these labels did NOT belong to -> {orphan.name}")
        shutil.copyfile(WS / "videos" / f"{video_clip}.mp4", media)
        print(f"  [media] {label_clip}.mp4 is now a copy of {video_clip}.mp4")
    print("  next:  python finetune/ft.py check --audit --mark " + label_clip)
    return 0


if __name__ == "__main__":
    sys.exit(main())
