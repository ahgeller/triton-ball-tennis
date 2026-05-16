#!/usr/bin/env python3
"""Pick high-value frames for hand labeling, extract PNGs, pre-fill annotation JSON.

Strategy: a frame is "valuable" if the pipeline used a non-detection source
(motion/carry/interp/guide), if it sits at a source transition, or if the local
velocity changed sharply (likely hit/bounce). Falls back to evenly-spaced
samples if not enough valuable frames exist.

Output:
  <out_dir>/frame_NNNNNN.png    - extracted frames (overlay-free, original video)
  <out_dir>/<clip>_starter.json - annotation JSON pre-filled with pipeline (x,y).
                                  User opens each PNG, checks the (x,y), and
                                  edits if wrong or marks visible=false.

Usage:
  python tools/extract_label_frames.py `
    --tracking-json output_videos/pomona_baseline_tracking.json `
    --video "input_videos/PomonaPitzer Women vs. UCSD-cut-merged-1773082079341.mp4" `
    --out-dir validation/labels/pomona_baseline/ `
    --n-frames 80
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HIGH_VALUE_SOURCES = {"motion", "carry", "interp", "guide"}


def _load_tracking(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    frames = data.get("frames", [])
    if not isinstance(frames, list):
        raise ValueError(f"{path} has no 'frames' list")
    return frames, data


def _is_present(row: Dict[str, Any]) -> bool:
    return bool(row.get("present")) and row.get("x") is not None and row.get("y") is not None


def _score_frame(frame: Dict[str, Any], prev: Optional[Dict[str, Any]],
                 next_: Optional[Dict[str, Any]]) -> float:
    """Higher score = more valuable to label."""
    score = 0.0
    if not _is_present(frame):
        return 0.0

    src = str(frame.get("source", ""))
    if src in HIGH_VALUE_SOURCES:
        score += 10.0
    if src == "det":
        score += 1.0  # det frames are still useful to verify, just less

    # Source transition (prev or next has different source)
    for neighbor in (prev, next_):
        if neighbor and _is_present(neighbor):
            if str(neighbor.get("source", "")) != src:
                score += 6.0
                break

    # Sharp local velocity change -> likely hit/bounce
    if prev and next_ and _is_present(prev) and _is_present(next_):
        vx_in = float(frame["x"]) - float(prev["x"])
        vy_in = float(frame["y"]) - float(prev["y"])
        vx_out = float(next_["x"]) - float(frame["x"])
        vy_out = float(next_["y"]) - float(frame["y"])
        accel = math.hypot(vx_out - vx_in, vy_out - vy_in)
        # Bigger jerk = more valuable. Cap so a single huge spike doesn't dominate.
        score += min(accel / 8.0, 8.0)

    # Low confidence is interesting
    conf = float(frame.get("conf", 0.0) or 0.0)
    if conf < 0.3:
        score += 2.0

    return score


def _pick_frames(frames: List[Dict[str, Any]], n: int) -> List[int]:
    """Return up to n frame indices chosen for labeling."""
    if not frames:
        return []
    scored: List[Tuple[float, int]] = []
    for i, row in enumerate(frames):
        prev = frames[i - 1] if i > 0 else None
        nxt = frames[i + 1] if i + 1 < len(frames) else None
        s = _score_frame(row, prev, nxt)
        if s > 0.0:
            scored.append((s, int(row.get("frame", i))))
    scored.sort(key=lambda x: x[0], reverse=True)

    picked: List[int] = []
    picked_set: set = set()
    # Enforce a minimum spacing so labels spread across the clip.
    min_spacing = max(2, len(frames) // max(n * 4, 1))
    for _, fidx in scored:
        if any(abs(fidx - p) < min_spacing for p in picked):
            continue
        picked.append(fidx)
        picked_set.add(fidx)
        if len(picked) >= n:
            break

    # Top up with evenly-spaced frames if not enough valuable ones.
    if len(picked) < n:
        step = max(1, len(frames) // max(n - len(picked), 1))
        for i in range(0, len(frames), step):
            fidx = int(frames[i].get("frame", i))
            if fidx in picked_set:
                continue
            picked.append(fidx)
            picked_set.add(fidx)
            if len(picked) >= n:
                break

    picked.sort()
    return picked


def _extract_pngs(video_path: Path, frame_indices: List[int], out_dir: Path) -> List[int]:
    """Extract PNGs for the given frame indices. Returns indices actually written."""
    import cv2  # deferred so --help works without the conda env
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    written: List[int] = []
    try:
        wanted = sorted(set(int(i) for i in frame_indices))
        # Single sequential pass is faster than per-frame seeks for large lists.
        target_iter = iter(wanted)
        try:
            target = next(target_iter)
        except StopIteration:
            return written

        cur = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if cur == target:
                fp = out_dir / f"frame_{cur:06d}.png"
                cv2.imwrite(str(fp), frame)
                written.append(cur)
                try:
                    target = next(target_iter)
                except StopIteration:
                    break
            elif cur > target:
                # Past the target somehow (shouldn't happen with monotonic read)
                try:
                    target = next(target_iter)
                except StopIteration:
                    break
            cur += 1
    finally:
        cap.release()
    return written


def _starter_annotation(
    video_rel: str,
    frames: List[Dict[str, Any]],
    picked: List[int],
) -> Dict[str, Any]:
    by_frame = {int(row.get("frame", -1)): row for row in frames}
    ball_rows: List[Dict[str, Any]] = []
    for fidx in picked:
        row = by_frame.get(int(fidx))
        if row and _is_present(row):
            ball_rows.append({
                "frame": int(fidx),
                "x": round(float(row["x"]), 2),
                "y": round(float(row["y"]), 2),
                "visible": True,
                "_pipeline_source": str(row.get("source", "")),
                "_pipeline_conf": round(float(row.get("conf", 0.0) or 0.0), 3),
                "_review": "VERIFY",
            })
        else:
            ball_rows.append({
                "frame": int(fidx),
                "visible": False,
                "_review": "VERIFY",
            })
    return {
        "_comment": "Starter annotation generated by tools/extract_label_frames.py. "
                    "For each frame: open the matching PNG, check (x,y) against the ball. "
                    "If correct, delete _review/_pipeline_* fields. If wrong, edit x/y. "
                    "If ball is not visible, set visible=false and remove x/y. "
                    "Then rename this file from <clip>_starter.json to <clip>.json.",
        "video": video_rel,
        "ball": ball_rows,
        "events": [],
        "ignore_ranges": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Pick label frames and pre-fill annotation JSON.")
    parser.add_argument("--tracking-json", required=True,
                        help="Path to tracking.json produced by now_main.py")
    parser.add_argument("--video", required=True,
                        help="Path to the source video (annotation 'video' field)")
    parser.add_argument("--out-dir", required=True,
                        help="Directory to write frame PNGs and starter annotation JSON")
    parser.add_argument("--n-frames", type=int, default=80,
                        help="Target number of frames to extract (default 80)")
    parser.add_argument("--clip-name", default=None,
                        help="Clip identifier used in the starter JSON filename "
                             "(defaults to the video stem)")
    args = parser.parse_args()

    tj_path = Path(args.tracking_json)
    video_path = Path(args.video)
    out_dir = Path(args.out_dir)
    if not tj_path.exists():
        print(f"[label_frames] tracking JSON not found: {tj_path}", file=sys.stderr)
        return 2
    if not video_path.exists():
        print(f"[label_frames] video not found: {video_path}", file=sys.stderr)
        return 2

    frames, _data = _load_tracking(tj_path)
    if not frames:
        print(f"[label_frames] tracking JSON has zero frames", file=sys.stderr)
        return 2

    picked = _pick_frames(frames, max(1, int(args.n_frames)))
    print(f"[label_frames] selected {len(picked)} frames "
          f"(target {args.n_frames}; first 10: {picked[:10]}...)")

    written = _extract_pngs(video_path, picked, out_dir)
    print(f"[label_frames] wrote {len(written)} PNGs to {out_dir}")

    clip = args.clip_name or video_path.stem
    starter = _starter_annotation(str(video_path).replace("\\", "/"), frames, written)
    starter_path = out_dir / f"{clip}_starter.json"
    with starter_path.open("w", encoding="utf-8") as f:
        json.dump(starter, f, indent=2)
    print(f"[label_frames] wrote starter annotation: {starter_path}")
    print(f"[label_frames] when done labeling, move it to "
          f"validation/annotations/{clip}.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
