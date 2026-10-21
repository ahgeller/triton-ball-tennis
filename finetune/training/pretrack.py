"""Pre-label new videos with the ball detector so you only correct, never click from scratch.

Drop videos into ``finetune/videos``, run this, and every clip without a
finished label file gets ``labels/<clip>_ball.csv.draft`` at the cadence the
model trains on (every 2nd frame of a 60 FPS clip, every frame at 30 FPS).

Two draft sources:

  raw       (default) the detector's own best guess on *every* label frame,
            with its confidence — the tool shows the point even when it is
            unsure so you accept or fix, never hunt.
  pipeline  the full tracker output (selector + fills); fewer wrong points,
            but frames the selector rejected come up empty.

Invisible/unknown frames are parked in the top-right corner (``W-1, 0``),
the convention the trainer and evaluate_archive.py already read.

    python finetune/training/pretrack.py                          # all unlabelled clips, raw GridTrackNet
    python finetune/training/pretrack.py --clips rally7           # one clip
    python finetune/training/pretrack.py --weights my_finetune.npz
    python finetune/training/pretrack.py --source pipeline
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

if __name__ == "__main__":          # run as a script: make the repository root importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from finetune.data import workspace as ws
from finetune.data.labels import DraftRow, write_draft as write_draft_rows
from finetune.data.workspace import VideoMeta, draft_csv, label_csv, video_meta, videos_in
from finetune.data.detect import ball_detector, detect_points
from tennis_tracker.frame_rate import require_phase_stride

RAW_DRAFT_CONF = 0.25          # raw drafts keep guesses down to this confidence, so nearly every frame has one
PIPELINE_CONF = 0.50           # the tracker's own default threshold

Guess = Tuple[float, float, str, float]     # x, y, where it came from, confidence
NO_GUESS = DraftRow(None, "none", 0.0)      # the draft row of a frame the detector had nothing for


def write_draft(rows_by_frame: Mapping[int, Optional[Guess]], draft_path: Path, stride: int,
                width: int, height: int, total: int) -> int:
    """Write one row per label frame (every `stride` frames from 0) through the label format's own
    draft writer: the guess where there is one, "no ball" otherwise. Returns how many frames got
    a guess, and says when a guess sits in the corner the files reserve for "no ball"."""
    rows = {}
    for frame in range(0, total, stride):
        guess = rows_by_frame.get(frame)
        rows[frame] = DraftRow((guess[0], guess[1]), guess[2], guess[3]) if guess else NO_GUESS
    guessed, parked = write_draft_rows(draft_path, rows, width, height)
    if parked:
        print(f"[warn] {draft_path.name}: {parked} guess(es) sit in the top-right corner the label files "
              f"reserve for 'no ball' and will read back as no ball")
    return guessed


def raw_rows(video: Path, weights: Path, conf: float, device: str, meta: VideoMeta) -> Dict[int, Guess]:
    """The detector's own best guess per frame, unshifted (no y offset): the draft is what it saw."""
    detector = ball_detector(weights, conf=conf, device=device)
    return {frame: (x, y, "det", score) for frame, (x, y, score) in detect_points(detector, video, meta).items()}


def pipeline_rows(video: Path, weights: Path, conf: Optional[float], device: str, json_path: Path) -> Dict[int, Guess]:
    """The full tracker's answer per frame it filled (selector + fills), as draft guesses."""
    from tennis_tracker.headless import run_headless     # the pipeline: CUDA and TensorRT, only for this source

    rows = run_headless(video, model_path=weights, conf=PIPELINE_CONF if conf is None else conf, device=device,
                        tracking_json=json_path).frame_rows()
    return {
        row["frame"]: (row["x"], row["y"], row.get("source", "det"), float(row.get("conf", 0.0)))
        for row in rows if row.get("present")
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clips", nargs="*", help="Clip stems to (re)track; default: every video without a final label CSV")
    parser.add_argument("--source", choices=("raw", "pipeline"), default="raw")
    parser.add_argument("--weights", type=Path, default=ws.current().repo_weights, help="GridTrackNet weights (.npz)")
    parser.add_argument("--conf", type=float, default=None,
                        help=f"Draft confidence threshold (raw default {RAW_DRAFT_CONF} so nearly every frame gets a guess)")
    parser.add_argument("--device", default="0")
    parser.add_argument("--force", action="store_true", help="Re-track even if a final label CSV exists")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Draft every video asked for into the workspace's labels folder (the workspace as it is
    now: ``TENNIS_WORKSPACE`` relocates it)."""
    args = build_parser().parse_args(argv)

    layout = ws.current()
    layout.videos.mkdir(exist_ok=True)
    layout.labels.mkdir(exist_ok=True)
    layout.runs.mkdir(exist_ok=True)
    weights = args.weights.resolve()
    if not weights.is_file():
        print(f"Weights not found: {weights}")
        return 1

    videos = list(videos_in(layout.videos).values())
    if args.clips:
        videos = [p for p in videos if p.stem in set(args.clips)]
    if not videos:
        print(f"No videos to track in {layout.videos}")
        return 1

    for video in videos:
        final = label_csv(video.stem)
        draft = draft_csv(video.stem)
        if final.is_file() and not args.force:
            print(f"[skip] {video.name}: {final.name} already exists (use --force to redo)")
            continue
        meta = video_meta(video)
        stride = require_phase_stride(meta.fps)
        print(f"[track] {video.name}: {meta.width}x{meta.height} @ {meta.fps:.2f} FPS, {meta.total} frames, "
              f"labelling every {stride} frame(s), source={args.source}")
        if args.source == "raw":
            rows = raw_rows(video, weights, RAW_DRAFT_CONF if args.conf is None else args.conf, args.device, meta)
        else:
            rows = pipeline_rows(video, weights, args.conf, args.device, layout.runs / f"{video.stem}_{weights.stem}.json")
        visible = write_draft(rows, draft, stride, meta.width, meta.height, meta.total)
        print(f"[draft] {draft.name}: {visible} guessed frames of {len(range(0, meta.total, stride))} label frames "
              f"-> python finetune/labeling/label_tool.py {video.stem}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
