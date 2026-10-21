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
from typing import Dict, Optional, Tuple

if __name__ == "__main__":          # run as a script: make the repository root importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from finetune.training.detect import ball_detector, detect_points
from finetune.data.fileio import atomic_csv
from finetune.data.labels import COLUMNS, frame_name, sentinel
from finetune.data.workspace import (LABELS, REPO_WEIGHTS, ROOT, RUNS, VIDEOS, VideoMeta, draft_csv, label_csv,
                                     video_meta, videos_in)
from tennis_tracker.frame_rate import require_phase_stride

RAW_DRAFT_CONF = 0.25          # raw drafts keep guesses down to this confidence, so nearly every frame has one
PIPELINE_CONF = 0.50           # the tracker's own default threshold
DRAFT_COLUMNS = (*COLUMNS, "source", "conf")

DraftRow = Tuple[float, float, str, float]     # x, y, where it came from, confidence


def write_draft(rows_by_frame: Dict[int, Optional[DraftRow]], draft_path: Path, stride: int,
                width: int, height: int, total: int) -> int:
    """rows_by_frame: frame -> (x, y, source, conf) or None. Returns how many frames got a guess."""
    visible = 0
    with atomic_csv(draft_path) as writer:
        writer.writerow(list(DRAFT_COLUMNS))
        for frame in range(0, total, stride):
            row = rows_by_frame.get(frame)
            if row:
                x, y, source, conf = row
                writer.writerow([frame_name(frame), f"{x:.2f}", f"{y:.2f}", source, f"{conf:.2f}"])
                visible += 1
            else:
                writer.writerow([frame_name(frame), *sentinel(width), "none", "0.00"])
    return visible


def raw_rows(video: Path, weights: Path, conf: float, device: str, meta: VideoMeta) -> Dict[int, DraftRow]:
    """The detector's own best guess per frame, unshifted (no y offset): the draft is what it saw."""
    detector = ball_detector(weights, conf=conf, device=device)
    return {frame: (x, y, "det", score) for frame, (x, y, score) in detect_points(detector, video, meta).items()}


def pipeline_rows(video: Path, weights: Path, conf: Optional[float], device: str, json_path: Path) -> Dict[int, DraftRow]:
    from tennis_tracker.config import Config
    from tennis_tracker.pipeline import run

    cfg = Config(
        input_video=str(video),
        output_video=str(json_path.with_suffix(".mp4")),
        tracking_json=str(json_path),
        model_path=str(weights),
        player_model_path=str(ROOT / "models" / "player.engine"),
        court_model_path=str(ROOT / "models" / "courtdetection.engine"),
        conf=PIPELINE_CONF if conf is None else conf,
        device=device,
        save_tracking_video=False,
        print_selector_tracks=False,
    )
    rows = run(cfg).frame_rows()
    return {
        row["frame"]: (row["x"], row["y"], row.get("source", "det"), float(row.get("conf", 0.0)))
        for row in rows if row.get("present")
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clips", nargs="*", help="Clip stems to (re)track; default: every video without a final label CSV")
    parser.add_argument("--source", choices=("raw", "pipeline"), default="raw")
    parser.add_argument("--weights", type=Path, default=REPO_WEIGHTS, help="GridTrackNet weights (.npz)")
    parser.add_argument("--conf", type=float, default=None,
                        help=f"Draft confidence threshold (raw default {RAW_DRAFT_CONF} so nearly every frame gets a guess)")
    parser.add_argument("--device", default="0")
    parser.add_argument("--force", action="store_true", help="Re-track even if a final label CSV exists")
    args = parser.parse_args()

    VIDEOS.mkdir(exist_ok=True)
    LABELS.mkdir(exist_ok=True)
    RUNS.mkdir(exist_ok=True)
    weights = args.weights.resolve()
    if not weights.is_file():
        print(f"Weights not found: {weights}")
        return 1

    videos = list(videos_in(VIDEOS).values())
    if args.clips:
        videos = [p for p in videos if p.stem in set(args.clips)]
    if not videos:
        print(f"No videos to track in {VIDEOS}")
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
            rows = pipeline_rows(video, weights, args.conf, args.device, RUNS / f"{video.stem}_{weights.stem}.json")
        visible = write_draft(rows, draft, stride, meta.width, meta.height, meta.total)
        print(f"[draft] {draft.name}: {visible} guessed frames of {len(range(0, meta.total, stride))} label frames "
              f"-> python finetune/labeling/label_tool.py {video.stem}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
