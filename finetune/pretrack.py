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

    python finetune/pretrack.py                          # all unlabelled clips, raw GridTrackNet
    python finetune/pretrack.py --clips rally7           # one clip
    python finetune/pretrack.py --weights my_finetune.npz
    python finetune/pretrack.py --source pipeline
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tennis_tracker.config import Config  # noqa: E402

WORKSPACE = Path(__file__).resolve().parent
VIDEOS = WORKSPACE / "videos"
LABELS = WORKSPACE / "labels"
RUNS = WORKSPACE / "runs"
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}
DEFAULT_WEIGHTS = ROOT / "models" / "gridtracknet_weights_torch.npz"


def label_stride(fps: float) -> int:
    if 57.0 <= fps <= 62.0:
        return 2
    if 22.0 <= fps <= 32.0:
        return 1
    raise ValueError(f"{fps:.2f} FPS is not 30 or 60; re-encode the clip first")


def video_meta(path: Path):
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


def write_draft(rows_by_frame, draft_path: Path, stride: int, width: int, height: int, total: int) -> int:
    """rows_by_frame: frame -> (x, y, source, conf) or None."""
    visible = 0
    with draft_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "ball_x", "ball_y", "source", "conf"])
        for frame in range(0, total, stride):
            row = rows_by_frame.get(frame)
            if row:
                x, y, source, conf = row
                writer.writerow([f"frame_{frame:03d}", f"{x:.2f}", f"{y:.2f}", source, f"{conf:.2f}"])
                visible += 1
            else:
                writer.writerow([f"frame_{frame:03d}", width - 1, 0, "none", "0.00"])
    return visible


def raw_rows(video: Path, weights: Path, conf: float, device: str, fps, width, height, total):
    from tennis_tracker import detectors as backends

    cfg = Config(conf=conf, device=device, gridtracknet_prepass_background=False)
    if weights.suffix.lower() in (".pt", ".pth"):
        backend = getattr(backends, "TOTNetBallDetector", None)
        if backend is None:
            raise RuntimeError("This checkout has no TOTNet backend; use a GridTrackNet .npz")
    else:
        backend = backends.GridTrackNetBallDetector
    detector = backend(str(weights), cfg)
    detector.prepare_video(video, fps, width, height, total)
    rows = {}
    for frame, dets in enumerate(detector.precomputed):
        if dets:
            box, score = dets[0]
            rows[frame] = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0, "det", float(score))
    return rows


def pipeline_rows(video: Path, weights: Path, conf, device: str, json_path: Path):
    from tennis_tracker.pipeline import run

    cfg = Config(
        input_video=str(video),
        output_video=str(json_path.with_suffix(".mp4")),
        tracking_json=str(json_path),
        model_path=str(weights),
        player_model_path=str(ROOT / "models" / "player.engine"),
        court_model_path=str(ROOT / "models" / "courtdetection.engine"),
        conf=0.50 if conf is None else conf,
        device=device,
        save_tracking_video=False,
        print_selector_tracks=False,
    )
    run(cfg)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    return {
        row["frame"]: (row["x"], row["y"], row.get("source", "det"), float(row.get("conf", 0.0)))
        for row in payload["frames"] if row.get("present")
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clips", nargs="*", help="Clip stems to (re)track; default: every video without a final label CSV")
    parser.add_argument("--source", choices=("raw", "pipeline"), default="raw")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="Detector weights (.npz GridTrackNet or .pt TOTNet)")
    parser.add_argument("--conf", type=float, default=None,
                        help="Draft confidence threshold (raw default 0.25 so nearly every frame gets a guess)")
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

    videos = sorted(p for p in VIDEOS.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES)
    if args.clips:
        videos = [p for p in videos if p.stem in set(args.clips)]
    if not videos:
        print(f"No videos to track in {VIDEOS}")
        return 1

    for video in videos:
        final = LABELS / f"{video.stem}_ball.csv"
        draft = LABELS / f"{video.stem}_ball.csv.draft"
        if final.is_file() and not args.force:
            print(f"[skip] {video.name}: {final.name} already exists (use --force to redo)")
            continue
        fps, width, height, total = video_meta(video)
        stride = label_stride(fps)
        print(f"[track] {video.name}: {width}x{height} @ {fps:.2f} FPS, {total} frames, "
              f"labelling every {stride} frame(s), source={args.source}")
        if args.source == "raw":
            rows = raw_rows(video, weights, 0.25 if args.conf is None else args.conf, args.device, fps, width, height, total)
        else:
            rows = pipeline_rows(video, weights, args.conf, args.device, RUNS / f"{video.stem}_{weights.stem}.json")
        visible = write_draft(rows, draft, stride, width, height, total)
        print(f"[draft] {draft.name}: {visible} guessed frames of {len(range(0, total, stride))} label frames "
              f"-> python finetune/label_tool.py {video.stem}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
