"""Score ball detectors and the full pipeline against the hand-labelled archive clips.

The archive (``gridtracknet_finetuning/archive``) holds ``videoN.mp4`` +
``videoN_ball.csv`` pairs labelled with the click tool: one row per labelled
frame, with invisible balls parked in the top-right corner.

Modes:
  raw       run each detector's prepass at threshold 0 and sweep thresholds
            offline (detector quality, independent of the selector)
  pipeline  run the complete tracker (no video) and score the per-frame
            output the renderer would draw (selector quality)

Caveat: the bundled GridTrackNet weights were fine-tuned on every archive clip
except video10, so only video10 is held-out for it; TOTNet never saw any.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2

ROOT = Path(__file__).resolve().parent
ARCHIVE = Path(r"C:\Users\Andrew\Desktop\gridtracknet_finetuning\archive")
MODELS = {
    "gridtracknet": ROOT / "models" / "gridtracknet_weights_torch.npz",
    "totnet": ROOT / "models" / "totnet_tennis_best.pt",
}
HIT_PX = 10.0      # prediction counts as correct within this many pixels (1080p scale)
WRONG_PX = 30.0    # beyond this it is tracking something else


def is_visible(x: float, y: float, width: int, height: int) -> bool:
    return not (x >= width * 0.95 and y <= height * 0.05)


def read_labels(path: Path, width: int, height: int) -> Dict[int, Optional[Tuple[float, float]]]:
    labels: Dict[int, Optional[Tuple[float, float]]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            index = int(re.fullmatch(r"frame_(\d+)", row["frame"].strip()).group(1))
            x, y = float(row["ball_x"]), float(row["ball_y"])
            labels[index] = (x, y) if is_visible(x, y, width, height) else None
    return labels


def natural_key(path: Path):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.stem)]


def list_clips(names: Optional[List[str]], archive: Path = ARCHIVE) -> List[Tuple[Path, Path]]:
    """Pairs of (video, label csv); accepts the flat archive layout or finetune/{videos,labels}."""
    label_dir = archive / "labels" if (archive / "labels").is_dir() else archive
    video_dirs = [archive / "videos", archive] if (archive / "videos").is_dir() else [archive]
    clips = []
    for csv_path in sorted(label_dir.glob("*_ball.csv"), key=natural_key):
        clip = csv_path.stem[: -len("_ball")]
        if names and clip not in names:
            continue
        videos = [p for d in video_dirs for p in d.glob(f"{clip}.*")
                  if p.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi", ".m4v"}]
        if not videos:
            print(f"[skip] no video for {csv_path.name}")
            continue
        clips.append((videos[0], csv_path))
    return clips


def video_meta(path: Path):
    capture = cv2.VideoCapture(str(path))
    meta = (
        float(capture.get(cv2.CAP_PROP_FPS)),
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    capture.release()
    return meta


def score(pairs, scale: float) -> Dict[str, float]:
    """pairs: [(label or None, prediction or None)], distances scaled to 1080p."""
    hit = wrong = miss = false_alarm = quiet = 0
    errors = []
    for label, prediction in pairs:
        if label is None:
            if prediction is None:
                quiet += 1
            else:
                false_alarm += 1
            continue
        if prediction is None:
            miss += 1
            continue
        error = math.hypot(prediction[0] - label[0], prediction[1] - label[1]) * scale
        errors.append(error)
        if error <= HIT_PX:
            hit += 1
        elif error > WRONG_PX:
            wrong += 1
    visible = hit + wrong + miss + sum(1 for e in errors if HIT_PX < e <= WRONG_PX)
    errors.sort()
    return {
        "visible": visible,
        "recall": hit / max(visible, 1),
        "wrong_obj": wrong / max(visible, 1),
        "miss": miss / max(visible, 1),
        "invisible": false_alarm + quiet,
        "false_alarm": false_alarm / max(false_alarm + quiet, 1),
        "median_px": errors[len(errors) // 2] if errors else float("nan"),
        "p90_px": errors[int(0.9 * len(errors))] if errors else float("nan"),
    }


LINE_PAIRS_14 = (
    (0, 4), (4, 6), (6, 1), (0, 2), (1, 3),
    (2, 5), (5, 7), (7, 3), (4, 8), (8, 10), (10, 5),
    (6, 9), (9, 11), (11, 7), (8, 12), (12, 9),
    (10, 13), (13, 11), (12, 13),
)
LINE_PX = 10.0  # "on a court line" if within this many 1080p pixels of one


def _inside_player(x: float, y: float, boxes) -> bool:
    values = boxes.values() if isinstance(boxes, dict) else (boxes or [])
    return any(b and len(b) >= 4 and b[0] <= x <= b[2] and b[1] <= y <= b[3] for b in values)


def _segment_distance(x: float, y: float, ax: float, ay: float, bx: float, by: float) -> float:
    dx, dy = bx - ax, by - ay
    length2 = dx * dx + dy * dy
    t = 0.0 if length2 <= 0 else max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / length2))
    return math.hypot(x - (ax + t * dx), y - (ay + t * dy))


def _near_court_line(x: float, y: float, keypoints, scale: float) -> bool:
    if not keypoints or len(keypoints) < 28:
        return False
    for a, b in LINE_PAIRS_14:
        ax, ay, bx, by = keypoints[2 * a], keypoints[2 * a + 1], keypoints[2 * b], keypoints[2 * b + 1]
        if (ax, ay) == (0.0, 0.0) or (bx, by) == (0.0, 0.0):
            continue
        if _segment_distance(x, y, ax, ay, bx, by) * scale <= LINE_PX:
            return True
    return False


def _classify(x: float, y: float, row, fallback_keypoints, scale: float) -> str:
    """What did a wrong/false output land on: player, line, or something else."""
    if _inside_player(x, y, row.get("player_boxes")):
        return "player"
    if _near_court_line(x, y, row.get("court_keypoints") or fallback_keypoints, scale):
        return "line"
    return "other"


def fmt(name: str, stats: Dict[str, float]) -> str:
    return (
        f"{name:14s} vis={stats['visible']:4d} recall={stats['recall']:.3f} "
        f"wrong={stats['wrong_obj']:.3f} miss={stats['miss']:.3f} | "
        f"invis={stats['invisible']:3d} falseAlarm={stats['false_alarm']:.3f} | "
        f"med={stats['median_px']:.1f}px p90={stats['p90_px']:.1f}px"
    )


def run_raw(models: List[str], clips, thresholds: List[float], device: str,
            weights: Optional[Dict[str, Path]] = None, stride: Optional[int] = None) -> None:
    from tennis_tracker.config import Config
    from tennis_tracker import detectors as backends

    per_model: Dict[str, List[Tuple[str, dict, list]]] = {name: [] for name in models}
    for name in models:
        cfg = Config(conf=0.0, device=device, gridtracknet_source_stride=stride, gridtracknet_prepass_background=False)
        path = (weights or {}).get(name) or MODELS[name]
        backend = backends.GridTrackNetBallDetector if name == "gridtracknet" else getattr(backends, "TOTNetBallDetector", None)
        if backend is None:
            print(f"[skip] {name}: this checkout has no TOTNet backend")
            continue
        detector = backend(str(path), cfg)
        for video_path, csv_path in clips:
            fps, width, height, total = video_meta(video_path)
            labels = read_labels(csv_path, width, height)
            started = time.perf_counter()
            detector.prepare_video(video_path, fps, width, height, total)
            elapsed = time.perf_counter() - started
            rows = []
            for frame, label in sorted(labels.items()):
                dets = detector.precomputed[frame] if frame < len(detector.precomputed) else []
                if dets:
                    box, conf = dets[0]
                    rows.append((label, ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0), conf))
                else:
                    rows.append((label, None, 0.0))
            per_model[name].append((video_path.stem, {"fps": fps, "width": width, "elapsed": elapsed}, rows))
            print(f"[{name}] {video_path.stem}: {total} frames in {elapsed:.1f}s", flush=True)
        del detector

    for name in models:
        print(f"\n=== {name}: threshold sweep (all clips pooled) ===")
        for threshold in thresholds:
            pairs = []
            for _, meta, rows in per_model[name]:
                scale = 1080.0 / meta["width"] * (1920.0 / 1080.0) if meta["width"] != 1920 else 1.0
                pairs.extend((label, pred if conf >= threshold else None) for label, pred, conf in rows)
            print(fmt(f"conf>={threshold:.2f}", score(pairs, 1.0)))
        print(f"\n=== {name}: per clip at conf>=0.50 ===")
        for clip, meta, rows in per_model[name]:
            pairs = [(label, pred if conf >= 0.5 else None) for label, pred, conf in rows]
            scale = 1920.0 / meta["width"]
            print(fmt(clip, score(pairs, scale)))
        # Systematic offset on confident hits (positive = prediction below/right of label).
        dx = dy = 0.0
        count = 0
        for _, meta, rows in per_model[name]:
            for label, pred, conf in rows:
                if label and pred and conf >= 0.5 and math.hypot(pred[0] - label[0], pred[1] - label[1]) <= HIT_PX:
                    dx += pred[0] - label[0]
                    dy += pred[1] - label[1]
                    count += 1
        if count:
            print(f"{name}: mean offset on hits dx={dx / count:+.2f}px dy={dy / count:+.2f}px (n={count})")


def run_pipeline(models: List[str], clips, conf: Dict[str, float], device: str, out_dir: Path) -> None:
    from tennis_tracker.config import Config
    from tennis_tracker.pipeline import run

    out_dir.mkdir(parents=True, exist_ok=True)
    for name in models:
        results = []
        for video_path, csv_path in clips:
            fps, width, height, total = video_meta(video_path)
            labels = read_labels(csv_path, width, height)
            json_path = out_dir / f"{video_path.stem}_{name}.json"
            if not json_path.is_file():
                cfg = Config(
                    input_video=str(video_path),
                    output_video=str(out_dir / f"{video_path.stem}_{name}.mp4"),
                    tracking_json=str(json_path),
                    model_path=str(MODELS[name]),
                    player_model_path=str(ROOT / "models" / "player.engine"),
                    court_model_path=str(ROOT / "models" / "courtdetection.engine"),
                    conf=conf[name],
                    device=device,
                    save_tracking_video=False,
                    print_selector_tracks=False,
                )
                run(cfg)
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            frames = {row["frame"]: row for row in payload["frames"]}
            fallback_keypoints = payload.get("last_valid_court_keypoints")
            scale = 1920.0 / width
            pairs = []
            by_source: Dict[str, Dict[str, int]] = {}
            landed: Dict[str, int] = {}
            bad_frames = []
            for frame, label in sorted(labels.items()):
                row = frames.get(frame)
                present = bool(row and row.get("present"))
                pred = (row["x"], row["y"]) if present else None
                pairs.append((label, pred))
                if present:
                    bucket = by_source.setdefault(row.get("source", "?"), {"hit": 0, "wrong": 0, "invisible": 0})
                    bad = None
                    if label is None:
                        bucket["invisible"] += 1
                        bad = "invisible"
                    elif math.hypot(pred[0] - label[0], pred[1] - label[1]) * scale > WRONG_PX:
                        bucket["wrong"] += 1
                        bad = "wrong"
                    else:
                        bucket["hit"] += 1
                    if bad:
                        where = _classify(pred[0], pred[1], row, fallback_keypoints, scale)
                        landed[where] = landed.get(where, 0) + 1
                        bad_frames.append((frame, bad, row.get("source"), where))
            results.append((video_path.stem, score(pairs, scale), by_source, pairs, landed, bad_frames))
        print(f"\n=== pipeline [{name}] per clip ===")
        pooled = []
        pooled_landed: Dict[str, int] = {}
        for clip, stats, by_source, pairs, landed, bad_frames in results:
            print(fmt(clip, stats), "| by source:", by_source, "| bad outputs landed on:", landed)
            if bad_frames:
                print("    bad frames:", ", ".join(f"{f}:{kind[0]}/{src}/{where}" for f, kind, src, where in bad_frames[:40]))
            pooled.extend(pairs)
            for key, value in landed.items():
                pooled_landed[key] = pooled_landed.get(key, 0) + value
        print(fmt("ALL", score(pooled, 1.0)), "| bad outputs landed on:", pooled_landed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("raw", "pipeline"), default="raw")
    parser.add_argument("--models", nargs="+", default=["gridtracknet", "totnet"], choices=sorted(MODELS))
    parser.add_argument("--clips", nargs="*", default=None, help="Clip stems, e.g. video10 video12")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument("--gridtracknet-conf", type=float, default=0.50)
    parser.add_argument("--totnet-conf", type=float, default=0.50)
    parser.add_argument("--device", default="0")
    parser.add_argument("--out-dir", default=str(ROOT / "output" / "archive_eval"))
    parser.add_argument("--archive", default=str(ARCHIVE),
                        help="Label set: the flat archive folder or a finetune-style folder with videos/ and labels/")
    parser.add_argument("--gridtracknet-weights", help="Score a different GridTrackNet .npz (raw mode)")
    parser.add_argument("--gridtracknet-stride", type=int, default=None,
                        help="Force the frame spacing fed to GridTrackNet (1 = consecutive frames even at 60 FPS)")
    args = parser.parse_args()

    clips = list_clips(args.clips, Path(args.archive))
    if not clips:
        print(f"No clips found in {args.archive}")
        return 1
    if args.mode == "raw":
        weights = {"gridtracknet": Path(args.gridtracknet_weights)} if args.gridtracknet_weights else None
        run_raw(args.models, clips, args.thresholds, args.device, weights, args.gridtracknet_stride)
    else:
        run_pipeline(
            args.models, clips,
            {"gridtracknet": args.gridtracknet_conf, "totnet": args.totnet_conf},
            args.device, Path(args.out_dir),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
