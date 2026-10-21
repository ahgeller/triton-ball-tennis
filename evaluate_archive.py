"""Score ball detectors and the full pipeline against the hand-labelled clips.

The label set holds ``videoN.mp4`` + ``videoN_ball.csv`` pairs labelled with the
click tool: one row per labelled frame, with invisible balls parked in the
top-right corner (see ``finetune/labels.py``). ``--archive`` picks it; the
default is ``$TENNIS_ARCHIVE`` when that is set, else this repo's own
``finetune/`` workspace. Pass ``--archive finetune`` explicitly for workspace
runs, as AGENTS.md asks.

Modes:
  raw       run each detector's prepass at threshold 0 and sweep thresholds
            offline (detector quality, independent of the selector)
  pipeline  run the complete tracker (no video) and score the per-frame
            output the renderer would draw (selector quality)

Caveat: the bundled GridTrackNet weights were fine-tuned on every archive clip
except video10, so only video10 is held-out for it.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from finetune.data_policy import is_verified, read_review_status
from finetune.detect import ball_detector, detect_points
from finetune.labels import read_labels
from finetune.scoring import HIT_PX, REFERENCE_HEIGHT, REFERENCE_WIDTH, WRONG_PX, score, scale_for_width
from finetune.workspace import VIDEO_SUFFIXES, clip_of, natural_key, video_meta

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT / "finetune"
ARCHIVE_VARIABLE = "TENNIS_ARCHIVE"
MODELS = {
    "gridtracknet": ROOT / "models" / "gridtracknet_weights_torch.npz",
}
PLAYER_ENGINE = ROOT / "models" / "player.engine"
COURT_ENGINE = ROOT / "models" / "courtdetection.engine"
DEFAULT_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
PER_CLIP_CONF = 0.5         # the per-clip table and the offset report use this threshold
BAD_FRAMES_LISTED = 40      # how many of a clip's bad frames the pipeline report names

ClipPair = Tuple[Path, Path]    # (video, label csv)


def default_archive() -> Path:
    """Hand-labelled clips: $TENNIS_ARCHIVE when set, else the in-repo workspace. Resolved when
    asked, not at import, so the environment and the folder are read as they are now."""
    override = os.environ.get(ARCHIVE_VARIABLE)
    return Path(override) if override else WORKSPACE


def list_clips(names: Optional[List[str]], archive: Path) -> List[ClipPair]:
    """Pairs of (video, label csv); accepts the flat archive layout or finetune/{videos,labels}."""
    label_dir = archive / "labels" if (archive / "labels").is_dir() else archive
    video_dirs = [archive / "videos", archive] if (archive / "videos").is_dir() else [archive]
    clips = []
    review_status = read_review_status(archive)
    for csv_path in sorted(label_dir.glob("*_ball.csv"), key=lambda path: natural_key(path.stem)):
        clip = clip_of(csv_path)
        if names and not any(fnmatch.fnmatchcase(clip, pattern) for pattern in names):
            continue
        if not is_verified(clip, review_status):
            print(f"[skip] {clip}: labels are still under review")
            continue
        videos = [p for d in video_dirs for p in d.glob(f"{clip}.*") if p.suffix.lower() in VIDEO_SUFFIXES]
        if not videos:
            print(f"[skip] no video for {csv_path.name}")
            continue
        clips.append((videos[0], csv_path))
    return clips


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


def _classify(x: float, y: float, row: dict, fallback_keypoints, scale: float) -> str:
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


def raw_rows(detector, clips: List[ClipPair], name: str) -> List[Tuple[str, dict, list]]:
    """Per clip: (stem, meta, [(label, prediction, confidence)]) from one detector's prepass."""
    out = []
    for video_path, csv_path in clips:
        meta = video_meta(video_path)
        labels = read_labels(csv_path, meta.width, meta.height)
        started = time.perf_counter()
        points = detect_points(detector, video_path, meta)
        elapsed = time.perf_counter() - started
        rows = []
        for frame, label in sorted(labels.items()):
            point = points.get(frame)
            rows.append((label, None if point is None else point[:2], 0.0 if point is None else point[2]))
        out.append((video_path.stem, {"fps": meta.fps, "width": meta.width, "elapsed": elapsed}, rows))
        print(f"[{name}] {video_path.stem}: {meta.total} frames in {elapsed:.1f}s", flush=True)
    return out


def report_raw(name: str, per_clip: List[Tuple[str, dict, list]], thresholds: List[float]) -> None:
    print(f"\n=== {name}: threshold sweep (all clips pooled) ===")
    for threshold in thresholds:
        pairs = []
        for _, meta, rows in per_clip:
            scale = REFERENCE_HEIGHT / meta["width"] * (REFERENCE_WIDTH / REFERENCE_HEIGHT) if meta["width"] != REFERENCE_WIDTH else 1.0
            pairs.extend((None if label is None else (label[0] * scale, label[1] * scale),
                          (pred[0] * scale, pred[1] * scale) if pred is not None and conf >= threshold else None)
                         for label, pred, conf in rows)
        print(fmt(f"conf>={threshold:.2f}", score(pairs, 1.0)))
    print(f"\n=== {name}: per clip at conf>={PER_CLIP_CONF:.2f} ===")
    for clip, meta, rows in per_clip:
        pairs = [(label, pred if conf >= PER_CLIP_CONF else None) for label, pred, conf in rows]
        print(fmt(clip, score(pairs, scale_for_width(meta["width"]))))
    # Systematic offset on confident hits (positive = prediction below/right of label).
    dx = dy = 0.0
    count = 0
    for _, meta, rows in per_clip:
        for label, pred, conf in rows:
            if label and pred and conf >= PER_CLIP_CONF and math.hypot(pred[0] - label[0], pred[1] - label[1]) <= HIT_PX:
                dx += pred[0] - label[0]
                dy += pred[1] - label[1]
                count += 1
    if count:
        print(f"{name}: mean offset on hits dx={dx / count:+.2f}px dy={dy / count:+.2f}px (n={count})")


def run_raw(models: List[str], clips: List[ClipPair], thresholds: List[float], device: str,
            weights: Optional[Dict[str, Path]] = None, stride: Optional[int] = None) -> None:
    per_model: Dict[str, List[Tuple[str, dict, list]]] = {}
    for name in models:
        detector = ball_detector((weights or {}).get(name) or MODELS[name], device=device, source_stride=stride)
        per_model[name] = raw_rows(detector, clips, name)
        del detector
    for name in models:
        report_raw(name, per_model[name], thresholds)


def tracker_config(video_path: Path, name: str, conf: float, device: str, out_dir: Path, json_path: Path):
    from tennis_tracker.config import Config
    return Config(
        input_video=str(video_path),
        output_video=str(out_dir / f"{video_path.stem}_{name}.mp4"),
        tracking_json=str(json_path),
        model_path=str(MODELS[name]),
        player_model_path=str(PLAYER_ENGINE),
        court_model_path=str(COURT_ENGINE),
        conf=conf,
        device=device,
        save_tracking_video=False,
        print_selector_tracks=False,
    )


def score_pipeline_clip(labels: Dict[int, Optional[Tuple[float, float]]], payload: dict, width: int):
    """One clip's pipeline output against its labels: (stats, by_source, 1080p pairs, landed, bad frames)."""
    frames = {row["frame"]: row for row in payload["frames"]}
    fallback_keypoints = payload.get("last_valid_court_keypoints")
    scale = scale_for_width(width)
    pairs = []
    by_source: Dict[str, Dict[str, int]] = {}
    landed: Dict[str, int] = {}
    bad_frames = []
    for frame, label in sorted(labels.items()):
        row = frames.get(frame)
        present = bool(row and row.get("present"))
        pred = (row["x"], row["y"]) if present else None
        pairs.append((label, pred))
        if not present:
            continue
        bucket = by_source.setdefault(row.get("source", "?"), {"hit": 0, "localized_error": 0, "wrong": 0, "invisible": 0})
        bad = None
        if label is None:
            bucket["invisible"] += 1
            bad = "invisible"
        else:
            error = math.hypot(pred[0] - label[0], pred[1] - label[1]) * scale
            if error > WRONG_PX:
                bucket["wrong"] += 1
                bad = "wrong"
            else:
                bucket["hit" if error <= HIT_PX else "localized_error"] += 1
        if bad:
            where = _classify(pred[0], pred[1], row, fallback_keypoints, scale)
            landed[where] = landed.get(where, 0) + 1
            bad_frames.append((frame, bad, row.get("source"), where))
    normalized_pairs = [(None if label is None else (label[0] * scale, label[1] * scale),
                         None if pred is None else (pred[0] * scale, pred[1] * scale))
                        for label, pred in pairs]
    return score(pairs, scale), by_source, normalized_pairs, landed, bad_frames


def run_pipeline(models: List[str], clips: List[ClipPair], conf: Dict[str, float], device: str,
                 out_dir: Path, reuse_json: bool = False) -> None:
    from tennis_tracker.pipeline import run

    out_dir.mkdir(parents=True, exist_ok=True)
    for name in models:
        results = []
        for video_path, csv_path in clips:
            meta = video_meta(video_path)
            labels = read_labels(csv_path, meta.width, meta.height)
            json_path = out_dir / f"{video_path.stem}_{name}.json"
            if json_path.is_file() and reuse_json:
                print(f"[reuse] {json_path.name} -- NOT re-running the tracker; "
                      f"these numbers are from whenever this file was written")
            else:
                run(tracker_config(video_path, name, conf[name], device, out_dir, json_path))
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            results.append((video_path.stem, *score_pipeline_clip(labels, payload, meta.width)))
        print(f"\n=== pipeline [{name}] per clip ===")
        pooled = []
        pooled_landed: Dict[str, int] = {}
        for clip, stats, by_source, pairs, landed, bad_frames in results:
            print(fmt(clip, stats), "| by source:", by_source, "| bad outputs landed on:", landed)
            if bad_frames:
                print("    bad frames:", ", ".join(f"{f}:{kind[0]}/{src}/{where}"
                                                   for f, kind, src, where in bad_frames[:BAD_FRAMES_LISTED]))
            pooled.extend(pairs)
            for key, value in landed.items():
                pooled_landed[key] = pooled_landed.get(key, 0) + value
        print(fmt("ALL", score(pooled, 1.0)), "| bad outputs landed on:", pooled_landed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("raw", "pipeline"), default="raw")
    parser.add_argument("--models", nargs="+", default=["gridtracknet"], choices=sorted(MODELS))
    parser.add_argument("--clips", nargs="*", default=None, help="Clip stems or globs, e.g. video10 'grid_match*'")
    parser.add_argument("--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--gridtracknet-conf", type=float, default=0.50)
    parser.add_argument("--device", default="0")
    parser.add_argument("--out-dir", default=str(ROOT / "output" / "archive_eval"))
    parser.add_argument("--archive", default=None,
                        help="Label set: the flat archive folder or a finetune-style folder with videos/ and "
                             f"labels/ (default: ${ARCHIVE_VARIABLE}, else finetune/)")
    parser.add_argument("--reuse-json", action="store_true",
                        help="Pipeline mode: score existing JSONs in --out-dir instead of re-running. "
                             "Off by default -- reusing them after a code change silently scores the old code.")
    parser.add_argument("--gridtracknet-weights", help="Score a different GridTrackNet .npz (raw mode)")
    parser.add_argument("--gridtracknet-stride", type=int, default=None,
                        help="Force the frame spacing fed to GridTrackNet (1 = consecutive frames even at 60 FPS)")
    args = parser.parse_args()

    archive = Path(args.archive) if args.archive else default_archive()
    clips = list_clips(args.clips, archive)
    if not clips:
        print(f"No clips found in {archive}")
        return 1
    print(f"[archive] {archive}  ({len(clips)} clips)")
    if args.mode == "raw":
        weights = {"gridtracknet": Path(args.gridtracknet_weights)} if args.gridtracknet_weights else None
        run_raw(args.models, clips, args.thresholds, args.device, weights, args.gridtracknet_stride)
    else:
        run_pipeline(
            args.models, clips,
            {"gridtracknet": args.gridtracknet_conf},
            args.device, Path(args.out_dir), reuse_json=bool(args.reuse_json),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
