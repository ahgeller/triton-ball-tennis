"""Score ball detectors and the full pipeline against the hand-labelled clips.

The label set holds ``videoN.mp4`` + ``videoN_ball.csv`` pairs labelled with the
click tool: one row per labelled frame, with invisible balls parked in the
top-right corner (see ``finetune/data/labels.py``). ``--archive`` picks it; the
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
from typing import TYPE_CHECKING, Dict, List, NamedTuple, Optional, Tuple

from finetune.data.data_policy import is_verified, read_review_status
from finetune.data.detect import ball_detector, detect_points
from finetune.data.labels import read_labels
from finetune.data.detector_metrics import HIT_PX, DetectorMetrics, Point, scale_for_width
from finetune.data import workspace as ws
from finetune.data.workspace import clip_of, find_video, natural_key, video_meta
from evaluation.pipeline_score import ClipScore, score_pipeline_clip
from tennis_tracker.paths import COURT_ENGINE, GRIDTRACKNET_WEIGHTS, PLAYER_ENGINE, ROOT

if TYPE_CHECKING:
    from tennis_tracker.config import Config

ARCHIVE_VARIABLE = "TENNIS_ARCHIVE"
MODELS = {
    "gridtracknet": GRIDTRACKNET_WEIGHTS,
}
DEFAULT_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
PER_CLIP_CONF = 0.5         # the per-clip table and the offset report use this threshold
BAD_FRAMES_LISTED = 40      # how many of a clip's bad frames the pipeline report names

ClipPair = Tuple[Path, Path]    # (video, label csv)


class LabelledDetection(NamedTuple):
    """One labelled frame against the detector: the label, the detector's point at any confidence
    (None when it saw nothing) and that point's confidence (0 when there is none)."""
    label: Point
    prediction: Point
    confidence: float


class RawClip(NamedTuple):
    """One clip's labelled frames against one detector's prepass, in frame order."""
    clip: str
    width: int
    fps: float
    elapsed: float            # seconds the prepass took
    rows: List[LabelledDetection]


def default_archive() -> Path:
    """Hand-labelled clips: $TENNIS_ARCHIVE when set, else the in-repo workspace. Resolved when
    asked, not at import, so the environment and the folder are read as they are now."""
    override = os.environ.get(ARCHIVE_VARIABLE)
    return Path(override) if override else ws.current().root


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
        video = next((found for found in (find_video(clip, folder) for folder in video_dirs) if found is not None), None)
        if video is None:
            print(f"[skip] no video for {csv_path.name}")
            continue
        clips.append((video, csv_path))
    return clips


def fmt(name: str, stats: DetectorMetrics) -> str:
    return (
        f"{name:14s} vis={stats.visible:4d} recall={stats.recall:.3f} "
        f"wrong={stats.wrong:.3f} miss={stats.miss:.3f} | "
        f"invis={stats.invisible:3d} falseAlarm={stats.false_alarm:.3f} | "
        f"med={stats.median_px:.1f}px p90={stats.p90_px:.1f}px"
    )


def raw_rows(detector, clips: List[ClipPair], name: str) -> List[RawClip]:
    """Each clip's labelled frames against one detector's prepass."""
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
            rows.append(LabelledDetection(label, None if point is None else point[:2], 0.0 if point is None else point[2]))
        out.append(RawClip(video_path.stem, meta.width, meta.fps, elapsed, rows))
        print(f"[{name}] {video_path.stem}: {meta.total} frames in {elapsed:.1f}s", flush=True)
    return out


def _pooled_pairs(per_clip: List[RawClip], threshold: float) -> list:
    """Every clip's (label, prediction) pairs at 1080p scale, a prediction under `threshold` a miss."""
    pairs = []
    for clip in per_clip:
        scale = scale_for_width(clip.width)
        pairs.extend((None if label is None else (label[0] * scale, label[1] * scale),
                      (pred[0] * scale, pred[1] * scale) if pred is not None and conf >= threshold else None)
                     for label, pred, conf in clip.rows)
    return pairs


def _hit_offset(per_clip: List[RawClip]) -> Tuple[float, float, int]:
    """(sum dx, sum dy, count) over confident hits: the systematic offset of predictions from
    labels (positive = prediction below/right of label), in native pixels."""
    dx = dy = 0.0
    count = 0
    for clip in per_clip:
        for label, pred, conf in clip.rows:
            if label and pred and conf >= PER_CLIP_CONF and math.hypot(pred[0] - label[0], pred[1] - label[1]) <= HIT_PX:
                dx += pred[0] - label[0]
                dy += pred[1] - label[1]
                count += 1
    return dx, dy, count


def report_raw(name: str, per_clip: List[RawClip], thresholds: List[float]) -> None:
    print(f"\n=== {name}: threshold sweep (all clips pooled) ===")
    for threshold in thresholds:
        print(fmt(f"conf>={threshold:.2f}", DetectorMetrics.of(_pooled_pairs(per_clip, threshold), 1.0)))
    print(f"\n=== {name}: per clip at conf>={PER_CLIP_CONF:.2f} ===")
    for clip in per_clip:
        pairs = [(label, pred if conf >= PER_CLIP_CONF else None) for label, pred, conf in clip.rows]
        print(fmt(clip.clip, DetectorMetrics.of(pairs, scale_for_width(clip.width))))
    dx, dy, count = _hit_offset(per_clip)
    if count:
        print(f"{name}: mean offset on hits dx={dx / count:+.2f}px dy={dy / count:+.2f}px (n={count})")


def run_raw(models: List[str], clips: List[ClipPair], thresholds: List[float], device: str,
            weights: Optional[Dict[str, Path]] = None, stride: Optional[int] = None) -> None:
    per_model: Dict[str, List[RawClip]] = {}
    for name in models:
        detector = ball_detector((weights or {}).get(name) or MODELS[name], device=device, source_stride=stride)
        per_model[name] = raw_rows(detector, clips, name)
        del detector
    for name in models:
        report_raw(name, per_model[name], thresholds)


def tracker_config(video_path: Path, name: str, conf: float, device: str, out_dir: Path, json_path: Path) -> "Config":
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


def run_pipeline(models: List[str], clips: List[ClipPair], conf: Dict[str, float], device: str,
                 out_dir: Path, reuse_json: bool = False) -> None:
    from tennis_tracker.pipeline import run

    out_dir.mkdir(parents=True, exist_ok=True)
    for name in models:
        results: List[Tuple[str, ClipScore]] = []
        for video_path, csv_path in clips:
            meta = video_meta(video_path)
            labels = read_labels(csv_path, meta.width, meta.height)
            json_path = out_dir / f"{video_path.stem}_{name}.json"
            if json_path.is_file() and reuse_json:
                print(f"[reuse] {json_path.name} -- NOT re-running the tracker; "
                      f"these numbers are from whenever this file was written")
                payload = json.loads(json_path.read_text(encoding="utf-8"))
            else:
                result = run(tracker_config(video_path, name, conf[name], device, out_dir, json_path))
                payload = {"frames": result.frame_rows(), "last_valid_court_keypoints": result.last_valid_court_keypoints}
            results.append((video_path.stem, score_pipeline_clip(labels, payload, meta.width)))
        print(f"\n=== pipeline [{name}] per clip ===")
        pooled = []
        pooled_landed: Dict[str, int] = {}
        for clip, scored in results:
            print(fmt(clip, scored.stats), "| by source:", scored.by_source, "| bad outputs landed on:", scored.landed)
            if scored.bad_frames:
                print("    bad frames:", ", ".join(f"{f}:{kind[0]}/{src}/{where}"
                                                   for f, kind, src, where in scored.bad_frames[:BAD_FRAMES_LISTED]))
            pooled.extend(scored.pairs)
            for key, value in scored.landed.items():
                pooled_landed[key] = pooled_landed.get(key, 0) + value
        print(fmt("ALL", DetectorMetrics.of(pooled, 1.0)), "| bad outputs landed on:", pooled_landed)


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
