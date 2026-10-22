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

``--holdout NAME`` scores a named held-out set from the workspace's ``holdouts.json``
(``finetune.data.holdouts``), the same clips a ``train_gridtracknet.py --holdout NAME`` run
never trained on. Raw mode reports, besides the pooled sweep and the per-clip table, a
``macro`` line (the per-clip mean, so a long clip does not outvote a short one), one
``source=<name>`` line per clips.csv source, the count beside every false-alarm rate (a rate
over five invisible labels is not a rate), and with ``--cell-histogram`` the recall by where the
ball sat within its 48x27 grid cell, the diagnostic for balls straddling a cell border.

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
from typing import Dict, List, NamedTuple, Optional, Sequence, Set, Tuple

from finetune.data.data_policy import clip_source, read_review_status, unscorable_reason
from finetune.data.detect import ball_detector, detect_points
from finetune.data.holdouts import holdout_clips
from finetune.data.labels import read_labels
from finetune.data.detector_metrics import HIT_PX, DetectorMetrics, Point, scale_for_width
from finetune.data import workspace as ws
from finetune.data.workspace import (HOLDOUTS_NAME, MANIFEST_NAME, clip_of, find_video, natural_key, read_exclusions,
                                     read_manifest, video_meta)
from evaluation.pipeline_score import ClipScore, score_pipeline_clip
from tennis_tracker.gridtracknet import GRID_COLS, GRID_ROWS
from tennis_tracker.paths import GRIDTRACKNET_WEIGHTS, ROOT

ARCHIVE_VARIABLE = "TENNIS_ARCHIVE"
MODELS = {
    "gridtracknet": GRIDTRACKNET_WEIGHTS,
}
DEFAULT_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
PER_CLIP_CONF = 0.5         # the per-clip table and the offset report use this threshold
BAD_FRAMES_LISTED = 40      # how many of a clip's bad frames the pipeline report names
CELL_BINS = 5               # --cell-histogram: bins across a grid cell, per axis

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
    height: int
    fps: float
    elapsed: float            # seconds the prepass took
    rows: List[LabelledDetection]


def default_archive() -> Path:
    """Hand-labelled clips: $TENNIS_ARCHIVE when set, else the in-repo workspace. Resolved when
    asked, not at import, so the environment and the folder are read as they are now."""
    override = os.environ.get(ARCHIVE_VARIABLE)
    return Path(override) if override else ws.current().root


def archive_folders(archive: Path) -> Tuple[Path, List[Path]]:
    """(the labels folder, the folders a video may sit in): finetune/{videos,labels} or the flat
    archive layout, where both share one folder."""
    label_dir = archive / "labels" if (archive / "labels").is_dir() else archive
    video_dirs = [archive / "videos", archive] if (archive / "videos").is_dir() else [archive]
    return label_dir, video_dirs


def _skip_reason(clip: str, status: Optional[dict], excluded: Set[str], include_excluded: bool) -> Optional[str]:
    """Why a clip's labels may not stand as ground truth (``data_policy.unscorable_reason``), or None."""
    return None if include_excluded else unscorable_reason(clip, status, excluded)


def list_clips(names: Optional[List[str]], archive: Path, include_excluded: bool = False,
               holdout: Optional[str] = None) -> List[ClipPair]:
    """Pairs of (video, label csv) whose labels may stand as ground truth (reviewed, not in
    exclude.txt; every clip with ``include_excluded``), narrowed to the stems or globs in `names`
    and to the members of the held-out set `holdout` (the archive's holdouts.json)."""
    label_dir, video_dirs = archive_folders(archive)
    review_status = read_review_status(archive)
    excluded = set(read_exclusions(archive))
    label_paths = sorted(label_dir.glob("*_ball.csv"), key=lambda path: natural_key(path.stem))
    held_out = None
    if holdout:
        labelled = [clip_of(path) for path in label_paths]
        held_out = set(holdout_clips(holdout, labelled, read_manifest(archive / MANIFEST_NAME), archive / HOLDOUTS_NAME))
    clips = []
    for csv_path in label_paths:
        clip = clip_of(csv_path)
        if names and not any(fnmatch.fnmatchcase(clip, pattern) for pattern in names):
            continue
        if held_out is not None and clip not in held_out:
            continue
        reason = _skip_reason(clip, review_status, excluded, include_excluded)
        if reason:
            print(f"[skip] {clip}: {reason}")
            continue
        video = next((found for found in (find_video(clip, folder) for folder in video_dirs) if found is not None), None)
        if video is None:
            print(f"[skip] no video for {csv_path.name}")
            continue
        clips.append((video, csv_path))
    return clips


def fmt(name: str, stats: DetectorMetrics, alarms: Optional[int] = None) -> str:
    """One metric line. ``alarms`` is the false-alarm count printed beside the rate (the rate's
    own numerator by default; a macro line passes the pooled count beside its mean rate)."""
    if alarms is None:
        alarms = round(stats.false_alarm * stats.invisible)
    return (
        f"{name:14s} vis={stats.visible:4d} recall={stats.recall:.3f} "
        f"wrong={stats.wrong:.3f} miss={stats.miss:.3f} | "
        f"invis={stats.invisible:3d} falseAlarm={stats.false_alarm:.3f} ({alarms}/{stats.invisible}) | "
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
        out.append(RawClip(video_path.stem, meta.width, meta.height, meta.fps, elapsed, rows))
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


def _clip_metrics(clip: RawClip, threshold: float) -> DetectorMetrics:
    """One clip's metrics at `threshold`, distances at 1080p."""
    pairs = [(label, pred if conf >= threshold else None) for label, pred, conf in clip.rows]
    return DetectorMetrics.of(pairs, scale_for_width(clip.width))


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def macro_metrics(per_clip: List[DetectorMetrics]) -> DetectorMetrics:
    """The per-clip mean of every rate and distance, over the clips that have the labels the rate
    is a share of (visible for recall / wrong / miss, invisible for the false-alarm rate), with the
    label counts summed: the line where a 900-label clip and a 36-label clip weigh the same."""
    with_ball = [stats for stats in per_clip if stats.visible]
    without = [stats for stats in per_clip if stats.invisible]
    located = [stats for stats in with_ball if not math.isnan(stats.median_px)]
    return DetectorMetrics(
        visible=sum(stats.visible for stats in per_clip),
        recall=_mean([stats.recall for stats in with_ball]),
        wrong=_mean([stats.wrong for stats in with_ball]),
        miss=_mean([stats.miss for stats in with_ball]),
        invisible=sum(stats.invisible for stats in per_clip),
        false_alarm=_mean([stats.false_alarm for stats in without]),
        median_px=_mean([stats.median_px for stats in located]),
        p90_px=_mean([stats.p90_px for stats in located]),
    )


def cell_histogram(per_clip: List[RawClip], threshold: float) -> Dict[str, List[DetectorMetrics]]:
    """Recall by where the labelled ball sits within its grid cell: for each axis, the visible
    labels binned by the fractional part of their grid coordinate (``CELL_BINS`` bins from the
    cell's leading edge to its trailing edge), each bin scored as a clip is. A network that
    only sees balls at cell centres shows it as low recall in the edge bins."""
    bins = {axis: [[] for _ in range(CELL_BINS)] for axis in ("x", "y")}
    for clip in per_clip:
        scale = scale_for_width(clip.width)
        cells = (GRID_COLS / clip.width, GRID_ROWS / clip.height)
        for label, pred, conf in clip.rows:
            if label is None:
                continue
            pair = ((label[0] * scale, label[1] * scale),
                    (pred[0] * scale, pred[1] * scale) if pred is not None and conf >= threshold else None)
            for axis, value, per_px in (("x", label[0], cells[0]), ("y", label[1], cells[1])):
                fraction = value * per_px % 1.0
                bins[axis][min(CELL_BINS - 1, int(fraction * CELL_BINS))].append(pair)
    return {axis: [DetectorMetrics.of(pairs, 1.0) for pairs in per_bin] for axis, per_bin in bins.items()}


def report_raw(name: str, per_clip: List[RawClip], thresholds: List[float],
               sources: Optional[Dict[str, str]] = None, histogram: bool = False) -> None:
    """The raw-mode report for one detector: the pooled sweep, the per-clip table with its macro
    and per-source lines (``sources``: clip -> clips.csv source), the hit offset, and with
    ``histogram`` the recall by position within the grid cell."""
    print(f"\n=== {name}: threshold sweep (all clips pooled) ===")
    for threshold in thresholds:
        print(fmt(f"conf>={threshold:.2f}", DetectorMetrics.of(_pooled_pairs(per_clip, threshold), 1.0)))
    print(f"\n=== {name}: per clip at conf>={PER_CLIP_CONF:.2f} ===")
    per_clip_stats = [_clip_metrics(clip, PER_CLIP_CONF) for clip in per_clip]
    for clip, stats in zip(per_clip, per_clip_stats):
        print(fmt(clip.clip, stats))
    pooled = DetectorMetrics.of(_pooled_pairs(per_clip, PER_CLIP_CONF), 1.0)
    print(fmt("macro", macro_metrics(per_clip_stats), alarms=round(pooled.false_alarm * pooled.invisible)))
    for source in sorted({(sources or {}).get(clip.clip, "") for clip in per_clip} - {""}):
        of_source = [clip for clip in per_clip if sources[clip.clip] == source]
        print(fmt(f"source={source}", DetectorMetrics.of(_pooled_pairs(of_source, PER_CLIP_CONF), 1.0)))
    dx, dy, count = _hit_offset(per_clip)
    if count:
        print(f"{name}: mean offset on hits dx={dx / count:+.2f}px dy={dy / count:+.2f}px (n={count})")
    if histogram:
        print(f"\n=== {name}: recall by position within the grid cell at conf>={PER_CLIP_CONF:.2f} ===")
        for axis, per_bin in cell_histogram(per_clip, PER_CLIP_CONF).items():
            for index, stats in enumerate(per_bin):
                low, high = index / CELL_BINS, (index + 1) / CELL_BINS
                print(f"{axis} {low:.1f}-{high:.1f}  n={stats.visible:5d} recall={stats.recall:.3f} "
                      f"wrong={stats.wrong:.3f} miss={stats.miss:.3f}")
            recalls = [stats.recall for stats in per_bin if stats.visible]
            if recalls:
                print(f"{axis} spread: {max(recalls) - min(recalls):.3f} between the best and the worst bin")


def run_raw(models: List[str], clips: List[ClipPair], thresholds: List[float], device: str,
            weights: Optional[Dict[str, Path]] = None, stride: Optional[int] = None,
            sources: Optional[Dict[str, str]] = None, histogram: bool = False) -> None:
    per_model: Dict[str, List[RawClip]] = {}
    for name in models:
        detector = ball_detector((weights or {}).get(name) or MODELS[name], device=device, source_stride=stride)
        per_model[name] = raw_rows(detector, clips, name)
        del detector
    for name in models:
        report_raw(name, per_model[name], thresholds, sources, histogram)


def run_pipeline(models: List[str], clips: List[ClipPair], conf: Dict[str, float], device: str,
                 out_dir: Path, reuse_json: bool = False, court_events: bool = False) -> None:
    """Track every clip (``headless.run_headless``) and score the per-frame output; with
    ``court_events`` each clip's tracking JSON also carries its court events, which
    ``score_court_events.py`` reads."""
    from tennis_tracker.headless import run_headless, tracking_payload   # the pipeline: CUDA and TensorRT, only here

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
                result = run_headless(video_path, model_path=MODELS[name], conf=conf[name], device=device,
                                      tracking_json=json_path, output_video=out_dir / f"{video_path.stem}_{name}.mp4",
                                      court_events=court_events)
                payload = tracking_payload(result)
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("raw", "pipeline"), default="raw")
    parser.add_argument("--models", nargs="+", default=["gridtracknet"], choices=sorted(MODELS))
    parser.add_argument("--clips", nargs="*", default=None, help="Clip stems or globs, e.g. video10 'grid_match*'")
    parser.add_argument("--holdout", metavar="NAME",
                        help="Only the clips of this named held-out set (the archive's holdouts.json), e.g. mixed_v1; "
                             "narrows --clips further when both are given")
    parser.add_argument("--cell-histogram", action="store_true",
                        help="Raw mode: also report recall by where the ball sat within its grid cell")
    parser.add_argument("--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--gridtracknet-conf", type=float, default=0.50)
    parser.add_argument("--device", default="0")
    parser.add_argument("--out-dir", default=str(ROOT / "output" / "archive_eval"))
    parser.add_argument("--archive", default=None,
                        help="Label set: the flat archive folder or a finetune-style folder with videos/ and "
                             f"labels/ (default: ${ARCHIVE_VARIABLE}, else finetune/)")
    parser.add_argument("--include-excluded", action="store_true",
                        help="Score every labelled clip, including those exclude.txt lists or the review policy "
                             "holds back (their numbers are not for choosing a model)")
    parser.add_argument("--reuse-json", action="store_true",
                        help="Pipeline mode: score existing JSONs in --out-dir instead of re-running. "
                             "Off by default -- reusing them after a code change silently scores the old code.")
    parser.add_argument("--court", action="store_true",
                        help="Pipeline mode: also compute the court events (landings, contacts, rally legs) and "
                             "write them into each clip's tracking JSON for score_court_events.py (needs CatBoost)")
    parser.add_argument("--gridtracknet-weights", help="Score a different GridTrackNet .npz (raw mode)")
    parser.add_argument("--gridtracknet-stride", type=int, default=None,
                        help="Force the frame spacing fed to GridTrackNet (1 = consecutive frames even at 60 FPS)")
    args = parser.parse_args(argv)

    archive = Path(args.archive) if args.archive else default_archive()
    clips = list_clips(args.clips, archive, args.include_excluded, args.holdout)
    if not clips:
        print(f"No clips found in {archive}")
        return 1
    print(f"[archive] {archive}  ({len(clips)} clips" + (f", held-out set {args.holdout}" if args.holdout else "") + ")")
    if args.mode == "raw":
        weights = {"gridtracknet": Path(args.gridtracknet_weights)} if args.gridtracknet_weights else None
        manifest = read_manifest(archive / MANIFEST_NAME)
        sources = {clip_of(csv_path): clip_source(clip_of(csv_path), manifest) for _, csv_path in clips}
        run_raw(args.models, clips, args.thresholds, args.device, weights, args.gridtracknet_stride, sources,
                args.cell_histogram)
    else:
        run_pipeline(
            args.models, clips,
            {"gridtracknet": args.gridtracknet_conf},
            args.device, Path(args.out_dir), reuse_json=bool(args.reuse_json), court_events=bool(args.court),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
