# Triton Ball Tennis

Tennis-ball tracking for UC San Diego match clips. This branch is the working
tracker: TensorRT YOLO finds the ball, players, and court; motion/ROI search and
physics recovery fill in the frames where the ball is tiny, blurred, or briefly
lost.

The output is an annotated video and, when requested, per-frame tracking JSON for
validation, benchmark runs, and downstream analytics.

## Current State

- Default branch: `Ball_Track`
- Stable entrypoint: `python now_main.py`
- Ball backend: TensorRT YOLO engine at `models/ball.engine`
- Player/court backends: TensorRT engines under `models/`
- Temporal TrackNet experiments live on the `temporal` branch and are not
  promoted here yet

This is a research/production hybrid: the command line is stable enough to run,
but thresholds and detector choices are still judged by validation reports, not
by vibes.

## Why This Pipeline Exists

Single-frame ball detection fails on tennis footage because the ball is often
only a few pixels, motion-blurred, hidden by players, or absent from every Nth
YOLO frame when speed settings are enabled. The tracker therefore combines:

1. TensorRT detections for ball, players, and court.
2. HSV S+V motion masks with static-region suppression.
3. ROI-localized search around the current ball state.
4. A selector that scores detections, motion blobs, guide tracks, physics carry,
   and short-gap interpolation.
5. Pass-2 rendering and JSON export for review.

Player and court detection run from the raw frame. Ball YOLO uses the boosted
preprocessed frame when preprocessing is enabled.

## Quick Start

Create the environment:

```powershell
conda env create -f environment.yml
conda activate tennis-tracker
```

Run the tracker:

```powershell
python now_main.py `
  --input input_videos/clip.mp4 `
  --output output_videos/clip_tracked.mp4
```

Write JSON for validation or analytics:

```powershell
python now_main.py `
  --input input_videos/clip.mp4 `
  --outputs none `
  --tracking-json output_videos/clip_tracking.json
```

Generate the useful debug videos:

```powershell
python now_main.py `
  --input input_videos/clip.mp4 `
  --outputs tracking,pre-yolo,motion,guide,motion-tracks `
  --tracking-json output_videos/clip_tracking.json `
  --info
```

Use `--no-nvenc` if video encoding fails on a machine without NVENC support.
Use `python now_main.py -h` for the full flag list.

## What To Inspect

| Need | Start here |
| --- | --- |
| Main runtime | `tennis_tracker/pipeline.py` |
| CLI flags | `tennis_tracker/cli.py` |
| Ball/player/court detectors | `tennis_tracker/detectors.py` |
| Candidate scoring and track selection | `ball_in_play_selector/` |
| Benchmark all labeled clips | `eval/README.md` |
| Label and validation workflow | `validation/README.md` |
| Bounce/hit and court-position sidecar | `analytics/bounce_events.py` |
| Concept map of the pipeline | `PIPELINE_STUDY_MAP.md` |
| Archived validation and 3D helpers | `3DtrackingV1/` |

## Tracking JSON

`--tracking-json` writes one row per frame. The important fields are:

```text
frame, present, x, y, conf, source, interpolated, bbox, search, guide_search, selection
```

Common `source` values:

| Source | Meaning |
| --- | --- |
| `det` | YOLO ball detection |
| `motion` | Motion blob accepted inside the search region |
| `carry` | Short physics prediction while the ball is lost |
| `guide` | Chosen-track guide fallback |
| `interp` | Short interpolated gap |

The `selection` object is the audit trail. When a frame looks wrong, inspect
that object before changing thresholds.

## Validation And Benchmarks

Run the fixture validator:

```powershell
python 3DtrackingV1/archived_tools/validate_tracking.py `
  --predictions validation/fixtures/sample_predictions.json `
  --annotations validation/fixtures/sample_annotations.json
```

Run the registered benchmark clips:

```powershell
python eval/benchmark.py
python eval/benchmark.py --skip-pipeline
python eval/benchmark.py --compare-latest --fail-on-regression
```

Benchmark results land in `eval/results/`. Per-clip validation reports land in
`validation/reports/`.

## Bounce And Hit Events

The tracker can feed a 2D event detector:

```powershell
python analytics/bounce_events.py `
  --tracking-json output_videos/clip_tracking.json `
  --output-json output_videos/clip_events.json `
  --render-video output_videos/clip_events.mp4
```

This sidecar finds likely bounces/hits from the selected 2D trajectory and maps
bounces onto court coordinates when court homography is available.

## Repository Layout

```text
now_main.py              Thin entrypoint into tennis_tracker.cli
tennis_tracker/          Runtime pipeline, detectors, motion, rendering, video I/O
ball_in_play_selector/   Track building, physics guide, scoring, final selection
models/                  Local YOLO/TensorRT model files
input_videos/            Local inputs; videos are ignored by git
output_videos/           Generated videos/JSON; ignored by git
validation/              Label fixtures and validation docs
eval/                    Multi-clip benchmark harness
analytics/               Bounce/hit detection and court mapping sidecar
3DtrackingV1/            Archived tools plus optional 3D trajectory work
```

Generated videos, validation reports, label scratch files, and training outputs
should stay out of git.

## Development Checks

```powershell
python -m pytest analytics/test_bounce_events.py
python -m py_compile analytics/bounce_events.py eval/benchmark.py tennis_tracker/pipeline.py
```

For behavior changes, run `eval/benchmark.py` before and after the change. The
benchmark is the gate for promoting detector, motion, or selector tweaks.
