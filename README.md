# Tennis Tracker

Production tennis-ball tracking pipeline for processed match clips.

The stable entrypoint is `now_main.py`. It delegates to `tennis_tracker.cli`,
which builds a two-pass pipeline:

1. Pass 1 collects ball detections, player boxes, court keypoints, raw motion,
   motion-boost masks, and optional debug metadata.
2. The selector in `ball_in_play_selector/` chooses the final ball position for
   each frame using detections, motion blobs, physics carry, guide tracks, and
   short-gap interpolation.
3. Pass 2 renders videos and writes optional per-frame tracking JSON.

## Project Layout

```text
now_main.py              Stable command-line entrypoint
tennis_tracker/          Runtime pipeline, detectors, motion preprocessing, rendering, video I/O
ball_in_play_selector/   Track building, physics guide, scoring, and final ball selection
models/                  Local model weights and TensorRT engines
input_videos/            Local video inputs, ignored by git
output_videos/           Generated videos/JSON, ignored by git
validation/              Label fixtures and regression-validation helpers
3DtrackingV1/            Archived helpers plus optional 3D trajectory sidecar work
PIPELINE_STUDY_MAP.md    Study guide for the math, libraries, and pipeline concepts
```

## Setup

Create the conda environment:

```powershell
conda env create -f environment.yml
conda activate tennis-tracker
```

The main runtime expects NVIDIA TensorRT Python bindings that match the
`.engine` files under `models/`. The default model paths are:

```text
models/ball.engine
models/player.engine
models/courtdetection.engine
```

Override them with `--model`, `--player-model`, and `--court-model`.

## Quick Run

Minimum tracking video:

```powershell
python now_main.py --input input_videos/clip.mp4 --output output_videos/clip_tracked.mp4
```

Tracking video plus per-frame JSON:

```powershell
python now_main.py `
  --input "input_videos/PomonaPitzer Women vs. UCSD-cut-merged-1773082079341.mp4" `
  --output output_videos/pomona_tracked.mp4 `
  --tracking-json output_videos/pomona_tracking.json
```

JSON only, no videos:

```powershell
python now_main.py `
  --input input_videos/clip.mp4 `
  --outputs none `
  --tracking-json output_videos/clip_tracking.json
```

Use `python now_main.py -h` for the complete flag list.

## Outputs

By default, only the final annotated `tracking` video is written. Use
`--outputs` to choose a set:

```powershell
# All debug videos
python now_main.py --input input_videos/clip.mp4 --outputs all

# A specific subset
python now_main.py --input input_videos/clip.mp4 --outputs tracking,pre-yolo,motion

# Override one output from an all run
python now_main.py --input input_videos/clip.mp4 --outputs all --no-motion-tracks-video
```

Named outputs:

| Name | Default path | Description |
| --- | --- | --- |
| `tracking` | `output_videos/prof_test.mp4` | Final annotated tracking video |
| `pre-yolo` | `output_videos/prof_test_yolo_input_debug.mp4` | Preprocessed frames used for ball YOLO input |
| `motion` | `output_videos/prof_test_motion_debug.mp4` | Raw/boost motion visualization |
| `guide` | `output_videos/prof_test_guide_debug.mp4` | Chosen-track guide progression |
| `motion-tracks` | `output_videos/prof_test_motion_tracks_debug.mp4` | Raw motion-track debug video |

Path flags:

```text
--output
--yolo-debug-path
--debug-path
--guide-path
--motion-tracks-path
```

Encoding uses NVENC when available. Pass `--no-nvenc` to fall back to CPU
encoding.

## Tracking JSON

Pass `--tracking-json path.json` to export frame-by-frame tracking data. Each
selected frame includes:

```text
frame, present, x, y, conf, source, interpolated, bbox, search, guide_search, selection
```

The `selection` object explains why the selector chose that frame's result. It
can include the source, reason, rejected candidate counts, stage, and local
context. This is useful for checking whether a point came from YOLO detection,
motion, carry, guide fallback, or interpolation.

Common `source` values:

| Source | Meaning |
| --- | --- |
| `det` | YOLO ball detection |
| `motion` | Motion blob accepted inside the search region |
| `carry` | Short physics prediction while temporarily lost |
| `guide` | Exact chosen-track guide fallback |
| `interp` | Interpolated gap fill |

## Debug Flags

Useful rendering and inspection flags:

```powershell
--draw-search-regions      # show selected-source search circles on tracking video
--hide-search-regions      # hide those circles
--draw-ball-trail          # draw selected ball trail
--no-ball-trail            # hide selected ball trail
--guide-video              # write guide debug video
--motion-video             # write motion debug video
--pre-yolo-video           # write ball-YOLO input debug video
--motion-tracks-video      # write motion-track debug video
--info                     # print timing breakdown
```

The main tracking output hides search-region circles by default. The guide
video still shows the richer guide/search debugging overlays.

## Speed Flags

Useful runtime controls:

```powershell
--skip-frame-yolo N        # run ball YOLO every Nth frame
--aux-on-yolo-frames       # sync player/court detection cadence to YOLO frames
--aux-force-interval N     # force auxiliary detection at least every N frames
--reader-prefetch N        # frame-reader queue depth
--cache-pass2-frames       # keep input frames in RAM for pass 2 when small enough
--pass2-cache-max-mb N     # RAM cache limit
--roi-motion               # restrict motion work to tracked ball ROIs
```

## Validation

The `validation/` folder contains label fixtures used to compare tracking JSON
against hand-labeled frames.

Run the lightweight fixture validator:

```powershell
python 3DtrackingV1/archived_tools/validate_tracking.py `
  --predictions validation/fixtures/sample_predictions.json `
  --annotations validation/fixtures/sample_annotations.json
```

Run the Pomona baseline helper when the source video is available locally:

```powershell
python 3DtrackingV1/archived_tools/run_validation.py --clip pomona_baseline
```

Compare two validation reports:

```powershell
python 3DtrackingV1/archived_tools/compare_reports.py `
  --before validation/reports/before.json `
  --after validation/reports/after.json `
  --fail-on-regression
```

Generated validation reports and labeling scratch files are ignored by git.

## 3DtrackingV1

`3DtrackingV1/` is not imported by the normal runtime. It contains archived
one-off helpers plus optional 3D trajectory tools:

```text
3DtrackingV1/archived_tools/   Validation, labeling, conversion, and probe helpers
3DtrackingV1/tools/            3D trajectory reconstruction and visualization tools
3DtrackingV1/tests/            Tests for the optional 3D/calibration sidecar
3DtrackingV1/calibration/      Camera calibration examples/guesses
```

The normal tracker should stay independent from this sidecar unless a change is
intentionally promoted into the runtime.

## Development Notes

- Keep tuned thresholds stable unless validation proves an improvement.
- Prefer `--tracking-json` plus validation reports for behavior comparisons.
- Keep generated videos, reports, label frames, and scratch outputs out of git.
- Use `PIPELINE_STUDY_MAP.md` when explaining the full pipeline, math, and
  libraries.
- Use `python -m compileall -q ball_in_play_selector tennis_tracker 3DtrackingV1`
  and `python -m unittest discover -s 3DtrackingV1\tests` for quick smoke checks.
