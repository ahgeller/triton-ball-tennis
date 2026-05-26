# Tennis Tracker

Production tennis-ball tracking pipeline for processed match clips.

The runtime keeps the tuned behavior in place: TensorRT YOLO ball detection,
player and court detection, motion-assisted recovery, ball-in-play selection,
and final tracking/video export.

## Project Layout

```text
tennis_tracker/          Runtime pipeline, detectors, motion preprocessing, rendering, video I/O
ball_in_play_selector/   Track building, physics guide, scoring, and final ball selection
models/                  Local model weights and TensorRT engines
input_videos/            Local video inputs
output_videos/           Generated outputs
3DtrackingV1/            Archived 3D experiments and old one-off helper scripts
```

`now_main.py` is the stable entrypoint. It delegates to `tennis_tracker.cli`.

## Setup

Create the conda environment:

```powershell
conda env create -f environment.yml
conda activate tennis-tracker
```

The main runtime expects NVIDIA TensorRT Python bindings that match the `.engine`
files under `models/`. If TensorRT is not importable, install it for your CUDA
runtime before running the full tracker.

## Run

Minimum:

```powershell
python now_main.py --input input_videos/clip.mp4 --output output_videos/clip_tracked.mp4
```

Typical run with per-frame JSON sidecar:

```powershell
python now_main.py `
  --input "input_videos/PomonaPitzer Women vs. UCSD-cut-merged-1773082079341.mp4" `
  --output output_videos/pomona_tracked.mp4 `
  --tracking-json output_videos/pomona_tracking.json
```

Models resolved from `models/` by default: `ball.engine`, `player.engine`,
`courtdetection.engine`. Override with `--model`, `--player-model`,
`--court-model`. See `python now_main.py -h` for the full flag set.

### Video outputs

By default only the `tracking` video is written. Select what's emitted with
`--outputs` (comma-separated), `all`, `none`, or individual `--<name>-video` /
`--no-<name>-video` toggles. Per-video flags override `--outputs`.

```powershell
# Default: just the annotated tracking video
python now_main.py --input input_videos/clip.mp4

# All five debug videos
python now_main.py --input input_videos/clip.mp4 --outputs all

# A specific subset
python now_main.py --input input_videos/clip.mp4 --outputs tracking,pre-yolo,motion

# No videos, JSON only (fastest pass for validation)
python now_main.py --input input_videos/clip.mp4 --outputs none `
  --tracking-json output_videos/clip_tracking.json

# Override one item out of an "all" run
python now_main.py --input input_videos/clip.mp4 --outputs all --no-motion-tracks-video
```

Named outputs and default paths (override with the matching `--*-path` flag):

| Name            | Default path                                       | Description                                |
| --------------- | -------------------------------------------------- | ------------------------------------------ |
| `tracking`      | `output_videos/prof_test.mp4` (`--output`)         | Final annotated tracking video             |
| `pre-yolo`      | `output_videos/prof_test_yolo_input_debug.mp4`     | Preprocessed frames fed to ball YOLO       |
| `motion`        | `output_videos/prof_test_motion_debug.mp4`         | Motion/debug visualization                 |
| `guide`         | `output_videos/prof_test_guide_debug.mp4`          | Selected-track guide progression           |
| `motion-tracks` | `output_videos/prof_test_motion_tracks_debug.mp4`  | Raw motion-track debug video               |

Path flags: `-o/--output`, `--yolo-debug-path`, `--debug-path`, `--guide-path`,
`--motion-tracks-path`. Encoder: NVENC by default; pass `--no-nvenc` to fall
back to a CPU encoder.

## Archived Helpers

The normal tracker does not import any scripts from the old `tools/` folder.
Those one-off validation, labeling, model-conversion, and motion-probe scripts
are archived under `3DtrackingV1/archived_tools/`.

Run the archived labeled Pomona baseline helper, if the validation labels are
present in your working tree:

```powershell
python 3DtrackingV1/archived_tools/run_validation.py --clip pomona_baseline
```

Compare reports after a change:

```powershell
python 3DtrackingV1/archived_tools/compare_reports.py `
  --before validation/reports/before.json `
  --after validation/reports/after.json `
  --fail-on-regression
```

Use `--info` on `now_main.py` or `--extra-now-args "--info"` through
the archived validation runner to print per-stage timing.

## Notes For Future Changes

- Keep tuned thresholds stable unless validation proves an improvement.
- Keep experimental 3D sidecars outside the runtime/tooling path; archived work lives under `3DtrackingV1/`.
- Keep generated videos, reports, label frames, and scratch outputs out of git.
- Prefer `--tracking-json` plus validation reports for behavior comparisons.
