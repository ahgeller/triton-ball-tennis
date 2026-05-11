# Tracking Validation

This folder holds hand-labeled clips and reports for measuring tracking quality before changing detector, motion, selector, or physics behavior.

## 0. One-shot loop (recommended)

After labels exist for a clip, run the full pipeline → validate → report in a single command:

```powershell
python tools/run_validation.py --clip pomona_baseline
```

This reads `validation/annotations/pomona_baseline.json`, locates the video via its `video` field, runs `now_main.py` with `--tracking-json`, then validates with thresholds and writes a tagged report to `validation/reports/pomona_baseline__<git_sha>.json`.

Useful flags:

```powershell
python tools/run_validation.py --clip pomona_baseline --skip-pipeline   # re-validate an existing tracking JSON without re-rendering
python tools/run_validation.py --clip pomona_baseline --min-recall 0.6 --max-mean-error 12 --max-p90-error 25
python tools/run_validation.py --clip pomona_baseline --extra-now-args "--info"
```

To compare two reports (before/after a change):

```powershell
python tools/compare_reports.py `
  --before validation/reports/pomona_baseline__abc1234.json `
  --after  validation/reports/pomona_baseline__def5678.json
```

Returns a table of recall, error stats, fill ratio, large jumps, FP count, accuracy buckets, and source-mix deltas with arrows for direction-of-better. Use `--fail-on-regression` in CI-style checks.

### Starting a label set for a new clip

Two paths:

**A. Smart pre-fill (recommended).** Run the pipeline once to produce a `tracking.json`, then use `extract_label_frames.py` to pick 50–100 high-value frames (non-det sources, source transitions, sharp velocity changes) and pre-fill the annotation JSON with the pipeline's (x, y) so you only correct them:

```powershell
python tools/extract_label_frames.py `
  --tracking-json output_videos/pomona_baseline_tracking.json `
  --video "input_videos/PomonaPitzer Women vs. UCSD-cut-merged-1773082079341.mp4" `
  --out-dir validation/labels/pomona_baseline/ `
  --n-frames 80 `
  --clip-name pomona_baseline
```

This writes 80 PNGs and `pomona_baseline_starter.json` into the out dir. Open each PNG, verify the pre-filled (x, y) against the ball, and edit if wrong (or set `visible: false` if the ball is not present). When done, remove the `_review`/`_pipeline_*` fields and move the file to `validation/annotations/pomona_baseline.json`.

**B. From scratch.** Copy `validation/annotations/pomona_baseline.template.json` to `validation/annotations/<clip>.json` and fill in the `ball` array (50–100 frames concentrated around failure moments — see §2 below for what to target).

The manual steps below are kept as reference for one-off invocations.

---

## 1. Export predictions

Run the normal pipeline with the optional tracking JSON output:

```powershell
python now_main.py `
  --input input_videos/your_clip.mp4 `
  --output output_videos/your_clip_tracked.mp4 `
  --tracking-json output_videos/your_clip_tracking.json `
  --info
```

The JSON contains video metadata, per-frame chosen ball positions, source labels (`det`, `motion`, `guide`, `carry`, `interp`), all scored tracks, and timing when `--info` is enabled.

It also contains `motion_diagnostics`, which summarizes why orange motion was or was not selected during detector gaps. Each exported diagnostic frame includes the active search circle, selected source, reason code, mask source, and top structured motion candidates.

## 2. Create annotations

Create one annotation file per clip under `validation/annotations/`.

```json
{
  "video": "input_videos/your_clip.mp4",
  "ball": [
    {"frame": 10, "x": 1012.5, "y": 438.0, "visible": true},
    {"frame": 11, "x": 1028.0, "y": 447.5, "visible": true},
    {"frame": 12, "visible": false}
  ],
  "events": [
    {"frame": 42, "type": "bounce"},
    {"frame": 87, "type": "hit"}
  ],
  "ignore_ranges": [
    {"start": 0, "end": 5}
  ]
}
```

Start small: label 50-100 frames across serves, rallies, net play, occlusion, adjacent-court interference, and parked-ball cases. Dense labels around failure moments are more valuable than sparse labels across an entire match.

## 3. Validate

```powershell
python tools/validate_tracking.py `
  --predictions output_videos/your_clip_tracking.json `
  --annotations validation/annotations/your_clip.json `
  --report-json validation/reports/your_clip_report.json `
  --max-mean-error 12 `
  --max-p90-error 25 `
  --min-recall 0.90
```

Primary metrics:

- `recall`: how often a visible labeled ball got a prediction.
- `mean_error_px`, `p90_error_px`: pixel accuracy against labels.
- `source_counts`: how much output comes from detector, motion, guide, carry, or interpolation.
- `large_jump_count`: likely track switches or unstable output.
- `false_positive_absent_frames`: predicted ball when the annotation says the ball is not visible.

For orange-path work, also inspect `motion_diagnostics.summary.reason_counts` in the prediction JSON. The most important reason codes are:

- `no_motion_mask`: the fast motion path did not see usable motion.
- `no_blob`: a mask existed, but no contour survived candidate extraction.
- `blob_too_far_from_search`: motion existed, but not near the active ball search.
- `carry_selected_over_motion_candidate`: orange evidence existed near the search, but carry won.
- `lost_despite_motion_candidate`: candidate evidence existed, but no source was selected.
