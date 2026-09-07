# Correctness fixes and verified-data runs — September 7, 2026

The raw-motion ball-color gate remains **disabled by default**, as requested.
Both CPU and CUDA fallback checks also default to disabled. The CUDA option now
works when explicitly enabled, but enabling it is not recommended by this run.
It does not alter the RGB input contract of GridTrackNet.

## Measured results

All comparisons below use only video13–video53: 41 manually verified clips,
9,292 source frames/label rows, 8,965 visible and 327 absent-ball labels.
Recall means localization within 10 pixels at a 1920-pixel reference width;
wrong-object means error above 30 reference pixels. The intermediate 10–30
pixel errors remain in the visible denominator. False alarms use absent labels
as their denominator. These are development measurements, not an untouched
published benchmark or proof of generalization to new recording sessions.

| Run | Localization recall | Wrong object | Missing | Absent-frame false alarms | Median / p90 error |
|---|---:|---:|---:|---:|---:|
| Raw detector, confidence 0.50 | 87.5% | 1.1% | 8.9% | 24.2% | 3.0 / 6.5 px |
| Corrected pipeline, color gate on | 87.0% | 0.8% | 9.3% | 23.9% | 2.8 / 6.6 px |
| Corrected pipeline, color gate **off** | **87.9%** | **0.8%** | **8.4%** | **23.9%** | **2.8 / 6.6 px** |

The first pipeline run loaded the prior enabled configuration before the user
changed the default. The explicit rerun verifies disabled configuration in
every exported JSON. Off improved recall by approximately 0.9 percentage points
over on, without a reported absent-frame false-alarm improvement from enabling
the gate. Leave it off. The false-alarm rate remains a material tracking issue.
Median/p90 describe emitted positions on visible labels, not missed frames.

Raw confidence sweeps also show the tradeoff: 0.30 gives 90.7% recall with
39.4% absent-frame false alarms; 0.70 gives 74.6% recall with 13.5% false alarms.
No threshold or weights were promoted. There is no complete matching pre-fix
run across these 41 clips, so this report does not attribute aggregate gains to
each correctness fix or compare them with the old frozen Pomona baseline.

## Implemented repairs

- Prepass frame ownership includes misses: overlapping tail batches cannot
  mutate already published results. Training scoring includes trailing frames.
- Training and archive scoring count intermediate localization errors and
  normalize pooled coordinates across resolutions. Model selection checks
  absent-ball false alarms as well as wrong-object errors.
- Writer enqueue/close detects a failed encoder worker even with a full queue;
  fallback writer initialization and process cleanup are checked.
- Kalman and projectile prediction share a damped transition for state and
  covariance. Drag has a fixed 30 FPS reference; source-frame velocity units
  remain compatible with the selector. Multi-step prediction agrees with
  repeated steps, including optional gravity-reference handling.
- Motion recovery counts connected-component pixels instead of rejecting tiny
  real blobs against a synthetic bounding-box area. CUDA allocations use the
  supplied frame device. Color filtering remains opt-in.
- JSON preserves detector measurement coordinates separately from fitted
  output and names the position kind. Detector-supported smoothing remains.
- Validation reports localization recall, presence recall and absence
  specificity. `--min-recall` now gates 10px localization; the optional
  `--min-presence-recall` provides an explicit presence gate. Legacy JSON
  aliases remain compatible; the frozen parity baseline was not refreshed.
- Label decoding establishes frame identity sequentially; identical images
  cannot redirect edits to another timestamp. Failed decoding cannot become
  a cached black annotation frame. Uncached backward jumps cost O(frame index).
- Realignment pins are hard constraints; reviewed rows cannot move or be
  dropped, and unreadable review anchors fail closed. Empty visible data is
  handled. Bulk label shifts refuse existing review/suspect sidecars instead
  of leaving stale frame references, and keep a distinct CSV backup.
- Review/suspect saves use temporary-file replacement. This is individual-file
  atomicity, not a multi-file transaction across media, CSV and sidecars.
- A separate `finetune/review_status.json` records the user's verified-video
  policy. Evaluation and default training honor it without editing exclusions
  or labels. Explicit inclusion of unverified training data does not authorize
  unverified validation. Validation expands whole manifest groups to prevent
  clip-level group leakage; recording IDs still need better curation.

## Verification

- `audit/test_fixes.py`: **15 synthetic regression tests passed**, including
  tail publication at both frame phases, metrics, writer failure, physics,
  CPU/CUDA color-gate parity, duplicate frames, hard pins, provenance,
  review eligibility and metadata-protected shifts.
- Tracker and training built-in self-tests passed.
- `audit/verify_fix_runs.py`: all 41 exports contain complete ordered source
  indices and color-off configuration, totaling 9,292 frames.
- An actual video13 run rendered the court overlay through NVENC. All 361
  output frames decoded at 1280×720; frame 180 was visually inspected.
  This validates rendering, not bounce-event or 3D accuracy.

Logs: `verified_raw_after_fixes.txt`, `verified_pipeline_after_fixes.txt`
(color on), `verified_pipeline_color_off.txt`, `fix_tests.txt`,
`training_self_test_after_fixes.txt`, `tracker_self_test_after_fixes.txt`,
`render_color_off_court.txt`, and `fix_run_verification.json` in this directory.
Final per-frame JSONs are in `output/verified_fixes_color_off_20260907`.
The rendered example is `output/verified_video13_color_off_court.mp4`.

## Next experiments, separate from correctness repairs

See `PROJECT_REVIEW.md` for the architecture rationale and primary sources.
Prioritize top-K candidate decoding and deterministic temporal fusion, then a
shared per-frame encoder with finer spatial features and lightweight temporal
convolutions. Test a local high-resolution branch if far-ball misses dominate.
Keep compatibility normalization and existing weights intact for this baseline.

Use recording-group holdouts and enough absent-ball frames to measure the
remaining false alarms. Measure cache JPEG/resize effects before changing the
training input distribution. Add camera-cut detection and per-shot court
geometry before trusting old court polygons across moving-camera footage.
Physics remains an image-space prior: time-scaled process noise, event-aware
uncertainty and camera-calibrated flight are further work. The disconnected
homography gravity helper was not enabled. Multi-GPU behavior was not tested
on this single-GPU machine; encoder failure testing does not prove cancellation
of every possible hung encoder. No labels were rewritten and no training or
model promotion was run during these fixes.
