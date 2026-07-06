# Motion Detection Research Log

Purpose: keep a running, repo-specific record of motion-detection research, observed improvements, failed ideas, and next actions. This file should be updated whenever a motion/speed experiment teaches us something, especially before promoting any change into the main pipeline.

## Working Agreement

- Main pipeline changes stay gated by evidence from tracking JSON, validation reports, and motion diagnostics.
- New motion/speed ideas should first live in a separate experiment file, not directly in `tennis_tracker.pipeline`.
- The most important proof points are FPS, visible-ball recall, pixel error on labeled frames, large jumps, false positives on absent frames, and `motion_diagnostics.summary.reason_counts`.
- Preserve the existing TensorRT/selector pipeline unless an experiment shows a measurable win.

## User-Provided Research Digest

The research brief argues that the best architecture for a mostly static tennis camera is a cheap predictor plus a cheap candidate generator around the existing YOLO detector.

Key takeaways:

- A constant-acceleration or projectile/Kalman predictor is the backbone for ROI placement.
- Frame differencing and background subtraction are most useful as candidate generators and recovery aids, not final truth.
- Motion masks plus size, shape, speed, and optional color scoring are a strong fit for tennis-ball recovery.
- Dense optical flow and correlation trackers are low priority for this problem because they are either too expensive or brittle on tiny blurred balls.
- Crop-based YOLO is where most speed should come from; motion should make the crop smarter and provide recovery candidates when YOLO misses.
- Periodic full-frame or widened search is needed to avoid permanent drift after sharp bounces, racket hits, or ROI loss.
- TrackNet-style and WASB-style methods validate the importance of temporal motion cues, but the practical first move is improving the wrapper around the existing YOLO pipeline.

## Research Pass - 2026-05-27

Primary docs checked before implementation:

- OpenCV background subtraction tutorial: background subtraction is intended for static-camera foreground masks, which matches the tennis-camera assumption.
- OpenCV motion-analysis docs: `createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=true)` and `createBackgroundSubtractorKNN(history=500, dist2Threshold=400, detectShadows=true)` are the built-in baseline APIs to compare.
- OpenCV morphology docs: opening is erosion followed by dilation and is useful for removing noise; closing can repair holes.
- OpenCV contour docs: moments, contour area, bounding rectangles, and perimeter-derived compactness are enough for cheap blob metrics.

Implementation choice from that research: build the first experiment around mask generation plus blob measurement, not around a tracker. The tracker is already complex in the main pipeline; the missing evidence is whether a different mask creates better candidate blobs at lower cost.

## Codebase Map

Entrypoint:

- `now_main.py` is only a stable wrapper around `tennis_tracker.cli.main`.
- `tennis_tracker.cli` turns command-line flags into `tennis_tracker.config.Config`.
- `tennis_tracker.pipeline.run` owns the two-pass runtime.

Runtime pass 1:

- Loads TensorRT ball, player, and court engines.
- Opens video through OpenCV and `ThreadedFrameReader`.
- Maintains HSV/S+V background models and variance.
- Computes raw motion and boost masks.
- Optionally restricts motion work to predicted ball ROIs through `ROIMotionTracker`.
- Brightens ball-like motion for YOLO input and dims static regions.
- Runs async TensorRT ball detection, plus player/court detection.
- Stores detections, masks, players, court keypoints, ROI boxes, and optional debug frames.

Selector:

- `ball_in_play_selector.tracking.build_detections` converts YOLO boxes into `Detection` objects and marks whether they overlap motion.
- `build_motion_tracks` extracts continuous centroid tracks from motion masks.
- `build_tracks` associates YOLO detections into candidate ball tracks with Kalman/projectile prediction.
- `score_tracks` ranks/blacklists tracks using motion, court, speed, player context, span, extent, and stationary rejection.
- `select_ball_in_play` chooses per-frame output from `det`, `motion`, `guide`, `carry`, `interp`, or lost.
- `BallKalmanFilter` in `physics.py` is a 4D state `[x, y, vx, vy]` filter with gravity/drag, Mahalanobis diagnostics, confidence-adaptive measurement noise, and a bounce heuristic.

Runtime pass 2:

- Renders tracking, pre-YOLO, motion, guide, and motion-track debug videos.
- Writes optional tracking JSON with full config, selected frames, tracks, source reasons, timing, and motion diagnostics.

Validation:

- Validation uses `validation/annotations/*.json`, tracking JSON, and archived helpers in `3DtrackingV1/archived_tools`.
- The most relevant workflow is pipeline -> `--tracking-json` -> `validate_tracking.py` -> `compare_reports.py`.
- For orange-path work, inspect `motion_diagnostics.summary.reason_counts` and diagnostic frames.

Archived/sidecar:

- `3DtrackingV1/tools` contains optional 3D reconstruction and visualization, not normal runtime.
- `3DtrackingV1/archived_tools/raw_motion_probe.cpp` is a useful prior motion probe: HSV/gray frame delta, adaptive S+V background, temporal gating, and connected-component filtering.

## Current Motion Implementation

Already implemented:

- Adaptive HSV/S+V background subtraction with variance thresholding.
- C++ probe-style temporal proof using max(gray diff, V diff, scaled S diff).
- Loose tennis-ball color support gate for raw motion.
- Boost mask filtering with connected components, area/aspect checks, tiny-blob preservation, and capped boost radius.
- Separate narrow selector-grade boost mask and wider YOLO-grade boost mask.
- CUDA preprocessing path with optional zero-copy preprocess-to-YOLO.
- ROI-based motion computation using `ROIMotionTracker`.
- Skip-frame YOLO and auxiliary detection cadence controls.
- Selector-side motion blob recovery with prediction search, blob scoring, player penalties, size checks, trajectory continuity, physics gates, guide fallback, and carry fallback.
- Motion diagnostics exported in tracking JSON.

Important implication: the repo is not missing the basic recommended architecture. Improvements should focus on measurement, simplification, and targeted candidate quality/speed changes.

## Main Risk Areas To Study

1. Motion candidate extraction may be doing too much shape filtering before we know whether the real ball is blurred or elongated.
2. ROI motion can save time, but stale or ghost ROIs can suppress true recovery or sweep up static clutter.
3. The current filter is constant-velocity Kalman plus gravity control rather than a full constant-acceleration state. It may still be enough, but bounce/hit recovery should be measured.
4. Background update and motion-freeze parameters may trade off between parked-ball ghost cleanup and weak-ball recall.
5. Color support is useful but brittle; any experiment should track whether it removes real balls in shadows or under lighting shifts.
6. The selector has many guardrails. A better raw mask may not improve final output if the selector rejects candidates later, so diagnostics must identify where loss happens.

## Recommended Separate-File Experiment

Create a standalone experiment script, likely `experiments/motion_speed_probe.py`, that does not import or mutate the main pipeline loop except for reusing small helper functions where safe.

First version should:

- Read a video path and optional frame range.
- Run several motion candidate generators on the same frames:
  - current adaptive S+V style,
  - simple 2-frame or 3-frame delta,
  - MOG2/KNN background subtractor,
  - optional hybrid delta plus HSV color score.
- Emit per-frame candidate blobs with area, aspect, fill ratio, compactness, centroid, bbox, color score, and timing.
- Save a lightweight JSON/CSV report plus optional debug video overlays.
- Compare candidates against existing tracking JSON or hand annotations when available.
- Produce a small summary table: FPS, candidate count, candidate near selected ball, missed selected ball, false candidates near players, and top failure reasons.

Only after that should we decide whether to promote a candidate generator into `tennis_tracker.motion` or adjust selector gates.

## Implementation Pass - 2026-05-27

Added `experiments/motion_speed_probe.py`.

The script compares these methods frame-by-frame:

- `adaptive_sv`: standalone CPU approximation of the repo's adaptive saturation/value background mask.
- `delta2`: two-frame max(gray, V, scaled-S) frame differencing.
- `delta3`: soft three-frame temporal differencing.
- `mog2`: OpenCV MOG2 background subtractor.
- `knn`: OpenCV KNN background subtractor.

Outputs:

- JSON summary/report with per-method timing, candidate counts, hit-rate against references, and per-frame candidates.
- Optional flat candidate CSV.
- Optional debug MP4 overlay with reference point and top candidates per method.

Scoring:

- Hand annotations take priority when passed with `--annotations`.
- Existing production tracking JSON can be passed with `--tracking-json`; this supplies fallback reference points and player boxes for near-player diagnostics.
- Candidate coordinates are reported in original video pixels even when `--downscale` is used.

## Probe Run - 2026-05-27

Environment:

- Used `conda run -n tennis-analysis`.
- Runtime check: Python from `C:\Users\Andrew\.conda\envs\tennis-analysis`, OpenCV 4.9.0, NumPy 1.26.4.

Clip/range:

- Video: `input_videos/PomonaPitzer Women vs. UCSD-cut-merged-1773082079341.mp4`.
- Annotation file: `validation/annotations/pomona_baseline.json`.
- First focused range: frames 780-999, which includes the annotated frame-870 motion-FP bug region.

Outputs:

- `output_videos/motion_probe_pomona_780_999.json`
- `output_videos/motion_probe_pomona_780_999.csv`
- `output_videos/motion_probe_pomona_780_999_debug.mp4`
- `output_videos/motion_probe_pomona_780_999_loose.json`
- `output_videos/motion_probe_pomona_780_999_loose.csv`
- `output_videos/motion_probe_pomona_780_999_fullres_diff.json`
- `output_videos/motion_probe_pomona_780_999_fullres_diff.csv`

Initial half-scale/default result at 24 px hit radius:

| Method | Hit Rate | Mean Best Dist | P90 Best Dist | Candidates/Frame | Mask ms | Extract ms |
|---|---:|---:|---:|---:|---:|---:|
| adaptive_sv | 4.8% | 226.1 | 411.8 | 4.98 | 10.29 | 0.92 |
| delta2 | 19.0% | 111.1 | 249.5 | 4.97 | 5.27 | 0.96 |
| delta3 | 19.0% | 113.4 | 398.1 | 4.95 | 9.16 | 0.83 |
| mog2 | 0.0% | 415.0 | 927.8 | 4.98 | 2.59 | 4.47 |
| knn | 0.0% | 396.1 | 1052.1 | 4.91 | 2.66 | 1.15 |

Loose half-scale result (`top_k=25`, no opening, looser shape filters):

| Method | Hit Rate @24px | Mean Best Dist | P90 Best Dist | Candidates/Frame |
|---|---:|---:|---:|---:|
| adaptive_sv | 19.0% | 49.0 | 91.8 | 21.53 |
| delta2 | 19.0% | 63.1 | 156.0 | 20.82 |
| delta3 | 19.0% | 56.7 | 127.7 | 15.14 |
| mog2 | 0.0% | 240.1 | 384.8 | 24.89 |
| knn | 0.0% | 125.7 | 214.9 | 24.55 |

Full-resolution diff/adaptive loose result:

| Method | Hit Rate @24px | Mean Best Dist | P90 Best Dist | Candidates/Frame | Mask ms |
|---|---:|---:|---:|---:|---:|
| adaptive_sv | 23.8% | 58.5 | 95.3 | 24.84 | 51.56 |
| delta2 | 14.3% | 82.1 | 219.6 | 24.88 | 27.46 |
| delta3 | 19.0% | 56.0 | 127.6 | 24.25 | 53.79 |

Interpretation:

- The 24 px hit radius is too strict for this motion-candidate probe. Motion blobs often land near the ball but not exactly on the annotation center, especially with blur/ghosting. For ROI generation, 50-60 px proximity is more meaningful than exact center accuracy.
- On the loose half-scale run, `adaptive_sv` and `delta3` find a candidate within 50 px on 13-15 of 21 labeled frames and within 60 px on 16 of 21 frames. This is much better than the headline 24 px hit rate suggests.
- `delta2` is the fastest useful candidate generator in this range; `delta3` is cleaner but roughly twice the mask cost at half scale.
- `MOG2` is not promising out of the box here. It is fast to mask but poor at placing useful top candidates after filtering.
- `KNN` gets closer under loose filtering but still trails the diff/adaptive methods.
- Full-resolution processing helps `adaptive_sv` slightly at 24 px but costs roughly 4-5x more than half-scale. For recovery ROI generation, half-scale plus a wider ROI may be a better speed/quality tradeoff.

Next experiment decision:

- Add a ranking/evaluation mode based on "candidate within ROI radius" rather than "candidate center equals annotation center".
- Consider a production candidate path that uses cheap half-scale `delta2`/`delta3` to propose a wider recovery ROI, while leaving exact ball localization to YOLO/selector.
- Do not promote MOG2/KNN yet.

## Research Note - Multi-Motion Fusion + Kalman Gating

Idea: run several cheap motion detectors at once, then fuse their candidates with a Kalman/projectile prediction so the system keeps motion that behaves like the ball and rejects motion that behaves like players/noise.

Relevant research/practice families:

- Temporal differencing: 2-frame differencing is fast and sensitive; 3-frame differencing adds temporal confirmation and reduces one-frame flicker/ghost noise.
- Background subtraction: MOG2/KNN model static background and expose foreground masks, but the first probe showed they are not automatically better for this clip.
- Motion attention: TrackNetV4 uses frame-differencing maps as motion attention, which supports the idea that even deep sports-ball trackers benefit from explicit motion cues.
- Kalman/data association: use the filter prediction and covariance to gate candidate measurements by Mahalanobis distance, then score candidate velocity against expected velocity.
- Multi-hypothesis/IMM-style tracking: useful conceptually around bounces/hits, where several motion models may be plausible; probably overkill before simple fusion is measured.

Recommended fusion level:

- Prefer candidate-level late fusion over pixel-level AND/OR as the first implementation.
- Pixel-level AND is too strict: a real ball may appear in delta2 but not MOG2/KNN.
- Pixel-level OR is too noisy: player limbs and court-line flicker accumulate.
- Candidate-level fusion lets each blob keep evidence fields: which masks supported it, distance to Kalman prediction, observed velocity, speed ratio, direction cosine, area/shape/color score, and player proximity.

Candidate scoring sketch:

```text
candidate_score =
  mask_support_score
  + kalman_gate_score
  + velocity_direction_score
  + speed_ratio_score
  + area_shape_score
  + color_soft_score
  - player_overlap_penalty
  - large_blob_penalty
```

Core motion-vector checks:

- Predicted center from Kalman/projectile: `p_pred`.
- Last accepted ball center: `p_last`.
- Candidate center: `p_cand`.
- Observed candidate velocity: `v_obs = (p_cand - p_last) / dt`.
- Predicted velocity from filter: `v_pred`.
- Direction match: cosine similarity between `v_obs` and `v_pred`.
- Speed match: `|v_obs| / max(|v_pred|, eps)`.
- Kalman innovation: `r = z - Hx`, `S = HPH^T + R`, `d2 = r^T S^-1 r`.

Practical interpretation:

- If a candidate is supported by `delta2` and close to the Kalman gate, keep it even if delta3 misses it.
- If a candidate is supported by multiple masks but is far outside the Kalman gate, treat it as likely player/background motion.
- If a bounce/hit is suspected, temporarily widen the gate and raise process noise instead of rejecting the candidate immediately.
- The output should be a ranked list of recovery ROIs, not a final ball point. YOLO or the selector should still do final localization.

Next experiment:

- Extend the experiment harness with a fusion mode that clusters candidates from `adaptive_sv`, `delta2`, `delta3`, and optionally KNN.
- Score clusters against annotation/reference using multiple radii: 24 px for exact point, 50-80 px for ROI usefulness.
- Report how often the best fused ROI contains/near-contains the labeled ball during YOLO-miss-like frames.

## Fusion Viewer Pass - 2026-05-27

Added fusion/debug-video support to `experiments/motion_speed_probe.py`.

New experiment features:

- `--fusion`: clusters candidates from multiple methods and scores fused clusters.
- `--fusion-guide-reference`: uses tracking JSON / annotation points as the per-frame prediction guide when available. This is closest to viewing "motion around the main YOLO/ROI path."
- `--fusion-seed-reference`: reseeds the experiment tracker from reference points after loss.
- Debug video now overlays:
  - colored raw motion-mask pixels,
  - top per-method candidates,
  - prediction/gate circle,
  - reference/tracking point,
  - fused candidates,
  - accepted fused ROI,
  - score breakdown text panel.

Full Pomona run:

```powershell
conda run -n tennis-analysis python experiments\motion_speed_probe.py `
  --input "input_videos\PomonaPitzer Women vs. UCSD-cut-merged-1773082079341.mp4" `
  --tracking-json output_videos\_inspection\strict_soft_tracking.json `
  --start-frame 0 `
  --max-frames 0 `
  --downscale 0.5 `
  --methods adaptive_sv,delta2,delta3,knn `
  --fusion `
  --fusion-seed-reference `
  --fusion-guide-reference `
  --top-k 20 `
  --fusion-top-k 8 `
  --open-size 1 `
  --min-area 3 `
  --max-area 5000 `
  --max-dim 220 `
  --max-aspect 8 `
  --min-fill 0.04 `
  --output-json output_videos\motion_fusion_full_pomona_guided.json `
  --debug-video output_videos\motion_fusion_full_pomona_guided_debug.mp4 `
  --no-progress
```

Guided full-run result:

| Metric | Value |
|---|---:|
| Frames processed | 2022 |
| Effective FPS | 12.02 |
| Fusion accepted frames | 1771 / 2022 |
| Fused clusters/frame | 8.00 |
| Fusion hit @24 px vs tracking reference | 88.5% |
| Fusion hit @50 px vs tracking reference | 97.1% |
| Fusion hit @80 px vs tracking reference | 97.8% |
| Accepted fusion hit @24 px | 88.0% |
| Accepted fusion hit @50 px | 96.5% |
| Accepted fusion hit @80 px | 97.8% |
| Mean accepted distance | 10.7 px |
| P90 accepted distance | 24.7 px |

Per-method hit rates against the same tracking reference:

| Method | Hit @24 px | Candidates/Frame | Mask ms | Extract ms |
|---|---:|---:|---:|---:|
| adaptive_sv | 89.2% | 15.61 | 12.30 | 2.00 |
| delta2 | 85.4% | 19.19 | 6.59 | 2.21 |
| delta3 | 92.6% | 16.82 | 12.05 | 1.54 |
| knn | 39.6% | 19.96 | 3.17 | 8.47 |

Unguided comparison:

- `output_videos/motion_fusion_full_pomona.json`
- `output_videos/motion_fusion_full_pomona_debug.mp4`
- Fusion best-cluster hit @50 px was 80.6%, but accepted-cluster hit @50 px was only 23.7%.
- Interpretation: candidate-level fusion can find useful motion, but an unguided tracker can drift. The main-pipeline ROI/Kalman guide is the important stabilizer.

Current conclusion:

- Candidate-level late fusion is promising when gated by the main tracker/ROI path.
- `delta3` is the strongest single method in this guided run; `delta2` is cheaper and still useful.
- KNN contributes extra candidates but is weak alone and slower to extract; keep it optional.
- The next production-style experiment should feed fused motion clusters into a crop/ROI proposal stage rather than replacing final ball localization.

## Immediate Next Steps

1. Run the current pipeline on a short known clip with `--tracking-json`, `--info`, and motion diagnostics enabled by JSON output.
2. Inspect the current baseline: effective FPS, source counts, motion reason counts, large jumps, and whether misses are due to `no_motion_mask`, `no_blob`, `blob_too_far_from_search`, or selector rejection.
3. Run `experiments/motion_speed_probe.py` against the same clip/range to benchmark candidate generators outside the production loop.
4. Use the experiment to choose one narrow change:
   - loosen/reshape blob filtering,
   - compare raw delta vs adaptive S+V for recall,
   - tune ROI widening/loss behavior,
   - tune color gate from hard support toward soft scoring,
   - or add a simple MOG2/KNN candidate mode if it beats current masks.
5. Promote only the winning change and validate with before/after reports.

## Open Questions

- Which clip/frame ranges show the perceived improvement or failure most clearly?
- Do we care first about higher FPS, fewer misses, fewer false parked-ball snaps, or smoother bounce recovery?
- Is the target run mode every-frame YOLO, skip-frame YOLO, or motion-heavy recovery with fewer YOLO calls?
- Should the first experiment compare against annotations, the selector's chosen tracking JSON, or both?

## Motion-Only Temporal Viewer Pass - 2026-05-26

User feedback on the first adaptive/delta3 motion-only render:

- Black/translucent views made the motion easy to see, but there were random full-frame / background flicker dumps.
- Simple area cleanup helped but did not remove persistent fence/tree/court-line flicker.
- The better direction is the prior `tracknetv4_test/motion_temporal_experiment.py` idea: require temporal support, not just a one-frame delta hit.

Why the old temporal experiment was slow:

- It used a stacked temporal window and percentile aggregation over the full image every frame.
- It ran several full-frame connected-component passes for baseline, temporal, rejected, hue/compactness, and top-K filtering.
- Optional stabilization added phase correlation / warping work.
- Debug output rendered a large multi-panel 1080p video.

Fast replacement added to `experiments/motion_speed_probe.py`:

- New `temporal` method, now the default method.
- Uses rolling binary support counts instead of `np.percentile`:
  - compute current V/S motion energy,
  - add a low-threshold support mask to a ring buffer,
  - accept current motion only when enough recent frames support it,
  - require `temporal-burst-support=2` so single-frame dumps do not pass as bursts.
- Added pure motion debug view:
  - `--debug-view motion`
  - `--motion-background black|translucent`
  - `--skip-candidates`
- Added visualization cleanup:
  - `--mask-keep-largest-components`
  - optimized small-component cleanup so it does not run twice when morphology is disabled.

Chosen command shape:

```powershell
conda run -n tennis-analysis python experiments\motion_speed_probe.py `
  --input "input_videos\PomonaPitzer Women vs. UCSD-cut-merged-1773082079341.mp4" `
  --start-frame 0 `
  --max-frames 0 `
  --downscale 0.5 `
  --methods temporal `
  --skip-candidates `
  --open-size 1 `
  --min-component-area 80 `
  --mask-keep-largest-components 8 `
  --prefilter gaussian `
  --prefilter-ksize 3 `
  --temporal-thresh 13 `
  --temporal-window 5 `
  --temporal-min-support 3 `
  --temporal-support-lo-frac 0.70 `
  --temporal-burst-mult 2.0 `
  --temporal-burst-support 2 `
  --temporal-soft-open-sum-thresh 2.5 `
  --temporal-post-dilate 1 `
  --debug-view motion `
  --motion-background black `
  --motion-visual-dilate 2 `
  --output-json output_videos\motion_temporal_fast_full_black.json `
  --debug-video output_videos\motion_temporal_fast_full_black.mp4 `
  --no-progress
```

Full Pomona outputs:

| Output | Frames | Effective FPS | Mask ms/frame |
|---|---:|---:|---:|
| `output_videos/motion_temporal_fast_full_black.mp4` | 2022 | 31.76 | 12.60 |
| `output_videos/motion_temporal_fast_full_translucent.mp4` | 2022 | 24.42 | 14.51 |

Current conclusion:

- The temporal support gate is the right replacement for raw adaptive/delta3 visualization when the goal is to see physical motion without single-frame dumps.
- The fastest useful version is `temporal + rolling support + skip candidates`.
- The remaining cost is mostly video decode/encode, color conversion, connected-component cleanup, and rendering at 1080p.
- If this becomes production-facing, run it on an ROI or lower-resolution mask and only run component cleanup after a high component-count/density trigger.

Superseded decision:

- The temporal idea is useful but still too slow/noisy for the current experiment direction.
- Removed temporal from `experiments/motion_speed_probe.py`; the default probe methods are back to `adaptive_sv` and `delta3`.
- Copied the original temporal experiment into `experiments/motion_temporal_experiment.py` as a parked standalone reference.
- Near-term work should stay on the simpler motion view and ROI/YOLO experiment path unless temporal support gets rewritten as ROI-only or GPU-first.

## Temporal Rewrite Check - 2026-05-26

File checked:

- `experiments/motion_temporal_experiment_rewrite.py`

Bug fixed:

- Script failed when run directly because it tried `.video_io`, then plain `video_io`.
- Patched the import path so direct script execution can fall back to `tennis_tracker.video_io`.

Runs completed on `input_videos/PomonaPitzer Women vs. UCSD-cut-merged-1773082079341.mp4`:

| Run | Frames | Size | Effective FPS | Notes |
|---|---:|---:|---:|---|
| `rewrite_full_fast` | 2021 | 1920x1080 | 34.60 | Full-res CUDA, `--fast`, no debug video |
| `rewrite_full_fast_960` | 2021 | 960x540 | 106.10 | CUDA, `--fast`, practical speed setting |
| `rewrite_roi_smoke_780_899` | 120 | 1920x1080 | 38.69 | ROI path works, but still does full-frame GPU mask compute |
| `rewrite_debug_330_419` | 90 | 960x540 | 42.52 | Debug comparison video generated |

Full-res bottlenecks:

- `cc_temporal`: 11.26 ms/frame
- `masks_gpu`: 9.06 ms/frame
- `transfer`: 2.89 ms/frame
- `extract_vs`: 2.27 ms/frame

960px bottlenecks:

- `cc_temporal`: 2.86 ms/frame
- `masks_gpu`: 2.43 ms/frame
- `extract_vs`: 1.40 ms/frame
- `transfer`: 0.80 ms/frame

Feedback:

- The rewrite is much better as an offline/debug experiment than the original because debug output is off by default and `--fast` skips the baseline arm.
- It is still not worth moving into the main hot path at full resolution: full-res no-video is only ~35 FPS, and any video render or YOLO work will push it lower.
- The practical mode is `--resize-max-side 960 --fast --no-stabilize-global-motion`, where it runs ~106 FPS without video output.
- ROI helps transfer but not enough, because full-frame GPU temporal masks are still computed before ROI cropping. A true fast production version would need ROI-first temporal state or tiled/local temporal history.
- Visual quality on the debug range is cleaner than raw delta/adaptive dumps, but it still detects player body motion and some scene motion. It is a candidate/confidence layer, not a final ball detector.
