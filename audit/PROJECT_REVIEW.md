# Tennis tracker: project, correctness and model review

Reviewed September 7, 2026 against the current, already modified worktree.

**Historical review:** the findings and reproduction results below describe
the pre-fix worktree. The subsequent authorized repairs, color-off default and
verified video13–video53 runs are recorded in [FIX_RESULTS.md](FIX_RESULTS.md).

**Recommendation: repair evaluation and label integrity first, then motion
candidate selection and time-consistent prediction, then experiment with a
better temporal detector.** A larger network is not yet the best-supported next
step. Several current metrics can reward the wrong model, and two label-tool
mechanisms can undermine otherwise careful annotation work.

This pass adds documentation and reproducible diagnostic scripts. It does not
modify production logic, existing labels, configuration, training exclusions or
model weights. Findings below distinguish reproduced defects, static code
findings, measured results and proposed experiments.

## 1. What the project is and how it works

The project is an offline, single-camera tennis ball tracker. It recognizes the
ball from short video sequences, filters detections into plausible trajectories,
uses image motion to recover some missing positions, and optionally draws court
events and trajectories. Its other substantial product is the workflow for
importing, prelabeling, manually correcting and fine-tuning those sequences.

### Detection and frame timing

`clean_tracker.py` constructs a `Config` and calls `tennis_tracker.pipeline.run`.
The default ball model is `models/gridtracknet_weights_torch.npz`, not the legacy
`ball.engine`. GridTrackNet has 8,891,471 convolution parameters. Five RGB images,
resized to 768x432, are concatenated into a 15-channel tensor. Thirteen 3x3
convolutions and four max-pooling stages produce 15 channels at 48x27: confidence,
x offset and y offset for each of the five frames. A decoded cell represents a
16x16 region of the resized input. Decoding keeps only the highest-confidence
cell for each frame; competing plausible locations are discarded before the
selector sees them.

The input contains temporal information but is processed by 2D convolutions
over concatenated frames. There is no recurrent state or transformer. The
width-axis normalization intentionally reproduces an unusual upstream export.
Its gamma, beta, mean and variance are buffers. Fine-tuning updates convolution
weights but not normalization values or statistics. Calling `train()` does not
change that. A fresh model starts with effectively fixed identity normalization,
not trainable conventional BatchNorm. This is a compatibility choice, not proof
that training cannot converge.

At about 30 FPS, frames are grouped in units of five. At about 60 FPS, independent
even/odd streams preserve approximately 30 FPS spacing while producing outputs
for both phases. A background prepass runs ahead of the main processing loop,
batching up to four units. This is throughput-oriented offline processing with
future-frame access, not a causal live tracker. Both code paths also accept
22–32 FPS as stride 1, so their advertised 30 FPS contract is broader in practice.

### Motion and selection

The main loop decodes the video separately, estimates a running background and
variance, and builds motion masks. Temporal differences, morphology, component
filters, court-side exclusions and ROI restrictions all affect which motion is
available. Raw RGB remains the GridTrackNet input. HSV brightening and static
dimming belong primarily to the alternative detector/debug paths.

TensorRT supplies player boxes and court keypoints. BoxMOT ByteTrack maintains
player IDs. A separate ROI tracker limits motion work around ball hypotheses;
full-frame searches periodically resume when the ball is lost. These ROIs
therefore influence recovery evidence even though they do not crop GridTrackNet.

The selector builds greedy nearest-cost tracks with continuity guards and a
four-state image-space Kalman filter. It scores tracks using observation count,
span, court proximity, motion/kinematic evidence and player proximity. It then
selects a timeline of tracks, fills suitably anchored short gaps and follows
brief tails only with motion support. Finally, a robust local polynomial refit
smooths or relocates positions. Selection is retrospective: it sees the full
sequence. Changing a later track can change earlier decisions.

The renderer consumes those selected positions. JSON records per-frame results,
sources, player boxes, court keypoints, configuration and optional diagnostics.
The optional court minibar combines CatBoost bounce candidates, kinematic turns
and player contacts into schematic rally legs. It is not a validated line-call
or serve-speed measurement system.

### Fine-tuning

`finetune/ft.py` coordinates import, drafts, manual labels, checks, camera tags,
evaluation, training and promotion. Training extracts JPEG frames at model size,
forms overlapping five-frame units advancing two label positions, applies a
shared horizontal flip and photometric augmentation, and optimizes a focal-like
confidence loss plus positive-cell offset loss with Adadelta and AMP.

Validation currently selects the minimum-loss epoch, then compares that epoch's
detector metrics with the starting model. Only those two checkpoints compete.
The winner is saved separately; `ft.py promote` copies it into the runtime model
location. There is no evidence that every epoch is compared by deployed tracking
quality. A checkpoint with worse validation loss but better localization can be
missed by this process.

## 2. Evidence boundaries and current measurements

### User's unfinished-label exclusions

The following are **not usable accuracy or model-selection evidence** until
reviewed: grid_match21, 24, 26, 31, 37, 40, 47, 49, 50, 55, 73, 78, 85, 86, 87,
89 and 92; video3, video8, video11 and video12. All bare match numbers refer to
GridTrackNet. "The one after grid_match31" remains unresolved; no additional
GridTrackNet clips were evaluated after this clarification.

video11 and video12 have not been reviewed at all. An earlier exploratory run
included them before the clarification. Those measurements are withdrawn as
quality evidence. `raw_eval.txt` was regenerated using **video10 only**. No
automatic annotation audit/fix or label realignment was run.

### Checks actually performed

| Check | Result | What it establishes |
|---|---|---|
| `clean_tracker.py --self-test` | Passed | Existing deterministic selector, decoder and court examples |
| `train_gridtracknet.py --self-test` | Passed | Existing target/loss/sampling/model-shape examples |
| `ft.py check`, before user exclusions | 86 files, 0 format problems | Syntax/cadence/range checks only; not label correctness |
| `audit/review_checks.py` | 13 issues reproduced | Synthetic, isolated current-behavior counterexamples |
| Full GPU Pomona run, JSON only | Completed | Default detector, TensorRT auxiliaries, CUDA motion, selection, JSON and annotation validation |
| Frozen Pomona parity check | Failed | Current output does not meet several historical gates |
| Raw video10 threshold sweep | Completed | One permitted workspace clip, current weights and labels |

The full run used Python 3.10.19, torch 2.10.0+cu128, NumPy 1.26.4, OpenCV 4.9.0,
FilterPy 1.4.5, TensorRT 10.15.1.29 and an RTX 5080 Laptop GPU. Full versions,
source hashes, label hashes and model hashes are in `snapshot.json`. The runtime
warned that its TensorRT plans were created for a different device model, but
this run completed. No engine rebuild was attempted.

On the 2,022-frame, approximately 59.7 FPS Pomona sample:

- 74 of 76 labeled visible frames had a selected position: presence recall 0.974.
- Mean error 2.61 px; median 1.89 px; p90 4.93 px; maximum 20.34 px.
- Three of four labeled absent frames had false outputs.
- 1,110 selected positions: 1,086 `det`, 15 `interp`, 9 `motion`; zero consecutive
  jumps over the validator's 120 px threshold.
- Runtime-reported total was 42.2 seconds. The detector prepass took 22.9 seconds
  and overlapped the main loop. These times must not be added as independent
  costs. This is one measured run, not a throughput distribution.
- The frozen parity gate failed recall, mean error, p90 error, within-5-px count
  and filled-frame count. Input and annotation hashes passed. The old baseline
  expected 687 present frames, while current output has 1,110. That does not prove
  every extra position is false; it does prove the frozen contract no longer
  passes. The cause cannot be assigned to one preexisting edit from this run.

Pomona has only 80 labeled frames and is not proof of generalization. Its four
absence labels are particularly inadequate for estimating false-positive rate.
Do not optimize to filled-frame count or smoothness alone: stationary clutter
and invented trajectories can improve both.

Permitted raw-detector results on video10, with 215 visible and 11 absent labels:

| Threshold | Recall within 10 px | Wrong object >30 px | Miss rate | False alarms on absent labels |
|---|---:|---:|---:|---:|
| 0.30 | 0.907 | 0.051 | 0.009 | 0.818 |
| 0.50 | 0.884 | 0.028 | 0.070 | 0.455 |
| 0.70 | 0.795 | 0.009 | 0.186 | 0.364 |

Errors are evaluated at a 1920-pixel-wide reference scale. This table is not a
recommendation to adopt 0.70: the recall loss is substantial and only 11 negative
labels were available. Source comments call video10 held out, but complete
checkpoint training lineage is not recorded, so historical independence is not
independently established. Video metadata advertised 227 frames; the decoder
delivered 226. Those 226 frames supplied the scored labels.

## 3. Reproduced findings and proposed fixes

`review_checks.py` uses synthetic frames, mocked detector outputs and small
in-memory objects. It does not test any excluded match. Its assertions confirm
the defects still exist; they are not a green correctness suite. All thirteen
outputs are saved in `review_checks.json`.

### P1 — Training recall has the wrong denominator

Location: `finetune/train_gridtracknet.py:404`, especially line 446.

Predictions with error above 10 px but at most 30 px increment neither hit, wrong
nor miss. Then `visible = hit + wrong + miss` silently removes those labeled
frames. A synthetic five-frame clip with one exact prediction and four 20 px
errors reports one visible frame and 100% recall. Correct recall is 20% over five
visible frames. The separate archive evaluator does include this middle band.

Fix: count ground-truth visible labels independently and explicitly report
localization errors in the middle band. Share one scoring definition between
training and archive evaluation. Regression: the five-frame example must have
denominator five and recall 0.2. Re-score candidate checkpoints after fixing it;
do not trust old winner comparisons as equivalent metrics.

### P1 — Realignment can move human-pinned frames

Locations: `finetune/realign.py:202`, `:291`, and its apply loop.

Pins use a finite penalty of 6,000, not a hard constraint. A 220-row synthetic
sequence chooses offset +1 for a pinned frame because enough other rows reward
the shift. Independently, non-unit-rate searches pass `pinned=None`; if rate 0.5
or 2 wins, pinned frame indices can change. Invisible rows are excluded from
the visible-row optimization, so reviewed absence frames lack equivalent
protection. Review sidecars are not consistently remapped with rewritten rows.

Fix: enforce the exact original source frame for every settled row under every
candidate rate, or reject incompatible transforms. Use impossible transition
costs for forbidden choices and report infeasibility. Never drop a settled row
as unplaceable. Prepare labels, media mapping and sidecar updates together before
replacing anything. Test visible and invisible pins, rate changes, collisions,
cut boundaries and repeated realignment. Do not apply this tool to repair the
user's review list automatically.

### P1 — Label-tool fingerprints confuse duplicate frames with wrong seeks

Locations: `finetune/label_tool.py:537`, `:566`, `:577`.

The reverse index maps a frame fingerprint to its first occurrence. Two identical
sequential decoded frames cause correct frame 1 to be recognized as frame 0 and
rejected. Frozen footage, duplicated source frames, or differences lost by the
96x54 fingerprint can all create this ambiguity. The fallback in `frame()` can
cache a black image after failed retries. Separately, a seek into previously
unseen footage can still be accepted using the decoder's own claimed index,
contrary to the stronger guarantee in its docstring.

Fix: distinguish trusted sequential decode from uncertain seeks. A repeated
fingerprint is not evidence to override a known sequential index. For seeks,
match a sequence against an anchor neighborhood, permitting multiple occurrences;
if ambiguous, decode forward from a verified anchor. Do not cache a failed read
as a valid frame or permit a label decision on an unverified placeholder. Test
duplicated frames and a simulated one-frame seek offset without a GUI.

### P1 — The progressive prepass mutates already published misses

Locations: `tennis_tracker/detectors.py:333`, `:340`, `:365`.

The tail overlap can revisit frames already declared final. Existing detections
are protected by `cached[source_index]`, but published empty detections are not.
A synthetic 23-frame run publishes frames below 20 with frame 18 absent, then
fills frame 18 while handling the tail. A consumer that already read frame 18
retains a miss; a later consumer sees a detection. The advertised equivalence of
background and synchronous prepasses is therefore not guaranteed.

Fix: maintain an explicit finalized status independent of detection presence.
Tail handling must predict only unresolved frames, or hold the overlapping
suffix unpublished until all contributing predictions are fused. Use the same
deterministic window policy for inference, labeling and evaluation. Test both
30/60 FPS phases with consumers at different schedules, including empty outputs.

### P1 — Async writer shutdown can deadlock after encoder failure

Location: `tennis_tracker/video_io.py:204`, also `write()` and `_drain()`.

`close()` blocks inserting its sentinel into a full queue before inspecting
`_thread_error`. A failed worker cannot drain that queue. The reproducer creates
exactly this state and confirms shutdown remains blocked until the probe frees
a slot. `write()` has a related check-then-block race if the consumer fails after
the initial error check.

Fix: use bounded queue waits that repeatedly check worker failure/liveness;
ensure shutdown can cancel pending work and release encoder resources in a
`finally` path. Surface the original encoder error. Also check OpenCV writer
`isOpened()`. Test failure with a full queue; do not infer correct shutdown from
a successful encoding run.

### P2 — Training evaluation drops trailing frames and differs from deployment

Location: `finetune/train_gridtracknet.py:404`, especially `unit.clear()` at 428.

The evaluator only emits complete five-frame units and never flushes the final
partial unit. A seven-frame perfect synthetic sequence scores 5/7 because the
last two frames are never inferred. Deployment supplies an overlapping trailing
window. At 60 FPS, each phase has its own tail. Training evaluation also omits
the runtime's configured 1.6 px-at-1080p y offset.

Fix: reuse a single prediction/windowing implementation, resolve the publication
bug first, and define a policy for clips shorter than five samples. If metrics
are intentionally for an uncalibrated raw model, name them as such and separately
evaluate the deployed decoder. Test lengths around 5/10-unit boundaries and odd
and even phases.

### P2 — Winner selection ignores absent-ball false alarms

Location: `finetune/train_gridtracknet.py:456`.

`better()` reads recall and wrong-object rate but ignores `false_alarm`. A
candidate moving from 0% to 100% false alarms beats the incumbent for a one-point
recall increase if wrong-object rate stays constant.

Fix: specify an explicit false-alarm acceptance constraint alongside visible
localization quality. Evaluate enough annotated no-ball frames for that gate to
be meaningful. Save counts, rates and clip-level results, not only one score.
Do not use this review's tiny negative subsets to decide a production tolerance.

### P2 — Small real motion blobs are rejected by synthetic detection area

Locations: `tennis_tracker/detectors.py` prepass radius and
`ball_in_play_selector/core.py:139`.

GridTrackNet predicts a point, then wraps it in a fixed resolution-scaled box.
At 1080p the box area is about 174.7 px². Motion contours must have at least
20% of that area: about 34.9 px². A 3x3-pixel ball blob has nine foreground pixels
but contour area only four, so it is rejected despite being centered exactly on
the prediction. One-pixel or line-like streaks can have zero contour area.

Fix: use connected-component pixel counts and a separately estimated ball-size
prior. Do not treat a fabricated detector box as a measured ball diameter. Test
2–5 pixel balls, elongated blur, close balls and neighboring player/line clutter.
The fix must retain precision; removing every area gate is not justified.

### P2 — CUDA motion ignores the configured ball-color gate

Locations: `tennis_tracker/motion.py:189`, helper at 790, CUDA path at 817.

The CPU refinement invokes the loose tennis-color gate. The CUDA helper exists
but is never called. With the gate enabled, a synthetic red moving patch leaves
100 raw foreground pixels in the CUDA implementation and zero in CPU refinement.
The test supplies CPU tensors to exercise the same tensor code without requiring
another device; the missing helper call is independently visible in source.

Fix: make both paths implement the stated policy and test their masks with
rounding tolerance. Benchmark on cleared labels before enabling a newly effective
hard gate: washed-out, night or blurred balls can fail color thresholds too.
Longer term, prefer a soft color likelihood over unconditional rejection.

### P2 — Drag changes with video frame rate

Locations: `ball_in_play_selector/physics.py:56`, `:76`, `:188`, `:202`.

Gravity scales with frame rate but drag stays 0.985 per frame. After one second,
horizontal velocity retains 63.5% at 30 FPS and 40.4% at 60 FPS for equivalent
initial motion. Changing the source frame rate therefore changes the physical
prior substantially.

Fix: express damping per second or convert a 30 FPS reference multiplier using
`drag_dt = drag_30 ** (30 * dt_seconds)`. Normalize covariance, motion windows and
background adaptation as well. Verify equivalent 30/60 FPS trajectories at equal
timestamps, including bounces and missing observations.

### P2 — ROI/projectile and Kalman predictions use different integration order

Locations: `ball_in_play_selector/physics.py:40`, `:163`, `:196`.

The projectile helper advances position before damping velocity; the Kalman
predictor damps first. Starting at (100,100) with velocity (30,-10), their
five-frame predictions differ by about 2.28 px. This discrepancy grows with
speed/horizon. The Kalman transition matrix also does not encode the drag used
to modify the state, so covariance propagation is inconsistent with the stated
state dynamics. `predict_dt()` does not reproduce optional depth-aware gravity.

Fix: implement one transition model for state and covariance and use it for ROI,
association and future prediction. Check one multi-step prediction against
repeated one-step predictions. See the physics recommendations below.

### P2 — Exported `det` positions are not always raw detections

Location: `ball_in_play_selector/core.py:419`.

The refit moves coordinates but only relabels detector points after a large
threshold, approximately 44 px at 1080p. A synthetic raw y=103 becomes 100.453
and remains `source='det'`. This is incompatible with interpreting green points
as untouched observations or scoring selected `det` coordinates as raw detector
accuracy. The bounding box may still represent the old location.

Fix: preserve immutable measurement coordinates and confidence, then export
filtered/display coordinates separately with explicit provenance. Evaluate each
layer against its own output. Keep the existing small smoothing behavior if it
helps rendering; make its meaning honest rather than disabling it blindly.

### P2 — Validator recall measures presence, not correct localization

Location: `validate_tracking.py:172`, `:188`.

A prediction more than 1,258 px from a visible label still gives recall 1.0.
The error statistics reveal it, but a `--min-recall` gate alone accepts it.
`absence_precision` is actually a true-negative rate over absent labels.

Fix: name these measures `presence_recall` and `absence_specificity`; add
localization recall at defined tolerances with all visible labels in the
denominator. Retain localization error distributions and explicit false alarms.
Change schemas/gates deliberately so existing consumers do not silently change
meaning.

## 4. Additional code and workflow findings

These are source/inventory findings, not additional match-quality measurements.

1. **Default training includes unchecked data.** The current manifest marks both
   video11 and video12 `custom-uncorrected`; video11 says eval-only. `find_clips`
   does not enforce that, and the default split puts both in training. One
   1,837-row clip has source `unknown`. Add explicit label-quality eligibility
   and split fields. This review records the defect without changing training
   policy on the user's behalf. The user's full review list is broader than
   `custom-uncorrected` and must be respected independently.
2. **Groups are metadata, not enforced splits.** Training splits by clip name
   and ignores `clips.csv` group/section. Multiple rallies of the same recording
   or public test game can enter train and validation separately. Existing
   `own-camera` grouping is too coarse to identify independent recording sessions.
   Add recording/session IDs and enforce disjoint groups. Do not call game10 a
   published test result after using its clips for training or model selection.
3. **Pooled raw sweeps ignore resolution scaling.** `evaluate_archive.py:225`
   computes a scale in the loop but calls `score(pairs, 1.0)`. Individual clip
   scores do scale correctly. Fix by scaling each pair before pooling, or
   pooling counts computed at each clip's reference scale. The retained
   video10-only evidence uses 1920-wide media and is unaffected.
4. **Training/inference resize mismatch.** Training uses INTER_AREA plus quality
   92 JPEG caching; inference uses OpenCV's default resize interpolation on the
   decoded source. For subpixel-sized balls this can change the signal. Share
   preprocessing, measure JPEG-cache effects, and version caches when changing it.
5. **Court geometry is reused across time.** Court detection is every 400 frames,
   low-confidence detections retain old geometry, and selector scoring uses the
   last valid polygon for the whole video. Pans/cuts can invalidate motion
   exclusions and court scores for earlier or later frames. Add shot boundaries,
   court confidence/reprojection checks and per-shot geometry.
6. **Depth-aware gravity is currently disconnected.** The selector accepts
   `court_keypoints` but does not attach them to its filters; no caller invokes
   `BallKalmanFilter.set_homography`. The unused homography helper assumes a
   corner order different from the renderer's active semantic order. Wiring it
   in without geometric validation would introduce another error.
7. **Non-default GPU path needs verification.** `preprocess_frame_cuda` uses
   `torch.device('cuda')` for new allocations while the pipeline uploads to the
   explicit configured device. This can mix devices in preprocessing/debug
   paths on multi-GPU machines. Use the input tensor's device consistently.
   Only device 0 was available for this pass; multi-GPU failure was not executed.
8. **Label-shift metadata is not transactional.** `ft.shift_labels` rewrites CSV
   indices without moving review/suspect sidecars, and does not keep a dedicated
   pre-shift CSV backup. Several sidecar writes and promotion copies are direct
   writes rather than temporary-file replacement. Protect labels and metadata
   together and validate a candidate model's shape/load before promotion.
9. **Reproducibility is incomplete.** There is no complete installation lock or
   checkpoint lineage record. Default run directories can overwrite prior run
   artifacts. The engine builder expects source `.pt` models that are not in the
   inspected file inventory; bundled engines alone do not guarantee rebuilds.
   Record package versions, seeds, input/label/model hashes, split membership,
   transforms and parent checkpoint. NPZ weights are inference artifacts, not
   complete optimizer/AMP-resume checkpoints.
10. **Some controls/comments are stale.** Auto-scaling overwrites several fields
    and the compact selector uses its own fill duration. A `motion_max_gap_frames`
    value is not a promise that this selector will fill that many frames.
    Background-channel references in the fine-tuning README do not describe the
    current 15-channel RGB model. Trace live callers before tuning a knob.

## 5. Motion and tracking changes worth trying

### First: preserve useful evidence

Fix small-component rejection and CPU/CUDA policy drift before widening search
radii. Keep component pixel count, elongation, brightness/color evidence and
centroid uncertainty. A blurred ball is a streak, so a circular shape prior
should weaken at high speed. Estimate measurement uncertainty along the streak
instead of inventing an exact point with a fixed box.

Retain a small number of separated local confidence maxima from GridTrackNet
instead of immediately throwing away all but one. Gate those candidates against
the trajectory, motion and court/player context. First try two or three candidates
with existing association; measure candidate recall and false switches before
adding a more elaborate assignment system. Top-K exposes ambiguity; it does not
make low-scoring cells trustworthy or guarantee a better selected track.

Use weak detections to continue an established track under tight geometric
support, with a stronger requirement to spawn/reacquire a track. A global
threshold reduction increases candidate recall and false alarms together, as
video10 illustrates. Continuation and new-track decisions need different evidence.

### Camera motion and recovery

Estimate background motion at reduced resolution while masking players and
candidate balls. Start with translation or a robust affine transform; use a
homography only when justified by the view. Warp the background/masks into the
current frame before differencing. Reset on cuts, failed alignment or abrupt
court changes. OpenCV provides feature-based transforms and ECC alignment;
confidence checks and fallbacks still belong to this application.
[OpenCV homography tutorial](https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html)
and [tracking/alignment APIs](https://docs.opencv.org/4.5.0/dc/d6b/group__video__track.html).

Represent motion search regions as uncertainty ellipses oriented by velocity,
with time-dependent growth and calibrated upper bounds. A fixed circular radius
cannot simultaneously handle a stationary near-player ball and a fast streak.
Reinitialize after a hit rather than letting uncertainty grow indefinitely.
Use local optical flow only as corroboration when texture is adequate; tiny
blurred balls and player edges are a poor basis for trusting flow alone.

Avoid connecting separate rallies across a cut or long gap. Conversely, do not
hard-reject every above-court ball: lobs legitimately project outside the court
polygon. Treat court/player features as likelihoods, not physical proof. Allow
an unresolved result to remain absent, and measure gap durations rather than
maximizing the count of nonempty outputs.

### Performance based on the measured run

The selector took roughly 0.73 seconds; preprocessing and auxiliary detection
dominated main-loop costs. Profile player detection intervals with prediction
between observations, mask transfer/packing and ROI components before rewriting
the selector for speed. Cache component summaries for frames searched repeatedly.
Keep per-stage timings honest about overlap: the near-zero main-loop ball detect
time is waiting/cache consumption, not the total neural inference cost.

Do not lower player frequency solely from this timing result: contact attribution
and near-player gap vetoes must remain accurate. Rebuild TensorRT engines for the
actual GPU only when the source models and export contract are available.

## 6. Physics: what can be justified from this camera

### A consistent image-space predictor is the immediate improvement

Use a state such as `[x, y, vx, vy]` with velocities in pixels/second and actual
elapsed time. A constant-velocity transition has position terms `vx*dt`, `vy*dt`.
For a fitted acceleration, add `0.5*a*dt²` to position and `a*dt` to velocity.
Propagate covariance with the same transition and time-scaled process noise;
for continuous white acceleration, each axis has the familiar
`q * [[dt³/3, dt²/2], [dt²/2, dt]]` position/velocity block.

Compare a constant-velocity baseline with locally fitted acceleration and the
current gravity prior on identical synthetic timestamped trajectories and later
on reviewed clips. Real flight often looks approximately curved over a short
interval, but global downward pixel acceleration is not a universal law. A
ball's movement toward/away from the camera changes its projection too.

Use normalized innovation and measurement uncertainty for association. At a
credible hit/bounce, reset or expand velocity uncertainty and refit the next
segment. Do not reflect vertical image velocity merely because one measurement
lies above a prediction. Racket hits, camera movement, depth change and detector
switches can look similar. Calibrate confidence-to-error empirically; heatmap
confidence alone is not a localization standard deviation.

### Court homography is not an airborne-ball calibration

A court homography maps points on the ground plane. It is appropriate for player
footpoints and a ball contact whose ground location is actually supported.
Applying it to an airborne ball does not recover the ball's ground projection
or height. The existing sideways pixels-per-meter helper is not the image
projection of vertical gravitational acceleration. The plane assumption follows
from the projective geometry documented in
[OpenCV's homography tutorial](https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html).

For measured 3D flight, obtain camera intrinsics/distortion and extrinsics from
known court geometry, then fit a world-space trajectory through the camera
projection with contact and timing constraints. With only one camera, depth and
height can remain ambiguous; quantify that uncertainty and validate against
independent measurements. A second synchronized view is the stronger path to
reliable 3D reconstruction. Do not introduce precise km/h, height or in/out
claims from the current pixel model.

Only after calibration and reliable events should quadratic aerodynamic drag,
surface-dependent restitution and tangential bounce losses be estimated. Spin/
Magnus forces add poorly observed parameters and should wait for evidence they
improve held-out trajectories. Do not tune tennis flight from shuttlecock data;
cross-sport visual pretraining and physical dynamics are different questions.

## 7. Fine-tuning and architecture roadmap

### Data and evaluation before architecture

Create an explicit label-quality state: reviewed, imported-unverified,
pseudo-labeled, disputed. Represent visible, occluded, out-of-frame and unknown
separately; the broad top-right sentinel currently makes some legitimate image
locations indistinguishable from absence. Keep old CSV compatibility through an
explicit conversion. Training may use partial labels, but unknown frames need
loss masking, not forced no-ball targets.

Finish the user's review queue before using those clips as evidence. Preserve
manual confirmations and source timestamps. A detector can propose an alignment
or suspicious frame; agreement with the same detector is not independent truth.
Keep human decisions authoritative. Split by recording session/match before
window creation, and reserve a final test set that is not used for threshold,
architecture, early-stop or checkpoint decisions.

Mine hard negatives and difficult positives from cleared training footage:
player shoes/rackets, court lines, background vehicles, ball carts, motion blur,
shadow transitions, net occlusion, high lobs, tiny far-court balls and rally ends.
Sample empty windows deliberately. Maintain fully reviewed random stretches
alongside uncertainty-driven labeling so evaluation does not contain only the
frames the model already found difficult. Increase verified diversity before
simply oversampling the same recording six times.

Use temporal-consistent augmentation: the same geometric crop/flip and exposure
transform across the five frames, with exact coordinate transforms. Add measured
compression, directional blur and occasional dropped/repeated-frame experiments
with explicit time handling. Avoid producing impossible motion through unrelated
per-frame spatial transforms. Match cache and runtime resize behavior first.

### Architecture experiments in order

| Experiment | Concrete change | Expected purpose, not promised gain | Promotion evidence |
|---|---|---|---|
| A: existing model, corrected loop | Fix metrics, provenance, tails, preprocessing and reviewed sampling | Establish a trustworthy baseline | Cleared grouped validation, negative-frame results, final test |
| B: better decoding | Top-K local peaks; overlapping-window fusion with deterministic frame ownership | Recover ambiguity and reduce unit-boundary discontinuities | Candidate recall, selected switches, latency, odd/even consistency |
| C: local detail branch | Native-resolution crop around an uncertain prediction, with occasional global reacquisition | Recover small/blurred ball detail lost at 768x432 | Far-ball and blur subsets; lost-track recovery; compute budget |
| D: new temporal encoder | Shared lightweight per-frame encoder, stride-4/8 features, small temporal convolutions and a heatmap/offset head | Preserve spatial detail and model temporal interaction explicitly | Against A/B at matched latency, memory and training-data budget |
| E: motion/background features | Optional aligned difference or background feature branch | Help fixed-camera disambiguation | Ablation on fixed and moving cameras, failed-alignment fallback |
| F: learned gap/event model | Short masked temporal module predicting uncertainty and candidate contact events | Improve gaps that remain after detector improvements | Gap duration/precision and event timing on independently labeled events |

My preferred next architecture after A/B is **D**, optionally paired with a small
detail branch if far-ball errors dominate. Keep a full-frame path so a wrong ROI
does not permanently hide the ball. A heatmap plus subpixel offset and separate
visibility/uncertainty output offers cleaner contracts than fabricated fixed
boxes. Begin with the same five-frame cadence so comparisons isolate architecture.
For live requirements, use a causal time window; for offline analysis, allow
lookahead but report its latency explicitly.

Do not replace the current normalization in a weight-compatible patch. For a
new architecture trained with small batches, GroupNorm is a reasonable candidate
because it normalizes within groups and does not depend on batch statistics;
its original paper evaluates the small-batch problem. This is an experiment,
not evidence that it will improve this tennis model.
[GroupNorm documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html),
[original paper](https://arxiv.org/abs/1803.08494).

Relevant prior work provides hypotheses rather than transferable benchmark wins:

- [GridTrackNet's original implementation](https://github.com/VKorpelshoek/GridTrackNet)
  is the reference for this five-frame grid detector and its conversion contract.
  Preserve a known-input/output fixture before changing layout or normalization.
- [TrackNetV3 by the paper's implementers](https://github.com/qaz812345/TrackNetV3)
  investigates background information and trajectory rectification for
  shuttlecocks. Those ideas motivate E/F, but its sport, data and missing-point
  assumptions differ from this tennis application.
- [TrackNetV4](https://arxiv.org/abs/2409.14543) studies learnable motion attention
  and fusion with visual features. It motivates a small motion branch ablation,
  not an immediate wholesale replacement or a claim of superiority here.

A large video transformer, full 3D CNN, end-to-end detector/physics/racket system,
or learned spin head would introduce several moving parts before the existing
failure modes are measured reliably. Escalate model complexity only when a
controlled experiment shows where the simpler candidate fails.

### Metrics that should select the winner

Report raw detector and selected trajectory separately. Use localization recall
at 5/10/20 reference pixels, false outputs on absent labels, wrong-object rate,
median/p90 localization error, and longest/median missing-run duration. Count
all ground-truth visible frames. Break results down by source, recording, camera
motion, near/far court, occlusion and blur rather than pooling everything alone.

Add track-switch counts and recovery latency; evaluate event precision/recall
with a declared frame tolerance on bounce/contact labels. Preserve real sharp
contacts when measuring smoothness. Report per-clip macro results alongside
micro counts, and bootstrap uncertainty by clip/session rather than treating
neighboring frames as independent samples. Require sufficient negatives before
claiming a false-positive improvement. Record latency, peak GPU memory and
decoder/preprocessing settings for every candidate.

Use a held-out validation set to choose epochs/thresholds; evaluate the frozen
test set only after selection. Save parent checkpoint hashes, full resume state,
data/split hashes and model configuration. Repeatedly tuning on video10 turns it
into validation, regardless of an old comment calling it a holdout.

## 8. Recommended implementation sequence and review limits

1. Protect settled labels and fix duplicate-frame handling. Add the user review
   exclusions to evaluation policy without rewriting annotations.
2. Unify scoring, tail handling and finalized prepass ownership. Re-score old
   candidate models on cleared data before another promotion.
3. Fix writer error shutdown, preserve raw coordinates, and make motion policies
   consistent. Keep these changes separately reviewable from model tuning.
4. Normalize dynamics by time, align prediction/covariance, fix tiny-blob evidence,
   and add per-shot camera handling. Compare each change against the corrected
   baseline; permit safe gaps and retain negative-frame checks.
5. Fine-tune the existing model on more verified diverse examples. Then compare
   decoding, detail-branch and temporal-encoder experiments one at a time.

The review covers the active runtime, tracking/physics, evaluation and main
labeling/training paths, with targeted source inspection of export and court
events. It is not an exhaustive proof of every GUI control, notebook cell,
legacy backend or device combination. No training run, model promotion,
multi-GPU execution, full label visual audit or production physics calibration
was completed. No excluded clip was compared after the user's clarification;
the earlier video11/video12 measurements were withdrawn. MP4 encoding was not exercised by the main
benchmark; writer failure handling was tested synthetically. Model architecture
gains and real-world bounce/line-call accuracy remain unmeasured proposals.

## Artifacts and reproduction

- `AGENTS.md`: project initialization, commands, contracts and user exclusions.
- `audit/review_checks.py`, `.json`, `.log`: thirteen synthetic reproductions.
- `audit/collect_snapshot.py`, `snapshot.json`: environment, source/model/data
  hashes and descriptive inventory; inventory counts are not accuracy evidence.
- `audit/raw_eval.txt`: the permitted video10-only threshold sweep.
- `output/review_20260907_tracking.json` and
  `output/review_20260907_validation.json`: full Pomona run and validation.

Run the two existing self-tests and `audit/review_checks.py` with the interpreter
listed in `AGENTS.md`. For the saved benchmark, use
`check_parity.py --existing --predictions output/review_20260907_tracking.json`.
That command currently fails the five gates listed above; do not describe it as
passing. Existing user changes and model weights remain intact.
