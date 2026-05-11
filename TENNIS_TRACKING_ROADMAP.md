# Tennis Tracking Roadmap

## Current Decision

Do not replace YOLO first.

The current YOLO/TensorRT detector is useful and already produces real ball candidates. The most important weakness is the layer around the detector: motion candidates, orange-path connection, low-confidence recovery, selector decisions, and source confidence. Physics is already good enough to support the next step, but it cannot rescue disconnected or poorly scored evidence.

Current constraints:

- Do not skip YOLO frames for now. The user wants up-to-date movement within roughly one frame.
- Do not reduce player detection cadence yet, even though it is expensive. Player movement freshness matters for the current workflow.
- Treat 3D reconstruction as a later product phase, after 2D ball tracking is reliable.
- Prioritize quality and consistency before aggressive speed shortcuts.

The next move is:

1. Keep YOLO as the baseline detector.
2. Improve motion diagnostics.
3. Turn motion blobs into structured candidates.
4. Fix orange motion-to-track connection during YOLO gaps.
5. Measure every change against exported tracking JSON and hand labels.

Rejected experiment:

- Loosening selector-side tiny motion blobs increased orange acceptance but created unstable vertical motion spikes. Do not reapply that change without an explicit temporal/velocity confidence gate.

Candidate external baseline:

- GridTrackNet: `https://github.com/VKorpelshoek/GridTrackNet`
- It is a TensorFlow/Keras temporal model using 5 input frames and 5 output locations at 768x432 input resolution.
- It should be tested as a separate detector baseline before being wired into the main selector.
- Because the current `tennis-analysis` env does not include TensorFlow/Keras, test it in a separate environment first.

GridTrackNet local test notes:

- Cloned to `.codex_tmp/GridTrackNet`.
- Created isolated conda env: `C:\Users\Andrew\.conda\envs\gridtracknet`.
- Native Windows TensorFlow loads the weights but cannot run inference on CPU because GridTrackNet uses channels-first max pooling.
- Converted the Keras model to ONNX at `.codex_tmp/gridtracknet.onnx`.
- ONNX CPU inference succeeded on the Pomona cut and exported `output_videos/pomona_gridtracknet_onnx_tracking.json`.
- Result at confidence threshold `0.5`: 1216 / 2022 filled frames, 60.1%, but 13 consecutive-present jumps over 120 px.
- Threshold sensitivity from the same output:
  - `0.5`: 1216 frames, 13 jumps.
  - `0.6`: 1129 frames, 8 jumps.
  - `0.7`: 1033 frames, 4 jumps.
  - `0.8`: 893 frames, 3 jumps.
  - `0.9`: 621 frames, 0 jumps.

Interpretation:

- GridTrackNet is promising as a high-confidence temporal candidate source.
- It should not replace the whole selector at low threshold; it also jumps to false high-confidence positions.
- Best next integration test is to feed GridTrackNet detections into the existing selector with a high confidence threshold and physics/court gating.
- For fair speed, use ONNXRuntime-GPU, TensorRT, or WSL2 GPU TensorFlow. The current ONNX CPU run is only a correctness baseline.

GridTrackNet GPU status:

- Installed `onnxruntime-gpu==1.21.0` with CUDA/cuDNN runtime extras into the isolated `gridtracknet` env.
- Added `tools/run_gridtracknet_onnx.py` to run the converted ONNX model and export both JSON and an overlay video.
- Native Windows CUDA path works when NVIDIA package DLL directories are added before ONNXRuntime session creation.
- Pomona full-cut CUDA run at threshold `0.90`:
  - Output video: `output_videos/pomona_gridtracknet_cuda_thr090_reference.mp4`
  - Output JSON: `output_videos/pomona_gridtracknet_cuda_thr090_tracking.json`
  - End-to-end with video writing: 58.6 fps.
  - Model inference time: 8.7 sec for 2022 frames.
  - No consecutive-present jumps above 120 px at threshold `0.90`.

## Baseline Snapshot

Baseline clip:

`input_videos/PomonaPitzer Women vs. UCSD-cut-merged-1773082079341.mp4`

Generated artifacts:

- `output_videos/pomona_baseline.mp4`
- `output_videos/pomona_baseline_tracking.json`

Measured baseline:

- Video: 2022 frames, 1920x1080, 59.7 fps.
- Runtime: 107.9 sec.
- Effective end-to-end speed: 18.7 fps.
- Filled tracking frames: 684 / 2022, 33.8%.
- Tracks built: 39.
- Output source mix:
  - `det`: 504 frames.
  - `motion`: 90 frames.
  - `carry`: 90 frames.
- Large jumps over 120 px: 0.

Main speed bottlenecks:

- Player/court auxiliary detection: 35.7 sec.
- Ball detection: 26.4 sec.
- Preprocess/motion: 22.5 sec.
- Selector: 2.47 sec.
- Pass 2 render/write: 12.9 sec.

Important environment note:

- `ffmpeg` was not found, so NVENC hardware encoding was not used.
- This hurts output writing speed, but it is not the main tracking-quality problem.

## Priority Order

### 1. Validation And Benchmarking

Goal:

Create a repeatable measurement loop before changing tracking behavior.

Required work:

- Use `--tracking-json` for every baseline/improvement run.
- Label 50-100 high-value frames per clip.
- Focus labels around:
  - orange disconnects.
  - YOLO misses.
  - wrong ball grabs.
  - carry overuse.
  - bounce/contact moments.
  - adjacent-court interference.
  - parked-ball confusion.

Metrics:

- Filled-frame ratio.
- Pixel error against labels.
- Recall on visible labeled ball frames.
- False positives when labeled ball is absent.
- Source mix: `det`, `motion`, `guide`, `carry`, `interp`.
- Track switch count or large jump count.
- Runtime by stage.

Acceptance:

- No behavioral change should be considered better unless it improves at least one quality metric without creating obvious regressions in another.

### 2. Motion Diagnostics

Goal:

Make orange failures explainable.

Required work:

- Add per-frame motion acceptance/rejection reason codes.
- Export those reason codes into tracking JSON or a companion debug JSON.
- Count reasons globally and by frame range.

Candidate reason codes:

- `no_motion_mask`
- `no_blob`
- `blob_too_small`
- `blob_too_large`
- `blob_bad_shape`
- `blob_too_far_from_prediction`
- `blob_too_far_from_guide`
- `physics_reject`
- `near_player_reject`
- `court_reject`
- `guide_conflict`
- `owner_conflict`
- `accepted_motion`

Why this comes first:

Right now it is too hard to tell whether orange fails because the mask is bad, the blob exists but is rejected, or the selector prefers carry/guide.

### 3. Structured Motion Candidates

Goal:

Promote motion from a raw mask fallback into a real candidate stream.

Required work:

- Extract motion candidates per frame.
- Score each candidate using explicit features.
- Keep top candidates for debug/export.

Candidate fields:

- `frame`
- `x`, `y`
- `area`
- `bbox`
- `compactness`
- `aspect_ratio`
- `fill_ratio`
- `motion_strength`
- `distance_to_prediction`
- `distance_to_guide`
- `distance_to_last_ball`
- `court_distance`
- `near_player`
- `score`
- `reject_reason`

Acceptance:

- We should be able to inspect a frame and answer: “Was the real ball present as a motion candidate, and if yes, why was it or was it not selected?”

### 4. Orange Motion-To-Track Connection

Goal:

Make orange continue the ball path during YOLO gaps when motion evidence is plausible.

Required work:

- Add motion latching to the active ball state.
- Prefer motion over carry when:
  - candidate is close to predicted path.
  - candidate continues recent velocity.
  - candidate has acceptable size/shape.
  - candidate is inside or near the court region.
  - candidate does not require impossible acceleration.
- Avoid switching orange onto unrelated moving objects.
- Export source transitions so gaps can be reviewed.

Core behavior change:

- `det -> motion -> det` should be a normal healthy sequence.
- `det -> carry -> det` should be used when no good motion candidate exists.
- `carry` should not dominate if a plausible orange candidate exists.

Acceptance:

- Higher filled-frame ratio.
- More valid `motion` frames during detector gaps.
- No increase in large jumps or parked-ball grabs.
- Better continuity around known failure frames.

### 5. Low-Confidence Detection Recovery

Goal:

Recover true balls that YOLO sees weakly instead of discarding them too early.

Research basis:

ByteTrack improves tracking by associating low-confidence detections instead of only high-confidence detections.

Required work:

- Keep low-confidence ball boxes as secondary candidates.
- Associate them only when track continuity, motion support, and physics agree.
- Separate detection threshold for:
  - visible output.
  - candidate association.
  - new-track creation.

Acceptance:

- Fewer fragmented tracks.
- Better recovery after occlusion/blur.
- No major increase in false positives.

### 6. Motion Mask Quality

Goal:

Improve the current fast CUDA S+V motion path without adopting the slower temporal experiment.

Required work:

- Improve tiny/blurred blob preservation.
- Tune blob filtering around ball-size components.
- Improve shape scoring for streaks, partial balls, and afterimages.
- Add optional local optical-flow check around predicted ball regions.
- Add camera-shake/global-motion compensation only if diagnostics show camera motion is harming masks.

Do not do yet:

- Do not merge the slow temporal experiment into the main path.
- Do not do full-frame dense optical flow as a first option.

Acceptance:

- More real ball motion candidates survive filtering.
- Fewer body/court/noise candidates survive filtering.
- Preprocess runtime does not rise significantly.

### 7. Physics Gate Refinement

Goal:

Use physics as a better judge after candidate quality improves.

Required work:

- Add uncertainty per source:
  - `det`: highest confidence when strong.
  - `motion`: medium confidence, depends on candidate score.
  - `carry`: low confidence.
  - `interp`: not a real observation.
  - `guide`: depends on whether exact or synthetic.
- Use bounce/hit state transitions more explicitly.
- Avoid computing final speed from carry/interp as if it were observed.
- Improve real-unit court-plane velocity once court calibration is stable.

Acceptance:

- Fewer impossible carry segments.
- Cleaner bounce/contact transitions.
- Speed estimates become less noisy and less dependent on synthetic points.

### 8. Selector Simplification

Goal:

Reduce tangled logic in `ball_in_play_selector/core.py`.

Required work:

- Split selector into clear stages:
  - candidate generation.
  - association.
  - state estimation.
  - track scoring.
  - final source selection.
  - interpolation/render output.
- Keep behavior equivalent first, then improve one stage at a time.

Acceptance:

- Easier debugging.
- Fewer hidden interactions between guide, carry, motion, and detection logic.
- Easier future testing.

### 9. Performance Cleanup

Goal:

Improve runtime without sacrificing tracking quality.

Required work:

- Fix `ffmpeg` discovery so NVENC is used.
- Later, tune player/court auxiliary detection cadence, since it was the largest timing block in the baseline.
- Do not change player cadence in the near term; current priority is fresh player movement.
- Benchmark ball engine sizes:
  - 1280 current.
  - 960.
  - 896.
- Reduce CPU/GPU sync in motion preprocessing.
- Avoid `--skip-frame-yolo` for now.

Important:

Players are low priority for quality right now. Use player boxes only as context until ball tracking is stable.

Acceptance:

- Higher effective FPS.
- No reduction in fill rate, recall, or pixel accuracy.

### 10. Detector Upgrade Path

Goal:

Decide later whether YOLO should be supplemented or replaced.

Near-term decision:

- Keep YOLO/TensorRT.
- Do not switch detector before fixing tracking/motion connection.

Future options:

- Add TrackNet-style heatmap ball detector.
- Add motion-attention model inspired by TrackNetV4.
- Use YOLO and TrackNet together:
  - YOLO for object candidates.
  - TrackNet/heatmap for tiny blurred frames.
  - Motion/physics for continuity.

Research notes:

- TrackNet was designed for tiny, blurry, high-speed sports balls.
- TrackNetV4 explicitly adds motion attention, which matches this repo's orange-motion problem.
- A hybrid detector is safer than a full switch.

Acceptance:

- Only replace or supplement YOLO after current tracking diagnostics prove detector misses are the dominant quality problem.

### 11. 3D Ecosystem

Goal:

Eventually produce accurate hit speeds, bounce locations, player locations, and 3D ball flight.

Reality:

Single-camera 2D cannot reliably recover true 3D ball height and speed without strong assumptions. True 3D requires synchronized cameras and calibration.

Required future components:

- Camera calibration.
- Court calibration.
- Multi-camera synchronization.
- Cross-camera ball association.
- 3D triangulation.
- 3D Kalman/physics filter.
- Shot/bounce/hit event model.
- Player foot-position tracking.
- Metric/report export.

Data entities:

- `CameraCalibration`
- `CourtCalibration`
- `FrameDetection`
- `MotionCandidate`
- `Track2D`
- `Track3D`
- `BallState`
- `PlayerTrack`
- `HitEvent`
- `BounceEvent`
- `MetricReport`

## Immediate Next Task

Fix orange motion-to-track connection without skipping YOLO frames.

Status:

- Implemented in `--tracking-json` export.
- New JSON section: `motion_diagnostics`.
- Baseline diagnostic artifact: `output_videos/pomona_motiondiag_tracking.json`.
- First behavior-changing fix: loosen selector-side tiny motion blob handling so preserved ball-sized motion is not rejected before physics can score it.

Concrete first milestone:

- Keep YOLO on every frame.
- Keep player movement freshness unchanged.
- Accept tiny connected motion components when they are close to the predicted/guide path.
- Keep physics, player-body, and search-radius gates active to avoid grabbing unrelated motion.
- Run the same Pomona clip and compare source mix, filled frames, and large jumps.

Expected result:

Orange should take over more often during YOLO gaps where a real tiny motion candidate exists, while `carry` remains a fallback instead of the dominant gap filler.

## References

- TrackNet: `https://arxiv.org/abs/1907.03698`
- TrackNetV4: `https://arxiv.org/abs/2409.14543`
- ByteTrack: `https://arxiv.org/abs/2110.06864`
- OpenCV optical flow: `https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html`
- OpenCV Kalman filter: `https://docs.opencv.org/3.4/dd/d6a/classcv_1_1KalmanFilter.html`
- NVIDIA VPI / Optical Flow: `https://docs.nvidia.com/vpi/html/index.html`
