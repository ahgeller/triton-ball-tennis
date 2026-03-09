# Tennis Pipeline Performance Plan (Baseline: ~30 FPS)

## Scope
- Input: `1920x1080`, ~30 FPS source
- Current measured pipeline: `~30 FPS` end-to-end (`pass1` bottleneck)
- Hardware context: RTX 5080 class laptop GPU

## Baseline Breakdown (Current)
- `pass1 ball detect`: ~15.3 ms/frame (largest cost)
- `pass1 preprocess`: ~10.5-11.5 ms/frame
- `pass1 aux detect` (player/court): ~1.0-1.3 ms/frame
- `pass1 decode/slide`: ~1.1-1.3 ms/frame
- `selector + pass2`: small compared to pass1

---

## Level 0: Lock Current Fast Baseline
### Goal
- Keep stable quality and avoid regressions while optimizing.

### Actions
- Keep non-strict FP16 engine setup (faster than strict FP16 IO in your tests).
- Keep current ROI and selector behavior.
- Keep timing enabled for every benchmark run (`--info`).

### Expected FPS
- **30-33 FPS** (current band)

---

## Level 1: Low-Risk Runtime Tuning (No model retrain)
### Goal
- Improve throughput without changing detection behavior.

### Actions
- Keep player cadence tuned (`5/10` or `8/15`) based on quality/perf tradeoff.
- Keep protect-mask caching (already done).
- Keep cross-frame async structure only if it benchmarks neutral/positive.
- Run A/B benchmarks over 5 runs and use median.

### Expected FPS
- **31-35 FPS**

---

## Level 2: Ball Inference Cost Reduction (Biggest ROI)
### Goal
- Reduce the ~15 ms/frame ball detect cost.

### Actions
- Rebuild ball engine on target machine/runtime only (already in place via builder).
- Try lower ball input size variants and compare quality:
  - `1280x1280` (current)
  - `960x960`
  - `896x896`
  - optional aspect-ratio-preserving engine (if retrained/exported accordingly)
- Keep same pipeline; only swap ball engine.

### Expected FPS
- Conservative: **34-40 FPS**
- Aggressive (if smaller input still acceptable): **40-50 FPS**

---

## Level 3: Pipeline Structure Optimization (Python still)
### Goal
- Reduce CPU/GPU sync points and duplicate work.

### Actions
- Keep ROI-only motion transfer/filtering (already done).
- Keep side/protect mask single-build path (already done).
- Validate async overlap placement and retain only if net-positive.
- Optional: skip-frame YOLO (`skip_frame_yolo=2`) only if quality allows.

### Expected FPS
- Without skip-frame YOLO: **35-45 FPS**
- With skip-frame YOLO=2 (quality-dependent): **45-60 FPS**

---

## Level 4: Partial DeepStream Migration
### Goal
- Move decode + infer path to DeepStream while keeping custom logic initially in probe code.

### Actions
- DeepStream handles:
  - decode
  - preprocess (basic)
  - TensorRT inference (ball/player/court)
- Keep custom motion/selector logic in Python or C++ probe first.

### Expected FPS
- **45-70 FPS** (depends on how much custom logic stays outside optimized plugins)

---

## Level 5: Full GPU-Centric DeepStream/C++ Pipeline
### Goal
- Maximize throughput and reduce Python overhead.

### Target Architecture
- `filesrc -> demux -> nvv4l2decoder -> nvstreammux -> nvinfer(ball) -> nvinfer(player) -> nvinfer(court) -> custom preprocess/selector plugin -> nvdsosd(optional) -> encoder`
- Keep frames as NVMM/GPU memory throughout.
- Store all per-frame state in DeepStream metadata; avoid CPU copies except final output/reporting.

### Implementation Plan
1. Define interfaces and metadata schema.
- Create C++ structs for:
  - ball detections
  - player boxes
  - court keypoints
  - motion mask stats
  - selector track state
- Define one metadata contract used by all plugins.

2. Build a custom GPU preprocess plugin (`GstBaseTransform`).
- Port HSV motion logic, ROI gating, and flicker suppression.
- Use CUDA kernels for mask ops and morphology.
- Output compact motion metadata rather than full-frame CPU masks.

3. Build a selector/state plugin in C++.
- Port track scoring and ball-in-play selection from `ball_in_play_selector.py`.
- Keep per-stream state machine in plugin context.
- Emit chosen ball point/bbox as metadata for renderer/encoder.

4. Integrate TensorRT engines directly in DeepStream config.
- Ball as primary infer.
- Player/court as secondary infer with configured intervals.
- Keep non-strict FP16 IO where it is faster in your benchmarks.

5. End-to-end GPU rendering/output.
- Draw overlays from metadata in GPU path.
- Encode with NVENC without host-frame round-trips.

6. Add profiling and guardrails.
- Add Nsight Systems trace points for each plugin stage.
- Export per-stage latency counters in logs.
- Add runtime toggles for:
  - motion preprocess on/off
  - selector on/off
  - ROI on/off
  - debug overlays on/off

### Delivery Phases
1. Phase A (Inference-only DS skeleton).
- Deliver decode + ball infer + encode pipeline.
- Match current ball detections on validation clips.
- Exit criteria: stable `45+ FPS`.

2. Phase B (GPU preprocess plugin).
- Add motion preprocess plugin and metadata output.
- Validate motion masks vs Python baseline.
- Exit criteria: no quality regression, `55+ FPS`.

3. Phase C (C++ selector plugin).
- Add full selector logic in C++.
- Validate chosen-track consistency vs Python baseline.
- Exit criteria: stable selection, `60+ FPS`.

4. Phase D (Optimization pass).
- Kernel tuning, stream overlap, memory pool tuning.
- Remove remaining host sync points.
- Exit criteria: target production FPS for hardware tier.

### Validation/Acceptance
- Same fixed clip suite used for all levels.
- Required metrics per build:
  - `filled/N`
  - track-switch count
  - false sideline selections
  - per-stage latency
  - end-to-end FPS
- Acceptance rule:
  - quality metrics not worse than current Python baseline
  - throughput meets phase FPS target

### Reality Checks Before Commit
1. Phase C scope warning (selector C++ port).
- The selector is large and behavior-heavy.
- Faithful C++ parity is likely a multi-week effort, not a quick task.
- Treat this as the highest schedule risk in Level 5.

2. Golden validation must start at Phase A.
- Do not wait until later phases.
- Use at least 5-6 diverse clips:
  - fast rallies
  - serves
  - net play
  - adjacent-court interference
  - camera shake/occlusion cases

3. Phase A may be sufficient.
- Do not assume B-D are required.
- If Phase A already meets throughput+quality goals, stop there.

4. Streaming constraint for Phase B motion logic.
- Current Python motion path uses 3-frame temporal logic (`prev/curr/next`).
- In streaming DeepStream you must choose upfront:
  - 2-frame motion (`prev/curr`), or
  - 1-frame buffered delay to preserve 3-frame logic.
- Decide this before plugin implementation.

5. Immediate pre-DeepStream check.
- Keep ball inference on every frame (`skip_frame_yolo=1`) by requirement.
- Do not use frame-skipping as a throughput strategy.

### Risks and Mitigations
- Risk: selector behavior drift during port.
- Mitigation: golden-output tests from Python reference runs.
- Risk: hidden sync points in plugins.
- Mitigation: Nsight profiling and explicit async stream policy.
- Risk: integration complexity.
- Mitigation: ship in phases A->D with rollback checkpoints.

### Expected FPS
- Phase A: **45-60 FPS**
- Phase B: **55-75 FPS**
- Phase C: **60-90 FPS**
- Phase D tuned: **70-110+ FPS** (hardware and model dependent)

### Effort Estimate by C++ Experience
- Strong C++/GStreamer/CUDA: ~2-4 weeks for A-C to stable parity.
- Moderate C++: ~4-8 weeks.
- Limited C++: ~2+ months unless scope is reduced to Phase A only.

---

## Quality Gates Per Level
- Ball trajectory continuity (no extra random switches)
- Ball-in-play selection stability
- Court-guided constraints preserved
- Filled-frame ratio not degraded
- Ball inference runs every frame (no skip-frame mode)

Run the same validation clip set and compare:
- `filled/N`
- track switch count
- false sideline picks
- total runtime and `pass1` stage timings

---

## Recommended Next Step
1. Stay on current non-strict FP16 engines.
2. Execute Level 2 (engine input-size sweep for ball model) with fixed quality checks.
3. Pick best speed/quality point, then decide whether DeepStream migration is worth complexity.
