# A ball detector built for this camera (proposal, 2026-08-29, research pass 2)

Everything below is driven by numbers measured on the hand-labelled archive
clips, not by papers. Baselines to beat (raw detector, `evaluate_archive.py --mode raw`):

| model | raw recall | wrong object | held-out video10 | prepass speed |
|---|---|---|---|---|
| GridTrackNet fine-tune (`models/gridtracknet_weights_torch.npz`, sha a128491d) | 96.5% | 0.6% | 88.4% / 2.8% | 90 fps at 768×432 |
| GridTrackNet original (pre-fine-tune, `_legacy.npz`) | see `output/confirm_best.log` | | | same |
| TOTNet tennis (official) | 75.3% | 7.6% | 82.3% / 6.5% | 10 fps |

Failure modes that remain after the selector fixes (from `evaluate_archive.py`):
far-court ball not detected (4–6 px at 1080p, 2–3 px after the 768×432 resize),
and confident detections on players' feet, court lines, logos and a parked van.
Hard constraints: fixed elevated camera behind the baseline, 1080p60, whole run
close to real time on an RTX 5080 laptop.

## What the fixed camera gives us that broadcast-video models throw away

1. **The court never moves.** A median background image of the clip is a near-perfect
   "everything that is not the ball" reference. Lines, logos, vans, nets and
   fences live in it; the ball never does. Classical detectors (DeepBall's
   candidate stage, the pre-TrackNet literature) used exactly this; the modern
   heat-map nets (TrackNet V2–V5, WASB, TOTNet, BlurBall) all dropped it because
   they target broadcast footage with moving cameras. BlurBall's own dataset is
   entirely static cameras and still does not use the background.
2. **The court region is known** (court detector + homography already in the
   pipeline). The ball's apparent size is a known function of image row.
3. **The camera is the same every session.** A model trained on this view does not
   need to generalise to broadcast zooms, so capacity can go to resolution.

## The design — "CourtTrackNet"

Keep GridTrackNet's output design (it is the reason it is fast and precise) and
change what it looks at.

### Input: two streams, not a crop (answer to "you'd lose the toss / balls off the side")

A hard crop to the court *would* lose serve tosses (they go above the far
fence), lobs, and balls that leave the sidelines. So the full frame is never
dropped:

- **Stream A — full frame at 768×432** (exactly today's input). Covers the
  toss, lobs, wide balls, the near court where the ball is already 8–12 px.
- **Stream B — far-court tile at native pixel density**: the region from the
  net line up to the far fence, full frame width, cut from the 1080p frame and
  resized to a fixed 1280×448 (in this view that is roughly a 1.2× *upscale* of
  the far court rather than the 2.5× downscale it gets today). The far-court
  ball becomes ~5 px instead of ~2 px.
- Both streams run the same network weights (fully convolutional, so the tile
  is just a different input size) in one batched pass; detections from B win
  inside the tile (they are more precise), A everywhere else. The selector
  sees one detection list per frame as today.
- Cost: stream B adds ~1.7× the pixels of stream A. With the TensorRT export
  below this stays well inside real time; without it, drop B to 960×336.

### Channels (per 5-frame window at 30 FPS spacing — measured better than 60 FPS consecutive: 97.0% vs 95.1% recall)

5 × RGB (15) + 4 **signed frame-difference** channels (brighten/darken of
frame *t* vs *t−1* and *t* vs *t+1* — TrackNetV5's motion-polarity idea,
essentially free) + 1 **background-difference** channel (|frame − running
median background|, background rebuilt every ~400 frames on the same cadence
as the court detector, re-aligned through the court homography so a windy
tripod does not poison it) + 1 **court-mask** channel. 21 channels in.

Static objects then look identical to the network in RGB *and* zero in the
background-difference channel; the ball is the only small bright blob that is
present in the difference channels and absent from the background.

### Backbone

GridTrackNet's VGG-style 13-conv trunk is the compute problem at higher input
size. Replace with a small encoder–decoder using depthwise-separable blocks
(MobileNetV3-style, ~3–5 M params, 3 down / 3 up with skips), FP16, exported
to **TensorRT**. Target: both streams together ≥ 100 fps of 5-frame windows on
the 5080 — cheaper per frame than today's PyTorch GridTrackNet. WASB/BlurBall
reach 79–95 fps on a 2080 Ti with a far heavier HRNet at 512×288, so this is
not ambitious.

### Head (GridTrackNet style, plus absence and blur)

- Grid over the input (16 px cells): per frame, per cell: ball-present logit +
  (dx, dy) sub-cell offset — exactly as now, so the tracker's decoder and the
  selector are reused unchanged after mapping tile → frame coordinates.
- **Per-frame visibility logit** (global-pooled head): trained on the "ball not
  visible" labels the click tool produces. The selector gets a calibrated
  "no ball in this frame" score instead of thresholding a heat-map peak — this
  is what kills the 0.59-confidence van.
- **Blur head (optional, from BlurBall)**: per positive cell predict streak
  half-length and orientation. BlurBall shows a fast ball is a streak in 62% of
  frames, that labelling the *centre* of the streak instead of its leading edge
  improves every model they tested, and that the blur vector is a velocity prior
  that cut their trajectory-fit error from 84 px to 53 px. For us that prior
  feeds the selector's association/Kalman step directly.

### Loss and data

Focal BCE on the grid (as now) + L1 on offsets in positive cells + BCE on
visibility (+ L1 on blur if used), with **hard-negative mining**: every frame
`evaluate_archive.py` flags as `wrong` or `false alarm` is up-weighted ×4 in the
next round. Position-aware sampling as in WASB: oversample windows where the
ball is small or blurred.

Labelling rule change (BlurBall): when the ball is a streak, click its
**centre**, not its front. `label_tool.py` already lets you do this; it is a
convention, not code.

External data that matches this view better than TrackNetV2's broadcast clips:
the IEEE DataPort multi-view tennis set (court-side camera + drone),
PadelTracker100 (single fixed camera, ~100k frames, padel ball is similar size).
Use them for pre-training only; your own clips decide.

### Post-processing

Unchanged selector (v3 rules), fed grid detections + visibility score (+ blur
velocity prior). Court homography → ground-plane bounce points come for free; a
3D ballistic lift per rally segment ("Where Is The Ball", TT3D) is an optional
later stage, not a prerequisite.

## Why not just train TrackNetV5 / TOTNet / WASB / BlurBall on the archive

- All resize the whole frame to ~512×288; the far-court ball is 1–2 px there.
  No head design fixes that — only resolution where the ball is small does.
- TOTNet outputs one frame per 5-frame pass (5× the compute per output frame);
  multi-frame-output heads (GridTrackNet, TrackNetV2/V5) are what make real
  time possible.
- None use the static-camera background, the cleanest signal against exactly
  the false positives seen here.
- Their heavy HRNet backbones buy robustness to camera motion you do not have.

## Build order (each step measurable with `evaluate_archive.py`)

0. Export the current GridTrackNet to TensorRT FP16 (speed headroom, no accuracy change).
1. **Background-difference + court-mask input channels on the existing
   GridTrackNet** (new input-channel weights zero-initialised so the pretrained
   net is untouched at step 0), plus the visibility head. Fine-tune on the
   archive + new clips. Expected: false alarms and wrong-object drop first.
2. Add stream B (far-court tile) with the same weights; merge detections.
   Expected: far-court recall.
3. Backbone swap only if step 2 is too slow after TensorRT.
4. Blur head + centre-of-streak labelling once there are enough fast-ball labels.
5. Keep labelling with `finetune/` between every step; retrain; keep-best.

Rough effort: step 1 a few days (data plumbing + a 21-channel first conv);
step 2 a day on top; step 3 a week including TensorRT export and timing.

## Sources consulted

- WASB: Tarashima et al., BMVC 2023 — https://arxiv.org/abs/2311.05237, https://github.com/nttcom/WASB-SBDT
- BlurBall: Gossard et al., CVPRW 2026 — https://arxiv.org/abs/2509.18387, https://github.com/cogsys-tuebingen/blurball
- TrackNetV4 — https://arxiv.org/abs/2409.14543 ; TrackNetV5 — https://arxiv.org/abs/2512.02789
- TOTNet — https://rbouadjenek.github.io/assets/pdf/YCVIU_104657.pdf
- DeepBall — https://arxiv.org/abs/1902.07304
- Where Is The Ball (CVPRW 2025) — https://openaccess.thecvf.com/content/CVPR2025W/CVSPORTS/papers/Ponglertnapakorn_Where_Is_The_Ball_3D_Ball_Trajectory_Estimation_From_2D_CVPRW_2025_paper.pdf
- TT3D — https://arxiv.org/abs/2504.10035
- Multi-view tennis dataset (drone + court camera) — https://ieee-dataport.org/documents/multi-view-tennis-ball-dataset-trajectory-estimation-drone-and-court-cameras-annotated
- PadelTracker100 — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12926558/
- YOLO-Net (RTMDet-light small tennis objects) — https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0335558
