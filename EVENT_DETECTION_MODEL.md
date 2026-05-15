# Event Detection Model — Architecture & Training Plan

Per-frame classification of **hits** (racket–ball contact) and **bounces** (ball–court contact) from an already-tracked tennis ball trajectory.

Input is what we already produce: tracking.json with per-frame ball `(u, v)`, per-frame court keypoints (→ homography), per-frame player bounding boxes. Output is a per-frame probability of each event class, post-processed into discrete events with `{frame, type, court_xy_m}`.

This document is the design contract for the model. Code lives in `tools/` (`event_features.py`, `event_dataset.py`, `event_model.py`, `train_event_model.py`, `infer_events.py`). Labels come from `tools/label_events.py`.

---

## 1. Why this approach (vs. the alternatives we ruled out)

| Option | Verdict |
|---|---|
| Pure heuristics on smoothed velocity | What we have. Plateaued at ~12 contacts on the Pomona clip; misses most bounces. |
| Image-based event spotting (TTNet-style 3D CNN on raw video) | Overkill for our setup — we already have an accurate ball tracker, so we don't need the network to learn detection from pixels. Throws away the structure we worked to extract. |
| TrackNetV3-style heatmap | TrackNet is a *tracker*, not an event head. Adding it as a secondary detector buys recall on lost frames, not event labels. Orthogonal. |
| Small temporal model on trajectory features (this doc) | Lowest cost, fastest iteration, matches what MonoTrack / table-tennis stroke detection / badminton hit detection literature converges on. ✓ |

All five recent papers I surveyed (MonoTrack, Liu et al. 2023 badminton, Kulkarni & Shenoy 2023 table tennis, Sun et al. 2024, Ganguli's tennis bounce LSTM) use the same pattern: **small temporal net (GRU / LSTM / TCN) on top of an already-tracked trajectory, 12–20 frame window, features = ball xy + (optional) pose / court, plus aggressive post-processing**. We follow that recipe.

---

## 2. Architecture — bidirectional dilated 1D TCN

**Bidirectional Temporal Convolutional Network** (Bai/Kolter/Koltun 2018, arXiv 1803.01271) is the right primitive:

- Convolutional, so it parallelizes — training is fast and inference is faster than RNNs.
- Dilated stacked kernels give a controllable, multi-scale receptive field. Hits and bounces have ~5–10 frame signatures; we want a receptive field 2–3× that.
- Acausal (we have the whole clip at inference) — symmetric padding gives the model symmetric context around each candidate frame.

### Block diagram

```
Input  (B, T=48, F=12)        per-frame feature vectors
   │
   ├─ LayerNorm over F
   │
   ▼
[Residual TCN block 1] dilation=1, kernel=3, channels=64
[Residual TCN block 2] dilation=2
[Residual TCN block 3] dilation=4
[Residual TCN block 4] dilation=8
[Residual TCN block 5] dilation=16             ← effective RF ≈ 63 frames
   │
   ▼
1×1 Conv → out_channels=2 (hit, bounce)
   │
   ▼
Sigmoid → (B, T, 2)           per-frame independent probabilities
```

Each residual block:

```
x ──► Conv1d(k=3, dilation=d, padding=same) ──► LayerNorm ──► GELU ──► Dropout(0.1)
   ──► Conv1d(k=3, dilation=d, padding=same) ──► LayerNorm ──► GELU ──► Dropout(0.1)
   ──► + (residual from input, 1×1 Conv if channel mismatch)
```

Parameter budget: ≈ 70 k. Trains in minutes on the user's RTX-class GPU even with heavy augmentation. Inference on a 4 000-frame clip is sub-second.

**Multi-label, not multi-class.** Hits and bounces can occur on adjacent frames (e.g., bounce followed by a half-volley return). Two independent sigmoid outputs avoid forcing the model to pick one.

---

## 3. Features per frame (F = 12)

All features are deterministic functions of tracking.json + the court homography we already compute. Anything missing (no detection, no homography) gets zero and a separate validity-mask channel — never silently filled with NaN.

| # | Feature | Computation | Normalization |
|---|---|---|---|
| 1 | ball u | `frames[i].x` | `/ image_width` |
| 2 | ball v | `frames[i].y` | `/ image_height` |
| 3 | ball du/dt | central finite diff over 1 frame | clip to ±50 px/frame, /50 |
| 4 | ball dv/dt | central finite diff | same |
| 5 | ball d²u/dt² | central second diff | clip ±10, /10 |
| 6 | ball d²v/dt² | same | same |
| 7 | ball court-X | homography(u, v) | `/ COURT_WIDTH_M` |
| 8 | ball court-Y | homography(u, v) | `/ COURT_LENGTH_M` |
| 9 | dist ball → nearest player (pixel) | min over players of pixel dist to bbox foot | `/ image_diagonal` |
|10 | dist ball → net line (pixel) | net is the projected `y_court = L/2` line | `/ image_diagonal` |
|11 | ball valid | 1 if raw detection present this frame, else 0 | — |
|12 | court valid | 1 if homography exists this frame, else 0 | — |

Rationale for each:
- **(1, 2)** absolute position — lets the model learn "near the net" / "near the baseline" priors.
- **(3, 4)** velocity — the primary signal: hits / bounces are velocity-discontinuity events.
- **(5, 6)** acceleration — distinguishes a free-flight curve (smooth a) from an impact (a spike).
- **(7, 8)** court-plane position — gives the model net / service-box awareness even when pixel u, v don't.
- **(9)** distance to the closest player — hits happen with the racket; bounces don't care about players. This feature lets the model separate the two.
- **(10)** distance to the net — bounces near the net are physically impossible; the model can learn the constraint.
- **(11, 12)** validity flags — so the model knows when zero-padding is "no data" vs. "ball at origin."

**Not included (yet):**
- Player pose keypoints. MonoTrack uses these. They'd help disambiguate near-player hits but require pose estimation as an extra dependency. Add later if recall on hits is poor.
- Audio. Tennis impacts are very audible — `librosa` + a tiny 1D CNN on the audio waveform could catch hits with high precision. Out of scope for v1.

---

## 4. Window + target encoding

**Window size: T = 48 frames** (≈ 0.8 s at 60 fps), centered on each candidate frame. Stride = 1 at training (every frame is a candidate, with the window slid around it).

**Gaussian-kernel soft targets.** For each labeled event at frame `f_e`, the target signal is

```
y_c(t) = max over labeled events e of class c of  exp(-(t - f_e)² / (2 σ²))
```

with **σ = 1.5 frames**. This:
- Tolerates the inevitable ±1 frame label noise from the human labeler.
- Provides dense gradient near the event instead of a single spiky positive.
- Pairs naturally with temporal NMS at inference (we look for the local max in the predicted heatmap).

This is the same scheme as TTNet's event spotter (arXiv 2004.09927) and CenterNet's keypoint heads (arXiv 1904.07850), adapted to 1D.

---

## 5. Loss

Per-frame **binary cross-entropy on the Gaussian-soft target**, summed over the two output channels:

```
L = Σ_c Σ_t  BCE( ŷ_c(t),  y_c(t) )
```

Class imbalance is mild here (~1 positive frame in 60 → after σ=1.5 smoothing, ~1 positive-mass frame in ~20) so plain BCE with the Gaussian target works. If recall lags after a first training run, drop in **focal BCE** (γ = 2, α = 0.25; Lin et al. 2017, arXiv 1708.02002) as a swap-in.

Decision rule for v1: ship with plain BCE on Gaussian targets, log focal-vs-plain as a v2 ablation only if needed.

---

## 6. Training data pipeline

```
tracking.json + court keypoints  ──►  event_features.py  ──►  (N_frames, 12) float32 array per clip
                                                                                    │
        validation/labels/<clip>_events.json  ──► event_dataset.py ──► (window, target) pairs
                                                                                    │
                                                                                    ▼
                                                                       train_event_model.py
                                                                          - bidirectional TCN
                                                                          - BCE on Gaussian target
                                                                          - Adam(1e-3), cosine decay
                                                                          - 50 epochs, batch 64
                                                                                    │
                                                                                    ▼
                                                                     models/event_tcn.pt
```

### Augmentations (applied online during training)

- **Time jitter**: shift the entire window by a uniform integer in [-2, +2] frames. Re-aligns the Gaussian target; teaches the model that the event center isn't exactly on the labeler's clicked frame.
- **Pixel noise**: add N(0, 2 px) to the raw ball u, v (and recompute downstream finite-diff features).
- **Velocity scaling**: multiply the whole window's u, v by U(0.95, 1.05) around the window center — simulates serve speed variation.
- **Horizontal flip**: mirror u → image_width − u, court_X → COURT_WIDTH_M − court_X. Doubles effective data size since tennis is left-right symmetric.
- **Dropout-by-frame**: randomly mask one ball detection per window (set validity flag to 0, zero positions) — teaches robustness to detector gaps.

### Splits

**Time-based, not random.** Events within the same rally are correlated; randomized splits leak labels through the rally boundary. Use the first 70 % of the clip for train, next 15 % for val, last 15 % for test. Across multiple clips, hold out one entire clip for test.

### Hyperparameters (starting point — tune from val)

| | Value |
|---|---|
| Optimizer | Adam, lr=1e-3, weight decay=1e-5 |
| Schedule | Cosine to 1e-5 over 50 epochs |
| Batch size | 64 windows |
| Window size T | 48 |
| Feature dim F | 12 |
| Hidden channels | 64 |
| Residual blocks | 5 (dilations 1, 2, 4, 8, 16) |
| Dropout | 0.1 |
| Gaussian σ | 1.5 frames |
| Early stopping | val event-F1 plateau for 10 epochs |

---

## 7. Inference + post-processing

1. **Compute features** for the whole clip from `tracking.json`.
2. **Window-stride-1 inference**: pad the feature array with zeros at both ends so every frame gets predicted exactly once. The model outputs `(T_total, 2)` probabilities. Acausal padding (zero before and after) is correct because we have the whole clip.
3. **Temporal NMS per class**: in each 1D heatmap channel, find local maxima with a 5-frame radius (`scipy.signal.find_peaks` with `distance=5`). Threshold at probability ≥ 0.5.
4. **Emit events**: each surviving peak becomes `{frame, type, prob}`. The ball's pixel position at that frame is the event's `ball_uv_pix`; project it via the per-frame homography to get `court_xy_m`.
5. **Optional rally-level consistency** (post-MVP): enforce alternating hits between players (MonoTrack does this with DP — bumps their accuracy 3 pp). Skip in v1.

Result is a JSON identical in schema to the labeler's output, ready for downstream consumption (mini-court rendering, statistics, etc.).

---

## 8. Evaluation

Two views:

### Per-event metrics (primary)
- **Precision / recall / F1**, with a predicted event counting as a match if it falls within **±3 frames (50 ms)** of a ground-truth event of the same class.
- Compute separately for hits and bounces.

### Per-frame metrics (sanity)
- Per-frame AUROC on the Gaussian-target regression. Useful for spotting training failures; doesn't reflect real-use behavior since users only care about discrete events.

### Localization quality
- **Median** and **P90 frame offset** between predicted event and matched GT. Should be ≤ 1 frame (the labeler's own noise floor).

### Targets
After labeling ~200–300 events across the Pomona clip + one or two more:

| Metric | Target v1 | Stretch |
|---|---|---|
| Hit F1 | ≥ 0.85 | ≥ 0.92 |
| Bounce F1 | ≥ 0.80 | ≥ 0.90 |
| P90 frame offset | ≤ 2 frames | ≤ 1 frame |

If either F1 < 0.80 on the test clip, the path forward is **add pose features** before adding more capacity to the net — every survey paper that beats us has pose-derived features.

---

## 9. Implementation plan (files in `tools/`)

| File | Lines (est.) | Purpose |
|---|---|---|
| `event_features.py` | ~150 | Pure feature extractor: tracking.json + court keypoints → (N, 12) float array. No torch dep. |
| `event_dataset.py` | ~120 | torch Dataset; takes a feature array + a labels JSON, yields (window, target) tensors with augmentation. |
| `event_model.py` | ~80 | TCN definition. ~70 lines of model code. |
| `train_event_model.py` | ~200 | Training loop: load multiple clips, split time-wise, train, save best-by-val checkpoint, log to a JSONL. |
| `infer_events.py` | ~120 | Load checkpoint + tracking.json, output an events JSON with the same schema as the labeler. |
| `eval_events.py` | ~100 | Compare predicted events to GT labels (per-event P/R/F1, frame offsets). |

Total: ~770 LoC. Fits in a long afternoon once labels exist.

**Dependencies:** PyTorch (already in the env), `scipy.signal` (already there). No new install required.

---

## 10. References

- **MonoTrack** — Liu et al., CVPR-W 2022. Single-camera badminton tracking with HitNet (GRU on 12-frame windows of trajectory + pose + court). https://arxiv.org/abs/2204.01899
- **TrackNetV3** — Chen et al., ACM MMAsia 2023. Trajectory rectification head; no event prediction. https://github.com/qaz812345/TrackNetV3
- **WASB-SBDT** — NTT, BMVC 2023. Detector + tracker only, no event head. https://arxiv.org/abs/2311.05237
- **TCN** — Bai, Kolter, Koltun 2018. The reference for dilated 1D convolutional sequence modeling. https://arxiv.org/abs/1803.01271
- **TTNet** — Voeikov et al. 2020. Multi-task image-based event spotter for table tennis (Gaussian-target event heatmap is the relevant precedent). https://arxiv.org/abs/2004.09927
- **CenterNet** — Zhou et al. 2019. Source of the Gaussian-kernel keypoint heatmap idea we're adapting to 1D. https://arxiv.org/abs/1904.07850
- **Focal Loss** — Lin et al. ICCV 2017. Fallback loss if event class imbalance bites. https://arxiv.org/abs/1708.02002
- **Liu et al. 2023, badminton hit detection** — TrackNet + YOLOv7 + rule-based refinement. 58.8 → 89.7 %. https://arxiv.org/abs/2307.16000
- **Kulkarni & Shenoy 2023, table-tennis stroke detection** — TCN on ball trajectory. 87 % on unseen strokes. https://arxiv.org/abs/2302.09657
- **Sun et al. 2024, badminton shot refinement** — Direction-flip + pose-derived swing fusion. https://pmc.ncbi.nlm.nih.gov/articles/PMC11244353/

---

## 11. What to do next (concrete)

1. Run `tools/label_events.py` on the Pomona clip. Aim for ~200 labeled events (rough mix: 100 hits, 80 bounces, 20 off-frame).
2. Write `tools/event_features.py`. Easiest to verify: dump features for a known rally and eyeball the velocity/acceleration columns at hits.
3. Write `tools/event_model.py` + `tools/event_dataset.py`. Run a 10-window overfit test (one batch, 200 steps) — train loss should go to ~0.
4. Write `tools/train_event_model.py`. Train on Pomona; report val F1 per class.
5. Write `tools/infer_events.py` + `tools/eval_events.py`. If F1 ≥ 0.80, ship and label the next clip.
6. If F1 < 0.80, add player-pose features (RTMPose or YOLOv8-pose; one extra feature vector per player per frame).
