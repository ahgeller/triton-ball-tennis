# Triton Tennis — Project Context

> **Purpose:** Drop this into a new AI conversation for full context.
> Last updated: 2026-03-08

---

## What This Project Is

A two-pass video processing pipeline that takes a raw tennis match recording and outputs:
- A clean tracking video with a ball marker drawn on every frame
- A motion-debug video (for development/tuning)
- A guide-debug video
- A YOLO-input-debug video

**Stack:**
- `now_main.py` — main pipeline entry point (~5 k lines)
- `ball_in_play_selector.py` — multi-hypothesis ball tracker (~5.4 k lines)
- YOLO ball detector running on TensorRT (`.engine` file at `models/ball.engine`)
- Court keypoint model (same TRT engine, 16 keypoints)
- Python env: `tennis-analysis` conda env

**Run command:**
```
python now_main.py -i "input_videos/VIDEO.mp4" [--debug-video] [--guide-video]
```

---

## Architecture Overview

### Pass 1 — Detection & Motion Collection
For each frame:
1. **Motion preprocessing** — weighted temporal averaging (WTA) background model, HSV channel differencing, morphological filtering → produces `raw_motion_u8` and `boost_mask_u8`
2. **YOLO detection** — TensorRT ball detector, skip-frame optimization (every Nth frame)
3. **Court keypoint detection** — 16-point court model gives corners/lines
4. **ROI motion tracker** — `ROIMotionTracker` tracks motion blobs, retroactively deletes "ghost" blobs that never reattach to the ball (ghost pruning)
5. Stores per-frame: `all_frame_dets`, `all_boost_masks`, `all_raw_motions`, `all_rois`, `all_ghost_rois`, `all_court_kps`, `all_player_boxes`

### Selector — `select_ball_in_play()` in `ball_in_play_selector.py`
Given all detections from Pass 1:
1. **Build tracks** — greedy or Ultralytics-backend tracker makes raw track hypotheses
2. **Merge tracks** — stitch fragments from the same ball across short gaps/bounces
3. **Score tracks** — multi-factor scoring: court-inside %, motion overlap %, near-player %, velocity consistency, etc.
4. **Timeline selection** — dynamic programming chain across the video to pick the best continuous track sequence
5. **Gap patching** — for frames where the ball isn't detected:
   - Kalman Filter prediction (`carry` source)
   - Motion blob search within KF search radius (`motion` source)
   - Guide circle tracking (`guide` source)
6. Returns `per_frame` results (one `FrameResult` per frame) + `chosen_track`

### Pass 2 — Rendering
Reads the video again, draws:
- Ball marker, trails, court overlays, player boxes
- Guide video: candidate tracks, ROI boxes, guide circles
- **Motion debug video** (`prof_test_motion_debug.mp4`): WTA-processed frame with overlays

---

## Key Classes & Functions

### `ball_in_play_selector.py`

| Symbol | What it does |
|---|---|
| `SelectorConfig` | All tunable params, auto-scaled from fps/resolution |
| `BallKalmanFilter` | FilterPy-based KF: state `[x, y, vx, vy]`, gravity + drag, depth-aware gravity via court homography |
| `Track` | Single ball track hypothesis; wraps `BallKalmanFilter` |
| `Detection` | Single YOLO hit with position, area, conf, frame index |
| `FrameResult` | Per-frame output: `cx/cy`, `source` (det/motion/carry/guide/interp), `bbox`, `search_cx/cy/radius` |
| `build_court_homography()` | Computes perspective transform from 4 court corners → normalized court plane |
| `court_px_per_meter()` | Given ball position + H matrix, returns local px/meter scale at that depth |
| `select_ball_in_play()` | Main entry point for the selector |
| `_stitch_track_chain()` | Merges a chain of tracks from timeline selection into one |
| `build_tracks()` / `build_tracks_ultra()` | Build raw track hypotheses from detections |
| `_find_motion_blob()` | Finds a motion blob in `raw_motion` near a predicted position |

### `now_main.py`

| Symbol | What it does |
|---|---|
| `BallTRTDetector` | TensorRT ball detector wrapper |
| `ROIMotionTracker` | Tracks motion ROI bounding boxes; retroactively deletes ghost blobs |
| `preprocess_frame()` | Returns `vis` debug image (WTA-processed frame with highlighted motion) |
| `filter_boost_mask()` | Filters raw motion mask by area/perspective/aspect ratio |
| `_commit_frame()` | Inner fn — appends frame data to all_* lists, runs ghost ROI pruning |
| `_build_court_polygon()` | Builds court convex hull from keypoints |
| `_build_ground_projection_model()` | Homography model for net line drawing |
| `guide_writer` | VideoWriter for `prof_test_guide_debug.mp4` |
| `dbg_writer` | VideoWriter for `prof_test_motion_debug.mp4` |

---

## What We Built / Changed (Chronological)

### 1. Weighted Temporal Averaging Motion Detection
- Replaced simple frame-diff with WTA background model (alpha-blend per channel)
- Zeroes alpha on moving pixels so foreground never bleeds into background
- HSV-channel differencing (V, S, H combined), 3-frame consensus

### 2. Ghost ROI Pruning
- `ROIMotionTracker` tracks motion blobs as short ROI tracks
- If a blob track never reattaches to a YOLO ball detection within a timeout, it's retroactively erased from `all_raw_motions` and `all_boost_masks` for all the frames it appeared in
- This kills fake motion from player bodies, crowd, etc. before the ball selector even sees it

### 3. FilterPy Kalman Filter Integration
- `BallKalmanFilter` replaces the old `_predict_projectile` + EMA custom physics
- State: `[x, y, vx, vy]`, control input: gravity on `vy`
- `Q` (process noise) and `P` (covariance) inflated on bounce detection (y-velocity sign flip or large Mahalanobis residual)
- Pink motion search radius = `sqrt(max(P[0,0], P[1,1])) * 3.0` — grows when uncertain, shrinks after detection
- `filterpy` package required (`pip install filterpy`)

### 4. Court Homography Depth-Aware Gravity
- `build_court_homography(court_keypoints)` uses 4 corners (TL/TR/BL/BR = keypoints 0,3,4,7)
- `court_px_per_meter()` computes local pixel scale at any ball position
- `BallKalmanFilter.predict()` scales gravity: `base_g * (local_px_per_m / ref_px_per_m)`
- Result: far-baseline balls get correct smaller pixel gravity, near-camera balls get larger
- Injected into all Track KFs after track merging in `select_ball_in_play`

### 5. Motion Blob Latching
- Pre-built `MotionTracks` from boost mask contours before selector runs
- Selector can "latch" onto these for `motion` source frames during gaps

### 6. SciPy B-spline Motion Trail Smoothing
- Motion trails in the debug video use `scipy.interpolate.splprep` + `gaussian_filter1d`
- Replaces old EMA-smoothed polyline rendering
- Do NOT modify this — user is happy with it

### 7. Motion Debug Visualizations (now_main.py pass2)
- Ball bounding box: colored by source (`det`=green, `motion`=blue, `carry`=orange, `guide`=yellow)
- Estimated ball box for non-det sources (uses `cached_anchor_radius`)
- KF physics search radius circle (magenta) with `r=N` label for all non-det frames
- **Green rectangles** = surviving `all_rois` motion detection boxes
- **Red "GHOST" rectangles** = `all_ghost_rois` boxes that were retroactively pruned
- `all_ghost_rois` list: filled retroactively when ghost pruning fires in `_commit_frame`

### 8. Player Detection (Partial)
- `boxmot` / BoT-SORT integration was attempted but unfinished
- Currently uses a simpler slot-assignment approach

---

## What Works Well ✅

- Full end-to-end pipeline runs in ~200–300s on a 3286-frame 1080p video (GPU)
- Ghost pruning significantly reduces false motion detections
- Kalman Filter gives smooth carry/interp during gaps
- Court homography injected — depth-aware gravity active
- Motion debug video shows: motion blobs, ghost ROIs, ball bbox, KF circle, all_rois boxes
- B-spline motion trails look good — don't touch

---

## What Doesn't Work / Known Issues ❌

- **Chosen track is id=1 with score=11** while better tracks exist (e.g. id=131 score=232) — timeline stitching issue; the final chosen track across the whole video is often suboptimal
- **`--draw-motion` flag doesn't exist** — user tried it, it's not a valid CLI arg
- **BoT-SORT player tracking** — half-implemented, not connected end-to-end
- **Ghost ROI red boxes** — newly added, not confirmed working yet by user
- **KF search radius for `interp`/`guide` frames** — wasn't populated before (now fixed for `carry`, but `guide` and `interp` still may be 0)
- **Ultralytics track builder** sometimes falls back to greedy; `Track.__init__` requires `cfg` arg — all known call sites patched

---

## Performance — How to Make It Faster

Current bottlenecks (in order of impact):

### 1. Pass 2 rendering is the slowest part (~200–290s)
- Most time is in the motion debug video (`dbg_writer`)
- `preprocess_frame()` is called every frame with full WTA reprocessing
- **Fix:** The WTA frames are already computed in Pass 1 — cache them or skip reprocessing in Pass 2 (just use stored motion masks directly)

### 2. Ghost ROI retroactive deletion
- When many ghost blobs are pruned, `_unpack_mask_u8` / `_pack_mask_u8` runs on old frames
- **Fix:** Batch the deletions, or do a single end-of-pass cleanup instead of rolling retroactive edits

### 3. Skip-frame YOLO (`--skip-frame-yolo N`)
- YOLO is already skip-framed; increasing N speeds things up but risks missing fast balls

### 4. Court KP detection every frame
- Court doesn't move — detect every N frames and cache

### 5. SciPy B-spline (`splprep`) per motion track per frame
- Relatively expensive for long tracks; pre-compute offline if needed

### 6. RAM frame cache (`--pass2-cache-max-mb`)
- If video fits in RAM, Pass 2 can skip video decode entirely
- Currently auto-enabled if estimated MB < limit

---

## Important Config Params (`SelectorConfig` in `ball_in_play_selector.py`)

| Param | Default | What it does |
|---|---|---|
| `gravity_base_ppf2_30fps` | 0.14 | Base gravity in px/frame² at 30fps |
| `gravity_drag_factor` | 0.985 | Velocity decay per frame (air resistance) |
| `gravity_enabled` | True | Whether KF uses gravity |
| `carry_interp_frames` | ~24 | Max frames to predict without a detection |
| `motion_search_base` | auto | Base radius for motion blob search |
| `track_builder_backend` | `ultra` | `ultra` (Ultralytics) or `greedy` |

---

## File Map

```
triton_tennis-main/
├── now_main.py               # Main pipeline (Pass 1 + Pass 2 + CLI)
├── ball_in_play_selector.py  # Selector, KF, Track scoring
├── models/
│   └── ball.engine           # TensorRT ball+court detector
├── input_videos/             # Input .mp4s
├── output_videos/            # Output (prof_test.mp4, *_motion_debug.mp4, etc.)
├── SaveFolder/               # Manual backup copies of main files
├── requirements.txt          # Core deps (filterpy, scipy, etc.)
├── environment.yml           # Conda env spec
└── PROJECT_CONTEXT.md        # This file
```

---

## How to Start a New Conversation

Paste this at the top:

> "I'm working on `triton_tennis-main`, a tennis ball tracking pipeline. Read `PROJECT_CONTEXT.md` in the repo root for full context. The main files are `now_main.py` (~5k lines) and `ball_in_play_selector.py` (~5.4k lines). Run command: `python now_main.py -i input_videos/VIDEO.mp4 --debug-video --guide-video` in the `tennis-analysis` conda env."
