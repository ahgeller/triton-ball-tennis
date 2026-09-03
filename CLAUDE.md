# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Run everything with the finetuning venv — the `python` on PATH is 3.14 without torch:

```powershell
$py = "C:\Users\Andrew\Desktop\gridtracknet_finetuning\.venv\Scripts\python.exe"   # py3.12, torch 2.11 cu128, TensorRT 10.15

& $py clean_tracker.py                                   # bundled Pomona sample -> output/tracking.mp4 + output/tracking.json
& $py clean_tracker.py --input path\to\match.mp4 --output output\x.mp4 --tracking-json output\x.json
& $py clean_tracker.py --no-video                        # JSON only (skips pass 2 entirely)
& $py clean_tracker.py --court                           # add the 2D court minibar (needs catboost + ctb_regr_bounce.cbm)
& $py clean_tracker.py --info                            # per-stage timing breakdown
& $py clean_tracker.py --device 1 --conf 0.55            # CUDA device index, GridTrackNet confidence threshold

& $py clean_tracker.py --self-test                       # the entire test suite (see below)
& $py check_parity.py                                    # frozen Pomona gate; --existing to reuse output/tracking.json (see caveat)
& $py validate_tracking.py --predictions output\tracking.json --annotations sample\pomona_annotations.json
& $py evaluate_archive.py --mode raw                     # detector-only benchmark on the hand-labelled archive clips, threshold sweep
& $py evaluate_archive.py --mode pipeline                # full tracker on the archive clips, scored per frame / per source / what bad outputs landed on
& $py evaluate_archive.py --mode raw --archive finetune  # same against finetune/labels; --gridtracknet-weights x.npz scores another model
& $py build_trt_engines.py                               # rebuild models/*.engine for a new GPU/TensorRT

& $py finetune\pretrack.py                               # draft labels (detector guess + confidence) for every video in finetune/videos
& $py finetune\label_tool.py <clip>                      # click-correct the draft -> finetune/labels/<clip>_ball.csv
& $py finetune\train_gridtracknet.py --val-clips <clip>  # fine-tune (or --from-scratch); keeps the better of new vs starting weights
```

Requires an NVIDIA CUDA GPU (RTX 5080 Laptop, 16 GiB here). `run()` raises if `--device` is not a CUDA index — there is no CPU fallback. `output/` and `AiFolder/` are gitignored.

## Ground truth and gates

`C:\Users\Andrew\Desktop\gridtracknet_finetuning\archive\videoN.mp4` + `videoN_ball.csv` are the user's hand-labelled clips (video1–10, 12; 60 FPS clips labelled every second frame; an invisible ball is parked in the top-right corner — see `is_visible`). **Judge tracking quality against these (`evaluate_archive.py`), not against previous `output/*.json` runs** — those contain the errors being fixed. The bundled GridTrackNet fine-tune was trained on every archive clip except video10, so only video10 is held-out for it.

`check_parity.py`'s BASELINE (recall 1.0, mean 1.82 px, 687 filled frames) predates the GridTrackNet switch and is not reproduced by committed HEAD or the working tree (2026-08-29: 72–73/76 labels, ~4 px mean, ~1000 filled frames); a fresh checkout also trips its sha check via CRLF. Treat it as a smoke test and compare candidate runs against a same-day run of the unmodified code with `validate_tracking.py`.

## Testing

There is no pytest/unittest framework. **All tests live in `clean_tracker.py::_self_test`** as inline assertions that pin the selector's behavioral contract (safe gap filling, conservative refusal to fill unanchored gaps, static-clutter rejection, timeline stitching incl. interleaved tracks, isolated-detection pruning, ROI probing cadence, bounce feature layout). Add new deterministic checks there rather than creating a test directory. `finetune/train_gridtracknet.py --self-test` covers the trainer.

## Architecture

Two entry points into one flow: `clean_tracker.py` builds a `Config` and calls `tennis_tracker.pipeline.run(cfg)`.

### GridTrackNet is a whole-video prepass, not a per-frame detector

`GridTrackNetBallDetector.prepare_video()` (`tennis_tracker/detectors.py`) decodes the entire video once, batches it through the network, and caches one detection per frame; `detect()` afterwards replays `precomputed[index]`. Since 2026-08-29 the prepass runs on a **worker thread with its own CUDA stream** so pass 1 starts immediately and `detect()` only blocks if pass 1 catches up (`Config.gridtracknet_prepass_background`); this took a 34 s clip from 51 s to 33.5 s `--no-video` with sub-pixel-identical output. Consequences worth knowing before touching detection code:

- The model consumes **five frames at 30 FPS spacing**. At 60 FPS input `source_stride = 2` and two independent phase streams (even/odd frames) are run so every frame gets an output. Any FPS outside ~22–32 or ~57–62 is rejected outright; `Config.gridtracknet_source_stride` overrides the stride for experiments.
- `uses_raw_frames = True`, so GridTrackNet sees raw RGB frames. The motion/preprocessing path in `tennis_tracker/motion.py` never alters its input — motion exists only as *evidence* (`Detection.on_motion`, boost masks) for the selector.
- Player and court detection go through TensorRT engines loaded directly by `_TensorRTRuntimeSession`. Engines are platform-specific.
- Where `--no-video` time goes on a 34 s 1080p60 clip: pass 1 ~29 s (player/court TensorRT ~11 s, CUDA motion preprocessing ~13 s), selector <1 s, prepass overlapped.

### Pipeline passes (`tennis_tracker/pipeline.py`, `run()`)

1. **Pass 1** — decode + preprocess (CUDA path preferred, CPU fallback), per-frame ball detections, boost/raw motion masks (ROI-limited via `ROIMotionTracker`), player boxes, court keypoints. Masks are stored run-length-packed (`_pack_mask_u8`). Ghost ROIs are erased in one batched sweep at the end of the pass. Decoded frames may be cached in RAM for pass 2 (`pass2_cache_max_mb`).
2. **Selector** — `select_ball_in_play()` runs once over the whole video, not incrementally.
3. **Optional court overlay** — only when `--court`: bounce prediction (CatBoost), player contacts, kinematic bounces, rally legs, minimap.
4. **Pass 2** — re-render with trail/overlays and write the video. Skipped entirely with `--no-video`.
5. `_write_tracking_json()` emits schema_version 1 consumed by `validate_tracking.py`, `check_parity.py` and `evaluate_archive.py` (per-frame rows carry `player_boxes` and 28-float `court_keypoints`).

### Ball-in-play selector (`ball_in_play_selector/`)

`core.select_ball_in_play()` is the whole contract, in order:

`build_detections` → `build_tracks` (association/gating) → `score_tracks` (court, player, motion, movement, coverage terms) → `_selected_tracks` (established / reacquired / rolling gates) → `_select_timeline_chain` (best non-overlapping sequence over time) → `_stitch_track_chain` → per-frame results → `_refine_trajectory` (robust local refit).

**The design invariant is conservatism**: an unresolved detector gap stays a gap. Interpolation only happens when a *real future detection* anchors the far endpoint, both endpoints are confident, and the direction is supported; motion recovery needs actual mask support near the predicted point. Do not "improve" continuity by relaxing these — the self-tests assert several of these refusals explicitly, and the selector is sensitive enough that every change needs before/after numbers from `evaluate_archive.py --mode pipeline` plus a same-day Pomona comparison.

Rules added 2026-08-29 (all pinned by self-tests; archive recall 89.2% → 95.8% at unchanged 0.4% wrong-object): (1) a detection isolated from its track by > ~0.1 s on both sides is dropped, and a fill longer than ~0.1 s needs corroborated anchors and ≤ 2× speed change (stops "jump to the van and back"); (2) a ≥3-frame gap whose incoming and outgoing directions oppose (a hit inside the gap) is not filled straight; (3) `_select_timeline_chain` judges handovers at the frame the successor starts, and its post-pass keeps tracks that share no observation frames with the chain when they are spatially continuous with it (the ball switching track at a bounce and back used to lose the whole middle segment).

Per-frame `FrameResult.source` drives everything downstream (renderer color, JSON, validation):

| source | meaning | trail color |
|---|---|---|
| `det` | raw GridTrackNet observation | green |
| `motion` | motion-mask-supported recovery | orange |
| `interp` / `carry` | bounded fill / evidence-bounded tail | cyan (physics) |

### Configuration

Two dataclasses hold all tuning; do not scatter new constants elsewhere.

- `tennis_tracker/config.py::Config` — runtime/pipeline knobs (paths, motion thresholds, ROI, encoding, debug outputs, prepass threading/stride).
- `ball_in_play_selector/config.py::SelectorConfig` — selector knobs. `auto_scale()` derives every resolution/FPS-dependent value from a **30 FPS, 1920×1080 (diag 2203)** reference; short temporal windows are expressed in seconds and converted to frames. New thresholds must be added the same way so 30 vs 60 FPS behavior stays consistent.

## Fine-tuning loop (`finetune/`)

`pretrack.py` (detector guess per label frame) → `label_tool.py` (SPACE = good, click = fix, both advance; persistent zoom; cursor placed on the guess) → `train_gridtracknet.py` / `GridTrackNet_FineTune_Colab.ipynb` (keeps the better of trained vs starting weights on held-out clips). Labels are at model cadence: every 2nd frame of 60 FPS video, every frame of 30 FPS. See `finetune/README.md`.

## Conventions

Four-space indent, `snake_case`/`PascalCase`/`UPPER_CASE`, type hints on public boundaries, imports grouped stdlib → third-party → local. No formatter or linter is configured; match nearby code. Reuse the existing package helpers before adding abstractions. See `AGENTS.md` for commit/PR expectations (short plain-English subjects, no Conventional Commit prefixes; PRs should list validation commands and results and call out GPU/TensorRT assumptions).
