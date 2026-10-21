# Project guide

## Purpose and entry points

This is an offline GPU tennis-video tracker, plus a labeling and GridTrackNet
fine-tuning workspace. `clean_tracker.py` runs detection, selection and optional
rendering. Outputs are an annotated MP4 and per-frame JSON. It is not a calibrated
3D ball-flight or line-calling system.

- `tennis_tracker/gridtracknet.py`: PyTorch port and NPZ loader. Five raw RGB
  frames at 768x432 become 15 input channels. Output is five confidence/x-offset/
  y-offset grids at 48x27. Width-axis normalization reproduces the upstream export;
  its values are buffers, not trainable BatchNorm parameters.
- `tennis_tracker/ball_detectors.py`: progressive GridTrackNet prepass and the
  single-frame TensorRT ball engine. At about 60 FPS, even and odd phases each
  run at about 30 FPS. Preserve source-frame indices and temporal order.
  `engine_session.py`: the TensorRT session the ball, player and court engines
  share; `engine_files.py`: locating engine files, their Ultralytics metadata
  and I/O tensor table (no CUDA); `engine_boxes.py`: decoding detection-engine
  outputs into boxes for the ball and player backends. `player_detector.py`:
  TensorRT players with ByteTrack association. `court_detector.py`: court
  keypoints every N frames, last good result cached. `paths.py`: the model asset
  paths every consumer (Config defaults, scripts, the workspace) spells once.
- `tennis_tracker/pipeline.py`: orders the stages, calls the selector and court
  events, reports. `run_context.py`: data the stages pass each other. `pass1.py`
  (with `frame_window.py`): decoding, background/motion state, ROI collection,
  detector calls. `pass2.py`: rendering pass. `export.py`: tracking JSON.
  `motion_debug.py`: motion-debug video. `court_geometry.py`: keypoints, side
  guides, polygon, homography. `court_minimap.py`: minimap. `rendering.py`
  draws only. `summaries.py`: the console track table and timing report.
- `ball_in_play_selector/`: candidate association, Kalman state, track scoring,
  timeline selection, bounded recovery, trajectory refit.
- `tennis_tracker/motion.py`, `tracking.py`: CUDA motion masks and ROI tracking.
- `tennis_tracker/court_overlay.py`: CatBoost/kinematic bounce candidates,
  player contacts and schematic rally legs.
- `finetune/refine.py`: model-free sub-pixel ball suggestions for the label tool
  (motion centroid, candidates chosen by trajectory, trust score).
- `finetune/cnn_refiner.py`: small CNN, trained only on frames the user checked, that places
  those suggestions where the user clicks. Versions live in
  `finetune/models/refiner/` (all kept, `current.txt` names the active one); retrain with
  `ft.py hints train` or menu option 10. Its self-check decides "trusted" (threshold set on
  folds of clips it never saw); refine.py still finds the ball and decides ball/no ball.
  In the label tool an unreviewed frame's suggestion is the mark SPACE takes (sure ones
  always, unsure ones within 3 px of the draft; see `Tool.offer_for` for the measurement);
  `x` keeps the draft, `y` takes a hint beside a reviewed label. `ft.py hints sheet <clip>`
  renders a zoomed PNG of where two versions disagree, for judging marks by eye. A trajectory
  prior on the hints was measured and makes them worse (see the module docstring).
- `finetune/ft.py`: workflow front door; `train_gridtracknet.py`: training;
  `pretrack.py`: drafts; `label_tool.py`: manual edits; `import_data.py` and
  `realign.py`: source conversion and temporal realignment. The label tool is
  split into `label_store.py` (a clip's rows and sidecars), `label_source.py`
  (one clip of a labelling run), `label_window.py` (window plumbing) and
  `label_hints.py` (the suggestions). `labels.py`: the label file format;
  `label_audit.py`: checks labels against the video and the raw detector;
  `workspace.py`: where the workspace keeps things and a clip's video facts;
  `scoring.py`: the detector metric; `detect.py`: the raw prepass over a clip;
  `data_policy.py`: which clips may be trained, validated or scored on;
  `fileio.py`: atomic workspace writes. `workspace.py` is also the one place
  `clips.csv` and `exclude.txt` are read and written (`TENNIS_WORKSPACE` relocates
  the workspace); `alignment.py`: the one label-vs-detector agreement the audit
  and `realign.py` share.
- `evaluation/`: `tracking.py` scores a tracking JSON against annotations in native
  pixels (`check_parity` freezes its numbers), `archive.py` scores the detector or
  the pipeline on workspace clips with `finetune/scoring.py`, `parity.py` is the
  Pomona gate. `validate_tracking.py`, `evaluate_archive.py` and `check_parity.py`
  at the root are thin entry points to them.

## Environment and commands

Run commands from the repository root. The verified local interpreter is
`C:\Users\Andrew\.conda\envs\tennis-analysis\python.exe`. Plain `python` currently
resolves to a different Python installation. Other machines should use their own
environment, not copy this user-specific path into application code.

Runtime dependencies include CUDA PyTorch, NumPy, OpenCV and TensorRT;
BoxMOT supplies player tracking. CatBoost is needed for `--court`. Ultralytics is
used by engine export. FFmpeg/NVENC can accelerate rendering. Versions and model/
source hashes for the review are in `audit/snapshot.json`. There is no complete
locked installation manifest. TensorRT engines may be GPU/version specific.

```powershell
& 'C:\Users\Andrew\.conda\envs\tennis-analysis\python.exe' -m pytest -q
& 'C:\Users\Andrew\.conda\envs\tennis-analysis\python.exe' clean_tracker.py --self-test
& 'C:\Users\Andrew\.conda\envs\tennis-analysis\python.exe' finetune/train_gridtracknet.py --self-test
& 'C:\Users\Andrew\.conda\envs\tennis-analysis\python.exe' clean_tracker.py --no-video --tracking-json output/new_review_tracking.json
& 'C:\Users\Andrew\.conda\envs\tennis-analysis\python.exe' evaluate_archive.py --archive finetune --mode raw --clips video13
```

`python -m pytest -q` (pytest required) runs every test: all suites live in
`tests/` (`pytest.ini` sets `testpaths` and puts the root on the import path),
use synthetic inputs only and need no GPU. `clean_tracker.py --self-test` runs
the selector, court-overlay, ROI and GridTrackNet modules from `tests/` through
pytest; `train_gridtracknet.py --self-test` is also run by
`tests/test_train_gridtracknet.py`. Put new tests in `tests/`, shared synthetic
fixtures in `tests/_helpers.py`. `audit/` keeps the review's record (reports,
logs, verification scripts); `audit/review_checks.py` documents the September
2026 defects and no longer runs.

Always specify `--archive finetune` for workspace evaluation: the evaluator can
otherwise prefer a separate Desktop archive. Use distinct output paths for new
runs. `check_parity.py` verifies a frozen Pomona benchmark; the September 7, 2026
worktree fails several of its gates. Do not silently refresh that baseline to
make a change pass.

## Label review status - October 18, 2026

The user has reviewed, by hand in the label tool:

- every video clip: `PomonaPitzer_Women_vs._UCSD-cut-merged-1778697988405` and
  `video1` through `video53`, including `video3`, `video8`, `video11` and
  `video12`, which were unreviewed before;
- every GridTrackNet clip from `grid_match1` through `grid_match31` (1, 4, 7, 8,
  16, 17, 19, 21, 22, 23, 24, 26, 31).

`grid_match36` is in progress: it is "the one after grid_match31" from
`review.txt`, and the user may have the label tool open on it, so never write
its files. The user also said "done 33"; no `grid_match33` exists, so that is
unresolved - do not guess what it means.

Until the user says otherwise, do not test, compare, tune thresholds, or select
models using `grid_match36` or the clips `review.txt` still lists as needing
work: `grid_match37`, `grid_match40`, `grid_match47`, `grid_match49`,
`grid_match50`, `grid_match55`, `grid_match73`, `grid_match78`, `grid_match85`,
`grid_match86`, `grid_match87`, `grid_match89`, `grid_match92`. The remaining
GridTrackNet clips above 31 (38, 42, 43, 44, 53) were never on that list, but
the user has not said they are reviewed either. Reviewed clips may be used for
measurement and validation; video13 and up remain the standard set.

`review.txt` holds the user's original notes. These rules concern evaluation; do
not silently rewrite labels or training exclusions as part of a review. Label
files for grid_match and video13+ clips are gitignored, so back them up before
any bulk change.

## Data and change discipline

Label CSVs use `frame_###,ball_x,ball_y`, native image pixels, zero-based source
frame indices, and a top-right sentinel for invisible balls. Labels on 60 FPS
clips may use either odd or even phase. Missing labels are not absent-ball
labels. Reviewed, edited and suspect flags live in sidecar JSON files.

Preserve hand edits, phase, source mapping and backups. The worktree already had
uncommitted edits to training/labeling tools, labels, manifests and Config before
this review. Do not revert those or promote/retrain weights as incidental cleanup.
`clips.csv` notes, `exclude.txt` and review sidecars convey different things:
passing the format checker does not establish annotation correctness.

Labels clicked in the label tool before commit 307f632 (2026-10-18) sat about half a
pixel up-left of the ball; commit 8c36b5e moved them back (backups:
`<csv>.before_halfpixel_fix`). Do not apply that correction again.

Keep RGB detector inputs unchanged unless an experiment explicitly changes the
model contract. Keep observations distinguishable from fitted positions. Short
unsupported gaps should remain absent. Physics is presently an image-space
prior; do not present its extrapolations as measured 3D positions or speeds.

See `audit/PROJECT_REVIEW.md` before modifying evaluation, prepass publication,
label frame identity, realignment pins, motion masks or physics. The audit's
`review_checks.py` records how the September 2026 defects were reproduced; it
exits without running; the tests that assert the fixes live next to each
module's tests (`tests/test_<module>.py`). Convert
relevant checks to desired-behavior regression tests when making fixes. The user has now authorized
fixing the reviewed defects and running validation. Architecture experiments
remain separate from the correctness fixes; do not promote a model implicitly.

The disabled manual perspective scaling, homography gravity, color gate,
flicker suppression and raw-component experiments were subsequently removed.
Keep automatic FPS/resolution normalization and supported debug outputs.
See `audit/CLEANUP_RESULTS.md` for rationale and measured output differences.
