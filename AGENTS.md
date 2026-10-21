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
  `headless.py`: the one Config and run for a caller that wants the tracker's
  answer and no video (`evaluation/archive.py`, `pretrack.py`).
- `tennis_tracker/pipeline.py`: orders the stages, calls the selector and court
  events, reports. `run_context.py`: data the stages pass each other. `pass1.py`
  (with `frame_window.py`): decoding, background/motion state, ROI collection,
  detector calls. `pass2.py`: rendering pass. `export.py`: tracking JSON.
  `motion_debug.py`: motion-debug video. `court_geometry.py`: keypoints, side
  guides, polygon, homography. `court_minimap.py`: minimap. `rendering.py`
  draws only: the canvas, colours, markers and HUD, with `court_drawing.py`
  (court layers), `trails.py` (ball, display-path and candidate trails) and
  `display_path.py` (the chosen track's display path) beside it.
  `summaries.py`: the console track table and timing report.
- `ball_in_play_selector/`: candidate association, Kalman state, track scoring,
  timeline selection, bounded recovery, trajectory refit.
- `tennis_tracker/motion.py`, `tracking.py`: CUDA motion masks and ROI tracking.
- `tennis_tracker/court_events.py`: CatBoost/kinematic bounce candidates,
  player contacts and schematic rally legs.
- `finetune/refiner/refine.py`: model-free sub-pixel ball suggestions for the label tool
  (motion centroid, candidates chosen by trajectory, trust score).
- `finetune/refiner/cnn_refiner.py`: small CNN, trained only on frames the user reviewed, that places
  those suggestions where the user clicks (`refiner_training.py` does the training; `hints.py`
  is their command line, the script `ft.py hints` runs). Versions live in
  `finetune/models/refiner/` (all kept, `current.txt` names the active one); retrain with
  `ft.py hints train` or menu option 10. Its self-check decides "trusted" (threshold set on
  folds of clips it never saw); refine.py still finds the ball and decides ball/no ball.
  In the label tool an unreviewed frame's suggestion is the mark SPACE takes (sure ones
  always, unsure ones within 3 px of the draft; see `Suggestions.offer_for` in
  `label_hints.py` for the measurement);
  `x` keeps the draft, `y` takes a hint beside a reviewed label. `ft.py hints sheet <clip>`
  (`refiner_sheet.py`) renders a zoomed PNG of where two versions disagree, for judging marks by eye. A trajectory
  prior on the hints was measured and makes them worse (see the module docstring).
- `finetune/ft.py`: workflow front door (launchers, script commands;
  `menu.py` is the interactive menu, `commands.py` has the in-process commands,
  `interpreter.py` the venv hand-over). The rest of the package is four
  subpackages. `finetune/labeling/`: `label_tool.py` (manual edits: the
  actions the key table names, the window and the loop), split into
  `label_session.py` (the one owner of the sitting's state: the rows, the
  cursor and its walk, the view geometry, the toggles, the suggestions, which
  the views and handlers read and write directly),
  `label_cli.py` (its command line), `label_store.py` (a clip's rows and
  sidecars), `label_source.py` (one clip of a labelling run, and the prefetcher
  that decodes ahead of the cursor), `label_window.py` (window plumbing and the
  key tables), `label_cursor.py` (parking the OS pointer on the ball),
  `label_keys.py` (what the keys do), `label_mouse.py` (what the mouse does),
  `label_frame_view.py` (the single-frame picture and HUD), `label_grid.py` and
  `label_sheet.py` (the grid and trajectory-sheet views, each owning its state,
  picture and keys), `label_draw.py` (the filters, marks and text every view
  draws with) and `label_hints.py` (the suggestions). `finetune/refiner/`: the
  suggestion model above (`refine.py`, `cnn_refiner.py`, `refiner_training.py`,
  `refiner_sheet.py`, `hints.py`). `finetune/data/`: `labels.py`, the label
  file format; `label_check.py`, checks a label file against its video's facts;
  `label_audit.py`, checks labels against the raw detector; `label_flags.py`, the frames an
  audit flagged for a person (the flag sidecar the label tool and the menu read); `detect.py`,
  the raw GridTrackNet prepass over one clip (torch only when a detector is built), which
  the audit, `realign.py`, `pretrack.py`, `evaluation/archive.py` and the trainer's winner
  metric all run; `workspace.py`, where the workspace keeps things and a clip's video
  facts, and the one place `clips.csv` and `exclude.txt` are read and written
  (`TENNIS_WORKSPACE` relocates the workspace); `detector_metrics.py`, the detector metric (`DetectorMetrics`);
  `data_policy.py`, which clips may be trained, validated or scored on;
  `fileio.py`, atomic workspace writes; `alignment.py`, the one label-vs-detector
  agreement the audit and `realign.py` share; `import_data.py` (its importers;
  `import_sources.py` holds what they share) and `realign.py` (with the pure
  solver in `realign_solve.py`), source conversion and temporal realignment. `finetune/training/`:
  `train_gridtracknet.py` (the training run: its command line, data, epochs and
  the winner; `finetune/train_gridtracknet.py` at the old path only hands over
  to it, for the command below), `loop.py` (the loss, one epoch, the validation
  loss and saving weights), `metrics.py` (the detector metric a run is judged
  by and the winner rule), `clips.py` (which clips
  it trains on and holds out), `frame_cache.py` (the per-clip cache of frames
  and training units), `dataset.py` (targets, augmentation, the Dataset and
  the sampler) and `pretrack.py` (drafts).
- `evaluation/`: `validate.py` scores a tracking JSON against annotations in native
  pixels (`check_parity` freezes its numbers), `archive.py` scores the detector or
  the pipeline on workspace clips with `finetune/data/detector_metrics.py` (`pipeline_score.py`
  scores one pipeline run's clip), `parity.py` is the Pomona gate. `validate_tracking.py`, `evaluate_archive.py` and `check_parity.py`
  at the root are thin entry points to them.

## Environment and commands

Run commands from the repository root. The verified local interpreter is
`C:\Users\Andrew\.conda\envs\tennis-analysis\python.exe`. Plain `python` currently
resolves to a different Python installation. Other machines should use their own
environment, not copy this user-specific path into application code.

Runtime dependencies include CUDA PyTorch, NumPy, OpenCV and TensorRT;
BoxMOT supplies player tracking. FilterPy is not one: the selector's Kalman filter
is NumPy (`ball_in_play_selector/physics.py`), and `tests/test_ball_kalman.py` only
compares against FilterPy when it is installed. pip's `opencv-python` serves the
tracker, but the label tool needs conda-forge's Qt build of OpenCV (see
`requirements.txt`). CatBoost is needed for `--court`. Ultralytics is
used by engine export. FFmpeg/NVENC can accelerate rendering. Versions and model/
source hashes for the review are in `audit/records/snapshot.json`. There is no complete
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
the selector, court-event, ROI and GridTrackNet modules from `tests/` through
pytest; `train_gridtracknet.py --self-test` is also run by
`tests/test_train_gridtracknet.py`. Put new tests in `tests/` as plain pytest
functions (fixtures and `parametrize`, not `unittest.TestCase`), shared synthetic
fixtures in `tests/_helpers.py`. `audit/` keeps the review's reports and
verification scripts; `audit/records/` holds their logs and results, and
`audit/records/review_checks_2026-09.py.txt` is the September 2026 defect
reproduction script, kept as text: it no longer runs.

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
label frame identity, realignment pins, motion masks or physics.
`audit/records/review_checks_2026-09.py.txt` records how the September 2026
defects were reproduced; its pointer to `tests/test_fixes.py` is out of date: the
tests that assert the fixes live next to each module's tests
(`tests/test_<module>.py`). Convert relevant checks to desired-behavior
regression tests when making fixes. The user has now authorized
fixing the reviewed defects and running validation. Architecture experiments
remain separate from the correctness fixes; do not promote a model implicitly.

The disabled manual perspective scaling, homography gravity, color gate,
flicker suppression and raw-component experiments were subsequently removed.
Keep automatic FPS/resolution normalization and supported debug outputs.
See `audit/CLEANUP_RESULTS.md` for rationale and measured output differences.
