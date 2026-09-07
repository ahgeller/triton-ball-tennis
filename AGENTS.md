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
- `tennis_tracker/detectors.py`: progressive GridTrackNet prepass and TensorRT
  player/court backends. At about 60 FPS, even and odd phases each run at about
  30 FPS. Preserve source-frame indices and temporal order.
- `tennis_tracker/pipeline.py`: decoding, motion/background state, ROI collection,
  selector invocation, optional court events, rendering, JSON export.
- `ball_in_play_selector/`: candidate association, Kalman state, track scoring,
  timeline selection, bounded recovery, trajectory refit.
- `tennis_tracker/motion.py`, `tracking.py`: CPU/CUDA motion masks and ROI tracking.
- `tennis_tracker/court_overlay.py`: CatBoost/kinematic bounce candidates,
  player contacts and schematic rally legs.
- `finetune/ft.py`: workflow front door; `train_gridtracknet.py`: training;
  `pretrack.py`: drafts; `label_tool.py`: manual edits; `import_data.py` and
  `realign.py`: source conversion and temporal realignment.

## Environment and commands

Run commands from the repository root. The verified local interpreter is
`C:\Users\Andrew\.conda\envs\tennis-analysis\python.exe`. Plain `python` currently
resolves to a different Python installation. Other machines should use their own
environment, not copy this user-specific path into application code.

Runtime dependencies include CUDA PyTorch, NumPy, OpenCV, FilterPy and TensorRT;
BoxMOT supplies player tracking. CatBoost is needed for `--court`. Ultralytics is
used by engine export. FFmpeg/NVENC can accelerate rendering. Versions and model/
source hashes for the review are in `audit/snapshot.json`. There is no complete
locked installation manifest. TensorRT engines may be GPU/version specific.

```powershell
& 'C:\Users\Andrew\.conda\envs\tennis-analysis\python.exe' clean_tracker.py --self-test
& 'C:\Users\Andrew\.conda\envs\tennis-analysis\python.exe' finetune/train_gridtracknet.py --self-test
& 'C:\Users\Andrew\.conda\envs\tennis-analysis\python.exe' clean_tracker.py --no-video --tracking-json output/new_review_tracking.json
& 'C:\Users\Andrew\.conda\envs\tennis-analysis\python.exe' evaluate_archive.py --archive finetune --mode raw --clips video13
```

Always specify `--archive finetune` for workspace evaluation: the evaluator can
otherwise prefer a separate Desktop archive. Use distinct output paths for new
runs. `check_parity.py` verifies a frozen Pomona benchmark; the September 7, 2026
worktree fails several of its gates. Do not silently refresh that baseline to
make a change pass.

## User-specified label review exclusions — September 7, 2026

Do not test, compare, tune thresholds, or select models using these clips until
the user says their review is finished:

`grid_match21`, `grid_match24`, `grid_match26`, `grid_match31`, `grid_match37`,
`grid_match40`, `grid_match47`, `grid_match49`, `grid_match50`, `grid_match55`,
`grid_match73`, `grid_match78`, `grid_match85`, `grid_match86`, `grid_match87`,
`grid_match89`, `grid_match92`, `video3`, `video8`, `video11`, `video12`.

The user also identified "the one after grid_match31" as wobbly; its exact ID is
unresolved. Do not infer that ID from sorting or assume another GridTrackNet
clip is cleared. The user subsequently confirmed that **video13 and all higher
numbered video clips are manually verified and accurate**. Use those for the
fix-validation runs; their numeric names refer to video clips, not grid_match
IDs. Other labels are still being worked on. Do not overwrite ongoing edits.
The user specifically says video11 and video12 have not been reviewed at all.
`review.txt` contains the original notes. These restrictions concern evaluation;
do not silently rewrite labels or training exclusions as part of a review.

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

Keep RGB detector inputs unchanged unless an experiment explicitly changes the
model contract. Keep observations distinguishable from fitted positions. Short
unsupported gaps should remain absent. Physics is presently an image-space
prior; do not present its extrapolations as measured 3D positions or speeds.

See `audit/PROJECT_REVIEW.md` before modifying evaluation, prepass publication,
label frame identity, realignment pins, motion masks or physics. The audit's
`review_checks.py` reproduces current defects with synthetic inputs: its passing
assertions do NOT mean those defects are fixed. Convert relevant checks to
desired-behavior regression tests when making fixes. The user has now authorized
fixing the reviewed defects and running validation. Architecture experiments
remain separate from the correctness fixes; do not promote a model implicitly.
