# GridTrackNet recipe experiments: what was built, what was measured, what is left

Written 2026-10-21, when the experiment chain was cancelled after its first two runs because
the full set would have taken another ten hours. The code is in; this page is the research so
far and the exact protocol to pick up later. The numbers that exist are also in the ledger
`audit/records/gridtracknet_recipe_2026-10.md`.

## Why

The detector the tracker runs (GridTrackNet, five RGB frames at 768x432) had three problems the
October 2026 review found:

1. It trained on one picture and ran on another: the training cache was INTER_AREA JPEG at
   quality 92, the tracker's prepass feeds INTER_LINEAR raw frames.
2. Nothing measured it on recordings it never trained on. The "standard set" (video13 and up)
   was training data for the bundled weights, so its numbers were not held out.
3. Checkpoints were chosen by validation loss, which only loosely tracks what the tracker gets
   (balls found, nothing fired on elsewhere).

The longer-term goal is a retrain from scratch on the workspace dataset. The bundled weights
were trained on some of these very videos, so fine-tuning them is not the goal and none of the
weights these runs produce are for deployment. Fine-tune runs are only the cheap instrument
(about 90 minutes each) for finding out which data choices help on this footage, and what they
find is likely, not certain, to carry over to a scratch run.

## What was built (five commits on the branch, all tests green after each)

| commit | what it does |
|---|---|
| `ee2fcec` | One resize rule: `tennis_tracker/gridtracknet.py` has `RESIZE_INTERPOLATION` and `resize_to_model`, which `frame_tensor` calls (inference unchanged). The trainer's frame cache (`finetune/training/frame_cache.py`, `DATA_VERSION = 5`) stores exactly that picture as lossless PNG, keyed on the video alone; units are derived from the labels every run and a label edit only extracts the frames it newly needs. 7.9 GB for the whole workspace. `photometric_augment` draws its noise as float32. |
| `176ff62` | `finetune/holdouts.json` names fixed held-out sets of whole recordings (`finetune/data/holdouts.py` reads it). `mixed_v1`: prof games 4 and 7 (video17 to 23, video43 to 46), video1 to video10, grid_match8, 16 and 22: 24 clips, 9,033 labels, 518 absent-ball. The trainer takes `--holdout NAME` and its split closes over the recording a clip was cut from (`recording_of`). `evaluate_archive.py --holdout NAME`; raw mode adds a `macro` line (per-clip mean), one `source=<name>` line per source, the count beside every false-alarm rate and `--cell-histogram` (recall by where the ball sat within its grid cell). `compare_eval_logs.py` reads the new lines. |
| `d5bcdbf` | After every scored epoch (`--score-every K`, the last always) the trainer scores `last.npz` on the held-out clips with the detector metric and keeps `best.npz` by `promotion_improves` (a tie keeps the earlier epoch); `best_loss.npz` keeps the loss's choice for the record. `results.csv` has the metric and timings per epoch; `winner.json` has `selected_by` and `best_epoch`. `--seed`. |
| `efc583f` | `--augment` is a set: `basic`, `strong`, `translate` (one integer shift up to 20 px at 768x432 for all five frames, border replicated, held so no label leaves the frame, labels moved in native px), `reverse` (time reversed), `compress` (JPEG re-encode at quality 35 to 85, blur at p 0.2). New draws come after the old ones so old runs reproduce. `--oversample-absent K` and `--oversample-false-alarms K` (units where the starting weights fired at conf 0.3 on a no-ball frame, one prepass before the run) as sampler weights. |
| `9b4985b` | `--all-clips`: nothing held out, nothing scored, `last.npz` is the result, `winner.json` says `selected_by: "recipe"` with the `--selection-run`'s details. README, `finetune/README.md` and AGENTS.md updated. |

Everything is reachable from the trainer and the evaluator; `tests/test_frame_cache.py`,
`tests/test_holdouts.py`, `tests/test_training_clips.py`, `tests/test_train_gridtracknet.py`,
`tests/test_training_dataset.py`, `tests/test_training_metrics.py`,
`tests/test_evaluate_archive.py` and `tests/test_compare_eval_logs.py` pin the behaviour.

The main checkout has one untracked file from this work, `finetune/holdouts.json` (a copy of the
branch's, needed because the trainer reads the record from the workspace), and the PNG cache
under `finetune/cache/data` (regenerable; `--prepare-only` rebuilds it in a few minutes).

## The protocol (what the chain was going to do)

All runs on `mixed_v1`, seed 7, 10 epochs, batch 2, Adadelta lr 1.0, `--score-every 1`,
fine-tuning the bundled weights on the 40 clips outside the set (prof games 2, 5, 6, 10; grid
1, 4, 7, 17, 19, 21, 23, 24, 26, 31; 5,308 units per epoch). One run is about 90 minutes on the
RTX 5080 Laptop GPU on mains (6 minutes an epoch plus 2 to 5 minutes of held-out scoring).

| run | flags on top of the baseline | question |
|---|---|---|
| E0 | none; the bundled weights in raw mode, INTER_LINEAR against INTER_AREA (`output/e0_area.py` flips `RESIZE_INTERPOLATION`, no code change) | what the resize mismatch costs |
| R0 | `--epochs 10 --seed 7` | the baseline on the matched cache |
| R1 | `--augment basic translate` | does the shift flatten the cell histogram and lift recall |
| R2 | `--augment basic reverse` | does time reversal help |
| R3 | `--augment basic compress` | does JPEG re-encoding help on uploaded footage |
| R4a | `--oversample-absent 3` | fewer false alarms from more absent-ball units |
| R4b | `--oversample-false-alarms 3` | fewer false alarms from the hard negatives |
| R5 | every accepted change together | the recipe, which must beat R0 by the same rule |

Acceptance: `promotion_improves` holds on the pooled `conf>=0.50` line against R0 (more recall,
wrong and false alarm not up by more than 0.005), macro recall not down, no `source=` line down
more than 0.010 recall, median px not up more than 0.3. R1 must also narrow the cell-histogram
spread. R4a and R4b target false alarm down with recall within 0.005. A margin under 0.010 is
repeated with `--seed 11` and must keep its sign. R6 (`strong` on top) only if night clips
dominate the misses in the per-clip table.

Then the honest number: R5's `best.npz` against the bundled weights on `mixed_v1` by source,
with the caveat that the bundled weights trained on video1 to 9, so their `source=custom` line
is not held out (prof games 4 and 7 and the three grid clips are).

### Commands

Run from the repository root with the conda interpreter. `output/run_recipe.sh` does one run
(train, then score `best.npz` on the held-out set with the histogram) and writes
`output/<run>/summary.txt`:

```powershell
bash output/run_recipe.sh R1 --epochs 10 --seed 7 --augment basic translate
bash output/run_recipe.sh R2 --epochs 10 --seed 7 --augment basic reverse
bash output/run_recipe.sh R3 --epochs 10 --seed 7 --augment basic compress
bash output/run_recipe.sh R4a --epochs 10 --seed 7 --oversample-absent 3
bash output/run_recipe.sh R4b --epochs 10 --seed 7 --oversample-false-alarms 3
```

Without the script, one run by hand:

```powershell
$env:TENNIS_WORKSPACE = 'C:\Users\Andrew\Desktop\clean_version_gridtracknet_restart\finetune'
& 'C:\Users\Andrew\.conda\envs\tennis-analysis\python.exe' finetune/train_gridtracknet.py --holdout mixed_v1 --epochs 10 --seed 7 --augment basic translate --run-dir output/R1/train --save output/R1/winner.npz
& 'C:\Users\Andrew\.conda\envs\tennis-analysis\python.exe' evaluate_archive.py --archive finetune --mode raw --holdout mixed_v1 --cell-histogram --gridtracknet-weights output/R1/train/best.npz
& 'C:\Users\Andrew\.conda\envs\tennis-analysis\python.exe' evaluation/compare_eval_logs.py output/R0/eval_best.log output/R1/eval_best.log
```

For the scratch retrain the same protocol applies with `--from-scratch` and many more epochs
(the upstream model trained for far longer than 10); the held-out set and the acceptance rule
do not change.

Before any run: on mains, and `nvidia-smi -q -d PERFORMANCE` showing the SW power cap not
active at idle (on battery the limit drops and nothing is comparable). The Windows power mode
was Balanced during the runs below, not Best performance.

## Results so far

### E0: the bundled weights on `mixed_v1`, raw mode at conf 0.50

| resize | pooled recall | wrong | false alarm | macro recall | custom | prof | grid | med px | prepass fps |
|---|---|---|---|---|---|---|---|---|---|
| INTER_LINEAR (what the tracker runs) | 0.925 | 0.012 | 0.210 (109/518) | 0.919 | 0.964 | 0.900 | 0.891 | 2.0 | 65 |
| INTER_AREA (what the old cache trained on) | 0.931 | 0.009 | 0.205 (106/518) | 0.924 | 0.967 | 0.905 | 0.905 | 2.0 | 58 |

The mismatch costs the bundled weights 0.6 points of pooled recall (1.4 on grid, where the
footage is sharpest), under the 2 point bar the plan set for switching the tracker to
INTER_AREA, and AREA's prepass runs at 58 source fps, under the 60 the tracker needs. The
recipe stays INTER_LINEAR on both paths; the matched cache removes the mismatch for anything
trained from now on.

Cell histogram (recall by where the ball sat within its grid cell, LINEAR): the edge bins are
about 3 points under the middle on both axes (x: 0.907 / 0.930 / 0.940 / 0.936 / 0.912;
y: 0.911 / 0.937 / 0.943 / 0.927 / 0.909; spread 0.033 and 0.034). That is the straddle cost the
translate augmentation (R1) is meant to shrink; it has not been measured.

Per source, the bundled weights are strongest on your own camera (0.964, but they trained on
video1 to 9) and weakest on grid (0.891, moving broadcast camera in grid_match8) and prof
(0.900, false alarm 0.278).

### R0: the baseline fine-tune on the matched cache, 9 of 10 epochs (cancelled during epoch 10)

Held-out `mixed_v1` after each epoch (`output/R0/train/results.csv`):

| epoch | recall | wrong | false alarm | med px |
|---|---|---|---|---|
| 1 | 0.938 | 0.011 | 0.305 | 1.89 |
| 2 | 0.937 | 0.008 | 0.266 | 1.86 |
| 3 | 0.936 | 0.008 | 0.268 | 1.83 |
| 4 | 0.937 | 0.008 | 0.297 | 1.85 |
| 5 | 0.934 | 0.008 | 0.270 | 1.87 |
| 6 | 0.938 | 0.009 | 0.268 | 1.85 |
| 7 | 0.937 | 0.008 | 0.259 | 1.85 |
| 8 | 0.934 | 0.007 | 0.247 | 1.84 |
| 9 | 0.936 | 0.007 | 0.230 | 1.83 |

What it says, as far as nine epochs can:

- Recall on footage the run never saw lands at 0.934 to 0.938 from epoch 1 on, about one point
  above the bundled weights (0.925), and does not move with more epochs. The loss keeps falling
  (train 0.000133 to 0.000084) while the held-out number does not: the loss is not the thing to
  select on, which is what commit 3 changed.
- False alarms are the cost: 0.305 after epoch 1 against the bundled 0.210, falling with every
  epoch to 0.230 by epoch 9 but still above the start. The checkpoint rule kept epoch 1 all the
  way (its recall was highest and later epochs did not beat it on recall), which means the rule
  prefers a 0.002 recall gain over a 0.075 false-alarm drop. Worth revisiting for the scratch
  retrain: the plan's R4a and R4b runs are exactly the lever for false alarms, and a selection
  rule that trades a little recall for a lot of false alarm may be wanted.
- No `--all-clips` run and no `ft.py promote`: nothing from this reached the tracker.

Timings (6 minutes an epoch, 2 to 5 minutes of scoring; epoch 9 took 33 minutes while the
machine was busy elsewhere) were measured with another session using the GPU for part of the
run, so they are indicative only.

## What is left

1. R1 to R4b, then R5 and the honest comparison, by the commands above (about 9 hours); or
   the same protocol under `--from-scratch` for the retrain, which is the real goal.
2. Decide the false-alarm question above before trusting a selection rule for the scratch run.
3. `check_parity.py` after any model change (its baseline is stale; do not refresh it to pass).
