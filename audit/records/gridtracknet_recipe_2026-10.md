# GridTrackNet recipe ledger, October 2026

**What this is for.** The user will retrain GridTrackNet from scratch on the workspace dataset:
the bundled weights were trained on some of these very videos, so fine-tuning them is not the
goal and no weights from these runs are meant for deployment. The runs below fine-tune only
because a fine-tune is the cheap instrument (about 100 minutes each) for measuring which data
choices help on this footage: the resize rule, the augmentations, the hard-negative sampling,
the checkpoint rule. What is kept is the code and these results. A flag that helps a fine-tune
is likely, not certain, to help a scratch run; the scratch retrain should re-check the margins
it cares about on the same held-out set.

The runs behind the detector recipe (plan item 2): every experiment, its flags and seed, what it
measured on the held-out set `mixed_v1` (`finetune/holdouts.json`: prof games 4 and 7, video1 to
video10, grid_match8, 16 and 22; 24 clips, 9,033 labels, 518 of them absent-ball) and what was
decided from it. Logs sit under `output/<run>/` (untracked): `train.log`, `train/results.csv`,
`train/config.json`, `train/winner.json`, `eval_best.log`.

Rules, from the plan: a change is accepted when `promotion_improves` holds on the pooled
`conf>=0.50` line against R0, macro recall is not down, no `source=` line drops more than 0.010
recall and the median px is not up more than 0.3; R1 must also flatten the cell histogram; R4a
and R4b target false alarm down with recall within 0.005. Margins under 0.010 are repeated with
seed 11 and must keep their sign. R5 is every accepted change together and must beat R0 by the
same rule. Nothing is promoted here; `ft.py promote` is the user's call.

Machine: RTX 5080 Laptop GPU (16 GB, driver 617.42, CUDA 12.8 torch 2.10.0), on mains. The
Windows power mode was Balanced (the AC overlay), not Best performance; `nvidia-smi` reported
the SW power cap not active before each run (power limit 104 W of a 120 W maximum). Another
session ran the court-event pipeline evaluation on the same GPU during the early runs, so the
timings below are not comparable between runs, only the metrics are.

All runs: 10 epochs, batch 2, Adadelta lr 1.0, `--score-every 1`, fine-tuning the bundled
weights (`models/gridtracknet_weights_torch.npz`), training on the 40 clips outside the set
(prof games 2, 5, 6, 10; grid 1, 4, 7, 17, 19, 21, 23, 24, 26, 31), the PNG cache of commit 1.

## Runs

| run | commit | flags | seed | best epoch | pooled recall / wrong / false alarm (n) | macro recall | custom / prof / grid recall | med px | decision |
|---|---|---|---|---|---|---|---|---|---|
| E0 linear | 9b4985b | bundled weights, raw, `--holdout mixed_v1 --cell-histogram` | - | - | 0.925 / 0.012 / 0.210 (109/518) | 0.919 | 0.964 / 0.900 / 0.891 | 2.0 | the reference; cell spread 0.033 (x) and 0.034 (y); prepass 65 fps |
| E0 area | 9b4985b | same, `RESIZE_INTERPOLATION` flipped to INTER_AREA by `output/e0_area.py` | - | - | 0.931 / 0.009 / 0.205 (106/518) | 0.924 | 0.967 / 0.905 / 0.905 | 2.0 | +0.6 recall, under the 2 point bar, prepass 58 fps: the recipe stays INTER_LINEAR |
| R0 | 9b4985b | `--epochs 10 --seed 7` (basic augmentation), cancelled in epoch 10 | 7 | 1 (of 9 scored) | 0.938 / 0.011 / 0.305 (epoch 1); 0.936 / 0.007 / 0.230 (epoch 9) | - | - | 1.9 | baseline: recall +1 point over the bundled weights from epoch 1 on, false alarms up and falling with epochs; not scored by source (cancelled) |
| R1 to R5 | - | see `docs/gridtracknet_recipe_experiments.md` | - | - | - | - | - | - | not run: the chain was cancelled on 2026-10-21 at the user's request (ten more hours); the protocol and commands are in the doc |

## E0: the current weights, INTER_LINEAR against INTER_AREA

Full tables, the cell histogram and the per-source reading are in
`docs/gridtracknet_recipe_experiments.md`. Logs: `output/E0/eval_linear.log`, `output/E0/eval_area.log`.

## Notes

- The epoch-by-epoch held-out numbers of R0 are in the doc and in `output/R0/train/results.csv`.
  The checkpoint rule kept epoch 1 (highest recall) although false alarms fell by 0.075 by
  epoch 9; whether the rule should trade a little recall for a large false-alarm drop is an open
  question for the scratch retrain.
