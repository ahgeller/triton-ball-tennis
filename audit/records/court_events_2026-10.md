# Court events with provenance: the measurements behind item 1 (October 21, 2026)

The October 2026 review's item 1: court events carried no provenance and were never exported,
so landings could not be scored. Five commits on top of main 2c10665, each measured before it
was kept, as the plan (`precious-dreaming-pebble`) requires. Every run below is on this laptop
(RTX 5080 Laptop GPU) on mains but on the Balanced power plan, so the GPU sat at 40 W and
about 1300 MHz: timings are not comparable with earlier logs, results are. The run folders
named below are under `output/` (gitignored) of the worktree the work was done in.

Two tracker runs of one clip on identical code are not byte-identical: the FP16 GridTrackNet
forward moves detections sub-pixel run to run, and the refit spreads that over whole runs of
rows (`gpu-reruns-are-not-bit-identical` in the memory notes). Output neutrality is therefore
judged by the golden harness (deterministic fakes, `tests/test_pipeline_run.py`) and, on the
real clip, by identical presence per frame, fill, track and chosen-track counts.

## 1a: provenance fields (8438b4a), output-neutral

- Full suite green with no golden edit.
- `clean_tracker.py --no-video` on `sample/VolyAvw-cut-merged-1771797159395.mp4` before and
  after (`output/court_events_1a/compare.txt`): presence identical on all 3286 rows, 2559
  filled both times, 48 tracks, chosen track 0, position kinds 131 measurement / 2318 filtered /
  110 recovered both times; 57 rows moved under 0.005 px (6 measurements, 3 confidences and
  3 boxes differ): the detector's noise.

## 1b: computed whenever --court is on, exported (a221ea7)

- Suite green after the two-line golden edit (`schema_version` 3, `court_events: {}` in both
  sections).
- `--court --no-video` on the sample (`output/court_events_1b/court.log`, `compare.txt`):
  12 confirmed bounces (18 candidates, 10 kinematic), 19 contacts, 14 legs (3 inferred,
  1 volley), the plan's reference counts; every event carries origin and position
  (bounces: 5 measurement, 5 filtered, 2 recovered; kinematic: 8 kink, 2 gap; contacts 17 kink,
  2 gap; legs 13 bounce with landing origin model, 1 volley with none).
- Against the run without `--court` at the same commit: `config` and the block differ, and
  2459 rows moved under 0.5 px plus one by 8.2 px, with presence, fill (2559), tracks (48) and
  the chosen track identical: the detector's noise again, the court stages only read the
  selection.
- `evaluate_archive.py --mode pipeline --court --clips video13` prints its detector line
  (recall 0.933, wrong 0.003, med 2.4 px) and 7 confirmed bounces, 5 legs.
- The reference run for 1d: `evaluate_archive.py --archive finetune --mode pipeline --court`
  over the 67 reviewed clips at commit 8c8cb6d (`output/court_events_1d/pipeline_1b.log` and
  `pipeline_1b/`): pooled recall 0.889, wrong 0.006, false alarm 0.216, med 1.7 px, p90 5.2 px
  (the October 21 baseline was 0.890 / 0.006 / 0.216 / 1.7 / 5.2); 277 confirmed bounces
  (319 candidates, 194 kinematic), 283 contacts, 240 legs (10 inferred, 5 volleys) in all.
  Of the 277 confirmed landings 274 sit on a frame whose selected point is the measurement
  itself and 3 on recovered frames, although only 1300 of the 27007 filled frames are
  unrefitted: the refit leaves the frame at a landing where the detector put it.

## 1c: landing labels and the scorer (8c8cb6d)

No landing label exists yet; `score_court_events.py` against the workspace reports so and exits 1.
The user labels Pomona, grid_match4, grid_match22, grid_match16 or grid_match24, and video6 or
video8 (see the task report for the file format).

## 1d1: events placed at the detector's measurement (7e45e1d), kept

`ball_point` returns the measurement when the frame has one. Selector untouched, so the
frames are identical by construction. Judged by recomputing events from the stored runs'
frames with `audit/records/court_events_recompute_diff_2026-10.py` (which reruns the stages
through `evaluation.court_events_score.recompute_block`):

- the 67 reviewed clips (`output/court_events_1d1/reviewed_recompute_vs_1b.txt`): no change
  at all, 0 of 67 clips differ, every one of the 277 confirmed landings on its frame and at
  its position (see the 274-of-277 note above: the landing frames were already measurements);
- the VolyAvw sample (`output/court_events_1d1/volyavw_recompute_vs_1b.txt`): the 12
  confirmed landings keep their frames; 5 sit at another position (the refit's correction,
  up to its bound); one more model candidate (frame 1450, score 0.37: at its measurement
  0.7 px below player 13's padded box, inside at the refitted point); one inferred leg
  (471 to 522) lost, its landing at court v 0.46 having moved into the net band (0.47 to
  0.53) where legs are not formed; leg (2944 to 2973) now reads as a serve (its end moved
  0.019 court units into the service box). Knife-edge cases of the position shift.

## 1d2: ball turns judged on measurements (0f6e543), decision pending

`sample_xy` feeds the velocities, the gap fit, the bounce features, the prominences and the
contact helpers. Frames identical by construction (the stages only read the selection), so
the detector metric of the reference run stands unchanged; the recompute uses its frames.
No landing labels existed when this was done, so the keep-or-revert rule of the plan (pooled
recall not down, precision down by at most 0.02, read per clip) could not be applied; the
counts are:

- the 67 reviewed clips (`output/court_events_1d2/reviewed_recompute_vs_1b.txt`), 53 of 67
  clips differ: 285 confirmed landings instead of 277 (347 candidates instead of 319, 192
  kinematic instead of 194), 332 contacts instead of 283, 249 legs instead of 240 (7 inferred
  instead of 10, 6 volleys instead of 5). Matched within 0.1 s: 212 landings on the same
  frame, 57 on another frame (the bounce features read the refitted neighbours before, the
  measurements now, so the model's peak moves a frame or two); 8 landings of the reference
  run are gone and 16 are new.
- the sample (`output/court_events_1d2/volyavw_recompute_vs_1b.txt`): 14 confirmed instead of
  12 (25 candidates, 12 kinematic), 22 contacts instead of 19, 14 legs; 8 landings on the
  same frame, 4 shifted, 2 new, none lost.

To decide: run the reference folder through `score_court_events.py --archive finetune
--out-dir output/court_events_1d/pipeline_1b` once the landing labels exist (the stored
blocks are the 1b/1d1 events), then the same with `--recompute --compare before.log` for 1d2,
and keep 1d2 only if pooled recall does not fall and precision falls by at most 0.02; else
revert 0f6e543 and record the numbers in `bounce_events.py`'s docstring.
