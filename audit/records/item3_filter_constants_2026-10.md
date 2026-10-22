# Item 3: per-frame constants in the selector and the ROI tracker (October 2026)

The review found that the selector's Kalman, refit, kink and ROI constants are per-frame pixels,
so 60 FPS footage runs a different tracker than 30 FPS footage. Fix 6b (scaling the Kalman
bounce prior) lost twice on the reviewed clips, so this item measures first and changes one
constant at a time, each with its own before and after on the 67 reviewed clips
(`evaluate_archive.py --archive finetune --mode pipeline`, compared with
`evaluation/compare_eval_logs.py` and `track_breaks.py --compare`).

Reference run: `output/fix1_no_court/pipeline` (main at 7695e5e, pooled recall 0.890, wrong
0.006, false alarm 0.216). Every run below is a fresh folder under `output/`.

## Where the misses are (reference run, `track_breaks.py`)

| group | labelled visible | missed | no track observed it | only an unchosen track | chosen track, pruned | switches | longest gap |
|---|---|---|---|---|---|---|---|
| 30 FPS clips | 13415 | 1186 (8.8 %) | 852 | 328 (28 % of misses) | 6 | 400 | 67 |
| 60 FPS clips | 7434 | 529 (7.1 %) | 285 | 241 (46 % of misses) | 3 | 68 | 111 |
| all 67 clips | 20849 | 1715 (8.2 %) | 1137 | 569 | 9 | 468 | 111 |

Reading: two thirds of the misses are frames no selector track ever observed, which is detector
recall (the raw detector misses 8.9 % of labelled balls) and not a selector constant. One third
are frames a track observed that the timeline did not choose, and that share is 46 % at 60 FPS
against 28 % at 30: the association fragments more at 60 FPS. The refit and the fills lose 9
frames in 20849, so the refit window, Huber bound and fill constants are not worth a run.

Worst clips by misses: Pomona 159 of 1575, grid_match21 128 of 496, video41 123 of 514, video39
118 of 553, video40 106 of 573 (all five mostly "no track"), grid_match31 93 of 228 (longest gap
111 frames at 60 FPS).

## Instrumentation (commits dd77fb8, a8049a5)

Every track now records the gate that turned its first detection away from the nearest live
track (`start.reason`, `start.rival_track`), how it ended (`end_reason`), which of the four
selection tests admitted it (`selection`) and whether the timeline used it (`in_timeline`);
every detected output row names its track (`track_id`). Tracking JSON schema 4. Output-neutral
by the golden harness (only the new keys differ) and by the 67-clip run below.

## The 60 FPS phases (instrumentation run, scratch analysis)

At 60 FPS the prepass runs the even and the odd frames as two 30 FPS streams, each with its own
sub-pixel bias. In 28 % of 60 FPS selector tracks the mean odd-to-even step differs from the
even-to-odd step by more than 20 % (6 % at 30 FPS); consecutive single-frame step ratios spread
0.38 to 1.94 at 60 FPS against 0.59 to 1.52 at 30, and 2.2 % fall outside the speed-ratio bounds
against 0.9 %. So every other step looks fast or slow to the speed-ratio test, which is what the
start histogram shows. The candidate that follows is judging that ratio on same-phase steps.

## Runs

| run | change | pooled recall / wrong / false alarm | 60 FPS misses on unchosen tracks | decision |
|---|---|---|---|---|
| `output/item3b_roi_predict` | ROI tracker predicts the frame being assigned (b8763e6, a bug fix) | 0.890 / 0.006 / 0.215 (hits 19137, unchanged) | 240 (238) | kept: a correctness fix with no measurable effect; nine clips moved within noise, one false alarm fewer on grid_match16 |
| `output/item3c_gate_cap_sec` | association gate-growth cap 0.2 s instead of 6 frames (agent commit d2ef917; 30 FPS identical, 60 FPS cap 12) | 0.889 / 0.006 / 0.215 (hits 19122 vs 19137) | 245 (240) | reverted: no gain; 60 FPS hits down 5 and 30 FPS, where the code is identical, down 10, so the whole move is run-to-run noise (about 15 hits pooled) |
| `output/item3d_speed_floor` | speed-ratio floor scaled by resolution only (agent commit 9a77668, kept on branch item3-speed-floor-rejected): 2.5 px/frame at 1080p at any rate, 1.67 at 720p | 0.892 / 0.006 / 0.216 (hits 19171 vs 19137) | 228 (240); 60 FPS track starts 230 (275) | reverted by the per-clip rule: grid_match8 recall -0.055 (29 frames: the association merged tracks differently at 260 to 518 and the timeline dropped 600 to 655), grid_match1 +8 frames but 4 wrong with a 25 px p90 tail (a slow track no longer judged took a line), grid_match31 longest gap 123 (111). Net +34 hits is real but bought with wrong associations on slow segments; a jitter-aware ratio (compare speeds with the jitter added to both) is the next idea, not run |
| `output/item3e_roi_limits` | ROI tracker lost limit 0.27 s (8 frames at 30 FPS, 16 at 60) and velocity reset scaled by resolution and rate (agent commit 9331620, kept on branch item3-roi-limits-neutral) | 0.890 / 0.006 / 0.215 (hits 19137, unchanged) | 238 (240) | reverted: no effect; every bucket within four frames, per-clip moves only on the two clips that flip every run |
| `output/item3f_kalman_velocity` | Kalman velocity process noise, initial velocity variance and bootstrap cap scaled by frame_scale squared (agent commit 28b6dca, kept on branch item3-kalman-velocity-rejected); 30 FPS bit-identical | 0.885 / 0.006 / 0.218 (hits 19044 vs 19137) | 331 (240); longest gap 167 (111) | reverted: a clear loss, all of it at 60 FPS (hits -91: grid_match1 recall -0.246, video7 -0.134, grid_match31 -0.039, video3 -0.032). The speed-ratio starts did not move (184 vs 186), so the filter's velocity is not what trips that test at 60 FPS |
| `output/item3g_per_time_gates` | gate-growth cap 0.2 s plus every per-missed-frame gate growth scaled per unit time (agent commits d2ef917 and de7a76c, kept together on branch item3-per-time-gates-neutral); 30 FPS identical, 60 FPS gates as strict as 30 FPS over a gap of the same duration | 0.890 / 0.006 / 0.215 (hits 19139 vs 19137) | 241 (240) | reverted: no effect at either rate; the start histogram is unchanged (60 FPS speed_ratio 186) |
| `output/item3h_same_phase` | the speed-ratio test judges a 60 FPS detection against the last same-phase observation (agent commit bac2a75, on the branch as c9e3d1e); 30 FPS bit-identical | 0.898 / 0.006 / 0.230 (hits 19322 vs 19137) | 83 (240); 60 FPS speed_ratio starts 90 (186); longest gap 96 (111) | the one real gain: +181 hits at 60 FPS (grid_match1 recall +0.285, grid_match31 +0.198, grid_match24 +0.051, Pomona +0.021), 30 FPS +4. Costs: false alarm 0.215 to 0.230 (about 21 filled frames with no ball, the longer tracks' tails), 60 FPS switches 69 to 75, grid_match7 -5 frames. Kept on the branch; the trade is the user's decision |
| `output/item3a_instrumentation` | the record above, no behaviour change | 0.890 / 0.006 / 0.216 (hits 19137 vs 19134) | 238 (reference 241) | output-neutral: six clips moved within the GPU noise bracket, no bucket moved by more than 7 frames |

## Why tracks start (instrumentation run, `track_breaks.py`)

Of 576 tracks that started because a live track's gate turned their first detection away:

| gate | all | 30 FPS | 60 FPS |
|---|---|---|---|
| speed_ratio | 315 | 129 | 186 (68 % of 60 FPS starts) |
| stale_distance | 95 | 63 | 32 |
| first (no live track) | 77 | 58 | 19 |
| gate | 56 | 23 | 33 |
| jump | 23 | 21 | 2 |
| stale_prediction | 10 | 7 | 3 |

The speed-ratio test (`_implausibility`) dominates, and twice as much at 60 FPS: it engages once a
track moves faster than `assoc_min_speed_for_dir` (2.5 px/frame at 30 FPS), a floor scaled down by
frame rate to 1.25 px/frame at 60 FPS, inside detector jitter. Candidates, each its own run:
the floor scaled by resolution only; the gate-growth cap in seconds; the per-missed-frame gate
growths per unit time; the ROI tracker's lost limit and reset speed per unit time. The refit and
kink constants are not run: the breakdown gives them 9 frames.

A correction found while writing the growth change: the per-missed-frame gate growths gave 60 FPS
twice the growth of 30 FPS for a gap of the same duration (twice the frames, the same growth each),
not half. Scaling them per unit time makes 60 FPS as strict as 30 FPS; whether that helps is for
its run to say, since the histogram puts the 60 FPS splits on the speed-ratio floor, not on gate growth.


## Where this leaves item 3

Kept on `item3-filter-constants`: the analysis script, the instrumentation (schema 4), the ROI
predictor fix (neutral, a bug) and the same-phase speed judgement (the gain above, pending the
user's call on its false-alarm cost). Set aside, each on a branch named for it: the speed floor
(`item3-speed-floor-rejected`), the Kalman velocity scaling (`item3-kalman-velocity-rejected`), the
ROI limits (`item3-roi-limits-neutral`) and the per-time gates pair (`item3-per-time-gates-neutral`).
Not run, by the breakdown: the refit window and bounds, the kink detector, the fill constants.

What the measurements say about the review's premise: making 60 FPS constants consistent with 30 FPS
in seconds changed nothing (three neutral runs) or hurt (the Kalman run). The 60 FPS losses came from
the detector's two phase streams disagreeing by a sub-pixel bias, which no per-frame constant fixes;
judging speed within one stream did. The remaining 60 FPS cost is the tails of longer tracks, a
selector fill question, and the next thing to measure if the false-alarm rise matters.
