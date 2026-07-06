# Benchmark Harness

One command to measure tracking quality + speed across every registered clip.
This is the gate for promoting any pipeline change: run it before and after,
and only keep changes the scoreboard justifies.

```powershell
# Full run (pipeline + validation for every enabled clip)
python eval/benchmark.py

# Re-score existing tracking JSONs without re-running the pipeline
python eval/benchmark.py --skip-pipeline

# Diff against the previous result and fail on any regression
python eval/benchmark.py --compare-latest --fail-on-regression
```

Results land in `eval/results/benchmark__<git_sha>.json`. Per-clip reports keep
their normal home in `validation/reports/`.

## Adding a clip

1. Run the pipeline once on the new clip with `--tracking-json`.
2. Pre-fill + correct labels (~80 frames, 10-15 min):
   `3DtrackingV1/archived_tools/extract_label_frames.py` then `label_assist.py`
   (see `validation/README.md`).
3. Register the clip in `eval/clips.json` with thresholds.

Coverage matters more than density: one ~80-frame label set per distinct
court/lighting/camera setup tells you more about generalization than another
500 labels on the same clip.
