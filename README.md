# Clean tracker

This folder is an isolated alternative tracker. It contains its own runtime
packages, TensorRT engines, validator, corrected sample labels, and sample clip.
It does not import code from the parent tracker.

Run from the repository root with the `tennis-analysis` environment:

```powershell
$py = 'C:\Users\Andrew\.conda\envs\tennis-analysis\python.exe'

& $py clean_version\clean_tracker.py --self-test

& $py clean_version\check_parity.py

& $py clean_version\clean_tracker.py `
  --annotations clean_version\sample\pomona_annotations.json

& $py clean_version\check_parity.py --existing
```

`check_parity.py` runs a fresh no-video inference before applying strict gates.
Use `--existing` for a quick recheck of the current JSON. `clean_tracker.py`
writes `clean_version/output/tracking.mp4`; its `--annotations` option produces a
general report, while the bundled Pomona pass/fail decision belongs to the strict
checker.

The selector is intentionally four stages: build tracks, keep motion-backed
tracks, fill established gaps, and extend a tail only while the current frame
still has motion evidence. Motion may correct a prediction by at most about 13 px and is blended at
75%; that correction is smoothed and decayed across frames so it cannot snap
back when a blob disappears. The main trail is green for detector positions,
orange for motion corrections, and cyan for physics fills. The strict checker
freezes the sample/label hashes and rejects incomplete, non-finite, inaccurate,
or temporally jerky output.

## Frozen result

| Metric | Existing tracker | Clean tracker |
|---|---:|---:|
| Visible recall | 76/76 | 76/76 |
| Mean error | 1.824 px | 1.571 px |
| p90 error | 2.862 px | 2.452 px |
| Maximum error | 20.417 px | 19.851 px |
| Within 5 px | 75/76 | 75/76 |
| Absence false positives | 3/4 | 3/4 |
| Large jumps | 1 | 0 |
| Acceleration p95 | 5.118 px/frame² | 4.284 px/frame² |
| Soft-point acceleration p95 | 7.674 px/frame² | 2.707 px/frame² |

The clean output contains 712 predicted frames versus 687 previously. Only four
absence labels exist, so the extra unlabeled coverage must be checked on more
matches before claiming broader precision improvement.
