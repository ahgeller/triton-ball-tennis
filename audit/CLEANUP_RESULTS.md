# Disabled-feature cleanup — September 7, 2026

Requested after checkpoint commit `a1eccbe`. The earlier project review and fix
report are historical records; this document describes the subsequent cleanup.

## Removed

- Manual `court_depth`, `court_side`, `x_scale_strength`, `y_scale_strength`
  blob-size adjustments. They required per-camera assumptions, were disabled,
  and offered no verified batch-wide benefit.
- The disconnected court-homography gravity path and its unused geometry
  helpers. A ground-plane scale is not a calibrated airborne-ball model.
- Raw-motion color gating on CPU/CUDA and its tuning fields. The verified
  color-on comparison lost recall; the supported behavior now retains motion
  regardless of yellow appearance.
- Disabled flicker suppression, its per-frame retained mask and controls.
- The disabled raw-component appearance/size filter and controls. Active
  temporal evidence, morphology and the separate boost-mask filter remain.
  Debug component boxes retain fixed display-only limits.

These options are removed from Config, not merely hidden. Old code constructing
Config with these keywords must stop passing them. Historical tracking JSONs
remain readable; they correctly record settings used when they were produced.

## Kept, deliberately

Automatic resolution/FPS normalization is active and reproducible from video
metadata; it is necessary for running mixed-resolution, 30/60 FPS collections.
It is fundamentally different from manually guessing camera perspective.
The fixed GridTrackNet input size, RGB contract and decoding offset remain
unchanged. Court overlays and court-side exclusion geometry are actively used
and are distinct from the removed calibration experiment.

Optional diagnostics, debug videos, synchronous fallbacks and CPU support have
concrete troubleshooting/portability uses. Disabled-by-default alone is not a
reason to delete those. New architecture, camera-motion compensation or
calibration should enter as a separately measured experiment rather than an
unused collection of runtime knobs.

## Additional repair

`filter_boost_mask` could return stale foreground in a reused output buffer
when the current raw mask was empty. Both empty-result paths now clear the
buffer. A synthetic regression starts with a filled buffer and verifies the
empty result, preventing motion evidence from leaking between frames.

## Validation

Sixteen regression tests and the tracker self-test pass. The cleanup rerun
uses only video13–video53, under `output/verified_cleanup_20260907`.
`verify_cleanup.py` checks complete ordered frame exports, removed settings,
and every frame record against the earlier color-off baseline.
Results are recorded in `cleanup_verification.json` and
`verified_pipeline_cleanup.txt`.

The completed cleanup run reports **87.8%** localization recall, **0.8%**
wrong-object rate, **8.5%** misses, **23.9%** absent-frame false alarms, and
**2.8 / 6.6 px** median/p90 error. All 9,292 exported frames are ordered and
complete. The earlier color-off run rounded to 87.9% recall.

Exact frame records are not identical: 4,218 differ, largely through tiny
floating-point coordinate/confidence differences. Of 8,283 positions present
in both runs, only 22 differ by more than 0.1 native pixels and one by more than
one pixel (maximum 13.88 px). Five presence decisions changed on video41,
frames 215–219. An unchanged-code repeat restored all five with the earlier
positions. This establishes existing run-to-run sensitivity; it does not
establish exact output equivalence or a cleanup-induced accuracy improvement.
The buffer fix is covered synthetically; current production callers do not
pass the optional reuse buffer.

The detector enables timing-based cuDNN kernel autotuning. A separate trial
disabled benchmarking and enabled deterministic cuDNN kernels. At the first
20 completed matched clips it accumulated 218.1 seconds versus 70.8 seconds in
the autotuned cleanup run. One overlapping repeat job and other workload
conditions mean this is not a controlled speed benchmark. Nevertheless the
observed cost was too large to adopt without a dedicated performance study.
The trial was stopped after 22 clip exports; its log is explicitly partial,
and its kernel settings were reverted. No new configuration toggle was added.
`verified_pipeline_cleanup_stable.txt` and `cleanup_stable_repeat_video41.txt`
record that experiment, not the shipped default. Bitwise reproducibility across
GPUs, libraries and separate runs remains unproven. Future work should measure
which hard selector boundaries amplify tiny input changes.

Local source MP4s and label safety copies are ignored by Git, preserved on disk.
Previously tracked sample videos remain tracked. No label rewriting, model
training or model promotion was performed for this cleanup.
