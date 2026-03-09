# Improve Later

## Model Improvements
- Improve ball model quality (data, training, robustness).
- Improve player/court model quality where it helps selector stability.
- Rebuild/benchmark TensorRT engines after model upgrades.

## Selector Behavior
- Bias fallback toward `motion` (orange) more often when both `motion` and `carry` are plausible.
- Reduce overuse of `carry` (blue) in short ambiguous gaps.
- Add explicit diagnostics for why motion was rejected on each frame.

## Motion Detection
- Improve isolated motion detection (small/brief blobs, thin trajectories, contact-zone cases).
- Improve motion near players without reintroducing false snaps.
- Revisit motion gate thresholds after model updates.

## Deepsearch / DeepStream Path
- Evaluate Deepsearch/DeepStream migration scope and expected FPS gain.
- Define phased port plan (inference first, then preprocess/selector).
- Keep parity tests against current Python pipeline before full switch.

