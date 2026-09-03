# Repository Guidelines

## Project Structure & Module Organization

`clean_tracker.py` is the main CLI and coordinates the full video pipeline. Core detection, motion recovery, rendering, tracking, and video I/O live in `tennis_tracker/`. Ball-track construction, scoring, and physics recovery live in `ball_in_play_selector/`. TensorRT engines are stored in `models/`; labeled fixtures and sample videos are in `sample/`; demo media belongs in `media/`. Generated videos, JSON, and reports go in `output/`, which is intentionally ignored by Git.

## Build, Test, and Development Commands

- `python clean_tracker.py` runs the bundled Pomona sample and writes artifacts under `output/`.
- `python clean_tracker.py --input path\to\match.mp4 --no-video` processes another match without rendering an MP4.
- `python clean_tracker.py --self-test` runs fast internal checks for tracking logic.
- `python check_parity.py` runs the frozen Pomona regression benchmark; use `--existing` to validate an existing `output/tracking.json` without rerunning inference.
- `python validate_tracking.py --predictions output\tracking.json --annotations sample\pomona_annotations.json` evaluates tracking output against labels.
- `python build_trt_engines.py` rebuilds platform-specific engines from matching `.pt` checkpoints when changing GPU or TensorRT versions.

Python 3.10, CUDA-enabled PyTorch, TensorRT 10, OpenCV, and NumPy are expected. FFmpeg is recommended for faster encoding.

## Coding Style & Naming Conventions

Use four-space indentation and standard Python conventions: `snake_case` for functions and variables, `PascalCase` for classes, and uppercase names for constants. Keep configuration in the existing dataclasses (`Config` and `SelectorConfig`) and reuse package helpers before adding new abstractions. Preserve type hints on public boundaries. No formatter or linter is configured, so match nearby code and keep imports grouped as standard library, third-party, then local modules.

## Testing Guidelines

There is no separate test framework or coverage threshold. Add the smallest deterministic self-test for new non-trivial logic, then run `--self-test` and `check_parity.py`. Changes to tracking quality should include benchmark metrics; changes to validation logic should be checked against the labeled sample. Do not commit generated `output/` files.

## Commit & Pull Request Guidelines

History uses short, plain-English subjects without Conventional Commit prefixes, such as `changing demo video`. Keep each commit focused. Pull requests should explain the behavior change, list validation commands and results, and call out GPU/TensorRT assumptions. Link relevant issues, and include an updated screenshot, GIF, or short clip when overlay or rendering behavior changes.
