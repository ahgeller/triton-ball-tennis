# Triton Ball Tennis

A GPU-accelerated tennis video analyzer that detects the court and players,
tracks the ball through fast or blurred motion, and exports an annotated video
with structured tracking data.

## Demo

[![Tennis tracking demo](media/tracking_demo.gif)](media/tracking_demo.mp4)

The overlay shows court keypoints, player boxes, and the selected ball trail:

- **Green:** detector positions
- **Orange:** motion corrections
- **Cyan:** bounded interpolation and physics recovery

## Features

- Tennis ball, player, and court detection from match video
- TensorRT inference with CUDA preprocessing
- Motion-assisted recovery when the detector loses the ball
- Ball-in-play track selection to reject static objects and false detections
- Annotated MP4 and per-frame tracking JSON output
- Automated accuracy, smoothness, and regression checks

## What makes it different

Many tennis-analyzer projects stop after drawing YOLO detections. This tracker
uses the detector as one signal inside a larger tracking system:

- **Better continuity:** motion, court position, player proximity, and trajectory
  physics help recover short detector gaps.
- **More efficient processing:** dynamic search regions, threaded frame reading,
  overlapped GPU inference, and asynchronous video encoding reduce unnecessary work.
- **More scalable code:** detection, motion, track selection, rendering, video I/O,
  and validation live in separate modules instead of one large script.
- **Measurable changes:** structured JSON and a frozen benchmark make model or
  tracking changes directly comparable.

## Requirements

- Python 3.10
- NVIDIA CUDA GPU
- PyTorch with CUDA
- TensorRT 10
- OpenCV and NumPy
- FFmpeg recommended for faster video encoding

The repository includes TensorRT engines for the ball, player, and court models.
Because TensorRT engines are platform-specific, they may need to be rebuilt for
a different GPU or TensorRT version.

## Usage

Run the bundled sample:

```powershell
python clean_tracker.py
```

Run another match:

```powershell
python clean_tracker.py --input path\to\match.mp4 --output output\tracking.mp4 --tracking-json output\tracking.json
```

Export JSON without rendering a video:

```powershell
python clean_tracker.py --input path\to\match.mp4 --tracking-json output\tracking.json --no-video
```

## Validation

```powershell
python clean_tracker.py --self-test
python check_parity.py
```

### Frozen benchmark

| Metric | Previous tracker | Current tracker |
|---|---:|---:|
| Visible recall | 76/76 | **76/76** |
| Mean error | 1.824 px | **1.571 px** |
| p90 error | 2.862 px | **2.452 px** |
| Large jumps | 1 | **0** |
| Acceleration p95 | 5.118 px/frame² | **4.284 px/frame²** |

These results are from the frozen Pomona sample. More labeled matches are needed
before treating them as general accuracy claims.

## Project structure

```text
clean_tracker.py              Main command-line entry point
tennis_tracker/               Detection, motion, rendering, and video pipeline
ball_in_play_selector/        Track construction, scoring, and physics recovery
models/                       Ball, player, and court TensorRT engines
sample/                       Sample videos and validation annotations
validate_tracking.py          General tracking evaluator
check_parity.py               Frozen regression benchmark
build_trt_engines.py          TensorRT engine builder
```

## Future implementation

- Improve and retrain the ball, player, and court models
- Evaluate alternative model types when they offer better accuracy or speed
- Add persistent player tracking across frames
- Add player identification so tracked players remain consistently labeled
