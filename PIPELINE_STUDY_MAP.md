# Triton Tennis Pipeline Study Map

This document maps the project from video input to final annotated output, then lists the concepts, math, libraries, and vocabulary worth researching so you can explain the system end to end.

## One-Sentence Summary

The project reads a tennis video, enhances moving ball-like pixels, runs TensorRT YOLO models for ball/player/court detection, builds many candidate ball tracks, scores and stitches the best in-play timeline, fills short gaps with motion and physics, then renders an annotated video and optional JSON.

## Runtime Pipeline Map

```text
now_main.py
  -> tennis_tracker.cli
    -> parse CLI flags into Config
    -> tennis_tracker.pipeline.run(cfg)

pipeline.run(cfg)
  1. Detect platform capabilities
     - CUDA, TensorRT, NVENC, FFmpeg

  2. Load models
     - ball.engine through TensorRTRuntimeBallDetector
     - player.engine through PlayerDetector
     - courtdetection.engine through CourtDetector

  3. Open video
     - OpenCV VideoCapture
     - ThreadedFrameReader prefetches frames

  4. Pass 1: collect evidence
     - maintain HSV/S+V background model
     - compute raw motion masks
     - filter masks into ball-sized boost blobs
     - brighten motion blobs and dim static regions before YOLO
     - run ball YOLO, player tracker, court detector
     - update ROI tracker so future motion search is local
     - store detections, masks, player boxes, court keypoints

  5. Ball-in-play selector
     - convert YOLO boxes to Detection objects
     - build continuous motion tracks from masks
     - build candidate ball tracks from detections
     - Kalman/projectile prediction supports association
     - score tracks by length, court position, motion, speed, extent, player context
     - blacklist stationary, sideline, and zero-motion tracks
     - choose a timeline of track fragments with dynamic programming
     - produce per-frame ball result: det, motion, guide, carry, interp, or lost

  6. Pass 2: render outputs
     - reopen or reuse cached frames
     - draw ball trail, detections, search regions, players, court, debug masks
     - write MP4 through NVENC, libx264, or OpenCV
     - optionally write tracking JSON

Optional sidecar:
  3DtrackingV1/tools
    -> consumes tracking JSON
    -> estimates camera/court geometry
    -> fits 3D ballistic arcs and event candidates
```

## Main Files

| Area | File | What to know |
|---|---|---|
| Entrypoint | `now_main.py` | Tiny wrapper that calls the CLI. |
| CLI/config | `tennis_tracker/cli.py`, `tennis_tracker/config.py` | Defines all command-line flags and runtime knobs. |
| Main runtime | `tennis_tracker/pipeline.py` | The orchestration layer. This is the best file to explain first. |
| Neural detectors | `tennis_tracker/detectors.py` | TensorRT engine loading, tensor preprocessing, YOLO output decoding, NMS. |
| Motion masks | `tennis_tracker/motion.py` | HSV/S+V motion detection, connected components, blob filtering, preprocessing. |
| ROI tracker | `tennis_tracker/tracking.py` | Keeps motion search local around predicted ball positions. |
| Rendering/video | `tennis_tracker/rendering.py`, `tennis_tracker/video_io.py` | Draws overlays and writes videos with FFmpeg/NVENC/OpenCV. |
| Selector core | `ball_in_play_selector/core.py` | Final per-frame ball choice logic. Densest part of the project. |
| Track building | `ball_in_play_selector/tracking.py` | Converts detections/masks into candidate tracks and guides. |
| Track scoring | `ball_in_play_selector/scoring.py` | Scores tracks and chooses/stitches the best timeline. |
| Physics | `ball_in_play_selector/physics.py` | Projectile prediction and Kalman filter. |
| Geometry helpers | `ball_in_play_selector/utils.py` | FPS scaling, homography, pixel-per-meter estimates. |
| 3D sidecar | `3DtrackingV1/tools/*` | Archived/optional 3D reconstruction experiments, not the normal runtime. |

## Concepts and Research Table

| Topic | Used where | What it does here | Core math or idea | Research terms |
|---|---|---|---|---|
| Python dataclasses | Config/model files | Stores settings and data records. | Structured records with typed fields. | Python dataclass |
| OpenCV video IO | `pipeline.py`, `video_io.py` | Reads frames and writes fallback video. | Frame as H x W x 3 BGR array. | OpenCV VideoCapture, VideoWriter |
| FFmpeg | `video_io.py` | Encodes MP4 output. | Raw BGR frames piped to encoder. | FFmpeg rawvideo, libx264 |
| NVENC | `video_io.py` | Hardware H.264 encoding on NVIDIA GPU. | GPU video encoder, not neural inference. | NVIDIA NVENC |
| NumPy arrays | Everywhere | CPU image/mask/math arrays. | Vectorized array operations. | NumPy ndarray |
| PyTorch tensors | `motion.py`, `detectors.py`, `video_io.py` | GPU image preprocessing and TensorRT buffers. | Multi-dimensional arrays on GPU. | PyTorch tensor, CUDA tensor |
| CUDA | `pipeline.py`, `motion.py`, `detectors.py` | Runs tensor operations and TensorRT inference on NVIDIA GPU. | Parallel GPU kernels. | CUDA streams, GPU memory |
| TensorRT | `detectors.py` | Runs compiled YOLO `.engine` models fast. | Optimized inference graph with fixed bindings. | NVIDIA TensorRT engine, bindings |
| ONNX | `models/*.onnx`, build script | Intermediate model format before TensorRT engine. | Portable neural network graph. | ONNX export |
| YOLO | Ball/player/court models | Detects ball boxes, player boxes, court keypoints. | CNN object detector outputs boxes and confidence. | YOLO, object detection |
| Confidence threshold | Detectors | Drops weak detections. | Keep if `conf >= threshold`. | detection confidence |
| Bounding boxes | Detectors/selector | Represents ball/player locations. | `[x1, y1, x2, y2]`, center, area. | bounding box coordinates |
| NMS | `detectors.py`, `motion.py` | Removes duplicate overlapping detections. | Keep high-score box, drop boxes with high IoU. | non maximum suppression, IoU |
| IoU | NMS/BallTracker | Measures box overlap. | `intersection / union`. | Intersection over Union |
| HSV color space | `motion.py` | Separates hue, saturation, brightness for motion/color tests. | Transform BGR to hue/saturation/value. | HSV color model |
| S+V motion detection | `motion.py` | Finds pixels whose saturation or brightness changed. | Compare `(current - background)^2` to threshold plus variance. | background subtraction |
| Weighted temporal average | `pipeline.py` | Maintains moving background and variance. | `bg = (1-a)bg + a*current`. | exponential moving average |
| Variance thresholding | `motion.py` | Avoids flagging noisy static pixels as motion. | Threshold grows with local variance. | adaptive background model |
| Frame delta gate | `motion.py` | Requires proof of frame-to-frame movement. | `max(gray diff, V diff, scaled S diff)`. | temporal differencing |
| Morphology | `motion.py` | Cleans masks. | Open/dilate/close binary masks with kernels. | erosion, dilation, morphological opening |
| Connected components | `motion.py` | Converts motion pixels into blobs. | Label contiguous foreground regions. | connectedComponentsWithStats |
| Contours and moments | `motion.py`, selector | Computes blob centers and shape. | `cx = M10 / M00`, `cy = M01 / M00`. | image moments, contour area |
| Blob filtering | `motion.py` | Keeps ball-like motion. | Area, aspect ratio, fill ratio, perspective-scaled size. | blob detection |
| ROI tracking | `tracking.py` | Limits motion work to where ball is expected. | Search box around predicted center. | region of interest tracking |
| Pixel velocity | selector/scoring | Estimates ball motion between frames. | `vx = dx/dt`, `vy = dy/dt`, `speed = sqrt(vx^2 + vy^2)`. | kinematics |
| Projectile prediction | `physics.py` | Predicts ball during gaps. | `x += vx`, `y += vy`, `vx *= drag`, `vy = vy*drag + g`. | projectile motion, drag |
| FPS normalization | `config.py`, `utils.py` | Keeps thresholds similar across frame rates. | Convert 30fps px/frame values by `30 / fps`. | frame-rate normalization |
| Frame diagonal scaling | Config/selector | Makes thresholds resolution-independent. | `diag = sqrt(width^2 + height^2)`. | normalized pixel thresholds |
| Kalman filter | `physics.py`, selector | Smoothly estimates `[x,y,vx,vy]` and uncertainty. | Predict with state transition, update with measurement. | Kalman filter |
| Covariance | Kalman filter | Tracks uncertainty in position/velocity. | Larger covariance means wider search. | covariance matrix |
| Mahalanobis distance | `physics.py`, selector gates | Detects outlier measurements. | `d^2 = r^T S^-1 r`. | Mahalanobis gating |
| Chi-square gate | `physics.py` | Soft/hard outlier thresholds. | 2D thresholds around 9.21 and 13.82. | chi-square distribution |
| Bounce heuristic | `physics.py` | Handles downward ball suddenly appearing higher. | Reflect vertical velocity by restitution. | coefficient of restitution |
| Homography | `utils.py`, rendering, 3D sidecar | Maps camera pixels to normalized court space. | 3x3 projective transform. | perspective transform, homography |
| Court polygon | rendering/scoring | Scores whether detections are in or near court. | Point-in-polygon signed distance. | `cv2.pointPolygonTest` |
| Player context | selector/scoring | Helps distinguish serve/hit context from background noise. | Distance from ball to expanded player boxes. | contextual tracking |
| ByteTrack | `detectors.py` | Tracks player boxes over frames. | Associates detections into persistent IDs. | ByteTrack, multi object tracking |
| Track association | `ball_in_play_selector/tracking.py` | Links detections into candidate ball tracks. | Greedy assignment by prediction distance and gates. | data association |
| Speed ratio gate | selector/tracking | Rejects implausible jumps. | Observed speed divided by expected speed. | motion gating |
| Direction cosine | selector/tracking | Compares directions. | `cos(theta)=dot(v1,v2)/(|v1||v2|)`. | cosine similarity |
| Acceleration gate | selector/tracking | Rejects abrupt velocity changes. | `sqrt((dvx)^2 + (dvy)^2)`. | acceleration |
| Track scoring | `scoring.py` | Picks likely in-play ball path. | Weighted score using observations, span, court, motion, speed, extent. | scoring function |
| Stationary rejection | `scoring.py`, `core.py` | Drops parked balls. | Low avg speed, low peak speed, small extent, low motion. | static object filtering |
| Timeline stitching | `scoring.py` | Joins fragments into one rally path. | Dynamic programming over track segments. | dynamic programming |
| Guide path | `tracking.py`, `core.py` | Per-frame hint from chosen track, with filled gaps. | Exact observations plus KF/linear fill. | trajectory guide |
| Motion fallback | `core.py` | Uses blobs when YOLO misses. | Search near predicted/guide position, then physics-gate blob. | detection fallback |
| Carry | `core.py` | Short predicted continuation when no detection/motion. | Physics prediction without measurement. | dead reckoning |
| Interpolation | `core.py`, `tracking.py` | Fills short gaps between real detections. | Linear or 3-point Lagrange quadratic. | interpolation, Lagrange polynomial |
| Output JSON | `pipeline.py` | Saves per-frame ball, source, search, players, court. | Structured audit trail. | JSON schema |
| 3D PnP | `3DtrackingV1/tools` | Estimates camera pose from court points. | Solve camera pose from 3D court template to 2D points. | solvePnP, camera extrinsics |
| Camera intrinsics | `3DtrackingV1/tools` | Optional calibrated camera matrix and distortion. | Pinhole camera matrix `K` plus radial/tangential distortion. | camera calibration |
| Reprojection error | 3D sidecar | Measures 3D fit quality in pixels. | Project 3D point to image, compare to observed 2D. | reprojection error |
| Least squares | 3D sidecar | Fits 3D ballistic arcs. | Minimize sum of residuals. | scipy least_squares |

## Tensor and TensorRT Primer

| Word | Meaning in this project | Plain-English explanation |
|---|---|---|
| Tensor | A multi-dimensional numeric array, often on GPU. | An image becomes numbers shaped like `[batch, channels, height, width]`. |
| CHW / HWC | Image memory layout. | OpenCV frames are HWC BGR. Neural models usually want CHW RGB. |
| FP32 / FP16 | 32-bit vs 16-bit floating point. | FP16 is less precise but much faster on NVIDIA GPUs. |
| CUDA stream | Ordered queue of GPU work. | Lets preprocessing/inference overlap with CPU work. |
| TensorRT engine | Compiled neural network optimized for one GPU/runtime. | Instead of running a generic PyTorch model, the project runs a pre-optimized `.engine`. |
| Binding | TensorRT input/output memory slot. | The code gives TensorRT GPU memory addresses for input and output tensors. |
| Async inference | Start inference now, collect result later. | The project launches current-frame inference and finishes the previous one to overlap work. |
| NMS output | Final filtered boxes. | Some engines output already-filtered boxes; others output raw predictions and the code runs NMS. |

## Model Assets

| File | Role |
|---|---|
| `models/ball.pt` | Original/training-time PyTorch YOLO ball model. |
| `models/ball.onnx` | Exported neutral graph format. |
| `models/ball.engine` | TensorRT-compiled ball detector used by runtime. |
| `models/player.engine` | TensorRT player detector. |
| `models/courtdetection.engine` | TensorRT court/keypoint detector. |

## What To Explain In Order

1. The system is not just YOLO. YOLO proposes detections, but the selector decides the actual in-play ball.
2. Motion preprocessing improves YOLO input and provides fallback evidence when YOLO misses.
3. Motion is determined by adaptive HSV/S+V background differencing, then cleaned into ball-sized blobs.
4. Ball movement is measured with pixel kinematics: displacement per frame, speed, acceleration, direction.
5. Prediction uses projectile physics and a Kalman filter to survive gaps.
6. Track scoring rejects stationary balls and background detections using court, player, motion, and speed context.
7. Timeline stitching joins the best fragments into one rally path.
8. Rendering is a second pass because the selector needs whole-clip context before final drawing.
9. TensorRT is the fast inference runtime for the YOLO engines; tensors are the GPU arrays passed into it.
10. The 3D folder is optional/archived sidecar work that consumes the tracking JSON for camera and trajectory experiments.

## Suggested Study Order

| Order | Learn this | Why |
|---|---|---|
| 1 | Image arrays, pixels, BGR vs RGB, HSV | Everything starts as video frames. |
| 2 | Bounding boxes, confidence, NMS, IoU | Needed to explain YOLO outputs. |
| 3 | Background subtraction and morphology | Needed to explain motion masks. |
| 4 | Contours, moments, connected components | Needed to explain ball-like motion blobs. |
| 5 | Kinematics: velocity, acceleration, direction cosine | Needed to explain track association. |
| 6 | Kalman filter basics | Needed to explain prediction and uncertainty. |
| 7 | Projectile motion with gravity/drag | Needed to explain carry and search positions. |
| 8 | Homography and court geometry | Needed to explain court-aware filtering. |
| 9 | Tensor, CUDA, TensorRT, FP16 | Needed to explain speed/acceleration. |
| 10 | Dynamic programming | Needed to explain timeline stitching. |
| 11 | Camera calibration/PnP/reprojection | Only needed for the optional 3D sidecar. |
