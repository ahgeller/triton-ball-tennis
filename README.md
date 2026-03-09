# Triton Tennis - Refactored Pipeline

The ball tracking pipeline has been refactored into smaller, more modular packages to improve maintainability while preserving the exact same functionality.

## How to Run

You can run the main pipeline exactly as before using the `now_main.py` entry point wrapper:

```bash
conda activate tennis-analysis
python now_main.py --input_video input_videos/TMP.mp4 --output_video output_videos/test.mp4
```

To see all available command-line arguments (which dynamically map to the settings below):
```bash
python now_main.py --help
```

## Settings and Configuration

The application is controlled by two main configuration objects. You can pass most of these via the command line or modify them programmatically.

### Main Pipeline Configuration (`now_main_pkg.config.Config`)

- **Paths:**
  - `input_video` / `output_video`: Input file and output render locations.
  - `model_path`: YOLO engine path for ball detection (`models/ball.engine`).
  - `player_model_path`, `court_model_path`: Engines for player/court tracking.
- **Ball Detector:**
  - `conf`: Confidence threshold for ball detection (default: 0.26).
  - `ball_backend`, `device`: Inference backend (e.g., `trt`, `auto`).
- **TensorRT Execution:**
  - `use_tensorrt`: Enable TensorRT support.
  - `tensorrt_half`: Enable FP16 logic inference.
  - `trt_async_execute`, `trt_async_slots`: Controls async TRT execution throughput.
- **Player & Court Detection:**
  - `player_detect_interval`, `court_detect_interval`: How often to re-detect to save GPU cycles.
  - `player_conf`, `court_conf`: Confidence thresholds.
- **Pre-processing:**
  - `enable_preprocess`: Enables HSV space modifications (saturation/value boosts) to highlight the tennis ball.
  - `motion_dilate`: Dilation kernel size for motion masking.
- **Motion Detection Analytics:**
  - `roi_motion_enabled`: Limit motion detection to cropped Regions of Interest (ROI) around the round's last known ball location.
  - `motion_thresh`, `motion_v_min`: Value thresholds for HSV background subtraction.
  - `boost_max_blob_area`, `blob_shape_filter`: Filters out implausibly large or distorted blobs.
- **Performance / Encoding:**
  - `use_nvenc`: Hardware-accelerated HVENC encoding.
  - `nvenc_preset`, `nvenc_bitrate`: Output quality adjustments.
  - `cache_input_frames_pass2`: Speeds up visual rendering by caching original frames in RAM.

### Ball-in-Play Selector Config (`ball_in_play_selector.config.SelectorConfig`)

This configuration dictates the heuristic scoring, kalman-filter tracking, and trajectory building for selecting the actual ball tracks vs surrounding background noise.

- **Association & Tracking:**
  - `assoc_gate_growth_cap_frames`, `assoc_min_speed_for_dir`: Match tracking bounds for Kalman Filter prediction distances.
  - `trail_stitch_enable`: Reconnects detached ball track trail segments algorithmically.
- **Trajectory Scoring Weights:**
  - `w_len`, `w_gaps`, `w_miss`: Core track continuity rewards/penalties.
  - `w_outside`, `w_inside`: Rewards trajectories that land generally inside the court.
  - `w_speed`, `w_peak_speed`: Validates fast tennis ball speeds vs slow static movement.
  - `w_player_court_joint`: Strong reward for the ball being realistically close to a player.
- **Motion Kinematics Fallback:**
  - `motion_use_min_det_speed`, `motion_search_vel_mult`: Adjusts motion threshold requirements dynamically based on ball speed momentum.
- **Physics Gravity Model:**
  - `gravity_enabled`, `gravity_base_ppf2_30fps`: Configures parabolic arc mapping to bridge tracking gaps over multiple missing observer frames.
  - `bounce_detect_min_down_speed`: Detects bounces by catching sudden upward velocity reversals.
  - `bounce_restitution`: Energy kept after an intense collision bounce.