# Imports
import argparse
import copy
import glob
import json
import math
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from collections import OrderedDict, namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import scipy.interpolate
from ball_in_play_selector import select_ball_in_play, FrameResult, _predict_projectile, SelectorConfig
HAS_NMS = False
_nms = None
try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except Exception:
    torch = None
    F = None
    HAS_TORCH = False

try:
    from boxmot import ByteTrack
except ImportError:
    print("[warning] boxmot not found. Player tracking will be disabled. Run 'pip install boxmot'")
    ByteTrack = None



@dataclass
class Config:
    # paths
    input_video: str = r"input_videos/TMP.mp4"
    output_video: str = "output_videos/prof_test.mp4"
    model_path: str = r"models/ball.engine"
    
    # Ball detector
    ball_class_name: Optional[str] = None
    conf: float = 0.26
    ball_backend: str = "trt"
    device: str = "auto"

    # TensorRT
    use_tensorrt: bool = True
    tensorrt_half: bool = True
    trt_async_execute: bool = True
    trt_async_slots: int = 2

    # Court perspective
    court_depth: Optional[str] = None
    court_side: Optional[str] = None
    y_scale_strength: float = 0.35
    x_scale_strength: float = 0.15

    # Player & court detection
    player_model_path: Optional[str] = "models/player.engine"
    court_model_path: Optional[str] = "models/courtdetection.engine"
    player_detect_interval: int = 30
    player_detect_interval_stable: int = 60  # interval when players haven't moved much
    player_stable_thresh_frac: float = 0.02  # movement < this * diag = "stable"
    court_detect_interval: int = 500
    court_conf: float = 0.10
    print_court_raw: bool = False
    court_remap_semantic_14: bool = False
    court_points_only: bool = False
    court_draw_indices: bool = False
    player_conf: float = 0.21
    player_iou: float = 0.10
    player_bbox_pad: int = 10
    draw_players: bool = True
    draw_court: bool = True
    num_players: int = 4

    # Ball tracking
    ball_max_jump: float = 0.25
    ball_iou_weight: float = 0.5
    ball_dist_weight: float = 0.3

    # preprocessing
    enable_preprocess: bool = True
    pre_sat_boost: float = 1.35
    pre_val_boost: float = 1.10
    pre_hue_shift: float = 0.4
    dim_static: float = 0.88
    static_sat_scale: float = 0.75
    motion_dilate: int = 5
    
    # motion detection - IMPROVED for better ball continuity
    wta_alpha: float = 0.02              # Lowered so slow-moving balls aren't absorbed into the background
    motion_thresh: float = 9.0           # Lower base threshold to catch subtle/slow movements
    motion_k_std: float = 2.5            # Lower std-dev multiplier so dimmer/slower balls are caught
    motion_v_min: float = 40.0           # Lowered to catch dimmer ball motion in shadows
    motion_temporal_soft: bool = False   # Disabled: this was causing the "lingering" yellow artifacts you noticed
    motion_temporal_lo_frac: float = 0.45
    motion_temporal_hi_mult: float = 1.25
    motion_flicker_suppress: bool = False # Disabled to prevent motion ghosts from lingering
    motion_flicker_min_area: int = 2
    motion_flicker_max_area: int = 350
    motion_flicker_prev_dilate: int = 12
    motion_flicker_keep_radius_frac: float = 0.14
    boost_max_blob_area: int = 1200
    boost_min_blob_area: int = 0
    
    # Ball persistence hysteresis settings
    ball_persist_frames: int = 20
    ball_confidence_decay: float = 0.85
    ball_reacquire_window: int = 15
    ball_min_confidence: float = 0.15

    # Speed: skip-frame ball YOLO (run every Nth frame, interpolate gaps)
    skip_frame_yolo: int = 1          # 1 = every frame
    skip_frame_require_roi: bool = True
    skip_preprocess_dim: bool = True
    aux_detect_on_yolo_frames: bool = True
    aux_force_interval: int = 6
    frame_reader_prefetch: int = 8

    # ROI-based motion — only compute motion/CC near ball
    roi_motion_enabled: bool = True
    roi_visible_radius_frac: float = 0.012   # ROI radius as frame diag frac when ball is visible
    roi_lost_radius_frac: float = 0.01      # ROI radius when ball is lost
    roi_lost_expand_per_frame: float = 0.0005 # grow lost ROI by this * diag per lost frame
    roi_max_radius_frac: float = 0.06       # cap ROI radius
    roi_motion_bleed_frac: float = 0.0       # extend the underlying cropping boundary for motion processing to capture long blurred streaks without affecting the tight visual box
    roi_fullframe_interval: int = 0         # 0 = never do full-frame fallback; N = every N frames

    # blob shape filtering - IMPROVED for small tennis balls
    blob_shape_filter: bool = True
    blob_erode_size: int = 4              # LOWERED from 3 to preserve small ball blobs
    blob_max_aspect: float = 5.0          # INCREASED from 4.0 for elongated motion blur
    blob_preserve_tiny: bool = True       # NEW: preserve very small ball-sized blobs
    blob_tiny_max_area: int = 120         # NEW: max area for "tiny" ball blobs to preserve

    # debug
    save_motion_debug: bool = False
    output_debug_path: str = "output_videos/prof_test_motion_debug.mp4"
    output_yolo_input_debug_path: str = "output_videos/prof_test_yolo_input_debug.mp4"
    debug_show_raw_motion: bool = False
    save_guide_video: bool = False
    output_guide_path: str = "output_videos/prof_test_guide_debug.mp4"
    guide_interp_max_gap: int = 12
    print_selector_tracks: bool = True
    selector_track_limit: int = 0  # 0 = print all

    # trail rendering / switch guards
    trail_hard_switch_x_frac: float = 0.30
    trail_hard_switch_y_frac: float = 0.30
    ball_marker_box_scale: float = 0.22
    ball_marker_min_radius: int = 3
    ball_marker_max_radius: int = 12
    drop_unattached_carry: bool = True
    carry_attach_max_frac: float = 0.016  # keep carry only if it ends this close to next DET

    # output encoding
    use_nvenc: bool = True
    nvenc_preset: str = "p1"
    nvenc_bitrate: str = "8M"
    use_async_writer: bool = True
    async_queue: int = 96
    cache_input_frames_pass2: bool = True  # keep decoded frames in RAM to avoid second decode
    pass2_cache_max_mb: int = 768
    progress_every: int = 200
    info_timing: bool = False

