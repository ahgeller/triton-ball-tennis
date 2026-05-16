from dataclasses import dataclass
from typing import Optional



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
    trt_async_slots: int = 3

    # Court perspective
    court_depth: Optional[str] = None
    court_side: Optional[str] = None
    y_scale_strength: float = 0.35
    x_scale_strength: float = 0.15

    # Player & court detection
    player_model_path: Optional[str] = "models/player.engine"
    court_model_path: Optional[str] = "models/courtdetection.engine"
    player_detect_interval: int = 1
    player_detect_interval_stable: int = 15  # interval when players haven't moved much
    player_stable_thresh_frac: float = 0.02  # movement < this * diag = "stable"
    court_detect_interval: int = 400
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
    pre_sat_boost: float = 1.55
    pre_val_boost: float = 1.04
    pre_hue_shift: float = 0.18
    dim_static: float = 0.88
    static_sat_scale: float = 0.75
    motion_dilate: int = 5
    
    # motion detection - IMPROVED for better ball continuity
    wta_alpha: float = 0.02               # Base background update rate at static pixels
    motion_freeze_alpha: float = 0.015    # Faster background/variance adaptation at motion pixels to let parked balls settle
    motion_thresh: float = 11.0           # Minimum pixel diff floor (additive term in threshold); too high makes system blind to very weak motion
    motion_k_std: float = 3.0             # Std-dev multiplier; balanced with additive floor so variance can suppress static-ball halos
    motion_v_min: float = 40.0           # Lowered to catch dimmer ball motion in shadows
    motion_temporal_soft: bool = False   # Disabled: this was causing the "lingering" yellow artifacts you noticed
    motion_temporal_lo_frac: float = 0.55
    motion_temporal_hi_mult: float = 1.35
    motion_flicker_suppress: bool = False # Disabled to prevent motion ghosts from lingering
    motion_flicker_min_area: int = 3
    motion_flicker_max_area: int = 220
    motion_flicker_prev_dilate: int = 9
    motion_flicker_keep_radius_frac: float = 0.11
    motion_raw_temporal_gate: bool = True  # C++ probe-style frame-to-frame proof before raw foreground becomes motion evidence
    motion_raw_temporal_hi: float = 18.0
    motion_raw_temporal_lo: float = 8.0
    motion_raw_temporal_very_hi: float = 36.0
    motion_raw_close_size: int = 2
    motion_raw_open_size: int = 0
    motion_raw_component_filter: bool = False  # Off by default; too strict for the main tracker so far
    motion_raw_component_min_area: int = 2
    motion_raw_component_max_area: int = 260
    motion_raw_component_max_dim: int = 38
    motion_raw_component_max_aspect: float = 5.5
    motion_raw_component_min_fill: float = 0.14
    motion_raw_ball_color_gate: bool = True  # Keep raw motion only near loose tennis-ball color support
    motion_raw_color_h_min: float = 18.0
    motion_raw_color_h_max: float = 75.0
    motion_raw_color_dilate: int = 5
    motion_raw_color_s_min: float = 0.12
    motion_raw_color_v_min: float = 0.18
    boost_max_blob_area: int = 600   # Reduced from 1200: large blobs (sqrt(1200/pi) 19px radius) overshoot the ball
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

    # ROI-based motion - only compute motion/CC near ball
    roi_motion_enabled: bool = True
    roi_visible_radius_frac: float = 0.1
    roi_lost_radius_frac: float = 0.2
    roi_lost_expand_per_frame: float = 0.0022 # grow lost ROI by this * diag per lost frame
    roi_max_radius_frac: float = 0.085       # cap ROI radius
    roi_motion_bleed_frac: float = 0.0       # extend the underlying cropping boundary for motion processing to capture long blurred streaks without affecting the tight visual box
    roi_fullframe_interval: int = 0         # 0 = never do full-frame fallback; N = every N frames

    # blob shape filtering - IMPROVED for small tennis balls
    blob_shape_filter: bool = True
    blob_erode_size: int = 3
    blob_max_aspect: float = 4.0
    blob_preserve_tiny: bool = True       # NEW: preserve very small ball-sized blobs
    blob_tiny_max_area: int = 120         # NEW: max area for "tiny" ball blobs to preserve

    # outputs / debug
    save_tracking_video: bool = True
    save_motion_debug: bool = False
    save_yolo_input_debug: bool = False
    output_debug_path: str = "output_videos/prof_test_motion_debug.mp4"
    output_yolo_input_debug_path: str = "output_videos/prof_test_yolo_input_debug.mp4"
    debug_show_raw_motion: bool = False
    debug_probe_motion_style: bool = True
    save_guide_video: bool = False
    output_guide_path: str = "output_videos/prof_test_guide_debug.mp4"
    save_motion_tracks_video: bool = False
    output_motion_tracks_debug_path: str = "output_videos/prof_test_motion_tracks_debug.mp4"
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
    tracking_json: Optional[str] = None
