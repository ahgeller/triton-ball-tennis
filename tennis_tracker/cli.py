from __future__ import annotations

import argparse
from typing import Iterable, Optional, Tuple

from .config import Config


OUTPUT_NAMES = {"tracking", "pre-yolo", "motion", "guide", "motion-tracks"}


def _parse_outputs(value: Optional[str]) -> Optional[Tuple[bool, bool, bool, bool, bool]]:
    if value is None:
        return None
    selected = {item.strip().lower() for item in value.split(",") if item.strip()}
    if not selected:
        raise argparse.ArgumentTypeError("--outputs cannot be empty")
    unknown = selected - OUTPUT_NAMES - {"all", "none"}
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown output(s): {', '.join(sorted(unknown))}. "
            f"Use any of: {', '.join(sorted(OUTPUT_NAMES))}, all, none."
        )
    if "all" in selected:
        selected = set(OUTPUT_NAMES)
    elif "none" in selected:
        selected = set()
    return (
        "tracking" in selected,
        "motion" in selected,
        "pre-yolo" in selected,
        "guide" in selected,
        "motion-tracks" in selected,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    d = Config()
    p = argparse.ArgumentParser(
        description="Tennis ball tracking pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("Paths")
    g.add_argument("-i", "--input", default=d.input_video, help="Input video")
    g.add_argument("-o", "--output", default=d.output_video, help="Tracking video path")
    g.add_argument("--model", default=d.model_path, help="Ball model path (.engine)")
    g.add_argument("--tracking-json", default=d.tracking_json,
                   help="Optional per-frame tracking/benchmark JSON path")

    g = p.add_argument_group("Outputs")
    g.add_argument(
        "--outputs",
        type=_parse_outputs,
        default=None,
        help="Comma-separated outputs: tracking, pre-yolo, motion, guide, motion-tracks, all, none",
    )
    g.add_argument("--tracking-video", dest="tracking_video", action="store_true", default=None,
                   help="Write the main tracking video")
    g.add_argument("--no-tracking-video", dest="tracking_video", action="store_false",
                   help="Skip the main tracking video")
    g.add_argument("--motion-video", dest="motion_video", action="store_true", default=None,
                   help="Write the motion debug video")
    g.add_argument("--no-motion-video", dest="motion_video", action="store_false",
                   help="Skip the motion debug video")
    g.add_argument("--pre-yolo-video", dest="pre_yolo_video", action="store_true", default=None,
                   help="Write the pre-YOLO/preprocessed input video")
    g.add_argument("--no-pre-yolo-video", dest="pre_yolo_video", action="store_false",
                   help="Skip the pre-YOLO/preprocessed input video")
    g.add_argument("--debug-video", dest="debug_video", action="store_true", default=None,
                   help="Legacy alias: write both motion and pre-YOLO debug videos")
    g.add_argument("--no-debug-video", dest="debug_video", action="store_false",
                   help="Disable legacy debug videos")
    g.add_argument("--debug-path", default=d.output_debug_path, help="Motion debug video path")
    g.add_argument("--yolo-debug-path", default=d.output_yolo_input_debug_path,
                   help="Pre-YOLO debug video path")
    g.add_argument("--guide-video", dest="guide_video", action="store_true", default=None,
                   help="Write guide progression debug video")
    g.add_argument("--no-guide-video", dest="guide_video", action="store_false",
                   help="Skip guide progression debug video")
    g.add_argument("--guide-path", default=d.output_guide_path, help="Guide progression video path")
    g.add_argument("--motion-tracks-video", dest="motion_tracks_video", action="store_true", default=None,
                   help="Write motion-track debug video")
    g.add_argument("--no-motion-tracks-video", dest="motion_tracks_video", action="store_false",
                   help="Skip motion-track debug video")
    g.add_argument("--motion-tracks-path", default=d.output_motion_tracks_debug_path,
                   help="Motion-track debug video path")

    g = p.add_argument_group("Detection")
    g.add_argument("--conf", type=float, default=d.conf, help="Ball confidence threshold")
    g.add_argument("--ball-backend", default=d.ball_backend, choices=["trt"],
                   help="Ball backend")
    g.add_argument("--device", default=d.device, help="Device: auto, cpu, mps, 0, 1")

    g = p.add_argument_group("Court Perspective")
    g.add_argument("--court-depth", default="none", choices=["top_far", "bot_far", "none"],
                   help="Far end of court")
    g.add_argument("--court-side", default="none", choices=["center_near", "left_far", "right_far", "none"],
                   help="Horizontal perspective")
    g.add_argument("--y-scale", type=float, default=d.y_scale_strength, help="Depth scale strength")
    g.add_argument("--x-scale", type=float, default=d.x_scale_strength, help="Side scale strength")

    g = p.add_argument_group("Preprocessing")
    g.add_argument("--no-preprocess", action="store_true", help="Disable motion preprocessing")
    g.add_argument("--sat-boost", type=float, default=d.pre_sat_boost, help="Saturation boost for motion blobs")
    g.add_argument("--val-boost", type=float, default=d.pre_val_boost, help="Brightness boost for motion blobs")
    g.add_argument("--hue-shift", type=float, default=d.pre_hue_shift, help="Hue shift toward yellow/green")
    g.add_argument("--dim-static", type=float, default=d.dim_static, help="Static-region dimming")
    g.add_argument("--static-sat-scale", type=float, default=d.static_sat_scale, help="Static-region desaturation")
    g.add_argument("--motion-thresh", type=float, default=d.motion_thresh, help="Motion sensitivity")
    g.add_argument("--motion-v-min", type=float, default=d.motion_v_min, help="Minimum V for motion pixels")
    g.add_argument("--motion-temporal-soft", dest="motion_temporal_soft",
                   action="store_true", default=d.motion_temporal_soft,
                   help="Use soft 3-frame temporal gate")
    g.add_argument("--motion-temporal-strict", dest="motion_temporal_soft",
                   action="store_false", help="Use strict 3-frame AND gate")
    g.add_argument("--motion-temporal-lo-frac", type=float, default=d.motion_temporal_lo_frac,
                   help="Soft temporal low threshold fraction")
    g.add_argument("--motion-temporal-hi-mult", type=float, default=d.motion_temporal_hi_mult,
                   help="Soft temporal strong-threshold multiplier")
    g.add_argument("--motion-flicker-suppress", dest="motion_flicker_suppress",
                   action="store_true", default=d.motion_flicker_suppress,
                   help="Suppress one-frame flicker blobs")
    g.add_argument("--no-motion-flicker-suppress", dest="motion_flicker_suppress",
                   action="store_false", help="Disable flicker suppression")
    g.add_argument("--motion-flicker-min-area", type=int, default=d.motion_flicker_min_area,
                   help="Always drop boost components smaller than this area")
    g.add_argument("--motion-flicker-max-area", type=int, default=d.motion_flicker_max_area,
                   help="Suppress unsupported components up to this area")
    g.add_argument("--motion-flicker-prev-dilate", type=int, default=d.motion_flicker_prev_dilate,
                   help="History support dilation kernel")
    g.add_argument("--motion-flicker-keep-radius", type=float, default=d.motion_flicker_keep_radius_frac,
                   help="Keep-zone radius as frame-diagonal fraction around predicted ball")
    g.add_argument("--motion-raw-temporal-gate", dest="motion_raw_temporal_gate",
                   action="store_true", default=d.motion_raw_temporal_gate,
                   help="Use frame-to-frame movement gate for raw motion")
    g.add_argument("--no-motion-raw-temporal-gate", dest="motion_raw_temporal_gate",
                   action="store_false", help="Disable raw temporal motion gate")
    g.add_argument("--motion-raw-temporal-hi", type=float, default=d.motion_raw_temporal_hi)
    g.add_argument("--motion-raw-temporal-lo", type=float, default=d.motion_raw_temporal_lo)
    g.add_argument("--motion-raw-temporal-very-hi", type=float, default=d.motion_raw_temporal_very_hi)
    g.add_argument("--motion-raw-close-size", type=int, default=d.motion_raw_close_size)
    g.add_argument("--motion-raw-component-filter", dest="motion_raw_component_filter",
                   action="store_true", default=d.motion_raw_component_filter)
    g.add_argument("--no-motion-raw-component-filter", dest="motion_raw_component_filter",
                   action="store_false")
    g.add_argument("--motion-raw-max-area", type=int, default=d.motion_raw_component_max_area)
    g.add_argument("--motion-raw-max-dim", type=int, default=d.motion_raw_component_max_dim)
    g.add_argument("--boost-max-blob", type=int, default=d.boost_max_blob_area)
    g.add_argument("--boost-min-blob", type=int, default=d.boost_min_blob_area)

    g = p.add_argument_group("Blob Filtering")
    g.add_argument("--no-shape-filter", action="store_true", help="Disable erosion + aspect filtering")
    g.add_argument("--blob-erode", type=int, default=d.blob_erode_size)
    g.add_argument("--blob-max-aspect", type=float, default=d.blob_max_aspect)

    g = p.add_argument_group("Player & Court Detection")
    g.add_argument("--player-model", default=d.player_model_path, help="Player model path (.engine)")
    g.add_argument("--court-model", default=d.court_model_path, help="Court model path (.engine)")
    g.add_argument("--player-interval", type=int, default=d.player_detect_interval)
    g.add_argument("--player-interval-stable", type=int, default=d.player_detect_interval_stable)
    g.add_argument("--num-players", type=int, default=d.num_players)
    g.add_argument("--court-interval", type=int, default=d.court_detect_interval)
    g.add_argument("--court-conf", type=float, default=d.court_conf)
    g.add_argument("--print-court-raw", dest="print_court_raw", action="store_true", default=d.print_court_raw)
    g.add_argument("--no-print-court-raw", dest="print_court_raw", action="store_false")
    g.add_argument("--court-remap-semantic-14", action="store_true", default=d.court_remap_semantic_14)
    g.add_argument("--court-points-only", action="store_true", default=d.court_points_only)
    g.add_argument("--court-indices", action="store_true", default=d.court_draw_indices)
    g.add_argument("--player-conf", type=float, default=d.player_conf)
    g.add_argument("--player-iou", type=float, default=d.player_iou)
    g.add_argument("--no-draw-players", action="store_true", help="Do not draw player boxes")
    g.add_argument("--no-draw-court", action="store_true", help="Do not draw court lines")

    g = p.add_argument_group("Acceleration")
    g.add_argument("--no-tensorrt", action="store_true", help="Disable TensorRT")
    g.add_argument("--no-nvenc", action="store_true", help="Disable NVENC encoding")
    g.add_argument("--trt-async", dest="trt_async_execute", action="store_true",
                   default=d.trt_async_execute, help="Use TensorRT async execution")
    g.add_argument("--no-trt-async", dest="trt_async_execute", action="store_false",
                   help="Use synchronous TensorRT execution")
    g.add_argument("--trt-async-slots", type=int, default=d.trt_async_slots)
    g.add_argument("--info", dest="info_timing", action="store_true", default=d.info_timing,
                   help="Print timing breakdown")
    g.add_argument("--no-info", dest="info_timing", action="store_false")

    g = p.add_argument_group("Debug Rendering")
    g.add_argument("--debug-show-raw-motion", dest="debug_show_raw_motion",
                   action="store_true", default=d.debug_show_raw_motion)
    g.add_argument("--debug-hide-raw-motion", dest="debug_show_raw_motion",
                   action="store_false")
    g.add_argument("--debug-probe-motion-style", dest="debug_probe_motion_style",
                   action="store_true", default=d.debug_probe_motion_style)
    g.add_argument("--debug-legacy-motion-trails", dest="debug_probe_motion_style",
                   action="store_false")
    g.add_argument("--guide-interp-gap", type=int, default=d.guide_interp_max_gap)
    g.add_argument("--print-selector-tracks", dest="print_selector_tracks",
                   action="store_true", default=d.print_selector_tracks)
    g.add_argument("--no-print-selector-tracks", dest="print_selector_tracks",
                   action="store_false")
    g.add_argument("--selector-track-limit", type=int, default=d.selector_track_limit)

    g = p.add_argument_group("Speed")
    g.add_argument("--skip-frame-yolo", type=int, default=d.skip_frame_yolo,
                   help="Run ball YOLO every Nth frame")
    g.add_argument("--skip-frame-require-roi", dest="skip_frame_require_roi",
                   action="store_true", default=d.skip_frame_require_roi)
    g.add_argument("--no-skip-frame-require-roi", dest="skip_frame_require_roi",
                   action="store_false")
    g.add_argument("--skip-preprocess-dim", dest="skip_preprocess_dim",
                   action="store_true", default=d.skip_preprocess_dim)
    g.add_argument("--no-skip-preprocess-dim", dest="skip_preprocess_dim",
                   action="store_false")
    g.add_argument("--aux-on-yolo-frames", dest="aux_detect_on_yolo_frames",
                   action="store_true", default=d.aux_detect_on_yolo_frames)
    g.add_argument("--no-aux-on-yolo-frames", dest="aux_detect_on_yolo_frames",
                   action="store_false")
    g.add_argument("--aux-force-interval", type=int, default=d.aux_force_interval)
    g.add_argument("--reader-prefetch", type=int, default=d.frame_reader_prefetch)
    g.add_argument("--cache-pass2-frames", dest="cache_input_frames_pass2",
                   action="store_true", default=d.cache_input_frames_pass2)
    g.add_argument("--no-cache-pass2-frames", dest="cache_input_frames_pass2",
                   action="store_false")
    g.add_argument("--pass2-cache-max-mb", type=int, default=d.pass2_cache_max_mb)

    g = p.add_argument_group("ROI Motion")
    g.add_argument("--roi-motion", dest="roi_motion_enabled", action="store_true",
                   default=d.roi_motion_enabled)
    g.add_argument("--no-roi-motion", dest="roi_motion_enabled", action="store_false")
    g.add_argument("--roi-visible-radius", type=float, default=d.roi_visible_radius_frac)
    g.add_argument("--roi-lost-radius", type=float, default=d.roi_lost_radius_frac)
    g.add_argument("--roi-fullframe-interval", type=int, default=d.roi_fullframe_interval)

    g = p.add_argument_group("Trail")
    g.add_argument("--trail-hard-switch-x-frac", type=float, default=d.trail_hard_switch_x_frac)
    g.add_argument("--trail-hard-switch-y-frac", type=float, default=d.trail_hard_switch_y_frac)
    g.add_argument("--ball-marker-scale", type=float, default=d.ball_marker_box_scale)
    g.add_argument("--ball-marker-min-r", type=int, default=d.ball_marker_min_radius)
    g.add_argument("--ball-marker-max-r", type=int, default=d.ball_marker_max_radius)

    return p


def _resolve_output_flags(args: argparse.Namespace) -> Tuple[bool, bool, bool, bool, bool]:
    tracking = True
    motion = False
    pre_yolo = False
    guide = False
    motion_tracks = False

    if args.outputs is not None:
        tracking, motion, pre_yolo, guide, motion_tracks = args.outputs
    if args.debug_video is True:
        motion = True
        pre_yolo = True
    elif args.debug_video is False:
        motion = False
        pre_yolo = False
    if args.tracking_video is not None:
        tracking = bool(args.tracking_video)
    if args.motion_video is not None:
        motion = bool(args.motion_video)
    if args.pre_yolo_video is not None:
        pre_yolo = bool(args.pre_yolo_video)
    if args.guide_video is not None:
        guide = bool(args.guide_video)
    if args.motion_tracks_video is not None:
        motion_tracks = bool(args.motion_tracks_video)
    return tracking, motion, pre_yolo, guide, motion_tracks


def config_from_args(args: argparse.Namespace) -> Config:
    save_tracking, save_motion, save_pre_yolo, save_guide, save_motion_tracks = _resolve_output_flags(args)
    return Config(
        input_video=args.input,
        output_video=args.output,
        model_path=args.model,
        tracking_json=args.tracking_json,
        save_tracking_video=save_tracking,
        save_motion_debug=save_motion,
        save_yolo_input_debug=save_pre_yolo,
        conf=args.conf,
        ball_backend=args.ball_backend,
        device=args.device,
        court_depth=None if args.court_depth == "none" else args.court_depth,
        court_side=None if args.court_side == "none" else args.court_side,
        y_scale_strength=args.y_scale,
        x_scale_strength=args.x_scale,
        enable_preprocess=not args.no_preprocess,
        pre_sat_boost=args.sat_boost,
        pre_val_boost=args.val_boost,
        pre_hue_shift=args.hue_shift,
        dim_static=args.dim_static,
        static_sat_scale=args.static_sat_scale,
        motion_thresh=args.motion_thresh,
        motion_v_min=args.motion_v_min,
        motion_temporal_soft=args.motion_temporal_soft,
        motion_temporal_lo_frac=args.motion_temporal_lo_frac,
        motion_temporal_hi_mult=args.motion_temporal_hi_mult,
        motion_flicker_suppress=args.motion_flicker_suppress,
        motion_flicker_min_area=args.motion_flicker_min_area,
        motion_flicker_max_area=args.motion_flicker_max_area,
        motion_flicker_prev_dilate=args.motion_flicker_prev_dilate,
        motion_flicker_keep_radius_frac=args.motion_flicker_keep_radius,
        motion_raw_temporal_gate=args.motion_raw_temporal_gate,
        motion_raw_temporal_hi=args.motion_raw_temporal_hi,
        motion_raw_temporal_lo=args.motion_raw_temporal_lo,
        motion_raw_temporal_very_hi=args.motion_raw_temporal_very_hi,
        motion_raw_close_size=max(0, int(args.motion_raw_close_size)),
        motion_raw_component_filter=args.motion_raw_component_filter,
        motion_raw_component_max_area=max(1, int(args.motion_raw_max_area)),
        motion_raw_component_max_dim=max(1, int(args.motion_raw_max_dim)),
        boost_max_blob_area=args.boost_max_blob,
        boost_min_blob_area=args.boost_min_blob,
        blob_shape_filter=not args.no_shape_filter,
        blob_erode_size=args.blob_erode,
        blob_max_aspect=args.blob_max_aspect,
        player_model_path=args.player_model,
        court_model_path=args.court_model,
        player_detect_interval=args.player_interval,
        player_detect_interval_stable=args.player_interval_stable,
        num_players=max(1, int(args.num_players)),
        court_detect_interval=args.court_interval,
        court_conf=args.court_conf,
        print_court_raw=args.print_court_raw,
        court_remap_semantic_14=args.court_remap_semantic_14,
        court_points_only=args.court_points_only,
        court_draw_indices=args.court_indices,
        player_conf=args.player_conf,
        player_iou=args.player_iou,
        draw_players=not args.no_draw_players,
        draw_court=not args.no_draw_court,
        use_tensorrt=not args.no_tensorrt,
        use_nvenc=not args.no_nvenc,
        trt_async_execute=args.trt_async_execute,
        trt_async_slots=max(1, int(args.trt_async_slots)),
        info_timing=args.info_timing,
        skip_frame_yolo=max(1, int(args.skip_frame_yolo)),
        skip_frame_require_roi=args.skip_frame_require_roi,
        skip_preprocess_dim=args.skip_preprocess_dim,
        aux_detect_on_yolo_frames=args.aux_detect_on_yolo_frames,
        aux_force_interval=max(1, int(args.aux_force_interval)),
        frame_reader_prefetch=max(2, int(args.reader_prefetch)),
        roi_motion_enabled=args.roi_motion_enabled,
        roi_visible_radius_frac=args.roi_visible_radius,
        roi_lost_radius_frac=args.roi_lost_radius,
        roi_fullframe_interval=args.roi_fullframe_interval,
        cache_input_frames_pass2=args.cache_input_frames_pass2,
        pass2_cache_max_mb=max(64, int(args.pass2_cache_max_mb)),
        output_debug_path=args.debug_path,
        output_yolo_input_debug_path=args.yolo_debug_path,
        debug_show_raw_motion=args.debug_show_raw_motion,
        debug_probe_motion_style=args.debug_probe_motion_style,
        save_guide_video=save_guide,
        output_guide_path=args.guide_path,
        guide_interp_max_gap=max(1, int(args.guide_interp_gap)),
        save_motion_tracks_video=save_motion_tracks,
        output_motion_tracks_debug_path=args.motion_tracks_path,
        print_selector_tracks=args.print_selector_tracks,
        selector_track_limit=max(0, int(args.selector_track_limit)),
        trail_hard_switch_x_frac=args.trail_hard_switch_x_frac,
        trail_hard_switch_y_frac=args.trail_hard_switch_y_frac,
        ball_marker_box_scale=args.ball_marker_scale,
        ball_marker_min_radius=args.ball_marker_min_r,
        ball_marker_max_radius=args.ball_marker_max_r,
    )


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    cfg = config_from_args(args)
    from .pipeline import run

    run(cfg)
