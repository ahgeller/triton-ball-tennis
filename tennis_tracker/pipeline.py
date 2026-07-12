import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from ball_in_play_selector import select_ball_in_play, FrameResult
try:
    import torch
    HAS_TORCH = True
except Exception:
    torch = None
    HAS_TORCH = False

from .config import Config
from .utils import _detect_device, _check_capabilities, _resolve_engine_path_for_ball, find_ball_class_id_from_names, _read_engine_names
from .detectors import BallDetectorBackend, CourtDetector, PlayerDetector, TensorRTRuntimeBallDetector
from .motion import filter_boost_mask, _pack_mask_u8, build_protect_mask, compute_motion_sv_from_hsv, refine_raw_motion_temporal_cpu, suppress_flicker_components, preprocess_frame_cuda, _unpack_mask_u8, build_court_side_protect_mask, apply_exclude_mask_u8, preprocess_frame
from .tracking import ROIMotionTracker
from .rendering import _is_soft_source, _trail_base_color, _get_track_color, _court_axis_spans, _build_court_polygon, _trail_jump_fracs, _draw_homography_net_line, _print_timing_summary, _trail_direction_break, _print_selector_track_summary, _trail_smooth_alpha, _build_ground_projection_model, _drop_unattached_soft_runs, _trail_prev2, _build_display_guide, COLOR_DET, COLOR_RAW, COLOR_MOTION, COLOR_SEARCH, COLOR_INTERP, COLOR_CARRY, COLOR_GUIDE, COLOR_GUIDE_INTERP, ENABLE_GAP_CONNECTORS
from .video_io import _cuda_frame_to_chw_f32, _PinnedFrameUploader, _cuda_vs_tensors, ThreadedFrameReader, VideoWriter


def _json_safe(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return str(value)


def _frame_result_to_json(frame_idx: int, result: Optional[FrameResult]) -> Dict[str, Any]:
    row: Dict[str, Any] = {"frame": int(frame_idx), "present": False}
    if result is None or bool(getattr(result, "debug_only", False)):
        if result is not None and bool(getattr(result, "debug_only", False)):
            row["debug_only"] = True
        return row

    row.update({
        "present": result.cx is not None and result.cy is not None,
        "x": None if result.cx is None else float(result.cx),
        "y": None if result.cy is None else float(result.cy),
        "conf": float(getattr(result, "conf", 0.0)),
        "source": str(getattr(result, "source", "")),
        "interpolated": bool(getattr(result, "interpolated", False)),
        "bbox": _json_safe(getattr(result, "bbox", None)),
        "search": {
            "x": float(getattr(result, "search_cx", 0.0)),
            "y": float(getattr(result, "search_cy", 0.0)),
            "radius": float(getattr(result, "search_radius", 0.0)),
        },
        "guide_search": {
            "x": float(getattr(result, "guide_search_cx", 0.0)),
            "y": float(getattr(result, "guide_search_cy", 0.0)),
            "radius": float(getattr(result, "guide_search_radius", 0.0)),
            "exact": bool(getattr(result, "guide_search_exact", False)),
            "frozen": bool(getattr(result, "guide_search_frozen", False)),
            "hold": bool(getattr(result, "guide_search_hold", False)),
        },
        "selection": _json_safe(getattr(result, "source_policy", {}) or {
            "source": str(getattr(result, "source", "")),
            "reason": str(getattr(result, "source_reason", "")),
            "reasons": list(getattr(result, "source_reasons", []) or []),
            "rejects": dict(getattr(result, "source_rejects", {}) or {}),
        }),
    })
    return row


def _track_to_json(track) -> Optional[Dict[str, Any]]:
    if track is None:
        return None
    observations = []
    for obs in getattr(track, "observations", []) or []:
        observations.append({
            "frame": int(getattr(obs, "frame", 0)),
            "x": float(getattr(obs, "cx", 0.0)),
            "y": float(getattr(obs, "cy", 0.0)),
            "bbox": [
                float(getattr(obs, "x1", 0.0)),
                float(getattr(obs, "y1", 0.0)),
                float(getattr(obs, "x2", 0.0)),
                float(getattr(obs, "y2", 0.0)),
            ],
            "conf": float(getattr(obs, "conf", 0.0)),
            "area": float(getattr(obs, "area", 0.0)),
            "on_motion": bool(getattr(obs, "on_motion", False)),
        })

    return {
        "track_id": int(getattr(track, "track_id", -1)),
        "score": float(getattr(track, "score", 0.0)),
        "num_obs": int(getattr(track, "num_obs", 0)),
        "span": int(getattr(track, "span", 0)),
        "first_frame": int(getattr(track, "first_frame", 0)),
        "last_obs_frame": int(getattr(track, "last_obs_frame", 0)),
        "score_breakdown": _json_safe(getattr(track, "score_breakdown", {}) or {}),
        "observations": observations,
    }


def _last_valid_court_keypoints(all_court_kps: Optional[List[Any]]) -> Optional[Any]:
    if not all_court_kps:
        return None
    for kps in reversed(all_court_kps):
        if kps and len(kps) >= 16:
            return kps
    return None


def _count_reason(reason_counts: Dict[str, int], reason: str) -> None:
    reason_counts[reason] = int(reason_counts.get(reason, 0)) + 1


def _mask_has_pixels(mask_obj) -> bool:
    mask = _unpack_mask_u8(mask_obj)
    return bool(mask is not None and cv2.countNonZero(mask) > 0)


def _extract_motion_candidates(
    mask_obj,
    mask_source: str,
    frame_idx: int,
    search_x: Optional[float],
    search_y: Optional[float],
    selected_x: Optional[float],
    selected_y: Optional[float],
    max_candidates: int = 5,
) -> List[Dict[str, Any]]:
    mask = _unpack_mask_u8(mask_obj)
    if mask is None or cv2.countNonZero(mask) <= 0:
        return []

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: List[Dict[str, Any]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area <= 0.0:
            continue
        x, y, w_box, h_box = cv2.boundingRect(contour)
        if w_box <= 0 or h_box <= 0:
            continue
        moments = cv2.moments(contour)
        if abs(float(moments.get("m00", 0.0))) <= 1e-9:
            continue
        cx = float(moments["m10"] / moments["m00"])
        cy = float(moments["m01"] / moments["m00"])
        perimeter = float(cv2.arcLength(contour, True))
        compactness = (
            float(4.0 * math.pi * area / max(perimeter * perimeter, 1e-9))
            if perimeter > 0.0 else 0.0
        )
        aspect_ratio = float(max(w_box, h_box) / max(min(w_box, h_box), 1))
        fill_ratio = float(area / max(float(w_box * h_box), 1.0))

        dist_to_search = None
        if search_x is not None and search_y is not None:
            dist_to_search = float(math.hypot(cx - float(search_x), cy - float(search_y)))
        dist_to_selected = None
        if selected_x is not None and selected_y is not None:
            dist_to_selected = float(math.hypot(cx - float(selected_x), cy - float(selected_y)))

        # Debug-only ranking: favor compact, filled components near the active search point.
        score = 0.0
        score += 50.0 * max(0.0, min(1.0, compactness))
        score += 25.0 * max(0.0, min(1.0, fill_ratio))
        score += 15.0 * min(area, 600.0) / 600.0
        if dist_to_search is not None:
            score -= min(dist_to_search, 300.0) / 6.0
        if aspect_ratio > 1.0:
            score -= max(0.0, aspect_ratio - 1.0) * 4.0

        candidates.append({
            "frame": int(frame_idx),
            "mask": str(mask_source),
            "x": cx,
            "y": cy,
            "bbox": [int(x), int(y), int(x + w_box), int(y + h_box)],
            "area": area,
            "compactness": compactness,
            "aspect_ratio": aspect_ratio,
            "fill_ratio": fill_ratio,
            "distance_to_search": dist_to_search,
            "distance_to_selected": dist_to_selected,
            "score": float(score),
        })

    candidates.sort(key=lambda item: float(item["score"]), reverse=True)
    return candidates[:max(1, int(max_candidates))]


def _choose_motion_mask(boost_mask, raw_motion) -> Tuple[Optional[Any], str, bool, bool]:
    has_boost = _mask_has_pixels(boost_mask)
    has_raw = _mask_has_pixels(raw_motion)
    if has_boost:
        return boost_mask, "boost", has_boost, has_raw
    if has_raw:
        return raw_motion, "raw", has_boost, has_raw
    return None, "none", has_boost, has_raw


def _result_position(result: Optional[FrameResult]) -> Tuple[Optional[float], Optional[float]]:
    if result is None or bool(getattr(result, "debug_only", False)):
        return None, None
    if result.cx is None or result.cy is None:
        return None, None
    return float(result.cx), float(result.cy)


def _result_search(
    result: Optional[FrameResult],
    prev_pos: Optional[Tuple[float, float]],
    diag: float,
) -> Tuple[Optional[float], Optional[float], float, str]:
    if result is not None and not bool(getattr(result, "debug_only", False)):
        sx = float(getattr(result, "search_cx", 0.0))
        sy = float(getattr(result, "search_cy", 0.0))
        sr = float(getattr(result, "search_radius", 0.0))
        if sr > 0.0 and (sx != 0.0 or sy != 0.0):
            return sx, sy, sr, "result_search"
        rx, ry = _result_position(result)
        if rx is not None and ry is not None:
            return rx, ry, max(24.0, 0.035 * diag), "result_position"
    if prev_pos is not None:
        return float(prev_pos[0]), float(prev_pos[1]), max(28.0, 0.045 * diag), "previous_position"
    return None, None, 0.0, "none"


def _motion_diagnostic_reason(
    selected_source: Optional[str],
    has_yolo: bool,
    mask_source: str,
    candidates: List[Dict[str, Any]],
    search_x: Optional[float],
    search_y: Optional[float],
    search_radius: float,
) -> str:
    if selected_source == "motion":
        return "accepted_motion"
    if has_yolo:
        return "yolo_detection_available"
    if mask_source == "none":
        return "no_motion_mask"
    if not candidates:
        return "no_blob"
    if search_x is None or search_y is None:
        return "no_search_anchor"

    best_dist = candidates[0].get("distance_to_search")
    if best_dist is not None:
        gate = max(18.0, float(search_radius) if search_radius > 0.0 else 0.0)
        if float(best_dist) > gate:
            return "blob_too_far_from_search"

    if selected_source in ("carry", "guide", "interp"):
        return f"{selected_source}_selected_over_motion_candidate"
    if selected_source:
        return "non_motion_selected_over_motion_candidate"
    return "lost_despite_motion_candidate"


def _build_motion_diagnostics(
    per_frame: List[Optional[FrameResult]],
    detections_by_frame: List[List[Tuple[list, float]]],
    boost_masks: List[Any],
    raw_motions: List[Any],
    width: int,
    height: int,
    max_candidates_per_frame: int = 5,
) -> Dict[str, Any]:
    diag = math.sqrt(float(width) ** 2 + float(height) ** 2)
    total_frames = max(len(per_frame), len(detections_by_frame), len(boost_masks), len(raw_motions))
    reason_counts: Dict[str, int] = {}
    mask_counts: Dict[str, int] = {}
    selected_source_counts: Dict[str, int] = {}
    diagnostic_frames: List[Dict[str, Any]] = []
    prev_pos: Optional[Tuple[float, float]] = None
    yolo_gap_frames = 0
    candidate_gap_frames = 0

    for frame_idx in range(total_frames):
        result = per_frame[frame_idx] if frame_idx < len(per_frame) else None
        selected_x, selected_y = _result_position(result)
        selected_source = None
        if result is not None and not bool(getattr(result, "debug_only", False)):
            selected_source = str(getattr(result, "source", "") or "")
            if selected_source:
                selected_source_counts[selected_source] = int(selected_source_counts.get(selected_source, 0)) + 1

        dets = detections_by_frame[frame_idx] if frame_idx < len(detections_by_frame) else []
        has_yolo = bool(dets)
        if not has_yolo:
            yolo_gap_frames += 1

        boost = boost_masks[frame_idx] if frame_idx < len(boost_masks) else None
        raw = raw_motions[frame_idx] if frame_idx < len(raw_motions) else None
        chosen_mask, mask_source, has_boost, has_raw = _choose_motion_mask(boost, raw)
        mask_counts[mask_source] = int(mask_counts.get(mask_source, 0)) + 1

        search_x, search_y, search_radius, search_source = _result_search(result, prev_pos, diag)
        candidates = _extract_motion_candidates(
            chosen_mask,
            mask_source,
            frame_idx,
            search_x,
            search_y,
            selected_x,
            selected_y,
            max_candidates=max_candidates_per_frame,
        )
        if candidates and not has_yolo and selected_source != "motion":
            candidate_gap_frames += 1

        reason = _motion_diagnostic_reason(
            selected_source,
            has_yolo,
            mask_source,
            candidates,
            search_x,
            search_y,
            search_radius,
        )
        _count_reason(reason_counts, reason)

        should_export_frame = (
            selected_source == "motion" or
            (not has_yolo and selected_source != "motion") or
            bool(candidates and not has_yolo)
        )
        if should_export_frame:
            diagnostic_frames.append({
                "frame": int(frame_idx),
                "selected_source": selected_source,
                "reason": reason,
                "has_yolo_detections": bool(has_yolo),
                "mask_source": mask_source,
                "has_boost_mask": bool(has_boost),
                "has_raw_motion": bool(has_raw),
                "search": {
                    "x": search_x,
                    "y": search_y,
                    "radius": float(search_radius),
                    "source": search_source,
                },
                "selected_position": {
                    "x": selected_x,
                    "y": selected_y,
                },
                "selection": _json_safe(getattr(result, "source_policy", {}) if result is not None else {}),
                "candidate_count_exported": int(len(candidates)),
                "candidates": candidates,
            })

        if selected_x is not None and selected_y is not None:
            prev_pos = (selected_x, selected_y)

    return {
        "schema_version": 1,
        "summary": {
            "frames_analyzed": int(total_frames),
            "yolo_gap_frames": int(yolo_gap_frames),
            "yolo_gap_frames_with_motion_candidate": int(candidate_gap_frames),
            "reason_counts": dict(sorted(reason_counts.items())),
            "mask_source_counts": dict(sorted(mask_counts.items())),
            "selected_source_counts": dict(sorted(selected_source_counts.items())),
            "max_candidates_per_frame": int(max_candidates_per_frame),
        },
        "frames": diagnostic_frames,
    }


def _write_tracking_json(
    path: str,
    cfg: Config,
    fps: float,
    width: int,
    height: int,
    total_frames: int,
    elapsed_sec: float,
    filled_frames: int,
    per_frame: List[Optional[FrameResult]],
    chosen_track,
    all_tracks,
    detections_by_frame: Optional[List[List[Tuple[list, float]]]] = None,
    boost_masks: Optional[List[Any]] = None,
    raw_motions: Optional[List[Any]] = None,
    court_keypoints_by_frame: Optional[List[Any]] = None,
    player_boxes_by_frame: Optional[List[Any]] = None,
    timing: Optional[Dict[str, float]] = None,
    pass2_frames_rendered: int = 0,
) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame_rows = []
    for i in range(int(total_frames)):
        row = _frame_result_to_json(i, per_frame[i] if i < len(per_frame) else None)
        if court_keypoints_by_frame is not None and i < len(court_keypoints_by_frame):
            row["court_keypoints"] = _json_safe(court_keypoints_by_frame[i])
        if player_boxes_by_frame is not None and i < len(player_boxes_by_frame):
            row["player_boxes"] = _json_safe(player_boxes_by_frame[i])
        frame_rows.append(row)

    payload = {
        "schema_version": 1,
        "video": {
            "input": str(cfg.input_video),
            "output": str(cfg.output_video),
            "fps": float(fps),
            "width": int(width),
            "height": int(height),
            "total_frames": int(total_frames),
        },
        "summary": {
            "filled_frames": int(filled_frames),
            "filled_percent": float(100.0 * filled_frames / max(1, total_frames)),
            "elapsed_sec": float(elapsed_sec),
            "effective_fps": float(total_frames / max(elapsed_sec, 1e-9)),
            "pass2_frames_rendered": int(pass2_frames_rendered),
            "chosen_track_id": (
                None if chosen_track is None else int(getattr(chosen_track, "track_id", -1))
            ),
            "track_count": int(len(all_tracks or [])),
        },
        "config": _json_safe(getattr(cfg, "__dict__", {})),
        "timing": _json_safe(timing or {}),
        "last_valid_court_keypoints": _json_safe(_last_valid_court_keypoints(court_keypoints_by_frame)),
        "chosen_track": _track_to_json(chosen_track),
        "tracks": [_track_to_json(t) for t in (all_tracks or [])],
        "motion_diagnostics": _build_motion_diagnostics(
            per_frame,
            detections_by_frame or [],
            boost_masks or [],
            raw_motions or [],
            int(width),
            int(height),
        ),
        "frames": frame_rows,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _validate_io_paths(cfg) -> Path:
    input_path = Path(cfg.input_video).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    outputs = []
    if getattr(cfg, "save_tracking_video", True):
        outputs.append(("tracking video", cfg.output_video))
    if cfg.save_motion_debug:
        outputs.append(("motion debug video", cfg.output_debug_path))
    if getattr(cfg, "save_yolo_input_debug", False):
        outputs.append(("pre-YOLO video", cfg.output_yolo_input_debug_path))
    if cfg.save_guide_video:
        outputs.append(("guide video", cfg.output_guide_path))
    if getattr(cfg, "save_motion_tracks_video", False):
        outputs.append(("motion-tracks video", cfg.output_motion_tracks_debug_path))
    if getattr(cfg, "tracking_json", None):
        outputs.append(("tracking JSON", cfg.tracking_json))

    seen = {}
    for label, value in outputs:
        path = Path(value).expanduser().resolve()
        if path == input_path:
            raise ValueError(f"{label} must not overwrite the input video: {path}")
        if path in seen:
            raise ValueError(f"{label} and {seen[path]} resolve to the same output path: {path}")
        seen[path] = label
    return input_path


def _detector_can_overlap(detector) -> bool:
    return bool(getattr(detector, "use_async", False) and int(getattr(detector, "async_slots", 1)) >= 2)


def run(cfg):
    input_path = _validate_io_paths(cfg)
    t0 = time.time()
    info_timing = bool(getattr(cfg, "info_timing", False))
    timing = {} if info_timing else None
    init_perf_t0 = time.perf_counter() if info_timing else 0.0

    # Platform detection
    device_str, _ = _detect_device(cfg.device)
    if not str(device_str).isdigit():
        raise RuntimeError("The tracking runtime requires an NVIDIA CUDA device index such as --device 0.")
    device_index = int(device_str)
    if not (HAS_TORCH and torch.cuda.is_available()) or device_index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device {device_index} is unavailable.")
    cfg.device = str(device_index)
    cfg = _check_capabilities(cfg, device_str)
    HAS_CUDA = HAS_TORCH and torch.cuda.is_available()
    is_cuda = device_str not in ("cpu", "mps")

    model_names = None
    detector: BallDetectorBackend
    ball_cls_id = None
    ball_cls_name = None

    # TensorRT-only ball runtime path.
    trt_engine_path = _resolve_engine_path_for_ball(cfg)
    if trt_engine_path is None:
        raise RuntimeError(
            f"Ball TensorRT engine not found for model_path='{cfg.model_path}'. "
            "Expected a .engine file."
        )
    if not (HAS_TORCH and torch.cuda.is_available()):
        raise RuntimeError("TensorRT runtime requires CUDA-enabled torch.")

    model_names = _read_engine_names(trt_engine_path)
    ball_cls_id, ball_cls_name = find_ball_class_id_from_names(model_names, cfg.ball_class_name)
    print(f"[init] Model classes: {model_names}")
    print(f"[init] Ball class: id={ball_cls_id}, name='{ball_cls_name}'")
    detector = TensorRTRuntimeBallDetector(
        str(trt_engine_path), cfg, ball_cls_id, names=model_names
    )

    if cfg.court_depth or cfg.court_side:
        parts = []
        if cfg.court_depth:
            parts.append(f"depth={cfg.court_depth} (y_strength={cfg.y_scale_strength})")
        if cfg.court_side:
            parts.append(f"side={cfg.court_side} (x_strength={cfg.x_scale_strength})")
        print(f"[init] Court perspective: {', '.join(parts)}")
    else:
        print("[init] Court perspective: disabled (no scaling)")

    # Open video
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[init] Video: {w}x{h} @ {fps:.1f}fps, ~{total} frames")

    court_det = CourtDetector(cfg)
    player_det = PlayerDetector(cfg, court_keypoints=None)

    # Writers (main output writer is created at pass 2 start to avoid long idle FFmpeg/NVENC process)
    writer = None
    dbg_writer = yolo_dbg_writer = guide_writer = motion_tracks_writer = None
    needs_pass2_outputs = bool(
        getattr(cfg, "save_tracking_video", True)
        or cfg.save_motion_debug
        or cfg.save_guide_video
        or getattr(cfg, "save_motion_tracks_video", False)
    )
    if cfg.save_motion_debug:
        os.makedirs(os.path.dirname(cfg.output_debug_path) or ".", exist_ok=True)
        dbg_writer = VideoWriter(cfg.output_debug_path, fps, w, h, cfg)
    if getattr(cfg, "save_yolo_input_debug", False):
        os.makedirs(os.path.dirname(cfg.output_yolo_input_debug_path) or ".", exist_ok=True)
        yolo_dbg_writer = VideoWriter(cfg.output_yolo_input_debug_path, fps, w, h, cfg)
    if cfg.save_guide_video:
        os.makedirs(os.path.dirname(cfg.output_guide_path) or ".", exist_ok=True)
        guide_writer = VideoWriter(cfg.output_guide_path, fps, w, h, cfg)
    if getattr(cfg, "save_motion_tracks_video", False):
        os.makedirs(os.path.dirname(cfg.output_motion_tracks_debug_path) or ".", exist_ok=True)
        motion_tracks_writer = VideoWriter(cfg.output_motion_tracks_debug_path, fps, w, h, cfg)

    # Preprocessing mode
    use_cuda = cfg.enable_preprocess and is_cuda and HAS_CUDA
    if use_cuda:
        print("[preprocess] CUDA S+V motion path")
        cuda_device = torch.device(f"cuda:{device_index}")
    else:
        print("[preprocess] CPU S+V motion path" if cfg.enable_preprocess else "[preprocess] disabled")

    # Threaded frame reader - overlaps decode with GPU inference
    reader = ThreadedFrameReader(cap, prefetch=max(2, int(cfg.frame_reader_prefetch)))
    frame_curr = reader.read()
    frame_next = reader.read() if frame_curr is not None else None
    if frame_curr is None:
        print("[error] No frames in video")
        cap.release()
        return

    prev_v = prev_s = curr_v_t = curr_s_t = next_v_t = next_s_t = None
    prev_frame_v_t = prev_frame_s_t = None
    curr_frame_gpu_t = next_frame_gpu_t = prev_frame_gpu_t = None
    frame_uploader = None
    if use_cuda:
        try:
            frame_uploader = _PinnedFrameUploader(frame_curr.shape[0], frame_curr.shape[1], cuda_device)
        except Exception:
            frame_uploader = None
        curr_frame_gpu_t = _cuda_frame_to_chw_f32(frame_curr, cuda_device, uploader=frame_uploader)
        curr_v_t, curr_s_t = _cuda_vs_tensors(None, cuda_device, gpu_tensor=curr_frame_gpu_t)
        if frame_next is not None:
            next_frame_gpu_t = _cuda_frame_to_chw_f32(frame_next, cuda_device, uploader=frame_uploader)
            next_v_t, next_s_t = _cuda_vs_tensors(None, cuda_device, gpu_tensor=next_frame_gpu_t)

    frame_prev_cpu = None
    hsv_prev = hsv_curr = hsv_next = None
    master_bg_v = master_bg_s = master_var_v = master_var_s = master_hsv = None
    if not use_cuda and cfg.enable_preprocess:
        hsv_curr = cv2.cvtColor(frame_curr, cv2.COLOR_BGR2HSV)
        if frame_next is not None:
            hsv_next = cv2.cvtColor(frame_next, cv2.COLOR_BGR2HSV)
    if info_timing:
        timing["init_total"] = time.perf_counter() - init_perf_t0
    pass1_perf_t0 = time.perf_counter() if info_timing else 0.0

    # ==============================================================
    # PASS 1 - Collect detections, preprocess frames, store metadata
    # ==============================================================
    print("[pass 1] Collecting detections...")
    all_frame_dets = []       # per-frame list of (bbox, conf)
    all_boost_masks = []      # per-frame boost mask (or None)
    all_raw_motions = []      # per-frame raw motion (or None)
    all_rois = []             # per-frame ROI boundary (or None)
    all_ghost_rois = []       # per-frame ghost-deleted ROI boxes (shown red in debug)
    all_player_boxes = []     # per-frame player bboxes
    all_player_boxes_preproc = []  # per-frame player boxes used during preprocessing
    all_court_kps = []        # per-frame court keypoints
    frame_idx = 0

    # ROI motion tracker - limits motion/CC to region near ball
    roi_tracker = ROIMotionTracker(cfg, w, h, fps)
    roi_used_count = 0
    fullframe_count = 0
    side_mask_cache = None
    side_mask_cache_key = None
    protect_mask_cuda_cache = None
    protect_mask_cuda_cache_key = None
    prev_boost_for_flicker = None
    prev_raw_motion_cuda = None  # CUDA boolean mask from previous frame - avoids CPU -> GPU re-upload in WTA
    prev_raw_motion_u8 = None    # CPU uint8 mask from previous frame for HSV WTA background
    collect_motion_stats = bool(cfg.save_motion_debug)
    motion_raw_px_before_exclude = 0
    motion_boost_px_before_exclude = 0
    motion_raw_px_after_exclude = 0
    motion_boost_px_after_exclude = 0
    motion_frames_raw = 0
    motion_frames_boost = 0
    pre_cuda_perf: Optional[Dict[str, float]] = {"pre_cuda_total": 0.0, "pre_cuda_raw_d2h": 0.0, "pre_cuda_cc_filter": 0.0} if info_timing else None

    # Skip-frame YOLO state
    skip_n = max(1, int(cfg.skip_frame_yolo))
    skip_require_roi = bool(cfg.skip_frame_require_roi)
    skip_yolo_saved = 0       # count of skipped YOLO frames
    skip_preprocess_dim = bool(cfg.skip_preprocess_dim)
    last_aux_detect_frame = -10**9
    aux_force_interval = max(1, int(cfg.aux_force_interval))
    use_cuda_pre_frame = bool(use_cuda and detector.supports_cuda_frame())
    can_overlap_ball_inference = _detector_can_overlap(detector)
    if use_cuda_pre_frame:
        print("[pass 1] CUDA zero-copy preprocess->YOLO path enabled")
    cache_pass2_frames = None
    if needs_pass2_outputs and cfg.cache_input_frames_pass2 and total > 0:
        est_mb = (float(total) * float(w) * float(h) * 3.0) / (1024.0 * 1024.0)
        if est_mb <= float(max(1, int(cfg.pass2_cache_max_mb))):
            cache_pass2_frames = []
            print(f"[pass 1] RAM frame cache enabled ({est_mb:.0f} MiB est)")
        else:
            print(f"[pass 1] RAM frame cache disabled ({est_mb:.0f} MiB > {int(cfg.pass2_cache_max_mb)} MiB)")

    pending_det = None

    def _protect_mask_key(court_kps_local):
        return tuple(
            int(round(float(v) * 4.0))
            for v in (court_kps_local or [])
        )

    def _commit_frame(record, dets):
        if info_timing:
            store_t0 = time.perf_counter()
        dropped_ghost_rois = roi_tracker.update_from_dets(dets, record["frame_idx"])
        all_frame_dets.append(dets)
        all_boost_masks.append(_pack_mask_u8(record["boost_mask_u8"], roi=record["roi_for_pack"]))
        all_raw_motions.append(_pack_mask_u8(record["raw_motion_u8"], roi=record["roi_for_pack"]))
        
        # Retroactively delete motion from ghost tracks that timed out
        if dropped_ghost_rois:
            for fr_idx, rx1, ry1, rx2, ry2 in dropped_ghost_rois:
                # We only delete strictly historical frames, not the current one being packed
                if 0 <= fr_idx < len(all_raw_motions) and fr_idx != record["frame_idx"]:
                    # (FPS FIX) We no longer unpack/pack masks here retroactively frame-by-frame.
                    # We only record the ghost ROI boxes, and do ONE batched mask erase at the end of Pass 1.
                                
                    # For all_rois (visual boxes on the guide video)
                    # We need to find the specific ROI in the list that matches these bounds
                    if all_rois[fr_idx] is not None:
                        new_rois = []
                        for r_box in all_rois[fr_idx]:
                            # Allow a tiny bit of float/int drift, though they should be identical ints
                            if abs(r_box[0]-rx1)<=2 and abs(r_box[1]-ry1)<=2 and abs(r_box[2]-rx2)<=2 and abs(r_box[3]-ry2)<=2:
                                # Record the ghost box for debug visualization and batched deletion
                                while len(all_ghost_rois) <= fr_idx:
                                    all_ghost_rois.append([])
                                all_ghost_rois[fr_idx].append((rx1, ry1, rx2, ry2))
                                continue  # This is the ghost ROI, delete it!
                            new_rois.append(r_box)
                        
                        all_rois[fr_idx] = new_rois if len(new_rois) > 0 else None

        all_rois.append(record.get("rois_list"))
        all_ghost_rois.append([])  # placeholder; ghosts filled retroactively
        all_player_boxes.append(record["player_dict"])
        all_player_boxes_preproc.append(record["player_boxes"])
        all_court_kps.append(record["court_kps"])
        if yolo_dbg_writer is not None:
            # Stream debug-preprocessed frames immediately to avoid huge RAM growth.
            if record["pre_frame"] is not None:
                yolo_dbg_writer.write(record["pre_frame"])
            elif record["pre_frame_cuda"] is not None:
                yolo_dbg_writer.write(record["pre_frame_cuda"].detach().cpu().numpy())
        if info_timing:
            timing["pass1_store"] = timing.get("pass1_store", 0.0) + (time.perf_counter() - store_t0)

    def _finish_pending(pending):
        if info_timing:
            det_t0 = time.perf_counter()
        dets = detector.detect_async_finish(pending["handle"])
        if info_timing:
            timing["pass1_ball_detect"] = timing.get("pass1_ball_detect", 0.0) + (time.perf_counter() - det_t0)
        _commit_frame(pending["record"], dets)

    while frame_curr is not None:
        # Keep previous pending handle; finish it after current preprocess/start
        # so frame N preprocess overlaps frame N-1 inference.
        prev_pending = pending_det
        pending_det = None

        raw_motion_u8 = boost_mask_u8 = None
        pre_frame_cuda = None
        pre_frame = frame_curr
        det_pending = None
        player_pending = None
        roi_for_pack = None

        if cache_pass2_frames is not None:
            try:
                cache_pass2_frames.append(frame_curr)
            except MemoryError:
                cache_pass2_frames = None
                print("[pass 1] RAM frame cache disabled (MemoryError)")

        # Update Weighted Temporal Average Backgrounds
        if use_cuda:
            if master_bg_v is None:
                # Initialize CUDA background at raw S+V values.
                master_bg_v = curr_v_t.clone()
                master_bg_s = curr_s_t.clone()

                thr = float(cfg.motion_thresh) / 255.0
                master_var_v = torch.full_like(master_bg_v, thr**2)
                master_var_s = torch.full_like(master_bg_s, thr**2)
            else:
                v_diff_sq = (curr_v_t - master_bg_v) ** 2
                s_diff_sq = (curr_s_t - master_bg_s) ** 2

                # Slow background updates where raw motion was detected last frame.
                motion_freeze_alpha = float(getattr(cfg, 'motion_freeze_alpha', 0.004))
                if prev_raw_motion_cuda is not None:
                    alpha_mask = torch.where(prev_raw_motion_cuda, motion_freeze_alpha, float(cfg.wta_alpha))
                else:
                    alpha_mask = float(cfg.wta_alpha)

                master_var_v.mul_(1.0 - alpha_mask).add_(alpha_mask * v_diff_sq)
                master_var_s.mul_(1.0 - alpha_mask).add_(alpha_mask * s_diff_sq)
                master_bg_v.mul_(1.0 - alpha_mask).add_(curr_v_t * alpha_mask)
                master_bg_s.mul_(1.0 - alpha_mask).add_(curr_s_t * alpha_mask)
        elif cfg.enable_preprocess and hsv_curr is not None:
            if master_hsv is None:
                master_hsv = hsv_curr.copy().astype(np.float32)
            else:
                if prev_raw_motion_u8 is not None:
                    # CPU HSV background: mirror CUDA behavior with reduced but non-zero
                    # updates at previous-frame motion pixels to avoid permanent ghosts.
                    motion_freeze_alpha = float(getattr(cfg, 'motion_freeze_alpha', 0.004))
                    alpha_arr = np.full_like(master_hsv[:, :, 0], cfg.wta_alpha, dtype=np.float32)
                    alpha_arr[prev_raw_motion_u8 > 0] = motion_freeze_alpha
                    alpha_arr = np.stack([alpha_arr] * 3, axis=-1)
                    master_hsv = master_hsv * (1.0 - alpha_arr) + hsv_curr.astype(np.float32) * alpha_arr
                else:
                    cv2.accumulateWeighted(hsv_curr, master_hsv, cfg.wta_alpha)

        # Detect ball - skip-frame YOLO for speed
        # When skip_n > 1: run YOLO every Nth frame, selector interpolates short gaps.
        _do_yolo = True
        if skip_n > 1 and frame_idx % skip_n != 0:
            # Only skip when ROI tracker says ball is visible (smooth motion)
            if skip_require_roi and any(
                getattr(track, "frames_since_det", 10**9) <= 3
                for track in getattr(roi_tracker, "tracks", [])
            ):
                _do_yolo = False
            elif not skip_require_roi:
                _do_yolo = False

        # Player + court detection: decouple from critical path by syncing with YOLO frames.
        if info_timing:
            aux_t0 = time.perf_counter()
        do_aux_detect = True
        if cfg.aux_detect_on_yolo_frames and not _do_yolo:
            do_aux_detect = (frame_idx - last_aux_detect_frame) >= aux_force_interval
        if do_aux_detect:
            court_kps = court_det.detect(frame_curr, frame_idx=frame_idx)
            player_det.set_court_keypoints(court_kps)
            player_pending = player_det.detect_async_start(
                frame_curr,
                frame_idx,
                frame_gpu_t=curr_frame_gpu_t if use_cuda else None,
            )
            player_boxes = [list(pb) for pb in player_det.cached_boxes] if player_det.cached_boxes else []
            last_aux_detect_frame = frame_idx
        else:
            court_kps = court_det.keypoints
            player_det.set_court_keypoints(court_kps)
            player_boxes = [list(pb) for pb in player_det.cached_boxes] if player_det.cached_boxes else []
        if info_timing:
            timing["pass1_aux_detect"] = timing.get("pass1_aux_detect", 0.0) + (time.perf_counter() - aux_t0)
        if info_timing:
            pre_t0 = time.perf_counter()
        if cfg.enable_preprocess:
            if info_timing:
                pre_mask_t0 = time.perf_counter()
            # Build court-side protect mask once via cache and reuse everywhere.
            if court_kps is not None and len(court_kps) >= 16:
                try:
                    side_key = tuple(int(round(float(court_kps[i]) * 4.0)) for i in range(16))
                except Exception:
                    side_key = None
            else:
                side_key = None
            if side_key != side_mask_cache_key:
                side_mask_cache = build_court_side_protect_mask(
                    frame_curr.shape[0], frame_curr.shape[1], court_kps
                )
                side_mask_cache_key = side_key
            side_mask = side_mask_cache
            protect_mask = side_mask
            if info_timing:
                timing["pass1_pre_mask_build"] = timing.get("pass1_pre_mask_build", 0.0) + (time.perf_counter() - pre_mask_t0)

            if use_cuda:
                protect_mask_cuda = None
                if protect_mask is not None:
                    protect_key = _protect_mask_key(court_kps)
                    if (protect_mask_cuda_cache is None) or (protect_key != protect_mask_cuda_cache_key):
                        protect_mask_cuda_cache = torch.from_numpy(protect_mask).to(cuda_device)
                        protect_mask_cuda_cache_key = protect_key
                    protect_mask_cuda = protect_mask_cuda_cache
                else:
                    protect_mask_cuda_cache = None
                    protect_mask_cuda_cache_key = None

                rois, rois_visual = roi_tracker.get_rois(frame_idx)
                if rois:
                    min_x = min(r[0] for r in rois)
                    min_y = min(r[1] for r in rois)
                    max_x = max(r[2] for r in rois)
                    max_y = max(r[3] for r in rois)
                    roi_for_pack = (min_x, min_y, max_x, max_y)
                    roi_used_count += len(rois)
                else:
                    roi_for_pack = None
                    fullframe_count += 1
                # Skip cosmetic dim_static when ball is visible - saves a GPU round-trip
                _skip_dim = (skip_preprocess_dim and
                             len(roi_tracker.tracks) > 0 and
                             roi_tracker.ball_visible)
                need_cpu_pre_frame = bool((yolo_dbg_writer is not None) or (not use_cuda_pre_frame))
                pre_frame, raw_motion_u8, boost_mask_u8, _, _, pre_frame_cuda = \
                    preprocess_frame_cuda(frame_curr, master_bg_v, master_bg_s,
                                          master_var_v, master_var_s, cfg,
                                          player_bboxes=player_boxes,
                                          court_keypoints=court_kps,
                                          protect_mask_cached=protect_mask,
                                          rois=rois,
                                          skip_dim=_skip_dim,
                                          return_cuda_frame=bool(use_cuda_pre_frame and _do_yolo),
                                          need_cpu_frame=need_cpu_pre_frame,
                                          frame_gpu_t=curr_frame_gpu_t,
                                          curr_v_cached=curr_v_t,
                                          curr_s_cached=curr_s_t,
                                          prev_frame_gpu_t=prev_frame_gpu_t,
                                          prev_frame_v_cached=prev_frame_v_t,
                                          prev_frame_s_cached=prev_frame_s_t,
                                          next_frame_gpu_t=next_frame_gpu_t,
                                          next_v_cached=next_v_t,
                                          next_s_cached=next_s_t,
                                          protect_mask_cuda_cached=protect_mask_cuda,
                                          perf=pre_cuda_perf)
            else:
                rois, rois_visual = roi_tracker.get_rois(frame_idx) if cfg.roi_motion_enabled else (None, None)
                if rois:
                    min_x = min(r[0] for r in rois)
                    min_y = min(r[1] for r in rois)
                    max_x = max(r[2] for r in rois)
                    max_y = max(r[3] for r in rois)
                    roi_for_pack = (min_x, min_y, max_x, max_y)
                    roi_used_count += len(rois)
                else:
                    roi_for_pack = None
                    fullframe_count += 1

                if master_hsv is not None:
                    master_hsv_u8 = master_hsv.astype(np.uint8)
                    if rois:
                        rm_full = np.zeros((frame_curr.shape[0], frame_curr.shape[1]), dtype=np.uint8)
                        rm_ungated_full = np.zeros_like(rm_full)
                        for r in rois:
                            rx1, ry1, rx2, ry2 = r
                            rm_roi_ungated = compute_motion_sv_from_hsv(
                                master_hsv_u8[ry1:ry2, rx1:rx2], hsv_curr[ry1:ry2, rx1:rx2], cfg.motion_thresh)
                            rm_roi = refine_raw_motion_temporal_cpu(
                                rm_roi_ungated,
                                frame_prev_cpu[ry1:ry2, rx1:rx2] if frame_prev_cpu is not None else None,
                                frame_curr[ry1:ry2, rx1:rx2],
                                frame_next[ry1:ry2, rx1:rx2] if frame_next is not None else None,
                                cfg,
                            )
                            np.maximum(rm_full[ry1:ry2, rx1:rx2], rm_roi, out=rm_full[ry1:ry2, rx1:rx2])
                            np.maximum(rm_ungated_full[ry1:ry2, rx1:rx2], rm_roi_ungated, out=rm_ungated_full[ry1:ry2, rx1:rx2])
                        rm = rm_full
                        rm_ungated = rm_ungated_full
                    else:
                        rm_ungated = compute_motion_sv_from_hsv(master_hsv_u8, hsv_curr, cfg.motion_thresh)
                        rm = refine_raw_motion_temporal_cpu(rm_ungated, frame_prev_cpu, frame_curr, frame_next, cfg)
                else:
                    rm = None
                    rm_ungated = None

                # Two boost masks:
                #   boost_mask_u8 - narrow, gated source -> returned to selector
                #     (preserves selector precision; frame-870-class FPs stay suppressed).
                #   boost_yolo_u8 - wide, ungated source -> drives the HSV brightening
                #     the YOLO input sees (recall-positive: rescues YOLO misses on frames
                #     where the temporal/color gate would have suppressed a real ball blob).
                raw_motion_u8 = rm
                def _build_boost(src):
                    if src is None:
                        return None
                    if rois:
                        out = np.zeros_like(src)
                        for r in rois:
                            rx1, ry1, rx2, ry2 = r
                            roi_slice = src[ry1:ry2, rx1:rx2]
                            if roi_slice.max() > 0:
                                filtered_roi = filter_boost_mask(
                                    roi_slice, cfg.boost_min_blob_area, cfg.boost_max_blob_area, cfg,
                                    player_bboxes=player_boxes)
                                np.maximum(out[ry1:ry2, rx1:rx2], filtered_roi, out=out[ry1:ry2, rx1:rx2])
                        return out if out.max() > 0 else None
                    result = filter_boost_mask(
                        src, cfg.boost_min_blob_area, cfg.boost_max_blob_area, cfg,
                        player_bboxes=player_boxes)
                    if result is None or result.max() == 0:
                        return None
                    return result

                boost_mask_u8 = _build_boost(rm)
                if rm_ungated is not None and rm_ungated is not rm:
                    boost_yolo_u8 = _build_boost(rm_ungated)
                else:
                    boost_yolo_u8 = boost_mask_u8

                pre_frame = preprocess_frame(frame_curr, raw_motion_u8, boost_yolo_u8, cfg,
                                              player_bboxes=player_boxes,
                                              court_keypoints=court_kps,
                                              hsv_cached=hsv_curr,
                                              protect_mask_cached=protect_mask)

            # Start current frame inference as soon as detector input is ready.
            # With per-pending CUDA events + output slots, this is safe and allows
            # overlap with post-mask CPU work below.
            if _do_yolo:
                if prev_pending is not None and not can_overlap_ball_inference:
                    _finish_pending(prev_pending)
                    prev_pending = None
                det_input = pre_frame_cuda if pre_frame_cuda is not None else pre_frame
                det_pending = detector.detect_async_start(det_input)

            # Don't exclude any court regions from motion detection - let the selector
            # decide what to do with sideline/alley motion (previously protect_mask
            # was blanking the outer left/right zone, blocking detection there).
            side_mask = None

            if info_timing:
                postmask_t0 = time.perf_counter()
            if collect_motion_stats and raw_motion_u8 is not None:
                motion_raw_px_before_exclude += int(cv2.countNonZero(raw_motion_u8))
                motion_frames_raw += 1
            if collect_motion_stats and boost_mask_u8 is not None:
                motion_boost_px_before_exclude += int(cv2.countNonZero(boost_mask_u8))
                motion_frames_boost += 1
            raw_motion_u8 = apply_exclude_mask_u8(raw_motion_u8, side_mask)
            boost_mask_u8 = apply_exclude_mask_u8(boost_mask_u8, side_mask)
            if cfg.motion_flicker_suppress:
                keep_mask_u8 = None
                if len(roi_tracker.tracks) > 0:
                    r_keep = int(round(max(4.0, cfg.motion_flicker_keep_radius_frac * roi_tracker.diag)))
                    if r_keep > 0:
                        keep_mask_u8 = np.zeros((frame_curr.shape[0], frame_curr.shape[1]), dtype=np.uint8)
                        for t in roi_tracker.tracks:
                            pred = t.predicted_center(roi_tracker.phys_cfg)
                            pcx = int(np.clip(round(pred[0]), 0, frame_curr.shape[1] - 1))
                            pcy = int(np.clip(round(pred[1]), 0, frame_curr.shape[0] - 1))
                            cv2.circle(keep_mask_u8, (pcx, pcy), r_keep, 255, -1, cv2.LINE_AA)
                boost_mask_u8 = suppress_flicker_components(
                    boost_mask_u8, prev_boost_for_flicker, keep_mask_u8, cfg
                )
                if boost_mask_u8 is not None and boost_mask_u8.max() > 0:
                    prev_boost_for_flicker = boost_mask_u8.copy()
                else:
                    prev_boost_for_flicker = None

            if collect_motion_stats and raw_motion_u8 is not None:
                motion_raw_px_after_exclude += int(cv2.countNonZero(raw_motion_u8))
            if collect_motion_stats and boost_mask_u8 is not None:
                motion_boost_px_after_exclude += int(cv2.countNonZero(boost_mask_u8))
            # Keep a CUDA copy of the current motion mask for next frame's WTA freeze.
            # This avoids re-uploading raw_motion_u8 (CPU -> CUDA) in the next iteration.
            if use_cuda:
                if raw_motion_u8 is not None:
                    prev_raw_motion_cuda = torch.from_numpy(raw_motion_u8).to(cuda_device, non_blocking=True) > 0
                else:
                    prev_raw_motion_cuda = None
            else:
                # CPU path: keep previous-frame raw motion for HSV WTA background updates.
                prev_raw_motion_u8 = raw_motion_u8.copy() if raw_motion_u8 is not None else None
            if info_timing:
                timing["pass1_pre_postmask"] = timing.get("pass1_pre_postmask", 0.0) + (time.perf_counter() - postmask_t0)

        if player_pending is not None:
            if info_timing:
                aux_finish_t0 = time.perf_counter()
            player_boxes = player_det.detect_async_finish(player_pending)
            player_pending = None
            if info_timing:
                timing["pass1_aux_detect"] = timing.get("pass1_aux_detect", 0.0) + (time.perf_counter() - aux_finish_t0)
        if info_timing:
            timing["pass1_preprocess"] = timing.get("pass1_preprocess", 0.0) + (time.perf_counter() - pre_t0)

        # Finish previous frame inference after current frame preprocess.
        if prev_pending is not None:
            _finish_pending(prev_pending)

        # Detect ball - skip-frame YOLO for speed.
        # Cross-frame overlap: launch current frame now, finish it next iteration.
        if _do_yolo:
            if det_pending is None:
                det_input = pre_frame_cuda if pre_frame_cuda is not None else pre_frame
                if info_timing:
                    det_t0 = time.perf_counter()
                det_pending = detector.detect_async_start(det_input)
                if info_timing:
                    timing["pass1_ball_detect"] = timing.get("pass1_ball_detect", 0.0) + (time.perf_counter() - det_t0)
            pending_det = {
                "handle": det_pending,
                "record": {
                    "frame_idx": frame_idx,
                    "raw_motion_u8": raw_motion_u8,
                    "boost_mask_u8": boost_mask_u8,
                    "roi_for_pack": roi_for_pack,
                    "rois_list": rois_visual if rois_visual is not None else roi_tracker.last_rois,
                    "player_dict": dict(player_det.get_player_dict()),
                    "player_boxes": [list(pb) for pb in player_boxes] if player_boxes else [],
                    "court_kps": court_kps,
                    "pre_frame": pre_frame,
                    "pre_frame_cuda": pre_frame_cuda,
                },
            }
        else:
            skip_yolo_saved += 1
            _commit_frame(
                {
                    "frame_idx": frame_idx,
                    "raw_motion_u8": raw_motion_u8,
                    "boost_mask_u8": boost_mask_u8,
                    "roi_for_pack": roi_for_pack,
                    "rois_list": rois_visual if rois_visual is not None else roi_tracker.last_rois,
                    "player_dict": dict(player_det.get_player_dict()),
                    "player_boxes": [list(pb) for pb in player_boxes] if player_boxes else [],
                    "court_kps": court_kps,
                    "pre_frame": pre_frame,
                    "pre_frame_cuda": pre_frame_cuda,
                },
                [],
            )

        frame_idx += 1
        if cfg.progress_every and frame_idx % cfg.progress_every == 0:
            elapsed = time.time() - t0
            print(f"  [pass1 {frame_idx}/{total}] {frame_idx/elapsed:.1f} fps")
        
        # Slide window
        if info_timing:
            slide_t0 = time.perf_counter()
        frame_prev_for_slide = frame_curr
        frame_curr = frame_next
        frame_next = reader.read()
        if use_cuda:
            prev_frame_gpu_t = curr_frame_gpu_t
            prev_frame_v_t, prev_frame_s_t = curr_v_t, curr_s_t
            curr_v_t, curr_s_t = next_v_t, next_s_t
            curr_frame_gpu_t = next_frame_gpu_t
            if frame_next is not None:
                next_frame_gpu_t = _cuda_frame_to_chw_f32(frame_next, cuda_device, uploader=frame_uploader)
                next_v_t, next_s_t = _cuda_vs_tensors(None, cuda_device, gpu_tensor=next_frame_gpu_t)
            else:
                next_v_t = next_s_t = None
                next_frame_gpu_t = None
        elif cfg.enable_preprocess:
            frame_prev_cpu = frame_prev_for_slide
            hsv_prev = hsv_curr
            hsv_curr = hsv_next
            hsv_next = cv2.cvtColor(frame_next, cv2.COLOR_BGR2HSV) if frame_next is not None else None
        if info_timing:
            timing["pass1_slide"] = timing.get("pass1_slide", 0.0) + (time.perf_counter() - slide_t0)

    # Drain last pending YOLO inference.
    if pending_det is not None:
        _finish_pending(pending_det)
        pending_det = None

    reader.release()
    N = frame_idx
    if info_timing:
        timing["pass1_total"] = time.perf_counter() - pass1_perf_t0
        if pre_cuda_perf is not None and float(pre_cuda_perf.get("pre_cuda_total", 0.0)) > 0.0:
            pre_total = float(pre_cuda_perf.get("pre_cuda_total", 0.0))
            pre_d2h = float(pre_cuda_perf.get("pre_cuda_raw_d2h", 0.0))
            pre_cc = float(pre_cuda_perf.get("pre_cuda_cc_filter", 0.0))
            pre_other = max(0.0, pre_total - pre_d2h - pre_cc)
            timing["pass1_pre_cuda_total"] = pre_total
            timing["pass1_pre_cuda_d2h"] = pre_d2h
            timing["pass1_pre_cuda_cc"] = pre_cc
            timing["pass1_pre_cuda_gpu_other"] = pre_other
    t_pass1 = time.time() - t0
    det_frames = sum(1 for d in all_frame_dets if d)
    print(f"[pass 1] Done: {N} frames in {t_pass1:.1f}s "
          f"({N/t_pass1:.1f} fps), detections in {det_frames} frames")
    if skip_n > 1:
        print(f"[pass 1] Skip-frame YOLO: {skip_yolo_saved}/{N} frames skipped "
              f"(every {skip_n}th frame, {100.0*skip_yolo_saved/max(N,1):.1f}% saved)")
    if cfg.roi_motion_enabled:
        print(f"[pass 1] ROI motion: {roi_used_count} ROI frames, "
              f"{fullframe_count} full-frame frames")
    if cfg.enable_preprocess and collect_motion_stats and motion_frames_raw > 0:
        raw_before_avg = motion_raw_px_before_exclude / max(motion_frames_raw, 1)
        raw_after_avg = motion_raw_px_after_exclude / max(motion_frames_raw, 1)
        raw_keep = 100.0 * motion_raw_px_after_exclude / max(motion_raw_px_before_exclude, 1)
        boost_before_avg = motion_boost_px_before_exclude / max(motion_frames_boost, 1)
        boost_after_avg = motion_boost_px_after_exclude / max(motion_frames_boost, 1)
        boost_keep = 100.0 * motion_boost_px_after_exclude / max(motion_boost_px_before_exclude, 1)
        print(
            "[pass 1] Motion pixels/frame (avg): "
            f"raw {raw_before_avg:.0f}->{raw_after_avg:.0f} ({raw_keep:.1f}% kept), "
            f"boost {boost_before_avg:.0f}->{boost_after_avg:.0f} ({boost_keep:.1f}% kept)"
        )
        
    print("[pass 1] Executing batched ghost ROI deletion...")
    batch_t0 = time.perf_counter()
    erased_frames = 0
    for fr_idx, ghosts in enumerate(all_ghost_rois):
        if not ghosts:
            continue
            
        rm_obj = all_raw_motions[fr_idx] if 0 <= fr_idx < len(all_raw_motions) else None
        bm_obj = all_boost_masks[fr_idx] if 0 <= fr_idx < len(all_boost_masks) else None
        
        if rm_obj is None and bm_obj is None:
            continue
        
        erased_frames += 1
        rm_unpacked = _unpack_mask_u8(rm_obj) if rm_obj is not None else None
        bm_unpacked = _unpack_mask_u8(bm_obj) if bm_obj is not None else None
        
        any_change = False
        for rx1, ry1, rx2, ry2 in ghosts:
            if rm_unpacked is not None and np.any(rm_unpacked[ry1:ry2, rx1:rx2]):
                rm_unpacked[ry1:ry2, rx1:rx2] = 0
                any_change = True
            if bm_unpacked is not None and np.any(bm_unpacked[ry1:ry2, rx1:rx2]):
                bm_unpacked[ry1:ry2, rx1:rx2] = 0
                any_change = True
        
        if not any_change:
            continue
                    
        if rm_unpacked is not None:
            if isinstance(rm_obj, tuple) and rm_obj[0] == "roi":
                all_raw_motions[fr_idx] = _pack_mask_u8(rm_unpacked, roi=rm_obj[4:8])
            else:
                all_raw_motions[fr_idx] = _pack_mask_u8(rm_unpacked, roi=None)
                    
        if bm_unpacked is not None:
            if isinstance(bm_obj, tuple) and bm_obj[0] == "roi":
                all_boost_masks[fr_idx] = _pack_mask_u8(bm_unpacked, roi=bm_obj[4:8])
            else:
                all_boost_masks[fr_idx] = _pack_mask_u8(bm_unpacked, roi=None)

    print(f"[pass 1] Batched ghost deletion cleaned {erased_frames} frames in {time.perf_counter() - batch_t0:.2f}s")
    
    # ==============================================================
    # SELECTOR - pick the in-play ball track
    # ==============================================================
    selector_perf_t0 = time.perf_counter() if info_timing else 0.0
    selector_poly_t0 = time.perf_counter() if info_timing else 0.0
    court_poly, last_kps = _build_court_polygon(all_court_kps, w, h)
    if info_timing:
        timing["selector_build_poly"] = timing.get("selector_build_poly", 0.0) + (time.perf_counter() - selector_poly_t0)

    print("[selector] Running ball-in-play selection...")
    selector_select_t0 = time.perf_counter() if info_timing else 0.0
    per_frame, chosen_track, all_tracks, motion_tracks_dbg = select_ball_in_play(
        all_frame_dets, fps, w, h,
        court_polygon=court_poly,
        boost_masks=all_boost_masks,
        raw_motions=all_raw_motions,
        player_boxes_by_frame=all_player_boxes,
        court_keypoints=last_kps,
        emit_guide_debug_meta=bool(cfg.save_guide_video),
        debug=bool(cfg.print_selector_tracks))
    if info_timing:
        timing["selector_select"] = timing.get("selector_select", 0.0) + (time.perf_counter() - selector_select_t0)

    print(f"[selector] {len(all_tracks)} total tracks")
    selector_post_t0 = time.perf_counter() if info_timing else 0.0
    if cfg.print_selector_tracks:
        _print_selector_track_summary(
            all_tracks,
            chosen_track,
            total_frames=N,
            limit=max(int(cfg.selector_track_limit), 0)
        )
    dropped_carry, dropped_motion = _drop_unattached_soft_runs(per_frame, cfg, w, h)
    if dropped_carry > 0:
        print(
            f"[selector] Dropped {dropped_carry} unattached carry frames "
            f"(attach <= {cfg.carry_attach_max_frac:.3f}*diag)"
        )
    if dropped_motion > 0:
        print(
            f"[selector] Dropped {dropped_motion} unattached motion frames "
            f"(attach <= {cfg.carry_attach_max_frac * 1.55:.3f}*diag)"
        )
    guide_interp_gap = max(
        max(1, int(cfg.guide_interp_max_gap)),
        min(24, max(10, int(round(float(fps) * 0.60))))
    )
    guide_map = _build_display_guide(
        chosen_track,
        N,
        max_interp_gap=guide_interp_gap,
        frame_w=w,
        frame_h=h
    )

    # Pre-process top candidate tracks for visualization (Pass 2)
    # Map: track_id -> { frame_idx: (cx, cy, is_obs) }
    vis_tracks = {}
    vis_track_list = sorted(all_tracks, key=lambda t: float(t.score), reverse=True)
    for trk in vis_track_list:
        tid = int(trk.track_id)
        vis_tracks[tid] = {}
        for o in trk.observations:
            vis_tracks[tid][o.frame] = (o.cx, o.cy, True)
    # Debug-display fallback: if chosen-track guide is sparse, fill from final per-frame
    # selector output so late-rally segments remain visible in guide video.
    if guide_writer is not None:
        for gi, gr in enumerate(per_frame):
            if gi in guide_map or gr is None or gr.cx is None or gr.cy is None:
                continue
            src = str(getattr(gr, "source", ""))
            exact = src in ("det", "motion", "guide")
            guide_map[gi] = (float(gr.cx), float(gr.cy), exact)

        # Note: we intentionally do NOT backfill guide_map from other (orange/blue) tracks.
        # The green guide must only reflect the chosen track's positions.  Non-chosen tracks
        # are already visible in the guide video via vis_tracks (their own colored dots/lines).
        # Backfilling would make the green guide "snap to" orange/blue positions in gaps,
        # which is the opposite of what we want (orange/blue should follow green, not vice versa).
    if info_timing:
        timing["selector_post"] = timing.get("selector_post", 0.0) + (time.perf_counter() - selector_post_t0)
        timing["selector_total"] = time.perf_counter() - selector_perf_t0

    # ==============================================================
    # PASS 2 - Render output video with debug visualization
    # ==============================================================
    video_outputs_enabled = needs_pass2_outputs
    if video_outputs_enabled:
        print("[pass 2] Rendering selected video output(s)...")
    else:
        print("[pass 2] Skipped (no video outputs selected)")
    pass2_perf_t0 = time.perf_counter() if info_timing else 0.0
    pass2_frames_rendered = 0
    if getattr(cfg, "save_tracking_video", True):
        os.makedirs(os.path.dirname(cfg.output_video) or ".", exist_ok=True)
        writer = VideoWriter(cfg.output_video, fps, w, h, cfg)
        print(f"[writer] Tracking encoder: {writer._encoder}")
    else:
        writer = None
    use_cached_frames = bool(video_outputs_enabled and cache_pass2_frames is not None and len(cache_pass2_frames) == N)
    if use_cached_frames:
        cap2 = None
        print(f"[pass 2] Using RAM frame cache ({len(cache_pass2_frames)} frames)")
    elif video_outputs_enabled:
        cap2 = cv2.VideoCapture(str(input_path))
        if cache_pass2_frames is not None:
            print("[pass 2] RAM frame cache incomplete; falling back to video decode")
    else:
        cap2 = None
    # list of (raw_x, raw_y, smooth_x, smooth_y, source, conf) or None for gaps
    trail = []
    trail_max = 50
    guide_trail = []  # list of (x, y, exact) or None for guide gaps
    guide_trail_max = max(40, trail_max + 20)
    last_guide_point_dbg = None  # (x, y, exact)
    # Display-only smoothing for the magenta motion-search center (orange MOT path).
    # This does not affect selector logic; it only stabilizes the debug overlay.
    last_motion_search_dbg = None  # (x, y)
    last_motion_search_dbg_frame = -1
    draw_main_search_regions = bool(getattr(cfg, "draw_search_regions", False))
    draw_main_ball_trail = bool(getattr(cfg, "draw_ball_trail", True))

    # Per-track short history for visualization: tid -> list of (x, y)
    vis_track_trails = {int(t.track_id): [] for t in vis_track_list}
    vis_trail_len = 120  # Long enough to show full rally segments in guide video
    # Pre-build score lookup for label rendering (avoids repeated attribute access per frame)
    vis_track_scores = {int(t.track_id): float(t.score) for t in vis_track_list}

    trail_kps = last_kps
    ground_model = None
    ground_model_kps_key = None
    pending_lost_gap = False
    # O(1) anchor state: latest estimated ball radius from green detections.
    cached_anchor_radius = float(max(cfg.ball_marker_min_radius, 4))
    cached_anchor_radius_cap = float(max(cfg.ball_marker_max_radius * 3, cfg.ball_marker_min_radius))

    raw_motion_tracks = []
    _dbg_protect_cache = None
    _dbg_protect_cache_key = None

    for fi in range(N if video_outputs_enabled else 0):
        if info_timing:
            read_t0 = time.perf_counter()
        if use_cached_frames:
            frame = cache_pass2_frames[fi]
        else:
            ret, frame = cap2.read()
            if not ret:
                break
        if info_timing:
            timing["pass2_read"] = timing.get("pass2_read", 0.0) + (time.perf_counter() - read_t0)

        if info_timing:
            render_t0 = time.perf_counter()
            guide_write_elapsed = 0.0
        dets = all_frame_dets[fi]
        frame_result = per_frame[fi]
        guide_state_result = frame_result
        display_result = None
        if frame_result is not None and not bool(getattr(frame_result, "debug_only", False)):
            display_result = frame_result
        frame_out = frame.copy()
        guide_frame = frame.copy() if guide_writer is not None else None
        court_kps = all_court_kps[fi]
        if court_kps and len(court_kps) >= 16:
            trail_kps = court_kps
        if trail_kps is not None and len(trail_kps) >= 16:
            try:
                kps_key = tuple(int(round(float(trail_kps[i]) * 4.0)) for i in range(16))
            except Exception:
                kps_key = None
        else:
            kps_key = None
        if kps_key != ground_model_kps_key:
            ground_model = _build_ground_projection_model(trail_kps, w, h)
            ground_model_kps_key = kps_key

        # Draw court
        if cfg.draw_court:
            if court_kps:
                court_det.keypoints = court_kps
                court_det.draw(frame_out)
                if guide_frame is not None:
                    court_det.draw(guide_frame)
        if ground_model is not None:
            _draw_homography_net_line(frame_out, ground_model)
            if guide_frame is not None:
                _draw_homography_net_line(guide_frame, ground_model)

        # Draw court polygon outline (thin grey)
        if court_poly is not None:
            cv2.drawContours(frame_out, [court_poly.astype(np.int32)], 0,
                             (100, 100, 100), 1)
            if guide_frame is not None:
                cv2.drawContours(guide_frame, [court_poly.astype(np.int32)], 0,
                                 (100, 100, 100), 1)


        # Draw players
        if cfg.draw_players:
            for pid, pbox in all_player_boxes[fi].items():
                if pbox is None or len(pbox) < 4:
                    continue
                px1, py1, px2, py2 = map(int, pbox)
                cv2.rectangle(frame_out, (px1, py1), (px2, py2), (0, 0, 255), 2)
                if guide_frame is not None:
                    cv2.rectangle(guide_frame, (px1, py1), (px2, py2), (0, 0, 255), 2)

        # Draw ALL raw detections (dim yellow thin boxes)
        for ball_box, ball_conf in dets:
            ix1, iy1, ix2, iy2 = map(int, ball_box)
            cv2.rectangle(frame_out, (ix1, iy1), (ix2, iy2), COLOR_RAW, 1)
            cv2.line(frame_out, (ix1, iy2), (ix2, iy2), (0, 0, 255), 1)
            if guide_frame is not None:
                cv2.rectangle(guide_frame, (ix1, iy1), (ix2, iy2), COLOR_RAW, 1)
                cv2.line(guide_frame, (ix1, iy2), (ix2, iy2), (0, 0, 255), 1)
                
        # Draw Motion ROIs (cleaned of ghost histories)
        if guide_frame is not None and all_rois[fi] is not None:
            for rx1, ry1, rx2, ry2 in all_rois[fi]:
                cv2.rectangle(guide_frame, (int(rx1), int(ry1)), (int(rx2), int(ry2)), (0, 255, 0), 2)
                cv2.putText(guide_frame, "ROI", (int(rx1), max(15, int(ry1) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Draw candidate tracks (circles + IDs)
        # Draw them underneath the chosen track result so the final choice pops on top.
        # ONLY draw on guide_frame to keep main output clean.
        if guide_frame is not None:
            for trk in vis_track_list:
                tid = int(trk.track_id)
                tdata = vis_tracks.get(tid, {})
                pt = tdata.get(fi)
                
                # Update trail
                is_active_frame = (pt is not None)
                if is_active_frame:
                    vis_track_trails[tid].append((pt[0], pt[1]))
                elif vis_track_trails[tid] and vis_track_trails[tid][-1] is not None:
                    # Add a break for gaps
                    vis_track_trails[tid].append(None)
                
                # Trim trail
                while len(vis_track_trails[tid]) > vis_trail_len:
                    vis_track_trails[tid].pop(0)
                    
                # Draw trail
                tcolor = _get_track_color(tid)
                trail_pts = vis_track_trails[tid]
                if len(trail_pts) > 1:
                    # Filter None
                    valid_segments = []
                    curr_seg = []
                    for p in trail_pts:
                        if p is None:
                            if len(curr_seg) > 1:
                                valid_segments.append(curr_seg)
                            curr_seg = []
                        else:
                            curr_seg.append(p)
                    if len(curr_seg) > 1:
                        valid_segments.append(curr_seg)
                        
                    for seg in valid_segments:
                        for i in range(1, len(seg)):
                            p0 = (int(seg[i-1][0]), int(seg[i-1][1]))
                            p1 = (int(seg[i][0]), int(seg[i][1]))
                            cv2.line(guide_frame, p0, p1, tcolor, 1, cv2.LINE_AA)

                # Draw current position and ID+score
                if is_active_frame:
                    cx, cy = int(pt[0]), int(pt[1])
                    is_chosen_trk = (chosen_track is not None and tid == int(chosen_track.track_id))
                    dot_r = 5 if is_chosen_trk else 3
                    dot_th = -1 if is_chosen_trk else 2
                    # Dark outline so label is readable on any background
                    cv2.circle(guide_frame, (cx, cy), dot_r + 2, (15, 15, 15), -1)
                    cv2.circle(guide_frame, (cx, cy), dot_r, tcolor, dot_th)
                    scr = vis_track_scores.get(tid, 0.0)
                    label = f"{tid}({scr:.0f})"
                    cv2.putText(guide_frame, label, (cx + 6, cy + 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (15, 15, 15), 2)
                    cv2.putText(guide_frame, label, (cx + 6, cy + 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, tcolor, 1)

        # Draw chosen result - color-coded by source
        if display_result is not None:
            rcx, rcy = int(display_result.cx), int(display_result.cy)
            trail_cx, trail_cy = rcx, rcy

            # Bottom-anchor for orange/blue paths using cached ball size.
            if display_result.source in ('motion', 'carry'):
                trail_cy = int(np.clip(
                    round(float(display_result.cy) + cached_anchor_radius), 0, h - 1))

            if display_result.source == 'det':
                # GREEN - YOLO detection
                marker_radius = 6
                if display_result.bbox:
                    bx1f, by1f, bx2f, by2f = map(float, display_result.bbox)
                    bx1, by1, bx2, by2 = map(int, (bx1f, by1f, bx2f, by2f))
                    # High-contrast box: dark halo + bright box + emphasized bottom edge.
                    cv2.rectangle(frame_out, (bx1, by1), (bx2, by2), (15, 15, 15), 4)
                    cv2.rectangle(frame_out, (bx1, by1), (bx2, by2), COLOR_DET, 2)
                    cv2.line(frame_out, (bx1, by2), (bx2, by2), (255, 255, 255), 2)
                    if guide_frame is not None:
                        cv2.rectangle(guide_frame, (bx1, by1), (bx2, by2), (15, 15, 15), 4)
                        cv2.rectangle(guide_frame, (bx1, by1), (bx2, by2), COLOR_DET, 2)
                        cv2.line(guide_frame, (bx1, by2), (bx2, by2), (255, 255, 255), 2)
                    # Anchor trail to bottom-center of the detected ball box.
                    trail_cx = int(np.clip(round((bx1f + bx2f) * 0.5), 0, w - 1))
                    trail_cy = int(np.clip(round(by2f), 0, h - 1))

                    bw = max(bx2f - bx1f, 1.0)
                    bh = max(by2f - by1f, 1.0)
                    det_anchor_radius = 0.5 * max(bw, bh)
                    cached_anchor_radius = float(np.clip(
                        0.80 * cached_anchor_radius + 0.20 * det_anchor_radius,
                        cfg.ball_marker_min_radius,
                        cached_anchor_radius_cap
                    ))

                    ball_px = max(int(round(bw)), int(round(bh)), 1)
                    marker_radius = int(np.clip(
                        round(ball_px * cfg.ball_marker_box_scale),
                        cfg.ball_marker_min_radius, cfg.ball_marker_max_radius))
                # Draw marker at measured center (not at bottom anchor) to avoid fake-ball look.
                cv2.circle(frame_out, (rcx, rcy), marker_radius + 2, (15, 15, 15), 2)
                cv2.circle(frame_out, (rcx, rcy), marker_radius, COLOR_DET, -1)
                cv2.putText(frame_out, "DET", (rcx + 10, rcy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_DET, 1)
                if guide_frame is not None:
                    cv2.circle(guide_frame, (rcx, rcy), marker_radius + 2, (15, 15, 15), 2)
                    cv2.circle(guide_frame, (rcx, rcy), marker_radius, COLOR_DET, -1)
                    cv2.putText(guide_frame, "DET", (rcx + 10, rcy - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_DET, 1)

            elif display_result.source == 'motion':
                # ORANGE - motion blob + search region
                m_rad = int(np.clip(round(cached_anchor_radius), 4, 12))
                cv2.circle(frame_out, (rcx, rcy), m_rad, COLOR_MOTION, -1)
                cv2.putText(frame_out, "MOT", (rcx + 10, rcy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_MOTION, 1)
                if guide_frame is not None:
                    cv2.circle(guide_frame, (rcx, rcy), m_rad, COLOR_MOTION, -1)
                    cv2.putText(guide_frame, "MOT", (rcx + 10, rcy - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_MOTION, 1)
                # Draw search region
                if display_result.search_radius > 0:
                    scx = int(display_result.search_cx)
                    scy = int(display_result.search_cy)
                    if last_motion_search_dbg is not None and last_motion_search_dbg_frame == (fi - 1):
                        pscx, pscy = last_motion_search_dbg
                        dx = scx - pscx
                        dy = scy - pscy
                        d = float((dx * dx + dy * dy) ** 0.5)
                        max_step_dbg = 64.0
                        if d > max_step_dbg and d > 1e-6:
                            s = max_step_dbg / d
                            scx = int(round(pscx + dx * s))
                            scy = int(round(pscy + dy * s))
                    last_motion_search_dbg = (scx, scy)
                    last_motion_search_dbg_frame = fi
                    if draw_main_search_regions:
                        cv2.circle(frame_out, (scx, scy),
                                   int(display_result.search_radius), COLOR_SEARCH, 1)
                        cv2.line(frame_out, (scx, scy), (rcx, rcy), COLOR_SEARCH, 1)
                        # Small cross at predicted position
                        cv2.drawMarker(frame_out, (scx, scy), COLOR_SEARCH,
                                       cv2.MARKER_CROSS, 12, 1)
                    if guide_frame is not None:
                        cv2.circle(guide_frame, (scx, scy),
                                   int(display_result.search_radius), COLOR_SEARCH, 1)
                        cv2.line(guide_frame, (scx, scy), (rcx, rcy), COLOR_SEARCH, 1)
                        cv2.drawMarker(guide_frame, (scx, scy), COLOR_SEARCH,
                                       cv2.MARKER_CROSS, 12, 1)
                else:
                    last_motion_search_dbg = None
                    last_motion_search_dbg_frame = -1

            elif display_result.source == 'interp':
                last_motion_search_dbg = None
                last_motion_search_dbg_frame = -1
                # YELLOW - prolonged guessed/stuck
                i_rad = int(np.clip(round(cached_anchor_radius), 4, 12))
                cv2.circle(frame_out, (rcx, rcy), i_rad, COLOR_INTERP, -1)
                if guide_frame is not None:
                    cv2.circle(guide_frame, (rcx, rcy), i_rad, COLOR_INTERP, -1)
            elif display_result.source == 'carry':
                last_motion_search_dbg = None
                last_motion_search_dbg_frame = -1
                # BLUE - short predicted carry when temporarily lost
                c_rad = int(np.clip(round(cached_anchor_radius), 4, 12))
                cv2.circle(frame_out, (rcx, rcy), c_rad, COLOR_CARRY, -1)
                cv2.putText(frame_out, "CAR", (rcx + 10, rcy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_CARRY, 1)
                if guide_frame is not None:
                    cv2.circle(guide_frame, (rcx, rcy), c_rad, COLOR_CARRY, -1)
                    cv2.putText(guide_frame, "CAR", (rcx + 10, rcy - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_CARRY, 1)

            # State circle (court-scaled radius) used by trail-only blue/yellow/green logic.
            if (
                display_result.source in ('det', 'carry', 'interp') and
                float(getattr(display_result, "search_radius", 0.0) or 0.0) > 0.0
            ):
                scx = int(round(float(getattr(display_result, "search_cx", rcx))))
                scy = int(round(float(getattr(display_result, "search_cy", rcy))))
                sr = int(round(float(getattr(display_result, "search_radius", 0.0))))
                if display_result.source == 'det':
                    scol = COLOR_DET
                elif display_result.source == 'carry':
                    scol = COLOR_CARRY
                else:
                    scol = COLOR_INTERP
                if draw_main_search_regions:
                    cv2.circle(frame_out, (scx, scy), sr, (15, 15, 15), 2, cv2.LINE_AA)
                    cv2.circle(frame_out, (scx, scy), sr, scol, 1, cv2.LINE_AA)
                if guide_frame is not None:
                    cv2.circle(guide_frame, (scx, scy), sr, (15, 15, 15), 2, cv2.LINE_AA)
                    cv2.circle(guide_frame, (scx, scy), sr, scol, 1, cv2.LINE_AA)

            # Piecewise arc smoothing:
            # smooth within segment, reset segment on jump or direction reversal.
            src = display_result.source
            conf = float(display_result.conf)
            prev, prev2 = _trail_prev2(trail)
            prev_src = None
            reset_smoothing = False
            if prev is not None:
                prev_src = prev[4]
                # Source transition: det resuming after carry/motion/interp.
                # Don't force-break - instead check if the positions are reasonably close.
                # A break only happens if the distance exceeds the jump threshold below.

            if prev is not None:
                if not pending_lost_gap:
                    x_span, y_span = _court_axis_spans(trail_kps, w, h)
                    dx = abs(trail_cx - prev[0])
                    dy = abs(trail_cy - prev[1])
                    jx_c, jy_c = _trail_jump_fracs(src, cfg)
                    jx_p, jy_p = _trail_jump_fracs(prev_src, cfg)
                    jx = max(jx_c, jx_p)
                    jy = max(jy_c, jy_p)
                    hard_x_px = max(jx * x_span, 34.0)
                    hard_y_px = max(jy * y_span, 26.0)
                    dist = float(np.hypot(dx, dy))
                    pair_diag = float(np.hypot(x_span, y_span))
                    source_transition = (src != prev_src)
                    bridge_px = max(0.035 * pair_diag, 18.0)
                    transition_px = max(0.020 * pair_diag, 14.0)

                    # Source transitions: connect if the jump is within a reasonable
                    # distance. Too tight = gaps everywhere. Too wide = connects to noise.
                    if source_transition:
                        if dist > bridge_px:
                            trail.append(None)
                            prev = None
                        elif prev_src in ("carry", "interp") and src == "det" and dist > 6.0:
                            trail.append(None)
                            prev = None
                        else:
                            reset_smoothing = True

                    if prev is not None:
                        if dx > hard_x_px or dy > hard_y_px:
                            # Allow small source-transition mismatch (e.g., blue->green reacquire).
                            if not (source_transition and dist <= bridge_px):
                                trail.append(None)
                                prev = None
                        elif prev2 is not None and _trail_direction_break(prev2, prev, (trail_cx, trail_cy)):
                            # Direction-change segment breaks should be stricter on confident sources only.
                            if not (_is_soft_source(src) or _is_soft_source(prev_src) or
                                    (source_transition and dist <= bridge_px * 1.4)):
                                trail.append(None)
                                prev = None

            if prev is None:
                smooth_x, smooth_y = trail_cx, trail_cy
            else:
                if pending_lost_gap or src == "carry" or reset_smoothing:
                    # Reacquire and carry frames stay exact.
                    smooth_x, smooth_y = trail_cx, trail_cy
                elif src == "det":
                    # Green trail always uses exact detection positions - no smoothing.
                    # Smoothing would pull the green line toward adjacent carry/interp
                    # positions and misrepresent the actual YOLO detection location.
                    smooth_x, smooth_y = trail_cx, trail_cy
                else:
                    # Smooth only non-green sources.
                    alpha_s = _trail_smooth_alpha(src)
                    sx = alpha_s * trail_cx + (1.0 - alpha_s) * prev[2]
                    sy = alpha_s * trail_cy + (1.0 - alpha_s) * prev[3]
                    smooth_x, smooth_y = int(round(sx)), int(round(sy))

            # When we recover from a fully-lost run, connect end->new-start in black.
            if pending_lost_gap and prev is not None:
                if ENABLE_GAP_CONNECTORS:
                    trail.append((trail_cx, trail_cy, smooth_x, smooth_y, 'gap', 0.0))
                else:
                    # Even with connectors disabled, hard-break the segment on loss.
                    trail.append(None)
            pending_lost_gap = False
            trail.append((trail_cx, trail_cy, smooth_x, smooth_y, src, conf))
        else:
            # det/motion/carry/guide all failed this frame; prepare a black connector
            # from the current segment end to the next recovered point.
            if trail and trail[-1] is not None:
                pending_lost_gap = True

        # Trim trail
        while len(trail) > trail_max:
            trail.pop(0)

        # Draw trail - render black gap bridges first, then normal colored trail.
        for draw_gap in (True, False):
            for i in range(1, len(trail)):
                if trail[i] is None or trail[i-1] is None:
                    continue
                src = trail[i][4]
                is_gap = (src == 'gap')
                if is_gap != draw_gap:
                    continue
                if is_gap:
                    continue
                p0 = (trail[i-1][2], trail[i-1][3])
                p1 = (trail[i][2], trail[i][3])

                # Guide view: show chosen trail state colors for all sources.
                if (not is_gap) and guide_frame is not None and src in ("det", "carry", "interp", "motion", "guide"):
                    gbase = _trail_base_color(src)
                    galpha = 0.55 + 0.45 * (i / len(trail))
                    gcolor = (int(gbase[0] * galpha), int(gbase[1] * galpha), int(gbase[2] * galpha))
                    gth = max(2, int(3 * galpha))
                    cv2.line(guide_frame, p0, p1, (15, 15, 15), gth + 1, cv2.LINE_AA)
                    cv2.line(guide_frame, p0, p1, gcolor, gth, cv2.LINE_AA)

                # Transfer selected state trails to main output.
                # Keep: green det, orange motion, blue carry, yellow interp, white guide.
                if (not is_gap) and src not in ("det", "motion", "carry", "guide", "interp"):
                    continue

                if draw_main_ball_trail:
                    alpha = 0.55 + 0.45 * (i / len(trail))
                    base = _trail_base_color(src)
                    color = (int(base[0] * alpha), int(base[1] * alpha), int(base[2] * alpha))
                    line_th = max(2, int(3 * alpha))
                    outline_th = line_th + 1
                    # Dark outline first so the trail stays readable against court/player textures.
                    cv2.line(frame_out, p0, p1, (15, 15, 15), outline_th, cv2.LINE_AA)
                    cv2.line(frame_out, p0, p1, color, line_th, cv2.LINE_AA)

        # HUD
        cv2.putText(frame_out, f"F{fi}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if display_result:
            chosen_id = int(chosen_track.track_id) if chosen_track else -1
            hud_text = f"ID:{chosen_id} {display_result.source} c={display_result.conf:.2f}"
            cv2.putText(frame_out, hud_text, (60, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if guide_frame is not None:
                 cv2.putText(guide_frame, f"F{fi}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                 cv2.putText(guide_frame, hud_text, (60, 25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Guide-debug stream: show chosen-track guide progression over time.
        guide_info = guide_map.get(fi)
        if guide_info is None and display_result is not None:
            # Keep guide-debug continuous when chosen-track guide ends early.
            # Render fallback prominently so end-of-video guide does not disappear visually.
            g_exact_fallback = True
            guide_info = (float(display_result.cx), float(display_result.cy), g_exact_fallback)
        if guide_info is not None:
            ggx, ggy, gexact = guide_info
            guide_pt_dbg = (int(round(ggx)), int(round(ggy)), bool(gexact))
            guide_trail.append(guide_pt_dbg)
            last_guide_point_dbg = guide_pt_dbg
        elif guide_trail and guide_trail[-1] is not None:
            guide_trail.append(None)
        while len(guide_trail) > guide_trail_max:
            guide_trail.pop(0)

        if guide_frame is not None:
            # Draw the actual guide-side detection gate/search circle used by the selector
            # (exact-guide lock or frozen-guide reacquire radius) when available.
            gate_dbg = None
            if guide_state_result is not None:
                gsr = float(getattr(guide_state_result, "guide_search_radius", 0.0) or 0.0)
                if gsr > 0.0:
                    gate_dbg = (
                        int(round(float(getattr(guide_state_result, "guide_search_cx", 0.0)))),
                        int(round(float(getattr(guide_state_result, "guide_search_cy", 0.0)))),
                        float(gsr),
                        bool(getattr(guide_state_result, "guide_search_exact", False)),
                        bool(getattr(guide_state_result, "guide_search_frozen", False)),
                        bool(getattr(guide_state_result, "guide_search_hold", False)),
                    )
            if gate_dbg is not None:
                if len(gate_dbg) >= 6:
                    gsx, gsy, gsr, gexact_dbg, gfrozen_dbg, ghold_dbg = gate_dbg[:6]
                else:
                    gsx, gsy, gsr, gexact_dbg, gfrozen_dbg = gate_dbg
                    ghold_dbg = False
                if ghold_dbg:
                    gate_col = (255, 0, 0)    # blue hold guide circle (selector logic)
                    gate_label = "GUIDE DET R (hold)"
                elif gfrozen_dbg:
                    gate_col = (0, 200, 255)  # orange-ish frozen reacquire circle
                    gate_label = "GUIDE DET R (frozen)"
                elif gexact_dbg:
                    gate_col = (0, 255, 0)    # green exact-guide lock
                    gate_label = "GUIDE DET R (exact)"
                else:
                    gate_col = (0, 255, 255)  # yellow soft/interp guide gate
                    gate_label = "GUIDE DET R (interp)"
                cv2.circle(guide_frame, (gsx, gsy), int(round(gsr)), (15, 15, 15), 2, cv2.LINE_AA)
                cv2.circle(guide_frame, (gsx, gsy), int(round(gsr)), gate_col, 1, cv2.LINE_AA)
                cv2.drawMarker(guide_frame, (gsx, gsy), gate_col, cv2.MARKER_CROSS, 10, 1)
                cv2.putText(
                    guide_frame, gate_label, (gsx + 10, gsy + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, gate_col, 1
                )

            if display_result is None:
                for i in range(1, len(guide_trail)):
                    p0 = guide_trail[i - 1]
                    p1 = guide_trail[i]
                    if p0 is None or p1 is None:
                        continue
                    if math.hypot(p1[0] - p0[0], p1[1] - p0[1]) > max(200.0, w * 0.15):
                        continue
                    seg_exact = bool(p0[2] and p1[2])
                    seg_color = COLOR_GUIDE if seg_exact else COLOR_GUIDE_INTERP
                    seg_th = 2 if seg_exact else 1
                    cv2.line(guide_frame, (p0[0], p0[1]), (p1[0], p1[1]),
                             (15, 15, 15), seg_th + 1, cv2.LINE_AA)
                    cv2.line(guide_frame, (p0[0], p0[1]), (p1[0], p1[1]),
                             seg_color, seg_th, cv2.LINE_AA)

            if guide_info is not None and display_result is None:
                ggx, ggy, gexact = guide_info
                gc = (int(round(ggx)), int(round(ggy)))
                gcol = COLOR_GUIDE if gexact else COLOR_GUIDE_INTERP
                gr = 6 if gexact else 4
                cv2.circle(guide_frame, gc, gr + 2, (15, 15, 15), 2)
                cv2.circle(guide_frame, gc, gr, gcol, -1)
            cv2.putText(guide_frame, f"F{fi}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            if chosen_track is not None:
                cv2.putText(guide_frame, f"track={chosen_track.track_id}",
                            (60, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1)
            if info_timing:
                guide_write_t0 = time.perf_counter()
            guide_writer.write(guide_frame)
            if info_timing:
                guide_write_dt = time.perf_counter() - guide_write_t0
                guide_write_elapsed += guide_write_dt
                timing["pass2_write_guide"] = timing.get("pass2_write_guide", 0.0) + guide_write_dt

        if info_timing:
            render_dt = time.perf_counter() - render_t0 - guide_write_elapsed
            timing["pass2_render"] = timing.get("pass2_render", 0.0) + max(0.0, render_dt)
            main_write_t0 = time.perf_counter()
        if writer is not None:
            writer.write(frame_out)
        if info_timing and writer is not None:
            timing["pass2_write_main"] = timing.get("pass2_write_main", 0.0) + (time.perf_counter() - main_write_t0)

        # Debug videos
        if dbg_writer is not None:
            if info_timing:
                dbg_write_t0 = time.perf_counter()
            raw_m = _unpack_mask_u8(all_raw_motions[fi])
            boost_m = _unpack_mask_u8(all_boost_masks[fi])
            pboxes = all_player_boxes_preproc[fi] if fi < len(all_player_boxes_preproc) else []
            # Cache the protect mask to avoid recomputing build_court_side_protect_mask
            # every frame - court keypoints rarely change.
            _dbg_protect_key = tuple(int(round(float(v)*4.0)) for v in (court_kps or [])) if court_kps else None
            if fi == 0 or _dbg_protect_key != _dbg_protect_cache_key:
                _dbg_protect_cache = build_protect_mask(
                    h, w, player_bboxes=pboxes, court_keypoints=court_kps,
                    player_pad=cfg.player_bbox_pad) if court_kps else None
                _dbg_protect_cache_key = _dbg_protect_key
            vis = preprocess_frame(frame, raw_m, boost_m,
                                    cfg,
                                    player_bboxes=pboxes,
                                    court_keypoints=court_kps,
                                    visualize=True,
                                    protect_mask_cached=_dbg_protect_cache,
                                    rois_dbg=all_rois[fi] if fi < len(all_rois) else None)

            # -- Draw motion detection ROI boxes onto vis --
            # Green = survived   Red = ghost-pruned (didn't reattach)
            if (not getattr(cfg, "debug_probe_motion_style", True)) and fi < len(all_rois) and all_rois[fi] is not None:
                for rx1, ry1, rx2, ry2 in all_rois[fi]:
                    cv2.rectangle(vis, (int(rx1), int(ry1)), (int(rx2), int(ry2)), (0, 255, 0), 1)

            if frame_result is not None:
                vis_fx = float(getattr(frame_result, 'cx', 0.0) or 0.0)
                vis_fy = float(getattr(frame_result, 'cy', 0.0) or 0.0)
                vis_bbox = getattr(frame_result, 'bbox', None)
                vis_src  = str(getattr(frame_result, 'source', ''))
                vis_sr   = float(getattr(frame_result, 'search_radius', 0.0) or 0.0)
                vis_scx  = float(getattr(frame_result, 'search_cx', 0.0) or vis_fx)
                vis_scy  = float(getattr(frame_result, 'search_cy', 0.0) or vis_fy)

                # Color by source
                src_colors = {
                    'det': (0, 220, 60), 'motion': (0, 160, 255),
                    'carry': (255, 160, 0), 'guide': (255, 255, 0),
                    'interp': (0, 220, 220),
                }
                ball_col = src_colors.get(vis_src, (200, 200, 200))

                if vis_fx > 0 or vis_fy > 0:
                    cx_i = int(round(vis_fx))
                    cy_i = int(round(vis_fy))

                    if vis_bbox is not None and len(vis_bbox) == 4:
                        # -- YOLO detection bounding box --
                        bx1, by1, bx2, by2 = (int(round(v)) for v in vis_bbox)
                        cv2.rectangle(vis, (bx1, by1), (bx2, by2), ball_col, 2)
                        cv2.putText(vis, vis_src[:3].upper(),
                                    (bx1, max(by1 - 4, 12)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, ball_col, 1)
                    else:
                        # -- Estimated ball box from cached radius (carry/motion/guide) --
                        br = max(int(round(cached_anchor_radius)), 6)
                        cv2.rectangle(vis, (cx_i - br, cy_i - br), (cx_i + br, cy_i + br),
                                      ball_col, 1)
                        cv2.putText(vis, vis_src[:3].upper(),
                                    (cx_i - br, max(cy_i - br - 3, 12)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, ball_col, 1)

                    # -- Ball center dot --
                    cv2.circle(vis, (cx_i, cy_i), 4, ball_col, -1)

                    # -- Physics / KF search radius circle --
                    # Show for ALL non-det frames. If search_radius is populated (carry/motion),
                    # use it; otherwise fall back to a reasonable estimate.
                    if vis_src != 'det':
                        draw_sr = int(round(vis_sr)) if vis_sr > 1 else max(int(round(cached_anchor_radius * 2.5)), 20)
                        draw_scx = int(round(vis_scx)) if vis_scx > 0 else cx_i
                        draw_scy = int(round(vis_scy)) if vis_scy > 0 else cy_i
                        # Dashed-style: draw half-opacity outer ring + crosshair
                        cv2.circle(vis, (draw_scx, draw_scy), draw_sr,
                                   (255, 0, 220), 1, cv2.LINE_AA)
                        cv2.drawMarker(vis, (draw_scx, draw_scy),
                                       (255, 0, 220), cv2.MARKER_CROSS, 12, 1)
                        # Label the search radius size
                        cv2.putText(vis, f"r={draw_sr}",
                                    (draw_scx + draw_sr + 3, draw_scy),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 220), 1)


            # Extract motion blob centroids from boost_m within each ROI.
            # Using actual contour centroids instead of ROI center/bottom gives the
            # correct ball position when motion is asymmetrically distributed in the ROI.
            current_blobs = []
            if (not getattr(cfg, "debug_probe_motion_style", True)) and fi < len(all_rois) and all_rois[fi] is not None:
                for rx1, ry1, rx2, ry2 in all_rois[fi]:
                    irx1, iry1 = int(rx1), int(ry1)
                    irx2, iry2 = int(rx2), int(ry2)
                    found_in_roi = False
                    if boost_m is not None and boost_m.size > 0:
                        roi_mask = boost_m[iry1:iry2, irx1:irx2]
                        if roi_mask.size > 0 and roi_mask.max() > 0:
                            cnts, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            for cnt in cnts:
                                area = cv2.contourArea(cnt)
                                if area < 1.0:
                                    continue
                                bx_r, by_r, bw_r, bh_r = cv2.boundingRect(cnt)
                                # Anchor to bottom-center of blob bounding rect,
                                # matching green trail which anchors at bottom of YOLO bbox.
                                cx = irx1 + bx_r + bw_r / 2.0
                                cy = iry1 + by_r + bh_r
                                blob_r = min(10.0, max(4.0, math.sqrt(area / math.pi)))
                                current_blobs.append((float(cx), float(cy), float(blob_r)))
                                found_in_roi = True
                    if not found_in_roi:
                        # Fallback: use ROI center when boost_m has no blobs here
                        blob_cx = (rx1 + rx2) / 2.0
                        blob_cy = (ry1 + ry2) / 2.0
                        blob_r = min(10.0, max(4.0, min(rx2 - rx1, ry2 - ry1) * 0.12))
                        current_blobs.append((float(blob_cx), float(blob_cy), float(blob_r)))

            if True:
                # Reduced from max(120, w*0.08) - the old value (~153px for 1920px video)
                # allowed unrelated motion regions to be merged into the same track.
                MAX_DIST = max(40.0, w * 0.025)
                unmatched = current_blobs.copy()
                for trk in raw_motion_tracks:
                    last_fi, last_x, last_y, last_r = trk[-1]
                    if fi - last_fi <= 3:  # Allow 3 frame gap
                        best_dist = MAX_DIST
                        best_match = None
                        for b in unmatched:
                            dist = math.hypot(b[0] - last_x, b[1] - last_y)
                            if dist < best_dist:
                                best_dist = dist
                                best_match = b
                        if best_match is not None:
                            trk.append((fi, best_match[0], best_match[1], best_match[2]))
                            unmatched.remove(best_match)
                for b in unmatched:
                    raw_motion_tracks.append([(fi, b[0], b[1], b[2])])
            
            # Periodically prune old tracks
            MAX_AGE = 150
            if fi % 100 == 0:
                raw_motion_tracks = [t for t in raw_motion_tracks if fi - t[-1][0] <= MAX_AGE]

            # Draw filtered, smoothed, and faded comet trails
            for trk in raw_motion_tracks:
                valid_pts = [pt for pt in trk if fi - pt[0] <= MAX_AGE]
                # Remove random isolated dots and very short noise tracks
                if len(valid_pts) < 2:
                    continue
                
                # Filter out points that are too dense to avoid splprep waves/crashing
                unique_pts = [valid_pts[0]]
                for pt in valid_pts[1:]:
                    if math.hypot(pt[1] - unique_pts[-1][1], pt[2] - unique_pts[-1][2]) > 2.0:
                        unique_pts.append(pt)
                
                # Always append the actual current head if it was filtered out
                if valid_pts[-1] not in unique_pts:
                    unique_pts.append(valid_pts[-1])
                
                if len(unique_pts) < 4:
                    # Fallback to straight lines if too few unique points
                    for i in range(1, len(unique_pts)):
                        age = fi - unique_pts[i][0]
                        ratio = max(0.0, 1.0 - (age / float(MAX_AGE)))
                        color = (int(0 * ratio), int(140 * ratio), int(255 * ratio))
                        th = max(1, int(4 * ratio))
                        pt1 = (int(unique_pts[i-1][1]), int(unique_pts[i-1][2]))
                        pt2 = (int(unique_pts[i][1]), int(unique_pts[i][2]))
                        if math.hypot(pt2[0]-pt1[0], pt2[1]-pt1[1]) < 60.0:
                            cv2.line(vis, pt1, pt2, color, th, cv2.LINE_AA)
                else:
                    try:
                        import scipy.ndimage
                        pts_arr = np.array([[pt[1], pt[2]] for pt in unique_pts])
                        x_raw, y_raw = pts_arr[:, 0], pts_arr[:, 1]
                        
                        # Apply a 1D Gaussian smooth to kill the high-frequency pixel jitter
                        # that causes B-splines to severely wiggle/wave
                        x_smooth = scipy.ndimage.gaussian_filter1d(x_raw, sigma=1.5)
                        y_smooth = scipy.ndimage.gaussian_filter1d(y_raw, sigma=1.5)
                        
                        # splprep generates a smooth B-spline. s > 0 allows the spline
                        # to relax further instead of forcing exact intersection
                        tck, u = scipy.interpolate.splprep([x_smooth, y_smooth], s=len(x_smooth))
                        
                        num_smooth_pts = max(100, len(unique_pts) * 5)
                        u_fine = np.linspace(0, 1.0, num_smooth_pts)
                        x_fine, y_fine = scipy.interpolate.splev(u_fine, tck)
                        
                        ages = [fi - pt[0] for pt in unique_pts]
                        ages_fine = np.interp(u_fine, u, ages)
                        
                        for i in range(1, len(x_fine)):
                            age = ages_fine[i]
                            ratio = max(0.0, 1.0 - (age / float(MAX_AGE)))
                            color = (int(0 * ratio), int(140 * ratio), int(255 * ratio))
                            th = max(1, int(4 * ratio))
                            
                            pt1 = (int(x_fine[i-1]), int(y_fine[i-1]))
                            pt2 = (int(x_fine[i]), int(y_fine[i]))
                            
                            if math.hypot(pt2[0]-pt1[0], pt2[1]-pt1[1]) < 80.0:
                                cv2.line(vis, pt1, pt2, color, th, cv2.LINE_AA)
                    except Exception as e:
                        # Fallback if splprep fails
                        for i in range(1, len(unique_pts)):
                            age = fi - unique_pts[i][0]
                            ratio = max(0.0, 1.0 - (age / float(MAX_AGE)))
                            color = (int(0 * ratio), int(140 * ratio), int(255 * ratio))
                            th = max(1, int(4 * ratio))
                            pt1 = (int(unique_pts[i-1][1]), int(unique_pts[i-1][2]))
                            pt2 = (int(unique_pts[i][1]), int(unique_pts[i][2]))
                            if math.hypot(pt2[0]-pt1[0], pt2[1]-pt1[1]) < 60.0:
                                cv2.line(vis, pt1, pt2, color, th, cv2.LINE_AA)
                
                # Draw the trail head prominently if active
                head_age = fi - valid_pts[-1][0]
                if head_age == 0:
                    sr_list = [pt[3] for pt in valid_pts[-10:]] # Only use recent for EMA
                    sr = sr_list[0]
                    for r in sr_list[1:]:
                        sr = 0.10 * r + 0.90 * sr
                    
                    pt_head = (int(valid_pts[-1][1]), int(valid_pts[-1][2]))
                    rad = max(2, int(sr))
                    cv2.circle(vis, pt_head, rad, (0, 140, 255), -1)

            dbg_writer.write(vis)
            if info_timing:
                timing["pass2_write_debug"] = timing.get("pass2_write_debug", 0.0) + (time.perf_counter() - dbg_write_t0)

        # -- Motion-tracks debug video: ONLY motion polylines + ROI box --
        if motion_tracks_writer is not None:
            mt_vis = frame.copy()
            WIN_PAST = 90  # frames of past trail drawn per track
            for trk in motion_tracks_dbg:
                if not trk.points:
                    continue
                # Quick reject: track hasn't started yet, or ended too long ago.
                if trk.points[0][0] > fi or trk.points[-1][0] < fi - WIN_PAST:
                    continue
                past_pts = [(int(p[1]), int(p[2])) for p in trk.points
                            if (fi - WIN_PAST) <= p[0] <= fi]
                if len(past_pts) < 2:
                    continue
                color = _get_track_color(int(trk.track_id))
                arr = np.asarray(past_pts, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(mt_vis, [arr], False, color, 2, cv2.LINE_AA)

            # ROI box(es) for this frame - cyan rectangles
            if fi < len(all_rois) and all_rois[fi] is not None:
                for rx1, ry1, rx2, ry2 in all_rois[fi]:
                    cv2.rectangle(mt_vis, (int(rx1), int(ry1)), (int(rx2), int(ry2)),
                                  (0, 255, 255), 2, cv2.LINE_AA)

            motion_tracks_writer.write(mt_vis)

        pass2_frames_rendered += 1
        if cfg.progress_every and (fi + 1) % cfg.progress_every == 0:
            elapsed = time.time() - t0
            print(f"  [pass2 {fi+1}/{N}] rendering...")

    if cap2 is not None:
        cap2.release()
    if writer is not None:
        writer.close()
    if dbg_writer:
        dbg_writer.close()
    if yolo_dbg_writer:
        yolo_dbg_writer.close()
    if guide_writer:
        guide_writer.close()
    if motion_tracks_writer:
        motion_tracks_writer.close()
    if info_timing:
        timing["pass2_total"] = time.perf_counter() - pass2_perf_t0

    elapsed = time.time() - t0
    filled = sum(1 for r in per_frame if r is not None and not bool(getattr(r, "debug_only", False)))
    if getattr(cfg, "tracking_json", None):
        _write_tracking_json(
            cfg.tracking_json,
            cfg,
            fps,
            w,
            h,
            N,
            elapsed,
            filled,
            per_frame,
            chosen_track,
            all_tracks,
            detections_by_frame=all_frame_dets,
            boost_masks=all_boost_masks,
            raw_motions=all_raw_motions,
            court_keypoints_by_frame=all_court_kps,
            player_boxes_by_frame=all_player_boxes,
            timing=timing,
            pass2_frames_rendered=pass2_frames_rendered,
        )
        print(f"[done] Tracking JSON: {cfg.tracking_json}")
    print(f"\n[done] {filled}/{N} frames filled ({100*filled/max(1,N):.1f}%)")
    print(f"[done] {elapsed:.1f}s total")
    if getattr(cfg, "save_tracking_video", True):
        print(f"[done] Tracking video: {cfg.output_video}")
    if cfg.save_motion_debug:
        print(f"[done] Motion debug:  {cfg.output_debug_path}")
    if getattr(cfg, "save_yolo_input_debug", False):
        print(f"[done] Pre-YOLO debug:  {cfg.output_yolo_input_debug_path}")
    if cfg.save_guide_video:
        print(f"[done] Guide Debug:  {cfg.output_guide_path}")
    if getattr(cfg, "save_motion_tracks_video", False):
        print(f"[done] Motion Tracks Debug:  {cfg.output_motion_tracks_debug_path}")
    if info_timing:
        _print_timing_summary(timing, elapsed, N, pass2_frames_rendered)
