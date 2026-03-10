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

from .config import Config
from .utils import _detect_device, _check_capabilities, _resolve_engine_path_for_ball, find_ball_class_id_from_names, _read_engine_names
from .detectors import BallDetectorBackend, CourtDetector, PlayerDetector, TensorRTRuntimeBallDetector
from .motion import filter_boost_mask, _pack_mask_u8, build_protect_mask, compute_motion_sv_from_hsv, suppress_flicker_components, preprocess_frame_cuda, _unpack_mask_u8, build_court_side_protect_mask, apply_exclude_mask_u8, preprocess_frame
from .tracking import ROIMotionTracker
from .rendering import _is_soft_source, _trail_base_color, _get_track_color, _court_axis_spans, _build_court_polygon, _trail_jump_fracs, _draw_homography_net_line, _print_timing_summary, _trail_direction_break, _print_selector_track_summary, _trail_smooth_alpha, _build_ground_projection_model, _drop_unattached_soft_runs, _trail_prev2, _build_display_guide, COLOR_DET, COLOR_RAW, COLOR_MOTION, COLOR_SEARCH, COLOR_INTERP, COLOR_CARRY, COLOR_GAP, COLOR_GUIDE, COLOR_GUIDE_INTERP, ENABLE_GAP_CONNECTORS, GAP_END_TRIM_PX
from .video_io import _cuda_frame_to_chw_f32, _PinnedFrameUploader, _cuda_vs_tensors, ThreadedFrameReader, VideoWriter


def run(cfg):
    t0 = time.time()
    info_timing = bool(getattr(cfg, "info_timing", False))
    timing = {} if info_timing else None
    init_perf_t0 = time.perf_counter() if info_timing else 0.0

    # Platform detection
    device_str, device_desc = _detect_device(cfg.device)
    cfg.device = device_str
    cfg = _check_capabilities(cfg, device_str)
    HAS_CUDA = HAS_TORCH and torch.cuda.is_available()
    is_cuda = device_str not in ("cpu", "mps")

    model_names = None
    detector: BallDetectorBackend
    ball_cls_id = None
    ball_cls_name = None

    # TensorRT-only ball runtime path.
    trt_engine_path = _resolve_engine_path_for_ball(cfg) if cfg.use_tensorrt else None
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
    cap = cv2.VideoCapture(cfg.input_video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[init] Video: {w}x{h} @ {fps:.1f}fps, ~{total} frames")

    court_det = CourtDetector(cfg)
    player_det = PlayerDetector(cfg, court_keypoints=None)

    # Writers (main output writer is created at pass 2 start to avoid long idle FFmpeg/NVENC process)
    writer = None
    dbg_writer = yolo_dbg_writer = guide_writer = None
    if cfg.save_motion_debug:
        os.makedirs(os.path.dirname(cfg.output_debug_path) or ".", exist_ok=True)
        dbg_writer = VideoWriter(cfg.output_debug_path, fps, w, h, cfg)
        os.makedirs(os.path.dirname(cfg.output_yolo_input_debug_path) or ".", exist_ok=True)
        yolo_dbg_writer = VideoWriter(cfg.output_yolo_input_debug_path, fps, w, h, cfg)
    if cfg.save_guide_video:
        os.makedirs(os.path.dirname(cfg.output_guide_path) or ".", exist_ok=True)
        guide_writer = VideoWriter(cfg.output_guide_path, fps, w, h, cfg)

    # Preprocessing mode
    use_cuda = cfg.enable_preprocess and is_cuda and HAS_CUDA
    if use_cuda:
        print("[preprocess] CUDA S+V motion path")
        cuda_device = torch.device("cuda")
    else:
        print("[preprocess] CPU S+V motion path" if cfg.enable_preprocess else "[preprocess] disabled")

    # Threaded frame reader — overlaps decode with GPU inference
    reader = ThreadedFrameReader(cap, prefetch=max(2, int(cfg.frame_reader_prefetch)))
    frame_curr = reader.read()
    frame_next = reader.read() if frame_curr is not None else None
    if frame_curr is None:
        print("[error] No frames in video")
        cap.release()
        return

    prev_v = prev_s = curr_v_t = curr_s_t = next_v_t = next_s_t = None
    curr_frame_gpu_t = next_frame_gpu_t = None
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

    hsv_prev = hsv_curr = hsv_next = None
    master_bg_v = master_bg_s = master_var_v = master_var_s = master_hsv = None
    if not use_cuda and cfg.enable_preprocess:
        hsv_curr = cv2.cvtColor(frame_curr, cv2.COLOR_BGR2HSV)
        if frame_next is not None:
            hsv_next = cv2.cvtColor(frame_next, cv2.COLOR_BGR2HSV)
    if info_timing:
        timing["init_total"] = time.perf_counter() - init_perf_t0
    pass1_perf_t0 = time.perf_counter() if info_timing else 0.0

    # ══════════════════════════════════════════════════════════════
    # PASS 1 — Collect detections, preprocess frames, store metadata
    # ══════════════════════════════════════════════════════════════
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

    # ROI motion tracker — limits motion/CC to region near ball
    roi_tracker = ROIMotionTracker(cfg, w, h, fps)
    roi_used_count = 0
    fullframe_count = 0
    side_mask_cache = None
    side_mask_cache_key = None
    protect_mask_cuda_cache = None
    protect_mask_cuda_cache_key = None
    prev_boost_for_flicker = None
    prev_raw_motion_cuda = None  # CUDA boolean mask from previous frame — avoids CPU→GPU re-upload in WTA
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
    if use_cuda_pre_frame:
        print("[pass 1] CUDA zero-copy preprocess->YOLO path enabled")
    cache_pass2_frames = None
    if cfg.cache_input_frames_pass2 and total > 0:
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

    while frame_curr is not None:
        # Keep previous pending handle; finish it after current preprocess/start
        # so frame N preprocess overlaps frame N-1 inference.
        prev_pending = pending_det
        pending_det = None

        raw_motion_u8 = boost_mask_u8 = None
        pre_frame_cuda = None
        pre_frame = frame_curr
        det_pending = None
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
                master_bg_v = curr_v_t.clone()
                master_bg_s = curr_s_t.clone()
                
                thr = float(cfg.motion_thresh) / 255.0
                master_var_v = torch.full_like(curr_v_t, thr**2)
                master_var_s = torch.full_like(curr_s_t, thr**2)
            else:
                v_diff_sq = (curr_v_t - master_bg_v)**2
                s_diff_sq = (curr_s_t - master_bg_s)**2
                
                # Freeze background updates where raw motion was detected last frame.
                # Use the CUDA raw_motion tensor returned by preprocess_frame_cuda (prev frame)
                # instead of re-uploading raw_motion_u8 from CPU every frame.
                if prev_raw_motion_cuda is not None:
                    alpha_mask = torch.where(prev_raw_motion_cuda, 0.0, float(cfg.wta_alpha))
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
                if raw_motion_u8 is not None:
                    # For OpenCV, we need to create a mask where alpha is 0 for motion pixels
                    alpha_arr = np.full_like(master_hsv[:,:,0], cfg.wta_alpha, dtype=np.float32)
                    alpha_arr[raw_motion_u8 > 0] = 0.0
                    alpha_arr = np.stack([alpha_arr]*3, axis=-1)
                    master_hsv = master_hsv * (1.0 - alpha_arr) + hsv_curr.astype(np.float32) * alpha_arr
                else:
                    cv2.accumulateWeighted(hsv_curr, master_hsv, cfg.wta_alpha)

        # Detect ball — skip-frame YOLO for speed
        # When skip_n > 1: run YOLO every Nth frame, selector interpolates short gaps.
        _do_yolo = True
        if skip_n > 1 and frame_idx % skip_n != 0:
            # Only skip when ROI tracker says ball is visible (smooth motion)
            if skip_require_roi and roi_tracker.last_pos is not None and roi_tracker.frames_since_det <= 3:
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
            player_boxes = player_det.detect(frame_curr, frame_idx)
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
                # Skip cosmetic dim_static when ball is visible — saves a GPU round-trip
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
                        for r in rois:
                            rx1, ry1, rx2, ry2 = r
                            rm_roi = compute_motion_sv_from_hsv(
                                master_hsv_u8[ry1:ry2, rx1:rx2], hsv_curr[ry1:ry2, rx1:rx2], cfg.motion_thresh)
                            np.maximum(rm_full[ry1:ry2, rx1:rx2], rm_roi, out=rm_full[ry1:ry2, rx1:rx2])
                        rm = rm_full
                    else:
                        rm = compute_motion_sv_from_hsv(master_hsv_u8, hsv_curr, cfg.motion_thresh)
                else:
                    rm = None

                raw_motion_u8 = rm
                if rm is not None:
                    if rois:
                        boost_mask_u8 = np.zeros_like(rm)
                        for r in rois:
                            rx1, ry1, rx2, ry2 = r
                            roi_slice = rm[ry1:ry2, rx1:rx2]
                            if roi_slice.max() > 0:
                                filtered_roi = filter_boost_mask(
                                    roi_slice, cfg.boost_min_blob_area, cfg.boost_max_blob_area, cfg,
                                    player_bboxes=player_boxes)
                                np.maximum(boost_mask_u8[ry1:ry2, rx1:rx2], filtered_roi, out=boost_mask_u8[ry1:ry2, rx1:rx2])
                        if boost_mask_u8.max() == 0:
                            boost_mask_u8 = None
                    else:
                        boost_mask_u8 = filter_boost_mask(
                            rm, cfg.boost_min_blob_area, cfg.boost_max_blob_area, cfg,
                            player_bboxes=player_boxes)
                else:
                    boost_mask_u8 = None

                pre_frame = preprocess_frame(frame_curr, raw_motion_u8, boost_mask_u8, cfg,
                                              player_bboxes=player_boxes,
                                              court_keypoints=court_kps,
                                              hsv_cached=hsv_curr,
                                              protect_mask_cached=protect_mask)

            # Start current frame inference as soon as detector input is ready.
            # With per-pending CUDA events + output slots, this is safe and allows
            # overlap with post-mask CPU work below.
            if _do_yolo:
                det_input = pre_frame_cuda if pre_frame_cuda is not None else pre_frame
                det_pending = detector.detect_async_start(det_input)

            # Don't exclude any court regions from motion detection — let the selector
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
            # This avoids re-uploading raw_motion_u8 (CPU→CUDA) in the next iteration.
            if use_cuda:
                if raw_motion_u8 is not None:
                    prev_raw_motion_cuda = torch.from_numpy(raw_motion_u8).to(cuda_device, non_blocking=True) > 0
                else:
                    prev_raw_motion_cuda = None
            if info_timing:
                timing["pass1_pre_postmask"] = timing.get("pass1_pre_postmask", 0.0) + (time.perf_counter() - postmask_t0)
        if info_timing:
            timing["pass1_preprocess"] = timing.get("pass1_preprocess", 0.0) + (time.perf_counter() - pre_t0)

        # Finish previous frame inference after current frame preprocess.
        if prev_pending is not None:
            if info_timing:
                det_t0 = time.perf_counter()
            dets_prev = detector.detect_async_finish(prev_pending["handle"])
            if info_timing:
                timing["pass1_ball_detect"] = timing.get("pass1_ball_detect", 0.0) + (time.perf_counter() - det_t0)
            _commit_frame(prev_pending["record"], dets_prev)

        # Detect ball — skip-frame YOLO for speed.
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
        
        # Prevent CUDA memory fragmentation from causing FPS decline over time
        if use_cuda and frame_idx % 500 == 0:
            torch.cuda.empty_cache()

        # Slide window
        if info_timing:
            slide_t0 = time.perf_counter()
        frame_curr = frame_next
        frame_next = reader.read()
        if use_cuda:
            curr_v_t, curr_s_t = next_v_t, next_s_t
            curr_frame_gpu_t = next_frame_gpu_t
            if frame_next is not None:
                next_frame_gpu_t = _cuda_frame_to_chw_f32(frame_next, cuda_device, uploader=frame_uploader)
                next_v_t, next_s_t = _cuda_vs_tensors(None, cuda_device, gpu_tensor=next_frame_gpu_t)
            else:
                next_v_t = next_s_t = None
                next_frame_gpu_t = None
        elif cfg.enable_preprocess:
            hsv_curr = hsv_next
            hsv_next = cv2.cvtColor(frame_next, cv2.COLOR_BGR2HSV) if frame_next is not None else None
        if info_timing:
            timing["pass1_slide"] = timing.get("pass1_slide", 0.0) + (time.perf_counter() - slide_t0)

    # Drain last pending YOLO inference.
    if pending_det is not None:
        if info_timing:
            det_t0 = time.perf_counter()
        dets_prev = detector.detect_async_finish(pending_det["handle"])
        if info_timing:
            timing["pass1_ball_detect"] = timing.get("pass1_ball_detect", 0.0) + (time.perf_counter() - det_t0)
        _commit_frame(pending_det["record"], dets_prev)
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
    
    # ══════════════════════════════════════════════════════════════
    # SELECTOR — pick the in-play ball track
    # ══════════════════════════════════════════════════════════════
    selector_perf_t0 = time.perf_counter() if info_timing else 0.0
    selector_poly_t0 = time.perf_counter() if info_timing else 0.0
    court_poly, last_kps = _build_court_polygon(all_court_kps, w, h)
    if info_timing:
        timing["selector_build_poly"] = timing.get("selector_build_poly", 0.0) + (time.perf_counter() - selector_poly_t0)

    print("[selector] Running ball-in-play selection...")
    selector_select_t0 = time.perf_counter() if info_timing else 0.0
    per_frame, chosen_track, all_tracks = select_ball_in_play(
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

        # Second-pass backfill: for frames still missing from guide_map (per_frame was also
        # None — e.g. the first frames right after a hit before the chosen track's span),
        # fill from the observations of the top candidate tracks so the guide video shows
        # *all* detected ball trajectories even when the selector's chosen guide didn't
        # reach back that far.
        if all_tracks:
            # Build per-frame lookup: frame -> (cx, cy, conf, exact) from all tracks,
            # preferring the highest-confidence detection at each frame.
            aux_by_frame: Dict[int, Tuple[float, float, bool]] = {}
            filtered_for_backfill = [t for t in all_tracks if float(getattr(t, "score_breakdown", {}).get("inside_strict_frac", getattr(t, "score_breakdown", {}).get("inside_frac", 0.0))) > 0.0 and float(getattr(t, "score_breakdown", {}).get("motion_frac", 0.0)) > 0.0]
            for trk in sorted(filtered_for_backfill, key=lambda t: float(t.score), reverse=True):
                for o in trk.observations:
                    f = int(o.frame)
                    if f in guide_map:
                        continue  # already covered by chosen track
                    prev = aux_by_frame.get(f)
                    if prev is None or float(o.conf) > float(prev[2]):
                        exact = bool(getattr(o, "on_motion", False)) or float(o.conf) > 0.3
                        aux_by_frame[f] = (float(o.cx), float(o.cy), exact)
            for f, (cx, cy, exact) in aux_by_frame.items():
                if f not in guide_map:
                    guide_map[f] = (cx, cy, exact)
    if info_timing:
        timing["selector_post"] = timing.get("selector_post", 0.0) + (time.perf_counter() - selector_post_t0)
        timing["selector_total"] = time.perf_counter() - selector_perf_t0

    # ══════════════════════════════════════════════════════════════
    # PASS 2 — Render output video with debug visualization
    # ══════════════════════════════════════════════════════════════
    print("[pass 2] Rendering output video...")
    pass2_perf_t0 = time.perf_counter() if info_timing else 0.0
    pass2_frames_rendered = 0
    os.makedirs(os.path.dirname(cfg.output_video) or ".", exist_ok=True)
    writer = VideoWriter(cfg.output_video, fps, w, h, cfg)
    print(f"[writer] Encoder: {writer._encoder}")
    use_cached_frames = bool(cache_pass2_frames is not None and len(cache_pass2_frames) == N)
    if use_cached_frames:
        cap2 = None
        print(f"[pass 2] Using RAM frame cache ({len(cache_pass2_frames)} frames)")
    else:
        cap2 = cv2.VideoCapture(cfg.input_video)
        if cache_pass2_frames is not None:
            print("[pass 2] RAM frame cache incomplete; falling back to video decode")
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

    for fi in range(N):
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

        # Draw chosen result — color-coded by source
        if display_result is not None:
            rcx, rcy = int(display_result.cx), int(display_result.cy)
            trail_cx, trail_cy = rcx, rcy

            # Bottom-anchor for orange/blue paths using cached ball size.
            if display_result.source in ('motion', 'carry'):
                trail_cy = int(np.clip(
                    round(float(display_result.cy) + cached_anchor_radius), 0, h - 1))

            if display_result.source == 'det':
                # GREEN — YOLO detection
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
                # ORANGE — motion blob + search region
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
                # YELLOW — prolonged guessed/stuck
                i_rad = int(np.clip(round(cached_anchor_radius), 4, 12))
                cv2.circle(frame_out, (rcx, rcy), i_rad, COLOR_INTERP, -1)
                if guide_frame is not None:
                    cv2.circle(guide_frame, (rcx, rcy), i_rad, COLOR_INTERP, -1)
            elif display_result.source == 'carry':
                last_motion_search_dbg = None
                last_motion_search_dbg_frame = -1
                # BLUE — short predicted carry when temporarily lost
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
                # Don't force-break — instead check if the positions are reasonably close.
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
                    # Light smoothing for green trail only on gentle continuation.
                    step = float(np.hypot(trail_cx - prev[0], trail_cy - prev[1]))
                    sharp_turn = bool(prev2 is not None and _trail_direction_break(
                        prev2, prev, (trail_cx, trail_cy)))
                    if prev_src == "det" and step <= max(18.0, 0.030 * float(np.hypot(w, h))) and not sharp_turn:
                        alpha_det = 0.82
                        sx = alpha_det * trail_cx + (1.0 - alpha_det) * prev[2]
                        sy = alpha_det * trail_cy + (1.0 - alpha_det) * prev[3]
                        smooth_x, smooth_y = int(round(sx)), int(round(sy))
                    else:
                        # Keep sharp turns/bounces exact.
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

        # Draw trail — render black gap bridges first, then normal colored trail.
        for draw_gap in (True, False):
            for i in range(1, len(trail)):
                if trail[i] is None or trail[i-1] is None:
                    continue
                src = trail[i][4]
                is_gap = (src == 'gap')
                if is_gap != draw_gap:
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

                if is_gap:
                    # Keep connector subtle and stop short of reacquire marker.
                    dx = float(p1[0] - p0[0])
                    dy = float(p1[1] - p0[1])
                    mag = float(np.hypot(dx, dy))
                    if mag > GAP_END_TRIM_PX + 1.0:
                        s = (mag - GAP_END_TRIM_PX) / mag
                        p1 = (int(round(p0[0] + dx * s)), int(round(p0[1] + dy * s)))
                    cv2.line(frame_out, p0, p1, COLOR_GAP, 2, cv2.LINE_AA)
                    continue

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
        writer.write(frame_out)
        if info_timing:
            timing["pass2_write_main"] = timing.get("pass2_write_main", 0.0) + (time.perf_counter() - main_write_t0)

        # Debug videos
        if dbg_writer is not None:
            if info_timing:
                dbg_write_t0 = time.perf_counter()
            raw_m = _unpack_mask_u8(all_raw_motions[fi])
            boost_m = _unpack_mask_u8(all_boost_masks[fi])
            pboxes = all_player_boxes_preproc[fi] if fi < len(all_player_boxes_preproc) else []
            # Cache the protect mask to avoid recomputing build_court_side_protect_mask
            # every frame — court keypoints rarely change.
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

            # ── Draw motion detection ROI boxes onto vis ──
            # Green = survived   Red = ghost-pruned (didn't reattach)
            if fi < len(all_rois) and all_rois[fi] is not None:
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
                        # ── YOLO detection bounding box ──
                        bx1, by1, bx2, by2 = (int(round(v)) for v in vis_bbox)
                        cv2.rectangle(vis, (bx1, by1), (bx2, by2), ball_col, 2)
                        cv2.putText(vis, vis_src[:3].upper(),
                                    (bx1, max(by1 - 4, 12)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, ball_col, 1)
                    else:
                        # ── Estimated ball box from cached radius (carry/motion/guide) ──
                        br = max(int(round(cached_anchor_radius)), 6)
                        cv2.rectangle(vis, (cx_i - br, cy_i - br), (cx_i + br, cy_i + br),
                                      ball_col, 1)
                        cv2.putText(vis, vis_src[:3].upper(),
                                    (cx_i - br, max(cy_i - br - 3, 12)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, ball_col, 1)

                    # ── Ball center dot ──
                    cv2.circle(vis, (cx_i, cy_i), 4, ball_col, -1)

                    # ── Physics / KF search radius circle ──
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


            # Extract ALL raw motion blobs from the motion ROIs and anchor them to bottom-center
            current_blobs = []
            if fi < len(all_rois) and all_rois[fi] is not None:
                for rx1, ry1, rx2, ry2 in all_rois[fi]:
                    blob_cx = (rx1 + rx2) / 2.0
                    blob_cy = ry2
                    area = max((rx2 - rx1) * (ry2 - ry1), 4.0)
                    blob_r = math.sqrt(area / math.pi)
                    current_blobs.append((float(blob_cx), float(blob_cy), float(blob_r)))

            if True:
                MAX_DIST = max(120.0, w * 0.08)
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
    if info_timing:
        timing["pass2_total"] = time.perf_counter() - pass2_perf_t0

    elapsed = time.time() - t0
    filled = sum(1 for r in per_frame if r is not None and not bool(getattr(r, "debug_only", False)))
    print(f"\n[done] {filled}/{N} frames filled ({100*filled/max(1,N):.1f}%)")
    print(f"[done] {elapsed:.1f}s total")
    print(f"[done] Output: {cfg.output_video}")
    if cfg.save_motion_debug:
        print(f"[done] Debug:  {cfg.output_debug_path}")
        print(f"[done] YOLO Input Debug:  {cfg.output_yolo_input_debug_path}")
    if cfg.save_guide_video:
        print(f"[done] Guide Debug:  {cfg.output_guide_path}")
    if info_timing:
        _print_timing_summary(timing, elapsed, N, pass2_frames_rendered)

def main():
    p = argparse.ArgumentParser(
        description="Tennis Ball Detection & Tracking Pipeline",
        formatter_class=argparse.RawTextHelpFormatter)

    g = p.add_argument_group("Paths")
    g.add_argument("-i", "--input", default="input_videos/TMP.mp4", help="Input video (default: input_videos/TMP.mp4)")
    g.add_argument("-o", "--output", default="output_videos/prof_test.mp4", help="Output video (default: output_videos/prof_test.mp4)")
    g.add_argument("--model", default="models/ball.engine",
                   help="Ball model path (.engine)")

    g = p.add_argument_group("Detection")
    g.add_argument("--conf", type=float, default=0.26, help="Confidence threshold 0.01-1.0 (default: 0.25)")
    g.add_argument("--ball-backend", default="trt", choices=["trt"],
                   help="Ball backend path (TensorRT only).")
    g.add_argument("--device", default="auto", help="Device: auto, cpu, mps, 0, 1 (default: auto)")

    g = p.add_argument_group("Court Perspective")
    g.add_argument("--court-depth", default="none", choices=["top_far", "bot_far", "none"],
                   help="Far end of court: top_far, bot_far, none (default: none)")
    g.add_argument("--court-side", default="none", choices=["center_near", "left_far", "right_far", "none"],
                   help="Horizontal perspective: center_near, left_far, right_far, none (default: none)")
    g.add_argument("--y-scale", type=float, default=0.35, help="Depth scale strength 0-1 (default: 0.35)")
    g.add_argument("--x-scale", type=float, default=0.15, help="Side scale strength 0-1 (default: 0.15)")

    g = p.add_argument_group("Preprocessing")
    g.add_argument("--no-preprocess", action="store_true", help="Disable motion preprocessing")
    g.add_argument("--sat-boost", type=float, default=1.55, help="Saturation boost for motion blobs (default: 1.55)")
    g.add_argument("--val-boost", type=float, default=1.04, help="Brightness boost for motion blobs (default: 1.04)")
    g.add_argument("--hue-shift", type=float, default=0.18, help="Hue shift toward yellow/green 0-1 (default: 0.18)")
    g.add_argument("--dim-static", type=float, default=0.88, help="Static region dimming 0-1 (default: 0.88)")
    g.add_argument("--static-sat-scale", type=float, default=0.75, help="Static region desat 0-1 (default: 0.75)")
    g.add_argument("--motion-thresh", type=float, default=22, help="Motion sensitivity 1-50 (default: 22)")
    g.add_argument("--motion-v-min", type=float, default=60.0,
                   help="Min V (brightness) to keep motion pixel (default: 60)")
    g.add_argument("--motion-temporal-soft", dest="motion_temporal_soft",
                   action="store_true", default=False,
                   help="Use soft 3-frame temporal gate (default: off)")
    g.add_argument("--motion-temporal-strict", dest="motion_temporal_soft",
                   action="store_false",
                   help="Use strict 3-frame AND gate")
    g.add_argument("--motion-temporal-lo-frac", type=float, default=0.55,
                   help="Soft temporal low threshold as fraction of motion-thresh (default: 0.55)")
    g.add_argument("--motion-temporal-hi-mult", type=float, default=1.35,
                   help="Soft temporal strong-threshold multiplier (default: 1.35)")
    g.add_argument("--motion-flicker-suppress", dest="motion_flicker_suppress",
                   action="store_true", default=False,
                   help="Suppress one-frame flicker blobs in boost mask (default: off)")
    g.add_argument("--no-motion-flicker-suppress", dest="motion_flicker_suppress",
                   action="store_false",
                   help="Disable flicker suppression")
    g.add_argument("--motion-flicker-min-area", type=int, default=3,
                   help="Always drop boost components smaller than this area (default: 3)")
    g.add_argument("--motion-flicker-max-area", type=int, default=220,
                   help="Suppress unsupported components up to this area (default: 220)")
    g.add_argument("--motion-flicker-prev-dilate", type=int, default=9,
                   help="History support dilation kernel for flicker suppression (default: 9)")
    g.add_argument("--motion-flicker-keep-radius", type=float, default=0.11,
                   help="Keep-zone radius as frame-diagonal fraction around predicted ball (default: 0.11)")
    g.add_argument("--boost-max-blob", type=int, default=800, help="Max blob area for ball candidate (default: 800)")
    g.add_argument("--boost-min-blob", type=int, default=10, help="Min blob area (default: 0)")

    g = p.add_argument_group("Blob Filtering")
    g.add_argument("--no-shape-filter", action="store_true", help="Disable erosion + aspect filtering")
    g.add_argument("--blob-erode", type=int, default=3, help="Erosion kernel size (default: 3)")
    g.add_argument("--blob-max-aspect", type=float, default=4.0, help="Max blob aspect ratio (default: 4.0)")

    g = p.add_argument_group("Player & Court Detection")
    g.add_argument("--player-model", default="models/player.engine",
                   help="Player model path (.engine)")
    g.add_argument("--court-model", default="models/courtdetection.engine",
                   help="Court model path (.engine)")
    g.add_argument("--player-interval", type=int, default=8, help="Player detect every N frames (default: 5)")
    g.add_argument("--player-interval-stable", type=int, default=15,
                   help="Player detect interval when stable (default: 10)")
    g.add_argument("--num-players", type=int, default=4, help="Max players to track (default: 4)")
    g.add_argument("--court-interval", type=int, default=400, help="Court detect every N frames (default: 400)")
    g.add_argument("--court-conf", type=float, default=0.10, help="Court detection confidence (default: 0.10)")
    g.add_argument("--print-court-raw", dest="print_court_raw", action="store_true")
    g.add_argument("--no-print-court-raw", dest="print_court_raw", action="store_false")
    g.add_argument("--court-remap-semantic-14", action="store_true", help="Remap 14-pt keypoints to semantic order")
    g.add_argument("--court-points-only", action="store_true", help="Draw only keypoints, no lines")
    g.add_argument("--court-indices", action="store_true", help="Show keypoint index labels")
    g.add_argument("--player-conf", type=float, default=0.21, help="Player confidence (default: 0.21)")
    g.add_argument("--player-iou", type=float, default=0.10, help="Player IoU threshold (default: 0.10)")
    g.add_argument("--no-draw-players", action="store_true", help="Don't draw player boxes")
    g.add_argument("--no-draw-court", action="store_true", help="Don't draw court lines")

    g = p.add_argument_group("Acceleration")
    g.add_argument("--no-tensorrt", action="store_true", help="Disable TensorRT")
    g.add_argument("--no-nvenc", action="store_true", help="Disable NVENC encoding")
    g.add_argument("--trt-async", dest="trt_async_execute", action="store_true",
                   help="Use TensorRT async execution on CUDA streams")
    g.add_argument("--no-trt-async", dest="trt_async_execute", action="store_false",
                   help="Use synchronous TensorRT execute_v2 path")
    g.add_argument("--trt-async-slots", type=int, default=3,
                   help="Number of async output slots for ball TRT (default: 3)")
    g.add_argument("--info", dest="info_timing", action="store_true",
                   help="Print per-stage timing breakdown")
    g.add_argument("--no-info", dest="info_timing", action="store_false",
                   help="Disable per-stage timing breakdown (default)")

    g = p.add_argument_group("Debug")
    g.add_argument("--debug-video", dest="debug_video", action="store_true", default=False, help="Save debug video (default: off)")
    g.add_argument("--no-debug-video", dest="debug_video", action="store_false", help="Disable debug video")
    g.add_argument("--debug-path", default="output_videos/prof_test_motion_debug.mp4", help="Debug video path")
    g.add_argument("--yolo-debug-path", default="output_videos/prof_test_yolo_input_debug.mp4", help="YOLO input debug path")
    g.add_argument("--debug-show-raw-motion", dest="debug_show_raw_motion", action="store_true", default=False,
                   help="Show noisy raw motion layer in debug video (default: off)")
    g.add_argument("--debug-hide-raw-motion", dest="debug_show_raw_motion", action="store_false",
                   help="Hide raw motion layer and show filtered boost only")
    g.add_argument("--guide-video", dest="guide_video", action="store_true", default=False, help="Save guide progression video (default: off)")
    g.add_argument("--no-guide-video", dest="guide_video", action="store_false", help="Disable guide progression video")
    g.add_argument("--guide-path", default="output_videos/prof_test_guide_debug.mp4", help="Guide progression video path")
    g.add_argument("--guide-interp-gap", type=int, default=12, help="Max gap (frames) to interpolate guide path for debug video")
    g.add_argument("--print-selector-tracks", dest="print_selector_tracks", action="store_true",
                   help="Print selector track table and chosen track")
    g.add_argument("--no-print-selector-tracks", dest="print_selector_tracks", action="store_false",
                   help="Disable selector track table printing")
    g.add_argument("--selector-track-limit", type=int, default=0,
                   help="How many selector tracks to print (0=all, default: 0)")

    g = p.add_argument_group("Speed")
    g.add_argument("--skip-frame-yolo", type=int, default=1,
                   help="Run ball YOLO every Nth frame (1=every, 2=skip half, default: 1)")
    g.add_argument("--skip-frame-require-roi", dest="skip_frame_require_roi",
                   action="store_true", help="Only skip YOLO when ROI tracker has ball (default)")
    g.add_argument("--no-skip-frame-require-roi", dest="skip_frame_require_roi",
                   action="store_false", help="Skip YOLO unconditionally every Nth frame")
    g.add_argument("--skip-preprocess-dim", dest="skip_preprocess_dim",
                   action="store_true", help="Skip cosmetic dim_static when ball visible (default)")
    g.add_argument("--no-skip-preprocess-dim", dest="skip_preprocess_dim",
                   action="store_false", help="Always apply dim_static in preprocess")
    g.add_argument("--aux-on-yolo-frames", dest="aux_detect_on_yolo_frames",
                   action="store_true", help="Run player/court detection mainly on YOLO frames (default)")
    g.add_argument("--no-aux-on-yolo-frames", dest="aux_detect_on_yolo_frames",
                   action="store_false", help="Run player/court detection every frame call")
    g.add_argument("--aux-force-interval", type=int, default=6,
                   help="Force player/court detection at least every N frames (default: 6)")
    g.add_argument("--reader-prefetch", type=int, default=8,
                   help="Threaded frame-reader queue depth (default: 8)")
    g.add_argument("--cache-pass2-frames", dest="cache_input_frames_pass2",
                   action="store_true", help="Cache decoded input frames in RAM to avoid pass2 decode (default)")
    g.add_argument("--no-cache-pass2-frames", dest="cache_input_frames_pass2",
                   action="store_false", help="Disable RAM frame cache and always decode in pass2")
    g.add_argument("--pass2-cache-max-mb", type=int, default=768,
                   help="Disable RAM frame cache if estimated usage exceeds this MB (default: 768)")
    g.set_defaults(
        skip_frame_require_roi=True,
        skip_preprocess_dim=True,
        aux_detect_on_yolo_frames=True,
        cache_input_frames_pass2=True,
    )

    g = p.add_argument_group("ROI Motion")
    g.add_argument("--roi-motion", dest="roi_motion_enabled", action="store_true",
                   help="Enable ROI-based motion detection")
    g.add_argument("--no-roi-motion", dest="roi_motion_enabled", action="store_false",
                   help="Disable ROI-based motion detection (use full-frame)")
    g.add_argument("--roi-visible-radius", type=float, default=0.1,
                   help="ROI radius when ball visible (frame diag frac, default: 0.06)")
    g.add_argument("--roi-lost-radius", type=float, default=0.2,
                   help="ROI radius when ball lost (frame diag frac, default: 0.14)")
    g.add_argument("--roi-fullframe-interval", type=int, default=0,
                   help="Full-frame CC every N frames (0=never, default: 0)")
    g.set_defaults(roi_motion_enabled=True)

    g = p.add_argument_group("Trail")
    g.add_argument("--trail-hard-switch-x-frac", type=float, default=0.30,
                   help="Break trail segment if per-frame x jump exceeds this * court x span (4->7)")
    g.add_argument("--trail-hard-switch-y-frac", type=float, default=0.30,
                   help="Break trail segment if per-frame y jump exceeds this * court y span (0->4)")
    g.add_argument("--ball-marker-scale", type=float, default=0.22,
                   help="Marker radius scale from ball bbox size (default: 0.22)")
    g.add_argument("--ball-marker-min-r", type=int, default=3,
                   help="Minimum marker radius (default: 3)")
    g.add_argument("--ball-marker-max-r", type=int, default=12,
                   help="Maximum marker radius (default: 12)")

    p.set_defaults(print_court_raw=False)
    p.set_defaults(print_selector_tracks=True)
    p.set_defaults(trt_async_execute=True)
    p.set_defaults(info_timing=False)
    args = p.parse_args()

    cfg = Config(
        input_video=args.input,
        output_video=args.output,
        model_path=args.model,
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
        save_motion_debug=args.debug_video,
        output_debug_path=args.debug_path,
        output_yolo_input_debug_path=args.yolo_debug_path,
        debug_show_raw_motion=args.debug_show_raw_motion,
        save_guide_video=args.guide_video,
        output_guide_path=args.guide_path,
        guide_interp_max_gap=max(1, int(args.guide_interp_gap)),
        print_selector_tracks=args.print_selector_tracks,
        selector_track_limit=max(0, int(args.selector_track_limit)),
        trail_hard_switch_x_frac=args.trail_hard_switch_x_frac,
        trail_hard_switch_y_frac=args.trail_hard_switch_y_frac,
        ball_marker_box_scale=args.ball_marker_scale,
        ball_marker_min_radius=args.ball_marker_min_r,
        ball_marker_max_radius=args.ball_marker_max_r,
    )
    run(cfg)