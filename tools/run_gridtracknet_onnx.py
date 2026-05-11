#!/usr/bin/env python3
"""Run GridTrackNet ONNX on a video and export an overlay video plus JSON."""

from __future__ import annotations

import argparse
import json
import math
import os
import site
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


WIDTH = 768
HEIGHT = 432
IMGS_PER_INSTANCE = 5
GRID_COLS = 48
GRID_ROWS = 27
GRID_SIZE_COL = WIDTH / GRID_COLS
GRID_SIZE_ROW = HEIGHT / GRID_ROWS


def _session_providers(name: str) -> Sequence[Any]:
    key = name.lower()
    if key == "cuda":
        return [
            ("CUDAExecutionProvider", {"cudnn_conv_use_max_workspace": "1"}),
            "CPUExecutionProvider",
        ]
    if key == "tensorrt":
        return [
            ("TensorrtExecutionProvider", {"trt_fp16_enable": "1"}),
            ("CUDAExecutionProvider", {"cudnn_conv_use_max_workspace": "1"}),
            "CPUExecutionProvider",
        ]
    return ["CPUExecutionProvider"]


def _add_nvidia_dll_directories() -> None:
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    added: List[str] = []
    roots: List[Path] = []
    for base in site.getsitepackages():
        roots.append(Path(base) / "nvidia")
    try:
        roots.append(Path(site.getusersitepackages()) / "nvidia")
    except Exception:
        pass
    for root in roots:
        if not root.exists():
            continue
        for bin_dir in root.glob("*/bin"):
            if bin_dir.is_dir():
                added.append(str(bin_dir))
                try:
                    os.add_dll_directory(str(bin_dir))
                except OSError:
                    pass
    if added:
        os.environ["PATH"] = os.pathsep.join(added) + os.pathsep + os.environ.get("PATH", "")


def _make_units(frames: Sequence[np.ndarray]) -> np.ndarray:
    units: List[List[np.ndarray]] = []
    for i in range(0, len(frames), IMGS_PER_INSTANCE):
        batch = frames[i:i + IMGS_PER_INSTANCE]
        if len(batch) != IMGS_PER_INSTANCE:
            break
        unit: List[np.ndarray] = []
        for frame in batch:
            resized = cv2.resize(frame, (WIDTH, HEIGHT))
            chw = np.moveaxis(resized, -1, 0)  # BGR, matching GridTrackNet DataGen.py.
            unit.extend([chw[0], chw[1], chw[2]])
        units.append(unit)
    return np.asarray(units, dtype=np.float32) / 255.0


def _decode_predictions(
    y_pred: np.ndarray,
    output_w: int,
    output_h: int,
    threshold: float,
) -> List[Dict[str, Any]]:
    y_pred = np.split(y_pred, IMGS_PER_INSTANCE, axis=1)
    y_pred = np.stack(y_pred, axis=2)
    y_pred = np.moveaxis(y_pred, 1, -1)
    conf_grid, x_grid, y_grid = np.split(y_pred, 3, axis=-1)
    conf_grid = np.squeeze(conf_grid, axis=-1)
    x_grid = np.squeeze(x_grid, axis=-1)
    y_grid = np.squeeze(y_grid, axis=-1)

    rows: List[Dict[str, Any]] = []
    for i in range(conf_grid.shape[0]):
        for j in range(conf_grid.shape[1]):
            conf = conf_grid[i][j]
            max_conf = float(np.max(conf))
            pred_row, pred_col = np.unravel_index(np.argmax(conf), conf.shape)
            x_offset = float(x_grid[i][j][pred_row][pred_col])
            y_offset = float(y_grid[i][j][pred_row][pred_col])
            x_pred = int((x_offset + pred_col) * GRID_SIZE_COL)
            y_pred_px = int((y_offset + pred_row) * GRID_SIZE_ROW)
            cand_x = int((x_pred / WIDTH) * output_w)
            cand_y = int((y_pred_px / HEIGHT) * output_h)
            present = max_conf >= threshold
            rows.append({
                "present": bool(present),
                "x": cand_x if present else None,
                "y": cand_y if present else None,
                "candidate_x": cand_x,
                "candidate_y": cand_y,
                "conf": max_conf,
                "source": "gridtracknet_onnx",
                "interpolated": False,
            })
    return rows


def _circle_color(conf: float) -> Tuple[int, int, int]:
    if conf >= 0.9:
        return (0, 255, 90)
    if conf >= 0.8:
        return (0, 220, 255)
    return (0, 130, 255)


def _draw_overlay(
    frame: np.ndarray,
    row: Dict[str, Any],
    trail: Sequence[Tuple[int, int]],
    frame_idx: int,
    provider: str,
    threshold: float,
    draw_candidates: bool,
    jump_from: Optional[Tuple[int, int]],
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, 0), (min(w, 520), 56), (0, 0, 0), -1)
    cv2.putText(
        out,
        f"GridTrackNet ONNX | {provider} | thr={threshold:.2f} | frame={frame_idx}",
        (14, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (230, 230, 230),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        f"conf={float(row.get('conf', 0.0)):.3f}",
        (14, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )

    for idx, pt in enumerate(trail[-18:]):
        alpha = (idx + 1) / max(len(trail[-18:]), 1)
        radius = max(2, int(round(2 + 4 * alpha)))
        cv2.circle(out, pt, radius, (0, int(180 + 60 * alpha), 255), -1, cv2.LINE_AA)

    if row.get("present"):
        pt = (int(row["x"]), int(row["y"]))
        color = _circle_color(float(row.get("conf", 0.0)))
        if jump_from is not None:
            cv2.line(out, jump_from, pt, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(out, "JUMP", (pt[0] + 12, pt[1] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.circle(out, pt, 10, color, 2, cv2.LINE_AA)
        cv2.circle(out, pt, 3, color, -1, cv2.LINE_AA)
    elif draw_candidates:
        pt = (int(row["candidate_x"]), int(row["candidate_y"]))
        cv2.circle(out, pt, 6, (145, 145, 145), 1, cv2.LINE_AA)
    return out


def _large_jump_summary(frames: Sequence[Dict[str, Any]], jump_px: float) -> Tuple[int, float]:
    jumps = 0
    max_step = 0.0
    last: Optional[Dict[str, Any]] = None
    for row in frames:
        if not row.get("present"):
            last = None
            continue
        if last is not None:
            dist = math.hypot(float(row["x"]) - float(last["x"]), float(row["y"]) - float(last["y"]))
            if dist > jump_px:
                jumps += 1
            max_step = max(max_step, dist)
        last = row
    return jumps, max_step


def run(args: argparse.Namespace) -> Dict[str, Any]:
    model_path = Path(args.model).resolve()
    input_path = Path(args.input).resolve()
    output_video = Path(args.output_video).resolve()
    output_json = Path(args.output_json).resolve()

    if args.provider.lower() in {"cuda", "tensorrt"}:
        _add_nvidia_dll_directories()

    import onnxruntime as ort

    if args.provider.lower() in {"cuda", "tensorrt"} and hasattr(ort, "preload_dlls"):
        ort.preload_dlls()

    sess = ort.InferenceSession(str(model_path), providers=_session_providers(args.provider))
    actual_provider = ",".join(sess.get_providers())
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_video.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    if not args.no_video:
        writer = cv2.VideoWriter(
            str(output_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open output video: {output_video}")

    frames_json: List[Dict[str, Any]] = []
    frame_buf: List[np.ndarray] = []
    raw_buf: List[np.ndarray] = []
    frame_start = 0
    trail: List[Tuple[int, int]] = []
    last_present_pt: Optional[Tuple[int, int]] = None
    infer_sec = 0.0
    pre_sec = 0.0
    decode_sec = 0.0
    write_sec = 0.0
    start = time.perf_counter()
    chunk_frames = max(IMGS_PER_INSTANCE, int(args.batch_groups) * IMGS_PER_INSTANCE)

    def flush(usable_only: bool = True) -> int:
        nonlocal frame_start, infer_sec, pre_sec, write_sec, trail, last_present_pt
        usable = (len(frame_buf) // IMGS_PER_INSTANCE) * IMGS_PER_INSTANCE
        if not usable and usable_only:
            return 0
        batch_frames = frame_buf[:usable]
        batch_raw = raw_buf[:usable]
        if not batch_frames:
            return 0
        t0 = time.perf_counter()
        units = _make_units(batch_frames)
        pre_sec += time.perf_counter() - t0
        t0 = time.perf_counter()
        y_pred = sess.run([output_name], {input_name: units})[0]
        infer_sec += time.perf_counter() - t0
        rows = _decode_predictions(y_pred, w, h, args.threshold)
        for off, row in enumerate(rows):
            row["frame"] = frame_start + off
            jump_from = None
            if row.get("present"):
                pt = (int(row["x"]), int(row["y"]))
                if last_present_pt is not None:
                    dist = math.hypot(pt[0] - last_present_pt[0], pt[1] - last_present_pt[1])
                    if dist > float(args.jump_px):
                        jump_from = last_present_pt
                trail.append(pt)
                last_present_pt = pt
            else:
                last_present_pt = None
            frames_json.append(row)
            if writer is not None:
                vis = _draw_overlay(
                    batch_raw[off],
                    row,
                    trail,
                    frame_start + off,
                    actual_provider,
                    args.threshold,
                    args.draw_candidates,
                    jump_from,
                )
                t1 = time.perf_counter()
                writer.write(vis)
                write_sec += time.perf_counter() - t1
        return usable

    while True:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        decode_sec += time.perf_counter() - t0
        if not ok:
            break
        frame_buf.append(frame)
        raw_buf.append(frame)
        if len(frame_buf) >= chunk_frames:
            used = flush()
            frame_start += used
            del frame_buf[:used]
            del raw_buf[:used]
            if args.progress_every and frame_start and frame_start % args.progress_every == 0:
                elapsed = time.perf_counter() - start
                print(f"[gridtracknet] {frame_start}/{total} frames, {frame_start / max(elapsed, 1e-6):.2f} fps")

    if frame_buf:
        used = flush()
        frame_start += used
        del frame_buf[:used]
        del raw_buf[:used]

    # Write tail frames that cannot form a 5-frame model instance.
    for frame in raw_buf:
        row = {
            "frame": len(frames_json),
            "present": False,
            "x": None,
            "y": None,
            "candidate_x": None,
            "candidate_y": None,
            "conf": 0.0,
            "source": "gridtracknet_onnx",
            "interpolated": False,
        }
        frames_json.append(row)
        if writer is not None:
            vis = _draw_overlay(frame, row, trail, row["frame"], actual_provider,
                                args.threshold, False, None)
            writer.write(vis)

    cap.release()
    if writer is not None:
        writer.release()

    elapsed = time.perf_counter() - start
    filled = sum(1 for row in frames_json if row.get("present"))
    jumps, max_step = _large_jump_summary(frames_json, args.jump_px)
    payload = {
        "schema_version": "gridtracknet_onnx_v1",
        "video": {
            "path": str(input_path),
            "width": w,
            "height": h,
            "fps": fps,
            "frames": total,
        },
        "summary": {
            "filled_frames": filled,
            "filled_percent": 100.0 * filled / max(total, 1),
            "elapsed_sec": elapsed,
            "effective_fps": len(frames_json) / max(elapsed, 1e-6),
            "inference_sec": infer_sec,
            "preprocess_sec": pre_sec,
            "decode_sec": decode_sec,
            "write_sec": write_sec,
            "large_jumps_consecutive_present": jumps,
            "max_consecutive_present_step_px": max_step,
            "threshold": float(args.threshold),
        },
        "config": {
            "model": str(model_path),
            "provider_requested": args.provider,
            "providers_actual": sess.get_providers(),
            "input_width": WIDTH,
            "input_height": HEIGHT,
            "frames_per_instance": IMGS_PER_INSTANCE,
            "draw_candidates": bool(args.draw_candidates),
        },
        "frames": frames_json,
    }
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload["summary"], indent=2))
    print(f"[gridtracknet] providers: {actual_provider}")
    if writer is not None:
        print(f"[gridtracknet] video: {output_video}")
    print(f"[gridtracknet] json: {output_json}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run GridTrackNet ONNX on a video.")
    parser.add_argument("--input", required=True, help="Input .mp4 video")
    parser.add_argument("--model", default=".codex_tmp/gridtracknet.onnx", help="GridTrackNet ONNX model")
    parser.add_argument("--output-video", required=True, help="Overlay output .mp4")
    parser.add_argument("--output-json", required=True, help="Tracking JSON output")
    parser.add_argument("--provider", choices=("cuda", "tensorrt", "cpu"), default="cuda")
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--batch-groups", type=int, default=20,
                        help="Number of 5-frame instances per inference call")
    parser.add_argument("--jump-px", type=float, default=120.0)
    parser.add_argument("--draw-candidates", action="store_true",
                        help="Draw low-confidence max-grid candidates too")
    parser.add_argument("--no-video", action="store_true", help="Only write JSON")
    parser.add_argument("--progress-every", type=int, default=500)
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
