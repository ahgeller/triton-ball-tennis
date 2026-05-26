"""Render an explainable tracking/3D diagnostic video.

This is meant for humans, not benchmarks.  It overlays the selected 2D track,
source/confidence, court keypoints, player boxes, event labels, and optional
trajectory3d_v2 output on top of the original video.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from trajectory3d_v1 import _draw_mini_court_overlay, _import_cv2, load_tracking_json
from trajectory3d_v2 import COURT_LINE_PAIRS_14


SOURCE_COLORS = {
    "det": (50, 235, 90),
    "motion": (0, 165, 255),
    "guide": (255, 230, 40),
    "carry": (255, 150, 40),
    "interp": (60, 220, 220),
    "": (200, 200, 200),
}

EVENT_COLORS = {
    "bounce": (0, 255, 255),
    "bounce_candidate": (0, 170, 220),
    "hit": (255, 80, 255),
    "hit_candidate": (180, 60, 200),
    "serve_toss_apex": (80, 180, 255),
}


def _json_load(path: Optional[str | Path]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_boxes(value: Any) -> List[List[float]]:
    if value is None:
        return []
    raw = list(value.values()) if isinstance(value, dict) else value
    boxes: List[List[float]] = []
    for box in raw or []:
        if box is None or len(box) < 4:
            continue
        try:
            boxes.append([float(v) for v in box[:4]])
        except Exception:
            continue
    return boxes


def _frame_rows_by_frame(tracking: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for row in tracking.get("frames", []) or []:
        try:
            out[int(row.get("frame", len(out)))] = row
        except Exception:
            continue
    return out


def _trajectory_rows_by_frame(trajectory: Optional[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    if not trajectory:
        return out
    for segment in trajectory.get("segments", []) or []:
        for row in segment.get("frames", []) or []:
            try:
                out[int(row["frame"])] = row
            except Exception:
                continue
    return out


def _events_by_frame(trajectory: Optional[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    out: Dict[int, List[Dict[str, Any]]] = {}
    if not trajectory:
        return out
    for ev in trajectory.get("events", []) or []:
        frame = ev.get("frame", ev.get("start_frame"))
        if frame is None:
            continue
        try:
            out.setdefault(int(frame), []).append(ev)
        except Exception:
            continue
    return out


def _draw_text_panel(frame, lines: Sequence[Tuple[str, Tuple[int, int, int]]]) -> None:
    cv2 = _import_cv2()
    x1, y1 = 14, 14
    line_h = 24
    width = 640
    height = 22 + line_h * len(lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x1 + width, y1 + height), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x1 + width, y1 + height), (210, 210, 210), 1)
    y = y1 + 26
    for text, color in lines:
        cv2.putText(frame, text, (x1 + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 1, cv2.LINE_AA)
        y += line_h


def _draw_court_keypoints(frame, kps: Optional[Sequence[float]]) -> None:
    cv2 = _import_cv2()
    if not kps or len(kps) < 8:
        return
    n = len(kps) // 2
    if n == 14:
        for a, b in COURT_LINE_PAIRS_14:
            if a >= n or b >= n:
                continue
            ax, ay = float(kps[a * 2]), float(kps[a * 2 + 1])
            bx, by = float(kps[b * 2]), float(kps[b * 2 + 1])
            if (abs(ax) > 1e-6 or abs(ay) > 1e-6) and (abs(bx) > 1e-6 or abs(by) > 1e-6):
                cv2.line(frame, (int(round(ax)), int(round(ay))), (int(round(bx)), int(round(by))), (60, 220, 60), 1, cv2.LINE_AA)
    for i in range(n):
        x, y = float(kps[i * 2]), float(kps[i * 2 + 1])
        if abs(x) <= 1e-6 and abs(y) <= 1e-6:
            continue
        p = (int(round(x)), int(round(y)))
        cv2.circle(frame, p, 3, (0, 255, 0), -1)
        cv2.putText(frame, str(i), (p[0] + 4, p[1] - 4), cv2.FONT_HERSHEY_PLAIN, 0.8, (210, 255, 210), 1)


def _draw_player_boxes(frame, boxes: Sequence[Sequence[float]]) -> None:
    cv2 = _import_cv2()
    for box in boxes:
        if len(box) < 4:
            continue
        x1, y1, x2, y2 = [int(round(float(v))) for v in box[:4]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 180, 60), 1, cv2.LINE_AA)


def _bad_trail_step(
    prev_pt: Tuple[int, float, float, str],
    cur_pt: Tuple[int, float, float, str],
    max_gap_frames: int = 5,
    max_speed_px_per_frame: float = 42.0,
    max_distance_px: float = 190.0,
) -> Tuple[bool, float, float]:
    df = max(1, int(cur_pt[0]) - int(prev_pt[0]))
    dist = math.hypot(float(cur_pt[1]) - float(prev_pt[1]), float(cur_pt[2]) - float(prev_pt[2]))
    speed = dist / float(df)
    bad = df > max_gap_frames or dist > max_distance_px or speed > max_speed_px_per_frame
    return bad, dist, speed


def _draw_trail(frame, trail: Sequence[Tuple[int, float, float, str]]) -> None:
    cv2 = _import_cv2()
    if len(trail) < 2:
        return
    for i in range(1, len(trail)):
        prev_pt = trail[i - 1]
        cur_pt = trail[i]
        bad, _, _ = _bad_trail_step(prev_pt, cur_pt)
        if bad:
            continue
        _, x0, y0, src0 = prev_pt
        _, x1, y1, src1 = cur_pt
        age = i / max(1, len(trail) - 1)
        base = SOURCE_COLORS.get(src1 or src0, (200, 200, 200))
        color = tuple(int(c * (0.35 + 0.65 * age)) for c in base)
        thickness = max(1, int(round(1 + 3 * age)))
        cv2.line(frame, (int(round(x0)), int(round(y0))), (int(round(x1)), int(round(y1))), color, thickness, cv2.LINE_AA)


def _draw_observation(frame, row: Optional[Dict[str, Any]]) -> Optional[Tuple[float, float, str]]:
    cv2 = _import_cv2()
    if not row or not row.get("present"):
        return None
    x = row.get("x")
    y = row.get("y")
    if x is None or y is None:
        return None
    x = float(x)
    y = float(y)
    source = str(row.get("source", "") or "")
    color = SOURCE_COLORS.get(source, (210, 210, 210))
    p = (int(round(x)), int(round(y)))
    bbox = row.get("bbox")
    if bbox and len(bbox) >= 4:
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    search = row.get("search") or {}
    sr = float(search.get("radius") or 0.0)
    if sr > 1:
        sx = float(search.get("x") or x)
        sy = float(search.get("y") or y)
        cv2.circle(frame, (int(round(sx)), int(round(sy))), int(round(sr)), (255, 0, 220), 1, cv2.LINE_AA)
    cv2.circle(frame, p, 7, (0, 0, 0), -1)
    cv2.circle(frame, p, 5, color, -1)
    cv2.drawMarker(frame, p, (255, 255, 255), cv2.MARKER_CROSS, 18, 1, cv2.LINE_AA)
    return x, y, source


def _traj_row_is_usable(traj_row: Optional[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    if not traj_row:
        return False, ["no_3d_row"]
    flags: List[str] = []
    if traj_row.get("xyz_m") is None:
        flags.append("no_xyz")
    ambiguity = str(traj_row.get("ambiguity") or "")
    if ambiguity == "high":
        flags.append("high_ambiguity")
    conf = traj_row.get("confidence")
    try:
        if conf is None or float(conf) < 0.25:
            flags.append("low_3d_confidence")
    except Exception:
        flags.append("bad_3d_confidence")
    residual = traj_row.get("residual_px")
    try:
        if residual is not None and float(residual) > 35.0:
            flags.append("large_3d_residual")
    except Exception:
        flags.append("bad_3d_residual")
    xyz = traj_row.get("xyz_m")
    try:
        if xyz and len(xyz) >= 3 and (float(xyz[2]) < -0.05 or float(xyz[2]) > 8.0):
            flags.append("implausible_3d_height")
    except Exception:
        flags.append("bad_xyz")
    return len(flags) == 0, flags


def _draw_trajectory_projection(frame, traj_row: Optional[Dict[str, Any]]) -> None:
    cv2 = _import_cv2()
    usable, _ = _traj_row_is_usable(traj_row)
    if not usable:
        return
    if not traj_row:
        return
    p = traj_row.get("projected_2d")
    if not p or len(p) < 2:
        return
    x, y = int(round(float(p[0]))), int(round(float(p[1])))
    cv2.circle(frame, (x, y), 9, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "3D", (x + 10, y - 8), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 0, 255), 1)


def render_visualization(
    tracking_json: str | Path,
    output_video: str | Path,
    input_video: Optional[str | Path] = None,
    trajectory_json: Optional[str | Path] = None,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    trail_frames: int = 90,
) -> None:
    cv2 = _import_cv2()
    tracking = load_tracking_json(tracking_json)
    trajectory = _json_load(trajectory_json)

    video_info = tracking.get("video") or {}
    source = Path(input_video) if input_video else Path(str(video_info.get("input", "")))
    if not source.exists():
        raise FileNotFoundError(f"Input video not found: {source}")
    fps = float(video_info.get("fps") or 30.0)
    total_frames = int(video_info.get("total_frames") or 0)
    if end_frame is None:
        end_frame = max(0, total_frames - 1)
    start_frame = max(0, int(start_frame))
    end_frame = max(start_frame, int(end_frame))

    tracking_rows = _frame_rows_by_frame(tracking)
    trajectory_rows = _trajectory_rows_by_frame(trajectory)
    events = _events_by_frame(trajectory)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {source}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or video_info.get("width") or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or video_info.get("height") or 0)
    out_path = Path(output_video)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    trail_2d: List[Tuple[int, float, float, str]] = []
    trail_3d: List[Tuple[float, float]] = []
    last_obs_for_quality: Optional[Tuple[int, float, float, str]] = None
    frame_idx = start_frame
    while frame_idx <= end_frame:
        ok, frame = cap.read()
        if not ok:
            break

        row = tracking_rows.get(frame_idx)
        traj_row = trajectory_rows.get(frame_idx)
        _draw_court_keypoints(frame, None if row is None else row.get("court_keypoints"))
        _draw_player_boxes(frame, [] if row is None else _normalize_boxes(row.get("player_boxes")))

        obs = None
        jump_warning = None
        if row and row.get("present") and row.get("x") is not None and row.get("y") is not None:
            obs = (float(row["x"]), float(row["y"]), str(row.get("source", "") or ""))
            cur_quality_pt = (frame_idx, obs[0], obs[1], obs[2])
            if last_obs_for_quality is not None:
                bad_jump, dist, speed = _bad_trail_step(last_obs_for_quality, cur_quality_pt)
                if bad_jump:
                    jump_warning = f"TRACK JUMP dist={dist:.0f}px speed={speed:.1f}px/f"
            last_obs_for_quality = cur_quality_pt
        if obs is not None:
            trail_2d.append((frame_idx, obs[0], obs[1], obs[2]))
            trail_2d = [t for t in trail_2d if frame_idx - t[0] <= trail_frames]
        _draw_trail(frame, trail_2d)
        _draw_observation(frame, row)
        _draw_trajectory_projection(frame, traj_row)

        traj_usable, traj_flags = _traj_row_is_usable(traj_row)
        xyz = None if traj_row is None or not traj_usable else traj_row.get("xyz_m")
        ball_xy_m = None
        ball_z_m = None
        if xyz and len(xyz) >= 3:
            ball_xy_m = (float(xyz[0]), float(xyz[1]))
            ball_z_m = float(xyz[2])
            trail_3d.append(ball_xy_m)
            if len(trail_3d) > trail_frames:
                trail_3d = trail_3d[-trail_frames:]
        _draw_mini_court_overlay(frame, ball_xy_m, ball_z_m, trail_3d, court_w=10.97, court_l=23.77)

        evs = events.get(frame_idx, [])
        for ev in evs[:4]:
            ex = ev.get("x")
            ey = ev.get("y")
            typ = str(ev.get("type", "event"))
            color = EVENT_COLORS.get(typ, (255, 255, 255))
            if ex is not None and ey is not None:
                p = (int(round(float(ex))), int(round(float(ey))))
                cv2.circle(frame, p, 14, color, 2, cv2.LINE_AA)
                cv2.putText(frame, typ, (p[0] + 14, p[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        source = "--" if row is None else str(row.get("source", "--") or "--")
        conf = None if row is None else row.get("conf")
        present = bool(row and row.get("present"))
        hyp = "--" if traj_row is None else str(traj_row.get("selected_hypothesis") or "--")
        ambiguity = "--" if traj_row is None else str(traj_row.get("ambiguity") or "--")
        residual = None if traj_row is None else traj_row.get("residual_px")
        lines = [
            (f"frame {frame_idx} / {total_frames - 1}   t={frame_idx / max(fps, 1e-9):.2f}s", (255, 255, 255)),
            (f"2D ball: {'present' if present else 'missing'}   source={source}   conf={float(conf):.2f}" if conf is not None else f"2D ball: {'present' if present else 'missing'}", SOURCE_COLORS.get(source, (220, 220, 220))),
            (f"3D: hypothesis={hyp}   ambiguity={ambiguity}   {'shown' if traj_usable else 'hidden'}", (255, 180, 255) if traj_usable else (150, 150, 150)),
            (
                "xyz: --"
                if xyz is None
                else f"xyz=({float(xyz[0]):.2f}, {float(xyz[1]):.2f}, {float(xyz[2]):.2f}) m   residual={float(residual):.1f}px" if residual is not None else f"xyz=({float(xyz[0]):.2f}, {float(xyz[1]):.2f}, {float(xyz[2]):.2f}) m",
                (210, 255, 210),
            ),
        ]
        if evs:
            lines.append(("events: " + ", ".join(str(e.get("type")) for e in evs[:4]), (0, 255, 255)))
        if jump_warning:
            lines.append((jump_warning, (0, 80, 255)))
        if not traj_usable and traj_flags:
            lines.append(("3D hidden: " + ", ".join(traj_flags[:3]), (160, 160, 160)))
        _draw_text_panel(frame, lines)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()


def main() -> int:
    parser = argparse.ArgumentParser(description="Render an explainable tracking diagnostic video.")
    parser.add_argument("--tracking-json", required=True)
    parser.add_argument("--trajectory-json", default=None, help="Optional trajectory3d_v2 JSON")
    parser.add_argument("--input-video", default=None, help="Override input video; defaults to tracking JSON video.input")
    parser.add_argument("--output-video", required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--trail-frames", type=int, default=90)
    args = parser.parse_args()

    render_visualization(
        tracking_json=args.tracking_json,
        trajectory_json=args.trajectory_json,
        input_video=args.input_video,
        output_video=args.output_video,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        trail_frames=args.trail_frames,
    )
    print(f"[visualize] Output video: {args.output_video}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
