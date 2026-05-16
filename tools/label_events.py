"""Tennis event labeling tool.

Produces training data for a small event-classification model:
per-frame ground-truth labels of HIT / BOUNCE / OFF-FRAME, plus an optional
court (X, Y) for the on-court events.

Workflow:
  python tools/label_events.py \
      --video       input_videos/<clip>.mp4 \
      --tracking    output_videos/<clip>_tracking.json \
      --output      validation/labels/<clip>_events.json

A cv2 window opens; navigate frames with the keyboard. On a frame where
something happened, press the appropriate key - the event is recorded
immediately and you keep moving. To attach a court position to a hit or
bounce, click on the mini-court overlay in the top-right; the click is
auto-attached to the most recent event of that frame.

Keys:
  RIGHT / SPACE / n  next frame
  LEFT  / b          previous frame
  SHIFT+RIGHT        +10 frames
  SHIFT+LEFT         -10 frames
  PgDn / ]           +60 frames (~1 second at 60 fps)
  PgUp / [           -60 frames
  Home               first frame
  End                last labeled frame

  h                  mark this frame as a HIT (player racket contact)
  v                  mark this frame as a BOUNCE (ball touched the court)
  f                  mark this frame as OFF-FRAME (ball gone / out of frame)
  Backspace          remove the event on this frame
  c                  clear the court (X, Y) attached to this frame's event

  s                  save and continue
  q / Esc            save and quit

Output JSON schema:
  {
    "video": <path>,
    "fps": <number>,
    "frame_count": <int>,
    "events": [
      {"frame": int, "time_sec": float, "type": "hit"|"bounce"|"off_frame",
       "court_xy_m": [x, y] | null,
       "ball_uv_pix": [u, v] | null,
       "player_boxes": {id: [x1, y1, x2, y2], ...} | null}
    ]
  }
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Make the package importable when the script is run as a file from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.mini_court import (
    MiniCourtLayout,
    compute_layout,
    draw_mini_court,
    draw_point,
)


# ---------- Color helpers ----------------------------------------------------

COLOR_HIT = (0, 255, 255)      # yellow
COLOR_BOUNCE = (0, 165, 255)   # orange
COLOR_OFF = (180, 180, 180)    # gray
COLOR_BALL_DETECT = (0, 255, 0)  # lime green

EVENT_COLORS = {
    "hit": COLOR_HIT,
    "bounce": COLOR_BOUNCE,
    "off_frame": COLOR_OFF,
}

EVENT_LABELS_VERBOSE = {
    "hit": "HIT (racket contact)",
    "bounce": "BOUNCE (ball on court)",
    "off_frame": "OFF-FRAME / lost",
}


# ---------- State ------------------------------------------------------------

@dataclass
class LabelEvent:
    frame: int
    type: str  # "hit" | "bounce" | "off_frame"
    court_xy_m: Optional[Tuple[float, float]] = None


@dataclass
class LabelState:
    by_frame: Dict[int, LabelEvent] = field(default_factory=dict)
    pending_click_frame: Optional[int] = None  # which frame the next click attaches to

    def set(self, frame: int, event_type: str) -> None:
        self.by_frame[frame] = LabelEvent(frame=frame, type=event_type)
        if event_type in ("hit", "bounce"):
            self.pending_click_frame = frame
        else:
            self.pending_click_frame = None

    def remove(self, frame: int) -> None:
        self.by_frame.pop(frame, None)
        if self.pending_click_frame == frame:
            self.pending_click_frame = None

    def clear_xy(self, frame: int) -> None:
        ev = self.by_frame.get(frame)
        if ev is not None:
            ev.court_xy_m = None
            self.pending_click_frame = frame if ev.type in ("hit", "bounce") else None

    def attach_xy(self, xy: Tuple[float, float]) -> Optional[int]:
        if self.pending_click_frame is None:
            return None
        ev = self.by_frame.get(self.pending_click_frame)
        if ev is None:
            return None
        ev.court_xy_m = xy
        attached_to = self.pending_click_frame
        self.pending_click_frame = None
        return attached_to

    def to_json(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for fr in sorted(self.by_frame):
            ev = self.by_frame[fr]
            out.append({
                "frame": int(fr),
                "type": ev.type,
                "court_xy_m": (
                    [float(ev.court_xy_m[0]), float(ev.court_xy_m[1])]
                    if ev.court_xy_m is not None else None
                ),
            })
        return out

    def load_from_json(self, events: List[Dict[str, Any]]) -> None:
        self.by_frame.clear()
        for e in events:
            fr = int(e["frame"])
            t = e["type"]
            xy = e.get("court_xy_m")
            ev = LabelEvent(frame=fr, type=t)
            if xy is not None and len(xy) >= 2:
                ev.court_xy_m = (float(xy[0]), float(xy[1]))
            self.by_frame[fr] = ev


# ---------- Tracking JSON access (read-only) --------------------------------

class TrackingLookup:
    """Per-frame access to tracking.json's ball detection + player boxes."""

    def __init__(self, tracking_json_path: Optional[Path]):
        self.frames: Dict[int, Dict[str, Any]] = {}
        if tracking_json_path is None or not tracking_json_path.exists():
            return
        try:
            with tracking_json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return
        for fr in data.get("frames", []) or []:
            try:
                idx = int(fr.get("frame"))
            except (TypeError, ValueError):
                continue
            self.frames[idx] = fr

    def ball_uv(self, frame: int) -> Optional[Tuple[float, float]]:
        fr = self.frames.get(frame)
        if fr is None or not fr.get("present"):
            return None
        try:
            return float(fr["x"]), float(fr["y"])
        except (KeyError, TypeError, ValueError):
            return None

    def player_boxes(self, frame: int) -> Dict[str, List[float]]:
        fr = self.frames.get(frame)
        if fr is None:
            return {}
        pb = fr.get("player_boxes") or {}
        if not isinstance(pb, dict):
            return {}
        out: Dict[str, List[float]] = {}
        for pid, bbox in pb.items():
            if bbox is None or len(bbox) < 4:
                continue
            try:
                out[str(pid)] = [float(bbox[0]), float(bbox[1]),
                                 float(bbox[2]), float(bbox[3])]
            except (TypeError, ValueError):
                continue
        return out


# ---------- Frame renderer --------------------------------------------------

def render_frame(
    base_frame: np.ndarray,
    frame_idx: int,
    fps: float,
    total_frames: int,
    state: LabelState,
    tracking: TrackingLookup,
    layout: MiniCourtLayout,
) -> np.ndarray:
    img = base_frame.copy()

    # 1) Source-frame overlay: highlight the ball detection (if present) + a
    # short trail of recent detections to help the eye lock on motion.
    trail_len = max(8, int(round(fps * 0.5)))
    trail: List[Tuple[float, float]] = []
    for j in range(max(0, frame_idx - trail_len), frame_idx + 1):
        b = tracking.ball_uv(j)
        if b is not None:
            trail.append(b)
    for p0, p1 in zip(trail, trail[1:]):
        cv2.line(img, (int(round(p0[0])), int(round(p0[1]))),
                 (int(round(p1[0])), int(round(p1[1]))),
                 (0, 200, 0), 2, cv2.LINE_AA)
    ball_now = tracking.ball_uv(frame_idx)
    if ball_now is not None:
        cv2.circle(img, (int(round(ball_now[0])), int(round(ball_now[1]))),
                   10, COLOR_BALL_DETECT, 2, cv2.LINE_AA)

    # 2) Player bboxes - useful for context (player IDs labeled).
    for pid, bbox in tracking.player_boxes(frame_idx).items():
        x1, y1, x2, y2 = (int(round(v)) for v in bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 180, 0), 1, cv2.LINE_AA)
        cv2.putText(img, f"P{pid}", (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 0), 1, cv2.LINE_AA)

    # 3) Mini-court overlay (always drawn last so it's on top).
    draw_mini_court(img, cv2, layout=layout)

    # 4) Past events visualized on the mini-court (faded by age).
    recent_window = 200  # frames
    for ev in state.by_frame.values():
        if abs(ev.frame - frame_idx) > recent_window:
            continue
        if ev.court_xy_m is None:
            continue
        age = abs(ev.frame - frame_idx) / recent_window
        color = EVENT_COLORS.get(ev.type, (200, 200, 200))
        faded = tuple(int(c * (1.0 - 0.6 * age)) for c in color)
        radius = 6 if ev.frame == frame_idx else 4
        draw_point(img, cv2, layout, ev.court_xy_m[0], ev.court_xy_m[1],
                   color=faded, radius=radius)

    # 5) Current frame's event banner.
    ev_here = state.by_frame.get(frame_idx)
    if ev_here is not None:
        color = EVENT_COLORS.get(ev_here.type, (255, 255, 255))
        label = EVENT_LABELS_VERBOSE.get(ev_here.type, ev_here.type.upper())
        xy_text = ""
        if ev_here.court_xy_m is not None:
            xy_text = f"  @ ({ev_here.court_xy_m[0]:.2f}, {ev_here.court_xy_m[1]:.2f}) m"
        cv2.rectangle(img, (10, 10), (640, 50), (0, 0, 0), -1)
        cv2.putText(img, f"{label}{xy_text}", (16, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    elif state.pending_click_frame == frame_idx:
        cv2.rectangle(img, (10, 10), (640, 50), (0, 0, 0), -1)
        cv2.putText(img, "Click mini-court to set position (or press 'c' to skip)",
                    (16, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 0), 1, cv2.LINE_AA)

    # 6) Status bar at bottom-left: frame index, time, count of labeled events.
    n_hits = sum(1 for e in state.by_frame.values() if e.type == "hit")
    n_bounces = sum(1 for e in state.by_frame.values() if e.type == "bounce")
    n_off = sum(1 for e in state.by_frame.values() if e.type == "off_frame")
    H = img.shape[0]
    bar_y = H - 80
    cv2.rectangle(img, (0, bar_y), (560, H), (0, 0, 0), -1)
    cv2.putText(img,
                f"frame {frame_idx} / {total_frames - 1}    t = {frame_idx / fps:6.2f}s",
                (12, bar_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(img,
                f"labeled: hits={n_hits}  bounces={n_bounces}  off={n_off}",
                (12, bar_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (200, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(img,
                "h=hit  v=bounce  f=off  Bksp=remove  c=clear xy  "
                "<-/-> step  []=+/-60  s=save  q=quit",
                (12, bar_y + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (160, 160, 160), 1, cv2.LINE_AA)
    return img


# ---------- Output -----------------------------------------------------------

def save_labels(out_path: Path, state: LabelState,
                video_path: Path, fps: float, frame_count: int) -> None:
    out: Dict[str, Any] = {
        "video": str(video_path),
        "fps": float(fps),
        "frame_count": int(frame_count),
        "events": state.to_json(),
        "schema_version": "events_v1",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


def load_labels_if_exists(path: Path, state: LabelState) -> bool:
    if not path.exists():
        return False
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return False
    state.load_from_json(data.get("events") or [])
    return True


# ---------- Main loop --------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Label hits / bounces / off-frame events for a tennis clip."
    )
    parser.add_argument("--video", required=True, help="Source MP4")
    parser.add_argument(
        "--tracking", default=None,
        help="Optional tracking.json (used to overlay ball detection + player boxes "
             "during labeling - strongly recommended).",
    )
    parser.add_argument("--output", required=True, help="Output labels JSON path")
    parser.add_argument(
        "--start-frame", type=int, default=0,
        help="Frame index to start at (default 0)",
    )
    args = parser.parse_args()

    video_path = Path(args.video).resolve()
    out_path = Path(args.output).resolve()
    tracking_path = Path(args.tracking).resolve() if args.tracking else None

    if not video_path.exists():
        raise SystemExit(f"video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"could not open video: {video_path}")
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise SystemExit("could not determine video frame count")

    tracking = TrackingLookup(tracking_path)
    layout = compute_layout(frame_w, frame_h)
    state = LabelState()

    resumed = load_labels_if_exists(out_path, state)
    if resumed:
        print(f"[label] resumed {len(state.by_frame)} existing events from {out_path}")

    # Cache frames by index for snappy back-and-forth scrubbing.
    frame_cache: Dict[int, np.ndarray] = {}
    cache_capacity = 512

    def read_frame(idx: int) -> Optional[np.ndarray]:
        if idx in frame_cache:
            return frame_cache[idx]
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, im = cap.read()
        if not ok or im is None:
            return None
        if len(frame_cache) >= cache_capacity:
            # Drop the oldest 64.
            for k in sorted(frame_cache)[:64]:
                frame_cache.pop(k, None)
        frame_cache[idx] = im
        return im

    win = "label_events"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    # Fit window to screen height so the entire frame is visible without
    # scrolling; preserve aspect ratio.
    target_h = 900
    target_w = int(round(frame_w * target_h / frame_h))
    cv2.resizeWindow(win, target_w, target_h)

    # Mouse: clicks on the mini-court attach an (X, Y) to the pending event.
    click_xy: Dict[str, Any] = {"received": None}

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        # The window may be resized; map back to original frame coords.
        try:
            wx, wy, ww, wh = cv2.getWindowImageRect(win)
        except Exception:
            ww, wh = target_w, target_h
        # Compute the scale that the window is displaying at.
        scale_x = frame_w / max(1, ww)
        scale_y = frame_h / max(1, wh)
        orig_x = int(x * scale_x)
        orig_y = int(y * scale_y)
        if layout.contains(orig_x, orig_y):
            world = layout.panel_to_world(orig_x, orig_y)
            if world is not None:
                click_xy["received"] = world

    cv2.setMouseCallback(win, on_mouse)

    idx = max(0, min(total - 1, int(args.start_frame)))
    dirty_since_save = False

    while True:
        base = read_frame(idx)
        if base is None:
            idx = max(0, idx - 1)
            continue

        img = render_frame(base, idx, fps, total, state, tracking, layout)
        cv2.imshow(win, img)
        key = cv2.waitKeyEx(20)

        if click_xy["received"] is not None:
            xy = click_xy["received"]
            click_xy["received"] = None
            attached_to = state.attach_xy(xy)
            if attached_to is not None:
                dirty_since_save = True

        if key < 0:
            # Window closed by the user-
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
            continue

        k = key & 0xFFFF

        # Navigation keys. cv2.waitKeyEx returns OS-specific extended codes for
        # arrow keys (Windows: 2424832 left, 2555904 right, etc.). We handle
        # both the ASCII and the Windows extended values.
        WIN_LEFT = 2424832
        WIN_RIGHT = 2555904
        WIN_UP = 2490368
        WIN_DOWN = 2621440

        moved = False
        if key in (ord('n'), ord(' '), 13, WIN_RIGHT):
            idx = min(total - 1, idx + 1)
            moved = True
        elif key in (ord('b'), WIN_LEFT):
            idx = max(0, idx - 1)
            moved = True
        elif key in (ord(']'),):
            idx = min(total - 1, idx + 60)
            moved = True
        elif key in (ord('['),):
            idx = max(0, idx - 60)
            moved = True
        elif key in (ord('>'), ord('.')):
            idx = min(total - 1, idx + 10)
            moved = True
        elif key in (ord('<'), ord(',')):
            idx = max(0, idx - 10)
            moved = True
        elif key == ord('h'):
            state.set(idx, "hit")
            dirty_since_save = True
        elif key == ord('v'):
            state.set(idx, "bounce")
            dirty_since_save = True
        elif key == ord('f'):
            state.set(idx, "off_frame")
            dirty_since_save = True
        elif key in (8, 127):  # Backspace
            state.remove(idx)
            dirty_since_save = True
        elif key == ord('c'):
            state.clear_xy(idx)
            dirty_since_save = True
        elif key == ord('s'):
            save_labels(out_path, state, video_path, fps, total)
            dirty_since_save = False
            print(f"[label] saved {len(state.by_frame)} events -> {out_path}")
        elif key in (ord('q'), 27):  # q or Esc
            break

        # Auto-advance after a labeling action (so the user can paddle quickly).
        # Skip auto-advance if the event opened a pending court click; let the
        # user click first.
        # (We don't auto-advance - the user controls navigation explicitly.
        # The above is a placeholder if we want to enable later.)
        _ = moved  # noqa: F841

    # Always save on exit.
    save_labels(out_path, state, video_path, fps, total)
    print(f"[label] saved {len(state.by_frame)} events -> {out_path}")
    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
