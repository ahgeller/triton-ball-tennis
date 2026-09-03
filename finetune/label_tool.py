"""Correct the detector's draft labels as fast as possible.

    python finetune/label_tool.py <clip>        # videos/<clip>.mp4 + labels/<clip>_ball.csv(.draft)

Every label frame (every 2nd frame of a 60 FPS clip, every frame at 30 FPS)
comes up with the detector's guess drawn and — on Windows — the mouse cursor
already sitting on it.  If it is right, press SPACE (or ENTER / c).  If it is
wrong, click where the ball is.  Both move to the next frame.  Zoom stays
where you set it and the view follows the ball between frames.

Controls:
  left click               ball is here -> save and go to next frame
  SPACE / ENTER / c        guess is good -> next frame
  v                        ball not visible on this frame -> next frame
  mouse wheel, + / -       smooth zoom in / out about the mouse / the ball (kept across frames)
  right click              re-centre the view here
  i / j / k / l            nudge the point one pixel up / left / down / right (stays on frame)
  n / d / right arrow      next frame        p / a / left arrow   previous frame
  f / b                    jump to next / previous unreviewed uncertain frame
  u                        undo last change on this frame
  t                        toggle trail       s   save        q / ESC   save and quit

Output: labels/<clip>_ball.csv (frame,ball_x,ball_y).  Review progress lives
in labels/<clip>_ball.review.json so you can quit and resume.  Everything is
autosaved every few seconds and on any exit; a frame you already dealt with
shows a REVIEWED (space) or EDITED (click / v / nudge) badge in the top-right
corner when you come back to it, and the top-left corner counts how many
frames are left to do.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

WORKSPACE = Path(__file__).resolve().parent
VIDEOS = WORKSPACE / "videos"
LABELS = WORKSPACE / "labels"
WINDOW = "label_tool"
HUD = 58
CONFIDENT_CONF = 0.70
VIEW_MAX_WIDTH = 1600
ZOOM_STEP = 1.25          # per wheel notch / keypress; multiplicative so every step feels the same size
ZOOM_MAX = 12.0
AUTOSAVE_SECONDS = 5.0    # write labels + review progress this often while there are unsaved changes

try:  # cursor placement is a Windows nicety; everything else works without it
    import ctypes
    _user32 = ctypes.windll.user32
except (AttributeError, ImportError, OSError):
    _user32 = None


class Row:
    __slots__ = ("frame", "x", "y", "visible", "source", "conf", "reviewed", "edited", "history")

    def __init__(self, frame: int, x: float, y: float, visible: bool, source: str, conf: float):
        self.frame, self.x, self.y, self.visible = frame, x, y, visible
        self.source, self.conf = source, conf
        self.reviewed = False
        self.edited = False
        self.history: List[tuple] = []

    @property
    def uncertain(self) -> bool:
        return not (self.visible and self.conf >= CONFIDENT_CONF)

    def snapshot(self) -> None:
        self.history.append((self.x, self.y, self.visible))

    def undo(self) -> None:
        if self.history:
            self.x, self.y, self.visible = self.history.pop()
            self.edited = bool(self.history)


def is_visible(x: float, y: float, width: int, height: int) -> bool:
    return not (x >= width * 0.95 and y <= height * 0.05)


def load_rows(csv_path: Path, width: int, height: int) -> List[Row]:
    rows = []
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        for record in csv.DictReader(handle):
            frame = int(re.fullmatch(r"frame_(\d+)", record["frame"].strip()).group(1))
            x, y = float(record["ball_x"]), float(record["ball_y"])
            visible = is_visible(x, y, width, height)
            conf = record.get("conf")
            rows.append(Row(frame, x if visible else width / 2.0, y if visible else height / 2.0, visible,
                            record.get("source") or ("det" if visible else "none"),
                            float(conf) if conf not in (None, "") else (1.0 if visible else 0.0)))
    rows.sort(key=lambda row: row.frame)
    return rows


def save_rows(csv_path: Path, rows: List[Row], width: int, height: int) -> None:
    temporary = csv_path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "ball_x", "ball_y"])
        for row in rows:
            if row.visible:
                writer.writerow([f"frame_{row.frame:03d}", f"{row.x:.2f}", f"{row.y:.2f}"])
            else:
                writer.writerow([f"frame_{row.frame:03d}", width - 1, 0])
    temporary.replace(csv_path)


def save_review(path: Path, rows: List[Row], cursor: int, zoom: float) -> None:
    path.write_text(json.dumps({
        "cursor": cursor,
        "zoom": zoom,
        "reviewed": [row.frame for row in rows if row.reviewed],
        "edited": [row.frame for row in rows if row.edited],
    }), encoding="utf-8")


def load_review(path: Path, rows: List[Row]):
    if not path.is_file():
        return None
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    reviewed, edited = set(state.get("reviewed", [])), set(state.get("edited", []))
    for row in rows:
        row.reviewed = row.frame in reviewed
        row.edited = row.frame in edited
    return state


class Tool:
    def __init__(self, clip: str, video: Path, rows: List[Row], width: int, height: int, out_csv: Path, review: Path):
        self.clip, self.rows, self.width, self.height = clip, rows, width, height
        self.out_csv, self.review = out_csv, review
        self.capture = cv2.VideoCapture(str(video))
        self.base_scale = min(1.0, VIEW_MAX_WIDTH / width)
        self.view_w = int(round(width * self.base_scale))
        self.view_h = int(round(height * self.base_scale))
        self.zoom = 1.0
        self.center = (width / 2.0, height / 2.0)
        self.origin = (0.0, 0.0)
        self.cursor = 0
        self.trail = True
        self.dirty = False
        self.frame_cache: Dict[int, np.ndarray] = {}
        self.pending_cursor_move = False
        self.last_save = time.monotonic()

    # ---------------------------------------------------------------- geometry
    @property
    def scale(self) -> float:
        return self.base_scale * self.zoom

    def clamp_view(self) -> None:
        span_w = self.view_w / self.scale
        span_h = self.view_h / self.scale
        cx = min(max(self.center[0], span_w / 2.0), max(self.width - span_w / 2.0, span_w / 2.0))
        cy = min(max(self.center[1], span_h / 2.0), max(self.height - span_h / 2.0, span_h / 2.0))
        self.center = (cx, cy)
        self.origin = (cx - span_w / 2.0, cy - span_h / 2.0)

    def to_view(self, x: float, y: float):
        return int(round((x - self.origin[0]) * self.scale)), int(round((y - self.origin[1]) * self.scale))

    def to_frame(self, vx: float, vy: float):
        return self.origin[0] + vx / self.scale, self.origin[1] + vy / self.scale

    def follow_ball(self) -> None:
        row = self.rows[self.cursor]
        if row.visible:
            self.center = (row.x, row.y)
        self.clamp_view()
        self.pending_cursor_move = True

    # ---------------------------------------------------------------- frames
    def frame(self, index: int) -> np.ndarray:
        cached = self.frame_cache.get(index)
        if cached is not None:
            return cached
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, image = self.capture.read()
        if not ok:
            image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        if len(self.frame_cache) > 48:
            self.frame_cache.pop(next(iter(self.frame_cache)))
        self.frame_cache[index] = image
        return image

    def render(self) -> np.ndarray:
        self.clamp_view()
        row = self.rows[self.cursor]
        image = self.frame(row.frame)
        x0, y0 = self.origin
        # Resize with exact sub-pixel origin handling via warpAffine.
        matrix = np.array([[self.scale, 0.0, -x0 * self.scale], [0.0, self.scale, -y0 * self.scale]], dtype=np.float64)
        display = cv2.warpAffine(image, matrix, (self.view_w, self.view_h),
                                 flags=cv2.INTER_LINEAR if self.zoom < 3 else cv2.INTER_NEAREST)
        if self.trail:
            for previous in self.rows[max(0, self.cursor - 12):self.cursor]:
                if previous.visible:
                    cv2.circle(display, self.to_view(previous.x, previous.y), 3, (200, 200, 60), 1, cv2.LINE_AA)
        if row.visible:
            colour = (0, 0, 255) if row.edited else (0, 220, 0) if row.reviewed else (0, 255, 255)
            cx, cy = self.to_view(row.x, row.y)
            radius = max(8, int(6 * self.zoom))
            cv2.circle(display, (cx, cy), radius, colour, 1, cv2.LINE_AA)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                cv2.line(display, (cx + dx * (radius - 3), cy + dy * (radius - 3)),
                         (cx + dx * (radius + 8), cy + dy * (radius + 8)), colour, 1)
        if row.reviewed or row.edited:
            badge = "EDITED" if row.edited else "REVIEWED"
            fill = (60, 140, 255) if row.edited else (70, 190, 70)
            (text_w, text_h), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            left = self.view_w - text_w - 26
            cv2.rectangle(display, (left, 10), (self.view_w - 10, 24 + text_h), fill, -1)
            cv2.putText(display, badge, (left + 8, 17 + text_h), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 0), 2, cv2.LINE_AA)
        reviewed = sum(r.reviewed for r in self.rows)
        remaining = len(self.rows) - reviewed
        counter = "ALL DONE" if remaining == 0 else f"{remaining} LEFT"
        fill, text_colour = ((70, 190, 70), (0, 0, 0)) if remaining == 0 else ((50, 50, 50), (255, 255, 255))
        (text_w, text_h), _ = cv2.getTextSize(counter, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(display, (10, 10), (26 + text_w, 24 + text_h), fill, -1)
        cv2.putText(display, counter, (18, 17 + text_h), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    text_colour, 2, cv2.LINE_AA)
        hud = np.zeros((HUD, self.view_w, 3), dtype=np.uint8)
        uncertain_left = sum(1 for r in self.rows if r.uncertain and not r.reviewed)
        state = "NOT VISIBLE" if not row.visible else f"({row.x:.1f}, {row.y:.1f})"
        tag = "EDITED" if row.edited else "ok" if row.reviewed else "unreviewed"
        line1 = (f"{self.clip}  frame {row.frame}  [{self.cursor + 1}/{len(self.rows)}]  {state}  "
                 f"guess {row.source} {row.conf:.2f}  {tag}  zoom x{self.zoom:.1f}{'  *unsaved*' if self.dirty else ''}")
        line2 = (f"reviewed {reviewed}/{len(self.rows)}  uncertain left {uncertain_left}   "
                 "click=fix+next  SPACE=good+next  v=invisible+next  f=next uncertain  wheel=zoom  q=save+quit")
        cv2.putText(hud, line1, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(hud, line2, (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1, cv2.LINE_AA)
        return np.vstack([display, hud])

    def place_cursor(self) -> None:
        """Put the OS mouse cursor on the current guess so a click is a one-pixel move."""
        self.pending_cursor_move = False
        row = self.rows[self.cursor]
        if _user32 is None or not row.visible:
            return
        try:
            wx, wy, _, _ = cv2.getWindowImageRect(WINDOW)
        except cv2.error:
            return
        if wx < 0 and wy < 0:
            return
        vx, vy = self.to_view(row.x, row.y)
        if 0 <= vx < self.view_w and 0 <= vy < self.view_h:
            _user32.SetCursorPos(int(wx + vx), int(wy + vy))

    # ---------------------------------------------------------------- edits
    def on_mouse(self, event, x, y, flags, param):
        if y >= self.view_h:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            fx, fy = self.to_frame(x, y)
            self.set_point(fx, fy)
            self.move(1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.center = self.to_frame(x, y)
            self.clamp_view()
        elif event == cv2.EVENT_MOUSEWHEEL:
            self.zoom_at(1 if flags > 0 else -1, float(x), float(y))

    def zoom_at(self, direction: int, vx: float, vy: float) -> None:
        """Zoom about the view point (vx, vy): whatever sits under the mouse stays under it."""
        fx, fy = self.to_frame(vx, vy)
        previous = self.zoom
        self.zoom = min(max(self.zoom * ZOOM_STEP ** direction, 1.0), ZOOM_MAX)
        if self.zoom == previous:
            return
        self.center = (fx + (self.view_w / 2.0 - vx) / self.scale,
                       fy + (self.view_h / 2.0 - vy) / self.scale)
        self.clamp_view()

    def zoom_key(self, direction: int) -> None:
        """Keyboard zoom pivots on the ball so it stays put on screen (view centre when invisible)."""
        row = self.rows[self.cursor]
        if row.visible:
            vx, vy = self.to_view(row.x, row.y)
            self.zoom_at(direction, float(vx), float(vy))
        else:
            self.zoom_at(direction, self.view_w / 2.0, self.view_h / 2.0)
        self.pending_cursor_move = True

    def set_point(self, x: float, y: float) -> None:
        row = self.rows[self.cursor]
        row.snapshot()
        row.x = float(min(max(x, 0.0), self.width - 1.0))
        row.y = float(min(max(y, 0.0), self.height - 1.0))
        row.visible = True
        row.reviewed = row.edited = True
        self.dirty = True

    def nudge(self, dx: float, dy: float) -> None:
        row = self.rows[self.cursor]
        if row.visible:
            self.set_point(row.x + dx, row.y + dy)
            self.pending_cursor_move = True

    def mark_invisible(self) -> None:
        row = self.rows[self.cursor]
        row.snapshot()
        row.visible = False
        row.reviewed = row.edited = True
        self.dirty = True

    def confirm(self) -> None:
        self.rows[self.cursor].reviewed = True
        self.dirty = True

    def move(self, delta: int) -> None:
        self.cursor = min(max(self.cursor + delta, 0), len(self.rows) - 1)
        self.follow_ball()

    def jump(self, direction: int) -> None:
        index = self.cursor + direction
        while 0 <= index < len(self.rows):
            row = self.rows[index]
            if row.uncertain and not row.reviewed:
                self.cursor = index
                self.follow_ball()
                return
            index += direction
        self.cursor = len(self.rows) - 1 if direction > 0 else 0
        self.follow_ball()

    def save(self, quiet: bool = False) -> None:
        save_rows(self.out_csv, self.rows, self.width, self.height)
        save_review(self.review, self.rows, self.cursor, self.zoom)
        self.dirty = False
        self.last_save = time.monotonic()
        if not quiet:
            print(f"[save] {self.out_csv} ({sum(r.visible for r in self.rows)} visible, "
                  f"{sum(r.reviewed for r in self.rows)}/{len(self.rows)} reviewed)")

    # ---------------------------------------------------------------- loop
    def run(self) -> None:
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        self.follow_ball()
        try:
            self.loop()
        finally:
            # Any way out - q, the window's X, Ctrl+C, a crash - keeps the work done so far.
            self.save()
            cv2.destroyAllWindows()

    def loop(self) -> None:
        while True:
            cv2.imshow(WINDOW, self.render())
            if self.pending_cursor_move:
                cv2.waitKey(1)
                self.place_cursor()
            key = cv2.waitKeyEx(30)
            if self.dirty and time.monotonic() - self.last_save > AUTOSAVE_SECONDS:
                self.save(quiet=True)
            if key == -1:
                try:
                    if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                        return
                except cv2.error:
                    return
                continue
            code = key & 0xFF if key < 0x10000 else key
            if code in (ord("q"), 27):
                break
            elif code in (ord(" "), 13, ord("c")):
                self.confirm()
                self.move(1)
            elif code == ord("v"):
                self.mark_invisible()
                self.move(1)
            elif code in (ord("n"), ord("d")) or key in (2555904, 83):
                self.move(1)
            elif code in (ord("p"), ord("a")) or key in (2424832, 81):
                self.move(-1)
            elif code == ord("f"):
                self.jump(1)
            elif code == ord("b"):
                self.jump(-1)
            elif code == ord("u"):
                self.rows[self.cursor].undo()
                self.dirty = True
                self.follow_ball()
            elif code in (ord("i"), ord("j"), ord("k"), ord("l")):
                dx, dy = {ord("i"): (0, -1), ord("j"): (-1, 0), ord("k"): (0, 1), ord("l"): (1, 0)}[code]
                self.nudge(dx, dy)
            elif code in (ord("+"), ord("=")):
                self.zoom_key(1)
            elif code in (ord("-"), ord("_")):
                self.zoom_key(-1)
            elif code == ord("t"):
                self.trail = not self.trail
            elif code == ord("s"):
                self.save()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clip", help="Clip stem, e.g. rally7 (videos/rally7.mp4)")
    parser.add_argument("--videos", type=Path, default=VIDEOS)
    parser.add_argument("--labels", type=Path, default=LABELS)
    parser.add_argument("--zoom", type=float, default=None, help="Starting zoom (default: last used, else 2)")
    args = parser.parse_args()

    videos = [p for p in args.videos.glob(f"{args.clip}.*") if p.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi", ".m4v"}]
    if not videos:
        print(f"No video named {args.clip}.* in {args.videos}")
        return 1
    video = videos[0]
    capture = cv2.VideoCapture(str(video))
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()

    final = args.labels / f"{args.clip}_ball.csv"
    draft = args.labels / f"{args.clip}_ball.csv.draft"
    source = final if final.is_file() else draft
    if not source.is_file():
        print(f"No labels for {args.clip}: run  python finetune/pretrack.py --clips {args.clip}  first")
        return 1
    rows = load_rows(source, width, height)
    review = args.labels / f"{args.clip}_ball.review.json"
    tool = Tool(args.clip, video, rows, width, height, final, review)
    state = load_review(review, rows)
    if state is not None:
        tool.cursor = min(int(state.get("cursor", 0)), len(rows) - 1)
        tool.zoom = float(state.get("zoom", 2.0))
    else:
        tool.zoom = 2.0
    if args.zoom:
        tool.zoom = args.zoom
    print(f"[open] {video.name}: {len(rows)} label frames, {sum(r.uncertain for r in rows)} uncertain (from {source.name})")
    tool.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
