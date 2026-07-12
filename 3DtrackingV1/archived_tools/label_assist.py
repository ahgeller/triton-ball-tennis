#!/usr/bin/env python3
"""Click-based hand labeler for ball annotation.

Reads a starter JSON produced by archived extract_label_frames.py, opens each
frame in a cv2 window with the pipeline's pre-filled (x, y) shown as a yellow
crosshair. You click on the ball to set a new position (red crosshair). When
done, the corrected positions are written to the output JSON in the canonical
annotation schema.

Controls:
  left click          Set ball center at click location (visible=True)
  i / j / k / l       Nudge label up / left / down / right by one pixel
  v                   Toggle visible flag (not-visible removes x/y on save)
  c                   Confirm pre-fill as-is (no change to x/y)
  u                   Undo last action on this frame
  n / right arrow / SPACE   Next frame
  p / left arrow      Previous frame
  s                   Save now (without quitting)
  q / ESC             Save and quit

Run:
  python 3DtrackingV1/archived_tools/label_assist.py `
    --starter validation/labels/pomona_baseline/pomona_baseline_starter.json `
    --frames-dir validation/labels/pomona_baseline `
    --out validation/annotations/pomona_baseline.json
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

WINDOW = "label_assist"
HUD_HEIGHT = 60


def _load_starter(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _strip_review_fields(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        clean = {k: row[k] for k in ("frame", "x", "y", "visible") if k in row}
        if not bool(clean.get("visible", True)):
            clean.pop("x", None)
            clean.pop("y", None)
        out.append(clean)
    return out


def _save(out_path: Path, doc: Dict[str, Any], rows: List[Dict[str, Any]]) -> None:
    final = {
        "video": doc.get("video", ""),
        "ball": _strip_review_fields(rows),
        "events": doc.get("events", []),
        "ignore_ranges": doc.get("ignore_ranges", []),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)


def _draw_hud(img, idx: int, total: int, row: Dict[str, Any], dirty: bool):
    import cv2
    h, w = img.shape[:2]
    bar = img.copy()
    cv2.rectangle(bar, (0, 0), (w, HUD_HEIGHT), (0, 0, 0), -1)
    cv2.addWeighted(bar, 0.6, img, 0.4, 0, img)

    visible = bool(row.get("visible", True))
    src = str(row.get("_pipeline_source", ""))
    conf = row.get("_pipeline_conf", "")
    confirmed = bool(row.get("_confirmed", False))
    x = row.get("x"); y = row.get("y")
    coord = f"({x:.1f},{y:.1f})" if (x is not None and y is not None) else "----"
    status = "VIS" if visible else "NOT-VIS"
    confirm_str = " CONFIRMED" if confirmed else ""
    dirty_str = " *unsaved*" if dirty else ""
    text = f"[{idx + 1}/{total}] f={row.get('frame')} src={src} conf={conf} pred={coord} {status}{confirm_str}{dirty_str}"
    cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    help_text = "CLICK=set   IJKL=nudge   v=visible   c=confirm   u=undo   n/SPACE=next   p=prev   autosaves"
    cv2.putText(img, help_text, (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, help_text, (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)


def _draw_open_marker(img, x: int, y: int, color, radius: int, tick_gap: int, tick_len: int):
    """Draw a hollow ring + 4 outward ticks. The pixel under (x, y) is untouched."""
    import cv2
    cv2.circle(img, (x, y), radius, color, 1, cv2.LINE_AA)
    # ticks point outward, leaving a clear gap of `tick_gap` around the center.
    cv2.line(img, (x, y - tick_gap), (x, y - tick_gap - tick_len), color, 1, cv2.LINE_AA)
    cv2.line(img, (x, y + tick_gap), (x, y + tick_gap + tick_len), color, 1, cv2.LINE_AA)
    cv2.line(img, (x - tick_gap, y), (x - tick_gap - tick_len, y), color, 1, cv2.LINE_AA)
    cv2.line(img, (x + tick_gap, y), (x + tick_gap + tick_len, y), color, 1, cv2.LINE_AA)


def _draw_markers(img, prefill: Tuple[Optional[float], Optional[float]],
                  current: Tuple[Optional[float], Optional[float]],
                  visible: bool):
    # Yellow = pipeline pre-fill (always shown for context). Larger, thinner.
    if prefill[0] is not None and prefill[1] is not None:
        _draw_open_marker(img, int(prefill[0]), int(prefill[1]),
                          (0, 220, 220), radius=18, tick_gap=22, tick_len=8)
    # Red = current labeled position (only if visible). Smaller, tighter.
    if visible and current[0] is not None and current[1] is not None:
        _draw_open_marker(img, int(current[0]), int(current[1]),
                          (0, 0, 255), radius=12, tick_gap=16, tick_len=6)


def main() -> int:
    parser = argparse.ArgumentParser(description="Click-based ball labeler.")
    parser.add_argument("--starter", required=True,
                        help="Starter JSON from archived extract_label_frames.py")
    parser.add_argument("--frames-dir", required=True,
                        help="Directory holding frame_NNNNNN.png files")
    parser.add_argument("--out", required=True,
                        help="Output annotation JSON path")
    parser.add_argument("--window-width", type=int, default=1600,
                        help="Fixed display width (default 1600)")
    args = parser.parse_args()

    import cv2

    starter_path = Path(args.starter)
    frames_dir = Path(args.frames_dir)
    out_path = Path(args.out)
    doc = _load_starter(starter_path)
    rows: List[Dict[str, Any]] = deepcopy(doc.get("ball", []))
    rows.sort(key=lambda r: int(r.get("frame", 0)))
    if not rows:
        print("[label_assist] starter has no rows", file=sys.stderr)
        return 2

    for row in rows:
        # Snapshot pre-fill for context display.
        row["_prefill_x"] = row.get("x")
        row["_prefill_y"] = row.get("y")
        row.setdefault("_confirmed", False)

    state = {
        "idx": 0,
        "history": {},  # frame_idx -> list of snapshots for undo
        "dirty": False,
    }

    def _autosave():
        _save(out_path, doc, rows)
        state["dirty"] = False

    def _record_undo(i: int):
        snap = deepcopy(rows[i])
        state["history"].setdefault(i, []).append(snap)

    def _set_position(i: int, x: float, y: float):
        _record_undo(i)
        rows[i]["x"] = float(x)
        rows[i]["y"] = float(y)
        rows[i]["visible"] = True
        rows[i]["_confirmed"] = True
        state["dirty"] = True
        _autosave()

    def _toggle_visible(i: int):
        _record_undo(i)
        rows[i]["visible"] = not bool(rows[i].get("visible", True))
        rows[i]["_confirmed"] = True
        state["dirty"] = True
        _autosave()

    def _confirm(i: int):
        _record_undo(i)
        rows[i]["_confirmed"] = True
        state["dirty"] = True
        _autosave()

    def _undo(i: int):
        stack = state["history"].get(i, [])
        if stack:
            rows[i] = stack.pop()
            state["dirty"] = True
            _autosave()

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)

    current_img_cache: Dict[int, Any] = {}
    current_scale: Dict[str, float] = {"sx": 1.0, "sy": 1.0}

    def _load_frame_image(i: int):
        f = int(rows[i].get("frame", -1))
        if i in current_img_cache:
            return current_img_cache[i]
        fp = frames_dir / f"frame_{f:06d}.png"
        if not fp.exists():
            print(f"[label_assist] missing PNG: {fp}", file=sys.stderr)
            return None
        img = cv2.imread(str(fp))
        if img is None:
            print(f"[label_assist] cv2 failed to read: {fp}", file=sys.stderr)
            return None
        current_img_cache[i] = img
        if len(current_img_cache) > 8:
            for k in list(current_img_cache.keys()):
                if k != i:
                    del current_img_cache[k]
                    break
        return img

    def _nudge(i: int, dx: float, dy: float):
        row = rows[i]
        if row.get("x") is None or row.get("y") is None:
            return
        img = _load_frame_image(i)
        if img is None:
            return
        h, w = img.shape[:2]
        _set_position(
            i,
            min(max(float(row["x"]) + dx, 0.0), float(w - 1)),
            min(max(float(row["y"]) + dy, 0.0), float(h - 1)),
        )

    def _redraw():
        i = state["idx"]
        img = _load_frame_image(i)
        if img is None:
            return
        h, w = img.shape[:2]
        view = img.copy()
        row = rows[i]
        prefill = (row.get("_prefill_x"), row.get("_prefill_y"))
        current = (row.get("x"), row.get("y"))
        visible = bool(row.get("visible", True))
        _draw_markers(view, prefill, current, visible)
        _draw_hud(view, i, len(rows), row, state["dirty"])

        target_w = int(args.window_width)
        scale = target_w / float(w)
        new_h = int(h * scale)
        view_scaled = cv2.resize(view, (target_w, new_h), interpolation=cv2.INTER_AREA)
        current_scale["sx"] = w / float(target_w)
        current_scale["sy"] = h / float(new_h)
        cv2.imshow(WINDOW, view_scaled)

    def _on_mouse(event, x, y, flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        i = state["idx"]
        full_x = float(x) * current_scale["sx"]
        full_y = float(y) * current_scale["sy"]
        _set_position(i, full_x, full_y)
        _redraw()

    cv2.setMouseCallback(WINDOW, _on_mouse)
    _redraw()

    print("[label_assist] keys: LMB=set, I/J/K/L=nudge, v=toggle visible, c=confirm, u=undo,")
    print("                 n/SPACE=next, p=prev, q/ESC=save+quit (every edit autosaves)")

    while True:
        key = cv2.waitKey(50)
        if key == -1:
            continue
        key &= 0xFF
        i = state["idx"]
        if key in (ord("q"), 27):  # q or ESC
            break
        elif key in (ord("n"), ord(" "), 83, 100):  # n, space, right arrow, 'd'
            state["idx"] = min(len(rows) - 1, i + 1)
            _redraw()
        elif key in (ord("p"), 81, ord("a")):  # p, left arrow, 'a'
            state["idx"] = max(0, i - 1)
            _redraw()
        elif key == ord("v"):
            _toggle_visible(i)
            _redraw()
        elif key == ord("c"):
            _confirm(i)
            _redraw()
        elif key == ord("u"):
            _undo(i)
            _redraw()
        elif key in (ord("i"), ord("j"), ord("k"), ord("l")):
            dx, dy = {
                ord("i"): (0.0, -1.0),
                ord("j"): (-1.0, 0.0),
                ord("k"): (0.0, 1.0),
                ord("l"): (1.0, 0.0),
            }[key]
            _nudge(i, dx, dy)
            _redraw()
        elif key == ord("s"):
            _save(out_path, doc, rows)
            state["dirty"] = False
            print(f"[label_assist] saved to {out_path}")
            _redraw()

    _save(out_path, doc, rows)
    print(f"[label_assist] saved to {out_path}")
    confirmed = sum(1 for r in rows if r.get("_confirmed"))
    not_vis = sum(1 for r in rows if not r.get("visible"))
    print(f"[label_assist] {confirmed}/{len(rows)} frames touched, "
          f"{not_vis} marked not-visible")
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
