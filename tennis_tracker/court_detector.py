"""The court keypoint TensorRT engine, run every N frames with the last good result cached.

Engine policy: see ``engine_session.open_engine_session``.
"""
from __future__ import annotations

import json
from typing import Any, List, Optional

import cv2
import numpy as np

from .config import Config
from .court_geometry import LINE_PAIRS_14
from .engine_session import open_engine_session

_NEVER = -10**9


class CourtDetector:
    """Runs the court keypoint TensorRT engine every N frames and caches the last good result.

    Engine policy: see ``open_engine_session``; while disabled ``detect``
    returns None and ``draw`` leaves the frame alone.
    """
    _MODEL_TO_SEMANTIC_14 = (0, 4, 6, 1, 2, 5, 7, 3, 8, 12, 9, 11, 13, 10)
    _LINE_PAIRS_14 = LINE_PAIRS_14
    # Rows are [x1, y1, x2, y2, conf, cls, kx, ky, kconf, ...].
    _KEYPOINT_COLUMN = 6
    _MIN_KEYPOINTS = 4

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.keypoints: Optional[List[float]] = None
        self.last_run_frame = _NEVER
        self.session = open_engine_session("court", cfg.court_model_path, cfg)
        if self.session is not None:
            remap = "enabled" if cfg.court_remap_semantic_14 else "disabled (raw model order)"
            print(f"[court] 14-point semantic remap: {remap}")

    @classmethod
    def _remap_kpt_indices(cls, xy: np.ndarray) -> np.ndarray:
        pts = np.asarray(xy, dtype=np.float32)
        if pts.shape[0] != len(cls._MODEL_TO_SEMANTIC_14):
            return pts
        out = np.zeros_like(pts)
        for model_i, semantic_i in enumerate(cls._MODEL_TO_SEMANTIC_14):
            out[semantic_i] = pts[model_i]
        return out

    def _print_raw_result(self, rows: np.ndarray, frame_idx: int) -> None:
        best_idx = int(np.argmax(rows[:, 4])) if rows.size else -1
        print(f"[court][raw] frame={frame_idx} rows={rows.shape[0]} best={best_idx}")
        if best_idx >= 0:
            r = rows[best_idx]
            print(json.dumps({
                "x1": float(r[0]), "y1": float(r[1]), "x2": float(r[2]), "y2": float(r[3]),
                "conf": float(r[4]), "cls": float(r[5]),
            }))

    def _report_miss(self, frame_idx: int) -> Optional[List[float]]:
        if self.cfg.print_court_raw:
            msg = "miss, using cached court" if self.keypoints else "no court detected"
            print(f"[court] frame={frame_idx}: {msg}")
        return self.keypoints

    def detect(self, frame: np.ndarray, frame_idx: Optional[int] = None) -> Optional[List[float]]:
        if self.session is None:
            return self.keypoints
        interval = max(1, int(self.cfg.court_detect_interval))
        if frame_idx is None:
            frame_idx = self.last_run_frame + interval
        if self.keypoints is not None and frame_idx != 0 and \
                (frame_idx - self.last_run_frame) < interval:
            return self.keypoints

        self.last_run_frame = frame_idx
        t, scale = self.session.preprocess_frame(frame)
        outputs = self.session.forward_tensor(t)
        self.session.wait()
        out = outputs[0].detach().float()
        rows = out[0].cpu().numpy() if out.dim() == 3 else np.empty((0, 0), dtype=np.float32)
        if rows.size == 0:
            return self.keypoints
        if self.cfg.print_court_raw:
            self._print_raw_result(rows, frame_idx)

        conf = rows[:, 4] if rows.shape[1] > 4 else np.zeros((rows.shape[0],), dtype=np.float32)
        best_i = int(np.argmax(conf))
        if float(conf[best_i]) < float(self.cfg.court_conf):
            return self._report_miss(frame_idx)

        row = rows[best_i]
        kpt_raw = row[self._KEYPOINT_COLUMN:]
        nk = int(kpt_raw.shape[0] // 3)
        if nk <= 0:
            return self.keypoints
        k = kpt_raw[: nk * 3].reshape(nk, 3)
        xy = k[:, :2].astype(np.float32)
        sx, sy = float(scale[0]), float(scale[1])
        xy[:, 0] *= sx
        xy[:, 1] *= sy
        if xy.shape[0] == 14 and self.cfg.court_remap_semantic_14:
            xy = self._remap_kpt_indices(xy)

        flat = np.asarray(xy, dtype=np.float32).reshape(-1).astype(float).tolist()
        if len(flat) >= 2 * self._MIN_KEYPOINTS:
            self.keypoints = flat
            return self.keypoints
        return self._report_miss(frame_idx)

    def draw(self, frame: np.ndarray) -> np.ndarray:
        if self.keypoints is None:
            return frame
        kps = self.keypoints
        n = len(kps) // 2

        for ki in range(0, len(kps) - 1, 2):
            x, y = int(kps[ki]), int(kps[ki + 1])
            if x > 0 or y > 0:
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                if self.cfg.court_draw_indices:
                    cv2.putText(frame, str(ki // 2), (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        if self.cfg.court_points_only:
            return frame

        line_pairs: Any = []
        if n == 14:
            line_pairs = self._LINE_PAIRS_14
        elif n >= self._MIN_KEYPOINTS:
            line_pairs = [(j, j + 1) for j in range(n - 1)]

        for a, b in line_pairs:
            if a < n and b < n:
                ax, ay = int(kps[a * 2]), int(kps[a * 2 + 1])
                bx, by = int(kps[b * 2]), int(kps[b * 2 + 1])
                if (ax > 0 or ay > 0) and (bx > 0 or by > 0):
                    cv2.line(frame, (ax, ay), (bx, by), (0, 255, 0), 2)
        return frame
