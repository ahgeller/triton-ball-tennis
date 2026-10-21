"""The court keypoint TensorRT engine, run every N frames with the last good result cached.

Engine policy: see ``engine_session.open_engine_session``.
"""
from __future__ import annotations

import json
from typing import List, Optional, Tuple

import numpy as np

from ..config import Config
from .engine_session import open_engine_session
from ..court.court_drawing import draw_court

_NEVER = -10**9


class CourtDetector:
    """Runs the court keypoint TensorRT engine every N frames and caches the last good result.

    Engine policy: see ``open_engine_session``; while disabled ``detect``
    returns None and ``draw`` leaves the frame alone.
    """
    _MODEL_TO_SEMANTIC_14 = (0, 4, 6, 1, 2, 5, 7, 3, 8, 12, 9, 11, 13, 10)
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

    def _print_raw_result(self, rows: np.ndarray, frame_index: int) -> None:
        best_idx = int(np.argmax(rows[:, 4])) if rows.size else -1
        print(f"[court][raw] frame={frame_index} rows={rows.shape[0]} best={best_idx}")
        if best_idx >= 0:
            r = rows[best_idx]
            print(json.dumps({
                "x1": float(r[0]), "y1": float(r[1]), "x2": float(r[2]), "y2": float(r[3]),
                "conf": float(r[4]), "cls": float(r[5]),
            }))

    def _report_miss(self, frame_index: int) -> Optional[List[float]]:
        """The cached keypoints after a frame without a usable court, reported
        when raw results are being printed.  Every miss path returns through here."""
        if self.cfg.print_court_raw:
            msg = "miss, using cached court" if self.keypoints else "no court detected"
            print(f"[court] frame={frame_index}: {msg}")
        return self.keypoints

    def detect(self, frame: np.ndarray, frame_index: int) -> Optional[List[float]]:
        """The court keypoints as of source frame ``frame_index``: the engine
        runs when ``cfg.court_detect_interval`` frames have passed since it
        last did, otherwise the cached result is returned."""
        if self.session is None:
            return self.keypoints
        interval = max(1, self.cfg.court_detect_interval)
        if self.keypoints is not None and frame_index != 0 and \
                (frame_index - self.last_run_frame) < interval:
            return self.keypoints

        self.last_run_frame = frame_index
        t, scale = self.session.preprocess_frame(frame)
        outputs = self.session.forward_tensor(t)
        self.session.wait()
        out = outputs[0].detach().float()
        rows = out[0].cpu().numpy() if out.dim() == 3 else np.empty((0, 0), dtype=np.float32)
        if rows.size == 0:
            return self._report_miss(frame_index)
        if self.cfg.print_court_raw:
            self._print_raw_result(rows, frame_index)

        conf = rows[:, 4] if rows.shape[1] > 4 else np.zeros((rows.shape[0],), dtype=np.float32)
        best_i = int(np.argmax(conf))
        if float(conf[best_i]) < self.cfg.court_conf:
            return self._report_miss(frame_index)

        flat = self._keypoints_of(rows[best_i], scale)
        if flat is None or len(flat) < 2 * self._MIN_KEYPOINTS:
            return self._report_miss(frame_index)
        self.keypoints = flat
        return self.keypoints

    def _keypoints_of(self, row: np.ndarray, scale: Tuple[float, float]) -> Optional[List[float]]:
        """The row's keypoints as a flat [x0, y0, x1, y1, ...] in frame pixels, remapped to the
        semantic order when configured; None when the row carries no keypoints."""
        kpt_raw = row[self._KEYPOINT_COLUMN:]
        nk = int(kpt_raw.shape[0] // 3)
        if nk <= 0:
            return None
        k = kpt_raw[: nk * 3].reshape(nk, 3)
        xy = k[:, :2].astype(np.float32)
        sx, sy = float(scale[0]), float(scale[1])
        xy[:, 0] *= sx
        xy[:, 1] *= sy
        if xy.shape[0] == 14 and self.cfg.court_remap_semantic_14:
            xy = self._remap_kpt_indices(xy)
        return np.asarray(xy, dtype=np.float32).reshape(-1).astype(float).tolist()

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Draw the cached keypoints on ``frame`` (``court_drawing.draw_court``); untouched while there are none."""
        return draw_court(frame, self.keypoints, self.cfg)
