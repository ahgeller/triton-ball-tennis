"""Ball, player and court detector backends.

Ball detection lives in ``ball_detectors`` (GridTrackNet and the single-frame
Ultralytics TensorRT engine behind ``BallDetectorBackend``) and is re-exported
here.  Player and court detection are TensorRT engines driven through
``_TensorRTRuntimeSession`` from ``engine_session``, which the ball engine
backend shares.

Engine policy (player and court): an unset model path skips that detector
with one message; a configured path whose file is missing or whose runtime
fails to initialise raises, so a broken install does not silently produce a
run without players or court.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from .ball_detectors import (  # noqa: F401  (re-exported for pipeline, run_context, pass1 and finetune)
    BallDetectorBackend,
    CachedDetections,
    Detection,
    FrameInput,
    GridTrackNetBallDetector,
    TensorRTBallPending,
    TensorRTRuntimeBallDetector,
    ball_engine_output_mode,
    decode_ball_predictions,
)
from .ball_detectors import _POSTPROCESSED_COLUMNS, nms_xyxy, xywh_to_xyxy
from .config import Config
from .court_geometry import LINE_PAIRS_14
from .engine_session import _TensorRTRuntimeSession, _open_engine_session

_NEVER = -10**9


def _load_bytetrack() -> Tuple[Optional[type], Optional[str]]:
    """boxmot is optional; player tracking is disabled without it.  Returns
    ``(ByteTrack, None)`` or ``(None, reason)``."""
    try:
        from boxmot import ByteTrack
    except ImportError:
        try:
            from boxmot.trackers.bbox.bytetrack import ByteTrack
        except ImportError as error:
            return None, str(error)
    return ByteTrack, None


@dataclass
class PlayerPending:
    """A player detection launched for one frame.

    ``outputs`` is None on frames between detector runs (``cfg.player_detect_interval``):
    the tracker is then updated with no detections and coasts its boxes.
    """

    frame: np.ndarray
    frame_idx: int
    width: int
    height: int
    court_kps: Optional[List[float]]
    outputs: Optional[List[torch.Tensor]] = None
    input_tensor: Optional[torch.Tensor] = None  # keeps the engine input alive until wait()


class PlayerDetector:
    """TensorRT player detection with ByteTrack association (no ReID).

    Engine policy: see ``_open_engine_session``.  Without boxmot the detector
    is disabled with one message, since the boxes are only useful tracked.
    ``detect_async_start`` returns None while disabled and
    ``detect_async_finish`` then returns no boxes.
    """

    def __init__(self, cfg: Config, court_keypoints: Optional[List[float]] = None):
        self.cfg = cfg
        self.court_kps = court_keypoints
        self.cached_boxes: List[List[float]] = []
        self._frame_diag = 1.0
        self.session: Optional[_TensorRTRuntimeSession] = None
        self.tracker: Optional[Any] = None
        # Detection cadence is controlled by the CLI/config, while ByteTrack
        # still receives an update every frame and coasts on empty detections.
        self._current_interval = max(1, int(cfg.player_detect_interval))
        # ByteTrack's live boxes keyed by track id.
        self.slots: Dict[int, List[float]] = {}

        if not cfg.player_model_path:
            print("[player] No engine configured, player tracking disabled")
            return
        bytetrack, reason = _load_bytetrack()
        if bytetrack is None:
            print(f"[player] boxmot not found ({reason}); player tracking disabled. Run 'pip install boxmot'")
            return
        self.session = _open_engine_session("player", cfg.player_model_path, cfg)
        self.tracker = bytetrack(
            min_conf=0.1,
            track_thresh=0.3,
            track_buffer=60,
            match_thresh=0.8,
        )
        print("[player] ByteTrack tracking initialized")

    @property
    def enabled(self) -> bool:
        return self.session is not None and self.tracker is not None

    def set_court_keypoints(self, court_keypoints: Optional[List[float]]) -> None:
        self.court_kps = court_keypoints

    def _decode_player_boxes(self, pred_t: torch.Tensor, frame_w: int, frame_h: int) -> np.ndarray:
        """Engine output -> ``[x1, y1, x2, y2, conf, cls]`` rows in frame pixels."""
        pred = pred_t.detach().float()
        if pred.dim() == 3 and pred.shape[-1] == _POSTPROCESSED_COLUMNS:
            rows = pred[0].cpu().numpy()
            if rows.size == 0:
                return np.empty((0, 6))
            conf = rows[:, 4]
            keep = conf >= float(self.cfg.player_conf)
            rows = rows[keep]
            return rows

        if pred.dim() == 3:
            p = pred[0]
            if p.shape[0] <= p.shape[1]:
                p = p.transpose(0, 1)
            p = p.cpu().numpy()
        else:
            return np.empty((0, 6))

        if p.shape[1] < 5:
            return np.empty((0, 6))

        boxes_xyxy = xywh_to_xyxy(p[:, :4].astype(np.float32))
        scores = p[:, 4].astype(np.float32)
        keep = scores >= float(self.cfg.player_conf)
        boxes_xyxy = boxes_xyxy[keep]
        scores = scores[keep]

        if boxes_xyxy.shape[0] == 0:
            return np.empty((0, 6))

        keep_idx = nms_xyxy(boxes_xyxy, scores, float(self.cfg.player_iou))
        boxes_xyxy = boxes_xyxy[keep_idx]
        scores = scores[keep_idx]

        sx = frame_w / float(max(self.session.input_w, 1))
        sy = frame_h / float(max(self.session.input_h, 1))
        boxes_xyxy[:, [0, 2]] *= sx
        boxes_xyxy[:, [1, 3]] *= sy
        boxes_xyxy[:, 0] = np.clip(boxes_xyxy[:, 0], 0, frame_w - 1)
        boxes_xyxy[:, 2] = np.clip(boxes_xyxy[:, 2], 0, frame_w - 1)
        boxes_xyxy[:, 1] = np.clip(boxes_xyxy[:, 1], 0, frame_h - 1)
        boxes_xyxy[:, 3] = np.clip(boxes_xyxy[:, 3], 0, frame_h - 1)

        # Format for boxmot: [x1, y1, x2, y2, conf, class_id]
        class_ids = np.zeros_like(scores)  # Usually 0 for 'person'
        return np.column_stack((boxes_xyxy, scores, class_ids))

    def detect_async_start(self, frame: np.ndarray, frame_idx: int,
                           frame_gpu_t: Optional[torch.Tensor] = None) -> Optional[PlayerPending]:
        if not self.enabled:
            return None

        h, w = frame.shape[:2]
        self._frame_diag = (w ** 2 + h ** 2) ** 0.5
        pending = PlayerPending(frame, int(frame_idx), int(w), int(h), self.court_kps)

        # The engine runs every N frames; the tracker is still finished every
        # frame, with empty detections in between.
        if frame_idx % self._current_interval != 0:
            return pending

        t = None
        if frame_gpu_t is not None:
            t, _ = self.session.preprocess_cuda_chw_frame(frame_gpu_t)
        if t is None:
            t, _ = self.session.preprocess_frame(frame)
        pending.outputs = self.session.forward_tensor(t)
        pending.input_tensor = t
        return pending

    def _keep_nearest_to_court(self, dets: np.ndarray, court_kps: List[float]) -> np.ndarray:
        """When the engine finds more people than players, keep the ones closest to a court keypoint."""
        scored = []
        for row in dets:
            bx1, by1, bx2, by2 = row[:4]
            bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
            dist = min(((bcx - court_kps[ki]) ** 2 + (bcy - court_kps[ki + 1]) ** 2) ** 0.5
                       for ki in range(0, len(court_kps), 2))
            scored.append((dist, row))
        scored.sort(key=lambda x: x[0])
        return np.array([r for _, r in scored[:self.cfg.num_players]])

    def detect_async_finish(self, pending: Optional[PlayerPending]) -> List[List[float]]:
        if pending is None:
            return []

        if pending.outputs is None:
            # Pass empty detections to tell the tracker to coast the boxes forward.
            dets = np.empty((0, 6))
        else:
            self.session.wait()
            dets = self._decode_player_boxes(pending.outputs[0], pending.width, pending.height)
            # Drop spectators before the tracker learns them.
            if pending.court_kps and len(dets) > self.cfg.num_players:
                dets = self._keep_nearest_to_court(dets, pending.court_kps)
            if len(dets) == 0:
                dets = np.empty((0, 6))

        # tracked_objects rows: [x1, y1, x2, y2, track_id, conf, cls, ind]
        tracked_objects = self.tracker.update(dets, pending.frame)

        boxes: List[List[float]] = []
        self.slots.clear()
        for track in tracked_objects:
            x1, y1, x2, y2, track_id, _conf, _cls, _ = track
            box = [float(x1), float(y1), float(x2), float(y2)]
            boxes.append(box)
            self.slots[int(track_id)] = box

        self.cached_boxes = boxes
        return boxes

    def detect(self, frame: np.ndarray, frame_idx: int) -> List[List[float]]:
        return self.detect_async_finish(self.detect_async_start(frame, frame_idx))

    def get_player_dict(self) -> Dict[int, List[float]]:
        return dict(self.slots)


class CourtDetector:
    """Runs the court keypoint TensorRT engine every N frames and caches the last good result.

    Engine policy: see ``_open_engine_session``; while disabled ``detect``
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
        self.session = _open_engine_session("court", cfg.court_model_path, cfg)
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
