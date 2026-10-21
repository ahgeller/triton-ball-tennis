"""TensorRT player detection with ByteTrack association.

Engine policy: see ``engine_session._open_engine_session``.  boxmot is optional;
without it the detector is disabled with one message, since the boxes are only
useful tracked.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .ball_detectors import _POSTPROCESSED_COLUMNS, nms_xyxy, xywh_to_xyxy
from .config import Config
from .engine_session import _TensorRTRuntimeSession, _open_engine_session


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
