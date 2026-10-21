"""TensorRT player detection with ByteTrack association.

Engine policy: see ``engine_session.open_engine_session``; boxmot follows the
same rule, since the boxes are only useful tracked: with a player engine
configured a missing boxmot raises, and only an unset engine path disables
player tracking.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ..config import Config
from .engine_boxes import POSTPROCESSED_COLUMNS, decode_engine_boxes, engine_output_mode
from .engine_session import TensorRTRuntimeSession, open_engine_session

_PLAYER_CLASS = 0  # the player ("person") class of the player engine


def _load_bytetrack() -> Tuple[Optional[type], Optional[str]]:
    """Returns ``(ByteTrack, None)``, or ``(None, reason)`` when boxmot is not importable."""
    try:
        from boxmot import ByteTrack
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
    frame_index: int
    width: int
    height: int
    court_kps: Optional[List[float]]
    outputs: Optional[List[torch.Tensor]] = None
    input_tensor: Optional[torch.Tensor] = None  # keeps the engine input alive until wait()


class PlayerDetector:
    """TensorRT player detection with ByteTrack association (no ReID).

    Engine policy: see ``open_engine_session``; a configured engine without
    boxmot raises too.  Without an engine the detector is disabled with one
    message: ``detect_async_start`` returns None and ``detect_async_finish``
    then returns no boxes.
    """

    def __init__(self, cfg: Config, court_keypoints: Optional[List[float]] = None):
        self.cfg = cfg
        self.court_kps = court_keypoints
        self.cached_boxes: List[List[float]] = []
        self._frame_diag = 1.0
        self.session: Optional[TensorRTRuntimeSession] = None
        self.tracker: Optional[Any] = None
        # Detection cadence is controlled by the CLI/config, while ByteTrack
        # still receives an update every frame and coasts on empty detections.
        self._current_interval = max(1, cfg.player_detect_interval)
        # ByteTrack's live boxes keyed by track id.
        self.slots: Dict[int, List[float]] = {}

        if not cfg.player_model_path:
            print("[player] No engine configured, player tracking disabled")
            return
        bytetrack, reason = _load_bytetrack()
        if bytetrack is None:
            raise RuntimeError(
                f"[player] player tracking requires boxmot ({reason}): pip install boxmot, "
                "or unset player_model_path to run without players"
            )
        self.session = open_engine_session("player", cfg.player_model_path, cfg)
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
        """Engine output -> ``[x1, y1, x2, y2, conf, cls]`` rows in frame pixels,
        clipped to the frame, for boxmot.  Class ``_PLAYER_CLASS`` only."""
        scale = (frame_w / float(max(self.session.input_w, 1)), frame_h / float(max(self.session.input_h, 1)))
        return decode_engine_boxes(pred_t, scale, self.cfg.player_conf, self.cfg.player_iou,
                                   engine_output_mode(pred_t), cls_filter=[_PLAYER_CLASS],
                                   frame_size=(frame_w, frame_h))

    def detect_async_start(self, frame: np.ndarray, frame_index: int,
                           frame_gpu_chw: Optional[torch.Tensor] = None) -> Optional[PlayerPending]:
        """Launch the engine on source frame ``frame_index`` (a host BGR frame,
        which ByteTrack also needs).  ``frame_gpu_chw`` is the same frame
        already on the GPU as CHW float in [0, 1], used instead when given."""
        if not self.enabled:
            return None

        h, w = frame.shape[:2]
        self._frame_diag = (w ** 2 + h ** 2) ** 0.5
        pending = PlayerPending(frame, int(frame_index), int(w), int(h), self.court_kps)

        # The engine runs every N frames; the tracker is still finished every
        # frame, with empty detections in between.
        if frame_index % self._current_interval != 0:
            return pending

        t = None
        if frame_gpu_chw is not None:
            t, _ = self.session.preprocess_cuda_chw_frame(frame_gpu_chw)
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
            dets = np.empty((0, POSTPROCESSED_COLUMNS))
        else:
            self.session.wait()
            dets = self._decode_player_boxes(pending.outputs[0], pending.width, pending.height)
            # Drop spectators before the tracker learns them.
            if pending.court_kps and len(dets) > self.cfg.num_players:
                dets = self._keep_nearest_to_court(dets, pending.court_kps)
            if len(dets) == 0:
                dets = np.empty((0, POSTPROCESSED_COLUMNS))

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

    def detect(self, frame: np.ndarray, frame_index: int) -> List[List[float]]:
        return self.detect_async_finish(self.detect_async_start(frame, frame_index))

    def player_boxes_by_slot(self) -> Dict[int, List[float]]:
        """A copy of ByteTrack's live boxes, keyed by track id."""
        return dict(self.slots)
