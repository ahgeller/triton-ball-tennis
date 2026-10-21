"""ROI tracking: where to look for motion, from where the detector last saw a ball.

One ``ROITrack`` per detected ball coasts on the selector's projectile model
between detections; ``ROIMotionTracker`` turns the live tracks into the
regions pass 1 computes motion in, probing the full frame periodically while
the ball is lost.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ball_in_play_selector import SelectorConfig, predict_projectile

from .config import Config

Point = Tuple[float, float]
Box = Tuple[int, int, int, int]
Detection = Tuple[Sequence[float], float]        # (xyxy box, confidence)
CoastingRoi = Tuple[int, int, int, int, int]     # (frame_idx, x1, y1, x2, y2)

# A jump faster than this (px/frame) is a new ball, not the tracked one moving.
_VELOCITY_RESET_SPEED_PX = 150.0
# Velocity blend between the latest step and the recent history.
_VELOCITY_STEP_WEIGHT, _VELOCITY_HISTORY_WEIGHT = 0.6, 0.4
_POSITION_HISTORY_LEN = 8
# Detection-to-track assignment: distance minus this many px per unit of confidence.
_ASSIGN_CONF_BONUS_PX = 20.0
# Acceptance radius: a fraction of the frame diagonal or the track's speed
# times a slack, capped relative to the selector's maximum ball speed.
_ASSIGN_BASE_DIAG_FRAC = 0.04
_ASSIGN_SPEED_SLACK = 1.5
_ASSIGN_MAX_SPEED_SLACK = 1.5
# Tracks drop after this many frames without a detection: 5 caused motion
# gaps at detector skip frames, 15 let coasting ROIs sweep incidental motion
# onto static balls.
_MAX_FRAMES_LOST = 8
# Two tracks predicted closer than this fraction of the diagonal are the same ball.
_DUPLICATE_TRACK_DIAG_FRAC = 0.025
# Visible-ball ROI: at most 1.5x the detection box, widened a little for low confidence.
_VISIBLE_BOX_SCALE = 1.5
_LOW_CONF_EXPAND = 0.25
# Lost-ball ROI: at least 4x the last box, so the search is genuinely wider than the ball.
_LOST_BOX_SCALE = 4.0
# ROIs covering this fraction of the frame or more are a full-frame search.
_FULL_FRAME_AREA_FRAC = 0.7


class ROITrack:
    """One ball hypothesis: last position, velocity and how long since the detector saw it."""

    def __init__(self, start_pos: Point, frame_idx: int, conf: float = 1.0, diag_box: float = 0.0) -> None:
        self.last_pos = start_pos
        self.last_vel: Point = (0.0, 0.0)
        self.frames_since_det = 0
        self.last_frame_idx = frame_idx
        self.pos_history: List[Tuple[float, float, int]] = [(start_pos[0], start_pos[1], frame_idx)]
        self.last_conf = conf
        self.last_diag_box = diag_box
        self.coasting_rois: List[CoastingRoi] = []

    def predict_roi(self, dt: int, phys_cfg: SelectorConfig) -> Point:
        """Predict the ball position ``dt`` frames on with the selector's gravity+drag
        projectile model, so the ROI and the selector agree on where the ball is going."""
        return predict_projectile(self.last_pos, self.last_vel, dt, phys_cfg)

    def predicted_center(self, phys_cfg: SelectorConfig) -> Point:
        return self.predict_roi(max(self.frames_since_det, 1), phys_cfg)

    def update(self, pos: Point, frame_idx: int, conf: float = 1.0, diag_box: float = 0.0) -> None:
        """Take a detection at ``pos``; the coasting history is cleared except this frame's ROI."""
        self.last_frame_idx = frame_idx
        self.last_conf = conf
        self.last_diag_box = diag_box
        # The ROI generated for this frame found the ball, so it was right; the rest were not.
        self.coasting_rois = [r for r in self.coasting_rois if r[0] == frame_idx]
        dt_since_last = max(self.frames_since_det + 1, 1)
        vx = (pos[0] - self.last_pos[0]) / dt_since_last
        vy = (pos[1] - self.last_pos[1]) / dt_since_last

        speed = (vx ** 2 + vy ** 2) ** 0.5
        if speed > _VELOCITY_RESET_SPEED_PX:
            self.pos_history.clear()
            self.pos_history.append((pos[0], pos[1], frame_idx))
            self.last_vel = (0.0, 0.0)
        else:
            self.pos_history.append((pos[0], pos[1], frame_idx))
            if len(self.pos_history) > _POSITION_HISTORY_LEN:
                self.pos_history.pop(0)

            if len(self.pos_history) >= 2:
                recent = self.pos_history[-2:]
                if recent[-1][2] != recent[0][2]:
                    dt_hist = recent[-1][2] - recent[0][2]
                    vx_hist = (recent[-1][0] - recent[0][0]) / max(dt_hist, 1)
                    vy_hist = (recent[-1][1] - recent[0][1]) / max(dt_hist, 1)
                    self.last_vel = (_VELOCITY_STEP_WEIGHT * vx + _VELOCITY_HISTORY_WEIGHT * vx_hist,
                                     _VELOCITY_STEP_WEIGHT * vy + _VELOCITY_HISTORY_WEIGHT * vy_hist)
                else:
                    self.last_vel = (vx, vy)
            else:
                self.last_vel = (vx, vy)

        self.last_pos = pos
        self.frames_since_det = 0


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _box_diagonal(bbox: Sequence[float]) -> float:
    return ((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2) ** 0.5


class ROIMotionTracker:
    """Tracks where each ball was last seen and provides the ROIs for motion detection.

    Every detection gets its own physics track, so several balls never share
    a region and a lost ball's region grows until it is found again or times out.
    """

    def __init__(self, cfg: Config, frame_w: int, frame_h: int, fps: float = 30.0) -> None:
        self.cfg = cfg
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.fps = fps
        self.diag = (frame_w ** 2 + frame_h ** 2) ** 0.5

        self.phys_cfg = SelectorConfig()
        self.phys_cfg.width = frame_w
        self.phys_cfg.height = frame_h
        self.phys_cfg.fps = fps
        self.phys_cfg.auto_scale()

        self.tracks: List[ROITrack] = []
        self._fullframe_counter = 0
        self.last_rois: Optional[List[Box]] = None
        self.last_rois_visual: Optional[List[Box]] = None
        self.ball_visible = False

    def _assign_detections(self, dets: Sequence[Detection], frame_idx: int) -> None:
        """Greedily match tracks to detections by predicted distance; the rest start new tracks."""
        unassigned: List[Dict[str, Any]] = [
            {"pos": ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0), "conf": conf, "bbox": bbox}
            for bbox, conf in dets
        ]
        for t in self.tracks:
            if not unassigned:
                t.frames_since_det += 1
                continue
            pred = t.predicted_center(self.phys_cfg)
            best_idx = min(range(len(unassigned)),
                           key=lambda i: _distance(unassigned[i]["pos"], pred) - unassigned[i]["conf"] * _ASSIGN_CONF_BONUS_PX)
            dist = _distance(unassigned[best_idx]["pos"], pred)
            # The acceptance radius follows the track's speed, within the selector's speed limit.
            current_speed = (t.last_vel[0] ** 2 + t.last_vel[1] ** 2) ** 0.5
            base_max_dist = max(_ASSIGN_BASE_DIAG_FRAC * self.diag, current_speed * _ASSIGN_SPEED_SLACK)
            max_dist = min(base_max_dist, self.phys_cfg.max_speed_px_per_frame * _ASSIGN_MAX_SPEED_SLACK)
            if dist < max_dist:
                d = unassigned.pop(best_idx)
                t.update(d["pos"], frame_idx, conf=d["conf"], diag_box=_box_diagonal(d["bbox"]))
            else:
                t.frames_since_det += 1
        for d in unassigned:
            self.tracks.append(ROITrack(d["pos"], frame_idx, conf=d["conf"], diag_box=_box_diagonal(d["bbox"])))

    def update_from_dets(self, dets: Sequence[Detection], frame_idx: int) -> List[CoastingRoi]:
        """Fold this frame's ball detections in.

        Returns the coasting ROIs ``(frame_idx, x1, y1, x2, y2)`` of every track
        that timed out or merged into another; the motion inside them was never
        ball evidence and the caller erases it.
        """
        if not dets:
            for t in self.tracks:
                t.frames_since_det += 1
        else:
            self._assign_detections(dets, frame_idx)

        dropped_rois: List[CoastingRoi] = []
        for t in self.tracks:
            if t.frames_since_det >= _MAX_FRAMES_LOST:
                dropped_rois.extend(t.coasting_rois)
        self.tracks = [t for t in self.tracks if t.frames_since_det < _MAX_FRAMES_LOST]

        # Tracks that drifted onto the same ball: keep the freshest, so a lost
        # expanding box cannot shadow a new tight box on the same ball.
        if len(self.tracks) > 1:
            keep: List[ROITrack] = []
            for t in sorted(self.tracks, key=lambda t: t.frames_since_det):
                pred = t.predicted_center(self.phys_cfg)
                duplicate = any(_distance(pred, kt.predicted_center(self.phys_cfg)) < _DUPLICATE_TRACK_DIAG_FRAC * self.diag
                                for kt in keep)
                if duplicate:
                    dropped_rois.extend(t.coasting_rois)    # a ghost that merged onto a real ball
                else:
                    keep.append(t)
            self.tracks = keep

        self.ball_visible = len(self.tracks) > 0 and any(t.frames_since_det == 0 for t in self.tracks)
        return dropped_rois

    def _track_radii(self, t: ROITrack) -> Tuple[float, float]:
        """``(visual, motion)`` radii for a track: tight around a visible ball, growing while lost."""
        cfg = self.cfg
        if t.frames_since_det == 0:
            base = cfg.roi_visible_radius_frac * self.diag
            if t.last_diag_box > 0:
                base = min(base, t.last_diag_box * _VISIBLE_BOX_SCALE)
            radius_visual = base * (1.0 + ((1.0 - t.last_conf) * _LOW_CONF_EXPAND))
        else:
            # A lost ball's own box is tiny; take the larger of the configured
            # base and a multiple of the box so the search really widens.
            base = cfg.roi_lost_radius_frac * self.diag
            if t.last_diag_box > 0:
                base = max(base, t.last_diag_box * _LOST_BOX_SCALE)
            growth = cfg.roi_lost_expand_per_frame * self.diag * t.frames_since_det
            radius_visual = min(base + growth, cfg.roi_max_radius_frac * self.diag)
        return radius_visual, radius_visual + cfg.roi_motion_bleed_frac * self.diag

    def _clip(self, x: float, y: float, radius: float, clamp_low: bool) -> Box:
        w, h = self.frame_w, self.frame_h
        if clamp_low:
            return (max(0, min(int(x - radius), w)), max(0, min(int(y - radius), h)),
                    max(0, min(int(x + radius), w)), max(0, min(int(y + radius), h)))
        return (max(0, int(x - radius)), max(0, int(y - radius)), min(w, int(x + radius)), min(h, int(y + radius)))

    def next_rois(self, frame_idx: int) -> Tuple[Optional[List[Box]], Optional[List[Box]]]:
        """Advance to ``frame_idx`` and return its motion ROIs.

        Returns ``(rois, rois_visual)``: the ``(x1, y1, x2, y2)`` regions to
        compute motion in (or None for the full frame) and their tight visual
        bounds for the debug video. Each call is one frame: it advances the
        periodic full-frame schedule and records the coasting ROI of every
        track that has lost the ball, so call it exactly once per frame.
        """
        if not self.cfg.roi_motion_enabled:
            return None, None
        force_fullframe = self._advance_fullframe_schedule()
        if not self.tracks:
            if force_fullframe:
                return None, None
            # Reuse the last ROI between periodic full-frame lost-ball probes.
            if self.last_rois is not None:
                return self.last_rois, self.last_rois_visual
            return None, None

        rois, rois_visual = self._track_rois(frame_idx)
        valid_rois = self._valid_rois(rois)
        if valid_rois is None:                   # one ROI covers the frame: compute motion full-frame
            self.last_rois = None
            return None, None
        self.last_rois = valid_rois if valid_rois else None
        self.last_rois_visual = rois_visual if rois_visual else None
        return (None, None) if force_fullframe else (self.last_rois, self.last_rois_visual)

    def _advance_fullframe_schedule(self) -> bool:
        """Count this frame against the periodic full-frame probe; whether this is a probe frame
        (the interval is up and no track has the ball in sight)."""
        force_fullframe = False
        if (self.cfg.roi_fullframe_interval > 0 and
                self._fullframe_counter >= self.cfg.roi_fullframe_interval):
            self._fullframe_counter = 0
            force_fullframe = not self.ball_visible
        self._fullframe_counter += 1
        return force_fullframe

    def _track_rois(self, frame_idx: int) -> Tuple[List[Box], List[Box]]:
        """``(motion, visual)`` ROIs about every track's predicted position at ``frame_idx``
        (tracks whose prediction or radius is not finite are skipped), recording the motion ROI
        of a track that has lost the ball as a coasting ROI."""
        rois: List[Box] = []
        rois_visual: List[Box] = []
        for t in self.tracks:
            pred_cx, pred_cy = t.predict_roi(max(1, frame_idx - t.last_frame_idx), self.phys_cfg)
            if not (math.isfinite(float(pred_cx)) and math.isfinite(float(pred_cy))):
                continue
            radius_visual, radius_motion = self._track_radii(t)
            if not (math.isfinite(float(radius_visual)) and math.isfinite(float(radius_motion))):
                continue
            rois_visual.append(self._clip(pred_cx, pred_cy, radius_visual, clamp_low=False))
            roi = self._clip(pred_cx, pred_cy, radius_motion, clamp_low=True)
            rois.append(roi)
            # Only coasting frames are recorded: recording every frame made the
            # retroactive erase O(N) when a track dropped.
            if t.frames_since_det > 0:
                t.coasting_rois.append((frame_idx, *roi))
        return rois, rois_visual

    def _valid_rois(self, rois: List[Box]) -> Optional[List[Box]]:
        """The ROIs with an area, or None when one of them is most of the frame."""
        valid_rois: List[Box] = []
        for rx1, ry1, rx2, ry2 in rois:          # already clipped to the frame by _clip
            w, h = rx2 - rx1, ry2 - ry1
            if w <= 0 or h <= 0:
                continue
            if w * h >= _FULL_FRAME_AREA_FRAC * (self.frame_w * self.frame_h):
                return None
            valid_rois.append((rx1, ry1, rx2, ry2))
        return valid_rois
