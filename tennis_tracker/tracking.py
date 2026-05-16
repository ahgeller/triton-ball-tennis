import math
from typing import List, Tuple
import numpy as np
from ball_in_play_selector import _predict_projectile, SelectorConfig
from .config import Config


class ROITrack:
    def __init__(self, start_pos: Tuple[float, float], frame_idx: int, conf: float = 1.0, diag_box: float = 0.0):
        self.last_pos = start_pos
        self.last_vel = (0.0, 0.0)
        self.frames_since_det = 0
        self.last_frame_idx = frame_idx
        self.pos_history = [(start_pos[0], start_pos[1], frame_idx)]
        self.max_history = 8
        self.last_conf = conf
        self.last_diag_box = diag_box
        self.coasting_rois = []  # Stores (frame_idx, rx1, ry1, rx2, ry2)

    def predict_roi(self, dt: int, phys_cfg: SelectorConfig) -> Tuple[float, float]:
        """Predict ball position using the same gravity+drag projectile model as
        the carry/blue-line path (ball_in_play_selector.physics._predict_projectile),
        so the ROI box and the blue carry trail agree on where the ball is going."""
        return _predict_projectile(self.last_pos, self.last_vel, dt, phys_cfg)
        
    def predicted_center(self, phys_cfg: SelectorConfig) -> Tuple[float, float]:
        dt = max(self.frames_since_det, 1)
        return self.predict_roi(dt, phys_cfg)
        
    def update(self, pos: Tuple[float, float], frame_idx: int, conf: float = 1.0, diag_box: float = 0.0):
        self.last_frame_idx = frame_idx
        self.last_conf = conf
        self.last_diag_box = diag_box
        # Successfully found a detection. Clear the ghost track history, but keep the current frame's
        # ROI if one was generated, because that ROI correctly found the ball.
        self.coasting_rois = [r for r in self.coasting_rois if r[0] == frame_idx]
        dt_since_last = max(self.frames_since_det + 1, 1)
        vx = (pos[0] - self.last_pos[0]) / dt_since_last
        vy = (pos[1] - self.last_pos[1]) / dt_since_last
        
        speed = (vx**2 + vy**2)**0.5
        if speed > 150.0:
            self.pos_history.clear()
            self.pos_history.append((pos[0], pos[1], frame_idx))
            self.last_vel = (0.0, 0.0)
        else:
            self.pos_history.append((pos[0], pos[1], frame_idx))
            if len(self.pos_history) > self.max_history:
                self.pos_history.pop(0)
                
            if len(self.pos_history) >= 2:
                recent = self.pos_history[-2:]
                if recent[-1][2] != recent[0][2]:
                    dt_hist = recent[-1][2] - recent[0][2]
                    vx_hist = (recent[-1][0] - recent[0][0]) / max(dt_hist, 1)
                    vy_hist = (recent[-1][1] - recent[0][1]) / max(dt_hist, 1)
                    self.last_vel = (0.6 * vx + 0.4 * vx_hist, 0.6 * vy + 0.4 * vy_hist)
                else:
                    self.last_vel = (vx, vy)
            else:
                self.last_vel = (vx, vy)
                
        self.last_pos = pos
        self.frames_since_det = 0

class ROIMotionTracker:
    """Tracks where the ball was last seen and provides an ROI for motion detection.
    
    IMPROVED: Multi-Object Tracker (MOT) capable. Assigns independent physics Tracks
    to every detection, preventing jumping between multiple balls entirely.
    """

    def __init__(self, cfg: Config, frame_w: int, frame_h: int, fps: float = 30.0):
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
        self.last_rois = None
        self.ball_visible = False

    def update_from_dets(self, dets, frame_idx):
        """Call after YOLO detection. Updates state based on whether ball was found."""
        if not dets:
            for t in self.tracks:
                t.frames_since_det += 1
        else:
            unassigned_dets = []
            for bbox, conf in dets:
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0
                unassigned_dets.append({"pos": (cx, cy), "conf": conf, "bbox": bbox})
                
            # Greedily match existing tracks to detections
            for t in self.tracks:
                if not unassigned_dets:
                    t.frames_since_det += 1
                    continue
                    
                pred = t.predicted_center(self.phys_cfg)
                best_idx = -1
                best_score = float('inf')
                
                for i, d in enumerate(unassigned_dets):
                    dist = ((d["pos"][0] - pred[0])**2 + (d["pos"][1] - pred[1])**2)**0.5
                    score = dist - (d["conf"] * 20.0)
                    if score < best_score:
                        best_score = score
                        best_idx = i
                        
                if best_idx != -1:
                    dist = ((unassigned_dets[best_idx]["pos"][0] - pred[0])**2 + (unassigned_dets[best_idx]["pos"][1] - pred[1])**2)**0.5
                    # Dynamic acceptance distance based on how fast the object is moving, capped at a reasonable limit
                    # If it's the exact same detection box, we allow slightly larger distances to prevent ghost tracks
                    # IMPROVED: Use relative court scaling (max_speed_px_per_frame) instead of hard-coded pixels
                    current_speed = (t.last_vel[0]**2 + t.last_vel[1]**2)**0.5
                    base_max_dist = max(0.04 * self.diag, current_speed * 1.5)
                    max_dist = min(base_max_dist, self.phys_cfg.max_speed_px_per_frame * 1.5)
                    
                    if dist < max_dist:
                        d = unassigned_dets.pop(best_idx)
                        bbox = d["bbox"]
                        diag_box = (((bbox[2]-bbox[0])**2 + (bbox[3]-bbox[1])**2)**0.5)
                        t.update(d["pos"], frame_idx, conf=d["conf"], diag_box=diag_box)
                    else:
                        t.frames_since_det += 1
                else:
                    t.frames_since_det += 1

            # Any remaining detections get a fresh, independent track
            for d in unassigned_dets:
                bbox = d["bbox"]
                diag_box = (((bbox[2]-bbox[0])**2 + (bbox[3]-bbox[1])**2)**0.5)
                self.tracks.append(ROITrack(d["pos"], frame_idx, conf=d["conf"], diag_box=diag_box))

        # Drop tracks that are hopelessly lost (timeout after 8 frames of no detections).
        # 5 caused motion gaps at YOLO skip frames; 15 let coasting ROIs sweep incidental motion onto static balls.
        max_lost = 8
        dropped_rois = []
        for t in self.tracks:
            if t.frames_since_det >= max_lost:
                dropped_rois.extend(t.coasting_rois)
        self.tracks = [t for t in self.tracks if t.frames_since_det < max_lost]
        
        # Deduplicate tracks that have drifted onto the exact same physical ball
        # This prevents a "lost" expanding box from shadowing a new "active" tight box for the exact same ball.
        if len(self.tracks) > 1:
            keep = []
            tracks_sorted = sorted(self.tracks, key=lambda t: t.frames_since_det)
            for t in tracks_sorted:
                is_duplicate = False
                pred = t.predicted_center(self.phys_cfg)
                for kt in keep:
                    k_pred = kt.predicted_center(self.phys_cfg)
                    dist = ((pred[0] - k_pred[0])**2 + (pred[1] - k_pred[1])**2)**0.5
                    if dist < (0.025 * self.diag):
                        is_duplicate = True
                        break
                if not is_duplicate:
                    keep.append(t)
                else:
                    # If this track was a ghost that merged onto a real ball, its unique ghost history was wrong
                    dropped_rois.extend(t.coasting_rois)
            self.tracks = keep

        self.ball_visible = len(self.tracks) > 0 and any(t.frames_since_det == 0 for t in self.tracks)
        
        return dropped_rois

    def get_rois(self, frame_idx):
        """Return a list of (x1, y1, x2, y2) ROIs or None for full-frame, plus their tight visual bounds."""
        if not self.cfg.roi_motion_enabled:
            return None, None

        force_fullframe = False
        if (self.cfg.roi_fullframe_interval > 0 and
                self._fullframe_counter >= self.cfg.roi_fullframe_interval):
            self._fullframe_counter = 0
            if not self.ball_visible:
                # Retain the exact same logic as tracking loss:
                if self.last_rois is not None:
                    return self.last_rois, self.last_rois_visual
                return None, None
            else:
                force_fullframe = True
        self._fullframe_counter += 1

        if not self.tracks:
            # IMPROVED: Instead of returning None (which triggers a full-frame motion sweep),
            # return the very last known ROI geometry so the motion filter stays constrained
            # to the last area the ball was seen.
            if self.last_rois is not None:
                return self.last_rois, self.last_rois_visual
            return None, None

        rois = []
        rois_visual = []
        for t in self.tracks:
            pipeline_dt = max(1, frame_idx - t.last_frame_idx)
            
            # Predict box location in real-time frame
            pred_cx, pred_cy = t.predict_roi(pipeline_dt, self.phys_cfg)
            if not (math.isfinite(float(pred_cx)) and math.isfinite(float(pred_cy))):
                continue
            
            if t.frames_since_det == 0:
                base = min(self.cfg.roi_visible_radius_frac * self.diag, t.last_diag_box * 1.5) if t.last_diag_box > 0 else self.cfg.roi_visible_radius_frac * self.diag
                # Expand slightly if confidence is low, but not massively
                radius_visual = base * (1.0 + ((1.0 - t.last_conf) * 0.25))
            else:
                # When lost, prefer the diag-frac base - the ball's own bbox is tiny and
                # would pin the search region to the ball's size, defeating the purpose of
                # widening on loss. Take the larger of (config base) and (last_diag_box * 4)
                # so the lost search region is genuinely wider than the ball.
                base = self.cfg.roi_lost_radius_frac * self.diag
                if t.last_diag_box > 0:
                    base = max(base, t.last_diag_box * 4.0)
                growth = self.cfg.roi_lost_expand_per_frame * self.diag * t.frames_since_det
                radius_visual = min(base + growth, self.cfg.roi_max_radius_frac * self.diag)

            radius_motion = radius_visual + self.cfg.roi_motion_bleed_frac * self.diag
            if not (math.isfinite(float(radius_visual)) and math.isfinite(float(radius_motion))):
                continue

            rx1_v = max(0, int(pred_cx - radius_visual))
            ry1_v = max(0, int(pred_cy - radius_visual))
            rx2_v = min(self.frame_w, int(pred_cx + radius_visual))
            ry2_v = min(self.frame_h, int(pred_cy + radius_visual))
            rois_visual.append((rx1_v, ry1_v, rx2_v, ry2_v))

            rx1_m = max(0, min(int(pred_cx - radius_motion), self.frame_w))
            ry1_m = max(0, min(int(pred_cy - radius_motion), self.frame_h))
            rx2_m = max(0, min(int(pred_cx + radius_motion), self.frame_w))
            ry2_m = max(0, min(int(pred_cy + radius_motion), self.frame_h))
            rois.append((rx1_m, ry1_m, rx2_m, ry2_m))
            # Only record coasting ROIs for frames where the ball was NOT detected.
            # Recording every frame caused O(N) retroactive work when the track was dropped.
            if t.frames_since_det > 0:
                t.coasting_rois.append((frame_idx, rx1_m, ry1_m, rx2_m, ry2_m))

        valid_rois = []
        for r in rois:
            rx1 = max(0, min(r[0], self.frame_w))
            ry1 = max(0, min(r[1], self.frame_h))
            rx2 = max(0, min(r[2], self.frame_w))
            ry2 = max(0, min(r[3], self.frame_h))
            
            w = rx2 - rx1
            h = ry2 - ry1
            
            if w <= 0 or h <= 0:
                continue
                
            if w * h >= 0.7 * (self.frame_w * self.frame_h):
                self.last_rois = None
                return None, None
                
            valid_rois.append((rx1, ry1, rx2, ry2))

        self.last_rois = valid_rois if valid_rois else None
        self.last_rois_visual = rois_visual if rois_visual else None
        return (None, None) if force_fullframe else (self.last_rois, self.last_rois_visual)

class BallTracker:
    """Reserved temporal ball tracker; selector currently drives final tracking."""

    def __init__(self, cfg: Config, frame_w: int, frame_h: int):
        self.cfg = cfg
        self.frame_w = max(2, int(frame_w))
        self.frame_h = max(2, int(frame_h))
        self.diag = (frame_w ** 2 + frame_h ** 2) ** 0.5
        self.max_jump = cfg.ball_max_jump * self.diag
        self.last_box = None
        self.last_center = None
        self.velocity = (0.0, 0.0)
        self.missing = 0

    @staticmethod
    def _center(box):
        return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

    def _clip_box(self, box):
        x1, y1, x2, y2 = map(float, box)
        bw, bh = max(2.0, x2 - x1), max(2.0, y2 - y1)
        cx = float(np.clip((x1 + x2) / 2, bw * 0.5, self.frame_w - bw * 0.5))
        cy = float(np.clip((y1 + y2) / 2, bh * 0.5, self.frame_h - bh * 0.5))
        return [cx - bw * 0.5, cy - bh * 0.5, cx + bw * 0.5, cy + bh * 0.5]

    def update(self, dets):
        """Pick best detection, return bbox or None."""
        best_box = None
        best_score = -1e9

        for bbox, conf in dets:
            cx, cy = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0

            if self.last_center is not None:
                dist = ((cx - self.last_center[0])**2 + (cy - self.last_center[1])**2)**0.5
                if dist > self.max_jump:
                    continue
                dist_pen = self.cfg.ball_dist_weight * dist / self.diag
            else:
                dist_pen = 0.0

            iou_bonus = 0.0
            if self.last_box is not None:
                lx1, ly1, lx2, ly2 = self.last_box
                inter = max(0, min(bbox[2],lx2)-max(bbox[0],lx1)) * \
                        max(0, min(bbox[3],ly2)-max(bbox[1],ly1))
                union = max(0,(bbox[2]-bbox[0])*(bbox[3]-bbox[1])) + \
                        max(0,(lx2-lx1)*(ly2-ly1)) - inter
                if union > 0:
                    iou_bonus = self.cfg.ball_iou_weight * inter / union

            score = conf + iou_bonus - dist_pen
            if score > best_score:
                best_score = score
                best_box = bbox

        if best_box is not None:
            best_box = self._clip_box(best_box)
            new_center = self._center(best_box)
            if self.last_center is not None:
                dvx = new_center[0] - self.last_center[0]
                dvy = new_center[1] - self.last_center[1]
                if abs(dvx) + abs(dvy) > 0.2:
                    ovx, ovy = self.velocity
                    self.velocity = (0.6 * ovx + 0.4 * dvx, 0.6 * ovy + 0.4 * dvy)
            self.last_box = best_box
            self.last_center = new_center
            self.missing = 0
            return best_box

        self.missing += 1
        self.last_box = None
        self.last_center = None
        self.velocity = (0.0, 0.0)
        return None

