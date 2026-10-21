"""Pass 2: render the tracking video and the optional debug videos from pass 1's evidence.

Frames come from the RAM cache pass 1 filled, or from a second decode of the
clip.  Every frame gets the court, players, the selected ball and its trail;
the track-debug video adds raw detections, ROIs, every candidate track and the
chosen track's display path; the motion-debug video shows the masks.
"""
from __future__ import annotations

import os
import time
from typing import Any, List, Optional

import cv2
import numpy as np

from .court.court_geometry import MIN_KEYPOINT_VALUES, build_ground_projection_model, court_keypoints_key
from .court.court_minimap import draw_court_minimap
from .motion import build_court_side_protect_mask
from .render.motion_debug import RawMotionTrails, draw_motion_masks, draw_selected_ball
from .pass1 import Pass1Result
from .court.court_drawing import draw_court_layer
from .render.rendering import (
    Canvas, MarkerSizer, SearchCentreSmoother, draw_hud, draw_motion_rois, draw_player_boxes, draw_raw_detections,
    draw_selected_result,
)
from .render.trails import BallTrail, CandidateTrackTrails, DisplayPathTrail
from .run_context import CourtEvents, RunContext, Selection
from .video_io import VideoWriter

COLOR_ROI_THIN = (0, 255, 0)


def _open_writer(ctx: RunContext, path: str) -> VideoWriter:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    video = ctx.video
    return VideoWriter(path, video.fps, video.width, video.height, ctx.cfg, encoder=ctx.encoder)


class _FrameSource:
    """Pass-2 frames from the RAM cache when it holds the whole clip, else a fresh decode."""

    def __init__(self, ctx: RunContext, cached: Optional[List[np.ndarray]], frame_count: int) -> None:
        self._cached = cached if (cached is not None and len(cached) == frame_count) else None
        self._capture = None
        if self._cached is not None:
            print(f"[pass 2] Using RAM frame cache ({len(self._cached)} frames)")
        else:
            self._capture = cv2.VideoCapture(str(ctx.video.path))
            if not self._capture.isOpened():
                raise RuntimeError(f"Could not reopen the input video for pass 2: {ctx.video.path}")
            if cached is not None:
                print("[pass 2] RAM frame cache incomplete; falling back to video decode")

    def read(self, frame_index: int) -> Optional[np.ndarray]:
        if self._cached is not None:
            return self._cached[frame_index]
        ok, frame = self._capture.read()
        return frame if ok else None

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()


class _GroundModelCache:
    """The court-plane homography for the latest valid keypoints, rebuilt only when they move."""

    def __init__(self, initial_kps: Optional[List[float]]) -> None:
        self.keypoints = initial_kps
        self._key: Any = None
        self.model = None

    def update(self, court_kps: Optional[List[float]]):
        if court_kps and len(court_kps) >= MIN_KEYPOINT_VALUES:
            self.keypoints = court_kps
        key = court_keypoints_key(self.keypoints)
        if key != self._key:
            self.model = build_ground_projection_model(self.keypoints)
            self._key = key
        return self.model


class _ProtectMaskCache:
    """The court-side protect mask for the motion-debug overlay, per keypoint set."""

    def __init__(self, height: int, width: int) -> None:
        self.height, self.width = height, width
        self._key: Any = None
        self.mask: Optional[np.ndarray] = None
        self._first = True

    def get(self, court_kps: Optional[List[float]]) -> Optional[np.ndarray]:
        key = court_keypoints_key(court_kps)
        if self._first or key != self._key:
            self.mask = build_court_side_protect_mask(self.height, self.width, court_kps) if court_kps else None
            self._key = key
            self._first = False
        return self.mask


class _Writers:
    """The video outputs this run asked for: the tracking video, the motion-debug video and the
    track-debug video, each None when it is off."""

    def __init__(self, ctx: RunContext) -> None:
        cfg = ctx.cfg
        self.main = self.motion_debug = self.track_debug = None
        if cfg.save_tracking_video:
            self.main = _open_writer(ctx, cfg.output_video)
            print(f"[writer] Tracking encoder: {self.main.encoder}")
        if cfg.save_motion_debug:
            self.motion_debug = _open_writer(ctx, cfg.output_debug_path)
        if cfg.save_track_debug_video:
            self.track_debug = _open_writer(ctx, cfg.output_track_debug_path)

    def close(self) -> None:
        for w in (self.main, self.motion_debug, self.track_debug):
            if w is not None:
                w.close()


class _FrameRenderer:
    """Draws and writes one frame after another, carrying the trails, the court-model caches
    and the marker sizes from frame to frame."""

    def __init__(self, ctx: RunContext, pass1: Pass1Result, selection: Selection, events: CourtEvents,
                 writers: _Writers) -> None:
        cfg, video = ctx.cfg, ctx.video
        self.cfg, self.video, self.timing = cfg, video, ctx.timing
        self.pass1, self.selection, self.events, self.writers = pass1, selection, events, writers
        self.ground_model = _GroundModelCache(selection.last_court_keypoints)
        self.protect_masks = _ProtectMaskCache(video.height, video.width)
        self.sizer = MarkerSizer(cfg)
        self.search_smoother = SearchCentreSmoother()
        self.ball_trail = BallTrail(cfg, video.fps, video.width, video.height)
        self.path_trail = DisplayPathTrail(video.fps, video.width)
        self.candidate_trails = CandidateTrackTrails(selection.all_tracks, selection.chosen_track)
        self.raw_motion_trails = RawMotionTrails(video.width)

    def _draw(self, fi: int, frame: np.ndarray, result, court_kps) -> Canvas:
        """The tracking frame and, when the track-debug video is on, its debug twin."""
        cfg, pass1, selection = self.cfg, self.pass1, self.selection
        canvas = Canvas(frame.copy(), frame.copy() if self.writers.track_debug is not None else None)
        model = self.ground_model.update(court_kps)

        draw_court_layer(canvas.both, cfg, court_kps, model, selection.court_polygon)
        if cfg.draw_players:
            draw_player_boxes(canvas.both, pass1.player_boxes[fi])
        # Rejected proposals, ROIs and candidate tracks belong only in the track-debug view;
        # candidates go under the chosen result so the final choice sits on top.
        draw_raw_detections(canvas.debug_only, pass1.detections[fi])
        draw_motion_rois(canvas.debug_only, pass1.rois[fi])
        if canvas.debug is not None:
            self.candidate_trails.advance_and_draw(canvas.debug, fi)
        if result is not None:
            draw_selected_result(canvas, result, fi, self.sizer, self.search_smoother, cfg.draw_search_regions)
        self.ball_trail.advance(result, self.ground_model.keypoints)
        self.ball_trail.draw(canvas, cfg.draw_ball_trail)
        if cfg.draw_court_minimap:
            draw_court_minimap(canvas.main, model, pass1.player_boxes[fi], self.events.rally_legs, fi, self.video.fps,
                               contact_events=self.events.contacts)

        # The track-debug stream follows the chosen track's display path while
        # the selector has nothing; the current result keeps it continuous.
        path_point = selection.display_path.get(fi)
        if path_point is None and result is not None:
            path_point = (float(result.cx), float(result.cy), True)
        self.path_trail.advance(path_point)
        if canvas.debug is not None and result is None:
            self.path_trail.draw(canvas.debug, path_point)
        draw_hud(canvas, fi, result, selection.chosen_track)
        return canvas

    def _write_motion_debug(self, fi: int, frame: np.ndarray, result, court_kps) -> None:
        cfg, pass1, timing = self.cfg, self.pass1, self.timing
        write_t0 = timing.start()
        raw_m = pass1.raw_motions[fi].unpack() if pass1.raw_motions[fi] is not None else None
        boost_m = pass1.boost_masks[fi].unpack() if pass1.boost_masks[fi] is not None else None
        rois = pass1.rois[fi]
        vis = draw_motion_masks(frame, raw_m, boost_m, cfg, court_keypoints=court_kps,
                                protect_mask=self.protect_masks.get(court_kps), rois=rois)
        probe_style = cfg.debug_probe_motion_style
        if not probe_style:
            for rx1, ry1, rx2, ry2 in rois or ():
                cv2.rectangle(vis, (int(rx1), int(ry1)), (int(rx2), int(ry2)), COLOR_ROI_THIN, 1)
        if result is not None:
            draw_selected_ball(vis, result, self.sizer.radius)
        self.raw_motion_trails.advance(fi, None if probe_style else rois, boost_m)
        self.raw_motion_trails.draw(vis, fi)
        self.writers.motion_debug.write(vis)
        timing.add("pass2_write_debug", write_t0)

    def render(self, fi: int, frame: np.ndarray) -> None:
        """Draw frame ``fi`` and write it to every output that is on."""
        timing, writers = self.timing, self.writers
        render_t0 = timing.start()
        debug_write_elapsed = 0.0
        result = self.selection.per_frame[fi]
        court_kps = self.pass1.court_keypoints[fi]
        canvas = self._draw(fi, frame, result, court_kps)

        if writers.track_debug is not None:
            write_t0 = timing.start()
            writers.track_debug.write(canvas.debug)
            debug_write_elapsed += timing.add("pass2_write_track_debug", write_t0)
        if timing.enabled:
            timing.values["pass2_render"] = (timing.values.get("pass2_render", 0.0)
                                             + max(0.0, time.perf_counter() - render_t0 - debug_write_elapsed))
        if writers.main is not None:
            write_t0 = timing.start()
            writers.main.write(canvas.main)
            timing.add("pass2_write_main", write_t0)
        if writers.motion_debug is not None:
            self._write_motion_debug(fi, frame, result, court_kps)


def render_pass2(ctx: RunContext, pass1: Pass1Result, selection: Selection, events: CourtEvents) -> int:
    """Write the selected video outputs; returns the number of frames rendered (0 when none is on)."""
    cfg, timing = ctx.cfg, ctx.timing
    n = pass1.frame_count
    if ctx.needs_pass2_outputs:
        print("[pass 2] Rendering selected video output(s)...")
    else:
        print("[pass 2] Skipped (no video outputs selected)")
    pass2_t0 = timing.start()

    writers = _Writers(ctx)
    rendered = 0
    if not ctx.needs_pass2_outputs:
        timing.add("pass2_total", pass2_t0)
        return rendered

    source = _FrameSource(ctx, pass1.cached_frames, n)
    renderer = _FrameRenderer(ctx, pass1, selection, events, writers)
    try:
        for fi in range(n):
            read_t0 = timing.start()
            frame = source.read(fi)
            if frame is None:
                break
            timing.add("pass2_read", read_t0)
            renderer.render(fi, frame)
            rendered += 1
            if cfg.progress_every and (fi + 1) % cfg.progress_every == 0:
                print(f"  [pass2 {fi+1}/{n}] rendering...")
    finally:
        source.release()
        writers.close()
    if rendered < n:
        print(f"[pass 2] Warning: the second decode gave {rendered} of the {n} frames pass 1 saw; "
              f"the last {n - rendered} frame(s) are missing from the video output(s)")
    timing.add("pass2_total", pass2_t0)
    return rendered
