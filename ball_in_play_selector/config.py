import math
from dataclasses import dataclass, field
from typing import Optional

# Diagonal of a 1920x1080 frame: the resolution every pixel threshold below
# was tuned at, so a threshold given as a fraction is multiplied by `diag` and
# one given in pixels by `diag / REFERENCE_DIAG_PX`.
REFERENCE_DIAG_PX = 2203.0
# Frame rate the per-frame speed thresholds were tuned at.
REFERENCE_FPS = 30.0
# Clamp on the frame-rate ratio so an odd clip (8 FPS, 240 FPS) cannot scale a
# per-frame threshold to nothing or to the whole frame.
_FPS_SCALE_MIN = 0.35
_FPS_SCALE_MAX = 3.0


def fps_scale(fps: float) -> float:
    """Multiplier that converts a 30 FPS px/frame quantity to `fps`."""
    scale = REFERENCE_FPS / max(float(fps), 1.0)
    return max(_FPS_SCALE_MIN, min(_FPS_SCALE_MAX, scale))


# Weights and fractions the selector multiplies as given; a negative one would
# silently invert a penalty.
_NON_NEGATIVE_FIELDS = (
    "timeline_min_track_score", "timeline_coverage_weight", "timeline_score_weight",
    "timeline_max_gap_frac", "timeline_overlap_tol_frac", "timeline_small_gap_frac",
    "timeline_small_gap_jump_reject_frac", "trail_jump_break_growth_frac", "bounce_restitution",
)
# Frame counts the selector divides by or uses as loop bounds.
_POSITIVE_FRAME_COUNT_FIELDS = ("trail_stitch_min_obs", "assoc_gate_growth_cap_frames")


@dataclass
class SelectorConfig:
    """Selector parameters.

    Fields declared with ``field(init=False)`` are derived from ``fps``,
    ``width`` and ``height`` when the config is constructed.  To change the
    video geometry afterwards, set those three and call :meth:`auto_scale`.
    """
    fps: float = 30.0
    width: int = 1920
    height: int = 1080
    diag: float = field(init=False)

    # -- Association --------------------------------------------------------
    max_gap_frames: int = field(init=False)           # misses before a track dies
    base_gate_px: float = field(init=False)           # association gate at dt=1
    gate_growth_px: float = field(init=False)         # gate growth per missed frame
    max_speed_px_per_frame: float = field(init=False) # absolute speed ceiling
    assoc_gate_growth_cap_frames: int = 6   # gate stops growing after this many misses
    assoc_min_speed_for_dir: float = 2.5    # 30 FPS px/frame below which direction is not judged
    assoc_speed_ratio_min: float = 0.15
    assoc_speed_ratio_max: float = 3.80
    assoc_dir_cos_min: float = -1.0         # bounces and racket hits reverse direction
    assoc_accel_frac: float = 0.09          # per-frame velocity change ceiling, diag fraction
    assoc_area_ratio_min: float = 0.15
    assoc_area_ratio_max: float = 6.00
    # A track that has sat still for eight observations may not absorb a
    # detection moving twice this fast (diag fraction per frame): that is a
    # ball passing a static false positive, not the false positive moving.
    assoc_static_speed_frac: float = 0.010
    # Trail creation / reconnect guards (build_tracks)
    trail_stitch_enable: bool = True
    trail_stitch_gap_frames: int = field(init=False)        # oldest finished track a detection may rejoin
    trail_stitch_min_stale_frames: int = field(init=False)  # youngest finished track a detection may rejoin
    trail_stitch_min_obs: int = 3
    trail_stitch_dist_frac: float = 0.06
    trail_stitch_max_step_frac: float = 0.11
    trail_stitch_pred_resid_frac: float = 0.08
    trail_jump_break_frac: float = 0.06
    trail_jump_break_growth_frac: float = 0.020

    # -- Scoring weights ------------------------------------------------------
    w_len: float = 1.0          # reward per observation (linear)
    w_span: float = 70.0        # reward for larger temporal span
    timeline_min_track_score: float = 2.0  # hard floor for timeline candidates
    timeline_coverage_weight: float = 1000.0  # emphasize covering more of the video
    timeline_score_weight: float = 6.0  # keep score as secondary preference
    timeline_max_gap_frac: float = 0.85  # tighter max forward gap between stitched tracks
    timeline_overlap_tol_frac: float = 0.050  # small overlap tolerance for handover
    timeline_small_gap_frac: float = 0.02  # "small gap" window for strict jump rejects
    timeline_small_gap_jump_reject_frac: float = 0.08  # reject cross-court jumps on tiny gaps
    timeline_switch_penalty: float = 34.0  # stronger cost for switching track ids
    timeline_gap_penalty: float = 180.0  # cost per uncovered time fraction between stitched tracks
    timeline_overlap_penalty: float = 130.0  # cost per overlap time fraction
    timeline_jump_penalty: float = 130.0  # stronger cost per normalized spatial jump between stitched tracks

    # Court margin: soft margin around the court polygon
    court_expand_px: float = field(init=False)

    # -- Motion fallback physics gate ------------------------------
    # Blob search radius, read by core._motion_search_radius.  A flat radius is
    # wrong in both directions: too wide for a slow ball beside a player, too
    # narrow for a fast one whose predicted point is off by a few percent of a
    # large step, so it grows with elapsed prediction and with ball speed.
    # Measured: a 22 px base with a 132 px cap put three motion frames on a
    # player, a line and the background for +0.002 recall, so the base stays
    # near the flat radius it replaced.  Widen only with numbers in hand.
    motion_search_base_frac: float = 0.007   # 15.4 px at 1080p
    motion_search_growth_px: float = 2.5     # per predicted frame beyond the first
    motion_search_vel_mult: float = 0.20     # per px/frame of ball speed
    motion_search_min_px: float = 14.0
    motion_search_max_frac: float = 0.030    # 66 px at 1080p
    motion_tail_miss_budget: int = 1         # consecutive blobless frames a tail may coast

    # A detection this confident and on the motion mask is corroborated by the
    # mask, so isolation from its own track does not condemn it.
    isolation_keep_conf: float = 0.80

    # -- Bounded fill -----------------------------------------------------
    # A long fill is a physics claim, so the ball may not change speed wildly
    # across it.  2x was too tight: gravity alone shifts |v| near apex and a
    # racket strike routinely exceeds it.
    fill_speed_ratio_min: float = 0.35
    fill_speed_ratio_max: float = 3.00

    # -- Projectile physics (gravity model) ---------------------------
    gravity_px_per_frame2: float = field(init=False)  # gravity in px/frame^2 at this fps/resolution
    gravity_base_ppf2_30fps: float = 0.55  # gravity at 30fps in px/frame^2 (tunable)
    gravity_enabled: bool = True         # enable parabolic prediction
    gravity_drag_factor: float = 0.985   # velocity retention per 30 FPS reference step (1.0 = no drag)
    bounce_restitution: float = 0.58      # reflected vy fraction on bounce

    # -- Track eligibility (core.selected_tracks) ----------------------
    # Seconds of observations a track needs before it may be selected at all.
    select_min_obs_sec: float = 0.25
    select_reacquire_obs_sec: float = 0.16
    select_min_obs_frames: int = field(init=False)        # from select_min_obs_sec
    select_reacquire_obs_frames: int = field(init=False)  # from select_reacquire_obs_sec
    select_established_motion_frac: float = 0.5   # raw mask overlap for a plain rally track
    select_reacquire_motion_frac: float = 0.25
    select_rolling_motion_frac: float = 0.15
    select_reacquire_conf: float = 0.55
    select_rolling_conf: float = 0.80
    select_outside_strict_max: float = 0.1  # "mostly outside the court" ceiling for a reacquisition
    select_inside_strict_min: float = 0.5   # "mostly inside the court" floor for a slow roll
    select_rolling_extent_frac: float = 0.02
    select_score_density_min: float = 0.5
    select_rolling_score_density_min: float = 0.12
    select_obs_density_min: float = 0.5
    # A ball can be unmistakable from its flight alone. The boost mask misses
    # small, distant and slow balls, and the two court-fraction gates above
    # leave a dead band (outside_strict_max .. inside_strict_min) that a rally
    # crossing the baseline lands in. Kinematic evidence covers both, so it is
    # held to a higher bar than the mask path: 0.60 is the most the extent term
    # alone can contribute (physics.kinematic_motion_frac), so this floor
    # demands real speed, not merely a wide footprint.
    select_kinematic_motion_frac: float = 0.75
    select_kinematic_extent_frac: float = 0.05

    def __post_init__(self) -> None:
        self.auto_scale()

    def _validate(self) -> None:
        """Reject values the selector's arithmetic assumes away (it never clamps them)."""
        for name in _NON_NEGATIVE_FIELDS:
            if getattr(self, name) < 0:
                raise ValueError(f"SelectorConfig.{name} must be >= 0, got {getattr(self, name)!r}")
        for name in _POSITIVE_FRAME_COUNT_FIELDS:
            if getattr(self, name) < 1:
                raise ValueError(f"SelectorConfig.{name} must be >= 1, got {getattr(self, name)!r}")

    def auto_scale(self) -> "SelectorConfig":
        """Derive the resolution/fps-dependent fields from fps, width and height."""
        self._validate()
        self.diag = math.sqrt(self.width ** 2 + self.height ** 2)
        diag = self.diag
        fps_now = max(float(self.fps), 1.0)
        frame_scale = fps_scale(fps_now)

        # Convert short temporal windows to frame counts from real time (seconds)
        # so behavior is more consistent across 30/60/120fps inputs.
        def _frames_for_seconds(sec: float, min_frames: int = 1, max_frames: Optional[int] = None) -> int:
            n = int(round(max(0.0, float(sec)) * fps_now))
            n = max(int(min_frames), n)
            if max_frames is not None:
                n = min(int(max_frames), n)
            return n

        self.max_gap_frames = _frames_for_seconds(20.0 / 30.0, min_frames=8, max_frames=120)
        self.base_gate_px = 0.085 * diag      # stricter to reduce cross-court track snaps
        self.gate_growth_px = 0.010 * diag    # slower growth on missed frames
        self.max_speed_px_per_frame = 0.16 * diag * frame_scale  # tighter speed ceiling normalized across FPS
        self.court_expand_px = 0.12 * diag
        self.select_min_obs_frames = _frames_for_seconds(self.select_min_obs_sec, min_frames=3)
        self.select_reacquire_obs_frames = _frames_for_seconds(self.select_reacquire_obs_sec, min_frames=10)
        self.trail_stitch_gap_frames = _frames_for_seconds(30.0 / 30.0, min_frames=6, max_frames=90)
        self.trail_stitch_min_stale_frames = _frames_for_seconds(3.0 / 30.0, min_frames=1, max_frames=12)
        # Gravity: scale from the 30 FPS, 1080p reference
        res_scale = diag / REFERENCE_DIAG_PX
        self.gravity_px_per_frame2 = (
            self.gravity_base_ppf2_30fps * res_scale * (frame_scale ** 2)
        )
        return self
