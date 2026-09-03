import math
from typing import Optional
from dataclasses import dataclass




@dataclass
class SelectorConfig:
    """All parameters auto-scale from fps/resolution."""
    fps: float = 30.0
    width: int = 1920
    height: int = 1080
    diag: float = 0.0

    # Association
    max_gap_frames: int = 0          # auto-set
    base_gate_px: float = 0.0       # auto-set
    gate_growth_px: float = 0.0     # auto-set
    max_speed_px_per_frame: float = 0.0  # auto-set (absolute ceiling)
    # IMPROVED: More lenient association parameters for better ball continuity
    assoc_gate_growth_cap_frames: int = 6  # INCREASED from 4 for better gap tolerance
    assoc_min_speed_for_dir: float = 2.5   # INCREASED from 2.0 for better direction checks
    assoc_speed_ratio_min: float = 0.15    # LOWERED from 0.18 for more lenient matching
    assoc_speed_ratio_max: float = 3.80    # INCREASED from 3.20 for fast ball tolerance
    assoc_dir_cos_min: float = -1.0        # ALLOW bounces and racket hits
    assoc_accel_frac: float = 0.09         # INCREASED from 0.07 for acceleration tolerance
    assoc_area_ratio_min: float = 0.15     # LOWERED from 0.20 for size variation
    assoc_area_ratio_max: float = 6.00     # INCREASED from 5.00 for close-up balls
    # Trail creation / reconnect guards (selector Step 1)
    trail_stitch_enable: bool = True
    trail_stitch_gap_frames: int = 30
    trail_stitch_min_stale_frames: int = 3
    trail_stitch_min_obs: int = 3
    trail_stitch_dist_frac: float = 0.06
    trail_stitch_max_step_frac: float = 0.11
    trail_stitch_pred_resid_frac: float = 0.08
    trail_jump_break_frac: float = 0.06
    trail_jump_break_growth_frac: float = 0.020
    # Scoring weights
    w_len: float = 1.0          # reward per observation (linear)
    w_gaps: float = 0.5         # penalty per gap segment
    w_miss: float = 0.1         # penalty per missing frame within span
    w_jump: float = 0.02        # penalty per px of sudden jump
    w_outside: float = 200.0    # BIG penalty for outside-court fraction
    w_inside: float = 200.0     # BIG reward for inside-court fraction
    w_redflag_outside_long: float = 420.0  # hard penalty: many obs with near-zero inside-court
    w_dom: float = 15.0         # reward for dominance
    w_motion_consistency: float = 10.0   # reward for smooth motion
    w_conf: float = 2.0         # reward for avg confidence
    w_motion_mask: float = 30.0  # reward for motion-mask overlap fraction
    w_speed: float = 90.0       # reward for plausible average speed
    w_peak_speed: float = 35.0  # reward for occasional fast motion segments
    w_slow: float = 40.0        # penalty for near-static tracks
    w_extent: float = 80.0      # reward for covering meaningful court extent
    w_movement: float = 55.0    # composite movement score from speed/peak/extent/smoothness
    w_player_prox: float = 18.0 # reward for being near players (serve/hit context)
    w_player_court_joint: float = 95.0  # strong reward: near player + near court together
    w_static_cluster: float = 220.0  # strong penalty for static clustered tracks
    w_coverage: float = 120.0   # reward for covering larger portion of video
    w_span: float = 70.0        # reward for larger temporal span
    w_short_track: float = 260.0  # penalty for very short tracks in long videos
    movement_merge_min_score: float = 10.0  # merge non-blacklisted tracks above this movement score
    movement_merge_min_avg_speed: float = 2.2  # 30fps-ref px/frame required for moving-merge eligibility
    movement_merge_min_peak_speed: float = 6.0  # 30fps-ref px/frame fallback moving-merge eligibility
    movement_merge_min_extent_frac: float = 0.06  # minimum spatial extent as frame diagonal fraction
    enable_global_moving_merge: bool = False  # keep tracks separate by default; only merge strict fragments
    period_split_gap_frac: float = 0.04  # gap as video fraction to start a new temporal period id
    enable_timeline_stitch: bool = True  # auto-pick best sequence of tracks over time
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
    timeline_period_switch_penalty: float = 12.0  # small extra cost for switching temporal periods
    timeline_tail_penalty: float = 0.0  # disabled
    timeline_head_penalty: float = 0.0  # disabled
    redflag_outside_min_obs: int = 80
    redflag_outside_max_inside_frac: float = 0.02
    min_track_obs_frac: float = 0.04
    min_track_obs_abs: int = 20

    # Hard stationary-track exclusion
    stationary_exclude_min_obs: int = 8 #changed from 8
    stationary_exclude_max_avg_speed: float = 1.6   # px/frame
    stationary_exclude_max_peak_speed: float = 3.6  # px/frame
    stationary_exclude_max_extent_frac: float = 0.055
    stationary_exclude_max_motion_frac: float = 0.18
    blocked_det_radius_px: float = 18.0

    # Context gating (camera-angle robust)
    far_player_reject_frac: float = 0.28      # dist to nearest player as frame diag frac
    toss_player_allow_frac: float = 0.16      # outside-court allowed only if near player
    outside_reject_expand_mult: float = 1.15  # stricter court-distance cutoff

    # Court margin
    court_expand_px: float = 0.0     # auto-set: soft margin around court polygon
    side_margin_frac: float = 0.12   # extreme side zone fraction

    # Interpolation
    interp_max_gap: int = 0          # auto-set

    # Motion trail bonus
    boost_mask_bonus: float = 1.5    # extra score when detection overlaps motion mask

    # Per-frame locking
    guide_lock_frac: float = 0.022   # lock to exact-guide det within this * frame diag
    carry_interp_frames: int = 5     # keep predicted position this many no-det frames
    guide_gate_exact_frac: float = 0.09  # strict detection gate when guide is exact at frame t
    guide_gate_soft_frac: float = 0.14   # wider detection gate for interpolated guide frames
    guide_spike_max_step_frac: float = 0.15  # guide outlier filter: max local speed (diag frac/frame)
    guide_spike_resid_frac: float = 0.08  # guide outlier filter: local residual threshold (diag frac)
    guide_spike_max_neighbor_gap: int = 6  # guide outlier filter: only test short local neighborhoods
    guide_spawn_static_speed_frac: float = 0.010  # jump-to-static guard: local speed threshold (diag frac/frame)
    guide_spawn_max_motion_frac: float = 0.20  # jump-to-static guard: max motion-mask support in local window
    guide_spawn_lookahead_obs: int = 3  # jump-to-static guard: lookahead observations
    det_hard_step_frac: float = 0.065  # hard reject per-frame det jumps this large (diag frac/frame)
    det_hard_step_growth_per_frame: float = 0.12  # grow hard det-jump budget with multi-frame gaps

    # -- Motion fallback physics gate ------------------------------
    motion_use_min_det_speed: float = 1.5   # min speed (px/frame) to trust motion direction
    motion_pred_residual_frac: float = 0.082 # allow more drift from prediction for far falling/reacquired balls
    motion_speed_ratio_min: float = 0.25    # min ratio (motion_speed / track_speed)
    motion_speed_ratio_max: float = 3.20    # tolerate stronger speed mismatch when prediction lags real drop
    motion_dir_cos_min: float = -1.0       # ALLOW bounces and racket hits
    motion_accel_frac: float = 0.075        # allow sharper one-frame motion changes before rejecting
    motion_guided_soft_resid_frac: float = 0.025
    motion_guided_blob_weight_hi: float = 0.85
    motion_guided_blob_weight_lo: float = 0.85
    motion_guided_step_ratio: float = 2.00
    motion_guided_min_step_px: float = 8.0
    motion_max_gap_frames: int = 14
    # Blob search radius, read by core._motion_search_radius.  A flat radius is
    # wrong in both directions: too wide for a slow ball beside a player, too
    # narrow for a fast one whose predicted point is off by a few percent of a
    # large step.  So it grows with how far the prediction has run and with how
    # fast the ball is moving.  These are deliberately tighter than the values
    # this block shipped with (never wired, never measured); widen only with
    # evaluate_archive.py numbers in hand.
    # Measured: a 22 px base with a 132 px cap put three motion frames on a
    # player, a line and the background on the UCSD clip, for +0.002 recall.
    # The shape is right, the scale was not -- keep the base near the flat
    # radius this replaced and let speed and elapsed prediction earn the rest.
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
    gravity_px_per_frame2: float = 0.0   # auto-set: gravity accel in px/frame^2
    gravity_base_ppf2_30fps: float = 0.55  # gravity at 30fps in px/frame^2 (tunable)
    gravity_enabled: bool = True         # enable parabolic prediction
    gravity_vel_history: int = 4         # frames of velocity history for gravity estimation
    gravity_drag_factor: float = 0.985   # per-frame horizontal drag (1.0 = no drag)
    gravity_adapt_enabled: bool = True   # adapt gravity from observed detection acceleration
    gravity_adapt_alpha: float = 0.12    # EMA factor for adaptive gravity updates
    gravity_adapt_min_mult: float = 0.45 # lower clamp vs base gravity
    gravity_adapt_max_mult: float = 2.40 # upper clamp vs base gravity
    gravity_depth_gain: float = 0.22     # perspective gain: bottom of frame gets stronger image gravity
    physics_vel_alpha_x: float = 0.45    # smoothing for horizontal detection velocity updates
    physics_vel_alpha_y: float = 0.72    # higher alpha to react faster to vertical drop changes
    bounce_detect_min_down_speed: float = 4.0  # 30fps-ref px/frame
    bounce_detect_min_up_speed: float = 1.5    # 30fps-ref px/frame
    bounce_restitution: float = 0.58      # reflected vy fraction on bounce
    bounce_tangent_damping: float = 0.94  # horizontal damping at bounce

    # -- Track eligibility (_selected_tracks) ----------------------
    # Seconds of observations a track needs before it may be selected at all.
    select_min_obs_sec: float = 0.25
    select_reacquire_obs_sec: float = 0.16
    select_min_obs_frames: int = 8         # auto-set from select_min_obs_sec (30 fps value)
    select_reacquire_obs_frames: int = 10  # auto-set from select_reacquire_obs_sec
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
    # alone can contribute (physics._kinematic_motion_frac), so this floor
    # demands real speed, not merely a wide footprint.
    select_kinematic_motion_frac: float = 0.75
    select_kinematic_extent_frac: float = 0.05

    # -- Reacquire gate (after loss) -- IMPROVED for better ball reacquisition --
    reacquire_gate_frames: int = 6          # INCREASED from 4 for longer reacquisition window
    reacquire_dist_frac: float = 0.075      # INCREASED from 0.055 for wider search
    reacquire_dist_growth_per_frame: float = 0.40  # INCREASED from 0.30 for faster expansion
    reacquire_area_ratio_min: float = 0.18  # LOWERED from 0.22 for size variation tolerance
    reacquire_area_ratio_max: float = 5.50  # INCREASED from 4.80 for close-up balls

    def auto_scale(self):
        """Set resolution/fps-dependent parameters."""
        self.diag = math.sqrt(self.width ** 2 + self.height ** 2)
        diag = self.diag
        fps_now = max(float(self.fps), 1.0)
        fps_scale = 30.0 / fps_now
        fps_scale = max(0.35, min(3.0, fps_scale))
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
        self.max_speed_px_per_frame = 0.16 * diag * fps_scale  # tighter speed ceiling normalized across FPS
        self.court_expand_px = 0.12 * diag
        self.interp_max_gap = _frames_for_seconds(4.0 / 30.0, min_frames=2, max_frames=16)
        self.motion_max_gap_frames = _frames_for_seconds(45.0 / 30.0, min_frames=3, max_frames=90)
        self.carry_interp_frames = _frames_for_seconds(8.0 / 30.0, min_frames=3, max_frames=24)
        self.reacquire_gate_frames = _frames_for_seconds(4.0 / 30.0, min_frames=2, max_frames=16)
        self.select_min_obs_frames = _frames_for_seconds(self.select_min_obs_sec, min_frames=3)
        self.select_reacquire_obs_frames = _frames_for_seconds(self.select_reacquire_obs_sec, min_frames=10)
        self.guide_spike_max_neighbor_gap = _frames_for_seconds(6.0 / 30.0, min_frames=2, max_frames=24)
        self.trail_stitch_gap_frames = _frames_for_seconds(30.0 / 30.0, min_frames=6, max_frames=90)
        self.trail_stitch_min_stale_frames = _frames_for_seconds(3.0 / 30.0, min_frames=1, max_frames=12)
        # Gravity: scale from 30fps reference
        res_scale = diag / 2203.0  # 2203 = diag of 1920x1080
        self.gravity_px_per_frame2 = (
            self.gravity_base_ppf2_30fps * res_scale * (fps_scale ** 2)
        )
        return self
