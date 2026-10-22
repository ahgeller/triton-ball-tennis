import math
import cv2
import numpy as np
from typing import Dict, List, NamedTuple, Optional, Tuple

from .config import SelectorConfig
from .models import Detection, Track
from .utils import PlayerBoxesByFrame
from .physics import kinematic_motion_frac, xy_dist

# Timeline handover: below ~1.5% of the diagonal per frame a track's velocity
# is too uncertain to compare speeds across a handover.
_MEANINGFUL_SPEED_DIAG_FRAC = 0.015
# Handover speed ratios outside this band mean two different balls.
_HANDOVER_SPEED_RATIO_MIN = 0.15
_HANDOVER_SPEED_RATIO_MAX = 3.5
# Handover distances never fall below 12 px, whatever the resolution.
_HANDOVER_MIN_DIST_PX = 12.0
# Overlapping tracks are the same ball when their median distance on shared
# frames is within 8% of the diagonal.
_SHARED_FRAME_DIST_DIAG_FRAC = 0.08
# Across a medium gap a jump may exceed the small-gap limit by this much.
_MEDIUM_GAP_EXTRA_JUMP_FRAC = 0.08
# A handover jump beyond 70% of the diagonal is a different ball at any gap.
_HANDOVER_MAX_JUMP_FRAC = 0.70
_HARD_SWITCH_PENALTY_MULT = 6.0
_JUMP_PENALTY_FRAC_CAP = 1.5
_MEANINGFUL_SPEED_MIN_PX = 2.0
_SPEED_MISMATCH_PENALTY_MULT = 0.5
# A successor spanning under max(6 frames, 0.5% of the clip) with fewer than
# ten observations is a tiny bridge and pays 0.4 switch penalties.
_TINY_BRIDGE_MIN_SPAN_FRAMES = 6
_TINY_BRIDGE_SPAN_FRAC = 0.005
_TINY_BRIDGE_MAX_OBS = 10
_TINY_BRIDGE_PENALTY_MULT = 0.4
# A skipped track is continuous with the chain when the chain observation
# nearest in time to its first observation (within 0.1 s, at least two
# frames) is within the handover distance; see _add_disjoint_tracks.
_HANDOVER_WINDOW_SEC = 0.1
_HANDOVER_MIN_FRAMES = 2
# A turn sharper than 0.6 pi (108 deg) between consecutive velocities is jitter.
_SHARP_TURN_PI_FRAC = 0.6
# A step more than 2.5x the median step is a jump.
_JUMP_MEDIAN_SPEED_MULT = 2.5
# score_tracks: player boxes are dilated by this much (px) for the
# near-player fraction, which only weakly multiplies the score.
_NEAR_PLAYER_MARGIN_PX = 60
_NEAR_PLAYER_MULT_FLOOR, _NEAR_PLAYER_MULT_RANGE = 0.70, 0.30


def _track_kinematics(trk: Track) -> Tuple[List[Tuple[float, float]], List[float], float, float]:
    """Return cached (velocities, speeds, avg_speed, peak_speed) for a track."""
    obs = trk.observations
    if len(obs) < 2:
        trk._kin_cache_key = (len(obs),)
        trk._cached_velocities = []
        trk._cached_speeds = []
        trk._cached_speed_stats = (0.0, 0.0)
        return trk._cached_velocities, trk._cached_speeds, 0.0, 0.0

    key = (
        len(obs),
        int(obs[0].frame),
        int(obs[-1].frame),
        float(obs[-1].cx),
        float(obs[-1].cy),
    )
    if trk._kin_cache_key == key:
        avg_speed, peak_speed = trk._cached_speed_stats
        return trk._cached_velocities, trk._cached_speeds, avg_speed, peak_speed

    velocities: List[Tuple[float, float]] = []
    speeds: List[float] = []
    total_dist = 0.0
    total_dt = 0
    peak = 0.0
    for i in range(1, len(obs)):
        dt = int(obs[i].frame - obs[i - 1].frame)
        if dt <= 0:
            continue
        dx = float(obs[i].cx - obs[i - 1].cx)
        dy = float(obs[i].cy - obs[i - 1].cy)
        vx = dx / dt
        vy = dy / dt
        dist = math.sqrt(dx * dx + dy * dy)
        spd = dist / dt
        velocities.append((vx, vy))
        speeds.append(spd)
        total_dist += dist
        total_dt += dt
        if spd > peak:
            peak = spd

    avg = total_dist / max(total_dt, 1)
    trk._kin_cache_key = key
    trk._cached_velocities = velocities
    trk._cached_speeds = speeds
    trk._cached_speed_stats = (avg, peak)
    return velocities, speeds, avg, peak

def _track_speed_stats(trk: Track) -> Tuple[float, float]:
    """Returns (avg_speed, peak_speed) in px/frame."""
    _, _, avg, peak = _track_kinematics(trk)
    return avg, peak

def _track_extent(trk: Track) -> float:
    """Spatial extent (diag of bbox containing all obs centers)."""
    if trk.num_obs < 2:
        return 0.0
    xs = [o.cx for o in trk.observations]
    ys = [o.cy for o in trk.observations]
    return math.sqrt((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2)


def is_near_player(
    x: float,
    y: float,
    player_boxes_by_frame: Optional[PlayerBoxesByFrame],
    frame_index: int,
    margin: float = 80,
) -> bool:
    """Check if (x,y) is near any player bbox at given frame.

    ``player_boxes_by_frame[frame_index]`` is a dict (slot -> xyxy) or a list of
    xyxy boxes; each box is dilated by ``margin`` and by twice its height
    upward (serve toss above the head).
    """
    if player_boxes_by_frame is None or frame_index >= len(player_boxes_by_frame):
        return False
    pboxes = player_boxes_by_frame[frame_index]
    if pboxes is None:
        return False
    # pboxes can be a dict (slot->bbox) or list of bboxes
    boxes = pboxes.values() if isinstance(pboxes, dict) else pboxes
    for pb in boxes:
        if pb is None or len(pb) < 4:
            continue
        # Expand player bbox upward generously (serve toss above head)
        px1, py1, px2, py2 = float(pb[0]), float(pb[1]), float(pb[2]), float(pb[3])
        ph = py2 - py1
        if (px1 - margin <= x <= px2 + margin and
                py1 - ph * 2 - margin <= y <= py2 + margin):
            return True
    return False


class _TimelineLimits(NamedTuple):
    """The gap and distance limits select_timeline_chain derives once per call."""
    diag: float
    tden: int            # total frames, at least 1
    max_gap: int         # frames a handover may skip
    overlap_tol: int     # frames two chained tracks may overlap
    small_gap: int
    medium_gap: int
    jump_reject: float   # handover jump (diagonal fraction) rejected across a small gap


def _timeline_limits(cfg: SelectorConfig, total_frames: int) -> _TimelineLimits:
    tden = max(total_frames, 1)
    small_gap = int(round(cfg.timeline_small_gap_frac * tden))
    return _TimelineLimits(
        diag=cfg.diag,
        tden=tden,
        max_gap=max(1, int(round(cfg.timeline_max_gap_frac * tden))),
        overlap_tol=int(round(cfg.timeline_overlap_tol_frac * tden)),
        small_gap=small_gap,
        medium_gap=max(small_gap + 1, int(round(small_gap * 2.0))),
        jump_reject=cfg.timeline_small_gap_jump_reject_frac,
    )


def _handover_transition(ti: Track, tj: Track, cfg: SelectorConfig, lim: _TimelineLimits) -> Optional[float]:
    """Score for chaining ``tj`` after ``ti`` (a penalty, <= 0), or None when it may not follow."""
    sj = int(tj.first_frame)
    j_start = tj.observations[0]
    si = int(ti.first_frame)
    ei = int(ti.last_obs_frame)
    if sj <= si:
        return None

    gap = sj - ei - 1
    if gap > lim.max_gap:
        return None
    if gap < -lim.overlap_tol:
        return None

    diag = lim.diag
    tden = lim.tden
    i_end = ti.observations[-1]
    dt = max(gap + 1, 1)
    if gap < 0:
        # The tracks interleave.  Judge the handover where tj actually
        # starts, from ti's last observation before that frame: ti's
        # span end may be one stray detection grabbed long after the
        # ball moved on, and that must not veto a good successor.
        before = [obs for obs in ti.observations if obs.frame < sj]
        if before:
            i_end = before[-1]
            dt = max(sj - i_end.frame, 1)
    shared_frames = set(obs.frame for obs in ti.observations) & set(
        obs.frame for obs in tj.observations
    )
    if gap < 0 and shared_frames:
        ti_by_frame = {obs.frame: obs for obs in ti.observations}
        tj_by_frame = {obs.frame: obs for obs in tj.observations}
        shared_distances = sorted(
            xy_dist(
                ti_by_frame[frame].cx,
                ti_by_frame[frame].cy,
                tj_by_frame[frame].cx,
                tj_by_frame[frame].cy,
            )
            for frame in shared_frames
        )
        d = shared_distances[len(shared_distances) // 2]
        if d > max(_HANDOVER_MIN_DIST_PX, _SHARED_FRAME_DIST_DIAG_FRAC * diag):
            return None
    else:
        pred_x = i_end.cx + ti.last_vel[0] * dt
        pred_y = i_end.cy + ti.last_vel[1] * dt
        d_raw = xy_dist(i_end.cx, i_end.cy, j_start.cx, j_start.cy)
        d_pred = xy_dist(pred_x, pred_y, j_start.cx, j_start.cy)
        d = min(d_raw, d_pred)
    dfrac = d / max(diag, 1.0)

    hard_discontinuity = (
        (gap <= lim.small_gap and dfrac > lim.jump_reject)
        or (gap <= lim.medium_gap and dfrac > (lim.jump_reject + _MEDIUM_GAP_EXTRA_JUMP_FRAC))
        or dfrac > _HANDOVER_MAX_JUMP_FRAC
    )
    # Simultaneous distant tracks are alternatives. Sequential distant
    # tracks are separate rallies/clips: keep both, but pay enough that
    # short noise cannot create a reset.
    if hard_discontinuity and gap < 0:
        return None

    if hard_discontinuity:
        trans = -_HARD_SWITCH_PENALTY_MULT * cfg.timeline_switch_penalty
    else:
        trans = -cfg.timeline_switch_penalty
        if gap < 0:
            trans -= cfg.timeline_overlap_penalty * (abs(gap) / float(tden))
        trans -= cfg.timeline_jump_penalty * min(dfrac, _JUMP_PENALTY_FRAC_CAP)
    if gap > 0:
        trans -= cfg.timeline_gap_penalty * (gap / float(tden))

    # -- Speed-continuity penalty --
    # A handover whose implied speed differs wildly from track i's
    # outgoing speed suggests two different balls.  Direction is not
    # judged: racket hits and bounces reverse it legitimately.
    exp_vx, exp_vy = ti.last_vel
    exp_speed = math.sqrt(exp_vx * exp_vx + exp_vy * exp_vy)
    obs_vx = (j_start.cx - i_end.cx) / max(dt, 1)
    obs_vy = (j_start.cy - i_end.cy) / max(dt, 1)
    obs_speed = math.sqrt(obs_vx * obs_vx + obs_vy * obs_vy)
    meaningful_speed = max(diag * _MEANINGFUL_SPEED_DIAG_FRAC, _MEANINGFUL_SPEED_MIN_PX)
    if (
        not hard_discontinuity
        and gap >= 0
        and exp_speed > meaningful_speed
        and obs_speed > meaningful_speed
    ):
        speed_ratio = obs_speed / max(exp_speed, 1e-6)
        if speed_ratio > _HANDOVER_SPEED_RATIO_MAX or speed_ratio < _HANDOVER_SPEED_RATIO_MIN:
            trans -= cfg.timeline_jump_penalty * _SPEED_MISMATCH_PENALTY_MULT

    # -- Tiny-bridge penalty --
    # A track with very few observations spanning a tiny window that creates
    # a large dead zone after it (the next viable track is far away) is likely
    # noise. Apply a small penalty proportional to how tiny the bridge is
    # relative to the gap it opens up on the far side.
    tj_span = max(int(tj.last_obs_frame) - int(tj.first_frame) + 1, 1)
    tiny_span = max(_TINY_BRIDGE_MIN_SPAN_FRAMES, int(round(_TINY_BRIDGE_SPAN_FRAC * tden)))
    if tj_span < tiny_span and tj.num_obs < _TINY_BRIDGE_MAX_OBS:
        trans -= cfg.timeline_switch_penalty * _TINY_BRIDGE_PENALTY_MULT
    return trans


def _add_disjoint_tracks(
    chain: List[Track], ordered: List[Track], cfg: SelectorConfig, lim: _TimelineLimits
) -> List[Track]:
    """Add back tracks the continuity chain skipped that are the only evidence for their frames.

    Continuity chooses between competing overlapping tracks. A validated,
    disjoint segment can be a separate rally and must not disappear merely
    because its endpoints do not connect to the surrounding rallies.

    Disjoint means the candidate observes frames the chain does not.  A
    track that lives inside another's span (the ball switched tracks at a
    bounce and switched back) is still the only evidence for its frames,
    so it is kept when its boundary is continuous with the chain.
    """
    chain = list(chain)
    chosen_ids = {id(track) for track in chain}
    chain_obs: Dict[int, Detection] = {}
    for track in chain:
        for obs in track.observations:
            chain_obs.setdefault(obs.frame, obs)
    handover_frames = max(_HANDOVER_MIN_FRAMES, int(round(cfg.fps * _HANDOVER_WINDOW_SEC)))
    handover_dist = max(_HANDOVER_MIN_DIST_PX, lim.jump_reject * lim.diag)

    def continuous(candidate: Track) -> bool:
        """Whether the candidate's boundary meets the chain.

        The chain observation nearest in time to the candidate's first
        observation decides (offsets grow one frame at a time, the earlier
        frame tried first): continuous when it lies within ``handover_dist``.
        The last observation is only consulted when no chain observation falls
        within ``handover_frames`` of the first one, so a far chain point next
        to the start vetoes a candidate whose end connects cleanly (pinned by
        test_disjoint_continuity_is_decided_at_the_start_edge).
        """
        for edge in (candidate.observations[0], candidate.observations[-1]):
            for offset in range(1, handover_frames + 1):
                for frame in (edge.frame - offset, edge.frame + offset):
                    other = chain_obs.get(frame)
                    if other is not None:
                        return xy_dist(edge.cx, edge.cy, other.cx, other.cy) <= handover_dist
        return False

    for candidate in sorted(ordered, key=lambda track: track.score, reverse=True):
        if id(candidate) in chosen_ids:
            continue
        span_disjoint = all(
            candidate.last_obs_frame < track.first_frame
            or candidate.first_frame > track.last_obs_frame
            for track in chain
        )
        shared = sum(obs.frame in chain_obs for obs in candidate.observations)
        if span_disjoint or (shared <= lim.overlap_tol and continuous(candidate)):
            chain.append(candidate)
            chosen_ids.add(id(candidate))
            for obs in candidate.observations:
                chain_obs.setdefault(obs.frame, obs)
    return chain


def select_timeline_chain(
    tracks: List[Track],
    cfg: SelectorConfig,
    total_frames: int
) -> List[Track]:
    """Pick an auto timeline of tracks maximizing score with time-progression continuity."""
    if not tracks:
        return []

    min_score = cfg.timeline_min_track_score
    eligible = [t for t in tracks if t.score >= min_score]
    if not eligible:
        return []
    if len(eligible) == 1:
        return [eligible[0]]

    lim = _timeline_limits(cfg, total_frames)
    ordered = sorted(eligible, key=lambda t: (t.first_frame, t.last_obs_frame, -t.score))
    n = len(ordered)
    cov_w = cfg.timeline_coverage_weight
    score_w = cfg.timeline_score_weight
    base = []
    for t in ordered:
        span_frames = max(int(t.last_obs_frame) - int(t.first_frame) + 1, 1)
        span_frac = span_frames / float(lim.tden)
        base_score = max(0.0, t.score)
        base.append(cov_w * span_frac + score_w * base_score)
    dp = base[:]
    prev = [-1] * n

    for j in range(n):
        for i in range(j):
            trans = _handover_transition(ordered[i], ordered[j], cfg, lim)
            if trans is None:
                continue
            cand = dp[i] + base[j] + trans
            if cand > dp[j]:
                dp[j] = cand
                prev[j] = i

    end_idx = max(range(n), key=lambda k: dp[k])
    idx_chain = []
    k = end_idx
    while k >= 0:
        idx_chain.append(k)
        k = prev[k]
    idx_chain.reverse()

    chain = _add_disjoint_tracks([ordered[k] for k in idx_chain], ordered, cfg, lim)
    return sorted(chain, key=lambda track: (track.first_frame, track.last_obs_frame))


def stitch_track_chain(chain: List[Track], cfg: SelectorConfig) -> Optional[Track]:
    """Merge a timeline chain into one track: per frame, the most confident observation."""
    if not chain:
        return None
    if len(chain) == 1:
        return chain[0]

    best_obs_by_frame: Dict[int, Detection] = {}
    for trk in chain:
        for obs in trk.observations:
            prev = best_obs_by_frame.get(obs.frame)
            if prev is None or obs.conf > prev.conf:
                best_obs_by_frame[obs.frame] = obs

    obs = [best_obs_by_frame[f] for f in sorted(best_obs_by_frame.keys())]
    if len(obs) < 2:
        return chain[0]

    # The chain's own record: it begins as its first track did and ends as its last did, and
    # the timeline used it by construction. ``selection`` stays empty: the chain was never judged.
    stitched = Track(track_id=chain[0].track_id, cfg=cfg,
                     start_reason=chain[0].start_reason, start_rival=chain[0].start_rival,
                     end_reason=chain[-1].end_reason, in_timeline=True)
    stitched.observations = obs
    stitched.last_pos = (obs[-1].cx, obs[-1].cy)
    prev = obs[-2]
    dt = max(obs[-1].frame - prev.frame, 1)
    stitched.last_vel = ((obs[-1].cx - prev.cx) / dt, (obs[-1].cy - prev.cy) / dt)
    stitched.score = sum(float(t.score) for t in chain)
    stitched.score_breakdown = {
        "total": stitched.score,
        "timeline_tracks": float(len(chain)),
        "start_frame": float(chain[0].first_frame),
        "end_frame": float(chain[-1].last_obs_frame),
        "start_frac": chain[0].score_value("start_frac"),
        "end_frac": chain[-1].score_value("end_frac"),
    }
    return stitched

def _court_fractions(
    track: Track, court_poly: Optional[np.ndarray], expand_px: float
) -> Tuple[float, float, float]:
    """Compute court occupancy fractions.

    Returns:
      near_court_frac: inside polygon or within expand_px margin (used for scoring).
      far_outside_frac: farther than 2x expand_px outside polygon.
      strict_inside_frac: strictly inside polygon (dist >= 0).

    Without a court polygon nothing is known about where the ball is, so the
    scoring fraction is 1: the court term then neither ranks tracks (every
    track in the clip shares it) nor pushes them under the absolute score
    density gates in ``core.selected_tracks`` and the timeline, which a
    "neutral" 0.5 did on every court-less clip. The band fractions stay 0.5,
    between the "mostly outside" and "mostly inside" bands the selector reads.
    """
    if len(track.observations) == 0:
        return 0.5, 0.5, 0.5
    if court_poly is None:
        return 1.0, 0.5, 0.5

    near_court = 0
    far_outside = 0
    strict_inside = 0
    for obs in track.observations:
        dist = cv2.pointPolygonTest(court_poly, (float(obs.cx), float(obs.cy)), True)
        # dist > 0 = inside polygon, dist < 0 = outside, magnitude = distance to edge
        if dist >= 0.0:
            strict_inside += 1
        if dist >= -expand_px:  # inside or within margin
            near_court += 1
        elif -dist > expand_px * 2:  # really far
            far_outside += 1

    n = len(track.observations)
    return near_court / n, far_outside / n, strict_inside / n

def _near_player_fractions(
    track: Track,
    player_boxes_by_frame: Optional[PlayerBoxesByFrame],
    court_poly: Optional[np.ndarray],
    expand_px: float,
) -> Tuple[float, float]:
    """Fractions of the observations beside a player, and beside a player while
    near the court (every near-player observation when there is no court)."""
    if player_boxes_by_frame is None or not track.observations:
        return 0.0, 0.0
    near_player_count = 0
    near_player_court_count = 0
    for obs in track.observations:
        if not is_near_player(obs.cx, obs.cy, player_boxes_by_frame, obs.frame, margin=_NEAR_PLAYER_MARGIN_PX):
            continue
        near_player_count += 1
        if court_poly is None:
            near_player_court_count += 1
        elif cv2.pointPolygonTest(court_poly, (float(obs.cx), float(obs.cy)), True) >= -expand_px:
            near_player_court_count += 1
    n = len(track.observations)
    return near_player_count / n, near_player_court_count / n


def _motion_consistency(track: Track) -> float:
    """Score 0-1 for how smooth/consistent the track motion is.
    High jitter = low score. Smooth arcs = high score."""
    obs = track.observations
    if len(obs) < 3:
        return 0.5

    velocities, _, _, _ = _track_kinematics(track)

    if len(velocities) < 2:
        return 0.5

    # Measure direction changes (angle between consecutive velocity vectors)
    angle_changes = []
    for i in range(1, len(velocities)):
        v1 = velocities[i-1]
        v2 = velocities[i]
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
        if mag1 < 0.5 or mag2 < 0.5:
            continue  # near-stationary, skip
        cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2 + 1e-9)))
        angle_changes.append(abs(math.acos(cos_angle)))

    if not angle_changes:
        return 0.5

    # Penalize frequent sharp direction changes (> 108 deg, i.e. 0.6 pi).
    # Tennis ball has smooth arcs with direction changes at bounces/hits
    # But random jitter has constant tiny direction changes
    sharp = sum(1 for a in angle_changes if a > math.pi * _SHARP_TURN_PI_FRAC)
    smooth_frac = 1.0 - sharp / len(angle_changes)
    return max(0.0, min(1.0, smooth_frac))

def _jump_penalty(track: Track) -> float:
    """Sum of unusually large jumps between consecutive observations."""
    obs = track.observations
    if len(obs) < 2:
        return 0.0
    penalty = 0.0
    _, speeds, _, _ = _track_kinematics(track)

    if not speeds:
        return 0.0

    # Penalize jumps above 2.5x the median speed (never below 5 px/frame)
    median_speed = sorted(speeds)[len(speeds) // 2]
    threshold = max(median_speed * _JUMP_MEDIAN_SPEED_MULT, 5.0)
    for s in speeds:
        if s > threshold:
            penalty += (s - threshold)
    return penalty

def score_tracks(
    tracks: List[Track],
    cfg: SelectorConfig,
    court_poly: Optional[np.ndarray] = None,
    player_boxes_by_frame: Optional[PlayerBoxesByFrame] = None,
    total_frames: int = 0
) -> List[Track]:
    """Score each track, sort by score descending.

    Positive-only scoring:
      base_add = obs_add + span_add
      score = base_add * inside_mult * motion_mult * near_player_mult

    with:
      - obs_add = w_len * observations
      - span_add = w_span * (span / total_frames)
      - inside_mult = near-court fraction: inside the polygon or within
        court_expand_px of it (strict inside is recorded but not multiplied,
        so near-edge detections do not destroy a track's score); 1 when the
        clip has no court polygon, so a missing court is not evidence
      - motion_mult = motion evidence: max(mask overlap fraction,
        physics.kinematic_motion_frac estimate)
      - near_player_mult = 0.70 + 0.30 * near-player fraction (weaker term)

    Every term lands in Track.score_breakdown under the keys documented in
    models.SCORE_BREAKDOWN_DEFAULTS.
    """
    if not tracks:
        return tracks

    last_index = max(total_frames - 1, 1)
    for trk in tracks:
        obs = trk.observations
        n = trk.num_obs
        span = trk.span

        coverage_frac = n / total_frames if total_frames > 0 else 0.0
        span_frac = span / total_frames if total_frames > 0 else 0.0

        inside_frac, _, strict_inside_frac = _court_fractions(
            trk, court_poly, cfg.court_expand_px
        )
        motion_overlap = sum(1 for o in obs if o.on_motion) / max(n, 1)
        near_player_frac, near_player_court_frac = _near_player_fractions(
            trk, player_boxes_by_frame, court_poly, cfg.court_expand_px
        )

        avg_speed, peak_speed = _track_speed_stats(trk)
        extent = _track_extent(trk)
        jump_penalty_raw = _jump_penalty(trk)
        motion_smooth = _motion_consistency(trk)
        motion_kin_frac = kinematic_motion_frac(
            avg_speed, peak_speed, extent, near_player_frac, cfg, cfg.diag
        )
        # Motion evidence: the mask overlap, or the kinematic estimate when the
        # mask misses a small, distant or slow ball.
        motion_evidence = max(motion_overlap, motion_kin_frac)

        # Positive additive terms requested by user.
        obs_add = cfg.w_len * n
        span_add = cfg.w_span * span_frac
        base_add = obs_add + span_add

        # Multipliers, each a fraction in [0, 1], with near-player intentionally
        # weaker.  The court term is the near-court fraction (inside OR within
        # the margin), not strict_inside_frac, so near-edge detections don't
        # destroy track scores.
        near_player_mult = _NEAR_PLAYER_MULT_FLOOR + _NEAR_PLAYER_MULT_RANGE * near_player_frac

        # Non-negative by construction: fractions times a positive sum.
        total_score = base_add * inside_frac * motion_evidence * near_player_mult

        trk.score = total_score
        trk.score_breakdown = {
            "total": float(total_score),
            "num_obs": n,
            "span": span,
            "inside_frac": float(inside_frac),
            "inside_strict_frac": float(strict_inside_frac),
            "motion_evidence": float(motion_evidence),
            "motion_overlap": float(motion_overlap),
            "motion_kin_frac": float(motion_kin_frac),
            "avg_speed_pxpf": float(avg_speed),
            "peak_speed_pxpf": float(peak_speed),
            "extent_px": float(extent),
            "jump_penalty": float(jump_penalty_raw),
            "near_player_frac": float(near_player_frac),
            "near_player_court_frac": float(near_player_court_frac),
            "coverage_frac": float(coverage_frac),
            "span_frac": float(span_frac),
            "start_frame": float(trk.first_frame),
            "end_frame": float(trk.last_obs_frame),
            "start_frac": trk.first_frame / last_index if total_frames > 0 else 0.0,
            "end_frac": trk.last_obs_frame / last_index if total_frames > 0 else 0.0,
            "obs_add": float(obs_add),
            "span_add": float(span_add),
            "additive_core": float(base_add),
            "motion_mult": float(motion_evidence),
            "inside_mult": float(inside_frac),
            "near_player_mult": float(near_player_mult),
            "movement_quality": float(motion_smooth),
        }

    tracks.sort(key=lambda t: t.score, reverse=True)
    return tracks


