"""Ball-in-play selector scenarios on synthetic detections.

These were the selector half of ``clean_tracker.py --self-test``. Several
fixtures reproduce measured tracks from real clips (noted per test); keep their
numbers and expectations as they are: they are regressions, not tunable values.
"""
from typing import List, Tuple

import numpy as np

from ball_in_play_selector import Detection, FrameResult, SelectorConfig, Track, select_ball_in_play
from ball_in_play_selector.association import build_tracks
from ball_in_play_selector.core import (
    direction_cosine,
    refine_trajectory,
    selected_tracks,
    trajectory_observations,
)
from ball_in_play_selector.scoring import select_timeline_chain

FRAME_SHAPE = (360, 640)
WIDTH, HEIGHT = FRAME_SHAPE[1], FRAME_SHAPE[0]
HD_WIDTH, HD_HEIGHT = 1920, 1080


def _moving_ball(frames: int = 12) -> Tuple[list, List[np.ndarray]]:
    """A confident 4 px ball moving 12 px/frame along y=20, with a matching motion mask."""
    detections = []
    masks = []
    for frame in range(frames):
        x = 10 + 12 * frame
        detections.append([([x, 20, x + 4, 24], 0.9)])
        mask = np.zeros(FRAME_SHAPE, dtype=np.uint8)
        mask[20:25, x:x + 5] = 255
        masks.append(mask)
    return detections, masks


def _empty_mask() -> np.ndarray:
    return np.zeros(FRAME_SHAPE, dtype=np.uint8)


def _cfg(fps: float, width: int = WIDTH, height: int = HEIGHT) -> SelectorConfig:
    return SelectorConfig(fps=fps, width=width, height=height).auto_scale()


def _line_track(track_id, frames, x_offset, score, cfg):
    """A track moving 1 px/frame along y=100 starting at ``x_offset``."""
    track = Track(track_id=track_id, cfg=cfg)
    track.observations = [
        Detection(frame, x_offset + frame, 100.0, 0, 0, 4, 4, 0.9, 16.0, True)
        for frame in frames
    ]
    track.last_vel = (1.0, 0.0)
    track.score = score
    track.score_breakdown = {"period_id": 0.0}
    return track


def test_moving_ball_is_selected_on_every_frame():
    detections, masks = _moving_ball()
    results, chosen, tracks = select_ball_in_play(
        detections, 30.0, WIDTH, HEIGHT, boost_masks=masks, raw_motions=masks
    )
    assert chosen is not None and tracks
    assert sum(result is not None for result in results) == len(detections)


def test_missing_court_is_not_evidence_against_the_ball():
    # grid_match26 (no court detected): the rally's longest track, 248 detections
    # with 74% on the motion mask, reached a score density of 0.47 and fell under the
    # 0.5 gate only because a missing polygon multiplied every score by a "neutral"
    # 0.5. The same rally scores the same with no court as with a court around it.
    frames = 1000
    detections, masks = _moving_ball(50)
    detections += [[] for _ in range(frames - 50)]
    masks += [_empty_mask() for _ in range(frames - 50)]
    players = [{1: (600, 300, 640, 360)}] * frames          # a player far from the ball
    court = np.array([[[0, 0]], [[639, 0]], [[639, 359]], [[0, 359]]], dtype=np.int32)
    with_court = select_ball_in_play(
        detections, 30.0, WIDTH, HEIGHT, court_polygon=court,
        boost_masks=masks, raw_motions=masks, player_boxes_by_frame=players,
    )
    without_court = select_ball_in_play(
        detections, 30.0, WIDTH, HEIGHT,
        boost_masks=masks, raw_motions=masks, player_boxes_by_frame=players,
    )
    assert with_court[1] is not None and without_court[1] is not None
    assert [result is not None for result in with_court[0]] == [result is not None for result in without_court[0]]
    assert without_court[1].score_value("inside_mult") == 1.0
    assert without_court[1].score_value("inside_strict_frac") == 0.5


def test_first_internal_gap_is_filled_between_real_endpoints():
    # The first internal gap is safe to fill because both real endpoints exist.
    detections, masks = _moving_ball()
    first_gap_detections = detections[:1] + [[], []] + detections[3:]
    results, _, _ = select_ball_in_play(
        first_gap_detections, 30.0, WIDTH, HEIGHT,
        boost_masks=masks, raw_motions=masks,
    )
    assert results[1] is not None and results[2] is not None


def test_motion_backed_odd_frames_beside_a_player_are_kept_at_60fps():
    # At 60 FPS, a motion-backed frame between two sampled detections remains
    # usable beside a player; unobserved near-player gaps stay conservative.
    sampled_detections = []
    sampled_masks = []
    sampled_players = []
    for frame in range(40):
        x = 100 + 4 * frame
        sampled_detections.append(
            [([x, 100, x + 4, 104], 0.9)] if frame % 2 == 0 else []
        )
        mask = _empty_mask()
        mask[100:105, x:x + 5] = 255
        sampled_masks.append(mask)
        sampled_players.append({1: (0, 0, 640, 300)})
    results, _, _ = select_ball_in_play(
        sampled_detections,
        60.0,
        WIDTH,
        HEIGHT,
        boost_masks=sampled_masks,
        raw_motions=sampled_masks,
        player_boxes_by_frame=sampled_players,
    )
    assert all(results[frame] is not None for frame in range(1, 38, 2))
    assert all(results[frame].source == "motion" for frame in range(1, 38, 2))


def test_direction_cosine_of_straight_and_reversed_paths():
    straight = [
        Detection(frame, float(frame * 10), 20.0, 0, 0, 4, 4, 0.9, 16.0, True)
        for frame in range(3)
    ]
    assert direction_cosine(*straight) == 1.0
    straight[2].cx = 0.0
    assert direction_cosine(*straight) == -1.0


def test_refit_repairs_a_target_switch_and_relabels_the_moved_detections():
    # The robust refit repairs a short target-switch block. The repair itself is
    # unbounded, but a detection dragged that far is no longer where the detector
    # fired, so it stops being reported as one: green in the overlay and scored
    # against the detector both mean "raw observation".
    smooth = [
        FrameResult(cx=float(frame * 10), cy=100.0, conf=0.9, source="det")
        for frame in range(15)
    ]
    smooth[7].cy = smooth[8].cy = 180.0
    refine_trajectory(smooth, 734.0)
    assert max(abs(smooth[frame].cy - 100.0) for frame in (7, 8)) < 10.0
    assert all(smooth[frame].source == "interp" for frame in (7, 8))
    assert all(smooth[frame].source == "det" for frame in range(15) if frame not in (7, 8))


def test_subpixel_refit_does_not_relabel_detections():
    # Ordinary sub-pixel smoothing must not relabel anything.
    nudged = [
        FrameResult(cx=float(frame * 10), cy=100.0, conf=0.9, source="det")
        for frame in range(15)
    ]
    nudged[7].cy = 103.0
    refine_trajectory(nudged, 734.0)
    assert all(result.source == "det" for result in nudged)


def test_one_motion_hit_after_loss_supports_one_point_not_a_tail():
    # One broad motion hit may support one physics point, never a blue tail
    # across later frames where the ball has disappeared.
    detections, masks = _moving_ball()
    lost_detections = detections[:10] + [[] for _ in range(5)]
    lost_masks = masks[:10] + [_empty_mask() for _ in range(5)]
    lost_masks[10][20:25, 130:135] = 255
    results, _, _ = select_ball_in_play(
        lost_detections, 30.0, WIDTH, HEIGHT,
        boost_masks=lost_masks, raw_motions=lost_masks,
    )
    assert results[10] is not None and results[10].source == "motion"
    assert all(result is None for result in results[11:])


def test_weak_detections_alone_select_nothing():
    weak = [[([10 + frame, 20, 14 + frame, 24], 0.2)] for frame in range(4)]
    results, chosen, _ = select_ball_in_play(weak, 30.0, WIDTH, HEIGHT)
    assert chosen is None and all(result is None for result in results)


def test_isolated_weak_wall_hit_does_not_anchor_a_connector():
    # A lone weak/no-motion wall hit cannot anchor a visible connector.
    isolated = Track(track_id=99, cfg=_cfg(30.0))
    isolated.observations = [
        Detection(frame, float(frame), 20.0, 0, 0, 4, 4, conf, 16.0, on_motion)
        for frame, conf, on_motion in (
            (0, 0.9, True), (1, 0.9, True), (10, 0.2, False),
            (20, 0.9, True), (21, 0.9, True),
        )
    ]
    assert [item.frame for item in trajectory_observations(isolated, 30.0)] == [0, 1, 20, 21]


def test_stale_reconnection_does_not_end_the_trail():
    # A confident detection reconnected 11 frames after the ball vanished (the
    # van case) is unverifiable: it may not end the trail or anchor a fill.
    stale_tail = Track(track_id=98, cfg=_cfg(60.0, HD_WIDTH, HD_HEIGHT))
    stale_tail.observations = [
        Detection(frame, 1000.0 + 4.0 * frame, 300.0, 0, 0, 4, 4, 0.95, 16.0, True)
        for frame in range(30)
    ] + [Detection(41, 1260.0, 230.0, 0, 0, 4, 4, 0.6, 16.0, True)]
    assert [item.frame for item in trajectory_observations(stale_tail, 60.0)][-1] == 29


def test_timeline_keeps_a_continuous_track_inside_another_tracks_hole():
    # A ball track living inside another track's span (the ball changed track
    # at a bounce and changed back) is the only evidence for its frames and
    # must survive timeline selection when it is continuous with the chain.
    chain_cfg = _cfg(30.0)
    outer = _line_track(5, list(range(0, 40)) + list(range(61, 100)), 100.0, 50.0, chain_cfg)
    inner = _line_track(6, list(range(40, 61)), 100.0, 20.0, chain_cfg)
    far = _line_track(7, list(range(40, 61)), 900.0, 20.0, chain_cfg)
    assert [track.track_id for track in select_timeline_chain([outer, inner], chain_cfg, 200)] == [5, 6]
    assert [track.track_id for track in select_timeline_chain([outer, far], chain_cfg, 200)] == [5]


def _point_track(track_id, points, score, cfg):
    """A track through the given ``(frame, x)`` points along y=100."""
    track = Track(track_id=track_id, cfg=cfg)
    track.observations = [
        Detection(frame, float(x), 100.0, 0, 0, 4, 4, 0.9, 16.0, True) for frame, x in points
    ]
    track.last_vel = (1.0, 0.0)
    track.score = score
    return track


def test_disjoint_continuity_is_decided_at_the_start_edge():
    # Pins scoring._add_disjoint_tracks.continuous as it behaves today: the
    # chain observation nearest in time to the candidate's FIRST observation
    # decides, and the last observation is never consulted when one exists.
    # A candidate whose start sits far from the chain but whose end meets it
    # is therefore dropped, while the mirror image is kept.  Whether the end
    # edge should also count is an open behaviour question, not settled here.
    chain_cfg = _cfg(30.0)
    outer = _line_track(5, list(range(0, 40)) + list(range(61, 100)), 100.0, 50.0, chain_cfg)
    inner_frames = range(40, 61)
    near_end = _point_track(
        6, [(frame, 900.0 if frame < 50 else 100.0 + frame) for frame in inner_frames], 20.0, chain_cfg
    )
    near_start = _point_track(
        7, [(frame, 100.0 + frame if frame <= 50 else 900.0) for frame in inner_frames], 20.0, chain_cfg
    )
    assert [track.track_id for track in select_timeline_chain([outer, near_end], chain_cfg, 200)] == [5]
    assert [track.track_id for track in select_timeline_chain([outer, near_start], chain_cfg, 200)] == [5, 7]


def test_weak_endpoint_without_motion_creates_no_interpolation():
    # Weak detector endpoints do not create blue interpolation without motion.
    detections, masks = _moving_ball()
    weak_anchor_detections = detections[:10] + [[], [([54, 20, 58, 24], 0.4)]]
    weak_anchor_masks = masks[:10] + [_empty_mask(), masks[11]]
    results, _, _ = select_ball_in_play(
        weak_anchor_detections, 30.0, WIDTH, HEIGHT,
        boost_masks=weak_anchor_masks, raw_motions=weak_anchor_masks,
    )
    assert results[10] is None


def test_slow_confident_ball_rolling_in_court_is_not_static_clutter():
    # A slow, high-confidence ball rolling inside the court is not static clutter.
    rolling = Track(track_id=100, cfg=_cfg(30.0))
    rolling.observations = [
        Detection(frame, 100.0 + frame, 200.0, 0, 0, 4, 4, 0.9, 16.0, frame % 5 == 0)
        for frame in range(30)
    ]
    rolling.score = 6.0
    rolling.score_breakdown = {
        "motion_overlap": 0.2,
        "inside_strict_frac": 0.9,
        "extent_px": 29.0,
    }
    assert rolling in selected_tracks([rolling])


def _sparse_track(track_id, frames):
    track = Track(track_id=track_id, cfg=_cfg(60.0, HD_WIDTH, HD_HEIGHT))
    track.observations = [
        Detection(frame, 1650.0 - 2.0 * index, 180.0, 0, 0, 4, 4, 0.6, 16.0, index % 3 == 0)
        for index, frame in enumerate(frames)
    ]
    track.score = 7.3
    track.score_breakdown = {"motion_overlap": 0.4, "inside_strict_frac": 0.0, "extent_px": 160.0}
    return track


def test_sparse_off_court_bystander_is_rejected_but_consecutive_reacquisition_is_kept():
    # Ten off-court detections sprinkled over seventy frames (a bystander) are
    # not a reacquired ball; the same ten on consecutive frames are.
    bystander = _sparse_track(102, [0, 2, 30, 55, 56, 59, 69, 70, 71, 72])
    reacquired_ball = _sparse_track(103, list(range(10)))
    assert bystander not in selected_tracks([bystander])
    assert reacquired_ball in selected_tracks([reacquired_ball])


def test_fast_short_track_is_selected():
    fast_short = Track(track_id=101, cfg=_cfg(30.0))
    fast_short.observations = [
        Detection(frame * 2, 100.0 + 12.0 * frame, 100.0, 0, 0, 4, 4, 0.9, 16.0, True)
        for frame in range(15)
    ]
    fast_short.score = 15.0
    fast_short.score_breakdown = {"motion_overlap": 1.0}
    assert fast_short in selected_tracks([fast_short])


def _airborne_track(track_id, motion_evidence, extent_px):
    """The measured numbers of the VolyAvw lob track (see the lob tests)."""
    track = Track(track_id=track_id, cfg=_cfg(60.0, HD_WIDTH, HD_HEIGHT))
    track.observations = [
        Detection(1940 + index, 945.0 + 1.5 * index, 200.0 - 0.5 * index,
                  0, 0, 4, 4, 0.88, 16.0, index % 5 == 0)
        for index in range(279)
    ]
    track.score = 281.0
    track.score_breakdown = {
        "motion_overlap": 0.391,
        "motion_evidence": motion_evidence,
        "inside_strict_frac": 0.251,
        "extent_px": extent_px,
    }
    return track


def test_lobbed_ball_crossing_the_baseline_is_selected():
    # A lobbed ball crossing the baseline sits in neither court-fraction band
    # (0.1 < inside_strict < 0.5) and the boost mask only catches it 39% of the
    # time, but its flight is unmistakable. These are the measured numbers of
    # the VolyAvw track that used to leave a 7.3 s hole in the middle of a rally.
    lobbed_ball = _airborne_track(104, 1.0, 663.0)
    assert lobbed_ball in selected_tracks([lobbed_ball])


def test_wide_but_slow_airborne_footprint_is_rejected():
    # The kinematic route is not a free pass: a wide but slow footprint scores at
    # most 0.60 from the extent term alone, which must stay below the gate.
    wide_and_slow = _airborne_track(105, 0.60, 663.0)
    assert wide_and_slow not in selected_tracks([wide_and_slow])


def test_fast_airborne_track_must_also_travel():
    # ... and a genuinely fast ball still needs to travel to qualify.
    fast_but_tiny = _airborne_track(106, 1.0, 40.0)
    assert fast_but_tiny not in selected_tracks([fast_but_tiny])


def test_timeline_prefers_the_continuous_overlap_and_keeps_a_reset_segment():
    selector_cfg = _cfg(30.0)
    prior = _line_track(1, range(0, 21), 100.0, 50.0, selector_cfg)
    false_overlap = _line_track(2, range(18, 41), 202.0, 45.0, selector_cfg)
    real_overlap = _line_track(3, range(18, 36), 100.0, 20.0, selector_cfg)
    reset_segment = _line_track(4, range(36, 60), 400.0, 45.0, selector_cfg)
    assert [track.track_id for track in select_timeline_chain(
        [prior, false_overlap, real_overlap, reset_segment], selector_cfg, 60
    )] == [1, 3, 4]


def test_timeline_keeps_a_strong_discontinuous_rally_segment():
    # A strong, non-overlapping rally segment must survive even when its
    # endpoints are spatially discontinuous with the surrounding rallies.
    # Numbers from the UCSD clip.
    ucsd_cfg = _cfg(60.0, HD_WIDTH, HD_HEIGHT)
    before = _line_track(22, range(2508, 2679), -1658.2, 74.0, ucsd_cfg)
    middle = _line_track(24, range(2740, 2823), -1909.6, 26.17, ucsd_cfg)
    after = _line_track(30, range(2920, 3023), -1946.2, 38.7, ucsd_cfg)
    assert [track.track_id for track in select_timeline_chain(
        [before, middle, after], ucsd_cfg, 3674
    )] == [22, 24, 30]


def test_stale_static_false_positive_does_not_absorb_a_later_moving_ball():
    # A stale static false positive must not absorb a later moving detection.
    separated = [[] for _ in range(34)]
    for frame in range(10):
        separated[frame] = [
            Detection(frame, 100.0, 100.0, 98, 98, 102, 102, 0.9, 16.0, False)
        ]
    for frame in range(31, 34):
        separated[frame] = [
            Detection(frame, 110.0 + frame - 31, 100.0, 0, 0, 4, 4, 0.9, 16.0, True)
        ]
    assert len(build_tracks(separated, _cfg(30.0))) == 2


def test_moving_ball_crossing_a_live_static_track_starts_its_own_track():
    # A moving ball crossing a live static false track must start its own track.
    crossing = [[] for _ in range(15)]
    for frame in range(10):
        crossing[frame] = [
            Detection(frame, 100.0, 100.0, 0, 0, 4, 4, 0.9, 16.0, False)
        ]
    for frame in range(10, 15):
        crossing[frame] = [
            Detection(frame, 112.0 + 8.0 * (frame - 10), 100.0, 0, 0, 4, 4, 0.9, 16.0, True)
        ]
    crossing_tracks = build_tracks(crossing, _cfg(30.0))
    assert len(crossing_tracks) == 2
    assert sorted(track.num_obs for track in crossing_tracks) == [5, 10]
