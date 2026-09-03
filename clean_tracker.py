"""Run the isolated clean tennis tracker."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from ball_in_play_selector import select_ball_in_play
from ball_in_play_selector.config import SelectorConfig
from ball_in_play_selector.core import (
    _direction_cosine,
    _refine_trajectory,
    _selected_tracks,
    _trajectory_observations,
)
from ball_in_play_selector.models import Detection, FrameResult, Track
from ball_in_play_selector.scoring import _select_timeline_chain
from ball_in_play_selector.tracking import build_tracks
from tennis_tracker.config import Config
from tennis_tracker.detectors import CourtDetector
from tennis_tracker.pipeline import run
from tennis_tracker.tracking import ROIMotionTracker


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Isolated clean tennis tracker")
    parser.add_argument("-i", "--input", default=str(ROOT / "sample" / "pomona.mp4"))
    parser.add_argument("-o", "--output", default=str(OUTPUT / "tracking.mp4"))
    parser.add_argument("--tracking-json", default=str(OUTPUT / "tracking.json"))
    parser.add_argument("--conf", type=float, default=0.50)
    parser.add_argument("--device", default="0")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--court", action="store_true", help="Draw the top-right 2D court minibar")
    parser.add_argument("--info", action="store_true")
    parser.add_argument("--annotations", help="Validate against this annotation JSON")
    parser.add_argument("--report-json", default=str(OUTPUT / "validation.json"))
    parser.add_argument("--self-test", action="store_true")
    return parser


def _self_test() -> None:
    from tennis_tracker.gridtracknet import self_test as gridtracknet_self_test
    from tennis_tracker.court_overlay import (
        _build_bounce_features,
        _cluster_bounce_candidates,
        _inside_player,
        _is_service_leg,
        _visible_completed_legs,
        build_rally_legs,
        draw_court_minimap,
        infer_kinematic_bounces,
    )

    gridtracknet_self_test()
    assert CourtDetector._LINE_PAIRS_14 == (
        (0, 4), (4, 6), (6, 1), (0, 2), (1, 3),
        (2, 5), (5, 7), (7, 3), (4, 8), (8, 10), (10, 5),
        (6, 9), (9, 11), (11, 7), (8, 12), (12, 9),
        (10, 13), (13, 11), (12, 13),
    )

    detections = []
    masks = []
    for frame in range(12):
        x = 10 + 12 * frame
        detections.append([([x, 20, x + 4, 24], 0.9)])
        mask = np.zeros((360, 640), dtype=np.uint8)
        mask[20:25, x:x + 5] = 255
        masks.append(mask)
    results, chosen, tracks, _ = select_ball_in_play(
        detections, 30.0, 640, 360, boost_masks=masks, raw_motions=masks
    )
    assert chosen is not None and tracks
    assert sum(result is not None for result in results) == len(detections)

    # The first internal gap is safe to fill because both real endpoints exist.
    first_gap_detections = detections[:1] + [[], []] + detections[3:]
    results, _, _, _ = select_ball_in_play(
        first_gap_detections, 30.0, 640, 360,
        boost_masks=masks, raw_motions=masks,
    )
    assert results[1] is not None and results[2] is not None

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
        mask = np.zeros((360, 640), dtype=np.uint8)
        mask[100:105, x:x + 5] = 255
        sampled_masks.append(mask)
        sampled_players.append({1: (0, 0, 640, 300)})
    results, _, _, _ = select_ball_in_play(
        sampled_detections,
        60.0,
        640,
        360,
        boost_masks=sampled_masks,
        raw_motions=sampled_masks,
        player_boxes_by_frame=sampled_players,
    )
    assert all(results[frame] is not None for frame in range(1, 38, 2))
    assert all(results[frame].source == "motion" for frame in range(1, 38, 2))

    straight = [
        Detection(frame, float(frame * 10), 20.0, 0, 0, 4, 4, 0.9, 16.0, True)
        for frame in range(3)
    ]
    assert _direction_cosine(*straight) == 1.0
    straight[2].cx = 0.0
    assert _direction_cosine(*straight) == -1.0

    # The robust refit repairs a short target-switch block without changing
    # the source classification used by the renderer.
    smooth = [
        FrameResult(cx=float(frame * 10), cy=100.0, conf=0.9, source="det")
        for frame in range(15)
    ]
    smooth[7].cy = smooth[8].cy = 180.0
    _refine_trajectory(smooth)
    assert max(abs(smooth[frame].cy - 100.0) for frame in (7, 8)) < 10.0
    assert all(result.source == "det" for result in smooth)

    # One broad motion hit may support one physics point, never a blue tail
    # across later frames where the ball has disappeared.
    lost_detections = detections[:10] + [[] for _ in range(5)]
    lost_masks = masks[:10] + [np.zeros((360, 640), dtype=np.uint8) for _ in range(5)]
    lost_masks[10][20:25, 130:135] = 255
    results, _, _, _ = select_ball_in_play(
        lost_detections, 30.0, 640, 360,
        boost_masks=lost_masks, raw_motions=lost_masks,
    )
    assert results[10] is not None and results[10].source == "motion"
    assert all(result is None for result in results[11:])

    weak = [[([10 + frame, 20, 14 + frame, 24], 0.2)] for frame in range(4)]
    results, chosen, _, _ = select_ball_in_play(weak, 30.0, 640, 360)
    assert chosen is None and all(result is None for result in results)

    # A lone weak/no-motion wall hit cannot anchor a visible connector.
    isolated = Track(track_id=99, cfg=SelectorConfig(fps=30.0, width=640, height=360).auto_scale())
    isolated.observations = [
        Detection(frame, float(frame), 20.0, 0, 0, 4, 4, conf, 16.0, on_motion)
        for frame, conf, on_motion in (
            (0, 0.9, True), (1, 0.9, True), (10, 0.2, False),
            (20, 0.9, True), (21, 0.9, True),
        )
    ]
    assert [item.frame for item in _trajectory_observations(isolated, 30.0)] == [0, 1, 20, 21]

    # A confident detection reconnected 11 frames after the ball vanished (the
    # van case) is unverifiable: it may not end the trail or anchor a fill.
    stale_tail = Track(track_id=98, cfg=SelectorConfig(fps=60.0, width=1920, height=1080).auto_scale())
    stale_tail.observations = [
        Detection(frame, 1000.0 + 4.0 * frame, 300.0, 0, 0, 4, 4, 0.95, 16.0, True)
        for frame in range(30)
    ] + [Detection(41, 1260.0, 230.0, 0, 0, 4, 4, 0.6, 16.0, True)]
    assert [item.frame for item in _trajectory_observations(stale_tail, 60.0)][-1] == 29

    # A ball track living inside another track's span (the ball changed track
    # at a bounce and changed back) is the only evidence for its frames and
    # must survive timeline selection when it is continuous with the chain.
    chain_cfg = SelectorConfig(fps=30.0, width=640, height=360).auto_scale()

    def holed_track(track_id, frames, x_offset, score):
        track = Track(track_id=track_id, cfg=chain_cfg)
        track.observations = [
            Detection(frame, x_offset + frame, 100.0, 0, 0, 4, 4, 0.9, 16.0, True)
            for frame in frames
        ]
        track.last_vel = (1.0, 0.0)
        track.score = score
        track.score_breakdown = {"period_id": 0.0}
        return track

    outer = holed_track(5, list(range(0, 40)) + list(range(61, 100)), 100.0, 50.0)
    inner = holed_track(6, list(range(40, 61)), 100.0, 20.0)
    far = holed_track(7, list(range(40, 61)), 900.0, 20.0)
    assert [track.track_id for track in _select_timeline_chain([outer, inner], chain_cfg, 200)] == [5, 6]
    assert [track.track_id for track in _select_timeline_chain([outer, far], chain_cfg, 200)] == [5]

    # Weak detector endpoints do not create blue interpolation without motion.
    weak_anchor_detections = detections[:10] + [[], [([54, 20, 58, 24], 0.4)]]
    weak_anchor_masks = masks[:10] + [
        np.zeros((360, 640), dtype=np.uint8), masks[11]
    ]
    results, _, _, _ = select_ball_in_play(
        weak_anchor_detections, 30.0, 640, 360,
        boost_masks=weak_anchor_masks, raw_motions=weak_anchor_masks,
    )
    assert results[10] is None

    selector_cfg = SelectorConfig(fps=30.0, width=640, height=360).auto_scale()

    # A slow, high-confidence ball rolling inside the court is not static clutter.
    rolling = Track(track_id=100, cfg=selector_cfg)
    rolling.observations = [
        Detection(frame, 100.0 + frame, 200.0, 0, 0, 4, 4, 0.9, 16.0, frame % 5 == 0)
        for frame in range(30)
    ]
    rolling.score = 6.0
    rolling.score_breakdown = {
        "motion_frac_raw": 0.2,
        "inside_strict_frac": 0.9,
        "extent_px": 29.0,
    }
    assert rolling in _selected_tracks([rolling], 30.0)

    # Ten off-court detections sprinkled over seventy frames (a bystander) are
    # not a reacquired ball; the same ten on consecutive frames are.
    def sparse_track(track_id, frames):
        track = Track(track_id=track_id, cfg=SelectorConfig(fps=60.0, width=1920, height=1080).auto_scale())
        track.observations = [
            Detection(frame, 1650.0 - 2.0 * index, 180.0, 0, 0, 4, 4, 0.6, 16.0, index % 3 == 0)
            for index, frame in enumerate(frames)
        ]
        track.score = 7.3
        track.score_breakdown = {"motion_frac_raw": 0.4, "inside_strict_frac": 0.0, "extent_px": 160.0}
        return track

    bystander = sparse_track(102, [0, 2, 30, 55, 56, 59, 69, 70, 71, 72])
    reacquired_ball = sparse_track(103, list(range(10)))
    assert bystander not in _selected_tracks([bystander], 60.0)
    assert reacquired_ball in _selected_tracks([reacquired_ball], 60.0)

    fast_short = Track(track_id=101, cfg=selector_cfg)
    fast_short.observations = [
        Detection(frame * 2, 100.0 + 12.0 * frame, 100.0, 0, 0, 4, 4, 0.9, 16.0, True)
        for frame in range(15)
    ]
    fast_short.score = 15.0
    fast_short.score_breakdown = {"motion_frac_raw": 1.0}
    assert fast_short in _selected_tracks([fast_short], 60.0)

    # A lobbed ball crossing the baseline sits in neither court-fraction band
    # (0.1 < inside_strict < 0.5) and the boost mask only catches it 39% of the
    # time, but its flight is unmistakable. These are the measured numbers of
    # the VolyAvw track that used to leave a 7.3 s hole in the middle of a rally.
    def airborne_track(track_id, motion_frac, extent_px):
        track = Track(track_id=track_id, cfg=SelectorConfig(fps=60.0, width=1920, height=1080).auto_scale())
        track.observations = [
            Detection(1940 + index, 945.0 + 1.5 * index, 200.0 - 0.5 * index,
                      0, 0, 4, 4, 0.88, 16.0, index % 5 == 0)
            for index in range(279)
        ]
        track.score = 281.0
        track.score_breakdown = {
            "motion_frac_raw": 0.391,
            "motion_frac": motion_frac,
            "inside_strict_frac": 0.251,
            "extent_px": extent_px,
        }
        return track

    lobbed_ball = airborne_track(104, 1.0, 663.0)
    assert lobbed_ball in _selected_tracks([lobbed_ball], 60.0)
    # The kinematic route is not a free pass: a wide but slow footprint scores at
    # most 0.60 from the extent term alone, which must stay below the gate.
    wide_and_slow = airborne_track(105, 0.60, 663.0)
    assert wide_and_slow not in _selected_tracks([wide_and_slow], 60.0)
    # ... and a genuinely fast ball still needs to travel to qualify.
    fast_but_tiny = airborne_track(106, 1.0, 40.0)
    assert fast_but_tiny not in _selected_tracks([fast_but_tiny], 60.0)

    def timeline_track(track_id, start, end, x_offset, score):
        track = Track(track_id=track_id, cfg=selector_cfg)
        track.observations = [
            Detection(frame, x_offset + frame, 100.0, 0, 0, 4, 4, 0.9, 16.0, True)
            for frame in range(start, end + 1)
        ]
        track.last_vel = (1.0, 0.0)
        track.score = score
        track.score_breakdown = {"period_id": 0.0}
        return track

    prior = timeline_track(1, 0, 20, 100.0, 50.0)
    false_overlap = timeline_track(2, 18, 40, 202.0, 45.0)
    real_overlap = timeline_track(3, 18, 35, 100.0, 20.0)
    reset_segment = timeline_track(4, 36, 59, 400.0, 45.0)
    assert [track.track_id for track in _select_timeline_chain(
        [prior, false_overlap, real_overlap, reset_segment], selector_cfg, 60
    )] == [1, 3, 4]

    # A strong, non-overlapping rally segment must survive even when its
    # endpoints are spatially discontinuous with the surrounding rallies.
    ucsd_cfg = SelectorConfig(fps=60.0, width=1920, height=1080).auto_scale()
    before = timeline_track(22, 2508, 2678, -1658.2, 74.0)
    middle = timeline_track(24, 2740, 2822, -1909.6, 26.17)
    after = timeline_track(30, 2920, 3022, -1946.2, 38.7)
    for track in (before, middle, after):
        track.cfg = ucsd_cfg
    assert [track.track_id for track in _select_timeline_chain(
        [before, middle, after], ucsd_cfg, 3674
    )] == [22, 24, 30]

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
    assert len(build_tracks(separated, selector_cfg)) == 2

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
    crossing_tracks = build_tracks(crossing, selector_cfg)
    assert len(crossing_tracks) == 2
    assert sorted(track.num_obs for track in crossing_tracks) == [5, 10]

    # Periodic full-frame scans belong to lost-ball recovery, not visible tracking.
    roi_cfg = Config(roi_fullframe_interval=2)
    roi_tracker = ROIMotionTracker(roi_cfg, 640, 360, 30.0)
    roi_tracker.last_rois = [(10, 10, 30, 30)]
    roi_tracker.last_rois_visual = list(roi_tracker.last_rois)
    assert roi_tracker.get_rois(0)[0] is not None
    assert roi_tracker.get_rois(1)[0] is not None
    assert roi_tracker.get_rois(2) == (None, None)

    # The bounce model consumes five samples at its native ~30 FPS cadence.
    bounce_probe = [None] * 9
    for frame, (x, y) in zip(range(0, 9, 2), ((10, 20), (13, 24), (17, 29), (22, 20), (28, 12))):
        bounce_probe[frame] = FrameResult(cx=x, cy=y, conf=0.9, source="det")
    features, feature_frames, stride = _build_bounce_features(
        bounce_probe, 60.0, 1280, 720
    )
    assert stride == 2 and feature_frames == [4]
    np.testing.assert_allclose(
        features[0],
        [4, 7, 5, 11, 0.8, 7 / 11, -5, -9, -9, -17, 5 / 9, 9 / 17],
        rtol=1e-5,
    )
    assert _cluster_bounce_candidates([4, 6, 8, 14], [0.4, 0.6, 0.8, 0.7], 2) == [
        (8, 0.8), (14, 0.7)
    ]
    strike_box = {1: (100, 100, 140, 200)}
    assert _inside_player(70, 90, strike_box)
    assert not _inside_player(120, 210, strike_box)
    assert _is_service_leg((0.80, 1.02), (0.25, 0.35), True)
    assert not _is_service_leg((0.80, 1.02), (0.25, 0.80), True)
    bounce_motion = [
        FrameResult(cx=10 + frame, cy=y, conf=0.9, source="det")
        for frame, y in enumerate((0, 4, 10, 6, 4))
    ]
    apex_motion = [
        FrameResult(cx=10 + frame, cy=y, conf=0.9, source="det")
        for frame, y in enumerate((10, 6, 0, 4, 10))
    ]
    assert len(infer_kinematic_bounces(
        bounce_motion, [{}] * 5, 30.0, 1920, 1080
    )) == 1
    assert not infer_kinematic_bounces(
        apex_motion, [{}] * 5, 30.0, 1920, 1080
    )
    gap_bounce = [
        FrameResult(cx=10, cy=0, conf=0.9, source="det"),
        FrameResult(cx=11, cy=4, conf=0.9, source="det"),
        None, None, None, None,
        FrameResult(cx=16, cy=4, conf=0.9, source="det"),
        FrameResult(cx=17, cy=2, conf=0.9, source="det"),
    ]
    assert len(infer_kinematic_bounces(
        gap_bounce, [{}] * len(gap_bounce), 30.0, 1920, 1080
    )) == 1
    test_kps = [10, 10, 110, 10, 10, 210, 110, 210] + [0] * 20
    inferred_leg = build_rally_legs(
        [
            (0, 90, 210, 1, 10.0, "kink"),
            (50, 35, 10, 2, 10.0, "kink"),
        ],
        [],
        [],
        [],
        [test_kps] * 80,
        30.0,
        120,
        220,
    )
    assert inferred_leg == []
    minibar_probe = np.zeros((360, 640, 3), dtype=np.uint8)
    minibar_leg = {
        "start_frame": 0,
        "end_frame": 30,
        "start": (0.80, 1.02),
        "end": (0.25, 0.35),
        "phase": "serve",
        "outcome": "bounce",
        "quality": "confirmed",
    }
    assert _visible_completed_legs([minibar_leg], [], 60, 60.0) == [minibar_leg]
    assert not _visible_completed_legs(
        [minibar_leg], [(61, 0, 0, 1, 1.0, "kink")], 61, 60.0
    )
    assert not _visible_completed_legs([minibar_leg], [], 91, 60.0)
    draw_court_minimap(
        minibar_probe,
        {"H_i2c": np.eye(3)},
        {1: (0.4, 0.3, 0.6, 0.8)},
        [minibar_leg],
        30,
        60.0,
    )
    assert np.any(minibar_probe[:, 400:]) and not np.any(minibar_probe[:, :300])


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.self_test:
        _self_test()
        print("clean_version self-test passed")
        return 0

    OUTPUT.mkdir(parents=True, exist_ok=True)
    cfg = Config(
        input_video=str(Path(args.input).resolve()),
        output_video=str(Path(args.output).resolve()),
        tracking_json=str(Path(args.tracking_json).resolve()),
        model_path=str(ROOT / "models" / "gridtracknet_weights_torch.npz"),
        player_model_path=str(ROOT / "models" / "player.engine"),
        court_model_path=str(ROOT / "models" / "courtdetection.engine"),
        conf=float(args.conf),
        device=str(args.device),
        save_tracking_video=not args.no_video,
        draw_court_minimap=bool(args.court),
        bounce_model_path=str(ROOT / "ctb_regr_bounce.cbm"),
        print_selector_tracks=False,
        info_timing=bool(args.info),
    )
    run(cfg)
    if not args.annotations:
        return 0
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "validate_tracking.py"),
            "--predictions", cfg.tracking_json,
            "--annotations", str(Path(args.annotations).resolve()),
            "--report-json", str(Path(args.report_json).resolve()),
        ],
        cwd=ROOT,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
