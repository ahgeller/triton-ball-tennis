"""ROI motion tracker scheduling (from ``clean_tracker.py --self-test``)."""
from tennis_tracker.config import Config
from tennis_tracker.roi_tracking import _FULL_FRAME_AREA_FRAC, ROIMotionTracker


def test_periodic_full_frame_scan_belongs_to_lost_ball_recovery():
    # Periodic full-frame scans belong to lost-ball recovery, not visible tracking.
    # next_rois advances the full-frame schedule, so each call is one frame.
    roi_tracker = ROIMotionTracker(Config(roi_fullframe_interval=2), 640, 360, 30.0)
    roi_tracker.last_rois = [(10, 10, 30, 30)]
    roi_tracker.last_rois_visual = list(roi_tracker.last_rois)
    assert roi_tracker.next_rois(0)[0] is not None
    assert roi_tracker.next_rois(1)[0] is not None
    assert roi_tracker.next_rois(2) == (None, None)


def test_fullframe_schedule_probes_only_while_the_ball_is_out_of_sight():
    roi_tracker = ROIMotionTracker(Config(roi_fullframe_interval=2), 640, 360, 30.0)
    assert [roi_tracker._advance_fullframe_schedule() for _ in range(4)] == [False, False, True, False]
    roi_tracker.ball_visible = True
    assert [roi_tracker._advance_fullframe_schedule() for _ in range(2)] == [False, False]
    assert ROIMotionTracker(Config(roi_fullframe_interval=0), 640, 360, 30.0)._advance_fullframe_schedule() is False


def test_valid_rois_drop_empty_boxes_and_give_up_on_a_frame_sized_one():
    roi_tracker = ROIMotionTracker(Config(), 640, 360, 30.0)
    assert roi_tracker._valid_rois([(10, 10, 30, 30), (5, 5, 5, 20), (0, 0, 40, 40)]) == [(10, 10, 30, 30), (0, 0, 40, 40)]
    side = int((_FULL_FRAME_AREA_FRAC * 640 * 360) ** 0.5) + 1
    assert roi_tracker._valid_rois([(10, 10, 30, 30), (0, 0, side, side)]) is None
    assert roi_tracker._valid_rois([]) == []
