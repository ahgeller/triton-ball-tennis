"""ROI motion tracker scheduling (from ``clean_tracker.py --self-test``)."""
from tennis_tracker.config import Config
from tennis_tracker.tracking import ROIMotionTracker


def test_periodic_full_frame_scan_belongs_to_lost_ball_recovery():
    # Periodic full-frame scans belong to lost-ball recovery, not visible tracking.
    # get_rois advances the full-frame schedule, so each call is one frame.
    roi_tracker = ROIMotionTracker(Config(roi_fullframe_interval=2), 640, 360, 30.0)
    roi_tracker.last_rois = [(10, 10, 30, 30)]
    roi_tracker.last_rois_visual = list(roi_tracker.last_rois)
    assert roi_tracker.get_rois(0)[0] is not None
    assert roi_tracker.get_rois(1)[0] is not None
    assert roi_tracker.get_rois(2) == (None, None)
