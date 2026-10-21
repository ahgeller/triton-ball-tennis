"""finetune/data/detector_metrics.py: the (label, prediction) metric and the 1080p scale factors."""
import math

from finetune.data.detector_metrics import HIT_PX, WRONG_PX, DetectorMetrics, scale_for_height, scale_for_width


def test_scale_factors_take_a_frame_to_1080p():
    assert scale_for_width(1280) == 1.5
    assert scale_for_height(720) == 1.5
    assert scale_for_width(1920) == 1.0


def test_error_exactly_at_hit_px_is_a_hit():
    stats = DetectorMetrics.of([((0.0, 0.0), (HIT_PX, 0.0))], 1.0)
    assert stats.visible == 1
    assert stats.recall == 1.0
    assert stats.wrong == 0.0
    assert stats.median_px == HIT_PX


def test_error_between_hit_and_wrong_is_neither():
    stats = DetectorMetrics.of([((0.0, 0.0), (WRONG_PX, 0.0))], 1.0)
    assert stats.visible == 1
    assert stats.recall == 0.0
    assert stats.wrong == 0.0


def test_error_just_past_wrong_px_is_a_wrong_object():
    stats = DetectorMetrics.of([((0.0, 0.0), (WRONG_PX + 0.01, 0.0))], 1.0)
    assert stats.wrong == 1.0
    assert stats.recall == 0.0


def test_scale_multiplies_the_error():
    assert DetectorMetrics.of([((0.0, 0.0), (8.0, 0.0))], 1.0).recall == 1.0      # 8 px at 1080p: a hit
    assert DetectorMetrics.of([((0.0, 0.0), (8.0, 0.0))], 1.5).recall == 0.0      # 12 px at 1080p: not


def test_prediction_on_an_invisible_ball_is_a_false_alarm():
    stats = DetectorMetrics.of([(None, (5.0, 5.0))], 1.0)
    assert stats.false_alarm == 1.0
    assert stats.invisible == 1
    assert stats.visible == 0


def test_no_prediction_on_an_invisible_ball_is_quiet():
    stats = DetectorMetrics.of([(None, None)], 1.0)
    assert stats.false_alarm == 0.0
    assert stats.invisible == 1


def test_no_prediction_on_a_visible_ball_is_a_miss():
    stats = DetectorMetrics.of([((1.0, 1.0), None)], 1.0)
    assert stats.miss == 1.0
    assert stats.visible == 1
    assert math.isnan(stats.median_px)


def test_empty_input_has_nan_distances_and_zero_counts():
    stats = DetectorMetrics.of([], 1.0)
    assert math.isnan(stats.median_px)
    assert math.isnan(stats.p90_px)
    assert stats.visible == 0 and stats.invisible == 0
    assert stats.recall == 0.0 and stats.false_alarm == 0.0


def test_percentiles_come_from_the_sorted_errors():
    pairs = [((0.0, 0.0), (float(e), 0.0)) for e in (9, 1, 5, 3, 7)]
    stats = DetectorMetrics.of(pairs, 1.0)
    assert stats.median_px == 5.0
    assert stats.p90_px == 9.0


def test_the_metric_is_a_named_record_with_one_vocabulary():
    stats = DetectorMetrics.of([((0.0, 0.0), (1.0, 0.0)), (None, None)], 1.0)
    assert isinstance(stats, DetectorMetrics)
    assert list(stats._asdict()) == ["visible", "recall", "wrong", "miss", "invisible", "false_alarm", "median_px", "p90_px"]
    assert stats._asdict()["wrong"] == 0.0 and "wrong_obj" not in stats._asdict()
