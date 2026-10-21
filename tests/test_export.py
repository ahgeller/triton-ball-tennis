"""tennis_tracker.export: the tracking JSON keeps the measurement apart from the fitted position."""
from types import SimpleNamespace

from ball_in_play_selector.core import frame_result
from tennis_tracker.export import frame_result_to_json


def test_export_preserves_measurement_after_position_changes():
    detection = SimpleNamespace(cx=100., cy=103., conf=.9, x1=98, y1=101, x2=102, y2=105)
    result = frame_result(100, 103, "det", detection=detection)
    result.cy = 100.5
    row = frame_result_to_json(0, result)
    assert row["measurement"]["y"] == 103.
    assert row["y"] == 100.5
    assert row["position_kind"] == "filtered"
