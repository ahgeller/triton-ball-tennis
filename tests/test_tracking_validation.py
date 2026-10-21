"""evaluation.tracking (validate_tracking.py): the tracking-JSON scorer on synthetic payloads."""
from evaluation.tracking import validate


def test_localization_recall_counts_wrong_positions_and_misses():
    predictions = {"frames": [{"frame": 0, "present": True, "x": 1000, "y": 1000}]}
    labels = {"ball": [{"frame": i, "visible": True, "x": 10, "y": 10} for i in range(2)]}
    summary = validate(predictions, labels, 200, 2)["summary"]
    assert summary["presence_recall"] == .5
    assert summary["localization_recall_10px"] == 0.
