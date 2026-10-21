"""evaluation.tracking (validate_tracking.py): the tracking-JSON scorer on synthetic payloads."""
from evaluation.tracking import summary_lines, validate


def test_localization_recall_counts_wrong_positions_and_misses():
    predictions = {"frames": [{"frame": 0, "present": True, "x": 1000, "y": 1000}]}
    labels = {"ball": [{"frame": i, "visible": True, "x": 10, "y": 10} for i in range(2)]}
    summary = validate(predictions, labels, 200, 2)["summary"]
    assert summary["presence_recall"] == .5
    assert summary["localization_recall_10px"] == 0.


def test_summary_lines_name_every_figure_and_absence_only_when_labelled():
    predictions = {"frames": [{"frame": 0, "present": True, "x": 10, "y": 10}, {"frame": 1, "present": False}]}
    labels = {"ball": [{"frame": 0, "visible": True, "x": 10, "y": 10}, {"frame": 1, "visible": False}]}
    summary = validate(predictions, labels, 200, 2)["summary"]
    lines = summary_lines(summary)
    assert all(line.startswith("[validation] ") for line in lines)
    assert "[validation] recall: 1.000" in lines and "[validation] large jumps: 0" in lines
    assert any(line.startswith("[validation] error px: mean=") for line in lines)
    assert lines[-1] == "[validation] absence false positives: 0/1"
    labels_without_absence = {"ball": labels["ball"][:1]}
    lines = summary_lines(validate(predictions, labels_without_absence, 200, 2)["summary"])
    assert not any("absence" in line for line in lines)
