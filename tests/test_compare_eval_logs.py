from evaluation.compare_eval_logs import compare, is_pooled, read_scores

BEFORE = """
[archive] finetune  (2 clips)
video13        vis= 300 recall=0.900 wrong=0.010 miss=0.090 | invis= 20 falseAlarm=0.250 | med=2.8px p90=6.0px
video14        vis= 150 recall=0.800 wrong=0.000 miss=0.200 | invis=  5 falseAlarm=0.000 | med=3.0px p90=7.5px
ALL            vis= 450 recall=0.867 wrong=0.007 miss=0.127 | invis= 25 falseAlarm=0.200 | med=2.9px p90=6.5px
"""
AFTER = """
video13        vis= 300 recall=0.950 wrong=0.010 miss=0.040 | invis= 20 falseAlarm=0.200 | med=2.7px p90=5.9px
video15        vis=  30 recall=1.000 wrong=0.000 miss=0.000 | invis=  0 falseAlarm=0.000 | med=1.0px p90=2.0px
ALL            vis= 330 recall=0.955 wrong=0.009 miss=0.036 | invis= 20 falseAlarm=0.200 | med=2.6px p90=5.7px
"""


COUNTED = """
video13        vis= 300 recall=0.900 wrong=0.010 miss=0.090 | invis= 20 falseAlarm=0.250 (5/20) | med=2.8px p90=6.0px
macro          vis= 450 recall=0.850 wrong=0.005 miss=0.145 | invis= 25 falseAlarm=0.125 (5/25) | med=2.9px p90=6.8px
source=prof    vis= 300 recall=0.900 wrong=0.010 miss=0.090 | invis= 20 falseAlarm=0.250 (5/20) | med=2.8px p90=6.0px
conf>=0.50     vis= 450 recall=0.867 wrong=0.007 miss=0.127 | invis= 25 falseAlarm=0.200 (5/25) | med=2.9px p90=6.5px
"""


def test_read_scores_takes_the_count_beside_the_false_alarm_rate_and_logs_without_one():
    scores = read_scores(COUNTED)
    assert list(scores) == ["video13", "macro", "source=prof", "conf>=0.50"]
    assert scores["video13"].false_alarm == 0.25 and scores["macro"].false_alarm == 0.125
    assert read_scores(BEFORE)["video13"].false_alarm == 0.25                  # the older line, without a count


def test_macro_and_source_lines_are_pooled_lines():
    assert is_pooled("ALL") and is_pooled("conf>=0.50") and is_pooled("macro") and is_pooled("source=prof")
    assert not is_pooled("video13") and not is_pooled("grid_match8")
    text = compare(read_scores(COUNTED), read_scores(COUNTED))
    rows = [line.split()[0] for line in text.splitlines() if line and not line.startswith(("clip", "-"))]
    assert rows == ["video13", "macro", "source=prof", "conf>=0.50"]       # the clip first, the pooled lines after the gap
    assert "\n\nmacro" in text


def test_read_scores_parses_every_metric_line():
    scores = read_scores(BEFORE)
    assert list(scores) == ["video13", "video14", "ALL"]
    assert scores["video13"].visible == 300 and scores["video13"].false_alarm == 0.25
    assert scores["ALL"].p90_px == 6.5


def test_compare_reports_deltas_and_clips_in_one_log_only():
    text = compare(read_scores(BEFORE), read_scores(AFTER))
    lines = [line for line in text.splitlines() if line.startswith("video13")]
    assert len(lines) == 1 and "0.950 (+0.050)" in lines[0] and "0.200 (-0.050)" in lines[0]
    assert "ALL" in text and "only in the first log: video14" in text and "only in the second log: video15" in text
    assert "video14" not in text.split("only in the first log")[0]
