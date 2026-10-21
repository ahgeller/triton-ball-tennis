from evaluation.compare_eval_logs import compare, read_scores

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
