"""The console summaries in tennis_tracker/summaries.py, read back through capsys."""
from types import SimpleNamespace

from tennis_tracker.summaries import _percent_or_dash, _track_row, print_selector_track_summary, print_timing_summary


def _track(track_id, score, breakdown):
    return SimpleNamespace(track_id=track_id, score=score, num_obs=5, span=9, first_frame=0, last_obs_frame=8,
                           score_breakdown=breakdown)


FULL = _track(1, 12.5, {"start_frac": 0.1, "end_frac": 0.5, "inside_strict_frac": 0.8,
                        "motion_evidence": 0.4, "near_player_frac": 0.2})
BARE = _track(2, 3.0, {})      # no breakdown: time fractions come from first/last frame


def test_track_table_without_tracks(capsys):
    print_selector_track_summary([], None, 10)
    assert capsys.readouterr().out.strip() == "[selector] Track table: none"


def test_track_table_marks_the_chosen_track(capsys):
    print_selector_track_summary([BARE, FULL], FULL, 10)
    out = capsys.readouterr().out.splitlines()
    rows = [line for line in out if line.startswith("  ") and not line.startswith("  sel")]
    assert rows[0].startswith("    *   1 ")           # sorted by score, chosen row starred
    assert rows[1].startswith("        2 ")
    assert "  10.0  50.0" in rows[0]                   # start/end fractions from the breakdown
    assert "   0.0  88.9" in rows[1]                   # ... and from frames 0..8 of 10 otherwise
    assert "   --    --     --" in rows[1]             # missing evidence columns print as --
    assert "[selector] Chosen track: id=1 score=12.5 obs=5 span=9 t=10.0%->50.0%" in out
    assert not any(line.startswith("[selector] Showing top") for line in out)


def test_track_table_limit_and_no_chosen_track(capsys):
    print_selector_track_summary([BARE, FULL], None, 10, limit=1)
    out = capsys.readouterr().out.splitlines()
    assert "[selector] Showing top 1 / 2 tracks" in out
    assert "[selector] Chosen track: none" in out
    assert sum(1 for line in out if line.startswith("  ") and not line.startswith("  sel")) == 1


def test_timing_summary_skips_zero_stages(capsys):
    print_timing_summary({"pass1_total": 2.0, "pass2_total": 0.0}, 4.0, 100, 50)
    out = capsys.readouterr().out
    assert "[info] Stage timing summary" in out
    pass1 = [line for line in out.splitlines() if "pass1 total" in line]
    assert len(pass1) == 1
    assert "2.000s" in pass1[0] and "50.0%" in pass1[0] and "20.00 ms/frame" in pass1[0]
    assert "pass2 total" not in out


def test_timing_summary_init_row_has_no_per_frame_cost(capsys):
    print_timing_summary({"init_total": 1.0}, 2.0, 10, 10)
    init = [line for line in capsys.readouterr().out.splitlines() if "init" in line]
    assert len(init) == 1 and "ms/frame" not in init[0]


def test_track_row_marks_the_chosen_track_and_dashes_unscored_columns():
    assert _percent_or_dash(0.25) == " 25.0" and _percent_or_dash(-1.0) == "   --"
    full = _track_row(FULL, True, 10).split()
    assert full[:4] == ["*", "1", "12.5", "5"] and full[-3:] == ["80.0", "40.0", "20.0"]
    bare = _track_row(BARE, False, 10).split()
    assert bare[:2] == ["2", "3.0"] and bare[-3:] == ["--", "--", "--"]
    assert bare[4:6] == ["0.0", "88.9"]                          # first/last frame over 9 intervals
