"""finetune.data.bounce_labels: the landing label file, read and written on temporary files, and the
blank points filled from the ball labels. No workspace file is read."""
import pytest

from finetune.data import workspace
from finetune.data.bounce_labels import (BounceLabel, between_frames, landing_frame, landing_name, landing_text,
                                         read_bounce_labels, resolve_points, write_bounce_labels)
from finetune.data.labels import LabelFormatError


def test_bounce_csv_names_the_landing_file_beside_the_ball_labels(tmp_path):
    assert workspace.bounce_csv("video6", tmp_path) == tmp_path / "video6_bounces.csv"
    assert workspace.bounce_csv("video6", tmp_path).parent == workspace.label_csv("video6", tmp_path).parent


def test_write_then_read_keeps_frames_in_order_and_points_optional(tmp_path):
    path = tmp_path / "video6_bounces.csv"
    assert write_bounce_labels(path, [BounceLabel(120), BounceLabel(35, (640.5, 300.25)), BounceLabel(80, None)]) == 3
    assert path.read_text(encoding="utf-8").splitlines() == [
        "frame,ball_x,ball_y", "frame_035,640.50,300.25", "frame_080,,", "frame_120,,",
    ]
    assert read_bounce_labels(path) == [BounceLabel(35, (640.5, 300.25)), BounceLabel(80, None), BounceLabel(120, None)]


def test_a_frame_given_twice_is_refused_before_the_file_is_touched(tmp_path):
    path = tmp_path / "video6_bounces.csv"
    write_bounce_labels(path, [BounceLabel(5)])
    # One with a point and one without: ordered by frame alone, not by comparing the points.
    with pytest.raises(ValueError, match="frame 80 is given twice"):
        write_bounce_labels(path, [BounceLabel(80, (1.0, 2.0)), BounceLabel(120), BounceLabel(80, None)])
    assert read_bounce_labels(path) == [BounceLabel(5, None)]


def test_a_landing_between_two_frames_is_written_and_read_as_a_half_frame(tmp_path):
    path = tmp_path / "video6_bounces.csv"
    assert write_bounce_labels(path, [BounceLabel(413), BounceLabel(412.5, (640.5, 300.25)), BounceLabel(3.5)]) == 3
    assert path.read_text(encoding="utf-8").splitlines() == [
        "frame,ball_x,ball_y", "frame_003.5,,", "frame_412.5,640.50,300.25", "frame_413,,",
    ]
    labels = read_bounce_labels(path)
    assert labels == [BounceLabel(3.5, None), BounceLabel(412.5, (640.5, 300.25)), BounceLabel(413, None)]
    assert [type(label.frame) for label in labels] == [float, float, int]      # a whole frame stays an int


def test_the_landing_cell_parser_takes_exactly_a_frame_or_a_half_frame():
    assert landing_frame("frame_0412") == 412 and isinstance(landing_frame("frame_0412"), int)
    assert landing_frame(" frame_0412.5 ") == 412.5
    for cell in ("frame_12.4", "frame_12.50", "frame_.5", "frame_12.", "412.5", "frame_12.5.5", ""):
        with pytest.raises(LabelFormatError, match="expected frame_<index> or frame_<index>.5"):
            landing_frame(cell)
    assert landing_name(412) == "frame_412" and landing_name(412.0) == "frame_412" and landing_name(3.5) == "frame_003.5"
    with pytest.raises(ValueError):
        landing_name(12.25)
    assert landing_text(412) == "412" and landing_text(412.0) == "412" and landing_text(412.5) == "412.5"
    assert between_frames(3, 4) == 3.5 and between_frames(4, 3) == 3.5
    assert between_frames(412, 414) == 413 and isinstance(between_frames(412, 414), int)


def test_a_hand_written_file_with_blank_lines_and_spaces_reads(tmp_path):
    path = tmp_path / "v_bounces.csv"
    path.write_text("frame,ball_x,ball_y\n\nframe_003, , \nframe_0100,12.5,7\n\n", encoding="utf-8")
    assert read_bounce_labels(path) == [BounceLabel(3, None), BounceLabel(100, (12.5, 7.0))]


@pytest.mark.parametrize("body", [
    "frame_010,5,",                    # one coordinate without the other
    "frame_010,,5",
    "frame_010,x,5",                   # a coordinate that is not a number
    "10,,",                            # a frame cell that is not frame_<index>
    "frame_010,,\nframe_010,,",        # a landing listed twice
    "frame_010.5,,\nframe_010.5,,",    # ... between the same two frames
    "frame_010.4,,",                   # a fraction the file has no meaning for
])
def test_a_malformed_row_is_an_error_naming_the_file(tmp_path, body):
    path = tmp_path / "v_bounces.csv"
    path.write_text("frame,ball_x,ball_y\n" + body + "\n", encoding="utf-8")
    with pytest.raises(LabelFormatError, match="v_bounces.csv"):
        read_bounce_labels(path)


def test_the_columns_must_be_the_ball_label_files(tmp_path):
    path = tmp_path / "v_bounces.csv"
    path.write_text("frame,x,y\nframe_001,,\n", encoding="utf-8")
    with pytest.raises(LabelFormatError, match="expected frame,ball_x,ball_y"):
        read_bounce_labels(path)


def test_resolve_points_fills_blank_points_from_the_ball_labels_or_the_other_phase():
    ball_labels = {10: (1.0, 2.0), 19: (7.0, 8.0), 21: (3.0, 4.0), 30: None, 42: (5.0, 6.0)}
    labels = [BounceLabel(10), BounceLabel(20), BounceLabel(30), BounceLabel(50), BounceLabel(41, (9.0, 9.0))]
    assert resolve_points(labels, ball_labels) == [
        BounceLabel(10, (1.0, 2.0)),       # the ball label at the landing's frame
        BounceLabel(20, (7.0, 8.0)),       # the other 60 FPS phase: the frame before, then the frame after
        BounceLabel(30, None),             # an invisible ball there, and no adjacent label: no point
        BounceLabel(50, None),
        BounceLabel(41, (9.0, 9.0)),       # its own point stands
    ]
    assert resolve_points([BounceLabel(42)], {43: (5.0, 6.0)}) == [BounceLabel(42, (5.0, 6.0))]


def test_resolve_points_puts_a_landing_between_two_frames_midway_between_their_ball_labels():
    ball_labels = {10: (1.0, 2.0), 11: (3.0, 6.0), 20: (7.0, 8.0), 31: None, 40: (9.0, 9.0)}
    labels = [BounceLabel(10.5), BounceLabel(20.5), BounceLabel(30.5), BounceLabel(50.5), BounceLabel(39.5, (0.0, 0.0))]
    assert resolve_points(labels, ball_labels) == [
        BounceLabel(10.5, (2.0, 4.0)),     # the midpoint of the labels on both frames
        BounceLabel(20.5, (7.0, 8.0)),     # only one of the two is labelled: its ball
        BounceLabel(30.5, None),           # an invisible ball on one, no label on the other
        BounceLabel(50.5, None),
        BounceLabel(39.5, (0.0, 0.0)),     # its own point stands
    ]
    assert resolve_points([BounceLabel(10.5)], {9: (1.0, 1.0), 12: (1.0, 1.0)}) == [BounceLabel(10.5, None)]   # no reach beyond the two
