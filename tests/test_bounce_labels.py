"""finetune.data.bounce_labels: the landing label file, read and written on temporary files, and the
blank points filled from the ball labels. No workspace file is read."""
import pytest

from finetune.data import workspace
from finetune.data.bounce_labels import BounceLabel, read_bounce_labels, resolve_points, write_bounce_labels
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
