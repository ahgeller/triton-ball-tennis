"""finetune/data/label_flags.py: the frame size a clip's labels are read at. The flags themselves are
covered with the audit and the label tool."""
from finetune.data.label_flags import clip_labels, label_frame_size
from finetune.data.labels import write_labels
from finetune.data.workspace import MANIFEST_COLUMNS, label_csv, write_manifest
from tests._helpers import write_clip

SIZE = (1280, 720)


def test_an_unlisted_clip_reads_its_sentinel_at_its_video_s_size(workspace_dir):
    # Read at the 1080p guess, a 720p sentinel (1279, 0) is a visible ball.
    write_clip(workspace_dir / "videos" / "rally.avi", 2, SIZE)
    write_labels(label_csv("rally"), {0: None, 1: (10.0, 20.0)}, *SIZE)
    assert label_frame_size("rally", {}) == (1280.0, 720.0)
    assert clip_labels("rally") == {0: None, 1: (10.0, 20.0)}


def test_the_manifest_row_s_size_is_taken_without_opening_the_video(workspace_dir):
    row = dict.fromkeys(MANIFEST_COLUMNS, "")
    row.update(clip="rally", width="1920", height="1080")
    write_manifest({"rally": row})
    assert label_frame_size("rally", row) == (1920.0, 1080.0)
    assert label_frame_size("nowhere", {}) == (1920.0, 1080.0)      # no row, no video: the reference
