"""finetune/labeling/label_keys.py: the key handler's own rules - direction keys across the
backends, which keys the single-frame view claims, and q with unsaved labels. The full key table
(every key, every view) is pinned in test_label_workflow.py through Tool.handle_key."""
from unittest.mock import patch

from finetune.labeling import label_window as win
from finetune.labeling.label_keys import KeyHandler
from finetune.labeling.label_tool import Tool
from finetune.labeling.label_window import KEYS_NEXT, KEYS_PREV
from tests._helpers import fake_source, fake_tool


def tool_over():
    return fake_tool(fake_source("a", 4, 64, 36, origin=(30.0, 20.0), step=(1.0, 1.0), conf=0.9))


def test_direction_keys_match_every_backends_arrow_codes():
    assert KeyHandler.key_direction(ord("n"), ord("n")) == 1 and KeyHandler.key_direction(ord("a"), ord("a")) == -1
    assert all(KeyHandler.key_direction(code, code & 0xFF if code < 0x10000 else code) == 1 for code in KEYS_NEXT)
    assert all(KeyHandler.key_direction(code, code & 0xFF if code < 0x10000 else code) == -1 for code in KEYS_PREV)
    assert KeyHandler.key_direction(ord("q"), ord("q")) == 0


def test_the_single_frame_view_claims_its_keys_and_leaves_the_shared_ones():
    tool = tool_over()
    keys = tool.keys
    assert keys.plain_key(ord("4"), 0) and tool.session.step == 4
    assert keys.plain_key(ord("d"), 1) and tool.session.cursor == 3                  # moves at the step, clipped to the end
    assert not keys.plain_key(ord("g"), 0) and not keys.edit_key(ord("g"))  # g is common_key's
    with patch.object(win, "shift_held", return_value=False):
        keys.common_key(ord("g"))
    assert not tool.session.follow


def test_q_warns_once_about_labels_that_could_not_be_saved(capsys):
    tool = tool_over()
    with patch.object(Tool, "save", lambda self: setattr(self.session, "dirty", True)):
        assert not tool.keys.quit_key() and tool.quit_warned
        assert "could not be saved" in capsys.readouterr().out
        assert tool.keys.quit_key()                                          # the second q quits anyway
