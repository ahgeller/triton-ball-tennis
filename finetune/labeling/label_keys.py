"""What the label tool's keys do: the guards every key passes, q, and the single-frame view's
keys, with the keys the grid shares. The grid's and the sheet's own keys are in ``label_grid`` and ``label_sheet``.

Every key names a tool action (``tool.confirm``, ``tool.move``, ``tool.jump``...): that is the
surface the key table in the tests pins, so the handler calls the tool by those names.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from finetune.labeling import label_window as win
from finetune.labeling.label_window import (KEY_TAB, KEYS_CONFIRM, KEYS_NEXT, KEYS_NUDGE, KEYS_PREV, KEYS_QUIT, KEYS_STEP,
                                            KEYS_ZOOM_IN, KEYS_ZOOM_OUT)

if TYPE_CHECKING:
    from finetune.labeling.label_tool import Tool


class KeyHandler:
    def __init__(self, tool: "Tool") -> None:
        self.tool = tool

    def handle(self, key: int) -> bool:
        """One key from waitKeyEx; True when the tool should quit. The guards and q are here; what
        a key does belongs to the view showing it (SheetView, GridView, plain_key), and the grid and
        the single frame then share common_key."""
        tool = self.tool
        code = key & 0xFF if key < 0x10000 else key
        if win.non_character_key(code):
            return False          # F1 is not p, F2 is not q: see NON_CHARACTER_KEYS
        if not tool.overview and not tool.view_ready() and code not in KEYS_QUIT:
            return False  # a mouse callback may have advanced while waitKey dispatched events
        if code in KEYS_QUIT:
            return self.quit_key()
        direction = self.key_direction(key, code)
        if tool.sheet:
            tool.sheet_view.handle_key(code, direction)
        elif tool.grid:
            tool.grid_view.handle_key(code, direction) or self.common_key(code)
        else:
            self.plain_key(code, direction) or self.common_key(code)
        return False

    @staticmethod
    def key_direction(key: int, code: int) -> int:
        """1 for a forward key (n, d, right arrow), -1 for a backward one (p, a, left arrow), else 0.
        Arrows: this build hands back the bare virtual-key codes (37/39), older Win32 builds pack
        them above bit 16, GTK and Qt have codes of their own. Match them all; none of them
        collide with a printable key the tool uses."""
        if code in (ord("n"), ord("d")) or key in KEYS_NEXT or code in KEYS_NEXT:
            return 1
        if code in (ord("p"), ord("a")) or key in KEYS_PREV or code in KEYS_PREV:
            return -1
        return 0

    def quit_key(self) -> bool:
        """q / ESC: save, and quit - unless labels could not be saved and that has not been said."""
        tool = self.tool
        tool.save()
        if tool.dirty and not tool.quit_warned:
            tool.quit_warned = True
            print("[warn] some labels could not be saved (see above). Close the file in the other "
                  "program and press q again; quitting anyway keeps them in <clip>_ball.unsaved.csv")
            return False
        return True

    def plain_key(self, code: int, direction: int) -> bool:
        """What a key does in the single-frame view; False for a key it shares with the grid."""
        tool = self.tool
        if self.edit_key(code):
            return True
        if code == ord("e"):
            tool.seek_disputed()
        elif code in (ord("f"), ord("b")):
            tool.jump(1 if code == ord("f") else -1)
        elif direction:
            tool.move(direction * tool.step)
        elif code in KEYS_STEP:
            tool.step = KEYS_STEP[code]
        elif code in KEYS_ZOOM_IN + KEYS_ZOOM_OUT:
            tool.zoom_key(1.0 if code in KEYS_ZOOM_IN else -1.0)
        else:
            return False
        return True

    def edit_key(self, code: int) -> bool:
        """The single-frame keys that say something about this frame's label and move on:
        SPACE, y, x, v and r. False for any other key."""
        tool = self.tool
        if code in KEYS_CONFIRM:
            tool.confirm()
            tool.move(1)
        elif code == ord("y"):
            if tool.take_suggestion():
                tool.move(1)
        elif code == ord("x"):
            tool.confirm(take_offer=False)    # the draft is right, not the hint on offer
            tool.move(1)
        elif code == ord("v"):
            tool.mark_invisible()
            tool.move(1)
        elif code == ord("r"):
            accepted = tool.accept_run()
            print(f"[run ] accepted {accepted} confident frame(s); stopped at frame "
                  f"{tool.rows[tool.cursor].frame}")
        else:
            return False
        return True

    def common_key(self, code: int) -> None:
        """The keys the grid and the single frame read the same way."""
        tool = self.tool
        toggles = {ord("w"): tool.toggle_sheet, KEY_TAB: tool.toggle_grid, ord("z"): tool.toggle_grid_size,
                   ord("t"): tool.toggle_trail, ord("g"): tool.toggle_follow, ord("h"): tool.cycle_filter,
                   ord("o"): tool.toggle_inset, ord("m"): tool.toggle_fullscreen, ord("s"): tool.save}
        if code in toggles:
            toggles[code]()
        elif code == ord("u"):
            tool.rows[tool.cursor].undo()
            tool.dirty = tool.source.dirty = True
            tool.follow_ball()
        elif code in KEYS_NUDGE:
            dx, dy = KEYS_NUDGE[code]
            step = .25 if (chr(code).isupper() or win.shift_held()) else 1.
            tool.nudge(dx * step, dy * step)
