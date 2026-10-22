# The label tool

`finetune/labeling/label_tool.py` corrects the detector's draft labels as fast as possible.

```powershell
python finetune/labeling/label_tool.py <clip>                        # videos/<clip>.mp4 + labels/<clip>_ball.csv
python finetune/labeling/label_tool.py video22..video53             # a whole run, one clip after another
python finetune/labeling/label_tool.py video22..video53 --uncertain # only the frames worth a look
python finetune/labeling/label_tool.py <clip> --grid                # start on the grid of crops
```

`ft.py label <clip>` runs the same thing with the paths filled in, and pre-labels
first when there is no draft yet.

Several clips run as one session: the last frame of one leads straight into the
first of the next, `n` and `p` cross the join, zoom carries over, and every edit is
written back to the clip the frame came from. The HUD names the clip you are in.

## A frame

Every label frame (every 2nd frame of a 60 FPS clip, every frame at 30 FPS) comes
up with the mark drawn and, on Windows, the mouse cursor already sitting on it. If
the mark is on the ball, press SPACE (or ENTER / `c`). If it is not, click where the
ball is. Both move to the next frame. The view keeps the ball in the middle and the
mouse goes with it, so the pointer is always sitting on the next mark. The mark is
dead centre on every frame, even one at the very edge of the picture: the view runs
off the frame there and that strip is drawn black; zoom stays where you set it. The
window is resizable (drag it, maximise it, or press `m` for full screen) and the
picture is redrawn at whatever size you give it.

A frame with no guess at all (the detector missed) gets a dashed "guess" ring where
the labels either side say the ball should be; the view centres there and the mouse
is parked on it, so if it is right a click is all it takes. Nothing is written until
you do click.

## Keys

| key | what it does |
|---|---|
| left click | the ball is here: save and go to the next frame |
| SPACE / ENTER / `c` | the mark is good: next frame. On an unreviewed frame with a hint on offer, the mark *is* the hint and SPACE moves the label onto it |
| `v` | the ball is not visible on this frame: next frame |
| `x` | the draft is right, not the hint on offer: confirm the label where it is and go on (on a reviewed frame the same as SPACE) |
| `y` | take the suggestion: move the point to the orange mark and go on, like a click there (see *Suggestions*) |
| `r` | accept the whole run of confident guesses up to the next iffy one |
| `n` / `d` / right arrow | next frame |
| `p` / `a` / left arrow | previous frame |
| `1` `2` `4` `6` `8` | how many frames those four move at a time; the x1..x8 buttons on the bar do the same. Holding `d` or `a` at x8 covers ground eight times as fast; SPACE, `v` and a click always move exactly one frame |
| `f` / `b` | jump to the next / previous unreviewed uncertain frame, and to any frame an audit (`ft.py check --audit --mark`) has flagged, drawn in magenta, even one you had reviewed before the audit. SPACE answers a flag as well as a click does (it is good where it is), and a later audit does not raise it again unless the label moves |
| `e` / SHIFT+`e` | jump to the next / previous frame whose label the suggestion disagrees with ("hint differs" on the bar), reviewed or not |
| `i` / `j` / `k` / `l` | nudge the point one pixel up / left / down / right (stays on the frame); with SHIFT held, a quarter of a pixel |
| `u` | undo the last change on this frame's label (not a landing mark: `/` again removes one) |
| `/` | this frame is a landing: the ball is on the ground here. Pressed again, it is not. The frame wears a LANDING badge (its grid tile too) and the bar counts the clip's landings. Pressed on the frame after (or before) one already marked, the ball bounced between the two: the two marks become one landing between them and both frames wear LANDING 1/2. When the landing has no ball label under it (one between two frames, or one on a frame labelled "no ball") the bar asks for a click where the ball touched the ground. Saved with the labels to `labels/<clip>_bounces.csv` (see *Landings*) |
| mouse wheel, `+` / `-` | smooth zoom in / out about the mouse / the ball (kept across frames) |
| right click | re-centre the view here |
| `g` / click the badge | auto-centre on / off. On (the default) every frame puts the mark in the middle of the screen with the mouse on it; off leaves the view and the mouse alone, for looking round a frame |
| `h` / click the badge | cycle the view filter: OFF, DIM WHITE (court lines, white kit and sky pulled down to a dull blue-grey so a yellow ball stops hiding on them), BALL POP (only tennis-ball yellows keep their colour), MOTION (what moved since the previous label frame lights up magenta). Remembered between sessions in `labels/label_tool.settings.json` |
| `o` / click the badge | overview inset on / off: while zoomed in, a small copy of the whole frame sits bottom-right with the part on screen boxed and the ball dotted. Click anywhere in it to bring that spot into the zoomed view, the quick way to reach a ball the detector put somewhere else entirely |
| TAB | toggle the grid: a page of crops around each ball (see *The grid*) |
| `w` | the trajectory sheet (see *The trajectory sheet*) |
| `m` | full screen on / off (the window also maximises normally) |
| `t` | toggle the trail of the previous labels |
| `s` | save |
| `q` / ESC | save and quit |

## The grid

TAB (or `--grid`) shows a page of crops around each ball, 8 or 16 to a page (`z`
switches). Click a crop to correct it without moving the crop; right click opens the
full frame; `1`-`9` or `,` / `.` pick the tile the keys act on; `y` takes its
suggestion and moves on, `x` keeps its draft instead; `v` marks it absent; `i` / `j`
/ `k` / `l` nudge it; `/` marks it as a landing; SPACE confirms the displayed page,
hints on offer (orange crosses) included; `a` / `d` change pages; `+` / `-` zoom the
crops.

## The trajectory sheet

`w` shows the whole frame with every label of 60 label frames drawn on it, joined
in order, and the suggestions as orange dots: a label off the ball's path stands
out at once. Click a point to open that frame; `a` / `d` move to the stretch before
/ after; `w` or right click goes back. It only finds frames, never edits.

## Suggestions

While you work, the tool measures each label frame's ball from the video itself:
the centre of what moved between that frame and its neighbours, looked for around
the current label and around where the labels either side put the ball
(`finetune/refiner/refine.py`). A small network trained on the frames you have reviewed
(`finetune/refiner/cnn_refiner.py`; "hint model" in the finetune menu retrains it) then puts
it where you would click and says whether it is sure. Without a trained version the
measurement is shown as it is.

On a frame you have not reviewed yet the suggestion is the mark: it is drawn as the
ball circle in orange, bright when it is sure, dull when only a guess; the mouse
parks on it, and SPACE (or a click without moving) takes it; the draft label it
replaces is ticked in grey beside it. So an unreviewed frame asks one thing, "is the
mark on the ball?": SPACE if it is, click where the ball is if not, `x` if the grey
draft was right after all. A sure suggestion is always offered; an unsure one is
offered only when it agrees with the draft within 3 px, where it is still the better
mark, and the dull colour says to look closely. The measurements behind those rules
are in `Suggestions.offer_for` (`finetune/labeling/label_hints.py`).

Once you have reviewed or edited a frame your label is the mark and the suggestion
is a hint beside it, four small ticks with its distance on the bar, and `y` moves the
label there. A label taken from a suggestion is recorded as source "refined". The
clip you are in is re-measured a moment after you change or confirm a label,
steered by the labels you have reviewed, so fixing a frame helps its neighbours. A
frame labelled "no ball" gets a suggestion only when the video clearly shows a ball
on its path. Where a trusted suggestion sits beside a label, the bar says whether
they agree: "hint ok" (within 3 px) or "hint differs" (further), and `e` walks
those. Agreeing never skips a frame or marks it reviewed; that is still your SPACE.
`r` (accept run) confirms the detector's own guesses, not the suggestions: it is for
frames you are not looking at.

## Landings

The court events a `--court` run exports are scored against the frames where the
ball hit the ground (`score_court_events.py`), and those frames are labelled here.
Step to the frame where the ball is on the ground and press `/`: the frame wears a
LANDING badge, the bar says `LANDING` beside its state and counts the clip's
landings, and `/` again unmarks it. Mark every landing of every rally, serves and
faults included; on a 60 FPS clip the label frame nearest the landing is close
enough (the scorer allows 0.1 s). The marks are saved with the labels to
`labels/<clip>_bounces.csv` with blank coordinates: the landing point is the ball
label of that frame (`finetune/data/bounce_labels.py`), so it follows the label if
you move it. A clip with no landings marked has no landing file, and the scorer
leaves it out; a file written by hand with coordinates keeps them.

The ball often lands between two frames: in the air on one, already up on the
next. Press `/` on both (the frame already marked, then `/` on the next or the
previous one): the two marks become one landing between them, saved as
`frame_0412.5` (on a 60 FPS clip labelled every other frame, the unlabelled frame
between the two), and both frames wear LANDING 1/2. `/` on either of the two frames
removes the landing.

A landing with no ball label under it - one between two frames, or one on a frame
labelled "no ball" (the ball hidden behind a player, say) - has no label to take its
point from, so the bar asks you to click where the ball touched the ground: the
scuff, or the spot under the ball. That click is the landing point, drawn as a
small cross and saved as the row's coordinates, and the frame stays up. Any other
key instead leaves the landing without a point, and the scorer takes the ball
labels around it: for a landing between two frames the midpoint of the two frames'
ball labels, or the one of them there is; for a landing on a frame without a ball
label the label of the frame before it, else of the frame after (`resolve_points`
in `finetune/data/bounce_labels.py`). From the grid, `/` marks and joins the same
way but asks for no click.

## `--uncertain`

`--uncertain` walks only the frames that want attention (no confident guess, or
flagged by `ft.py check --audit --mark`) and skips confident predictions, which may
still be wrong. `n` / `p` / SPACE step between those frames; the rest of the labels
are untouched and still saved. Imported CSVs without confidence are treated as
unknown, not as perfect detections. Use the regular or `--grid` view to check small
centering errors in every imported label.

## Output

`labels/<clip>_ball.csv` (`frame,ball_x,ball_y`). Review progress lives in
`labels/<clip>_ball.review.json` so you can quit and resume, and the landings in
`labels/<clip>_bounces.csv` (see *Landings*). Everything is autosaved
every few seconds and on any exit; a frame you already dealt with shows a REVIEWED
(space) or EDITED (click / `v` / nudge) badge in the top-right corner when you come
back to it, and the top-left corner counts how many frames are left to do.
