# Precision label review — September 8, 2026

The user's main task is correcting slightly misplaced centers across many
existing CSV labels, with variable footage quality. This pass adds human review
tools rather than automatically smoothing, propagating or accepting coordinates.

## Use

Launch the existing labeling tool with `--grid`, for example:

```powershell
& 'C:\Users\Andrew\.conda\envs\tennis-analysis\python.exe' finetune/label_tool.py video13 --grid
```

The same option passes through `finetune/ft.py label <clip> --grid`.
Substitute the clip you actually want to edit; this example does not run itself.

- **Tab:** switch between one full-frame view and eight enlarged crops.
- **Grid left click:** correct that tile's ball center without advancing or
  moving the crop underneath the pointer. The blue border identifies selection.
- **Grid right click:** inspect the selected frame in the normal full view.
- **Space:** confirm only the displayed group and advance to the next page.
  Inspect every tile first; this is your approval, not an automatic quality check.
- **A/D:** previous/next page without approving it. **+/-:** crop zoom.
- **I/J/K/L:** up/left/down/right by one native pixel; hold **Shift** for 0.25 px.
- **U:** undo the selected row's last action; **V:** selected ball absent.

Crop centers stay fixed while coordinates change. Native pixel enlargement and
an open-center marker help inspect small offsets without covering the center.
Extra zoom or decimal coordinates cannot recover detail missing from blurry or
compressed footage. Mixed-resolution clips can be reviewed in the same session;
each page stays within one clip and saves native coordinates to its own CSV.

## Fixes and evidence

- Frame navigation cancels the previous zoom anchor and pending cursor feedback,
  finishes its zoom animation, then centers the new target.
- A displayed-frame/transform check blocks queued mouse or keyboard edits from
  modifying a newly selected frame before that image is displayed. Native Qt
  wheel compensation still runs for a rejected stale wheel event.
- Resizing clears stale zoom anchors. Crop pages also invalidate old hit boxes
  immediately, so rapid input cannot approve or edit an unseen next page.
- Undo restores coordinates, visibility and prior reviewed/edited/suspect flags.
- Confidence/source evidence survives saves in the review sidecar. Imported
  CSVs with no confidence no longer become fabricated confidence-1 detections.
- Accept-run marks each affected source dirty, honors filtered navigation, and
  includes the final row. Missing frames remain protected from edits/approval.
- Recently revisited frames refresh their cache position; cache limits respect
  resolution, and switching clips releases the previous clip's decoded cache.
- Rendered crop pages are reused until their content or geometry changes.

`audit/test_label_workflow.py` contains 10 passing synthetic tests for zoomed
navigation, queued clicks, resizing, wheel handling, crop coordinate mapping,
unseen/missing pages, mixed resolutions, undo, evidence persistence and multi-clip
acceptance. The existing 16 regression checks also pass. A synthetic sharp/blurred
crop preview was rendered and inspected. No real label files were modified by
these tests, and no unfinished clips were used for comparisons or model selection.

The shared worktree also contains newer window, seek, display-filter and labeling
changes made during this work; these were retained. The headless tests establish
geometry and state behavior, not a complete live Qt/Windows/DPI verification.
Live cursor placement and the window's independent native zoom still need a
hands-on check in the user's actual display setup. The UI has not been launched
against the user's active editing session.
