# Fine-tuning workspace

Drop videos in, let the detector pre-label them, fix the few frames it got
wrong, train, keep the best model.

**New here? Double-click `finetune.bat`** (or run `finetune` in cmd,
`.\finetune.ps1` in PowerShell, with no arguments) — an interactive menu shows
what is labelled and walks through add / label / check / eval / train / promote.

The same actions exist as commands for scripts and power use:

```powershell
.\finetune.ps1 status                 # or:  python finetune\ft.py status   (any python; it re-runs itself in the venv)
```

| command | what it does |
|---|---|
| `ft.py status` | every clip by source, own-camera clips with review progress, which model the tracker uses |
| `ft.py import all` | pull every labelled clip that exists on this PC into the workspace (see *Data* below) |
| `ft.py add C:\clips\rally7.mp4` | copy a new video in (30 or 60 FPS) and pre-label it |
| `ft.py label rally7` | open the click tool; pre-labels first if there is no draft yet |
| `ft.py check` | validate every label file against its video (cadence, range, columns); `--audit [--fix]` cross-checks with the detector |
| `ft.py camera` | measure camera motion per clip and tag `clips.csv` `fixed` / `moving` (the background channel only learns from fixed) |
| `ft.py eval` | raw detector recall / wrong / false-alarm on the own-camera clips (`--all`, `--clips`, `--weights`) |
| `ft.py train ...` | fine-tune or train from scratch; keeps the better of new vs starting weights |
| `ft.py promote` | make the winner the tracker's model (`models/gridtracknet_weights_torch.npz`, old one backed up) |
| `ft.py paths` | which python, which folders |

Everything that is not `ft.py` is still callable directly (`pretrack.py`,
`label_tool.py`, `train_gridtracknet.py`, `import_data.py`); `ft.py` just picks
the interpreter and fills in the paths. Unknown options pass straight through
(`ft.py train --epochs 3 --lr 0.3`).

## Layout

```
finetune/
  videos/     <clip>.mp4              30 or 60 FPS, any resolution           (gitignored)
  labels/     <clip>_ball.csv         the training label: frame,ball_x,ball_y (native px; invisible = top-right corner)
              <clip>_ball.csv.draft   detector guess + confidence per frame, written by pretrack.py
              <clip>_ball.review.json your review progress + zoom
  clips.csv   one row per clip: source, group, size, fps, stride, label counts (written by import_data.py)
  models/     gridtracknet_best.npz   the winner of the last training run
  runs/       <tag>/results.csv, config.json, winner.json, best.npz, last.npz
  cache/      768x432 JPEG frames, one folder per clip; rebuilt per clip when its video/labels change (safe to delete)
```

Own labels (`video*_ball.csv`) are tracked in git; imported public labels
(`tnv2_*`, `grid_*`) and all videos are not — `ft.py import all` recreates them.

## Data (after `ft.py import all`)

| source (`clips.csv`) | clips | label frames | visible | what it is |
|---|---:|---:|---:|---|
| `custom` | 11 | 4,106 | 3,844 | your archive clips video1–10, 12 (fixed camera, 1080p; video12 is 720p) |
| `custom-uncorrected` | 1 | 300 | 245 | `video11`: 768×432 PNGs from the TrackNetV5 prep, labels never click-corrected — hold-out/eval only |
| `grid` | 32 | 17,509 | 16,217 | a partial copy of GridTrackNet's public set (amateur phone + TV clips, 1080p); the full 100-match set is downloadable (`DATASETS.md` #1). 21 local matches ship labels without media; a few have labels from a different cut of the video — `ft.py check --audit` finds them |
| `tracknetv2-badminton` | 201 | 91,214 | 78,952 | **not imported by `all`** (removed 2026-08-31 as unneeded — `ft.py import tracknetv2` brings it back): the TrackNetV2 **badminton** set (V5Test's README called it tennis — it is not), broadcast shuttlecock rallies, 1280×720 @ 30 FPS. Cross-sport "tiny fast object" pre-training only |

Sources are read from `V5Test/` and `gridtracknet_finetuning/archive` (paths in
`import_data.py`); the importer converts TrackNet-style `Frame,Visibility,X,Y`
labels, scales GridTrackNet's 1280×720 labels to the 1080p videos, and maps its
60 FPS matches to frames 1, 3, 5, … (that is how its `FrameGenerator.py` sampled
them — verified against the shipped PNGs). Visibility 3 (occluded, estimated
position) becomes "not visible" unless `--keep-occluded`.

`ft.py check --audit` runs the detector over every tennis clip and reports how
often it agrees with the labels, trying temporal shifts of ±4 label frames. A
clip whose labels are simply offset is shifted with `--fix`; one whose labels
belong to another cut of the video is written to `exclude.txt`, which
`train_gridtracknet.py` honours by default (`--include-excluded` overrides).

Any downloaded dataset in a TrackNet layout (`Label.csv` beside frames, or
`csv/*_ball.csv` + `video/*.mp4`) imports the same way:

```powershell
.\finetune.ps1 import tracknet --src D:\datasets\tracknet_tennis --prefix tn --fps 30
```

`DATASETS.md` ranks what is worth downloading.

## Frame by frame or every other?

The network looks at **5 frames spaced at 30 FPS** — it needs that much
displacement between frames to see the ball as a moving thing. So:

- 60 FPS video → labels on **every second frame** (`frame_000, frame_002, …`,
  or `001, 003, …` — either phase works).
- 30 FPS video → labels on **every frame**.

`pretrack.py` and `label_tool.py` only ever show you those frames. The *tracker
output* is still per frame at 60 FPS (it runs the even and odd frame streams
separately). `ft.py check` flags files that break this rule.

## Workflow: processing a new video

1. **Add + pre-label**

   ```powershell
   .\finetune.ps1 add "C:\wherever\night_session.mp4"      # --name night1 to pick the clip name
   ```

   Copies the video into `videos/` and immediately pre-labels it: the current
   detector runs over the whole clip (~1–2 min per minute of video) and writes
   `labels/<clip>_ball.csv.draft` with a guess + confidence for every label
   frame. Low-confidence guesses are kept on purpose so you accept or move a
   point, never hunt. The video must be **30 or 60 FPS** (any resolution) —
   if it is not, the command prints the exact `ffmpeg -vf fps=30` line to
   re-encode first. One rally per clip works best (that is what the archive
   is), but longer videos are fine. You can also just drop files into
   `videos/` and run `.\finetune.ps1 prelabel`.

   Pre-label with your **newest** model — the better the model, the less you
   click: `.\finetune.ps1 prelabel --clips night1 --weights
   finetune\models\gridtracknet_best.npz` (`--source pipeline` uses the full
   tracker instead of the raw detector).

2. **Correct** — `.\finetune.ps1 label night1`. Each label frame comes up
   zoomed on the guess with the mouse cursor already on it.
   - guess is right → **SPACE** (next frame)
   - guess is wrong → **click** on the ball (saves and goes to next frame)
   - ball hidden or out of frame → **v** (next frame) — do label these, the
     "no ball here" frames are what stop the network firing on feet and vans
   - `f` / `b` jump to the next / previous unreviewed *uncertain* frame
     (no guess or confidence < 0.7) if you only want to check the doubtful ones
   - **mouse wheel** zooms smoothly about the mouse and the zoom stays; each
     new frame comes up centred on the guess; right-click re-centres
   - `i j k l` nudge one pixel, `u` undo, `n`/`p` step, `s` save, `q` save + quit
   You can quit any time (even by closing the window or Ctrl+C) — everything is
   autosaved every few seconds. A frame you already handled shows a green
   REVIEWED (space) or orange EDITED (click / v / nudge) badge in the top-right
   corner when you come back to it, and the top-left corner counts down how
   many frames are left to do. A streaked
   (motion-blurred) ball gets the click at the **centre of the streak**.
   Output: `labels/night1_ball.csv`. A rally is typically ~15–20 min of
   clicking, much less when the detector is already good on that footage.

3. **Sanity-check**

   ```powershell
   .\finetune.ps1 check --clips night1     # cadence / coordinate range / columns
   .\finetune.ps1 camera --clips night1    # tag fixed/moving in clips.csv (background channel needs this)
   .\finetune.ps1 eval --clips night1      # how the current detector scores on it (--weights for a candidate)
   ```

4. **Train** — `ft.py train`. Defaults: fine-tune the tracker's current
   weights on every clip in the workspace, hold out `video10`, 10 epochs.
   Useful recipes:

   ```powershell
   .\finetune.ps1 train --sources custom --val-clips video10 video11            # own footage only
   .\finetune.ps1 train --sources custom grid --oversample custom=6 --val-clips video10 video11   # tennis only, own footage x6
   .\finetune.ps1 train --sources custom grid --cameras fixed --augment strong --val-clips video10 video11   # fixed cameras, lighting jitter
   .\finetune.ps1 train --oversample custom=6 tracknetv2-badminton=0.25 --epoch-units 20000 --val-clips video10 video11
   .\finetune.ps1 train --from-scratch --epochs 30 --epoch-units 40000           # new network on everything
   .\finetune.ps1 train --val-clips video10 "grid_match4*" --run-dir finetune\runs\mixed --lr 0.3
   ```

   - `--sources` picks sources from `clips.csv`; `--cameras fixed` keeps only
     clips tagged by `ft.py camera` (moving-camera clips are fine for the
     RGB-only GridTrackNet but useless for a background-difference channel);
     `--exclude-clips` drops stems or globs; `--oversample source=weight`
     changes the draw probability per source (0 removes it from training);
     `--epoch-units N` fixes how many windows an epoch draws (an epoch over
     everything is ~50k windows, about an hour on the 5080).
   - `--augment strong` adds gamma / contrast / brightness / colour-cast /
     noise / blur jitter, applied identically to all five frames of a window,
     to cover dark courts and other lighting you have not labelled yet. The
     default `basic` is the old behaviour (flip + ±15% gain).
   - Frames are cached per clip under `cache/`, so adding one clip does not
     re-extract the other 244. `--prepare-only` builds the cache and stops.
   - The run ends by scoring the best-loss checkpoint **and** the starting
     weights on the held-out clips with the detector metric, and only the
     winner is written to `models/gridtracknet_best.npz`. A run that lowers
     recall or adds wrong-object detections cannot replace a good model.
     `runs/<tag>/winner.json` records the decision.
   - Colab: `GridTrackNet_FineTune_Colab.ipynb` runs the same trainer; put
     `videos/`, `labels/`, `clips.csv`, `models/` and `code/` under
     `MyDrive/tennis_finetune/`.

5. **Use it** — `.\finetune.ps1 promote` copies the winner over
   `models/gridtracknet_weights_torch.npz` (the old file is kept as
   `*_prev_<timestamp>.npz`) — only worth doing when `runs/<tag>/winner.json`
   says the new weights won. Then `.\finetune.ps1 eval` and
   `python check_parity.py`. Go back to step 1 with `--weights` pointing at
   the new model. `.\finetune.ps1 status` at any point shows what is
   labelled, what is still a draft, and which model the tracker is using.

## Tips

- Label whole rallies, including frames where the ball is hidden or out of
  frame (`v`). Those "no ball here" frames are what stop the network firing on
  players' feet, court lines and vans.
- Keep at least one full own-camera clip held out (`--val-clips`) and never
  train on it — that is the only honest number you get. video10 and video11 are
  the ones the bundled model has never seen.
- Mixed resolutions are fine; coordinates are native pixels and the trainer
  resizes to 768×432 internally.
- Only ever compare models on the same held-out clips; `ft.py status` shows the
  last runs' validation clips next to their numbers.
