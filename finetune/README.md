# Fine-tuning workspace

Drop videos in, let the detector pre-label them, fix the few frames it got
wrong, train, keep the best model. Everything here uses the same label format
as `gridtracknet_finetuning/archive` (`frame,ball_x,ball_y`, invisible ball
parked top-right), so old and new clips train together.

```
finetune/
  videos/     <clip>.mp4              you add these (30 or 60 FPS, any resolution)
  labels/     <clip>_ball.csv.draft   written by pretrack.py (detector guess + confidence per frame)
              <clip>_ball.csv         written by label_tool.py  (the training label)
              <clip>_ball.review.json your review progress + zoom
  models/     gridtracknet_best.npz   written by train_gridtracknet.py / the Colab notebook
  runs/       tracker JSON per clip, training logs
  cache/      extracted 768x432 training frames (safe to delete)
```

Run every command with the finetuning venv:
`$py = "C:\Users\Andrew\Desktop\gridtracknet_finetuning\.venv\Scripts\python.exe"`

## Frame by frame or every other?

The network looks at **5 frames spaced at 30 FPS** — it needs that much
displacement between frames to see the ball as a moving thing. So:

- 60 FPS video → labels on **every second frame** (`frame_000, frame_002, …`).
- 30 FPS video → labels on **every frame**.

`pretrack.py` and `label_tool.py` only ever show you those frames. The
*tracker output* is still per frame at 60 FPS (it runs the even and odd frame
streams separately). Your existing archive follows the same rule.

## Workflow

1. **Pre-label**
   `& $py finetune\pretrack.py` — every video in `videos/` without a final CSV
   gets `labels/<clip>_ball.csv.draft` with the detector's guess for every
   label frame and its confidence (`--source raw`, the default). Low-confidence
   guesses are kept on purpose: you accept or move a point, you never hunt.
   `--weights some.npz` pre-labels with a different model (use your latest
   fine-tune here — the better the model, the less you click),
   `--source pipeline` uses the full tracker instead, `--clips a b` picks
   clips, `--force` redoes finished ones.

2. **Correct** — `& $py finetune\label_tool.py <clip>`
   Each label frame comes up zoomed on the guess, with the mouse cursor
   already on it.
   - guess is right → **SPACE** (next frame)
   - guess is wrong → **click** on the ball (saves and goes to next frame)
   - ball not visible → **v** (next frame)
   - **mouse wheel** zooms and the zoom stays; the view follows the ball
     between frames; right-click re-centres
   - `f` / `b` jump to the next / previous unreviewed *uncertain* frame
     (no guess or confidence < 0.7) if you only want to check the doubtful ones
   - `i j k l` nudge one pixel, `u` undo, `n`/`p` step, `s` save, `q` save + quit
   You can quit any time; progress and zoom are remembered.
   Output: `labels/<clip>_ball.csv`.

3. **Check** (optional) — `& $py evaluate_archive.py --mode raw --archive finetune`
   scores the current detector against your new labels (recall / wrong / false-alarm).

4. **Train**
   - Locally: `& $py finetune\train_gridtracknet.py --val-clips <clip>`
     (fine-tunes `models/gridtracknet_weights_torch.npz`); `--from-scratch`
     trains a new network instead.
   - Colab: open `GridTrackNet_FineTune_Colab.ipynb`; put `videos/`,
     `labels/`, `models/` and `code/` (`train_gridtracknet.py` +
     `tennis_tracker/gridtracknet.py`) under `MyDrive/tennis_finetune/`,
     run top to bottom.

   Either way the run ends by scoring the trained checkpoint **and** the
   starting weights on the held-out clip(s) with the detector metric, and only
   the winner is written to `--save` (`finetune/models/gridtracknet_best.npz`).
   A run that lowers recall or adds wrong-object detections cannot replace a
   good model. `runs/<tag>/winner.json` records the decision.

5. **Use it** — copy the winner over `models/gridtracknet_weights_torch.npz`
   in the repo (rename the old one first), re-run the archive eval, then go
   back to step 1 with `--weights` pointing at it.

## Tips

- Label whole rallies, including frames where the ball is hidden or out of
  frame (`v`). Those "no ball here" frames are what stop the network firing on
  players' feet, court lines and vans.
- Keep at least one full clip held out (`--val-clips`) and never train on it —
  that is the only honest number you get.
- Mixed resolutions are fine; coordinates are native pixels and the trainer
  resizes to 768×432 internally.
