"""The labelling and GridTrackNet fine-tuning workspace.

``ft.py`` is the front door: the launchers and the commands that run another script;
``menu.py`` is its interactive menu, ``commands.py`` holds the commands that work in this process
and ``interpreter.py`` finds the python with cv2 and torch to hand over to. The modules behind
them are grouped by what they own:

- ``finetune.data``: the label files and their sidecars, the workspace layout, the review
  policy, the audit and the dataset ingest (stdlib + numpy; nothing here needs cv2 or torch at
  import, which is what lets ``ft.py`` start under any interpreter);
- ``finetune.labeling``: the label tool and its suggestion models (cv2, torch);
- ``finetune.training``: the trainer, the raw prepass over a clip and the draft writer.

Modules import each other as ``finetune.<package>.<module>``. The scripts among them
(``labeling/label_tool.py``, ``labeling/cnn_refiner.py``, ``training/pretrack.py``,
``training/train_gridtracknet.py``, ``data/realign.py``, ``data/import_data.py``) can also be run
as ``python finetune/<package>/<script>.py`` from the repository root: each one puts the
repository root on ``sys.path`` only when it is the program being run, never as a side effect
of being imported.
"""
