"""Scoring the tracker and the detector against hand labels.

Two scorers live here on purpose, because they answer different questions and one of them is
frozen:

- ``evaluation.tracking`` scores an exported tracking JSON against an annotation JSON in native
  pixels (presence recall, 5/10/20 px localisation buckets, absence false positives, jumps).
  ``check_parity`` gates the bundled Pomona benchmark on its numbers, so they are not to change.
- ``evaluation.archive`` scores the raw detector or the whole pipeline on the fine-tuning
  workspace's labelled clips with ``finetune.scoring`` (1080p-normalised hit / wrong-object /
  false-alarm), the same metric training uses to pick a winner.

The repository root keeps ``validate_tracking.py``, ``check_parity.py`` and ``evaluate_archive.py``
as thin entry points to these modules, so the documented commands keep working.
"""
