"""The label data layer: the label file format and its sidecars (``labels``), where the
workspace keeps things (``workspace``), atomic writes (``fileio``), which clips may be trained
or scored on (``data_policy``), the detector metric (``scoring``), the label-vs-detector
agreement (``alignment``), the label audit (``label_audit``), and the two scripts that bring
clips in and re-time them (``import_data``, built on ``import_sources``, and ``realign``).

The modules ``ft.py`` imports (``labels``, ``workspace``, ``fileio``, ``data_policy``,
``scoring``, ``label_audit``) need nothing beyond the standard library at import, so ``ft.py``
loads under any interpreter; ``alignment`` (numpy) is imported where it is used, and the two
scripts are only ever run.
"""
