"""The label tool and what it draws on: ``label_tool`` (the tool, run as a script: its state,
navigation and edits), split into ``label_cli`` (its command line), ``label_store`` (a clip's rows
and sidecars), ``label_source`` (one clip of a labelling run, and decoding ahead of the cursor),
``label_window`` (window plumbing and the key tables), ``label_cursor`` (parking the OS pointer on
the ball), ``label_keys`` (what the keys do), ``label_mouse`` (what the mouse does),
``label_frame_view`` (the single-frame picture and HUD), ``label_grid`` and ``label_sheet`` (the grid
and the trajectory sheet, each owning its state, picture and keys), ``label_draw`` (what every
view draws with) and ``label_hints`` (the suggestions, measured by ``finetune.refiner``). Every key
and what the marks mean: ``LABEL_TOOL.md`` beside these.
"""
