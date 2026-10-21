"""The label tool and what it draws on: ``label_tool`` (the tool, run as a script), split into
``label_store`` (a clip's rows and sidecars), ``label_source`` (one clip of a labelling run),
``label_window`` (window plumbing and the key tables), ``label_views`` (the grid and the trajectory
sheet, each owning its state, picture and keys), ``label_draw`` (what every view draws with) and
``label_hints`` (the suggestions); ``refine`` measures the model-free suggestions,
``cnn_refiner`` is the small network that places them and ``refiner_training`` trains it. Every key and what the marks mean:
``LABEL_TOOL.md`` beside these.
"""
