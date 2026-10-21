"""The label tool and what it draws on: ``label_tool`` (the tool, run as a script), split into
``label_store`` (a clip's rows and sidecars), ``label_source`` (one clip of a labelling run),
``label_window`` (window plumbing) and ``label_hints`` (the suggestions); ``refine`` measures the
model-free suggestions and ``cnn_refiner`` is the small network that places them. Every key and
what the marks mean: ``LABEL_TOOL.md`` beside these.
"""
