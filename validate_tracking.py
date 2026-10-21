"""Validate exported tennis tracking JSON against hand-labeled frames (``evaluation.validate``)."""
from evaluation.validate import main, validate   # noqa: F401 - the names scripts and tests import from here

if __name__ == "__main__":
    raise SystemExit(main())
