"""Strict structural and quality gate for the bundled Pomona benchmark (``evaluation.parity``)."""
from evaluation.parity import BASELINE, SMOOTHNESS_LIMITS, check, main   # noqa: F401

if __name__ == "__main__":
    raise SystemExit(main())
