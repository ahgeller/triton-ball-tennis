"""now_main package exports.

The heavy video pipeline imports OpenCV/TensorRT-related modules, so expose the
legacy package-level names lazily.  This keeps lightweight sidecar modules
importable in environments that only need parsing or JSON post-processing.
"""

__all__ = ["main", "run", "Config"]


def __getattr__(name):
    if name in {"main", "run"}:
        from .app import main, run

        return {"main": main, "run": run}[name]
    if name == "Config":
        from .config import Config

        return Config
    raise AttributeError(name)
