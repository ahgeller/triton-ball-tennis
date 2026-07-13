"""Production tennis tracking runtime.

The heavy video pipeline imports OpenCV/TensorRT-related modules, so expose
runtime entry points lazily. Lightweight modules such as ``Config`` remain
importable for CLI/help and tests without initializing detector backends.
"""

__all__ = ["run", "Config"]


def __getattr__(name):
    if name == "run":
        from .pipeline import run

        return run
    if name == "Config":
        from .config import Config

        return Config
    raise AttributeError(name)
