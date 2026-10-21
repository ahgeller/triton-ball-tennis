"""Make the repository root importable so tests see ``tennis_tracker`` as the CLI does, and the
fixtures every suite shares. Tests are plain pytest functions with fixtures and ``parametrize``;
``unittest.TestCase`` is not used."""
import contextlib
import sys
from pathlib import Path

import pytest

# The bootstrap: pytest.ini's pythonpath does the same, but a test run started another way
# (an IDE on one file, say) still needs the repository on sys.path before the project imports.
_REPOSITORY = Path(__file__).resolve().parent.parent
if str(_REPOSITORY) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY))

from tennis_tracker.paths import ROOT  # noqa: E402  (needs the bootstrap above)

assert ROOT == _REPOSITORY, "tests/ must sit directly under the repository tennis_tracker.paths names"


@pytest.fixture
def workspace_dir(tmp_path, monkeypatch):
    """Point the fine-tuning workspace (``finetune.data.workspace``) at a fresh folder under tmp_path,
    with empty ``labels/`` and ``videos/`` and the tracker's weights redirected to ``models/tracker.npz``
    under it. The finetune modules read every path from ``workspace.current()`` when called, so
    replacing that one function moves all of them; no real label, video or model is touched.
    Returns the folder."""
    from finetune.data import workspace

    root = tmp_path / "workspace"
    (root / "labels").mkdir(parents=True)
    (root / "videos").mkdir()
    relocated = workspace.Workspace(root, repo_weights=root / "models" / "tracker.npz")
    monkeypatch.setattr(workspace, "current", lambda: relocated)
    return root


class _NoOpStream:
    """Stands in for ``torch.cuda.Stream``; every method is a no-op."""

    def __init__(self, *args, **kwargs):
        pass

    def synchronize(self):
        return None

    def wait_stream(self, other):
        return None

    def wait_event(self, event):
        return None

    def record_event(self, event=None):
        return event if event is not None else _NoOpEvent()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _NoOpEvent:
    """Stands in for ``torch.cuda.Event``; ``record``/``synchronize``/``wait`` do nothing."""

    def __init__(self, *args, **kwargs):
        pass

    def record(self, stream=None):
        return None

    def synchronize(self):
        return None

    def wait(self, stream=None):
        return None

    def query(self):
        return True


@pytest.fixture
def fake_cuda(monkeypatch):
    """Let CUDA-only code run on the CPU.

    ``torch.device("cuda:N")`` resolves to the CPU device, ``torch.cuda``
    reports one available device, and its streams, events and synchronisation
    become no-ops.  Pinned-memory allocation is left alone: on a machine
    without a CUDA driver it raises, which is the fallback path the pipeline
    already handles.  Returns the real ``torch.device`` so a test can build
    one when it needs it.
    """
    import torch

    real_device = torch.device

    def device(*args, **kwargs):
        spec = args[0] if args else kwargs.get("type")
        if isinstance(spec, str) and spec.startswith("cuda"):
            return real_device("cpu")
        if isinstance(spec, real_device) and spec.type == "cuda":
            return real_device("cpu")
        return real_device(*args, **kwargs)

    monkeypatch.setattr(torch, "device", device)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: "fake-cuda")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: _NoOpStream())
    monkeypatch.setattr(torch.cuda, "default_stream", lambda device=None: _NoOpStream())
    monkeypatch.setattr(torch.cuda, "Stream", _NoOpStream)
    monkeypatch.setattr(torch.cuda, "Event", _NoOpEvent)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream=None: contextlib.nullcontext())
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", False)
    return real_device
