"""The Colab notebook stages every repository module the trainer imports.

train_gridtracknet.py imports a handful of finetune.* and tennis_tracker.* modules (some inside
functions); the notebook's staging cell lists the files it needs as REQUIRED. This walks the
trainer's internal import closure from the source and checks the two agree, so a new import in
the trainer fails here instead of with ModuleNotFoundError on Colab."""
import ast
import json
from pathlib import Path
from typing import Iterator, List, Set

from tennis_tracker.paths import ROOT

NOTEBOOK = ROOT / "finetune" / "GridTrackNet_FineTune_Colab.ipynb"
PACKAGES = ("finetune", "tennis_tracker")
TRAINER = "finetune.training.train_gridtracknet"


def module_path(name: str) -> Path:
    path = ROOT / (name.replace(".", "/") + ".py")
    return path if path.is_file() else ROOT / name.replace(".", "/") / "__init__.py"


def module_name(path: Path) -> str:
    """The dotted name of a module file under the repository (``tennis_tracker/config.py`` ->
    ``tennis_tracker.config``; a package's ``__init__.py`` is the package)."""
    parts = path.relative_to(ROOT).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def internal_imports(path: Path) -> Iterator[str]:
    """Every finetune.* / tennis_tracker.* module `path` imports, at module level or inside a
    function, relative imports (``from .config import Config``) resolved against `path`."""
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in PACKAGES:
                    yield alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level:                       # relative: the package `path` sits in, `level - 1` levels up
                package = module_name(path.parent / "__init__.py").split(".")
                base = ".".join(package[:len(package) - (node.level - 1)])
                module = f"{base}.{node.module}" if node.module else base
            else:
                module = node.module or ""
            if not module or module.split(".")[0] not in PACKAGES:
                continue
            yield module
            for alias in node.names:            # `from finetune import refine` names a module too
                if (ROOT / module.replace(".", "/") / f"{alias.name}.py").is_file():
                    yield f"{module}.{alias.name}"


def import_closure(module: str) -> Set[str]:
    seen: Set[str] = set()
    pending = [module]
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        pending.extend(internal_imports(module_path(name)))
    for name in list(seen):                      # importing a submodule imports every package above it
        parts = name.split(".")
        seen.update(".".join(parts[:depth]) for depth in range(1, len(parts)))
    return seen


def staged_files() -> List[str]:
    """The REQUIRED list in the notebook's staging cell."""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "code" and "REQUIRED = [" in source:
            tree = ast.parse(source.replace("!pip", "pass  # pip"))
            for node in tree.body:
                if isinstance(node, ast.Assign) and node.targets[0].id == "REQUIRED":
                    return ast.literal_eval(node.value)
    raise AssertionError("the notebook has no staging cell with a REQUIRED list")


def test_the_notebook_stages_the_trainers_whole_import_closure():
    needed = {module_path(name).relative_to(ROOT).as_posix() for name in import_closure(TRAINER)}
    staged = staged_files()
    assert set(staged) == needed, f"missing from the notebook: {sorted(needed - set(staged))}; " \
                                  f"staged but not imported: {sorted(set(staged) - needed)}"
    assert all((ROOT / name).is_file() for name in staged)


def test_the_notebook_runs_the_trainer_as_a_module_from_the_staged_code():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = "".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert "'-m', 'finetune.training.train_gridtracknet'" in code
    assert "cwd=str(CODE)" in code
    assert "CODE / 'train_gridtracknet.py'" not in code        # no longer copied to code/ on its own
