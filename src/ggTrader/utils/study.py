"""Study plumbing: find the repo root, put a study's code on ``sys.path``, load a driver by path.

Study code lives in ``research/<category>/<study>/code`` (shared category helpers in
``research/<category>/_lib``). Nothing under ``src/`` may import this module: it is for drivers,
study tests and study conftests only.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

CODE_DIR = "code"
LIB_DIR = "_lib"


def _start(file: str | Path) -> Path:
    """The folder a file lives in (a folder is used as given)."""
    path = Path(file).resolve()
    return path if path.is_dir() else path.parent


def repo_root(file: str | Path) -> Path:
    """The first ancestor of ``file`` that holds both ``pyproject.toml`` and ``.git``.

    ``.git`` may be a folder or, in a linked worktree, a file.

    Raises:
        FileNotFoundError: no ancestor looks like a repository root.
    """
    start = _start(file)
    for parent in (start, *start.parents):
        if (parent / "pyproject.toml").is_file() and (parent / ".git").exists():
            return parent
    raise FileNotFoundError(f"no repository root above {file}")


def _nearest(file: str | Path, names: tuple[str, ...]) -> Path | None:
    """The nearest ancestor folder whose name is in ``names``."""
    start = _start(file)
    return next((p for p in (start, *start.parents) if p.name in names), None)


def bootstrap(file: str | Path) -> Path:
    """Put a study's own code (and its category library) at the front of ``sys.path``.

    The nearest ancestor of ``file`` named ``code`` puts ``<study>/code`` and, if it exists,
    ``<category>/_lib`` on the path; one named ``_lib`` puts that folder. Calling it again changes
    nothing.

    Returns:
        The repository root, found by walking up from ``file``.
    """
    anchor = _nearest(file, (CODE_DIR, LIB_DIR))
    if anchor is not None:
        wanted = [anchor]
        if anchor.name == CODE_DIR:
            lib = anchor.parent.parent / LIB_DIR
            if lib.is_dir():
                wanted.append(lib)
        have = set(sys.path)
        sys.path[:0] = [str(p) for p in wanted if str(p) not in have]
    return repo_root(file)


def study_dir(file: str | Path) -> Path:
    """The study folder: the folder above the ``code`` folder that holds ``file``.

    Raises:
        ValueError: ``file`` is not inside a ``code`` folder.
    """
    code = _nearest(file, (CODE_DIR,))
    if code is None:
        raise ValueError(f"{file} is not inside a study 'code' folder")
    return code.parent


def load_driver(path: str | Path) -> ModuleType:
    """Import a driver script by path (never by name) and return the module.

    The module is named ``study_<category>_<study>_<stem>`` and registered in ``sys.modules``
    (dataclasses and pickling need that). Each call builds a fresh module.
    """
    file = Path(path).resolve()
    code = _nearest(file, (CODE_DIR,))
    parts = [code.parent.parent.name, code.parent.name] if code is not None else []
    name = re.sub(r"\W", "_", "_".join(["study", *parts, file.stem]))
    spec = importlib.util.spec_from_file_location(name, file)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load a driver from {file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module
