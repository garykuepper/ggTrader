"""Keep the public ggTrader imports used by research studies resolvable."""

from __future__ import annotations

import importlib
from pathlib import Path

from tests.study_surface import KNOWN_UNRESOLVED, StudyImport, study_imports

ROOT = Path(__file__).resolve().parents[2]


def _resolves(item: StudyImport) -> bool:
    """Return whether the imported module and optional name still resolve."""
    try:
        module = importlib.import_module(item.module)
    except (ImportError, ModuleNotFoundError):
        return False
    if item.name is None or hasattr(module, item.name):
        return True
    try:
        importlib.import_module(f"{item.module}.{item.name}")
    except (ImportError, ModuleNotFoundError):
        return False
    return True


def test_every_study_import_resolves() -> None:
    imports = study_imports(ROOT)
    unresolved = {
        (item.module, item.name)
        for item in imports
        if not _resolves(item) and (item.module, item.name or "") not in KNOWN_UNRESOLVED
    }
    assert not unresolved


def test_known_unresolved_is_still_unresolved() -> None:
    assert all(not _resolves(StudyImport("", 0, module, name)) for module, name in KNOWN_UNRESOLVED)


def test_scan_finds_imports() -> None:
    imports = study_imports(ROOT)
    assert len(imports) >= 150
    assert any(
        (item.module, item.name) == ("ggTrader.lab.persist", "get_engine") for item in imports
    )
