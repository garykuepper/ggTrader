"""Boundary: the paper trader reaches vectorbt only through `ggTrader.engine` (spec 5.2 rule 5).

`vectorbt` and `vectorbt_rust` are imported only under `src/ggTrader/engine/`; the `.vbt`
accessor only in `engine/` and `lab/strategies/`; and nothing in `paper/` imports the legacy
simulator, the legacy indicator module or the legacy ensemble (they are deleted after the
engine rebuild, which would otherwise break `import ggTrader.paper.trader`).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PAPER = Path(__file__).resolve().parents[2] / "src" / "ggTrader" / "paper"
FILES = sorted(PAPER.glob("*.py"))

BANNED_MODULES = ("vectorbt", "vectorbt_rust")
LEGACY_MODULES = (
    "ggTrader.lab.simulate",
    "ggTrader.lab.sweep",
    "ggTrader.lab.metrics",
    "ggTrader.lab.harness",
    "ggTrader.lab.strategies.indicators",
    "ggTrader.lab.strategy",
)
# The legacy ensemble module was ``ggTrader.lab.strategies.ensemble``; since the P9 flattening that
# name is the engine ensemble, so it is no longer a banned import (the legacy file is deleted).


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def test_paper_package_is_found():
    assert len(FILES) >= 10 and (PAPER / "signal_runner.py") in FILES


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_paper_never_imports_vectorbt_directly(path):
    bad = sorted(n for n in _imports(path) if n.split(".")[0] in BANNED_MODULES or n == "vectorbt")
    assert not bad, f"{path.name} imports {bad}; go through ggTrader.engine"


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_paper_never_touches_the_vbt_accessor(path):
    attrs = {n.attr for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.Attribute)}
    assert "vbt" not in attrs, f"{path.name} uses the .vbt accessor"


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_paper_does_not_import_legacy_lab_modules(path):
    bad = sorted(n for n in _imports(path) if n in LEGACY_MODULES)
    assert not bad, f"{path.name} imports legacy modules {bad}"


def test_signal_path_uses_the_engine_and_the_v2_ensemble():
    runner = _imports(PAPER / "signal_runner.py")
    assert "ggTrader.engine" in runner or "ggTrader.engine.selfcheck" in runner
    assert "ggTrader.paper.live_signals" in runner
    assert "ggTrader.lab.strategies.ensemble" in _imports(PAPER / "live_signals.py")
    assert "ggTrader.engine.portfolio" in _imports(PAPER / "overlay.py") or (
        "ggTrader.engine" in _imports(PAPER / "overlay.py")
    )
