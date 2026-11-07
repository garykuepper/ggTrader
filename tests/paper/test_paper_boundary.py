"""Boundary: the live signal path is built on `ggTrader.engine` and the engine ensemble.

That `paper/` never imports `vectorbt` or uses the `.vbt` accessor is enforced for every file under
`src/` by `tests/engine/test_boundary.py` (and by ruff TID251); the per-file repeats of those two
checks were removed in S1. What only this file checks is the ban on the legacy lab modules and on
`ggTrader.lab.strategy`, kept as ONE test over every paper file (it was 16 per-file cases).
"""

from __future__ import annotations

import ast
from pathlib import Path

#: Names `paper/` must never import: the deleted legacy simulator modules (an import of one fails
#: at once, but this names the rule) and `lab.strategy`, the legacy config module.
LEGACY_MODULES = (
    "ggTrader.lab.simulate",
    "ggTrader.lab.sweep",
    "ggTrader.lab.metrics",
    "ggTrader.lab.harness",
    "ggTrader.lab.strategies.indicators",
    "ggTrader.lab.strategy",
)

PAPER = Path(__file__).resolve().parents[2] / "src" / "ggTrader" / "paper"
FILES = sorted(PAPER.glob("*.py"))


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


def test_paper_does_not_import_legacy_lab_modules():
    bad = {f.name: sorted(n for n in _imports(f) if n in LEGACY_MODULES) for f in FILES}
    bad = {name: modules for name, modules in bad.items() if modules}
    assert not bad, f"paper imports legacy modules: {bad}"


def test_signal_path_uses_the_engine_and_the_v2_ensemble():
    runner = _imports(PAPER / "signal_runner.py")
    assert "ggTrader.engine" in runner or "ggTrader.engine.selfcheck" in runner
    assert "ggTrader.paper.live_signals" in runner
    assert "ggTrader.lab.strategies.ensemble" in _imports(PAPER / "live_signals.py")
    assert "ggTrader.engine.portfolio" in _imports(PAPER / "overlay.py") or (
        "ggTrader.engine" in _imports(PAPER / "overlay.py")
    )
