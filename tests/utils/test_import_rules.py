"""Import rules that keep the base engine free of study plumbing (AST, no imports executed)."""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "ggTrader"

#: Study-side helpers that the engine, walk-forward code, strategies, paper trader and CLI avoid.
STUDY_SIDE = (
    "significance",
    "fred_data",
    "binding",
    "study",
    "manifest",
    "io",
)
#: Packages (relative to src/ggTrader) that must stay free of the study-side helpers.
BASE_PACKAGES = ("engine", "lab/wfo", "lab/strategies", "paper", "cli")


def _imported_modules(path: Path) -> list[tuple[int, str]]:
    """Every module a file imports as ``(line, dotted name)``; ``from a import b`` adds ``a.b``."""
    out: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            out += [(node.lineno, alias.name) for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out.append((node.lineno, node.module))
            out += [(node.lineno, f"{node.module}.{alias.name}") for alias in node.names]
    return out


def _is_banned(name: str, banned: tuple[str, ...]) -> bool:
    return any(
        name == f"ggTrader.utils.{b}" or name.startswith(f"ggTrader.utils.{b}.") for b in banned
    )


def _files(package: str) -> list[Path]:
    return sorted((SRC / package).rglob("*.py"))


def test_base_packages_exist_and_are_not_empty() -> None:
    for package in BASE_PACKAGES:
        assert _files(package), f"{package} has no python files; the rule would check nothing"


def test_base_packages_never_import_study_side_helpers() -> None:
    hits = [
        f"{path.relative_to(SRC)}:{line} imports {name}"
        for package in BASE_PACKAGES
        for path in _files(package)
        for line, name in _imported_modules(path)
        if _is_banned(name, STUDY_SIDE)
    ]
    assert hits == []


def test_nothing_under_src_imports_the_study_module() -> None:
    hits = [
        f"{path.relative_to(SRC)}:{line} imports {name}"
        for path in sorted(SRC.rglob("*.py"))
        if path.name != "study.py"
        for line, name in _imported_modules(path)
        if _is_banned(name, ("study",))
    ]
    assert hits == []


def test_moved_utils_modules_import_lab_only_inside_functions() -> None:
    for name in ("significance.py", "fred_data.py"):
        tree = ast.parse((SRC / "utils" / name).read_text())
        top_level = [
            f"{name}:{node.lineno}"
            for node in tree.body
            if (isinstance(node, ast.ImportFrom) and (node.module or "").startswith("ggTrader.lab"))
            or (
                isinstance(node, ast.Import)
                and any(a.name.startswith("ggTrader.lab") for a in node.names)
            )
        ]
        assert top_level == []
