"""Scan study imports; checks imported names only, not module.attr uses."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

KNOWN_UNRESOLVED: frozenset[tuple[str, str]] = frozenset()


@dataclass(frozen=True)
class StudyImport:
    """One ggTrader module or imported name referenced by study code."""

    path: str
    line: int
    module: str
    name: str | None


def study_imports(root: Path) -> list[StudyImport]:
    """Return ggTrader imports found beneath study code and shared libraries."""
    paths = set(root.glob("research/*/*/code/**/*.py"))
    paths.update(root.glob("research/*/_lib/**/*.py"))
    imports: list[StudyImport] = []
    for path in sorted(paths):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        relative_path = path.relative_to(root).as_posix()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(
                    StudyImport(relative_path, node.lineno, alias.name, None)
                    for alias in node.names
                    if alias.name == "ggTrader" or alias.name.startswith("ggTrader.")
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module == "ggTrader" or node.module.startswith("ggTrader."):
                    imports.extend(
                        StudyImport(relative_path, node.lineno, node.module, alias.name)
                        for alias in node.names
                        if alias.name != "*"
                    )
    return imports
