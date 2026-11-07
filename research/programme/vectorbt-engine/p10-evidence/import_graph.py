"""P10: who imports a module? AST plus string references over src, scripts, tests and ggt.py.

Reads source text only (nothing is imported or run). Usage:
    python import_graph.py <repo-root> <module-path> [<module-path> ...]
prints one JSON object per module: lines, importers split into src/scripts/tests, and the
research reports/docs that cite the file by name.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path


def module_name(root: Path, path: Path) -> str:
    """Dotted module name of a file under ``root/src``."""
    parts = list(path.relative_to(root / "src").with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def referenced_names(tree: ast.AST) -> set[str]:
    """Dotted names imported by a tree, plus dotted string constants that look like modules."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names |= {f"{node.module}.{a.name}" for a in node.names}
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value.startswith("ggTrader."):
                names.add(node.value)
    return names


def kept_files(root: Path) -> list[tuple[Path, str]]:
    """Python files that count as kept code, with their kind."""
    out: list[tuple[Path, str]] = []
    for kind, folder in (("src", "src"), ("scripts", "scripts"), ("tests", "tests")):
        for path in (root / folder).rglob("*.py"):
            if "__pycache__" not in path.parts:
                out.append((path, kind))
    out.append((root / "ggt.py", "scripts"))
    return out


def main(argv: list[str] | None = None) -> int:
    """Print the importer table for the given module files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("modules", nargs="+", help="paths relative to root, e.g. src/ggTrader/x.py")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    trees: dict[Path, tuple[str, set[str]]] = {}
    for path, kind in kept_files(root):
        try:
            trees[path] = (kind, referenced_names(ast.parse(path.read_text())))
        except (SyntaxError, UnicodeDecodeError):
            continue
    docs = [p for folder in ("docs", "research", "README.md", "AGENTS.md") for p in _text_files(root / folder)]
    for rel in args.modules:
        target = root / rel
        hits: dict[str, list[str]] = {"src": [], "scripts": [], "tests": []}
        # Scripts and tests are entry points: nothing imports them by name.
        mod = module_name(root, target) if target.is_relative_to(root / "src") else ""
        for path, (kind, names) in trees.items():
            if path == target or not mod:
                continue
            if any(n == mod or n.startswith(mod + ".") for n in names):
                hits[kind].append(str(path.relative_to(root)))
        cited = sorted(
            str(p.relative_to(root))
            for p in docs
            if re.search(re.escape(Path(rel).name), _read(p))
            and "p10-evidence" not in p.parts
            and not str(p.relative_to(root)).startswith("docs/changelog")
        )
        json.dump(
            {"path": rel, "lines": len(target.read_text().splitlines()), "importers": hits, "cited_in": cited},
            sys.stdout,
        )
        sys.stdout.write("\n")
    return 0


def _text_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return [p for p in path.rglob("*") if p.suffix in {".md", ".txt", ".csv"} and p.is_file()] if path.exists() else []


def _read(path: Path) -> str:
    try:
        return path.read_text(errors="ignore")
    except OSError:
        return ""


if __name__ == "__main__":
    raise SystemExit(main())
