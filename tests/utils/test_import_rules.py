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
LAYER_RULES = {
    "utils": ("lab", "paper", "cli"),
    "data": ("lab", "paper", "cli"),
    "engine": ("lab", "paper", "cli"),
    "lab": ("paper", "cli"),
}
LAYER_EXCEPTIONS: frozenset[tuple[str, int, str]] = frozenset(
    {
        ("utils/fred_data.py", 92, "ggTrader.lab.persist"),
        ("utils/fred_data.py", 118, "ggTrader.lab.persist"),
        ("utils/fred_data.py", 136, "ggTrader.lab.persist"),
        ("utils/significance.py", 49, "ggTrader.lab.wfo.report"),
    }
)


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


def _layer_imports(relative: str, tree: ast.AST) -> list[tuple[int, str]]:
    """Return (line, dotted module) for every ggTrader import form in one source file.

    Covers `import a.b`, `from a.b import x`, `from ggTrader import lab` (the module is then
    `ggTrader.lab`), relative imports resolved against the file's package, and
    `importlib.import_module("ggTrader....")` string literals.
    """
    parts = relative.removesuffix(".py").split("/")
    package = parts[:-1] if parts[-1] != "__init__" else parts[:-1]
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = ["ggTrader", *package[: len(package) - (node.level - 1)]]
            else:
                base = []
            module = [*base, *(node.module.split(".") if node.module else [])]
            if not module:
                continue
            found.append((node.lineno, ".".join(module)))
            if module == ["ggTrader"] or (node.level and not node.module):
                found.extend((node.lineno, ".".join([*module, a.name])) for a in node.names)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, (ast.Attribute, ast.Name))
            and getattr(node.func, "attr", getattr(node.func, "id", "")) == "import_module"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            found.append((node.lineno, node.args[0].value))
    return found


def test_layer_rules() -> None:
    hits: set[tuple[str, int, str]] = set()
    for path in sorted(SRC.rglob("*.py")):
        relative = path.relative_to(SRC).as_posix()
        package = relative.split("/", maxsplit=1)[0]
        if package not in LAYER_RULES:
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for line, name in _layer_imports(relative, tree):
            for target in LAYER_RULES[package]:
                prefix = f"ggTrader.{target}"
                if name == prefix or name.startswith(f"{prefix}."):
                    hits.add((relative, line, name))

    exceptions = hits & LAYER_EXCEPTIONS
    unexpected = hits - LAYER_EXCEPTIONS
    stale = LAYER_EXCEPTIONS - hits
    assert not unexpected and not stale, (
        f"Unexpected layer imports: {sorted(unexpected)}; "
        f"stale exceptions: {sorted(stale)}; "
        f"exceptions found: {sorted(exceptions)}"
    )


def test_layer_scan_sees_every_import_form() -> None:
    source = (
        "import ggTrader.lab.persist\n"
        "from ggTrader import lab, paper\n"
        "from ggTrader.lab import data\n"
        "from ..lab import wfo\n"
        "from .. import lab as l2\n"
        "import importlib\n"
        "importlib.import_module('ggTrader.paper.trader')\n"
    )
    modules = {m for _, m in _layer_imports("utils/sample.py", ast.parse(source))}
    assert {
        "ggTrader.lab.persist",
        "ggTrader.lab",
        "ggTrader.paper",
        "ggTrader.paper.trader",
    } <= modules
    # `from a.b import x` yields the module `a.b`; `from ..lab import x` resolves to ggTrader.lab.
    assert len([m for _, m in _layer_imports("utils/sample.py", ast.parse(source))]) >= 8
