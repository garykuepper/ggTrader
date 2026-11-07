"""L0: vectorbt is imported only inside ggTrader.engine (spec 5.2), enforced two ways.

Ruff TID251 bans the import per file; this AST test catches what ruff cannot: imports at any
depth (including function-local ones), dynamic import strings, the global ``.vbt`` accessor
(reachable from any module once vectorbt is imported anywhere), the raw indicator
constructors and ``Portfolio`` entry points, and the private ``vectorbt._engine``.

The one file that still imports vectorbt (the legacy-interpreter reference runner) is listed in
``LEGACY_ONLY`` and in ruff.toml's per-file-ignores. The list shrank from eight entries to this one
in P5c: this test fails if the entry no longer violates anything, and if ruff.toml and
``LEGACY_ONLY`` disagree.
"""

from __future__ import annotations

import ast
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCAN_ROOTS = ("src", "scripts", "tests")
#: Study code is scanned too: it must reach vectorbt only through ``ggTrader.engine`` like the rest.
SCAN_GLOBS = ("research/*/*/code", "research/*/_lib")
SCAN_FILES = ("ggt.py",)
ENGINE_DIR = "src/ggTrader/engine/"
ENGINE_TESTS = "tests/engine/"
STRATEGIES_DIR = "src/ggTrader/lab/strategies/"
SELFCHECK = "src/ggTrader/engine/selfcheck.py"
BANNED_MODULES = ("vectorbt", "vectorbt_rust")
CONSTRUCTORS = {"RSI", "BBANDS", "MACD", "ATR", "MA", "MSTD"}
DYNAMIC_IMPORTERS = {"__import__", "import_module", "find_spec", "reload"}
#: Modules whose names are importer functions that a file may rebind (``import_module as load``).
IMPORTER_MODULES = {"importlib", "importlib.util", "importlib.machinery", "builtins"}
#: Call names that reach an attribute by a string: ``getattr(x, "vbt")``, ``attrgetter("vbt")``.
ATTRIBUTE_BY_NAME = {"getattr", "attrgetter", "__getattribute__"}
EXEC_CALLS = {"exec", "eval", "compile"}
STRING_IMPORT = re.compile(r"\b(?:import|from)\s+vectorbt")
PORTFOLIO_CALLS = {"from_signals", "from_orders", "from_holding", "from_order_func"}

#: Files that may import vectorbt outside ``ggTrader.engine``. The allow-list is down to the one
#: documented exception: ``scripts/legacy_reference_runs.py`` runs under the legacy 0.28.5
#: interpreter against the tree extracted from git tag legacy-engine-vbt0285 and imports vectorbt
#: by design. Keep in step with ruff.toml; it must not grow.
LEGACY_ONLY = {
    "scripts/legacy_reference_runs.py",
}


def _is_banned(module: str | None) -> bool:
    return bool(module) and module.split(".")[0] in BANNED_MODULES


def _const_str(node: ast.AST | None) -> str | None:
    """The string a node evaluates to when it is built from literals only (else ``None``)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _const_str(node.left), _const_str(node.right)
        return left + right if left is not None and right is not None else None
    if isinstance(node, ast.JoinedStr):
        parts = [_const_str(v) for v in node.values]
        return "".join(parts) if all(p is not None for p in parts) else None
    return None


def _importer_aliases(tree: ast.AST) -> dict[str, str]:
    """``from importlib import import_module as load`` gives ``{"load": "import_module"}``."""
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "") in IMPORTER_MODULES:
            for alias in node.names:
                if alias.name in DYNAMIC_IMPORTERS:
                    aliases[alias.asname or alias.name] = alias.name
    return aliases


def _call_strings(node: ast.Call) -> list[str]:
    """Every literal-built string among a call's positional and keyword arguments."""
    values = [*node.args, *(kw.value for kw in node.keywords)]
    return [text for text in map(_const_str, values) if text is not None]


def violations(source: str, rel: str) -> list[str]:
    """Boundary violations in one file's source, as ``"line: what"`` strings."""
    tree = ast.parse(source)
    found: list[str] = []
    in_engine = rel.startswith(ENGINE_DIR) or rel.startswith(ENGINE_TESTS)
    vbt_attr_ok = in_engine or rel.startswith(STRATEGIES_DIR)
    aliases = _importer_aliases(tree)

    for node in ast.walk(tree):
        line = getattr(node, "lineno", 0)
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_banned(alias.name) and not in_engine:
                    found.append(f"{line}: import {alias.name}")
                if alias.name.startswith("vectorbt.labels"):
                    found.append(f"{line}: vectorbt.labels is forward-looking and banned")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if _is_banned(module) and not in_engine:
                found.append(f"{line}: from {module} import ...")
            if module.startswith("vectorbt.labels") or (
                module == "vectorbt" and any(a.name == "labels" for a in node.names)
            ):
                found.append(f"{line}: vectorbt.labels is forward-looking and banned")
            private = module == "vectorbt._engine" or (
                module == "vectorbt" and any(a.name == "_engine" for a in node.names)
            )
            if private and rel != SELFCHECK and not rel.startswith(ENGINE_TESTS):
                found.append(f"{line}: vectorbt._engine is private to selfcheck.py")
        elif isinstance(node, ast.Attribute):
            if node.attr == "vbt" and not vbt_attr_ok:
                found.append(f"{line}: .vbt accessor outside engine/ and lab/strategies/")
            if node.attr in PORTFOLIO_CALLS | {"Portfolio"} and not in_engine:
                value = ast.unparse(node.value)
                if value.endswith("Portfolio") or node.attr == "Portfolio":
                    found.append(f"{line}: Portfolio entry point outside engine/")
            if node.attr == "run" and not in_engine:
                value = ast.unparse(node.value)
                if value.split(".")[-1] in CONSTRUCTORS and (
                    value.startswith(("vbt.", "vectorbt.")) or value in CONSTRUCTORS
                ):
                    found.append(f"{line}: raw indicator constructor {value}.run outside engine/")
        elif isinstance(node, ast.Call):
            found += _call_violations(node, line, aliases, in_engine, vbt_attr_ok)
        elif isinstance(node, ast.Subscript) and not in_engine:
            key = _const_str(node.slice)
            if ast.unparse(node.value).endswith("sys.modules") and _is_banned(key):
                found.append(f"{line}: dynamic module string {key!r}")
    return found


def _call_violations(
    node: ast.Call, line: int, aliases: dict[str, str], in_engine: bool, vbt_attr_ok: bool
) -> list[str]:
    """Violations that hide in a call: dynamic imports, string attribute access, exec."""
    found: list[str] = []
    callee = ast.unparse(node.func).split(".")[-1]
    strings = _call_strings(node)
    if callee in ATTRIBUTE_BY_NAME and "vbt" in strings and not vbt_attr_ok:
        found.append(
            f"{line}: getattr-style access to the .vbt accessor outside engine/ and strategies"
        )
    if in_engine:
        return found
    name = aliases.get(callee, callee) if isinstance(node.func, ast.Name) else callee
    if name in DYNAMIC_IMPORTERS:
        for text in strings:
            if _is_banned(text):
                found.append(f"{line}: dynamic module string {text!r}")
    elif callee == "get" and ast.unparse(node.func).endswith("sys.modules.get"):
        found += [f"{line}: dynamic module string {t!r}" for t in strings if _is_banned(t)]
    elif callee in EXEC_CALLS and any(STRING_IMPORT.search(t) for t in strings):
        found.append(f"{line}: string-executed import of vectorbt")
    return found


def _python_files() -> list[Path]:
    files = [ROOT / name for name in SCAN_FILES if (ROOT / name).exists()]
    for top in SCAN_ROOTS:
        files += sorted((ROOT / top).rglob("*.py"))
    for pattern in SCAN_GLOBS:
        for folder in sorted(ROOT.glob(pattern)):
            files += sorted(folder.rglob("*.py"))
    return [f for f in files if ".venv" not in f.parts and "__pycache__" not in f.parts]


def _scan_repo() -> dict[str, list[str]]:
    out = {}
    for path in _python_files():
        rel = path.relative_to(ROOT).as_posix()
        # tests/engine/test_boundary.py names the banned modules in its own fixtures.
        found = violations(path.read_text(), rel)
        if found:
            out[rel] = found
    return out


def test_only_the_engine_package_touches_vectorbt() -> None:
    offenders = {f: v for f, v in _scan_repo().items() if f not in LEGACY_ONLY}
    engine_private = {
        f: v
        for f, v in offenders.items()
        if f.startswith(ENGINE_DIR) and any("private" in x for x in v)
    }
    assert not offenders, f"vectorbt boundary violated: {offenders}"
    assert not engine_private


def test_study_code_and_libraries_are_scanned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A vectorbt import or ``.vbt`` call in research/*/*/code or research/*/_lib is a violation."""
    code = tmp_path / "research/alloc/study/code/run.py"
    lib = tmp_path / "research/alloc/_lib/helper.py"
    clean = tmp_path / "research/alloc/study/code/clean.py"
    frozen = tmp_path / "research/alloc/artifacts-2026/old.py"
    for path, source in (
        (code, "import vectorbt\n"),
        (lib, "x = close.vbt.crossed_above(y)\n"),
        (clean, "import pandas\n"),
        (frozen, "import vectorbt\n"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    monkeypatch.setattr(sys.modules[__name__], "ROOT", tmp_path)
    assert set(_scan_repo()) == {
        "research/alloc/study/code/run.py",
        "research/alloc/_lib/helper.py",
    }


def test_legacy_only_allowlist_has_no_stale_entries() -> None:
    """An entry whose violations are gone must be deleted (the list may reach zero)."""
    scanned = _scan_repo()
    stale = sorted(f for f in LEGACY_ONLY if f not in scanned)
    assert not stale, f"remove from LEGACY_ONLY and ruff.toml per-file-ignores: {stale}"


def test_ruff_per_file_ignores_match_the_legacy_only_list() -> None:
    config = tomllib.loads((ROOT / "ruff.toml").read_text())
    ignores = config["lint"]["per-file-ignores"]
    in_ruff = {
        path
        for path, rules in ignores.items()
        if "TID251" in rules and not path.startswith((ENGINE_DIR, ENGINE_TESTS))
    }
    assert in_ruff == LEGACY_ONLY
    assert {"vectorbt", "vectorbt_rust"} <= set(config["lint"]["flake8-tidy-imports"]["banned-api"])
    assert "TID251" in config["lint"]["select"]


# --- the detector itself ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("import vectorbt\n", "import vectorbt"),
        ("import vectorbt as vbt\n", "import vectorbt"),
        ("import vectorbt.indicators\n", "import vectorbt.indicators"),
        ("from vectorbt import RSI\n", "from vectorbt import"),
        ("from vectorbt.portfolio import nb\n", "from vectorbt.portfolio import"),
        ("import vectorbt_rust\n", "import vectorbt_rust"),
        ("def f():\n    import vectorbt\n", "import vectorbt"),
        ("class A:\n    def f(self):\n        from vectorbt import x\n", "from vectorbt import"),
        ("x = close.vbt.crossed_above(y)\n", ".vbt accessor"),
        ("x = importlib.import_module('vectorbt')\n", "dynamic module string"),
        ("x = __import__('vectorbt.portfolio')\n", "dynamic module string"),
        ("m = sys.modules['vectorbt']\n", "dynamic module string"),
        ("spec = importlib.util.find_spec('vectorbt_rust')\n", "dynamic module string"),
        ('x = getattr(close, "vbt")\n', "getattr"),
        ('x = getattr(close, "vbt", None)\n', "getattr"),
        ('x = getattr(close, attr="vbt")\n', "getattr"),
        ('f = operator.attrgetter("vbt")\n', "getattr"),
        ('x = close.__getattribute__("vbt")\n', "getattr"),
        ('x = getattr(close, "v" + "bt")\n', "getattr"),
        ('x = importlib.import_module(name="vectorbt")\n', "dynamic module string"),
        ('x = importlib.import_module("vector" + "bt")\n', "dynamic module string"),
        ('x = importlib.import_module(f"vectorbt")\n', "dynamic module string"),
        ('x = builtins.__import__("vectorbt")\n', "dynamic module string"),
        ('x = importlib.__import__("vectorbt_rust")\n', "dynamic module string"),
        (
            "from importlib import import_module as load\nx = load('vectorbt')\n",
            "dynamic module string",
        ),
        ("import importlib as il\nx = il.import_module('vectorbt')\n", "dynamic module string"),
        ("import importlib.util as u\nspec = u.find_spec('vectorbt')\n", "dynamic module string"),
        ('exec("import vectorbt")\n', "string-executed import"),
        ('code = compile("from vectorbt import RSI", "x", "exec")\n', "string-executed import"),
        ("m = sys.modules.get('vectorbt')\n", "dynamic module string"),
        ("from vectorbt import _engine\n", "private"),
        ("from vectorbt import labels\n", "forward-looking"),
        ("import vectorbt.labels.generators\n", "forward-looking"),
        ("x = vbt.RSI.run(c)\n", "raw indicator constructor"),
        ("x = vbt.Portfolio.from_signals(c)\n", "Portfolio entry point"),
        ("from ggTrader.engine import portfolio\nPortfolio = 3\nx = Portfolio\n", ""),
        ('x = getattr(close, "values")\n', ""),
        ('x = importlib.import_module("pandas")\n', ""),
        ("from importlib import import_module as load\nx = load('pandas')\n", ""),
        ('exec("import os")\n', ""),
        ("x = importlib.import_module(name)\n", ""),  # unresolvable; ruff cannot see it either
    ],
)
def test_detector_flags_each_form(source: str, expected: str) -> None:
    found = violations(source, "src/ggTrader/lab/new_module.py")
    if expected:
        assert any(expected in v for v in found), found
    else:
        assert found == []


def test_getattr_vbt_is_allowed_inside_the_engine_and_strategies() -> None:
    source = 'x = getattr(close, "vbt")\n'
    assert violations(source, "src/ggTrader/engine/indicators.py") == []
    assert violations(source, "src/ggTrader/lab/strategies/ensemble.py") == []
    assert violations(source, "src/ggTrader/lab/wfo.py") != []


def test_detector_allows_the_engine_and_its_tests() -> None:
    source = "import vectorbt as vbt\nx = c.vbt.crossed_above(1.0)\nvbt.RSI.run(c)\n"
    assert violations(source, "src/ggTrader/engine/indicators.py") == []
    assert violations(source, "tests/engine/test_indicators.py") == []


def test_private_engine_is_allowed_only_in_selfcheck() -> None:
    source = "from vectorbt import _engine\n"
    assert violations(source, SELFCHECK) == []
    assert violations(source, "src/ggTrader/engine/config.py") != []


def test_vbt_accessor_is_allowed_in_strategies_but_not_wfo() -> None:
    source = "x = close.vbt.crossed_above(y)\n"
    assert violations(source, "src/ggTrader/lab/strategies/ensemble.py") == []
    assert violations(source, "src/ggTrader/lab/wfo.py") != []
    assert violations(source, "scripts/some_driver.py") != []


def test_unrelated_strings_and_comments_are_not_flagged() -> None:
    source = (
        '"""vectorbt is great."""\nx = "the vectorbt engine"\n# import vectorbt\n'
        'pkg = "vectorbt"\nrow = {"vectorbt": "1.2.0"}\nfetch("vectorbt")\n'
    )
    assert violations(source, "src/ggTrader/lab/new_module.py") == []


# --- ruff TID251 agrees --------------------------------------------------------------------


def _ruff() -> str | None:
    return shutil.which("ruff", path=str(Path(sys.executable).parent)) or shutil.which("ruff")


@pytest.mark.skipif(_ruff() is None, reason="ruff (dev extra) not installed")
def test_ruff_bans_vectorbt_outside_the_engine_and_allows_it_inside() -> None:
    def run(stdin_name: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [_ruff(), "check", "--no-cache", "--stdin-filename", stdin_name, "-"],
            input=(
                "import vectorbt\nfrom vectorbt import RSI\n\n\ndef f():\n"
                "    import vectorbt_rust\n    return vectorbt, RSI, vectorbt_rust\n"
            ),
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

    banned = run("src/ggTrader/lab/new_module.py")
    assert banned.returncode != 0 and banned.stdout.count("TID251") == 3, banned.stdout
    assert run("src/ggTrader/engine/new_module.py").returncode == 0
    assert run("tests/engine/test_new.py").returncode == 0
    assert run("scripts/legacy_reference_runs.py").returncode == 0  # the documented exception
