"""Every live ``scripts/*.py`` and study driver handles ``--help`` before it does any work.

CLAUDE.md: scripts need a ``main()`` guard, ``argparse`` and a graceful ``--help``. On
2026-11-06 a loop that ran ``--help`` over the scripts started a real backfill in one that
ignored the flag, so this file never runs a script it has not first proved safe:

1. **Static (AST), no execution.** The script (or, for the four thin wrappers in
   ``DELEGATES``, the library function it hands over to) builds an ``ArgumentParser``; the
   module top level calls nothing but a short list of harmless constructors; there is an
   ``if __name__ == "__main__"`` guard; and the entry function reaches ``parse_args`` before
   any other call outside an argparse/path vocabulary.
2. **Subprocess ``--help``,** only for scripts that pass step 1, with ``env -i`` style
   isolation: no HOME, no database or broker variables, a scratch working directory, a 20 s
   timeout. Exit code 0 and ``usage`` on stdout.

The scanned set is ``scripts/*.py`` plus every ``research/*/*/code/*.py`` and
``research/*/_lib/*.py`` that has an ``if __name__ == "__main__"`` guard (a library module without
one is not a command). Archived scripts (``research/programme/vectorbt-engine/legacy-scripts``)
are out of scope. A count balance ties this set to ``research/CODEMOVED.csv``: a script that moved
out of ``scripts/`` must still be scanned at its new path.
"""

from __future__ import annotations

import ast
import csv
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CODEMOVED = ROOT / "research/CODEMOVED.csv"
LAB = ROOT / "src/ggTrader/lab"

#: Thin wrappers whose parser lives in the library, keyed by repo-relative path:
#: script -> (library file, entry function). A slice that moves a wrapper or its library updates
#: the entry in the same commit.
DELEGATES = {
    "scripts/event_timing_audit.py": (LAB / "event_timing_audit.py", "main"),
    "scripts/insider_acceptance_diagnostic.py": (LAB / "event_acceptance_diagnostic.py", "main"),
    "scripts/intraday_observe.py": (LAB / "intraday_observer.py", "main"),
    "scripts/intraday_observer_job.py": (LAB / "intraday_observer.py", "job_main"),
}

#: Calls a module top level may make: constants, paths, regexes, loggers, sys.path set-up.
MODULE_CALLS = frozenset(
    {
        "re.compile", "Path", "Path(__file__).resolve", "sys.path.insert", "sys.path.append",
        "str", "frozenset", "set", "tuple", "dict", "any", "tempfile.gettempdir", "date",
        "date.today", "logging.getLogger", "find_project_root", "os.environ.get", "pd.Timestamp",
        "matplotlib.use", "bootstrap", "FastMCP", "FigureSpec", "OptionBlock", "Point", "WfoPolicy",
        "i.startswith", "'\\n\\n'.join", "','.join",
    }
)  # fmt: skip

#: Calls allowed in the entry function before ``parse_args`` (building the parser, not working).
PRE_PARSE_CALLS = frozenset(
    {
        "argparse.ArgumentParser", "ArgumentParser", "parser.add_argument", "p.add_argument",
        "ap.add_argument", "parser.add_subparsers", "sub.add_parser", "subparsers.add_parser",
        "parser.add_mutually_exclusive_group", "parser.set_defaults", "parser.parse_args",
        "parser.parse_known_args", "p.parse_args", "ap.parse_args", "parse_args", "build_parser",
        "_build_parser", "make_parser", "Path", "str", "int", "float", "bool", "len",
        "logging.basicConfig", "(__doc__ or '').strip", "__doc__.splitlines", "argparse.FileType",
        "sorted", "date.today", "Path(__file__).resolve",
        "__doc__.split", "__doc__.strip", "type",
    }
)  # fmt: skip


def _is_main_guard(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.If)
        and ast.unparse(node.test).replace('"', "'") == "__name__ == '__main__'"
    )


def _has_main_guard(path: Path) -> bool:
    return any(_is_main_guard(n) for n in ast.parse(path.read_text()).body)


def scanned_scripts(root: Path) -> list[Path]:
    """``scripts/*.py`` and the study drivers: ``code/*.py`` and ``_lib/*.py`` with a main guard."""
    live = [*root.joinpath("scripts").glob("*.py"), *root.joinpath("scripts/one_off").glob("*.py")]
    study = [*root.glob("research/*/*/code/*.py"), *root.glob("research/*/_lib/*.py")]
    return sorted([*live, *(p for p in study if _has_main_guard(p))])


def unscanned_moves(rows: list[dict[str, str]], scanned: set[str]) -> list[str]:
    """CODEMOVED rows that moved a ``scripts/*.py`` file to a ``.py`` path the scan misses."""
    return [
        f"{r['old_path']} -> {r['new_path']}"
        for r in rows
        if r["old_path"].startswith("scripts/")
        and r["old_path"].endswith(".py")
        and r["new_path"].endswith(".py")
        and r["new_path"] not in scanned
    ]


def _calls(node: ast.AST) -> list[str]:
    return [ast.unparse(n.func) for n in ast.walk(node) if isinstance(n, ast.Call)]


def _functions(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    return {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}


def _entry_name(tree: ast.Module) -> str:
    """The module function the main guard calls (``main()``, ``sys.exit(main())``, ...)."""
    guard = next(n for n in tree.body if _is_main_guard(n))
    funcs = _functions(tree)
    for node in ast.walk(guard):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in funcs:
            return node.func.id
    raise AssertionError("the main guard calls no function defined in the module")


def _first_work_before_parse(
    func: ast.FunctionDef, funcs: dict[str, ast.FunctionDef]
) -> str | None:
    """The first call that is not argparse vocabulary and precedes ``parse_args`` (or None)."""
    for stmt in func.body:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            continue  # docstring
        calls = _calls(stmt)
        if any(c.endswith("parse_args") or c.endswith("parse_known_args") for c in calls):
            return None
        # A local helper that itself calls parse_args counts as the parse step.
        for name in calls:
            helper = funcs.get(name)
            if helper is not None and any(c.endswith("parse_args") for c in _calls(helper)):
                return None
        for call in calls:
            if call not in PRE_PARSE_CALLS:
                return call
    return "<no parse_args call in the entry function>"


def _static_problems(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    problems: list[str] = []
    if not any(_is_main_guard(n) for n in tree.body):
        return ['no `if __name__ == "__main__":` guard']
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)):
            continue
        if _is_main_guard(node) or (
            isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
        ):
            continue
        problems += [f"module-level call {c}" for c in _calls(node) if c not in MODULE_CALLS]
    rel = path.relative_to(ROOT).as_posix()
    if rel in DELEGATES:
        lib_path, entry_name = DELEGATES[rel]
        entry_tree = ast.parse(lib_path.read_text())
    else:
        entry_tree = tree
        try:
            entry_name = _entry_name(tree)
        except AssertionError as exc:
            return [*problems, str(exc)]
    if not any(c.endswith("ArgumentParser") for c in _calls(entry_tree)):
        problems.append("no ArgumentParser")
    funcs = _functions(entry_tree)
    if entry_name not in funcs:
        return [*problems, f"entry function {entry_name} not found"]
    work = _first_work_before_parse(funcs[entry_name], funcs)
    if work is not None:
        problems.append(f"{work} runs before parse_args in {entry_name}()")
    return problems


SCRIPTS = scanned_scripts(ROOT)


def _ids(paths: list[Path]) -> list[str]:
    return [p.relative_to(ROOT).as_posix() for p in paths]


@pytest.mark.parametrize("script", SCRIPTS, ids=_ids(SCRIPTS))
def test_script_has_a_parser_a_main_guard_and_parses_first(script: Path) -> None:
    problems = _static_problems(script)
    assert not problems, f"{script.relative_to(ROOT)}: {problems}"


def test_every_moved_script_is_still_scanned() -> None:
    """The count balance: a script CODEMOVED moved from scripts/ to a .py path stays covered."""
    with CODEMOVED.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    scanned = {p.relative_to(ROOT).as_posix() for p in SCRIPTS}
    assert unscanned_moves(rows, scanned) == []


def test_the_balance_check_finds_a_moved_script_that_is_not_scanned() -> None:
    rows = [
        {"old_path": "scripts/a.py", "new_path": "research/x/y/code/a.py"},
        {"old_path": "scripts/b.py", "new_path": "research/x/y/code/run.py"},
        {"old_path": "scripts/c.py", "new_path": ""},
        {"old_path": "scripts/d.sh", "new_path": "research/x/y/code/d.sh"},
        {"old_path": "tests/e.py", "new_path": "research/x/y/code/tests/e.py"},
    ]
    assert unscanned_moves(rows, {"research/x/y/code/run.py"}) == [
        "scripts/a.py -> research/x/y/code/a.py"
    ]


def test_study_drivers_are_scanned_only_with_a_main_guard(tmp_path: Path) -> None:
    code = tmp_path / "research/a/s/code"
    lib = tmp_path / "research/a/_lib"
    for folder in (code, lib, tmp_path / "scripts"):
        folder.mkdir(parents=True)
    (code / "run.py").write_text('if __name__ == "__main__":\n    pass\n')
    (code / "helper.py").write_text("X = 1\n")
    (lib / "tool.py").write_text("if __name__ == '__main__':\n    pass\n")
    (lib / "mod.py").write_text("X = 1\n")
    (tmp_path / "scripts" / "s.py").write_text("X = 1\n")
    found = [p.relative_to(tmp_path).as_posix() for p in scanned_scripts(tmp_path)]
    assert found == ["research/a/_lib/tool.py", "research/a/s/code/run.py", "scripts/s.py"]


def _statically_safe() -> list[Path]:
    return [p for p in SCRIPTS if not _static_problems(p)]


SAFE = _statically_safe()


def _clean_env() -> dict[str, str]:
    """``env -i`` plus only what a Python process needs: no HOME, no DB or broker variables."""
    return {
        "PATH": "/usr/bin:/bin",
        "PYTHONPATH": str(ROOT / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "LC_ALL": "C.UTF-8",
    }


@pytest.mark.slow
@pytest.mark.parametrize("script", SAFE, ids=_ids(SAFE))
def test_script_help_exits_zero_with_usage(script: Path, tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        env=_clean_env(),
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-800:]
    assert "usage" in result.stdout.lower(), result.stdout[:300]
    assert not os.listdir(tmp_path), "--help wrote into its working directory"
