"""Every live ``scripts/*.py`` and study driver handles ``--help`` before it does any work.

CLAUDE.md: scripts need a ``main()`` guard, ``argparse`` and a graceful ``--help``. On
2026-11-06 a loop that ran ``--help`` over the scripts started a real backfill in one that
ignored the flag, so this file never runs a script it has not first proved safe:

1. **Static (AST), no execution.** The script (or, for the four thin wrappers in
   ``DELEGATES``, the library function it hands over to) builds an ``ArgumentParser``; the
   module top level calls nothing but a short list of harmless constructors; there is an
   ``if __name__ == "__main__"`` guard; and the entry function reaches ``parse_args`` before
   any other call outside an argparse/path vocabulary. "Top level" means everything that runs
   on import: class bodies, decorators, defaults and annotations, not function bodies. A script
   that imports a ``LIBRARIES`` module (directly or through another) is safe only if that library
   passes the same import-time check, since ``--help`` imports it.
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
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CODEMOVED = ROOT / "research/CODEMOVED.csv"
INTRADAY_OBSERVER = ROOT / "research/intraday/spy-momentum-prospective-observer/code"
EVENT_TIMING = ROOT / "research/audits/event-timing/code"

#: Thin wrappers whose parser lives in the library, keyed by repo-relative path:
#: script -> (library file, entry function). A slice that moves a wrapper or its library updates
#: the entry in the same commit.
DELEGATES = {
    "research/audits/event-timing/code/run.py": (EVENT_TIMING / "event_timing_audit.py", "main"),
    "research/audits/event-timing/code/insider_acceptance_diagnostic.py": (
        EVENT_TIMING / "event_acceptance_diagnostic.py",
        "main",
    ),
    "research/intraday/spy-momentum-prospective-observer/code/intraday_observe.py": (
        INTRADAY_OBSERVER / "intraday_observer.py",
        "main",
    ),
    "research/intraday/spy-momentum-prospective-observer/code/intraday_observer_job.py": (
        INTRADAY_OBSERVER / "intraday_observer.py",
        "job_main",
    ),
}

#: Import-only library modules in ``scripts/``: imported by a sibling script (the whole-job
#: broker and driver, which are scanned), never run, so they get no fake CLI. Each is checked
#: instead to have no main guard and to do no work at import time. Adding a guard to one means
#: it became a command: remove it from this set so the parser checks apply.
LIBRARIES = frozenset({"scripts/whole_job_admission.py", "scripts/whole_job_prototype.py"})

#: Calls a module may make on import (top level, class bodies, decorators, defaults): constants,
#: paths, regexes, loggers, sys.path set-up, standard-library decorators.
MODULE_CALLS = frozenset(
    {
        "re.compile", "Path", "Path(__file__).resolve", "sys.path.insert", "sys.path.append",
        "str", "frozenset", "set", "tuple", "dict", "any", "tempfile.gettempdir", "date",
        "date.today", "datetime.date", "logging.getLogger", "find_project_root", "repo_root",
        "os.environ.get", "pd.Timestamp", "ZoneInfo",
        "matplotlib.use", "bootstrap", "FastMCP", "FigureSpec", "OptionBlock", "Point", "WfoPolicy",
        "i.startswith", "'\\n\\n'.join", "','.join",
        # Standard-library decorators and class-body helpers: they only wrap, nothing runs.
        "dataclass", "dataclasses.dataclass", "field", "dataclasses.field", "property",
        "classmethod", "staticmethod", "contextmanager", "contextlib.contextmanager",
        "lru_cache", "functools.lru_cache", "cache", "functools.cache",
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
        "add_membership_argument",
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
    """``scripts/*.py`` but ``LIBRARIES``, and study drivers: ``code/``, ``_lib/`` with a guard."""
    live = [*root.joinpath("scripts").glob("*.py"), *root.joinpath("scripts/one_off").glob("*.py")]
    live = [p for p in live if p.relative_to(root).as_posix() not in LIBRARIES]
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


class _ImportTimeCalls(ast.NodeVisitor):
    """Collect what a module runs when it is imported, skipping bodies that run only when called.

    Executed on import: every top-level statement, class bodies (nested too), decorators (a bare
    ``@name`` is a call of ``name``), default values and annotations of functions and methods,
    lambda defaults, class bases and keywords (``metaclass=M`` is a call of ``M``), the context
    managers of a ``with``, and a class's ``__init_subclass__`` hook. Deferred: the bodies of
    functions, methods and lambdas, and the ``if __name__ == "__main__"`` block. Operator and
    attribute dunders are not modelled; the call names are matched against ``MODULE_CALLS``.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    def visit_Call(self, node: ast.Call) -> None:
        self.calls.append(ast.unparse(node.func))
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        if _is_main_guard(node):
            return  # runs only as __main__, never on import
        self.generic_visit(node)

    def _decorators(self, decorators: list[ast.expr]) -> None:
        for dec in decorators:
            if not isinstance(dec, ast.Call):
                self.calls.append(ast.unparse(dec))
            self.visit(dec)

    def _signature(self, args: ast.arguments, returns: ast.expr | None = None) -> None:
        for default in [*args.defaults, *args.kw_defaults]:
            if default is not None:
                self.visit(default)
        every = [*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg]
        for arg in every:
            if arg is not None and arg.annotation is not None:
                self.visit(arg.annotation)
        if returns is not None:
            self.visit(returns)

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._decorators(node.decorator_list)
        self._signature(node.args, node.returns)
        # The body is deferred.

    visit_AsyncFunctionDef = visit_FunctionDef  # noqa: N815

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._signature(node.args)  # the body is deferred

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._decorators(node.decorator_list)
        for base in node.bases:
            self.visit(base)
        for keyword in node.keywords:
            if keyword.arg == "metaclass":
                self.calls.append(ast.unparse(keyword.value))
            self.visit(keyword.value)
        for stmt in node.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
                stmt.name == "__init_subclass__"
            ):
                self.calls.append("__init_subclass__")
            self.visit(stmt)  # a class body runs when the class statement does

    def visit_With(self, node: ast.With | ast.AsyncWith) -> None:
        for item in node.items:
            if not isinstance(item.context_expr, ast.Call):
                self.calls.append(ast.unparse(item.context_expr))  # its __enter__ runs
        self.generic_visit(node)

    visit_AsyncWith = visit_With  # noqa: N815


def _module_level_work(tree: ast.Module) -> list[str]:
    """Import-time calls outside ``MODULE_CALLS`` (work that would run on import)."""
    visitor = _ImportTimeCalls()
    for node in tree.body:
        visitor.visit(node)
    return [f"module-level call {c}" for c in visitor.calls if c not in MODULE_CALLS]


def _imported_libraries(tree: ast.Module, libraries: frozenset[str]) -> list[str]:
    """``libraries`` entries the module imports, by any import form, anywhere in the file.

    A library is matched by its module name (the file stem) as the last part of an imported
    dotted name, an imported name (``from scripts import lib``), or a string constant naming it
    (``importlib.import_module("scripts.lib")``). Matching is deliberately loose: a false match
    only withholds a ``--help`` run, a miss would let one import an unvalidated library.
    """
    stems = {Path(lib).stem: lib for lib in libraries}
    found: set[str] = set()

    def match(dotted: str | None) -> None:
        if dotted and dotted.rsplit(".", 1)[-1] in stems:
            found.add(stems[dotted.rsplit(".", 1)[-1]])

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                match(alias.name)
        elif isinstance(node, ast.ImportFrom):
            match(node.module)
            for alias in node.names:
                match(alias.name)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            match(node.value)
    return sorted(found)


def _library_problems(
    library: str,
    root: Path = ROOT,
    libraries: frozenset[str] = LIBRARIES,
    seen: set[str] | None = None,
) -> list[str]:
    """Why ``library`` is not a safe import-only module, including the libraries it imports."""
    seen = set() if seen is None else seen
    if library in seen:
        return []
    seen.add(library)
    path = root / library
    if not path.is_file():
        return [f"{library}: listed in LIBRARIES but missing"]
    tree = ast.parse(path.read_text())
    problems = [f"{library}: {p}" for p in _module_level_work(tree)]
    if any(_is_main_guard(n) for n in tree.body):
        problems.append(f"{library}: has a main guard: it is a command, remove it from LIBRARIES")
    for other in _imported_libraries(tree, libraries):
        problems += _library_problems(other, root, libraries, seen)
    return problems


def _static_problems(
    path: Path, root: Path = ROOT, libraries: frozenset[str] = LIBRARIES
) -> list[str]:
    tree = ast.parse(path.read_text())
    if not any(_is_main_guard(n) for n in tree.body):
        return ['no `if __name__ == "__main__":` guard']
    problems = _module_level_work(tree)
    rel = path.relative_to(root).as_posix()
    if rel in DELEGATES:
        lib_path, entry_name = DELEGATES[rel]
        entry_tree = ast.parse(lib_path.read_text())
    else:
        entry_tree = tree
        entry_name = ""
    # An importer of a library that fails validation is unsafe too: running its --help would
    # import (and so run) the library, whatever the library's own test reported.
    imported = {*_imported_libraries(tree, libraries), *_imported_libraries(entry_tree, libraries)}
    seen: set[str] = set()
    for library in sorted(imported):
        problems += [f"imports {p}" for p in _library_problems(library, root, libraries, seen)]
    if not entry_name:
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


@pytest.mark.parametrize("library", sorted(LIBRARIES))
def test_script_library_is_import_only(library: str) -> None:
    """A ``LIBRARIES`` module exists, is not a command, and does no work when imported."""
    assert _library_problems(library) == []


def test_library_exemption_skips_only_listed_modules(tmp_path: Path) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in ("whole_job_admission.py", "whole_job_prototype.py", "other.py"):
        (scripts / name).write_text("X = 1\n")
    found = [p.relative_to(tmp_path).as_posix() for p in scanned_scripts(tmp_path)]
    assert found == ["scripts/other.py"]


def _work(source: str) -> list[str]:
    return _module_level_work(ast.parse(textwrap.dedent(source)))


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("class A:\n    X = load()\n", "load"),
        ("class A:\n    class B:\n        Y = fetch()\n", "fetch"),
        ("@register\ndef f():\n    pass\n", "register"),
        ("@app.route('/x')\ndef f():\n    pass\n", "app.route"),
        ("@tag\nclass A:\n    pass\n", "tag"),
        ("class A:\n    @hook\n    def m(self):\n        pass\n", "hook"),
        ("def f(x=connect()):\n    pass\n", "connect"),
        ("def f(*, x=connect()):\n    pass\n", "connect"),
        ("async def f(x=connect()):\n    pass\n", "connect"),
        ("def f(x: resolve()):\n    pass\n", "resolve"),
        ("def f() -> resolve():\n    pass\n", "resolve"),
        ("class A(make_base()):\n    pass\n", "make_base"),
        ("class A(metaclass=Meta):\n    pass\n", "Meta"),
        ("class A:\n    def __init_subclass__(cls):\n        pass\n", "__init_subclass__"),
        ("class A:\n    def m(self, x=connect()):\n        pass\n", "connect"),
        ("f = lambda x=connect(): x\n", "connect"),
        ("with lock:\n    pass\n", "lock"),
        ("try:\n    import a\nexcept ImportError:\n    a = fallback()\n", "fallback"),
        ("if True:\n    @register\n    def f():\n        pass\n", "register"),
    ],
)
def test_import_time_work_is_found(source: str, expected: str) -> None:
    """Class bodies, decorators, defaults, annotations, bases and metaclasses run on import."""
    assert f"module-level call {expected}" in _work(source)


@pytest.mark.parametrize(
    "source",
    [
        "def f():\n    connect()\n",
        "async def f():\n    await connect()\n",
        "class A:\n    def m(self):\n        connect()\n",
        "class A:\n    def m(self):\n        def inner(x=connect()):\n            pass\n",
        "f = lambda: connect()\n",
        "if __name__ == '__main__':\n    connect()\n",
        "def f():\n    @register\n    def g():\n        pass\n",
        "@dataclass(frozen=True)\nclass A:\n    x: int = 0\n",
        "class P(Protocol):\n    def run(self) -> None: ...\n",
    ],
)
def test_deferred_bodies_are_not_import_time_work(source: str) -> None:
    """Function, method and lambda bodies and the main guard run only when called."""
    assert _work(source) == []


def _write(path: Path, source: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source))
    return path


#: A command that builds its parser first and imports the synthetic library ``scripts/lib.py``.
IMPORTER = """\
    import argparse

    from scripts.lib import VALUE


    def main() -> None:
        parser = argparse.ArgumentParser()
        parser.parse_args()


    if __name__ == "__main__":
        main()
"""


@pytest.mark.parametrize(
    "library_source",
    [
        "VALUE = open({sentinel!r}, 'w')\n",
        "class Holder:\n    VALUE = open({sentinel!r}, 'w')\n",
        "def touch(x=open({sentinel!r}, 'w')):\n    pass\n",
        "VALUE = 1\nif __name__ == '__main__':\n    pass\n",
    ],
)
def test_an_unsafe_library_gates_its_importer(tmp_path: Path, library_source: str) -> None:
    """An importer of a library that fails validation is not statically safe, so no --help runs.

    The fixtures are only parsed: the sentinel file would exist if any of them were executed.
    """
    sentinel = tmp_path / "ran"
    _write(tmp_path / "scripts/lib.py", library_source.format(sentinel=str(sentinel)))
    importer = _write(tmp_path / "scripts/cmd.py", IMPORTER)
    libraries = frozenset({"scripts/lib.py"})
    problems = _static_problems(importer, tmp_path, libraries)
    assert any(p.startswith("imports scripts/lib.py:") for p in problems), problems
    assert _statically_safe([importer], tmp_path, libraries) == []
    assert not sentinel.exists()


@pytest.mark.parametrize(
    "import_line",
    [
        "from scripts.lib import VALUE",
        "from scripts import lib",
        "import scripts.lib",
        "import lib",
        "from . import lib",
        "from .lib import VALUE",
        "import importlib\nMOD = 'scripts.lib'",
    ],
)
def test_every_import_form_of_a_library_is_gated(tmp_path: Path, import_line: str) -> None:
    _write(tmp_path / "scripts/lib.py", "VALUE = work()\n")
    importer = _write(
        tmp_path / "scripts/cmd.py",
        IMPORTER.replace("from scripts.lib import VALUE", import_line.replace("\n", "\n    ")),
    )
    assert _statically_safe([importer], tmp_path, frozenset({"scripts/lib.py"})) == []


def test_a_library_imported_through_another_library_gates_the_importer(tmp_path: Path) -> None:
    _write(tmp_path / "scripts/lib.py", "from scripts.deep import Y\nVALUE = 1\n")
    _write(tmp_path / "scripts/deep.py", "Y = work()\n")
    importer = _write(tmp_path / "scripts/cmd.py", IMPORTER)
    libraries = frozenset({"scripts/lib.py", "scripts/deep.py"})
    problems = _static_problems(importer, tmp_path, libraries)
    assert "imports scripts/deep.py: module-level call work" in problems, problems
    assert _statically_safe([importer], tmp_path, libraries) == []


def test_a_missing_library_gates_its_importer(tmp_path: Path) -> None:
    importer = _write(tmp_path / "scripts/cmd.py", IMPORTER)
    assert _statically_safe([importer], tmp_path, frozenset({"scripts/lib.py"})) == []


def test_a_clean_library_leaves_its_importer_safe(tmp_path: Path) -> None:
    _write(
        tmp_path / "scripts/lib.py",
        "from dataclasses import dataclass\n\n\n@dataclass\nclass VALUE:\n    x: int = 0\n",
    )
    importer = _write(tmp_path / "scripts/cmd.py", IMPORTER)
    assert _statically_safe([importer], tmp_path, frozenset({"scripts/lib.py"})) == [importer]


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


def _statically_safe(
    scripts: list[Path], root: Path = ROOT, libraries: frozenset[str] = LIBRARIES
) -> list[Path]:
    """The scripts whose ``--help`` may run: no static problem, own or of a library they import."""
    return [p for p in scripts if not _static_problems(p, root, libraries)]


SAFE = _statically_safe(SCRIPTS)


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
    # Re-checked at run time too, in case a script or library changed after collection.
    assert _static_problems(script) == []
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
