"""Every live ``scripts/*.py`` and study driver handles ``--help`` before it does any work.

CLAUDE.md: scripts need a ``main()`` guard, ``argparse`` and a graceful ``--help``. On
2026-11-06 a loop that ran ``--help`` over the scripts started a real backfill in one that
ignored the flag, so this file never runs a script it has not first checked:

1. **Static (AST), no execution.** The script (or, for the four thin wrappers in
   ``DELEGATES``, the library function it hands over to) builds an ``ArgumentParser``; the
   module top level calls nothing but a short list of harmless constructors (a list of names
   trusted only while the module does not rebind them); there is an
   ``if __name__ == "__main__"`` guard; and the entry function reaches ``parse_args`` before
   any other call outside an argparse/path vocabulary. "Top level" means everything that runs
   on import: class bodies, decorators, defaults and annotations, not function bodies.
2. **Library importers are never executed.** A static check cannot prove that importing a
   module is free of side effects (a descriptor default, a shadowed or monkeypatched wrapper,
   an operator dunder, a ``__class_getitem__`` all run code without a visible call). So a script
   that may import a ``LIBRARIES`` module, directly, through any other repository module, or
   through a dynamic import whose target folds to a constant string, gets the static checks of
   step 1 only. Its ``--help`` is not run: it is pinned in ``HELP_NOT_RUN`` and reported as a
   skip, never as a pass.
3. **Subprocess ``--help``,** only for scripts that pass step 1 and are not withheld by step 2,
   with ``env -i`` style isolation (no HOME, no database or broker variables, a scratch working
   directory, a 20 s timeout) under ``HELP_GUARD``: an audit hook that ends the process
   (``os._exit``, which no ``except`` can swallow) before a module named like a library is
   imported, before a file named like one (source or ``__pycache__`` bytecode) is opened, and
   before any child process starts. That backs step 2 up for loads it cannot see, such as an
   import whose module name comes from a variable. Exit code 0 and ``usage`` on stdout.

Limits, on purpose: step 1 is a lint against accidental work, not a proof; an unrelated script
can still run code on import by the routes listed in step 2, and its imported ``ggTrader`` and
third-party modules are not checked at all. Step 2 finds imports by name (any dotted component
equal to a library stem, any folded string naming one) and follows repository modules by
suffix, so it over-approximates; it does not evaluate variables, ``sys.path`` edits or
``sys.modules`` tricks. The tripwire matches file names, so a copy or hard link of a library
under another name, or loading through ``ctypes``, is not caught.

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
import re
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
#: broker and driver), never run, so they get no fake CLI. Each is linted to have no main guard
#: and no visible import-time work; that lint is not a proof, so a script that imports one never
#: has its ``--help`` run (``HELP_NOT_RUN``). Adding a guard to one means it became a command:
#: remove it from this set so the parser checks apply.
LIBRARIES = frozenset({"scripts/whole_job_admission.py", "scripts/whole_job_prototype.py"})

#: Scanned scripts whose ``--help`` is intentionally not executed, with the reason. They still get
#: every static check. The test that pins this set fails when a script starts or stops importing a
#: library, so the list is always the current one.
HELP_NOT_RUN = {
    "scripts/whole_job_broker.py": "imports scripts/whole_job_admission.py",
    "scripts/whole_job_driver.py": "imports scripts/whole_job_prototype.py",
}

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

#: Allowlisted standard-library names, trusted only as the object of that name: the module may
#: bind one at module or class scope only by ``import <module>`` or ``from <module> import <name>``
#: with the module given here (``None``: a builtin, which it may not bind at all). Any other binding
#: (def, class, assignment, another import, a star import, ``global``, an attribute store on the
#: module or on ``builtins``) shadows the name, and its calls count as work.
STDLIB_ORIGINS: dict[str, str | None] = {
    "dataclass": "dataclasses", "field": "dataclasses", "contextmanager": "contextlib",
    "lru_cache": "functools", "cache": "functools", "Path": "pathlib", "date": "datetime",
    "ZoneInfo": "zoneinfo", "dataclasses": "dataclasses", "contextlib": "contextlib",
    "functools": "functools", "re": "re", "sys": "sys", "os": "os", "logging": "logging",
    "tempfile": "tempfile", "datetime": "datetime",
    "property": None, "classmethod": None, "staticmethod": None, "str": None,
    "frozenset": None, "set": None, "tuple": None, "dict": None, "any": None,
}  # fmt: skip

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

#: Calls that load a module by name (first argument) or by file path.
NAME_LOADERS = frozenset({"import_module", "__import__", "find_spec", "run_module"})
PATH_LOADERS = frozenset(
    {
        "spec_from_file_location",
        "run_path",
        "SourceFileLoader",
        "load_source",
        "SourcelessFileLoader",
    }
)

#: Directories never searched for repository modules.
PRUNED_DIRS = frozenset({".git", ".venv", "venv", "node_modules", "__pycache__"})

#: Exit status of a ``--help`` run that ``HELP_GUARD`` stopped.
TRIPWIRE_EXIT = 97

#: ``python -c HELP_GUARD <script> --help`` runs the script as ``python <script> --help`` would
#: (``sys.argv``, ``sys.path[0]``, ``__main__``), after installing an audit hook that ends the
#: process before a library is imported or opened (by name, so ``__pycache__`` bytecode too) or a
#: child process starts. ``os._exit`` cannot be caught, so a script cannot turn a trip into a pass.
HELP_GUARD = """\
import os, sys

STEMS = frozenset({stems!r})
SPAWN = frozenset({{
    "subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.fork",
    "os.forkpty",
}})


def trip(event, detail):
    os.write(2, ("help tripwire: %s %r\\n" % (event, detail)).encode())
    os._exit({exit_code})


def hook(event, args):
    if event == "import":
        if args[0] and args[0].rsplit(".", 1)[-1] in STEMS:
            trip(event, args[0])
    elif event == "open":
        target = args[0]
        if isinstance(target, int):
            return
        try:
            name = os.path.basename(os.fsdecode(target))
        except TypeError:
            trip(event, target)
        if name.split(".", 1)[0] in STEMS:
            trip(event, name)
    elif event in SPAWN:
        trip(event, args[:1])


sys.addaudithook(hook)
script = sys.argv[1]
sys.argv = sys.argv[1:]
sys.path[0] = os.path.dirname(os.path.realpath(script))
import runpy

runpy.run_path(script, run_name="__main__")
"""


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
    functions, methods and lambdas, and the ``if __name__ == "__main__"`` block. Operator,
    attribute and descriptor dunders are not modelled; the call names are matched against
    ``MODULE_CALLS``.
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


class _ShadowedNames(ast.NodeVisitor):
    """``STDLIB_ORIGINS`` names that a module rebinds at module or class scope (see there)."""

    def __init__(self) -> None:
        self.names: set[str] = set()

    def _bind(self, name: str | None) -> None:
        if name in STDLIB_ORIGINS:
            self.names.add(name)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self._bind(node.id)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            root = node.value
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id in {"builtins", "__builtins__"}:
                self.names.update(n for n, origin in STDLIB_ORIGINS.items() if origin is None)
            elif isinstance(root, ast.Name):
                self._bind(root.id)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._bind(node.name)
        for expr in [*node.decorator_list, *node.args.defaults, *node.args.kw_defaults]:
            if expr is not None:
                self.visit(expr)
        for inner in ast.walk(node):  # ``global cache`` in a body rebinds the module's name
            if isinstance(inner, ast.Global):
                for name in inner.names:
                    self._bind(name)

    visit_AsyncFunctionDef = visit_FunctionDef  # noqa: N815

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._bind(node.name)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        self._bind(node.name)
        self.generic_visit(node)

    def visit_MatchAs(self, node: ast.MatchAs) -> None:
        self._bind(node.name)
        self.generic_visit(node)

    def visit_MatchStar(self, node: ast.MatchStar) -> None:
        self._bind(node.name)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            bound = alias.asname or alias.name.split(".")[0]
            trusted = alias.asname is None and STDLIB_ORIGINS.get(bound) == bound
            if not trusted:
                self._bind(bound)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module if node.level == 0 else None
        for alias in node.names:
            if alias.name == "*":
                self.names.update(n for n, origin in STDLIB_ORIGINS.items() if origin != module)
                continue
            bound = alias.asname or alias.name
            trusted = alias.name == bound and module is not None
            if not (trusted and STDLIB_ORIGINS.get(bound) == module):
                self._bind(bound)


def _module_level_work(tree: ast.Module) -> list[str]:
    """Import-time calls outside ``MODULE_CALLS``, or of an allowlisted name the module shadows."""
    visitor = _ImportTimeCalls()
    shadows = _ShadowedNames()
    for node in tree.body:
        visitor.visit(node)
        shadows.visit(node)
    work = []
    for call in visitor.calls:
        root = re.match(r"[A-Za-z_]\w*", call)
        if call not in MODULE_CALLS:
            work.append(f"module-level call {call}")
        elif root is not None and root.group() in shadows.names:
            work.append(f"module-level call {call} (shadows the standard-library name)")
    return work


def _library_problems(library: str, root: Path = ROOT) -> list[str]:
    """Lint findings that make ``library`` not an import-only module. A lint, not a proof."""
    path = root / library
    if not path.is_file():
        return [f"{library}: listed in LIBRARIES but missing"]
    tree = ast.parse(path.read_text())
    problems = [f"{library}: {p}" for p in _module_level_work(tree)]
    if any(_is_main_guard(n) for n in tree.body):
        problems.append(f"{library}: has a main guard: it is a command, remove it from LIBRARIES")
    return problems


def _static_problems(path: Path, root: Path = ROOT) -> list[str]:
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


def _repo_modules(root: Path) -> dict[tuple[str, ...], list[Path]]:
    """Every repository ``.py`` file under each dotted-name suffix it could be imported as.

    ``a/b/c.py`` is filed under ``c``, ``b.c`` and ``a.b.c``; a package ``a/b/__init__.py`` under
    ``b`` and ``a.b``. Matching suffixes rather than resolving ``sys.path`` over-approximates, which
    is the safe direction here: a spurious match can only withhold a ``--help`` run.
    """
    index: dict[tuple[str, ...], list[Path]] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in PRUNED_DIRS and not d.startswith(".")]
        for filename in filenames:
            if not filename.endswith(".py"):
                continue
            path = Path(dirpath, filename)
            parts = path.relative_to(root).with_suffix("").parts
            if parts[-1] == "__init__":
                parts = parts[:-1]
            for start in range(len(parts)):
                index.setdefault(parts[start:], []).append(path)
    return index


def _fold(node: ast.AST) -> str | None:
    """The value of a string expression built only from constants (``+``, f-string, ``join``)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _fold(node.left), _fold(node.right)
        return None if left is None or right is None else left + right
    if isinstance(node, ast.JoinedStr):
        pieces = [_fold(v.value if isinstance(v, ast.FormattedValue) else v) for v in node.values]
        return None if None in pieces else "".join(pieces)  # type: ignore[arg-type]
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "join"
        and len(node.args) == 1
        and isinstance(node.args[0], (ast.List, ast.Tuple))
    ):
        sep = _fold(node.func.value)
        items = [_fold(e) for e in node.args[0].elts]
        return None if sep is None or None in items else sep.join(items)  # type: ignore[arg-type]
    return None


def _imports(path: Path, tree: ast.Module) -> tuple[list[str], list[Path]]:
    """Dotted names ``path`` imports (statements and constant dynamic loads), and the files it
    names by relative import or by a constant loader path."""
    names: list[str] = []
    files: list[Path] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names += [node.module, *(f"{node.module}.{a.name}" for a in node.names)]
        elif isinstance(node, ast.ImportFrom):
            base = path.parent
            for _ in range(node.level - 1):
                base = base.parent
            module = base.joinpath(*node.module.split(".")) if node.module else base
            files += [base / "__init__.py", module.with_suffix(".py"), module / "__init__.py"]
            for alias in node.names:
                files += [
                    (module / alias.name).with_suffix(".py"),
                    module / alias.name / "__init__.py",
                ]
                names.append(alias.name)  # matched against library stems only
            names += node.module.split(".") if node.module else []
        elif isinstance(node, ast.Call):
            func = ast.unparse(node.func).rsplit(".", 1)[-1]
            folded = [_fold(arg) for arg in node.args]
            if func in NAME_LOADERS and folded and folded[0]:
                names.append(folded[0])
            elif func in PATH_LOADERS:
                for sub in node.args:
                    for piece in ast.walk(sub):
                        value = _fold(piece)
                        if value and value.endswith(".py"):
                            names.append(Path(value).stem)
                            files.append(path.parent / value)
    return names, [f for f in files if f.is_file()]


def _library_imports(
    script: Path,
    root: Path = ROOT,
    libraries: frozenset[str] = LIBRARIES,
    index: dict[tuple[str, ...], list[Path]] | None = None,
) -> list[str]:
    """Why ``script`` may import a ``libraries`` module, following repository modules it imports.

    A file in the closure gates the script when it is a library, imports a dotted name with a
    component equal to a library stem, or holds a constant (folded) string naming one. Empty
    means none was found, not that none is imported: see the module docstring for what is missed.
    """
    index = _repo_modules(root) if index is None else index
    stems = {Path(lib).stem for lib in libraries}
    library_files = {(root / lib).resolve(): lib for lib in libraries}
    named = re.compile(r"(?<![\w])(" + "|".join(map(re.escape, sorted(stems))) + r")(?![\w])")
    parent: dict[Path, Path | None] = {script.resolve(): None}
    queue = [script.resolve()]
    reasons: set[str] = set()

    def via(path: Path) -> str:
        chain = []
        node: Path | None = path
        while node is not None:
            chain.append(
                node.relative_to(root).as_posix() if node.is_relative_to(root) else str(node)
            )
            node = parent[node]
        return " -> ".join(reversed(chain))

    while queue:
        path = queue.pop(0)
        if path in library_files:
            reasons.add(f"{via(path)}: imports {library_files[path]}")
            continue
        try:
            tree = ast.parse(path.read_text())
        except (SyntaxError, UnicodeDecodeError, ValueError) as exc:
            reasons.add(f"{via(path)}: cannot be parsed ({type(exc).__name__})")
            continue
        names, files = _imports(path, tree)
        for name in names:
            if stems & set(name.split(".")):
                reasons.add(f"{via(path)}: imports {name}")
        for node in ast.walk(tree):
            value = _fold(node)
            if value is not None and named.search(value):
                reasons.add(f"{via(path)}: names {named.search(value).group()} in a string")
        candidates = list(files)
        for name in names:
            parts = tuple(name.split("."))
            for end in range(1, len(parts) + 1):
                candidates += index.get(parts[:end], [])
        for candidate in candidates:
            candidate = candidate.resolve()
            if candidate not in parent:
                parent[candidate] = path
                queue.append(candidate)
    return sorted(reasons)


def _help_withheld(
    scripts: list[Path], root: Path = ROOT, libraries: frozenset[str] = LIBRARIES
) -> dict[Path, list[str]]:
    """The scripts whose ``--help`` must not run because they may import a library, and why."""
    index = _repo_modules(root)
    found = {p: _library_imports(p, root, libraries, index) for p in scripts}
    return {p: reasons for p, reasons in found.items() if reasons}


def _help_runnable(
    scripts: list[Path], root: Path = ROOT, libraries: frozenset[str] = LIBRARIES
) -> list[Path]:
    """The scripts whose ``--help`` may run: no static problem and no possible library import."""
    safe = [p for p in scripts if not _static_problems(p, root)]
    withheld = _help_withheld(safe, root, libraries)
    pinned = {root / rel for rel in HELP_NOT_RUN} if root == ROOT else set()
    return [p for p in safe if p not in withheld and p not in pinned]


SCRIPTS = scanned_scripts(ROOT)


def _ids(paths: list[Path]) -> list[str]:
    return [p.relative_to(ROOT).as_posix() for p in paths]


@pytest.mark.parametrize("script", SCRIPTS, ids=_ids(SCRIPTS))
def test_script_has_a_parser_a_main_guard_and_parses_first(script: Path) -> None:
    problems = _static_problems(script)
    assert not problems, f"{script.relative_to(ROOT)}: {problems}"


@pytest.mark.parametrize("library", sorted(LIBRARIES))
def test_script_library_passes_the_import_time_lint(library: str) -> None:
    """A ``LIBRARIES`` module exists, is not a command, and shows no import-time work.

    A lint only: passing it does not make the library safe to import, and it gates nothing.
    """
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
    ("source", "expected"),
    [
        ("def cache(f):\n    return f\n\n\n@cache\ndef g():\n    pass\n", "cache"),
        ("from evil import cache\n\n\n@cache\ndef g():\n    pass\n", "cache"),
        ("from evil import dataclass\n\n\n@dataclass\nclass A:\n    pass\n", "dataclass"),
        ("import evil as dataclasses\n\n\n@dataclasses.dataclass\nclass A:\n    pass\n",
         "dataclasses.dataclass"),
        ("from functools import lru_cache as cache\n\n\n@cache\ndef g():\n    pass\n", "cache"),
        ("from evil import *\n\n\n@dataclass\nclass A:\n    pass\n", "dataclass"),
        ("dataclass = hooks.run\n\n\n@dataclass\nclass A:\n    pass\n", "dataclass"),
        ("class property:\n    pass\n\n\nclass A:\n    x = property(1)\n", "property"),
        ("import builtins\nbuiltins.str = hooks.run\nX = str(1)\n", "str"),
        ("import functools\nfunctools.cache = hooks.run\n\n\n@functools.cache\ndef g():\n"
         "    pass\n", "functools.cache"),
        ("def f():\n    global cache\n    cache = hooks.run\n\n\n@cache\ndef g():\n    pass\n",
         "cache"),
        ("class A:\n    field = hooks.run\n    x: int = field()\n", "field"),
    ],
)  # fmt: skip
def test_a_shadowed_standard_library_name_is_import_time_work(source: str, expected: str) -> None:
    """An allowlisted name trusts the standard-library object, not whatever the module bound."""
    assert any(p.startswith(f"module-level call {expected} (shadows") for p in _work(source))


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
        "from dataclasses import dataclass, field\n\n\n@dataclass\nclass A:\n"
        "    x: list = field(default_factory=list)\n",
        "import functools\n\n\n@functools.cache\ndef g():\n    pass\n",
        "from functools import cache\n\n\n@cache\ndef g():\n    pass\n",
        "def f():\n    field = 1\n    str = 2\n    return field, str\n",
        "from dataclasses import *\n\n\n@dataclass\nclass A:\n    pass\n",
    ],
)
def test_deferred_bodies_are_not_import_time_work(source: str) -> None:
    """Function, method and lambda bodies and the main guard run only when called."""
    assert _work(source) == []


def _write(path: Path, source: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source))
    return path


#: The synthetic library of the fixtures below, and the stem its importers name.
LIB = "scripts/job_lib.py"
LIBS = frozenset({LIB})

#: A command that builds its parser first and imports the synthetic library ``scripts/job_lib.py``.
IMPORTER = """\
    import argparse

    from scripts.job_lib import VALUE


    def main() -> None:
        parser = argparse.ArgumentParser()
        parser.parse_args()


    if __name__ == "__main__":
        main()
"""


def _importer(tmp_path: Path, import_line: str, name: str = "cmd.py") -> Path:
    lines = import_line.replace("\n", "\n    ")
    return _write(
        tmp_path / "scripts" / name, IMPORTER.replace("from scripts.job_lib import VALUE", lines)
    )


def _assert_withheld(tmp_path: Path, importer: Path, sentinel: Path | None = None) -> list[str]:
    assert _static_problems(importer, tmp_path) == []  # the static checks alone would pass it
    reasons = _help_withheld([importer], tmp_path, LIBS).get(importer, [])
    assert reasons, "the importer was not withheld"
    assert _help_runnable([importer], tmp_path, LIBS) == []
    if sentinel is not None:
        assert not sentinel.exists(), "a fixture was executed"
    return reasons


#: Library bodies that run code on import without a call the lint can attribute to them. The
#: ``{sentinel}`` file would exist if any were executed; they are only parsed.
BYPASSES = {
    "shadowed cache": """\
        def cache(func):
            open({sentinel!r}, "w").close()
            return func


        @cache
        def value():
            return 1
    """,
    "shadowed dataclass": """\
        from scripts.helper import hook as dataclass


        @dataclass
        class Config:
            x: int = 0
    """,
    "dataclass descriptor": """\
        from dataclasses import dataclass

        from scripts.helper import DEFAULT


        @dataclass
        class Config:
            x: int = DEFAULT
    """,
}

#: The unlisted helper the bypasses import: ``hook`` and ``DEFAULT.__get__`` / ``__set_name__``
#: run when ``@dataclass`` processes ``Config``.
HELPER = """\
    class Hook:
        def __set_name__(self, owner, name):
            open({sentinel!r}, "w").close()

        def __get__(self, obj, owner):
            open({sentinel!r}, "w").close()
            return 0


    DEFAULT = Hook()


    def hook(cls):
        open({sentinel!r}, "w").close()
        return cls
"""


@pytest.mark.parametrize("bypass", sorted(BYPASSES))
def test_a_lint_bypassing_library_still_withholds_its_importer(tmp_path: Path, bypass: str) -> None:
    """Sol's bypasses: whatever the lint says about the library, its importer is never executed."""
    sentinel = tmp_path / "ran"
    _write(tmp_path / LIB, BYPASSES[bypass].format(sentinel=str(sentinel)))
    _write(tmp_path / "scripts/helper.py", HELPER.format(sentinel=str(sentinel)))
    importer = _write(tmp_path / "scripts/cmd.py", IMPORTER)
    _assert_withheld(tmp_path, importer, sentinel)
    lint = _library_problems(LIB, tmp_path)
    if bypass == "dataclass descriptor":
        assert lint == []  # the lint cannot see it, which is why the lint gates nothing
    else:
        assert lint, "the shadowed decorator should be reported by the lint"


def test_a_clean_library_still_withholds_its_importer(tmp_path: Path) -> None:
    """Passing the lint is not proof: no importer of a library has its --help run."""
    _write(
        tmp_path / LIB,
        "from dataclasses import dataclass\n\n\n@dataclass\nclass VALUE:\n    x: int = 0\n",
    )
    importer = _write(tmp_path / "scripts/cmd.py", IMPORTER)
    assert _library_problems(LIB, tmp_path) == []
    _assert_withheld(tmp_path, importer)


@pytest.mark.parametrize(
    "import_line",
    [
        "from scripts.job_lib import VALUE",
        "from scripts import job_lib",
        "import scripts.job_lib",
        "import job_lib",
        "from . import job_lib",
        "from .job_lib import VALUE",
        "import importlib\nMOD = 'scripts.job_lib'",
    ],
)
def test_every_import_form_of_a_library_is_withheld(tmp_path: Path, import_line: str) -> None:
    _write(tmp_path / LIB, "VALUE = 1\n")
    _assert_withheld(tmp_path, _importer(tmp_path, import_line))


def test_a_missing_library_still_withholds_its_importer(tmp_path: Path) -> None:
    _assert_withheld(tmp_path, _write(tmp_path / "scripts/cmd.py", IMPORTER))


@pytest.mark.parametrize(
    ("files", "import_line"),
    [
        ({"scripts/helper.py": "from scripts.job_lib import VALUE\n"}, "import scripts.helper"),
        ({"scripts/helper.py": "from job_lib import VALUE\n"}, "import helper"),
        ({"scripts/helper.py": "from scripts.deeper import X\n", "scripts/deeper.py":
          "import scripts.job_lib\nX = 1\n"}, "from scripts.helper import X"),
        ({"scripts/pkg/__init__.py": "from scripts.job_lib import VALUE\n",
          "scripts/pkg/sub.py": "Y = 1\n"}, "from scripts.pkg.sub import Y"),
        ({"scripts/pkg/__init__.py": "", "scripts/pkg/sub.py": "from ..job_lib import VALUE\n"},
         "from scripts.pkg import sub"),
        ({"research/a/_lib/shared.py": "import job_lib\n"}, "import shared"),
        ({"scripts/helper.py": "import importlib\nM = importlib.import_module('scripts.' + "
          "'job_' + 'lib')\n"}, "import helper"),
        ({"scripts/helper.py": "import runpy\nrunpy.run_path('deeper.py')\n",
          "scripts/deeper.py": "from scripts.job_lib import VALUE\n"}, "import helper"),
    ],
)  # fmt: skip
def test_a_library_reached_through_an_unlisted_module_is_withheld(
    tmp_path: Path, files: dict[str, str], import_line: str
) -> None:
    """Unlisted intermediaries (modules, packages, relative imports, other directories)."""
    _write(tmp_path / LIB, "VALUE = 1\n")
    for rel, source in files.items():
        _write(tmp_path / rel, source)
    reasons = _assert_withheld(tmp_path, _importer(tmp_path, import_line))
    assert any(" -> " in r for r in reasons), reasons


@pytest.mark.parametrize(
    "import_line",
    [
        "import importlib\nMOD = importlib.import_module('scripts.' + 'job_' + 'lib')",
        "import importlib\nMOD = importlib.import_module(f\"scripts.{'job_lib'}\")",
        "import importlib\nMOD = importlib.import_module('.'.join(['scripts', 'job_lib']))",
        "MOD = __import__('scripts.job' + '_lib')",
        "import importlib.util\n"
        "SPEC = importlib.util.spec_from_file_location('m', 'job_' + 'lib.py')",
        "import runpy\nNS = runpy.run_path('job_lib' + '.py')",
    ],
)
def test_a_computed_dynamic_import_of_a_library_is_withheld(
    tmp_path: Path, import_line: str
) -> None:
    """A loader whose argument folds to a constant naming a library gates the script."""
    sentinel = tmp_path / "ran"
    _write(tmp_path / LIB, f"open({str(sentinel)!r}, 'w').close()\n")
    importer = _write(
        tmp_path / "scripts/cmd.py",
        IMPORTER.replace(
            "from scripts.job_lib import VALUE\n", import_line.replace("\n", "\n    ")
        ),
    )
    # The loader runs at module level, so the static check fails as well; the point here is that
    # the library gate sees through the computed name on its own.
    assert _help_withheld([importer], tmp_path, LIBS).get(importer)
    assert _help_runnable([importer], tmp_path, LIBS) == []
    assert not sentinel.exists()


def test_a_computed_import_of_another_module_is_not_withheld(tmp_path: Path) -> None:
    """The folding is specific: a computed name that is not a library leaves the script runnable."""
    _write(tmp_path / LIB, "VALUE = 1\n")
    _write(
        tmp_path / "scripts/helper.py",
        "import importlib\nM = importlib.import_module('scripts.' + 'other')\n",
    )
    importer = _importer(tmp_path, "import helper")
    assert _help_withheld([importer], tmp_path, LIBS) == {}
    assert _help_runnable([importer], tmp_path, LIBS) == [importer]


def _guarded_help(
    script: Path, cwd: Path, stems: frozenset[str], **env: str
) -> subprocess.CompletedProcess:
    guard = HELP_GUARD.format(stems=sorted(stems), exit_code=TRIPWIRE_EXIT)
    return subprocess.run(
        [sys.executable, "-c", guard, str(script), "--help"],
        env={**_clean_env(), **env},
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )


#: Unlisted helpers that load the library by a name the static gate cannot evaluate.
TRIPWIRE_HELPERS = {
    "import by variable name": "import importlib\nimport os\n\n"
    "M = importlib.import_module(os.environ['FIXTURE_MODULE'])\n",
    "load by variable path": "import importlib.util\nimport os\n\n"
    "S = importlib.util.spec_from_file_location('m', os.environ['FIXTURE_PATH'])\n"
    "S.loader.exec_module(importlib.util.module_from_spec(S))\n",
    "exec of the source": "import os\n\nexec(open(os.environ['FIXTURE_PATH']).read())\n",
    "swallowed failure": "import importlib\nimport os\n\ntry:\n"
    "    importlib.import_module(os.environ['FIXTURE_MODULE'])\nexcept BaseException:\n    pass\n",
    "child process": "import os\nimport subprocess\nimport sys\n\n"
    "subprocess.run([sys.executable, os.environ['FIXTURE_PATH']], check=False)\n",
}


@pytest.mark.parametrize("case", sorted(TRIPWIRE_HELPERS))
def test_the_tripwire_stops_a_library_load_the_static_gate_misses(
    tmp_path: Path, case: str
) -> None:
    """The guard ends --help before the library's file is read; the library never runs.

    This runs the synthetic importer (in tmp_path, under the clean environment), never the
    library: the sentinel proves it. The static gate is blind to these by design (documented).
    """
    sentinel = tmp_path / "ran"
    library = _write(tmp_path / LIB, f"open({str(sentinel)!r}, 'w').close()\n")
    _write(tmp_path / "scripts/helper.py", TRIPWIRE_HELPERS[case])
    importer = _importer(tmp_path, "import helper")
    assert _help_runnable([importer], tmp_path, LIBS) == [importer]  # the static gate misses it
    work = tmp_path / "work"
    work.mkdir()
    result = _guarded_help(
        importer, work, frozenset({"job_lib"}), FIXTURE_MODULE="job_lib", FIXTURE_PATH=str(library)
    )
    assert result.returncode == TRIPWIRE_EXIT, (result.returncode, result.stderr[-800:])
    assert "help tripwire" in result.stderr
    assert not sentinel.exists(), "the library ran"


def test_the_tripwire_lets_a_plain_help_through(tmp_path: Path) -> None:
    _write(tmp_path / LIB, "VALUE = 1\n")
    _write(tmp_path / "scripts/helper.py", "import json\n\nX = json.dumps({})\n")
    importer = _importer(tmp_path, "import helper")
    work = tmp_path / "work"
    work.mkdir()
    result = _guarded_help(importer, work, frozenset({"job_lib"}))
    assert result.returncode == 0, result.stderr[-800:]
    assert "usage: cmd.py" in result.stdout


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


WITHHELD = _help_withheld(SCRIPTS)
RUNNABLE = _help_runnable(SCRIPTS)


def test_help_not_run_lists_exactly_the_library_importers() -> None:
    """The pinned list matches what the gate finds, so a new importer is named in a commit."""
    found = {p.relative_to(ROOT).as_posix(): r for p, r in WITHHELD.items()}
    assert sorted(found) == sorted(HELP_NOT_RUN), found
    for rel, reason in HELP_NOT_RUN.items():
        assert any(reason.split()[-1] in r for r in found[rel]), (rel, found[rel])
    assert not {ROOT / rel for rel in HELP_NOT_RUN} & set(RUNNABLE)


@pytest.mark.parametrize("script", sorted(HELP_NOT_RUN))
def test_help_not_run_is_reported(script: str) -> None:
    """Reported, not passed: these scripts get the static checks only."""
    pytest.skip(f"--help intentionally not executed for {script}: {HELP_NOT_RUN[script]}")


def _clean_env() -> dict[str, str]:
    """``env -i`` plus only what a Python process needs: no HOME, no DB or broker variables."""
    return {
        "PATH": "/usr/bin:/bin",
        "PYTHONPATH": str(ROOT / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "LC_ALL": "C.UTF-8",
    }


@pytest.fixture(scope="session")
def mpl_config_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A matplotlib config directory whose font cache is already built.

    ``_clean_env`` has no HOME, so a script that imports matplotlib would build the font cache
    inside its guarded ``--help`` run, and on a fresh runner that spawns ``fc-list`` (the tripwire
    stops it). Building the cache here, outside the guard, keeps the tripwire strict.
    """
    path = tmp_path_factory.mktemp("mplconfig")
    subprocess.run(
        [sys.executable, "-c", "import matplotlib.font_manager"],
        env={**_clean_env(), "MPLCONFIGDIR": str(path)},
        capture_output=True,
        timeout=120,
        check=False,
    )
    return path


@pytest.mark.slow
@pytest.mark.parametrize("script", RUNNABLE, ids=_ids(RUNNABLE))
def test_script_help_exits_zero_with_usage(
    script: Path, tmp_path: Path, mpl_config_dir: Path
) -> None:
    # Re-checked at run time too, in case a script or a module it imports changed after collection.
    assert _static_problems(script) == []
    assert _library_imports(script) == []
    stems = frozenset(Path(lib).stem for lib in LIBRARIES)
    result = _guarded_help(script, tmp_path, stems, MPLCONFIGDIR=str(mpl_config_dir))
    assert result.returncode != TRIPWIRE_EXIT, f"tripwire: {result.stderr[-800:]}"
    assert result.returncode == 0, result.stderr[-800:]
    assert "usage" in result.stdout.lower(), result.stdout[:300]
    assert not os.listdir(tmp_path), "--help wrote into its working directory"
