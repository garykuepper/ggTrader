"""Contracts for the legacy seams, new-module imports and the production boundary (Task 50).

Immutable behavioral set (hash-recorded at Task 51). Nothing here regenerates a golden or a
digest, and every test passes on today's code: no `xfail`, no skip. Where a rule concerns modules
that do not exist yet, the test *discovers* them (``src/ggTrader/paper/*.py`` minus
`EXISTING_MODULES`), so it checks nothing today and everything from the task that creates a
module on. Each rule is proved on synthetic sources in
`test_the_new_module_scanner_detects_each_violation` (and its siblings), so a pass over an empty
set is a visible pass and not a silent one.
"""

from __future__ import annotations

import ast
import copy
import os
import re
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

from ggTrader.paper import ledger as ledger_module
from ggTrader.paper import trader as trader_module
from ggTrader.paper.corporate_actions import CorporateActions
from ggTrader.paper.execution import OrderExecutor
from ggTrader.paper.risk import RiskConfig, RiskGuard
from ggTrader.paper.trader import PaperTrader
from tests.paper import rules_support as rs
from tests.paper.equivalence.run_with import implementation
from tests.paper.rules_support import IMPLS
from tests.paper.transcript import harness
from tests.paper.transcript.harness import Scenario
from tests.paper.transcript.injected import HarnessLedger
from tests.paper.transcript.scenarios import SCENARIOS

REPO = Path(__file__).resolve().parents[2]
PAPER_DIR = REPO / "src" / "ggTrader" / "paper"
TESTS_DIR = REPO / "tests"
PACKAGE = "ggTrader.paper"
TRADER_MODULE = "ggTrader.paper.trader"
TRADER_LOGGER = "ggTrader.paper.trader"

#: The modules of `src/ggTrader/paper/` at the phase's base commit. Anything else in the folder is a
#: module Phase 5 added, and is held to the new-module rules.
EXISTING_MODULES = frozenset(
    {
        "__init__",
        "alpaca_broker",
        "cash_sweep",
        "catastrophe_stop",
        "config",
        "constants",
        "dividend_check",
        "gates",
        "ledger",
        "live_bars",
        "live_signals",
        "models",
        "notifier",
        "overlay",
        "persist",
        "ports",
        "risk",
        "signal_runner",
        "split_check",
        "trader",
        "untradable",
    }
)

#: Imports a new module never makes: the database, the network, the broker, the real notifier, the
#: signal runner (reached only through `RunHooks`) and the research lab.
FORBIDDEN_TOP_LEVEL = frozenset({"psycopg2", "sqlalchemy", "requests", "httpx"})
FORBIDDEN_TOP_LEVEL_PREFIX = "alpaca"
FORBIDDEN_DOTTED = (
    "ggTrader.paper.persist",
    "ggTrader.paper.alpaca_broker",
    "ggTrader.paper.notifier",
    "ggTrader.paper.signal_runner",
    "ggTrader.lab",
)
#: Clocks and ids come through `RunHooks`; a new module imports neither.
HOOKED_MODULES = frozenset({"time", "uuid"})
CLOCK_CALL_ATTRS = frozenset({"now", "today"})
ENV_ATTRS = frozenset({"environ", "getenv"})
#: The module-level settings accessors of the per-run snapshot, by module.
SETTINGS_ACCESSORS: dict[str, frozenset[str]] = {
    "cash_sweep": frozenset(
        {"sweep_enabled", "sweep_symbol", "reserve_pct", "min_clip_usd", "buy_trigger_pct"}
    ),
    "catastrophe_stop": frozenset({"catastrophe_stop_enabled", "catastrophe_stop_pct"}),
    "untradable": frozenset({"configured_untradable"}),
}
#: Modules the tests patch by attribute on the trader namespace: new code calls them as
#: `module.fn(...)`, never `from module import fn`.
ATTRIBUTE_ONLY_MODULES = ("ggTrader.paper.cash_sweep", "ggTrader.paper.catastrophe_stop")
#: Files where a generator frame could turn a collaborator's `StopIteration` into `RuntimeError`.
GENERATOR_FREE_FILES = frozenset(
    {"planning.py", "order_flow.py", "preflight.py", "portfolio.py", "risk_assessor.py"}
)
GENEXP_ONLY_FILE = "reporter.py"
#: The generator expressions `_run` has in region R7 at 73c35417, moved verbatim into `reporter.py`
#: (`ast.unparse` text, and the line each has in trader.py at that commit). Each may appear once.
GENEXP_ALLOWED: tuple[tuple[str, int], ...] = (
    (
        "(corrected_updated_positions[sym]['market_value'] - "
        "updated_positions[sym]['market_value'] for sym in split_corrections "
        "if sym in updated_positions)",
        1462,
    ),
    ("(isinstance(p, (int, float)) and (not isinstance(p, bool)) for p in pnl_values)", 1491),
    ("(f'{u}={w:.0%}' for u, w in weights.items())", 1508),
)
EXCEPTION_FREE_FILE = "preflight.py"

#: The five names that move down to `constants.py` / `run_helpers.py`; `trader.py` keeps them bound.
MOVED_CONSTANTS: dict[str, tuple[str, int]] = {
    "_MAX_SIGNAL_AGE_DAYS": ("MAX_SIGNAL_AGE_DAYS", 4),
    "_STALE_PENDING_DAYS": ("STALE_PENDING_DAYS", 5),
    "_SPLIT_LOOKBACK_DAYS": ("SPLIT_LOOKBACK_DAYS", 14),
}
MOVED_HELPERS: dict[str, str] = {"_as_date": "as_date", "_account_read_halt": "account_read_halt"}
MOVED_ALIASES = frozenset(MOVED_CONSTANTS) | frozenset(MOVED_HELPERS)

#: Names a test or the harness may patch on the trader namespace, beyond the 26 persist names.
REQUIRED_TRADER_NAMES = (
    "find_split_applied_symbols",
    "catastrophe_stop",
    "cash_sweep",
    "time",
    "datetime",
    "uuid4",
    "_today_et",
    "_now_pt",
    "generate_core_signals",
    "refresh_benchmark_tape",
    "live_membership_source",
    "require_live_membership_spec",
    "AlpacaBroker",
    "TelegramNotifier",
    "DryRunNotifier",
    "PaperTrader",
    "LIVE_MEMBERSHIP_SPEC",
    "run_paper_trading",
    *sorted(MOVED_ALIASES),
)
#: The step objects that own the bodies of the former private trader seams (retired in refactor
#: completion II). A test replaces the named method on the class; the run reaches it there.
STEP_OWNERS: dict[str, type] = {
    "OrderExecutor": OrderExecutor,
    "CorporateActions": CorporateActions,
}
#: The eight private `PaperTrader` methods that tests used to assign or patch by name. They are
#: gone; the step objects above own the bodies.
RETIRED_SEAMS = (
    "_submit_order",
    "_reconcile_order_intents",
    "_flag_if_stale",
    "_reconcile_pending_orders",
    "_poll_orders",
    "_missed_exits",
    "_compute_split_corrections",
    "_accrue_dividends",
)


# =========================================================================== source scanners


@dataclass(frozen=True)
class Finding:
    """One violation a scanner found: the rule, the line and a short detail."""

    rule: str
    line: int
    detail: str = ""


def _dotted(node: ast.AST) -> str | None:
    """``a.b.c`` for a chain of attribute accesses on a name, else ``None``."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _resolve_from(node: ast.ImportFrom, package: str) -> str:
    """The absolute dotted module an ``ImportFrom`` names (relative forms resolved in `package`)."""
    module = node.module or ""
    if node.level == 0:
        return module
    parts = package.split(".")
    base = parts[: len(parts) - (node.level - 1)] if node.level > 1 else parts
    return ".".join([*base, module] if module else base)


def _import_names(tree: ast.AST, package: str) -> Iterator[tuple[ast.AST, str]]:
    """Every dotted name an import statement can bring in, one tuple per candidate.

    ``from a.b import c`` yields ``a.b`` and ``a.b.c`` (``c`` may be a submodule), so
    ``from ggTrader.paper import trader`` is seen as an import of ``ggTrader.paper.trader``.
    Function-local imports are included: the walk visits every node.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node, alias.name
        elif isinstance(node, ast.ImportFrom):
            base = _resolve_from(node, package)
            yield node, base
            for alias in node.names:
                if alias.name != "*":
                    yield node, f"{base}.{alias.name}"


def _is_or_under(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(prefix + ".")


def _is_forbidden_import(name: str) -> bool:
    top = name.split(".")[0]
    if top in FORBIDDEN_TOP_LEVEL or top.startswith(FORBIDDEN_TOP_LEVEL_PREFIX):
        return True
    return any(_is_or_under(name, prefix) for prefix in FORBIDDEN_DOTTED)


def _logger_name_ok(node: ast.Call, resolve: Callable[[str | None], str | None]) -> bool:
    if len(node.args) != 1 or node.keywords:
        return False
    return resolve(_dotted(node.args[0])) == "ggTrader.paper.constants.TRADER_LOGGER_NAME"


def _import_aliases(nodes: Iterable[ast.AST]) -> dict[str, str]:
    """Map imported local names to their fully qualified origins."""
    aliases: dict[str, str] = {}
    for node in nodes:
        if isinstance(node, ast.Import):
            for alias in node.names:
                aliases[alias.asname or alias.name.split(".")[0]] = (
                    alias.name if alias.asname else alias.name.split(".")[0]
                )
        elif isinstance(node, ast.ImportFrom):
            base = _resolve_from(node, PACKAGE)
            for alias in node.names:
                if alias.name != "*":
                    aliases[alias.asname or alias.name] = f"{base}.{alias.name}"
    return aliases


def scan_new_module(source: str, filename: str) -> list[Finding]:
    """The rule violations of one new `paper/` module (`filename` is its base name)."""
    tree = ast.parse(source)
    found: list[Finding] = []
    # This conservative file-wide map does not model scope or later rebinding.
    aliases = _import_aliases(ast.walk(tree))

    def resolve(name: str | None) -> str | None:
        if name is None:
            return None
        first, separator, rest = name.partition(".")
        return aliases.get(first, first) + (separator + rest if separator else "")

    def add(rule: str, node: ast.AST, detail: str = "") -> None:
        found.append(Finding(rule, getattr(node, "lineno", 0), detail))

    for node, name in _import_names(tree, PACKAGE):
        if _is_forbidden_import(name):
            add("forbidden-import", node, name)
        if _is_or_under(name, TRADER_MODULE):
            add("imports-trader", node, name)
        if name.split(".")[0] in HOOKED_MODULES:
            add("time-or-uuid-import", node, name)
        if name == "importlib" or name.startswith("importlib."):
            add("dynamic-import", node, name)
        if (
            isinstance(node, ast.ImportFrom)
            and _resolve_from(node, PACKAGE) in ATTRIBUTE_ONLY_MODULES
        ):
            add("from-import-of-attribute-only-module", node, name)
        if isinstance(node, ast.ImportFrom) and name == "os.environ":
            add("environment-read", node, name)
        if isinstance(node, ast.ImportFrom) and name == "os.getenv":
            add("environment-read", node, name)
        if isinstance(node, ast.ImportFrom):
            module = _resolve_from(node, PACKAGE).split(".")[-1]
            accessor = name.rsplit(".", 1)[-1]
            if accessor in SETTINGS_ACCESSORS.get(module, frozenset()):
                add("settings-accessor", node, name)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in CLOCK_CALL_ATTRS:
                add("clock-call", node, f".{func.attr}(")
            if isinstance(func, ast.Name) and func.id == "__import__":
                add("dynamic-import", node, "__import__")
            if resolve(_dotted(func)) == "logging.getLogger" or (
                isinstance(func, ast.Name)
                and resolve(func.id) in ("getLogger", "logging.getLogger")
            ):
                if not _logger_name_ok(node, resolve):
                    add("logger-name", node, ast.unparse(node))
        if isinstance(node, ast.Attribute):
            owner = resolve(_dotted(node.value))
            module = owner.split(".")[-1] if owner else None
            if module in SETTINGS_ACCESSORS and node.attr in SETTINGS_ACCESSORS[module]:
                add("settings-accessor", node, f"{module}.{node.attr}")
            if node.attr in ENV_ATTRS:
                add("environment-read", node, node.attr)
            dotted = resolve(_dotted(node))
            if dotted is not None and _is_or_under(dotted, TRADER_MODULE):
                add("imports-trader", node, dotted)
            if dotted is not None and _is_forbidden_import(dotted):
                add("forbidden-import", node, dotted)
        if isinstance(node, ast.Name) and node.id in ENV_ATTRS:
            add("environment-read", node, node.id)

    if filename == EXCEPTION_FREE_FILE:
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    base_name = (
                        base.attr if isinstance(base, ast.Attribute) else getattr(base, "id", "")
                    )
                    if base_name in ("Exception", "BaseException") or base_name.endswith(
                        ("Error", "Exception")
                    ):
                        add("exception-class", node, f"{node.name}({base_name})")

    if filename in GENERATOR_FREE_FILES or filename == GENEXP_ONLY_FILE:
        allowed = (
            Counter(text for text, _line in GENEXP_ALLOWED)
            if filename == GENEXP_ONLY_FILE
            else Counter()
        )
        for node in ast.walk(tree):
            if isinstance(node, ast.Yield | ast.YieldFrom):
                add("generator", node, type(node).__name__)
            elif isinstance(node, ast.GeneratorExp):
                text = ast.unparse(node)
                if allowed[text] > 0:
                    allowed[text] -= 1
                else:
                    add("generator", node, text)

    return sorted(found, key=lambda f: (f.line, f.rule, f.detail))


def _defined_names(tree: ast.Module) -> dict[str, ast.AST]:
    """Module-level names bound by an assignment or a ``def``, with the node that binds them."""
    names: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names[target.id] = node
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names[node.target.id] = node
        elif isinstance(node, ast.FunctionDef):
            names[node.name] = node
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    names[alias.asname or alias.name] = alias
    return names


def _binds_to(node: ast.AST, home_name: str, home_module: str, imports: dict[str, str]) -> bool:
    """Check that a trader alias resolves to its canonical home."""
    if isinstance(node, ast.alias):
        return imports.get(node.asname or node.name) == f"ggTrader.paper.{home_module}.{home_name}"
    if not isinstance(node, ast.Assign | ast.AnnAssign) or node.value is None:
        return False
    value = _dotted(node.value)
    if value is None:
        return False
    first, sep, rest = value.partition(".")
    resolved = imports.get(first, first) + (sep + rest if sep else "")
    return resolved == f"ggTrader.paper.{home_module}.{home_name}"


def _literal_int(node: ast.AST) -> int | None:
    value = getattr(node, "value", None)
    if isinstance(value, ast.Constant) and isinstance(value.value, int):
        return value.value
    return None


def home_findings(
    constants_src: str, run_helpers_src: str | None, trader_src: str
) -> list[Finding]:
    """Check that each moved constant and helper has exactly one home.

    Before the move (`run_helpers_src` is ``None``) `trader.py` holds the five names as it did at
    the base commit and `constants.py` holds none of the three constants. Once `run_helpers.py`
    exists, `constants.py` defines the constants, `run_helpers.py` the helpers, and `trader.py`
    only binds the old names to them (no literal, no ``def``).
    """
    constants = _defined_names(ast.parse(constants_src))
    trader_tree = ast.parse(trader_src)
    trader = _defined_names(trader_tree)
    imports = _import_aliases(trader_tree.body)
    helpers = _defined_names(ast.parse(run_helpers_src)) if run_helpers_src is not None else None
    found: list[Finding] = []

    for old, (new, value) in MOVED_CONSTANTS.items():
        in_trader = trader.get(old)
        in_constants = constants.get(new)
        if helpers is None:
            if in_trader is None:
                found.append(Finding("name-lost", 0, old))
            elif _literal_int(in_trader) != value:
                found.append(Finding("value-drift", getattr(in_trader, "lineno", 0), old))
            if in_constants is not None:
                found.append(Finding("duplicated", getattr(in_constants, "lineno", 0), new))
        else:
            if in_constants is None:
                found.append(Finding("missing-home", 0, new))
            elif _literal_int(in_constants) != value:
                found.append(Finding("value-drift", getattr(in_constants, "lineno", 0), new))
            if in_trader is None:
                found.append(Finding("name-lost", 0, old))
            elif not _binds_to(in_trader, new, "constants", imports):
                found.append(Finding("duplicated", getattr(in_trader, "lineno", 0), old))

    for old, new in MOVED_HELPERS.items():
        in_trader = trader.get(old)
        if helpers is None:
            if not isinstance(in_trader, ast.FunctionDef):
                found.append(Finding("name-lost", 0, old))
        else:
            if not isinstance(helpers.get(new), ast.FunctionDef):
                found.append(Finding("missing-home", 0, new))
            if in_trader is None:
                found.append(Finding("name-lost", 0, old))
            elif not _binds_to(in_trader, new, "run_helpers", imports):
                found.append(Finding("duplicated", getattr(in_trader, "lineno", 0), old))
    return found


def _trader_module_aliases(tree: ast.AST) -> set[str]:
    """Local names a file binds to the trader module (``import ... as x``, ``from ... import``)."""
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == TRADER_MODULE:
                    aliases.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            for alias in node.names:
                if f"{node.module}.{alias.name}" == TRADER_MODULE:
                    aliases.add(alias.asname or alias.name)
    return aliases


_PATCH_CALLS = ("patch.object", "monkeypatch.setattr", "setattr", "mocker.patch.object")
_DOTTED_REF = re.compile(re.escape(TRADER_MODULE) + r"\.(\w+)")


def patch_targets(source: str) -> tuple[set[str], set[str]]:
    """The names a test source patches or assigns on the trader module and on `PaperTrader`.

    Returns ``(module_names, class_names)``. Module names come from dotted strings
    (``"<module>.name[.more]"``), from ``patch.object``/``setattr`` and attribute assignment on a
    local alias of the module, and from the keys of a dict named ``replacements``.
    """
    tree = ast.parse(source)
    aliases = _trader_module_aliases(tree)
    module_names = set(_DOTTED_REF.findall(source))
    class_names: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and _dotted(node.func) in _PATCH_CALLS
            and len(node.args) >= 2
        ):
            owner, name = node.args[0], node.args[1]
            if isinstance(name, ast.Constant) and isinstance(name.value, str):
                if isinstance(owner, ast.Name) and owner.id in aliases:
                    module_names.add(name.value)
                elif isinstance(owner, ast.Name) and owner.id == "PaperTrader":
                    class_names.add(name.value)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id in aliases
                ):
                    module_names.add(target.attr)
                if isinstance(target, ast.Name) and target.id == "replacements":
                    module_names.update(_dict_keys(node.value))
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == "replacements":
                module_names.update(_dict_keys(node.value))
    return module_names, class_names


def _dict_keys(node: ast.AST | None) -> set[str]:
    """The string keys of a dict literal (empty for anything else)."""
    if not isinstance(node, ast.Dict):
        return set()
    return {
        key.value
        for key in node.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }


def alias_patches(source: str) -> list[Finding]:
    """Patches or assignments of the five moved names on the trader module."""
    tree = ast.parse(source)
    aliases = _trader_module_aliases(tree)
    found = [
        Finding("alias-patched", source[: m.start()].count("\n") + 1, m.group(1))
        for m in _DOTTED_REF.finditer(source)
        if m.group(1) in MOVED_ALIASES
    ]
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and _dotted(node.func) in _PATCH_CALLS
            and len(node.args) >= 2
        ):
            owner, name = node.args[0], node.args[1]
            if (
                isinstance(owner, ast.Name)
                and owner.id in aliases
                and isinstance(name, ast.Constant)
                and name.value in MOVED_ALIASES
            ):
                found.append(Finding("alias-patched", node.lineno, name.value))
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id in aliases
                    and target.attr in MOVED_ALIASES
                ):
                    found.append(Finding("alias-patched", node.lineno, target.attr))
    return found


IN_MEMORY_LEDGER = "InMemoryLedger"
#: Production calls that build a trader; none may hand it a ledger.
LEDGER_ENTRY_POINTS = frozenset({"PaperTrader", "run_paper_trading"})
LEDGER_HOME = Path("src/ggTrader/paper/ledger.py")


def in_memory_ledger_references(source: str) -> list[Finding]:
    """Where a production source names `InMemoryLedger` or passes a ``ledger=`` argument."""
    tree = ast.parse(source)
    found: list[Finding] = []
    for node in ast.walk(tree):
        line = getattr(node, "lineno", 0)
        if isinstance(node, ast.Name) and node.id == IN_MEMORY_LEDGER:
            found.append(Finding("in-memory-ledger", line, "name"))
        elif isinstance(node, ast.Attribute) and node.attr == IN_MEMORY_LEDGER:
            found.append(Finding("in-memory-ledger", line, "attribute"))
        elif isinstance(node, ast.alias) and IN_MEMORY_LEDGER in (node.name, node.asname):
            found.append(Finding("in-memory-ledger", line, "import"))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if IN_MEMORY_LEDGER in node.value:
                found.append(Finding("in-memory-ledger", line, "string"))
        elif isinstance(node, ast.Call):
            callee = (_dotted(node.func) or "").split(".")[-1]
            if callee in LEDGER_ENTRY_POINTS and any(k.arg == "ledger" for k in node.keywords):
                found.append(Finding("ledger-argument", line, callee))
    return found


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _new_modules() -> list[Path]:
    return sorted(p for p in PAPER_DIR.glob("*.py") if p.stem not in EXISTING_MODULES)


# =========================================================================== a driven run


@dataclass
class Driven:
    """One run with the trader kept, so a test can read what it recorded and what it raised."""

    run: harness._Run
    trader: PaperTrader
    result: dict | None
    error: Exception | None

    def view(self) -> tuple[list[dict], dict | None, str | None]:
        """Everything a run is compared by: the recorded calls, the result and the error text."""
        error = None if self.error is None else f"{type(self.error).__name__}: {self.error}"
        return self.run.recorder.calls, self.result, error


def drive(
    impl_name: str,
    scenario: Scenario,
    configure: Callable[[PaperTrader, harness._Run, ExitStack], None] | None = None,
) -> Driven:
    """Build the trader of `scenario` in the event environment, let `configure` act, then run it.

    `configure` runs after construction and before `run()`, inside every patch of the environment,
    so it can assign a private method on the instance, patch the class, or replace a module-level
    name the way a test does. Only `Exception` is caught (a `HarnessError` fails the test).
    """
    result: dict | None = None
    error: Exception | None = None
    with rs.event_environment(scenario) as run, ExitStack() as stack:
        s = run.s
        with implementation(impl_name):
            trader = PaperTrader(
                run.broker,  # type: ignore[arg-type]
                run.notifier,
                dry_run=s.dry_run,
                read_only=s.read_only,
                membership=s.membership,
                ledger=HarnessLedger(run),
            )
            if configure is not None:
                configure(trader, run, stack)
            try:
                result = trader.run()
            except Exception as exc:
                error = exc
        return Driven(run=run, trader=trader, result=result, error=error)


def _same_run(a: Driven, b: Driven) -> bool:
    """True when two runs recorded the same calls, returned the same result and raised the same."""
    return a.view() == b.view()


# =========================================================================== patch targets


def test_every_patch_target_still_resolves_in_the_trader_namespace():
    """Every name a test or the harness patches on the trader module is still an attribute of it.

    A moved body that stops importing a name the tests patch would leave the patch pointing at
    nothing (`patch` raises, or an assignment silently does nothing). The scan reads every file
    under `tests/` for dotted strings, ``patch.object``/``setattr``/assignment on a local alias of
    the module, and the harness's ``replacements`` dict.
    """
    module_names: dict[str, set[str]] = {}
    class_names: dict[str, set[str]] = {}
    for path in sorted(TESTS_DIR.rglob("*.py")):
        names, classes = patch_targets(_read(path))
        rel = str(path.relative_to(REPO))
        if names:
            module_names[rel] = names
        if classes:
            class_names[rel] = classes
    assert "tests/paper/transcript/harness.py" in module_names, "the harness patches nothing?"
    assert sum(len(names) for names in module_names.values()) > 60  # the scan reaches the tests
    missing = {
        (rel, name)
        for rel, names in module_names.items()
        for name in names
        if not hasattr(trader_module, name)
    }
    assert not missing, (
        f"patched names that no longer exist in the trader namespace: {sorted(missing)}"
    )
    missing_methods = {
        (rel, name)
        for rel, names in class_names.items()
        for name in names
        if not hasattr(PaperTrader, name)
    }
    assert not missing_methods, (
        f"patched PaperTrader methods that are gone: {sorted(missing_methods)}"
    )


def test_the_patch_target_scanner_finds_each_form():
    dotted = "ggTrader.paper.trader"
    source = (
        "from unittest.mock import patch\n"
        "from ggTrader.paper import trader as tm\n"
        "from ggTrader.paper.trader import PaperTrader\n"
        f'@patch("{dotted}.string_name")\n'
        "def t(monkeypatch):\n"
        '    patch.object(tm, "object_name", 1)\n'
        '    monkeypatch.setattr(tm, "setattr_name", 1)\n'
        "    tm.assigned_name = 1\n"
        '    patch.object(PaperTrader, "_method_name", 1)\n'
        '    replacements = {"dict_name": 1, "other_dict_name": 2}\n'
        '    replacements: dict[str, object] = {"annotated_dict_name": 1}\n'
        f'    x = "{dotted}.PaperTrader._nested"\n'
    )
    names, classes = patch_targets(source)
    assert names == {
        "string_name",
        "object_name",
        "setattr_name",
        "assigned_name",
        "dict_name",
        "other_dict_name",
        "annotated_dict_name",
        "PaperTrader",
    }
    assert classes == {"_method_name"}
    assert patch_targets("x = 1\n") == (set(), set())


def retired_seam_uses(source: str) -> list[tuple[int, str]]:
    """Lines that assign, patch or set a retired seam name on any object, as ``(line, name)``.

    `patch.object(PaperTrader, "_poll_orders")` fails loudly (the class has no such attribute), but
    `trader._poll_orders = Mock()` or `patch.object(trader, "_poll_orders", ...)` on an instance
    silently intercepts nothing, so those forms are flagged wherever they appear.
    """
    hits: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Attribute) and target.attr in RETIRED_SEAMS:
                    hits.append((node.lineno, target.attr))
        elif isinstance(node, ast.Call) and len(node.args) >= 2:
            name = node.args[1]
            if (
                _dotted(node.func) in (*_PATCH_CALLS, "patch.multiple", "mock.patch.object")
                and isinstance(name, ast.Constant)
                and name.value in RETIRED_SEAMS
            ):
                hits.append((node.lineno, name.value))
    return sorted(hits)


def test_no_test_assigns_or_patches_a_retired_seam_name():
    """A leftover `trader._poll_orders = Mock()` would silently stop intercepting anything."""
    found = {
        str(path.relative_to(REPO)): hits
        for path in sorted(TESTS_DIR.rglob("*.py"))
        if (hits := retired_seam_uses(_read(path)))
    }
    assert not found, f"tests still assign or patch a retired seam name: {found}"


def test_the_retired_seam_scanner_finds_each_form():
    source = (
        "from unittest.mock import patch\n"
        "trader._poll_orders = 1\n"
        "self.trader._missed_exits: object = 1\n"
        'patch.object(trader, "_submit_order", 1)\n'
        'monkeypatch.setattr(trader, "_flag_if_stale", 1)\n'
        'setattr(trader, "_accrue_dividends", 1)\n'
        "trader._other = 1\n"
        'patch.object(trader, "_other", 1)\n'
    )
    assert retired_seam_uses(source) == [
        (2, "_poll_orders"),
        (3, "_missed_exits"),
        (4, "_submit_order"),
        (5, "_flag_if_stale"),
        (6, "_accrue_dividends"),
    ]
    assert retired_seam_uses("x = 1\n") == []


def test_required_trader_names_stay_bound():
    """The names the plan lists as patchable are still bound; the eight private seams are gone."""
    missing = [name for name in REQUIRED_TRADER_NAMES if not hasattr(trader_module, name)]
    assert not missing, missing
    for name in RETIRED_SEAMS:
        assert not hasattr(PaperTrader, name), name


def test_trader_persist_names_unchanged():
    """The protocol and PostgreSQL ledger expose the same persistence operations."""
    from ggTrader.paper.persist import PgPaperLedger

    names = {
        name
        for name, value in ledger_module.PaperLedger.__dict__.items()
        if not name.startswith("_") and callable(value)
    }
    assert names
    actual = {
        name
        for name, value in PgPaperLedger.__dict__.items()
        if not name.startswith("_") and callable(value)
    }
    assert names == actual


# =========================================================================== the legacy seams


@pytest.mark.parametrize("impl_name", IMPLS)
class TestLegacySeamsInterceptTheRun:
    """What a test assigns or patches after construction still wins inside the run.

    Each case runs a minimal scenario twice, in the event environment: untouched, and with a
    spy that delegates to the real thing. The spy must be called (the run reached it through the
    instance, the class or the module namespace, not through something captured at construction)
    and, because it only delegates, the run must record exactly what the untouched one did.
    """

    SEAMS = (
        pytest.param("OrderExecutor.submit_order", "plain_buys_and_sells", id="submit_order"),
        pytest.param("OrderExecutor.reconcile_order_intents", "plain_buys_and_sells", id="intents"),
        pytest.param(
            "OrderExecutor.reconcile_pending_orders",
            "plain_buys_and_sells",
            id="pending_orders",
        ),
        pytest.param("OrderExecutor.poll_orders", "plain_buys_and_sells", id="poll_orders"),
        pytest.param("OrderExecutor.flag_if_stale", "legacy_pending_stale", id="flag_if_stale"),
        pytest.param("CorporateActions.missed_exits", "missed_exit_catch_up", id="missed_exits"),
        pytest.param(
            "CorporateActions.split_corrections", "split_correction", id="split_corrections"
        ),
        pytest.param("CorporateActions.accrue_dividends", "new_dividend_accrual", id="dividends"),
    )

    @pytest.mark.parametrize(("method", "scenario_name"), SEAMS)
    def test_private_method_seam(self, impl_name, method, scenario_name):
        """Replacing a step method on its class (`Owner.method`) intercepts the run."""
        scenario = SCENARIOS[scenario_name]
        baseline = drive(impl_name, scenario)
        calls: list[tuple] = []

        def configure(_trader: PaperTrader, _run: harness._Run, stack: ExitStack) -> None:
            owner_name, _, attr = method.partition(".")
            owner = STEP_OWNERS[owner_name]
            step_function = getattr(owner, attr)

            def step_spy(self: Any, *args: Any, **kwargs: Any) -> Any:
                calls.append((args, kwargs))
                return step_function(self, *args, **kwargs)

            stack.enter_context(patch.object(owner, attr, step_spy))

        intercepted = drive(impl_name, scenario, configure)
        assert calls, f"{method} was never reached through the step object"
        assert _same_run(baseline, intercepted)

    def test_submit_order_seam_sees_every_order_and_replaces_the_broker_call(self, impl_name):
        """A replaced `OrderExecutor.submit_order` gets every order's arguments, once per order."""
        scenario = SCENARIOS["plain_buys_and_sells"]
        baseline = drive(impl_name, scenario)
        expected = len(rs.submissions(_transcript_of(baseline)))
        assert expected >= 2
        seen: list[tuple] = []

        def configure(_trader: PaperTrader, _run: harness._Run, stack: ExitStack) -> None:
            real = OrderExecutor.submit_order

            def spy(
                self: OrderExecutor, side: str, symbol: str, amount: float, reason: str | None
            ) -> str:
                seen.append((side, symbol, amount, reason))
                return real(self, side, symbol, amount, reason)

            stack.enter_context(patch.object(OrderExecutor, "submit_order", spy))

        intercepted = drive(impl_name, scenario, configure)
        assert len(seen) == expected
        assert {side for side, *_ in seen} == {"BUY", "SELL"}
        assert _same_run(baseline, intercepted)

    def test_reconcile_pending_orders_replacement_decides_the_pending_symbols(self, impl_name):
        """A replacement of `reconcile_pending_orders` is the source of the symbols skipped."""
        scenario = SCENARIOS["plain_buys_and_sells"]
        baseline = drive(impl_name, scenario)
        bought = [s for side, s, _ in rs.submissions(_transcript_of(baseline)) if side == "BUY"]
        assert bought, "the scenario buys something"
        blocked = bought[0]
        replacement = Mock(return_value={blocked})

        def configure(_trader: PaperTrader, _run: harness._Run, stack: ExitStack) -> None:
            stack.enter_context(
                patch.object(OrderExecutor, "reconcile_pending_orders", replacement)
            )

        intercepted = drive(impl_name, scenario, configure)
        replacement.assert_called_once_with()
        after = [s for side, s, _ in rs.submissions(_transcript_of(intercepted)) if side == "BUY"]
        assert blocked not in after
        assert blocked in bought

    def test_risk_guard_replacement_and_config_replacement_change_the_slot_count(self, impl_name):
        """`_risk` and `_risk.cfg`, assigned after construction, are the limits the run applies."""
        scenario = SCENARIOS["plain_buys_and_sells"]
        baseline = drive(impl_name, scenario)
        default_buys = len(baseline.result["buys"])
        assert default_buys >= 2

        def replace_guard(trader: PaperTrader, _run: harness._Run, _stack: ExitStack) -> None:
            trader._risk = RiskGuard(RiskConfig(max_positions=1))

        def replace_config(trader: PaperTrader, _run: harness._Run, _stack: ExitStack) -> None:
            trader._risk.cfg = RiskConfig(max_positions=1)

        guarded = drive(impl_name, scenario, replace_guard)
        configured = drive(impl_name, scenario, replace_config)
        assert len(guarded.result["buys"]) < default_buys
        assert _same_run(guarded, configured)

    def test_dry_run_assigned_after_construction_selects_the_dry_path(self, impl_name):
        """`_dry_run` set to ``True`` after construction takes the dry path: no order, no lock."""
        scenario = SCENARIOS["plain_buys_and_sells"]
        assert scenario.dry_run is False
        live = drive(impl_name, scenario)
        assert rs.submissions(_transcript_of(live))
        assert "lock.enter" in _targets(live)

        def configure(trader: PaperTrader, _run: harness._Run, _stack: ExitStack) -> None:
            trader._dry_run = True

        dry = drive(impl_name, scenario, configure)
        assert dry.error is None, dry.error
        targets = _targets(dry)
        assert "broker.submit_buy" not in targets
        assert "broker.submit_sell" not in targets
        assert "lock.enter" not in targets
        assert any("DRY RUN" in message for message in _messages(dry))

    def test_paper_lock_assigned_a_mock_is_the_one_verified(self, impl_name):
        """A `_paper_lock` that is a `MagicMock` has `verify()` called once per order submitted."""
        lock = MagicMock()
        scenario = SCENARIOS["plain_buys_and_sells"]

        def configure(trader: PaperTrader, _run: harness._Run, stack: ExitStack) -> None:
            real = OrderExecutor.submit_order

            def submit_with_the_mock(self: OrderExecutor, *args: Any, **kwargs: Any) -> str:
                trader._paper_lock = lock
                return real(self, *args, **kwargs)

            stack.enter_context(patch.object(OrderExecutor, "submit_order", submit_with_the_mock))

        driven = drive(impl_name, scenario, configure)
        orders = len(rs.submissions(_transcript_of(driven)))
        assert orders >= 2
        assert lock.verify.call_count == orders
        # The fake run lock the harness installed was not the one asked to verify.
        assert "lock.verify" not in _targets(driven)

    @pytest.mark.parametrize(
        ("name", "scenario_name"),
        [
            pytest.param("generate_core_signals", "plain_buys_and_sells", id="signals"),
            pytest.param("refresh_benchmark_tape", "plain_buys_and_sells", id="tape"),
            pytest.param("_today_et", "plain_buys_and_sells", id="today_et"),
            pytest.param("_now_pt", "plain_buys_and_sells", id="now_pt"),
            pytest.param("uuid4", "plain_buys_and_sells", id="uuid4"),
            pytest.param("find_split_applied_symbols", "split_correction", id="find_split_applied"),
            pytest.param("log_snapshot", "plain_buys_and_sells", id="persist_name"),
        ],
    )
    def test_a_string_patch_of_a_module_name_made_after_construction_wins(
        self, impl_name, name, scenario_name
    ):
        scenario = SCENARIOS[scenario_name]
        baseline = drive(impl_name, scenario)
        seen: list[tuple] = []

        def configure(_trader: PaperTrader, _run: harness._Run, stack: ExitStack) -> None:
            target = _trader._ledger if name == "log_snapshot" else trader_module
            real = getattr(target, name)

            def spy(*args: Any, **kwargs: Any) -> Any:
                seen.append((args, kwargs))
                return real(*args, **kwargs)

            if name == "log_snapshot":
                stack.enter_context(patch.object(target, name, spy))
            else:
                stack.enter_context(patch(f"{TRADER_MODULE}.{name}", spy))

        intercepted = drive(impl_name, scenario, configure)
        assert seen, f"the run never called the patched {name}"
        assert _same_run(baseline, intercepted)


def _transcript_of(driven: Driven) -> dict:
    """The run in the shape `rules_support` readers take (``{"calls": [...]}``)."""
    return {"calls": driven.run.recorder.calls}


def _targets(driven: Driven) -> list[str]:
    return [c["target"] for c in driven.run.recorder.calls]


def _messages(driven: Driven) -> list[str]:
    return rs.messages(_transcript_of(driven))


# =========================================================================== logging


def test_moved_code_logs_on_the_trader_logger():
    """The trader's logger has the literal name; every new module logger uses the constant."""
    assert trader_module._log.name == TRADER_LOGGER
    assert TRADER_LOGGER == "ggTrader.paper.trader"
    for path in _new_modules():
        loggers = [f for f in scan_new_module(_read(path), path.name) if f.rule == "logger-name"]
        assert not loggers, (path.name, loggers)


# =========================================================================== new-module rules


def test_new_modules_import_only_what_they_may():
    """Every discovered new module passes every scanner rule (none exists before Task 52)."""
    problems = {
        path.name: findings
        for path in _new_modules()
        if (findings := scan_new_module(_read(path), path.name))
    }
    assert not problems, problems


def test_the_set_of_discovered_modules_is_the_folder_minus_the_frozen_list():
    """Every module of the frozen list exists; any other file in the folder is discovered."""
    on_disk = {p.stem for p in PAPER_DIR.glob("*.py")}
    assert EXISTING_MODULES <= on_disk, sorted(EXISTING_MODULES - on_disk)
    assert {p.stem for p in _new_modules()} == on_disk - EXISTING_MODULES


CLEAN_SOURCE = '''\
"""A clean new module."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from ggTrader.paper import cash_sweep, catastrophe_stop
from ggTrader.paper.constants import TRADER_LOGGER_NAME
from ggTrader.paper.models import Position
from ggTrader.paper.ports import Notifier

_log = logging.getLogger(TRADER_LOGGER_NAME)


class Planner:
    """A plain class: no generator, no exception base, no clock."""

    def advance(self, cfg, hooks):
        today = hooks.today_et()
        sweep = cfg.reserve_pct()
        return [today, sweep, cash_sweep.compute_sweep_sell_for_funding, catastrophe_stop]
'''

# (rule, file name, source) per violation: the scanner must report the rule for the source.
VIOLATIONS: tuple[tuple[str, str, str], ...] = (
    ("forbidden-import", "x.py", "import psycopg2\n"),
    ("forbidden-import", "x.py", "from sqlalchemy import text\n"),
    ("forbidden-import", "x.py", "import requests\n"),
    ("forbidden-import", "x.py", "import httpx\n"),
    ("forbidden-import", "x.py", "import alpaca_trade_api\n"),
    ("forbidden-import", "x.py", "from alpaca.trading.client import TradingClient\n"),
    ("forbidden-import", "x.py", "from ggTrader.paper import persist\n"),
    ("forbidden-import", "x.py", "from ggTrader.paper.persist import log_trade\n"),
    ("forbidden-import", "x.py", "from ggTrader.paper.alpaca_broker import AlpacaBroker\n"),
    ("forbidden-import", "x.py", "from ggTrader.paper.notifier import TelegramNotifier\n"),
    (
        "forbidden-import",
        "x.py",
        "from ggTrader.paper.signal_runner import generate_core_signals\n",
    ),
    ("forbidden-import", "x.py", "from ggTrader.lab import something\n"),
    ("forbidden-import", "x.py", "from ggTrader.lab.data import fetch_stock_ohlcv\n"),
    ("forbidden-import", "x.py", "from ggTrader import lab\n"),
    ("forbidden-import", "x.py", "from . import persist\n"),
    ("forbidden-import", "x.py", "from .notifier import DryRunNotifier\n"),
    ("imports-trader", "x.py", "import ggTrader.paper.trader\n"),
    ("imports-trader", "x.py", "import ggTrader.paper.trader as t\n"),
    ("imports-trader", "x.py", "from ggTrader.paper import trader\n"),
    ("imports-trader", "x.py", "from ggTrader.paper import trader as t\n"),
    ("imports-trader", "x.py", "from ggTrader.paper.trader import PaperTrader\n"),
    ("imports-trader", "x.py", "from . import trader\n"),
    ("imports-trader", "x.py", "from .trader import PaperTrader\n"),
    (
        "imports-trader",
        "x.py",
        "def f():\n    from ggTrader.paper import trader\n    return trader\n",
    ),
    ("imports-trader", "planning.py", "def f():\n    from ggTrader.paper import trader\n"),
    ("imports-trader", "x.py", "import ggTrader\nx = ggTrader.paper.trader.PaperTrader\n"),
    (
        "dynamic-import",
        "x.py",
        "import importlib\nm = importlib.import_module('ggTrader.paper.trader')\n",
    ),
    ("dynamic-import", "x.py", "from importlib import import_module\n"),
    ("dynamic-import", "x.py", "m = __import__('ggTrader.paper.trader')\n"),
    ("time-or-uuid-import", "x.py", "import time\n"),
    ("time-or-uuid-import", "x.py", "import uuid\n"),
    ("time-or-uuid-import", "x.py", "from time import sleep\n"),
    ("time-or-uuid-import", "x.py", "from uuid import uuid4\n"),
    ("clock-call", "x.py", "from datetime import datetime\nx = datetime.now()\n"),
    ("clock-call", "x.py", "from datetime import date\nx = date.today()\n"),
    ("clock-call", "x.py", "import datetime\nx = datetime.datetime.now(None)\n"),
    ("environment-read", "x.py", "import os\nx = os.environ['A']\n"),
    ("environment-read", "x.py", "import os\nx = os.getenv('A')\n"),
    ("environment-read", "x.py", "from os import environ\n"),
    ("environment-read", "x.py", "from os import getenv\n"),
    (
        "settings-accessor",
        "x.py",
        "from ggTrader.paper import cash_sweep\nx = cash_sweep.sweep_enabled()\n",
    ),
    (
        "settings-accessor",
        "x.py",
        "from ggTrader.paper import cash_sweep\nx = cash_sweep.sweep_symbol()\n",
    ),
    (
        "settings-accessor",
        "x.py",
        "from ggTrader.paper import cash_sweep\nx = cash_sweep.reserve_pct()\n",
    ),
    (
        "settings-accessor",
        "x.py",
        "from ggTrader.paper import cash_sweep\nx = cash_sweep.min_clip_usd()\n",
    ),
    (
        "settings-accessor",
        "x.py",
        "from ggTrader.paper import cash_sweep\nx = cash_sweep.buy_trigger_pct()\n",
    ),
    (
        "settings-accessor",
        "x.py",
        "from ggTrader.paper import catastrophe_stop\n"
        "x = catastrophe_stop.catastrophe_stop_enabled()\n",
    ),
    (
        "settings-accessor",
        "x.py",
        "from ggTrader.paper import catastrophe_stop\nx = catastrophe_stop.catastrophe_stop_pct()",
    ),
    (
        "settings-accessor",
        "x.py",
        "from ggTrader.paper import untradable\nx = untradable.configured_untradable()\n",
    ),
    (
        "settings-accessor",
        "x.py",
        "from ggTrader.paper.untradable import configured_untradable\n",
    ),
    (
        "from-import-of-attribute-only-module",
        "x.py",
        "from ggTrader.paper.cash_sweep import compute_sweep_sell_for_funding\n",
    ),
    (
        "from-import-of-attribute-only-module",
        "x.py",
        "from ggTrader.paper.catastrophe_stop import find_catastrophe_stops\n",
    ),
    ("from-import-of-attribute-only-module", "x.py", "from .cash_sweep import x\n"),
    ("exception-class", "preflight.py", "class Halted(Exception):\n    pass\n"),
    ("exception-class", "preflight.py", "class RunHalted(BaseException):\n    pass\n"),
    ("exception-class", "preflight.py", "class Stop(ValueError):\n    pass\n"),
    ("generator", "planning.py", "def f():\n    yield 1\n"),
    ("generator", "planning.py", "def f():\n    yield from g()\n"),
    ("generator", "planning.py", "x = sum(i for i in range(3))\n"),
    ("generator", "order_flow.py", "x = list(i for i in range(3))\n"),
    ("generator", "preflight.py", "x = any(i for i in range(3))\n"),
    ("generator", "portfolio.py", "def f():\n    yield 1\n"),
    ("generator", "risk_assessor.py", "x = all(i for i in range(3))\n"),
    ("generator", "reporter.py", "def f():\n    yield 1\n"),
    ("generator", "reporter.py", "x = sum(i for i in range(3))\n"),
    ("logger-name", "x.py", "import logging\n_log = logging.getLogger(__name__)\n"),
    ("logger-name", "x.py", "import logging\n_log = logging.getLogger('ggTrader.paper.trader')\n"),
    ("logger-name", "x.py", "import logging\n_log = logging.getLogger()\n"),
    ("logger-name", "x.py", "from logging import getLogger\n_log = getLogger(__name__)\n"),
    ("logger-name", "x.py", "import logging as log\n_log = log.getLogger(__name__)\n"),
    ("imports-trader", "x.py", "from ggTrader import paper\nx = paper.trader.PaperTrader\n"),
    ("forbidden-import", "x.py", "import ggTrader.paper as gp\nx = gp.persist.log_trade\n"),
    ("logger-name", "x.py", "from logging import getLogger as get_log\nx = get_log(__name__)\n"),
    (
        "settings-accessor",
        "x.py",
        "from ggTrader.paper import cash_sweep as sweep\nx = sweep.reserve_pct()\n",
    ),
    ("settings-accessor", "x.py", "from . import untradable as u\nx = u.configured_untradable()\n"),
    (
        "settings-accessor",
        "x.py",
        "from ggTrader.paper import cash_sweep\nf = cash_sweep.reserve_pct\n",
    ),
)


def _rules(source: str, filename: str) -> set[str]:
    return {f.rule for f in scan_new_module(source, filename)}


def test_the_new_module_scanner_detects_each_violation():
    """Each rule fires on a synthetic source that breaks it, and the clean source passes."""
    assert scan_new_module(CLEAN_SOURCE, "planning.py") == []
    assert scan_new_module(CLEAN_SOURCE, "preflight.py") == []
    for rule, filename, source in VIOLATIONS:
        assert rule in _rules(source, filename), (rule, filename, source)
    assert {rule for rule, _f, _s in VIOLATIONS} == {
        "forbidden-import",
        "imports-trader",
        "dynamic-import",
        "time-or-uuid-import",
        "clock-call",
        "environment-read",
        "settings-accessor",
        "from-import-of-attribute-only-module",
        "exception-class",
        "generator",
        "logger-name",
    }


def test_the_scanner_names_the_line_and_keeps_function_local_imports():
    source = "x = 1\n\ndef f():\n    from ggTrader.paper import trader\n"
    findings = scan_new_module(source, "x.py")
    assert [(f.rule, f.line) for f in findings] == [("imports-trader", 4)]


def test_the_scanner_accepts_what_new_modules_may_do():
    """The allowed forms are not flagged: models, ports, constants, module attributes, `cfg`."""
    allowed = (
        "from ggTrader.paper.models import RunContext\n",
        "from ggTrader.paper.ports import Broker, Notifier\n",
        "from ggTrader.paper.gates import SignalGateError, late_run_reason\n",
        "from ggTrader.paper.config import TraderConfig\n",
        "from ggTrader.paper.risk import RiskGuard\n",
        "from ggTrader.paper import cash_sweep\nx = cash_sweep.compute_sweep_sell_for_funding\n",
        "from ggTrader.paper import catastrophe_stop\nx = catastrophe_stop.find_stops\n",
        "def f(cfg):\n    return cfg.reserve_pct(), cfg.min_clip_usd(), cfg.buy_trigger_pct()\n",
        "def f(h):\n    return h.now_pt(), h.today_et(), h.utcnow(), h.new_uuid()\n",
        "from ggTrader.paper import trader_helpers\n",  # another module: dotted match, not prefix
        "from ggTrader.paper.models import Position\nclass Skip(Position):\n    pass\n",
        "import logging\nfrom ggTrader.paper.constants import TRADER_LOGGER_NAME\n"
        "_log = logging.getLogger(TRADER_LOGGER_NAME)\n",
        "import logging\nfrom ggTrader.paper import constants\n"
        "_log = logging.getLogger(constants.TRADER_LOGGER_NAME)\n",
        "import logging as log\nfrom ggTrader.paper.constants import TRADER_LOGGER_NAME as NAME\n"
        "_log = log.getLogger(NAME)\n",
        "class Halted:\n    pass\n",
    )
    for source in allowed:
        assert scan_new_module(source, "preflight.py") == [], source
    # A hooks field named `utcnow` is not the clock method `now`.
    assert scan_new_module("def f(h):\n    return h.utcnow()\n", "x.py") == []


def test_the_scanner_allows_exactly_the_listed_generator_expressions_in_reporter():
    """`reporter.py` may hold each of the three R7 generator expressions once, and nothing else."""
    texts = [text for text, _line in GENEXP_ALLOWED]
    assert len(texts) == 3
    assert [line for _t, line in GENEXP_ALLOWED] == sorted(line for _t, line in GENEXP_ALLOWED)
    body = "\n".join(f"x{i} = {text}" for i, text in enumerate(texts))
    assert scan_new_module(body, "reporter.py") == []
    assert [f.rule for f in scan_new_module(body + f"\ny = {texts[0]}\n", "reporter.py")] == [
        "generator"
    ]
    assert [f.rule for f in scan_new_module("z = (a for a in b)\n", "reporter.py")] == ["generator"]
    # The same expressions are not allowed anywhere else the rule applies.
    assert _rules(body, "planning.py") == {"generator"}
    # Elsewhere (a module with no generator rule) they are not this scanner's business.
    assert scan_new_module(body, "execution.py") == []


# =========================================================================== one home


def test_moved_constants_and_helpers_have_one_home():
    """Before the move `trader.py` holds the names; after it each name has one home."""
    constants_src = _read(PAPER_DIR / "constants.py")
    helpers_path = PAPER_DIR / "run_helpers.py"
    helpers_src = _read(helpers_path) if helpers_path.exists() else None
    findings = home_findings(constants_src, helpers_src, _read(PAPER_DIR / "trader.py"))
    assert not findings, findings


def test_the_home_check_distinguishes_before_and_after_the_move():
    before_trader = (
        "_SPLIT_LOOKBACK_DAYS = 14\n_STALE_PENDING_DAYS = 5\n_MAX_SIGNAL_AGE_DAYS = 4\n"
        "def _as_date(value):\n    return value\n"
        "def _account_read_halt(notifier, exc):\n    return None\n"
    )
    plain_constants = "TERMINAL_STATUSES = ()\n"
    assert home_findings(plain_constants, None, before_trader) == []
    # Before the move: a lost name, a changed value and a half-made copy are each found.
    assert {
        f.rule
        for f in home_findings(
            plain_constants, None, before_trader.replace("_SPLIT_LOOKBACK_DAYS = 14\n", "")
        )
    } == {"name-lost"}
    assert {
        f.rule for f in home_findings(plain_constants, None, before_trader.replace("= 14", "= 15"))
    } == {"value-drift"}
    assert {
        f.rule
        for f in home_findings(plain_constants + "MAX_SIGNAL_AGE_DAYS = 4\n", None, before_trader)
    } == {"duplicated"}
    # After the move.
    constants = (
        plain_constants
        + "TRADER_LOGGER_NAME = 'x'\nMAX_SIGNAL_AGE_DAYS = 4\n"
        + "STALE_PENDING_DAYS = 5\nSPLIT_LOOKBACK_DAYS = 14\n"
    )
    helpers = (
        "def as_date(value):\n    return value\n"
        "def account_read_halt(notifier, exc):\n    return None\n"
    )
    after_trader = (
        "from ggTrader.paper import constants, run_helpers\n"
        "_MAX_SIGNAL_AGE_DAYS = constants.MAX_SIGNAL_AGE_DAYS\n"
        "_STALE_PENDING_DAYS = constants.STALE_PENDING_DAYS\n"
        "_SPLIT_LOOKBACK_DAYS = constants.SPLIT_LOOKBACK_DAYS\n"
        "_as_date = run_helpers.as_date\n_account_read_halt = run_helpers.account_read_halt\n"
    )
    assert home_findings(constants, helpers, after_trader) == []
    wrong_owner = after_trader.replace("run_helpers.as_date", "unrelated.as_date")
    assert {f.rule for f in home_findings(constants, helpers, wrong_owner)} == {"duplicated"}
    wrong_constant = after_trader.replace(
        "constants.MAX_SIGNAL_AGE_DAYS", "other.MAX_SIGNAL_AGE_DAYS"
    )
    assert {f.rule for f in home_findings(constants, helpers, wrong_constant)} == {"duplicated"}
    wrong_import = after_trader.replace(
        "from ggTrader.paper import constants, run_helpers",
        "from ggTrader.paper import constants\nfrom ggTrader.paper import ledger as run_helpers",
    )
    assert [(f.rule, f.detail) for f in home_findings(constants, helpers, wrong_import)] == [
        ("duplicated", "_as_date"),
        ("duplicated", "_account_read_halt"),
    ]
    direct = after_trader.replace(
        "_as_date = run_helpers.as_date",
        "from ggTrader.paper.run_helpers import as_date as _as_date",
    )
    assert home_findings(constants, helpers, direct) == []
    plain = after_trader.replace(
        "_as_date = run_helpers.as_date",
        "from ggTrader.paper.run_helpers import as_date\n_as_date = as_date",
    )
    assert home_findings(constants, helpers, plain) == []
    wrong_plain = plain.replace(
        "from ggTrader.paper.run_helpers import as_date", "def as_date(x): return x"
    )
    assert [(f.rule, f.detail) for f in home_findings(constants, helpers, wrong_plain)] == [
        ("duplicated", "_as_date")
    ]
    typed = after_trader.replace(
        "_MAX_SIGNAL_AGE_DAYS = constants.MAX_SIGNAL_AGE_DAYS",
        "_MAX_SIGNAL_AGE_DAYS: int = constants.MAX_SIGNAL_AGE_DAYS",
    )
    assert home_findings(constants, helpers, typed) == []
    # Created the helpers module but left the literals and defs in the trader: duplicates.
    assert {f.rule for f in home_findings(constants, helpers, before_trader)} == {"duplicated"}
    # Created the helpers module but not the constants.
    assert {f.rule for f in home_findings(plain_constants, helpers, after_trader)} == {
        "missing-home"
    }
    # Helper missing from the new module, or an alias dropped from the trader.
    assert {f.rule for f in home_findings(constants, "x = 1\n", after_trader)} == {"missing-home"}
    assert {
        f.rule
        for f in home_findings(
            constants, helpers, after_trader.replace("_as_date = run_helpers.as_date\n", "")
        )
    } == {"name-lost"}
    # A changed value in the new home.
    assert {
        f.rule for f in home_findings(constants.replace("= 14", "= 15"), helpers, after_trader)
    } == {"value-drift"}


def test_no_test_patches_an_alias_of_a_moved_name():
    """A patch of an alias of a moved name would not reach the new modules; none exists."""
    patched = {
        str(path.relative_to(REPO)): findings
        for path in sorted(TESTS_DIR.rglob("*.py"))
        if (findings := alias_patches(_read(path)))
    }
    assert not patched, patched


def test_the_alias_patch_scanner_finds_each_form():
    dotted = "ggTrader.paper.trader"
    forms = (
        f'patch("{dotted}._as_date")\n',
        f'patch("{dotted}._account_read_halt", new=1)\n',
        f'x = "{dotted}._MAX_SIGNAL_AGE_DAYS"\n',
        "from ggTrader.paper import trader as tm\ntm._STALE_PENDING_DAYS = 1\n",
        "from ggTrader.paper import trader as tm\nfrom unittest.mock import patch\n"
        'patch.object(tm, "_SPLIT_LOOKBACK_DAYS", 3)\n',
        "from ggTrader.paper import trader\ntrader._as_date = None\n",
    )
    for source in forms:
        assert alias_patches(source), source
    clean = (
        f'patch("{dotted}.generate_core_signals")\n',
        "from ggTrader.paper import trader as tm\ntm.generate_core_signals = None\n",
        "x = _as_date(1)\n",
    )
    for source in clean:
        assert alias_patches(source) == [], source


# ========================================================================= environment snapshot

#: ``(scenario, variable, value)``: a setting a run reads once, in a scenario where changing it from
#: the start visibly changes the run (the control), so an ignored late change proves something.
SNAPSHOT_CASES = (
    pytest.param("sweep_buy", "CASH_SWEEP_ENABLED", "false", id="sweep_enabled"),
    pytest.param("sweep_buy", "SWEEP_SYMBOL", "IVV", id="sweep_symbol"),
    pytest.param("sweep_buy", "SWEEP_CASH_RESERVE_PCT", "0.9", id="reserve"),
    pytest.param("sweep_buy", "SWEEP_MIN_CLIP_USD", "1000000", id="min_clip"),
    pytest.param("sweep_buy", "SWEEP_BUY_TRIGGER_PCT", "0.99", id="buy_trigger"),
    pytest.param("sweep_funding_sell", "SWEEP_CASH_RESERVE_PCT", "0.9", id="funding_reserve"),
    pytest.param("plain_buys_and_sells", "UNTRADABLE_SYMBOLS", "NVDA", id="untradable"),
    pytest.param("catastrophe_stop", "CATASTROPHE_STOP_PCT", "-0.9", id="stop_pct"),
    pytest.param("catastrophe_stop", "CATASTROPHE_STOP_ENABLED", "false", id="stop_enabled"),
)


def _with_env(scenario: Scenario, changes: dict[str, str]) -> Scenario:
    return replace(copy.deepcopy(scenario), env={**scenario.env, **changes})


def _mutate_after_first_account_read(changes: dict[str, str]) -> Callable[..., None]:
    """A `configure` for `drive`: the first `get_account` changes the environment, then answers."""

    def configure(_trader: PaperTrader, run: harness._Run, _stack: ExitStack) -> None:
        real = run.broker.get_account
        state = {"changed": False}

        def get_account() -> dict:
            if not state["changed"]:
                state["changed"] = True
                os.environ.update(changes)
            return real()

        run.broker.get_account = get_account  # type: ignore[method-assign]

    return configure


@pytest.mark.parametrize("impl_name", IMPLS)
@pytest.mark.parametrize(("scenario_name", "variable", "value"), SNAPSHOT_CASES)
def test_env_changed_after_the_first_broker_call_is_ignored_by_every_stage(
    impl_name, scenario_name, variable, value
):
    """A setting changed once the run is under way changes nothing: the first snapshot is used.

    The control sets the same value before the run and must differ from the baseline (otherwise
    the setting is inert in that scenario and the comparison would prove nothing).
    """
    scenario = SCENARIOS[scenario_name]
    baseline = drive(impl_name, scenario)
    control = drive(impl_name, _with_env(scenario, {variable: value}))
    assert not _same_run(baseline, control), (
        f"{variable}={value} does not matter in {scenario_name}"
    )
    late = drive(impl_name, scenario, _mutate_after_first_account_read({variable: value}))
    assert _same_run(baseline, late)


@pytest.mark.parametrize("impl_name", IMPLS)
@pytest.mark.parametrize(
    "scenario_name",
    ["sweep_buy", "sweep_funding_sell", "sweep_sell_and_strategy_sell", "catastrophe_stop"],
)
def test_all_seven_settings_changed_together_after_the_first_broker_call_are_ignored(
    impl_name, scenario_name
):
    changes = {
        "SWEEP_SYMBOL": "IVV",
        "UNTRADABLE_SYMBOLS": "AAPL,MSFT",
        "CATASTROPHE_STOP_PCT": "-0.01",
        "SWEEP_MIN_CLIP_USD": "1000000",
        "SWEEP_BUY_TRIGGER_PCT": "0.99",
        "SWEEP_CASH_RESERVE_PCT": "0.9",
        "CASH_SWEEP_ENABLED": "false",
    }
    scenario = SCENARIOS[scenario_name]
    baseline = drive(impl_name, scenario)
    assert not _same_run(baseline, drive(impl_name, _with_env(scenario, changes)))
    late = drive(impl_name, scenario, _mutate_after_first_account_read(changes))
    assert _same_run(baseline, late)


# =========================================================================== production boundary


def _production_sources() -> list[Path]:
    paths = sorted((REPO / "src" / "ggTrader").rglob("*.py"))
    paths += sorted((REPO / "scripts").glob("*.py"))
    paths.append(REPO / "ggt.py")
    return paths


def test_in_memory_ledger_is_never_reachable_from_production_code():
    """`InMemoryLedger` appears only in `ledger.py`, and no production call passes ``ledger=``.

    Static only: a runtime refusal would change behavior. The scan covers `src/ggTrader`, the
    scripts and `ggt.py`; the entry points (`run_paper_trading`, the CLI) are among them.
    """
    paths = _production_sources()
    assert REPO / LEDGER_HOME in paths
    assert REPO / "src/ggTrader/cli/main.py" in paths
    assert REPO / "ggt.py" in paths
    outside = {}
    for path in paths:
        if path == REPO / LEDGER_HOME:
            continue
        findings = in_memory_ledger_references(_read(path))
        if findings:
            outside[str(path.relative_to(REPO))] = findings
    assert not outside, outside
    # The definition is where it is supposed to be, so the scan is looking for something real.
    assert IN_MEMORY_LEDGER in _read(REPO / LEDGER_HOME)
    assert hasattr(ledger_module, IN_MEMORY_LEDGER)


def test_the_in_memory_ledger_scanner_finds_each_form():
    forms = (
        "from ggTrader.paper.ledger import InMemoryLedger\n",
        "from ggTrader.paper.ledger import InMemoryLedger as L\n",
        "from ggTrader.paper import ledger\nx = ledger.InMemoryLedger()\n",
        "x = InMemoryLedger()\n",
        "names = ['InMemoryLedger']\n",
        "run_paper_trading(dry_run=True, ledger=make())\n",
        "cli.run_paper_trading(ledger=make())\n",
        "PaperTrader(broker, notifier, ledger=object())\n",
    )
    for source in forms:
        assert in_memory_ledger_references(source), source
    clean = (
        "x = 1\n",
        "def run(ledger=None):\n    return ledger\n",
        "ledger = make()\n",
        "build_proposal(ledger=make())\n",  # an unrelated `ledger` argument
    )
    for source in clean:
        assert in_memory_ledger_references(source) == [], source
