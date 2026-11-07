"""Registry of the strategies on the engine contract, built by scanning the package.

This is the only strategy registry (the legacy ``STRATEGY_REGISTRY`` was deleted in P5c). It is
filled by :func:`discover`, which the package ``__init__`` runs on import: every module in
``ggTrader/lab/strategies/`` that is not a helper (:data:`HELPER_MODULES`, or a name starting
with ``_``) must define at least one class that implements the strategy contract
(``name``, ``kind``, ``grid`` and ``build``; see ``contract.SignalStrategy`` and
``WeightStrategy``). Dropping a new module into the package is all it takes for
``ggt lab --strategy NAME`` to find it. A module that is malformed raises
:class:`StrategyModuleError` naming the file.

Names equal the legacy registry names; the 28 legacy names that were not ported (OD11(b)) are
listed in ``LEGACY_ONLY_STRATEGIES`` so the CLI can say where they still run.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.util
import inspect
import pkgutil
from pathlib import Path
from types import ModuleType
from typing import Any, TypeVar

REGISTRY: dict[str, type] = {}

#: Modules of the package that are infrastructure, not strategies (not scanned for classes).
#: Any other module whose name starts with ``_`` is a helper too.
HELPER_MODULES: frozenset[str] = frozenset({"contract", "leak", "registry", "vote", "voters"})

#: The values a strategy's ``kind`` may take.
KINDS: tuple[str, ...] = ("signals", "weights")

LEGACY_TAG = "legacy-engine-vbt0285"

#: Legacy registry names that were not ported to the engine contract (OD11(b), P4-completion).
#: They are reproducible at ``LEGACY_TAG`` with ``scripts/legacy_env.sh``; their code and tests
#: were removed from the main tree in P5c. The research reports are kept.
LEGACY_ONLY_STRATEGIES: tuple[str, ...] = (
    "commodity_trend",
    "congress_trades",
    "dual_momentum",
    "ensemble_conviction",
    "ensemble_ic",
    "ensemble_kelly",
    "fomc_drift",
    "fx_hedge_overlay",
    "headline_sentiment",
    "idio_vol",
    "index_deletion_fade",
    "insider_cluster_buy",
    "leveraged_rotation_nasdaq100",
    "leveraged_rotation_russell2000",
    "leveraged_rotation_sp500",
    "leveraged_trend_nasdaq100",
    "leveraged_trend_russell2000",
    "leveraged_trend_sp500",
    "max_effect",
    "month_end_treasury",
    "mtf_reversion",
    "overnight_gap",
    "pairs_stat_arb",
    "pead",
    "short_interest",
    "short_volume_ratio",
    "treasury_curve",
    "wfo_tournament",
)

T = TypeVar("T", bound=type)


class StrategyModuleError(RuntimeError):
    """A module in the strategies package does not implement the strategy contract."""


def contract_problems(cls: type) -> list[str]:
    """What a class lacks to implement the strategy contract (empty when it does).

    The contract is the class attributes ``name`` (str), ``kind`` ("signals" or "weights") and
    ``grid`` (``ParamGrid``) plus a callable ``build`` that takes ``(bars, grid)`` for signal
    strategies and ``(bars, grid, rebal, eligible)`` for weight strategies.
    """
    from ggTrader.lab.strategies.contract import ParamGrid

    problems: list[str] = []
    name = getattr(cls, "name", None)
    if not isinstance(name, str) or not name:
        problems.append("`name` must be a non-empty string class attribute")
    kind = getattr(cls, "kind", None)
    if kind not in KINDS:
        problems.append(f"`kind` must be one of {KINDS}, got {kind!r}")
    if not isinstance(getattr(cls, "grid", None), ParamGrid):
        problems.append("`grid` must be a ParamGrid class attribute")
    build = getattr(cls, "build", None)
    if not callable(build):
        problems.append("`build` must be a method")
    elif kind in KINDS:
        n_args = len([p for p in inspect.signature(build).parameters if p != "self"])
        need = 2 if kind == "signals" else 3
        if n_args < need:
            problems.append(
                f"`build` of a {kind} strategy takes at least {need} arguments, has {n_args}"
            )
    return problems


def _looks_like_strategy(cls: type) -> bool:
    """A public class that has a ``build`` method or a ``kind``: it is meant to be a strategy."""
    return not cls.__name__.startswith("_") and (
        callable(getattr(cls, "build", None)) or hasattr(cls, "kind")
    )


def _strategy_modules(package: ModuleType) -> list[str]:
    """Names of the strategy modules in ``package``, sorted."""
    return sorted(
        info.name
        for info in pkgutil.iter_modules(package.__path__)
        if not info.ispkg and info.name not in HELPER_MODULES and not info.name.startswith("_")
    )


def _module_classes(module: ModuleType) -> list[type]:
    """Strategy classes defined in ``module`` (not imported into it), in definition order."""
    return [
        obj
        for obj in vars(module).values()
        if inspect.isclass(obj) and obj.__module__ == module.__name__ and _looks_like_strategy(obj)
    ]


def discover(package: ModuleType | None = None) -> dict[str, type]:
    """Scan the strategies package and rebuild :data:`REGISTRY`.

    Order is deterministic: modules sorted by name, classes in definition order within a
    module; classes added with :func:`register` (outside the package) follow, in the order
    they were registered.

    Raises:
        StrategyModuleError: a module cannot be imported, defines no strategy class, defines a
            class that breaks the contract, or reuses a name another module already took; the
            message names the file.
    """
    if package is None:
        package = importlib.import_module("ggTrader.lab.strategies")
    outside = {n: c for n, c in REGISTRY.items() if not c.__module__.startswith(package.__name__)}
    found: dict[str, type] = {}
    owner: dict[str, str] = {}
    for mod_name in _strategy_modules(package):
        spec = importlib.util.find_spec(f"{package.__name__}.{mod_name}")
        path = Path(spec.origin) if spec and spec.origin else Path(f"{mod_name}.py")
        try:
            module = importlib.import_module(f"{package.__name__}.{mod_name}")
        except Exception as exc:
            raise StrategyModuleError(f"{path}: cannot import the module: {exc!r}") from exc
        classes = _module_classes(module)
        if not classes:
            raise StrategyModuleError(
                f"{path}: defines no class that implements the strategy contract "
                "(needs `name`, `kind`, `grid` and `build`). If this module is a helper, "
                "start its file name with an underscore"
            )
        for cls in classes:
            problems = contract_problems(cls)
            if problems:
                raise StrategyModuleError(f"{path}: class {cls.__name__}: " + "; ".join(problems))
            if cls.name in found:  # type: ignore[attr-defined]
                raise StrategyModuleError(
                    f"{path}: strategy name {cls.name!r} is already defined in "  # type: ignore[attr-defined]
                    f"{owner[cls.name]}"  # type: ignore[attr-defined]
                )
            found[cls.name] = cls  # type: ignore[attr-defined]
            owner[cls.name] = str(path)  # type: ignore[attr-defined]
    REGISTRY.clear()
    REGISTRY.update(found)
    for name, cls in outside.items():
        if name in REGISTRY and REGISTRY[name] is not cls:
            raise StrategyModuleError(f"registered strategy {name!r} clashes with {owner[name]}")
        REGISTRY[name] = cls
    return REGISTRY


def register(cls: T) -> T:
    """Class decorator for a strategy defined outside the package (scratch work, a study).

    Strategies inside ``ggTrader/lab/strategies/`` do not need it: :func:`discover` finds them.

    Raises:
        StrategyModuleError: the class breaks the contract.
        ValueError: the name is taken by another class.
    """
    problems = contract_problems(cls)
    if problems:
        raise StrategyModuleError(f"class {cls.__name__}: " + "; ".join(problems))
    name = cls.name  # type: ignore[attr-defined]
    if name in REGISTRY and REGISTRY[name] is not cls:
        raise ValueError(f"strategy {name!r} is already registered")
    REGISTRY[name] = cls
    return cls


def get(name: str) -> Any:
    """A fresh instance of a registered strategy.

    Raises:
        ValueError: the name is not registered; the message names the legacy tag that still
            runs the unported ones.
    """
    if name not in REGISTRY:
        raise ValueError(
            f"strategy {name!r} is not on the engine contract; legacy-only strategies stay "
            f"runnable at git tag {LEGACY_TAG} (scripts/legacy_env.sh) (registered: "
            f"{sorted(REGISTRY)})"
        )
    return REGISTRY[name]()


def signal_names() -> tuple[str, ...]:
    """Registered strategies that produce entry/exit signals."""
    return tuple(n for n, c in REGISTRY.items() if c.kind == "signals")  # type: ignore[attr-defined]


def weight_names() -> tuple[str, ...]:
    """Registered strategies that produce target weights."""
    return tuple(n for n, c in REGISTRY.items() if c.kind == "weights")  # type: ignore[attr-defined]


def _package_imports(path: Path, package: str, package_dir: Path) -> list[Path]:
    """Files of the sibling modules in ``package`` that the module at ``path`` imports."""
    try:
        tree = ast.parse(path.read_text())
    except (OSError, SyntaxError):
        return []
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module == package:
                names |= {alias.name for alias in node.names}
            elif node.module.startswith(package + "."):
                names.add(node.module[len(package) + 1 :].split(".")[0])
        elif isinstance(node, ast.Import):
            names |= {
                alias.name[len(package) + 1 :].split(".")[0]
                for alias in node.names
                if alias.name.startswith(package + ".")
            }
    return [
        package_dir / f"{name}.py"
        for name in sorted(names)
        if (package_dir / f"{name}.py").is_file()
    ]


def source_files(cls: type) -> list[Path]:
    """Source files a strategy is made of: its module, its base classes' modules and the
    sibling modules of ``ggTrader.lab.strategies`` they import (``voters``, ``vote``,
    ``contract``...), followed transitively. ``registry.py`` itself is not part of a strategy."""
    package = __name__.rsplit(".", 1)[0]
    package_dir = Path(__file__).resolve().parent
    files: dict[Path, None] = {}
    queue: list[Path] = []
    for klass in cls.__mro__:
        if klass is object:
            continue
        path = getattr(inspect.getmodule(klass), "__file__", None)
        if path and not path.endswith("__init__.py"):
            queue.append(Path(path).resolve())
    while queue:
        path = queue.pop(0)
        if path in files or path == Path(__file__).resolve():
            continue
        files[path] = None
        queue.extend(_package_imports(path, package, package_dir))
    return sorted(files)


def source_sha256(cls: type) -> dict[str, Any]:
    """Hash of the strategy's source file(s), for the provenance block of a result.

    Returns:
        ``{"strategy_sha256": <hash of the sorted per-file hashes>, "strategy_files":
        {<file name>: <sha256 of the file>}}``; a file that cannot be read is skipped.
    """
    per_file: dict[str, str] = {}
    for path in source_files(cls):
        try:
            per_file[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            continue
    combined = hashlib.sha256("".join(f"{k}:{v}\n" for k, v in sorted(per_file.items())).encode())
    return {"strategy_sha256": combined.hexdigest(), "strategy_files": per_file}
