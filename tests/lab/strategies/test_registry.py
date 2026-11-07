"""Registry built by discovery: properties of every registered strategy, errors, order."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from ggTrader.lab.strategies import REGISTRY, get, signal_names, weight_names
from ggTrader.lab.strategies.contract import ParamGrid
from ggTrader.lab.strategies.registry import (
    LEGACY_ONLY_STRATEGIES,
    LEGACY_TAG,
    StrategyModuleError,
    contract_problems,
    discover,
    source_sha256,
)

#: The names ported at P4 (OD11); the registry may hold more, never fewer.
PORTED = {
    "ensemble",
    "bb_reversion",
    "rsi_reversion",
    "ema_cross",
    "macd_divergence",
    "volume_bb_reversion",
    "conviction_bb",
    "cross_asset_sleeve",
    "retail_attention",
    "xs_momentum",
}
#: Grid sizes of the ported ten (unchanged by the move to discovery).
GRID_SIZES = {
    "conviction_bb": 36,
    "cross_asset_sleeve": 3,
    "ensemble": 48,
    "retail_attention": 6,
    "ema_cross": 16,
    "bb_reversion": 16,
    "rsi_reversion": 27,
    "macd_divergence": 8,
    "volume_bb_reversion": 12,
    "xs_momentum": 12,
}


def test_the_ported_ten_are_registered() -> None:
    assert PORTED <= set(REGISTRY)


def test_every_registered_strategy_has_name_kind_grid_and_build() -> None:
    assert REGISTRY
    for name, cls in REGISTRY.items():
        assert contract_problems(cls) == [], name
        assert cls.name == name
        assert cls.kind in ("signals", "weights")
        assert isinstance(cls.grid, ParamGrid) and len(cls.grid) >= 1, name
        assert callable(cls.build)
        assert get(name).name == name


def test_kinds_match_the_legacy_target_kind() -> None:
    """The legacy registry's ``target_kind`` for the ported ten (frozen when it was deleted)."""
    weights = {"cross_asset_sleeve", "retail_attention", "xs_momentum"}
    for name in PORTED:
        assert REGISTRY[name].kind == ("weights" if name in weights else "signals"), name
    assert weights <= set(weight_names())
    assert PORTED - weights <= set(signal_names())
    assert set(signal_names()) | set(weight_names()) == set(REGISTRY)
    assert not set(signal_names()) & set(weight_names())


def test_ported_grid_sizes_are_unchanged() -> None:
    for name, size in GRID_SIZES.items():
        assert len(REGISTRY[name].grid) == size, name


def test_discovery_order_is_deterministic() -> None:
    first = list(REGISTRY)
    assert list(discover()) == first
    assert first.index("conviction_bb") < first.index("xs_momentum")  # module-name order


def test_the_38_legacy_names_are_the_ported_ten_plus_the_28_legacy_only() -> None:
    assert len(LEGACY_ONLY_STRATEGIES) == 28
    assert len(set(LEGACY_ONLY_STRATEGIES)) == 28
    assert not set(LEGACY_ONLY_STRATEGIES) & set(REGISTRY)
    assert len(PORTED) + len(LEGACY_ONLY_STRATEGIES) == 38


def test_get_returns_instances_and_unported_names_name_the_legacy_tag() -> None:
    assert get("ensemble").name == "ensemble"
    for name in LEGACY_ONLY_STRATEGIES:
        with pytest.raises(ValueError, match=LEGACY_TAG):
            get(name)


# ---- discovery of a new module, and the errors for malformed ones -----------------------------

GOOD = textwrap.dedent(
    '''
    """A scratch strategy."""
    from typing import Any, ClassVar

    from ggTrader.lab.strategies.contract import ParamGrid


    class Scratch:
        name: ClassVar[str] = "scratch_discovered"
        kind: ClassVar[str] = "signals"
        grid: ClassVar[ParamGrid] = ParamGrid.from_sweep({"n": [1, 2]})
        live: ClassVar[dict[str, Any] | None] = None

        def build(self, bars, grid=None):
            raise NotImplementedError
    '''
)


@pytest.fixture
def scratch_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The real package extended with a scratch directory; registry restored afterwards."""
    import sys

    import ggTrader.lab.strategies as pkg

    saved = dict(REGISTRY)
    before = set(sys.modules)
    monkeypatch.setattr(pkg, "__path__", [*pkg.__path__, str(tmp_path)])
    yield tmp_path
    REGISTRY.clear()
    REGISTRY.update(saved)
    for name in set(sys.modules) - before:
        if name.startswith("ggTrader.lab.strategies.scratch_"):
            del sys.modules[name]


def _write(directory: Path, name: str, source: str) -> Path:
    path = directory / f"{name}.py"
    path.write_text(source)
    return path


def test_a_new_module_is_found_without_editing_the_package(scratch_package: Path) -> None:
    _write(scratch_package, "scratch_good", GOOD)
    discover()
    assert "scratch_discovered" in REGISTRY
    assert get("scratch_discovered").kind == "signals"
    assert list(REGISTRY).index("scratch_discovered") > list(REGISTRY).index("ensemble")


def test_a_module_with_no_strategy_class_is_an_error_naming_the_file(
    scratch_package: Path,
) -> None:
    path = _write(scratch_package, "scratch_empty", '"""Nothing here."""\nX = 1\n')
    with pytest.raises(StrategyModuleError, match=r"scratch_empty\.py") as err:
        discover()
    assert str(path) in str(err.value)


def test_a_class_missing_the_grid_is_an_error_naming_the_file(scratch_package: Path) -> None:
    bad = GOOD.replace('grid: ClassVar[ParamGrid] = ParamGrid.from_sweep({"n": [1, 2]})', "")
    _write(scratch_package, "scratch_nogrid", bad)
    with pytest.raises(StrategyModuleError, match=r"scratch_nogrid\.py.*`grid`"):
        discover()


def test_a_module_that_does_not_import_is_an_error_naming_the_file(scratch_package: Path) -> None:
    _write(scratch_package, "scratch_broken", "import this_module_does_not_exist_xyz\n")
    with pytest.raises(StrategyModuleError, match=r"scratch_broken\.py.*cannot import"):
        discover()


def test_a_duplicate_name_is_an_error_naming_both_files(scratch_package: Path) -> None:
    _write(scratch_package, "scratch_one", GOOD)
    _write(scratch_package, "scratch_two", GOOD)
    with pytest.raises(StrategyModuleError, match=r"scratch_two\.py.*scratch_one\.py"):
        discover()


def test_an_underscore_module_is_a_helper_and_is_not_scanned(scratch_package: Path) -> None:
    _write(scratch_package, "_helper", '"""Shared code."""\nX = 1\n')
    discover()


def test_weight_build_needs_the_rebalance_argument() -> None:
    class Bad:
        name = "x"
        kind = "weights"
        grid = ParamGrid.from_sweep({"a": [1]})

        def build(self, bars, grid):
            return None

    assert any("at least 3" in p for p in contract_problems(Bad))


def test_source_hash_names_the_files_and_differs_between_strategies() -> None:
    info = source_sha256(REGISTRY["ensemble"])
    assert len(info["strategy_sha256"]) == 64
    assert {"ensemble.py", "voters.py", "vote.py", "contract.py"} <= set(info["strategy_files"])
    assert source_sha256(REGISTRY["ensemble"]) == info
    assert source_sha256(REGISTRY["xs_momentum"])["strategy_sha256"] != info["strategy_sha256"]
