"""The five voter strategies against the frozen legacy output: grids, then every combination."""

from __future__ import annotations

import pytest

from ggTrader.lab.strategies.voter_strategies import (
    BbReversion,
    EmaCross,
    MacdDivergence,
    RsiReversion,
    VolumeBbReversion,
)
from ggTrader.lab.wfo.inputs import combo_name
from tests.lab.strategies.helpers import (
    diff_after_burn_in,
    frozen_cells,
    legacy_grids,
    make_bars,
)

PAIRS = [
    (EmaCross, 16),
    (BbReversion, 16),
    (RsiReversion, 27),
    (MacdDivergence, 8),
    (VolumeBbReversion, 12),
]


@pytest.fixture(scope="module")
def bars():
    return make_bars(seed=33, n_bars=800, n_syms=8, holes=True)


@pytest.mark.parametrize(("new_cls", "size"), PAIRS)
def test_grid_equals_the_frozen_legacy_build_grid(new_cls, size) -> None:
    assert new_cls.grid.combos() == legacy_grids()[new_cls.name]["grid"]
    assert len(new_cls.grid) == size


@pytest.mark.parametrize(("new_cls", "size"), PAIRS)
def test_every_combination_equals_the_frozen_legacy_after_burn_in(bars, new_cls, size) -> None:
    out = new_cls().build(bars)
    n = len(bars.symbols)
    keys = list(new_cls.grid.values)
    assert len(out.combos) == size
    for i, combo in enumerate(out.combos):
        key = f"{new_cls.name}|{combo_name(new_cls.name, {k: combo[k] for k in keys})}"
        sl = slice(i * n, (i + 1) * n)
        entries = diff_after_burn_in(
            out.entries.to_numpy()[:, sl], frozen_cells(key, "entries"), bars.close
        )
        exits = diff_after_burn_in(
            out.exits.to_numpy()[:, sl], frozen_cells(key, "exits"), bars.close
        )
        assert entries == 0 and exits == 0


@pytest.mark.parametrize(("new_cls", "size"), PAIRS)
def test_defaults_equal_the_frozen_legacy_constructor(bars, new_cls, size) -> None:
    from ggTrader.lab.strategies.contract import ParamGrid

    legacy = legacy_grids()[new_cls.name]["constructor_attributes"]
    for key, value in new_cls.defaults.items():
        assert legacy[key] == value
    out = new_cls().build(bars, ParamGrid.single(new_cls.defaults))
    # the legacy to_targets masks entries to the last bar only; compare exits, which are unmasked
    old_exits = frozen_cells(f"{new_cls.name}_defaults", "exits")
    assert diff_after_burn_in(out.exits.to_numpy(), old_exits, bars.close) == 0


def test_validity_rules_remove_fast_not_below_slow() -> None:
    assert all(c["ema_fast"] < c["ema_slow"] for c in EmaCross.grid.combos())
    assert all(c["macd_fast"] < c["macd_slow"] for c in MacdDivergence.grid.combos())
