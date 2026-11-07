"""conviction_bb on the engine contract: signals equal bb_reversion's, sizes equal the legacy."""

from __future__ import annotations

import numpy as np
import pytest

from ggTrader.lab.strategies.conviction_bb import ConvictionBb, conviction_sizes
from ggTrader.lab.wfo.inputs import combo_name
from tests.lab.strategies.helpers import (
    diff_after_burn_in,
    frozen_cells,
    frozen_sizes,
    legacy_grids,
    make_bars,
    settled,
)

pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture(scope="module")
def bars():
    return make_bars(seed=51, n_bars=800, n_syms=8, holes=True)


def test_grid_equals_legacy() -> None:
    assert ConvictionBb.grid.combos() == legacy_grids()["conviction_bb"]["grid"]
    assert len(ConvictionBb.grid) == 36


def test_every_combination_matches_frozen_legacy_signals_and_sizes(bars) -> None:
    out = ConvictionBb().build(bars)
    n = len(bars.symbols)
    keys = list(ConvictionBb.grid.values)
    ok = settled(bars.close)
    worst = 0.0
    for i, combo in enumerate(out.combos):
        key = f"conviction_bb|{combo_name('conviction_bb', {k: combo[k] for k in keys})}"
        old_entries, old_exits = frozen_cells(key, "entries"), frozen_cells(key, "exits")
        sl = slice(i * n, (i + 1) * n)
        assert diff_after_burn_in(out.entries.to_numpy()[:, sl], old_entries, bars.close) == 0
        assert diff_after_burn_in(out.exits.to_numpy()[:, sl], old_exits, bars.close) == 0
        new_size = out.size.to_numpy()[:, sl]
        old_size = frozen_sizes(key)
        both = ok & out.entries.to_numpy()[:, sl] & old_entries
        assert not np.isnan(new_size[both]).any()
        worst = max(worst, float(np.nanmax(np.abs(new_size[both] - old_size[both]), initial=0.0)))
        assert np.isnan(new_size[~out.entries.to_numpy()[:, sl]]).all()  # NaN = base size
    assert worst < 1e-12


def test_size_interpolates_between_min_and_max_by_depth() -> None:
    close = np.array([[100.0], [90.0], [80.0], [100.0]])
    lower = np.full_like(close, 95.0)
    middle = np.full_like(close, 105.0)  # half-width 10
    entries = np.array([[True], [True], [True], [False]])
    sizes = conviction_sizes(close, lower, middle, entries, 0.01, 0.05)
    assert sizes[0, 0] == pytest.approx(0.01)  # above the band: depth 0 clips to min
    assert sizes[1, 0] == pytest.approx(0.01 + 0.5 * 0.04)  # 5 below / 10
    assert sizes[2, 0] == pytest.approx(0.05)  # 15 below clips to depth 1
    assert np.isnan(sizes[3, 0])


def test_zero_band_width_gives_nan_not_inf() -> None:
    close = np.array([[1.0]])
    sizes = conviction_sizes(close, close, close, np.array([[True]]), 0.01, 0.05)
    assert np.isnan(sizes[0, 0])
