"""The frozen golden inputs exist, are used by the factories, and still match the generators.

If a factory's default changes (say ``start``), its key stops matching, the factory silently falls
back to live generation and the CPU-dependent bit-exact failures return. These tests make that
loud. Equality with the live generator is checked to 1e-12, which any CPU meets; the frozen files
exist precisely because it cannot be checked bit for bit everywhere.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.engine import panels
from tests.frozen_inputs import load_frozen
from tests.lab.strategies import helpers


@pytest.mark.parametrize("spec", panels.FROZEN_PANELS, ids=str)
def test_frozen_panel_is_served_and_matches_the_generator(spec) -> None:
    seed, n_bars, n_syms, holes = spec
    key = panels.panel_key(seed, n_bars, n_syms, holes, panels.START)
    frozen = load_frozen(panels.PANELS_FILE, key, panels.PANEL_ARRAYS)
    assert frozen is not None, f"{key} is missing from {panels.PANELS_FILE.name}"
    served = panels.make_panel(seed=seed, n_bars=n_bars, n_syms=n_syms, holes=holes)
    live = panels.generate_panel_arrays(seed, n_bars, n_syms, holes=holes)
    for name in panels.PANEL_ARRAYS:
        assert np.array_equal(served[name].to_numpy(), frozen[name], equal_nan=True)
        np.testing.assert_allclose(frozen[name], live[name], rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("spec", helpers.FROZEN_BARS, ids=str)
def test_frozen_bars_are_served_and_match_the_generator(spec) -> None:
    seed, n_bars, n_syms, holes, symbols = spec
    key = helpers.bars_key(seed, n_bars, n_syms, holes, symbols)
    frozen = load_frozen(helpers.BARS_FILE, key, helpers.BAR_ARRAYS)
    assert frozen is not None, f"{key} is missing from {helpers.BARS_FILE.name}"
    bars = helpers.make_bars(
        seed=seed, n_bars=n_bars, n_syms=n_syms, holes=holes, symbols=list(symbols or []) or None
    )
    live = helpers.generate_bar_arrays(seed, n_bars, n_syms, holes=holes)
    for name in helpers.BAR_ARRAYS:
        assert np.array_equal(getattr(bars, name).to_numpy(), frozen[name], equal_nan=True)
        np.testing.assert_allclose(frozen[name], live[name], rtol=1e-12, atol=0.0)
