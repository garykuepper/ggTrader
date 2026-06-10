"""Parity tests for metrics._profit_factor_raw vs vbt's native Trades.profit_factor.

_profit_factor_raw replaces the monkey-patched profit_factor from the deleted
vbt_patches.py. It must match native vbt semantics exactly: no trades -> NaN,
all-win -> inf, all-loss -> 0.0, mixed -> gross_profit / |gross_loss|.
"""

import numpy as np
import pandas as pd
import vectorbt as vbt

from ggTrader.core.metrics import _profit_factor_raw


def _native_reference(trades) -> pd.Series:
    """vbt's Trades.profit_factor() formula (vectorbt/portfolio/trades.py) on writable
    copies. The real accessor mutates its arrays in place and crashes when vbt hands
    back read-only views — the very bug _profit_factor_raw exists to avoid — so the
    parity reference replicates the documented formula instead of calling it."""
    total_win = np.array(np.atleast_1d(np.asarray(trades.winning.pnl.sum())), dtype=float)
    total_loss = np.array(np.atleast_1d(np.asarray(trades.losing.pnl.sum())), dtype=float)
    has_values = np.atleast_1d(np.asarray(trades.count())) > 0
    total_win[np.isnan(total_win) & has_values] = 0.0
    total_loss[np.isnan(total_loss) & has_values] = 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        result = total_win / np.abs(total_loss)
    return pd.Series(result, index=trades.wrapper.grouper.get_columns())


def _build_grid_portfolio() -> vbt.Portfolio:
    """Grouped portfolio with 4 combos: all-win, mixed, all-loss, no-trade."""
    n = 80
    idx = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
    cols = pd.MultiIndex.from_product([[0, 1, 2, 3], ["A", "B"]], names=["param_combo", "symbol"])

    close = pd.DataFrame(100.0, index=idx, columns=cols)
    ramp_up = 100.0 + np.arange(n)  # monotonically rising -> long trades win
    ramp_down = 100.0 - 0.5 * np.arange(n)  # falling -> long trades lose
    rng = np.random.default_rng(7)
    wiggle = 100.0 + np.cumsum(rng.normal(0, 2.0, n))

    for sym in ["A", "B"]:
        close[(0, sym)] = ramp_up  # combo 0: all wins
        close[(1, sym)] = wiggle  # combo 1: mixed
        close[(2, sym)] = ramp_down  # combo 2: all losses
        close[(3, sym)] = ramp_up  # combo 3: no trades (no entries)

    entries = pd.DataFrame(False, index=idx, columns=cols)
    exits = pd.DataFrame(False, index=idx, columns=cols)
    for combo in [0, 1, 2]:
        for sym in ["A", "B"]:
            entries.loc[idx[5], (combo, sym)] = True
            exits.loc[idx[30], (combo, sym)] = True
            entries.loc[idx[40], (combo, sym)] = True
            exits.loc[idx[70], (combo, sym)] = True

    return vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        init_cash=10_000.0,
        fees=0.0,
        slippage=0.0,
        freq="1d",
        cash_sharing=True,
        group_by=cols.droplevel(-1),
    ).copy()


def test_profit_factor_raw_matches_native_vbt():
    pf = _build_grid_portfolio()
    ours = _profit_factor_raw(pf)
    native = _native_reference(pf.trades)

    assert list(ours.index) == list(native.index)
    for label in native.index:
        o, n = float(ours.loc[label]), float(native.loc[label])
        if np.isnan(n):
            assert np.isnan(o), f"combo {label}: expected NaN, got {o}"
        elif np.isinf(n):
            assert np.isinf(o), f"combo {label}: expected inf, got {o}"
        else:
            assert o == n, f"combo {label}: {o} != {n}"


def test_profit_factor_raw_edge_semantics():
    pf = _build_grid_portfolio()
    ours = _profit_factor_raw(pf)

    assert np.isinf(ours.loc[0]), "all-win combo must be inf"
    assert np.isfinite(ours.loc[1]) and ours.loc[1] > 0, "mixed combo must be finite > 0"
    assert ours.loc[2] == 0.0, "all-loss combo must be 0.0"
    assert np.isnan(ours.loc[3]), "no-trade combo must be NaN"


def test_profit_factor_raw_handles_readonly_pnl():
    """The whole point of the replacement: never mutate vbt's arrays in place."""
    pf = _build_grid_portfolio()
    pnl_arr = pf.trades.pnl.values
    pnl_arr.flags.writeable = False
    try:
        result = _profit_factor_raw(pf)
        assert len(result) == 4
    finally:
        pnl_arr.flags.writeable = True
