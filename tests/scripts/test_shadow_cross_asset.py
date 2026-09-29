"""Shadow 80/20 tracker: the simulated book must match a hand-computed path."""

import importlib.util

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location("shadow", "scripts/shadow_cross_asset.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Daily growth per symbol: distinct so any weight mistake shows up.
_GROWTH = {"SPY": 0.010, "TLT": -0.005, "GLD": 0.004, "PDBC": 0.0, "BIL": 0.0001}


def _ohlcv(index: pd.DatetimeIndex) -> pd.DataFrame:
    frames = {}
    for sym, g in _GROWTH.items():
        close = 100.0 * (1 + g) ** np.arange(len(index))
        frames[sym] = pd.DataFrame(
            {"open": close, "high": close, "low": close, "close": close, "volume": 1e6},
            index=index,
        )
    df = pd.concat(frames, axis=1)
    df.columns.names = ["symbol", "field"]
    return df


def _hand_path(close: pd.DataFrame, trade_bars: list[pd.Timestamp]) -> pd.Series:
    """Buy the 80/20 weights at each trade bar's close; drift in between."""
    w = pd.Series({"SPY": 0.8, "TLT": 0.2 / 3, "GLD": 0.2 / 3, "PDBC": 0.2 / 3, "BIL": 0.0})
    value, shares, out = 1.0, None, []
    for ts in close.index:
        if shares is not None:
            value = float((shares * close.loc[ts, w.index]).sum())
        if ts in trade_bars:
            shares = w * value / close.loc[ts, w.index]
        out.append(value)
    return pd.Series(out, index=close.index)


def test_shadow_curve_matches_hand_computed_rebalance():
    idx = pd.bdate_range("2026-09-29", "2026-10-20", tz="UTC")
    ohlcv = _ohlcv(idx)
    curve = mod.shadow_curve(ohlcv, idx[0], idx[-1], cost_bp=0.0)

    # Decisions at the start and at the 09-30 month-end; each trades next bar.
    trade_bars = [idx[1], idx[idx > pd.Timestamp("2026-09-30", tz="UTC")][0]]
    expected = _hand_path(ohlcv.xs("close", axis=1, level="field"), trade_bars)
    assert curve.to_numpy() == pytest.approx(expected.to_numpy(), rel=1e-9)


def test_compare_table_rebases_every_book_to_start():
    idx = pd.date_range("2026-09-29", periods=3, tz="UTC")
    books = {
        "shadow": pd.Series([1.0, 1.1, 0.99], index=idx),
        "live": pd.Series([100.0, 90.0, 95.0], index=idx),
    }
    table = mod.compare(books, start_value=1000.0)
    assert table.loc["shadow", "value"] == pytest.approx(990.0)
    assert table.loc["live", "return_pct"] == pytest.approx(-5.0)
    assert table.loc["live", "max_dd_pct"] == pytest.approx(-10.0)
