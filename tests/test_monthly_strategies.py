"""Unit tests for pluggable monthly strategies (synthetic data, no network)."""

import json

import numpy as np
import pandas as pd

from ggTrader.research.equity_wfo import STOCK_BASE_CONFIG
from ggTrader.research.monthly_strategies import (
    CrossSectionalMomentum,
    DualMomentum,
    simulate_hold_weights,
)
from ggTrader.research.monthly_walkforward import MonthlyHarnessConfig


def make_ohlcv(prices: dict) -> pd.DataFrame:
    """Build a (symbol, field) MultiIndex OHLCV frame from close-price series."""
    frames = {}
    for sym, close in prices.items():
        frames[sym] = pd.DataFrame(
            {
                "open": close.shift(1).fillna(close.iloc[0]),
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": np.full(len(close), 1e6),
            },
            index=close.index,
        )
    out = pd.concat(frames, axis=1)
    out.columns = out.columns.set_names(["symbol", "field"])
    return out


def _idx(n: int, start: str = "2020-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="B", tz="UTC")


def test_momentum_ranks_strongest_first_with_equal_weights():
    idx = _idx(300)
    ohlcv = make_ohlcv(
        {
            "UP": pd.Series(np.linspace(10, 30, 300), index=idx),
            "FLAT": pd.Series(np.full(300, 20.0), index=idx),
            "DOWN": pd.Series(np.linspace(30, 10, 300), index=idx),
        }
    )
    cfg = MonthlyHarnessConfig(top_n=3)
    strat = CrossSectionalMomentum(cfg, STOCK_BASE_CONFIG)
    sels = strat.select(idx[-1], ohlcv, ["UP", "FLAT", "DOWN"])
    assert [s["symbol"] for s in sels] == ["UP", "FLAT", "DOWN"]
    assert sels[0]["momentum"] > 0.0 > sels[2]["momentum"]
    assert all(abs(s["weight"] - 1.0 / 3.0) < 1e-12 for s in sels)
    json.dumps(sels)  # selections must be JSON-able for checkpoints


def test_momentum_skip_window_is_actually_skipped():
    idx = _idx(300)
    base = pd.Series(np.linspace(10, 20, 300), index=idx)
    jumped = base.copy()
    jumped.iloc[-21:] = jumped.iloc[-21:] * 2.0  # move entirely inside skip window
    ohlcv = make_ohlcv({"BASE": base, "JUMP": jumped})
    cfg = MonthlyHarnessConfig(top_n=2)
    strat = CrossSectionalMomentum(cfg, STOCK_BASE_CONFIG, lookback=252, skip=21)
    sels = {s["symbol"]: s["momentum"] for s in strat.select(idx[-1], ohlcv, ["BASE", "JUMP"])}
    assert sels["BASE"] == sels["JUMP"]


def test_momentum_respects_top_n_and_short_history():
    idx = _idx(300)
    short_idx = idx[-100:]  # < lookback+1 bars -> ineligible
    ohlcv = make_ohlcv(
        {
            "A": pd.Series(np.linspace(10, 40, 300), index=idx),
            "B": pd.Series(np.linspace(10, 30, 300), index=idx),
            "C": pd.Series(np.linspace(10, 20, 300), index=idx),
            "NEW": pd.Series(np.linspace(10, 99, 100), index=short_idx),
        }
    )
    cfg = MonthlyHarnessConfig(top_n=2)
    strat = CrossSectionalMomentum(cfg, STOCK_BASE_CONFIG)
    sels = strat.select(idx[-1], ohlcv, ["A", "B", "C", "NEW"])
    assert [s["symbol"] for s in sels] == ["A", "B"]
    assert all(abs(s["weight"] - 0.5) < 1e-12 for s in sels)


def test_simulate_hold_weights_matches_hand_computed_returns():
    idx = _idx(42, start="2021-01-01")
    a = pd.Series(100.0 * 1.01 ** np.arange(42), index=idx)  # +1%/day
    b = pd.Series(np.full(42, 50.0), index=idx)  # flat
    ohlcv = make_ohlcv({"A": a, "B": b})
    asof, month_end = idx[20], idx[-1]
    frictionless = {**STOCK_BASE_CONFIG, "FEES": 0.0, "SLIPPAGE": 0.0}
    rets, diags = simulate_hold_weights(ohlcv, {"A": 0.5, "B": 0.5}, asof, month_end, frictionless)

    # Buys at the close of the first forward bar (idx[21]); B is flat, so the
    # portfolio return is exactly half of A's appreciation from that bar.
    expected_total = 0.5 * (a.iloc[-1] / a.loc[idx[21]] - 1.0)
    total = float((1.0 + rets).prod() - 1.0)
    assert abs(total - expected_total) < 1e-9
    assert rets.index.min() > asof and rets.index.max() <= month_end
    assert diags["n_positions"] == 2
    assert 0.90 <= diags["avg_exposure"] <= 1.01


def test_simulate_hold_weights_empty_inputs_are_flat():
    idx = _idx(42, start="2021-01-01")
    ohlcv = make_ohlcv({"A": pd.Series(np.full(42, 10.0), index=idx)})
    rets, diags = simulate_hold_weights(ohlcv, {}, idx[20], idx[-1], STOCK_BASE_CONFIG)
    assert rets.empty
    assert diags["n_positions"] == 0


def test_dual_momentum_drops_only_negative_momentum():
    idx = _idx(300)
    ohlcv = make_ohlcv(
        {
            "UP": pd.Series(np.linspace(10, 30, 300), index=idx),
            "FLAT": pd.Series(np.full(300, 20.0), index=idx),
            "DOWN": pd.Series(np.linspace(30, 10, 300), index=idx),
        }
    )
    cfg = MonthlyHarnessConfig(top_n=3)
    strat = DualMomentum(cfg, STOCK_BASE_CONFIG)
    sels = strat.select(idx[-1], ohlcv, ["UP", "FLAT", "DOWN"])
    assert [s["symbol"] for s in sels] == ["UP", "FLAT"]  # FLAT momentum == 0.0 kept
    # weights are NOT renormalized — DOWN's slot stays in cash
    assert all(abs(s["weight"] - 1.0 / 3.0) < 1e-12 for s in sels)


def test_dual_momentum_all_negative_is_flat_month():
    idx = _idx(300)
    ohlcv = make_ohlcv(
        {
            "D1": pd.Series(np.linspace(30, 10, 300), index=idx),
            "D2": pd.Series(np.linspace(40, 20, 300), index=idx),
        }
    )
    cfg = MonthlyHarnessConfig(top_n=2)
    strat = DualMomentum(cfg, STOCK_BASE_CONFIG)
    sels = strat.select(idx[-60], ohlcv.loc[: idx[-60]], ["D1", "D2"])
    assert sels == []
    rets, diags = strat.simulate(ohlcv, sels, idx[-60], idx[-1])
    assert rets.empty and diags["n_positions"] == 0
