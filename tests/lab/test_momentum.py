import json

import numpy as np
import pandas as pd

from ggTrader.lab.strategies.momentum import (
    CrossSectionalMomentum,
    DualMomentum,
    build_strategy,
)
from ggTrader.lab.strategy import LabConfig


def make_ohlcv(prices: dict) -> pd.DataFrame:
    frames = {}
    for sym, close in prices.items():
        frames[sym] = pd.DataFrame(
            {
                "open": close,
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


def _idx(n, start="2020-01-01"):
    return pd.date_range(start, periods=n, freq="B", tz="UTC")


def test_momentum_ranks_strongest_first_equal_weight():
    idx = _idx(300)
    ohlcv = make_ohlcv(
        {
            "UP": pd.Series(np.linspace(10, 30, 300), index=idx),
            "FLAT": pd.Series(np.full(300, 20.0), index=idx),
            "DOWN": pd.Series(np.linspace(30, 10, 300), index=idx),
        }
    )
    strat = CrossSectionalMomentum(LabConfig(top_n=3))
    sels = strat.select(idx[-1], ohlcv, ["UP", "FLAT", "DOWN"])
    assert [s["symbol"] for s in sels] == ["UP", "FLAT", "DOWN"]
    assert all(abs(s["weight"] - 1 / 3) < 1e-12 for s in sels)
    json.dumps(sels)  # must be JSON-able


def test_momentum_ignores_rows_after_asof():
    idx = _idx(300)
    ohlcv = make_ohlcv(
        {
            "UP": pd.Series(np.linspace(10, 30, 300), index=idx),
            "DOWN": pd.Series(np.linspace(30, 10, 300), index=idx),
        }
    )
    asof = idx[-30]
    strat = CrossSectionalMomentum(LabConfig(top_n=2))
    unmasked = strat.select(asof, ohlcv, ["UP", "DOWN"])
    truncated = strat.select(asof, ohlcv.loc[:asof], ["UP", "DOWN"])
    assert json.dumps(unmasked, sort_keys=True) == json.dumps(truncated, sort_keys=True)


def test_dual_momentum_drops_negative():
    idx = _idx(300)
    ohlcv = make_ohlcv(
        {
            "UP": pd.Series(np.linspace(10, 30, 300), index=idx),
            "FLAT": pd.Series(np.full(300, 20.0), index=idx),
            "DOWN": pd.Series(np.linspace(30, 10, 300), index=idx),
        }
    )
    sels = DualMomentum(LabConfig(top_n=3)).select(idx[-1], ohlcv, ["UP", "FLAT", "DOWN"])
    assert [s["symbol"] for s in sels] == ["UP", "FLAT"]
    assert all(abs(s["weight"] - 1 / 3) < 1e-12 for s in sels)  # NOT renormalized


def test_to_targets_lays_weights_after_asof_and_zeros_drops():
    idx = _idx(120)
    ohlcv = make_ohlcv(
        {
            "A": pd.Series(np.linspace(10, 20, 120), index=idx),
            "B": pd.Series(np.linspace(10, 15, 120), index=idx),
        }
    )
    strat = CrossSectionalMomentum(LabConfig(top_n=1))
    t1, t2 = idx[60], idx[90]
    plans = {
        t1: [{"symbol": "A", "weight": 1.0, "momentum": 1.0}],
        t2: [{"symbol": "B", "weight": 1.0, "momentum": 0.5}],
    }
    targets = strat.to_targets(plans, ohlcv)
    first_after_t1 = ohlcv.index[ohlcv.index > t1][0]
    first_after_t2 = ohlcv.index[ohlcv.index > t2][0]
    assert targets.loc[first_after_t1, "A"] == 1.0
    assert targets.loc[first_after_t2, "A"] == 0.0  # dropped -> exit
    assert targets.loc[first_after_t2, "B"] == 1.0
    assert targets.notna().any(axis=1).sum() == 2  # only two rebalance rows carry orders


def test_build_strategy_dispatch():
    assert build_strategy("xs_momentum", LabConfig()).name == "xs_momentum"
    assert build_strategy("dual_momentum", LabConfig()).name == "dual_momentum"
    try:
        build_strategy("nope", LabConfig())
        assert False, "expected ValueError"
    except ValueError:
        pass
