# tests/lab/test_signals.py
import json

import numpy as np
import pandas as pd

from ggTrader.lab.strategies.signals import EmaCrossSignal, build_signal_strategy
from ggTrader.lab.strategy import LabConfig, SignalTargets


def _idx(n, start="2020-01-01"):
    return pd.date_range(start, periods=n, freq="B", tz="UTC")


def _ohlcv(symbols, n=600):
    idx = _idx(n)
    frames = {}
    for i, s in enumerate(symbols):
        close = pd.Series(100.0 * (1 + 0.0003 * (i + 1)) ** np.arange(n), index=idx)
        frames[s] = pd.DataFrame(
            {
                "open": close,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": np.full(n, 1e6),
            },
            index=idx,
        )
    out = pd.concat(frames, axis=1)
    out.columns = out.columns.set_names(["symbol", "field"])
    return out


def test_ema_cross_select_returns_eligible_symbols():
    ohlcv = _ohlcv(["A", "B", "C"])
    strat = EmaCrossSignal(LabConfig(min_history_bars=400))
    asof = ohlcv.index[-1]
    sels = strat.select(asof, ohlcv, ["A", "B", "C"])
    assert [s["symbol"] for s in sels] == ["A", "B", "C"]
    assert all("ema_fast" in s and "ema_slow" in s for s in sels)
    assert all(s["weight"] == 0.0 for s in sels)


def test_ema_cross_select_respects_min_history():
    ohlcv = _ohlcv(["A"], n=200)  # fewer bars than min_history_bars=400
    strat = EmaCrossSignal(LabConfig(min_history_bars=400))
    sels = strat.select(ohlcv.index[-1], ohlcv, ["A"])
    assert sels == []


def test_ema_cross_to_targets_returns_signal_targets():
    ohlcv = _ohlcv(["A", "B"])
    strat = EmaCrossSignal(LabConfig(min_history_bars=100))
    asof1 = ohlcv.index[300]
    asof2 = ohlcv.index[450]
    plans = {
        asof1: [{"symbol": "A", "weight": 0.0, "ema_fast": 20, "ema_slow": 50}],
        asof2: [
            {"symbol": "A", "weight": 0.0, "ema_fast": 20, "ema_slow": 50},
            {"symbol": "B", "weight": 0.0, "ema_fast": 20, "ema_slow": 50},
        ],
    }
    result = strat.to_targets(plans, ohlcv)
    assert isinstance(result, SignalTargets)
    assert result.entries.shape == result.exits.shape
    assert set(result.entries.columns) == {"A", "B"}
    assert result.entries.dtypes.eq(bool).all()
    assert result.exits.dtypes.eq(bool).all()


def test_ema_cross_no_lookahead():
    ohlcv = _ohlcv(["A"])
    strat = EmaCrossSignal(LabConfig(min_history_bars=100))
    asof = ohlcv.index[-30]
    full = strat.select(asof, ohlcv.loc[:asof], ["A"])
    truncated = strat.select(asof, ohlcv.loc[:asof].copy(), ["A"])
    unmasked = strat.select(asof, ohlcv, ["A"])
    assert (
        json.dumps(full, sort_keys=True)
        == json.dumps(truncated, sort_keys=True)
        == json.dumps(unmasked, sort_keys=True)
    )


def test_build_signal_strategy_dispatch():
    cfg = LabConfig()
    assert build_signal_strategy("ema_cross", cfg).name == "ema_cross"
    try:
        build_signal_strategy("bogus", cfg)
        assert False, "expected ValueError"
    except ValueError:
        pass
