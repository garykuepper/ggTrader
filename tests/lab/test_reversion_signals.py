# tests/lab/test_reversion_signals.py
import numpy as np
import pandas as pd

from ggTrader.lab.strategies.signals import (
    BollingerReversionSignal,
    RsiReversionSignal,
    _bb_signals,
    _rsi_signals,
    build_signal_strategy,
)
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


def _mean_reverting_close(n=200):
    """Generate a price series that mean-reverts around 100."""
    np.random.seed(42)
    noise = np.random.normal(0, 3.0, n)
    price = np.full(n, 100.0)
    for i in range(1, n):
        price[i] = price[i - 1] + 0.1 * (100.0 - price[i - 1]) + noise[i]
    return price


# --- BollingerReversionSignal tests ---


def test_bb_reversion_select_returns_eligible():
    ohlcv = _ohlcv(["A", "B", "C"])
    strat = BollingerReversionSignal(LabConfig(min_history_bars=400))
    sels = strat.select(ohlcv.index[-1], ohlcv, ["A", "B", "C"])
    assert [s["symbol"] for s in sels] == ["A", "B", "C"]
    assert all("bb_period" in s and "bb_std" in s for s in sels)


def test_bb_reversion_select_respects_min_history():
    ohlcv = _ohlcv(["A"], n=200)
    strat = BollingerReversionSignal(LabConfig(min_history_bars=400))
    sels = strat.select(ohlcv.index[-1], ohlcv, ["A"])
    assert sels == []


def test_bb_reversion_to_targets_returns_signal_targets():
    ohlcv = _ohlcv(["A", "B"])
    strat = BollingerReversionSignal(LabConfig(min_history_bars=100))
    plans = {
        ohlcv.index[300]: [
            {"symbol": "A", "weight": 0.0, "bb_period": 20, "bb_std": 2.0},
        ],
        ohlcv.index[450]: [
            {"symbol": "A", "weight": 0.0, "bb_period": 20, "bb_std": 2.0},
            {"symbol": "B", "weight": 0.0, "bb_period": 20, "bb_std": 2.0},
        ],
    }
    result = strat.to_targets(plans, ohlcv)
    assert isinstance(result, SignalTargets)
    assert result.entries.shape == result.exits.shape
    assert set(result.entries.columns) == {"A", "B"}
    assert result.entries.dtypes.eq(bool).all()
    assert result.exits.dtypes.eq(bool).all()


def test_bb_signals_entry_on_lower_band_cross():
    """Entry fires when price crosses below the lower Bollinger Band."""
    idx = _idx(100)
    base = np.full(100, 100.0)
    base[50] = 80.0  # sharp drop below lower band
    close = pd.DataFrame({"A": base}, index=idx)
    entries, exits = _bb_signals(close, period=20, std=2.0)
    assert entries["A"].iloc[50], "Should enter on sharp drop below lower band"


def test_bb_signals_exit_on_sma_cross():
    """Exit fires when price crosses back above the SMA."""
    idx = _idx(100)
    base = np.full(100, 100.0)
    base[50] = 80.0  # drop below lower band
    base[51:] = 100.0  # snap back to mean
    close = pd.DataFrame({"A": base}, index=idx)
    _, exits = _bb_signals(close, period=20, std=2.0)
    exit_bars = exits["A"].iloc[51:55]
    assert exit_bars.any(), "Should exit when price returns to SMA"


def test_bb_signals_no_entry_in_warmup():
    """No signals during the warmup period (before enough bars for rolling)."""
    idx = _idx(50)
    close = pd.DataFrame({"A": np.full(50, 100.0)}, index=idx)
    entries, exits = _bb_signals(close, period=20, std=2.0)
    assert not entries["A"].iloc[:20].any()


def test_bb_sweep_params():
    params = BollingerReversionSignal.sweep_params()
    assert "bb_period" in params
    assert "bb_std" in params
    assert len(params["bb_period"]) >= 3
    assert len(params["bb_std"]) >= 3


def test_bb_sweep_signals_produces_all_combos():
    ohlcv = _ohlcv(["A", "B"])
    strat = BollingerReversionSignal(LabConfig(min_history_bars=100))
    combos = [
        {"bb_period": 10, "bb_std": 1.5},
        {"bb_period": 20, "bb_std": 2.0},
    ]
    symbols = ["A", "B"]
    result = strat.sweep_signals(combos, symbols, ohlcv)
    assert len(result) == 2
    for key, st in result.items():
        assert isinstance(st, SignalTargets)
        assert set(st.entries.columns) == {"A", "B"}


def test_bb_wider_std_fewer_entries():
    """Wider bands should produce fewer entry signals."""
    idx = _idx(500)
    np.random.seed(42)
    close = pd.DataFrame({"A": 100.0 + np.cumsum(np.random.normal(0, 1, 500))}, index=idx)
    entries_tight, _ = _bb_signals(close, period=20, std=1.5)
    entries_wide, _ = _bb_signals(close, period=20, std=3.0)
    assert entries_tight["A"].sum() >= entries_wide["A"].sum()


# --- RsiReversionSignal tests ---


def test_rsi_reversion_select_returns_eligible():
    ohlcv = _ohlcv(["A", "B"])
    strat = RsiReversionSignal(LabConfig(min_history_bars=400))
    sels = strat.select(ohlcv.index[-1], ohlcv, ["A", "B"])
    assert [s["symbol"] for s in sels] == ["A", "B"]
    assert all("rsi_period" in s and "rsi_oversold" in s and "rsi_exit" in s for s in sels)


def test_rsi_reversion_select_respects_min_history():
    ohlcv = _ohlcv(["A"], n=200)
    strat = RsiReversionSignal(LabConfig(min_history_bars=400))
    sels = strat.select(ohlcv.index[-1], ohlcv, ["A"])
    assert sels == []


def test_rsi_reversion_to_targets_returns_signal_targets():
    ohlcv = _ohlcv(["A", "B"])
    strat = RsiReversionSignal(LabConfig(min_history_bars=100))
    plans = {
        ohlcv.index[300]: [
            {"symbol": "A", "weight": 0.0, "rsi_period": 14, "rsi_oversold": 30, "rsi_exit": 50},
        ],
    }
    result = strat.to_targets(plans, ohlcv)
    assert isinstance(result, SignalTargets)
    assert result.entries.dtypes.eq(bool).all()


def test_rsi_signals_entry_on_oversold():
    """Entry fires when RSI drops below oversold threshold."""
    idx = _idx(100)
    np.random.seed(7)
    price = 100.0 + np.cumsum(np.random.normal(0.05, 0.5, 100))
    price[50:55] = [price[49], price[49] - 3, price[49] - 7, price[49] - 12, price[49] - 18]
    price[55:] = price[54] + np.cumsum(np.random.normal(0.2, 0.3, 45))
    close = pd.DataFrame({"A": price}, index=idx)
    entries, _ = _rsi_signals(close, period=14, oversold=30, exit_level=50)
    assert entries["A"].any(), "Should have entries when RSI drops below 30"


def test_rsi_signals_exit_on_neutral():
    """Exit fires when RSI crosses above exit level."""
    idx = _idx(200)
    price = _mean_reverting_close(200)
    close = pd.DataFrame({"A": price}, index=idx)
    _, exits = _rsi_signals(close, period=14, oversold=30, exit_level=50)
    assert exits["A"].any(), "Should have exits when RSI returns above 50"


def test_rsi_signals_no_entry_in_warmup():
    """No signals in the first rsi_period bars."""
    idx = _idx(50)
    close = pd.DataFrame({"A": np.linspace(100, 90, 50)}, index=idx)
    entries, _ = _rsi_signals(close, period=14, oversold=30, exit_level=50)
    assert not entries["A"].iloc[:14].any()


def test_rsi_sweep_params():
    params = RsiReversionSignal.sweep_params()
    assert "rsi_period" in params
    assert "rsi_oversold" in params
    assert "rsi_exit" in params


def test_rsi_sweep_signals_produces_all_combos():
    ohlcv = _ohlcv(["A"])
    strat = RsiReversionSignal(LabConfig(min_history_bars=100))
    combos = [
        {"rsi_period": 7, "rsi_oversold": 20, "rsi_exit": 50},
        {"rsi_period": 14, "rsi_oversold": 30, "rsi_exit": 60},
    ]
    result = strat.sweep_signals(combos, ["A"], ohlcv)
    assert len(result) == 2
    for st in result.values():
        assert isinstance(st, SignalTargets)


def test_rsi_lower_oversold_fewer_entries():
    """Lower oversold threshold should produce fewer entry signals."""
    idx = _idx(500)
    price = _mean_reverting_close(500)
    close = pd.DataFrame({"A": price}, index=idx)
    entries_loose, _ = _rsi_signals(close, period=14, oversold=30, exit_level=50)
    entries_strict, _ = _rsi_signals(close, period=14, oversold=20, exit_level=50)
    assert entries_loose["A"].sum() >= entries_strict["A"].sum()


# --- Registry tests ---


def test_build_signal_strategy_bb_reversion():
    strat = build_signal_strategy("bb_reversion", LabConfig())
    assert strat.name == "bb_reversion"
    assert strat.target_kind == "signals"


def test_build_signal_strategy_rsi_reversion():
    strat = build_signal_strategy("rsi_reversion", LabConfig())
    assert strat.name == "rsi_reversion"
    assert strat.target_kind == "signals"
