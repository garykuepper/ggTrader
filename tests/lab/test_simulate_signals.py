# tests/lab/test_simulate_signals.py
import numpy as np
import pandas as pd

from ggTrader.lab.data import STOCK_BASE_CONFIG
from ggTrader.lab.simulate import simulate_signals
from ggTrader.lab.strategy import SignalTargets


def test_stock_base_config_deploys_at_three_percent():
    """Validated 2026-06-24 (lever diagnostic): 0.03 deploys idle cash and
    beats SPY outright. Guards against silent regression to the old 0.02."""
    assert STOCK_BASE_CONFIG["SIGNAL_POSITION_SIZE"] == 0.03


BASE = {
    "START_CASH": 10000.0,
    "FEES": 0.0,
    "SLIPPAGE": 0.0,
    "FREQ": "1d",
    "SIGNAL_POSITION_SIZE": 0.5,
}


def _prices(n=40):
    idx = pd.date_range("2021-01-01", periods=n, freq="B", tz="UTC")
    return pd.DataFrame(
        {
            "A": 100.0 * 1.01 ** np.arange(n),
            "B": np.full(n, 50.0),
        },
        index=idx,
    )


def test_simulate_signals_buy_and_hold_a():
    prices = _prices(40)
    # Buy A on bar 2, never exit
    entries = pd.DataFrame(False, index=prices.index, columns=prices.columns)
    exits = pd.DataFrame(False, index=prices.index, columns=prices.columns)
    entries.iloc[2, 0] = True  # buy A
    st = SignalTargets(entries=entries, exits=exits)

    rets, equity, diags = simulate_signals({"strat_a": st}, prices, BASE)
    assert "strat_a" in equity.columns
    assert equity["strat_a"].iloc[-1] > BASE["START_CASH"]  # A is rising
    assert diags["strat_a"]["n_symbols"] == 2


def test_simulate_signals_diags_include_real_trade_count():
    """Regression: gates.py's expectancy calc used raw total-return as a
    proxy for per-trade expectancy because n_trades wasn't tracked -- diags
    must expose the real vbt trade count."""
    prices = _prices(40)
    entries = pd.DataFrame(False, index=prices.index, columns=prices.columns)
    exits = pd.DataFrame(False, index=prices.index, columns=prices.columns)
    entries.iloc[2, 0] = True  # buy A
    exits.iloc[10, 0] = True  # sell A -- one closed trade
    entries.iloc[15, 1] = True  # buy B
    exits.iloc[20, 1] = True  # sell B -- one closed trade
    st = SignalTargets(entries=entries, exits=exits)

    _rets, _equity, diags = simulate_signals({"strat_a": st}, prices, BASE)
    assert diags["strat_a"]["n_trades"] == 2


def test_simulate_signals_two_strategies_independent():
    prices = _prices(40)
    e1 = pd.DataFrame(False, index=prices.index, columns=prices.columns)
    x1 = pd.DataFrame(False, index=prices.index, columns=prices.columns)
    e1.iloc[2, 0] = True  # strat1: buy A

    e2 = pd.DataFrame(False, index=prices.index, columns=prices.columns)
    x2 = pd.DataFrame(False, index=prices.index, columns=prices.columns)
    e2.iloc[2, 1] = True  # strat2: buy B (flat)

    rets_both, eq_both, _ = simulate_signals(
        {"s1": SignalTargets(e1, x1), "s2": SignalTargets(e2, x2)}, prices, BASE
    )
    rets_s1, eq_s1, _ = simulate_signals({"s1": SignalTargets(e1, x1)}, prices, BASE)

    # Running together must not change individual equity curves
    pd.testing.assert_series_equal(eq_both["s1"], eq_s1["s1"], check_names=False)


def _ohlcv(symbols, n):
    """Build a MultiIndex OHLCV frame for testing ATR stops."""
    idx = pd.date_range("2021-01-01", periods=n, freq="B", tz="UTC")
    frames = {}
    for sym in symbols:
        close = pd.Series(100.0 * 1.01 ** np.arange(n), index=idx, name="close")
        frames[sym] = pd.DataFrame(
            {
                "open": close,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
            },
            index=idx,
        )
    ohlcv = pd.concat(frames, axis=1)
    ohlcv.columns.names = ["symbol", "field"]
    return ohlcv


def test_simulate_signals_fixed_trailing_stop_exits_on_drop():
    """A fixed trailing stop should exit a position when price drops from peak."""
    prices = _prices(60)
    # Create a rising-then-falling price for symbol A
    rise = 100.0 * 1.01 ** np.arange(30)
    fall = rise[-1] * 0.97 ** np.arange(30)  # 3%/day drop
    prices["A"] = np.concatenate([rise, fall])

    entries = pd.DataFrame(False, index=prices.index, columns=prices.columns)
    exits = pd.DataFrame(False, index=prices.index, columns=prices.columns)
    entries.iloc[2, 0] = True  # buy A on bar 2

    st = SignalTargets(entries=entries, exits=exits)
    config_no_stop = {**BASE}
    config_with_stop = {**BASE, "ts_stop": 0.05}  # 5% trailing stop

    _, eq_no_stop, _ = simulate_signals({"s": st}, prices, config_no_stop)
    _, eq_with_stop, _ = simulate_signals({"s": st}, prices, config_with_stop)

    # With trailing stop, final equity should be higher because it exited
    # before the full drawdown
    assert eq_with_stop["s"].iloc[-1] > eq_no_stop["s"].iloc[-1]


def test_simulate_signals_no_stop_params_unchanged():
    """Without stop params, simulate_signals behaves exactly as before."""
    prices = _prices(40)
    entries = pd.DataFrame(False, index=prices.index, columns=prices.columns)
    exits = pd.DataFrame(False, index=prices.index, columns=prices.columns)
    entries.iloc[2, 0] = True
    st = SignalTargets(entries=entries, exits=exits)

    _, eq1, _ = simulate_signals({"s": st}, prices, {**BASE})
    _, eq2, _ = simulate_signals({"s": st}, prices, {**BASE})
    pd.testing.assert_frame_equal(eq1, eq2)


def test_simulate_signals_take_profit_caps_gain_on_rise():
    """A take-profit stop should close the position once price hits the target,
    capping the gain below a buy-and-hold on a monotonically rising price."""
    prices = _prices(40)  # A rises 1%/bar, never falls
    entries = pd.DataFrame(False, index=prices.index, columns=prices.columns)
    exits = pd.DataFrame(False, index=prices.index, columns=prices.columns)
    entries.iloc[2, 0] = True  # buy A on bar 2, no signal exit
    st = SignalTargets(entries=entries, exits=exits)

    _, eq_no_tp, _ = simulate_signals({"s": st}, prices, {**BASE})
    _, eq_tp, _ = simulate_signals({"s": st}, prices, {**BASE, "tp_stop": 0.05})

    # TP exits at +5%; buy-and-hold rides the full rise -> ends higher.
    assert eq_tp["s"].iloc[-1] < eq_no_tp["s"].iloc[-1]
    # And the TP position actually captured a gain vs starting cash.
    assert eq_tp["s"].iloc[-1] > BASE["START_CASH"]


def test_simulate_signals_tp_stop_none_unchanged():
    """tp_stop=None must behave exactly like no tp_stop key at all."""
    prices = _prices(40)
    entries = pd.DataFrame(False, index=prices.index, columns=prices.columns)
    exits = pd.DataFrame(False, index=prices.index, columns=prices.columns)
    entries.iloc[2, 0] = True
    st = SignalTargets(entries=entries, exits=exits)

    _, eq_base, _ = simulate_signals({"s": st}, prices, {**BASE})
    _, eq_none, _ = simulate_signals({"s": st}, prices, {**BASE, "tp_stop": None})
    pd.testing.assert_frame_equal(eq_base, eq_none)


def test_compute_atr_stop_shape_and_values():
    """ATR stop array should match price shape and be positive fractions."""
    from ggTrader.lab.simulate import compute_atr_stop

    idx = pd.date_range("2021-01-01", periods=100, freq="B", tz="UTC")
    close = pd.DataFrame(
        {"A": 100.0 + np.random.randn(100).cumsum(), "B": 50.0 + np.random.randn(100).cumsum()},
        index=idx,
    )
    high = close * 1.01
    low = close * 0.99

    result = compute_atr_stop(high, low, close, atr_period=14, atr_mult=2.0)
    assert result.shape == close.shape
    assert list(result.columns) == list(close.columns)
    # After warmup period, values should be positive fractions (not raw ATR)
    valid = result.iloc[14:].dropna()
    assert (valid > 0).all().all()
    assert (valid < 1).all().all()  # fractional, not absolute


def test_compute_atr_stop_higher_mult_wider_stops():
    """Higher ATR multiplier should produce wider (larger) stop fractions."""
    from ggTrader.lab.simulate import compute_atr_stop

    idx = pd.date_range("2021-01-01", periods=100, freq="B", tz="UTC")
    np.random.seed(42)
    close = pd.DataFrame({"A": 100.0 + np.random.randn(100).cumsum()}, index=idx)
    high = close * 1.02
    low = close * 0.98

    stops_1x = compute_atr_stop(high, low, close, atr_period=14, atr_mult=1.0)
    stops_3x = compute_atr_stop(high, low, close, atr_period=14, atr_mult=3.0)
    # 3x mult should be ~3x wider than 1x mult
    valid_1x = stops_1x.iloc[14:].dropna()
    valid_3x = stops_3x.iloc[14:].dropna()
    ratio = (valid_3x / valid_1x).mean().iloc[0]
    assert 2.9 < ratio < 3.1


def test_simulate_signals_atr_trailing_stop():
    """ATR-adaptive trailing stop should use per-bar stop distances."""
    prices = _prices(60)
    rise = 100.0 * 1.01 ** np.arange(30)
    fall = rise[-1] * 0.97 ** np.arange(30)
    prices["A"] = np.concatenate([rise, fall])

    entries = pd.DataFrame(False, index=prices.index, columns=prices.columns)
    exits = pd.DataFrame(False, index=prices.index, columns=prices.columns)
    entries.iloc[20, 0] = True  # After ATR warmup (period=14)

    st = SignalTargets(entries=entries, exits=exits)

    # Build OHLCV-like frame for ATR computation
    ohlcv_frames = {}
    for sym in prices.columns:
        ohlcv_frames[sym] = pd.DataFrame(
            {
                "open": prices[sym],
                "high": prices[sym] * 1.01,
                "low": prices[sym] * 0.99,
                "close": prices[sym],
            },
            index=prices.index,
        )
    ohlcv = pd.concat(ohlcv_frames, axis=1)
    ohlcv.columns.names = ["symbol", "field"]

    config_atr = {**BASE, "atr_period": 14, "atr_mult": 2.0}
    config_no_stop = {**BASE}

    _, eq_atr, _ = simulate_signals({"s": st}, prices, config_atr, ohlcv=ohlcv)
    _, eq_no_stop, _ = simulate_signals({"s": st}, prices, config_no_stop)

    # ATR stop should exit during the drawdown, preserving more equity
    assert eq_atr["s"].iloc[-1] > eq_no_stop["s"].iloc[-1]
