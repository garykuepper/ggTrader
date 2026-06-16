# tests/lab/test_simulate_signals.py
import numpy as np
import pandas as pd

from ggTrader.lab.simulate import simulate_signals
from ggTrader.lab.strategy import SignalTargets

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
