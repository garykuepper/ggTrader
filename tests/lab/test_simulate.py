import numpy as np
import pandas as pd

from ggTrader.lab.simulate import simulate_weights

BASE = {"START_CASH": 10000.0, "FEES": 0.0, "SLIPPAGE": 0.0, "FREQ": "1d"}


def _prices(n=40):
    idx = pd.date_range("2021-01-01", periods=n, freq="B", tz="UTC")
    return pd.DataFrame(
        {
            "A": 100.0 * 1.01 ** np.arange(n),  # +1%/day
            "B": np.full(n, 50.0),  # flat
        },
        index=idx,
    )


def _targets(prices, first_weights):
    t = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    t.iloc[1] = [first_weights.get(c, 0.0) for c in prices.columns]
    return t


def test_simulate_weights_matches_hand_computed_return():
    prices = _prices()
    targets = {"x": _targets(prices, {"A": 0.5, "B": 0.5})}
    rets, equity, diags = simulate_weights(targets, prices, BASE)
    # Buy at bar 1; B flat, so portfolio total return is half of A's appreciation.
    expected = 0.5 * (prices["A"].iloc[-1] / prices["A"].iloc[1] - 1.0)
    total = float(equity["x"].iloc[-1] / BASE["START_CASH"] - 1.0)
    assert abs(total - expected) < 1e-6
    assert diags["x"]["n_strategies"] == 1


def test_simulate_weights_runs_strategies_simultaneously_and_equally():
    prices = _prices()
    together = {"x": _targets(prices, {"A": 1.0}), "y": _targets(prices, {"B": 1.0})}
    r_both, eq_both, _ = simulate_weights(together, prices, BASE)
    r_x, eq_x, _ = simulate_weights({"x": together["x"]}, prices, BASE)
    # Grouped multi-strategy run must equal the strategy run alone (vectorization guard).
    pd.testing.assert_series_equal(eq_both["x"], eq_x["x"], check_names=False)
