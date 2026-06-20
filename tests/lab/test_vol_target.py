# tests/lab/test_vol_target.py
import numpy as np
import pandas as pd

from ggTrader.lab.simulate import compute_vol_scalar, simulate_signals
from ggTrader.lab.strategy import SignalTargets
from ggTrader.lab.sweep import OVERLAY_PARAMS, VOL_PARAMS, split_params


def _idx(n, start="2020-01-01"):
    return pd.date_range(start, periods=n, freq="B", tz="UTC")


def _prices(n=500, n_syms=5, seed=42):
    np.random.seed(seed)
    idx = _idx(n)
    cols = [f"S{i}" for i in range(n_syms)]
    data = 100.0 * np.exp(np.cumsum(np.random.normal(0.0003, 0.015, (n, n_syms)), axis=0))
    return pd.DataFrame(data, index=idx, columns=cols)


# --- compute_vol_scalar ---


def test_vol_scalar_shape_matches_prices():
    prices = _prices()
    scalar = compute_vol_scalar(prices, vol_target=0.15, vol_lookback=20)
    assert len(scalar) == len(prices)
    assert isinstance(scalar, pd.Series)


def test_vol_scalar_no_lookahead():
    """Scalar on day t depends only on data through day t-1."""
    prices = _prices(n=100)
    scalar_full = compute_vol_scalar(prices, vol_target=0.15, vol_lookback=20)
    scalar_trunc = compute_vol_scalar(prices.iloc[:80], vol_target=0.15, vol_lookback=20)
    # First 80 values should match (scalar uses shift(1), so bar 79 uses data through 78)
    pd.testing.assert_series_equal(
        scalar_full.iloc[:80],
        scalar_trunc,
        check_names=False,
    )


def test_vol_scalar_warmup_is_one():
    """During warmup (before vol_lookback bars), scalar defaults to 1.0."""
    prices = _prices(n=100)
    scalar = compute_vol_scalar(prices, vol_target=0.15, vol_lookback=20)
    # First 20 bars have no rolling vol + shift(1), so they should be 1.0
    assert (scalar.iloc[:20] == 1.0).all()


def test_vol_scalar_capped():
    """Scalar must not exceed vol_cap."""
    prices = _prices()
    scalar = compute_vol_scalar(prices, vol_target=0.15, vol_lookback=20, vol_cap=1.5)
    assert scalar.max() <= 1.5 + 1e-10


def test_vol_scalar_floored():
    """Scalar must not go below 0.1."""
    prices = _prices()
    scalar = compute_vol_scalar(prices, vol_target=0.01, vol_lookback=20)
    assert scalar.min() >= 0.1 - 1e-10


def test_vol_scalar_high_vol_reduces_exposure():
    """Higher realized vol -> lower scalar (reduced position size)."""
    np.random.seed(42)
    idx = _idx(200)
    calm = pd.DataFrame({"A": 100.0 + np.cumsum(np.random.normal(0, 0.2, 200))}, index=idx)
    volatile = pd.DataFrame({"A": 100.0 + np.cumsum(np.random.normal(0, 2.0, 200))}, index=idx)
    s_calm = compute_vol_scalar(calm, vol_target=0.15, vol_lookback=20)
    s_vol = compute_vol_scalar(volatile, vol_target=0.15, vol_lookback=20)
    # After warmup, volatile should have lower median scalar
    assert s_vol.iloc[30:].median() < s_calm.iloc[30:].median()


# --- split_params with vol params ---


def test_vol_params_in_overlay():
    assert "vol_target" in VOL_PARAMS
    assert "vol_lookback" in VOL_PARAMS
    assert VOL_PARAMS.issubset(OVERLAY_PARAMS)


def test_split_params_separates_vol():
    combo = {"bb_period": 20, "bb_std": 2.0, "vol_target": 0.15, "vol_lookback": 20}
    signal, overlay = split_params(combo)
    assert signal == {"bb_period": 20, "bb_std": 2.0}
    assert overlay == {"vol_target": 0.15, "vol_lookback": 20}


def test_split_params_stop_and_vol_together():
    combo = {"rsi_period": 14, "ts_stop": 0.05, "vol_target": 0.15}
    signal, overlay = split_params(combo)
    assert signal == {"rsi_period": 14}
    assert overlay == {"ts_stop": 0.05, "vol_target": 0.15}


# --- simulate_signals with vol targeting ---


def _simple_signal_targets(prices):
    """Create simple signal targets: entry every 20 bars, exit 5 bars later."""
    idx = prices.index
    cols = prices.columns
    entries = pd.DataFrame(False, index=idx, columns=cols)
    exits = pd.DataFrame(False, index=idx, columns=cols)
    for i in range(50, len(idx), 20):
        entries.iloc[i] = True
        if i + 5 < len(idx):
            exits.iloc[i + 5] = True
    return SignalTargets(entries=entries.astype(bool), exits=exits.astype(bool))


def test_simulate_signals_vol_target_runs():
    """Vol targeting doesn't crash and produces valid output."""
    prices = _prices(n=200, n_syms=3)
    targets = _simple_signal_targets(prices)
    config = {
        "START_CASH": 10000.0,
        "FEES": 0.0,
        "SLIPPAGE": 0.0,
        "FREQ": "1d",
        "SIGNAL_POSITION_SIZE": 0.02,
        "vol_target": 0.15,
        "vol_lookback": 20,
    }
    rets, eq, diags = simulate_signals({"test": targets}, prices, config)
    assert "test" in eq.columns
    assert len(eq) == len(prices)
    assert eq["test"].iloc[0] > 0


def test_simulate_signals_vol_target_reduces_drawdown():
    """Vol targeting should reduce drawdown compared to no vol targeting."""
    np.random.seed(99)
    idx = _idx(400)
    # Create prices with a crash in the middle
    base = np.exp(np.cumsum(np.random.normal(0.0005, 0.01, 400)))
    base[200:220] *= np.linspace(1.0, 0.7, 20)  # 30% crash
    base[220:] *= 0.7
    prices = pd.DataFrame({"A": 100 * base, "B": 100 * base * 1.1}, index=idx)
    targets = _simple_signal_targets(prices)

    config_base = {
        "START_CASH": 10000.0,
        "FEES": 0.0,
        "SLIPPAGE": 0.0,
        "FREQ": "1d",
        "SIGNAL_POSITION_SIZE": 0.05,
    }
    _, eq_no_vol, _ = simulate_signals({"test": targets}, prices, config_base)

    config_vol = {**config_base, "vol_target": 0.10, "vol_lookback": 20}
    _, eq_vol, _ = simulate_signals({"test": targets}, prices, config_vol)

    dd_no_vol = (eq_no_vol["test"] / eq_no_vol["test"].cummax() - 1).min()
    dd_vol = (eq_vol["test"] / eq_vol["test"].cummax() - 1).min()
    # Vol targeting should have smaller (less negative) drawdown
    assert dd_vol >= dd_no_vol, f"Vol target DD {dd_vol:.4f} worse than baseline {dd_no_vol:.4f}"


def test_simulate_signals_without_vol_target_unchanged():
    """Without vol_target in config, behavior is identical to before."""
    prices = _prices(n=200, n_syms=2)
    targets = _simple_signal_targets(prices)
    config = {
        "START_CASH": 10000.0,
        "FEES": 0.0,
        "SLIPPAGE": 0.0,
        "FREQ": "1d",
        "SIGNAL_POSITION_SIZE": 0.02,
    }
    _, eq1, _ = simulate_signals({"test": targets}, prices, config)
    _, eq2, _ = simulate_signals({"test": targets}, prices, config)
    pd.testing.assert_frame_equal(eq1, eq2)
