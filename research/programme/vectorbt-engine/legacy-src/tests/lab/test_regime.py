# tests/lab/test_regime.py
import numpy as np
import pandas as pd

from ggTrader.lab.regime import classify_regime, compute_regime_scalar


def _spy(n=400, start=100.0, drift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    rets = drift + rng.normal(0, 0.01, n)
    idx = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
    return pd.Series(start * np.exp(np.cumsum(rets)), index=idx)


def test_trend_up_when_above_ema():
    spy = _spy(drift=0.002, seed=1)  # steady uptrend
    reg = classify_regime(spy, ema_period=50)
    # After warmup, an uptrending series sits above its EMA → trend "up"
    tail = reg["trend"].dropna().iloc[-50:]
    assert (tail == "up").mean() > 0.8


def test_regime_is_lookahead_safe():
    """Appending future bars must not change any past label."""
    spy = _spy(n=400, seed=2)
    reg_full = classify_regime(spy, ema_period=50)
    reg_trunc = classify_regime(spy.iloc[:300], ema_period=50)
    # Past labels (well after warmup) must be identical whether or not the
    # future exists. Compare the overlap, skipping warmup NaNs.
    a = reg_full["label"].iloc[100:300]
    b = reg_trunc["label"].iloc[100:300]
    pd.testing.assert_series_equal(a, b)


def test_compute_regime_scalar_maps_labels():
    spy = _spy(drift=0.002, seed=1)
    scalar_map = {
        "up_calm": 2.0,
        "up_normal": 1.5,
        "up_turbulent": 1.0,
        "down_calm": 1.0,
        "down_normal": 0.7,
        "down_turbulent": 0.5,
    }
    s = compute_regime_scalar(spy, scalar_map, ema_period=50, default=1.0)
    assert s.index.equals(spy.index)
    assert s.notna().all()  # default fills warmup
    assert set(s.unique()).issubset(set(scalar_map.values()) | {1.0})
    # Uptrend series should spend most time at an up_* scalar (>= 1.0)
    assert (s.iloc[-50:] >= 1.0).mean() > 0.8
