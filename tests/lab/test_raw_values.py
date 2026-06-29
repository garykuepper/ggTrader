"""Tests for raw-value extractors feeding the IC weighting."""

import numpy as np
import pandas as pd

from ggTrader.lab.strategies.indicators import (
    bb_raw,
    ema_raw,
    macd_raw,
    rsi_raw,
    vbb_raw,
)


def _idx(n, start="2020-01-01"):
    return pd.date_range(start, periods=n, freq="B", tz="UTC")


def _close(n=120, n_syms=4, seed=7):
    np.random.seed(seed)
    idx = _idx(n)
    cols = {}
    for i in range(n_syms):
        cols[f"S{i}"] = 100.0 * np.exp(np.cumsum(np.random.normal(0, 0.012, n)))
    return pd.DataFrame(cols, index=idx)


def _volume(close, seed=8):
    np.random.seed(seed)
    return pd.DataFrame(
        np.random.randint(1000, 9000, size=close.shape).astype(float),
        index=close.index,
        columns=close.columns,
    )


def test_rsi_raw_higher_when_more_oversold():
    """A net-falling series (oversold) ranks ABOVE a net-rising one.

    Both series carry small periodic counter-moves so each has gains AND
    losses — otherwise a perfect monotonic trend yields avg_loss/avg_gain == 0
    and RSI is NaN (a degenerate, downstream-harmless case the IC drops).
    """
    idx = _idx(60)
    t = np.arange(60, dtype=float)
    falling = pd.Series(100.0 - t * 0.5, index=idx)
    falling.iloc[::5] += 1.0  # small bounces -> nonzero gains
    rising = pd.Series(60.0 + t * 0.5, index=idx)
    rising.iloc[::5] -= 1.0  # small dips -> nonzero losses
    close = pd.DataFrame({"DOWN": falling, "UP": rising})
    raw = rsi_raw(close, period=14)
    assert np.isfinite(raw["DOWN"].iloc[-1])
    assert np.isfinite(raw["UP"].iloc[-1])
    assert raw["DOWN"].iloc[-1] > raw["UP"].iloc[-1]


def test_bb_raw_higher_below_lower_band():
    """A price driven below its lower band ranks ABOVE one sitting at its mean."""
    idx = _idx(40)
    np.random.seed(1)
    base = 100.0 + np.random.normal(0, 0.5, 40)
    mid = pd.Series(base, index=idx)
    low = mid.copy()
    low.iloc[-1] = base[-1] - 10.0  # last bar dumps well below the lower band
    close = pd.DataFrame({"MID": mid, "LOW": low})
    raw = bb_raw(close, period=20, std=2.0)
    assert np.isfinite(raw["MID"].iloc[-1])
    assert np.isfinite(raw["LOW"].iloc[-1])
    assert raw["LOW"].iloc[-1] > raw["MID"].iloc[-1]


def test_ema_raw_positive_in_uptrend():
    idx = _idx(80)
    up = pd.Series(np.linspace(50, 150, 80), index=idx)
    close = pd.DataFrame({"UP": up})
    raw = ema_raw(close, fast=20, slow=50)
    assert raw["UP"].iloc[-1] > 0


def test_macd_raw_higher_in_uptrend_than_downtrend():
    """MACD histogram is higher for an accelerating uptrend than a downtrend."""
    idx = _idx(80)
    t = np.arange(80.0)
    up = pd.Series(100.0 * np.exp(0.01 * t), index=idx)
    down = pd.Series(100.0 * np.exp(-0.01 * t), index=idx)
    close = pd.DataFrame({"UP": up, "DOWN": down})
    raw = macd_raw(close, fast=12, slow=26, signal_period=9)
    assert raw["UP"].iloc[-1] > raw["DOWN"].iloc[-1]


def test_vbb_raw_higher_below_lower_band():
    """vbb_raw: a price below its lower band ranks ABOVE one at its mean."""
    idx = _idx(40)
    np.random.seed(2)
    base = 100.0 + np.random.normal(0, 0.5, 40)
    mid = pd.Series(base, index=idx)
    low = mid.copy()
    low.iloc[-1] = base[-1] - 10.0
    close = pd.DataFrame({"MID": mid, "LOW": low})
    vol = pd.DataFrame(5000.0, index=idx, columns=close.columns)
    raw = vbb_raw(close, vol, period=20, std=2.0, vol_period=20)
    assert np.isfinite(raw["MID"].iloc[-1])
    assert np.isfinite(raw["LOW"].iloc[-1])
    assert raw["LOW"].iloc[-1] > raw["MID"].iloc[-1]
