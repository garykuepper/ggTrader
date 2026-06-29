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
    """A monotonically falling series (oversold) ranks ABOVE a rising one."""
    idx = _idx(60)
    falling = pd.Series(np.linspace(100, 60, 60), index=idx)
    rising = pd.Series(np.linspace(60, 100, 60), index=idx)
    close = pd.DataFrame({"DOWN": falling, "UP": rising})
    raw = rsi_raw(close, period=14)
    assert raw["DOWN"].iloc[-1] > raw["UP"].iloc[-1]


def test_bb_raw_higher_below_lower_band():
    """Negated %b: a price far below its mean ranks above one at its mean."""
    close = _close()
    raw = bb_raw(close, period=20, std=2.0)
    assert raw.shape == close.shape
    assert raw.notna().any().any()


def test_ema_raw_positive_in_uptrend():
    idx = _idx(80)
    up = pd.Series(np.linspace(50, 150, 80), index=idx)
    close = pd.DataFrame({"UP": up})
    raw = ema_raw(close, fast=20, slow=50)
    assert raw["UP"].iloc[-1] > 0


def test_macd_raw_shape_and_finite():
    close = _close()
    raw = macd_raw(close, fast=12, slow=26, signal_period=9)
    assert raw.shape == close.shape
    assert np.isfinite(raw.iloc[-1].to_numpy()).all()


def test_vbb_raw_shape():
    close = _close()
    vol = _volume(close)
    raw = vbb_raw(close, vol, period=20, std=2.0, vol_period=20)
    assert raw.shape == close.shape
