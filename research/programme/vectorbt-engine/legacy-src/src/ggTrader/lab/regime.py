"""Lookahead-safe market-regime classifier (SPY trend + volatility)."""

from __future__ import annotations

import numpy as np
import pandas as pd

TREND_UP = "up"
TREND_DOWN = "down"
VOL_BUCKETS = ("calm", "normal", "turbulent")


def classify_regime(
    spy_close: pd.Series,
    ema_period: int = 200,
    vol_lookback: int = 20,
    vol_window: int = 252,
) -> pd.DataFrame:
    """SPY close -> per-bar regime (trend x vol_bucket). All values lagged 1 bar.

    trend: close vs its EMA(ema_period). vol_bucket: trailing realized vol
    (vol_lookback) bucketed by its own expanding 33/66 percentiles over the
    last vol_window bars (causal). Everything shifted 1 bar so the label at t
    uses only data through t-1's close -> usable to size the t entry.
    """
    ema = spy_close.ewm(span=ema_period, adjust=False).mean()
    trend = pd.Series(np.where(spy_close > ema, TREND_UP, TREND_DOWN), index=spy_close.index)

    rets = spy_close.pct_change(fill_method=None)
    realized = rets.rolling(vol_lookback, min_periods=vol_lookback).std() * np.sqrt(252)
    lo = realized.rolling(vol_window, min_periods=vol_lookback).quantile(0.33)
    hi = realized.rolling(vol_window, min_periods=vol_lookback).quantile(0.66)
    vol_bucket = pd.Series("normal", index=spy_close.index, dtype=object)
    vol_bucket[realized <= lo] = "calm"
    vol_bucket[realized >= hi] = "turbulent"

    out = pd.DataFrame({"trend": trend, "vol_bucket": vol_bucket}, index=spy_close.index).shift(1)
    out["label"] = out["trend"].str.cat(out["vol_bucket"], sep="_")
    return out


def compute_regime_scalar(
    spy_close: pd.Series,
    scalar_map: dict[str, float],
    ema_period: int = 200,
    vol_lookback: int = 20,
    vol_window: int = 252,
    default: float = 1.0,
) -> pd.Series:
    """Per-bar exposure multiplier from regime label via scalar_map."""
    reg = classify_regime(spy_close, ema_period, vol_lookback, vol_window)
    return reg["label"].map(scalar_map).astype(float).fillna(default)
