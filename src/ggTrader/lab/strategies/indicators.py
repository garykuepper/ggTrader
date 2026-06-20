# src/ggTrader/lab/strategies/indicators.py
"""Shared vectorized indicator functions used by multiple signal strategies."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def extract_close(data: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
    """Extract a (time x symbol) close-price DataFrame from multi-level OHLCV data."""
    have = set(data.columns.get_level_values(0))
    return pd.concat({s: data[s]["close"] for s in symbols if s in have}, axis=1)


def eligible_symbols(data: pd.DataFrame, eligible: List[str], min_history_bars: int) -> List[str]:
    """Filter eligible symbols to those present in data with enough history."""
    have = set(data.columns.get_level_values(0).unique())
    return [s for s in eligible if s in have and len(data[s]["close"].dropna()) >= min_history_bars]


def bb_signals(close: pd.DataFrame, period: int, std: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Vectorized Bollinger Band entry/exit signals."""
    sma = close.rolling(window=period, min_periods=period).mean()
    rolling_std = close.rolling(window=period, min_periods=period).std()
    lower = sma - std * rolling_std

    prev_above = close.shift(1) >= lower.shift(1)
    now_below = close < lower
    entries = (prev_above & now_below).fillna(False).astype(bool)

    prev_below = close.shift(1) < sma.shift(1)
    now_above = close >= sma
    exits = (prev_below & now_above).fillna(False).astype(bool)

    return entries, exits


def rsi_signals(
    close: pd.DataFrame, period: int, oversold: int, exit_level: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Vectorized RSI entry/exit signals."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    prev_above = rsi.shift(1) >= oversold
    now_below = rsi < oversold
    entries = (prev_above & now_below).fillna(False).astype(bool)

    prev_below = rsi.shift(1) < exit_level
    now_above = rsi >= exit_level
    exits = (prev_below & now_above).fillna(False).astype(bool)

    return entries, exits


def ema_signals(
    close: pd.DataFrame, ema_fast: int, ema_slow: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """EMA crossover entry/exit signals."""
    ema_f = close.ewm(span=ema_fast, adjust=False).mean()
    ema_s = close.ewm(span=ema_slow, adjust=False).mean()
    entries = ((ema_f > ema_s) & (ema_f.shift(1) <= ema_s.shift(1))).fillna(False)
    exits = ((ema_f < ema_s) & (ema_f.shift(1) >= ema_s.shift(1))).fillna(False)
    return entries.astype(bool), exits.astype(bool)
