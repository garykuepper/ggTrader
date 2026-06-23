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


def bb_strength(close: pd.DataFrame, period: int, std: float) -> pd.DataFrame:
    """Normalized depth below the lower Bollinger Band, clipped to [0, 1]."""
    sma = close.rolling(window=period, min_periods=period).mean()
    rolling_std = close.rolling(window=period, min_periods=period).std()
    band_width = std * rolling_std
    depth = (sma - std * rolling_std - close) / band_width.replace(0, np.nan)
    return depth.clip(lower=0.0, upper=1.0)


def rsi_strength(close: pd.DataFrame, period: int, oversold: int) -> pd.DataFrame:
    """Normalized RSI depth below oversold threshold, clipped to [0, 1]."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    depth = (oversold - rsi) / oversold
    return depth.clip(lower=0.0, upper=1.0)


def ema_strength(close: pd.DataFrame, ema_fast: int, ema_slow: int) -> pd.DataFrame:
    """Normalized EMA gap (fast above slow) / slow, clipped to [0, 1]."""
    ema_f = close.ewm(span=ema_fast, adjust=False).mean()
    ema_s = close.ewm(span=ema_slow, adjust=False).mean()
    gap = (ema_f - ema_s) / ema_s.replace(0, np.nan)
    return gap.clip(lower=0.0, upper=1.0)


def macd_signals(
    close: pd.DataFrame,
    fast: int,
    slow: int,
    signal_period: int,
    divergence_window: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """MACD bullish divergence entry + histogram-crosses-zero exit."""
    macd_line = (
        close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    )
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line

    price_low = close.rolling(window=divergence_window, min_periods=divergence_window).min()
    hist_low = histogram.rolling(window=divergence_window, min_periods=divergence_window).min()

    price_at_low = close <= price_low
    hist_above_low = histogram > hist_low
    entries = (price_at_low & hist_above_low).fillna(False).astype(bool)

    # Mask out warmup period (slow + signal + divergence_window bars)
    warmup = slow + signal_period + divergence_window
    entries = entries.copy()
    entries.iloc[:warmup] = False

    prev_above = histogram.shift(1) >= 0
    now_below = histogram < 0
    exits = (prev_above & now_below).fillna(False).astype(bool)

    return entries, exits


def macd_strength(close: pd.DataFrame, fast: int, slow: int, signal_period: int) -> pd.DataFrame:
    """Normalized absolute MACD histogram magnitude, clipped to [0, 1]."""
    macd_line = (
        close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    )
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    hist_max = histogram.abs().rolling(window=50, min_periods=1).max()
    strength = histogram.abs() / hist_max.replace(0, np.nan)
    strength = strength.clip(lower=0.0, upper=1.0)
    # Mask out warmup period (slow bars) — set to NaN
    strength = strength.copy()
    strength.iloc[:slow] = np.nan
    return strength


def extract_volume(data: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
    """Extract a (time x symbol) volume DataFrame from multi-level OHLCV data."""
    have = set(data.columns.get_level_values(0))
    return pd.concat({s: data[s]["volume"] for s in symbols if s in have}, axis=1)


def volume_bb_signals(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    bb_period: int,
    bb_std: float,
    vol_period: int,
    vol_mult: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """BB reversion entry gated by a volume spike above vol_mult * SMA(volume)."""
    sma = close.rolling(window=bb_period, min_periods=bb_period).mean()
    rolling_std = close.rolling(window=bb_period, min_periods=bb_period).std()
    lower = sma - bb_std * rolling_std

    prev_above = close.shift(1) >= lower.shift(1)
    now_below = close < lower
    bb_entry = (prev_above & now_below).fillna(False)

    vol_avg = volume.rolling(window=vol_period, min_periods=vol_period).mean()
    vol_spike = volume > (vol_mult * vol_avg)

    entries = (bb_entry & vol_spike).fillna(False).astype(bool)

    prev_below_sma = close.shift(1) < sma.shift(1)
    now_above_sma = close >= sma
    exits = (prev_below_sma & now_above_sma).fillna(False).astype(bool)

    return entries, exits


def volume_bb_strength(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    bb_period: int,
    bb_std: float,
    vol_period: int,
) -> pd.DataFrame:
    """BB depth strength scaled by volume spike intensity, clipped to [0, 1]."""
    bb_str = bb_strength(close, bb_period, bb_std)
    vol_avg = volume.rolling(window=vol_period, min_periods=vol_period).mean()
    vol_ratio = (volume / vol_avg.replace(0, np.nan)).clip(upper=3.0) / 3.0
    combined = (bb_str + vol_ratio) / 2.0
    return combined.clip(lower=0.0, upper=1.0)


def _weekly_rsi(close: pd.DataFrame, period: int) -> pd.DataFrame:
    """Compute RSI on weekly-resampled close, forward-filled to daily index."""
    weekly = close.resample("W").last().dropna(how="all")
    delta = weekly.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.reindex(close.index, method="ffill")


def mtf_signals(
    close: pd.DataFrame,
    weekly_rsi_period: int,
    weekly_rsi_oversold: int,
    weekly_rsi_exit: int,
    daily_bb_period: int,
    daily_bb_std: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Multi-timeframe reversion: weekly RSI oversold + daily BB breakdown."""
    w_rsi = _weekly_rsi(close, weekly_rsi_period)

    sma = close.rolling(window=daily_bb_period, min_periods=daily_bb_period).mean()
    rolling_std = close.rolling(window=daily_bb_period, min_periods=daily_bb_period).std()
    lower = sma - daily_bb_std * rolling_std

    weekly_oversold = w_rsi < weekly_rsi_oversold
    daily_below_bb = close < lower

    entries = (weekly_oversold & daily_below_bb).fillna(False).astype(bool)

    weekly_recovered = w_rsi >= weekly_rsi_exit
    prev_weekly_below = w_rsi.shift(1) < weekly_rsi_exit
    weekly_exit = (weekly_recovered & prev_weekly_below).fillna(False)

    prev_below_sma = close.shift(1) < sma.shift(1)
    now_above_sma = close >= sma
    daily_exit = (prev_below_sma & now_above_sma).fillna(False)

    exits = (weekly_exit | daily_exit).fillna(False).astype(bool)

    return entries, exits


def mtf_strength(
    close: pd.DataFrame,
    weekly_rsi_period: int,
    weekly_rsi_oversold: int,
    daily_bb_period: int,
    daily_bb_std: float,
) -> pd.DataFrame:
    """Average of weekly RSI depth and daily BB depth, clipped to [0, 1]."""
    w_rsi = _weekly_rsi(close, weekly_rsi_period)
    rsi_depth = ((weekly_rsi_oversold - w_rsi) / weekly_rsi_oversold).clip(lower=0.0, upper=1.0)
    bb_str = bb_strength(close, daily_bb_period, daily_bb_std)
    combined = (rsi_depth + bb_str) / 2.0
    return combined.clip(lower=0.0, upper=1.0)
