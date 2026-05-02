"""Stock-specific regime masks and correlation logic."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import pandas as pd
import yfinance as yf


def _compute_spy_regime_mask(
    ohlcv: pd.DataFrame,
    config: Dict[str, Any],
) -> pd.Series | None:
    """Compute a boolean SPY EMA(n) regime mask aligned to ohlcv.index.

    Fetches SPY data via yfinance if not present in ohlcv.
    True = SPY above its EMA (bull market).
    """
    logger = logging.getLogger("ggTraderLive")
    bench_symbol = config.get("BENCHMARK_SYMBOL", "SPY")
    n_warmup = int(config.get("EMA_WARMUP_BARS", 200))
    
    spy_series: pd.Series | None = None
    
    # 1. Try to get from ohlcv if present
    if bench_symbol in ohlcv.columns.get_level_values(0):
        spy_series = ohlcv[bench_symbol]["close"]
        
    # 2. Fall back to yfinance fetch
    if spy_series is None or len(spy_series) < n_warmup:
        try:
            start_date = ohlcv.index[0] - pd.Timedelta(days=n_warmup * 2)
            end_date = ohlcv.index[-1]
            spy_df = yf.download(bench_symbol, start=start_date, end=end_date, interval="1d", auto_adjust=True, progress=False)
            if not spy_df.empty:
                spy_series = spy_df["Close"]
                # Convert index to UTC to match ohlcv
                if spy_series.index.tz is None:
                    spy_series.index = spy_series.index.tz_localize("UTC")
                else:
                    spy_series.index = spy_series.index.tz_convert("UTC")
        except Exception as e:
            logger.error(f"Failed to fetch SPY for regime filter: {e!r}")

    if spy_series is None or spy_series.empty:
        return None

    # 3. Compute EMA
    spy_ema = spy_series.ewm(span=n_warmup, adjust=False).mean()
    
    # 4. Determine regime
    short_span = config.get("SPY_REGIME_FILTER_SHORT_EMA")
    if short_span:
        spy_ema_short = spy_series.ewm(span=int(short_span), adjust=False).mean()
        regime_raw = spy_ema_short > spy_ema
    else:
        regime_raw = spy_series > spy_ema
        
    # 5. Reindex to match original ohlcv
    return regime_raw.reindex(ohlcv.index, method="ffill").fillna(False)


def _compute_vix_regime_mask(
    ohlcv: pd.DataFrame,
    config: Dict[str, Any],
) -> pd.Series | None:
    """Compute a boolean VIX regime mask aligned to ohlcv.index.

    True = VIX < threshold (low volatility = bullish/stable).
    """
    logger = logging.getLogger("ggTraderLive")
    vix_symbol = "^VIX"
    threshold = config.get("VIX_REGIME_THRESHOLD", 25)
    
    try:
        start_date = ohlcv.index[0] - pd.Timedelta(days=10)
        end_date = ohlcv.index[-1]
        vix_df = yf.download(vix_symbol, start=start_date, end=end_date, interval="1d", auto_adjust=True, progress=False)
        if vix_df.empty:
            return None
        
        vix_series = vix_df["Close"]
        if vix_series.index.tz is None:
            vix_series.index = vix_series.index.tz_localize("UTC")
        else:
            vix_series.index = vix_series.index.tz_convert("UTC")
            
        regime_raw = vix_series < threshold
        return regime_raw.reindex(ohlcv.index, method="ffill").fillna(False)
    except Exception as e:
        logger.error(f"Failed to fetch VIX for regime filter: {e!r}")
        return None


def _compute_spy_correlations(ohlcv: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, float]:
    """Return per-stock correlation of daily returns vs SPY."""
    bench_symbol = config.get("BENCHMARK_SYMBOL", "SPY")
    try:
        # Extract closing prices
        close = ohlcv.xs("close", axis=1, level=1)
        
        # Ensure SPY is in the data
        if bench_symbol not in close.columns:
            # We would need to fetch SPY here to compute correlations if it's missing
            # For now return empty dict, meaning no tiered filtering
            return {}
            
        returns = close.pct_change()
        spy_ret = returns[bench_symbol]
        corr = returns.corrwith(spy_ret)
        return corr.to_dict()
    except Exception:
        return {}
