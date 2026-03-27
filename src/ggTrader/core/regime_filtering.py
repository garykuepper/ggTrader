"""BTC regime masks and altcoin index filtering."""

from typing import Any, Dict

import pandas as pd


def _compute_btc_regime_mask(
    ohlcv: pd.DataFrame,
    config: Dict[str, Any],
) -> "pd.Series | None":
    """Compute a boolean BTC EMA(n) regime mask aligned to ``ohlcv.index``.

    Fetches ``EMA_WARMUP_BARS`` extra bars before the first index timestamp so
    the EMA is fully warm from bar 0 of the actual data range.  Returns a
    boolean Series (True = BTC above EMA = bull regime) or None on failure.
    """
    bench_symbol = config.get("BENCHMARK_SYMBOL", "BTC-USD")
    n_warmup = int(config.get("EMA_WARMUP_BARS", 200))
    interval_str = config.get("INTERVAL", "4h")
    try:
        interval_hours = int(interval_str.rstrip("h"))
    except ValueError:
        interval_hours = 4
    warmup_td = pd.Timedelta(hours=interval_hours * n_warmup)

    # Try to get BTC close from the passed ohlcv first (fastest path).
    btc_series: "pd.Series | None" = None
    if bench_symbol in ohlcv.columns.get_level_values(0):
        raw = ohlcv[[bench_symbol]].xs("close", axis=1, level=1, drop_level=True)
        btc_series = raw.iloc[:, 0]

    # Fall back to DB fetch with warmup extension.
    data_start = pd.to_datetime(ohlcv.index[0])
    data_end = pd.to_datetime(ohlcv.index[-1])
    if data_start.tz is None:
        data_start = data_start.tz_localize("UTC")
    if data_end.tz is None:
        data_end = data_end.tz_localize("UTC")
    warmup_start = data_start - warmup_td

    try:
        from ggTrader.data.historical.timescaledb_loader import TimescaleDBLoader
        loader = TimescaleDBLoader()
        btc_ohlcv = loader.fetch_ohlcv(
            symbols=[bench_symbol],
            interval=interval_str,
            start_date=warmup_start,
            end_date=data_end,
        )
        if not btc_ohlcv.empty:
            raw_db = btc_ohlcv.xs("close", axis=1, level=1, drop_level=True).iloc[:, 0]
            # Use DB data (has warmup prepended); fall back to ohlcv-embedded above only
            # if the DB fetch is shorter than what we already have.
            if len(raw_db) >= len(btc_series or []):
                btc_series = raw_db
    except Exception as _e:
        if btc_series is None:
            print(f"  [BTC Regime] WARNING: failed to load {bench_symbol} from DB: {_e}")

    if btc_series is None or btc_series.dropna().empty:
        return None

    # Compute EMA over the full (warmup-extended) series, then trim to ohlcv.index.
    btc_ema = btc_series.ewm(span=n_warmup, adjust=False).mean()
    regime_raw = btc_series > btc_ema

    # Normalize timezone so reindex matches ohlcv.index (which may be tz-naive).
    ohlcv_tz = ohlcv.index.tz
    regime_tz = regime_raw.index.tz
    if ohlcv_tz is None and regime_tz is not None:
        regime_raw = regime_raw.tz_convert("UTC").tz_localize(None)
    elif ohlcv_tz is not None and regime_tz is None:
        regime_raw = regime_raw.tz_localize("UTC").tz_convert(ohlcv_tz)

    regime_bull = regime_raw.reindex(ohlcv.index, fill_value=False)
    return regime_bull


def _compute_btc_correlations(ohlcv: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, float]:
    """Return per-symbol Pearson correlation of 4h returns vs BTC-USD returns.

    Used to decide which coins should have the BTC regime filter applied.
    Coins below BTC_REGIME_FILTER_MIN_CORRELATION use the altcoin index filter or trade freely.
    """
    bench_symbol = config.get("BENCHMARK_SYMBOL", "BTC-USD")
    try:
        close = ohlcv.xs("close", axis=1, level=1)
        close = close[[c for c in close.columns if str(c).endswith("-USD")]]
        if bench_symbol not in close.columns:
            return {}
        returns = close.pct_change()
        btc_ret = returns[bench_symbol]
        corr = returns.corrwith(btc_ret)
        return corr.to_dict()
    except Exception:
        return {}


def _compute_altcoin_index_mask(
    ohlcv: pd.DataFrame,
    config: Dict[str, Any],
) -> "pd.Series | None":
    """Compute a boolean regime mask based on an equal-weighted altcoin index EMA.

    Builds an equal-weighted price index from all coins in ohlcv excluding BTC-USD,
    normalised to start at 1.0. Applies EMA(EMA_WARMUP_BARS) and returns True where
    the index is above its EMA (bull regime). Used for mid-correlation coins that
    move more with the broader altcoin market than with BTC specifically.
    """
    bench_symbol = config.get("BENCHMARK_SYMBOL", "BTC-USD")
    n_warmup = int(config.get("EMA_WARMUP_BARS", 200))
    try:
        close = ohlcv.xs("close", axis=1, level=1)
        # Exclude BTC from the index — it has its own filter
        alt_cols = [c for c in close.columns if str(c).endswith("-USD") and c != bench_symbol]
        if not alt_cols:
            return None
        # Equal-weighted: normalise each coin to 1.0 at first valid bar, then average
        normed = close[alt_cols].div(close[alt_cols].bfill().iloc[0])
        alt_index = normed.mean(axis=1)
        alt_ema = alt_index.ewm(span=n_warmup, adjust=False).mean()
        regime = alt_index > alt_ema
        return regime
    except Exception as e:
        print(f"  [Altcoin Index] WARNING: failed to compute index: {e}")
        return None
