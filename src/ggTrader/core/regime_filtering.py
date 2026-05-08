"""BTC leader-regime mask and per-coin correlations.

Coins whose return correlation to BTC clears ``LEADER_CORR_THRESHOLD`` are
gated by the BTC EMA regime; below the threshold they trade freely. The
2-tier logic lives in ``orchestrator._apply_tiered_regime_mask``; this module
produces the underlying mask and correlations. Generic ``_compute_leader_*``
helpers are kept so research tools (e.g. correlation matrix scripts) can
reuse them for arbitrary leader symbols.
"""

from typing import Any, Dict

import pandas as pd


def _compute_leader_regime_mask(
    ohlcv: pd.DataFrame,
    config: Dict[str, Any],
    leader_symbol: str,
    short_ema_key: str,
) -> "pd.Series | None":
    """Compute a boolean ``close > EMA(n)`` regime mask for any leader symbol.

    Fetches ``EMA_WARMUP_BARS`` extra bars before the first index timestamp so
    the EMA is fully warm. ``short_ema_key`` is the config key naming the
    optional short-EMA span (e.g., ``BTC_REGIME_FILTER_SHORT_EMA``); when
    that's set, the mask is ``ema_short > ema_long`` instead of ``close > ema_long``.
    Returns a bool Series aligned to ``ohlcv.index`` or ``None`` on failure.
    """
    n_warmup = int(config.get("EMA_WARMUP_BARS", 200))
    interval_str = config.get("INTERVAL", "4h")
    try:
        interval_hours = int(interval_str.rstrip("h"))
    except ValueError:
        interval_hours = 4
    warmup_td = pd.Timedelta(hours=interval_hours * n_warmup)

    # Try to get the leader close from the passed ohlcv first (fastest path).
    series: "pd.Series | None" = None
    if leader_symbol in ohlcv.columns.get_level_values(0):
        raw = ohlcv[[leader_symbol]].xs("close", axis=1, level=1, drop_level=True)
        series = raw.iloc[:, 0]

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
        leader_ohlcv = loader.fetch_ohlcv(
            symbols=[leader_symbol],
            interval=interval_str,
            start_date=warmup_start,
            end_date=data_end,
        )
        if not leader_ohlcv.empty:
            raw_db = leader_ohlcv.xs("close", axis=1, level=1, drop_level=True).iloc[:, 0]
            # Use DB data (has warmup prepended); fall back to ohlcv-embedded above only
            # if the DB fetch is shorter than what we already have.
            if len(raw_db) >= len(series or []):
                series = raw_db
    except Exception as _e:
        if series is None:
            print(f"  [{leader_symbol} Regime] WARNING: failed to load from DB: {_e}")

    if series is None or series.dropna().empty:
        return None

    ema_long = series.ewm(span=n_warmup, adjust=False).mean()
    short_span = config.get(short_ema_key, None)
    if short_span is not None:
        ema_short = series.ewm(span=int(short_span), adjust=False).mean()
        regime_raw = ema_short > ema_long
    else:
        regime_raw = series > ema_long

    # Normalize timezone so reindex matches ohlcv.index.
    ohlcv_tz = ohlcv.index.tz
    regime_tz = regime_raw.index.tz
    if ohlcv_tz is None and regime_tz is not None:
        regime_raw = regime_raw.tz_convert("UTC").tz_localize(None)
    elif ohlcv_tz is not None and regime_tz is None:
        regime_raw = regime_raw.tz_localize("UTC").tz_convert(ohlcv_tz)

    return regime_raw.reindex(ohlcv.index, fill_value=False)


def _compute_btc_regime_mask(
    ohlcv: pd.DataFrame,
    config: Dict[str, Any],
) -> "pd.Series | None":
    """BTC EMA regime mask. Uses ``BENCHMARK_SYMBOL`` (default BTC-USD) and
    ``BTC_REGIME_FILTER_SHORT_EMA``."""
    return _compute_leader_regime_mask(
        ohlcv, config,
        leader_symbol=config.get("BENCHMARK_SYMBOL", "BTC-USD"),
        short_ema_key="BTC_REGIME_FILTER_SHORT_EMA",
    )


def _compute_leader_correlations(
    ohlcv: pd.DataFrame,
    leader_symbol: str,
) -> Dict[str, float]:
    """Per-symbol Pearson correlation of returns vs ``leader_symbol`` returns."""
    try:
        close = ohlcv.xs("close", axis=1, level=1)
        close = close[[c for c in close.columns if str(c).endswith("-USD")]]
        if leader_symbol not in close.columns:
            return {}
        returns = close.pct_change()
        leader_ret = returns[leader_symbol]
        return returns.corrwith(leader_ret).to_dict()
    except Exception:
        return {}


def _compute_btc_correlations(ohlcv: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, float]:
    """Per-symbol correlation to BTC-USD returns."""
    return _compute_leader_correlations(
        ohlcv, config.get("BENCHMARK_SYMBOL", "BTC-USD")
    )


