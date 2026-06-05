"""OHLCV loading from TimescaleDB (and optional CCXT tail for recent validation)."""

from __future__ import annotations

import os
from typing import Any, List

import pandas as pd

from ggTrader.data.historical.timescaledb_loader import TimescaleDBLoader
from ggTrader.utils.config import load_symbols_from_json

# CCXT normalises Kraken markets to standard symbols (BTC/USD, DOGE/USD, etc.);
# no base aliasing is needed when calling exchange.fetch_ohlcv via CCXT.
_KRAKEN_BASE_ALIASES: dict[str, str] = {}


def _resolve_exchange_id() -> str:
    """Research data venue, from the EXCHANGE env var (matches the live trader and the
    universe fetch). Defaults to 'kraken' so legacy behavior is unchanged when unset.
    This is what aligns WFO backtests to the venue we actually execute on."""
    return (os.getenv("EXCHANGE") or "kraken").lower()


def _venue_for_exchange(exchange_id: str) -> str:
    """Map a CCXT exchange id to the DB `venue` string (mirrors CachedExchangeLoader.venue)."""
    return "binanceus_spot" if exchange_id == "binanceus" else "kraken_spot"


def load_data_and_setup(config: dict) -> pd.DataFrame:
    """
    Loads symbols and fetches 4h OHLCV data from TimescaleDBLoader.

    Args:
        config (dict): Configuration dictionary containing:
            - 'SYMBOLS': List of symbol strings (optional).
            - 'SYMBOLS_FILE': Path to JSON file with symbols (optional).
            - 'INTERVAL': Time interval (e.g., '4h').
            - 'START_DATE': Start date string.
            - 'END_DATE': End date string.

    Returns:
        pd.DataFrame: A DataFrame containing the OHLCV data for the requested symbols.
    """
    # Priority: Direct SYMBOLS list -> SYMBOLS_FILE JSON
    if "SYMBOLS" in config and config["SYMBOLS"]:
        symbols = config["SYMBOLS"]
    elif "SYMBOLS_FILE" in config and config["SYMBOLS_FILE"]:
        symbols = load_symbols_from_json(config["SYMBOLS_FILE"])
        if symbols is None:
            raise ValueError(f"Symbols file '{config['SYMBOLS_FILE']}' not found or invalid.")
    else:
        raise ValueError("Config must contain 'SYMBOLS' (list) or 'SYMBOLS_FILE' (path).")

    if not symbols:
        raise ValueError("No symbols provided or symbols list is empty.")

    from ggTrader.data.live.cached_loader import CachedExchangeLoader

    # Read/refresh from the venue we trade on (binanceus_spot), not the kraken default.
    # The cached loader auto-tails fresh bars from the matching exchange and persists them.
    loader = CachedExchangeLoader(exchange_id=_resolve_exchange_id())

    ohlcv_df = loader.fetch_ohlcv(
        symbols=symbols,
        interval=config["INTERVAL"],
        start_date=pd.to_datetime(config["START_DATE"]).tz_localize("UTC"),
        end_date=pd.to_datetime(config["END_DATE"]).tz_localize("UTC"),
        limit=None,
    )

    if ohlcv_df.empty:
        raise ValueError(
            f"No OHLCV data returned from database.\n"
            f"  Symbols requested: {symbols[:5]}{'...' if len(symbols) > 5 else ''}\n"
            f"  Date range: {config['START_DATE']} to {config['END_DATE']}\n"
            f"  Interval: {config['INTERVAL']}\n"
            f"  Check that the database is running and contains data for these symbols."
        )

    # Ensure all requested symbols exist in the dataframe as top-level columns.
    # New coins with short histories (e.g. SUI) might be completely missing
    # depending on the requested date range. Padding them with NaNs prevents KeyErrors.
    existing_symbols = set(ohlcv_df.columns.get_level_values(0))
    missing_symbols = [s for s in symbols if s not in existing_symbols]

    if missing_symbols:
        metrics = ["open", "high", "low", "close", "volume", "trades"]
        # Only use metrics that actually exist in the dataframe
        actual_metrics = [m for m in metrics if m in ohlcv_df.columns.get_level_values(1)]

        # Create a MultiIndex of missing combinations
        import itertools

        missing_tuples = list(itertools.product(missing_symbols, actual_metrics))
        missing_idx = pd.MultiIndex.from_tuples(missing_tuples)

        # Create an empty dataframe with NaNs and concatenate
        empty_df = pd.DataFrame(index=ohlcv_df.index, columns=missing_idx, dtype=float)
        ohlcv_df = pd.concat([ohlcv_df, empty_df], axis=1)

        # Sort columns to maintain consistency
        ohlcv_df.sort_index(axis=1, inplace=True)

    return ohlcv_df


def build_mover_mask(
    ohlcv_df: pd.DataFrame,
    config: dict,
    top_n: int = 20,
) -> pd.DataFrame:
    """Build a boolean mask of daily top-N movers aligned to the OHLCV index.

    The DB is queried once for daily rankings, then forward-filled to match
    the intraday frequency of the OHLCV data (e.g. 4h bars).
    """
    start = pd.to_datetime(config["START_DATE"]).tz_localize("UTC")
    end = pd.to_datetime(config["END_DATE"]).tz_localize("UTC")

    loader = TimescaleDBLoader()
    daily_mask = loader.get_daily_mover_mask(
        start=start, end=end, top_n=top_n, venue=_venue_for_exchange(_resolve_exchange_id())
    )

    if daily_mask.empty:
        raise ValueError("No mover data returned from the database.")

    # Align symbols: only keep columns present in the OHLCV data
    ohlcv_symbols = ohlcv_df.columns.get_level_values(0).unique()
    common = daily_mask.columns.intersection(ohlcv_symbols)
    daily_mask = daily_mask[common]

    # Reindex to the intraday OHLCV index and forward-fill within each day
    mask = daily_mask.reindex(ohlcv_df.index).ffill().fillna(False)
    return mask.astype(bool)


from typing import Optional, Tuple  # noqa: E402


def load_data_with_movers(config: dict) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Loads data and optionally builds a mover mask based on config['USE_MOVERS'].

    Args:
        config: Configuration dictionary.

    Returns:
        Tuple of (ohlcv DataFrame, mover_mask DataFrame or None)
    """
    print("Loading data...")
    ohlcv = load_data_and_setup(config)

    mover_mask = None
    use_movers = config.get("USE_MOVERS", 0)
    if use_movers > 0:
        print(f"Building dynamic top-{use_movers} mover mask...")
        try:
            mover_mask = build_mover_mask(ohlcv, config, top_n=use_movers)
        except Exception as e:
            print(f"Warning: mover mask build failed: {e}")

    return ohlcv, mover_mask


def _symbol_bases_from_config(config: dict) -> List[str]:
    """Resolve base asset names (e.g. BTC) for CCXT from prepared ``config['SYMBOLS']``."""
    if "SYMBOLS" in config and config["SYMBOLS"]:
        raw = list(config["SYMBOLS"])
    elif "SYMBOLS_FILE" in config and config["SYMBOLS_FILE"]:
        loaded = load_symbols_from_json(config["SYMBOLS_FILE"]) or []
        max_sym = int(config.get("MAX_SYMBOLS") or len(loaded))
        raw = loaded[:max_sym]
    else:
        raise ValueError("Config must contain SYMBOLS or SYMBOLS_FILE for hybrid load.")
    bases: List[str] = []
    for s in raw:
        if not s:
            continue
        s = str(s).strip()
        if "/" in s:
            bases.append(s.split("/")[0].strip())
        elif "-" in s:
            bases.append(s.split("-")[0].strip())
        else:
            bases.append(s)
    return bases


def _tsdb_column_key(base: str, quote: str = "USD") -> str:
    """Match TimescaleDBLoader pivot column names (e.g. BTC-USD)."""
    b = base.upper()
    return f"{b}-{quote}"


def _ccxt_pair_for_exchange(base: str, quote: str, exchange: Any) -> str:
    """Return the CCXT pair string for a base symbol.

    Applies exchange-specific base aliases (e.g. Kraken uses XBT instead of BTC,
    XDG instead of DOGE).  For all other exchanges the pair is ``base/quote``.
    """
    exchange_id = getattr(exchange, "id", "") if exchange is not None else ""
    if exchange_id == "kraken":
        base = _KRAKEN_BASE_ALIASES.get(base.upper(), base)
    return f"{base}/{quote}"


def _ccxt_fetch_paginated(
    exchange,
    pair: str,
    interval: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    page_limit: int = 1000,
) -> pd.DataFrame:
    """Fetch OHLCV from ``exchange`` in pages until ``end`` (inclusive window)."""
    rows: list = []
    since_ms = int(pd.Timestamp(start).timestamp() * 1000)
    end_ms = int(pd.Timestamp(end).timestamp() * 1000)
    while since_ms <= end_ms:
        batch = exchange.fetch_ohlcv(pair, timeframe=interval, since=since_ms, limit=page_limit)
        if not batch:
            break
        rows.extend(batch)
        last_ts = batch[-1][0]
        if last_ts <= since_ms:
            break
        since_ms = last_ts + 1
        if last_ts >= end_ms:
            break
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("datetime").drop(columns=["timestamp"])
    df = df[(df.index >= start.tz_convert("UTC")) & (df.index <= end.tz_convert("UTC"))]
    return df


def load_hybrid_validation_ohlcv(
    config: dict,
    validation_start: pd.Timestamp,
    validation_end: pd.Timestamp,
    *,
    use_ccxt_tail: bool = False,
    quote: str = "USD",
) -> pd.DataFrame:
    """
    Load OHLCV for ``[validation_start, validation_end]`` from TimescaleDB.

    If ``use_ccxt_tail`` is True, append bars from the configured exchange (EXCHANGE env;
    binanceus when live on Binance.US) via CCXT from the latest DB timestamp through
    ``validation_end`` (deduped; CCXT wins on overlap). Column names match TimescaleDB
    (e.g. BTC-USD).
    """
    exchange_id = _resolve_exchange_id()
    bases = _symbol_bases_from_config(config)
    if not bases:
        raise ValueError("No symbols resolved for hybrid validation load.")

    ts_start = validation_start
    if ts_start.tz is None:
        ts_start = pd.Timestamp(ts_start).tz_localize("UTC")
    else:
        ts_start = ts_start.tz_convert("UTC")

    ts_end = validation_end
    if ts_end.tz is None:
        ts_end = pd.Timestamp(ts_end).tz_localize("UTC")
    else:
        ts_end = ts_end.tz_convert("UTC")

    formatted = [b if "-" in b or "/" in b else f"{b}-{quote}" for b in bases]

    loader = TimescaleDBLoader()
    tsdb = loader.fetch_ohlcv(
        symbols=formatted,
        interval=config["INTERVAL"],
        start_date=ts_start,
        end_date=ts_end,
        venue=_venue_for_exchange(exchange_id),
    )

    if not use_ccxt_tail:
        if tsdb.empty:
            raise ValueError(
                f"No TimescaleDB OHLCV for recent validation window {ts_start} .. {ts_end}."
            )
        return tsdb.sort_index()

    from ggTrader.data.live.cached_loader import CachedExchangeLoader

    cache_loader = CachedExchangeLoader(exchange_id=exchange_id)

    try:
        combined = cache_loader.fetch_ohlcv(
            symbols=formatted,
            interval=config["INTERVAL"],
            start_date=ts_start,
            end_date=ts_end,
            quote=quote,
            limit=None,
        )
    except Exception as e:
        print(f"Warning: Cached loader failed: {e}. Falling back to DB-only results.")
        return tsdb

    # Ensure column names have the expected format
    if not combined.empty:
        new_cols = []
        for col in combined.columns:
            sym, field = col
            # Normalize to symbol-quote (e.g. BTC-USD)
            if "-" not in sym and "/" not in sym:
                new_cols.append((f"{sym}-{quote}", field))
            else:
                new_cols.append((sym.replace("/", "-"), field))

        combined.columns = pd.MultiIndex.from_tuples(new_cols)

        # Deduplicate columns (important if both BTC and BTC-USD existed due to loader mismatch)
        if combined.columns.duplicated().any():
            # Group by column name and take the first non-null or just first
            combined = combined.loc[:, ~combined.columns.duplicated(keep="last")]

    return combined.sort_index()
