"""Kraken deposit / withdrawal ledger fetching with local caching.

Used by the daily PnL report to compute "true trading PnL" by subtracting
external capital flows (deposits, withdrawals) from balance snapshots.
Without this, manual deposits show up as fake trading profit.

Cache strategy:
- Persistent JSON file at data/live/kraken_ledger.json
- TTL of 1 hour by default (deposits change rarely; trading activity does not)
- Incremental fetch using ``since=last_known_timestamp + 1`` so we don't
  re-pull the entire history on every report
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger("ggTraderLive")


def _build_exchange() -> Any:
    """Construct a CCXT Kraken instance with API keys from environment.

    Mirrors execution_engine init but standalone so the report builder can
    work without depending on the live trader code path.
    """
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    from ggTrader.data.live.cached_loader import CachedExchangeLoader

    loader = CachedExchangeLoader(exchange_id="kraken")
    exchange = loader.exchange
    if not exchange.apiKey:
        exchange.apiKey = os.getenv("KRAKEN_KEY")
        exchange.secret = os.getenv("KRAKEN_SECRET")
    if not exchange.apiKey or not exchange.secret:
        raise RuntimeError(
            "Kraken API keys missing — set KRAKEN_KEY and KRAKEN_SECRET in .env"
        )
    return exchange


def _normalise_entry(entry: dict) -> Optional[dict]:
    """Normalise a CCXT ledger / deposit / withdrawal entry to a flat dict.

    Returns None if the entry isn't a deposit/withdrawal we care about.
    """
    etype = (entry.get("type") or "").lower()
    if etype not in ("deposit", "withdrawal", "transfer"):
        return None

    ts = entry.get("timestamp")
    if ts is None:
        # Try the datetime string fallback
        dt = entry.get("datetime")
        if dt:
            try:
                ts = int(pd.Timestamp(dt).timestamp() * 1000)
            except Exception:
                return None
        else:
            return None

    amount = entry.get("amount")
    if amount is None:
        return None
    try:
        amount = float(amount)
    except (TypeError, ValueError):
        return None

    currency = entry.get("currency") or entry.get("code") or "USD"

    return {
        "timestamp_ms": int(ts),
        "type": etype,
        "currency": currency,
        "amount": amount,
        "id": entry.get("id"),
    }


def _fetch_all_ledger_entries(exchange: Any, since_ms: int) -> list[dict]:
    """Pull all deposit + withdrawal entries from Kraken since the given timestamp.

    Uses ``fetch_deposits`` and ``fetch_withdrawals`` directly because
    ``fetch_ledger`` returns mostly ``type='trade'`` entries by default and
    requires extra Kraken-specific params to filter to deposits/withdrawals.
    """
    entries: list[dict] = []
    seen_ids: set[str] = set()

    for fn_name, default_type in (
        ("fetch_deposits", "deposit"),
        ("fetch_withdrawals", "withdrawal"),
    ):
        try:
            fn = getattr(exchange, fn_name, None)
            if fn is None:
                continue
            since = since_ms
            while True:
                batch = fn(code=None, since=since, limit=50)
                if not batch:
                    break
                for raw in batch:
                    # Ensure type is set even if the exchange response omits it
                    if not raw.get("type"):
                        raw["type"] = default_type
                    norm = _normalise_entry(raw)
                    if norm is None:
                        continue
                    key = norm.get("id") or f"{norm['timestamp_ms']}_{norm['amount']}"
                    if key in seen_ids:
                        continue
                    seen_ids.add(key)
                    entries.append(norm)
                if len(batch) < 50:
                    break
                since = batch[-1].get("timestamp", since) + 1
                time.sleep(1)  # Kraken rate limit
        except Exception as e:
            logger.warning(f"  [Ledger] {fn_name} failed: {e!r}")

    entries.sort(key=lambda x: x["timestamp_ms"])
    return entries


def _default_cache_path() -> str:
    """User-writable cache location that works on both host and container.

    Uses XDG_CACHE_HOME if set, else ~/.cache/ggtrader/. Falls back to
    /tmp/ggtrader_cache/ if HOME isn't writable (rare).
    """
    cache_root = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    candidate = os.path.join(cache_root, "ggtrader", "kraken_ledger.json")
    try:
        os.makedirs(os.path.dirname(candidate), exist_ok=True)
        return candidate
    except OSError:
        return "/tmp/ggtrader_cache/kraken_ledger.json"


def fetch_kraken_ledger_cached(
    cache_path: Optional[str] = None,
    cache_ttl_seconds: int = 3600,
    exchange: Any = None,
) -> pd.DataFrame:
    """Return a DataFrame of all deposits and withdrawals, using a local cache.

    Columns: ``timestamp`` (UTC tz-aware), ``type`` (deposit/withdrawal),
    ``currency``, ``amount``, ``id``, ``signed_amount`` (positive for deposits,
    negative for withdrawals).

    The cache is incremental — on each call we only fetch entries newer than
    the latest cached timestamp, so the report is fast even with a long history.

    Args:
        cache_path: Local JSON cache file path.
        cache_ttl_seconds: How fresh the cache must be to skip fetching at all.
        exchange: Optional pre-built CCXT exchange. If None, builds one from env.
    """
    if cache_path is None:
        cache_path = _default_cache_path()
    cache_p = Path(cache_path)
    try:
        cache_p.parent.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        # Caller passed a path we can't create — fall through and disable
        # persistent caching for this run. Reads/writes below will silently
        # no-op via the existing exception handlers.
        pass
    now_ms = int(time.time() * 1000)

    cached: dict = {"entries": [], "last_fetch_ms": 0, "last_entry_ts_ms": 0}
    if cache_p.exists():
        try:
            cached = json.loads(cache_p.read_text())
        except Exception as e:
            logger.warning(f"  [Ledger] cache read failed ({e!r}) — refetching all")

    cache_age_seconds = (now_ms - int(cached.get("last_fetch_ms", 0))) / 1000
    if cache_age_seconds < cache_ttl_seconds and cached.get("entries"):
        # Cache is fresh, use as-is
        entries = cached["entries"]
    else:
        # Fetch new entries since the last known one (or all if cache is empty)
        if exchange is None:
            try:
                exchange = _build_exchange()
            except Exception as e:
                logger.warning(
                    f"  [Ledger] could not build exchange ({e!r}) — using cached only"
                )
                entries = cached.get("entries", [])
                return _entries_to_df(entries)

        since_ms = int(cached.get("last_entry_ts_ms", 0))
        # Add a small overlap window so we don't miss entries that arrived late
        if since_ms > 0:
            since_ms = max(0, since_ms - 60_000)  # 1 min overlap
        new_entries = _fetch_all_ledger_entries(exchange, since_ms)

        # Merge: dedupe by id (or by timestamp+amount for entries without id)
        existing_keys = set()
        for e in cached.get("entries", []):
            key = e.get("id") or f"{e['timestamp_ms']}_{e['amount']}"
            existing_keys.add(key)

        merged = list(cached.get("entries", []))
        for e in new_entries:
            key = e.get("id") or f"{e['timestamp_ms']}_{e['amount']}"
            if key not in existing_keys:
                merged.append(e)
                existing_keys.add(key)

        merged.sort(key=lambda x: x["timestamp_ms"])
        last_ts = merged[-1]["timestamp_ms"] if merged else 0

        cached = {
            "entries": merged,
            "last_fetch_ms": now_ms,
            "last_entry_ts_ms": last_ts,
        }
        try:
            cache_p.write_text(json.dumps(cached, indent=2))
        except Exception as e:
            logger.warning(f"  [Ledger] cache write failed: {e!r}")
        entries = merged

    return _entries_to_df(entries)


def _entries_to_df(entries: list[dict]) -> pd.DataFrame:
    if not entries:
        return pd.DataFrame(
            columns=["timestamp", "type", "currency", "amount", "id", "signed_amount"]
        )
    df = pd.DataFrame(entries)
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df["signed_amount"] = df.apply(
        lambda r: float(r["amount"]) if r["type"] == "deposit" else -float(r["amount"]),
        axis=1,
    )
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df[["timestamp", "type", "currency", "amount", "id", "signed_amount"]]


def cumulative_net_deposits_usd(ledger_df: pd.DataFrame) -> pd.Series:
    """Build a cumulative net-deposits time series in USD.

    Non-USD entries are silently ignored (we'd need historical FX rates to
    convert them; the user's primary capital flows are USD per the gg account).
    Returns a Series indexed by timestamp (UTC) with cumulative net flow.
    """
    if ledger_df is None or ledger_df.empty:
        return pd.Series(dtype=float)
    usd_only = ledger_df[ledger_df["currency"].isin(["USD", "ZUSD"])]
    if usd_only.empty:
        return pd.Series(dtype=float)
    cum = usd_only.set_index("timestamp")["signed_amount"].sort_index().cumsum()
    return cum
