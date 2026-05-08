"""Kraken deposit / withdrawal ledger fetching with TimescaleDB caching.

Used by the daily PnL report to compute "true trading PnL" by subtracting
external capital flows (deposits, withdrawals) from balance snapshots.
Without this, manual deposits show up as fake trading profit.

Cache strategy:
- Rows persisted in the ``kraken_ledger`` table (PK = ledger_id)
- TTL of 1 hour by default — older than that and we re-poll Kraken
- Incremental fetch using ``MAX(timestamp)`` from the table as the cursor
  so we don't re-pull the entire history on every report
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta, timezone
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


def _ledger_key(entry: dict) -> str:
    """Stable identifier for an entry — Kraken-supplied ``id`` if present,
    else a synthetic ``ts_amount`` fallback."""
    return entry.get("id") or f"{entry['timestamp_ms']}_{entry['amount']}"


def _read_ledger_from_db() -> tuple[list[dict], Optional[datetime]]:
    """Return (rows-as-entries, last_fetched_at) from the kraken_ledger table."""
    from sqlalchemy import text
    from ggTrader.utils.result_db_manager import ResultDBManager

    m = ResultDBManager()
    try:
        with m.engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT ledger_id, timestamp, type, currency, amount,
                           signed_amount, last_fetched_at
                    FROM kraken_ledger
                    ORDER BY timestamp
                    """
                )
            ).fetchall()
    except Exception as e:
        logger.warning(f"  [Ledger] DB read failed ({e!r})")
        return [], None
    if not rows:
        return [], None
    entries = []
    latest_fetched = None
    for r in rows:
        entries.append({
            "id": r[0],
            "timestamp_ms": int(r[1].timestamp() * 1000),
            "type": r[2],
            "currency": r[3],
            "amount": float(r[4]) if r[4] is not None else 0.0,
            "signed_amount": float(r[5]) if r[5] is not None else 0.0,
        })
        if r[6] is not None and (latest_fetched is None or r[6] > latest_fetched):
            latest_fetched = r[6]
    return entries, latest_fetched


def _write_ledger_to_db(new_entries: list[dict]) -> None:
    """Upsert new entries into the kraken_ledger table."""
    if not new_entries:
        return
    from sqlalchemy import text
    from ggTrader.utils.result_db_manager import ResultDBManager

    m = ResultDBManager()
    sql = text(
        """
        INSERT INTO kraken_ledger
            (ledger_id, timestamp, type, currency, amount, signed_amount, last_fetched_at)
        VALUES (:lid, :ts, :type, :curr, :amt, :sa, now())
        ON CONFLICT (ledger_id) DO UPDATE SET
            last_fetched_at = now()
        """
    )
    rows = []
    for e in new_entries:
        amt = float(e["amount"])
        signed = amt if e["type"] == "deposit" else -amt
        rows.append({
            "lid": _ledger_key(e),
            "ts": datetime.fromtimestamp(e["timestamp_ms"] / 1000.0, tz=timezone.utc),
            "type": e["type"],
            "curr": e["currency"],
            "amt": amt,
            "sa": signed,
        })
    try:
        with m.engine.begin() as conn:
            conn.execute(sql, rows)
    except Exception as e:
        logger.warning(f"  [Ledger] DB write failed: {e!r}")


def fetch_kraken_ledger_cached(
    cache_path: Optional[str] = None,  # retained for API compat; ignored
    cache_ttl_seconds: int = 3600,
    exchange: Any = None,
) -> pd.DataFrame:
    """Return a DataFrame of all deposits and withdrawals (DB-backed).

    Columns: ``timestamp`` (UTC tz-aware), ``type`` (deposit/withdrawal),
    ``currency``, ``amount``, ``id``, ``signed_amount`` (positive for deposits,
    negative for withdrawals).

    The cache lives in the ``kraken_ledger`` table. We poll Kraken at most once
    per ``cache_ttl_seconds`` and incrementally append new entries using the
    most recent stored timestamp as the cursor.
    """
    db_entries, last_fetched = _read_ledger_from_db()
    now = datetime.now(timezone.utc)
    is_fresh = (
        last_fetched is not None
        and (now - last_fetched) < timedelta(seconds=cache_ttl_seconds)
        and db_entries
    )

    if is_fresh:
        return _entries_to_df(db_entries)

    if exchange is None:
        try:
            exchange = _build_exchange()
        except Exception as e:
            logger.warning(
                f"  [Ledger] could not build exchange ({e!r}) — using DB cache only"
            )
            return _entries_to_df(db_entries)

    if db_entries:
        # 1-minute overlap to catch late-arriving entries
        since_ms = max(0, db_entries[-1]["timestamp_ms"] - 60_000)
    else:
        since_ms = 0

    new_entries = _fetch_all_ledger_entries(exchange, since_ms)
    if new_entries:
        _write_ledger_to_db(new_entries)
        # Re-merge: existing + new (dedupe by stable key)
        keys = {_ledger_key(e) for e in db_entries}
        for e in new_entries:
            if _ledger_key(e) not in keys:
                db_entries.append(e)
                keys.add(_ledger_key(e))
        db_entries.sort(key=lambda x: x["timestamp_ms"])
    else:
        # Bump last_fetched_at on existing rows so we don't re-poll for a TTL window.
        if db_entries:
            from sqlalchemy import text
            from ggTrader.utils.result_db_manager import ResultDBManager
            try:
                m = ResultDBManager()
                with m.engine.begin() as conn:
                    conn.execute(text("UPDATE kraken_ledger SET last_fetched_at = now()"))
            except Exception:
                pass

    return _entries_to_df(db_entries)


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
