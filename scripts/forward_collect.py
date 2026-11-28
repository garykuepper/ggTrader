"""Collect dated stock listings, raw SIP bars and weekly share counts."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import logging
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas_market_calendars as calendars
import requests
import yfinance as yf
from dotenv import load_dotenv

LOG = logging.getLogger(__name__)
NYSE = calendars.get_calendar("NYSE")
DIRECTORY = "https://www.nasdaqtrader.com/dynamic/SymDir/"
ALPACA = "https://data.alpaca.markets/v2/stocks/bars"


def sessions(start: dt.date, end: dt.date) -> list[dt.date]:
    """Return NYSE trading dates inclusive of both endpoints."""
    return [x.date() for x in NYSE.valid_days(start_date=start, end_date=end)]


def observation_day() -> dt.date:
    """Return today in the New York market timezone."""
    return dt.datetime.now(ZoneInfo("America/New_York")).date()


def is_shares_day(day: dt.date) -> bool:
    """Collect shares on the first trading day of each week."""
    monday = day - dt.timedelta(days=day.weekday())
    return sessions(monday, day) == [day]


def common_candidate(row: dict[str, Any]) -> bool:
    """Select explicit common or ordinary share names, excluding other securities."""
    name = row["security_name"].lower()
    return (
        not row["is_etf"]
        and not row["is_test"]
        and not re.search(r"\b(warrants?|units?|rights?|preferred)\b", name)
        and bool(re.search(r"\b(common|ordinary) (stock|shares?)\b", name))
    )


def digest(value: Any) -> str:
    """Hash a canonical JSON representation."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


class Sources:
    """Fetch free listing, raw SIP and shares data with bounded request pacing."""

    def __init__(self, checkpoint: Path | None) -> None:
        self.checkpoint = checkpoint
        self.requests = {"listing": 0, "alpaca": 0, "yahoo": 0}
        self.raw = []
        self.errors = []

    def get(self, url: str, *, kind: str, **kwargs: Any) -> requests.Response:
        """Retry rate limits while respecting the source's request interval."""
        for attempt in range(5):
            time.sleep(0.5 if kind == "alpaca" else 1)
            self.requests[kind] += 1
            response = requests.get(url, timeout=60, **kwargs)
            if response.status_code != 429 or attempt == 4:
                response.raise_for_status()
                return response
            retry = response.headers.get("Retry-After", "")
            time.sleep(float(retry) if retry.isdecimal() else 10 * 2**attempt)
        raise RuntimeError("source request failed")

    def listing(self) -> list[dict[str, Any]]:
        """Read both Nasdaq directory files and retain non-ETF, non-test listings."""
        import pandas as pd

        rows = []
        for filename, symbol in [("nasdaqlisted.txt", "Symbol"), ("otherlisted.txt", "ACT Symbol")]:
            response = self.get(DIRECTORY + filename, kind="listing")
            self.raw.append(response.text)
            frame = pd.read_csv(io.StringIO(response.text), sep="|", dtype=str).fillna("")
            for item in frame.to_dict("records"):
                name = item[symbol]
                if not name or name.startswith("File Creation Time"):
                    continue
                if item["ETF"] == "Y" or item["Test Issue"] == "Y":
                    continue
                rows.append(
                    {
                        "symbol": name,
                        "exchange": item.get("Exchange", "Q"),
                        "security_name": item["Security Name"],
                        "is_etf": False,
                        "is_test": False,
                    }
                )
        if not rows:
            raise ValueError("Nasdaq directories returned no eligible listings")
        return list({x["symbol"]: x for x in rows}.values())

    def bars(self, symbols: list[str], day: dt.date) -> list[dict[str, Any]]:
        """Fetch raw SIP bars, splitting malformed batches without losing valid symbols."""
        headers = {
            "APCA-API-KEY-ID": os.getenv("APCA_API_KEY_ID", ""),
            "APCA-API-SECRET-KEY": os.getenv("APCA_API_SECRET_KEY", ""),
        }
        if not all(headers.values()):
            raise ValueError("Alpaca market-data credentials are missing")
        checkpoint = (
            self.checkpoint.with_name(self.checkpoint.stem + "_bars.json")
            if self.checkpoint
            else None
        )
        if checkpoint is not None:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
        cache = json.loads(checkpoint.read_text()) if checkpoint and checkpoint.exists() else {}
        if cache.get("day") != str(day) or cache.get("symbols") != symbols:
            cache = {"day": str(day), "symbols": symbols, "batches": {}}
        rows = []
        for offset in range(0, len(symbols), 200):
            key = str(offset)
            if key not in cache["batches"]:
                batch_rows = []
                batch_raw = []
                batches = [symbols[offset : offset + 200]]
                while batches:
                    batch = batches.pop(0)
                    params = {
                        "symbols": ",".join(batch),
                        "timeframe": "1Day",
                        "feed": "sip",
                        "adjustment": "raw",
                        "start": str(day),
                        "end": str(day + dt.timedelta(days=1)),
                        "limit": 10000,
                    }
                    while True:
                        try:
                            response = self.get(
                                ALPACA, kind="alpaca", headers=headers, params=params
                            )
                        except requests.HTTPError as exc:
                            if exc.response is None or exc.response.status_code != 400:
                                status = (
                                    exc.response.status_code
                                    if exc.response is not None
                                    else "unknown"
                                )
                                raise RuntimeError(
                                    f"Alpaca request failed: HTTP {status}"
                                ) from None
                            if len(batch) > 1:
                                mid = len(batch) // 2
                                batches[0:0] = [batch[:mid], batch[mid:]]
                            else:
                                self.errors.append(
                                    {"source": "alpaca", "symbol": batch[0], "error": "http_400"}
                                )
                            break
                        payload = response.json()
                        batch_raw.append(payload)
                        for symbol, bars in (payload.get("bars") or {}).items():
                            for bar in bars:
                                if bar["t"][:10] == str(day):
                                    batch_rows.append(
                                        dict(
                                            symbol=symbol,
                                            open=bar["o"],
                                            high=bar["h"],
                                            low=bar["l"],
                                            close=bar["c"],
                                            volume=bar["v"],
                                        )
                                    )
                        if not payload.get("next_page_token"):
                            break
                        params["page_token"] = payload["next_page_token"]
                cache["batches"][key] = {"rows": batch_rows, "raw": batch_raw}
                if checkpoint is not None:
                    temporary = checkpoint.with_suffix(".tmp")
                    temporary.write_text(json.dumps(cache))
                    temporary.replace(checkpoint)
            completed = cache["batches"][key]
            rows.extend(completed["rows"])
            self.raw.extend(completed["raw"])
        return rows

    def shares(self, symbols: list[str], day: dt.date) -> list[dict[str, Any]]:
        """Fetch a weekly observed share count, checkpointing each completed symbol."""
        if self.checkpoint is not None:
            self.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        cache = (
            json.loads(self.checkpoint.read_text())
            if self.checkpoint is not None and self.checkpoint.exists()
            else {}
        )
        if cache.get("day") != str(day) or cache.get("symbols") != symbols:
            cache = {"day": str(day), "symbols": symbols, "values": {}}
        for symbol in symbols:
            if symbol in cache["values"]:
                continue
            value = None
            for attempt in range(5):
                try:
                    time.sleep(1)
                    self.requests["yahoo"] += 1
                    values = yf.Ticker(symbol).get_shares_full(
                        start=str(day - dt.timedelta(days=550)), end=str(day + dt.timedelta(days=1))
                    )
                    if values is not None and not values.empty:
                        value = int(values.sort_index().iloc[-1])
                    break
                except Exception as exc:
                    if type(exc).__name__ != "YFRateLimitError" or attempt == 4:
                        self.errors.append(
                            {"source": "yahoo", "symbol": symbol, "error": type(exc).__name__}
                        )
                        break
                    time.sleep(10 * 2**attempt)
            cache["values"][symbol] = {"shares": value, "observed_day": str(observation_day())}
            if self.checkpoint is not None:
                tmp = self.checkpoint.with_suffix(".tmp")
                tmp.write_text(json.dumps(cache))
                tmp.replace(self.checkpoint)
        self.raw.append(cache["values"])
        return [
            {
                "symbol": symbol,
                "shares": value["shares"],
                "observed_day": dt.date.fromisoformat(value["observed_day"]),
            }
            for symbol, value in cache["values"].items()
            if value["shares"] is not None
        ]


def collect(day: dt.date, dry_run: bool, source: Any, conn: Any = None) -> dict[str, Any]:
    """Collect one trading day; the dry run never opens or writes a database."""
    if not sessions(day, day):
        return {
            "listed": 0,
            "bars": 0,
            "gaps": 0,
            "newly_ended": 0,
            "alert": False,
            "holiday": True,
        }
    if not dry_run and conn is None:
        raise ValueError("A database connection is required outside dry-run")
    if not dry_run:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_lock(hashtext('forward_collect'))")
        try:
            return _collect_locked(day, source, conn)
        finally:
            with conn.cursor() as cur:
                cur.execute("SELECT pg_advisory_unlock(hashtext('forward_collect'))")
    return _gather(day, source, [], dry_run=True)[0]


def _gather(day: dt.date, source: Any, previous: list[dict], dry_run: bool) -> tuple:
    """Fetch source data and calculate coverage against tracked symbols."""
    listing = source.listing()
    candidates = [x["symbol"] for x in listing if common_candidate(x)]
    if not candidates:
        raise ValueError("No explicit common-stock candidates in listing")
    tracked = {x["symbol"] for x in previous if x["ended_on"] is None}
    target = sorted(set(candidates) | tracked)
    bars = source.bars(target, day)
    returned = {x["symbol"] for x in bars}
    gaps = sorted(set(target) - returned)
    shares = source.shares(candidates, day) if is_shares_day(day) else []
    summary = {
        "listed": len(listing),
        "candidates": len(candidates),
        "bars": len(returned),
        "gaps": len(gaps),
        "shares": len(shares),
        "newly_ended": 0 if not dry_run else None,
        "alert": bool(target) and len(returned & set(target)) / len(target) < 0.9,
    }
    return summary, listing, bars, gaps, shares


def _collect_locked(day: dt.date, source: Any, conn: Any) -> dict[str, Any]:
    """Write a complete collection atomically after checking the daily lock."""
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM forward.loads WHERE day = %s LIMIT 1", (day,))
        if cur.fetchone():
            return {
                "already_collected": True,
                "listed": 0,
                "bars": 0,
                "gaps": 0,
                "newly_ended": 0,
                "alert": False,
            }
        cur.execute(
            "SELECT DISTINCT ON (symbol) symbol, first_seen, last_seen, ended_on "
            "FROM forward.symbol_status ORDER BY symbol, status_id DESC"
        )
        previous = [
            dict(zip(("symbol", "first_seen", "last_seen", "ended_on"), row, strict=True))
            for row in cur.fetchall()
        ]
    summary, listing, bars, gaps, shares = _gather(day, source, previous, dry_run=False)
    previous_map = {x["symbol"]: x for x in previous}
    prior_sessions = sessions(day - dt.timedelta(days=20), day)[-5:]
    raw = getattr(source, "raw", [listing, bars, shares])
    normalized = {"listing": listing, "bars": bars, "gaps": gaps, "shares": shares}
    try:
        code = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        code = "unknown"
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO forward.loads (day, source, raw_sha256, normalized_sha256, "
                "code_commit, actor) VALUES (%s,%s,%s,%s,%s,%s) RETURNING load_id",
                (
                    day,
                    "nasdaq+alpaca-sip+yahoo",
                    digest(raw),
                    digest(normalized),
                    code,
                    os.getenv("USER", "unknown"),
                ),
            )
            load = cur.fetchone()[0]
            for row in listing:
                cur.execute(
                    "INSERT INTO forward.listing_snapshots VALUES (%s,%s,%s,%s,%s,%s,%s)",
                    (
                        load,
                        day,
                        row["symbol"],
                        row["exchange"],
                        row["security_name"],
                        row["is_etf"],
                        row["is_test"],
                    ),
                )
            for row in bars:
                cur.execute(
                    "INSERT INTO forward.daily_bars VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    (
                        load,
                        day,
                        row["symbol"],
                        "sip",
                        row["open"],
                        row["high"],
                        row["low"],
                        row["close"],
                        row["volume"],
                    ),
                )
            for symbol in gaps:
                cur.execute(
                    "INSERT INTO forward.gaps VALUES (%s,%s,%s,%s)",
                    (load, day, symbol, "missing_bar"),
                )
            for row in shares:
                cur.execute(
                    "INSERT INTO forward.shares_outstanding VALUES (%s,%s,%s,%s,%s)",
                    (load, row.get("observed_day", day), row["symbol"], row["shares"], "yahoo"),
                )
            for row in bars:
                symbol = row["symbol"]
                old = previous_map.get(symbol)
                cur.execute(
                    "INSERT INTO forward.symbol_status (symbol, first_seen, last_seen, ended_on) "
                    "VALUES (%s,%s,%s,NULL)",
                    (symbol, old["first_seen"] if old else day, day),
                )
            for symbol in gaps:
                old = previous_map.get(symbol)
                if not old:
                    cur.execute(
                        "INSERT INTO forward.symbol_status "
                        "(symbol, first_seen, last_seen, ended_on) VALUES (%s,%s,%s,NULL)",
                        (symbol, day, day),
                    )
                    continue
                if old["ended_on"] is not None:
                    continue
                cur.execute(
                    "SELECT count(DISTINCT day) FROM forward.gaps "
                    "WHERE symbol = %s AND day = ANY(%s)",
                    (symbol, prior_sessions),
                )
                if len(prior_sessions) == 5 and cur.fetchone()[0] == 5:
                    cur.execute(
                        "INSERT INTO forward.symbol_status "
                        "(symbol, first_seen, last_seen, ended_on) VALUES (%s,%s,%s,%s)",
                        (symbol, old["first_seen"], old["last_seen"], day),
                    )
                    summary["newly_ended"] += 1
    return summary


def last_closed_day(now: dt.datetime) -> dt.date:
    """Find the most recent NYSE session whose close has passed."""
    schedule = NYSE.schedule(start_date=now.date() - dt.timedelta(days=14), end_date=now.date())
    closed = schedule[schedule.market_close <= now]
    if closed.empty:
        raise ValueError("No recently closed NYSE session")
    return closed.index[-1].date()


def main() -> None:
    """Parse arguments and run the collector."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--day", type=dt.date.fromisoformat, help="NYSE trading day (YYYY-MM-DD)")
    parser.add_argument(
        "--dry-run", action="store_true", help="Fetch only; do not access the database"
    )
    args = parser.parse_args()
    load_dotenv("/home/flynn/ggTrader/.env")
    day = args.day or last_closed_day(dt.datetime.now(dt.timezone.utc))
    source = Sources(None if args.dry_run else Path("results/forward_collection") / f"{day}.json")
    started = time.monotonic()
    if args.dry_run:
        result = collect(day, True, source)
    else:
        import psycopg2

        from ggTrader.utils.config import get_db_connection_string

        today = dt.datetime.now(ZoneInfo("America/New_York")).date()
        if day != today or day != last_closed_day(dt.datetime.now(dt.timezone.utc)):
            raise SystemExit(
                "Collection requires the current NYSE day after its close; no backfill"
            )
        url = get_db_connection_string().replace("postgresql+psycopg2://", "postgresql://")
        with psycopg2.connect(url) as conn:
            conn.autocommit = True
            result = collect(day, False, source, conn)
    result.update(
        day=str(day),
        elapsed_seconds=round(time.monotonic() - started, 2),
        requests=source.requests,
        source_errors=source.errors,
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
