"""Collect dated stock listings, raw SIP bars and weekly share counts."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import logging
import os
import subprocess
import tempfile
import time
from collections.abc import Iterator
from contextlib import closing, contextmanager
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas_market_calendars as calendars
import requests
import yfinance as yf
from dotenv import load_dotenv
from psycopg2.extras import execute_values

from ggTrader.data.forward import common_candidate

LOG = logging.getLogger(__name__)
NYSE = calendars.get_calendar("NYSE")
DIRECTORY = "https://www.nasdaqtrader.com/dynamic/SymDir/"
ALPACA = "https://data.alpaca.markets/v2/stocks/bars"
DAILY_SOURCE = "nasdaq+alpaca-sip"
SHARES_SOURCE = "yahoo-shares"
LISTING_GAP_SYMBOL = "::listing-missing::"
REPO_ROOT = Path(__file__).resolve().parents[1]


def sessions(start: dt.date, end: dt.date) -> list[dt.date]:
    """Return NYSE trading dates inclusive of both endpoints."""
    return [x.date() for x in NYSE.valid_days(start_date=start, end_date=end)]


def bar_window(day: dt.date, now: dt.datetime | None = None) -> tuple[str, str]:
    """Return a fixed UTC window ending three hours after the scheduled close."""
    schedule = NYSE.schedule(start_date=day, end_date=day)
    if schedule.empty:
        raise ValueError("Not an NYSE trading day")
    close = schedule.iloc[0]["market_close"].to_pydatetime()
    now = now or dt.datetime.now(dt.timezone.utc)
    if now < close + dt.timedelta(minutes=15):
        raise ValueError("SIP bars need at least 15 minutes after the market close")
    end = close + dt.timedelta(hours=3)
    if now < end + dt.timedelta(minutes=15):
        end = close
    return f"{day}T00:00:00Z", end.strftime("%Y-%m-%dT%H:%M:%SZ")


def observation_day() -> dt.date:
    """Return today in the New York market timezone."""
    return dt.datetime.now(ZoneInfo("America/New_York")).date()


def shares_week(day: dt.date, observed: dt.date) -> dt.date:
    """Return Monday only for a day in the observation's current week."""
    week = day - dt.timedelta(days=day.weekday())
    if week != observed - dt.timedelta(days=observed.weekday()) or day > observed:
        raise ValueError("Shares observations must belong to the current week")
    return week


def digest(value: Any) -> str:
    """Hash a canonical JSON representation."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def retry_delay(value: str, now: dt.datetime, attempt: int) -> float:
    """Interpret a Retry-After delay or HTTP date, with exponential fallback."""
    if value.isdecimal():
        return min(120.0, float(value))
    try:
        when = parsedate_to_datetime(value)
        if when.tzinfo is not None:
            return min(120.0, max(0.0, (when - now).total_seconds()))
    except (TypeError, ValueError, OverflowError):
        pass
    return min(120.0, float(10 * 2**attempt))


class Sources:
    """Fetch free listing, raw SIP and shares data with bounded request pacing."""

    def __init__(self, checkpoint: Path | None) -> None:
        self.checkpoint = checkpoint
        self.requests = {"listing": 0, "alpaca": 0, "yahoo": 0}
        self.raw = []
        self.errors = []
        self.progress = {}

    def get(self, url: str, *, kind: str, **kwargs: Any) -> requests.Response:
        """Retry rate limits while respecting the source's request interval."""
        for attempt in range(5):
            time.sleep(0.5 if kind == "alpaca" else 1)
            self.requests[kind] += 1
            try:
                response = requests.get(url, timeout=60, **kwargs)
            except (requests.ConnectionError, requests.Timeout):
                if attempt == 4:
                    raise
                time.sleep(retry_delay("", dt.datetime.now(dt.timezone.utc), attempt))
                continue
            if (response.status_code != 429 and response.status_code < 500) or attempt == 4:
                response.raise_for_status()
                return response
            retry = response.headers.get("Retry-After", "")
            time.sleep(retry_delay(retry, dt.datetime.now(dt.timezone.utc), attempt))
        raise RuntimeError("source request failed")

    def _save_raw_listing(self, day: dt.date, filename: str, text: str) -> None:
        """Keep the first dated raw listing without silently replacing it."""
        if self.checkpoint is None:
            return
        raw_dir = REPO_ROOT / "data" / "forward_raw" / str(day)
        raw_dir.mkdir(parents=True, exist_ok=True)
        target = raw_dir / filename
        if target.exists() and target.read_text() != text:
            raise ValueError(f"Raw listing already exists with different contents: {filename}")
        if not target.exists():
            temporary = target.with_suffix(".tmp")
            temporary.write_text(text)
            temporary.replace(target)

    @staticmethod
    def _listing_rows(text: str, symbol: str) -> list[dict[str, Any]]:
        """Parse one directory without treating the ticker NA as missing."""
        import pandas as pd

        frame = pd.read_csv(
            io.StringIO(text), sep="|", dtype=str, keep_default_na=False, na_filter=False
        )
        rows = []
        for item in frame.to_dict("records"):
            name = item[symbol]
            if not name or name.startswith("File Creation Time"):
                continue
            rows.append(
                {
                    "symbol": name,
                    "exchange": item.get("Exchange", "Q"),
                    "security_name": item["Security Name"],
                    "is_etf": item["ETF"] == "Y",
                    "is_test": item["Test Issue"] == "Y",
                }
            )
        return rows

    def _persist_listing_pair(self, day: dt.date, raw: list[str]) -> list[str]:
        """Publish a complete raw pair atomically; preserve any older partial file."""
        if self.checkpoint is None:
            return raw
        root = REPO_ROOT / "data" / "forward_raw"
        root.mkdir(parents=True, exist_ok=True)
        folder = root / str(day)
        names = ("nasdaqlisted.txt", "otherlisted.txt")
        if folder.exists():
            chosen = []
            for name, text in zip(names, raw, strict=True):
                path = folder / name
                if path.exists():
                    text = path.read_text()
                else:
                    self._save_raw_listing(day, name, text)
                chosen.append(text)
            return chosen
        with tempfile.TemporaryDirectory(prefix=f".{day}-", dir=root) as staging:
            stage = Path(staging)
            for name, text in zip(names, raw, strict=True):
                (stage / name).write_text(text)
            stage.rename(folder)
        return raw

    def saved_listing(self, day: dt.date) -> list[dict[str, Any]] | None:
        """Read the dated listing checkpoint or its durable raw pair."""
        if self.checkpoint is None:
            return None
        path = self.checkpoint.with_name(self.checkpoint.stem + "_listing.json")
        raw = None
        if path.exists():
            data = json.loads(path.read_text())
            if data.get("day") == str(day):
                raw = data["raw"]
        if raw is None:
            folder = REPO_ROOT / "data" / "forward_raw" / str(day)
            paths = [folder / "nasdaqlisted.txt", folder / "otherlisted.txt"]
            if not all(item.is_file() for item in paths):
                return None
            raw = [item.read_text() for item in paths]
        if len(raw) != 2:
            raise ValueError("Saved listing checkpoint lacks both raw directories")
        raw = self._persist_listing_pair(day, raw)
        rows = []
        for text, symbol in zip(raw, ("Symbol", "ACT Symbol"), strict=True):
            rows.extend(self._listing_rows(text, symbol))
        if not rows:
            raise ValueError("Saved Nasdaq directories contain no listings")
        self.raw.extend(raw)
        return list({row["symbol"]: row for row in rows}.values())

    def listing(self) -> list[dict[str, Any]]:
        """Fetch both directories before preserving the dated raw pair."""
        day = observation_day()
        saved = self.saved_listing(day)
        if saved is not None:
            return saved
        if self.checkpoint is not None and self.checkpoint.stem != str(day):
            raise ValueError("Cannot save a current listing for a different day")
        raw = [
            self.get(DIRECTORY + name, kind="listing").text
            for name in ("nasdaqlisted.txt", "otherlisted.txt")
        ]
        if observation_day() != day:
            raise ValueError("Listing observation date changed while fetching the directories")
        rows = []
        for text, symbol in zip(raw, ("Symbol", "ACT Symbol"), strict=True):
            rows.extend(self._listing_rows(text, symbol))
        if not rows:
            raise ValueError("Nasdaq directories returned no listings")
        raw = self._persist_listing_pair(day, raw)
        rows = []
        for text, symbol in zip(raw, ("Symbol", "ACT Symbol"), strict=True):
            rows.extend(self._listing_rows(text, symbol))
        listing = list({row["symbol"]: row for row in rows}.values())
        self.raw.extend(raw)
        if self.checkpoint is not None:
            path = self.checkpoint.with_name(self.checkpoint.stem + "_listing.json")
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix(".tmp")
            temp.write_text(json.dumps({"day": str(day), "listing": listing, "raw": raw}))
            temp.replace(path)
        return listing

    def invalidate_bars(self) -> None:
        """Discard a low-coverage response so the next attempt can refetch it."""
        if self.checkpoint is not None:
            path = self.checkpoint.with_name(self.checkpoint.stem + "_bars.json")
            if path.exists():
                temporary = path.with_suffix(".tmp")
                temporary.write_text("{}")
                temporary.replace(path)

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
        start, end = bar_window(day)
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
                        "start": start,
                        "end": end,
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
            self.progress["bars"] = len(rows)
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
            cache = {
                "day": str(day),
                "symbols": symbols,
                "values": {},
                "unavailable": [],
                "listing_day": getattr(self, "share_listing_day", None),
            }
            if self.checkpoint is not None:
                temporary = self.checkpoint.with_suffix(".tmp")
                temporary.write_text(json.dumps(cache))
                temporary.replace(self.checkpoint)
        unavailable = set(cache.get("unavailable", []))
        for symbol, record in list(cache["values"].items()):
            if record["shares"] is None:
                unavailable.add(symbol)
                del cache["values"][symbol]
        failed = 0
        for symbol in symbols:
            if symbol in cache["values"] or symbol in unavailable:
                continue
            value = None
            failed_symbol = False
            for attempt in range(5):
                try:
                    time.sleep(1)
                    self.requests["yahoo"] += 1
                    observed = observation_day()
                    values = yf.Ticker(symbol).get_shares_full(
                        start=str(observed - dt.timedelta(days=550)),
                        end=str(observed + dt.timedelta(days=1)),
                    )
                    if values is not None:
                        values = values.dropna().sort_index()
                        if not values.empty and values.iloc[-1] >= 0:
                            value = int(values.iloc[-1])
                    break
                except Exception as exc:
                    if type(exc).__name__ != "YFRateLimitError" or attempt == 4:
                        error = type(exc).__name__
                        self.errors.append({"source": "yahoo", "symbol": symbol, "error": error})
                        failed += 1
                        failed_symbol = True
                        LOG.warning("Yahoo shares failed for %s: %s", symbol, error)
                        break
                    time.sleep(10 * 2**attempt)
            if failed_symbol:
                continue
            if value is None:
                unavailable.add(symbol)
            else:
                unavailable.discard(symbol)
                cache["values"][symbol] = {"shares": value, "observed_day": str(observation_day())}
            cache["unavailable"] = sorted(unavailable)
            if self.checkpoint is not None:
                tmp = self.checkpoint.with_suffix(".tmp")
                tmp.write_text(json.dumps(cache))
                tmp.replace(self.checkpoint)
        if symbols and not getattr(self, "share_is_retry", False) and failed / len(symbols) > 0.1:
            raise RuntimeError(f"Yahoo shares errors exceed 10%: {failed} of {len(symbols)}")
        self.raw.append({"values": cache["values"], "unavailable": sorted(unavailable)})
        return [
            {
                "symbol": symbol,
                "shares": value["shares"],
                "observed_day": dt.date.fromisoformat(value["observed_day"]),
            }
            for symbol, value in cache["values"].items()
            if value["shares"] is not None
        ]


def acquire_lock(conn: Any, kind: str = "daily") -> None:
    """Try the collector lock briefly rather than waiting indefinitely."""
    for attempt in range(5):
        with conn.cursor() as cur:
            cur.execute("SELECT pg_try_advisory_lock(hashtext(%s))", (f"forward_collect:{kind}",))
            if cur.fetchone()[0]:
                return
        if attempt < 4:
            time.sleep(1)
    raise TimeoutError("Forward collection advisory lock is busy after five attempts")


def release_lock(conn: Any, kind: str = "daily") -> None:
    """Release the lock without obscuring a preceding exception."""
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_unlock(hashtext(%s))", (f"forward_collect:{kind}",))
    except Exception:
        LOG.exception("Could not unlock the forward collection advisory lock")


def collect(
    day: dt.date,
    dry_run: bool,
    source: Any,
    conn: Any = None,
    shares_sample: int | None = None,
    listing_override: list[dict] | None = None,
    listing_day: dt.date | None = None,
) -> dict[str, Any]:
    """Collect one trading day; the dry run never opens or writes a database."""
    if shares_sample is not None and (not dry_run or shares_sample < 1):
        raise ValueError("shares sample must be positive and is dry-run only")
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
        acquire_lock(conn)
        try:
            return _collect_locked(day, source, conn, listing_override, listing_day)
        finally:
            release_lock(conn)
    return _gather(day, source, [], dry_run=True, shares_sample=shares_sample)[0]


def _gather(
    day: dt.date,
    source: Any,
    previous: list[dict],
    dry_run: bool,
    shares_sample: int | None = None,
    listing_override: list[dict] | None = None,
    tracked_gaps: set[str] | None = None,
    never_traded: set[str] | None = None,
) -> tuple:
    """Fetch source data and calculate coverage against tracked symbols."""
    listing = listing_override if listing_override is not None else source.listing()
    candidates = [x["symbol"] for x in listing if common_candidate(x)]
    if not candidates:
        raise ValueError("No explicit common-stock candidates in listing")
    tracked = {x["symbol"] for x in previous if x["ended_on"] is None}
    ended = {x["symbol"] for x in previous if x["ended_on"] is not None}
    target = sorted(
        set(candidates) | ((tracked | (tracked_gaps or set())) - ended - (never_traded or set()))
    )
    if hasattr(source, "progress"):
        source.progress.update(listed=len(listing), candidates=len(candidates))
    shares = source.shares(candidates[:shares_sample], day) if shares_sample is not None else []
    if hasattr(source, "progress"):
        source.progress["shares"] = len(shares)
    fetched = source.bars(target, day)
    unique = {}
    for bar in fetched:
        symbol = bar["symbol"]
        if symbol in unique and unique[symbol] != bar:
            if hasattr(source, "invalidate_bars"):
                source.invalidate_bars()
            raise ValueError(f"conflicting duplicate bars for {symbol}; checkpoint invalidated")
        unique[symbol] = bar
    bars = list(unique.values())
    returned = {x["symbol"] for x in bars}
    gaps = sorted(set(target) - returned)
    summary = {
        "listed": len(listing),
        "candidates": len(candidates),
        "bars": len(returned),
        "gaps": len(gaps),
        "shares": len(shares),
        "newly_ended": 0 if not dry_run else None,
        "alert": len(returned & set(candidates)) / len(candidates) < 0.9,
    }
    return summary, listing, bars, gaps, shares


@contextmanager
def write_transaction(conn: Any) -> Iterator[None]:
    """Commit prior reads and atomically commit or roll back one load."""
    original = conn.autocommit
    conn.commit()
    conn.autocommit = False
    try:
        yield
        conn.commit()
    except BaseException:
        if not conn.closed:
            conn.rollback()
        raise
    finally:
        if not conn.closed:
            conn.autocommit = original


def _collect_locked(
    day: dt.date,
    source: Any,
    conn: Any,
    listing_override: list[dict] | None = None,
    listing_day: dt.date | None = None,
) -> dict[str, Any]:
    """Write a complete collection atomically after checking the daily lock."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM forward.loads WHERE day = %s AND source <> %s LIMIT 1",
            (day, SHARES_SOURCE),
        )
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
            "SELECT DISTINCT ON (symbol) symbol, min(first_seen) OVER (PARTITION BY symbol), "
            "last_seen, ended_on "
            "FROM forward.symbol_status WHERE first_seen <= %s AND last_seen <= %s "
            "AND (ended_on IS NULL OR ended_on <= %s) "
            "ORDER BY symbol, COALESCE(ended_on, last_seen) DESC, status_id DESC",
            (day, day, day),
        )
        previous = [
            dict(zip(("symbol", "first_seen", "last_seen", "ended_on"), row, strict=True))
            for row in cur.fetchall()
        ]
        cur.execute(
            "SELECT DISTINCT g.symbol FROM forward.gaps g "
            "JOIN forward.loads l ON l.load_id=g.load_id "
            "WHERE g.reason='missing_bar' AND l.day = (SELECT max(day) FROM forward.loads "
            "WHERE day < %s AND source <> %s)",
            (day, SHARES_SOURCE),
        )
        tracked_gaps = {row[0] for row in cur.fetchall()}
        cur.execute(
            "SELECT DISTINCT g.symbol FROM forward.gaps g "
            "WHERE g.reason='never_traded' AND g.day < %s AND NOT EXISTS "
            "(SELECT 1 FROM forward.daily_bars b WHERE b.symbol=g.symbol "
            "AND b.day > g.day AND b.day < %s)",
            (day, day),
        )
        never_traded = {row[0] for row in cur.fetchall()}
    summary, listing, bars, gaps, shares = _gather(
        day,
        source,
        previous,
        dry_run=False,
        listing_override=listing_override,
        tracked_gaps=tracked_gaps,
        never_traded=never_traded,
    )
    if summary["alert"]:
        if hasattr(source, "invalidate_bars"):
            source.invalidate_bars()
        raise ValueError(f"Coverage below 90% for {day}; no load written")
    previous_map = {x["symbol"]: x for x in previous}
    prior_sessions = sessions(day - dt.timedelta(days=20), day)[-5:]
    raw = getattr(source, "raw", [listing, bars, shares])
    normalized = {"listing": listing, "bars": bars, "gaps": gaps, "shares": shares}
    code = git_revision()
    with write_transaction(conn):
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO forward.loads (day, source, raw_sha256, normalized_sha256, "
                "code_commit, actor) VALUES (%s,%s,%s,%s,%s,%s) RETURNING load_id",
                (
                    day,
                    (f"{DAILY_SOURCE}:catchup-from:{listing_day}" if listing_day else DAILY_SOURCE),
                    digest([raw, {"listing_from": listing_day}] if listing_day else raw),
                    digest(normalized),
                    code,
                    os.getenv("USER", "unknown"),
                ),
            )
            load = cur.fetchone()[0]
            execute_values(
                cur,
                "INSERT INTO forward.listing_snapshots "
                "(load_id, day, symbol, exchange, security_name, is_etf, is_test) VALUES %s",
                [
                    (
                        load,
                        day,
                        row["symbol"],
                        row["exchange"],
                        row["security_name"],
                        row["is_etf"],
                        row["is_test"],
                    )
                    for row in listing
                ],
                page_size=1000,
            )
            if bars:
                execute_values(
                    cur,
                    "INSERT INTO forward.daily_bars "
                    "(load_id, day, symbol, feed, open, high, low, close, volume) VALUES %s",
                    [
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
                        )
                        for row in bars
                    ],
                    page_size=1000,
                )
            for symbol in gaps:
                reason = (
                    "never_traded"
                    if symbol in never_traded and symbol not in previous_map
                    else "missing_bar"
                )
                if (
                    symbol not in previous_map
                    and symbol not in never_traded
                    and len(prior_sessions) == 5
                ):
                    cur.execute(
                        "SELECT count(DISTINCT day) FROM forward.gaps WHERE symbol=%s "
                        "AND reason='missing_bar' AND day=ANY(%s)",
                        (symbol, prior_sessions[:-1]),
                    )
                    if cur.fetchone()[0] == 4:
                        reason = "never_traded"
                cur.execute(
                    "INSERT INTO forward.gaps (load_id, day, symbol, reason) VALUES (%s,%s,%s,%s)",
                    (load, day, symbol, reason),
                )
            if listing_day is not None:
                cur.execute(
                    "INSERT INTO forward.gaps (load_id, day, symbol, reason) VALUES (%s,%s,%s,%s)",
                    (load, day, LISTING_GAP_SYMBOL, "listing_missing"),
                )
            if bars:
                execute_values(
                    cur,
                    "INSERT INTO forward.symbol_status "
                    "(symbol, first_seen, last_seen, ended_on) VALUES %s",
                    [
                        (
                            row["symbol"],
                            previous_map[row["symbol"]]["first_seen"]
                            if row["symbol"] in previous_map
                            else day,
                            day,
                            None,
                        )
                        for row in bars
                    ],
                    page_size=1000,
                )
            for symbol in gaps:
                old = previous_map.get(symbol)
                if not old:
                    continue
                if old["ended_on"] is not None:
                    continue
                cur.execute(
                    "SELECT count(DISTINCT day) FROM forward.gaps "
                    "WHERE symbol = %s AND reason = 'missing_bar' AND day = ANY(%s)",
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


def saved_listing(conn: Any, day: dt.date, *, strictly_before: bool = False) -> tuple:
    """Return the most recent real, not reconstructed, listing no later than a date."""
    op = "<" if strictly_before else "<="
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT load_id, day FROM forward.loads WHERE day {op} %s "
            "AND source IN (%s, %s) AND EXISTS "
            "(SELECT 1 FROM forward.listing_snapshots s WHERE s.load_id = forward.loads.load_id) "
            "ORDER BY day DESC, load_id DESC LIMIT 1",
            (day, DAILY_SOURCE, "nasdaq+alpaca-sip+yahoo"),
        )
        found = cur.fetchone()
        if not found:
            raise ValueError(f"No earlier saved listing available for {day}")
        load, listing_day = found
        cur.execute(
            "SELECT symbol, exchange, security_name, is_etf, is_test "
            "FROM forward.listing_snapshots WHERE load_id = %s ORDER BY symbol",
            (load,),
        )
        rows = [
            dict(
                zip(("symbol", "exchange", "security_name", "is_etf", "is_test"), row, strict=True)
            )
            for row in cur.fetchall()
        ]
    return listing_day, rows


def git_revision() -> str:
    """Identify the code version when available."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def collect_shares(day: dt.date, source: Any, conn: Any) -> dict[str, Any]:
    """Append weekly share observations, retrying unresolved symbols without duplicates."""
    week = day - dt.timedelta(days=day.weekday())
    acquire_lock(conn, "shares")
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT load_id FROM forward.loads WHERE day=%s AND source=%s",
                (week, SHARES_SOURCE),
            )
            loads = [row[0] for row in cur.fetchall()]
            if loads:
                cur.execute(
                    "SELECT symbol FROM (SELECT DISTINCT ON (symbol) symbol, outcome FROM ("
                    "SELECT symbol, 'value' AS outcome, load_id FROM forward.shares_outstanding "
                    "WHERE load_id=ANY(%s) UNION ALL "
                    "SELECT symbol, reason AS outcome, load_id FROM forward.gaps "
                    "WHERE load_id=ANY(%s) AND reason IN ('shares_failed','shares_unavailable')"
                    ") outcomes ORDER BY symbol, load_id DESC) latest "
                    "WHERE outcome='shares_failed'",
                    (loads, loads),
                )
                candidates = sorted(row[0] for row in cur.fetchall())
                if not candidates:
                    return {
                        "already_collected": True,
                        "shares": 0,
                        "remaining": 0,
                        "message": "Weekly shares complete; no retryable symbols remain",
                    }
                checkpoint = getattr(source, "checkpoint", None)
                previous = (
                    json.loads(checkpoint.read_text())
                    if checkpoint is not None and checkpoint.exists()
                    else {}
                )
                listing_day = previous.get("listing_day")
            else:
                checkpoint = getattr(source, "checkpoint", None)
                previous = (
                    json.loads(checkpoint.read_text())
                    if checkpoint is not None and checkpoint.exists()
                    else {}
                )
                if previous.get("day") == str(week) and previous.get("listing_day"):
                    listing_day = dt.date.fromisoformat(previous["listing_day"])
                    candidates = previous["symbols"]
                else:
                    listing_day, listing = saved_listing(conn, day)
                    candidates = sorted(row["symbol"] for row in listing if common_candidate(row))
        source.share_is_retry = bool(loads)
        if hasattr(source, "checkpoint"):
            source.share_listing_day = str(listing_day) if listing_day is not None else None
        if not candidates:
            raise ValueError("No eligible symbols for weekly shares")
        before_errors = len(getattr(source, "errors", []))
        shares = source.shares(candidates, week)
        successes = {row["symbol"] for row in shares}
        errors = getattr(source, "errors", [])[before_errors:]
        failed = sorted(
            {item["symbol"] for item in errors if item.get("source") == "yahoo"}
            & set(candidates) - successes
        )
        unavailable = sorted(set(candidates) - successes - set(failed))
        if not shares and not loads:
            raise ValueError("Yahoo returned no shares; no load was recorded")
        if not shares and not unavailable:
            return {"shares": 0, "remaining": len(candidates), "message": "No shares resolved"}
        raw = getattr(source, "raw", shares)
        with write_transaction(conn):
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO forward.loads (day, source, raw_sha256, normalized_sha256, "
                    "code_commit, actor) VALUES (%s,%s,%s,%s,%s,%s) RETURNING load_id",
                    (
                        week,
                        SHARES_SOURCE,
                        digest([raw, listing_day]),
                        digest([shares, failed, unavailable]),
                        git_revision(),
                        os.getenv("USER", "unknown"),
                    ),
                )
                load = cur.fetchone()[0]
                if shares:
                    execute_values(
                        cur,
                        "INSERT INTO forward.shares_outstanding "
                        "(load_id, day, symbol, shares, source) VALUES %s",
                        [
                            (
                                load,
                                row.get("observed_day", day),
                                row["symbol"],
                                row["shares"],
                                "yahoo",
                            )
                            for row in shares
                        ],
                        page_size=1000,
                    )
                missing = (
                    [] if loads else [(load, day, symbol, "shares_failed") for symbol in failed]
                )
                missing += [(load, day, symbol, "shares_unavailable") for symbol in unavailable]
                if missing:
                    execute_values(
                        cur,
                        "INSERT INTO forward.gaps (load_id, day, symbol, reason) VALUES %s",
                        missing,
                        page_size=1000,
                    )
        return {
            "shares": len(shares),
            "candidates": len(candidates),
            "remaining": len(failed),
            "unavailable": len(unavailable),
            "listing_day": str(listing_day) if listing_day is not None else None,
            "week": str(week),
        }
    finally:
        release_lock(conn, "shares")


def catch_up(
    end_day: dt.date,
    source_factory: Any,
    conn: Any,
    *,
    start_day: dt.date | None = None,
    current_day: dt.date | None = None,
) -> dict[str, Any]:
    """Fill up to ten missing sessions, never using a later listing."""
    current_day = current_day or end_day
    with conn.cursor() as cur:
        cur.execute(
            "SELECT min(day) FROM forward.loads WHERE source <> %s AND day <= %s",
            (SHARES_SOURCE, end_day),
        )
        earliest = cur.fetchone()[0]
    first = start_day or (earliest + dt.timedelta(days=1) if earliest else end_day)
    if first > end_day:
        return {"collected": [], "remaining": 0, "message": "Already current"}
    with conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT day FROM forward.loads WHERE source <> %s AND day BETWEEN %s AND %s",
            (SHARES_SOURCE, first, end_day),
        )
        existing = {row[0] for row in cur.fetchall()}
    missing = [day for day in sessions(first, end_day) if day not in existing]
    done = []
    failures = []
    today = [day for day in missing if day == current_day]
    older = [day for day in missing if day != current_day]
    for day in today + older[:10]:
        try:
            source = source_factory(day)
            if day == current_day:
                result = collect(day, False, source, conn)
            else:
                own = source.saved_listing(day) if hasattr(source, "saved_listing") else None
                if own is not None:
                    result = collect(day, False, source, conn, listing_override=own)
                else:
                    listing_day, listing = saved_listing(conn, day, strictly_before=True)
                    result = collect(
                        day, False, source, conn, listing_override=listing, listing_day=listing_day
                    )
            if not result.get("already_collected"):
                done.append(day)
        except Exception as exc:
            LOG.error("Forward collection failed for %s: %s", day, type(exc).__name__)
            failures.append({"day": day, "error": str(exc), "error_type": type(exc).__name__})
    remaining = max(0, len(older) - 10)
    message = (
        f"Stopped after 10 older trading days; {remaining} remain" if remaining else "Up to date"
    )
    if failures:
        message = f"{len(failures)} day(s) failed; retry needed; {remaining} older days remain"
    return {"collected": done, "remaining": remaining, "message": message, "failures": failures}


def last_closed_day(now: dt.datetime) -> dt.date:
    """Find the most recent NYSE session whose close has passed."""
    schedule = NYSE.schedule(start_date=now.date() - dt.timedelta(days=14), end_date=now.date())
    closed = schedule[schedule.market_close + dt.timedelta(minutes=15) <= now]
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
    parser.add_argument("--shares-sample", type=int, help="Dry-run sample size for Yahoo shares")
    parser.add_argument("--shares-only", action="store_true", help="Collect weekly shares only")
    args = parser.parse_args()
    if args.shares_sample is not None and (not args.dry_run or args.shares_sample < 1):
        parser.error("--shares-sample requires --dry-run and a positive size")
    if args.shares_only and (args.dry_run or args.shares_sample is not None):
        parser.error("--shares-only cannot be combined with dry-run or a sample")
    load_dotenv(REPO_ROOT / ".env")
    day = args.day or (
        observation_day() if args.shares_only else last_closed_day(dt.datetime.now(dt.timezone.utc))
    )
    checkpoint_day = day - dt.timedelta(days=day.weekday()) if args.shares_only else day
    checkpoint_name = f"shares-{checkpoint_day}.json" if args.shares_only else f"{day}.json"
    source = Sources(
        None if args.dry_run else REPO_ROOT / "results" / "forward_collection" / checkpoint_name
    )
    started = time.monotonic()
    if args.dry_run:
        try:
            result = collect(day, True, source, shares_sample=args.shares_sample)
        except Exception as exc:
            result = {
                "day": str(day),
                "error": str(exc),
                "error_type": type(exc).__name__,
                "elapsed_seconds": round(time.monotonic() - started, 2),
                "requests": source.requests,
                "progress": source.progress,
                "source_errors": source.errors,
            }
            print(json.dumps(result))
            raise SystemExit(1) from None
    else:
        import psycopg2

        from ggTrader.utils.config import get_db_connection_string

        today = observation_day()
        if day > today:
            raise SystemExit("Cannot collect a future day")
        if args.shares_only:
            shares_week(day, today)
        if args.shares_only and not sessions(day, day):
            print(json.dumps({"day": str(day), "message": "Not an NYSE trading day"}))
            return
        if not args.shares_only and day > last_closed_day(dt.datetime.now(dt.timezone.utc)):
            raise SystemExit("Daily collection requires an NYSE session after its close")
        url = get_db_connection_string().replace("postgresql+psycopg2://", "postgresql://")
        with closing(psycopg2.connect(url)) as conn:
            conn.autocommit = True
            if args.shares_only:
                result = collect_shares(day, source, conn)
            else:
                sources = []

                def make_source(target_day: dt.date) -> Sources:
                    instance = Sources(
                        REPO_ROOT / "results" / "forward_collection" / f"{target_day}.json"
                    )
                    sources.append(instance)
                    return instance

                result = catch_up(day, make_source, conn, current_day=today)
                for instance in sources:
                    for key, value in instance.requests.items():
                        source.requests[key] += value
                    source.errors.extend(instance.errors)
    result.update(
        day=str(day),
        elapsed_seconds=round(time.monotonic() - started, 2),
        requests=source.requests,
        source_errors=source.errors,
    )
    print(json.dumps(result, default=str))
    if result.get("failures"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
