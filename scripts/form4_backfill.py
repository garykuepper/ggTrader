"""One-time historical backfill: SEC EDGAR Form 4 insider transactions for
the SP500 universe, 2015-present. Genuinely large -- potentially 100k+
individual filing fetches across ~750 issuers -- so this is deliberately
resumable (skips filings already cached) and rate-limited (SEC's own
guidance caps automated access around 10 req/sec; this stays comfortably
under that via a shared token-bucket limiter across a small thread pool).
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from sqlalchemy import text

from ggTrader.lab.data import equity_universe_between
from ggTrader.lab.form4_data import (
    cache_filing,
    ensure_schema,
    list_form4_filings,
    load_ticker_cik_map,
)
from ggTrader.lab.persist import get_engine

EARLIEST_FILING_DATE = "2015-01-01"
MAX_REQUESTS_PER_SEC = 6.0
MAX_WORKERS = 5


class RateLimiter:
    """Shared token-bucket limiter so a thread pool's AGGREGATE request
    rate stays under SEC's guidance, regardless of worker count."""

    def __init__(self, max_per_sec: float) -> None:
        self._min_interval = 1.0 / max_per_sec
        self._lock = threading.Lock()
        self._last_call = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_call = time.monotonic()


def _cached_accessions(symbol: str) -> set[str]:
    with get_engine().connect() as conn:
        rows = conn.execute(
            text("SELECT DISTINCT accession_number FROM form4_transactions WHERE symbol = :s"),
            {"s": symbol},
        ).fetchall()
    return {r[0] for r in rows}


def main() -> None:
    ensure_schema()
    limiter = RateLimiter(MAX_REQUESTS_PER_SEC)

    def rate_limited_fetch(url: str) -> str:
        limiter.wait()
        from ggTrader.lab.form4_data import _default_http_fetch

        return _default_http_fetch(url)

    eval_start = pd.Timestamp(EARLIEST_FILING_DATE, tz="UTC")
    eval_end = pd.Timestamp.now(tz="UTC")
    symbols = equity_universe_between(eval_start, eval_end, universe="sp500")
    print(f"Backfilling Form 4 data for {len(symbols)} SP500 (ever-member) symbols", flush=True)

    ticker_cik = load_ticker_cik_map(http_fetch=rate_limited_fetch)
    print(f"Resolved {len(symbols)} symbols against {len(ticker_cik)} CIKs", flush=True)

    total_rows = 0
    for i, symbol in enumerate(symbols):
        cik = ticker_cik.get(symbol.replace("-", "."))  # yfinance normalizes '.' -> '-'
        if cik is None:
            cik = ticker_cik.get(symbol)
        if cik is None:
            print(f"  [{i + 1}/{len(symbols)}] {symbol}: no CIK found, skipping", flush=True)
            continue

        try:
            filings = list_form4_filings(cik, http_fetch=rate_limited_fetch)
        except Exception as e:  # noqa: BLE001 -- one bad issuer must not kill the whole backfill
            print(f"  [{i + 1}/{len(symbols)}] {symbol}: ERROR listing filings: {e}", flush=True)
            continue

        filings = [f for f in filings if f["filingDate"] >= EARLIEST_FILING_DATE]
        already = _cached_accessions(symbol)
        todo = [f for f in filings if f["accessionNumber"] not in already]
        if not todo:
            print(
                f"  [{i + 1}/{len(symbols)}] {symbol}: {len(filings)} filings, all already cached",
                flush=True,
            )
            continue

        symbol_rows = 0
        errors = 0
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {
                pool.submit(
                    cache_filing,
                    cik,
                    f["accessionNumber"],
                    f["xml_filename"],
                    f["filingDate"],
                    rate_limited_fetch,
                ): f
                for f in todo
            }
            for fut in as_completed(futures):
                try:
                    symbol_rows += fut.result()
                except Exception:  # noqa: BLE001 -- one bad filing must not kill the symbol
                    errors += 1

        total_rows += symbol_rows
        print(
            f"  [{i + 1}/{len(symbols)}] {symbol}: {len(todo)} filings fetched "
            f"({symbol_rows} rows, {errors} errors), {len(already)} already cached",
            flush=True,
        )

    print(f"Done. {total_rows} total new rows cached.", flush=True)


if __name__ == "__main__":
    main()
