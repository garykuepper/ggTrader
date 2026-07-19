"""One-time historical backfill: House Periodic Transaction Reports
(STOCK Act), 2015-present, via the House Clerk's public bulk annual index.
Much smaller volume than the Form 4 backfill (~750-1500 PTR filings/year
across ALL House members combined, vs. Form 4's per-issuer volume) --
realistically well under an hour, not the ~24-hour Form 4 run.

Resumable (skips already-cached filings) and rate-limited (same
shared-token-bucket pattern as form4_backfill.py, conservative for a
government site with no documented rate-limit guidance).
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from sqlalchemy import text

from ggTrader.lab.house_ptr_data import cache_filing, ensure_schema, fetch_year_index
from ggTrader.lab.persist import get_engine

EARLIEST_YEAR = 2015
MAX_REQUESTS_PER_SEC = 4.0
MAX_WORKERS = 4


class RateLimiter:
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


def _cached_doc_ids() -> set[str]:
    with get_engine().connect() as conn:
        rows = conn.execute(text("SELECT DISTINCT doc_id FROM house_ptr_transactions")).fetchall()
    return {r[0] for r in rows}


def main() -> None:
    ensure_schema()
    limiter = RateLimiter(MAX_REQUESTS_PER_SEC)

    def rate_limited_fetch(url: str) -> bytes:
        limiter.wait()
        from ggTrader.lab.house_ptr_data import _default_http_fetch

        return _default_http_fetch(url)

    current_year = pd.Timestamp.now().year
    years = list(range(EARLIEST_YEAR, current_year + 1))

    all_filings: list[dict] = []
    for year in years:
        try:
            filings = fetch_year_index(year, http_fetch=rate_limited_fetch)
        except Exception as e:  # noqa: BLE001
            print(f"{year}: ERROR fetching index: {e}", flush=True)
            continue
        print(f"{year}: {len(filings)} PTR filings", flush=True)
        all_filings.extend(filings)

    already = _cached_doc_ids()
    todo = [f for f in all_filings if f["doc_id"] not in already]
    print(
        f"Total {len(all_filings)} PTR filings, {len(already)} already cached, "
        f"{len(todo)} to fetch",
        flush=True,
    )

    total_rows = 0
    errors = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(
                cache_filing,
                f["year"],
                f["doc_id"],
                f["last"],
                f["first"],
                f["state_dst"],
                f["filing_date"],
                rate_limited_fetch,
            ): f
            for f in todo
        }
        done = 0
        for fut in as_completed(futures):
            done += 1
            try:
                total_rows += fut.result()
            except Exception:  # noqa: BLE001
                errors += 1
            if done % 100 == 0:
                print(f"  {done}/{len(todo)} filings processed, {errors} errors", flush=True)

    print(f"Done. {total_rows} total rows cached, {errors} errors.", flush=True)


if __name__ == "__main__":
    main()
