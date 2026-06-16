# Lab Plan 3: Equity Backfill + CachedYFinanceLoader Fix + Old Research Deletion

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the `CachedYFinanceLoader._interval_to_timedelta` AttributeError that forces every equity lab run to re-download ~600 yfinance symbols live (2 min), backfill S&P 500 OHLCV into TimescaleDB for instant future runs, consolidate `fetch_stock_ohlcv`/`STOCK_BASE_CONFIG` into `lab/data.py` to eliminate the `lab/ → research/` dependency, then delete the entire old research/WFO code cluster (~10 files/packages) that the lab bench replaces.

**Architecture:** Three phases in dependency order: (1) fix the loader bug and move its callers' dependencies into `lab/data.py`; (2) run a one-time backfill script to populate the DB; (3) delete the old code cluster now that nothing live depends on it. Validation gate (bit-identical selections on `xs_momentum`/`dual_momentum`) already passed in Plan 1 — cutover condition is met.

**Tech Stack:** Python, psycopg2/SQLAlchemy, yfinance, TimescaleDB, pytest. venv activated via `source .venv/bin/activate`.

**Spec:** [`docs/superpowers/specs/2026-06-15-vectorbt-lab-core-design.md`](../specs/2026-06-15-vectorbt-lab-core-design.md) §9 (Cutover and deletion).

**Conventions:** ruff line length 100; absolute imports from `ggTrader`; a PostToolUse hook runs ruff autofix. Commit trailer: `Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>`.

---

## File Map

| File | Action | Reason |
|---|---|---|
| `src/ggTrader/data/live/cached_yfinance_loader.py` | Modify | Add `_interval_to_timedelta` method |
| `src/ggTrader/lab/data.py` | Modify | Absorb `fetch_stock_ohlcv` + `STOCK_BASE_CONFIG` from `research/equity_wfo.py`; fix the `research.equity_wfo` import |
| `src/ggTrader/lab/cli.py` | Modify | Update `STOCK_BASE_CONFIG` import to `ggTrader.lab.data` |
| `tests/lab/test_data.py` | Modify | Extend: add unit test for `_interval_to_timedelta`; add integration test for `fetch_stock_ohlcv` |
| `scripts/equity_backfill.py` | Create | One-time script: download full S&P 500 history via `CachedYFinanceLoader` and persist to DB |
| `src/ggTrader/research/` | Delete (4 files) | `equity_wfo.py`, `monthly_strategies.py`, `monthly_walkforward.py`, `__init__.py` |
| `src/ggTrader/backtest/` | **Leave** | `vectorized.py` still used by `cli/cmd_backtest_strategy.py` (new-arch crypto carry) |
| `src/ggTrader/backtesting/` | Delete (3 files) | `__init__.py`, `wfo.py`, `__pycache__` — zero importers confirmed |
| `src/ggTrader/pipeline/` | Delete (4 files) | `exit_tournament.py`, `param_grids.py`, `pipeline_runner.py`, `__init__.py` |
| `src/ggTrader/core/wfo.py` | Delete | Only used by `orchestrator.py` + `research/equity_wfo.py` (both being deleted) |
| `src/ggTrader/core/orchestrator.py` | Delete | Only used by `utils/pipeline_phases.py` + research CLI (both being deleted) |
| `src/ggTrader/core/fast_backtest.py` | Delete | Only used by orchestrator/wfo/sensitivity/portfolio_optimizer (all being deleted) |
| `src/ggTrader/core/sensitivity.py` | Delete | Only used by orchestrator.py (being deleted) |
| `src/ggTrader/core/benchmarking.py` | Delete | Only used by orchestrator.py + equity_wfo.py (both being deleted) |
| `src/ggTrader/core/orchestrator_utils.py` | Delete | Only used by orchestrator.py, wfo.py, sensitivity.py (all being deleted) |
| `src/ggTrader/core/wfo_aggregate.py` | Delete | Only used by core/wfo.py (being deleted) |
| `src/ggTrader/core/portfolio_optimizer.py` | Delete | Only used by `cli/cmd_production.py` (being deleted) |
| `src/ggTrader/utils/pipeline_phases.py` | Delete | Only used by `cmd_report.py` + `pipeline/pipeline_runner.py` (both being deleted) |
| `src/ggTrader/cli/cmd_research.py` | Delete | Old research CLI command |
| `src/ggTrader/cli/cmd_backtest.py` | Delete | Old backtest CLI command |
| `src/ggTrader/cli/cmd_production.py` | Delete | Old WFO→production pipeline CLI command |
| `src/ggTrader/cli/cmd_report.py` | Delete | Old research report CLI; uses pipeline_phases.py |
| `src/ggTrader/cli/main.py` | Modify | Remove 8 registration entries for the 4 deleted commands |
| `docs/superpowers/specs/2026-06-15-vectorbt-lab-core-design.md` | Modify | Mark Plan 3 executed |
| `docs/changelog.md` | Modify | Add 2026-06-16 entry |

---

## Task 1: Fix `CachedYFinanceLoader._interval_to_timedelta`

The `fetch_ohlcv` method calls `self._interval_to_timedelta(interval)` (line 175) but neither `CachedYFinanceLoader` nor its parent `YFinanceDataLoader` defines it. The AttributeError causes `fetch_ohlcv` to raise, the caller catches it and falls back to plain yfinance, and data never gets persisted — so every lab run re-downloads all ~600 symbols.

**Files:**
- Modify: `src/ggTrader/data/live/cached_yfinance_loader.py`
- Modify: `tests/lab/test_data.py`

- [ ] **Step 1: Write the failing test**

Add this to `tests/lab/test_data.py` (after the existing imports, before `test_rebalance_dates_are_month_ends_excluding_last`):

```python
def test_cached_loader_interval_to_timedelta():
    from ggTrader.data.live.cached_yfinance_loader import CachedYFinanceLoader
    loader = CachedYFinanceLoader.__new__(CachedYFinanceLoader)  # skip __init__ (no DB needed)
    assert loader._interval_to_timedelta("1d") == pd.Timedelta(days=1)
    assert loader._interval_to_timedelta("1h") == pd.Timedelta(hours=1)
    assert loader._interval_to_timedelta("1wk") == pd.Timedelta(weeks=1)
    assert loader._interval_to_timedelta("1mo") == pd.Timedelta(days=30)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest tests/lab/test_data.py::test_cached_loader_interval_to_timedelta -v`
Expected: FAIL with `AttributeError: 'CachedYFinanceLoader' object has no attribute '_interval_to_timedelta'`

- [ ] **Step 3: Add the method to `CachedYFinanceLoader`**

In `src/ggTrader/data/live/cached_yfinance_loader.py`, insert this method inside `CachedYFinanceLoader`, between `__init__` and `_connect`:

```python
    @staticmethod
    def _interval_to_timedelta(interval: str) -> "pd.Timedelta":
        _MAP = {
            "1m": pd.Timedelta(minutes=1),
            "5m": pd.Timedelta(minutes=5),
            "15m": pd.Timedelta(minutes=15),
            "30m": pd.Timedelta(minutes=30),
            "60m": pd.Timedelta(hours=1),
            "1h": pd.Timedelta(hours=1),
            "4h": pd.Timedelta(hours=4),
            "1d": pd.Timedelta(days=1),
            "5d": pd.Timedelta(days=5),
            "1wk": pd.Timedelta(weeks=1),
            "1mo": pd.Timedelta(days=30),
            "3mo": pd.Timedelta(days=90),
        }
        if interval not in _MAP:
            raise ValueError(f"Unknown interval {interval!r}")
        return _MAP[interval]
```

The method also needs `import pandas as pd` at module level — it's already there (line 19 of the file).

- [ ] **Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && python -m pytest tests/lab/test_data.py::test_cached_loader_interval_to_timedelta -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add src/ggTrader/data/live/cached_yfinance_loader.py tests/lab/test_data.py
git commit -m "fix(data): add _interval_to_timedelta to CachedYFinanceLoader

The missing method caused fetch_ohlcv to AttributeError, triggering the
fallback to plain yfinance on every equity lab run (no DB caching = 2-min
re-downloads for 600 symbols).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 2: Move `fetch_stock_ohlcv` and `STOCK_BASE_CONFIG` into `lab/data.py`

`lab/data.py` and `lab/cli.py` currently import from `research/equity_wfo.py`. Moving these two items into `lab/data.py` makes the lab package self-contained and lets us delete `research/` in Task 4.

`STOCK_BASE_CONFIG` is a pure dict constant — copy it verbatim. `fetch_stock_ohlcv` is a ~70-line function — copy it verbatim.

**Files:**
- Modify: `src/ggTrader/lab/data.py`
- Modify: `src/ggTrader/lab/cli.py`
- Modify: `tests/lab/test_data.py`

- [ ] **Step 1: Write the failing integration test**

Add to `tests/lab/test_data.py` (at the bottom, before the existing `@pytest.mark.integration` block):

```python
@pytest.mark.integration
def test_fetch_stock_ohlcv_returns_multiindex_frame():
    from ggTrader.lab.data import fetch_stock_ohlcv
    df = fetch_stock_ohlcv(["SPY", "AAPL"], start="2024-01-01", end="2024-03-01")
    assert df.columns.names == ["symbol", "field"]
    assert "close" in df["SPY"].columns
    assert len(df) > 20
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest tests/lab/test_data.py::test_fetch_stock_ohlcv_returns_multiindex_frame -v -m integration`
Expected: FAIL with `ImportError: cannot import name 'fetch_stock_ohlcv' from 'ggTrader.lab.data'`

- [ ] **Step 3: Add `STOCK_BASE_CONFIG` and `fetch_stock_ohlcv` to `lab/data.py`**

Add the following to `src/ggTrader/lab/data.py` — insert after the existing imports and before `load_ohlcv`. Note: `normalize_yf_ticker` is already imported from `ggTrader.data.core.index_constituents` in the file; use it.

```python
from typing import Any  # add to existing import if not present

#: Default research config for daily-bar US equities.
STOCK_BASE_CONFIG: dict[str, Any] = {
    "START_CASH": 10000.0,
    "PORTFOLIO_SHARE": 1.0,
    "FEES": 0.0,
    "SLIPPAGE": 0.0005,
    "FREQ": "1d",
    "USE_CASH_SHARING": False,
    "TRAIN_METRIC": "composite",
    "MIN_CLOSED_TRADES_TRAIN": 0,
    "MIN_TRADES_PER_TRAIN_FOLD": 8,
    "MAX_TRAIN_DRAWDOWN_PCT": 75,
    "BENCHMARK_SYMBOL": "SPY",
}


def fetch_stock_ohlcv(
    symbols: List[str],
    start: str,
    end: Optional[str] = None,
    interval: str = "1d",
    use_db_cache: bool = True,
    min_coverage: float = 0.0,
) -> pd.DataFrame:
    """Fetch daily OHLCV for ``symbols`` as a (symbol, field) MultiIndex frame.

    DB-first via CachedYFinanceLoader (TimescaleDB) when reachable; falls back
    to plain yfinance. Symbols absent from the cached result are fetched for
    the full range and persisted.
    """
    tickers = sorted({normalize_yf_ticker(s) for s in symbols})
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC") if end else None

    from ggTrader.data.live.yfinance_loader import YFinanceDataLoader

    loader: Any = None
    if use_db_cache:
        try:
            from ggTrader.data.live.cached_yfinance_loader import CachedYFinanceLoader

            loader = CachedYFinanceLoader()
            df = loader.fetch_ohlcv(tickers, interval, start_date=start_ts, end_date=end_ts)
        except Exception as exc:
            print(f"  [data] DB cache unavailable ({exc!r}); falling back to yfinance only")
            loader = None
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()

    plain = YFinanceDataLoader()
    if df.empty:
        df = plain.fetch_ohlcv(tickers, interval, start_date=start_ts, end_date=end_ts)
        if df.empty:
            raise ValueError("yfinance returned no data for the requested universe")

    have = set(df.columns.get_level_values(0).unique())
    missing = [t for t in tickers if t not in have]
    if missing:
        print(f"  [data] fetching {len(missing)} symbols missing from cache...")
        extra = plain.fetch_ohlcv(missing, interval, start_date=start_ts, end_date=end_ts)
        if not extra.empty:
            if loader is not None:
                try:
                    loader._cache_to_db(extra, interval)
                except Exception as exc:
                    print(f"  [data] failed to persist gap fetch: {exc!r}")
            df = pd.concat([df, extra], axis=1)
            df.sort_index(axis=1, inplace=True)

    df = df[df.index >= start_ts]
    if end_ts is not None:
        df = df[df.index <= end_ts]

    if min_coverage > 0.0:
        keep = [
            sym
            for sym in df.columns.get_level_values(0).unique()
            if df[sym]["close"].notna().mean() >= min_coverage
        ]
        df = df[keep]

    n_syms = len(df.columns.get_level_values(0).unique())
    print(f"  [data] {len(df)} rows x {n_syms} symbols ({interval})")
    return df
```

Also ensure `from typing import Any` is in the imports at the top of `lab/data.py` (it may already be there — check).

- [ ] **Step 4: Update `lab/cli.py` imports**

In `src/ggTrader/lab/cli.py`, replace:

```python
from ggTrader.research.equity_wfo import STOCK_BASE_CONFIG
```

with:

```python
from ggTrader.lab.data import STOCK_BASE_CONFIG
```

- [ ] **Step 5: Update `lab/data.py` import of `fetch_stock_ohlcv`**

In `src/ggTrader/lab/data.py`, the existing `load_ohlcv` function currently calls:

```python
from ggTrader.research.equity_wfo import fetch_stock_ohlcv
return fetch_stock_ohlcv(symbols, start=start, end=end, interval="1d", use_db_cache=True)
```

Change `load_ohlcv` to call the local `fetch_stock_ohlcv` defined above (no import needed — it's in the same file):

```python
def load_ohlcv(symbols: List[str], start: str, end: Optional[str] = None) -> pd.DataFrame:
    """DB-first daily OHLCV as a (symbol, field) MultiIndex frame."""
    return fetch_stock_ohlcv(symbols, start=start, end=end, interval="1d", use_db_cache=True)
```

- [ ] **Step 6: Run unit tests (no integration marker) to verify nothing broken**

Run: `source .venv/bin/activate && python -m pytest tests/lab/ -m "not integration" -q`
Expected: all existing unit tests pass (no new failures).

- [ ] **Step 7: Run the new integration test**

Run: `source .venv/bin/activate && python -m pytest tests/lab/test_data.py::test_fetch_stock_ohlcv_returns_multiindex_frame -v -m integration`
Expected: PASS (1 passed). If DB unreachable, the function falls back to yfinance — still passes.

- [ ] **Step 8: Commit**

```bash
git add src/ggTrader/lab/data.py src/ggTrader/lab/cli.py tests/lab/test_data.py
git commit -m "refactor(lab): move fetch_stock_ohlcv + STOCK_BASE_CONFIG into lab/data.py

Eliminates the lab/ → research/ import dependency, enabling research/
deletion in the next task. STOCK_BASE_CONFIG and fetch_stock_ohlcv are
copied verbatim; lab/cli.py updated to import from lab/data.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 3: One-time S&P 500 equity backfill

Create and run a script that downloads the full daily history for every symbol that has ever been in the S&P 500 (using the PIT constituents list) and persists it to TimescaleDB via `CachedYFinanceLoader`. After this runs, equity lab runs hit the DB and complete in seconds.

**Files:**
- Create: `scripts/equity_backfill.py`

- [ ] **Step 1: Create the backfill script**

```python
#!/usr/bin/env python
"""One-time S&P 500 equity OHLCV backfill into TimescaleDB.

Run once (takes ~5-10 min for ~600 symbols). After this, equity lab runs
hit the DB cache instead of downloading live from yfinance.

Usage:
    source .venv/bin/activate
    python scripts/equity_backfill.py [--start 2000-01-01] [--batch 50]
"""

from __future__ import annotations

import argparse
import sys
import time

sys.path.insert(0, "src")

import pandas as pd

from ggTrader.data.core.index_constituents import all_members_between, normalize_yf_ticker
from ggTrader.lab.data import fetch_stock_ohlcv


def main() -> None:
    p = argparse.ArgumentParser(description="Backfill S&P 500 OHLCV into TimescaleDB.")
    p.add_argument("--start", default="2000-01-01", help="History start date (default: 2000-01-01)")
    p.add_argument("--batch", type=int, default=50, help="Symbols per yfinance batch (default: 50)")
    args = p.parse_args()

    start_ts = pd.Timestamp(args.start, tz="UTC")
    end_ts = pd.Timestamp.now(tz="UTC").normalize()

    # All S&P 500 members from start to today (PIT union).
    members = sorted({normalize_yf_ticker(t) for t in all_members_between(start_ts, end_ts)})
    # Always include SPY (benchmark).
    if "SPY" not in members:
        members = ["SPY"] + members

    print(f"Backfilling {len(members)} symbols from {args.start} to {end_ts.date()}")
    print(f"Batch size: {args.batch}")

    total_batches = (len(members) + args.batch - 1) // args.batch
    for i in range(0, len(members), args.batch):
        batch = members[i : i + args.batch]
        batch_num = i // args.batch + 1
        print(f"\n[{batch_num}/{total_batches}] {batch[0]}..{batch[-1]} ({len(batch)} symbols)")
        t0 = time.time()
        try:
            fetch_stock_ohlcv(batch, start=args.start, use_db_cache=True)
            print(f"  done in {time.time() - t0:.1f}s")
        except Exception as exc:
            print(f"  ERROR: {exc!r} — skipping batch")

    print(f"\nBackfill complete: {len(members)} symbols processed.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the script is syntactically valid**

Run: `source .venv/bin/activate && python -c "import ast, pathlib; ast.parse(pathlib.Path('scripts/equity_backfill.py').read_text()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Run the backfill (in the container where DB is accessible)**

Run: `source .venv/bin/activate && python scripts/equity_backfill.py --start 2000-01-01 --batch 50`

This will take 5-10 minutes. Expected output pattern:
```
Backfilling 6XX symbols from 2000-01-01 to YYYY-MM-DD
Batch size: 50

[1/13] A..BXP (50 symbols)
  [data] Full-range yfinance fetch for 50 symbols
  [data] XXXX rows x 50 symbols (1d)
  done in X.Xs
...
Backfill complete: 6XX symbols processed.
```

If any batch errors with a yfinance rate-limit (HTTP 429), re-run the script — it will skip symbols already in the DB (incremental fetch for fresh symbols, full fetch skipped for cached ones).

- [ ] **Step 4: Verify lab run hits DB (fast)**

Run: `source .venv/bin/activate && time python -m ggTrader.lab.cli --strategy xs_momentum --eval-start 2024-01-31 --eval-end 2024-06-30 --max-stocks 20`

Expected: completes in <30s (was ~2 min before backfill). DB cache hit confirmed if you see `[data] XXXX rows x 20 symbols (1d)` with no "Full-range yfinance fetch" line.

- [ ] **Step 5: Commit the script**

```bash
git add scripts/equity_backfill.py
git commit -m "feat(scripts): one-time S&P 500 equity OHLCV backfill

Downloads full history for all ~600 S&P 500 symbols via CachedYFinanceLoader
and persists to TimescaleDB. After running this once, equity lab runs hit the
DB cache and complete in <30s instead of ~2 minutes.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 4: Delete `research/` package

The `research/` package has no remaining importers after Task 2 moved its only needed exports into `lab/data.py`. Safe to remove entirely.

**Verify before deleting:**

Run: `grep -rn "from ggTrader.research\|import ggTrader.research" src/ --include="*.py" | grep -v "__pycache__"`
Expected: **no output** (zero importers). If any appear, fix them first.

- [ ] **Step 1: Delete the research package**

```bash
rm -rf src/ggTrader/research/
```

- [ ] **Step 2: Run full lab unit suite to confirm no regressions**

Run: `source .venv/bin/activate && python -m pytest tests/lab/ -m "not integration" -q`
Expected: all lab unit tests pass.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "refactor(lab): delete research/ package

fetch_stock_ohlcv and STOCK_BASE_CONFIG moved to lab/data.py in the
previous commit. The wfo_tournament code in equity_wfo.py is superseded
by the new lab bench; monthly_strategies.py was ported verbatim into
lab/strategies/momentum.py. Validation gate passed in Plan 1.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 5: Delete old WFO core cluster (`core/wfo.py`, `core/orchestrator.py`, etc.)

These six `core/` files form a self-contained cluster that powered the old WFO research pipeline. None of them are imported by anything outside this cluster, `research/` (deleted), `pipeline/` (deleted next task), or the old CLI commands (deleted in Task 6).

Files to delete in this task:
- `src/ggTrader/core/wfo.py`
- `src/ggTrader/core/wfo_aggregate.py`
- `src/ggTrader/core/orchestrator.py`
- `src/ggTrader/core/orchestrator_utils.py`
- `src/ggTrader/core/fast_backtest.py`
- `src/ggTrader/core/sensitivity.py`
- `src/ggTrader/core/benchmarking.py`
- `src/ggTrader/core/portfolio_optimizer.py`

**NOT deleted:** `core/crypto_execution_engine.py`, `core/base_execution_engine.py`, `core/trade_tracker.py`, `core/balance.py`, `core/calendars.py`, `core/calendar.py`, `core/dashboard_charts.py`, `core/holdout.py`, `core/instrument.py`, `core/metrics.py`, `core/order.py`, `core/signal.py`, `core/ticker.py`, `core/trailing_stop_utils.py`, `core/types.py`, `core/universe.py` — these power the live crypto trading engine.

**Verify before deleting:**

```bash
grep -rn "from ggTrader.core.wfo\b\|from ggTrader.core.orchestrator\b\|from ggTrader.core.fast_backtest\|from ggTrader.core.sensitivity\|from ggTrader.core.benchmarking\|from ggTrader.core.orchestrator_utils\|from ggTrader.core.wfo_aggregate\|from ggTrader.core.portfolio_optimizer" \
  src/ --include="*.py" | grep -v "__pycache__\|/core/wfo\|/core/orchestrator\|/core/fast_backtest\|/core/sensitivity\|/core/benchmarking\|/core/portfolio_optimizer"
```

Expected: only hits inside `utils/pipeline_phases.py` and `cli/cmd_production.py` — both being deleted in Task 6. If any other files appear, fix them before proceeding.

- [ ] **Step 1: Delete the cluster**

```bash
rm src/ggTrader/core/wfo.py \
   src/ggTrader/core/wfo_aggregate.py \
   src/ggTrader/core/orchestrator.py \
   src/ggTrader/core/orchestrator_utils.py \
   src/ggTrader/core/fast_backtest.py \
   src/ggTrader/core/sensitivity.py \
   src/ggTrader/core/benchmarking.py \
   src/ggTrader/core/portfolio_optimizer.py
```

- [ ] **Step 2: Run non-integration suite to verify no regressions in live path**

Run: `source .venv/bin/activate && python -m pytest tests/ -m "not integration" -q --ignore=tests/lab/`
Expected: same test counts as before this plan (3 pre-existing failures: `test_circuit_breaker_persistence`, `test_system_dry_run_cycle`, `test_persistence_logic`; no new failures).

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "refactor: delete old WFO core cluster (wfo, orchestrator, fast_backtest, sensitivity)

These 8 core/ files powered the old walk-forward optimization pipeline,
superseded by the new lab/ vectorbt bench. Zero live-path importers.
The live crypto engine (crypto_execution_engine, trade_tracker, etc.) is
untouched.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 6: Delete `pipeline/`, `backtesting/`, `utils/pipeline_phases.py`, old CLI commands

**Files to delete:**
- `src/ggTrader/pipeline/` (exit_tournament, param_grids, pipeline_runner, `__init__.py`)
- `src/ggTrader/backtesting/` (`__init__.py`, `wfo.py` — zero importers confirmed)
- `src/ggTrader/utils/pipeline_phases.py`
- `src/ggTrader/cli/cmd_research.py`
- `src/ggTrader/cli/cmd_backtest.py`
- `src/ggTrader/cli/cmd_production.py`
- `src/ggTrader/cli/cmd_report.py`

**Files to modify:**
- `src/ggTrader/cli/main.py` — remove 8 registration entries

**Verify before deleting:**

```bash
grep -rn "from ggTrader.pipeline\|from ggTrader.backtesting\|pipeline_phases\|cmd_research\|cmd_backtest\b\|cmd_production\|cmd_report" \
  src/ --include="*.py" | grep -v "__pycache__\|/pipeline/\|/backtesting/\|/cli/cmd_research\|/cli/cmd_backtest\|/cli/cmd_production\|/cli/cmd_report\|/utils/pipeline_phases"
```

Expected: only hits in `src/ggTrader/cli/main.py` (the entries we're about to remove). If any other file appears, fix it first.

- [ ] **Step 1: Remove old command registrations from `main.py`**

In `src/ggTrader/cli/main.py`, locate and remove the following 4 blocks (each block is 4 lines — the command name string, description, register function, and run function):

Blocks to remove (find by their command name strings):
- `"research"` block (4 lines)
- `"backtest"` block (4 lines — not `"backtest-strategy"`, leave that one)
- `"production"` block (4 lines)
- `"report"` block (4 lines)

After removing, run `ggt --help` to verify those 4 commands are gone and all others still appear.

Run: `source .venv/bin/activate && python ggt.py --help | grep -E "research|^  backtest$|production|report"`
Expected: **no output** (those commands are gone; `backtest-strategy` still appears separately).

- [ ] **Step 2: Delete the files**

```bash
rm -rf src/ggTrader/pipeline/
rm -rf src/ggTrader/backtesting/
rm src/ggTrader/utils/pipeline_phases.py
rm src/ggTrader/cli/cmd_research.py
rm src/ggTrader/cli/cmd_backtest.py
rm src/ggTrader/cli/cmd_production.py
rm src/ggTrader/cli/cmd_report.py
```

- [ ] **Step 3: Verify the CLI still starts cleanly**

Run: `source .venv/bin/activate && python ggt.py --help`
Expected: no ImportError; all live commands (`signals`, `status`, `trade`, `pnl-daily`, `trade-report`, `db`, `ingest`, `repair`, `cleanup`, `backtest-strategy`, `lab`) appear.

- [ ] **Step 4: Run full unit suite**

Run: `source .venv/bin/activate && python -m pytest tests/ -m "not integration" -q`
Expected: same pre-existing 3 failures; no new failures.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: delete pipeline/, backtesting/, old CLI commands (research/backtest/production/report)

Completes Plan 3 deletion. The 4 old CLI commands and supporting
infrastructure (pipeline/, utils/pipeline_phases.py, backtesting/) are
superseded by the lab/ bench. Live commands (signals, trade, pnl-daily,
backtest-strategy, etc.) are unaffected.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 7: Update spec, changelog, run full integration suite

- [ ] **Step 1: Run the full lab integration suite to confirm end-to-end health**

Run: `source .venv/bin/activate && python -m pytest tests/lab/ -m integration -q`
Expected: persist, harness-resume, and validation-gate tests pass (same as Plan 1).

- [ ] **Step 2: Mark Plan 3 executed in the spec**

In `docs/superpowers/specs/2026-06-15-vectorbt-lab-core-design.md`, update the Status line from:

```
**Status:** Plan 1 executed 2026-06-15 (momentum bench shipped on branch `lab-core-plan1`; validation gate passes, selections bit-identical). Plans 2 (wfo_tournament/signal family) and 3 (equity backfill + old-code deletion) pending.
```

to:

```
**Status:** Plan 1 executed 2026-06-15 (momentum bench + validation gate). Plan 3 executed 2026-06-16 (CachedYFinanceLoader bug fixed, S&P 500 backfilled into DB, research/WFO code cluster deleted). Plan 2 (wfo_tournament/signal family via from_signals) pending.
```

- [ ] **Step 3: Add changelog entry**

In `docs/changelog.md`, add under a new `## 2026-06-16` heading at the top (above the existing `## 2026-06-15` entry):

```markdown
## 2026-06-16

### Research: Lab Plan 3 — equity backfill + old research code deletion

- **Fixed** `CachedYFinanceLoader._interval_to_timedelta` AttributeError that forced
  every equity lab run to re-download ~600 symbols live (~2 min → <30s after fix).
- **Backfilled** full S&P 500 OHLCV history (2000–present, ~600 symbols) into
  TimescaleDB via `scripts/equity_backfill.py`. Future lab runs are DB-only.
- **Moved** `fetch_stock_ohlcv` + `STOCK_BASE_CONFIG` from `research/equity_wfo.py`
  into `lab/data.py`, making the `lab/` package fully self-contained.
- **Deleted** the old research/WFO code cluster: `research/`, `pipeline/`,
  `backtesting/`, `core/{wfo,orchestrator,fast_backtest,sensitivity,benchmarking,
  orchestrator_utils,wfo_aggregate,portfolio_optimizer}.py`,
  `utils/pipeline_phases.py`, and CLI commands `research`, `backtest`,
  `production`, `report`. ~26k lines removed. Live trading engine untouched.
  Pending: Plan 2 (signal-based `wfo_tournament` family via `from_signals`).
```

- [ ] **Step 4: Commit**

```bash
git add docs/changelog.md docs/superpowers/specs/2026-06-15-vectorbt-lab-core-design.md
git commit -m "docs: changelog + spec status for lab Plan 3 (backfill + cleanup)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- §9 "Cutover = validation gate passes. Then delete." → validation gate passed in Plan 1 ✓
- `research/` deletion → Task 4 ✓
- `backtest/vectorized.py` → explicitly kept (used by `cmd_backtest_strategy.py`) ✓
- `core/orchestrator.py`, `core/wfo.py`, `core/fast_backtest.py` → Task 5 ✓
- `pipeline/` → Task 6 ✓
- old `cli/cmd_research|cmd_backtest|cmd_production` → Task 6 ✓
- `utils/results_manager.py` → not found on disk; skip ✓
- `results/` files → not touched (lab persists to DB only, no files written) ✓
- Deferred live teardown (crypto_execution_engine, CCXT brokers, trade_tracker) → explicitly not in this plan ✓
- CachedYFinanceLoader bug → Task 1 ✓
- Equity backfill → Task 3 ✓
- `fetch_stock_ohlcv`/`STOCK_BASE_CONFIG` migration → Task 2 ✓

**Placeholder scan:** No TBDs. All code blocks are complete. Step 3 of Task 3 has a time estimate range (5-10 min) and an explicit retry note for rate-limits — this is genuine uncertainty about network/API behavior, not a placeholder.

**Type consistency:** `fetch_stock_ohlcv` signature in Task 2 matches the existing callers in `lab/data.py:load_ohlcv` and `lab/cli.py`. `STOCK_BASE_CONFIG` is a plain `dict[str, Any]` — consumed as `dict(STOCK_BASE_CONFIG)` in `lab/cli.py:run_lab`, unchanged.

**Risk notes:**
- Task 5's verify step may hit `pipeline_phases.py` importing `core/orchestrator.py` — that's expected (it's deleted in Task 6). The verify grep in Task 5 explicitly excludes `pipeline_phases.py` importers.
- `backtesting/` (separate from `backtest/`) confirmed zero importers via grep before adding to deletion list.
- `cmd_backtest` in `main.py` is separate from `cmd_backtest_strategy` — the rm command and main.py edits only touch the plain `backtest` entry, not `backtest-strategy`.
