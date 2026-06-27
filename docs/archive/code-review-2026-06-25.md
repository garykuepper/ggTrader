> **ARCHIVED 2026-06-27 — STALE.** All CRITICAL findings in this review (EnsembleConvictionSignal voters bug, check_concentration dead code, AVGO duplicate) were verified FIXED in current code. Retained for historical context only.

# Code Review: ggTrader

**Date:** 2026-06-25
**Reviewer:** Senior Engineer
**Scope:** Full codebase analysis

---

## CRITICAL BUGS

### 1. `EnsembleConvictionSignal` ignores the `voters` parameter

**File:** `src/ggTrader/lab/strategies/ensemble.py:306-370`

`_generate_signals_with_sizes()` hardcodes all 6 sub-signals (BB, RSI, EMA, MACD, VBB, MTF) regardless of `self.voters`. The constructor accepts `voters` and stores it, but the method never reads it. If someone configures `voters=FIVE_VOTERS` (the production default), MTF is still computed and counted in the vote. This silently reintroduces the "consistently harmful voter" the ablation study removed.

### 2. Duplicate `AVGO` in S&P 500 symbols

**File:** `src/ggTrader/data/core/stock_constants.py:37` and line 53

`AVGO` appears twice in `SP500_SYMBOLS`. This causes duplicate entries when the list is used as a universe, potentially skewing position sizing or signal aggregation.

### 3. `cmd_ingest.py` is non-functional dead code

**File:** `src/ggTrader/cli/cmd_ingest.py:15-36`

The `run_ingest` function creates a `PostgresIngestor` but never calls any method on it. The loop body is entirely commented out (`# ingestor.sync_symbol_ohlcv(sym)`). The `--days` CLI parameter is parsed but never referenced. This command exists, parses args, does nothing, and prints "Ingestion complete."

---

## HIGH SEVERITY

### 4. Hardcoded database credentials

**File:** `src/ggTrader/data/live/cached_yfinance_loader.py:28-32`

`_default_connection_string()` falls back to `"postgresql://ggtrader:ggtrader@localhost:5433/ggtrader"` when `GGTRADER_DB_URL` env var is unset. Plaintext credentials in source code. An attacker who gains repo read access gets the DB credentials.

### 5. Stale/dead architecture references in pyproject.toml

**File:** `pyproject.toml:71-92`

Mypy overrides reference 18+ modules (`ggTrader.core.instrument`, `ggTrader.execution.kraken_spot`, `ggTrader.features.base`, etc.) that do not exist in the source tree. These are remnants of a previous crypto-centric architecture. They create dead configuration and would mislead any new developer reading the config.

### 6. `_ema_combo_is_sharpe` creates full vbt.Portfolio per combo — a performance sink

**File:** `src/ggTrader/lab/strategies/signals.py:32`

Creates a full `vectorbt.Portfolio` object just to get a Sharpe ratio for 4 EMA combos per rebalance. In a WFO run with thousands of folds x rebalance dates, this adds massive overhead. A closed-form Sharpe calculation (mean/std of returns) would be orders of magnitude faster.

### 7. Crypto-to-equity transition is incomplete

Multiple files reference the old crypto focus:

- `src/ggTrader/data/core/constants.py` built around Kraken pairs, `STABLE_BASES`, Kraken-style `ZUSD` prefixes
- `src/ggTrader/data/historical/postgres_ingestor.py` is a Kraken OHLCV ingester (pair-splitting, quote-detection)
- `src/ggTrader/cli/cmd_ingest.py` still hardcodes `["BTC-USD", "ETH-USD"]`
- The project description says "algorithmic trading bot" but all active work is US equities

---

## MEDIUM SEVERITY

### 8. Duplicated threshold constant

`DEFAULT_THRESHOLD = 0.55` is duplicated in:

- `src/ggTrader/paper/feature_gate.py:12`
- `src/ggTrader/lab/train_gate.py` (hardcoded in training logic)

No single source of truth. If one changes, the other drifts silently.

### 9. Duplicated strategy registries

Strategy class-to-name mappings exist in two places:

- `src/ggTrader/lab/strategies/signals.py` (via `_build_signal_registry`)
- `src/ggTrader/lab/cli.py` (via `cls_map` dict)

Adding a new strategy requires updating both, which will be forgotten.

### 10. Bare `except Exception` patterns

Found in at least 5 files (`yfinance_loader.py`, `postgres_ingestor.py`, `cached_yfinance_loader.py`, `tiingo_loader.py`, `cmd_ingest.py`). These silently swallow errors, making debugging nearly impossible.

```python
# yfinance_loader.py:62
except Exception:
    df = pd.DataFrame()
```

### 11. Module-level mutable state

- `src/ggTrader/lab/persist.py:_ENGINE` — module-level mutable global, thread-unsafe
- `src/ggTrader/data/core/index_constituents.py:_history_cache` — module-level cache, no invalidation mechanism
- `src/ggTrader/lab/data.py:STOCK_BASE_CONFIG` — mutable dict at module level, any importer can mutate it for all users

### 12. Monolithic functions >150 lines

- `PaperTrader.run` (`src/ggTrader/paper/trader.py:34-149`)
- `run_wfo` (`src/ggTrader/lab/wfo.py`, ~250 lines with deeply nested fold loop)
- `run_lab` (`src/ggTrader/lab/cli.py`, 183 lines, three modes + two strategy types)
- `run_sweep` (`src/ggTrader/lab/sweep.py`, ~100 lines, two code paths)

### 13. No `__all__` in any `__init__.py`

Almost all package `__init__.py` files are empty. This means `from ggTrader.lab import *` imports everything, including internal helpers. No public API surface is defined.

---

## LOW SEVERITY

### 14. Missing test coverage for critical components

- `CachedYFinanceLoader` — no tests
- `PostgresIngestor` — no tests
- `TiingoDataLoader` — minimal/cursory tests only
- `cmd_ingest.py`, `cmd_db.py` — no tests
- No integration tests are run by default (excluded via `-m 'not integration'` marker)

### 15. RSI recomputed 3+ times per strategy evaluation

`src/ggTrader/lab/strategies/indicators.py` recomputes RSI independently in `rsi_signals`, `rsi_strength`, and `_weekly_rsi`. For a 500-symbol universe x 6+ strategy configurations in WFO, this is significant redundant computation.

### 16. Inconsistent type annotations

- Widespread use of `Dict[str, Any]` / `dict[str, Any]` where `TypedDict` would be appropriate
- No `Final`, `Literal`, or `TypeVar` usage
- `get_account()` returns bare `dict` with no schema

### 17. `print()` for error reporting instead of `logging`

Multiple files use `print()` for diagnostics/errors instead of Python's `logging` module. This means log levels, formatting, and output routing are not configurable.

### 18. PostgreSQL connection leak risk

`src/ggTrader/data/live/cached_yfinance_loader.py:_cache_to_db` opens a connection with manual `conn.close()` — but exceptions before the close call leak connections. No context manager (`with conn:`) usage.

### 19. Market-hours constants declared but never used

`src/ggTrader/data/core/stock_constants.py:15-19` defines `MARKET_OPEN_HOUR`, `MARKET_CLOSE_HOUR`, etc., but no code anywhere checks market hours before trading.

### 20. `conftest.py` is minimal

Only sets `matplotlib.use("Agg")`. No fixtures for DB mocking, test data generation, or shared strategy configuration.

---

## WHAT'S DONE WELL

- **Lookahead protection** — `regime.py` shifts values by 1, `leak_check` exists in harness, point-in-time universe selection via SP500 constituent history
- **Statistical rigor** — NDH, DSR gates, WFE circuit breaker, composite scoring — production-grade overfitting defense
- **Per-symbol cache freshness** — `CachedYFinanceLoader` evaluates freshness per-symbol, avoiding truncation when new symbols are added
- **Risk guardrails** — `RiskGuard` with drawdown halt, daily loss limit, concentration limits, position sizing
- **Vectorized design** — No per-bar iteration anywhere; all signal computation is numpy/pandas vectorized
- **Comprehensive documentation** — `docs/architecture.md`, `docs/cli_reference.md`, `docs/changelog.md` are thorough
- **Research-first workflow** — Strategies are validated in the lab before reaching paper trading, with parameter sweep + WFO + statistical gates

---

## SUMMARY

| Severity | Count | Action |
|----------|-------|--------|
| Critical | 3 | Fix immediately |
| High | 4 | Fix before next deploy |
| Medium | 6 | Schedule |
| Low | 7 | Nice to have |

**Critical:** Fix the `EnsembleConvictionSignal` voters bug (MTF silently pollutes the 5-voter ensemble), deduplicate `AVGO`, and either repair or remove `cmd_ingest.py`.

**High:** Remove hardcoded DB credentials, clean up stale pyproject.toml config, optimize `_ema_combo_is_sharpe`, and finish the crypto-to-equity migration.
