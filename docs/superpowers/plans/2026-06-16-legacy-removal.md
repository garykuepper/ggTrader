# Legacy Code Removal & Lab-First Architecture

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove all legacy trading/backtesting/strategy code, keeping only the lab research bench and its data dependencies. Shrink ~18.6K LOC of legacy source to ~0, leaving a clean lab-first codebase.

**Architecture:** The lab (`src/ggTrader/lab/`) is fully self-contained — it imports only `data.core.index_constituents`, `data.live.yfinance_loader`, `data.live.cached_yfinance_loader`, `data.historical.timescaledb_loader`, and `utils.config`/`utils.paths`. Everything else in `src/ggTrader/` is dead code. The CLI will be rewritten from the legacy Typer+argparse shim to a single `ggt lab` command.

**Tech Stack:** Python, vectorbt, Typer CLI, pytest

## Global Constraints

- The lab tests (`tests/lab/`) must continue to pass after every task.
- Do NOT modify any file under `src/ggTrader/lab/` or `tests/lab/`.
- Keep `data/` modules that lab transitively imports (see dependency chain below).
- Keep `scripts/equity_backfill.py` (imports lab). All other `scripts/*.py` are legacy.
- Docker image rebuild is out of scope for this plan (separate step after cleanup).

### Lab dependency chain (KEEP these files):

```
lab/persist.py → utils/config.py → utils/paths.py
lab/data.py → data/core/index_constituents.py (no further ggTrader deps)
lab/data.py → data/live/yfinance_loader.py → data/core/base_loader.py (leaf)
                                            → data/core/stock_constants.py (leaf)
lab/data.py → data/live/cached_yfinance_loader.py → data/live/yfinance_loader.py (above)
                                                   → data/historical/timescaledb_loader.py → data/core/base_loader.py (above)
                                                                                           → utils/config.py (above)
```

### Files to KEEP (source):
- `src/ggTrader/__init__.py`
- `src/ggTrader/lab/` (entire directory, untouched)
- `src/ggTrader/data/__init__.py`
- `src/ggTrader/data/core/__init__.py`
- `src/ggTrader/data/core/base_loader.py`
- `src/ggTrader/data/core/constants.py` (used by `postgres_ingestor`)
- `src/ggTrader/data/core/index_constituents.py`
- `src/ggTrader/data/core/stock_constants.py`
- `src/ggTrader/data/historical/__init__.py`
- `src/ggTrader/data/historical/timescaledb_loader.py`
- `src/ggTrader/data/historical/postgres_ingestor.py` (data pipeline — still useful for ingesting OHLCV)
- `src/ggTrader/data/live/__init__.py` (will be cleaned — remove legacy re-exports)
- `src/ggTrader/data/live/yfinance_loader.py`
- `src/ggTrader/data/live/cached_yfinance_loader.py`
- `src/ggTrader/utils/__init__.py`
- `src/ggTrader/utils/config.py`
- `src/ggTrader/utils/paths.py`
- `src/ggTrader/utils/db_engine.py` (used by `cmd_db`, small utility)
- `src/ggTrader/cli/__init__.py`
- `src/ggTrader/cli/main.py` (will be rewritten)
- `src/ggTrader/cli/cmd_db.py` (will be trimmed — remove legacy trade_tracker/wfo_cache deps)
- `src/ggTrader/cli/cmd_ingest.py` (data pipeline — keep)

### Files to KEEP (tests):
- `tests/lab/` (entire directory, untouched)
- `tests/test_index_constituents.py` (tests kept module)
- `tests/test_timescaledb_loader.py` (tests kept module)
- `tests/conftest.py` (will be simplified — remove legacy fixture if unused by kept tests)

### Files to KEEP (scripts):
- `scripts/equity_backfill.py` (imports lab)
- `scripts/daily_pnl_report.sh`, `scripts/daily_pnl_report_stocks.sh` (cron — keep shells, they'll just fail gracefully if container lacks code)
- All `.sh` scripts that don't import Python legacy code

### Files to REMOVE (source — ~18K LOC):
- `src/ggTrader/backtest/` (entire)
- `src/ggTrader/core/` (entire)
- `src/ggTrader/execution/` (entire)
- `src/ggTrader/features/` (entire)
- `src/ggTrader/indicators/` (entire)
- `src/ggTrader/strategies/` (entire)
- `src/ggTrader/portfolio/` (entire)
- `src/ggTrader/risk/` (entire)
- `src/ggTrader/sizing/` (entire)
- `src/ggTrader/config/` (entire)
- `src/ggTrader/data/cache/` (entire — wfo_cache)
- `src/ggTrader/data/sources/` (entire — kraken_futures)
- `src/ggTrader/data/store/` (entire — migrations)
- `src/ggTrader/data/live/exchange_loader.py`
- `src/ggTrader/data/live/cached_loader.py`
- `src/ggTrader/data/core/venue_listings.py`
- `src/ggTrader/cli/cmd_backtest_strategy.py`
- `src/ggTrader/cli/cmd_cleanup.py`
- `src/ggTrader/cli/cmd_dashboard.py`
- `src/ggTrader/cli/cmd_pnl_daily.py`
- `src/ggTrader/cli/cmd_repair.py`
- `src/ggTrader/cli/cmd_signals.py`
- `src/ggTrader/cli/cmd_status.py`
- `src/ggTrader/cli/cmd_trade.py`
- `src/ggTrader/cli/cmd_trade_report.py`
- `src/ggTrader/cli/entrypoints.py`
- `src/ggTrader/cli/script_entry.py`
- `src/ggTrader/utils/fear_greed.py`
- `src/ggTrader/utils/kraken_ledger.py`
- `src/ggTrader/utils/live_metrics.py`
- `src/ggTrader/utils/notifier.py`
- `src/ggTrader/utils/pipeline_run_history.py`
- `src/ggTrader/utils/pipeline_status_logger.py`
- `src/ggTrader/utils/plotting.py`
- `src/ggTrader/utils/pnl_report_builder.py`
- `src/ggTrader/utils/report_generator.py`
- `src/ggTrader/utils/result_db_manager.py`
- `src/ggTrader/utils/results_manager.py`
- `src/ggTrader/utils/run_config.py`
- `src/ggTrader/utils/setup.py`
- `src/ggTrader/utils/state_manager.py`

### Files to REMOVE (tests — ~29 files):
- ALL files in `tests/` except `tests/lab/`, `tests/test_index_constituents.py`, `tests/test_timescaledb_loader.py`
- `tests/backtesting/` (entire)
- `tests/integration/` (entire)
- `tests/strategies/` (entire)
- `tests/unit/` (entire)

### Files to REMOVE (scripts — legacy Python):
- `scripts/auto_trader.py`
- `scripts/phase4_comparison.py`
- `scripts/view_results.py`
- `scripts/equity_wfo_research.py`
- `scripts/sp500_monthly_walkforward.py`
- `scripts/run_cross_sectional_research.py`
- `scripts/run_walk_forward_optimization.py`
- `scripts/gate_replay.py`
- `scripts/investigate_live_trades.py`
- `scripts/strategy_usage_stats.py`
- `scripts/binanceus_smoke_test.py`
- `scripts/coin_correlation_matrix.py`
- `scripts/backfill_kraken_futures.py`
- `scripts/backfill_kraken_csv.py`
- `scripts/backfill_spot_btc.py`
- `scripts/backfill_spot_coinbase.py`
- `scripts/backfill_binanceus.py`
- `scripts/backfill_binanceus_universe.py`
- `scripts/backfill_binanceus_from_kraken.py`
- `scripts/binanceus_spread_depth.py`
- `scripts/cleanup_project.py`
- `scripts/scorecard_step1.py`
- `scripts/analyze_portfolio_performance.py`
- `scripts/analyze_profile.py`
- `scripts/profile_wfo.sh`
- `scripts/archive/` (entire)

### Scripts to KEEP:
- `scripts/equity_backfill.py`
- `scripts/update_universe_ccxt.py` (data maintenance — no ggTrader imports)
- `scripts/update_venue_listings.py` (data maintenance)
- `scripts/daily_pnl_report.sh` / `scripts/daily_pnl_report_stocks.sh` (cron wrappers)
- `scripts/update_docker.sh` (infra)
- `scripts/amcfilebot.sh` (media pipeline, unrelated)
- Any other `.sh` scripts not listed above

---

### Task 1: Delete legacy source modules

**Files:**
- Delete: `src/ggTrader/backtest/` (entire)
- Delete: `src/ggTrader/core/` (entire)
- Delete: `src/ggTrader/execution/` (entire)
- Delete: `src/ggTrader/features/` (entire)
- Delete: `src/ggTrader/indicators/` (entire)
- Delete: `src/ggTrader/strategies/` (entire)
- Delete: `src/ggTrader/portfolio/` (entire)
- Delete: `src/ggTrader/risk/` (entire)
- Delete: `src/ggTrader/sizing/` (entire)
- Delete: `src/ggTrader/config/` (entire)
- Delete: `src/ggTrader/data/cache/` (entire)
- Delete: `src/ggTrader/data/sources/` (entire)
- Delete: `src/ggTrader/data/store/` (entire)
- Delete: `src/ggTrader/data/live/exchange_loader.py`
- Delete: `src/ggTrader/data/live/cached_loader.py`
- Delete: `src/ggTrader/data/core/venue_listings.py`
- Delete: Legacy utils (see REMOVE list above — 13 files)
- Delete: Legacy CLI commands (see REMOVE list above — 10 files)
- Modify: `src/ggTrader/data/live/__init__.py` — remove legacy re-exports

- [ ] **Step 1: Delete legacy source directories**

```bash
cd /home/flynn/ggTrader
rm -rf src/ggTrader/backtest src/ggTrader/core src/ggTrader/execution \
       src/ggTrader/features src/ggTrader/indicators src/ggTrader/strategies \
       src/ggTrader/portfolio src/ggTrader/risk src/ggTrader/sizing \
       src/ggTrader/config src/ggTrader/data/cache src/ggTrader/data/sources \
       src/ggTrader/data/store
```

- [ ] **Step 2: Delete legacy data/live loaders and venue_listings**

```bash
rm src/ggTrader/data/live/exchange_loader.py \
   src/ggTrader/data/live/cached_loader.py \
   src/ggTrader/data/core/venue_listings.py
```

- [ ] **Step 3: Delete legacy utils**

```bash
rm src/ggTrader/utils/fear_greed.py \
   src/ggTrader/utils/kraken_ledger.py \
   src/ggTrader/utils/live_metrics.py \
   src/ggTrader/utils/notifier.py \
   src/ggTrader/utils/pipeline_run_history.py \
   src/ggTrader/utils/pipeline_status_logger.py \
   src/ggTrader/utils/plotting.py \
   src/ggTrader/utils/pnl_report_builder.py \
   src/ggTrader/utils/report_generator.py \
   src/ggTrader/utils/result_db_manager.py \
   src/ggTrader/utils/results_manager.py \
   src/ggTrader/utils/run_config.py \
   src/ggTrader/utils/setup.py \
   src/ggTrader/utils/state_manager.py
```

- [ ] **Step 4: Delete legacy CLI commands**

```bash
rm src/ggTrader/cli/cmd_backtest_strategy.py \
   src/ggTrader/cli/cmd_cleanup.py \
   src/ggTrader/cli/cmd_dashboard.py \
   src/ggTrader/cli/cmd_pnl_daily.py \
   src/ggTrader/cli/cmd_repair.py \
   src/ggTrader/cli/cmd_signals.py \
   src/ggTrader/cli/cmd_status.py \
   src/ggTrader/cli/cmd_trade.py \
   src/ggTrader/cli/cmd_trade_report.py \
   src/ggTrader/cli/entrypoints.py \
   src/ggTrader/cli/script_entry.py
```

- [ ] **Step 5: Clean data/live/__init__.py**

Replace content with:

```python
"""Live data fetching via yfinance."""
```

(Remove the `CachedExchangeLoader` and `LiveExchangeLoader` re-exports.)

- [ ] **Step 6: Verify lab imports still resolve**

```bash
cd /home/flynn/ggTrader
python3 -c "from ggTrader.lab.cli import run_lab; print('OK')"
```

Expected: `OK` — all transitive imports resolve.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor: remove legacy trading/backtesting code (~18K LOC)

Keep only lab/ research bench and its data dependencies.
Removed: backtest, core, execution, features, indicators, strategies,
portfolio, risk, sizing, config, legacy CLI commands, legacy utils."
```

---

### Task 2: Delete legacy tests and scripts

**Files:**
- Delete: All test files listed in REMOVE (tests) above
- Delete: All script files listed in REMOVE (scripts) above
- Modify: `tests/conftest.py` — simplify to matplotlib backend only

- [ ] **Step 1: Delete legacy test directories**

```bash
cd /home/flynn/ggTrader
rm -rf tests/backtesting tests/integration tests/strategies tests/unit
```

- [ ] **Step 2: Delete legacy top-level test files**

```bash
rm tests/test_bear_market_forgiveness.py \
   tests/test_broadcasting.py \
   tests/test_cagr_benchmark.py \
   tests/test_circuit_breaker.py \
   tests/test_cli_routing.py \
   tests/test_exchange_loader_dedup.py \
   tests/test_exit_tournament_wfo.py \
   tests/test_fast_backtest.py \
   tests/test_fast_backtest_fix.py \
   tests/test_holdout.py \
   tests/test_live_trading_system.py \
   tests/test_live_trading_unit.py \
   tests/test_metrics_returns_extraction.py \
   tests/test_orchestrator.py \
   tests/test_pipeline_new_functionality.py \
   tests/test_profit_factor_raw.py \
   tests/test_rank_composite.py \
   tests/test_refactor_verification.py \
   tests/test_sensitivity_trade_gate_alignment.py \
   tests/test_signals.py \
   tests/test_state_manager.py \
   tests/test_utils_managers.py \
   tests/test_utils_plotting.py \
   tests/test_utils_setup.py \
   tests/test_vbt_patches.py \
   tests/test_vectorized_architecture.py \
   tests/test_venue_listings.py \
   tests/test_wfo_aggregate_gates.py \
   tests/test_wfo_robustness_selection.py \
   tests/test_wfo_trade_counts_gate.py
```

- [ ] **Step 3: Simplify conftest.py**

Replace `tests/conftest.py` with:

```python
import matplotlib

matplotlib.use("Agg")
```

(The `sample_ohlcv_data` fixture is not used by any kept test.)

- [ ] **Step 4: Delete legacy Python scripts**

```bash
rm scripts/auto_trader.py \
   scripts/phase4_comparison.py \
   scripts/view_results.py \
   scripts/equity_wfo_research.py \
   scripts/sp500_monthly_walkforward.py \
   scripts/run_cross_sectional_research.py \
   scripts/run_walk_forward_optimization.py \
   scripts/gate_replay.py \
   scripts/investigate_live_trades.py \
   scripts/strategy_usage_stats.py \
   scripts/binanceus_smoke_test.py \
   scripts/coin_correlation_matrix.py \
   scripts/backfill_kraken_futures.py \
   scripts/backfill_kraken_csv.py \
   scripts/backfill_spot_btc.py \
   scripts/backfill_spot_coinbase.py \
   scripts/backfill_binanceus.py \
   scripts/backfill_binanceus_universe.py \
   scripts/backfill_binanceus_from_kraken.py \
   scripts/binanceus_spread_depth.py \
   scripts/cleanup_project.py \
   scripts/scorecard_step1.py \
   scripts/analyze_portfolio_performance.py \
   scripts/analyze_profile.py \
   scripts/profile_wfo.sh
rm -rf scripts/archive
```

- [ ] **Step 5: Run kept tests**

```bash
python3 -m pytest tests/lab/ tests/test_index_constituents.py tests/test_timescaledb_loader.py -v --tb=short
```

Expected: All lab tests pass (28u + 10i), plus index_constituents and timescaledb_loader tests.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor: remove legacy tests and scripts

Keep lab tests (38), index_constituents, timescaledb_loader.
Remove 29 legacy test files, 25 legacy scripts, scripts/archive/."
```

---

### Task 3: Rewrite CLI and pyproject.toml entry points

**Files:**
- Modify: `src/ggTrader/cli/main.py` — replace legacy Typer shim with clean `ggt lab` command
- Modify: `pyproject.toml` — remove dead entry points
- Modify: `src/ggTrader/cli/cmd_db.py` — remove imports of deleted modules (trade_tracker, result_db_manager, wfo_cache)

**Interfaces:**
- Consumes: `lab.cli.run_lab` (unchanged), `cli.cmd_db` (trimmed), `cli.cmd_ingest` (unchanged)

- [ ] **Step 1: Rewrite main.py**

Replace `src/ggTrader/cli/main.py` with:

```python
"""``ggt`` CLI — lab-first research toolkit."""

from __future__ import annotations

import sys

import typer

app = typer.Typer(
    name="ggt",
    help="ggTrader — vectorbt research lab",
    no_args_is_help=True,
    add_completion=False,
)


@app.command(
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
        "help_option_names": [],
    },
)
def lab(ctx: typer.Context) -> None:
    """Run a lab strategy walk-forward."""
    from ggTrader.lab.cli import run_lab

    run_lab(list(ctx.args))


@app.command(
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
        "help_option_names": [],
    },
)
def ingest(ctx: typer.Context) -> None:
    """Ingest OHLCV data into TimescaleDB."""
    import argparse

    from ggTrader.cli.cmd_ingest import register_ingest_parser, run_ingest

    parser = argparse.ArgumentParser(prog="ggt", add_help=False)
    subs = parser.add_subparsers(dest="command")
    register_ingest_parser(subs)
    ns = parser.parse_args(["ingest", *ctx.args])
    run_ingest(ns)


@app.command(
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
        "help_option_names": [],
    },
)
def db(ctx: typer.Context) -> None:
    """TimescaleDB management commands."""
    import argparse

    from ggTrader.cli.cmd_db import register_db_parser, run_db

    parser = argparse.ArgumentParser(prog="ggt", add_help=False)
    subs = parser.add_subparsers(dest="command")
    register_db_parser(subs)
    ns = parser.parse_args(["db", *ctx.args])
    run_db(ns)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Trim cmd_db.py — remove legacy imports**

In `src/ggTrader/cli/cmd_db.py`, find all references to deleted modules and remove the corresponding subcommands:

Remove these subcommand registrations and their handler blocks:
- `sync-live` (uses `TradeTracker`)
- `migrate-wfo-cache` (uses `wfo_cache`)
- `purge-wfo-cache` (uses `wfo_cache`)

Keep: `diag`, `clean`, `truncate`, `compression`, `export`, and any subcommands that only use `sqlalchemy` + `utils.config`.

Verify no remaining `from ggTrader.core` or `from ggTrader.utils.result_db_manager` imports.

- [ ] **Step 3: Update pyproject.toml entry points**

Replace the `[project.scripts]` section with:

```toml
[project.scripts]
ggt = "ggTrader.cli.main:main"
```

Remove all `ggtrader-*` entry points (backtest, wfo, sensitivity, pipeline, compare-strategies, view-results, pipeline-status, manage-data, manage-db, live).

- [ ] **Step 4: Verify CLI works**

```bash
python3 -c "from ggTrader.cli.main import main; print('CLI loads OK')"
python3 -m ggTrader.cli.main --help
```

Expected: Shows `lab`, `ingest`, `db` commands.

- [ ] **Step 5: Run all kept tests**

```bash
python3 -m pytest tests/ -v --tb=short
```

Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor: rewrite CLI for lab-first architecture

ggt lab: run lab walk-forward strategies
ggt ingest: OHLCV data ingestion (kept)
ggt db: TimescaleDB management (trimmed)
Removed 10 legacy entry points from pyproject.toml."
```

---

### Task 4: Clean up docs and agents.md

**Files:**
- Modify: `agents.md` — remove references to deleted modules, update architecture description
- Modify: `docs/roadmap.md` — mark legacy sections as archived, update status table
- Modify: `docs/architecture.md` — rewrite for lab-first layout
- Modify: `docs/cli_reference.md` — rewrite for 3-command CLI
- Delete: `docs/live_trading_guide.md` (all legacy)
- Delete: `docs/installation.md` (if it references legacy setup)
- Keep: `docs/equity_monthly_walkforward.md` (research results, reference)
- Keep: `docs/changelog.md` (history)

- [ ] **Step 1: Update agents.md**

Remove or update sections that reference:
- The 3-phase WFO pipeline (Phase 1/2/3 optimization)
- `indicators/strategies.py`, `core/orchestrator.py`, `core/wfo.py`
- Legacy CLI commands (`trade`, `signals`, `backtest-strategy`, etc.)
- The live trading Docker workflow (`docker compose run --rm ggtrader_live python -u ggt.py research`)

Add a section describing the lab architecture:
- `lab/` is the research bench
- `ggt lab --strategy <name>` runs walk-forward
- Data comes from yfinance (equities) + TimescaleDB (crypto OHLCV)
- Results persisted to `lab_runs` / `lab_periods` tables

- [ ] **Step 2: Rewrite docs/architecture.md**

Replace the legacy module map with the current layout:

```
src/ggTrader/
├── lab/              # Research bench (vectorbt-first)
│   ├── cli.py        # CLI entry point
│   ├── data.py       # Universe + OHLCV loading
│   ├── harness.py    # Walk-forward driver
│   ├── metrics.py    # Performance analytics
│   ├── persist.py    # DB persistence
│   ├── simulate.py   # Vectorized portfolio simulation (vbt)
│   ├── strategy.py   # Strategy protocol + config
│   └── strategies/   # Strategy implementations
├── data/             # Data loading infrastructure
│   ├── core/         # Base loader, SP500 constituents, constants
│   ├── historical/   # TimescaleDB loader + ingestor
│   └── live/         # yfinance loader
├── utils/            # Config, paths, DB engine
└── cli/              # CLI (ggt lab | ingest | db)
```

- [ ] **Step 3: Rewrite docs/cli_reference.md**

Document the 3 remaining commands:
- `ggt lab --strategy <name> [--eval-start DATE] [--eval-end DATE] [--top-n N] [--lookback N] [--skip N] [--max-stocks N]`
- `ggt ingest [--days N]`
- `ggt db <diag|clean|truncate|compression|export>`

- [ ] **Step 4: Update docs/roadmap.md**

Add a prominent note at the top: "Legacy code removed 2026-06-16. The codebase is now lab-only. §2a-c (WFO pipeline, Phase 2, TRX) are archived — they reference deleted code."

Mark completed: "Strategy library redesign (§2d) — infrastructure shipped (lab bench). Signal research is the next step."

- [ ] **Step 5: Delete docs/live_trading_guide.md**

```bash
rm docs/live_trading_guide.md
```

- [ ] **Step 6: Check docs/installation.md**

If it references legacy setup, rewrite for lab-only. If it's generic (pip install, Docker), keep and trim.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "docs: update for lab-first architecture

Rewrite architecture, CLI reference, roadmap for post-cleanup state.
Remove live_trading_guide (legacy). Update agents.md."
```

---

### Task 5: Rebuild Docker image and final validation

**Files:**
- Modify: `Dockerfile` — simplify if legacy deps can be dropped
- Modify: `.dockerignore` — no changes expected

- [ ] **Step 1: Check if Dockerfile needs changes**

Review `Dockerfile` for references to deleted files or unnecessary dependencies. The `COPY . .` should just work with fewer files. Check if `ggt.py` still works as entry point.

- [ ] **Step 2: Rebuild Docker image**

```bash
cd /home/flynn/ggTrader
docker compose build ggtrader_live
```

Expected: Build succeeds.

- [ ] **Step 3: Verify lab runs in container**

```bash
docker compose run --rm -w /app ggtrader_live python -c "from ggTrader.lab.cli import run_lab; print('OK')"
docker compose run --rm -w /app ggtrader_live python -m pytest tests/lab/ --tb=short -q
```

Expected: Import succeeds, all lab tests pass.

- [ ] **Step 4: Verify CLI in container**

```bash
docker compose run --rm -w /app ggtrader_live python ggt.py --help
docker compose run --rm -w /app ggtrader_live python ggt.py lab --help
```

Expected: Shows 3 commands (lab, ingest, db).

- [ ] **Step 5: LOC count — before/after comparison**

```bash
find src/ggTrader -name '*.py' | xargs wc -l | tail -1
find tests -name '*.py' | xargs wc -l | tail -1
```

Report the final LOC count (expect ~1.5K source, ~800 tests).

- [ ] **Step 6: Commit (if Dockerfile changed)**

```bash
git add -A
git commit -m "build: rebuild Docker image for lab-first codebase"
```

---

## Post-Cleanup Checklist

After all tasks complete:
- [ ] `python3 -m pytest tests/ -v` — all tests pass
- [ ] `ggt lab --help` — shows strategy choices
- [ ] `ggt --help` — shows lab, ingest, db
- [ ] No import errors from any kept module
- [ ] Docker image builds and lab tests pass inside container
- [ ] `ruff check src/ tests/` — no lint errors
