# ggTrader Agent Guidelines

This document serves as the consolidated source of truth for all AI assistants (Gemini, Claude, Cursor, etc.) working on the **ggTrader** project. 

---

## 1. Project Context & Core Architecture

ggTrader is an algorithmic crypto trading bot designed for Binance.US and Kraken. It utilizes a Walk-Forward Optimization (WFO) pipeline, a tiered regime filtering system, and mirrors live trading state to TimescaleDB.

* **Unified CLI**: All operations are routed through `ggt.py`.
* **Execution Engine**: The live trading bot (`ggt trade`) is a long-running process that manages its own lifecycle, including monthly recalibrations.
* **State Management**:
  * **Primary Logs**: Uses `data/live/` (inside the container) for trade logs, balance snapshots, and active positions.
  * **Database Mirroring**: Live trade events (buy/sell), orders, and balance snapshots are mirrored to TimescaleDB in real-time for observability.
  * **Backfill**: Use `ggt db sync-live` to import existing CSV history into the database.
* **Observability (Grafana)**:
  * **Dashboard**: Accessible at `http://localhost:3002`.
  * **Data Source**: Connects to the `ggtrader` TimescaleDB instance via `host.docker.internal:5433` (within Docker) or `localhost:5433` (on host).
  * **Panels**: Real-time Equity Curve, PnL per Trade (categorized dots), and Recent Closed Trades.
  * **Provisioning**: Configurations are stored in `grafana/provisioning/`.

---

## 2. Rules & Deployment Nuances

* **Docker-First Environment**: 
  * The `ggtrader_live` container **does not** volume-mount the `src/` directory.
  * **Updates**: Code changes made on the host must be copied into the container via `docker cp` (e.g., `docker cp src/ggTrader/path/to/file.py ggtrader_live:/app/src/ggTrader/path/to/file.py`) or applied by rebuilding the container (`docker compose build --no-cache && docker compose up -d`).
  * **Logs**: Active container logs are written to `/app/logs/live_trader.log` inside the container, which maps to the host's `logs/` directory.
* **Monthly Recalibration (WFO)**:
  * **Automation**: On the 1st of every month, the `ExecutionEngine` internally triggers a full research and production pipeline run.
  * **Process**:
    1. **Phase 1 (WFO)**: Optimizes parameters for all symbols. Note: The Regime Filter is **NOT** applied during per-fold optimization.
    2. **Phase 2 (Validation)**: Validates selected parameters in a combined portfolio. The Regime Filter **IS** applied here.
    3. **Phase 3 (Recent Data)**: Evaluates performance on the most recent data (YTD) with filters active.
  * **Reloading**: The bot reloads the new parameters automatically once Phase 2/3 complete. No container restart is required.
* **WFO Cache**:
  * Run `ggt db purge-wfo-cache` when changing scoring configurations (composite weights, fold consistency, OOS alpha, N_SPLITS) since cached results use old settings.
  * The WFO cache resides in the TimescaleDB `wfo_cache` table.
* **Market Regime Filtering**:
  * The bot employs a tiered correlation-based filter to prevent trading in unfavorable market conditions.
  * *Note: The historical BTC regime gating filter was disabled on 2026-05-23, but the bull/bear status is kept in daily PnL reports for context.*
* **Daily Operations**:
  * **PnL Reports**: Triggered at 06:00 AM local time via `scripts/daily_pnl_report.sh`. Reports include BTC/ETH prices and the Fear & Greed index, pulled live via `ccxt`.
  * **Sync**: The report builder automatically syncs recent trade history from the configured exchange to ensure local CSVs are accurate.

---

## 3. Development Guidelines & Config Changes

* **Config Changes**: Test one at a time with a research run between each. Do not bundle multiple config tweaks so that results can be properly attributed.
* **New Strategies**: Can be bundled together since they are purely additive and do not affect existing strategy scoring.
* **Research Runs**: Run inside Docker (`docker compose run --rm ggtrader_live python -u ggt.py research`) to ensure `host.docker.internal` DB connections resolve correctly.
* **Backtesting Architecture**:
  * **Production per-coin pipeline**: `FastBacktest` is the engine for backtesting, sensitivity analysis, and the live monthly recalibration WFO. The old Optuna-based `WalkForwardOptimizer` for this pipeline is archived.
  * **Experimental cross-sectional strategies** (`strategies/momentum/`, `strategies/regime/`): use the standalone `WalkForwardOptimizer` in `src/ggTrader/backtesting/wfo.py`, which drives `vectorbt` directly (not FastBacktest). This is a separate research track from the production pipeline — do not conflate the two.
  * Always pass `config=CONSTANTS` for portfolio-level settings; keep signal `params` separate.
  * Use `cash_sharing=True` and `group_by=True` — never independent cash pools per symbol.
  * The `Trading` engine is legacy (paper/live trading only).
* **Dynamic Mover Masking**:
  * Use `build_mover_mask(ohlcv, config, top_n=N)` for daily top-N mover masks.
  * Pass masks to `FastBacktest(mover_mask=mask)` to zero out entries for non-qualifying symbols.
  * Masks are precomputed via a single SQL query (`get_daily_mover_mask`), not per-bar.

---

## 4. Coding Standards

### Python Standards
1. **Strict PEP 8 Adherence**:
   * Use 4 spaces for indentation.
   * Maximum line length of 88-100 characters (Ruff/Black format).
   * Snake_case for functions/variables, CamelCase for classes, UPPER_CASE for constants.
   * **Filenames**: Must be lowercase with underscores (e.g., `trading.py`, not `Trading.py`).
   * Group imports: Standard library, Third-party, Local (alphabetically sorted).
2. **Type Hinting**: Use type hints for all function arguments and return values.
3. **Documentation**:
   * **Module Docstrings**: One-line summary of the file's purpose at the top.
   * **Function/Class Docstrings**: Brief summary of purpose (Google Style).
   * **Inline Comments**: Explain *WHY*, not *WHAT*. Avoid obvious comments.
4. **Single Source of Truth**: Core logic must live in `src/`. Do not duplicate definitions across files.
5. **Orchestration Boilerplate**: Scripts in `scripts/` must follow a standard template: `sys.path` setup, `main()` orchestration, and `argparse` for inputs. Every script must handle the `--help` flag gracefully.
6. **Results Management**: Use `ResultsManager` for all artifact, metric, and plot saving to ensure consistent output directory structures and metadata logging.
7. **Symbol Normalization**:
   * All scripts, notebooks, and models MUST use standardized asset symbols (e.g., `BTC`, `ETH`).
   * Use the `SYMBOL_MAPPING` table (consolidated in `data_manager.py`) to convert exchange-specific prefixed symbols (e.g., `XBT`, `XETH`, `ZUSD`) to standard tickers.
   * Output artifacts saved to `data/` or returned to the user must use the normalized symbol.
8. **SQL Query Patterns**:
   * When querying the `ohlcv` table, use standard PostgreSQL/TimescaleDB functions for asset separation:
     * Asset Part: `split_part(symbol, '-', 1)`
     * Quote Part: `split_part(symbol, '-', 2)`
   * Aggregation: Prefer Notional Volume (`volume * close`) for cross-asset ranking to account for price discrepancies.
9. **Path Safety**: Always use `os.path.join` or `pathlib.Path` for file paths. Resolve project root dynamically.
10. **Error Handling**: Data loading functions must raise descriptive exceptions (e.g., `ValueError`, `FileNotFoundError`) on failure. Avoid returning `None` from functions expected to return iterables (handle empty data by returning empty structures).
11. **Vectorization First**: Avoid iterating over rows in DataFrames for signal calculation. Use `vectorbt`, `numpy`, or `pandas` vectorized operations. Use `SignalFactory.run()` for indicator logic.

### Jupyter Notebook Standards
1. **Imports from `src`**: Notebooks must import core logic and indicators from `src`. Do not define complex strategy classes inline. Notebooks are for orchestration, analysis, and visualization only.
2. **Path Setup**: Always include the standard `sys.path` setup block at the top to resolve project root.
3. **Sequential Execution**: Notebooks must run top-to-bottom without errors.

---

## 5. Documentation Standards

* **Single Source of Truth**: Core architectural changes (e.g., database transitions, new CLI commands) must be reflected across all documentation in `docs/` and `README.md`.
* **Changelog**: Add an entry to `docs/changelog.md` whenever strategies, config values, param grids, or infrastructure are changed. Include what changed, why, and research results if a run was done.
* **Future Tweaks Plan**: Update the "Current Live Configuration" section of `docs/future_tweaks_plan.md` when new research params go live. Add new experiment ideas as they arise; remove or mark completed experiments that have been tested.
* **CLI Alignment**: Whenever the `ggt` CLI is updated, corresponding guides in `docs/UNIFIED_PIPELINE.md` must be updated immediately.
* **Standardized Reporting**: All `ggt research` summaries provided to the user must include:
  * **Executive Metrics**: Aggregate Return, Win Rate %, Sharpe Ratio, and Max Drawdown.
  * **Top Performers**: List the top 5 assets by Risk-Adjusted Return.
  * **Optimization Insights**: Prevailing optimal param ranges (e.g., PSAR acceleration, ATR multipliers).
  * **Visual Evidence**: Embed key portfolio PNG plots directly in the report.
  * **File Artifacts**: Explicit links to `run_results.json` and the `plots/` directory.
