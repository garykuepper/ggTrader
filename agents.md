# ggTrader Agent Guidelines

This document serves as the consolidated source of truth for all AI assistants (Gemini, Claude, Cursor, etc.) working on the **ggTrader** project. 

---

## 1. Project Context & Core Architecture

**As of 2026-06-16:** ggTrader is a **research-first lab-based algorithmic trading platform**. Legacy execution engines, live trading, and strategy orchestration code have been removed. The codebase now contains only the **lab** — a vectorbt-based research bench for walk-forward optimization of trading strategies.

* **Single CLI entry point**: `ggt.py` with three commands:
  * `ggt lab --strategy <name>` — run walk-forward optimization on a strategy over a historical universe (equities or crypto)
  * `ggt ingest` — pull OHLCV data into TimescaleDB
  * `ggt db <subcommand>` — database administration (diagnostics, cleanup, compression, export)
* **Lab Architecture**: 
  * **Data Layer**: `src/ggTrader/data/` loads OHLCV from TimescaleDB (crypto) or yfinance (equities). SP500 constituents are sourced from a static registry.
  * **Strategy Layer**: `src/ggTrader/lab/strategies/` implements momentum and signal-based strategies. Each strategy is a callable that produces entry/exit signals over historical data.
  * **Simulation Layer**: `src/ggTrader/lab/simulate.py` uses `vectorbt.Portfolio` for fast vectorized backtesting.
  * **Walk-Forward Harness**: `src/ggTrader/lab/harness.py` runs overlapping monthly folds, computes metrics, and persists results to `lab_runs` and `lab_periods` TimescaleDB tables.
  * **Metrics & Reporting**: `src/ggTrader/lab/metrics.py` computes Sharpe, Calmar, max drawdown, and win rate. Results are written to JSON and optionally to markdown.
* **State Storage**: All lab run results (strategy name, parameters, performance metrics, timestamps) are persisted to TimescaleDB `lab_runs` and `lab_periods` tables. No JSON file fallback.

---

## 2. Rules & Deployment Nuances

* **Docker Research Environment**: 
  * The `ggtrader_live` container has the lab code copied in. For local development on the host, install with `pip install -e .` into a virtual environment.
  * **Research command**: `docker compose run --rm ggtrader_live python ggt.py lab --strategy <name>`
  * **Database connectivity**: Inside Docker, the TimescaleDB connection string uses `host.docker.internal:5433`. On the host, use `localhost:5433`.
* **Lab Run Workflow**:
  1. **Strategy Selection**: Choose a strategy from the registry (`wfo_tournament`, `xs_momentum`, `dual_momentum`, `ema_cross`, `wfo_tournament_signal` for equities; additional strategies may be added to `src/ggTrader/lab/strategies/`).
  2. **Universe Selection**: The lab auto-generates the trading universe for the eval period — SP500 constituents for equities (sourced from `data/universe/sp500_constituents_history.csv.gz`), or top-volume coins for crypto.
  3. **Walk-Forward Execution**: Overlapping monthly folds run in-memory using vectorbt. Each fold trains on historical data (in-sample) and validates on held-out future data (out-of-sample).
  4. **Persistence**: `lab_runs` table stores run metadata (strategy, config, timestamps). `lab_periods` table stores per-fold performance metrics. Results are not written to disk.
* **Strategy Parameters**:
  * Use `--top-n` to control universe size (default 50 for equities, all eligible for crypto).
  * Use `--lookback` to set momentum calculation window (default 252 days).
  * Use `--skip` for rebalance frequency (default 21 trading days).
  * Use `--eval-start` and `--eval-end` to pin the evaluation window (default 2021-01-31 → today).
  * Use `--max-stocks` to cap universe at a specific count (optional, for diagnostic runs).

---

## 3. Development Guidelines & Strategy Addition

* **Adding a New Strategy**:
  * Implement the `Strategy` protocol in `src/ggTrader/lab/strategies/` (see `momentum.py` or `signals.py` as examples).
  * The strategy callable takes `(universe, ohlcv, cfg: LabConfig)` and returns `(weights, signals)` — vectorized arrays suitable for `vectorbt.Portfolio`.
  * Register the strategy name in the corresponding `STRATEGY_NAMES` list.
  * New strategies are immediately available via `ggt lab --strategy <new_name>`.
* **Data Access**:
  * Equities OHLCV: use `load_ohlcv()` from `src/ggTrader/lab/data.py`, which pulls from yfinance and caches locally.
  * Crypto OHLCV: TimescaleDB loader in `src/ggTrader/data/historical/timescaledb_loader.py` fetches keyed by venue and interval.
  * SP500 constituents: sourced from `data/universe/sp500_constituents_history.csv.gz` (updated infrequently; refresh via `download_sp500_history()` in `data/core/index_constituents.py`).
* **Backtesting with vectorbt**:
  * Lab uses `vectorbt.Portfolio` for fast in-memory simulation. Passed a 2D signal array (dates × symbols) and OHLCV data.
  * Weights are set daily (or per-rebalance) — the portfolio auto-rebalances on entry/exit signals.
  * No per-symbol state — vectorbt is stateless. Any state needed across bars (e.g., time-in-trade) must be computed before simulation as a separate signal.
  * Always use cash sharing and group_by for multi-symbol portfolios (prevents independent cash pools).
* **Performance Analysis**:
  * Lab computes Sharpe, Calmar, max drawdown, win rate, and monthly returns via `metrics.py`.
  * Monthly folds allow out-of-sample validation — the walk-forward harness handles fold generation and metrics aggregation.

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
11. **Vectorization First**: Avoid iterating over rows in DataFrames for signal calculation. Use `vectorbt`, `numpy`, or `pandas` vectorized operations. All strategy signals must be fully vectorized arrays (dates × symbols) before being passed to `vectorbt.Portfolio`.

### Jupyter Notebook Standards
1. **Imports from `src`**: Notebooks must import core logic and indicators from `src`. Do not define complex strategy classes inline. Notebooks are for orchestration, analysis, and visualization only.
2. **Path Setup**: Always include the standard `sys.path` setup block at the top to resolve project root.
3. **Sequential Execution**: Notebooks must run top-to-bottom without errors.

---

## 5. Documentation Standards

* **Single Source of Truth**: Core architectural changes (e.g., database transitions, new CLI commands) must be reflected across all documentation in `docs/` and `README.md`.
* **Changelog**: Add an entry to `docs/changelog.md` whenever strategies, data sources, or lab infrastructure changes. Include what changed, why, and research results if available.
* **CLI Reference**: Keep `docs/cli_reference.md` synchronized with actual `ggt` commands and flags. Document `ggt lab`, `ggt ingest`, and `ggt db` subcommands.
* **Architecture Guide**: Maintain `docs/architecture.md` as the authority on lab structure, data flow, and module responsibilities.
* **Standardized Lab Reporting**: Lab runs produce `lab_runs` and `lab_periods` TimescaleDB table entries (timestamped, immutable). Optional: generate markdown summary with:
  * **Executive Metrics**: Mean monthly return, Sharpe, max drawdown, win rate across all folds.
  * **Top Performers**: List top 5 stocks/coins by risk-adjusted return.
  * **Optimization Insights**: Parameter ranges that passed selection gates.
  * **Visual Evidence**: Equity curve and drawdown plots (if generated).
