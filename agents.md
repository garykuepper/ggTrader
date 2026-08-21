# ggTrader Agent Guidelines

This document serves as the consolidated source of truth for all AI assistants (Gemini, Claude, Cursor, etc.) working on the **ggTrader** project. 

## Role

In this project, you are a **senior quant trading analyst**: skeptical of unvalidated edges, allergic to survivorship/lookahead bias, and blunt about NO-GO results rather than shading them positive. Default to the same rigor a lead quant would demand before proposing anything for live capital — cite honest OOS metrics against the SPY baseline, call out gate failures and regime halts explicitly, and don't understate a strategy's collapse to soften the deliverable.

---

## 1. Project Context & Core Architecture

**As of 2026-08-21:** ggTrader is a **research-first** algorithmic trading platform with a small live execution path. The legacy crypto execution engines and strategy orchestration code were removed on 2026-06-16, leaving the **lab** — a vectorbt-based research bench for walk-forward optimization — as the bulk of the codebase.

A **paper-trading path was added back on 2026-06-20** and has been trading a live Alpaca paper account since. It lives in `src/ggTrader/paper/` (`trader.py`, `alpaca_broker.py`, `signal_runner.py`, `overlay.py`, `risk.py`, `persist.py`, `notifier.py`, `feature_gate.py`, `split_check.py`, `dividend_check.py`) and runs on cron via `scripts/paper_trade.sh` at 12:45 PT, Mon–Fri. Treat it as production: it places real orders against the broker and writes `paper_trades` / `paper_snapshots`.

* **Single CLI entry point**: `ggt.py` with four commands:
  * `ggt lab --strategy <name>` — run walk-forward optimization on a strategy over a historical universe (equities or crypto)
  * `ggt lab --blend "<strategy>@<universe>,..."` — blend multiple `strategy@universe` sleeves through the gated WFO with the inverse-vol/target-vol overlay (`--target-vol`/`--blend-window`/`--max-leverage`), persisted as a `blend:` lab run. (Infra for orthogonal sleeves; equity-only diversification is a closed NO-GO.)
  * `ggt ingest` — **non-functional stub, do not use.** `cli/cmd_ingest.py:26-31` hardcodes a two-symbol crypto list with the actual `sync_symbol_ohlcv` call commented out, then prints "Ingestion complete." It reports success having written nothing. Equity data is loaded on demand through `CachedYFinanceLoader` instead (see Data Layer below); crypto ingestion is parked along with the rest of the crypto arc.
  * `ggt db <subcommand>` — database administration (diagnostics, cleanup, compression, export)
  * `ggt paper [--live]` — run the ensemble paper-trading strategy against Alpaca. Without `--live` it is a dry run that logs intended orders; `--live` places them. This is the deployed production command.
* **Lab Architecture**: 
  * **Data Layer**: `src/ggTrader/data/` loads OHLCV from TimescaleDB (crypto) or yfinance (equities). SP500 constituents are sourced from a static registry.
  * **Strategy Layer**: `src/ggTrader/lab/strategies/` implements momentum and signal-based strategies. Each strategy is a callable that produces entry/exit signals over historical data.
  * **Simulation Layer**: `src/ggTrader/lab/simulate.py` uses `vectorbt.Portfolio` for fast vectorized backtesting.
  * **Walk-Forward Harness**: `src/ggTrader/lab/harness.py` runs overlapping monthly folds, computes metrics, and persists results to TimescaleDB.
  * **Metrics & Reporting**: `src/ggTrader/lab/metrics.py` computes Sharpe, Calmar, max drawdown, and win rate.
* **State Storage**: Lab run results are persisted to TimescaleDB by `src/ggTrader/lab/persist.py`, which creates **seven** tables: `lab_runs` (run metadata), `lab_plans`, `lab_returns`, `lab_equity`, `lab_summary`, `lab_sweeps`, and `lab_sweep_combos`. There is **no** `lab_periods` table — earlier revisions of this file named one repeatedly and it has never existed.

---

## 2. Rules & Deployment Nuances

* **Docker Research Environment**: 
  * The `ggtrader_live` container is the **production paper-trading** container — it is what cron runs at 12:45 PT. It has the whole package copied in (not bind-mounted), so **code changes do not reach it until the image is rebuilt and pulled**: push to `main`, let `.github/workflows/docker-build.yml` publish `ghcr.io/garykuepper/ggtrader:latest` (~2.5 min), then `docker compose pull && docker compose up -d`. Forgetting this ships stale code to live; it has bitten twice.
  * For local development and research on the host, install with `pip install -e .` into a virtual environment and run natively — research does **not** need Docker.
  * **Research command** (if you do want the container): `docker compose run --rm ggtrader_live python ggt.py lab --strategy <name>`
  * **Database connectivity**: Inside Docker, the TimescaleDB connection string uses `host.docker.internal:5433`. On the host, use `localhost:5433`.
* **Lab Run Workflow**:
  1. **Strategy Selection**: Choose a strategy from `STRATEGY_REGISTRY` in `src/ggTrader/lab/strategies/__init__.py` — **36 entries** as of 2026-08-21, so read the registry rather than trusting any list written here. `ensemble` is the deployed live strategy. Most of the rest are closed NO-GO research; check `docs/research/RESEARCH_SNAPSHOT.md` before spending a run on one.
  2. **Universe Selection**: The lab auto-generates the trading universe for the eval period — SP500 constituents for equities (sourced from `data/universe/sp500_constituents_history.csv.gz`), or top-volume coins for crypto.
  3. **Walk-Forward Execution**: Overlapping monthly folds run in-memory using vectorbt. Each fold trains on historical data (in-sample) and validates on held-out future data (out-of-sample).
  4. **Persistence**: `lab_runs` stores run metadata (strategy, config, timestamps); per-fold and per-sweep detail goes to `lab_plans` / `lab_returns` / `lab_equity` / `lab_summary` / `lab_sweeps` / `lab_sweep_combos`. The harness itself writes nothing to disk — but research **drivers** under `scripts/` do record their results as JSON (e.g. `scripts/anchor_fix_reproduction_wfo.py` → `docs/research/_anchor_fix_reproduction_results.json`), and any new driver should do the same so a cited number has a raw artifact behind it.
* **Strategy Parameters**:
  * Use `--top-n` to control universe size (`lab/cli.py:70`, default 50 for **both** markets — it is not market-conditional).
  * Use `--lookback` to set momentum calculation window (default 252 days).
  * Use `--skip` for rebalance frequency (default 21 trading days).
  * Use `--eval-start` and `--eval-end` to pin the evaluation window (default 2021-01-31 → today).
  * Use `--max-stocks` to cap universe at a specific count (optional, for diagnostic runs).

---

## 3. Development Guidelines & Strategy Addition

* **Adding a New Strategy**:
  * Implement the `Strategy` protocol in `src/ggTrader/lab/strategies/` (see `momentum.py` or `signals.py` as examples).
  * The `Strategy` protocol requires `name`, `target_kind` ("weights" or "signals"), `select(asof, data, eligible) -> Plan`, and `to_targets(plans, data) -> DataFrame | SignalTargets`.
  * Weight strategies return a `pd.DataFrame` (time x symbol, float weights). Signal strategies return `SignalTargets(entries, exits)` with boolean frames.
  * Register the strategy in the single-source `STRATEGY_REGISTRY` dict in `src/ggTrader/lab/strategies/__init__.py` (one line: `"<name>": <Class>`). Signal-vs-weight membership and the CLI name lists derive automatically from each class's `target_kind` (see `strategies/registry.py`); the old `signals.py`/`momentum.py` name tuples are now lazy shims over this one dict.
  * New strategies are immediately available via `ggt lab --strategy <new_name>` (and as a `--blend` sleeve).
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
5. **Scripts**: Utility scripts in `scripts/` should have a `main()` entry point and `argparse` for inputs.
6. **Symbol Normalization**: Use the `SYMBOL_MAPPING` table in `data/core/constants.py` to convert exchange-specific prefixed symbols (e.g., `XBT`, `XETH`) to standard tickers (`BTC`, `ETH`).
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
* **CLI Reference**: Keep `docs/cli_reference.md` synchronized with actual `ggt` commands and flags. Document `ggt lab`, `ggt db`, and `ggt paper` subcommands.
* **Architecture Guide**: Maintain `docs/architecture.md` as the authority on lab structure, data flow, and module responsibilities.

### The research documents that actually matter

These are the most actively maintained docs in the repo and the ones to read
*first* when picking up work. Earlier revisions of this file omitted all four.

* **`docs/next_steps.md`** — the current worklist, deliberately only 1–2 steps deep. Start here.
* **`docs/roadmap.md`** — long-form history and the strategy status table.
* **`docs/research/RESEARCH_SNAPSHOT.md`** — the roster of everything tried and its verdict. Regenerate it with the `research-snapshot` skill rather than hand-editing; it drifts otherwise.
* **`docs/research/<YYYY-MM-DD>-<slug>[-nogo].md`** — one report per closed study, following `docs/research/TEMPLATE-research-report.md`. `WEB_RESEARCH_CANDIDATES.md` is the accumulating backlog of untried ideas.

### Research process rules — these are hard-won, do not skip

* **Pin `--eval-start` and `--eval-end` on any run whose number you will cite.** `--eval-end` defaults to "now" and therefore drifts, which is why the window behind the once-headline 1.12/1.14 Sharpe figures is unrecoverable. Two runs quoting different windows are not comparable, and SPY's own Sharpe moving between runs is the tell.
* **Watch `--max-leverage`.** It defaults to **2.0**, but production runs at **1.0**. Quoting a blend number computed at the default overstates the deployed config. `lab/blend.py:84-87` carries an inline warning; this trap has been hit twice.
* **Record raw results to JSON** next to the report, as the `scripts/*_wfo.py` drivers do, so a cited figure always has an artifact behind it.
* **Report NO-GO plainly.** Most candidates fail. A clearly-written NO-GO with honest OOS numbers is a successful outcome, not a failed one.
* **Standardized Lab Reporting**: Lab runs produce timestamped, immutable TimescaleDB entries (`lab_runs` plus the per-fold/sweep tables listed in §1). Optional: generate markdown summary with:
  * **Executive Metrics**: Mean monthly return, Sharpe, max drawdown, win rate across all folds.
  * **Top Performers**: List top 5 stocks/coins by risk-adjusted return.
  * **Optimization Insights**: Parameter ranges that passed selection gates.
  * **Visual Evidence**: Equity curve and drawdown plots (if generated).
