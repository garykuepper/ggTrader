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
4. **Linting is a gate, not a suggestion**: code must pass `ruff check .` and `ruff format .`. Unused imports, unused variables, and lines over 100 characters must be resolved before merging — `ruff check --fix` handles most of it. This is enforced automatically by the `PostToolUse` hook in `.claude/hooks/ruff-fix.sh`, and config lives in `ruff.toml` (`line-length = 100`, rules `E`/`F`/`W`/`I`).
5. **Single Source of Truth**: Core logic must live in `src/`. Do not duplicate definitions across files.
6. **Module structure**: constants at module level (top), not buried inside functions; execution logic behind an `if __name__ == "__main__":` guard.
7. **Scripts**: Utility scripts in `scripts/` should have a `main()` entry point and `argparse` for inputs, and every script must handle `--help` gracefully.
8. **Resilient, resumable scripts**: long-running work (backfills, ingestion, WFO drivers) must log and continue on recoverable errors rather than crashing, and must checkpoint for resumability if it can run longer than ~5 minutes. Several drivers in `scripts/` run for 30-60 minutes; losing an hour to an unhandled exception at minute 55 is the failure this prevents.
9. **Imports**: absolute, rooted at `src` (`from ggTrader.<package> import ...`) for cross-package access; relative imports (`from .module import X`) only inside a package's `__init__.py`. Avoid circular imports by keeping `utils` free of domain logic.
10. **Dependency discipline**: record new libraries in `pyproject.toml` as you add them. Never `pip install` something the project then depends on without recording it.
11. **Symbol Normalization**: Use the `SYMBOL_MAPPING` table in `data/core/constants.py` to convert exchange-specific prefixed symbols (e.g., `XBT`, `XETH`) to standard tickers (`BTC`, `ETH`).
12. **Path Safety**: Always use `os.path.join` or `pathlib.Path` for file paths. Resolve project root dynamically.
13. **Error Handling**: Data loading functions must raise descriptive exceptions (e.g., `ValueError`, `FileNotFoundError`) on failure. Avoid returning `None` from functions expected to return iterables (handle empty data by returning empty structures).
14. **Vectorization First**: Avoid iterating over rows in DataFrames for signal calculation. Use `vectorbt`, `numpy`, or `pandas` vectorized operations. All strategy signals must be fully vectorized arrays (dates × symbols) before being passed to `vectorbt.Portfolio`.

### SQL conventions on the `ohlcv` table

Symbols are stored as `ASSET-QUOTE` (e.g. `BTC-USD`). Split them with
PostgreSQL/TimescaleDB functions rather than string slicing in Python:

* **Asset**: `split_part(symbol, '-', 1)`
* **Quote**: `split_part(symbol, '-', 2)`
* **Cross-asset ranking**: prefer notional volume (`volume * close`) over raw
  `volume`, so a $3 stock and a $600 stock are comparable.

Also note `ohlcv.timestamp` is `timestamp WITHOUT time zone` holding **naive
UTC**. Never write a tz-aware datetime into it — Postgres will rebase the
offset into the session timezone and silently shift every bar. This caused a
whole-tape corruption; see `data/live/cached_yfinance_loader.py:215-226`.

### Never delete

`data/` and any database configuration are off-limits to automated cleanup.
Generated artifacts under `results/` and stray logs are fair game once
confirmed unreferenced — but check for inbound references from tracked docs
first, since several research reports cite log files by name.

---

## 5. Documentation Standards

* **Single Source of Truth**: Core architectural changes (e.g., database transitions, new CLI commands) must be reflected across all documentation in `docs/` and `README.md`.
* **Where docs live**: supplemental documentation belongs in `docs/`, linked with relative paths so it resolves both on disk and on GitHub. Point-in-time documents (snapshots, one-off reports) go to `docs/archive/` once superseded.
* **Periodic review**: after any major refactor, audit the docs and purge references to deleted modules, archived scripts, and removed commands. This is not optional housekeeping — stale instructions actively mislead. The 2026-08-21 audit found `AGENTS.md` itself asserting a `lab_periods` table that has never existed and denying the existence of the live trading path, and four separate competing ruleset files naming symbols deleted months earlier.
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
