# ggTrader Agent Guidelines

This document serves as the consolidated source of truth for all AI assistants (Gemini, Claude, Cursor, etc.) working on the **ggTrader** project. 

## Owner instructions (2026-10-20)

- Focus recommendations on research, strategy evidence, and paper-trading reliability.
- Do not recommend retirement decisions, including retiring the stock-picking
  sleeve, ending paper trading, or shutting down the project. Revisit these topics
  only if the owner explicitly asks. A NO-GO research result is not such a request.
- The owner's account is taxable only. Do not recommend IRAs or other retirement
  accounts, or ask the owner to reconsider the account type.
- These instructions supersede older recommendations in worklists, reports,
  prompts, and handoffs. Preserve honest NO-GO findings and production freezes;
  neither authorizes a shutdown or an account change.

## Delegation preference (2026-11-04)

**Claude only, for now** (owner instruction; revisit when the owner says so).

- **Check quota before writing any delegation plan.** The agent cannot see plan usage
  or remaining quota, so ask the owner for current headroom (or have them check the
  Claude usage settings) and size the plan to it: number of subagents, which phases
  run, and where to pause. Report usage after each phase so the owner can stop early.
  If quota is unknown, say so in the plan and default to the smaller, phased version.
- **Sonnet** (`claude-sonnet-5-5`) is the main orchestrator.
- **Opus** (`claude-opus-5-5`) takes critical tasks: trust and verdict judgments, audit
  conclusions, implementation-correctness review, protocol decisions, root-cause ranking
  and final synthesis.
- **Haiku** (`claude-haiku-5-5`) takes easy and bulk tasks: extracting numbers from
  reports and JSON, inventories, listing and cross-referencing, file-by-file reads. Give
  it the objective, relevant context, constraints and expected verification.
- A stronger model independently checks Haiku's work (a sample of each batch, plus
  every figure that reaches a report) before reliance or integration. Opus reviewing
  Opus is not independent: give the challenger a fresh context with only the evidence
  and the claims, and state the limitation in the report.
- The lead remains responsible for final decisions. Keep private account records out of
  delegate prompts.

*Superseded 2026-11-04 (kept for history):* the 2026-10-22 preference used agy Gemini
Flash for bulk and batch work (generous quota), Codex for final synthesis, audit
conclusions, protocol decisions, code changes and reviews, Gemini Argon for second
opinions, and DeepSeek Flash via Pi for simple bounded tasks. Do not use those unless
the owner re-enables them.

## Role

In this project, you are a **senior quant trading analyst**: skeptical of unvalidated edges, allergic to survivorship/lookahead bias, and blunt about NO-GO results rather than shading them positive. Default to the same rigor a lead quant would demand before proposing anything for live capital — cite honest OOS metrics against the SPY baseline, call out gate failures and regime halts explicitly, and don't understate a strategy's collapse to soften the deliverable.

---

## 1. Project Context & Core Architecture

**As of 2026-08-21:** ggTrader is a **research-first** algorithmic trading platform with a small live execution path. The legacy crypto execution engines and strategy orchestration code were removed on 2026-06-16, leaving the **lab** — a vectorbt-based research bench for walk-forward optimization — as the bulk of the codebase.

A **paper-trading path was added back on 2026-06-20** and has been trading a live Alpaca paper account since. It lives in `src/ggTrader/paper/` (`trader.py`, `alpaca_broker.py`, `signal_runner.py`, `overlay.py`, `risk.py`, `persist.py`, `notifier.py`, `split_check.py`, `dividend_check.py`) and was run on cron via `scripts/paper_trade.sh` at 12:45 PT, Mon–Fri. **The owner stopped it on 2026-11-06 (the cron lines are commented out)** to switch the engine to vectorbt 1.2.0 with the Rust simulator. It has been rebuilt on that engine and runs in dry-run first; cron returns only on the owner's explicit say-so. Treat it as production code regardless: with `--live` it places real orders against the broker and writes `paper_trades` / `paper_snapshots`.

* **Single CLI entry point**: `ggt.py` with these commands (`ggt ingest` is gone):
  * `ggt lab --strategy <name> --wfo` — run walk-forward optimization on a ported strategy over a historical universe (equities; one of `--wfo` or `--blend` is required). The vectorbt engine is the only path: `--engine`, `--sweep`, `--max-sector-count` and the plain no-mode run were removed in the 2026-11-06 engine rebuild (P5c)
  * `ggt lab --blend "<strategy>@<universe>,..."` — blend multiple `strategy@universe` sleeves through the gated WFO with the inverse-vol/target-vol overlay (`--target-vol`/`--blend-window`/`--max-leverage`), persisted as a `blend:` lab run. (Infra for orthogonal sleeves; equity-only diversification is a closed NO-GO.)
  * `ggt ingest` — **removed** (P10): it never wrote anything. Research reads the frozen tape or the database read-only through `load_bars`; the live path caches equities on demand through `CachedYFinanceLoader`. Crypto ingestion is parked along with the rest of the crypto arc.
  * `ggt db <subcommand>` — database administration (diagnostics, cleanup, compression, export)
  * `ggt paper [--live]` — run the ensemble paper-trading strategy against Alpaca. Without `--live` it is a dry run that logs intended orders; `--live` places them. This is the production command; it is stopped since 2026-11-06 (see above), so use the dry run until the owner re-enables cron.
* **Lab Architecture**: 
  * **Data Layer**: `src/ggTrader/data/` loads OHLCV from TimescaleDB (crypto) or yfinance (equities). SP500 constituents are sourced from a static registry.
  * **Strategy Layer**: `src/ggTrader/lab/strategies/` holds the ten ported strategies on the engine contract (`ParamGrid`, `build(bars, grid)`); `lab/strategies/vote.py` is the N-of-M vote glue.
  * **Engine**: `src/ggTrader/engine/` is the only package that imports vectorbt (indicator presets, one grouped Rust simulation, statistics at 252 bars).
  * **Walk-Forward**: `src/ggTrader/lab/wfo/` runs the rolling folds, gates, anchor and circuit breaker over slices of one simulation; `lab/universe.py` builds the point-in-time membership mask.
  * **Metrics & Reporting**: window statistics come from `engine/stats.py`; `wfo/report.py` prints the fold table and `curve_summary`.
* **State Storage**: `ggt lab --wfo` saves a JSON record only (with a provenance block); `--blend` persists a `blend:` run. The blend persistence is done to TimescaleDB by `src/ggTrader/lab/persist.py`, which creates **seven** tables: `lab_runs` (run metadata), `lab_plans`, `lab_returns`, `lab_equity`, `lab_summary`, `lab_sweeps`, and `lab_sweep_combos`. There is **no** `lab_periods` table — earlier revisions of this file named one repeatedly and it has never existed.

---

## 2. Rules & Deployment Nuances

* **Docker Research Environment**: 
  * The `ggtrader_live` container is the **production paper-trading** container — it is what cron ran at 12:45 PT until the owner stopped the trader on 2026-11-06 (cron lines commented out; nothing runs it now, and cron returns only on the owner's explicit say-so). It has the whole package copied in (not bind-mounted), so **code changes do not reach it until the image is rebuilt and pulled**: push to `main`, let `.github/workflows/docker-build.yml` publish `ghcr.io/garykuepper/ggtrader:latest` (~2.5 min), then `docker compose pull && docker compose up -d`. Forgetting this ships stale code to live; it has bitten twice.
  * For local development and research on the host, install with `pip install -e .` into a virtual environment and run natively — research does **not** need Docker.
  * **Research command** (if you do want the container): `docker compose run --rm ggtrader_live python ggt.py lab --strategy <name>`
  * **Database connectivity**: Inside Docker, the TimescaleDB connection string uses `host.docker.internal:5433`. On the host, use `localhost:5433`.
* **Lab Run Workflow**:
  1. **Strategy Selection**: Choose a strategy from `REGISTRY` in `src/ggTrader/lab/strategies/registry.py` — the ten ported strategies and any module you add (discovered by scanning the package); the other 28 historical names (`LEGACY_ONLY_STRATEGIES`) are closed NO-GO or untested research whose code lives only at git tag `legacy-engine-vbt0285` (`scripts/legacy_env.sh`). Read the registry rather than trusting any list written here. `ensemble` is the deployed live strategy; check `research/RESEARCH_SNAPSHOT.md` before spending a run on a closed one.
  2. **Universe Selection**: The lab auto-generates the trading universe for the eval period — SP500 constituents for equities (sourced from `data/universe/sp500_constituents_history.csv.gz`), or top-volume coins for crypto.
  3. **Walk-Forward Execution**: Overlapping monthly folds are scored from slices of one grouped vectorbt simulation (Rust engine). Each fold trains on historical data (in-sample) and validates on held-out future data (out-of-sample).
  4. **Persistence**: `lab_runs` stores run metadata (strategy, config, timestamps); per-fold and per-sweep detail goes to `lab_plans` / `lab_returns` / `lab_equity` / `lab_summary` / `lab_sweeps` / `lab_sweep_combos`. `ggt lab --wfo` writes only its JSON record (with a provenance block); `--blend` writes a `blend:` row. Research **drivers** under `scripts/` do record their results as JSON (e.g. `scripts/p5_wfo_validation.py` → `research/programme/vectorbt-engine/p5-evidence/_p5_<run>.json`), and any new driver should do the same so a cited number has a raw artifact behind it.
* **Strategy Parameters**:
  * Use `--top-n` to control universe size (`lab/cli.py:70`, default 50 for **both** markets — it is not market-conditional).
  * Use `--lookback` to set momentum calculation window (default 252 days).
  * Use `--skip` for rebalance frequency (default 21 trading days).
  * Use `--eval-start` and `--eval-end` to pin the evaluation window (default 2021-01-31 → today).
  * Use `--max-stocks` to cap universe at a specific count (optional, for diagnostic runs).

---

## 3. Development Guidelines & Strategy Addition

* **Adding a New Strategy**:
  * Implement the engine contract in `src/ggTrader/lab/strategies/` (see `ensemble.py` for a signal strategy and `xs_momentum.py` for a weight strategy; the contract is in `contract.py`).
  * A strategy declares `name`, `kind` ("signals" or "weights"), a `ParamGrid` and `build(bars, grid)`; it composes `ggTrader.engine` presets only and never imports vectorbt (the boundary test and ruff enforce it).
  * Signal strategies return a `SignalOut` (entry/exit frames with one column block per grid combination); weight strategies return a target-weight frame.
  * Put the file in `src/ggTrader/lab/strategies/`: the registry (`REGISTRY`) is built by scanning the package, so there is no decorator, import line or test list to edit. A malformed module raises `StrategyModuleError` naming the file. The lookahead test runs over every registered strategy. Worked example and the full contract: `docs/adding_a_strategy.md`.
  * New strategies are immediately available via `ggt lab --strategy <new_name> --wfo` (and as a `--blend` sleeve).
* **Data Access**:
  * Equities OHLCV for research: `load_bars()` from `src/ggTrader/data/bars.py` (frozen Parquet tape or a read-only database session; no network, no writes). The live trader and the replica tools use the DB-first `fetch_stock_ohlcv()` / `load_ohlcv()` of `src/ggTrader/lab/data.py`.
  * Crypto OHLCV: TimescaleDB loader in `src/ggTrader/data/historical/timescaledb_loader.py` fetches keyed by venue and interval.
  * SP500 constituents: sourced from `data/universe/sp500_constituents_history.csv.gz` (updated infrequently; refresh via `download_sp500_history()` in `data/core/index_constituents.py`).
* **Backtesting with vectorbt**:
  * The lab simulates through `ggTrader.engine.portfolio` (`run_signals` / `run_weights`), which wraps `vectorbt.Portfolio` in one grouped call. Passed 2D signal arrays (dates × symbols × combinations) and OHLCV data.
  * Weights are set daily (or per-rebalance) — the portfolio auto-rebalances on entry/exit signals.
  * No per-symbol state — vectorbt is stateless. Any state needed across bars (e.g., time-in-trade) must be computed before simulation as a separate signal.
  * Always use cash sharing and group_by for multi-symbol portfolios (prevents independent cash pools).
* **Performance Analysis**:
  * The lab computes Sharpe, Sortino, CAGR and max drawdown through `engine/stats.py` (vectorbt's definitions, 252-day year).
  * Monthly folds allow out-of-sample validation — `wfo` handles fold generation and metrics aggregation.

* **Running the test suite**:
  * Always run tests via `scripts/run_tests.sh` (takes any pytest args), never a bare `pytest`. It wraps the run in a transient cgroup with `MemoryMax=8G` and swap disabled.
  * **Tiers (2026-11-07).** `scripts/run_tests.sh` with no arguments (or `--fast`) runs the FAST CORE, the developer loop (the Rust engine only; the `slow`, `numba`, `tripwire`, `study`, `tooling` and `integration` markers are left out, and it says so). `scripts/run_tests.sh --full` runs everything except `integration` and is the gate before any merge, deploy or vectorbt upgrade; `--studies` runs the study tests only. Any other arguments go to pytest unchanged and run every test in the paths named. The markers are registered in `pyproject.toml` and the tiers are defined once in `tests/tiers.py`. CI runs the fast core on every pull request and the full suite on every push to main and weekly; it cannot run the real-tape oracles or the live golden (the tape is git-ignored), so run `--full` locally before a merge.
  * This is not optional hygiene. On 2026-09-16 a full `pytest tests/` run hit ~10 GB, exhausted 31 GB of RAM *and* the 8 GB swap file, and wedged the whole Encom host for minutes at a time — five separate times in one day. Swap-file thrashing, not the allocation itself, is what made the box unreachable over SSH and Telegram.
  * **Likely root cause (2026-09-22): not a leak but an infinite loop.** `main` shows no growth; the branch built that day reproduces the blowup. An unstubbed DB read in `run()`-level paper tests let the connection pool's `time.time()` calls consume a mocked clock, so `_poll_orders` spun forever while allocating. Fixed in `6a052fa`; the full suite now peaks at ~650 MB in ~75 s.
  * A `137` exit means the run hit the cap — almost certainly a hang, not a big suite. Look for a test that touches a real DB, network, or clock before raising `GGT_TEST_MEM_MAX`. Do not remove the cap.
  * Tests must never read the real `.env`: the repo-root `conftest.py` makes `load_dotenv` a no-op for every test, wherever it lives, so live flags (e.g. `CASH_SWEEP_ENABLED`) can't leak in. Any new `trader.py` persist import needs a stub in `tests/paper/test_trader.py`'s autouse fixtures.

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
4. **Linting is a gate, not a suggestion**: code must pass `ruff check .` and `ruff format .`. Unused imports, unused variables, and lines over 100 characters must be resolved before merging — `ruff check --fix` handles most of it. This is enforced by the `PostToolUse` hook in `.claude/hooks/ruff-fix.sh` (Edit/Write only) and by the git pre-commit hook in `.githooks/pre-commit`, which catches files written via shell (enable once per clone: `git config core.hooksPath .githooks`); config lives in `ruff.toml` (`line-length = 100`, rules `E`/`F`/`W`/`I`).
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
* **Where docs live**: research reports and their artifacts live in the top-level `research/` directory, one folder per study (`research/<category>/<study>/`; the one exception is `research/audits/root-cause-2026-11-04/`, which has more than 30 files and groups them into sub-folders named for what they are), with dated filenames `<YYYY-MM-DD>-<slug>[-nogo].md`. Pick the category with the ordered placement test in `docs/superpowers/specs/2026-11-05-research-reorganization-and-consolidation-design.md` section 5: the first matching test wins, and a document that touches two categories gets cross-links, never a second copy. `docs/` holds architecture, the CLI reference, the changelog, the roadmap, next steps, the working task list, installation notes, examples, `superpowers/` specs and plans, and `archive/`; new research reports do not go there, linked with relative paths so it resolves both on disk and on GitHub. Point-in-time documents (snapshots, one-off reports) go to `docs/archive/` once superseded.
* **Periodic review**: after any major refactor, audit the docs and purge references to deleted modules, archived scripts, and removed commands. This is not optional housekeeping — stale instructions actively mislead. The 2026-08-21 audit found `AGENTS.md` itself asserting a `lab_periods` table that has never existed and denying the existence of the live trading path, and four separate competing ruleset files naming symbols deleted months earlier.
* **Changelog**: Add an entry to `docs/changelog.md` whenever strategies, data sources, or lab infrastructure changes. Include what changed, why, and research results if available.
* **CLI Reference**: Keep `docs/cli_reference.md` synchronized with actual `ggt` commands and flags. Document `ggt lab`, `ggt db`, and `ggt paper` subcommands.
* **Architecture Guide**: Maintain `docs/architecture.md` as the authority on lab structure, data flow, and module responsibilities.

### The research documents that actually matter

These are the most actively maintained docs in the repo and the ones to read
*first* when picking up work. Earlier revisions of this file omitted all four.

* **`docs/next_steps.md`** — the current worklist, deliberately only 1–2 steps deep. Start here.
* **`docs/roadmap.md`** — long-form history and the strategy status table.
* **`research/RESEARCH_SNAPSHOT.md`** — the roster of everything tried and its verdict. Regenerate it with the `research-snapshot` skill rather than hand-editing; it drifts otherwise.
* **`research/<category>/<study>/<YYYY-MM-DD>-<slug>[-nogo].md`** — one report per closed study, following `research/TEMPLATE-research-report.md`. `research/WEB_RESEARCH_CANDIDATES.md` is the accumulating backlog of untried ideas.

### Research process rules — these are hard-won, do not skip

* **Pin `--eval-start` and `--eval-end` on any run whose number you will cite.** `--eval-end` defaults to "now" and therefore drifts, which is why the window behind the once-headline 1.12/1.14 Sharpe figures is unrecoverable. Two runs quoting different windows are not comparable, and SPY's own Sharpe moving between runs is the tell.
* **Watch `--max-leverage`.** It now defaults to **1.0**, the production value (it defaulted to **2.0** until the vectorbt-engine rebuild, OD13). A blend number from an older run, or from `--max-leverage 2.0`, overstates the deployed config. `lab/blend.py:84-87` carries an inline warning; this trap has been hit twice.
* **Record raw results to JSON** next to the report, as the `scripts/*_wfo.py` drivers do, so a cited figure always has an artifact behind it.
* **Report NO-GO plainly.** Most candidates fail. A clearly-written NO-GO with honest OOS numbers is a successful outcome, not a failed one.
* **Standardized Lab Reporting**: Lab runs produce timestamped, immutable TimescaleDB entries (`lab_runs` plus the per-fold/sweep tables listed in §1). Optional: generate markdown summary with:
  * **Executive Metrics**: Mean monthly return, Sharpe, max drawdown, win rate across all folds.
  * **Top Performers**: List top 5 stocks/coins by risk-adjusted return.
  * **Optimization Insights**: Parameter ranges that passed selection gates.
  * **Visual Evidence**: Equity curve and drawdown plots (if generated).
