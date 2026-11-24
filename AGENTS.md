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

## Research rules reset (2026-11-20)

Owner instruction of 2026-11-20. The owner did not understand the old evidence rules
and wiped them for **new** work.

- **No longer binding for new work:** Part A, the t-stat bars (Holm, Newey-West), the
  "+1.0 percentage point a year since 1994" bar, the B0/B1 shadow stages and the
  "six months of clean paper trading" wait. Do not hold a new idea to them, and do not
  quote them to the owner as requirements.
- **The four plain rules that replace them:**
  1. Test only on data the strategy never saw while it was being tuned.
  2. Use only what was known on each day, and include stocks that later died.
  3. Subtract trading costs and taxes, and compare to SPY over the same dates.
  4. Paper trade first. The owner decides when a strategy has earned real money.
- **Talk to the owner in plain words.** No jargon, no unexplained labels (Part A, PIT,
  Holm, WFO, bps). If a term is needed, say what it means in the same sentence.
- **History stays.** Old reports, verdicts and NO-GO findings were scored under the old
  rules and stay on disk as history. Do not delete them and do not re-score them.
  Production freezes are unchanged: the trader stays stopped until the owner says so.

## Delegation preference (2026-11-08)

Owner instruction of 2026-11-08; revisit when the owner says so.

- **Check quota before writing any delegation plan.** The agent cannot see plan usage
  or remaining quota, so ask the owner for current headroom (or have them check the
  Claude usage settings) and size the plan to it: number of subagents, which phases
  run, and where to pause. Report usage after each phase so the owner can stop early.
  If quota is unknown, say so in the plan and default to the smaller, phased version.
- **Builders:** ChatGPT Codex Terra (`gpt-6.1-terra`), ChatGPT Codex Luna
  (`gpt-6-luna`), Claude Sonnet (`claude-sonnet-5-5`) and Claude Haiku
  (`claude-haiku-5-5`).
- **Reviewers:** ChatGPT Codex Sol (`gpt-6.1-sol`) and Claude Opus (`claude-opus-5-5`).
  Opus also takes the critical judgments: trust and verdict calls, audit conclusions,
  protocol decisions, root-cause ranking and final synthesis.
- **Orchestrator:** Claude Sonnet. It re-verifies every slice itself (reads the diff, reruns
  the checks) and does not pass on a builder's or reviewer's claim unchecked.
- **A reviewer never reviews its own model family's build.** Codex output (Terra, Luna) is
  reviewed by Opus; Claude output (Sonnet, Haiku) is reviewed by Sol. Anything headed for a
  merge gets both a Sol review and an Opus review.
- **Haiku** takes bulk mechanical work: extracting numbers from reports and JSON,
  inventories, listing and cross-referencing, file-by-file reads. Give it the objective,
  relevant context, constraints and expected verification. A stronger model samples each
  batch and checks every figure that reaches a report before it is relied on.
- Give a reviewer a fresh context with only the evidence and the claims; a model checking
  its own earlier work is not independent, and the report should say so when it happens.
- **Antigravity re-enabled for building and reading (2026-11-24).** Owner: "give agy some work too". Gemini 4 Neon may build self-contained tasks and Gemini 4 Argon may act as an extra reader. A Gemini build is reviewed by Sol or Opus. The superseded notes below stay as history. Revisit when the owner says so.
- The lead remains responsible for final decisions. Keep private account records out of
  delegate prompts.

*Superseded 2026-11-08 (kept for history):* the 2026-11-04 preference was "Claude only":
Sonnet orchestrated, Opus took critical tasks (trust and verdict judgments, audit
conclusions, implementation-correctness review, protocol decisions, root-cause ranking and
final synthesis), Haiku took easy and bulk tasks, and a stronger model independently
checked Haiku's work. Codex models were not used under it.

*Superseded 2026-11-04 (kept for history):* the 2026-10-22 preference used agy Gemini
Flash for bulk and batch work (generous quota), Codex for final synthesis, audit
conclusions, protocol decisions, code changes and reviews, Gemini Argon for second
opinions, and DeepSeek Flash via Pi for simple bounded tasks. Do not use those unless
the owner re-enables them.

## Local reference lookup

For trading and finance fundamentals, search the `gg-library` MCP using `Trading` and
`Mathematics-Statistics`, broadening subjects when useful. Read relevant passages and cite
their sources and locators. If the MCP is unavailable, use the `gg-library` skill's CLI
fallback. Use the existing `research` workflow for online research.

## Role

In this project, you are a **senior quant trading analyst**: skeptical of unvalidated edges, allergic to survivorship/lookahead bias, and blunt about NO-GO results rather than shading them positive. Default to the same rigor a lead quant would demand before proposing anything for live capital — cite honest OOS metrics against the SPY baseline, call out gate failures and regime halts explicitly, and don't understate a strategy's collapse to soften the deliverable.

---

## 1. Project Context & Core Architecture

**As of 2026-08-21:** ggTrader is a **research-first** algorithmic trading platform with a small live execution path. The legacy crypto execution engines and strategy orchestration code were removed on 2026-06-16, leaving the **lab** — a vectorbt-based research bench for walk-forward optimization — as the bulk of the codebase.

A **paper-trading path was added back on 2026-06-20** and has been trading a live Alpaca paper account since. It lives in `src/ggTrader/paper/` (`trader.py`, a short orchestrator since the Phase 5 run split of 2026-11-20, `alpaca_broker.py`, `signal_runner.py`, `overlay.py`, `risk.py`, `persist.py`, `notifier.py`, `split_check.py`, `dividend_check.py`, plus the signal, gate and risk helpers `cash_sweep.py`, `catastrophe_stop.py`, `gates.py`, `live_bars.py`, `live_signals.py` and `untradable.py`; the step modules the Phase 5 run split added (`preflight.py`, `portfolio.py`, `risk_assessor.py`, `planning.py`, `order_flow.py`, `execution.py`, `corporate_actions.py`, `reporter.py`) with their helpers `seams.py` and `run_helpers.py`; and the shared pieces from earlier phases (`models.py`, `ports.py`, `ledger.py` and `config.py` from Phase 4; `constants.py` from phases 0 to 2); `docs/architecture.md` has the tree and the seam rules) and was run on cron via `scripts/paper_trade.sh` at 12:45 PT, Mon–Fri. **The owner stopped it on 2026-11-06 (the cron lines are commented out)** to switch the engine to vectorbt 1.2.0 with the Rust simulator. It has been rebuilt on that engine and runs in dry-run first; cron returns only on the owner's explicit say-so. Treat it as production code regardless: with `--live` it places real orders against the broker and writes `paper_trades` / `paper_snapshots`.

* **Single CLI entry point**: `ggt.py` with these commands (`ggt ingest` is gone):
  * `ggt lab --strategy <name> --wfo` — run walk-forward optimization on a ported strategy over a historical universe (equities; one of `--wfo` or `--blend` is required). The vectorbt engine is the only path: `--engine`, `--sweep`, `--max-sector-count` and the plain no-mode run were removed in the 2026-11-06 engine rebuild (P5c)
  * `ggt lab --blend "<strategy>@<universe>,..."` — blend multiple `strategy@universe` sleeves through the gated WFO with the inverse-vol/target-vol overlay (`--target-vol`/`--blend-window`/`--max-leverage`), persisted as a `blend:` lab run. (Infra for orthogonal sleeves; equity-only diversification is a closed NO-GO.)
  * `ggt ingest` — **removed** (P10): it never wrote anything. Research reads the frozen tape or the database read-only through `load_bars`; the live path caches equities on demand through `CachedYFinanceLoader`. Crypto ingestion is parked along with the rest of the crypto arc.
  * `ggt db <subcommand>` — database administration (diagnostics, cleanup, compression, export)
  * `ggt db load-sp500` and `ggt db membership {apply-ddl,verify,status,diff,export,reinstate,check-upstream,check-holdings,weekly,proposal}` — the versioned S&P 500 membership list (load a vintage, compare vintages, cross-check against the SPY fund holdings, propose changes; a vintage is written only by `load-sp500 --commit` or `membership reinstate --commit`, never by the weekly job). Flags and behavior: `docs/cli_reference.md`.
  * `ggt paper [--live]` — run the ensemble paper-trading strategy against Alpaca. Without `--live` it is a dry run that logs intended orders; `--live` places them. This is the production command; it is stopped since 2026-11-06 (see above), so use the dry run until the owner re-enables cron.
* **Lab Architecture**: 
  * **Data Layer**: `src/ggTrader/data/` loads OHLCV from TimescaleDB (crypto) or yfinance (equities). S&P 500 membership is read through `MembershipSource` (`data/membership.py`) from the PostgreSQL `universe` schema (the latest or a pinned vintage) or from a hash-checked frozen fixture; the live trader's gate is `data/membership_gate.py`.
  * **Strategy Layer**: `src/ggTrader/lab/strategies/` holds the ten ported strategies on the engine contract (`ParamGrid`, `build(bars, grid)`); `lab/strategies/vote.py` is the N-of-M vote glue.
  * **Engine**: `src/ggTrader/engine/` is the only package that imports vectorbt (indicator presets, one grouped Rust simulation, statistics at 252 bars).
  * **Walk-Forward**: `src/ggTrader/lab/wfo/` runs the rolling folds, gates, anchor and circuit breaker over slices of one simulation; `lab/universe.py` builds the point-in-time membership mask.
  * **Metrics & Reporting**: window statistics come from `engine/stats.py`; `wfo/report.py` prints the fold table and `curve_summary`.
  * **Shared helpers**: `src/ggTrader/utils/` holds the helpers that studies and the lab share: `significance.py`, `fred_data.py`, `io.py` (atomic writes), `results.py`, `manifest.py` and `binding.py` (hash and provenance records), and `study.py` (study plumbing: `bootstrap`, `repo_root`, `study_dir`).
  * **Study code is not in `src`**: since the study-isolation refactor (2026-11-07) `src/ggTrader/lab/` holds no study modules. Each study keeps its own modules, drivers and tests in its folder under `research/`: a `code/` folder (tests in `code/tests/`), plus a `_lib/` folder per category for helpers that several studies of that category share. The moved screens, audits, tax-lot library, replica tools and intraday family now live there; `research/CODEMOVED.csv` lists every moved file with its hash before and after. Nothing in `src/` imports a study.
* **State Storage**: `ggt lab --wfo` saves a JSON record only (with a provenance block); `--blend` persists a `blend:` run. The blend persistence is done to TimescaleDB by `src/ggTrader/lab/persist.py`, which creates **seven** tables: `lab_runs` (run metadata), `lab_plans`, `lab_returns`, `lab_equity`, `lab_summary`, `lab_sweeps`, and `lab_sweep_combos`. There is **no** `lab_periods` table — earlier revisions of this file named one repeatedly and it has never existed. The S&P 500 membership list lives apart from the lab tables, in PostgreSQL schema `universe` (`loads`, `membership`, `snapshot_dates`, `load_blobs`, `load_evidence`, `checks`, `check_blobs`, `symbol_changes`, `source_snapshots`). Its tables are append-only: a refresh is a new vintage (1 the frozen file import, 2 the upstream download to 2026-08-18, 3 a curated gap-fill to 2026-11-04), never an edit of an existing one (the only permitted update marks a replaced row `superseded_by_load_id`). DDL is `sql/universe_schema.sql`, applied by `ggt db membership apply-ddl`.

### Database map (measured 2026-11-22; Kaggle update 2026-11-23)

Read this before writing any query or importer. Everything below was read from the live database on that date; row counts are approximate (`reltuples`) and drift. Re-measure before you quote one.

* **Server.** One PostgreSQL 16 database named `ggtrader` with the TimescaleDB 2.25.1 extension (an add-on that splits big time-series tables into compressed chunks; it is not a separate database). Container `ggtrader_db` (`timescale/timescaledb:latest-pg16`), host port 5433. Its data lives in the Docker volume `ggtraderdb_ggtrader_pgdata` on the **main drive** (455 GB, 83% full, about 75 GB free on 2026-11-22). On 2026-11-22 total size was **14 GB**, of which the `ohlcv` hypertable was 12 GB; after the import it was 17 GB on 2026-11-23. `wal_level=replica`, `max_wal_size` 1 GB, archiving off. The original tables are in `public` (49 tables) and `universe` (9 tables); the Kaggle import added the four schemas below on 2026-11-23.
* **Sizing trap.** `pg_total_relation_size()` and `pg_class` row counts do **not** include compressed hypertable chunks: `ohlcv` shows 16 kB and `-1` rows there while holding 295 million rows. Use `hypertable_size('public.ohlcv')` for size and `count(*)` or the `timescaledb_information` views for rows.
* **`public.ohlcv`: the existing bars table** (hypertable, 7-day chunks, 1,402 chunks, compression policy after 7 days, compressed by `symbol, interval`, ordered by `timestamp DESC`). Columns: `timestamp` (**timestamp without time zone**, naive UTC; a client set to Pacific time shows midnight as 08:00Z, the stored value is midnight), `symbol`, `interval`, `open`, `high`, `low`, `close`, `volume` (all `double precision`, nullable), `trades` (integer), `venue`. Primary key `(timestamp, symbol, interval, venue)`, so **the same symbol and day can exist under more than one venue**; there is also an index on `timestamp DESC`. What it holds:

  | venue | interval | rows | symbols | span |
  | :--- | :--- | ---: | ---: | :--- |
  | `yfinance` | `1d` | 5,777,834 | 1,443 | 2000-01-03 to 2026-11-05 (**the equity tape; every stock tool reads this**) |
  | `kraken_spot` | 1m, 5m, 15m, 30m, 1h, 4h, 12h, 1d | about 289 million | 1,097 to 1,611 per interval | 2023-01 to 2026-06 (crypto, parked) |
  | `binanceus_spot` | 4h, 1d | about 271,000 | 53 to 78 | 2023-01 to 2026-06 (crypto, parked) |

  `ggTrader.data.bars.load_bars` (the research entry point) hard-codes `VENUE = "yfinance"` and `INTERVAL = "1d"`, selects `open, high, low, close, volume` for a list of symbols, and **raises** if stored stamps are not unique per `(timestamp, symbol)`, not midnight, or fall on a weekend. `load_bars` defaults to `source="db"`, reading Yahoo rows through the newest available row rather than stopping at the snapshot’s 2026-10-30 end. Select the frozen Parquet tape named in `data/snapshots/MANIFEST.json` with `--snapshot`, `source="snapshot"`, or `--source snapshot`; **on 2026-11-22 that file (`ohlcv_20261106.parquet`) was not on disk**, only the manifest, so a run that needs it fails until it is rebuilt (`scripts/make_tape_snapshot.py`). The snapshot is a test fixture from the engine rebuild (605 S&P 500 symbols, 2019-03 to 2026-10), not research data. The combined source described below also reads a published CRSP + Yahoo release. Symbols are tickers in the form `normalize_yf_ticker` produces (for example `BRK-B`); the live loader that fills the equity rows (`data/live/yfinance_loader.py`) calls yfinance with `auto_adjust=True`, so the bars are meant to be split- and dividend-adjusted (the stored values were not audited when this map was written).
* **Other bars and crypto tables.** `perp_ohlcv` and `funding_rates` (perpetual futures, tiny), `ohlcv_no_data` (symbol/interval pairs checked and found empty, 173 rows), `symbol_inception` (first date per symbol, 141 rows), `ohlcv_yf_preshift_backup_20260820` (548 MB copy of the equity tape from before the 2026-08-20 one-day-shift fix: **keep it**, it is evidence).
* **Alternative data (equities).** `form4_transactions` (insider trades, 746,813 rows, 1,309 symbols), `house_ptr_transactions` (congressional trades, 36,263), `short_volume` (daily, 1,186,767 rows, 2018-08 to 2026-07, 674 symbols), `short_interest` (83,958 rows, 2020-04 to 2026-06, 629 symbols), `earnings_surprise` (61,299 rows, 1998-12 to 2026-07, 762 symbols), `fred_series` (74,641 rows, 11 series, 1934 to 2026-09), `fear_greed_index`, `google_trends_interest`, `news_headlines` with `headline_sentiment_scores`, `correlation_matrices`. **Dates in `form4_transactions` and `house_ptr_transactions` include impossible values** (years 0015 and 2031 in the first, 3031 in the second): filter them and never trust a min or max date there.
* **Lab results.** `lab_runs` (292), `lab_plans`, `lab_summary`, `lab_sweeps`, `lab_sweep_combos`, and the hypertables `lab_returns` and `lab_equity`; see State Storage above. Older engine and crypto-era results sit in `runs` (1,017), `trades`, `orders`, `equity_curves`, `performance_metrics`, `wfo_splits`, `wfo_windows`, `wfo_cache`, `study_results`, `regime_states`, `kraken_ledger`, `live_balance_snapshots`, `live_positions_snapshot`.
* **Paper trader state.** `paper_snapshots` (80 days), `paper_trades` (183), `paper_pending_orders`, `paper_rebalance_state`, `paper_risk_state`, `paper_split_state`, `paper_dividend_accruals`, `system_state`, `universe_cache`, plus `*_backup_*` copies taken before earlier fixes (keep them).
* **`universe` schema.** The versioned S&P 500 membership list: `membership` (1,255 rows), `snapshot_dates` (8,156), `source_snapshots` (57), `loads`, `load_blobs`, `load_evidence`, `checks`, `check_blobs`, `symbol_changes`. It is the model for any new versioned import: a `loads` row per load (source, SHA-256 of the raw and normalized content, normalizer version, counts, loader git SHA, who and when), append-only triggers that fire for every role, idempotent DDL, no `timestamp without time zone`.
* **Backups.** A full dump of the database was taken to `/media/thesix/ggTrader-data/ggtraderdb-backup-20261122/` on 2026-11-22 (9.7 GB, readable, SHA-256 recorded; a full restore drill has not been run). It is a one-off, not a schedule. The system backup (`borgmatic`, config `~/borgmatic_config.yaml`) **excludes Docker volumes** and the folder `~/ggtraderdb`. That folder holds a single 14 GB dump directory from 2026-02-18; the only database dump I found (2026-11-22). No scheduled dump of `ggtrader` turned up in cron, the user timers or the scripts. Treat the live volume as unbacked-up until the owner confirms otherwise, and say so before any bulk load or destructive change.
* **Kaggle import (loaded 2026-11-23).** Apply the schemas with `ggt db import apply-ddl`: `ref` holds loads, release records, symbol maps and splice links; `market` holds `daily_bars` and `stock_spans`; `kaggle_crsp.daily` holds raw CRSP and `kaggle_chars.mcgill` holds McGill. Load 1 is CRSP daily: 68,879,302 rows, zero quarantined, 25,754 stocks, 1968-01-02 to 2024-12-31. Load 2 is McGill-FIAM: 273,373 rows, 3,349 stocks, 2000-01-31 to 2023-12-29, 166 columns; only the symbol map and closing check 3 read it. Load 3 contains 68,879,302 derived CRSP bars. Read completed loads through `kaggle_crsp.daily_of(load_id)` and `kaggle_chars.mcgill_of(load_id)`.
* **Published release 1.** It contains 916 `linked_yahoo`, 495 `yahoo`, 24,838 `placeholder` and 3,034 `spell` columns; 72 links were demoted. The CRSP file covers 25,754 stocks, of which 21,950 stop before its end. `load_bars(source="combined", release_id=None)` selects the newest published release. `ggt lab --source combined --release-id N --dead-stock-loss FLOAT` selects a release and loss from 0 to 1. The lab sells dead stocks on their last bar and prevents later orders; the default sale at the last price is optimistic about what holders recover. A run extending past the CRSP end, 2024-12-31, is refused when a loaded column ends on that date. A combined read has a fixed cost of about 3–5 seconds; 916 stocks over all history took about two minutes and 1.8 GB. The rule and its limits are in `docs/kaggle_import_runbook.md`.
* **Storage and Yahoo check (2026-11-23).** The database grew from 14 GB to 17 GB; compressed `kaggle_crsp.daily` is 1.9 GB and `market.daily_bars` 0.9 GB, with about 150 GB free. The Yahoo tape is unchanged: 5,777,834 yfinance daily rows for 1,443 symbols, 2000-01-03 to 2026-11-05. Reads with main's own code matched in 13 of 13 batches, using SHA-256 over all five fields. Raw Kaggle and Stooq downloads remain under `/media/thesix/ggTrader-data/market-data/`; the CRSP copy has no open, high, low or volume.

---

## 2. Rules & Deployment Nuances

* **Docker Research Environment**: 
  * The `ggtrader_live` container is the **production paper-trading** container — it is what cron ran at 12:45 PT until the owner stopped the trader on 2026-11-06 (cron lines commented out; nothing runs it now, and cron returns only on the owner's explicit say-so). It has the whole package copied in (not bind-mounted), so **code changes do not reach it until the image is rebuilt and pulled**: push to `main`, let `.github/workflows/docker-build.yml` publish `ghcr.io/garykuepper/ggtrader:latest` (~2.5 min), then `docker compose pull && docker compose up -d`. Forgetting this ships stale code to live; it has bitten twice.
  * For local development and research on the host, install with `pip install -e .` into a virtual environment and run natively — research does **not** need Docker.
  * **Research command** (if you do want the container): `docker compose run --rm ggtrader_live python ggt.py lab --strategy <name>`
  * **Database connectivity**: Inside Docker, the TimescaleDB connection string uses `host.docker.internal:5433`. On the host, use `localhost:5433`.
* **Lab Run Workflow**:
  1. **Strategy Selection**: Choose a strategy from `REGISTRY` in `src/ggTrader/lab/strategies/registry.py` — the ten ported strategies and any module you add (discovered by scanning the package); the other 28 historical names (`LEGACY_ONLY_STRATEGIES`) are closed NO-GO or untested research whose code lives only at git tag `legacy-engine-vbt0285` (`scripts/legacy_env.sh`). Read the registry rather than trusting any list written here. `ensemble` is the deployed live strategy; check `research/RESEARCH_SNAPSHOT.md` before spending a run on a closed one.
  2. **Universe Selection**: The lab auto-generates the trading universe for the eval period — S&P 500 constituents for equities, read point-in-time from the membership list through `MembershipSource` (`ggTrader.data.membership`), or top-volume coins for crypto. On `ggt lab` the source is always named, never defaulted: `--membership db` (latest vintage), `db:<load_id>` (a pinned vintage) or `fixture:<name>`, else the `GGT_MEMBERSHIP` environment variable, else the run stops with `MembershipSourceNotSelected`. The two oracle drivers (`scripts/p5_wfo_validation.py`, `scripts/p7_live_golden.py`) deliberately default to the frozen fixture `sp500_v1_2026-06-02` so the reference results cannot drift. The old file `data/universe/sp500_constituents_history.csv.gz` is the immutable vintage-1 fixture `sp500_v1_2026-06-02` (the basis of every committed reference number); it is never edited. `--expect-membership-sha` pins a run to one exact list.
  3. **Walk-Forward Execution**: Overlapping monthly folds are scored from slices of one grouped vectorbt simulation (Rust engine). Each fold trains on historical data (in-sample) and validates on held-out future data (out-of-sample).
  4. **Persistence**: `lab_runs` stores run metadata (strategy, config, timestamps); per-fold and per-sweep detail goes to `lab_plans` / `lab_returns` / `lab_equity` / `lab_summary` / `lab_sweeps` / `lab_sweep_combos`. `ggt lab --wfo` writes only its JSON record (with a provenance block); `--blend` writes a `blend:` row. Research **drivers** (in a study's `code/` folder, or under `scripts/` for the engine evidence drivers) do record their results as JSON (e.g. `scripts/p5_wfo_validation.py` → `research/programme/vectorbt-engine/p5-evidence/_p5_<run>.json`), and any new driver should do the same so a cited number has a raw artifact behind it.
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
* **Adding a study** (a research question with its own brief, report and code; not a new engine strategy):
  * Scaffold it with `scripts/study.py new CATEGORY SLUG --kind {screen,wfo,event,audit,prospective} --placement ROW`. The placement row comes from the table in `research/README.md`. It creates the study folder with a README, a `study.toml` card, a brief from `research/TEMPLATE-brief.md`, `code/run.py`, one module, and `code/tests/`.
  * The driver `code/run.py` calls `ggTrader.utils.study.bootstrap(__file__)` before importing its sibling modules (this puts `code/` and the category's `_lib/` on `sys.path` and returns the repo root; never hand-roll `parents[N]` or `sys.path` edits). It has `main()`, `argparse`, a `__main__` guard, no work at import time, and a pinned `--eval-end` (never "today").
  * A study imports `ggTrader` (engine, utils, data); it never imports vectorbt directly, and never imports another study's `code/` (shared code goes in the category's `_lib/`). Put its tests in `code/tests/` with `pytestmark = pytest.mark.study` (the scaffold does); they run in `--studies` and `--full`, not in the fast core.
  * `scripts/study.py check STUDY_FOLDER` checks the naming, the card, the brief's registered parameters, the driver rules and the import boundaries. When the study closes, `scripts/study.py close` writes `PIN.json` (commit, lock hash, data hashes, command) and `scripts/study.py repro` re-runs it in a scratch clone.
  * Moving a file out of `src/`, `scripts/` or `tests/` into a study adds a row to `research/CODEMOVED.csv` (old path, new path, hashes). Evidence files that pin a code hash resolve through it; `scripts/verify_pins.py` checks all pins.
* **Data Access**:
  * Equities OHLCV for research: `load_bars()` from `src/ggTrader/data/bars.py` (frozen Parquet tape or a read-only database session; no network, no writes). The live equity-price path is `ggTrader.data.live.fetch_chain` (`OhlcvFetchChain`: database cache, then yfinance, then Tiingo for the gaps), imported directly by `src/ggTrader/paper/signal_runner.py` and `overlay.py`. The replica tools use `fetch_stock_ohlcv()` / `load_ohlcv()` of `src/ggTrader/lab/data.py`; `lab.data.fetch_stock_ohlcv` stays as a thin wrapper over that chain with the previous public arguments.
  * Crypto OHLCV: TimescaleDB loader in `src/ggTrader/data/historical/timescaledb_loader.py` fetches keyed by venue and interval.
  * SP500 constituents: read through `ggTrader.data.membership` from the database vintages (`db`, `db:<load_id>`) or the frozen fixture `fixture:sp500_v1_2026-06-02` (the file `data/universe/sp500_constituents_history.csv.gz`, never edited), chosen with `--membership` or `GGT_MEMBERSHIP` and never defaulted; a refresh is a new vintage (`ggt db load-sp500`, or `ggt db load-sp500 --from-events` for a curated one).
* **Backtesting with vectorbt**:
  * The lab simulates through `ggTrader.engine.portfolio` (`run_signals` / `run_weights`), which wraps `vectorbt.Portfolio` in one grouped call. Passed 2D signal arrays (dates × symbols × combinations) and OHLCV data.
  * Weights are set daily (or per-rebalance) — the portfolio auto-rebalances on entry/exit signals.
  * No per-symbol state — vectorbt is stateless. Any state needed across bars (e.g., time-in-trade) must be computed before simulation as a separate signal.
  * Always use cash sharing and group_by for multi-symbol portfolios (prevents independent cash pools).
* **Performance Analysis**:
  * The lab computes Sharpe, Sortino, CAGR and max drawdown through `engine/stats.py` (vectorbt's definitions, 252-day year).
  * Monthly folds allow out-of-sample validation — `wfo` handles fold generation and metrics aggregation.

* **Running the test suite**:
  * Always run tests via `scripts/run_tests.sh` (takes any pytest args), never a bare `pytest`. It wraps the run in a transient cgroup with `MemoryMax=4G` (2026-11-18; `GGT_TEST_MEM_MAX`, it was 8G) and swap disabled.
  * **Tiers (2026-11-07).** `scripts/run_tests.sh` with no arguments (or `--fast`) runs the FAST CORE, the developer loop (the Rust engine only; the `slow`, `numba`, `tripwire`, `study`, `tooling` and `integration` markers are left out, and it says so). `scripts/run_tests.sh --full` runs everything except `integration` and is the gate before any merge, deploy or vectorbt upgrade; `--studies` runs the study tier: the cases marked `study` plus the tests that live inside the studies (`code/tests/` of each study and `_lib/tests/` of each category, collected explicitly and in importlib mode because file names may repeat), and fails if it collects fewer than `STUDIES_MIN_COLLECTED` (in `tests/tiers.py`). `--full` runs that study half too. Any other arguments go to pytest unchanged and run every test in the paths named. The markers are registered in `pyproject.toml` and the tiers are defined once in `tests/tiers.py`. CI runs the fast core on every pull request and the full suite on every push to main and weekly; it cannot run the real-tape oracles or the live golden (the tape is git-ignored), so run `--full` locally before a merge.
  * This is not optional hygiene. On 2026-09-16 a full `pytest tests/` run hit ~10 GB, exhausted 31 GB of RAM *and* the 8 GB swap file, and wedged the whole Encom host for minutes at a time — five separate times in one day. Swap-file thrashing, not the allocation itself, is what made the box unreachable over SSH and Telegram.
  * **Likely root cause (2026-09-22): not a leak but an infinite loop.** `main` shows no growth; the branch built that day reproduces the blowup. An unstubbed DB read in `run()`-level paper tests let the connection pool's `time.time()` calls consume a mocked clock, so `_poll_orders` spun forever while allocating. Fixed in `6a052fa`; a full gate (core plus studies, with the real-tape tests) was measured on 2026-11-18 at 1.84 GiB or more in about 7 minutes: a 2G cap kills the real-tape live-golden test, 4G does not.
  * **Agent cap scope (2026-11-19).** The runners start their scope with `systemd-run --slice-inherit`, so it lands inside the caller's slice: a test run started from a T3 agent sits under `app-t3code-agents-*.slice` and counts against the T3 agent memory cap. Tests started from an SSH shell are not in that slice: they run outside the T3-specific agent cap, but still inside Encom's outer user-slice limit (`user-1000.slice`) and the per-run `MemoryMax`.
  * A `137` or `143` exit is inconclusive: it may be a cap kill, an oomd kill, the runner's time limit or an external signal. If it was a hang rather than a big suite, look for a test that touches a real DB, network, or clock before raising `GGT_TEST_MEM_MAX`. Do not remove the cap.
  * **One heavy job at a time (2026-11-18).** Two parallel `tests/paper/` runs, started by parallel builders, each grew toward their own cap, together needed more than Encom had free, and nothing was killed: the host thrashed until it was power-cycled. The cause of the growth was a logger left at INFO by the `ggt` CLI callback: on Python 3.12 every log record reads `time.time()`, which starved paper tests that stub the global clock, so `_poll_orders` never reached its deadline (the root `conftest.py` now restores the `ggTrader` logger after every test). A per-run cap cannot stop two runs adding up, so `scripts/run_tests.sh` and `scripts/run_capped.sh` share a guard (`scripts/_heavy_job.sh`): a lock in the git directory, shared by every worktree, lets one job run at a time (a second waits `GGT_HEAVY_WAIT` seconds, default 1800, then exits 75); a job exits 75 at once when `MemAvailable` is below its cap plus `GGT_HEAVY_MARGIN` (default 4G); each pytest run is stopped after `GGT_TEST_MAX_SECONDS` (default 1800) and a `run_capped.sh` job after `GGT_RUN_MAX_SECONDS` (default 7200), exit 143; and each test is stopped after `GGT_TEST_TIMEOUT` seconds (default 180; `@pytest.mark.timeout_seconds(n)` overrides) so a hang fails the test instead of eating memory. Exit 75 means busy or short of memory, not a failure; exit 137 or 143 means the run was killed, cause unverified (possibly a hang). CI (`GGT_TEST_NO_CGROUP=1`) skips the lock, the preflight and the runner's time limit; the per-test timeout still applies there.
  * **Rules for builders and delegates (2026-11-18).** Run tests only as `scripts/run_tests.sh <small targeted paths>`. Never run `tests/paper/`, a whole tier or an oracle driver unless the brief names it: the orchestrator runs those, one at a time. If a run prints `waiting for another heavy ggTrader job`, wait; never kill another run and never bypass the scripts. Put this block in every delegated brief that may run tests.
  * Tests must never read the real `.env`: the repo-root `conftest.py` makes `load_dotenv` a no-op for every test, wherever it lives, so live flags (e.g. `CASH_SWEEP_ENABLED`) can't leak in. `tests/paper/test_trader.py` has an autouse guard (`_unstubbed_persist_guard`) that replaces the default `PgPaperLedger` factory and fails any unstubbed persistence call, even if the trader swallows the error; give each test an `InMemoryLedger` or `SpyLedger` (`tests/paper/support.py`).

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
5. **Single Source of Truth**: Engine logic lives in `src/`; a study's logic lives in that study's `code/` folder (or the category's `_lib/` when several studies share it). Do not duplicate definitions across files.
6. **Module structure**: constants at module level (top), not buried inside functions; execution logic behind an `if __name__ == "__main__":` guard.
7. **Scripts**: Utility scripts in `scripts/` should have a `main()` entry point and `argparse` for inputs, and every script must handle `--help` gracefully. A study's drivers may live in that study's `code/` folder instead; the same rules apply and `scripts/study.py check` enforces them.
8. **Resilient, resumable scripts**: long-running work (backfills, ingestion, WFO drivers) must log and continue on recoverable errors rather than crashing, and must checkpoint for resumability if it can run longer than ~5 minutes. Several drivers (in `scripts/` or in a study's `code/` folder) run for 30-60 minutes; losing an hour to an unhandled exception at minute 55 is the failure this prevents.
9. **Imports**: absolute, rooted at `src` (`from ggTrader.<package> import ...`) for cross-package access; relative imports (`from .module import X`) only inside a package's `__init__.py`. Avoid circular imports by keeping `utils` free of domain logic.
10. **Dependency discipline**: record new libraries in `pyproject.toml` as you add them. Never `pip install` something the project then depends on without recording it.
11. **Path Safety**: Always use `os.path.join` or `pathlib.Path` for file paths. Resolve project root dynamically.
12. **Error Handling**: Data loading functions must raise descriptive exceptions (e.g., `ValueError`, `FileNotFoundError`) on failure. Avoid returning `None` from functions expected to return iterables (handle empty data by returning empty structures).
13. **Vectorization First**: Avoid iterating over rows in DataFrames for signal calculation. Use `vectorbt`, `numpy`, or `pandas` vectorized operations. All strategy signals must be fully vectorized arrays (dates × symbols) before being passed to `vectorbt.Portfolio`.

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
* **Where docs live**: research reports and their artifacts, and the code of the study that produced them, live in the top-level `research/` directory, one folder per study (`research/<category>/<study>/`; the one exception is `research/audits/root-cause-2026-11-04/`, which has more than 30 files and groups them into sub-folders named for what they are), with dated filenames `<YYYY-MM-DD>-<slug>[-nogo].md`. Pick the category with the ordered placement test in `docs/superpowers/specs/2026-11-05-research-reorganization-and-consolidation-design.md` section 5: the first matching test wins, and a document that touches two categories gets cross-links, never a second copy. `docs/` holds architecture, the CLI reference, the changelog, the roadmap, next steps, the working task list, installation notes, examples, `superpowers/` specs and plans, and `archive/`; new research reports do not go there, linked with relative paths so it resolves both on disk and on GitHub. Point-in-time documents (snapshots, one-off reports) go to `docs/archive/` once superseded.
* **Periodic review**: after any major refactor, audit the docs and purge references to deleted modules, archived scripts, and removed commands. This is not optional housekeeping — stale instructions actively mislead. The 2026-08-21 audit found `AGENTS.md` itself asserting a `lab_periods` table that has never existed and denying the existence of the live trading path, and four separate competing ruleset files naming symbols deleted months earlier.
* **Changelog**: Add an entry to `docs/changelog.md` whenever strategies, data sources, or lab infrastructure changes. Include what changed, why, and research results if available.
* **CLI Reference**: Keep `docs/cli_reference.md` synchronized with actual `ggt` commands and flags. Document `ggt lab`, `ggt db`, and `ggt paper` subcommands.
* **Architecture Guide**: Maintain `docs/architecture.md` as the authority on lab structure, data flow, and module responsibilities.

### The research documents that actually matter

These are the most actively maintained docs in the repo and the ones to read
*first* when picking up work. Earlier revisions of this file omitted all four.

* **`docs/next_steps.md`** — the current worklist, deliberately only 1–2 steps deep. Start here.
* **`docs/roadmap.md`** — the North Star, where things stand, the open tracks and a condensed timeline; the older long-form history is `docs/archive/roadmap-2026-06-to-2026-11.md`.
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
