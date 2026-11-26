# ggTrader agent instructions

## Production freeze and owner instructions

- Keep the production paper trader and its cron stopped until the owner explicitly authorizes resumption. Use dry-run only during the freeze and leave production paper state untouched. Never place live orders. During the freeze, send no orders to the Alpaca paper broker.
- Paper trade first. The owner decides when a strategy has earned real money.
- Focus recommendations on research, strategy evidence and paper-trading reliability. Discuss retiring the stock-picking sleeve, ending paper trading or shutting down the project only at the owner's explicit request.
- The account is taxable only. Do not recommend IRAs or other retirement accounts or ask the owner to reconsider the account type.
- These owner instructions override older worklists, reports, prompts and handoffs. Preserve honest NO-GO results and the production freeze. A NO-GO gives no authority to shut down or change the account.
- For an authorized deployment, read [Architecture](docs/architecture.md) and the [CLI reference](docs/cli_reference.md). The production image bundles the code. Ship from `main` through
  `.github/workflows/docker-build.yml` and pull and recreate the container with the built image before expecting changed code in production.

## Research

- Start in [Next steps](docs/next_steps.md). Consult the [roadmap](docs/roadmap.md), [verdict roster](research/RESEARCH_SNAPSHOT.md) and [candidate backlog](research/WEB_RESEARCH_CANDIDATES.md). Regenerate the roster with the `research-snapshot` skill.
- For trading and finance fundamentals, search the `gg-library` MCP under `Trading` and `Mathematics-Statistics`. Broaden if useful, read the relevant passages and cite sources with locators. Use the skill's CLI fallback if MCP is unavailable and the `research` workflow for online research.
- Use data the strategy never saw during tuning. Use only information available on each date and include stocks that later died. Subtract trading costs and taxes and compare against SPY on the same dates.
- The research reset retired Part A, the Holm and Newey–West t-statistic bars, the historical excess-return threshold, the B0/B1 shadow stages and the six-month clean paper-trading wait for new work. Do not cite them as requirements.
- Keep old reports, scores and verdicts, including NO-GO findings, as history. Never delete or rescore them.
- Talk to the owner in plain words. Define any necessary technical term in the same sentence.
- Pin both `--eval-start` and `--eval-end` for every result you cite. Compare results over identical dates and save raw JSON next to the report.
- Report NO-GO plainly with honest held-out results against SPY. Name failed gates and regime halts, and report any collapse at its measured magnitude.
- Check leverage against the production value of 1.0 when interpreting blend runs. Label higher-leverage results and do not present them as deployed performance.
- WFO writes provenance-bearing JSON only. Blends persist a `blend:` run and associated tables through `src/ggTrader/lab/persist.py`. New evidence drivers also write raw JSON.
- An optional lab summary gives mean monthly return, Sharpe, max drawdown and win rate across folds, the top five assets by risk-adjusted return, passing parameter ranges and equity and drawdown plots if generated.

## Data sources and lab

- Use `load_bars()` from `src/ggTrader/data/bars.py` for research equities. It is read-only and uses no network. Its default is `source="db"`, which reads the latest available Yahoo daily bars (`venue=yfinance`, `interval=1d`) and rejects duplicates, non-midnight stamps and weekends.
- Select the frozen tape explicitly with `--snapshot`, `--source snapshot` or `source="snapshot"`. It is an engine test fixture. Check `data/snapshots/MANIFEST.json` and the file before a run that needs it. Rebuild with `scripts/make_tape_snapshot.py` if needed.
- For the published CRSP + Yahoo tape including dead stocks, choose `source="combined"`. The default is the latest published CRSP + Yahoo release. Pin `--release-id` for a specific release and document the `--dead-stock-loss`
  assumption. The lab sells dead stocks on their last valid bar, blocks later orders and defaults to an optimistic last-price recovery. Check the CRSP end guard and release limitations in the
  [Kaggle import runbook](docs/kaggle_import_runbook.md) before running or interpreting combined data.
- For paper pricing, preserve the cache → yfinance → Tiingo gap-fill chain in `ggTrader.data.live.fetch_chain` and the thin `fetch_stock_ohlcv()` compatibility wrapper in `ggTrader.lab.data` with its existing arguments. Use the TimescaleDB historical loader keyed by venue and interval for crypto.
- Select point-in-time S&P 500 membership through `MembershipSource` with `--membership db`, `db:<load_id>` or `fixture:<name>`, or set `GGT_MEMBERSHIP`. Missing selection stops the run. Use `--expect-membership-sha` for an exact pin.
- Keep `data/universe/sp500_constituents_history.csv.gz` immutable. The validation and paper golden drivers default to its frozen `sp500_v1_2026-06-02` fixture.
- Before running the lab, read the strategy registry and verdict roster. Legacy names refer to the `legacy-engine-vbt0285` tag. See the [CLI reference](docs/cli_reference.md) for the current flags and defaults of the single `ggt` entry point.
- Use an editable install in a host virtualenv for research. For an optional container run, use
  `docker compose run --rm ggtrader_live python ggt.py lab --strategy <name> --wfo --membership fixture:sp500_v1_2026-06-02`.
  Database endpoints use `localhost:5433` on the host and `host.docker.internal:5433` in Docker.
- Use grouped Rust simulation through `ggTrader.engine.portfolio` with cash sharing and `group_by` for multi-symbol portfolios. Precompute cross-bar state as a signal. Score held-out monthly folds and use
  `engine/stats.py` for metrics with 252 trading bars per year. Read [Architecture](docs/architecture.md) before changing engine, lab or paper seams.

## Database

- Before querying or writing an importer, inspect the live schema read-only. Remeasure reported counts and use `hypertable_size()` and chunk-aware counts or views for compressed hypertables. Consult the [dated inventory](docs/archive/database-inventory-2026-11-23.md) for historical evidence.
- Filter by venue and interval because bars can share a symbol and timestamp across venues. Write naive UTC into `ohlcv.timestamp` and never supply a tz-aware datetime. Use normalized Yahoo equity tickers. Only
  for crypto venues with `ASSET-QUOTE`, split with SQL `split_part`. Prefer `volume * close` for cross-asset ranking.
- In queries on `form4_transactions` and `house_ptr_transactions`, filter impossible dates. Never trust unfiltered date extrema.
- Treat the live volume as unbacked-up until the owner confirms otherwise. State this before any bulk load or destructive change.
- Keep `data/`, database configuration, database backup copies, paper-state `*_backup_*` tables and pre-shift equity-tape evidence (`ohlcv_yf_preshift_backup_20260820`) out of automated cleanup. Check inbound references from tracked docs before removing unreferenced generated results or stray logs.
- Refresh S&P 500 membership by appending a vintage. Only `load-sp500 --commit` or `membership reinstate --commit` writes one. Weekly jobs only propose. The only existing-row update marks `superseded_by_load_id`.
- Use `sql/universe_schema.sql` and the [CLI reference](docs/cli_reference.md) for membership DDL and curated loads. For versioned imports, use append-only triggers for every role, idempotent DDL and timezone-aware timestamps.
  Record one loads row per import with source, raw and normalized SHA-256, normalizer version, counts, loader Git SHA, actor and time.
- For import backup gates, DDL, release operations and complete-load readers (`kaggle_crsp.daily_of` and `kaggle_chars.mcgill_of`), read the [Kaggle import runbook](docs/kaggle_import_runbook.md).

## Implementation and documentation

- Before adding a strategy, read [Adding a strategy](docs/adding_a_strategy.md) for the contract, discovery and lookahead checks. Only `src/ggTrader/engine/` may import vectorbt.
- Keep engine logic in `src/` and study logic in its own `code/` or the category's shared `_lib/`. Keep definitions single-sourced and keep `src/` and other studies from importing a study's private code.
- Scaffold studies with `scripts/study.py new CATEGORY SLUG --kind KIND --placement ROW` using the placement table in [Research](research/README.md). Call `ggTrader.utils.study.bootstrap(__file__)` before sibling imports. Avoid manual path edits and import-time work.
- Put study tests in `code/tests/` with `pytestmark = pytest.mark.study`. Pin the driver evaluation end. Drivers also follow the script rules below. Use `scripts/study.py check` for validation, `close` to write the reproducibility pin and `repro` for scratch-clone reproduction.
- Record moves from `src/`, `scripts/` or `tests/` into studies in `research/CODEMOVED.csv` with old and new paths and hashes. Use `scripts/verify_pins.py` to verify evidence references.
- Use PEP 8, four spaces, lower_snake_case files, snake_case functions and variables, CamelCase classes and UPPER_CASE constants. Group and alphabetize standard, third-party and local imports. Type every argument and return value.
- Use a one-line module docstring, brief Google-style function and class docstrings and comments that explain why. Keep lines within the Ruff limit of 100 characters. Pass `ruff check .` and Ruff formatting. Enable `.githooks` once per clone via `git config core.hooksPath .githooks`.
- Put constants at module level and guard script execution with `if __name__ == "__main__"`. Give scripts `main()`, `argparse` and working `--help`.
- Log and continue on recoverable errors. Checkpoint work that can take more than about five minutes so it can resume.
- Use absolute `ggTrader` cross-package imports and reserve relative imports for a package's `__init__.py`. Keep domain logic out of `utils` to avoid cycles. Record every new dependency in `pyproject.toml`.
- Build paths with `pathlib.Path` or `os.path.join` and resolve the root dynamically. Raise descriptive data loading errors and return empty collections for empty iterable results.
- Vectorize signal calculation with the engine, NumPy or pandas. Supply fully vectorized date × symbol signals to simulation.
- Reflect core architecture changes in [README](README.md) and affected docs. After major refactors, audit stale paths and commands. Keep the [architecture guide](docs/architecture.md) and [CLI reference](docs/cli_reference.md) synchronized.
- Update the [changelog](docs/changelog.md) for strategy, data-source or lab infrastructure changes. Include what changed, why and research results when available.
- Keep a single study home and closed-study report under `research/<category>/<study>/`. Use the first matching
  [placement rule](docs/superpowers/specs/2026-11-05-research-reorganization-and-consolidation-design.md), relative cross-links and `<YYYY-MM-DD>-<slug>[-nogo].md` with
  [the report template](research/TEMPLATE-research-report.md). The existing root-cause audit retains its subfolders. Archive superseded point-in-time docs in `docs/archive/`.

## Tests and heavy jobs

- Always use `scripts/run_tests.sh` for tests. Never use bare pytest. The default or `--fast` runs the fast core. Use `--studies` for marked and in-study tests with its minimum-collection guard. Run `--full`
  locally before merge, deploy or vectorbt upgrade. It includes the study tier and excludes integration. Other arguments pass through to pytest.
- CI runs fast on PRs and full on pushes to main and weekly, but lacks the real tape for local oracles. Read `tests/tiers.py` and the [CLI reference](docs/cli_reference.md) for the tier definitions.
- Run one heavy job at a time across all worktrees. Use `scripts/run_capped.sh` for heavy drivers. Both wrappers share the git-directory lock and require available memory above the cap by `GGT_HEAVY_MARGIN`
  (default 4G). A contender waits up to `GGT_HEAVY_WAIT` (default 1800). Exit 75 means busy or insufficient memory.
- The test scope uses MemoryMax=4G by default (`GGT_TEST_MEM_MAX`), with swap disabled. It inherits the caller's slice, so T3 runs also count against the agent cap. SSH runs still count against Encom's user slice and the per-run cap.
- The time limits are `GGT_TEST_MAX_SECONDS` (default 1800) and `GGT_RUN_MAX_SECONDS` (default 7200). Individual tests use `GGT_TEST_TIMEOUT` (default 180), with `@pytest.mark.timeout_seconds(n)` for a test-specific override.
- CI's `GGT_TEST_NO_CGROUP=1` skips the lock, free-memory preflight and runner time limit. The per-test timeout still applies in CI.
- Exits 137 and 143 mean termination with cause unverified. Check for cap, oomd, time limit or an external signal. Investigate real database, network and clock use in a hang before raising the memory cap. Keep the cap enabled.
- Builders and delegates run only targeted paths through `scripts/run_tests.sh`. Run the whole paper suite, any tier or oracle only if named in the brief. Otherwise the orchestrator runs them serially. If the
  runner reports another heavy job, wait without killing it or bypassing the wrappers. Include these rules in every test-running brief.
- Tests must never read the real `.env`. Preserve the root dotenv stub and inject `InMemoryLedger` or `SpyLedger` in paper trader tests, keeping the unstubbed persistence guard enabled.

## Delegation and financial privacy

- Follow the home instructions and `~/docs/MODEL_STRATEGY.md` for quotas, routing and independent review. The lead owns final decisions. Keep private account records out of delegate prompts.
- Obtain two independent reviews from different vendors for anything headed for merge.
- As orchestrator, read each slice's diff and rerun the checks yourself.
- Use a stronger model to sample every batch of bulk mechanical work and check every figure that reaches a report.
- Disclose self-review in the report and never count it as independent.
- Small-quota critical-only models, including Astra and Argon-class models, require the owner's approval each time. Use them only if the first-choice reviewer has failed or is unavailable.
