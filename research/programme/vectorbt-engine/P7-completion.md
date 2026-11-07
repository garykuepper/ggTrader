# P7 completion note: the paper trader's signal path rebuilt on the engine (code and tests)

**Date:** 2026-11-06. **Phase:** P7 of the [vectorbt engine rebuild](../../../docs/superpowers/specs/2026-11-06-vectorbt-engine-design.md) (sections 8.2-8.4), code-and-tests part. **Branch:** `vbt12/engine` (nothing committed, staged, tagged or pushed).
Terms: **OD** = open decision in the spec. **Golden** = P0's `live_golden_legacy.json`: for the last 120 sessions, what the pre-rebuild `generate_signals` returned on the 365-day window. **Tape** = the frozen daily-bars file `data/snapshots/ohlcv_20261106.parquet`. **Gate** = a check that stops the run with no orders. **Read-only dry run** = `ggt paper` without `--live` after this change: logs intended orders, writes and sends nothing.

**What was NOT done, by design:** the trader was not run, started, dry-run or pointed at the broker; no order, no Alpaca call, no Telegram message, no database access (the suite stubs them), no Docker build/pull/run, no crontab edit, `.env` not read, `scripts/paper_trade.sh` untouched. Running the actual dry run and re-enabling cron are the owner's (OD18).

## Verdict against the exit criteria

| Criterion | Result |
|---|---|
| Rebuilt signal runner reproduces the P0 live golden (120 sessions of buys, sells, `last_exit`) on the frozen tape | **Met, exactly: 120 of 120 sessions equal** (1,028 buys, 5,519 sells; also `as_of` and `universe_size`). Run through the real `signal_runner.generate_signals` (engine self-check, production sanity bounds, v2 ensemble with the live defaults), 31 s. `p7-evidence/p7_live_golden.json`, test `tests/paper/test_live_golden.py`. |
| Fail closed on every S7 section 4 failure mode | **Met** (table below). |
| Overlay equity parity against the legacy function | **Met**: value-for-value equal (`rtol=1e-12`) on 14 symbols x 700 bars. |
| Nothing in `src/ggTrader/paper` imports vectorbt directly | **Met** (`tests/paper/test_paper_boundary.py`, also bans the legacy simulator, ensemble and indicator imports). |
| Whole suite green via `scripts/run_tests.sh` | **Met: 3,036 passed**, 30 integration deselected, 0 failed, 12 min 32 s (`scripts/run_tests.sh tests/`, 8 GB cap; includes the concurrent P5 and parity agents' tests present in the worktree at that time). `tests/paper` alone collects 458 tests, of which about 140 are new or changed in this phase (95 in the five new files: gates 23, live bars 5, untradable 12, live golden 2, boundary 53). |
| Ruff clean | `ruff check .` and `ruff format --check .` clean over the whole repo. |
| Changelog, docs, skills | `docs/changelog.md` (top entry), `docs/cli_reference.md`, `docs/architecture.md`, `.claude/skills/ggtrader-deploy`, `.claude/skills/daily-trader-check` (the skills exist in the repo, so no `P7-skill-edits.md` is needed). |

## What was built

New files (all under `src/ggTrader/paper/` unless noted):
- `gates.py`: `SanityConfig`, `SignalGateError`, `vet_bars`, `vet_signal_counts`, `late_run_reason`.
- `live_bars.py`: live loader frame to the engine `Bars` container.
- `live_signals.py`: `live_ensemble_frames` (the single place the live ensemble frames are built), `history_eligibility`, `subset_bars`.
- `untradable.py`: `UNTRADABLE_SYMBOLS`, broker status resolution, the "asset is not active" error classifier.
- `scripts/p7_live_golden.py`: the evidence driver (argparse, `--help`, writes JSON).
- Tests: `tests/paper/test_gates.py`, `test_live_bars.py`, `test_untradable.py`, `test_live_golden.py`, `test_paper_boundary.py`; additions to `test_signal_runner.py`, `test_trader.py`, `test_overlay.py`, `test_cli_paper.py`.

Changed: `signal_runner.py` (rebuilt), `overlay.py` (rerouted), `trader.py` (gates, untradable, `read_only`), `notifier.py` (`DryRunNotifier`), `cli/main.py` (`--dry-run-writes`, `--notify`), `Dockerfile`, `docker-compose.yaml`, `.github/workflows/docker-build.yml`, two skills, three docs.

### Signal path (spec 8.2)
`generate_signals` order: (1) `require_rust_engine()` (calls `engine.selfcheck`; the configured engine must be `rust`; any failure raises `SignalGateError`); (2) universe and bars through the existing live loader `fetch_stock_ohlcv` (DB-first, yfinance gap fetch, the partial 12:45 bar), reshaped into `Bars`; (3) `vet_bars`; (4) `Ensemble().build(bars, Ensemble.live_grid())` (OD14: five voters, `min_agree=2`, `min_agree_exit=2`, divergence window 20, volume multiple 2.0); (5) `buys`/`sells` on the last bar among symbols with at least 60 valid closes, sorted alphabetically, `last_exit` from the whole window, as before; (6) `vet_signal_counts`; (7) the unchanged ML feature gate. The cap (`MAX_POSITIONS`, code default 30, environment override) and the alphabetical selection in `trader.py` are untouched. There is no legacy fallback: `EnsembleSignal` is no longer imported anywhere under `paper/`.

**Gap recorded (item 1):** `data.bars` cannot serve the live partial-bar case (it is a read-only reader of a frozen tape or a read-only DB session, with no network), so the existing live loader stays for the fetch step only. The spec's move of the loader chain to `data/live/` was not done: this checkout's project hook blocks writes under `data/live/` ("live trader runtime state"), so `live_bars.py` lives under `paper/`. The loader chain itself still lives in `lab/data.py`; P5's clean-up must keep `fetch_stock_ohlcv` and `STOCK_BASE_CONFIG` there (or move them with their importers `signal_runner.py` and `overlay.py`).

### Sanity gates and the incomplete-bar rule (item 2)
| Gate | Rule (production bound) | Golden margin (120 sessions) |
|---|---|---|
| `engine_selfcheck`, `engine_not_rust` | self-check must pass and report `rust` | n/a |
| `empty_frame` | the loader returned no rows or no symbols | n/a |
| `duplicate_days`, `unsorted_index`, `weekend_bars` | index unique, ascending, Monday to Friday | none seen |
| `small_universe` | at least 450 symbols loaded and at least 90% of those requested | smallest 500 |
| `short_history` | at least 240 bars | n/a (window has ~252) |
| `last_bar_coverage` | final bar has a close for at least 95% of loaded symbols | lowest 98.8% |
| `no_eligible_symbols` | at least one symbol with 60 valid closes | n/a |
| `absurd_signal_count` | at most 100 buys and 250 sells | largest 37 and 127 |
| late run | signals ready after 12:55 PT (or 5 minutes before the broker clock's `next_close` when that is earlier, a half day): no orders | n/a |

**Not-yet-complete bar versus bad tape.** The snapshot day with a close for 1 of 646 symbols is a bar the vendor had not filled in, not a corrupt tape. Rule: if the final bar is below the coverage floor **and** is dated today (ET) or later **and** the bar before it is at or above the floor, the final bar is dropped and signals use the previous session (`trimmed_partial_bar: True` in the result and the log; the trader's existing freshness gate already tolerates that lag). A thin final bar of any other date, a thin previous bar as well, or any structural problem is a bad tape and stops the run. Decision for the owner: the alternative is to stop on any thin final bar (stricter, would sometimes skip a day the rule would trade). The bounds are ESTIMATES from the golden margins above and are to be tuned from the dry-run record (spec 11).

### Failure modes (S7 section 4) and where each is tested
| Failure | Behaviour | Test |
|---|---|---|
| Rust missing or minor-mismatched, engine silently numba | self-check raises or reports non-rust: `SignalGateError`, no data fetched, no orders, Telegram alert | `TestEngineSelfcheck`, `TestFailClosed` (`engine_selfcheck`, `engine_not_rust`) |
| Long first-call compile (numba) | cannot happen: numba is refused; late-run gate bounds any other delay | `TestLateRun`, `TestFailClosed.test_late_run_places_no_orders` |
| pandas 3 datetime/dtype changes | the whole suite and the golden run on pandas 3; write path unit test `tests/data/test_cached_yfinance_loader.py` (naive-UTC records) | golden, existing loader tests |
| Data gaps: missing symbol, stale or duplicate bars | `vet_bars` gates; the freshness gate (as_of within 4 days) is unchanged | `TestVetBars`, `TestSignalPathOnTheEngine` |
| Wrong but plausible signals | the golden (120 of 120) plus the absurd-count gate | `test_live_golden.py` |
| Memory | a 500 x 252 build is small (the golden runs 120 builds in 31 s inside the 8 GB test cap) | n/a |
| Half-run (crash between sells and buys) | unchanged: signals are computed in full before the first order | existing pending-order tests |
| Silent library drift on rebuild | lock-based image, `uv pip check`, selfcheck and trader import at build time and in CI (text only) | Dockerfile, workflow |
| Cron never fires | out of scope (no cron edit); the S7 13:10 heartbeat is not built | n/a |
| Any engine/signal exception | propagates; the trader alerts ("no orders placed" for a gate, "signal generation error" otherwise), re-raises (non-zero exit, the shell trap pages) | `TestFailClosed` |

### Untradable symbols, AVB (OD17, item 4)
The old trader asked the broker for `tradable: false` on **held** names only (AVB, `trader.py:609` at HEAD), so a **buy** signal on a dead ticker (AVB's tape is EQR's since 2026-07-20) would have been submitted and rejected; a sell of a name the status check missed became an entry in `errors`. Now: the untradable set is `UNTRADABLE_SYMBOLS` (environment) plus the broker status for held names and buy candidates; those are excluded from slots, sells and buys (a held-but-frozen name's own buy signal is dropped too, found by a test); and a rejected order whose text says "not active"/"not tradable" (HTTP 422, 403 or no status) is skipped, alerted once per run and returned as `untradable`, not an error. A 422 for another reason is still an error. Tests: `TestUntradableAvb`, `test_untradable.py`. Whether the owner wants AVB converted or removed at the broker (OD17 (b)) is still open; the trader handles it either way.

### Overlay (item 3)
`overlay.compute_sleeve_curve` builds the signals through `live_ensemble_frames` and simulates with `engine.run_signals`; `lab.simulate`, `lab.strategy` and the legacy ensemble are no longer imported by `paper/`. Sweep/A23 code is not in `paper/overlay.py` and was not touched (the cash sweep is `cash_sweep.py`, pure sizing, unchanged). The overlay is the blend path only; the deployed core does not call it.

### Docker, compose, workflow (item 5; text only)
`Dockerfile`: imports the trader path and runs `ggt.py paper --help`; `uv pip check`; `scripts/check_vbt_engine.py --engine rust` and `python -m ggTrader.engine.selfcheck --engine rust` as root and as `appuser` (the one tiny `from_signals` is the self-check's probe). `docker-compose.yaml`: `image: ghcr.io/garykuepper/ggtrader:${GGTRADER_TAG:-latest}` (`latest` is only the bootstrap default; the deploy skill treats an unset or `latest` tag as not deployed); no other change, the service still sleeps. `docker-build.yml`: builds locally, runs the four smoke commands against the finished image, then pushes with the short sha, `sha-<40 hex>` (new) and `latest`.

## Dry-run side-effect inventory (item 7; read from the code, not run)

Line numbers are `src/ggTrader/paper/trader.py` at the branch base (`HEAD`, identical to production) unless another file is named. "Old dry run" = `ggt paper` without `--live` before this change.

**Broker orders: none.** Every `submit_*` call sits under `if self._dry_run` (catastrophe stop 726, sweep sell 810-811, sells 856-858, buys 916-923, sweep buy 954-961). Broker **reads** do happen: account (586, 943, 1029), positions (587, 1031), clock (510), assets for tradability (609), orders for reconcile (184), corporate actions and activities (335, 438).

**Writes by the old dry run** (each is a real write to the production database):
| # | Write | Where | Risk |
|---|---|---|---|
| 1 | DDL `CREATE TABLE IF NOT EXISTS` x7 (`paper_*`) | 495 -> `persist.py:77-83` | none in practice (idempotent) |
| 2 | Price tape: every bar the live loader fetches is upserted into `ohlcv`, plus `symbol_inception` rows and DDL | 519 -> `lab/data.py:103` -> `cached_yfinance_loader.py:356, 377` (`_cache_to_db` :205-285), `:370` (`_record_inceptions` :151), `:128` (DDL) | **highest**: the shared tape; the 2026-08 whole-tape day-shift corruption came from this writer; pandas-3 write path only unit-tested, not against Postgres |
| 3 | Benchmark tape: SPY, TLT, GLD, DBC, IEF, 30 days, upserted (overwrites the trailing partial bar) | 526 -> `signal_runner.py:53` | same as 2 |
| 4 | `paper_snapshots` row, **upsert by `run_date`** | 1089 -> `persist.py:204-222` | overwrites a live run's row for the same ET date; the snapshot feeds the next run's daily-loss baseline (`get_latest_snapshot`, 685), total P&L (`get_earliest_snapshot`), and the split and dividend evidence (`get_snapshot_history`) |
| 5 | `paper_risk_state` peak value (upsert id=1) | 678 -> `persist.py:263-274` | peak is monotone, but it moves the drawdown-halt baseline |
| 6 | `paper_split_state` insert and delete | 393, 326, 380 | changes which corrections later runs apply |
| 7 | `paper_dividend_accruals` insert | 460 | idempotent (`ON CONFLICT DO NOTHING`) |
| 8 | Reconcile of pending orders: books a fill into `paper_trades` (202), deletes the pending row (213), marks it stale (152) | 170-215 | only if an order is pending; books real fills, harmless but a state change |
| 9 | `paper_trades` / `paper_pending_orders` rows for submitted orders | 1005, 1017 | not reached: a dry run submits nothing |

**Telegram messages sent by the old dry run:** "Market closed" (515); "signal generation error" (521); overlay fallback (531, blend only); stale data (563); ML gate (579); untradable holdings (615); halted (672); daily limit (692); the labelled "DRY RUN" lines for catastrophe stop (729), sweep sell (812), sell (858), buy (918) and sweep buy (956); the **unlabelled** full "Paper Portfolio Summary" (1071) and "Risk" line (1083), which read exactly like a live run; and reconcile alerts for real fills and stale orders (146, 203).

**What the read-only dry run does now** (default for `ggt paper`; `--live` unchanged): items 1, 3-8 are skipped through `PaperTrader._write` (the names of what was skipped come back in `skipped_writes` and are printed); item 2 is avoided because `generate_signals(persist_tape=False)` fetches from yfinance directly into memory and never constructs the caching loader; reconcile only reads the pending symbols and does not call the broker or book anything; Telegram goes through `DryRunNotifier`, which logs and forwards nothing (`--notify` forwards only the free-text lines, prefixed `[DRY RUN]`, never the summary or order alerts); the late-run gate only warns. Reads (account, positions, assets, corporate actions, the database tables) still happen. `--dry-run-writes` restores the old behaviour for an owner who wants the snapshot history; it is refused together with `--live`. Tests: `TestReadOnlyDryRun` (nothing in a list of 11 persistence functions is called; the dry and live paths give identical intended orders; the legacy-writes and live defaults are unchanged), `TestRunPaperTradingReadOnly`, `TestPaperCliFlags`, `TestTapeIsNotWrittenWhenAskedNot`.

Not covered by a test: that the read-only dry run touches no database table at all in a real process (the persistence functions are patched, and a read-only database session was not used); the read path still opens ordinary connections for the SELECTs.

## Decisions I made (change them if you disagree)
1. **A dry run is read-only by default**, with `--dry-run-writes` as the opt-out (the brief allowed either direction). Reason: the spec's own inventory shows the old dry run overwrites live state; the deploy skill already told people not to use it as a check.
2. **A thin final bar dated today is trimmed, not fatal** (rule above); the owner may prefer to stop.
3. **The late-run gate only warns in a dry run**, so the path can be rehearsed at any hour.
4. **`latest` stays the compose default** (the brief's example); the immutable tag is enforced by the deploy skill, not by compose, so a checkout without `GGTRADER_TAG` still starts.
5. **Untradable detection also covers buy candidates and the configured list**, a behaviour change from the old trader (held names only).
6. `live_bars.py` placed under `paper/` (the `data/live/` hook), see above.

## What remains for the owner
- Run the actual dry run (`ggt paper`, read-only by default) and review it against spec 8.4 item 3; tune the gate bounds from it. Nothing here ran against the live loader, the broker or the database.
- Decide OD17 (b): have the broker convert or remove AVB, or keep treating it as untradable (set `UNTRADABLE_SYMBOLS=AVB` in `.env` as belt and braces).
- Choose whether a thin final bar of today should be trimmed (current) or stop the run.
- Set `GGTRADER_TAG` to a sha tag at the first deploy; the deploy and image steps are text only and untested until a build runs.
- Re-enable cron only when you say so.

## Uncertain or unfinished
- The live loader's real output (tz, duplicate days on the partial bar when the cached and fresh rows meet) was never seen here; `bars_from_legacy_frame` keeps duplicates so the gate fails closed on them, which is safe but could stop a healthy day. The dry run will show it.
- `Ensemble` is imported from `ggTrader.lab.strategies.v2.ensemble` in `paper/live_signals.py` (and the boundary test names that path). P5 plans to move the v2 modules up; it must update that one import.
- The S7 13:10 PT heartbeat for a missing run, the scratch-Postgres round trip of `_cache_to_db` and `refresh_benchmark_tape` under pandas 3 (spec 8.2), and a `docker-compose run` timeout are not built: they need Docker, a database or the crontab.
- `ggtrader-deploy/evals/evals.json` still asserts "does not use `ggt paper` dry run as the check"; the skill still prefers `verify_live_path.py`, so it stays consistent, but the eval was not re-run.
- Items for the P5 and parity agents: none required in their files. After this phase `paper/` imports from `lab/` only `lab.data` (`fetch_stock_ohlcv`, `STOCK_BASE_CONFIG`), `lab.allocation` (blend only), `lab.persist.get_engine` and `lab.strategies.v2.ensemble`, so P5 may delete `lab/simulate.py`, `lab/strategy.py`, `lab/strategies/ensemble.py` and `lab/strategies/indicators.py` without breaking `import ggTrader.paper.trader` (the boundary test bans the old imports). Other tests and `lab/replica.py` (which imports `paper/` on the allow-list) were not examined.
