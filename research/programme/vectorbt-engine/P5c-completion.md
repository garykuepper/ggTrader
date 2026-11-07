# P5c completion note: the legacy engine is deleted, the vectorbt engine is the only path

**Date:** 2026-11-06. **Phase:** P5c of the [vectorbt engine rebuild](../../../docs/superpowers/specs/2026-11-06-vectorbt-engine-design.md). **Branch:** `vbt12/engine` (nothing committed, staged, tagged or pushed; files were removed from the working tree with plain `rm`).
Terms: **WFO** = walk-forward optimisation. **OOS** = out-of-sample. **Tape** = the frozen daily-bars file `data/snapshots/ohlcv_20261106.parquet`. **Oracle** = an independent reference implementation or frozen output a test compares with. **Tag** = git tag `legacy-engine-vbt0285` (pre-migration `main`, vectorbt 0.28.5).

Ledger: [P5c-ledger.csv](P5c-ledger.csv) (path, action, reason, evidence, lines; built **before** any deletion by `p5c-evidence/build_ledger.py` from the AST import graph `p5c-evidence/import_graph.py`). Closure check: `p5c-evidence/closure_before.txt` (every kept file that imported a deleted file was ported first) and `p5c-evidence/dangling_imports_after.txt` (after deletion the only dangling `ggTrader.*` imports are in `scripts/legacy_reference_runs.py`, the documented legacy-interpreter exception). A safety copy of the whole pre-P5c working tree is `pre_p5c_tree.tgz` in the session scratchpad (the tag does not contain the P1 to P7 work, so git could not restore it; the branch is uncommitted).

## Ledger summary

| Action | Source | Tests | Scripts | Lines |
|---|---:|---:|---:|---:|
| deleted | 34 files | 48 files | 2 files | 7,261 + 10,209 + 351 |
| archived to `legacy-scripts/` | | | 32 files | 7,343 |
| ported (kept, edited) | 9 | 19 | 4 | 1,593 + 3,335 + 931 |
| added (oracles, goldens, tests) | | 9 | | |
| kept (referenced, unchanged) | | 1 | 50 | |

25,164 lines of Python left the main tree (the archive keeps the 7,343 script lines as research evidence).

**Size before and after** (`*.py`, `wc -l`; "before" is the working tree at the start of P5c, which already held P1 to P7; "tag" is `legacy-engine-vbt0285`):

| Tree | Tag | Before P5c | After P5c | Change in P5c |
|---|---:|---:|---:|---:|
| `src/ggTrader` | 28,953 | 35,873 | 28,340 | -7,533 |
| `src/ggTrader/lab` | 23,451 | 27,445 | 19,912 | -7,533 |
| `scripts/` | 15,620 | 20,118 | 12,397 | -7,721 |
| `tests/` | 33,542 | 42,901 | 33,967 | -8,934 |

The engine path is now smaller than the legacy one it replaced, but the honest net result against the tag is small (`src` 28,953 to 28,340): the new `engine/`, `wfo_v2/`, `strategies/v2/`, `data/bars.py` and the P7 `paper/` code roughly equal what was deleted. The gain is one simulation path, one registry and one set of definitions, not fewer lines.

## What was deleted and why

* `lab/simulate.py`, `sweep.py`, `harness.py`, `wfo.py`, `metrics.py`, `strategies/indicators.py`, `strategies/signals.py`, `strategies/registry.py`, `ic_weights.py`, `kelly.py`, and the 17 legacy strategy classes plus `ensemble*.py`, `conviction.py`, `momentum.py`, `cross_asset_sleeve.py`, `retail_attention.py`. `STRATEGY_REGISTRY` is gone; `lab/strategies/v2/registry.py` is the only registry and now carries `LEGACY_ONLY_STRATEGIES` (the 28 names) so `ggt lab` names the tag.
* `lab/strategy.py` keeps `LabConfig` only (`Plan`, `SignalTargets`, `Strategy`, `max_sector_count` went).
* The O(T^2) `pit_entry_mask` went with `sweep.py`.
* Tests of the deleted modules (48 files) and two already-broken scripts (`update_universe_ccxt.py`, `update_venue_listings.py`).

## What was kept on purpose, and why (not deleted although the brief allowed it)

* `lab/data.py` is unchanged: `load_ohlcv`, `collapse_daily_duplicates`, `eligible_at` and `negative_cache.py` are still called by two kept tools, `scripts/production_replica_audit.py` and `scripts/replica_parity_check.py` (grep evidence in the ledger). `eligible_at` is also the independent oracle of `tests/lab/test_universe.py`. `fetch_stock_ohlcv` and `STOCK_BASE_CONFIG` stay for the live trader.
* Data loaders and backfill scripts for the unported strategies (`earnings_surprise_data`, `form4_data`, `house_ptr_data`, `headline_sentiment_data`, `short_interest_data`, `short_volume_data`, `fred_data`, `fomc_calendar`, `google_trends_data` and their `*_backfill.py` scripts) stay: they collect data into tables, they are not strategy code, and a registered re-test would need them.
* `lab/regime.py`, `lab/event_*`, screens, `tax_lots`, `a23_rebalance`, `century_leverage`, intraday tooling: untouched.
* `scripts/engine_parity_evidence.py` stays because the vectorbt upgrade runbook runs it (it is the one check that saw the 1.1.0 Bollinger error on real data). Its indicators and voters sections now use pure-pandas oracles in `tests/oracles/legacy_voters.py` and its output is **identical** to the committed `p2-evidence/p2_engine_parity_20261106.json` (both sections, compared on the tape with 150 symbols). The simulator section was dropped; the original is archived as `engine_parity_evidence_p2_full.py`.
* `scripts/legacy_reference_runs.py` stays as the single boundary exception.

## Ported kept modules and their equality evidence

| Module | Change | Evidence |
|---|---|---|
| `lab/replica.py` `build_signal_panel` | signals from the live engine ensemble (`paper.live_signals.live_ensemble_frames`) plus the monthly PIT entry mask | legacy panel captured before deletion (119 SP500 symbols, tape, 2022-01 to 2026-04, window 2024-01-02 to 2026-04-29): **entries identical in every cell, replica equity bit-identical (max abs diff 0.0)**; 121 exit cells differ, all between 2022-01-04 and 2022-03-11 (indicator burn-in before the window). `p5c-evidence/replica_port_check.json` |
| `lab/train_gate.py` | entry labels from the engine ensemble | `_build_dataset` tests unchanged; the ML gate stays falsified and off |
| `lab/blend.py` | engine path only; table and persisted summary use `wfo_v2.report.curve_summary` | 6 orchestration tests rewritten for the engine path; real-tape blend test passes |
| `lab/significance.py` | CAGR difference uses `curve_summary` | tests; **definition change**: bar-annualised CAGR (vectorbt wins, standing rule) |
| `lab/wfo_policy_synthetic.py` and its script | breaker functions from `wfo_v2.breaker` | 8 tests unchanged |
| `scripts/shadow_cross_asset.py` | static 80/20 arm via `CrossAssetSleeve` and `engine.run_weights` | the hand-computed-rebalance test passes unchanged (rel 1e-9) |
| `scripts/production_replica_audit.py` | `curve_summary` | its 11 tests |
| `lab/cli.py` | see below | `tests/lab/test_cli.py`, real-tape CLI tests |

## CLI

`ggt lab` now needs `--wfo` or `--blend`. **Removed:** `--engine {legacy,vbt}`, `--sweep`, `--max-sector-count` (the engine path never honoured it, `lab/universe.py` documents the omission) and the plain no-mode run (one simulation persisted to `lab_runs`). `--strategy` accepts the ten ported names; the 28 legacy-only names exit with the tag and `scripts/legacy_env.sh`. `--sweep-param`, `--snapshot`, `--result-json`, `--max-leverage` (default 1.0) are kept. Documented in `docs/cli_reference.md`, `docs/changelog.md`, `AGENTS.md`, `README.md`.

## Coverage kept and coverage lost in the test suite

The legacy comparison tests were turned into frozen-oracle tests *before* the legacy code was removed (capture scripts in `p5c-evidence/`, each a record of how the golden was made; they now run only at the tag):

* `tests/engine/golden/legacy_simulate.npz`: equity and trade counts of `simulate_signals`/`simulate_weights` for plain, trailing, take-profit, both, fees, conviction sizes, ATR stop, weights and the empty group; `test_portfolio.py` compares bit for bit, as before.
* `tests/lab/strategies_v2/golden/legacy_{grids.json,signals.npz,weights.npz}`: grids and constructor defaults of the ten ported strategies; per-combination entries and exits of the ensemble (48), the five voters and `conviction_bb` (with sizes); weight targets and equity of `xs_momentum`, `cross_asset_sleeve`, `retail_attention`. The "equal after burn-in" checks are unchanged.
* `tests/lab/wfo_v2/golden/legacy_logic_units.json`: folds, composite scores, live-winner pick, 300 breaker histories x 2 policies, 200 shadow re-entry cases, grid arrays, the fold table text.
* `tests/paper/golden/legacy_overlay_curve.npz`: the overlay sleeve curve (value for value, rtol 1e-12).
* `tests/oracles/`: the legacy `curve_stats` and the five voter signal functions as pure-pandas oracles.
* `tests/lab/wfo_v2/test_policy_units.py`: 34 hand-written known-answer tests (WFE, circuit breaker, shadow re-entry, composite score, live-winner pick, folds, table) carried over from the deleted `test_wfo.py`.

**Lost (and recoverable at the tag):** every unit test of the 28 unported strategies and of the deleted hand-written indicators; `test_wfo.py`'s legacy sweep, gate and anchor integration tests, the monkeypatched `WfoPolicy` run tests and the legacy anchor-simulation tests (the engine equivalents in `tests/lab/wfo_v2/` cover the same behaviours on a closed-form oracle); the 864-curve equity parity proof against `lab.wfo._sweep_fold` (`p5_equity_parity.py`, recorded in `p5-evidence/`); P4 strategy-level parity on the real tape (`p4_strategy_parity.py`, recorded in `p4-evidence/`). Regression protection for those claims now rests on the frozen goldens above, the P7 live golden, and `scripts/p5_wfo_validation.py` against the P0 references.

## Test suite and lint

`scripts/run_tests.sh tests/` (8 GB cap, no swap): **2,602 passed, 0 failed, 19 deselected** (integration marks) in 11 min 30 s. P5b ended at 3,068 passed, 30 deselected; the 466 fewer tests are the deleted legacy tests net of 34 policy-unit, 3 oracle and about 30 changed or added tests. A final re-run of the engine boundary, oracle, strategy, wfo_v2, CLI and script tests after the last import-order fixes: 603 passed. `ruff check .` and `ruff format --check .` are clean (306 files).

## Boundary

The vectorbt import allow-list went from **eight entries to one**: only `scripts/legacy_reference_runs.py` (runs under the legacy 0.28.5 interpreter against the tree extracted from the tag). The six scripts and two modules that used to be listed were archived, deleted or ported; `tests/scripts/test_check_vbt_engine.py` moved to `tests/engine/test_check_vbt_engine_script.py`. `ruff.toml` and `tests/engine/test_boundary.py` (`LEGACY_ONLY`) agree, and the test fails if the entry goes stale.

## End-to-end verification (no database, frozen tape)

| Check | Result |
|---|---|
| 17-fold core through the public path `ggt.py lab --strategy ensemble --wfo --eval-start 2021-01-31 --eval-end 2026-04-30` | OOS Sharpe 0.39 (0.391), CAGR 3.5%, MaxDD -9.6%, gates 1/17, anchor 17/17; SPY 0.78 / 13.0% / -22.1%. Fold table, winners, gates and the printed table text **equal** the committed guarded evidence (`p5c-evidence/cli_reproduction_check.txt`). |
| Guarded DSR-0 run (`scripts/p5_wfo_validation.py`) | Sharpe 0.64 (0.637), CAGR 3.4%, MaxDD -5.3%, gates 17/17, anchor 0/17. |
| All 13 validation runs (core17, DSR-0, smoke2 and the ten per-strategy references) re-run into a scratch folder | **0 differing fields** against `p5-evidence/guarded/` (timings and provenance aside): `p5c-evidence/guarded_reproduction_check.txt`. |
| 2-fold smoke through the CLI | Sharpe 0.62, CAGR 7.5%, MaxDD -7.0%, gates 0/2, anchor 2/2; SPY -0.52 / -15.1% / -20.6%. Identical to the guarded smoke. |
| P7 live golden (`scripts/p7_live_golden.py`, 120 sessions) | 120 of 120 equal. |
| `ggt.py lab --help`, `paper --help`, `db --help`; `--help` of the changed and evidence scripts | all exit 0 |
| P4 successor | `p4_strategy_parity.py` needs the legacy classes, so it is archived with `p4-evidence/`; its claims are carried by the frozen goldens above. |

Neither guarded variant beats SPY on Sharpe, CAGR or drawdown-adjusted return; that is a research result, unchanged, not a retirement recommendation.

## Incident to disclose

While checking `--help` across `scripts/` I ran every script with `--help`, including 14 that have no argument parser and ignore the flag. `scripts/fomc_calendar_backfill.py` therefore ran its backfill and appended three dates (2026-07-29, 2026-09-16, 2026-10-28) to `data/universe/fomc_meeting_dates.csv`. I removed exactly those three lines; `git diff` on the file is empty again (content equals `HEAD`; the modification time changed). `fetch_midcap400_snapshot.py` failed before writing. The database statistics (`pg_stat_user_tables`) show 0 inserts, updates and deletes on the backfill target tables (`earnings_surprise`, `form4_transactions`, `house_ptr_transactions`, `headline_sentiment_scores`, `google_trends_interest`, `ohlcv`), and no other file under `data/` or `research/` changed. I stopped the loop after `net_issuance_screen.py`. The scripts after it alphabetically (`short_interest_backfill.py`, `short_volume_backfill.py` and others) were never run. After this I checked `--help` only on scripts that have an argument parser.

## Owner decisions this triggered

1. **Retire the last legacy loader?** The two replica tools still use `lab.data.load_ohlcv` (DB-first with yfinance and Tiingo gap fetch, price top-ups written to `ohlcv`, negative cache). Moving them to `data.bars` (read-only) removes `load_ohlcv`, `collapse_daily_duplicates` and `negative_cache.py`, but they would stop fetching missing symbols from the network. Default: leave as is.
2. **Data loaders and backfill scripts for the 28 unported strategies** are kept (they feed tables, not strategies). Archive them as well? Default: keep.
3. **Statistic definitions moved with the port.** `blend` tables and `significance.cagr_diff_pp` (the momentum, factor and net-issuance screens) now use vectorbt's bar-annualised CAGR and Sortino. Past reports used calendar-year CAGR; differences are about 0.1 point. Re-quote nothing without re-running.
4. **Naming.** `lab/strategies/v2/` and `lab/wfo_v2/` are now the only implementations; the `v2` names are historical. Renaming them (about 40 files of imports plus docs) is mechanical and was not done.
5. **`persist.py`** still creates `lab_sweeps`/`lab_sweep_combos` and has sweep writers that nothing calls. Drop them, or keep the schema for history (default: keep).
6. **`--max-sector-count` is gone**, so there is no sector cap on the engine path (`xs_momentum` in particular). Wanted back?
7. **Upgrade runbook.** P8's note R4 names `scripts/p5_equity_parity.py` (archived). The drill's equity check should be `p5_wfo_validation.py` against the frozen JSON plus the engine goldens; the `vbt-upgrade` skill already uses `engine_parity_evidence.py`, which still works.
8. **14 scripts violate the "every script handles `--help`" rule** (`earnings_surprise_backfill`, `event_timing_audit`, `fetch_midcap400_snapshot`, `fomc_calendar_backfill`, `form4_backfill`, `google_trends_backfill`, `headline_sentiment_backfill`, `house_ptr_backfill`, `insider_acceptance_diagnostic`, `intraday_observe`, `intraday_observer_job`, `mcp_server`, `short_interest_backfill`, `short_volume_backfill`). The last incident shows the cost; adding `argparse` is a small separate change.
9. **Commit the branch soon.** Everything from P1 to P5c is uncommitted; the tag restores only the pre-migration code.

## Could not delete, and why

`lab/data.py` pieces and `negative_cache.py` (item 1 above), `lab/metrics.curve_stats` (deleted; a verbatim copy lives in `tests/oracles/`), `scripts/legacy_reference_runs.py` (documented exception), `tests/paper/test_paper_boundary.py` (only names the deleted modules as banned strings).
