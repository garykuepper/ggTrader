# P2b completion note: fixes for the independent P0-P3 review and the P4 chunker finding

**Date:** 2026-11-06. **Branch:** `vbt12/engine` (nothing committed, staged, tagged or pushed). **Inputs:** the review (4 MAJOR, 12 MINOR, 0 blockers, 21 mutants all killed), the lead's decisions 1-7, the P4 chunker finding. Terms: **E1** = the configured-to-legacy baseline epoch. **ulp** = unit in the last place (about 1e-16 relative for a double). **WFO** = walk-forward optimisation. **PIT** = point-in-time. **Tape** = the frozen daily-bars file `data/snapshots/ohlcv_20261106.parquet` (sha256 `6573e840...789b2e`).

## Result

| Exit criterion | Result |
|---|---|
| Whole suite via `scripts/run_tests.sh tests/` | **2,912 passed, 1 failed**, 30 integration deselected. The one failure is `tests/engine/test_boundary.py::test_only_the_engine_package_touches_vectorbt`, caused by another agent's untracked script `scripts/new_stack_reference_runs.py` (imports `vectorbt`, `vectorbt._engine` and `vectorbt_rust` directly; lines 50, 51, 70, 133, 137). I did not touch it. Fix it by routing the script through `ggTrader.engine` (or listing it in `TRANSITIONAL` and `ruff.toml` until P5). Without that file the whole suite is green (the engine suite alone: 431 tests, was 264). |
| `ruff check .` and `ruff format --check .` | clean (379 files) |
| Changelog | entry at the top of `docs/changelog.md`; golden reason recorded there |
| Tests written first | yes for every behaviour change (each new test was seen failing before the fix or, for the guard tests, killed a mutant) |

## Review findings: action and evidence

| Finding | Action | Test / evidence |
|---|---|---|
| **MAJOR-1** `window_stats` window convention | `window_stats(returns, a, b)` now equals `curve_stats(equity.iloc[a:b])`: returns from the second bar (`n_obs = b - a - 1`), CAGR dates from the window's first and last bar. **New finding while testing on real tape:** vectorbt's returns-based max drawdown starts its path after the first return and misses a fall from the base bar (AOS 2022-01-31..2023-01-31: engine -35.1%, legacy -35.8%; DISCA 0.0 against -3.6%). Fixed by giving the drawdown path a leading zero return. Random-series tests had not hit it because their peaks come late. `tests/engine/scenarios.py` is unchanged; its `mini_wfo` golden now scores the legacy window, so `engine_scenarios.json` was regenerated with a recorded reason (simulation curves unchanged). | `tests/engine/test_legacy_defs.py`: window_stats against `curve_stats` for every statistic on 4 seeds x 5 windows, the first-bar-peak examples, 60 real symbols x 4 windows (assert 1e-12), real engine equity. `p2b-evidence/window_convention_check.py` on 599 symbols x 4 windows = 2,367 comparisons: max abs difference Sharpe 1.9e-14, volatility 3.7e-13, max drawdown 2.9e-15, CAGR 2.7e-14, Sortino 0.0 (`window_check.json`). |
| **MAJOR-2** E1 vs Sortino and CAGR | New transitional `engine/legacy_defs.py` (header says TRANSITIONAL, deleted at the end of P6). `window_stats(..., definitions="legacy" \| "vbt")`, default `"legacy"` (constant `stats.DEFAULT_DEFINITIONS`). `provenance` stamps `definitions` per statistic (Sharpe, volatility, max drawdown, Sortino, CAGR, window convention, DSR) from `stats.definition_labels`. | `test_legacy_defs.py`: Sortino equals `curve_stats` exactly on 6 seeds x 5 windows and on real tape; CAGR to 1e-12; `"vbt"` equals vectorbt's accessors; the two sets differ on a documented 8-return example (Sortino 2.75 vs 3.54, CAGR 12.5% vs 82.2%); expiry guard `test_module_expires_with_epoch_e1`; only `stats.py` imports the module. `test_provenance.py`: per-statistic stamps, vbt vs legacy set differ in exactly Sortino and CAGR. |
| **MAJOR-3** workflow token | `permissions: {}` at workflow level; `contents: read` for `detect` and `try`; write only for `report`. `try` has no secrets, `enable-cache: false`, `persist-credentials: false`; `detect` also `persist-credentials: false`. `report` never installs or runs the release and refuses an artifact `pyproject.toml` that changes more than vectorbt lines, then copies exactly two files. All five actions pinned to commit SHAs. | `test_upgrade_tooling.py` (+6 tests: permissions per job, no secret in `try`, no install in `report`, credentials, full-SHA pins with a release comment, header lists the pins). The guard's shell logic was run on three sample diffs. **Pins VERIFIED 2026-11-06** with unauthenticated GETs of `api.github.com/repos/<repo>/commits/<tag>`: checkout v6.1.0 `d23441a4...`, setup-python v6.3.0 `ece7cb06...`, setup-uv v7.6.0 `37802adc...`, upload-artifact v5.0.0 `330a01c4...`, download-artifact v5.0.0 `634f93cb...`; release-notes tag form `v1.2.0` exists on `polakowo/vectorbt`. **Not run:** the workflow itself (no GitHub run from here). |
| **MAJOR-4** P1 exit criterion | A separate job runs the 17-fold parity (not mine). Lead acceptance recorded in the spec (Appendix E): the event-study means (9e-16) and the `idio_vol` tie order for 3 near-zero-variance delisted names are class (ii) outside the WFO path. | spec Appendix E |
| MINOR-1 `compare` | tape sha difference is `TAPE_DIFF` (not comparable; an unrecorded tape counts as different); any `definitions` difference is `CONTRACT_DIFF`; the worst flag wins by rank. | `test_provenance.py`: tape, definitions, a silent edit of one definition, and a full flag-versus-comparable table. |
| MINOR-2 vol keys | `run_signals` and `run_weights` raise `ValueError` for `vol_target`, `vol_lookback`, `vol_cap` (non-None); `run_weights` also for `ts_stop`, `atr_mult`, `tp_stop` (it has no stops). `SIGNAL_POSITION_SIZE` and the other config noise stay accepted (`STOCK_BASE_CONFIG` carries them). | `test_portfolio.py` (vol keys, stop keys on weights, `None` stays allowed). |
| MINOR-3 ATR after gaps | Documented in `atr_stop_fraction`; pinned: on a seeded gapped panel the stops differ from legacy in exactly six cells (bar 13 on four series, bar 117 after a gap, bar 163 after a late listing). | `test_atr_stop_differs_from_legacy_only_at_documented_structural_cells` |
| MINOR-4 terminal exits | metadata has `biases.exit_price` (optimistic) and `biases.same_bar_entry_suppression` (favourable lookahead, with the count), plus `lookahead`, `uses_knowledge_of_series_end`. | `tests/lab/test_universe.py` (+2). The OD7 P6 run must report the count. |
| MINOR-5 inclusive end | `data.bars.legacy_exclusive_end(date)` (previous weekday, naive UTC); `load_bars` default stays inclusive and documents it. | `tests/data/test_bars.py`: weekday/weekend/aware inputs, load window, and the real tape against P0's `oos_last` 2026-04-29. |
| MINOR-6 tz-aware span | `_utc()` converts (naive is read as UTC, aware is converted, not shifted). | 4 forms parametrised, including US/Pacific. |
| MINOR-7 DSR flag | spec 5.4 item 6, OD2 and Appendix A corrected to `bias=False`; code unchanged. | spec |
| MINOR-8 zero-trade scope | **Not fixed, by design:** a P5 decision (zero entries vs zero exposure; apply to every composite statistic). Documented in `stats.sharpe`. | docstring, spec Appendix E |
| MINOR-9 boundary test | AST test now catches `getattr(x, "vbt")` (also keyword and string-built names), `operator.attrgetter`, `__getattribute__`, `import_module` with a keyword or string-built name, aliased imports of `import_module`/`find_spec`/`__import__`, `builtins.__import__`, `sys.modules.get`, and `exec`/`compile` strings. Negative cases stay clean. | `test_boundary.py` (22 new parametrised cases, 5 of them negative, plus an engine/strategies allowance test) |
| MINOR-10 image hygiene | `Dockerfile` base pinned by the Docker Hub multi-arch index digest (resolved 2026-11-06); Dependabot ignores `python` major and minor. **Not built.** | `test_upgrade_tooling.py` (+2) |
| MINOR-11 changelog defect | bracketed supersession note appended (history kept) in six places, listed below. | grep |
| MINOR-12 one-bar fold overlap | no E1 change; recorded as a P6 definition question (spec 5.4 item 8). | spec |
| **P4 chunker finding** | each chunk's equity columns are copied (not viewed), the `Portfolio` and records are deleted, and `gc.collect()` plus `malloc_trim(0)` run per chunk. A group is never split (a property test over random group sizes). `WORKING_ARRAYS` 8 to 12. | table below; `test_each_chunk_is_released_before_the_next_one_runs` (weak references: no finished chunk alive when the next simulates), `test_six_chunks_equal_one_call_bit_for_bit`, `test_planned_chunks_partition_the_groups_and_never_split_one`. |

Files with the supersession note (all minimal, bracketed, history kept): `docs/changelog.md`, `docs/superpowers/plans/2026-10-20-momentum-retest.md`, `research/audits/production-replica/2026-10-19-production-replica-audit.md`, `.../2026-10-19-production-replica-audit-nogo.md`, `research/stock-signals/momentum/2026-10-20-momentum-retest.md`, `.../2026-10-20-momentum-french-screen-nogo.md`. No hit in `docs/next_steps.md`, `docs/roadmap.md`, `README.md`, `AGENTS.md`, `research/RESEARCH_INDEX.md`, `research/RESEARCH_SNAPSHOT.md` or `src/`. The S7 report in `research/programme/vectorbt-engine/reports/` quotes the phrase as a finding about it and was left alone.

## Chunker memory: full ensemble grid, 48 combinations x 605 symbols x 1,929 bars

Driver `p4-evidence/chunker_memory.py` (one fresh process per run, 6-9 GiB cgroup, swap off, frozen tape). Process baseline before the run: 1.39 GB (inputs and tiled copies). "Over baseline" is the figure the budget is meant to bound.

| Run | Budget | Chunks | Peak RSS | Over baseline |
|---|---:|---:|---:|---:|
| chunks kept (the P4 behaviour, same estimate) | 2 GiB | 3 | 5,241 MB | 3,850 MB (1.8x the budget) |
| **released (new)** | **2 GiB** | 3 | **3,381 MB** | **1,982 MB (budget 2,147 MB)** |
| chunks kept | 5 GiB | 2 | 5,783 MB | 4,391 MB |
| **released (new)** | **5 GiB** | 2 | **5,781 MB** | **4,391 MB (budget 5,369 MB)** |
| single call (no chunking) | none | 1 | 6,139 MB | 4,744 MB |

Equity and trades are hash-identical in all five runs (`p2b_memory_summary.json`). At 5 GiB the first chunk (44 of 48 groups) dominates, so releasing changes nothing there; at 2 GiB it removes 1.9 GB. The old estimate (8 working arrays) put the single call at 4.03 GB against 4.74 GB measured; 12 gives about 12% headroom, which is why the 5 GiB budget now splits into two chunks. For P5 under a 6 GB cap use a budget of about 3 GiB (expected peak about 4.5 GB: baseline 1.4 GB plus a chunk), not the 5 GiB default whose absolute peak is 5.8 GB.

## Second independent tests for the six single-test protections, mutation-checked

Each mutation was applied to a scratch copy of the worktree (no `.venv`, tape symlinked, other agent's script removed) and `tests/engine` was run (`p2b-evidence/mutate_engine.py`, results in `mutation_results.json`). **Survivors: 0 of 13.**

| Protection | Mutation | Killed by (tests that fail) |
|---|---|---|
| zero-trade guard, frame path | `> 0` to `>= 0` | `test_zero_trades_window_is_nan` (original) and `test_zero_trade_guard_on_the_scalar_path_and_with_a_missing_count` (new) |
| zero-trade guard, scalar path | `== 0` to `== -1` | `test_zero_trade_guard_on_the_scalar_path...` and `test_zero_trade_guard_accepts_a_plain_integer_count_for_a_series` (new; the original never reached this branch) |
| trade-count window end | `< end` to `<= end` | original `test_trade_counts_use_the_entry_index...`, plus new `test_trade_count_window_is_half_open_on_a_real_simulation` and `test_trade_counts_equal_a_brute_force_count_for_random_windows` |
| mask before time stop | time exit from the raw entries | original `test_pit_mask_is_applied_before_the_time_stop` and the golden, plus new `test_mask_then_time_stop_equals_prefiltered_entries_and_differs_from_the_other_order` |
| contract-change flag | `CONTRACT_DIFF` comparable | original `test_contract_diff_is_not_comparable` and 9 more including the new flag matrix |
| version-mismatch error | drop the filter in `configure()` | original `test_mismatch_warning_becomes_an_error`, plus new simulator-path and fresh-interpreter tests |
| selfcheck probe | report the configured engine | original `test_probe_fails_when_the_kernel_runs_on_another_engine`, plus new `test_probe_reports_the_engine_that_ran_not_the_one_configured` (both directions) |
| *new:* window includes the first return | `iloc[start+1:end]` to `iloc[start:end]` | 31 tests |
| *new:* drawdown without the base bar | | 2 tests (`test_max_drawdown_counts_the_first_bar_as_a_peak...`, the real-tape sweep) |
| *new:* legacy Sortino replaced by vbt | | 22 tests |
| *new:* tape ignored by `compare` | | 2 tests |
| *new:* vol keys ignored | | 3 tests |
| *new:* chunk kept alive | delete the release | 1 test (the weak-reference test; memory is otherwise only visible in the RSS table above) |

## Not fixed, and why

- **Bit-for-bit window statistics.** Not achievable with vectorbt's accessors: the engine and `curve_stats` sum in different orders, so Sharpe, volatility, max drawdown and CAGR agree to a few ulp (max 3.7e-13 absolute, volatility), not exactly. Only Sortino is bit-identical, because `legacy_defs` mirrors the pandas calls. Re-implementing the maths in numpy would breach the "no re-implemented vectorbt maths" anti-goal, so the spec's class (i) row now states the ulp floor (7.1) and the tests use 1e-12 relative.
- **MAJOR-4 (the 17-fold run on the new stack).** Another job; this note only records the lead's acceptance in the spec.
- **MINOR-8, MINOR-12.** Deferred to P5 and P6 as decisions, documented.
- **Docker image and workflow.** Neither built nor run here; only the text, the pins and the shell guard were checked.
- **The red boundary test** above belongs to `scripts/new_stack_reference_runs.py`.
- Real-tape tests skip when the git-ignored parquet is absent (as the existing P3 tests do).

## Files touched

`src/ggTrader/engine/{stats,legacy_defs (new),provenance,portfolio}.py`, `src/ggTrader/data/bars.py`, `src/ggTrader/lab/universe.py`; tests `tests/engine/{test_legacy_defs (new),test_stats,test_provenance,test_portfolio,test_boundary,test_config,test_selfcheck,test_upgrade_tooling}.py`, `tests/data/test_bars.py`, `tests/lab/test_universe.py`, `tests/engine/golden/engine_scenarios.json`; `.github/workflows/vbt-upgrade-watch.yml`, `.github/dependabot.yml`, `Dockerfile`; docs `docs/changelog.md`, `docs/architecture.md`, the spec; evidence `research/programme/vectorbt-engine/p2b-evidence/` and `p4-evidence/{chunker_memory.py,p2b_memory_*.json}`.
