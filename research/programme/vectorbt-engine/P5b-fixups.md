# P5b completion note: fixes for the independent review of the walk-forward rewrite (P5)

**Date:** 2026-11-06. **Branch:** `vbt12/engine` (nothing committed, staged, tagged or pushed). **Inputs:** the independent review (0 BLOCKER, 3 MAJOR, 7 MINOR, 4 INFO) and the lead's decisions 1-6. Terms: **WFO** = walk-forward optimisation. **OOS** = out-of-sample. **NDH** = plateau-density gate. **DSR** = deflated Sharpe ratio gate. **Anchor** = the defensive fallback parameter set, refit on data up to each fold's `train_end`. **Selection window** = a window whose statistics feed a choice (anchor, train, live); the **test window** is the measurement of the deployed book. **E1** = first baseline epoch. **Tape** = `data/snapshots/ohlcv_20261106.parquet`.

## Result

| Exit criterion | Result |
|---|---|
| Whole suite via `scripts/run_tests.sh tests/` | **3,068 passed, 0 failed**, 30 integration deselected, 13 min 21 s (P5/P7 baseline 3,036; +32 net: 23 new wfo_v2 tests, 8 engine stats tests and the toy-oracle second grid) |
| `ruff check .` / `ruff format --check .` | clean (412 files) |
| Mutation check (scratch copy, 11 mutants) | **11 of 11 killed**, including the two the review found surviving (M2 full-sample anchor, M7 DSR 0.70) |
| Changelog, spec Appendix E, P5-completion update | done (`docs/changelog.md` top entry; spec 5.6 note and Appendix E "P5b outcome"; P5-completion top note and section 10) |
| `src/ggTrader/paper/` | untouched |

## Findings: action and evidence

| Finding | Action | Test / evidence |
|---|---|---|
| **MAJOR-1** degenerate Sortino wins selection; spec 5.6 zero-trade half not wired | Lead decision implemented. `engine.stats.window_stats(..., n_trades, min_losing_days)`: a selection window (anchor, train, live) blanks **Sharpe and Sortino** of a combination with no entry inside the window, and the **Sortino** of one with fewer than `engine.config.MIN_LOSING_DAYS` = 20 losing days (strictly negative returns). Vectorbt's formula is unchanged above the threshold. NaN is floored by the composite's existing rule (`_floor_nan` to the grid's worst finite value; no new statistic). Overridable: `run_wfo(min_losing_days=...)`, `window_stats(..., min_losing_days=...)`, driver flag `--min-losing-days` (0 = off). Stamped: `diagnostics["guards"]` in every WFO result and `sortino_guard` / `zero_trade_guard` in the provenance `definitions` (so a guarded result is a different, stamped measurement; `compare` flags it `CONTRACT_DIFF` against an unguarded block). The test window keeps only the flat-curve guard, as does `curve_summary` (both measure a book that exists; a carried-in position earns real returns without an entry). Calmar is not computed anywhere in the engine, so nothing else is affected. | `tests/engine/test_stats.py` (+8: default 20, threshold inclusive, zero returns are not losing days, unchanged above threshold, both guards in `window_stats`, series scalar count); `tests/lab/wfo_v2/test_p5b_review_fixes.py` (scorer-level: no-entry and few-losing-day rows blank in train/anchor/live, not in test; overridable; guard changes the winner; stamp). Toy oracle (`test_toy_known_answers.py`) now applies the guard from its own entry count and has a second grid whose entries fall inside the train windows (the carried-in toy ranks every combination NaN). Before/after below. |
| **MAJOR-2** full-sample (lookahead) anchor passed all tests | New tests (a) `test_perturbing_bars_after_an_interior_train_end_leaves_that_anchor_unchanged[0,1]`: bars after fold k's `train_end` are shaken with a seed chosen (by the closed-form oracle) so that an anchor fitted to the last evaluation day would be another combination; the recorded anchor of folds <= k, their winners and train scores must equal the unperturbed run and the oracle's causal anchor. (b) `test_every_fold_records_its_anchor...` compares every fold's anchor with an independent closed-form anchor. (c) `test_the_anchor_equals_a_fresh_simulation_on_data_truncated_at_train_end` recomputes each ensemble anchor from a fresh simulation on bars truncated at `train_end`. (d) `test_bars_after_an_interior_train_end_cannot_change_that_folds_selection_or_anchor` (ensemble, whole `run_wfo`). The golden is not the guard. DSR default: `test_the_default_dsr_threshold_is_point_eight` (signature) and `test_a_default_run_hands_the_default_threshold_to_the_gate` (what the run passes to the gate). | Mutation check, scratch copy of the worktree, new test file only: **M2 killed** by (a) (both parametrisations); **M7 killed** by both DSR tests. Note: on the unperturbed toy, the anchor at the last evaluation day equals the causal one, so (b) and (c) alone do not kill M2; (a) does. |
| **MAJOR-3** per-fold anchor not recorded | Each `fold_results` row: `anchor_combo`, `anchor_params`, `anchor_reason` (`"gates_failed"`, `"halted"` or `None` when the winner was deployed; `"halted"` wins when both apply, as in the deploy rule), `deployed_combo` (`"none"` if no anchor exists), `halt_cleared` (the shadow re-entry ended the halt after this fold). `result_payload` adds `anchor_history`. The printed table gets a separate section "Anchor Per Fold" **after** the legacy sections, showing the anchor parameters that vary across folds and the reason; `format_wfo_table` without the new argument is byte-for-byte the legacy text (existing equality test unchanged). Callers that parse the table (`ablation_voters.parse_table` and the midcap scripts) match fold rows with `^\d+\s` and the aggregate lines by label: the new lines start with spaces and the word "fold", so they are not seen. | `test_every_fold_records_its_anchor_and_why_it_was_used`, `test_the_anchor_history_is_in_the_json_payload_and_the_report`, `test_the_default_table_text_is_still_the_legacy_text`. Guarded 17-fold JSON: `p5-evidence/guarded/_p5_core17_ensemble_sp500.json` (`fold_table[*].anchor_*`, `anchor_history`). The anchor was one of 5 parameter sets across the 17 folds. |
| MINOR-1 ties decided by 1e-13 noise | Composite rounded to 1e-9 (`select.TIE_DECIMALS`) before the arg-max in `best_index` and `pick_live_winner`; the first combination in grid order wins a tie. | `test_a_tie_within_float_noise_goes_to_the_first_combination`; mutant G6 killed. Visible in the 17-fold run: fold 8 flips between two combinations with equal Sharpe to 1e-13 in the zero-entry-only run (no metric moves). |
| MINOR-2 combination order unchecked | `check_combination_order`: a strategy whose built combinations are not in `ParamGrid.combos()` order raises `ValueError` (signal build, and weight blocks). | `test_a_strategy_that_orders_combinations_differently_is_refused`, `..._in_the_grid_order_is_accepted`; mutant G9 killed. |
| MINOR-3 anchor can be a dead combination | `pick_anchor` drops combinations whose Sharpe is NaN (never traded, or flat); none left means `NO_ANCHOR`. **This differs from legacy by design** (owner's flat-curve intent). With no anchor the book stays in cash: a flat curve at the running equity over the test window (new `Scorer.window_dates`), so the OOS record and the SPY comparison keep their days. | `test_a_combination_that_never_traded_cannot_be_the_anchor`, `test_cagr_filter_still_applies...`, `test_all_dead_means_no_anchor`, `test_a_run_with_no_anchor_sits_in_cash_over_the_test_windows`; mutant G5 killed. Evidence: `ref_retail_attention` unchanged. |
| MINOR-4 DSR default untested | See MAJOR-2. | |
| MINOR-5 golden is self-referential | Not fixable by design (a golden is a change detector). Test docstring now says so and that the independent tests carry the proof. Regenerated for the guards with a recorded reason (also in the JSON and the changelog); the old reason is kept inside the new one. | `tests/lab/wfo_v2/golden/mini_wfo.json`. What changed (scratch comparison, old golden in the scratchpad): winners and live combination in all three scenarios, `dsr_value` and `is_sharpe` values, and in the open-gates scenario fold 2's OOS Sharpe (NaN, now 0.22: consistent with a never-traded anchor no longer being deployed, not separately verified) and gates; OOS equity unchanged in the other two. |
| MINOR-6 stale `--max-leverage` text | `AGENTS.md` line 197 (`CLAUDE.md` is a symlink to it) now says the default is 1.0 (2.0 until the engine rebuild, OD13): that one sentence only. `combine_sleeves` default aligned to 1.0 (every caller passes it explicitly; its tests pass 2.0 explicitly or are indifferent); `target_vol_scale` keeps 2.0 because a test pins it and it is a primitive no CLI path reaches without the explicit value. Changelog mentions both. | `tests/lab/test_allocation.py` green. |
| MINOR-7 symbol count | `P5-completion.md` says 599 loaded (605 is the tape's count); the memory test keeps 605 x 1,929 (conservative). | |
| INFO-1 first `--blend` alters `lab_runs` schema | Not changed. The first `--blend` run runs `ALTER TABLE ... ADD COLUMN IF NOT EXISTS provenance` (additive, idempotent). Owner should know before the first blend (changelog says so). | |
| INFO-2 OOS stitching carries no switching cost | Not changed (same semantics as legacy). Read OOS figures as "not tradable as stitched". | |
| INFO-3 fold dates drift to the 30th | Not changed (legacy-identical). Flag for P6. | |
| INFO-4 WFE uses the winner's IS Sharpe against the deployed OOS Sharpe | Not changed (legacy-identical). In the 17-fold core this trips the halt after fold 5, and it never clears. | |

## Before / after, 17-fold SP500 ensemble (frozen tape, eval 2021-01-31 to 2026-04-30, window ends on the 2026-04-29 bar, 48 combinations, vectorbt definitions, `STOCK_BASE_CONFIG`)

Script `scripts/p5b_guard_compare.py`; raw: `p5-evidence/_p5_core17_*.json` (unguarded, as P5), `p5-evidence/guarded_zero_entry_only/` (`--min-losing-days 0`), `p5-evidence/guarded/` (default guards), `p5-evidence/guarded/_p5b_before_after.json`, `_p5b_all_runs_before_after.json`.

| Run | Variant | Gates | Anchor folds | Winners changed vs unguarded | OOS Sharpe / CAGR / MaxDD |
|---|---|---:|---:|---|---|
| core | unguarded (P5) | 1/17 (fold 12) | 17 | - | 0.391 / 3.50% / -9.58% |
| core | zero-entry rule only | 1/17 | 17 | fold 8 (a 1e-13 tie) | 0.391 / 3.50% / -9.58% |
| core | **guarded = E1 candidate** | 1/17 (fold 12) | 17 | folds 8, 9, 16, 17 | **0.391 / 3.50% / -9.58%** |
| DSR threshold 0 | unguarded (P5) | 16/17 | 1 | - | 0.886 / 4.91% / -5.28% |
| DSR threshold 0 | zero-entry rule only | 16/17 | 1 | fold 8 | 0.886 / 4.91% / -5.28% |
| DSR threshold 0 | **guarded** | 17/17 | 0 | folds 8, 9, 16, 17 | **0.637 / 3.41% / -5.28%** |
| SPY, same days | | | | | 0.779 / 13.02% / -22.09% |

The guarded dsr0 numbers match the reviewer's independent prediction (0.637, 3.41%, 17/17). The primary result is anchor-driven, so it cannot move: the winner never reaches the money after the breaker trips in fold 5. The dsr0 0.886 was the artefact (fold 8's one-entry, four-losing-day winner failed NDH and routed a lucky anchor, +7.6% in Q4 2023, into the book). Both variants are below SPY on Sharpe, CAGR and drawdown-adjusted return; this is a research result, not a retirement recommendation. The other eleven validation runs (smoke and nine strategies) are unchanged in every winner, gate, anchor use and OOS figure.

## Not fixed, and why

* Carried-in-only windows. A selection window in which a combination holds only positions entered before the window has real returns but no entry; spec 5.6 as written blanks its Sharpe and Sortino. The ensemble never hits this in practice (hundreds of entries per year across the universe), but a buy-and-hold style strategy would rank NaN in every window after the first. Recorded; the rule is the lead's.
* The Sortino threshold 20 is the reviewer's measurement on one run (it changes exactly folds 8, 9, 16, 17); other strategies, universes and window lengths may want another value. It is a parameter and is stamped.
* `target_vol_scale` default 2.0 (see MINOR-6).
* Scripts that still import `lab.wfo` (legacy) were not switched.

## P7 uncertainties (listed, not touched; no `paper/` file changed)

From `P7-completion.md`, still open and needing the owner, Docker, a database or the crontab:

1. The read-only dry run was never run, so the live loader's real output (timezone, duplicate days when cached and fresh rows meet on the partial bar) is unseen; `bars_from_legacy_frame` keeps duplicates so the gate fails closed, which could stop a healthy day.
2. Gate bounds (450 symbols, 95% last-bar coverage, 100 buys and 250 sells, 12:55 PT late-run cut-off) are estimates from the 120-session golden margins; tune from the dry-run record.
3. Thin final bar dated today is trimmed (not fatal): an owner choice.
4. OD17 (b): convert or remove AVB at the broker, or keep `UNTRADABLE_SYMBOLS=AVB`.
5. `GGTRADER_TAG` must be set to a sha tag at the first deploy; the Dockerfile, compose and workflow changes are text only and were never built.
6. Not built: the 13:10 PT missing-run heartbeat, the scratch-Postgres round trip of `_cache_to_db` and `refresh_benchmark_tape` under pandas 3, a `docker-compose run` timeout.
7. `paper/live_signals.py` imports `Ensemble` from `lab.strategies.v2.ensemble`; whoever moves the v2 modules must update that one import and the boundary test's path.
8. `ggtrader-deploy/evals/evals.json` was not re-run after the dry-run semantics change.
9. Not covered by a test: that the read-only dry run touches no table at all in a real process (persistence functions are patched; reads still open ordinary connections).
10. Cron stays off until the owner says so (OD18).

## Files

Changed: `src/ggTrader/engine/{stats,config}.py`, `src/ggTrader/lab/wfo_v2/{run,scorers,select,anchor,report}.py`, `src/ggTrader/lab/allocation.py`, `scripts/p5_wfo_validation.py` (`--min-losing-days`), `AGENTS.md`, `docs/changelog.md`, the spec (5.6 note, Appendix E), `P5-completion.md`, `tests/engine/test_stats.py`, `tests/lab/wfo_v2/{test_toy_known_answers,test_leak_golden_memory}.py`, the golden. Created: `scripts/p5b_guard_compare.py`, `tests/lab/wfo_v2/test_p5b_review_fixes.py`, `p5-evidence/guarded/`, `p5-evidence/guarded_zero_entry_only/`, this note. Mutation harness (scratch only): the scratchpad `mut5b/mutate5b.py`.
