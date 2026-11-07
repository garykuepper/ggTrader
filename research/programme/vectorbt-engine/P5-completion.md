# P5 completion note: the walk-forward on the vectorbt engine

**Date:** 2026-11-06. **Phase:** P5 of the [vectorbt engine rebuild](../../../docs/superpowers/specs/2026-11-06-vectorbt-engine-design.md) (section 9). **Branch:** `vbt12/engine` (nothing committed, staged, tagged or pushed). Legacy `lab/wfo.py`, `sweep.py`, `simulate.py`, `gates.py`, `metrics.py`, `harness.py` are unchanged and still the CLI default.

> **Update 2026-11-06 (P5b), read this first.** An independent review found that the winner of folds 8, 9, 16 and 17 was chosen through a degenerate Sortino (1 to 14 losing days), and that spec 5.6's zero-trade rule had not been wired in. With the guards of [P5b-fixups.md](P5b-fixups.md) the **guarded** 17-fold core result is the E1 candidate: **OOS Sharpe 0.391, CAGR 3.50%, MaxDD -9.58%, gates 1/17, anchor 17/17 (unchanged), SPY 0.779 / 13.02% / -22.09%**. The DSR-threshold-0 diagnostic below (0.886) was the artefact: guarded it is **0.637 / 3.41% / -5.28%, gates 17/17, anchor 0/17**. The unguarded numbers in sections 3b and 4 are kept as the record of what P5 first produced; the guarded evidence is in `p5-evidence/guarded/` (`_p5b_before_after.json`). The "605 symbols loaded" in section 2 is 599 (605 is the tape's symbol count).

Terms: **WFO** = walk-forward optimisation. **OOS** = out-of-sample. **PIT** = point-in-time membership. **NDH** = the plateau-density gate. **DSR** = deflated Sharpe ratio gate. **Anchor** = the defensive fallback parameter set. **Tape** = the frozen daily-bars file `data/snapshots/ohlcv_20261106.parquet` (sha256 `6573e840...789b2e`). **ulp floor** = agreement to the last digits of a double (1e-12 relative is used as the pass bar). **OD12** = trades counted per scoring window. **E1** = the first baseline epoch.

## 0. Owner decision received during the phase (2026-11-06) and what it changed

The coordinator relayed: **vectorbt's definitions are the baseline; where a ggTrader definition conflicts with vectorbt's, vectorbt wins and ours is deleted.** I did the following, in this order:

1. `definitions="vbt"` became the default for the new WFO (`run_wfo`) and for `engine.stats.window_stats`. The 252-day year and the flat-curve finite guard stay (a calendar parameter and a safety guard).
2. The correctness proof became **equity-level parity**, which does not depend on any statistic definition (section 2). Winners and gates are no longer required to match the legacy references; differences are reported and attributed (section 4).
3. The legacy-definition path was run **once as a diagnostic** (all runs, evidence files carry the suffix `_legacydefs`), then **deleted**: `src/ggTrader/engine/legacy_defs.py`, the `definitions="legacy"` branch and `definitions=` argument of `stats.window_stats`, `tests/engine/test_legacy_defs.py` (with its expiry-guard test), the `stat_definitions` argument of `provenance`, and the `definitions` argument everywhere in `wfo_v2` and the validation driver. `provenance.definitions()` stamps vectorbt's definitions only; the cross-definition comparability test was replaced by a "silent definition edit is not comparable" test. `lab/metrics.curve_stats` is untouched and is the oracle for the Sharpe/volatility/drawdown conventions only.
4. The vectorbt-definition 17-fold run is the main result and the **E1 baseline candidate** (section 4).

Because the legacy path is gone, the diagnostic cannot be re-run from this tree; the evidence JSON (`*_legacydefs*`, `_p5_diff_*`) is the record. The driver scripts were stripped of the switch after the diagnostics were recorded.

## 1. What was built

New package `src/ggTrader/lab/wfo_v2/` (naming follows `lab/strategies/v2/`):

| Module | Content |
|---|---|
| `folds.py` | `generate_folds` (legacy recurrence unchanged), `fold_windows` (calendar windows to bar positions via `engine.windows.slice_positions`; anchor window is `(0, train_end)`) |
| `select.py` | `composite_score` (unchanged maths), first-max pick, `pick_live_winner` |
| `gates.py` | NDH lattice glue around `lab.gates.ndh_check` (reused, not duplicated), DSR gate through `engine.stats.dsr` (configured to reproduce `gates.dsr_check`); a flat window's DSR is reported as 0.0 as the legacy PSR did |
| `anchor.py` | `AnchorSet`, `pick_anchor` (min drawdown among CAGR > 4%) |
| `breaker.py` | `WfoPolicy`, `WfoState`, circuit breaker, 2-of-3 shadow re-entry, WFE (unchanged logic) |
| `scorers.py` | `SignalScorer` (one grouped full-history simulation sliced per window) and `WeightScorer` (fresh-start stacked windows) |
| `run.py`, `report.py` | `run_wfo` (legacy call signature), `WfoResult` (legacy fields plus `diagnostics`), `format_wfo_table` (identical text), JSON payload |
| `inputs.py`, `cli_support.py` | `Bars` or legacy-frame input, PIT masks, `ggt lab --engine vbt` glue |

Other changes: `engine/stats.total_return` (added); `lab/cli.py` (`--engine {legacy,vbt}`, `--snapshot`, `--result-json`, `--max-leverage` default 1.0 per OD13); `lab/blend.py` (`engine=`, `snapshot=`, leverage default 1.0, the legacy body moved verbatim into `_legacy_curves`; the vbt path stamps `lab_runs.provenance`; **no migration was run**); `ruff.toml` and `tests/engine/test_boundary.py` (the other agent's unrouted `scripts/new_stack_reference_runs.py` joins the transitional vectorbt allow-list; the script itself is untouched); docs (`changelog.md`, `cli_reference.md`, `architecture.md`).

Scripts (argparse, `--help`, resumable, JSON behind every number): `scripts/p5_wfo_validation.py` (runs and compares against the P0 JSON), `scripts/p5_equity_parity.py` (legacy-vs-new equity, two phases, checkpointed per window end), `scripts/p5_tie_audit.py`.

## 2. The correctness proof: equity-level parity (17-fold SP500 ensemble)

Setup: P0's configuration (frozen tape, eval 2021-01-31 to 2026-04-30, window ends on the 2026-04-29 bar, 599 symbols loaded (the tape holds 605), 48-combination ensemble grid, `STOCK_BASE_CONFIG`, PIT mask `pit_mask(min_history_bars=400)`).

**Causality (the crux, F's finding).** Slicing one full-history simulation equals simulating each window on data truncated at its end because every indicator, the PIT mask, the time stop, the stops and the simulator are causal. Not assumed: `tests/lab/wfo_v2/test_causality.py` rebuilds signals and simulates on bars truncated at three ends and demands **bit-identical** equity for all 48 combinations from the first bar, equal cumulative trade counts and equal window statistics, on a synthetic panel with NaN holes and on 40 real-tape symbols, for five exit variants: none, `td_stop=10`, trailing stop 5%, ATR trailing stop 2x14, take-profit 8%. All 10 pass. A deliberately forward-looking strategy fails the same check (the test has teeth). I did not need to stop or adapt anything.

**Full-size proof** (`scripts/p5_equity_parity.py`, evidence `_p5_equity_parity_core17_ensemble_sp500.json`): the new single simulation against the legacy `lab.wfo._sweep_fold` (the legacy per-window code, run on this stack) re-simulated at each of the **18 distinct window ends** of the 17 folds (every `train_end` and `test_end`; the anchor, train sweep and live sweep are slices of these):

| Check | Result |
|---|---|
| per-combination equity from the first loaded bar to each end, 48 combinations x 18 ends = 864 curves | **864 of 864 bit-identical**, max abs difference **0.0** |
| legacy `pf.trades.count()` against the new count of entries before that end | **equal for all 864** |
| test windows (single-combination legacy simulations) | covered by the OOS curves: the new stitched OOS equity of the legacy-definition runs equals P0's legacy OOS equity with max abs difference **0.0** over all 1,065 bars (both the anchor-dominated and the DSR-threshold-0 winner-path runs) |
| weight path (fresh-start stacked windows) | `tests/lab/wfo_v2/test_legacy_equivalence.py` on real tape: `xs_momentum` (30 symbols, 12 combinations) and `cross_asset_sleeve` (5 ETFs, 3 combinations): train, test and anchor windows bit-identical to legacy `_sweep_fold_weights`, and trade counts equal; full-size: P0's `ref_xs_momentum`, `ref_cross_asset_sleeve` and `ref_cross_asset_sleeve_17fold` OOS equity bit-identical (diff 0.0) |

A finding on the weight path while proving trade counts: vectorbt writes one trade record per exit order, so a position reduced after `end` is several records in a full-window simulation and one open record in a simulation that stops at `end` (the legacy count). `WeightScorer.trade_counts` counts records closed before `end` plus each position open at `end` once, which reproduces the legacy count exactly (found because the first version overcounted the anchor, 124 against 106; no decision used it).

**OD12.** Per-window trade counts (entries inside the window) are what NDH's expectancy now uses. For every fold of every run the NDH gate was also evaluated with the legacy whole-simulation count: **zero decisions flipped in all 26 runs** (`diagnostics.od12_ndh_flips_folds` is empty everywhere), as the sign argument predicts.

## 3. Comparison with the P0 legacy references (classification)

### 3a. With the legacy Sortino/CAGR forms (the diagnostic, `*_legacydefs`)

| Run | Folds identical | Winners / gates / anchor / halt | OOS equity max abs diff | Aggregate (Sharpe, CAGR, MaxDD, SPY) | Class |
|---|---:|---|---:|---|---|
| `core17_ensemble_sp500` | 17/17 | 17/17 each | 0.0 | equal to 1e-12 | identical |
| `core17_ensemble_sp500_dsr0` | 17/17 | 17/17 each | 0.0 | equal to 1e-12 | identical |
| `smoke2_ensemble_sp500` | 2/2 | all | 0.0 | equal | identical |
| `ref_cross_asset_sleeve`, `_17fold`, `ref_ema_cross`, `ref_macd_divergence`, `ref_rsi_reversion`, `ref_volume_bb_reversion`, `ref_xs_momentum` | all | all | 0.0 | equal | identical |
| `ref_bb_reversion` | 0/2 | all identical | 0.0 | equal | **tie** (see below) |
| `ref_conviction_bb` | 0/2 | all identical | 1.1e-11 | equal | **tie** + conviction size rounding (P4) |
| `ref_retail_attention` | 0/2 | all identical | 0.0 | flat book, equal | **representation** (below) |

Classification of the three runs with a field-level difference (winners, gates, anchor, halt, OOS equity and aggregates are all identical in each):

* `ref_bb_reversion`, `ref_conviction_bb`: only the winner's `train_score` differs (about 4e-8 and 2e-6 relative). `_p5_equity_parity_ref_bb_reversion.json` and `_ref_conviction_bb.json` show 2 of 16 (bb) and all 36 (conviction, size rounding 1e-13) non-deployed combinations whose equity differs from the legacy sweep, which moves the min-max normalisation range of the composite. `scripts/p5_tie_audit.py` (`_p5_tie_audit.json`) on the exact bars: bb exits differ in 32 cells, **4 inside the 200-bar burn-in, 28 ties, 0 real**; conviction exits 72 cells, **9 burn-in, 63 ties, 0 real**; entries 0. A tie is a close within 1e-9 of its threshold (vectorbt requires a strict cross, the legacy form is non-strict; P4's documented class (ii)).
* `ref_retail_attention`: `dsr_value` was `null` (NaN) against the legacy 0.0 for a flat book. Legacy PSR returns 0.0 when the standard error is not finite; fixed in `gates.run_dsr` (a non-finite DSR is reported as 0.0, still a failed gate). The `_legacydefs` JSON predates the fix; the current run is identical to the reference.

### 3b. With vectorbt's definitions (the main result, E1 baseline candidate)

Same simulations; Sortino (a composite-score term) and CAGR (the anchor's >4% filter and the reported CAGR) now come from vectorbt. Sharpe, volatility and drawdown are the same in both sets (max difference between the sets: 0.0, `attribution.max_sharpe_diff_between_sets`).

| Run | Winners differing from P0 | Gates (vbt / P0) | Anchor folds (vbt / P0) | OOS Sharpe / CAGR % / MaxDD % | SPY Sharpe |
|---|---:|---|---|---|---:|
| **`core17_ensemble_sp500`** | 4 (folds 8, 9, 16, 17) | 1/17 / 1/17 | 17 / 17 | **0.391 / 3.50 / -9.58** (P0: 0.391 / 3.49 / -9.58) | 0.779 |
| `core17_ensemble_sp500_dsr0` | 4 (same folds) | 16/17 / 17/17 | 1 / 0 | **0.886 / 4.91 / -5.28** (P0: 0.622 / 3.31 / -5.28) | 0.779 |
| `smoke2_ensemble_sp500` | 0 | 0/2 / 0/2 | 2 / 2 | 0.623 / 7.5 / -7.0 | -0.515 |
| `ref_rsi_reversion` | 1 (fold 2) | 0/2 / 0/2 | 2 / 2 | 0.334 / 4.40 / -7.14 | -0.515 |
| other eight strategies | 0 | equal to P0 | equal | Sharpe and drawdown equal to P0; CAGR within 0.1 pt | |

Evidence: `_p5_<run>.json`, `_p5_compare_<run>.json` (against P0), `_p5_diff_<run>.json` (against the legacy-definition twin, with the attribution).

## 4. Definition diagnostic: what Sortino and CAGR changed (17-fold ensemble)

Attribution method (`_p5_diff_core17_ensemble_sp500*.json`, key `attribution`): one simulation, each fold's train sweep scored with both sets; an ablation scores the legacy composite with only the Sortino column swapped to vectorbt's.

* **All 4 winner flips are explained by Sortino alone** (the Sortino-only ablation reproduces the vbt winner in 4 of 4). Folds 8, 9, 16: the legacy winner's Sortino was higher under the legacy form (for example fold 8: 1.03 against 0.65 for the other combination) and lower under vectorbt's (3.01 against 3.53). **Fold 17** is the cautionary one: the vbt winner has no legacy Sortino (fewer than two down days, legacy NaN) and a vectorbt Sortino of **28.6**. vectorbt's downside deviation over all observations rewards a combination with almost no losing days; this is the owner-chosen definition, but it is what moved the fold.
* **No anchor changed because of CAGR** (0 anchor flips in the attribution: the CAGR differences, at most about 0.1 pt, never crossed the 4% filter in a way that changed the minimum-drawdown pick). The one anchor-use change in the dsr0 run (fold 8) is downstream of a winner change.
* **No gate decision changed without a winner change** (`gate_changes_without_a_winner_change` is empty in every run). In the primary run NDH flips in fold 8 (one winner change) but the fold still fails DSR, so gates stay 1/17. In the dsr0 run the same winner change fails NDH in fold 8, giving 16/17 gates and one anchor fold, and moves the OOS result from Sharpe 0.622 to 0.886 and CAGR 3.31% to 4.91%. SPY on the same days: Sharpe 0.779, CAGR 13.0%. The verdict against SPY is unchanged in both (below SPY on Sharpe, CAGR and drawdown-adjusted return); the dsr0 variant is a diagnostic, not a production configuration.
* SPY's own CAGR moves from 12.96% to 13.02% (bar-annualised against calendar days) with no other change.

## 5. Timings, memory, simulations (frozen tape, box shared with other jobs, load 3 to 4 on 4 cores)

| Run | Legacy (P0) | New | Notes |
|---|---:|---:|---|
| 17-fold ensemble wall time (`/usr/bin/time -v`) | 1,421 s (about 24 min) | **22.2 s** total process (18.2 s in the driver; engine 17.3 s: build 0.9 s, one simulation 13.7 s in 2 chunks, all folds' scoring 2.6 s) | about 64x faster; target (well under 10 min) met |
| peak RSS, 17-fold | 4,971 MB (needed a gc hook; 6 GB cap killed the first attempt) | **3,649 MB** (cap 6 GB, swap off, chunk budget 3 GiB) | below the 6 GB criterion |
| grouped simulations | 52 (2 full-grid sweeps + 1 test per fold, +1 live) | **1** (2 memory chunks) | 4 for a weight strategy (anchor, train, test, live stacks) |
| 2-fold smoke | 363 s | 7-9 s | |
| `ref_xs_momentum` 2-fold | 182 s | 15-16 s | |

The 17-fold legacy-definition runs took 18-19 s in the driver; the legacy per-window equity re-simulation for the parity proof took 24-52 s for each of 18 ends.

## 6. Tests (all through `scripts/run_tests.sh`)

* New: `tests/lab/wfo_v2/` with 72 tests (24 logic units, 9 toy oracle, 11 causality, 3 legacy equality, 7 leak/golden/memory, 10 weights/report/inputs, 8 CLI): pure logic against the legacy functions (folds, composite, live pick, breaker on 300 random histories, shadow re-entry, NDH lattice, DSR), a closed-form toy oracle (independent equity, fold-by-fold expected winners, gates, anchor, breaker and OOS stitching), causality (10 variants plus the teeth test), legacy equity equivalence on real tape (3), leak tests through the whole WFO (bars after `eval_end`, bars after the last fold), a 3-scenario regression golden (`golden/mini_wfo.json`), memory budget (a tiny budget splits the simulation into chunks without changing a number; planned chunks of the real 29,040-column run fit 3 GiB), weight windows (fresh start, stacking changes no number, anchor is a prefix), report format, inputs, CLI wiring. Real-tape tests skip with a reason when the parquet is absent.
* Engine: the `legacy_defs` tests and the expiry guard were removed with the module; `test_stats`/`test_provenance` adjusted (the "differs from the legacy form" assertions are gone because the legacy form is gone). No other existing test was weakened or deleted.
* Whole suite via `scripts/run_tests.sh tests/`: **3,036 passed, 0 failed**, 30 integration deselected, 12 min 33 s (P4 baseline 2,732 plus P2b/P5 additions, minus the removed legacy-definition tests). `ruff check .` and `ruff format --check .` clean (410 files).

## 7. Uncertain or unfinished

* **Winners/gates changed on 4 of 17 folds relative to the old numbers** by decision (Sortino). The primary 17-fold result is anchor-dominated (anchor 17/17), so the changed winners do not move its OOS equity at all; only the dsr0 diagnostic shows a visible shift. This should be read as "E1 is a different measurement", not as evidence about edge. The fold-17 Sortino of 28.6 is worth a glance by the owner.
* **Coverage of the equity proof.** Bit-identity is proven for the ensemble grid on the real tape (18 ends x 48 combinations) and for the weight path on a 30-symbol sample plus the P0 full-size OOS curves. Not run: every one of the other eight strategies' full grids against the legacy at every end (their OOS curves are bit-identical in the P0 comparison, and bb/conviction have the classified ties).
* **Mask provenance.** The PIT mask is `lab.universe.pit_mask`, proven equal to `eligible_at` in P3 (0 mismatches); the equity parity inherits that proof rather than re-running `pit_entry_mask` (about 140 s per run).
* **`last_exit`, trader, `--sweep`, plain `walkforward` runs** are not touched (P7 and legacy).
* **`oos_score`** is the legacy always-0.0 composite of a one-element list; kept for compatibility.
* **Combination order** follows the legacy product over sorted parameter names, so first-max tie-breaks match; a caller's differently ordered list is re-ordered.
* **Compatibility not exercised:** the legacy scripts still import `lab.wfo`; none was switched (they are research drivers, listed below).
* The constituent file is 150 days old against the tape (warning suppressed in `pit_mask(on_stale="ignore")` for parity; the age check record belongs to P6/OD5).

## 8. Deletion list for the lead (nothing deleted; legacy is still the default)

Becomes dead once `--engine vbt` is the default and the 28 unported strategies are archived at the legacy tag:

* `src/ggTrader/lab/wfo.py` (entire file, 1,076 lines) after switching its importers to `lab.wfo_v2`: `lab/blend.py` (`_legacy_curves`), `lab/cli.py`, `lab/wfo_policy_synthetic.py` and `scripts/wfo_policy_synthetic.py` (`WfoPolicy`, `WfoState`, breaker functions now in `wfo_v2.breaker`), and the research drivers `anchor_fix_reproduction_wfo.py`, `audit_reruns_wfo.py`, `core_spy_blend.py`, `cross_asset_sleeve_wfo.py`, `lever_diagnostic.py`, `midcap_research.py`, `midcap_bias_calibration.py`, `month_end_treasury_wfo.py`, `position_sizing_wfo.py`, `ablation_voters.py`, the `*_correlation_check.py` and `*_research.py` drivers (most import unported strategies: archive them at the tag instead of porting).
* `src/ggTrader/lab/sweep.py`: `pit_entry_mask`, `sweep_signal_group`, `group_by_stop_config`, `run_sweep` (the `--sweep` mode), `format_results_table`; `build_grid`, `combo_name`, `build_combo_lookup`, `split_params` are still imported by the legacy strategy classes and scripts, and `combo_name` now also exists in `wfo_v2.inputs`.
* `src/ggTrader/lab/simulate.py` (`simulate_signals`, `simulate_weights`, `compute_atr_stop`, `compute_vol_scalar`) once the legacy strategies, `harness.py`, `idio_vol.py`, the P4 parity script and the tests that import it are gone (`tests/lab/test_simulate*.py`, `test_vol_target.py`, `test_conviction.py`, `strategies_v2` parity tests, `tests/paper/test_overlay.py` via `paper/overlay.py`, which P7 rebuilds).
* `src/ggTrader/lab/harness.py` (`walkforward`, `leak_check`) with `tests/lab/test_harness*.py`; non-WFO runs become a one-fold WFO.
* Kept on purpose: `lab/gates.py` (NDH and the DSR oracle), `lab/metrics.curve_stats` (oracle for Sharpe/volatility/drawdown conventions; `significance.py`, `blend.py` and scripts use it), `lab/data.py` (rebalance dates, base config), `lab/universe.py`.
* Tests deleted together with the legacy modules: `tests/lab/test_wfo.py`, `test_wfo_policy_synthetic.py`, `test_sweep.py`, `tests/lab/wfo_v2/test_legacy_equivalence.py` and the legacy-comparison halves of `test_logic_units.py`/`test_report_inputs_weights.py` (marked in their docstrings).
* After all legacy importers go: remove `src/ggTrader/lab/simulate.py` and `lab/strategies/signals.py` from `TRANSITIONAL` and `ruff.toml`; the boundary test fails if an entry goes stale.
* Scripts that become evidence-only: `scripts/legacy_reference_runs.py`, `scripts/new_stack_reference_runs.py` (both on the transitional list), `scripts/p5_equity_parity.py` (needs `lab.wfo`).

## 9. Files

Created: `src/ggTrader/lab/wfo_v2/*` (11 files), `tests/lab/wfo_v2/*` (10 files and `golden/mini_wfo.json`), `scripts/p5_wfo_validation.py`, `scripts/p5_equity_parity.py`, `scripts/p5_tie_audit.py`, this note, `p5-evidence/` (about 75 files: per-run `_p5_<run>.json` with provenance, `_legacydefs` twins, `_p5_compare_*`, `_p5_diff_*`, `_p5_equity_parity_*`, `_p5_tie_audit.json`, OOS equity CSVs, and two `.npz` of the new equities).
Modified: `src/ggTrader/engine/{stats,provenance,config}.py`, `src/ggTrader/lab/{cli,blend}.py`, `tests/engine/{test_stats,test_provenance,test_boundary}.py`, `ruff.toml`, `docs/{changelog,cli_reference,architecture}.md`, the spec (Appendix E).
Deleted: `src/ggTrader/engine/legacy_defs.py`, `tests/engine/test_legacy_defs.py`.

## 10. P5b update (2026-11-06): guarded E1 candidate

Same configuration as section 2 (frozen tape, eval 2021-01-31 to 2026-04-30, 48-combination ensemble grid, vectorbt definitions). The guards: selection windows (anchor, train, live) blank Sharpe and Sortino of a combination with no entry inside the window (spec 5.6) and Sortino below 20 losing days (`engine.config.MIN_LOSING_DAYS`). Evidence: `p5-evidence/guarded/` (default guards), `p5-evidence/guarded_zero_entry_only/` (zero-entry rule only, `--min-losing-days 0`), `p5-evidence/guarded/_p5b_before_after.json`, `_p5b_all_runs_before_after.json`.

| Run | Variant | Gates | Anchor folds | Winners differing from the unguarded run | OOS Sharpe / CAGR / MaxDD | SPY (same days) |
|---|---|---:|---:|---|---|---|
| core17 | unguarded (section 3b) | 1/17 | 17 | - | 0.391 / 3.50% / -9.58% | 0.779 / 13.02% / -22.09% |
| core17 | zero-entry rule only | 1/17 | 17 | fold 8 | 0.391 / 3.50% / -9.58% | same |
| core17 | **guarded (E1 candidate)** | 1/17 | 17 | folds 8, 9, 16, 17 | **0.391 / 3.50% / -9.58%** | same |
| core17 DSR threshold 0 | unguarded (section 3b) | 16/17 | 1 | - | 0.886 / 4.91% / -5.28% | same |
| core17 DSR threshold 0 | zero-entry rule only | 16/17 | 1 | fold 8 | 0.886 / 4.91% / -5.28% | same |
| core17 DSR threshold 0 | **guarded** | 17/17 | 0 | folds 8, 9, 16, 17 | **0.637 / 3.41% / -5.28%** | same |

* The four folds the guard changes are exactly the four the section 4 attribution blamed on "vectorbt Sortino". In the zero-entry-only run fold 8 changes between two combinations with the same Sharpe to 1e-13 (a float-noise tie, review MINOR-1); no metric moves.
* The primary result is anchor-driven (the winner never reaches the money after the breaker trips in fold 5), so the guard cannot move it. The dsr0 run, where winners are deployed, loses its 0.886: in fold 8 the one-entry, four-losing-day winner failed NDH, routing a lucky anchor (+7.6% in Q4 2023) into the book. Guarded, every gate passes and the Sharpe lands at 0.637, close to the legacy-definition diagnostic (0.622).
* Neither variant beats SPY (Sharpe 0.779, CAGR 13.0%). A research result, not a retirement recommendation.
* The eleven other runs (smoke and the nine other strategies; `_p5b_all_runs_before_after.json`) are unchanged in winners, gates, anchor use and OOS figures. `ref_retail_attention` (a book that never trades) has no anchor now (a never-traded combination is no longer the "safest" fallback, MINOR-3); the book stays in cash, so its flat out-of-sample record and SPY comparison are unchanged.
* Recorded in each fold row now: `anchor_combo`, `anchor_params`, `anchor_reason`, `deployed_combo`, `halt_cleared` (MAJOR-3). In the guarded core run the anchor was one of 5 parameter sets across the 17 folds.
