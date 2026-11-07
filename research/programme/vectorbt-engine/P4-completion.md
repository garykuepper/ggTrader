# P4 completion note: strategy port onto the engine contract, `ensemble` first

**Date:** 2026-11-06. **Phase:** P4 of the [vectorbt engine rebuild](../../../docs/superpowers/specs/2026-11-06-vectorbt-engine-design.md) (section 9). **Branch:** `vbt12/engine` (nothing committed, staged, tagged or pushed).
Terms: **PIT** = point-in-time. **OD** = open decision in the spec. **Tape** = the frozen daily-bars file `data/snapshots/ohlcv_20261106.parquet` (sha256 `6573e840...789b2e`). **Burn-in** = the first 200 bars after a symbol's own first valid close. **Tie** = a compared quantity within 1e-9 of its threshold on the flip bar or the bar before. **Real** = a difference after the burn-in that is not a tie.

## Verdict against the exit criteria

| Criterion (spec 7 / P4) | Result |
|---|---|
| (a) entry/exit frames vs legacy, defaults and a sample of grid combinations, >= 150 symbols x full window, ZERO after burn-in except documented ties | **Met, with far more than the minimum.** Every valid combination of every ported signal strategy (not a sample of 6) plus the defaults, on two 200-symbol samples x 1,929 bars. **0 real differences** in all 10 strategy/sample reports; after-burn-in differences are exact float ties only (tables below). |
| (b) same signals into `engine.run_signals` and `lab.simulate.simulate_signals`: equal equity | **Met.** 24 of 24 cases bit-identical equity and equal trade counts (ensemble 6, conviction_bb 3, five voters 3 each). |
| (c) P0 signal golden (ensemble) | **Met.** 150 symbols x 1,929 bars: ensemble entries differ in 2 of 4,673 cells, exits in 204 of 24,322; every one in the burn-in except 1 tie (BIL, 2021-05-21). |
| (d) live-defaults control config reproduces P0's live golden, last 120 sessions | **Met, exactly.** `buys`, `sells` and `last_exit` identical in **120 of 120** sessions (1,028 buys, 5,519 sells, 100 short-history symbol-sessions of 60-199 bars included, 0 `last_exit` differences). |
| (e) leak / lookahead, generalised from `lab.harness.leak_check`; vbt label functions forbidden | **Met.** Truncation and perturbation checks on real bars (40 dates x 80 symbols, 8 signal configurations, plus weights): 0 violations. A deliberately forward-looking strategy is caught by both variants. An AST test bans `vectorbt.labels` and the label functions in strategy code. |
| (f) grid sizes and validity rules equal the legacy `build_grid` for every ported strategy | **Met.** Combinations equal in content and order for all 10 (ensemble 48, ema_cross 16 of 20 raw, bb_reversion 16, rsi_reversion 27, macd_divergence 8, volume_bb_reversion 12, conviction_bb 36, xs_momentum 12, cross_asset_sleeve 3, retail_attention 6). |
| (g) full ensemble grid x 500+ symbols within the chunking budget | **Met for the build; see finding 1 for the engine.** 48 combinations x 605 symbols x 1,929 bars = 29,040 columns build in 1.1 s, 112 MB of signals, process RSS 0.96 GB. The engine ran it in one call under the 6 GB cgroup (peak RSS 5.7 GB); with a 1.5 GiB budget it split into 3 chunks and still peaked at 4.7 GB. |
| Tests via `scripts/run_tests.sh` | `tests/lab/strategies_v2`: **102 passed** (about 17 s). Whole suite: **2,732 passed**, 30 integration deselected, 0 failed, 9 min 44 s (2,630 in P2 plus 102 new). |
| Ruff | `ruff check .` and `ruff format --check .` clean (375 files). |
| Changelog | entry added at the top of `docs/changelog.md`; `docs/architecture.md` has a new contract section and tree entries. |

## What was built (new files only; no legacy file was edited)

- `src/ggTrader/lab/strategies/vote.py`: `VoterArrays`, `threshold`, `vote` (N-of-M over precomputed boolean arrays, independent RSI exit that also counts as an exit vote, `min_agree_exit=None` means `min_agree`, a per-voter-parameter vote-sum cache, stacking into one wide frame).
- `src/ggTrader/lab/strategies/v2/` (the new registry is `REGISTRY_V2`; naming decision: a `v2` subpackage next to the legacy modules, so P5 deletes the legacy files and moves these up):
  `contract.py` (`ParamGrid`, `Lt` rule, `resolve_combos`, `group_columns`, `tile_columns`, `SignalOut`, `next_bar_positions`, `stack_weight_plans`, `simulate_signal_out`), `voters.py`, `ensemble.py`, `voter_strategies.py`, `conviction_bb.py`, `xs_momentum.py`, `cross_asset_sleeve.py`, `retail_attention.py`, `leak.py`, `registry.py`.
- `tests/lab/strategies_v2/` (9 modules, 102 tests). `scripts/p4_strategy_parity.py` (evidence driver, 8 sections). `research/programme/vectorbt-engine/p4-evidence/` (10 JSON files with provenance, plus `chunker_experiment.py`).
- Docs: `docs/changelog.md`, `docs/architecture.md`.

Contract decisions the spec left open (all documented in the module docstrings):
- A strategy declares `grid: ParamGrid` (the legacy `sweep_params()` values, including pinned single-valued names, plus validity rules as data: `Lt("ema_fast", "ema_slow")`, `Lt("macd_fast", "macd_slow")`) and `defaults` (legacy constructor defaults for names the grid does not list, e.g. `rsi_exit`). `grid.combos()` equals `build_grid`. Column levels are the **swept** names in sorted order plus `symbol`; a one-combination grid (live) uses a constant `combo` level.
- `SignalOut` carries `combos` (the parameter dict per group), `size` (`None` = the caller's `SIGNAL_POSITION_SIZE`, a frame = per-bar sizes with NaN = base size), stop metadata and `td_stop`. The spec sketch's default `size=0.03` would override the lab base config, so `None` keeps legacy behaviour.
- Weight strategies take `build(bars, grid, rebal, eligible=None)`: an **extra `eligible` argument** (a PIT mask, `lab.universe.pit_mask`) because the ranking must see only eligible names; masking the weights afterwards would pick different names. Weights land on the first bar after each decision date; columns are the union of symbols selected by any combination, a symbol a combination never selected stays at 0.0 there.
- Voters are defined once (`voters.py`) and shared by the ensemble and the five standalone strategies. Each indicator runs once per distinct parameter tuple.
- `Ensemble.grid` is the 48-combination WFO grid (`divergence_window` 10, `vol_mult` 1.5); `Ensemble.live` / `Ensemble.live_grid()` are the deployed defaults (20 and 2.0, `min_agree_exit` 2), which are outside the grid (checked in `p4_grids.json`). OD14 reference rows are both available.
- The volume-BB double vote is kept (OD3a): `vbb` entry = Bollinger entry AND volume spike (a strict subset of the plain entry, tested), its exit array is identical to the plain Bollinger exit, and both vote.

## Comparison results (all JSON in `p4-evidence/`, tape-only, no database)

### (a) Signal frames, 2 samples x 200 symbols x 1,929 bars, all valid combinations, difference classes

Sample A = first 200 symbols alphabetically; sample B = seeded random 200 of the rest (recent listings and dead names included). Cells are summed over all parameter tuples; the voter rows compare the new voter arrays with the legacy `*_signals` functions, the strategy-level check repeats it through the real legacy `sweep_signals`.

| Strategy (sample A / B) | Tuples | Entry cells that differ: burn-in / tie / real | Exit cells that differ: burn-in / tie / real |
|---|---:|---|---|
| bb_reversion | 16 | 0 / 0 / 0 and 0 / 0 / 0 | 4 / 48 / 0 and 0 / 52 / 0 |
| rsi_reversion | 27 | 642 / 0 / 0 and 648 / 0 / 0 | 2,814 / 0 / 0 and 3,003 / 0 / 0 |
| ema_cross | 16 | 5,020 / 0 / 0 and 5,433 / 0 / 0 | 4,839 / 0 / 0 and 5,257 / 0 / 0 |
| macd_divergence | 8 | 131 / 0 / 0 and 91 / 0 / 0 | 3,514 / 0 / 0 and 3,414 / 0 / 0 |
| volume_bb_reversion | 12 | 0 / 0 / 0 and 0 / 0 / 0 | 0 / 18 / 0 and 0 / 36 / 0 |
| conviction_bb | 36 | 0 / 0 / 0 and 0 / 0 / 0 | 9 / 108 / 0 and 0 / 117 / 0 |
| **ensemble, 48 grid combinations** | 48 | **20 / 0 / 0 and 12 / 0 / 0** | 21,204 / 48 / 0 and 22,380 / 144 / 0 |
| **ensemble, live-defaults control** | 1 | 3 / 0 / 0 and 3 / 0 / 0 | 272 / 1 / 0 and 296 / 3 / 0 |

Strategy-level `sweep_signals` check: `entries_xor_after_burn_in` is 0 for every strategy in both samples; `exits_xor_after_burn_in` is 0 except bb_reversion (48 and 52) and volume_bb_reversion (18 and 36), which are exactly the tie counts above.

**Classification of every difference.**
1. *Burn-in (all the RSI, EMA, MACD and most ensemble-exit differences).* Where the legacy form emits values from bar 0 (EMA and MACD seeded at the first bar, RSI valid from bar `n`), vectorbt masks `w-1` bars (EMA), 26+9 bars (MACD) or `2n-1` bars (RSI, first valid bar 27 for n=14) as NaN (spec OD1: adopt vectorbt's warm-up). Every such cell is within 200 bars of the symbol's own first close, so a 400-bar PIT history rule never reaches them.
2. *Tie (bb, volume-BB and ensemble exits, conviction exits).* Legacy exits when `close >= SMA` after a bar below; vectorbt requires a strict cross. When a close equals its own 10-30 bar average exactly (a flat T-bill ETF such as BIL; or a single-day print equal to the mean to the last bit or to within 1e-13, e.g. SW, V, PNW, UAA, CTXS) the legacy exits on the tie bar and vectorbt exits on the next bar. Each such event shows as a pair of differing cells (the tie bar and the bar after); the audit treats a cell as a tie when the distance to the threshold is below 1e-9 on the flip bar or the previous bar (examples with distances are in `tie_examples` of `p4_signals.json`). One further case: a volume spike exactly on its threshold (vbb) is covered by the same rule.
3. *Real:* none. No unexplained difference after the burn-in in any strategy, combination or sample, nor against the P0 golden.

Ensemble flips are attributed to voters: a flipped ensemble cell counts as explained only if at least one voter differs at that cell, and it inherits the voter's class (cells with no differing voter would be reported real; there are none).

### (b) Simulator equality (`p4_simulator.json`)

PIT mask `pit_mask(min_history_bars=400)` applied before the time stop; `STOCK_BASE_CONFIG` (3% size, 0.0005 slippage, no fees); 200 symbols x 1,929 bars.

| Strategy | Cases (combinations) | Same signals: engine vs `lab.simulate` | Legacy signals vs new signals (each in its own path) |
|---|---:|---|---|
| ensemble | 6 | 6/6 bit-identical, trade counts equal | 6/6 bit-identical equity |
| bb / rsi / ema / macd / volume_bb | 3 each | 15/15 bit-identical | 15/15 bit-identical |
| conviction_bb (per-bar sizes) | 3 | 3/3 bit-identical | 0/3 bit-identical; max abs equity difference 3.6e-11 on 10,000+ (sizes differ by about 1e-13 because the band half-width is `middle - lower` of the Bollinger preset instead of `std * rolling std`); trade counts equal |

### (c) P0 signal golden (`p4_golden.json`, live-default parameters, first 150 symbols)

| Voter | Entries differ (burn-in / tie / real) | Exits differ (burn-in / tie / real) |
|---|---|---|
| bb | 0 | 0 / 1 / 0 |
| rsi | 27 / 0 / 0 | 82 / 0 / 0 |
| ema | 177 / 0 / 0 | 176 / 0 / 0 |
| macd | 5 / 0 / 0 | 325 / 0 / 0 |
| vbb | 0 | 0 / 1 / 0 |
| ensemble (golden 4,673 entries, 24,322 exits) | 2 / 0 / 0 | 203 / 1 / 0 |

### (d) Live golden (`p4_live.json`): 0 of 120 sessions differ in buys, sells or `last_exit`; the order caps (alphabetical, `MAX_POSITIONS`) are untouched by this phase, so the intended orders after the cap are also unchanged for those sessions.

### Weights (`p4_weights.json`, 63 decision dates of the 17-fold window, PIT universe)
- `xs_momentum`, 12 combinations: target weights equal the legacy `select`/`to_targets` frames exactly (max abs difference 0.0) and the stacked equity from `engine.run_weights` equals `simulate_weights` bit for bit; trade counts equal. The new build takes 2.0 s against 123 s for the legacy per-date `select` loop (the legacy lists were fed the same eligibility).
- `cross_asset_sleeve`: trend arm (3 lookbacks) and the static arm: targets and equity bit-identical.
- `retail_attention`: the `google_trends_interest` table has 0 rows, so the real book is flat cash and no real-data run exists. The logic is ported faithfully (copy of the spike score, `available_as_of`, quintile bucket) and tested on synthetic interest data against the legacy class for all 6 combinations (targets equal); empty data gives no columns. The real loader was not called (a read-only session refuses its `CREATE TABLE IF NOT EXISTS`).

## Not ported (legacy-only at tag `legacy-engine-vbt0285`), 28 of 38 registry names

`commodity_trend`, `congress_trades`, `dual_momentum`, `ensemble_conviction`, `ensemble_ic`, `ensemble_kelly`, `fomc_drift`, `fx_hedge_overlay`, `headline_sentiment`, `idio_vol`, `index_deletion_fade`, `insider_cluster_buy`, `leveraged_rotation_nasdaq100`, `leveraged_rotation_russell2000`, `leveraged_rotation_sp500`, `leveraged_trend_nasdaq100`, `leveraged_trend_russell2000`, `leveraged_trend_sp500`, `max_effect`, `month_end_treasury`, `mtf_reversion`, `overnight_gap`, `pairs_stat_arb`, `pead`, `short_interest`, `short_volume_ratio`, `treasury_curve`, `wfo_tournament`. `registry.get()` raises an error naming the tag. Inside the ported set, not carried over: the `mtf` voter, `exits_enabled=False` and a swept `td_stop` (a fixed `td_stop` is carried in `SignalOut` for the engine), and the weight path's `max_sector_count`.

## Findings and problems

1. **Engine chunker does not follow its budget (engine finding; not patched).** `engine/portfolio.py::_run_chunks` keeps each chunk's `Portfolio` object alive after reading its results. With the full ensemble grid x 605 symbols, a 5 GiB budget runs in one call and peaks at 5.7 GB RSS (the single-call estimate is 3.76 GiB; the baseline process is about 1 GB); a 1.5 GiB budget splits into 3 chunks and still peaks at 4.7 GB (`p4_memory*.json`). Freeing each chunk (`del pf, value; gc.collect(); malloc_trim(0)` after `trade_frames.append(records)`, tested by wrapping the function in `p4-evidence/chunker_experiment.py`) gave peaks of 5.10 GB (as is), 3.55 GB (3 chunks) and 2.73 GB (5 chunks) with equity identical in all three runs (`np.array_equal`). The tiled `close`, mask and a forward-filled copy also sit outside the estimate. This did not block P4 (the run fits the 6 GB cap), so I did not touch `engine/`; it matters for the P5 17-fold run (criterion: peak under 6 GB). The P2 reviewer should decide the fix.
2. **Workspace slip (corrected).** While creating the first test file one `cd` failed in a chained command and `tests/lab/strategies_v2/test_contract.py` was written to `/home/flynn/ggTrader/test_contract.py` (the read-only production checkout). I moved it into the worktree straight away; `git status` of the production checkout is again exactly the three untracked paths it had at the start. Nothing else touched it.
3. **No defect found in `engine/`, `universe.py`, `bars.py` or `snapshot.py`.** They worked as specified. Two usage notes for P5: `run_signals` needs `close`, `mask`, `size` already tiled to the grid columns (`contract.tile_columns`; `contract.simulate_signal_out` does it), and vectorbt indicator output frames must be copied to a local before use (re-noted from P2 finding 4; `voters.py` does so).
4. The vbt strict-cross tie behaviour (finding in the table above) is a permanent, documented class (ii) difference, not a bug; it only appears for a close that equals its own average to the last bit.

## Uncertain or unfinished

- The live golden proves `Ensemble.build()` with the one-combination live grid reproduces the legacy `generate_signals` results; `paper/signal_runner.py` and the trader are not changed (P7). The golden test slices the window and applies the 60-bar history and membership rules itself, as the legacy golden script did, rather than through a P7 extraction function.
- The comparison covers 400 of the 605 loaded symbols per strategy (two samples), all bars. The memory run covers all 605. A full-universe signal diff over every combination was not run (the spec's risk 1 asks for it; the two samples found 0 real differences).
- `conviction_bb` equity differs from the legacy by up to 3.6e-11 (class (ii), explained above); if P5 needs bit-identical conviction sizes the band half-width must be taken from the legacy formula, which re-implements indicator maths and was not done.
- Weight-path ties in ranking (equal momentum scores) are resolved by symbol order here and by eligibility-list order in the legacy; no tie occurred in the 12 x 63 selections.
- No new WFO was run (P5). No definition was changed; only OD1(a) presets were used.
