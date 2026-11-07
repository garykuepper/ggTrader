# F: Verification of reports A-D and the vectorbt decision for ggTrader

Date: 2026-11-05. Auditor: Opus 5.5, read-only on `/home/flynn/ggTrader` (branch `docs/accept-synthesis`, HEAD `8eb2eb5d`).
New evidence produced for this document lives in `vbt-research/F-checks/`:
`f_real_checks.py` + `.log` (real cached SP500 tape, 507 symbols x 860 bars), `f_primitives.py` + `f_primitives_out.txt`,
`d3_rerun.txt`, `pd3.rst` (pandas 3.0 release-note source).

**One-paragraph answer.** vectorbt cannot be made the backbone of ggTrader beyond what it already is (the portfolio
simulator). Under 2% of the source (about 390 lines, counted below) has *any* vectorbt equivalent, and most of
those equivalents match only with non-default flags. The large wins the owner wants (speed, less code to maintain)
come from three **pure refactors that use no new vectorbt feature at all**: vectorise the PIT mask, stop
re-simulating the same grid, and memoise indicators inside a sweep. All three reproduce today's numbers exactly. The
vectorbt swaps (indicator kernels, returns accessor) are optional second steps. They change numbers at the 1e-9
level or by definition, and the indicator swap touches the live trader.

---

## (a) Verification scorecard

Legend: VERIFIED = holds as stated; PARTLY = substance holds but a qualifier, count or framing is off; WRONG;
COULD-NOT-CHECK.

### C (cheaper model): inventory of ggTrader hand-rolled code. 18 claims checked

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| C1 | lab = 18,591 lines | VERIFIED | `wc -l lab/*.py lab/strategies/*.py` = 18,591 |
| C2 | paper = 3,236 lines | VERIFIED | `wc -l paper/*.py` = 3,236 |
| C3 | ~3,999 hand-rolled primitive lines | PARTLY | Re-ran `count_primitives.py`: exactly 3,999, method = AST spans incl. docstrings. It is a reproducible count of *functions C chose to label*, and the label mixes in policy and accounting (TaxLotBook 378, replica 195, RiskGuard, A23 Book, gates, breaker, cash sweep, split/dividend). **Not** a measure of what vbt could replace: that subset is ~388 lines (see b). |
| C4 | ATR at `simulate.py:149` is a simple rolling mean of TR | VERIFIED | `tr.rolling(window=atr_period, min_periods=atr_period).mean()` at line 149 |
| C5 | Stochastic/Keltner/OBV/Donchian/VWAP/Calmar absent from lab/ and paper/ | VERIFIED | case-insensitive grep: 0 hits |
| C6 | `signals.py:75` bare `except Exception` returns -inf for every combo; `init_cash=10000.0` hard-coded | VERIFIED | lines 63 and 75-76. Consequence C understates: `max()` over all -inf then silently picks `_EMA_COMBOS[0]` |
| C7 | Sharpe: no risk-free, ddof=1, x sqrt(252); Sortino: std of negative returns only | VERIFIED | `metrics.py:19-29` |
| C8 | NDH plateau exemption, density 0.85 / variance 0.20, lines 105-112 | VERIFIED | `gates.py:105-112` |
| C9 | DSR per-period scaling and 0.80 pass, lines 206-210 | VERIFIED | `gates.py:205-210` |
| C10 | WFE is None when IS Sharpe < 0.4 | VERIFIED | `wfo.py:44-56` |
| C11 | Folds: rolling 12m train / 3m test / 3m step | VERIFIED | `wfo.py:20-21, 194-210` |
| C12 | `target_vol_scale` = clip(target/vol, 0, max_lev), 0.0 on invalid | VERIFIED | `allocation.py:38-48` |
| C13 | Kelly f* = W - (1-W)/R, NaN until min trades and both signs | VERIFIED | `kelly.py:105-112` |
| C14 | RiskConfig defaults 30 / 0.033 / 0.05 / 0.03 / 0.15 | VERIFIED | `paper/risk.py:13-21` |
| C15 | Default voters = FIVE_VOTERS (MTF excluded) | VERIFIED | `ensemble.py:31-42` |
| C16 | overlay target_vol 0.068, max_leverage 1.0; base_size fallback 0.02 | VERIFIED | `overlay.py:74-78`, `simulate.py:249` |
| C17 | "Only four Portfolio constructors appear in src/" | WRONG (minor) | There are three: `simulate.py:61` from_orders, `simulate.py:277` from_signals, `signals.py:57` from_signals. C's own table lists three. |
| C18 | Paper path imports lab EnsembleSignal and simulate, with no direct vbt import | VERIFIED | `overlay.py:11-14`, `signal_runner.py:10-12`, no vectorbt import in paper/ |

C: **16 VERIFIED, 1 PARTLY, 1 WRONG.** C's line references survived a branch change: C read `research/visual-reports`, which no longer
exists locally, and every sampled line still matched HEAD. **Trust: HIGH for facts and locations, LOW for the headline
3,999 figure as a deletion estimate.** C missed two things in code it says it read:
- `n_trades` is counted over the whole simulation, not the scoring window. See D3.
- The sweep path ANDs the PIT mask *after* the time stop. `sweep_signals` calls `_generate_signals` with no `entry_mask`,
  and `sweep.py:160-169` masks afterwards. This contradicts `_generate_signals`' own docstring and the `to_targets` path.
  It affects only grids that sweep `td_stop`; the default ensemble grid does not. I observed this in the code and did not measure it.

### A (cheaper model): vectorbt 0.28.5 inventory. 10 claims checked against installed source

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| A1 | `split_ranges_into_sets`:21, RangeSplitter:108, RollingSplitter:181, ExpandingSplitter:228 | VERIFIED | `generic/splitters.py` |
| A2 | `.vbt.split`:1335, `range_split`:1454, `rolling_split`:1538, `expanding_split`:1566 | VERIFIED | `generic/accessors.py` |
| A3 | Indicators MA:76 MSTD:166 BBANDS:236 RSI:363 STOCH:461 MACD:575 ATR:689 OBV:776 | VERIFIED | `indicators/basic.py` |
| A4 | Portfolio:1496, from_orders:1601, from_signals:2022, from_holding:3024, from_random_signals:3038, from_order_func:3145 | VERIFIED | `portfolio/base.py` |
| A5 | from_signals has 58 params | WRONG (minor) | `inspect.signature` gives 59 excluding `cls` (D says 59, correctly) |
| A6 | No engine setting in 0.28.5 | VERIFIED | grep "engine": only a giphy docstring |
| A7 | DSR `nb_trials` defaults to the column count, `var_sharpe` taken across columns | VERIFIED | `returns/accessors.py` deflated_sharpe_ratio source. A **missed** that it feeds scipy's excess kurtosis into a formula that needs non-excess kurtosis (`(kurtosis-1)/4`, `returns/metrics.py:24-26`). D caught this. |
| A8 | `param_product=False` default, `factory.py:1375` | VERIFIED | |
| A9 | `ParamLoc` at `base/indexing.py:160` | VERIFIED | |
| A10 | No time stop, no purge/embargo | VERIFIED | grep `time_stop|max_hold|td_stop|purg|embargo`: 0 code hits |

A: **9 VERIFIED, 1 WRONG (minor).** **Trust: HIGH for API locations. It inventories what exists and does not judge numeric semantics.**

### D: vectorbt WFO capabilities. 9 claims

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| D1 | Anchor (`wfo.py:656`) and train (`:675`) sweeps run the full grid on the same slice | VERIFIED | `compute_anchor_set` passes `ohlcv.loc[:train_end]`, then `_sweep_fold` slices `loc[:full_end]` = the same slice. Train slices `ohlcv.loc[:train_end]`. Same grid, same `entry_mask`, both simulate from the first loaded bar. The two simulations are byte-identical work. Only the scoring window differs (anchor scores from the first loaded bar, including pre-eval warm-up). |
| D2 | "re-simulates the same 48-combo grid ... 3 times per fold" | PARTLY | Per fold: two identical full-grid sims plus one *single-combo* test sim on `loc[:test_end]`, plus one full-grid live sweep per run. 2 folds = 7 sweeps (profile: `_sweep_fold` ncalls=7). |
| D3 | One full-history sim sliced per fold gives identical equity | PARTLY, and the equity part is VERIFIED on real code. See the crux below. |
| D4 | The 0.0 diff from its d4 test proves it for ggTrader | WRONG as evidence | d4 used D's own synthetic `Ens` indicator and raw `from_signals`, with no PIT mask, stops or `simulate_signals`. A causal toy is 0.0 by construction, so it could not expose a non-causal step in ggTrader's pipeline. My real-code test closes that gap (below). |
| D5 | Primitive table (SMA, ddof, EMA, RSI 2n-1, MACD, ATR, Sharpe 252/365, Sortino, CAGR, flat-Sharpe inf) | VERIFIED | Re-ran `d3_primitive_equivalence.py`: output identical to D's (`F-checks/d3_rerun.txt`). Independent script on different data (`f_primitives_out.txt`): SMA 1.3e-11; BB lower with vbt default ddof=0 is off by 1.09, with ddof=1 by 1.2e-9; RSI default is off by 36.1, RSI(27, ewm=True) has diff 0.0 but its first valid bar is 27 instead of 14 (104 cells differ); MACD with both ewm flags is off by 0.12 early and 8.9e-16 after bar 200; ATR(ewm=False) 2.1e-13, default ewm 1.69; Sharpe 252 equal to 10 digits, default 365 = 0.834 vs 0.693; Sortino 1.173 vs 1.012; CAGR 15.51 vs 14.94; flat curve Sharpe NaN vs inf; RSI-14 entries 141 vs 140, 0 differences after bar 100. |
| D6 | 0.28.5 DSR uses excess kurtosis where non-excess is needed | VERIFIED | source above |
| D7 | Live code uses these same RSI/MACD/BB functions | VERIFIED | `signal_runner.py:11` and `overlay.py:13` import `EnsembleSignal`, whose `_generate_signals` calls `indicators.py` |
| D8 | Splitters are bar-count based, the index is lost, `generate_folds` does not go away | VERIFIED (by source reading of `splitters.py:206-222`, `accessors.py:1403`). I did not re-run d1. |
| D9 | ~100-200 honestly deletable lines | VERIFIED as plausible | My upper bound for spans with *any* vbt equivalent is 388 lines incl. docstrings (b). Net deletion after wrapper flags and parity tests is in D's range (estimate). |

D: **6 VERIFIED, 2 PARTLY, 1 WRONG (as evidence).** **Trust: MEDIUM-HIGH.** It found the most
important thing in this whole exercise (the duplicate simulation), but proved it with a test that could not fail.

**The crux (D3). Is a sliced full-history simulation equivalent to today's per-window simulation?**
- The premise that today's simulation "starts fresh with cash at window start" is **false for the signal path**. `_sweep_fold`
  simulates `ohlcv.loc[:window_end]` from the **first loaded bar** (about 700 calendar days before eval start) and then scores
  a rescaled slice. Positions and cash carry into every window, including the OOS test window. Equivalence therefore
  needs only causality, not a fresh-start match.
- I read the path for anything non-causal and found nothing in the equity path. All indicators are rolling/ewm/shift; the MACD
  warm-up mask is counted from the data start; ATR and vol-scalar are rolling/shift/ffill; the PIT mask is precomputed per day;
  combos are separate vbt groups, so cash is never shared *across* combos.
- **Real-data test** (`f_real_checks.py`): real `_sweep_fold`, 120 real SP500 symbols, real-equivalent PIT mask, 80 combos
  (the 48-grid plus `td_stop=10`, `atr_mult=2`, `ts_stop+tp_stop`, `vol_target` variants). Prefix simulation to each
  `train_end` compared with one simulation to `eval_end`, sliced. Result for both folds: **max abs equity diff 0.0, max abs
  Sharpe diff 0.0**.
- **But `n_trades` differs in 72 of 80 combos** (for example 344 vs 485). `pf.trades.count()` counts every trade in the simulation,
  not the scoring window. That number feeds expectancy = window return / n_trades (`wfo.py:579-580`), which the NDH gate
  uses only as `> 0` (`gates.py:92`). Its sign equals the sign of the window's total return, so **gate decisions cannot
  change**, but the stored expectancy values will. The current value is already definitionally wrong, because it includes trades before
  `train_start`.
- **The weight path is NOT equivalent.** `_sweep_fold_weights` makes plans only from `window_start`
  (`rebalance_dates(..., window_start, window_end)`), so it starts flat at each window. Slicing a full-history weight simulation would carry
  positions in. De-duplicate the weight path only within a fold (anchor = train), never across folds.

### B: vectorbt 1.x / Rust. 8 claims

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| B1 | Summary timing tables match raw outputs | VERIFIED | `small_calls_table.txt`, `portfolio_*_pass*.txt` match the report's cells |
| B2 | Bit-identical simulator outputs on 5 stacks (compat 8/8, fuzz 102/102) | VERIFIED from raw files | `compat_compare.txt`, `fuzz_compare.txt`. I did not re-run them. |
| B3 | Methodology: warm-up excluded, same inputs | VERIFIED | first call reported separately, 150-call median; seeded inputs; equal checksums across stacks. Shared box (load 3-10 on 4 cores) is disclosed; the stacks ran sequentially, not interleaved, so treat differences under 10% as noise, as B says. |
| B4 | "Fold-sized" benchmark = what ggTrader does | WRONG (label) | 280 x 150 columns x 3 groups is far smaller than ggTrader's real call (~1,000-1,800 bars x ~24,000 columns, 48 groups; ~2.7 s per call in the profile). B's own 4,800-column, 48-group case is closer and shows Rust *slower* on 1.0.0 (1.08 s vs 0.47 s), so the headline survives. |
| B5 | Headline: Rust gives ggTrader nothing | VERIFIED | above, plus the vbt share of wall time is about 11% unprofiled (section 6), so the ceiling is small even if Rust were faster |
| B6 | pandas 3: bare `pct_change()` no longer pads | VERIFIED | pandas 3.1.1 source `generic.py:11617` has `fill_method: None = None`; `[1,nan,2]` gives `[nan,nan,nan]` (run). Deprecated in the 2.1 notes (`fill_method` deprecated, GH 53491). The 3.0 notes cover it only obliquely (removal of `limit`, line 1192), so B's citation of the 3.0 notes is thin. The behaviour claim is right. |
| B7 | pandas 3: `agg(np.std)` becomes NumPy ddof=0 | VERIFIED | 3.0 notes `v3.0.0.rst:1165` ("will no longer replace NumPy functions"); probe gives 0.5 vs 0.7071 |
| B8 | 14 bare `pct_change()` in src/, 30 in scripts/; pandas-ta never imported | VERIFIED | my grep: 14 / 30; **44 calls in src+scripts lack an explicit `fill_method`** (the bare calls plus `pct_change(periods)` forms); pandas_ta imports = 0 |
| B9 | vectorbt 1.0.0 accepts pandas<3 | COULD-NOT-CHECK independently | I relied on B's working venv100. I did confirm that 0.28.5 declares `pandas<3.0,>=2.0`. |

B: **7 VERIFIED (B2 from raw files only), 1 WRONG (label), 1 COULD-NOT-CHECK.** **Trust: HIGH.** Its one error is the label on the fold-size benchmark, which does not change its conclusion.

### Profile (claim 6)

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| P1 | 65% PIT (`pit_entry_mask`/`eligible_at`) / 27% signals / 4% from_signals of 440 s | VERIFIED | `wfo.prof`: `eligible_at` 286.1 s (65.0%), `_generate_signals` 114.8 s (26.1%), `from_signals` 19.1 s (4.3%), `simulate_signals` 28.5 s (6.5%) |
| P2 | cProfile does not distort the split materially | PARTLY | Measured overhead (same tape, profiled vs plain): `eligible_at` **2.04x**, `_generate_signals` **1.45x**, `simulate_signals` **1.06x**. Estimated unprofiled split: PIT ~140 s (~55%), signals ~79 s (~31%), simulate ~27 s (~11%), other ~8 s; total ~254 s. The ranking holds; the PIT share is overstated by about 10 points. |
| P3 | The 2-fold profile represents a real 17-fold run | WRONG (implicit) | `eligible_at` is O(T^2), because it calls `past[sym].dropna()` on a growing past. Sweeps scale with folds (2 x 17 + 1 = 35 sweeps). In a 17-fold run the sweeps probably dominate. This is an estimate: about 1,250 s profiled for the PIT mask vs about 1,500-2,000 s for sweeps. |

**Overall scorecard (49 claims): 39 VERIFIED, 4 PARTLY, 5 WRONG, 1 COULD-NOT-CHECK.** All five WRONG items are minor
(two parameter/constructor counts, a benchmark label, an implicit framing, a test that could not fail). None changes a recommendation.

**Independence limitation.** I am Opus checking work by cheaper models (A, C) and by peers (B, D). The checks are not fully
independent:
1. I chose claims from the reports' own tables, so anything they never looked at I mostly did not look at either. The
   two findings C missed came from reading the code paths the crux required.
2. My causality test was designed after reading D's hypothesis, though on ggTrader's real code and data rather than D's toy.
3. I did not re-run B's benchmarks or fuzz suite, and I did not run vectorbt 1.x at all.
4. The overhead ratios come from 40 dates and one combo each.
5. The "decisions cannot change" argument for `n_trades` is analytic. It has not been run on a 17-fold reference.

---

## (b) Primitive-by-primitive verdict

REPLACE = switch to vectorbt. KEEP = keep the hand-written code. GLUE = keep a thin hand-written layer around a vectorbt call.
Lines are spans incl. docstrings; deletions are **estimates**.

| Primitive | Verdict | vectorbt API (0.28.5) | What shifts | Code deleted (est.) |
|---|---|---|---|---|
| Universe / PIT mask (`sweep.pit_entry_mask`, `data.eligible_at`) | KEEP, **rewrite vectorised** (no vbt feature) | none; vbt has no membership concept | nothing: 0 mismatches on 123 sampled dates; builds in 1.0 s vs ~140 s unprofiled | ~0 net; replaces an O(T^2) loop |
| Folds (`generate_folds`) | KEEP | `range_split(start_idxs, end_idxs)` needs integer positions and drops the index | n/a | 0 (vbt version is longer) |
| Grid (`build_grid`, `_is_valid_combo`) | KEEP | `param_product=True` covers only the product, not the validity rules | none | ~3 |
| SMA / rolling std | GLUE (optional) | `vbt.MA.run(w)`, `vbt.MSTD.run(w, ddof=1)` | ~1e-11 to 1e-9 float noise | small |
| Bollinger / %B | GLUE (optional) | `vbt.BBANDS.run(w, alpha=k, ddof=1)` | 1.2e-9; **default ddof=0 is off by 1.09** | ~15 |
| RSI (Wilder) | GLUE (optional) | `vbt.RSI.run(window=2n-1, ewm=True)` | values exact; first valid bar 27 instead of 14; zero-loss gives 100 vs NaN | ~20; dedup the 4 copies regardless |
| EMA | GLUE (optional) | `vbt.MA.run(w, ewm=True)` | values exact; NaN for the first w-1 bars where pandas emits values | ~5 |
| MACD | GLUE (optional) | `vbt.MACD.run(..., macd_ewm=True, signal_ewm=True)` | 0.12 early, about 1e-15 after 200 bars (different EWM seed) | ~15 |
| ATR (stops) | REPLACE (low value) | `vbt.ATR.run(h,l,c,w, ewm=False).atr` | 2e-13; first bar 13 vs 14 (vbt's first TR is H-L) | ~20 |
| Volume BB, divergence, strengths, raw scores | KEEP | none | n/a | 0 |
| Signal combination (N-of-M vote, RSI independent exit) | KEEP | none (`.vbt.signals.AND/OR` add nothing) | n/a | 0 |
| Time stop (`td_stop`) | KEEP | none in OSS (PRO only) | n/a | 0 |
| Trailing / ATR / TP stops | already vbt | `from_signals(sl_stop, sl_trail, tp_stop)` | n/a | 0 |
| Sizing (percent, conviction) | already vbt + GLUE | `size`, `size_type="percent"` | n/a | 0 |
| Vol targeting scalar | KEEP | `.vbt.returns(year_freq='252 days').rolling_annualized_volatility` covers only the rolling std (3.8e-15) | none | ~3, not worth it |
| Portfolio simulation | already vbt | `Portfolio.from_signals/from_orders` | n/a | 0 |
| Live position cap (`MAX_POSITIONS`) | KEEP | only `from_order_func` with a numba callback | n/a | 0 |
| Sharpe / ann. vol / max DD | REPLACE (optional) | `.vbt.returns(freq='D', year_freq='252 days').sharpe_ratio() / annualized_volatility() / max_drawdown()` plus an `inf -> NaN` guard | equal to about 10 digits; **flat curve gives inf (would win `idxmax`)** | ~10 |
| Sortino | KEEP (or owner decision) | `sortino_ratio()` uses a different definition | 1.17 vs 1.01; feeds the composite score at weight 0.3 and **can change fold winners** | 0 |
| CAGR | KEEP (or owner decision) | `annualized()` counts bars/252 instead of calendar days/365.25 | 15.51 vs 14.94 | 0 |
| DSR / PSR | KEEP | `deflated_sharpe_ratio` has wrong kurtosis in 0.28.5 and per-column trial semantics | 0.6236 vs 0.6022 on D's sample | 0 |
| Data loading | KEEP | `YFData` is tz-aware 05:00 UTC, multi-symbol fails, no cache/PIT | n/a | 0 |
| Persistence (TimescaleDB, JSON) | KEEP | vbt can only pickle | n/a | 0 |
| Gates / anchor / circuit breaker / shadow re-entry / composite | KEEP | none (OSS or PRO) | n/a | 0 |
| Paper trader (risk, sweep, splits, dividends, broker) | KEEP | none | n/a | 0 |
| Tax lots, A23, replica | KEEP | none (no tax/cash flows in OSS) | n/a | 0 |

Spans with any vbt equivalent (indicators 207, gates DSR 76, metrics 26, simulate 45, folds 17, grid 11, alloc vol 6) total
**388 lines = 1.8% of 21,827**. Realistic net deletion after compatibility wrappers and parity tests: **~100-200 lines (estimate)**.

---

## (c) Can ggTrader be "100% vectorbt plus glue"?

**No. Realistically ~2% of the code moves to vectorbt; the simulator (already vbt) is the only large piece it carries.**
Hard blockers:
- **PRO-only features:** calendar/purged splitters, the CV apply-and-select loop, time stops, dividends/cash flows, stitched
  OOS portfolios, parameterising arbitrary functions, chunking, and the agent tooling. These are excluded by the owner's no-subscription rule.
- **Behaviour differences:** 365-day year, ddof=0, SMA-based "RSI/MACD" defaults, +inf Sharpe, Sortino/CAGR definitions,
  DSR kurtosis. Each swap needs flags and a parity test, and two (Sortino, CAGR) change published numbers by definition.
- **No vbt concept exists for:** PIT membership, the NDH/DSR gates, anchor, breaker, tax lots, the live broker, persistence.
- **Live coupling:** any indicator change is a live-signal change (`signal_runner` -> `EnsembleSignal`).

What each option serves (time numbers are estimates for the 2-fold profile config, ~254 s unprofiled):

| Option | Speed | Upstream maintenance | Agent-friendliness |
|---|---|---|---|
| Pure refactors: vectorise PIT, de-duplicate sweeps, memoise indicators (no vbt) | **Large**: PIT ~140 s -> ~1 s (measured build 1.0 s); sweeps 7 -> 1-2; indicators ~240 -> ~8 computations per sweep (48 combos x 5 voters vs about 2 bb + 1 rsi + 2 ema + 1 macd + 2 vbb unique; estimate). Plausibly ~254 s -> ~30-40 s (estimate) | neutral (less code on the hot path) | better: simpler data flow |
| vbt indicator kernels | 1.7-7x on the rolling parts (B); marginal once memoised | trades ~200 owned lines for dependence on vbt defaults that changed in 1.1.1 / 1.2.0 (rolling std, ewm std, OBV) | worse unless wrapped: silent defaults are the classic agent trap |
| vbt returns accessor for Sharpe/vol/DD | negligible (`curve_stats` 0.6 s total) | small win | neutral if wrapped with explicit `year_freq` |
| Rust engine / vbt 1.2.0 | ~0; the 1.2.0 numba kernel is 2.6x slower per B | negative (pandas-3 port) | none (agent features are PRO) |

---

## (d) Ranked, phased plan with safeguards

**Stack:** stay on **vectorbt 0.28.5**, pinned. 1.0.0 adds nothing ggTrader uses. 1.2.0 forces pandas 3, a large
silent-change surface, and a slower numba kernel. B, D and my source read agree.

**Phase 0: safeguards before any code change (no result change).**
1. **Reference oracle:** a pinned-window WFO artefact (JSON) on the current code: per-fold winner, anchor, IS/OOS Sharpe,
   NDH density/variance, DSR, WFE, the spliced OOS curve, and n_trades. Run it twice, a 2-fold config (as `prof_wfo.py`) and the full
   17-fold `--eval-start 2021-01-31 --eval-end 2026-04-30`. Record package versions next to it.
2. **Live golden:** freeze an OHLCV snapshot. Run `EnsembleSignal.to_targets` / `signal_runner` logic over the last ~60
   trading days with a 365-day lookback and store entries, exits and sizes. Any later change must reproduce this.
3. **Pins:** `pandas` and `numpy` are unbounded in `pyproject.toml`. Today only the transitive caps hold them (vbt 0.28.5 needs
   `pandas<3`; pandas-ta pins numba 0.61.2, which caps numpy<2.3). Pin both explicitly. Delete the dead `pandas-ta` (0 imports).
4. **pandas-3 trap, cheap part now:** give all 44 `pct_change` sites in src/scripts an explicit choice. Use `fill_method=None`
   where the input has no NaN gaps; where it does, use `.ffill()` first to keep today's padding. Prove each with a unit test on 2.3.3,
   since the outputs must match today. The live path has 3 sites (`overlay.py` 2, `feature_gate.py` 1); `idio_vol` and `max_effect` run on gappy
   price panels. Add a test lane with `-W error::FutureWarning`. Defer the actual pandas-3 migration.

**Phase 1: pure speed-ups, no vbt feature. Gate: oracle bit-for-bit.**
- 1a. **Vectorise `pit_entry_mask`:** `close.notna().cumsum() >= min_history_bars`, ANDed with daily membership. Gate: mask
  identical on every date of the oracle run (I verified 123 sampled dates). Lab-only.
- 1b. **Anchor reuses the train sweep's simulation.** They are the same simulation, so every output is identical, `n_trades` included. Gate:
  oracle bit-for-bit. Applies to both signal and weight paths.
- 1c. **Memoise indicators inside `EnsembleSignal.sweep_signals`:** compute each unique indicator series once, then threshold
  and vote per combo. Same pandas calls, so bit-for-bit. **This touches the file the live trader imports.** Gate: oracle plus
  live golden identical, and deploy through `ggtrader-deploy`. It could instead live in a lab-only subclass to keep the live path untouched.
- 1d. **One full-history simulation per (strategy, stop group) for all folds, the live sweep and the test slices (signal path only).** Equity is exact
  (verified). `n_trades` must be computed per scoring window from `pf.trades` records (entry index inside the window), or it
  silently changes. **Changes stored expectancy values, so this needs an owner decision.** Gate:
  every fold decision identical on the 17-fold oracle, all equity/Sharpe bit-for-bit, expectancy recorded as re-defined.
  Memory: about 24k columns x 1,800 bars with dense ATR `sl_stop` arrays. D measured 3.1 GB on synthetic data; measure at real size under
  the 6 GB cap before adopting.
- Also fix independently: the `signals.py:75` silent fallback (log and re-raise in research).

**Phase 2: vbt-native swaps, optional. Gate: tolerance plus zero decision change.**
- 2a. A single `lab/vbt_compat.py` that holds every vbt call with explicit flags (`year_freq='252 days'`, `ddof=1`,
  `ewm=True` + `window=2n-1`, `macd_ewm=signal_ewm=True`, `ATR ewm=False`, inf->NaN). This is the agent-friendliness lever.
- 2b. Sharpe/vol/max DD through the returns accessor. Gate: |diff| < 1e-12 and identical winners on the oracle. Lab only.
- 2c. Indicator kernels. Gate: values within 1e-9 after warm-up, **0 entry/exit differences on the full SP500 tape after a
  200-bar burn-in, and 0 differences in the live golden.** If any signal flips, stop. This is the step that touches live trading and needs an owner decision.
- 2d. ATR stop through `vbt.ATR(ewm=False)`. The first-bar difference changes early stop levels, so gate it on the oracle.

**Phase 3: owner decisions only (re-baseline required).** Changing the Sortino or CAGR definition, the DSR source, or any move to pandas 3 /
vbt 1.2. Each produces new numbers that must be re-baselined against SPY on pinned windows. None is needed for the owner's goals.

**Do not:** adopt Rust, vbt splitters, `YFData`, or the vbt DSR in 0.28.5.

Result-changing steps that need an owner decision: **1d** (stored expectancy values only), **2c** (live signals if any flip),
and all of **Phase 3**. Everything else must reproduce exactly or it does not ship.

---

## (e) Three biggest open uncertainties

1. **The real split and memory of a 17-fold run.** My estimate says sweeps, not PIT, dominate at 17 folds, and the single
   full-history simulation's peak memory at about 24k columns with ATR arrays is unmeasured. *Cheapest experiment:* run `prof_wfo.py`
   with `--eval-end 2026-04-30` without cProfile (wall time plus `/usr/bin/time -v` peak RSS), then once more after 1a+1b.
2. **Whether vbt's 1e-11 indicator noise flips any real signal** at a threshold crossing (BB lower band, RSI 30/50) on
   production data. My synthetic test found 0 flips after burn-in, but that is not production evidence. *Cheapest experiment:* compute the
   ensemble's entries/exits on the full SP500 tape with pandas vs a vbt-kernel variant and count XOR cells after 200 bars.
   Lab-only, about 1 minute.
3. **Decision invariance of 1d on the full oracle.** The NDH sign argument for `n_trades` is analytic, and there is one more
   pre-existing divergence (sweep path masks after the time stop) that could interact with `td_stop` grids. *Cheapest
   experiment:* run the 17-fold oracle twice (current code vs 1b+1d) and diff the fold-table JSON. Separately, run a `td_stop` grid
   with the mask applied before vs after the time stop and count the changed exits.
