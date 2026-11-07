# vectorbt for a walk-forward-optimisation (WFO) lab: how far it carries ggTrader, and how agent-friendly it is

Date: 2026-11-05. Installed: vectorbt 0.28.5 (`/home/flynn/ggTrader/.venv/lib/python3.12/site-packages/vectorbt/`, pandas 2.3.3, numpy 2.2.6, Python 3.12.3). 1.x was inspected from the PyPI wheel `vectorbt-1.2.0` downloaded into the scratchpad (not installed; it needs numpy>=2.4.6 and pandas>=3.0.3, `v12/vectorbt-1.2.0.dist-info/METADATA`). All scripts are in `D-examples/` next to this file; every measured number below comes from a script output saved there. No project file was modified; no package was installed into the project venv.

Paths: `D-examples/` = `/tmp/claude-1000/-home-flynn-ggTrader/6c9c1710-29af-4adf-b77b-ad700b6fcc23/scratchpad/vbt-research/D-examples/`. ggTrader paths are relative to `/home/flynn/ggTrader`.

---

## 0. Bottom line

1. **vectorbt can carry the *simulation, the grid broadcasting and the per-fold metric reductions*. It cannot carry the WFO *policy* layer** (gates, composite score, anchor, circuit breaker, PIT membership, tax). ggTrader already uses vbt for the heavy part (one grouped `from_signals` call per stop-config group, `sweep.py:130-181`, `simulate.py:277-290`).
2. **Honest deletable code is small: roughly 100-200 of the ~3,000 lines I read (~5%), and each deletion changes numbers in warm-up/NaN/definition edge cases** (section 3). The biggest saving is *structural*, not a deletion: wfo.py re-simulates the same 48-combo grid on nested prefixes 3 times per fold, and I verified (max abs equity diff 0.0) that one full-history simulation sliced per fold is identical (section 1.5). That is a code *simplification* that needs no vbt feature.
3. vbt's splitters are **bar-count based, not calendar based**; for ggTrader's 12m/3m calendar folds you still hand-write the fold dates and feed them to `range_split` as integer positions, so `generate_folds` (17 lines, `wfo.py:194-210`) does not go away.
4. **Four silent traps** if swapping to vbt defaults: Sharpe annualises with **365** not 252 (`_settings.py:401`); Sharpe returns **+inf** when std==0 (a combo that never traded wins `idxmax`); `MSTD`/`BBANDS` default `ddof=0`; vbt RSI/ATR/MACD "ewm" is **span-based** and its warm-up differs (RSI needs `window=2n-1` to equal Wilder). 0.28.5's `deflated_sharpe_ratio` feeds *excess* kurtosis into a formula that expects non-excess (fixed in 1.1.1).
5. **Agents**: the open-source package ships *no* agent artefacts (no llms.txt, AGENTS.md, skill, MCP). The agent tooling (MCP server, `vbt agents install` skills, llms.txt) is **PRO-only**. Community skills/MCPs exist for the OSS package (section 5).

---

## 1. Walk-forward in vectorbt (open-source 0.28.5)

### 1.1 What exists (exact)

| Item | Where | What it does |
|---|---|---|
| `RangeSplitter`, `RollingSplitter`, `ExpandingSplitter` | `generic/splitters.py:108, 181, 228` | Yield integer position arrays; nothing else. No calendar logic, purge, embargo. |
| `Series/DataFrame.vbt.split(splitter, ...)` | `generic/accessors.py:1335` | Calls the splitter, then **stacks every split as columns** with a `split_idx` level; sets are `pd.concat`-ed along axis=1. |
| `.vbt.range_split / rolling_split / expanding_split` | `generic/accessors.py:1454, 1538, 1566` | Convenience wrappers over `split`. |
| `split_ranges_into_sets(start_idxs, end_idxs, set_lens, left_to_right)` | `generic/splitters.py:21` | Cuts each window into train/(valid)/test by length or fraction. |
| Official example | https://github.com/polakowo/vectorbt/blob/master/examples/WalkForwardOptimization.ipynb (listed in repo tree: `examples/WalkForwardOptimization.ipynb`) | Uses `rolling_split(n=30, window_len=365*2, set_lens=(180,), left_to_right=False)`. |

Behaviours that matter (verified by running `D-examples/d1_wfo_splitter.py`, output `d1_output_20sym.txt`):

* **Fixed bars, not months.** `rolling_split(window_len=315, set_lens=(252,), n=17)` returned train (252, S) / test (63, S) blocks. With `n=` it picks windows by `np.round(np.linspace(0, len-window, n))` (`splitters.py:218-222`), so **stride is not fixed**: measured strides 86/87 bars (not the 63 bars of a 3-month roll). Without `n`, stride is 1 bar (`start_idxs = np.arange(...)`, `splitters.py:206-207`). There is no `step`/`stride` parameter.
* **Calendar folds need `range_split(start_idxs=..., end_idxs=...)`** with *integer positions*. Passing a `pd.Index` of dates requires exact index membership (`find_first_occurrence`, `splitters.py:135-142`), and ggTrader's fold dates (e.g. 2021-01-31, a Sunday) are not trading days, so I converted with `index.searchsorted`. Windows of unequal length are **NaN-padded at the tail** (docstring example `accessors.py:1488-1498`; measured window lengths 385/386 bars in an array of 389).
* **Index is lost.** `split` does `.reset_index(drop=True)` (`accessors.py:1403, 1409`; same in 1.2.0 `accessors.py:1718, 1724`). Dates live only in the returned list of per-split `DatetimeIndex`es; you map position back to date yourself.
* ggTrader's own `generate_folds(2021-01-31 -> 2026-04-30)` gives **17 folds** (`wfo.py:194-210`; confirmed by run). `rolling_split(n=17)` also gives 17 but with different, non-calendar boundaries.
* Splitters are the only WFO feature: **no** purge/embargo (grep "purge|embargo" over 0.28.5 = no hits; 1.2.0 only mentions them in a docstring tip that points to PRO, `v12/vectorbt/generic/splitters.py:8-12`), **no** train/test apply-and-merge helper, **no** cross-validation decorator. vectorbt.dev itself says splitters "do not fit models or prevent leakage automatically" (https://vectorbt.dev/getting-started/features/). GitHub discussion #575 (https://github.com/polakowo/vectorbt/discussions/575) is a user asking how to assemble the stitched OOS portfolio after the example notebook: the stitching glue is left to the user.

### 1.2 Working example: 17 folds x 48-combo grid

`D-examples/d1_wfo_splitter.py` (synthetic mean-reverting prices, 4-voter ensemble written as a vbt `IndicatorFactory`, `param_product=True` over `min_agree [2,3,4] x min_agree_exit [1,2] x bb_alpha [2.0,2.5] x rsi_oversold [25,30] x ema_fast [10,20]` = 48 combos):

1. `ggTrader.generate_folds` -> 17 calendar folds.
2. `close.vbt.range_split(start_idxs, end_idxs)` where start = train_start minus 60 warm-up bars (via `searchsorted`), end = test_end.
3. `Ens.run(win, ..., param_product=True)` -> entries `(389, S*17*48)`; column levels `['ens_min_agree', 'ens_min_agree_exit', 'ens_bb_alpha', 'ens_rsi_oversold', 'ens_ema_fast', 'split_idx', None]`.
4. **One** `vbt.Portfolio.from_signals(win, entries, exits, size=0.03, size_type='percent', cash_sharing=True, group_by=[0,1,2,3,4,5])` -> **816 groups = 48 combos x 17 folds**, fresh `init_cash` per (combo, fold).
5. `pf.returns()` -> `(389, 816)`; per fold `rets.xs(k, level='split_idx')`, slice train/test rows by date, `.vbt.returns(freq='D', year_freq='252 days').sharpe_ratio()`; winner = `idxmax` of train Sharpe, read its test Sharpe.

**Answer to "single vectorised call across all combos and folds?": yes.** Both the indicator run and the portfolio simulation are single calls whose *columns* are the Cartesian product (combos x folds x symbols). Cash is shared per group via `group_by` + `cash_sharing=True` (a list of level positions is accepted: `base/column_grouper.py:36-46`). Gotcha: `group_by` level *names* must match exactly (the factory prefixes names with the lower-cased class name, `ens_...`); a wrong name makes vbt silently fall through to "group_by and index must have the same length" (`column_grouper.py:34-48`), which cost me one failed run.

### 1.3 Memory and speed (measured, 6 GB cgroup cap, `systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0`)

Stacked-fold layout (T=389 rows incl. 60 warm-up bars; 17 folds x 48 combos x S symbols; no stops; scalar size):

| symbols | columns | `Ens.run` | `from_signals` | peak RSS |
|---|---|---|---|---|
| 20 | 16,320 | 1.1 s | 4.1 s | 549 MB |
| 100 | 81,600 | 5.0 s | 6.9 s | 1,055 MB |
| 200 | 163,200 | 9.6 s | 11.0 s | 1,689 MB |
| **500** | **408,000** | **23.3 s** | **17.3 s** | **3,642 MB** |

(`d1_output_500sym_mem.txt`; earlier identical run showed 25.7 s / 19.2 s / 3,642 MB.) Fits the 6 GB cap, but note why it is as large as it is:

* Folds overlap by 12 of 15 months, so stacking copies the data ~3.9x: 389 x 408,000 = 158.7 M cells vs 1,695 x 24,000 = 40.7 M cells for one full-history layout.
* Every *dense* float64 array of that shape is 389 x 408,000 x 8 B = **1.27 GB** (arithmetic, not measured). The measured 3.6 GB survived because `size` is a scalar and signals are bool/int8. vbt broadcasts only `close` densely and keeps other args "raw" with flexible indexing (`portfolio/base.py:2885-2895`, comment "Only close is broadcast, others can remain unchanged"). A **per-column** `size`/`sl_stop` DataFrame (e.g. ATR stops that depend on a swept `atr_mult`, or a vol-target scalar per combo) would be a dense 1.27 GB each: expect ~+1.3 GB per such array (extrapolation, UNVERIFIED for your exact config).
* The better layout (section 1.5) is one full-history simulation: **24,000 columns, 48 groups, 3,062 MB peak, 8.6 s + 6.8 s + 3.2 s (indicators, simulate, returns) and 0.2 s for all 17 folds' slice-and-select** (`d5_output_500sym.txt`). The numbers are synthetic-data timings; ggTrader's real per-fold cost was not measured here.

### 1.4 What the stacked layout does and does not reproduce

* Each fold starts with **fresh cash and no open positions at the window start** (60 warm-up bars have `entries[:60]=False` in my demo). ggTrader instead simulates from the beginning of the loaded history up to `window_end` and scores the `[window_start, window_end]` slice (`wfo.py:272, 299`), i.e. positions and cash carry into the fold. These differ by construction; neither is "wrong", but the metrics will not match unless you choose the carry-over layout.
* NaN-padded tails (ragged calendar windows) are harmless to `from_signals` but must be excluded from the metric slices; I slice by date lists.

### 1.5 Finding that does not need any vbt feature: nested prefixes are re-simulated 3x per fold

`wfo.py` per fold: `compute_anchor_set(..., ohlcv.loc[:fold.train_end], ...)` (`wfo.py:656`), then train sweep via `_sweep_fold_dispatch` (`wfo.py:675`) whose `_sweep_fold` uses `ohlcv.loc[:window_end]` (`wfo.py:272`), then the test sweep (`wfo.py:757`) on `ohlcv.loc[:test_end]`. The anchor and train sweeps are the **same grid over the same prefix** (anchor scores the whole prefix, train scores `[train_start, train_end]`). Signals and vbt's order simulation are causal, so equity at bar t is independent of later bars.

`D-examples/d4_prefix_causality.py` checks this with vbt: 48-combo grid, 40 symbols, folds 1, 9, 17: prefix simulation to `train_end` vs the full-history simulation sliced, **max abs equity diff = 0.000e+00 and max abs Sharpe diff over 48 combos = 0.000e+00 for all three folds** (`d4_output.txt`), where the fold Sharpe is computed with ggTrader's own `curve_stats` on the rescaled slice exactly as `wfo.py:299-304` does. So one full-history simulation per (strategy, universe, stop-config group) can feed anchor, train and test scoring for all 17 folds. Caveats: holds for the signal-strategy path with a precomputed `entry_mask` (`wfo.py:632-640`); the weight-strategy path (`_sweep_fold_weights`) calls `select()` per date and was not tested; the demo has no PIT mask or ATR stops (stops are also causal, UNVERIFIED in this test).

---

## 2. Parameter sweeps: how parameters become columns (`D-examples/d2_param_sweeps.py`, output `d2_output.txt`)

| Mechanism | Result (run) |
|---|---|
| `Indicator.run(close, window=[10,20,50], ewm=[False,True], param_product=True)` | 6 combos x 4 symbols = 24 columns; levels `['ma_window','ma_ewm', None]`. Factory docs `indicators/factory.py:277` (param_product example), pipeline `run_pipeline(..., param_product=...)` `factory.py:1375, 1686-1688`. |
| Same without `param_product` | Params are **zipped**; lists of unequal length raise `ValueError: Parameters at index 1 have length 2 that cannot be broadcast to 3`. |
| `Indicator.run_combs(close, window=[5,10,20,50], r=2, short_names=['fast','slow'])` | All C(4,2)=6 unordered pairs of ONE list (fast<slow by construction), returns `r` indicator instances (`factory.py:2204, 2991-3060`). It cannot filter on arbitrary validity rules, so `_is_valid_combo` (`sweep.py:38-61`: ts_stop xor atr_mult, exit_z < entry_z, never-exits guard, fast<slow) has no vbt equivalent. |
| `Portfolio.from_signals(..., sl_stop=pd.Index([0.02,0.05,0.10], name='sl_stop'))` | **Fails**: the Index is a shape `(3,)` array and `ValueError: shape mismatch ... (500, 4) and ... (3,)`. Portfolio arguments in the OSS library are *broadcast*, they do not *expand* into a product. |
| Manual product: `entries.vbt.tile(3, keys=sl)` plus `sl_stop` as a `(1, n_cols)` row | Works; levels `['sl_stop','rsi_window',None]`, 12 columns (`base/accessors.py:283` `tile`). |
| Idiomatic: put the stop in the IndicatorFactory as an *output* (`output_names=['entries','exits','sl_stop']`) and pass `sl_stop=ind.sl_stop` | Works; the swept params appear as column levels for free; 6 groups from `rsi_lo=[25,30] x sl=[.02,.05,.10]`. This is how a stop/vol-target grid is added without hand-built tiling. |
| Selecting a winner | `pf.sharpe_ratio()` returns a Series indexed by the group levels; `sh.replace([inf,-inf], nan).idxmax()` gives the params tuple; `pf.stats(column=best, group_by=[0,1])` gives a full 22-row summary (Total Return, Max Drawdown, Win Rate, Sharpe, Calmar, Omega, Sortino ...; `portfolio/base.py:4581`). `pf.sharpe_ratio(year_freq='252 days')` changes -0.938 to -0.780 for the first test group (ratio 1.2026 = sqrt(365/252)). |

For a swept `atr_mult` the stop array would be per-combo (`time x symbol x combo`): in vbt this is simply the IndicatorFactory-output route above; ggTrader instead loops over stop-config groups (`sweep.py:88-98 group_by_stop_config`, one `from_signals` call per group). vbt could fold those groups into one call at the memory cost noted in 1.3.

---

## 3. Primitive-by-primitive: vbt equivalent and numeric agreement (`D-examples/d3_primitive_equivalence.py`, output `d3_output.txt`, `d3_results.csv`)

Test data: 6 synthetic symbols x 1,200 bars (seed 7). ggTrader functions imported read-only from `src/ggTrader/lab/strategies/indicators.py`, `lab/metrics.py`, `lab/gates.py`, `lab/simulate.py`. "Warm-up" = leading all-NaN bars (ggTrader / vbt). "max abs diff" is on cells valid in both frames unless stated.

| Primitive | vbt 0.28.5 equivalent | Direct equivalent? | Numerically equal? (measured) |
|---|---|---|---|
| SMA / rolling mean | `vbt.MA.run(close, w)` (`indicators/basic.py:76`) | Yes | **Yes**: max abs diff 4.576e-12; warm-up 19/19 |
| Rolling std (pandas ddof=1) | `vbt.MSTD.run(close, w, ddof=1)` (`basic.py:166`, default `ddof=0`) | Yes, with flag | Default **No**: max abs diff 4.439e-01. With `ddof=1`: 3.797e-10 |
| Bollinger lower band / %b | `vbt.BBANDS.run(close, window=20, alpha=2.0, ddof=1)` (`basic.py:236`; `.percent_b`) | Yes, with flag | Default `ddof=0`: max abs diff 8.879e-01 (corr 0.99999906). With `ddof=1`: lower 7.635e-10, `-%b` vs `bb_raw` 2.007e-10; warm-up 19/19 |
| EMA (`ewm(span, adjust=False)`) | `vbt.MA.run(close, w, ewm=True)` | Yes | Values **exact** (max abs diff 0.0 on jointly valid cells) but vbt is NaN for the first `w-1` bars (19 bars x 6 = 114 cells) while pandas emits from bar 0 |
| RSI (ggTrader: Wilder, `ewm(alpha=1/n, min_periods=n, adjust=False)`) | `vbt.RSI.run(close, window, ewm=...)` (`basic.py:363`, `nb.py rsi_*`) | Only via a trick | vbt default (`ewm=False`, SMA/Cutler) **No**: max abs diff 30.99, corr 0.889. `ewm=True, window=14` (span 14) **No**: 21.01, corr 0.9456. `ewm=True, window=27` (span 2n-1 == alpha 1/14) **Yes**: max abs diff 0.0, but warm-up 27 vs 14, so 78 cells differ for 13 bars/symbol. Entry signals (RSI14<30 crossing): ggTrader 98 vs vbt 96, difference 2 (both in burn-in), **0 after bar 150**. Zero-loss edge: monotonic series gives ggTrader `NaN` (`avg_loss.replace(0, nan)`), vbt `100.0` |
| MACD histogram (12,26,9) | `vbt.MACD.run(close, 12,26,9, macd_ewm=True, signal_ewm=True).hist` (`basic.py:575`) | Only with both flags | Default (`macd_ewm=False`, i.e. SMA MACD) **No**: max abs diff 5.482, corr 0.787. With both EWM flags: max abs diff 1.095e-01 over all valid cells, **2.647e-04 after 60 bars, 0.0 after 200**; warm-up 33 vs 0. Cause: vbt's signal EWM starts at the first non-NaN MACD value (bar ~25) while pandas starts at bar 0, a different initial condition that decays exponentially. ggTrader masks `slow+signal+divergence_window` bars (`indicators.py:203-205`) so only part of the gap is hidden |
| ATR (ggTrader: SMA of true range in `compute_atr_stop`, `simulate.py:123-151`) | `vbt.ATR.run(high, low, close, w, ewm=False).atr` (`basic.py:689`); default `ewm=True` | Yes, with `ewm=False` | `ewm=False`: max abs diff 1.159e-13. Default `ewm=True`: 1.232e+00 (corr 0.989). Warm-up 14 vs 13: vbt's first true range is `H-L` (no prev close) so it starts one bar earlier (6 cells). Wilder ATR (alpha 1/14) = `vbt.ATR(window=27, ewm=True)`: max abs diff 3.401e-05 after 150-bar burn-in |
| Stochastic %K/%D (not in current ggTrader: removed in commit 82931a40; reference = textbook rolling min/max) | `vbt.STOCH.run(high, low, close, k_window, d_window)` (`basic.py:461`) | Yes | %K max abs diff 0.0; %D (SMA) 3.524e-12; warm-up 13/13, 15/15 |
| OBV (not in current ggTrader; reference = cumsum(sign(diff)*volume)) | `vbt.OBV.run(close, volume)` (`basic.py:776`, `nb.obv_custom_nb`) | Partly | Without ties: constant offset = first-bar volume (raw max abs diff 8.972e+04, **0.0 after subtracting `volume.iloc[0]`**). With tied closes vbt counts an unchanged close as **+volume** (`close < prev` is False): 1,919 tied bars in my test, max abs diff 3.336e+07 vs textbook; the "tie=+volume" rule reproduces vbt exactly (0.0). Release notes list "OBV on unchanged closes" as fixed in 1.2.0 (https://github.com/polakowo/vectorbt/releases) |
| Keltner channel (not in current ggTrader, removed with the legacy engine) | **None built in** (`dir(vbt)` has no KELTNER/KC) | No (compose `MA(ewm)+ATR`) | n/a |
| Rolling realised vol (vol-target scalar, `simulate.py:104-120`) | `series.vbt.returns(freq='D', year_freq='252 days').rolling_annualized_volatility(window, minp)` | Yes for the vol; the scalar/clip/shift/size wiring stays hand-written | max abs diff 3.775e-15 |
| Sharpe | `ret.vbt.returns(freq='D', year_freq='252 days').sharpe_ratio()` (`returns/accessors.py:445`, `nb.py sharpe_ratio_1d_nb`) | Yes, with `year_freq='252 days'` | **Yes**: 0.9285248657 vs 0.9285248657. Default year_freq `'365 days'` (`_settings.py:401`) gives 1.1174798488 (x sqrt(365/252)). Zero variance: ggTrader `NaN`, vbt `inf` (`nb.py:sharpe_ratio_1d_nb`) |
| Sortino | `.sortino_ratio()` (`accessors.py:541`) | **Different definition** | **No**: ggTrader 1.5241396507 = mean/std(negative returns only); vbt 1.3876899651 = mean/sqrt(mean(min(r,0)^2)) x sqrt(ann). Swapping changes every published Sortino and the composite score (`wfo.py:222-248`, weight 0.3) |
| Max drawdown | `.max_drawdown()` (`accessors.py:845`) | Yes | **Yes**: -16.5403780426% both |
| Annualised vol | `.annualized_volatility()` | Yes | **Yes**: 17.10510773% both |
| CAGR | `.annualized()` (`accessors.py:341`, bar-count/252 based) | **Different definition** | **No**: ggTrader 16.123176% (calendar days/365.25) vs vbt 15.509146% (800 bars/252 years) on the same curve |
| Calmar | `.calmar_ratio()` (`accessors.py:393`) | vbt-only (ggTrader `curve_stats` has no Calmar despite the module docstring in CLAUDE.md) | n/a (vbt 0.937654 on test curve) |
| Deflated Sharpe | `.deflated_sharpe_ratio()` (`accessors.py:477`), `returns/metrics.py:18-26` | **Different semantics** | 48 random columns, best column SR 2.585: ggTrader `dsr_check` 0.623630 vs vbt 0.602228. vbt deflates across the **columns of the frame** (`nb_trials=n_cols`, `var_sharpe=np.var(sharpe across columns)`); ggTrader uses `n_trials=len(grid)` with null SE `1/sqrt(T-1)`. Feeding vbt's raw function `var_sharpe=1/(T-1)` and non-excess kurtosis reproduces ggTrader **exactly** (0.6236297914 vs 0.6236297914, diff 0.0). 0.28.5 passes scipy's *excess* kurtosis into a formula that needs non-excess (`accessors.py:510` vs 1.2.0 `accessors.py:696` `fisher=False`; release notes for 1.1.1 say results change); effect on this sample 0.60222791 -> 0.60125239 |
| Trailing stop / take-profit | `from_signals(sl_stop=, sl_trail=True, tp_stop=)` | Yes, **ggTrader already uses it** (`simulate.py:204-245`) | n/a |
| Time stop (`td_stop`) | **None** (`td_stop` not an argument of `Portfolio.from_signals`; grep "td_stop|time_stop" over 0.28.5 = no hits) | No | ggTrader's glue `_apply_time_stop` (`ensemble.py:211-221`) is the standard workaround; PRO lists "time stops" (see 6) |
| Benchmark comparison | `pf.stats()` shows "Benchmark Return [%]"; returns accessor has `information_ratio`, `beta`, `alpha` given `benchmark_rets` (`accessors.py:567, 601, 629`) | Partly | ggTrader's `metrics.benchmark` (monthly hit rate vs SPY, `metrics.py:41-56`) is not provided; the SPY curve rebase is trivial pandas anyway |
| Data loading | `vbt.YFData.download` (`data/custom.py:177`), `vbt.AlpacaData` (`custom.py:765`) | Partly | `YFData.download('SPY')` and `('AAPL')` worked singly: index tz-aware **UTC stamped 05:00** (NY midnight), columns incl. `Dividends`, `Stock Splits`, and `Capital Gains` for ETFs. `YFData.download(['SPY','AAPL'])` **failed**: `ValueError: Symbols have mismatching columns` (vbt 0.28.5 + yfinance 1.2.0). AlpacaData not run (no keys): UNVERIFIED. Neither does what `CachedYFinanceLoader` + TimescaleDB do (naive-UTC midnight stamps, caching, PIT, split/dividend checks); the CLAUDE.md warning about tz-aware writes into the naive `ohlcv.timestamp` column applies directly to vbt's 05:00-UTC index |

Net for the ensemble: RSI, BB, EMA, MACD and (hand-rolled) SMA/ATR all have vbt kernels, **but each is exact only with non-default flags and a different warm-up**. Because `paper/signal_runner.py:11,86-95` imports the same `EnsembleSignal`, any swap changes *live* signals in warm-up and NaN edge cases (new index members, post-gap bars). That is a deployment-risk item, not a research one, and it should be gated by a signal-parity test, not by trust.

Also noted while reading: `indicators.py` computes the Wilder RSI **four separate times** (`rsi_signals` :118, `rsi_strength` :161, `_weekly_rsi` :281, `rsi_raw` :342) and MACD three times; consolidating those is a plain refactor worth the same lines as any vbt swap and carries no numeric risk.

### 3.1 Honest line-count view (estimates, labelled as such)

| Hand-written piece | Lines (read) | Replaceable by vbt? | Estimated net deletion |
|---|---|---|---|
| `generate_folds` (`wfo.py:194-210`) | 17 | No: calendar months, tz, Sunday dates need the same glue around `range_split` | ~0 or negative |
| `build_grid` product (`sweep.py:64-75`) | ~12 | `param_product=True` replaces `itertools.product` only; validity filter (`sweep.py:38-61`) stays | ~3 |
| Indicator bodies (`indicators.py`) | 391 total | RSI/MACD/BB/ATR kernels exist but need flags + warm-up compensation; edge cases differ | ~60-100 after a risky swap; ~same via internal dedup |
| `compute_atr_stop` (`simulate.py:123-151`) | ~29 | `vbt.ATR(ewm=False).atr` is equal to 1.2e-13 | ~20 |
| `compute_vol_scalar` rolling std | 22 | `rolling_annualized_volatility` equal to 3.8e-15, wiring stays | ~3 |
| `curve_stats` (`metrics.py:13-38`) | 26 | Sharpe/MaxDD/AnnVol equal; **Sortino and CAGR differ in definition** | 0 unless the owner accepts re-defining the headline metrics |
| DSR (`gates.py:122-215`) | ~95 | Core formula exists in `returns/metrics.py` and reproduces ggTrader exactly with the right inputs, but ggTrader's `psr`/`expected_max_sr` wrap it with the grid-size trials | ~0-30 |
| NDH gate, `composite_score`, circuit breaker, shadow re-entry, anchor, live-param pick | `gates.py:12-120`, `wfo.py:44-165, 213-248, 465-523, 894-975` | **No vbt equivalent** | 0 |
| `simulate_weights/simulate_signals` glue | ~300 | Already vbt; glue is column stacking, PIT masks, conviction sizes | 0 (already thin) |

Estimated total: **~100-200 lines (about 5% of the ~3,000 read)**, versus the structural simplification in 1.5 (one simulation instead of ~3 per fold). The research value of "thin glue" is therefore in the *data flow*, not the primitives.

---

## 4. What vectorbt does not cover for a WFO lab

Verified by grepping 0.28.5 (`grep -rIli` over the package) and, where noted, reading 1.2.0 and the PRO documentation index.

| Need | Status in OSS 0.28.5 |
|---|---|
| Point-in-time index membership / constituent masks / survivorship | **Truly absent** (no "constituent", "universe" except a prose comment at `portfolio/nb.py:2568`, "survivor", "point-in-time"). Handle with a boolean mask ANDed into entries (what `pit_entry_mask`, `sweep.py:101-127`, does); the mask is just a DataFrame to vbt. PRO's docs list "Realistic backtests: avoid look-ahead bias, impossible fills, and survivorship bias" as a *guide page*, not a membership dataset (https://vectorbt.pro/llms.txt). |
| Gate logic (NDH plateau test, DSR with grid-size trials), composite score, anchor set, circuit breaker with shadow re-entry, `WfoPolicy` | **Absent**. Nothing like it in OSS or PRO's published feature index (PRO has "deflated Sharpe" and "permutation tests", `vectorbt.pro/llms.txt`, not these policies). |
| Tax lots, wash-sale, after-tax returns | **Absent** (grep "tax|wash": no hits; the "lot" hits are English words/variables). Not found in PRO's feature index either. |
| Dividends / cash flows in the portfolio | **Absent in OSS** (no cash-flow input to `from_signals`; `data/custom.py` mentions dividends only as fetched columns). PRO: "cash deposits, withdrawals, and dividends" (https://vectorbt.pro/getting-started/vectorbt-pro-vs-vectorbt). |
| Corporate-action (split) correctness | **Not found** beyond what yfinance returns; ggTrader's `paper/split_check.py` stays. |
| Time-based exit, purge/embargo | Absent in OSS (see 3, 1.1). |
| Position-count cap across symbols (live `MAX_POSITIONS=3`) | **Not found** as a parameter; needs `from_order_func` with a Numba callback (`portfolio/base.py:3145`) or pre-thinning the signals. |
| Multi-sleeve blend with inverse-vol / target-vol overlay (`lab/blend.py`) | **Absent**; trivial pandas on top of per-sleeve value series. |
| Persistence of runs to TimescaleDB, run lineage, JSON artefacts | **Absent** (vbt can pickle objects only). |
| Stitched OOS portfolio from folds | **No helper in OSS** (discussion #575); PRO documents `vbt.PF.row_stack` and continuing a portfolio across windows (https://vectorbt.pro/zh/features/optimization/time-series-cross-validation/ search snippet). |
| Reproducible window pinning | n/a; remains the caller's job (CLAUDE.md research rules). |

---

## 5. AI-agent angle

### 5.1 Official, open-source package: NONE FOUND (as of 2026-11-05)

* Repository tree (342 paths, `vbt-research/tree.txt`, from `gh api repos/polakowo/vectorbt/git/trees/master?recursive=1`): grep for `agent|llm|claude|cursor|mcp|skill|copilot|gemini` returns nothing. No `AGENTS.md`, `CLAUDE.md`, `.cursor`, skills or MCP code. `https://raw.githubusercontent.com/polakowo/vectorbt/master/AGENTS.md` and `CLAUDE.md` return HTTP 404; `https://vectorbt.dev/llms.txt` and `llms-full.txt` return HTTP 404.
* The only agent mentions are marketing/README prose: "Built for traders, researchers, and AI agents" and "AI agent-driven workflows" (`readme_master.md` lines 61, 142, from the master README), and a "Composable Python API" paragraph at https://vectorbt.dev/getting-started/features/: "usable from a notebook, a batch script, a test, or a Python tool called by an AI agent ... VectorBT does not require an agent framework".
* Release cadence is active: v1.0.0 2026-04-22, v1.1.0 2026-07-05, v1.1.1 2026-09-26, v1.1.2 2026-10-09, v1.2.0 2026-10-19 (`gh api repos/polakowo/vectorbt/releases`); repo pushed 2026-11-03, ~9.7k stars, license reported as NOASSERTION (Apache 2.0 with Commons Clause).

### 5.2 PRO-only agent features (note only)

From https://vectorbt.pro/features/ai/mcp-server.md and https://vectorbt.pro/features/ai/ai-research-assistant.md and the homepage: an MCP server (`vbt mcp config claude-code`; tools `search`, `find`, `resolve_refnames`, `get_page`, `get_attrs`, `get_source`, `list_skills`, `get_skill`, `run_code`, `get_environment`), `vbt agents install` that writes version-matched skills (the homepage lists `vbt-analysis, vbt-backtesting, vbt-basics, vbt-callbacks, vbt-cross-validation, vbt-custom-simulators, vbt-data, vbt-indicators, vbt-numba, vbt-optimization, ...` into `.claude/skills` and `.agents/skills`), `vbt.chat` (ChatVBT), and **members-only** `llms.txt` indexes / Markdown docs. I could still fetch `https://vectorbt.pro/llms.txt` (HTTP 200, 11,949 bytes; it is a public index of PRO docs pages, including "Machine-readable exports": `/llms-docs.txt`, `/auth.md`), but the page text says the full knowledge assets are for members. None of this applies to the open-source install.

### 5.3 Community artefacts for the OSS package (not endorsements)

| Artefact | Facts (GitHub API 2026-11-05) |
|---|---|
| https://github.com/marketcalls/vectorbt-backtesting-skills | Skills (setup, backtest, optimize, quick-stats, strategy-compare, a `vectorbt-expert` rules skill with templates incl. a walk-forward template) for Claude Code and 40+ agents; 220 stars, last push 2026-07-12; targets TA-Lib + OpenAlgo/yfinance/CCXT. |
| https://github.com/agiprolabs/claude-trading-skills (vectorbt skill) | MIT, 513 stars, last push 2026-09-03; its walk-forward section is a simple 70/30 split (search snippet). |
| https://github.com/wshobson/maverick-mcp | MIT, 719 stars, pushed 2026-11-02; FastMCP server using vectorbt for backtests (README: "Walk-Forward Optimization", "15+ built-in strategies"); I did not review its correctness. |
| https://github.com/singhvedant/vectorbt-ai-tool, https://github.com/Erikvirya/alpaca-mcp-quant | Small (0 stars) MCP/skill wrappers; the latter exposes `execute_vectorbt_strategy(symbol, strategy_code)` in a restricted sandbox and enforces a 1-bar signal lag. Low maturity. |
| Reddit/blog anecdote (r/algotrading-style snapshot, via WebSearch) | Generated backtests often run clean while hiding one-bar look-ahead; anecdotal, not a study. |

No reputable write-up measuring LLM hallucination rates for vectorbt specifically was found (searches 2026-11-05): **NONE FOUND**.

### 5.4 Verifiable agent-friendliness facts of 0.28.5 (measured with `ast` on the installed package)

* 87 files, 52,709 lines, 1,543 functions/methods; **1,359 public**, of which **1,245 (92%) have docstrings** and **1,307 (96%) are fully type-annotated** (using the `vectorbt._typing as tp` aliases). No `py.typed` marker (`os.path.exists(.../py.typed)` = False), so type checkers and agents that read installed signatures get the annotations only by inspection.
* API surface is wide: `Portfolio.from_signals` has **59 parameters** (`inspect.signature`), 127 public attributes on `Portfolio`, 100 top-level names in `vbt`. Docstrings in `generic/accessors.py` and `indicators/factory.py` carry runnable `pycon` examples (`splitters`/`range_split` examples at `accessors.py:1454-1536`), which helps.
* Practical friction for an agent, all observed in this session: accessor-based API (`.vbt.returns(...)`, `.vbt.tile`) is invisible to `grep` for method definitions; level/param naming is generated at runtime (`ens_min_agree`); wrong `group_by` names fail with an unrelated message (section 1.2); silent defaults (365-day year, `ddof=0`, SMA-based "ewm=False", `+inf` Sharpe) produce plausible but wrong numbers, which is the dangerous failure mode for an agent that checks "does it run".
* `import vectorbt` took 5.7 s here (numba-heavy), so agent loops that import in fresh subprocesses pay this per call.

---

## 6. vectorbt PRO vs open-source (note only; the owner refuses paid data/subscriptions; this is not a purchase recommendation)

From the official comparison page https://vectorbt.pro/getting-started/vectorbt-pro-vs-vectorbt (fetched 2026-11-05) and https://vectorbt.pro/llms.txt:

* **Cross-validation**: "The community edition's rolling and expanding splitters divide arrays into windows. PRO adds a full cross-validation workflow: define rolling, expanding, calendar-based, purged, or custom splits, run your research function on them, select parameters on each training set, and evaluate ... on its test set." (`vbt.Splitter.from_rolling/from_expanding/from_grouper`, `@vbt.cv_split`, purge/embargo, `PF.row_stack` for stitched OOS). This is the part that would remove `generate_folds` and the fold-loop glue.
* **Parameter sweeps over any function** (`@vbt.parameterized`, `vbt.Param`, chunking, resumable runs) vs OSS `param_product` limited to IndicatorFactory.
* **Simulation**: time stops, stop ladders, limit/stop-entry orders, cash deposits/withdrawals/dividends, continuing a portfolio across windows.
* **Performance/memory**: chunking over threads/processes/Ray with caching, which would address the 408,000-column memory shape in 1.3.
* **AI tooling**: MCP server, agent skills, llms.txt (5.2).
* **Source/support**: full source code, private docs, Discord. Distribution is license-gated (the OSS package is "the open-source community edition of VectorBT PRO", GitHub description).

### 6.1 What open-source 1.x adds over 0.28.5 (verified by diffing the 1.2.0 wheel and the release notes)

* Public API delta (def/class names, non-numba): `engine` argument/method on accessors, `Portfolio`, records; `ReturnsAccessor.ulcer_index`, `martin_ratio`, rolling variants, `resample_total_return`; `ohlcv_accessors` first/last price/volume; Rust-engine dispatch helpers (`_engine.py`, `*/dispatch.py`). **No new splitter, no purge, no new Portfolio constructor, no param product on Portfolio** (`generic/splitters.py` identical API; `diff` of class/def lists above).
* Optional Rust engine: `pip install "vectorbt[rust]"`, `engine='auto'|'numba'|'rust'`, "precompiled kernels for supported operations", and the docs state timings "do not establish a general speed advantage" and "Callback-based operations ... require Numba" (https://vectorbt.dev/getting-started/features/). Whether your `from_signals` call with `sl_stop`/`tp_stop`/`cash_sharing`/`group_by` is Rust-supported: UNVERIFIED.
* pandas 3 / numpy 2.4 / Python 3.14 support in 1.1.0; **1.x requires pandas>=3.0.3 and numpy>=2.4.6** (`v12/vectorbt-1.2.0.dist-info/METADATA`), so it cannot be dropped into the current venv (pandas 2.3.3). `pyproject.toml:39` pins `vectorbt>=0.28.5,<0.29` after the 2026-09 breakage; that pin is still right unless the whole stack moves to pandas 3.
* Numerical changes relevant to you: DSR kurtosis fix and NaN handling (1.1.1), EWM std ddof handling and OBV tie fix (1.2.0), "empty risk and index metrics now return NaN" (1.1.0), more stable rolling std (1.1.1), breakeven trades now counted in expectancy (1.2.0). Anything you cite from a vbt metric changes across the 0.28 to 1.x line: pin and record the version next to every number.

---

## 7. If the goal is "delete code, keep thin glue": ranked, with risk

1. **Structural, no vbt feature needed, biggest payoff**: simulate each (strategy, universe, stop-config group) once over the full history and slice per fold for anchor, train and test (verified exact, 1.5; layout and cost in 1.3). Removes ~2 of every 3 fold simulations.
2. **Metrics**: use vbt's returns accessor only with `year_freq='252 days'`, and guard `inf`/`NaN` (zero-variance Sharpe is `+inf` in vbt, `NaN` in `curve_stats`; with 20 synthetic symbols 16-20 of 48 combos per fold had `inf` train Sharpe and `idxmax` would have chosen them: `d1_output_20sym.txt` column `n_inf_train`). Do **not** swap Sortino/CAGR without owner sign-off, because the definitions differ (3).
3. **Indicators**: leave as is for production parity; if replaced, add a bit-parity test on warm-up and zero-loss edges using the recipes in section 3 (`RSI window=2n-1, ewm=True`; `MSTD/BBANDS ddof=1`; `MACD macd_ewm=signal_ewm=True`; `ATR ewm=False`). Dedupe the four RSI copies regardless.
4. **Keep hand-written**: fold dates, grid validity filter, gates, composite score, anchor, circuit breaker, PIT mask, tax code, weights-path (`select()` per date).
5. **Add column-product plumbing via IndicatorFactory outputs** (stops, vol targets) if you want a single call over stop configs (2), accepting the dense-array memory cost in 1.3.

---

## 8. Reproduction

Run from `/home/flynn/ggTrader` with the project venv (read-only use; `PYTHONPATH=src` for ggTrader imports):

```
systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 .venv/bin/python D-examples/d1_wfo_splitter.py --symbols 20
systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 .venv/bin/python D-examples/d1_wfo_splitter.py --symbols 500 --mem-only
systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 .venv/bin/python D-examples/d2_param_sweeps.py
systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 --setenv=PYTHONPATH=src .venv/bin/python D-examples/d3_primitive_equivalence.py
systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 --setenv=PYTHONPATH=src .venv/bin/python D-examples/d4_prefix_causality.py
systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 .venv/bin/python D-examples/d5_full_history_wfo.py --symbols 500
```

Outputs saved: `d1_output_20sym.txt`, `d1_output_500sym_mem.txt`, `d2_output.txt`, `d3_output.txt`, `d3_results.csv`, `d4_output.txt`, `d5_output_500sym.txt`.

Limits of this study: synthetic data only (no real ggTrader tape, no PIT mask, no ATR stops in the memory runs; strategy Sharpe values printed by the scripts are meaningless); Stochastic/Keltner/OBV are compared to textbook definitions because ggTrader no longer contains them; 1.2.0 was read, not executed; AlpacaData not executed; Rust engine not tested; PRO claims are from vendor pages and were not verified against PRO code.

## Sources (web)

* https://github.com/polakowo/vectorbt (README, tree, releases API)
* https://github.com/polakowo/vectorbt/releases
* https://vectorbt.dev/getting-started/features/
* https://vectorbt.pro/getting-started/vectorbt-pro-vs-vectorbt
* https://vectorbt.pro/llms.txt
* https://vectorbt.pro/features/ai/mcp-server.md
* https://vectorbt.pro/features/ai/ai-research-assistant.md
* https://vectorbt.pro/features/optimization/time-series-cross-validation/
* https://github.com/polakowo/vectorbt/blob/master/examples/WalkForwardOptimization.ipynb
* https://github.com/polakowo/vectorbt/discussions/575
* https://github.com/marketcalls/vectorbt-backtesting-skills
* https://github.com/agiprolabs/claude-trading-skills
* https://github.com/wshobson/maverick-mcp
* https://github.com/singhvedant/vectorbt-ai-tool
* https://github.com/Erikvirya/alpaca-mcp-quant
