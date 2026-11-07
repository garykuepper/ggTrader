# vectorbt 0.28.5 -> 1.x and the optional Rust engine: what is delivered, what it would cost

Date: 2026-11-05. Scope: ggTrader (`/home/flynn/ggTrader`, read-only for this work). Nothing in the repo or in `/home/flynn/ggTrader/.venv` was touched; all installs are in throw-away venvs under the scratchpad (listed in section 9).

Every claim carries a URL or `file:line`. Anything I did not verify is marked **UNVERIFIED**. "repo@v1.2.0" means a local `git clone` of https://github.com/polakowo/vectorbt checked out at tag `v1.2.0` (commit `a52a984`); line numbers refer to that tag.

---

## 0. Bottom line (read this first)

1. **Numerics: no change for what ggTrader calls.** `Portfolio.from_orders` (targetpercent, `call_seq='auto'`, `cash_sharing`, `group_by`), `Portfolio.from_signals` (`size_type='percent'`, trailing `sl_stop`, `tp_stop`), `pf.value()`, `pf.trades.count()` gave **bit-identical** outputs on five stacks (0.28.5; 1.0.0 numba; 1.0.0 Rust; 1.2.0 numba; 1.2.0 Rust). That holds for 5 large synthetic benchmark cases, 102 randomised configs (NaN holes, shorts, reversals, fees, slippage, `call_seq='auto'`), and ggTrader's own `lab/simulate.py` loaded unmodified from the repo (section 3.1).
2. **Rust is not a speed-up for ggTrader.** At WFO-fold size (280 days x 150 columns, 3 groups) a full `from_orders + .copy() + value() + trades.count()` call is 27-31 ms on 0.28.5/1.0.0 and 34-41 ms on 1.2.0; Rust is 6-7% slower than numba on 1.0.0 and 2-14% faster on 1.2.0 (section 3.2). The project's own published benchmark shows Rust at **0.70x-1.25x** of numba for `simulate_from_signals` / `simulate_from_orders` (repo@v1.2.0 `benchmarks/BENCHMARKS.md`, last three rows). With `from_*` at ~4% of wall time (owner's profile), the ceiling is ~4%; realistic gain is well under 1%.
3. **The one place Rust helps is a stack artefact.** vectorbt 1.0.0's numba kernel is 2.6x slower on numba 0.68.1/numpy 2.5.4 than on numba 0.61.2/numpy 2.2.6 (3.7 ms vs 9.8 ms per `simulate_from_orders` call, same vectorbt code). Rust runs the same kernel in ~4.5 ms regardless of numba. If you go to the pandas-3 stack, Rust partly repairs a regression you would otherwise have taken.
4. **You can get vectorbt 1.0.0 + Rust today on the current pandas 2.3.3 / numpy 2.2.6 / numba 0.61.2 stack.** 1.0.0 declares `pandas>=2.0,<3.0`, `numpy>=1.23`, `numba>=0.60` (repo `pyproject.toml@v1.0.0`); I installed and ran it. **1.1.0+ hard-requires `pandas>=3.0.3`, `numpy>=2.4.6`, `numba>=0.66`, `Python>=3.11`** (`pyproject.toml@v1.1.0`..`@v1.2.0`, confirmed from the installed 1.2.0 wheel metadata). 1.0.0 gives nothing you use that 0.28.5 lacks, so this is only a way to test Rust, not a reason to move.
5. **Moving to 1.2.0 is a full pandas-3 migration with one hard dependency conflict:** ggTrader declares `pandas-ta`, whose newest release (0.4.71b0, 2025-09-14) pins `numba==0.61.2`; numba 0.61.2 requires `numpy<2.3`. That cannot coexist with vectorbt 1.2.0. `pandas_ta` is **never imported** in `src/`, `scripts/` or `tests/` (grep), so deleting the dead pin removes the blocker. The rest of the cost is pandas 3 itself (section 6), which silently changes several numbers (default `pct_change`, `stack`, `agg(np.std)`, epoch units).
6. **vectorbt's built-in indicators are not what ggTrader needs.** ggTrader imports only `Portfolio.from_signals/from_orders/from_holding`; no `vbt.MA/RSI/...` and no `.vbt` accessor anywhere in `src`, `scripts`, `tests` (grep). Measured, vbt's rolling mean/std/RSI/BBANDS under numba are 1.7x-7x faster than the equivalent pandas calls (EWM is level with pandas), but Rust is slower than numba in 12 of 15 indicator rows at 1900x500, level in 2 and faster in 1 (RSI, 1.7x).
7. **AI-agent integration: none in the open-source package.** A README sentence says "Built for ... AI agents"; the MCP server, documentation chat and CLI exist only in the paid PRO edition (section 5).

Recommendation in one line: **stay on 0.28.5 for the live image and the WFO; if the owner wants Rust anyway, test it on 1.0.0 (same stack, `engine="rust"`, no other change); do not take 1.2.0 until/unless a pandas-3 port is wanted for its own reasons.** Details and a staged path in section 7.

---

## 1. Method and machine

- Machine: Intel Core i5-7500 @ 3.40 GHz, **4 cores / 4 threads**, AVX2, 31 GB RAM, Linux 6.8, Python 3.12.3 (`lscpu`). The box was shared with other jobs while I ran: load average was 3.0-10.5 on 4 cores. Absolute times are therefore inflated and noisy; I report best-of-N, repeated passes, and treat differences under ~10% as noise.
- Neither vectorbt numba kernels nor vectorbt-rust kernels use multiple cores (no `parallel=True`/`prange` in `vectorbt/`; no rayon/`std::thread` in `rust/src`, though the Rust code releases the GIL via `py.detach`, e.g. `rust/src/indicators.rs:63`). So core count does not enter the numbers.
- Heavy runs wrapped in `systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0`.
- Isolated venvs (all under `.../scratchpad/`): `venv0285` (vectorbt 0.28.5, numpy 2.2.6, pandas 2.3.3, numba 0.61.2, plotly 6.6.0 = ggTrader's pins), `venv100` (vectorbt 1.0.0 + vectorbt-rust 1.0.0 on the same old stack), `venv100b` (1.0.0 + numba 0.68.1/numpy 2.5.4, pandas 2.3.3; isolates the numba effect), `venv120` (vectorbt 1.2.0 + vectorbt-rust 1.2.0 resolved by pip: pandas 3.1.1, numpy 2.5.4, numba 0.68.1, llvmlite 0.50.0, plotly 7.1.0). Extra libs (sqlalchemy 2.0.48, pyarrow, yfinance 1.2.0) installed in `venv120` and `venv0285` only, for the pandas probe.
- ggTrader's installed versions for comparison (read from its venv metadata): vectorbt 0.28.5, numpy 2.2.6, pandas 2.3.3, numba 0.61.2, llvmlite 0.44.0, plotly 6.6.0, pyarrow 23.0.1, yfinance 1.2.0, SQLAlchemy 2.0.48.
- All scripts and raw outputs: `.../scratchpad/vbt-research/B-bench/` (section 9).

---

## 2. Task 1 - behaviour changes 0.28.5 -> 1.2.0

Sources: releases API https://api.github.com/repos/polakowo/vectorbt/releases (7 releases: v0.28.4, v0.28.5, v1.0.0, v1.1.0, v1.1.1, v1.1.2, v1.2.0); compare https://github.com/polakowo/vectorbt/compare/v0.28.5...v1.2.0 (cloned: 133 commits, 210 files, +99,150/-27,566 lines, most of it Rust, docs, tests and a black reformat; `vectorbt/` alone: 98 files, +20,777/-10,388). Release pages: https://github.com/polakowo/vectorbt/releases/tag/v1.0.0 , `.../v1.1.0`, `.../v1.1.1`, `.../v1.1.2`, `.../v1.2.0`.

### 2.1 Changes that can alter numbers or break an API ggTrader uses

| Release | Change (PR / commit) | Touches ggTrader? | Measured effect (mine) |
|---|---|---|---|
| v1.0.0 (2026-04-22) | Optional Rust engine, global `engine` setting `auto/numba/rust` plus per-call `engine=`; `dispatch.py` in every subpackage; soft dtype casting; FlexArray (release notes, v1.0.0) | **Yes, silently**: default is `engine="auto"` (`vectorbt/_settings.py:116`), so merely installing `vectorbt-rust` flips every supported call to Rust with no code change | Identical outputs; see 3.1 |
| v1.0.0 | New static no-callback simulator `simulate_from_signals_nb`; `Portfolio.from_signals` uses it unless a `signal_func_nb` / `adjust_sl_func_nb` / `adjust_tp_func_nb` is passed (`vectorbt/portfolio/base.py:3000-3006`) | **Yes**: ggTrader's `from_signals` takes the static path | Identical to 0.28.5 on 5 benchmark cases and 102 fuzz cases |
| v1.0.0 | `true_range_nb` single pass; `between_two_ranges_nb` O(n) (release notes) | No (indicator/signal helpers, unused) | not tested |
| v1.1.0 (2026-07-05) | Python 3.14, pandas 3, NumPy 2.4+, Numba 0.66+ support; **minimum versions raised**: `numpy>=2.4.6`, `pandas>=3.0.3,<4.0`, `numba>=0.66`, `Python>=3.11,<3.15` (`pyproject.toml@v1.1.0`; commit `4025314`) | **Blocking** for the current venv | `pip` resolved to pandas 3.1.1 / numpy 2.5.4 / numba 0.68.1 |
| v1.1.0 | `init_temp_records` flag for simulations, default `False` (`_settings.py:422`, PR-less commit `6008777`); `from_order_func` record buffers initialised (#860) | No | none |
| v1.1.0 | "Return NaN for empty risk/index metrics" (commit `275a7d3`); NaN-aware top/bottom-N (#857, #861) | Only if you compute risk metrics on empty input | not tested |
| v1.1.1 (2026-09-26) | Welford rolling std (#864); `expanding_mean/std` NaN when `minp` > length (#873); Plotly 7 template fix (Mapbox entries removed) | Rolling std only via vbt indicators (unused). The Plotly fix matters: ggTrader's `pyproject.toml` caps `plotly<7` because 0.28/1.0 crashed at import under plotly 7; 1.1.1+ removes that cause (release notes v1.1.1) | `rolling_std` max abs diff vs 0.28.5 = 1.96e-9 (rel 7.8e-10) |
| v1.1.1 | **Deflated Sharpe corrected** (non-excess kurtosis, ignore NaN) (#872): "Results change" | Only if `deflated_sharpe_ratio` is used (it is not) | measured 2.67e-4 abs / 0.21% rel difference |
| v1.1.2 (2026-10-09) | `is_sorted_nb` NaN parity (#868); NaN trade-plot markers (#869); Best/Worst Trade doc (#875); `cond_value_at_risk` ignores NaN (#879); keep intraday time converting datetime64 (#877) | No | none |
| v1.2.0 (2026-10-19) | **EWM std `ddof` handling fixed (#884)** | Only via `ewm_std` (unused) | `ewm_std(ddof=0)`: max abs diff 3.16, max rel 29%, NaN mask differs; `ddof=1` (default) identical |
| v1.2.0 | **Breakeven trades included in expectancy (#885)**; `profit_factor`/`expectancy` copy reductions before mutation (commit `9ea4daa`) | Only if `pf.trades.expectancy()` / `profit_factor()` are used. ggTrader's gate computes expectancy itself and calls only `trades.count()` (`src/ggTrader/lab/simulate.py:_add_trade_counts`) | not exercised |
| v1.2.0 | Selected-group membership preserved (#887); unsorted `unstack` labels preserved (#888); active drawdowns ending in NaN (#886); OBV on unchanged closes (#883); ulcer index / Martin ratio added (#881) | `#887` affects `pf[...]`/column selection on grouped portfolios (ggTrader does `.copy()` + `.value()`, no selection) | none seen |
| v0.28.5 (already pinned) | `gross_exposure` uses abs asset value for grouped portfolios (#837) | already live | n/a |

Not in release notes but in the diff: plotting refactor "backend -> renderer" (`0aefc0f`), backend-agnostic figure protocol (`9e89146`). ggTrader does not plot through vectorbt (UNVERIFIED for notebooks outside the repo).

Two pandas-3 shortcuts inside vectorbt itself: `GenericAccessor.ewm_mean(adjust=False)` returns `pandas.ewm(...).mean()` and never reaches vbt/Rust when `pd.__version__ >= 3` (`vectorbt/generic/accessors.py:454-455`), and `describe` does the same (`accessors.py:1190-1195`). So on pandas 3 an `engine=` argument is silently ignored for those calls (measured: 0 Rust kernel calls).

### 2.2 ggTrader API surface vs the changes

Used (grep of `src`, `scripts`, `tests`): `vbt.Portfolio.from_orders` (`lab/simulate.py`, scripts), `vbt.Portfolio.from_signals` (`lab/simulate.py`, `lab/strategies/signals.py:57`, scripts), `vbt.Portfolio.from_holding` (`scripts/leveraged_trend_research.py:43`), `pf.value()`, `pf.trades.count()`, `pf.copy()`. Arguments used in `lab/simulate.py`: `cash_sharing=True`, `group_by=pd.Index(...)`, `call_seq="auto"`, `size_type="targetpercent"/"percent"`, `sl_stop`, `sl_trail`, `tp_stop`, `freq`, `fees`, `slippage`, `init_cash`. **None of these arguments was renamed or removed between 0.28.5 and 1.2.0** (all ran unmodified, section 3.1).

---

## 3. Task 3 - hands-on measurements

### 3.1 Parity (the result that matters most)

| Check | Stacks compared | Result |
|---|---|---|
| 5 large benchmark cases (1900 days; 500 cols one shared-cash group; 4,800 cols = 48 groups x 100 symbols; trailing `sl_stop`; `size_type='percent'`; `from_orders` targetpercent + `call_seq='auto'`); `np.array_equal(pf.value(), ...)`, order and trade counts | 0.28.5 vs 1.0.0 numba, 1.0.0 Rust, 1.2.0 numba, 1.2.0 Rust, 1.2.0 `auto` | **All identical** (max abs diff 0.0; same orders/trades, e.g. 145,933 orders / 73,380 trades in the 48-group signal case) |
| 120 randomised configs (`fuzz_parity.py`): random shapes, NaN holes and leading NaNs, 1-4 groups, `cash_sharing`, fees, slippage, fixed fees, size types amount/value/percent/targetpercent, trailing SL, TP, longonly/shortonly/both, `accumulate`, `upon_opposite_entry`, alt `price`, `call_seq='auto'` | same 5 stacks | 102 configs run; **102/102 value arrays, trade counts and order counts identical** on every stack. 18 configs raised on all stacks (`from_signals` + `targetpercent` is unsupported everywhere). Only difference: the exception class. numba raises `ValueError`, Rust raises `RejectedOrderError` for the same message (`fuzz_raw.txt`) |
| ggTrader's own `src/ggTrader/lab/simulate.py` (`simulate_weights`, incl. the empty-strategy short-circuit, and `simulate_signals` with trailing stop, plain, and trailing+TP), loaded by path, synthetic data, 3+2 strategies | 0.28.5; 1.0.0 numba/Rust; 1.2.0 numba/Rust | Returns and equity arrays **identical** in all 8 outputs on all 5 stacks (`compat_compare.txt`). Includes `.copy()`, `pct_change(fill_method=None)`, `pd.concat`, MultiIndex columns, pandas-3 string column labels |

Caveats: synthetic random-walk prices, <= 4,800 columns, no corporate-action edge cases, no real PIT membership mask. It shows engine and version parity of the simulator, not parity of ggTrader's data pipeline under pandas 3 (section 6).

Rust dispatch was verified by wrapping every `*_rs` callable in `vectorbt_rust.{generic,indicators,labels,portfolio,records,returns,signals}` with a counter (`common.install_rust_counters`; vectorbt imports kernels lazily at call time, e.g. `vectorbt/portfolio/dispatch.py` `from vectorbt_rust.portfolio import simulate_from_signals_rs`). For `engine="rust"` and for `engine="auto"` the counter showed `portfolio.simulate_from_signals_rs` (or `simulate_from_orders_rs`) plus 11 more Rust portfolio/records kernels per portfolio (`value_rs`, `cash_grouped_rs`, `col_map_rs`, ...); with `engine="numba"` the counter stayed at 0.

### 3.2 Portfolio timings (steady state, seconds, best of 3-5, shared box)

Synthetic: 1,900 days; `sim` = the `Portfolio.from_*` call; `value` = first `pf.value()`; ranges are the spread across 1-3 passes (`portfolio_*.json`, `portfolio_*_pass*.json`).

| Case | 0.28.5 numba (sim / value) | 1.0.0 numba | 1.0.0 Rust | 1.2.0 numba | 1.2.0 Rust | 1.2.0 `auto` |
|---|---|---|---|---|---|---|
| signals, 1 group x 500 sym, trailing SL, 3% percent | 0.07-0.08 / 0.12-0.13 | 0.05 / (JIT) | 0.05 / 0.07 | 0.14-0.15 / 0.15-0.23 | 0.11 / 0.16-0.17 | 0.11 / 0.17 |
| signals, 1 group x 500 sym, no stops | 0.08 / 0.08-0.09 | 0.04 / 0.08 | 0.05 / 0.09 | 0.13-0.14 / 0.08-0.09 | 0.10-0.11 / 0.10-0.11 | 0.11 / 0.10 |
| signals, 48 groups x 100 sym (4,800 cols), trailing SL per column | 0.77-0.80 / 0.58-0.78 | 0.47 / 0.86 | 1.08 / 0.59 | 1.32-1.39 / 0.73-1.43 | 1.31-1.42 / 1.18-1.36 | 1.33 / 1.20 |
| orders targetpercent + `call_seq='auto'`, 48 groups x 100 sym | 1.11-1.39 / 0.64-1.04 | 1.15 / 0.57 | 1.44 / 0.50 | 2.31-2.51 / 0.75-1.00 | 1.57 / 1.15-1.21 | 1.44 / 1.15 |
| orders targetpercent + `call_seq='auto'`, 1 group x 500 sym | 0.33-0.36 / 0.08-0.13 | 0.35 / 0.09 | 0.28 / 0.06 | 0.47-0.50 / 0.07-0.11 | 0.31-0.34 / 0.10-0.12 | 0.33 / 0.11 |

Cold start (first call in a fresh process; numba compiles, Rust does not): 0.28.5 numba 4.1 s (first trailing-SL case); 1.0.0 numba 10.3 s; 1.2.0 numba 13.8 s on the first pass, 5.2-5.3 s on later passes (`value()` first call: 7.2 s, then 0.15-0.23 s, so part of the compile cost is cached on disk between runs; mechanism UNVERIFIED); Rust: 0.2-0.3 s in the same position.

**Fold-sized calls (what ggTrader actually does).** `bench_small_calls.py`: 280 days x 150 columns, 3 groups, 150 repeated calls of `from_*(...).copy()` + `value()` + `trades.count()`; median per call (ms), two passes where shown (`small_calls_table.txt`):

| Call | 0.28.5 | 1.0.0 numba | 1.0.0 Rust | 1.2.0 numba | 1.2.0 Rust |
|---|---|---|---|---|---|
| `from_orders` targetpercent, `call_seq='auto'` | 27.8 / 28.1 | 29.5 | 31.3 | 40.7 / 39.8 | 35.9 / 34.2 |
| `from_signals` trailing SL, 3% percent | 27.1 / 28.2 | 28.5 | 30.5 | 38.1 / 37.6 | 36.9 / 36.9 |

Reading: at this size the call is dominated by Python/pandas wrapping, not by the simulation kernel. Rust is 6-7% slower than numba on 1.0.0 and 2-14% faster on 1.2.0 (orders 12-14%, signals 2-3%); 1.2.0 numba is 33-46% slower per call than 0.28.5 (38% slower than 1.0.0 numba on the orders call).

**Why 1.2.0 numba is slower, isolated.** cProfile of `from_orders` (`profile_small_call_out.txt`), 60 calls: time inside `portfolio/dispatch.py:simulate_from_orders` (the numba kernel) = 3.7 ms/call on vectorbt 1.0.0 + numba 0.61.2 + numpy 2.2.6; **9.5 ms/call on 1.2.0 + numba 0.68.1; 9.8 ms/call on vectorbt 1.0.0 + numba 0.68.1 + numpy 2.5.4 (pandas 2.3.3)**; Rust kernel (`simulate_from_orders_rs`) 4.5 ms/call. `vectorbt/portfolio/nb.py` changed by only +89/-15 lines between v1.0.0 and v1.2.0 (`git diff --stat`), so the slowdown follows numba/llvmlite/numpy, not vectorbt code. (The root cause inside numba/LLVM is UNVERIFIED.)

### 3.3 Indicators: vbt numba vs vbt Rust vs pandas (1.2.0 venv, 1,900 days x 500 symbols; sweeps 100 symbols x 48 windows = 4,800 columns)

Times in ms, best of 5, two independent runs shown (`indicators_venv120_run1.json`, `..._run2.json`). Max abs diff is Rust vs numba / numba vs pandas on the same inputs. "Rust calls" = counted `*_rs` invocations (0 means Rust did not run).

| Case | numba ms | Rust ms | pandas ms | max abs diff Rust-numba | max abs diff numba-pandas |
|---|---|---|---|---|---|
| rolling mean, window 20 | 4.2 / 4.8 | 14.1 / 13.8 | 29.9 / 29.7 | 6.1e-11 | 6.1e-11 |
| rolling std, window 20, ddof 1 | 14.6 / 14.4 | 20.1 / 19.0 | 37.9 / 38.5 | 0 | 0 |
| EWM mean span 20, `adjust=False`, via `.vbt.ewm_mean` accessor | 15.4 / 15.9 | 15.4 / 15.7 | 15.3 / 15.3 | 0 | 0; **Rust calls = 0** (pandas-3 shortcut, `accessors.py:454`) |
| EWM mean span 20, `adjust=False`, via `generic.dispatch.ewm_mean` | 6.4 / 6.4 | 9.4 / 11.2 | 15.1 / 15.5 | 0 | 0 |
| EWM mean span 20, `adjust=True` | 7.7 / 7.5 | 11.6 / 11.4 | 15.7 / 15.6 | 0 | 0 |
| `MA.run` window 20 | 11.7 / 12.0 | 15.1 / 15.7 | 29.0 / 29.5 | 6.1e-11 | 6.1e-11 |
| `MA.run` ewm=True | 12.8 / 12.6 | 17.8 / 18.2 | 14.8 / 15.3 | 0 | 0 (vbt NaN-pads the first window-1 rows, pandas does not) |
| `MSTD.run` window 20 (vbt default ddof=0) | 19.0 / 19.8 | 23.4 / 23.3 | 39.0 / 38.0 | 0 | 0 vs pandas `ddof=0`; 4.05 vs `ddof=1` |
| `RSI.run` 14 (default `ewm=False`: simple-MA, not Wilder) | 51.6 / 50.0 | **30.1 / 30.0** | 86.3 / 89.7 | 7.2e-12 | 7.2e-12 |
| `BBANDS.run` 20, alpha 2 | 28.9 / 29.7 | 36.7 / 40.9 | 71.7 / 74.3 | 6.1e-11 | 6.1e-11 |
| `MACD.run` 12/26/9 (default `macd_ewm=False`, i.e. simple MA) | 21.6 / 24.5 | 30.3 / 32.3 | n/a | 4.3e-11 | n/a |
| `MACD.run` with `macd_ewm=True, signal_ewm=True` | 32.8 / 29.3 | 39.2 / 33.7 | 32.5 / 33.3 | 0 | 0 (NaN-pad difference as above) |
| 48-window sweep `MA.run` | 68.6 / 66.3 | 98.5 / 95.0 | 293 / 307 | 8.6e-11 | 8.7e-11 |
| 48-window sweep `RSI.run` | 219 / 240 | 216 / 211 | 901 / 938 | 1.3e-11 | 1.3e-11 |
| 48-window sweep `BBANDS.run` | 333 / 344 | 461 / 455 | n/a | 8.6e-11 | n/a |

Findings. (1) vbt numba beats the equivalent pandas `.rolling` by 1.7x-7x (rolling mean 7x, rolling std 2.6x, RSI 1.7x, `MA.run`/`BBANDS.run` 2.4x, 48-window sweeps 4-4.3x) and is level with pandas `.ewm` (which is already C). (2) Rust matches numba to ~1e-11 (floating-point order), not bit-for-bit, for rolling mean/RSI/MACD/BBANDS; bit-identical for std and EWM here. (3) Rust is **slower than numba** at this shape in 12 of 15 rows, level in 2 (the pandas-3 EWM shortcut, where Rust is never called, and the RSI sweep), faster in 1 (`RSI.run`); rolling mean is 3.3x slower. The project's own matrix is also mixed: `generic.rolling_mean` 0.30x-2.30x, `rolling_std` 0.42x-2.09x, `ewm_mean` 0.83x-1.09x, `indicators.ma` 0.50x-1.18x, `rsi_apply` 1.10x-3.53x (repo@v1.2.0 `benchmarks/BENCHMARKS.md`). I did not diagnose why my rolling-mean case is worse than their closest shapes (they have no 500-column column; memory layout of pandas frames is one candidate, **UNVERIFIED**). (4) vbt indicator defaults differ from textbook: `RSI` uses a simple MA unless `ewm=True`; `MACD` uses simple MAs unless `macd_ewm`/`signal_ewm=True` (signatures printed from the installed package).

### 3.4 How to verify that Rust actually ran

There is no public flag or log. Options that worked: (a) wrap the `*_rs` callables as in `common.install_rust_counters`; (b) check `vectorbt._engine.is_rust_available()` (`vectorbt/_engine.py:51`); (c) call with `engine="rust"`, which raises instead of falling back for most unsupported inputs (but see 4.3 for exceptions).

---

## 4. Task 2 - the Rust engine

### 4.1 What it is

- Separate wheel `vectorbt-rust` (PyO3 + `numpy` + `ndarray` + `rand`, `rust/Cargo.toml`; release profile LTO, codegen-units 1, opt-level 3, strip). Kernels are drop-in re-implementations of vectorbt's numba kernels; "Treat Numba as the reference" (`rust/README.md`, "New kernels"). About 15.6k lines of Rust (`wc -l rust/src/*.rs`).
- Coverage in the installed 1.2.0 wheel (counted by me): **245 `*_rs` functions**: generic 83, indicators 19, labels 13, portfolio 51, records 14, returns 43, signals 22. v1.0.0 release notes list: generic (rolling, expanding, EWM, groupby, drawdowns, ranges, apply), indicators (MA, RSI, Bollinger, ATR, MACD, Stochastic, OBV), labels, signals (random, stop, OHLC stop generation), portfolio (order-based and signal-based simulation, reservoir sampling, SL/TP), records, returns (https://github.com/polakowo/vectorbt/releases/tag/v1.0.0).
- Single-threaded. The Rust code releases the GIL but does not parallelise (no rayon; `grep` found `py.detach` only).
- License string on PyPI: `Apache-2.0 WITH Commons-Clause-1.0` (https://pypi.org/pypi/vectorbt-rust/1.2.0/json); the repo's `LICENSE.md:1` is the Commons Clause (no selling the software or a service whose value derives substantially from it). Internal research use is not "Sell", but the owner should read it before ever hosting a product on it.

### 4.2 The `engine` setting

- Global: `vbt.settings["engine"]`, default `"auto"` (`vectorbt/_settings.py:116`). Per call: `engine=` on `Portfolio.from_orders/from_signals`, indicator `.run`, accessors; accessors and `Portfolio` also keep an instance-level default (`Portfolio.engine`, `base.py:1589`). Resolver: `vectorbt/_engine.py:341` `resolve_engine`.
  - `"numba"`: always numba.
  - `"rust"`: raises `ImportError` if `vectorbt-rust` is missing or version-incompatible, raises `ValueError("... Use engine='numba'.")` if the call's inputs are unsupported.
  - `"auto"`: Rust when `is_rust_available()` and the call is supported; otherwise numba, silently.
- Randomised functions are special: `auto` stays on numba "to preserve legacy NumPy/Numba random streams" (`_engine.py:323-330`, `rust/README.md` "Usage"); a `from_*` with `reject_prob > 0` takes that route (`portfolio/dispatch.py:570`, `:793`). Explicit `engine="rust"` uses a different random stream.

### 4.3 What causes (silent) fallback to numba

Observed with `fallback_probe.py` on 1.2.0 (`fallback_probe_out.txt`); "kernel calls" = counted `simulate_from_*_rs` invocations:

| Situation | `engine="auto"` | `engine="rust"` |
|---|---|---|
| float64 close, standard args | Rust | Rust |
| float32 close | Rust (soft cast to float64) | Rust |
| int64 close | numba, silent | `ValueError: Rust engine requires close to be float64-compatible` |
| `call_seq='auto'` with cash sharing | Rust | Rust |
| `adjust_sl_func_nb` (numba callback) | numba, silent | **numba, silent, no error** (the callback branch bypasses the dispatcher, `base.py:3000-3006,3140+`) |
| `reject_prob=0.2` | numba | Rust (different random stream) |
| pandas-3 `str` column labels, read-only numpy inputs | Rust | Rust |
| Simulated `vectorbt-rust` 1.1.9 next to vectorbt 1.2.0 | numba + one `UserWarning` "version mismatch; falling back to numba" | `ImportError: vectorbt-rust is not installed...` (misleading text; the package is installed but minor-version-incompatible) |

Version coupling: the resolver compares `major.minor` only (`_engine.py:51-74`); the `[rust]` extra pins `vectorbt-rust==<same version>` (wheel metadata). A mismatched upgrade of one package therefore degrades performance silently (one warning) rather than failing. Callback functions (`signal_func_nb`, `adjust_*_func_nb`, `from_order_func`) are never Rust (`_engine.py:83-85` `callback_unsupported_with_rust`).

### 4.4 Platform wheels

PyPI `vectorbt-rust 1.2.0`: wheels for CPython 3.11-3.14 on `manylinux_2_17_x86_64` (917 KB for cp312), `manylinux_2_17_aarch64`, macOS x86_64 and arm64, Windows amd64, plus an sdist (https://pypi.org/pypi/vectorbt-rust/1.2.0/json). `Requires-Python >=3.11,<3.15`; `numpy>=2.4.6`. `vectorbt-rust 1.0.0` accepts Python >=3.10 and `numpy>=1.23` (installed metadata in `venv100`). Linux x86_64 + Python 3.12 installed and ran here without a compiler. No `-C target-cpu=native` or SIMD-specific flag is set in `rust/Cargo.toml`; I did not inspect the build flags in CI (**UNVERIFIED**).

### 4.5 Parity guarantees and tests

- Statement of intent: "Treat Numba as the reference" and "Add or update ... parity, fallback, explicit-error, and layout-sensitive tests" (`rust/README.md`). `tests/test_engine.py` is 2,674 lines with 102 `def test_` functions using `assert_array_equal` / `assert_allclose` between engines (e.g. `:227-285`, `:467`).
- CI: `.github/workflows/tests.yml` runs the whole suite on ubuntu/windows/macos x Python 3.11-3.14 twice: once numba-only, once with a locally built release Rust extension (`rust-engine` job). Tests only run on PRs and merges (commit `965e746`).
- Parity is **allclose, not bit-exact, for floating-point reductions** (their tests use `assert_allclose`; my measurement: 1e-11 differences for rolling mean/RSI/MACD/BBANDS). For the order/signal simulators I found exact equality in every case I ran (3.1).
- Release notes mention "Improved Rust engine packaging and parity coverage" (v1.1.0) and fixes like "Fix NaN-aware top/bottom N parity across engines" (#861), "Fix is_sorted_nb NaN parity" (#868), "Rust: aligned expanding mean/std across engines when minp exceeds rows" (v1.1.1): parity bugs were being found and fixed through 1.1.x, so Rust is a young code path.

### 4.6 Published benchmark numbers (quoted)

Source: repo@v1.2.0 `benchmarks/BENCHMARKS.md` (generated 2026-04-23, commit `b213e07`). Settings line: "Window: 20, NaN ratio: 5%, Repeat: 5, Seed: 42, Layout: view, Suite: core". Columns are rows x columns, from 100x1 up to 10Kx100 (and 100Kx10). **Hardware, OS, Python/NumPy/Rust versions are not stated in the file** (grep for CPU/hardware terms found none; their own `benchmarks/README.md` says to record them). "speedup = numba_s / rust_s".

- Overall: count 2,050 case-shape pairs; min 0.10x; **median 1.38x; mean 2.20x**; max 30.42x.
- Per shape medians range 1.22x-1.66x; per-shape minima 0.10x-0.46x (`Per-Config Statistics`).
- Rows relevant here (min-max across the 10 shapes): `portfolio.simulate_from_orders` 0.73x-1.15x; `portfolio.simulate_from_signals` 0.70x-1.23x; `simulate_from_signals_ls` 0.73x-1.25x; `portfolio.value` 0.98x-2.83x; `generic.rolling_mean` 0.30x-2.30x; `generic.rolling_std` 0.42x-2.09x; `generic.ewm_mean` 0.83x-1.09x; `indicators.rsi_apply` 1.10x-3.53x; `macd_apply` 0.77x-1.98x; `bb_apply` 1.20x-3.58x; `signals.generate_rand` 4.71x-30.42x (random signal generation is where the large wins are); `returns.rolling_*` 2.8x-11.4x.
- The headline average is carried by returns/rolling-metric, random-generation and records kernels. The simulator that ggTrader uses is at parity. My measurements are consistent with this (3.2).
- Their numbers are kernel micro-benchmarks, "best-of-repeat ... steady-state kernel cost rather than cold-start effects" (`benchmarks/README.md`). End-to-end Python overhead is outside them, which is why my fold-sized call (3.2) shows less than even their small-shape rows.

---

## 5. Task 4 - AI-agent / LLM / MCP material

Found (all cited):

- Marketing text only in the open-source package: README "Built for both human researchers and AI agents" (`README.md:61`) and "Combine building blocks for rapid experimentation and AI agent-driven workflows" (`README.md:131`); the same sentence is in the PyPI description (https://pypi.org/pypi/vectorbt/json).
- Community video links on the docs resources page, not product features: "Self-improving AI backtesting: OpenAlgo, DuckDB, Claude Code, and VectorBT in an automated research loop" and "OpenAlgo backtesting skills" (`docs/docs/getting-started/resources.md:~462-471`).
- **PRO edition only (paid, private repo)**: "Built-in documentation search and chat, a command-line interface, and an MCP server that lets compatible coding assistants inspect APIs and run code" (`docs/docs/getting-started/upgrade.md:41`, comparison row "Coding tools"); the community edition row says "Python APIs and public documentation".
- The PRO site publishes an `llms.txt` index (https://vectorbt.pro/llms.txt returned HTTP 200, `text/plain`, a navigation outline of PRO pages).

NONE FOUND for the open-source edition: no `llms.txt`/`llms-full.txt`/`AGENTS.md`/MCP manifest on https://vectorbt.dev (HTTP 404 for `/llms.txt`, `/llms-full.txt`, `/AGENTS.md`, `/.well-known/mcp.json`, `/robots.txt`; the sitemap has 102 URLs, none with agent/llm/mcp/skill in the path), and no files named llms/AGENTS/CLAUDE/cursor/mcp/skill in the repo (`git ls-files`).

---

## 6. Task 5 - migration risk: pandas 2.3.3 / numpy 2.2.6 / numba 0.61.2 -> pandas>=3.0.3 / numpy>=2.4.6 / numba>=0.66

### 6.1 Dependency feasibility

- vectorbt 1.2.0 runtime requirements (installed metadata): `numpy>=2.4.6`, `pandas<4.0,>=3.0.3`, `numba>=0.66`, `Python>=3.11,<3.15`, `plotly>=4.12.0`, scipy, scikit-learn, etc. `vectorbt-rust` extra pins `==1.2.0`.
- **Hard conflict: `pandas-ta`.** ggTrader's `pyproject.toml:22` lists `pandas-ta`; the only PyPI release line (0.4.67b0 2025-09-03, 0.4.71b0 2025-09-14, https://pypi.org/pypi/pandas-ta/json) requires `numba==0.61.2`, `numpy>=2.2.6`, `pandas>=2.3.2`, Python >=3.12. numba 0.61.2 itself requires `numpy<2.3,>=1.24` and `llvmlite<0.45` (ggTrader venv metadata). Unsatisfiable together with vectorbt 1.2.0. Mitigation: `pandas_ta` has **zero imports** in `src/`, `scripts/` and `tests/` (grep); only `pyproject.toml` and old docs mention it. Remove the declaration.
- Other declared dependencies and their pandas/numpy bounds (from the installed metadata): yfinance 1.2.0 `pandas>=1.3.0`; statsmodels 0.15.0 `pandas>=1.4`; alpaca-py 0.43.4 `pandas>=1.5.3`; pandas-market-calendars 5.5.0 `pandas>=1.1`; exchange_calendars 4.13.2 `numpy>=1.26.4, pandas>=1.5.0`; scikit-learn 1.8.0 `numpy>=1.24.1`; lightgbm, hmmlearn, optuna `numpy` unbounded; ta 0.11.0, mplfinance, seaborn, pytrends unbounded. Metadata allows pandas 3 for all of them, but **I only exercised yfinance** (below). Behavioural compatibility of the others under pandas 3 is **UNVERIFIED**.
- Docker: the live image is rebuilt from `pyproject.toml`; the 2026-09-23 incident (comments at `pyproject.toml:23-24` and `:38`) shows an unpinned rebuild already pulled vectorbt 1.0.0 and plotly 7 once. Any move needs exact pins.

### 6.2 What pandas 3 changes (sources: https://pandas.pydata.org/docs/whatsnew/v3.0.0.html released 2026-01-21; https://pandas.pydata.org/docs/whatsnew/v3.1.0.html released 2026-10-22; https://pandas.pydata.org/docs/user_guide/migration.html) and what I measured in `pandas3_probe.py` (same script, pandas 2.3.3 vs 3.1.1; `pandas3_probe_venv0285.txt`, `pandas3_probe_venv120.txt`)

**Silent numeric or shape changes (most dangerous):**

| Idiom | pandas 2.3.3 | pandas 3.1.1 | Note |
|---|---|---|---|
| `pct_change()` default with NaN gap `[1, nan, 2]` | `[nan, 0.0, 1.0]` (forward-fill, FutureWarning) | `[nan, nan, nan]` | Removal of default fill method. ggTrader's lab already passes `fill_method=None` (`simulate.py`), but grep finds `pct_change()` with no argument at 14 places in `src/` and 30 in `scripts/` |
| `groupby(...).agg(np.std)` | 0.7071 (pandas ddof=1, FutureWarning) | **0.5** (NumPy ddof=0) | "apply/agg/transform will no longer replace NumPy functions" (3.0 notes). None found by grep in ggTrader |
| `DataFrame.stack()` with NaN | drops NaN: 1 row | keeps NaN: 2 rows | `future_stack` default True (3.0 notes). grep: 2 `.stack(` uses in `src/` |
| datetime resolution | `datetime64[ns]` | `datetime64[us]`; `Timestamp('2024-01-02').unit` `s`->`us` | `idx.astype('int64')[0]`: 1704153600000000000 -> **1704153600000000** (1000x smaller). 3.1.0 changes numeric-input resolution again (to us) |
| `yf.download(...)` index | `datetime64[ns]` | `datetime64[s]` (!) | With yfinance 1.2.0 on live network data. Also emits `Pandas4Warning` for `inplace=` inside yfinance. Any `.astype('int64')`/`.view('i8')` epoch conversion changes units; ggTrader has a documented history of timestamp corruption on the `ohlcv` table (CLAUDE.md "ohlcv.timestamp is naive UTC") so every write path needs a unit-aware test |
| tz-aware `.values` | `datetime64[ns]` UTC | `datetime64[us]` + `Pandas4Warning` (deprecated, 3.1 notes) | Use `.dt.tz_convert(None).to_numpy()` |
| `groupby` on categorical key | unobserved category `z` present (value 0.0) | `z` absent | `observed=True` default. grep: `observed=` appears in 2 scripts only |

**Loud breakage (raises):** `Series[int]` is always a label lookup (`KeyError: 0` on a DatetimeIndex series and on an int index); `groupby(axis=1)` removed (TypeError); `resample('M')` and `freq='H'` raise (use `ME`, `h`); `fillna(method=...)`, `DataFrame.applymap` removed; `copy=` keywords deprecated. grep of ggTrader: 0 hits for `fillna(method`, `resample('M')`, `freq='H'`, `applymap`, `axis=` in groupby (scripts and src), so this class looks small. Note: grep patterns are approximate, not an audit.

**Copy-on-Write:** chained assignment `df['a'][mask] = 0` stops working (measured: `[1,0,0]` -> `[1,2,3]`, only a `ChainedAssignmentError` warning); a subset no longer mutates its parent (`s = df['a']; s.iloc[0] = 99` leaked in 2.3.3, not in 3.1.1). `Series.to_numpy()` of a column is **read-only** (measured `writeable: False`). Any code that mutates `.values` / `.to_numpy()` in place raises `ValueError: assignment destination is read-only` (migration guide, "Read-only NumPy arrays"). Numba kernels and vectorbt accepted read-only inputs in my probe (`from_signals` with a read-only close array ran, Rust and numba). `inplace=True` on drop/rename/set_index/etc. is deprecated in 3.1 (16 hits in `src/`), harmless for now (warning).

**String dtype:** string columns and string `Index` (symbols!) become dtype `str` (NaN sentinel), not `object`; `dtype == object` checks fail (measured `False`). In my probe sqlite round trip gave `str` for text columns; `None` read back as `NaN` instead of `None`. grep: `dtype == object` / `infer_objects` / `dtype=object`: 2 hits in `src/`.

**MultiIndex OHLCV:** `ohlcv['AAA']['close']` and `pd.concat({sym: df}, axis=1)` worked unchanged; the 3.0 notes list `MultiIndex.dtype` constructor-arg removal and several MultiIndex bug fixes only. The `yfinance` 2-ticker download returned the same `['Price','Ticker']` column levels on both versions. `lab/simulate.py` (which uses `MultiIndex.from_product`, `pd.concat`, `set_axis`, `reindex().ffill()`) ran bit-identically (3.1).

**to_sql / read_sql:** the 3.0 notes list only additions (`if_exists="delete_rows"`, generic `DatabaseError`); my sqlite round trip (SQLAlchemy 2.0.48) worked, `method='multi'` worked, `chunksize` worked, `parse_dates` returned `datetime64[us]`. **Not tested against PostgreSQL/TimescaleDB with psycopg2 (UNVERIFIED)**, which is where the naive-UTC `timestamp WITHOUT time zone` hazard lives.

**Other:** timezone objects become `zoneinfo` instead of pytz (measured); `Day` offset is calendar-day; `Series.view`/`ravel` removed; `groupby` mean/sum may differ in last digits (no Kahan, 3.1 notes, GH 65103); `DataFrame.sum/mean(axis=1)` on bool+numeric mix dtype changes (3.1). Python >=3.11 is already satisfied (3.12.3).

### 6.3 What would be needed (ordered)

1. Remove `pandas-ta` from `pyproject.toml` (dead).
2. Build a second venv/Docker target with pandas 3.1.x, numpy 2.5.x, numba 0.68.x, plotly (7 works with vectorbt >= 1.1.1), vectorbt 1.2.0 (+ optional rust); pin exactly.
3. Run the full test suite via `scripts/run_tests.sh`, with `-W error::FutureWarning`-style triage first on 2.3.3 (pandas recommends "upgrade to pandas 2.3 and ensure your code is working without warnings first", 3.0 notes).
4. Fix the silent-change list in 6.2 (default `pct_change()`, epoch units, `stack`, string-dtype checks) and re-run a pinned-window WFO on both stacks, comparing equity curves, before trusting any new number. The existing fold results are the regression oracle.
5. Verify the Postgres write/read path against a scratch database, including timestamp units and the naive-UTC rule.
6. Re-baseline wall time: the numba kernel got 2.6x slower on the newer numba (3.2); with Rust that is repaired.

Effort estimate: UNVERIFIED (I did not port the code base); the risk is dominated by silent numeric/unit changes rather than by loud errors.

---

## 7. Assessment and options for the owner

| Option | What changes | Benefit | Risk | My view |
|---|---|---|---|---|
| A. Stay on vectorbt 0.28.5 | nothing | none needed: `from_*` is ~4% of WFO wall | none; already what produced every published number | **Default** |
| B. Test vectorbt 1.0.0 + `vectorbt-rust==1.0.0` on the current stack | new pins; run with `engine="rust"` explicitly | cold start 0.2 s instead of 4-10 s; ~0 steady-state change (measured -6% to +8%) | `auto` flips on silently when the wheel is installed; error-class change (`RejectedOrderError`); 1.0.0 still lacks the plotly-7 fix (keep `plotly<7`) | Low risk, low value. Only worth it as a measured experiment |
| C. vectorbt 1.2.0 + pandas 3 stack | section 6 | access to new risk metrics (ulcer index, Martin ratio) and bug fixes that ggTrader does not use | Large, silent-change surface; numba kernel 2.6x slower unless Rust is on; dead `pandas-ta` pin must go | Not now. Revisit when a pandas-3 port is wanted for other reasons |
| D. "Use vbt built-ins more" (indicators, vbt Rust kernels) for the 27% rolling/EWM share | rewrite hand-written pandas indicator math to `vbt.MA/RSI/...` | 1.7-7x on the rolling/RSI pieces in numba (3.3) | Defaults differ from textbook (RSI/MACD use simple MAs); NaN-padding differs from pandas; the profile says indicators are 27% of wall and the PIT universe mask is 65%, which no vectorbt feature addresses | Targeted numba use (not Rust) could cut part of the 27%; the 65% mask is a pandas/numpy problem. Rust makes rolling mean 3x slower here |

Amdahl cap from the owner's profile: removing `from_signals` entirely saves 4%; replacing the 27% indicator share with vbt-numba at the 1.7-7x I measured would save roughly 11-23% of wall (assuming the 27% is dominated by rolling mean/std/RSI-type work, which I did not verify). The 65% PIT mask is outside vectorbt's feature set.

Not recommending anything about retiring any sleeve or the project; none of the above touches strategy decisions.

---

## 8. Unverified or incomplete (do not rely on these)

- Which PostgreSQL/TimescaleDB write path behaves how under pandas 3 (only sqlite tested).
- Behaviour of lightgbm, statsmodels, hmmlearn, optuna, ta, pandas-market-calendars, exchange_calendars, alpaca-py under pandas 3; only yfinance 1.2.0 `download` was exercised.
- Whether ggTrader's full WFO (real data, real PIT mask, ~17 folds) reproduces under 1.2.0/pandas 3: I ran `lab/simulate.py` on synthetic data only.
- Why numba 0.68.1/llvmlite 0.50.0/numpy 2.5.4 make the order-simulation kernel 2.6x slower than 0.61.2/0.44/2.2.6 (isolated to that bundle, not to vectorbt code; sub-cause unknown).
- Why Rust rolling mean is 3x slower than numba at 1900x500 here; memory layout is only a guess.
- Hardware, OS and toolchain behind the published `BENCHMARKS.md` (not stated).
- Whether the numba compile cache explains the drop in first-call time across passes (13.8 s -> 5.2 s).
- Effort/duration of the pandas-3 port.
- The remaining 2% of release-diff semantics I did not read line by line: I relied on release notes, the commit list and empirical parity for the paths ggTrader uses; unused modules (labels, signals generators, records, plotting) were not tested for drift except the indicator/metric set in `drift_compare.txt`.
- Timing is from a shared 4-core machine (load average 3-10); treat ratios within ~10% as noise.

---

## 9. Artifacts (all under `/tmp/claude-1000/-home-flynn-ggTrader/6c9c1710-29af-4adf-b77b-ad700b6fcc23/scratchpad/`)

Report: `vbt-research/B-vectorbt-1x-and-rust.md`.
Scripts and raw outputs in `vbt-research/B-bench/`:

- Parity: `bench_portfolio.py`, `portfolio_venv*_*.json/.npz/.txt` (incl. `*_pass2/3`), `fuzz_parity.py`, `fuzz_venv*.npz(.errs.json)`, `fuzz_compare.txt`, `fuzz_raw.txt`, `compat_ggtrader_simulate.py`, `compat_*.npz`, `compat_raw.txt`, `compat_compare.txt`.
- Speed: `bench_small_calls.py`, `small_calls_raw.jsonl`, `small_calls_table.txt`, `profile_small_call.py`, `profile_small_call_out.txt`, `bench_indicators.py`, `indicators_venv120_run1.json`, `indicators_venv120_run2.json`, `indicators_raw_run1.txt`, `indicators_raw_run2.txt`, `check_exactness.py`, `check_exactness_out.txt`.
- Drift and fallback: `indicator_drift.py`, `drift_*.npz`, `drift_compare.txt`, `fallback_probe.py`, `fallback_probe_out.txt`.
- pandas 3: `pandas3_probe.py`, `pandas3_probe_venv0285.txt`, `pandas3_probe_venv120.txt`.
- Install logs: `pip_install*.log`. Shared helpers: `common.py`.
- Source clones: `gh/repo` (vectorbt at `v1.2.0`), `gh/releases.json`, `gh/pd3.txt`, `gh/pd31.txt`, `gh/mig.txt`.
