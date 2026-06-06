# WFO Backtest Profiling Report — 2026-06-05

**Goal:** answer the standing request to "maximize VectorBT usage / eradicate Python
loops / optimize `Portfolio.from_signals`" with *measured* data instead of assumptions,
and give a go/no-go on any refactor.

**TL;DR:** The original prescription is aimed at the wrong target. `Portfolio.from_signals`
(the simulation) is **7.9%** of runtime and the lone pure-Python `fixed_sl_tp` loop is
**~0.1%** (1.2 s). The real bottleneck — **~58% of runtime** — is **vectorbt metric-accessor
overhead**: the per-fold train-metric computation calls 4–5 separate vbt accessors
(`sortino_ratio`, `total_return`, `max_drawdown`, `trades.profit_factor`), each of which
re-derives returns and re-runs vbt's config/type-checking/array-wrapping machinery. The
high-payoff change is to compute the composite metric from raw arrays once per fold, not
to vectorize the simulation or numba-ify `fixed_sl_tp`.

---

## Method

- **Tool:** `cProfile` (stdlib; only profiler in the image).
- **Critical correction to the harness:** `ggt.py research` *always* shells the WFO
  compute out to `scripts/run_walk_forward_optimization.py` via `subprocess.run` —
  **even with `--no-parallel`** ([cmd_research.py:393-415](../src/ggTrader/cli/cmd_research.py#L393-L415)).
  Profiling `ggt.py research` therefore captures only the parent blocked in `waitpid`
  (verified: 693.8 s / 695 s = **99.8% in `posix.waitpid`**, zero backtest frames). The
  fix was to point cProfile **directly at the worker script in-process**. `scripts/profile_wfo.sh`
  was updated to do this.
- **Workload:** `BTC-USD,ETH-USD,SOL-USD`, `EXCHANGE=binanceus`, 2023-06-05 → 2026-06-05,
  `--phase1 --no-progress`. 3 coins × 33 strategy×exit combos × 10 folds = stable
  per-function proportions.
- **Profile artifact:** `results/profiling/wfo_20260605_103005_primary.prof` (3.9 MB),
  re-loadable via `pstats`; summary in `results/profiling/wfo_20260605_103005_primary.md`.
- cProfile adds overhead (here ~1.5×: 693 s uninstrumented worker → **1048.8 s** profiled).
  Read **proportions**, not absolute times.

> **Caveat — numba is invisible to cProfile.** `atr_trailing` and `trailing_stop` exits
> run as `@njit` kernels ([signals.py:262,312](../src/ggTrader/indicators/signals.py#L262))
> whose compiled time does **not** appear in cProfile. So `fixed_sl_tp` (pure Python) shows
> its *full* cost while the numba exits look ~free even when doing real work. This biases the
> profile *toward* `fixed_sl_tp` — and it's still only 1.9%, which only strengthens the
> conclusion below.

---

## Phase breakdown (by cumulative time, % of 1048.8 s profiled)

| Phase | Function | cum (s) | % |
|---|---|---|---|
| Total worker | `run_walk_forward_optimization.main` | 1040.3 | 99.2% |
| WFO loop | `wfo._execute_wfo_loop` | 993.7 | 94.7% |
| Per-fold | `wfo._process_wfo_fold` (760 folds) | 1017.4 | 97.0% |
| Data load | `load_data_and_setup` | 1.5 | 0.1% |
| ccxt tail fetch | `fetch_ohlcv` (7 calls) | 3.8 | 0.4% |

Data loading is negligible. Effectively all time is in the per-fold WFO inner loop.

## Top-3 bottlenecks (ranked, with evidence)

### 1. vectorbt metric-accessor overhead — ~58% of runtime
The dominant cost chain:

| Function | cum (s) | % | calls |
|---|---|---|---|
| `portfolio/base.py:get_returns_acc` | 604.4 | 57.6% | 4 034 |
| `portfolio/base.py:returns` | 467.0 | 44.5% | 4 704 |
| `portfolio/base.py:value` | 457.8 | 43.6% | 5 376 |
| `utils/decorators.py:wrapper` | 885.5 | 84.4% | **1 592 792** |

Self-time confirms it's Python object churn inside vbt's accessor layer, not numerics:

| Self-time hot spot | self (s) | calls |
|---|---|---|
| `object.__dir__` | 121.7 | 7 941 593 |
| `builtins.isinstance` | 54.7 | 280 037 471 |
| pandas `DatetimeIndex.__iter__` | 35.3 | 30 465 642 |
| `copy.deepcopy` (+`_deepcopy_dict`) | 43.4 | — |
| vbt `is_subclass_of`/`is_instance_of`/`assert_instance_of` | ~88 | ~107 M |
| vbt `config.__init__` / `merge_dicts` / `convert_to_dict` | ~67 | ~23 M |

**Root cause** ([metrics.py:134 `_train_metric_series`](../src/ggTrader/core/metrics.py#L134),
reached via [sensitivity.py:58](../src/ggTrader/core/sensitivity.py#L58)): with
`TRAIN_METRIC=composite` (the live setting), each fold computes the composite from **four
separate vbt accessor calls** — `pf.sortino_ratio()`, `pf.total_return()`,
`pf.max_drawdown()`, `pf.trades.profit_factor()`. Every call re-runs vbt's
`get_returns_acc` → `returns` → `value` chain, which re-wraps arrays
(`array_wrapper.__init__` ×461 k), rebuilds configs (`config.__init__` ×1.05 M), deepcopies,
and runs millions of `__dir__`/`isinstance` type assertions. The actual reductions
(Sharpe/Sortino/etc.) are trivial; the overhead is vbt's per-call object machinery,
multiplied by 760 folds.

### 2. `Portfolio.from_signals` (the simulation) — 7.9% (82.8 s, 2 692 calls)
The thing the original prompt wanted to optimize. Already uses `cash_sharing`, single-group
`group_by`, `size_type="percent"`. At <8% it is **not** where the time goes.

### 3. Signal generation — ~1.9% combined
`compute_entries` 14.5 s (1.4%), all `compute_exits` 5.0 s (0.5%) — of which the lone
pure-Python `FixedSLTPExit.compute_exits` loop ([strategies.py:1183](../src/ggTrader/indicators/strategies.py#L1183))
is just **1.2 s (~0.1%)**. The loop the prompt targeted is rounding error. (Even this
slightly over-counts it vs the invisible numba exits — see caveat.)

---

## Already optimal — do not touch

- **`Portfolio.from_signals` config** — `cash_sharing=True`, single-group `group_by`,
  `size_type="percent"` ([wfo.py:908](../src/ggTrader/core/wfo.py#L908),
  [orchestrator.py:690](../src/ggTrader/core/orchestrator.py#L690)); correctly
  `group_by=False` for per-coin WFO ([wfo.py:830](../src/ggTrader/core/wfo.py#L830)).
- **Vectorized train path** — `USE_VECTORIZED=True` is already the default on the WFO hot
  path ([wfo.py:81](../src/ggTrader/core/wfo.py#L81)).
- **Numba exits** — `_atr_trailing_stop..._numba` / `_trailing_stop..._numba`
  ([signals.py:262,312](../src/ggTrader/indicators/signals.py#L262)) already compiled.
- **Param-grid `itertools.product` sweeps** — correct vectorized-batch pattern, not hot.
- **Invalid prompt prescriptions** — `call_seq="market"` is not valid in vectorbt 1.0.0;
  `group_by=True` everywhere would corrupt per-coin WFO. Reject both.

---

## Recommendation

**Go:** Collapse the per-fold composite-metric computation. Extract the returns/equity
matrix from the portfolio **once per fold** and compute Sortino, Calmar (total_return /
|max_drawdown|), and profit factor with plain numpy over columns, instead of 4 independent
vbt accessor passes. Target: [metrics.py `_train_metric_series`](../src/ggTrader/core/metrics.py#L134)
+ the `_calmar_ratio_series` / `_profit_factor_series` helpers.
- **Effort:** moderate (1 function family; reuse a single `pf.returns()`/`pf.value()`
  result + trade records). Keep the existing rank-composite math identical — only change
  *how* the inputs are obtained.
- **Payoff:** large. The metric accessors are ~58% of runtime; deriving all metrics from a
  single returns/value extraction should remove most of the redundant `get_returns_acc`/
  `value`/config-deepcopy passes. Plausible ~1.5–2.5× speedup on the WFO train phase
  (≈97% of total). Verify by re-running this profile after the change.
- **Risk:** medium — must reproduce vbt's Sortino/Calmar/PF definitions exactly. Gate with
  a before/after equality check on the metric Series for a fixed seed/run.

**No-go (for now):**
- **Numba-ify `fixed_sl_tp`** — 1.9% ceiling; not worth the risk until bottleneck #1 is
  fixed. Revisit only if it becomes a meaningful share afterward.
- **Rewriting `Portfolio.from_signals`** — already optimal; 7.9% ceiling.

---

## Outcome (implemented & validated, same day)

The "Go" recommendation was implemented (commits `dd674f1`, `86a6c95`, + the OOS-stats follow-up).
`core/metrics.py` `_returns_based_metrics` / `_fold_stats_metrics` extract `pf.returns()` once per
fold and use vbt's own kernels; `wfo._process_wfo_fold` was updated likewise. Verified **bit-identical**
to the per-call accessors by `tests/test_metrics_returns_extraction.py`.

End-to-end cold re-profile (cache purged, same 3-coin workload):

| | profiled runtime | get_returns_acc | from_signals |
|---|---|---|---|
| Baseline | 1048.8 s | 595 s (57%, 4034 calls) | 82.8 s (7.9%) |
| Optimized | **507.2 s (2.07×)** | 4.3 s (0.9%, 14 calls) | 84.9 s (16.7%) |

All 76 per-combo gate verdicts identical to baseline (same PASS/FAIL + failure reasons); only
`wfe` differs at ≤1e-13 (floating-point reassociation). `Portfolio.from_signals` (the simulation)
is now the largest single item — as predicted, the remaining headroom is small and not worth the
risk of numba-ifying `fixed_sl_tp` (still ~0.1%).

## Reproduce

```bash
# inside the container (src/scripts bind-mounted), profiles the WORKER in-process:
docker compose run --rm \
  -v "$PWD/src:/app/src" -v "$PWD/scripts:/app/scripts" \
  -e EXCHANGE=binanceus -e END_DATE=2026-06-05 \
  ggtrader_live bash scripts/profile_wfo.sh "BTC-USD,ETH-USD,SOL-USD"

# analyze any .prof:
python3 scripts/analyze_profile.py results/profiling/<file>.prof --top 30
```
