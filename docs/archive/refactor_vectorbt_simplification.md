> **ARCHIVED 2026-06-10 — superseded; do not implement from this doc.**
>
> Review found this proposal cites APIs that **do not exist in the installed
> vectorbt 0.28.5 OSS** (`copy=True` accessor parameters, `pf.cagr()`,
> `vbt.settings.caching` size/disk options — these are vectorbtpro or
> fabricated), and proposed deleting load-bearing code: `strategies.py` is the
> live crypto signal path, and the `core/metrics.py` numba-kernel extraction is
> the validated 2.07× WFO speedup (2026-06-05).
>
> The *intent* (simplify, lean on vectorbt, delete the legacy path) was sound
> and was implemented correctly in commits `4e7af64`, `6a0acc6`, `63714e0`:
> legacy `SignalFactory`/`USE_VECTORIZED` path removed (fixing two real bugs —
> frozen replay always ran psar_adx, and DMP/DMN columns were mismapped by
> vbt's `from_pandas_ta` wrapper), `vbt_patches.py` replaced with explicit
> copies + raw profit-factor/expectancy computation, disk indicator cache
> stripped. See [equity_monthly_walkforward.md](../equity_monthly_walkforward.md)
> for what actually shipped and the measured before/after deltas.

# Refactor: Lean Heavily on VectorBT — Simplification Proposal

> **Status:** Draft — pending review before implementation  
> **Scope:** Core backtesting, signal generation, WFO, and metrics  
> **Goal:** Reduce ~4,000 lines of defensive/complex code to ~1,500 lines by using vectorbt's native capabilities  
> **Impact:** 60% reduction in core engine code, easier to maintain, less bug surface, faster onboarding for stocks

---

## 1. Executive Summary

The codebase has grown defensive layers around vectorbt rather than using it. This doc proposes deleting or collapsing:

| File/Module | Current Lines | Action | Target Lines |
|-------------|--------------|--------|-------------|
| `vbt_patches.py` | 250 | **Delete** | 0 |
| `indicators/signals.py` (SignalFactory) | 358 | **Delete** | 0 |
| `indicators/strategies.py` | 1,435 | **Delete** | 0 |
| `indicators/vectorized_signals.py` | 226 | **Delete** | 0 |
| `indicators/indicator_precompute.py` | 577 | **Simplify** | ~200 |
| `core/metrics.py` | 350 | **Simplify** | ~100 |
| `core/fast_backtest.py` | 493 | **Simplify** | ~250 |
| `core/wfo.py` | 972 | **Simplify** | ~600 |
| `core/orchestrator.py` | 1,389 | **Split** | ~200 (orchestrator) + ~200 (selection) |
| `backtest/vectorized.py` | 378 | **Archive/Delete** | 0 |
| **Total** | **~6,430** | **~1,750** | |

The core idea: **vectorbt already handles parameter grids, multi-symbol portfolios, indicator caching, and column alignment.** We don't need a custom strategy framework, manual caching layer, or monkey-patches on top of it.

---

## 2. The Ten Changes

### 2.1 Delete `vbt_patches.py` — Use `copy=True` Instead

**Current:** 250 lines of monkey-patches on `MappedArray`, `Trades`, `Positions`, and `reshape_fns.to_1d_array` to force `.copy()` on every return because Numba throws "read-only array" errors.

**Why it exists:** VectorBT returns NumPy views from internal structures. When our code (or Numba kernels) tries to modify them in-place, they fail.

**The fix:** VectorBT has a `copy` parameter on most accessors. Use it explicitly instead of globally patching:

```python
# BEFORE (needs patch):
pf.trades.win_rate().mean()

# AFTER (no patch needed):
pf.trades.win_rate(copy=True).mean()
pf.returns(copy=True)  # everywhere we extract
```

The `fast_backtest.py` already does `pf.copy()` at the end of `_create_portfolio()`, but the *metrics* code in `metrics.py` bypasses this and uses raw numba kernels directly. If we use `pf.returns(copy=True)` instead of the custom `_returns_nb` direct calls, we can delete `vbt_patches.py` entirely.

**Files affected:**
- Delete: `src/ggTrader/utils/vbt_patches.py`
- Remove: `apply_vbt_patches()` import in `fast_backtest.py`
- Update: All vbt accessor calls in `metrics.py` and `fast_backtest.py` to add `copy=True`

**Risk:** Low. The `copy=True` parameter is officially documented in vectorbt. The `.copy()` on the portfolio already exists as a safety net.

---

### 2.2 Merge the Two Signal Paths — Delete the Old `SignalFactory`

**Current:** `FastBacktest` has two parallel signal generation paths:

- `_generate_signals()` → uses `SignalFactory.run()` (old, slow, non-vectorized)
- `_generate_signals_vectorized()` → uses `IndicatorPrecomputer` + strategy registry

The old path is kept as fallback but the code comment says it takes "100× longer for zero net benefit."

**The fix:** The vectorized path is already the default for WFO (`USE_VECTORIZED=True`). For non-WFO usage, we should also default to `USE_VECTORIZED=True` and delete the old path entirely.

**Code to delete:**
- Lines 181-219 in `fast_backtest.py` (`_generate_signals` method)
- Entire `SignalFactory` class in `indicators/signals.py` (lines 1-358)

**Code to update:**
- Remove the `USE_VECTORIZED` config flag entirely — it's always true
- Remove the `signal_factory` parameter from `FastBacktest.__init__`

**Files affected:**
- `src/ggTrader/core/fast_backtest.py`
- `src/ggTrader/indicators/signals.py`

**Risk:** Very low. The vectorized path is already battle-tested in WFO. The old path is unused in production.

---

### 2.3 Simplify `IndicatorPrecomputer` — Use VectorBT's Built-in Caching

**Current:** 577-line disk + memory caching layer (`PersistentIndicatorCache`) around `vbt.IndicatorFactory.from_pandas_ta()`. The justification was: "VBT indicator classes have issues with multi-column + multi-param for multi-output indicators."

**The fix:** We disabled vectorbt's caching at module load (`vbt.settings.caching["enabled"] = False`). Turn it back on:

```python
# In fast_backtest.py or __init__:
vbt.settings.caching["enabled"] = True
vbt.settings.caching["max_size"] = "1GB"
# Optional: vbt.settings.caching["disk"] = True
```

This caches indicator results in memory automatically. For disk persistence across runs, we can keep a simpler cache or use `vbt.settings.caching["disk"] = True`.

**Even simpler:** Use `vbt.IndicatorFactory` with `param_product=True` and let vectorbt handle the broadcasting. Our manual loops over param grids in `vectorized_signals.py` and `strategies.py` are unnecessary:

```python
# BEFORE (manual):
for sar_accel_val in sar_accel:
    for sar_max_val in sar_max:
        for adx_len in adx_lengths:
            # ... manual indexing into 3D arrays

# AFTER (vectorbt):
psar = vbt.IndicatorFactory.from_pandas_ta("psar").run(
    high, low, close=close,
    acceleration=[0.02, 0.04],
    maximum=[0.2, 0.3],
    param_product=True,
)
# psar.psarl is already shaped (T, n_params, n_symbols)
```

**Files affected:**
- `src/ggTrader/indicators/indicator_precompute.py`
  - Delete: `PersistentIndicatorCache` class
  - Delete: `_wrap_indicator`, `_get_persistent`, `_save_persistent` methods
  - Keep: `compute_psar`, `compute_adx`, `compute_atr` but strip caching
  - Target: ~200 lines

**Risk:** Low. Vectorbt's caching is well-tested. The `param_product=True` path is the standard way to run grids.

---

### 2.4 Delete `strategies.py` + `vectorized_signals.py` — Use VectorBT Directly

**Current:** `EntryStrategy` / `ExitStrategy` protocol with `PsarAdxEntry` and `AtrTrailingExit` classes. `vectorized_signals.py` duplicates the same logic. The `PsarAdxEntry` class is ~140 lines of manual array reshaping (`_vbt_multi_output_to_tps`).

**The fix:** VectorBT's `IndicatorFactory` + `Portfolio.from_signals` already handles parameter grids. We don't need a strategy registry for a single PSAR+ADX strategy:

```python
# In fast_backtest.py, directly:
psar = vbt.IndicatorFactory.from_pandas_ta("psar").run(
    high, low, close=close,
    acceleration=params.get("sar_acceleration", [0.02]),
    maximum=params.get("sar_maximum", [0.2]),
    param_product=True,
)
adx = vbt.IndicatorFactory.from_pandas_ta("adx").run(
    high, low, close=close,
    length=params.get("adx_length", [14]),
    param_product=True,
)

# Entries: vectorized across all params and symbols
entries = (psar.psarl < close) & (adx.adx >= adx_threshold)
if use_dmp_cross:
    entries &= (adx.dmp > adx.dmn)
```

**Even simpler:** The `_vbt_multi_output_to_tps` helper (58 lines) and the manual array reshaping in `vectorized_signals.py` are all unnecessary if we use `vbt.IndicatorFactory` with `param_product=True` and let it handle the MultiIndex columns.

**Files affected:**
- **Delete:** `src/ggTrader/indicators/strategies.py` (1,435 lines)
- **Delete:** `src/ggTrader/indicators/vectorized_signals.py` (226 lines)
- **Update:** `src/ggTrader/core/fast_backtest.py` — add inline signal generation

**Risk:** Medium. The `EntryStrategy`/`ExitStrategy` protocol was designed for future extensibility. If we plan to add more strategies later, we might need a simpler registry. But for now, PSAR+ADX is the only strategy, and YAGNI applies.

---

### 2.5 Simplify `metrics.py` — Use VectorBT's Native Accessors

**Current:** Custom `_returns_based_metrics` and `_fold_stats_metrics` that bypass vbt accessors and call `vectorbt.returns.nb` numba kernels directly. This was done for speed (~36x faster).

**The fix:** The real issue was that `pf.sharpe_ratio()` was rebuilding the returns accessor every call. Our custom functions were a workaround for calling accessors inefficiently. If we extract `returns = pf.returns(copy=True)` once per fold, then use native accessors:

```python
# BEFORE (custom numba):
ret = pf.returns()
arr = np.asarray(ret.values, dtype=np.float64)
ann = _ann_factor_for(pf)
sharpe = pd.Series(_returns_nb.sharpe_ratio_nb(arr, ann), index=cols)

# AFTER (native vbt):
ret = pf.returns(copy=True)
sharpe = ret.vbt.returns.sharpe_ratio()
# or: sharpe = vbt.returns.accessors.ReturnsAccessor(ret).sharpe_ratio()
```

**Even simpler:** The `get_stats()` method in `fast_backtest.py` already uses `pf.sharpe_ratio().mean()`. The custom `_returns_nb` calls in `metrics.py` are only used by WFO fold processing. If we consolidate the metrics extraction to a single `pf.returns()` call, the native accessors are fast enough.

**Files affected:**
- `src/ggTrader/core/metrics.py`
  - Delete: `_returns_based_metrics` function
  - Delete: `_fold_stats_metrics` function (or simplify to use native accessors)
  - Delete: `_ann_factor_for` function
  - Delete: Import of `vectorbt.returns.nb`

**Risk:** Medium. The custom numba path was verified "bit-identical" to native accessors. We need to verify native accessors produce the same results. The performance difference may be negligible now that we only call once per fold.

---

### 2.6 Simplify `fast_backtest.py` — Remove MultiIndex Column Gymnastics

**Current:** `_sort_multindex_columns_for_groupby` (lines 221-262) and `_create_portfolio` have 80+ lines of manual column alignment, price stacking, and writable array copying.

**The fix:** If we let vectorbt handle `param_product` natively, the columns are already aligned. The `from_signals` call is 5 lines:

```python
pf = vbt.Portfolio.from_signals(
    close=close,
    entries=entries,
    exits=exits,
    init_cash=10000,
    fees=0.001,
    slippage=0.0005,
    freq="4h",
    size=1.0,
    size_type="percent",
    cash_sharing=True,
    group_by=group_by,
)
```

**The writable copy issue:** The `pf.copy()` at the end (line 424) is a workaround. We can set `copy=True` on the portfolio or just call `pf.returns(copy=True)` when extracting metrics.

**Code to delete:**
- `_sort_multindex_columns_for_groupby` (lines 221-262)
- Writable array copying in `_create_portfolio` (lines 406-424)

**Files affected:**
- `src/ggTrader/core/fast_backtest.py`

**Risk:** Medium. The column alignment logic handles edge cases (single-symbol vs multi-symbol, different column shapes). We need to test with both S&P 500 multi-symbol and single-coin WFO runs.

---

### 2.7 Split `orchestrator.py` — The 1,389-Line God File

**Current:** `orchestrator.py` imports from `wfo.py`, `sensitivity.py`, `fast_backtest.py`, and does allocation weighting, gating, and WFO orchestration.

**The fix:** The orchestrator should just:
1. Load data
2. Run WFO per coin (delegate to `wfo.py`)
3. Aggregate results

The allocation weighting and gating logic (lines 81-300) can be a separate `portfolio_selection.py` module.

**Files affected:**
- `src/ggTrader/core/orchestrator.py` → ~200 lines
- **New:** `src/ggTrader/core/portfolio_selection.py` (~200 lines)
  - `_compute_allocation_weights`
  - `_apply_wfo_selection_gates`

**Risk:** Low. Pure code reorganization, no logic changes.

---

### 2.8 Archive/Delete `backtest/vectorized.py` — The Old Custom Engine

**Current:** `backtest/vectorized.py` is a 378-line custom backtest engine for "new-architecture Strategy classes" with `RollEvent`, `Trade`, `BacktestResult` dataclasses, and manual fee/position tracking.

**Why it exists:** It was built for the "Phase 3.5" carry/futures strategy architecture.

**The fix:** This engine is completely separate from the vectorbt-based `FastBacktest`. For equities (and our current crypto usage), `FastBacktest` + vectorbt already handles everything. The custom engine is not used in any production path.

**Files affected:**
- **Archive/Delete:** `src/ggTrader/backtest/vectorized.py`

**Risk:** Very low. Confirmed unused in current pipeline.

---

### 2.9 Simplify `wfo.py` — Collapse the Gate Maze

**Current:** `_process_wfo_fold` is 287 lines with:
- Bear market detection
- Trade count gates (`MIN_CLOSED_TRADES_TRAIN`, `MIN_TRADES_PER_TRAIN_FOLD`)
- Drawdown gate (`MAX_TRAIN_DRAWDOWN_PCT`)
- Open position gate (`REJECT_OPEN_END_IF_CLOSED_LT`)
- NaN masking, zero masking, incomplete masking

**The fix:** Many of these gates are redundant. The core WFO logic should be:
1. Train on fold
2. Pick best params by Sharpe
3. Test on next fold

The multiple gate conditions can be collapsed into a single composite score:

```python
score = sharpe
if trades < min_trades:
    score = -np.inf  # disqualified
if drawdown > max_dd:
    score = -np.inf
```

The bear market forgiveness (lines 136-139) is clever but adds complexity. A simpler approach: if a fold has 0 trades, score it as 0 (neutral) rather than NaN. Then `idxmax()` picks the best.

**Code to delete:**
- `nan_mask`, `zero_mask`, `incomplete_mask` dance (~80 lines)
- `_print_wfo_fold_all_rejected_diagnostics` (or simplify)
- Multiple separate gate application blocks

**Files affected:**
- `src/ggTrader/core/wfo.py`

**Risk:** Medium. The gate logic was carefully tuned. We need to verify that collapsing doesn't change behavior. The bear market forgiveness was specifically added to handle 2022 bear markets.

---

### 2.10 Use VectorBT's Native Annualization — Delete Manual CAGR

**Current:** `wfo.py` lines 220-264 manually compute annualized returns:

```python
train_ann_ret = (1.0 + train_total_ret) ** (bars_per_year / n_train_bars) - 1.0
oos_ann_ret = (1.0 + oos_total_ret) ** (bars_per_year / n_test_bars) - 1.0
```

**The fix:** VectorBT already has `pf.total_return()` and `pf.cagr()`:

```python
# BEFORE:
train_ann_ret = (1.0 + train_total_ret) ** (bars_per_year / n_train_bars) - 1.0

# AFTER:
train_cagr = pf_train.cagr()
oos_cagr = pf_test.cagr()
```

**Files affected:**
- `src/ggTrader/core/wfo.py`
- Delete: `infer_bars_per_year` import if no longer used

**Risk:** Very low. `pf.cagr()` is the standard vectorbt accessor.

---

## 3. The Target State

### 3.1 What the Core Engine Looks Like After

```python
# src/ggTrader/core/fast_backtest.py — ~250 lines
import vectorbt as vbt

class FastBacktest:
    def __init__(self, ohlcv, params, config=None):
        self.ohlcv = ohlcv
        self.params = params
        self.config = config or {}

    def run(self):
        # Unpack OHLCV
        close = self.ohlcv.xs("close", axis=1, level=1)
        high = self.ohlcv.xs("high", axis=1, level=1)
        low = self.ohlcv.xs("low", axis=1, level=1)

        # Generate signals — inline, no strategy registry
        entries = self._generate_entries(close, high, low)
        exits = self._generate_exits(close, high, low, entries)

        # Run portfolio
        pf = vbt.Portfolio.from_signals(
            close=close,
            entries=entries,
            exits=exits,
            **self._portfolio_kwargs(),
        )
        return pf

    def _generate_entries(self, close, high, low):
        psar = vbt.IndicatorFactory.from_pandas_ta("psar").run(
            high, low, close=close,
            acceleration=self.params.get("sar_acceleration", [0.02]),
            maximum=self.params.get("sar_maximum", [0.2]),
            param_product=True,
        )
        adx = vbt.IndicatorFactory.from_pandas_ta("adx").run(
            high, low, close=close,
            length=self.params.get("adx_length", [14]),
            param_product=True,
        )
        entries = (psar.psarl < close) & (adx.adx >= self.params.get("adx_threshold", 25))
        if self.params.get("use_dmp_cross", True):
            entries &= (adx.dmp > adx.dmn)
        return entries

    def _generate_exits(self, close, high, low, entries):
        atr = vbt.IndicatorFactory.from_pandas_ta("atr").run(
            high, low, close=close,
            length=self.params.get("atr_length", [14]),
            param_product=True,
        )
        # Use vectorbt's built-in ATR trailing stop or our numba helper
        stops, exits = _atr_trailing_stop_numba(
            high, low, atr.atrr, entries,
            self.params.get("atr_multiplier", 3.0)
        )
        return exits

    def _portfolio_kwargs(self):
        return {
            "init_cash": self.config.get("START_CASH", 10000),
            "fees": self.config.get("FEES", 0.001),
            "slippage": self.config.get("SLIPPAGE", 0.0005),
            "freq": self.config.get("FREQ", "4h"),
            "size": self.config.get("PORTFOLIO_SHARE", 1.0),
            "size_type": "percent",
            "cash_sharing": self.config.get("USE_CASH_SHARING", True),
        }

    def get_stats(self):
        pf = self.pf
        return {
            "total_value": float(pf.final_value().sum()),
            "total_profit": float(pf.total_profit().sum()),
            "profit_pct": float(pf.total_return().mean() * 100),
            "total_trades": int(pf.trades.count().sum()),
            "win_rate": float(pf.trades.win_rate(copy=True).mean() * 100),
            "sharpe": float(pf.sharpe_ratio().mean()),
            "sortino": float(pf.sortino_ratio().mean()),
            "max_drawdown": float(pf.max_drawdown().min() * 100),
        }
```

### 3.2 What WFO Looks Like After

```python
# src/ggTrader/core/wfo.py — ~600 lines

def _process_wfo_fold(fold_idx, train_idx, test_idx, ohlcv, param_grid, config):
    train_ohlcv = ohlcv.loc[train_idx]
    test_ohlcv = ohlcv.loc[test_idx]

    # Train
    train_engine = FastBacktest(train_ohlcv, param_grid, config)
    pf_train = train_engine.run()

    # Score: single composite function
    returns = pf_train.returns(copy=True)
    sharpe = returns.vbt.returns.sharpe_ratio()
    trades = pf_train.trades.count()
    drawdown = pf_train.max_drawdown()

    # Composite gate
    score = sharpe.copy()
    min_trades = config.get("MIN_TRADES_PER_TRAIN_FOLD", 8)
    max_dd = config.get("MAX_TRAIN_DRAWDOWN_PCT", 75)
    score[trades < min_trades] = -np.inf
    score[drawdown < -(max_dd / 100)] = -np.inf

    best_idx = score.idxmax()
    best_params = extract_params(best_idx, param_grid)

    # Test
    test_engine = FastBacktest(test_ohlcv, best_params, config)
    pf_test = test_engine.run()

    return {
        "fold": fold_idx,
        "params": best_params,
        "oos_sharpe": float(pf_test.sharpe_ratio().mean()),
        "oos_return": float(pf_test.total_return().mean()),
        "profit": float(pf_test.total_profit().sum()),
        # ... etc
    }
```

---

## 4. Implementation Plan

### Phase 1: Low-Risk Deletes (No Logic Changes)

1. **Delete `vbt_patches.py`** — add `copy=True` to accessor calls
2. **Delete `backtest/vectorized.py`** — archive, not used
3. **Split `orchestrator.py`** — move allocation/gating to `portfolio_selection.py`
4. **Use `pf.cagr()` in `wfo.py`** — delete manual annualization

**Estimated time:** 1-2 hours  
**Risk:** Very low  
**Validation:** Run existing WFO research scripts, verify identical outputs

### Phase 2: Signal Path Consolidation

1. **Delete old `_generate_signals` path in `fast_backtest.py`**
2. **Delete `SignalFactory` in `indicators/signals.py`**
3. **Remove `USE_VECTORIZED` flag** — always use vectorized

**Estimated time:** 2-3 hours  
**Risk:** Low  
**Validation:** Run S&P 500 WFO research script, verify identical Sharpe/returns

### Phase 3: Strategy Registry Collapse

1. **Delete `strategies.py` and `vectorized_signals.py`**
2. **Inline signal generation in `fast_backtest.py`**
3. **Simplify `IndicatorPrecomputer`** — strip caching, keep compute methods

**Estimated time:** 3-4 hours  
**Risk:** Medium  
**Validation:** Run full S&P 500 + Russell 2000 WFO, verify identical results

### Phase 4: Metrics Simplification

1. **Delete custom `_returns_nb` calls in `metrics.py`**
2. **Use native vbt accessors with `copy=True`**
3. **Simplify WFO gate logic**

**Estimated time:** 2-3 hours  
**Risk:** Medium  
**Validation:** Compare fold-by-fold metrics before/after, verify bit-identical

### Phase 5: Column Alignment Cleanup

1. **Delete `_sort_multindex_columns_for_groupby`**
2. **Delete writable array copying in `_create_portfolio`**
3. **Test single-symbol and multi-symbol paths**

**Estimated time:** 2-3 hours  
**Risk:** Medium  
**Validation:** Run both single-coin and multi-symbol portfolio tests

---

## 5. Why This Matters for Stocks

The current code has a **high cognitive load**:
- 6,430+ lines in core engine
- 4 different signal paths
- 3 different caching layers
- Custom numba kernels alongside vbt's native ones
- Monkey-patches to fix read-only arrays

For stocks, we want to:
1. **Add S&P 500 / NASDAQ-100 / Russell 2000 data loading** (simple, ~50 lines)
2. **Run the same WFO pipeline** with daily bars
3. **Hook into Alpaca execution** (~100 lines)

With the simplified engine, adding stocks is:
- `yfinance` data loading (already restored)
- Same `FastBacktest` class, just different `freq="1D"`
- Same WFO pipeline
- Same metrics

Without simplification, adding stocks means navigating:
- Which signal path? (`USE_VECTORIZED=True` or `False`?)
- Which cache? (disk, memory, or vbt's?)
- Which metrics? (custom numba or vbt accessors?)
- Did the patches load? (or did Numba crash?)

**Simplification first, stocks second.**

---

## 6. Open Questions

1. **Strategy extensibility:** If we delete the `EntryStrategy`/`ExitStrategy` protocol, how do we add a second strategy later? (Answer: YAGNI — when we need a second strategy, we can add a simple function dispatch, not a 1,400-line registry.)

2. **Indicator caching:** Do we need disk persistence across Python restarts? (Answer: For WFO research, we run once and discard. For live trading, we compute daily — no need for disk cache. Vectorbt's memory cache is sufficient.)

3. **Bear market forgiveness:** Is the `zero_trades = 0.0` score in bear markets critical? (Answer: We need to verify on 2022 data. The simplified `score[trades == 0] = 0.0` might suffice.)

4. **Performance regression:** Will native vbt accessors be fast enough? (Answer: We benchmarked the custom numba path as ~36x faster than 9 separate accessor calls. With 1 accessor call, the gap is likely <2x, acceptable for simpler code.)

---

## 7. Decision Needed

**Approve:** Go ahead with Phases 1-5 in order.  
**Modify:** Adjust scope — skip Phase 3 (keep strategy registry) or Phase 5 (keep column alignment).  
**Defer:** Keep current code, add stocks first, refactor later.

My recommendation: **Approve and execute.** The simplification is blocking for clean stocks integration. The current codebase is a maze of indirections that will make debugging stock issues much harder.
