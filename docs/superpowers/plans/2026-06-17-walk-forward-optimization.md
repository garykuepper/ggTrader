# Walk-Forward Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add honest out-of-sample measurement to signal strategy sweeps via rolling train/test folds, composite scoring, OOS concatenation, and recommended live params.

**Architecture:** A new `src/ggTrader/lab/wfo.py` module owns all WFO logic — fold generation, composite scoring, per-fold train/test simulation, OOS aggregation, live param selection, and output formatting. It calls existing infrastructure (`simulate_signals`, `sweep.py` helpers, `curve_stats`). CLI gets a `--wfo` flag (mutually exclusive with `--sweep`).

**Tech Stack:** Python 3.12, vectorbt, pandas, numpy. Docker for test execution: `docker compose build ggtrader_live && docker compose run --rm ggtrader_live python -m pytest tests/lab/test_wfo.py -v`

## Global Constraints

- No new dependencies — only use existing lab modules
- No modifications to `simulate.py`, `sweep.py`, `metrics.py`, `strategy.py`, or `persist.py`
- All simulation must use existing `simulate_signals` — fully vectorized, one grouped vbt call per stop config
- Trailing stops (ts_stop, atr_mult/atr_period) handled identically to existing sweep path via `split_params` grouping
- Signal strategies only (ema_cross, wfo_tournament); weight strategies raise an error if `--wfo` is used
- Tests run in Docker: `docker compose build ggtrader_live` before first test run (source is baked into image, not mounted)
- Absolute imports from `src` (e.g., `from ggTrader.lab.wfo import ...`)
- Strict ruff linting; no unused imports, no trailing whitespace

---

### Task 1: Fold Generation and Composite Scoring

**Files:**
- Create: `src/ggTrader/lab/wfo.py`
- Create: `tests/lab/test_wfo.py`

**Interfaces:**
- Consumes: nothing from other tasks
- Produces:
  - `Fold` NamedTuple with fields `train_start`, `train_end`, `test_start`, `test_end` (all `pd.Timestamp`)
  - `generate_folds(eval_start: pd.Timestamp, eval_end: pd.Timestamp, train_years: int = 3, test_years: int = 1) -> List[Fold]`
  - `composite_score(metrics_list: List[Dict[str, float]]) -> List[float]`
  - Constants: `TRAIN_YEARS = 3`, `TEST_YEARS = 1`

- [ ] **Step 1: Write failing tests for `generate_folds`**

```python
# tests/lab/test_wfo.py
import pandas as pd
import pytest

from ggTrader.lab.wfo import Fold, generate_folds


def test_generate_folds_count_and_boundaries():
    """11-year span with 3yr train / 1yr test → 8 folds, no overlap."""
    start = pd.Timestamp("2015-01-01", tz="UTC")
    end = pd.Timestamp("2026-01-01", tz="UTC")
    folds = generate_folds(start, end)
    assert len(folds) == 8
    for f in folds:
        assert f.test_start == f.train_end  # no gap
        train_years = (f.train_end - f.train_start).days / 365.25
        test_years = (f.test_end - f.test_start).days / 365.25
        assert abs(train_years - 3.0) < 0.05
        assert abs(test_years - 1.0) < 0.05
    # No fold's test_end exceeds data_end
    assert all(f.test_end <= end for f in folds)
    # Folds slide by 1 year
    for i in range(1, len(folds)):
        delta_years = (folds[i].train_start - folds[i - 1].train_start).days / 365.25
        assert abs(delta_years - 1.0) < 0.05


def test_generate_folds_short_data_returns_fewer():
    """Only 3.5 years of data → not enough for train+test, returns 0 folds."""
    start = pd.Timestamp("2020-01-01", tz="UTC")
    end = pd.Timestamp("2023-06-01", tz="UTC")
    folds = generate_folds(start, end)
    assert len(folds) == 0


def test_generate_folds_exact_4_years():
    """Exactly 4 years → 1 fold."""
    start = pd.Timestamp("2020-01-01", tz="UTC")
    end = pd.Timestamp("2024-01-01", tz="UTC")
    folds = generate_folds(start, end)
    assert len(folds) == 1
    assert folds[0].train_start == start
    assert folds[0].test_end == end
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `docker compose build ggtrader_live && docker compose run --rm ggtrader_live python -m pytest tests/lab/test_wfo.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ggTrader.lab.wfo'`

- [ ] **Step 3: Implement `generate_folds` and `Fold`**

```python
# src/ggTrader/lab/wfo.py
"""Walk-forward optimization: rolling train/test folds with composite scoring."""

from __future__ import annotations

import math
from typing import Any, Dict, List, NamedTuple, Type

import pandas as pd

TRAIN_YEARS = 3
TEST_YEARS = 1


class Fold(NamedTuple):
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def generate_folds(
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    train_years: int = TRAIN_YEARS,
    test_years: int = TEST_YEARS,
) -> List[Fold]:
    """Rolling fixed-width folds. Slides forward by test_years each step."""
    folds: List[Fold] = []
    cursor = eval_start
    while True:
        train_end = cursor + pd.DateOffset(years=train_years)
        test_end = train_end + pd.DateOffset(years=test_years)
        if test_end > eval_end:
            break
        folds.append(Fold(cursor, train_end, train_end, test_end))
        cursor += pd.DateOffset(years=test_years)
    return folds
```

- [ ] **Step 4: Run fold tests to verify they pass**

Run: `docker compose run --rm ggtrader_live python -m pytest tests/lab/test_wfo.py -v`
Expected: 3 PASSED

- [ ] **Step 5: Write failing tests for `composite_score`**

Add to `tests/lab/test_wfo.py`:

```python
from ggTrader.lab.wfo import composite_score


def test_composite_score_ranking():
    """Combo with best sharpe+sortino and least drawdown scores highest."""
    metrics = [
        {"sharpe": 1.0, "sortino": 1.2, "max_drawdown_pct": -10.0},
        {"sharpe": 0.5, "sortino": 0.6, "max_drawdown_pct": -20.0},
        {"sharpe": 0.2, "sortino": 0.3, "max_drawdown_pct": -30.0},
    ]
    scores = composite_score(metrics)
    assert len(scores) == 3
    assert scores[0] > scores[1] > scores[2]
    # Best combo should get close to 1.0 (all normalized to 1.0)
    # 0.5 * 1.0 + 0.3 * 1.0 - 0.2 * 0.0 = 0.8
    assert abs(scores[0] - 0.8) < 1e-9


def test_composite_score_single_combo():
    """Single combo: min==max for all metrics, all normalized to 0.0, score=0.0."""
    metrics = [{"sharpe": 0.5, "sortino": 0.6, "max_drawdown_pct": -15.0}]
    scores = composite_score(metrics)
    assert len(scores) == 1
    assert scores[0] == 0.0


def test_composite_score_nan_handling():
    """NaN sharpe/sortino treated as worst (floor of range)."""
    metrics = [
        {"sharpe": 1.0, "sortino": 1.0, "max_drawdown_pct": -10.0},
        {"sharpe": float("nan"), "sortino": float("nan"), "max_drawdown_pct": -20.0},
    ]
    scores = composite_score(metrics)
    assert scores[0] > scores[1]
```

- [ ] **Step 6: Run to verify they fail**

Run: `docker compose run --rm ggtrader_live python -m pytest tests/lab/test_wfo.py -v`
Expected: 3 new tests FAIL — `ImportError: cannot import name 'composite_score'`

- [ ] **Step 7: Implement `composite_score`**

Add to `src/ggTrader/lab/wfo.py`:

```python
def _min_max_normalize(values: List[float]) -> List[float]:
    """Min-max scale to [0, 1]. Returns all 0.0 if min == max."""
    lo = min(values)
    hi = max(values)
    if hi == lo:
        return [0.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def composite_score(metrics_list: List[Dict[str, float]]) -> List[float]:
    """Composite rank: 0.5*norm(sharpe) + 0.3*norm(sortino) - 0.2*norm(|maxdd|).

    NaN values are replaced with the worst value in each metric's range.
    """
    sharpes: List[float] = []
    sortinos: List[float] = []
    drawdowns: List[float] = []
    for m in metrics_list:
        sharpes.append(m.get("sharpe", float("nan")))
        sortinos.append(m.get("sortino", float("nan")))
        drawdowns.append(abs(m.get("max_drawdown_pct", 0.0)))

    def _floor_nan(vals: List[float]) -> List[float]:
        finite = [v for v in vals if not math.isnan(v)]
        floor = min(finite) if finite else 0.0
        return [floor if math.isnan(v) else v for v in vals]

    sharpes = _floor_nan(sharpes)
    sortinos = _floor_nan(sortinos)
    drawdowns = _floor_nan(drawdowns)

    ns = _min_max_normalize(sharpes)
    no = _min_max_normalize(sortinos)
    nd = _min_max_normalize(drawdowns)

    return [0.5 * ns[i] + 0.3 * no[i] - 0.2 * nd[i] for i in range(len(metrics_list))]
```

- [ ] **Step 8: Run all tests to verify they pass**

Run: `docker compose run --rm ggtrader_live python -m pytest tests/lab/test_wfo.py -v`
Expected: 6 PASSED

- [ ] **Step 9: Commit**

```bash
git add src/ggTrader/lab/wfo.py tests/lab/test_wfo.py
git commit -m "feat(lab): fold generation and composite scoring for WFO"
```

---

### Task 2: Train-Fold Sweep and OOS Simulation (`run_wfo`)

**Files:**
- Modify: `src/ggTrader/lab/wfo.py`
- Modify: `tests/lab/test_wfo.py`

**Interfaces:**
- Consumes:
  - `Fold`, `generate_folds`, `composite_score` from Task 1
  - `from ggTrader.lab.sweep import split_params, combo_name` — `split_params(combo: Dict) -> tuple[Dict, Dict]`, `combo_name(strategy_name: str, params: Dict) -> str`
  - `from ggTrader.lab.simulate import simulate_signals` — `simulate_signals(targets: Dict[str, SignalTargets], prices: pd.DataFrame, base_config: Dict, ohlcv: pd.DataFrame | None = None) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]`
  - `from ggTrader.lab.metrics import curve_stats` — `curve_stats(curve: pd.Series) -> Dict[str, float]` with keys `sharpe`, `sortino`, `max_drawdown_pct`, `cagr_pct`, `total_return_pct`, `ann_vol_pct`
  - Strategy classes have `sweep_signals(self, combos: list[dict], symbols: list[str], data: pd.DataFrame) -> dict[str, SignalTargets]`
- Produces:
  - `run_wfo(strategy_name, strategy_cls, cfg, ohlcv, spy_close, eval_start, eval_end, market, base_config, grid) -> str` — prints the WFO table and returns it
  - Internal: `_train_fold_sweep(...)` helper that runs all combos on a train window and returns per-combo metrics + the winner

- [ ] **Step 1: Write failing integration test**

Add to `tests/lab/test_wfo.py`:

```python
import numpy as np

from ggTrader.lab.strategy import LabConfig, SignalTargets
from ggTrader.lab.wfo import run_wfo


def _ohlcv(symbols, n):
    """Build a (symbol, field) MultiIndex OHLCV DataFrame with trending prices."""
    idx = pd.date_range("2014-01-01", periods=n, freq="B", tz="UTC")
    frames = {}
    for i, s in enumerate(symbols):
        base = 100.0 + i * 10
        close = base * (1 + 0.0003 * np.arange(n))
        frames[s] = pd.DataFrame(
            {
                "open": close * 0.999,
                "high": close * 1.005,
                "low": close * 0.995,
                "close": close,
                "volume": np.full(n, 1e6),
            },
            index=idx,
        )
    return pd.concat(frames, axis=1)


class _TinySignal:
    """Minimal signal strategy for testing: buy on bar 2, never exit."""

    name = "tiny"
    target_kind = "signals"

    def __init__(self, cfg):
        self.cfg = cfg

    @classmethod
    def sweep_params(cls):
        return {"param_a": [1, 2]}

    def sweep_signals(self, combos, symbols, data):
        from ggTrader.lab.sweep import combo_name

        close = pd.concat(
            {s: data[s]["close"] for s in symbols if s in data.columns.get_level_values(0)},
            axis=1,
        )
        result = {}
        for combo in combos:
            entries = pd.DataFrame(False, index=close.index, columns=close.columns)
            exits = pd.DataFrame(False, index=close.index, columns=close.columns)
            # Buy on bar 2 — after warmup
            entries.iloc[2] = True
            key = combo_name(self.name, combo)
            result[key] = SignalTargets(entries=entries.astype(bool), exits=exits.astype(bool))
        return result


def test_run_wfo_integration():
    """Full WFO with tiny strategy: 2014-2020 data, 3yr/1yr → 3 folds."""
    symbols = ["X", "Y"]
    n = 252 * 7  # ~7 years of daily bars
    ohlcv = _ohlcv(symbols, n)
    spy_close = ohlcv["X"]["close"].copy()  # use X as SPY proxy
    cfg = LabConfig(top_n=10, lookback=20, skip=5, min_history_bars=10)
    base_config = {
        "START_CASH": 10000.0,
        "FEES": 0.0,
        "SLIPPAGE": 0.0,
        "FREQ": "1d",
        "SIGNAL_POSITION_SIZE": 0.5,
    }
    eval_start = ohlcv.index[0]
    eval_end = ohlcv.index[-1]
    grid = [{"param_a": 1}, {"param_a": 2}]

    output = run_wfo(
        "tiny",
        _TinySignal,
        cfg,
        ohlcv,
        spy_close,
        str(eval_start.date()),
        str(eval_end.date()),
        "test",
        base_config,
        grid,
    )
    assert "WFO:" in output
    assert "OOS Aggregate:" in output
    assert "Recommended Live Params" in output
    # Should have at least 1 fold
    assert "Fold" in output
```

- [ ] **Step 2: Run to verify it fails**

Run: `docker compose build ggtrader_live && docker compose run --rm ggtrader_live python -m pytest tests/lab/test_wfo.py::test_run_wfo_integration -v`
Expected: FAIL — `ImportError: cannot import name 'run_wfo'`

- [ ] **Step 3: Implement the train-fold helper `_sweep_fold`**

Add to `src/ggTrader/lab/wfo.py`:

```python
from ggTrader.lab.metrics import curve_stats
from ggTrader.lab.simulate import simulate_signals
from ggTrader.lab.strategy import LabConfig, SignalTargets
from ggTrader.lab.sweep import combo_name, split_params


def _sweep_fold(
    strategy_name: str,
    strat_instance: Any,
    ohlcv: pd.DataFrame,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    base_config: Dict[str, Any],
    grid: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Run all combos on a single time window and return per-combo metrics.

    Signal generation uses all data up to window_end (for EMA warmup).
    Scoring uses only [window_start, window_end).
    Returns list of dicts with keys: 'combo', 'params', and all curve_stats keys.
    """
    from collections import defaultdict

    ohlcv_window = ohlcv.loc[:window_end]
    symbols = sorted(ohlcv_window.columns.get_level_values(0).unique())
    prices = pd.concat(
        {s: ohlcv_window[s]["close"] for s in symbols},
        axis=1,
    )

    start_cash = float(base_config["START_CASH"])

    # Group by stop config (same logic as sweep.py)
    stop_groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for combo in grid:
        _, stop_p = split_params(combo)
        stop_key = tuple(sorted(stop_p.items()))
        stop_groups[stop_key].append(combo)

    all_eq: Dict[str, pd.Series] = {}
    for stop_key, group_combos in stop_groups.items():
        stop_config = dict(stop_key)
        signal_combos = [split_params(c)[0] for c in group_combos]
        seen: set = set()
        unique_signal: List[Dict[str, Any]] = []
        for sc in signal_combos:
            k = tuple(sorted(sc.items()))
            if k not in seen:
                seen.add(k)
                unique_signal.append(sc)
        targets = strat_instance.sweep_signals(unique_signal, symbols, ohlcv_window)
        group_targets: Dict[str, SignalTargets] = {}
        for combo in group_combos:
            signal_p, _ = split_params(combo)
            signal_key = combo_name(strategy_name, signal_p)
            full_key = combo_name(strategy_name, combo)
            group_targets[full_key] = targets[signal_key]

        sim_config = {**base_config, **stop_config}
        ohlcv_arg = ohlcv_window if "atr_mult" in stop_config else None
        _rets, eq, _diag = simulate_signals(
            group_targets, prices, sim_config, ohlcv=ohlcv_arg
        )
        for key in group_targets:
            all_eq[key] = eq[key]

    # Score each combo over the scoring window only
    results: List[Dict[str, Any]] = []
    for key, eq_series in all_eq.items():
        eq_window = eq_series.loc[window_start:window_end].dropna()
        if len(eq_window) < 2:
            continue
        # Rescale to start at start_cash for consistent metrics
        eq_scaled = start_cash * (eq_window / eq_window.iloc[0])
        metrics = curve_stats(eq_scaled)
        combo_params = next(c for c in grid if combo_name(strategy_name, c) == key)
        results.append({"combo": key, "params": combo_params, **metrics})
    return results
```

- [ ] **Step 4: Implement `run_wfo`**

Add to `src/ggTrader/lab/wfo.py`:

```python
def run_wfo(
    strategy_name: str,
    strategy_cls: Type,
    cfg: LabConfig,
    ohlcv: pd.DataFrame,
    spy_close: pd.Series,
    eval_start: str,
    eval_end: str,
    market: str,
    base_config: Dict[str, Any],
    grid: List[Dict[str, Any]],
) -> str:
    """Main WFO entry point: fold, train, test, concatenate, report."""
    eval_start_ts = pd.Timestamp(eval_start, tz="UTC")
    eval_end_ts = pd.Timestamp(eval_end, tz="UTC")
    folds = generate_folds(eval_start_ts, eval_end_ts)
    if not folds:
        return f"WFO: {strategy_name} | no valid folds (need >= {TRAIN_YEARS + TEST_YEARS} years)"

    strat_instance = strategy_cls(cfg)
    start_cash = float(base_config["START_CASH"])
    fold_results: List[Dict[str, Any]] = []
    oos_curves: List[pd.Series] = []
    oos_running_value = start_cash
    fold_winners: List[Dict[str, Any]] = []

    for i, fold in enumerate(folds):
        # Train: sweep all combos on train window
        train_metrics = _sweep_fold(
            strategy_name, strat_instance, ohlcv,
            fold.train_start, fold.train_end, base_config, grid,
        )
        if not train_metrics:
            continue

        # Pick winner by composite score
        scores = composite_score(train_metrics)
        best_idx = max(range(len(scores)), key=lambda j: scores[j])
        winner = train_metrics[best_idx]

        # Test: simulate winner only on data up to test_end, score test window
        winner_grid = [winner["params"]]
        test_metrics = _sweep_fold(
            strategy_name, strat_instance, ohlcv,
            fold.test_start, fold.test_end, base_config, winner_grid,
        )
        oos_score = 0.0
        oos_sharpe = float("nan")
        if test_metrics:
            oos_score = composite_score(test_metrics)[0]
            oos_sharpe = test_metrics[0].get("sharpe", float("nan"))

            # Build continuous OOS equity curve
            test_ohlcv = ohlcv.loc[:fold.test_end]
            symbols = sorted(test_ohlcv.columns.get_level_values(0).unique())
            prices = pd.concat({s: test_ohlcv[s]["close"] for s in symbols}, axis=1)
            signal_combos = [split_params(winner["params"])[0]]
            _, stop_p = split_params(winner["params"])
            targets = strat_instance.sweep_signals(signal_combos, symbols, test_ohlcv)
            key = combo_name(strategy_name, signal_combos[0])
            full_key = combo_name(strategy_name, winner["params"])
            sim_config = {**base_config, **stop_p}
            ohlcv_arg = test_ohlcv if "atr_mult" in stop_p else None
            _r, eq, _d = simulate_signals(
                {full_key: targets[key]}, prices, sim_config, ohlcv=ohlcv_arg
            )
            eq_test = eq[full_key].loc[fold.test_start:fold.test_end].dropna()
            if len(eq_test) > 0:
                normalized = oos_running_value * (eq_test / eq_test.iloc[0])
                oos_curves.append(normalized)
                oos_running_value = float(normalized.iloc[-1])

        fold_results.append({
            "fold_num": i + 1,
            "train_start": fold.train_start,
            "train_end": fold.train_end,
            "test_start": fold.test_start,
            "test_end": fold.test_end,
            "winner_combo": winner["combo"],
            "winner_params": winner["params"],
            "train_score": scores[best_idx],
            "oos_score": oos_score,
        })
        fold_winners.append(winner)

    # Concatenate OOS curves and score
    if oos_curves:
        oos_equity = pd.concat(oos_curves)
        oos_equity = oos_equity[~oos_equity.index.duplicated(keep="last")]
        oos_metrics = curve_stats(oos_equity)
        spy_oos = spy_close.reindex(oos_equity.index).ffill().dropna()
        if len(spy_oos) > 1:
            spy_curve = start_cash * (spy_oos / spy_oos.iloc[0])
            spy_metrics = curve_stats(spy_curve)
        else:
            spy_metrics = {"sharpe": float("nan"), "cagr_pct": float("nan"), "max_drawdown_pct": float("nan")}
    else:
        oos_metrics = {"sharpe": float("nan"), "cagr_pct": float("nan"), "max_drawdown_pct": float("nan")}
        spy_metrics = oos_metrics.copy()

    # Recommended live params
    live = select_live_params(
        strategy_name, strategy_cls, cfg, ohlcv, eval_end, base_config, grid, fold_winners,
    )

    table = format_wfo_table(
        fold_results, oos_metrics, spy_metrics, live, strategy_name, len(grid), len(folds),
    )
    print(table)
    return table
```

- [ ] **Step 5: Implement `select_live_params`**

Add to `src/ggTrader/lab/wfo.py`:

```python
def select_live_params(
    strategy_name: str,
    strategy_cls: Type,
    cfg: LabConfig,
    ohlcv: pd.DataFrame,
    eval_end: str,
    base_config: Dict[str, Any],
    grid: List[Dict[str, Any]],
    fold_winners: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Train on the most recent TRAIN_YEARS window and pick the composite winner."""
    eval_end_ts = pd.Timestamp(eval_end, tz="UTC")
    live_train_start = eval_end_ts - pd.DateOffset(years=TRAIN_YEARS)
    strat_instance = strategy_cls(cfg)

    train_metrics = _sweep_fold(
        strategy_name, strat_instance, ohlcv,
        live_train_start, eval_end_ts, base_config, grid,
    )
    if not train_metrics:
        return {"combo": "none", "params": {}, "train_metrics": {}, "stability": 0}

    scores = composite_score(train_metrics)
    best_idx = max(range(len(scores)), key=lambda j: scores[j])
    winner = train_metrics[best_idx]

    # Stability: count how many WFO folds selected the same combo
    stability = sum(1 for fw in fold_winners if fw["combo"] == winner["combo"])

    return {
        "combo": winner["combo"],
        "params": winner["params"],
        "train_metrics": {
            k: winner[k] for k in ("sharpe", "cagr_pct", "max_drawdown_pct")
            if k in winner
        },
        "stability": stability,
        "train_start": live_train_start,
        "train_end": eval_end_ts,
    }
```

- [ ] **Step 6: Implement `format_wfo_table`**

Add to `src/ggTrader/lab/wfo.py`:

```python
def format_wfo_table(
    fold_results: List[Dict[str, Any]],
    oos_metrics: Dict[str, float],
    spy_metrics: Dict[str, float],
    live_params: Dict[str, Any],
    strategy_name: str,
    n_combos: int,
    n_folds: int,
) -> str:
    """Render per-fold table + OOS aggregate + recommended live params."""
    lines = [
        f"WFO: {strategy_name} | {n_combos} combos x {n_folds} folds"
        f" | rolling {TRAIN_YEARS}yr/{TEST_YEARS}yr",
        "",
        f"{'Fold':<6}{'Train Window':<20}{'Test Window':<20}"
        f"{'Winner':<36}{'Train':>7}{'OOS':>7}",
        "─" * 96,
    ]
    for r in fold_results:
        ts = r["train_start"].strftime("%Y-%m")
        te = r["train_end"].strftime("%Y-%m")
        os_ = r["test_start"].strftime("%Y-%m")
        oe = r["test_end"].strftime("%Y-%m")
        # Shorten winner name: strip strategy prefix
        short = r["winner_combo"].replace(f"{strategy_name}__", "")
        if len(short) > 34:
            short = short[:31] + "..."
        lines.append(
            f"{r['fold_num']:<6}{ts} → {te:<13}{os_} → {oe:<13}"
            f"{short:<36}{r['train_score']:>7.2f}{r['oos_score']:>7.2f}"
        )

    lines.append("")
    lines.append(
        f"OOS Aggregate: Sharpe {oos_metrics.get('sharpe', float('nan')):.2f}"
        f" | CAGR {oos_metrics.get('cagr_pct', float('nan')):.1f}%"
        f" | MaxDD {oos_metrics.get('max_drawdown_pct', float('nan')):.1f}%"
    )
    lines.append(
        f"SPY baseline:  Sharpe {spy_metrics.get('sharpe', float('nan')):.2f}"
        f" | CAGR {spy_metrics.get('cagr_pct', float('nan')):.1f}%"
        f" | MaxDD {spy_metrics.get('max_drawdown_pct', float('nan')):.1f}%"
    )

    # Recommended live params
    lines.append("")
    lines.append("── Recommended Live Params " + "─" * 71)
    ts = live_params.get("train_start")
    te = live_params.get("train_end")
    ts_str = ts.strftime("%Y-%m") if ts else "?"
    te_str = te.strftime("%Y-%m") if te else "?"
    lines.append(f"Train window: {ts_str} → {te_str}")
    lines.append(f"Winner:       {live_params.get('combo', 'none')}")
    tm = live_params.get("train_metrics", {})
    lines.append(
        f"Train Sharpe: {tm.get('sharpe', float('nan')):.2f}"
        f" | CAGR {tm.get('cagr_pct', float('nan')):.1f}%"
        f" | MaxDD {tm.get('max_drawdown_pct', float('nan')):.1f}%"
    )
    lines.append(
        f"Stability:    selected in {live_params.get('stability', 0)}/{len(fold_results)} folds"
    )

    return "\n".join(lines)
```

- [ ] **Step 7: Run the integration test**

Run: `docker compose build ggtrader_live && docker compose run --rm ggtrader_live python -m pytest tests/lab/test_wfo.py -v`
Expected: all tests PASS (6 from Task 1 + 1 integration = 7)

- [ ] **Step 8: Add test for `select_live_params` stability counting**

Add to `tests/lab/test_wfo.py`:

```python
from ggTrader.lab.wfo import select_live_params


def test_select_live_params_uses_recent_window():
    """Live params trained on most recent 3yr; stability counts matching fold winners."""
    symbols = ["X", "Y"]
    n = 252 * 7
    ohlcv = _ohlcv(symbols, n)
    cfg = LabConfig(top_n=10, lookback=20, skip=5, min_history_bars=10)
    base_config = {
        "START_CASH": 10000.0,
        "FEES": 0.0,
        "SLIPPAGE": 0.0,
        "FREQ": "1d",
        "SIGNAL_POSITION_SIZE": 0.5,
    }
    grid = [{"param_a": 1}, {"param_a": 2}]
    # Simulate fold_winners that all picked param_a=1
    fold_winners = [
        {"combo": "tiny__param_a1", "params": {"param_a": 1}},
        {"combo": "tiny__param_a1", "params": {"param_a": 1}},
    ]
    eval_end = str(ohlcv.index[-1].date())
    result = select_live_params(
        "tiny", _TinySignal, cfg, ohlcv, eval_end, base_config, grid, fold_winners,
    )
    assert "combo" in result
    assert "stability" in result
    assert isinstance(result["stability"], int)
    assert "train_metrics" in result
    assert "sharpe" in result["train_metrics"]
```

- [ ] **Step 9: Run all tests**

Run: `docker compose run --rm ggtrader_live python -m pytest tests/lab/test_wfo.py -v`
Expected: 8 PASSED

- [ ] **Step 10: Commit**

```bash
git add src/ggTrader/lab/wfo.py tests/lab/test_wfo.py
git commit -m "feat(lab): run_wfo with train/test folds, OOS aggregation, and live params"
```

---

### Task 3: CLI Integration (`--wfo` flag)

**Files:**
- Modify: `src/ggTrader/lab/cli.py:17-43` (argument parser)
- Modify: `src/ggTrader/lab/cli.py:46-113` (run_lab function)
- Modify: `tests/lab/test_wfo.py` (add CLI tests)

**Interfaces:**
- Consumes:
  - `from ggTrader.lab.wfo import run_wfo` (from Task 2)
  - `from ggTrader.lab.sweep import build_grid` — existing
  - `_parse_sweep_params` — existing in cli.py
  - `SIGNAL_STRATEGY_NAMES` — existing tuple of signal strategy names
- Produces:
  - `--wfo` CLI flag, mutually exclusive with `--sweep`
  - Validation: `--wfo` with a weight strategy raises `SystemExit`

- [ ] **Step 1: Write failing CLI tests**

Add to `tests/lab/test_wfo.py`:

```python
from ggTrader.lab.cli import build_arg_parser


def test_cli_parser_accepts_wfo_flag():
    p = build_arg_parser()
    args = p.parse_args(["--strategy", "ema_cross", "--wfo"])
    assert args.wfo is True
    assert args.sweep is False


def test_cli_parser_wfo_and_sweep_mutually_exclusive():
    p = build_arg_parser()
    with pytest.raises(SystemExit):
        p.parse_args(["--strategy", "ema_cross", "--wfo", "--sweep"])


def test_cli_parser_wfo_with_sweep_param():
    p = build_arg_parser()
    args = p.parse_args([
        "--strategy", "ema_cross", "--wfo",
        "--sweep-param", "atr_mult=1.5,2.0",
    ])
    assert args.wfo is True
    assert len(args.sweep_param) == 1
```

- [ ] **Step 2: Run to verify they fail**

Run: `docker compose build ggtrader_live && docker compose run --rm ggtrader_live python -m pytest tests/lab/test_wfo.py::test_cli_parser_accepts_wfo_flag -v`
Expected: FAIL — `AttributeError: ... object has no attribute 'wfo'`

- [ ] **Step 3: Add `--wfo` flag to argument parser**

Modify `src/ggTrader/lab/cli.py`. Replace the `--sweep` argument definition and add a mutually exclusive group:

Replace this block in `build_arg_parser()`:

```python
    p.add_argument(
        "--sweep",
        action="store_true",
        default=False,
        help="Run parameter sweep instead of single walk-forward.",
    )
```

With:

```python
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--sweep",
        action="store_true",
        default=False,
        help="Run parameter sweep instead of single walk-forward.",
    )
    mode.add_argument(
        "--wfo",
        action="store_true",
        default=False,
        help="Walk-forward optimization: rolling train/test folds with OOS scoring.",
    )
```

- [ ] **Step 4: Add WFO branch to `run_lab`**

In `src/ggTrader/lab/cli.py`, add the `--wfo` handling after the existing `--sweep` block (after line 94, before the single-run `if args.strategy in SIGNAL_STRATEGY_NAMES:` block).

Insert this block between the sweep return and the single-run code:

```python
    if args.wfo:
        from ggTrader.lab.strategies.signals import EmaCrossSignal, WfoTournamentSignal, SIGNAL_STRATEGY_NAMES as _SIG_NAMES
        from ggTrader.lab.sweep import build_grid
        from ggTrader.lab.wfo import run_wfo

        if args.strategy not in _SIG_NAMES:
            raise SystemExit(f"--wfo only supports signal strategies: {_SIG_NAMES}")
        cls_map = {
            "ema_cross": EmaCrossSignal,
            "wfo_tournament": WfoTournamentSignal,
        }
        strategy_cls = cls_map[args.strategy]
        overrides = _parse_sweep_params(args.sweep_param)
        grid = build_grid(strategy_cls, overrides=overrides if overrides else None)
        print(f"WFO: {args.strategy} | {len(grid)} param combos")
        result = run_wfo(
            args.strategy,
            strategy_cls,
            cfg,
            ohlcv,
            spy_close,
            eval_start=str(eval_start.date()),
            eval_end=str(eval_end.date()),
            market=args.market,
            base_config=dict(STOCK_BASE_CONFIG),
            grid=grid,
        )
        return result
```

- [ ] **Step 5: Run CLI tests**

Run: `docker compose build ggtrader_live && docker compose run --rm ggtrader_live python -m pytest tests/lab/test_wfo.py -v`
Expected: all 11 tests PASS

- [ ] **Step 6: Run the full test suite to check for regressions**

Run: `docker compose run --rm ggtrader_live python -m pytest tests/ -v`
Expected: all tests PASS (existing ~77 + 11 new)

- [ ] **Step 7: Commit**

```bash
git add src/ggTrader/lab/cli.py tests/lab/test_wfo.py
git commit -m "feat(lab): --wfo CLI flag for walk-forward optimization"
```
