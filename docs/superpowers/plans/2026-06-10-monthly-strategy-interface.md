# Strategy-Agnostic Monthly Walk-Forward Harness — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generalize `research/monthly_walkforward.py` so any `MonthlyStrategy` (select + simulate) plugs into the honest harness; ship the WFO tournament as plug-in #1 (behavior-identical) plus `CrossSectionalMomentum` and `DualMomentum`.

**Architecture:** New module `research/monthly_strategies.py` holds a `MonthlyStrategy` Protocol, a shared `simulate_hold_weights()` simulator, and the three strategies. The harness keeps data loading, PIT eligibility, month loop, checkpoints, leak check, stitching, and benchmarking; it passes `ohlcv.loc[:asof]` into `select()` so strategies cannot peek. Diagnostics gain `avg_exposure`.

**Tech Stack:** Python 3.11, vectorbt 0.28.5 OSS, pandas, pytest. Spec: `docs/superpowers/specs/2026-06-10-monthly-strategy-interface-design.md`.

**Conventions:** repo uses ruff (line length 100), absolute imports from `src`, tests run as `source .venv/bin/activate && python -m pytest tests/ -m 'not integration' -q`. A long-running research process is executing old code from memory — do not touch `results/monthly_wf/sp500_monthly_v1/` or kill any process.

**File map:**
- Create: `src/ggTrader/research/monthly_strategies.py` (protocol, helpers, 3 strategies, factory)
- Create: `tests/test_monthly_strategies.py`
- Modify: `src/ggTrader/research/monthly_walkforward.py` (harness slimming; `select_for_month`/`simulate_forward_month`/`_select_worker` move out)
- Modify: `scripts/sp500_monthly_walkforward.py` (`--strategy`, momentum knobs)

---

### Task 1: Test helper + `CrossSectionalMomentum.select`

**Files:**
- Create: `tests/test_monthly_strategies.py`
- Create: `src/ggTrader/research/monthly_strategies.py`

- [x] **Step 1: Write the failing tests** (new file `tests/test_monthly_strategies.py`)

```python
"""Unit tests for pluggable monthly strategies (synthetic data, no network)."""

import json

import numpy as np
import pandas as pd

from ggTrader.research.equity_wfo import STOCK_BASE_CONFIG
from ggTrader.research.monthly_strategies import (
    CrossSectionalMomentum,
    DualMomentum,
    simulate_hold_weights,
)
from ggTrader.research.monthly_walkforward import MonthlyHarnessConfig


def make_ohlcv(prices: dict) -> pd.DataFrame:
    """Build a (symbol, field) MultiIndex OHLCV frame from close-price series."""
    frames = {}
    for sym, close in prices.items():
        frames[sym] = pd.DataFrame(
            {
                "open": close.shift(1).fillna(close.iloc[0]),
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": np.full(len(close), 1e6),
            },
            index=close.index,
        )
    out = pd.concat(frames, axis=1)
    out.columns = out.columns.set_names(["symbol", "field"])
    return out


def _idx(n: int, start: str = "2020-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="B", tz="UTC")


def test_momentum_ranks_strongest_first_with_equal_weights():
    idx = _idx(300)
    ohlcv = make_ohlcv(
        {
            "UP": pd.Series(np.linspace(10, 30, 300), index=idx),
            "FLAT": pd.Series(np.full(300, 20.0), index=idx),
            "DOWN": pd.Series(np.linspace(30, 10, 300), index=idx),
        }
    )
    cfg = MonthlyHarnessConfig(top_n=3)
    strat = CrossSectionalMomentum(cfg, STOCK_BASE_CONFIG)
    sels = strat.select(idx[-1], ohlcv, ["UP", "FLAT", "DOWN"])
    assert [s["symbol"] for s in sels] == ["UP", "FLAT", "DOWN"]
    assert sels[0]["momentum"] > 0.0 > sels[2]["momentum"]
    assert all(abs(s["weight"] - 1.0 / 3.0) < 1e-12 for s in sels)
    json.dumps(sels)  # selections must be JSON-able for checkpoints


def test_momentum_skip_window_is_actually_skipped():
    idx = _idx(300)
    base = pd.Series(np.linspace(10, 20, 300), index=idx)
    jumped = base.copy()
    jumped.iloc[-21:] = jumped.iloc[-21:] * 2.0  # move entirely inside skip window
    ohlcv = make_ohlcv({"BASE": base, "JUMP": jumped})
    cfg = MonthlyHarnessConfig(top_n=2)
    strat = CrossSectionalMomentum(cfg, STOCK_BASE_CONFIG, lookback=252, skip=21)
    sels = {s["symbol"]: s["momentum"] for s in strat.select(idx[-1], ohlcv, ["BASE", "JUMP"])}
    assert sels["BASE"] == sels["JUMP"]


def test_momentum_respects_top_n_and_short_history():
    idx = _idx(300)
    short_idx = idx[-100:]  # < lookback+1 bars -> ineligible
    ohlcv = make_ohlcv(
        {
            "A": pd.Series(np.linspace(10, 40, 300), index=idx),
            "B": pd.Series(np.linspace(10, 30, 300), index=idx),
            "C": pd.Series(np.linspace(10, 20, 300), index=idx),
            "NEW": pd.Series(np.linspace(10, 99, 100), index=short_idx),
        }
    )
    cfg = MonthlyHarnessConfig(top_n=2)
    strat = CrossSectionalMomentum(cfg, STOCK_BASE_CONFIG)
    sels = strat.select(idx[-1], ohlcv, ["A", "B", "C", "NEW"])
    assert [s["symbol"] for s in sels] == ["A", "B"]
    assert all(abs(s["weight"] - 0.5) < 1e-12 for s in sels)
```

- [x] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tests/test_monthly_strategies.py -q`
Expected: FAIL/ERROR with `ModuleNotFoundError: No module named 'ggTrader.research.monthly_strategies'`

- [x] **Step 3: Implement** (new file `src/ggTrader/research/monthly_strategies.py`)

```python
"""Pluggable monthly strategies for the honest walk-forward harness.

A MonthlyStrategy maps (data <= T, point-in-time eligible universe) to
next-month selections, then simulates the forward month with those frozen
selections. The select/simulate split is what makes the generic leak check
possible: ``select`` must be a pure function of data <= T.

The harness (research/monthly_walkforward.py) guarantees ``select`` only ever
receives data truncated to <= asof.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np
import pandas as pd
import vectorbt as vbt

from ggTrader.core.fast_backtest import FastBacktest
from ggTrader.core.orchestrator_utils import _to_native
from ggTrader.research.equity_wfo import grid_books, wfo_strategy_tournament_one_stock


class MonthlyStrategy(Protocol):
    """Contract for a strategy runnable by run_monthly_walkforward."""

    name: str

    def select(
        self, asof: pd.Timestamp, ohlcv: pd.DataFrame, eligible: List[str]
    ) -> List[Dict[str, Any]]:
        """JSON-able selection records (each with at least "symbol"); data <= asof."""
        ...

    def simulate(
        self,
        ohlcv: pd.DataFrame,
        selections: List[Dict[str, Any]],
        asof: pd.Timestamp,
        month_end: pd.Timestamp,
    ) -> Tuple[pd.Series, Dict[str, Any]]:
        """Daily portfolio returns for (asof, month_end] plus diagnostics."""
        ...


def _portfolio_exposure(pf) -> pd.Series:
    """Fraction of capital deployed per bar: 1 - cash/value (grouped portfolio)."""
    cash, value = pf.cash(), pf.value()
    if isinstance(cash, pd.DataFrame):
        cash = cash.iloc[:, 0]
    if isinstance(value, pd.DataFrame):
        value = value.iloc[:, 0]
    return 1.0 - cash / value


class CrossSectionalMomentum:
    """12-1 cross-sectional momentum: top-N by trailing return, equal weight."""

    name = "xs_momentum"

    def __init__(
        self,
        cfg,
        base_config: Dict[str, Any],
        lookback: int = 252,
        skip: int = 21,
    ) -> None:
        self.cfg = cfg
        self.base_config = base_config
        self.lookback = lookback
        self.skip = skip

    def select(
        self, asof: pd.Timestamp, ohlcv: pd.DataFrame, eligible: List[str]
    ) -> List[Dict[str, Any]]:
        scores: Dict[str, float] = {}
        for sym in eligible:
            closes = ohlcv[sym]["close"].dropna()
            if len(closes) < self.lookback + 1:
                continue
            past = float(closes.iloc[-(self.lookback + 1)])
            recent = float(closes.iloc[-(self.skip + 1)])
            if past <= 0.0 or not np.isfinite(past) or not np.isfinite(recent):
                continue
            scores[sym] = recent / past - 1.0
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[: self.cfg.top_n]
        if not ranked:
            return []
        weight = 1.0 / len(ranked)
        return [{"symbol": s, "weight": weight, "momentum": m} for s, m in ranked]

    def simulate(
        self,
        ohlcv: pd.DataFrame,
        selections: List[Dict[str, Any]],
        asof: pd.Timestamp,
        month_end: pd.Timestamp,
    ) -> Tuple[pd.Series, Dict[str, Any]]:
        weights = {s["symbol"]: float(s["weight"]) for s in selections}
        return simulate_hold_weights(ohlcv, weights, asof, month_end, self.base_config)
```

(`simulate_hold_weights`, `DualMomentum`, `WfoTournamentStrategy`, and
`build_strategy` are added in Tasks 2-4 and 6; for this step also add a
temporary stub so the import in the test file resolves:)

```python
def simulate_hold_weights(ohlcv, weights, asof, month_end, base_config):
    raise NotImplementedError  # implemented in Task 2


class DualMomentum(CrossSectionalMomentum):
    name = "dual_momentum"  # behavior added in Task 3
```

- [x] **Step 4: Run the three Task-1 tests; verify they pass**

Run: `source .venv/bin/activate && python -m pytest tests/test_monthly_strategies.py -q`
Expected: 3 passed

- [x] **Step 5: Commit**

```bash
git add src/ggTrader/research/monthly_strategies.py tests/test_monthly_strategies.py
git commit -m "feat(research): MonthlyStrategy protocol + cross-sectional momentum select"
```

---

### Task 2: `simulate_hold_weights` (buy-and-hold-for-the-month simulator)

**Files:**
- Modify: `src/ggTrader/research/monthly_strategies.py` (replace the stub)
- Modify: `tests/test_monthly_strategies.py` (append tests)

- [x] **Step 1: Write the failing tests** (append to `tests/test_monthly_strategies.py`)

```python
def test_simulate_hold_weights_matches_hand_computed_returns():
    idx = _idx(42, start="2021-01-01")
    a = pd.Series(100.0 * 1.01 ** np.arange(42), index=idx)  # +1%/day
    b = pd.Series(np.full(42, 50.0), index=idx)  # flat
    ohlcv = make_ohlcv({"A": a, "B": b})
    asof, month_end = idx[20], idx[-1]
    frictionless = {**STOCK_BASE_CONFIG, "FEES": 0.0, "SLIPPAGE": 0.0}
    rets, diags = simulate_hold_weights(ohlcv, {"A": 0.5, "B": 0.5}, asof, month_end, frictionless)

    # Buys at the close of the first forward bar (idx[21]); B is flat, so the
    # portfolio return is exactly half of A's appreciation from that bar.
    expected_total = 0.5 * (a.iloc[-1] / a.loc[idx[21]] - 1.0)
    total = float((1.0 + rets).prod() - 1.0)
    assert abs(total - expected_total) < 1e-9
    assert rets.index.min() > asof and rets.index.max() <= month_end
    assert diags["n_positions"] == 2
    assert 0.90 <= diags["avg_exposure"] <= 1.01


def test_simulate_hold_weights_empty_inputs_are_flat():
    idx = _idx(42, start="2021-01-01")
    ohlcv = make_ohlcv({"A": pd.Series(np.full(42, 10.0), index=idx)})
    rets, diags = simulate_hold_weights(ohlcv, {}, idx[20], idx[-1], STOCK_BASE_CONFIG)
    assert rets.empty
    assert diags["n_positions"] == 0
```

- [x] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tests/test_monthly_strategies.py -q`
Expected: 2 new tests FAIL with `NotImplementedError` (3 old pass)

- [x] **Step 3: Replace the stub with the implementation**

```python
def simulate_hold_weights(
    ohlcv: pd.DataFrame,
    weights: Dict[str, float],
    asof: pd.Timestamp,
    month_end: pd.Timestamp,
    base_config: Dict[str, Any],
) -> Tuple[pd.Series, Dict[str, Any]]:
    """Buy target weights at the first bar after ``asof``, hold to ``month_end``.

    Symbols with no price at the first forward bar are dropped (their weight
    stays in cash). Mid-month gaps are forward-filled.
    """
    empty = pd.Series(dtype=float)
    month_mask = (ohlcv.index > asof) & (ohlcv.index <= month_end)
    if not month_mask.any() or not weights:
        return empty, {"n_positions": 0, "n_trades": 0, "avg_exposure": 0.0}

    have = set(ohlcv.columns.get_level_values(0))
    close = pd.concat(
        {s: ohlcv[s]["close"] for s in weights if s in have}, axis=1
    ).loc[month_mask].ffill()
    close = close.dropna(axis=1)  # NaN after ffill == no price at month start
    if close.shape[1] == 0:
        return empty, {"n_positions": 0, "n_trades": 0, "avg_exposure": 0.0}

    size = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    size.iloc[0] = [weights[s] for s in close.columns]
    pf = vbt.Portfolio.from_orders(
        close=close,
        size=size,
        size_type="targetpercent",
        init_cash=float(base_config["START_CASH"]),
        fees=float(base_config["FEES"]),
        slippage=float(base_config["SLIPPAGE"]),
        freq=base_config["FREQ"],
        cash_sharing=True,
        group_by=np.full(close.shape[1], 0),
        call_seq="auto",
    ).copy()

    returns = pf.returns()
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]
    diags = {
        "n_positions": int(close.shape[1]),
        "n_trades": int(pf.trades.count().sum()),
        "avg_exposure": float(_portfolio_exposure(pf).mean()),
        "month_return_pct": float((1.0 + returns).prod() - 1.0) * 100,
    }
    return returns, diags
```

- [x] **Step 4: Run tests; verify all pass**

Run: `source .venv/bin/activate && python -m pytest tests/test_monthly_strategies.py -q`
Expected: 5 passed

- [x] **Step 5: Commit**

```bash
git add src/ggTrader/research/monthly_strategies.py tests/test_monthly_strategies.py
git commit -m "feat(research): hold-weights month simulator with exposure diagnostics"
```

---

### Task 3: `DualMomentum`

**Files:**
- Modify: `src/ggTrader/research/monthly_strategies.py`
- Modify: `tests/test_monthly_strategies.py` (append)

- [x] **Step 1: Write the failing tests**

```python
def test_dual_momentum_drops_only_negative_momentum():
    idx = _idx(300)
    ohlcv = make_ohlcv(
        {
            "UP": pd.Series(np.linspace(10, 30, 300), index=idx),
            "FLAT": pd.Series(np.full(300, 20.0), index=idx),
            "DOWN": pd.Series(np.linspace(30, 10, 300), index=idx),
        }
    )
    cfg = MonthlyHarnessConfig(top_n=3)
    strat = DualMomentum(cfg, STOCK_BASE_CONFIG)
    sels = strat.select(idx[-1], ohlcv, ["UP", "FLAT", "DOWN"])
    assert [s["symbol"] for s in sels] == ["UP", "FLAT"]  # FLAT momentum == 0.0 kept
    # weights are NOT renormalized — DOWN's slot stays in cash
    assert all(abs(s["weight"] - 1.0 / 3.0) < 1e-12 for s in sels)


def test_dual_momentum_all_negative_is_flat_month():
    idx = _idx(300)
    ohlcv = make_ohlcv(
        {
            "D1": pd.Series(np.linspace(30, 10, 300), index=idx),
            "D2": pd.Series(np.linspace(40, 20, 300), index=idx),
        }
    )
    cfg = MonthlyHarnessConfig(top_n=2)
    strat = DualMomentum(cfg, STOCK_BASE_CONFIG)
    sels = strat.select(idx[-60], ohlcv.loc[: idx[-60]], ["D1", "D2"])
    assert sels == []
    rets, diags = strat.simulate(ohlcv, sels, idx[-60], idx[-1])
    assert rets.empty and diags["n_positions"] == 0
```

- [x] **Step 2: Run; verify the first test fails** (DualMomentum currently inherits select unchanged)

Run: `source .venv/bin/activate && python -m pytest tests/test_monthly_strategies.py -q`
Expected: `test_dual_momentum_drops_only_negative_momentum` FAILS (DOWN present)

- [x] **Step 3: Implement** (replace the Task-1 placeholder class)

```python
class DualMomentum(CrossSectionalMomentum):
    """Cross-sectional momentum + absolute filter: negative-momentum picks go to cash.

    Weights are NOT renormalized — a dropped pick's 1/N slot stays in cash,
    so the portfolio de-risks as breadth deteriorates.
    """

    name = "dual_momentum"

    def select(
        self, asof: pd.Timestamp, ohlcv: pd.DataFrame, eligible: List[str]
    ) -> List[Dict[str, Any]]:
        picks = super().select(asof, ohlcv, eligible)
        return [p for p in picks if p["momentum"] >= 0.0]
```

- [x] **Step 4: Run tests; verify all pass**

Run: `source .venv/bin/activate && python -m pytest tests/test_monthly_strategies.py -q`
Expected: 7 passed

- [x] **Step 5: Commit**

```bash
git add src/ggTrader/research/monthly_strategies.py tests/test_monthly_strategies.py
git commit -m "feat(research): dual momentum (absolute filter, cash fallback)"
```

---

### Task 4: `WfoTournamentStrategy` (move existing behavior)

**Files:**
- Modify: `src/ggTrader/research/monthly_strategies.py`
- Modify: `tests/test_monthly_strategies.py` (append)

The bodies of `_select_worker`, `select_for_month`, and `simulate_forward_month`
move from `monthly_walkforward.py` (current lines 76-269) **verbatim except**:
eligibility/coverage moves harness-side (Task 5); `select` receives
pre-truncated data; `simulate` adds `avg_exposure`. Do not edit
`monthly_walkforward.py` in this task — both copies coexist until Task 5.

- [x] **Step 1: Write the failing test** (append)

```python
def _trending_ohlcv(n: int = 600, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = _idx(n, start="2018-01-02")
    prices = {}
    for i, sym in enumerate(["AAA", "BBB"]):
        drift = 0.0006 * (i + 1)
        steps = rng.normal(drift, 0.02, n)
        prices[sym] = pd.Series(100.0 * np.exp(np.cumsum(steps)), index=idx)
    return make_ohlcv(prices)


def test_wfo_tournament_strategy_is_deterministic_and_jsonable():
    from ggTrader.research.monthly_strategies import WfoTournamentStrategy

    ohlcv = _trending_ohlcv()
    asof = ohlcv.index[-40]
    cfg = MonthlyHarnessConfig(
        top_n=2,
        entries=["psar_adx"],
        exits=["atr_trailing"],
        grid_book="coarse",
        n_splits=4,
        n_jobs=1,
        lookback_bars=400,
    )
    strat = WfoTournamentStrategy(cfg, STOCK_BASE_CONFIG)
    past = ohlcv.loc[:asof]
    s1 = strat.select(asof, past, ["AAA", "BBB"])
    s2 = strat.select(asof, past.copy(), ["AAA", "BBB"])
    assert json.dumps(s1, sort_keys=True, default=str) == json.dumps(
        s2, sort_keys=True, default=str
    )
    json.dumps(s1)
    if len(s1) >= 2:  # gates may reject synthetic series; determinism is the contract
        assert s1[0]["oos_robustness"] >= s1[1]["oos_robustness"]
    rets, diags = strat.simulate(ohlcv, s1, asof, ohlcv.index[-1])
    assert "avg_exposure" in diags
```

- [x] **Step 2: Run to verify it fails**

Run: `source .venv/bin/activate && python -m pytest tests/test_monthly_strategies.py::test_wfo_tournament_strategy_is_deterministic_and_jsonable -q`
Expected: FAIL with `ImportError: cannot import name 'WfoTournamentStrategy'`

- [x] **Step 3: Implement** — add to `monthly_strategies.py` (module-level worker for pickling):

```python
def _select_worker(args: Tuple) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Tournament for one stock on its trailing window (process-pool worker)."""
    (
        symbol,
        sym_ohlcv,
        base_config,
        entries,
        exits,
        entry_book,
        exit_book,
        n_splits,
        test_ratio,
    ) = args
    try:
        res = wfo_strategy_tournament_one_stock(
            sym_ohlcv, base_config, entries, exits, entry_book, exit_book, n_splits, test_ratio
        )
    except Exception as exc:
        print(f"    [{symbol}] tournament failed: {exc!r}")
        return symbol, None
    best = res.get("best")
    if best is None or not np.isfinite(best.get("oos_robustness", float("nan"))):
        return symbol, None

    avg_hold = float("nan")
    try:
        replay_cfg = {
            **base_config,
            "ENTRY_STRATEGY": best["entry"],
            "EXIT_STRATEGY": best["exit"],
        }
        engine = FastBacktest(sym_ohlcv, best["params"], config=replay_cfg)
        engine.run(show_progress=False)
        avg_hold = engine.get_stats().get("avg_holding_days", float("nan"))
    except Exception:
        pass

    return symbol, {
        "symbol": symbol,
        "entry": best["entry"],
        "exit": best["exit"],
        "params": _to_native(best["params"]),
        "oos_robustness": float(best["oos_robustness"]),
        "fold_consistency": float(best["fold_consistency"]),
        "is_robustness": float(best["is_robustness"]),
        "avg_holding_days": float(avg_hold) if np.isfinite(avg_hold) else None,
    }


class WfoTournamentStrategy:
    """Per-stock entry x exit WFO tournament; top-N by OOS robustness."""

    name = "wfo_tournament"

    def __init__(
        self,
        cfg,
        base_config: Dict[str, Any],
        entry_book: Optional[Dict[str, Dict[str, Any]]] = None,
        exit_book: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        if entry_book is None or exit_book is None:
            entry_book, exit_book = grid_books(cfg.grid_book)
        self.cfg = cfg
        self.base_config = base_config
        self.entry_book = entry_book
        self.exit_book = exit_book

    def select(
        self, asof: pd.Timestamp, ohlcv: pd.DataFrame, eligible: List[str]
    ) -> List[Dict[str, Any]]:
        cfg = self.cfg
        jobs = [
            (
                sym,
                ohlcv[[sym]].tail(cfg.lookback_bars),
                self.base_config,
                cfg.entries,
                cfg.exits,
                self.entry_book,
                self.exit_book,
                cfg.n_splits,
                cfg.test_ratio,
            )
            for sym in eligible
        ]
        results: List[Dict[str, Any]] = []
        if cfg.n_jobs > 1:
            with ProcessPoolExecutor(max_workers=cfg.n_jobs) as pool:
                for _sym, rec in pool.map(_select_worker, jobs, chunksize=1):
                    if rec is not None:
                        results.append(rec)
        else:
            for job in jobs:
                _sym, rec = _select_worker(job)
                if rec is not None:
                    results.append(rec)
        results.sort(key=lambda r: r["oos_robustness"], reverse=True)
        return results[: cfg.top_n]

    def simulate(
        self,
        ohlcv: pd.DataFrame,
        selections: List[Dict[str, Any]],
        asof: pd.Timestamp,
        month_end: pd.Timestamp,
    ) -> Tuple[pd.Series, Dict[str, Any]]:
        cfg, base_config = self.cfg, self.base_config
        empty = pd.Series(dtype=float)
        month_index = ohlcv.index[(ohlcv.index > asof) & (ohlcv.index <= month_end)]
        if len(month_index) == 0 or not selections:
            return empty, {"n_positions": 0, "n_trades": 0, "avg_exposure": 0.0}

        all_entries, all_exits, all_close = [], [], []
        for sel in selections:
            sym = sel["symbol"]
            if sym not in ohlcv.columns.get_level_values(0):
                continue
            window = ohlcv[[sym]].loc[:month_end].tail(cfg.warmup_bars + len(month_index))
            sig_cfg = {
                **base_config,
                "ENTRY_STRATEGY": sel["entry"],
                "EXIT_STRATEGY": sel["exit"],
            }
            try:
                engine = FastBacktest(window, sel["params"], config=sig_cfg)
                engine.run(show_progress=False)
            except Exception as exc:
                print(f"    [{sym}] forward signal generation failed: {exc!r}")
                continue
            entries = engine.entries.droplevel("param_combo", axis=1)
            exits = engine.exits.droplevel("param_combo", axis=1)
            entries.loc[entries.index <= asof] = False  # no pre-month positions
            all_entries.append(entries)
            all_exits.append(exits)
            all_close.append(window.xs("close", axis=1, level=1, drop_level=True))

        if not all_entries:
            return empty, {"n_positions": 0, "n_trades": 0, "avg_exposure": 0.0}

        entries_df = pd.concat(all_entries, axis=1).fillna(False)
        exits_df = pd.concat(all_exits, axis=1).fillna(False)
        close_df = pd.concat(all_close, axis=1)

        pf = vbt.Portfolio.from_signals(
            close=close_df,
            entries=entries_df,
            exits=exits_df,
            init_cash=float(base_config["START_CASH"]),
            fees=float(base_config["FEES"]),
            slippage=float(base_config["SLIPPAGE"]),
            freq=base_config["FREQ"],
            size=cfg.max_position_pct,
            size_type="percent",
            cash_sharing=True,
            group_by=np.full(entries_df.shape[1], 0),
        ).copy()

        returns = pf.returns()
        if isinstance(returns, pd.DataFrame):
            returns = returns.iloc[:, 0]
        in_month = (returns.index > asof) & (returns.index <= month_end)
        month_returns = returns.loc[in_month]
        exposure = _portfolio_exposure(pf).loc[in_month]

        dur = np.array(pf.trades.duration.values, dtype=np.float64, copy=True)
        diags = {
            "n_positions": int(len(selections)),
            "n_trades": int(pf.trades.count().sum()),
            "avg_holding_days": float(dur.mean()) if dur.size else None,
            "avg_exposure": float(exposure.mean()) if len(exposure) else 0.0,
            "month_return_pct": float((1.0 + month_returns).prod() - 1.0) * 100,
        }
        return month_returns, diags
```

- [x] **Step 4: Run the new test (allow ~1 min; numba warmup)**

Run: `source .venv/bin/activate && python -m pytest tests/test_monthly_strategies.py -q`
Expected: 8 passed

- [x] **Step 5: Commit**

```bash
git add src/ggTrader/research/monthly_strategies.py tests/test_monthly_strategies.py
git commit -m "feat(research): WFO tournament as a MonthlyStrategy plug-in"
```

---

### Task 5: Harness refactor (`monthly_walkforward.py`)

**Files:**
- Modify: `src/ggTrader/research/monthly_walkforward.py`

Changes: delete `_select_worker`/`select_for_month`/`simulate_forward_month`
(now in `monthly_strategies.py`); add `_eligibility`; `run_monthly_walkforward`
and `leak_check` take a `strategy` (default: `WfoTournamentStrategy` for
backward compatibility); collect `avg_exposure` into the summary (including
from `coverage.json` on resumed months).

- [x] **Step 1: Check for external importers of the moved functions**

Run: `grep -rn "select_for_month\|simulate_forward_month\|_select_worker" src/ scripts/ tests/ --include="*.py" | grep -v monthly_walkforward.py | grep -v monthly_strategies.py`
Expected: no output (only the CLI imports `run_monthly_walkforward`/`leak_check`/`MonthlyHarnessConfig`). If anything appears, update those importers in this task.

- [x] **Step 2: Apply the refactor.** Keep `MonthlyHarnessConfig`,
`_month_end_selection_dates`, `stitch_equity_curve`, `benchmark_vs_spy`,
`selection_turnover` unchanged. Replace imports and the three moved functions;
new/changed code:

```python
# imports: drop ProcessPoolExecutor, FastBacktest, _to_native, vbt,
# ENTRY/EXIT_REGISTRY stay (config defaults), and add:
from ggTrader.research.monthly_strategies import MonthlyStrategy, WfoTournamentStrategy
# equity_wfo imports shrink to: STOCK_BASE_CONFIG, fetch_stock_ohlcv


def _eligibility(
    asof: pd.Timestamp, past: pd.DataFrame, cfg: MonthlyHarnessConfig
) -> Tuple[List[str], Dict[str, Any]]:
    """PIT members at asof with enough history in ``past`` (data <= asof)."""
    members = [normalize_yf_ticker(m) for m in sp500_members_asof(asof)]
    have = set(past.columns.get_level_values(0).unique())
    eligible: List[str] = []
    for sym in members:
        if sym not in have:
            continue
        if len(past[sym]["close"].dropna()) >= cfg.min_history_bars:
            eligible.append(sym)
    if cfg.max_stocks is not None:
        eligible = sorted(eligible)[: cfg.max_stocks]
    coverage = coverage_stats(members, eligible)
    coverage["asof"] = str(asof.date())
    return eligible, coverage
```

`run_monthly_walkforward` signature and month loop (rest of the function body
— data loading, checkpoint resume, summary writing — stays as-is except where
shown):

```python
def run_monthly_walkforward(
    cfg: MonthlyHarnessConfig,
    base_config: Optional[Dict[str, Any]] = None,
    strategy: Optional[MonthlyStrategy] = None,
) -> Dict[str, Any]:
    base_config = {**STOCK_BASE_CONFIG, **(base_config or {})}
    strategy = strategy or WfoTournamentStrategy(cfg, base_config)
    run_dir = Path(cfg.checkpoint_dir) / cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Strategy: {strategy.name}")
    ...
    # inside the month loop, the refit branch becomes:
        if refit:
            past = ohlcv.loc[:asof]
            eligible, coverage = _eligibility(asof, past, cfg)
            selections = strategy.select(asof, past, eligible)
            active_selections = selections
        else:
            selections, coverage = active_selections, {"reused_previous_selection": True}

        rets, diags = strategy.simulate(ohlcv, selections, asof, month_end)
    ...
    # resume branch additionally recovers exposure diagnostics:
        if ret_path.exists() and sel_path.exists():
            ...
            cov_path = month_dir / "coverage.json"
            if cov_path.exists():
                prev = json.loads(cov_path.read_text()).get("diagnostics", {})
                if prev.get("avg_exposure") is not None:
                    exposures.append(float(prev["avg_exposure"]))
            ...
    # in the freshly-computed branch, after diags are known:
        if diags.get("avg_exposure") is not None:
            exposures.append(float(diags["avg_exposure"]))
```

Summary additions (`exposures: List[float] = []` initialized next to
`month_returns`; combo counts guarded for non-WFO records):

```python
    combo_counts: Dict[str, int] = {}
    for sels in selections_by_month:
        for s in sels:
            if "entry" in s and "exit" in s:
                key = f"{s['entry']}+{s['exit']}"
                combo_counts[key] = combo_counts.get(key, 0) + 1

    summary = {
        "config": {**cfg.__dict__},
        "strategy": strategy.name,
        "report": report,
        "avg_exposure": float(np.mean(exposures)) if exposures else None,
        ...  # rest unchanged
    }
```

`leak_check` becomes strategy-generic — same data setup, then:

```python
def leak_check(
    cfg: MonthlyHarnessConfig,
    base_config: Optional[Dict[str, Any]] = None,
    strategy: Optional[MonthlyStrategy] = None,
) -> bool:
    """Selections at T must be identical with and without post-T data loaded."""
    base_config = {**STOCK_BASE_CONFIG, **(base_config or {})}
    strategy = strategy or WfoTournamentStrategy(cfg, base_config)
    ...  # unchanged through sel_dates / asof
    eligible, _ = _eligibility(asof, ohlcv.loc[:asof], cfg)
    full = strategy.select(asof, ohlcv.loc[:asof], eligible)
    truncated = strategy.select(asof, ohlcv.loc[:asof].copy(deep=True), eligible)
    # also verify untruncated data changes nothing if a strategy mis-truncates:
    unmasked = strategy.select(asof, ohlcv, eligible)
    ok = (
        json.dumps(full, sort_keys=True, default=str)
        == json.dumps(truncated, sort_keys=True, default=str)
        == json.dumps(unmasked, sort_keys=True, default=str)
    )
    print("LEAK CHECK:", "PASS — selections identical" if ok else "FAIL — selections differ!")
    return ok
```

(Note: `unmasked` intentionally passes full data — defense in depth; every
strategy must be invariant to post-T rows even though the harness truncates.)

- [x] **Step 3: Run the full unit suite**

Run: `source .venv/bin/activate && python -m pytest tests/ -m 'not integration' -q`
Expected: 268 passed (260 baseline + 8 new), 3 pre-existing failures (`test_circuit_breaker_persistence`, `test_system_dry_run_cycle`, `test_persistence_logic`) — same as baseline

- [x] **Step 4: Commit**

```bash
git add src/ggTrader/research/monthly_walkforward.py
git commit -m "refactor(research): harness is strategy-agnostic; exposure in diagnostics"
```

---

### Task 6: CLI `--strategy` + factory

**Files:**
- Modify: `src/ggTrader/research/monthly_strategies.py` (factory)
- Modify: `scripts/sp500_monthly_walkforward.py`

- [x] **Step 1: Add the factory** (end of `monthly_strategies.py`)

```python
STRATEGY_NAMES = ("wfo_tournament", "xs_momentum", "dual_momentum")


def build_strategy(
    name: str,
    cfg,
    base_config: Dict[str, Any],
    mom_lookback: int = 252,
    mom_skip: int = 21,
) -> "MonthlyStrategy":
    if name == "wfo_tournament":
        return WfoTournamentStrategy(cfg, base_config)
    if name == "xs_momentum":
        return CrossSectionalMomentum(cfg, base_config, mom_lookback, mom_skip)
    if name == "dual_momentum":
        return DualMomentum(cfg, base_config, mom_lookback, mom_skip)
    raise ValueError(f"Unknown strategy {name!r}. Available: {STRATEGY_NAMES}")
```

- [x] **Step 2: Update the CLI.** In `scripts/sp500_monthly_walkforward.py`:

```python
# new imports
from ggTrader.research.equity_wfo import STOCK_BASE_CONFIG  # noqa: E402
from ggTrader.research.monthly_strategies import STRATEGY_NAMES, build_strategy  # noqa: E402

# new args (after --grid):
    p.add_argument("--strategy", choices=STRATEGY_NAMES, default="wfo_tournament")
    p.add_argument("--mom-lookback", type=int, default=252)
    p.add_argument("--mom-skip", type=int, default=21)

# run_id default changes:
        run_id=args.run_id
        or (f"sp500_quick_{args.strategy}" if args.quick else f"sp500_{args.strategy}"),

# quick-mode entries/exits narrowing only applies to the WFO strategy:
    if args.quick:
        ...
        if args.entries == "all" and args.strategy == "wfo_tournament":
            cfg.entries = ["psar_adx", "ema_cross"]
        if args.exits == "all" and args.strategy == "wfo_tournament":
            cfg.exits = ["atr_trailing"]

# strategy construction + call sites:
    strategy = build_strategy(
        args.strategy, cfg, dict(STOCK_BASE_CONFIG), args.mom_lookback, args.mom_skip
    )
    if args.leak_check:
        ok = leak_check(cfg, strategy=strategy)
        raise SystemExit(0 if ok else 1)
    summary = run_monthly_walkforward(cfg, strategy=strategy)

# summary printing: guard the WFO-only fields:
    if summary.get("avg_exposure") is not None:
        print(f"avg exposure: {summary['avg_exposure']:.2f}")
    if summary["combo_selection_counts"]:
        print("combo selection counts (top 10):")
        for combo, n in list(summary["combo_selection_counts"].items())[:10]:
            print(f"  {combo:<40} {n}")
```

- [x] **Step 3: Lint + suite**

Run: `source .venv/bin/activate && ruff check src/ggTrader/research/ scripts/sp500_monthly_walkforward.py tests/test_monthly_strategies.py && python -m pytest tests/ -m 'not integration' -q`
Expected: ruff clean; 268 passed, 3 pre-existing failures

- [x] **Step 4: Commit**

```bash
git add src/ggTrader/research/monthly_strategies.py scripts/sp500_monthly_walkforward.py
git commit -m "feat(research): --strategy CLI flag (wfo_tournament|xs_momentum|dual_momentum)"
```

---

### Task 7: End-to-end verification (quick momentum run + leak check)

**Files:** none (verification only). Note: the big `sp500_monthly_v1` run is
using the CPU — keep these to `--jobs 1`; momentum needs no pool anyway.

- [x] **Step 1: Leak check for xs_momentum** (uses DB-cached data; ~2 min)

Run: `source .venv/bin/activate && python -u scripts/sp500_monthly_walkforward.py --quick --strategy xs_momentum --leak-check --jobs 1`
Expected: `LEAK CHECK: PASS — selections identical`, exit 0

- [x] **Step 2: Quick xs_momentum run**

Run: `source .venv/bin/activate && python -u scripts/sp500_monthly_walkforward.py --quick --strategy xs_momentum --jobs 1`
Expected: completes in minutes; summary JSON prints with `avg_exposure` ≈ 0.95-1.0; checkpoints under `results/monthly_wf/sp500_quick_xs_momentum/`

- [x] **Step 3: Quick dual_momentum run**

Run: `source .venv/bin/activate && python -u scripts/sp500_monthly_walkforward.py --quick --strategy dual_momentum --jobs 1`
Expected: completes; avg_exposure <= xs_momentum's (cash fallback)

- [x] **Step 4: Backward-compat smoke: WFO strategy still selects** (slowest step, ~10 min)

Run: `source .venv/bin/activate && python -u scripts/sp500_monthly_walkforward.py --quick --strategy wfo_tournament --leak-check --jobs 1`
Expected: `LEAK CHECK: PASS`

- [x] **Step 5: Update changelog + commit.** Append under `## 2026-06-10` in `docs/changelog.md`:

```markdown
### Research: strategy-agnostic monthly harness + momentum plug-ins

The honest monthly walk-forward harness now takes any `MonthlyStrategy`
(select on data <= T, simulate the forward month) — spec:
[2026-06-10-monthly-strategy-interface-design.md](file:///home/flynn/ggTrader/docs/superpowers/specs/2026-06-10-monthly-strategy-interface-design.md).
The WFO tournament moved to `research/monthly_strategies.py` unchanged; new
plug-ins `xs_momentum` (12-1 cross-sectional momentum, top-50 equal weight,
fully invested) and `dual_momentum` (negative-momentum picks go to cash).
CLI: `--strategy`, `--mom-lookback`, `--mom-skip`. Diagnostics and summaries
now report `avg_exposure`; the leak check is strategy-generic and also feeds
untruncated data to `select` as defense in depth.
```

```bash
git add docs/changelog.md docs/superpowers/plans/2026-06-10-monthly-strategy-interface.md
git commit -m "docs: changelog for strategy-agnostic harness; mark plan executed"
```

---

## Self-review (done at write time)

- **Spec coverage:** §2 interface → Task 1/4; §3.1 harness → Task 5; §3.2 WFO plug-in → Task 4; §3.3 xs momentum → Tasks 1-2; §3.4 dual momentum → Task 3; §3.5 CLI → Task 6; §5 testing → Tasks 1-4 (unit), Task 7 (leak checks per strategy, quick runs); equivalence is covered by verbatim code move + Task 4 determinism test + Task 7 Step 4 (spec's checkpoint-fixture idea replaced — fixture would need network/DB in unit tests).
- **Placeholders:** none; every code step shows the code.
- **Type consistency:** `select(asof, ohlcv, eligible)` / `simulate(ohlcv, selections, asof, month_end)` used identically in Tasks 1-6; `MonthlyHarnessConfig` fields referenced (`top_n`, `lookback_bars`, `entries`, `exits`, `grid_book`, `n_splits`, `n_jobs`, `test_ratio`, `warmup_bars`, `max_position_pct`, `min_history_bars`, `max_stocks`) all exist in the current dataclass.
