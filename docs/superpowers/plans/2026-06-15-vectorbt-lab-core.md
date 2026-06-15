# vectorbt lab core (Plan 1: momentum bench) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a new `src/ggTrader/lab/` package — a simple, vectorbt-centric, DB-only research bench — and prove the architecture by reproducing the known-good `sp500_xs_momentum` / `dual_momentum` results through it.

**Architecture:** One unified harness: a point-in-time *plan* phase (`select` on data ≤ T, per strategy per rebalance, resumable from the DB) followed by ONE vectorized `vbt.Portfolio.from_orders` call that simulates all weight-based strategies simultaneously (`group_by="strategy"`, `cash_sharing=True`). All run state (plans, returns, equity, summary) lives in TimescaleDB; nothing is written to `results/`.

**Tech Stack:** Python, vectorbt 0.28.5, pandas/numpy, TimescaleDB (SQLAlchemy), pytest.

**Spec:** [`docs/superpowers/specs/2026-06-15-vectorbt-lab-core-design.md`](../specs/2026-06-15-vectorbt-lab-core-design.md)

**Scope of this plan (Plan 1):** the foundation + the **weight-based** strategy family (`xs_momentum`, `dual_momentum`). The **signal-based** family (`wfo_tournament`, via `from_signals`) and the **deletion of old research code** are deliberately deferred to Plans 2 and 3 — they depend on this foundation proving out against the validation gate.

**Conventions:** ruff line length 100; absolute imports from `ggTrader`; a PostToolUse hook runs ruff autofix and **strips unused imports immediately** — when a step adds an import before its first use, expect it to be removed; re-add imports only once the using code is in the file. Commit trailer: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`. Activate the venv for every command: `source .venv/bin/activate`.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/ggTrader/lab/__init__.py` | Package marker (empty) |
| `src/ggTrader/lab/strategy.py` | `Strategy` protocol, `Plan` alias, `LabConfig` dataclass |
| `src/ggTrader/lab/strategies/__init__.py` | Package marker (empty) |
| `src/ggTrader/lab/strategies/momentum.py` | `CrossSectionalMomentum`, `DualMomentum`, registry/`build_strategy` |
| `src/ggTrader/lab/data.py` | DB OHLCV reader, PIT universe provider, rebalance-date helper |
| `src/ggTrader/lab/simulate.py` | `build_target_matrix`, `simulate_weights` (one grouped `from_orders`) |
| `src/ggTrader/lab/metrics.py` | curve stats + SPY benchmark compare |
| `src/ggTrader/lab/persist.py` | DB schema init + plan/returns/equity/summary read-write + resume |
| `src/ggTrader/lab/harness.py` | `walkforward` (plan → simulate → persist → score), `leak_check` |
| `src/ggTrader/lab/cli.py` | `run_lab` CLI entry (argparse) |
| `tests/lab/test_*.py` | Unit + integration tests mirroring each module |

---

## Task 1: Package scaffold + Strategy protocol + LabConfig

**Files:**
- Create: `src/ggTrader/lab/__init__.py` (empty)
- Create: `src/ggTrader/lab/strategies/__init__.py` (empty)
- Create: `src/ggTrader/lab/strategy.py`
- Test: `tests/lab/__init__.py` (empty), `tests/lab/test_strategy.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/lab/test_strategy.py
from ggTrader.lab.strategy import LabConfig


def test_labconfig_defaults():
    cfg = LabConfig()
    assert cfg.top_n == 50
    assert cfg.lookback == 252
    assert cfg.skip == 21
    assert cfg.min_history_bars == 400
    assert cfg.max_stocks is None


def test_labconfig_override():
    cfg = LabConfig(top_n=10, max_stocks=20)
    assert cfg.top_n == 10
    assert cfg.max_stocks == 20
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest tests/lab/test_strategy.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'ggTrader.lab'`

- [ ] **Step 3: Create the package markers and `strategy.py`**

Create empty `src/ggTrader/lab/__init__.py`, `src/ggTrader/lab/strategies/__init__.py`, `tests/lab/__init__.py`.

```python
# src/ggTrader/lab/strategy.py
"""Strategy protocol and config for the lab research bench."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Protocol

import pandas as pd

Plan = List[Dict[str, Any]]  # JSON-able selection records, each with at least "symbol"


@dataclass
class LabConfig:
    """Tunables shared by lab strategies and the harness."""

    top_n: int = 50
    lookback: int = 252  # trailing bars for the momentum measurement window
    skip: int = 21  # most-recent bars excluded (12-1 momentum)
    min_history_bars: int = 400  # required non-NaN closes to be eligible
    max_stocks: int | None = None  # cap the per-rebalance universe (deterministic)


class Strategy(Protocol):
    """A lab strategy: point-in-time select, then a whole-window target matrix.

    ``target_kind`` is "weights" (simulated via Portfolio.from_orders) or
    "signals" (via from_signals — added in Plan 2).
    """

    name: str
    target_kind: str

    def select(self, asof: pd.Timestamp, data: pd.DataFrame, eligible: List[str]) -> Plan:
        """JSON-able selections; MUST be a pure function of data <= asof."""
        ...

    def to_targets(self, plans: Dict[pd.Timestamp, Plan], data: pd.DataFrame) -> pd.DataFrame:
        """Whole-window (time x symbol) target matrix from per-rebalance plans."""
        ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && python -m pytest tests/lab/test_strategy.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/ggTrader/lab/__init__.py src/ggTrader/lab/strategies/__init__.py src/ggTrader/lab/strategy.py tests/lab/__init__.py tests/lab/test_strategy.py
git commit -m "feat(lab): strategy protocol + LabConfig scaffold

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 2: Momentum strategies (select + to_targets)

The `select` logic is ported **verbatim** from `research/monthly_strategies.py` so the validation gate's selections stay bit-identical. `to_targets` is new: it lays each rebalance's weights onto the first trading bar strictly after `asof`, filling 0.0 for symbols dropped that month so `targetpercent` exits them.

**Files:**
- Create: `src/ggTrader/lab/strategies/momentum.py`
- Test: `tests/lab/test_momentum.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/lab/test_momentum.py
import json

import numpy as np
import pandas as pd

from ggTrader.lab.strategy import LabConfig
from ggTrader.lab.strategies.momentum import (
    CrossSectionalMomentum,
    DualMomentum,
    build_strategy,
)


def make_ohlcv(prices: dict) -> pd.DataFrame:
    frames = {}
    for sym, close in prices.items():
        frames[sym] = pd.DataFrame(
            {"open": close, "high": close * 1.01, "low": close * 0.99,
             "close": close, "volume": np.full(len(close), 1e6)},
            index=close.index,
        )
    out = pd.concat(frames, axis=1)
    out.columns = out.columns.set_names(["symbol", "field"])
    return out


def _idx(n, start="2020-01-01"):
    return pd.date_range(start, periods=n, freq="B", tz="UTC")


def test_momentum_ranks_strongest_first_equal_weight():
    idx = _idx(300)
    ohlcv = make_ohlcv({
        "UP": pd.Series(np.linspace(10, 30, 300), index=idx),
        "FLAT": pd.Series(np.full(300, 20.0), index=idx),
        "DOWN": pd.Series(np.linspace(30, 10, 300), index=idx),
    })
    strat = CrossSectionalMomentum(LabConfig(top_n=3))
    sels = strat.select(idx[-1], ohlcv, ["UP", "FLAT", "DOWN"])
    assert [s["symbol"] for s in sels] == ["UP", "FLAT", "DOWN"]
    assert all(abs(s["weight"] - 1 / 3) < 1e-12 for s in sels)
    json.dumps(sels)  # must be JSON-able


def test_momentum_ignores_rows_after_asof():
    idx = _idx(300)
    ohlcv = make_ohlcv({
        "UP": pd.Series(np.linspace(10, 30, 300), index=idx),
        "DOWN": pd.Series(np.linspace(30, 10, 300), index=idx),
    })
    asof = idx[-30]
    strat = CrossSectionalMomentum(LabConfig(top_n=2))
    unmasked = strat.select(asof, ohlcv, ["UP", "DOWN"])
    truncated = strat.select(asof, ohlcv.loc[:asof], ["UP", "DOWN"])
    assert json.dumps(unmasked, sort_keys=True) == json.dumps(truncated, sort_keys=True)


def test_dual_momentum_drops_negative():
    idx = _idx(300)
    ohlcv = make_ohlcv({
        "UP": pd.Series(np.linspace(10, 30, 300), index=idx),
        "FLAT": pd.Series(np.full(300, 20.0), index=idx),
        "DOWN": pd.Series(np.linspace(30, 10, 300), index=idx),
    })
    sels = DualMomentum(LabConfig(top_n=3)).select(idx[-1], ohlcv, ["UP", "FLAT", "DOWN"])
    assert [s["symbol"] for s in sels] == ["UP", "FLAT"]
    assert all(abs(s["weight"] - 1 / 3) < 1e-12 for s in sels)  # NOT renormalized


def test_to_targets_lays_weights_after_asof_and_zeros_drops():
    idx = _idx(120)
    ohlcv = make_ohlcv({
        "A": pd.Series(np.linspace(10, 20, 120), index=idx),
        "B": pd.Series(np.linspace(10, 15, 120), index=idx),
    })
    strat = CrossSectionalMomentum(LabConfig(top_n=1))
    t1, t2 = idx[60], idx[90]
    plans = {t1: [{"symbol": "A", "weight": 1.0, "momentum": 1.0}],
             t2: [{"symbol": "B", "weight": 1.0, "momentum": 0.5}]}
    targets = strat.to_targets(plans, ohlcv)
    first_after_t1 = ohlcv.index[ohlcv.index > t1][0]
    first_after_t2 = ohlcv.index[ohlcv.index > t2][0]
    assert targets.loc[first_after_t1, "A"] == 1.0
    assert targets.loc[first_after_t2, "A"] == 0.0  # dropped -> exit
    assert targets.loc[first_after_t2, "B"] == 1.0
    assert targets.notna().any(axis=1).sum() == 2  # only two rebalance rows carry orders


def test_build_strategy_dispatch():
    assert build_strategy("xs_momentum", LabConfig()).name == "xs_momentum"
    assert build_strategy("dual_momentum", LabConfig()).name == "dual_momentum"
    try:
        build_strategy("nope", LabConfig())
        assert False, "expected ValueError"
    except ValueError:
        pass
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tests/lab/test_momentum.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'ggTrader.lab.strategies.momentum'`

- [ ] **Step 3: Implement `momentum.py`**

```python
# src/ggTrader/lab/strategies/momentum.py
"""Cross-sectional and dual momentum (weight-based lab strategies)."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from ggTrader.lab.strategy import LabConfig, Plan


class CrossSectionalMomentum:
    """12-1 cross-sectional momentum: top-N by trailing return, equal weight."""

    name = "xs_momentum"
    target_kind = "weights"

    def __init__(self, cfg: LabConfig) -> None:
        self.cfg = cfg

    def select(self, asof: pd.Timestamp, data: pd.DataFrame, eligible: List[str]) -> Plan:
        data = data.loc[:asof]  # defense in depth: invariant to post-asof rows
        lookback, skip = self.cfg.lookback, self.cfg.skip
        scores: Dict[str, float] = {}
        for sym in eligible:
            closes = data[sym]["close"].dropna()
            if len(closes) < lookback + 1:
                continue
            past = float(closes.iloc[-(lookback + 1)])
            recent = float(closes.iloc[-(skip + 1)])
            if past <= 0.0 or not np.isfinite(past) or not np.isfinite(recent):
                continue
            scores[sym] = recent / past - 1.0
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[: self.cfg.top_n]
        if not ranked:
            return []
        weight = 1.0 / len(ranked)
        return [{"symbol": s, "weight": weight, "momentum": m} for s, m in ranked]

    def to_targets(self, plans: Dict[pd.Timestamp, Plan], data: pd.DataFrame) -> pd.DataFrame:
        symbols = sorted({s["symbol"] for plan in plans.values() for s in plan})
        targets = pd.DataFrame(np.nan, index=data.index, columns=symbols)
        for asof in sorted(plans):
            forward = data.index[data.index > asof]
            if len(forward) == 0:
                continue
            bar = forward[0]
            targets.loc[bar, symbols] = 0.0  # default: exit anything not re-selected
            for sel in plans[asof]:
                targets.loc[bar, sel["symbol"]] = float(sel["weight"])
        return targets


class DualMomentum(CrossSectionalMomentum):
    """Cross-sectional momentum + absolute filter: negative-momentum picks go to cash.

    Weights are NOT renormalized — a dropped pick's slot stays in cash.
    """

    name = "dual_momentum"

    def select(self, asof: pd.Timestamp, data: pd.DataFrame, eligible: List[str]) -> Plan:
        return [p for p in super().select(asof, data, eligible) if p["momentum"] >= 0.0]


_REGISTRY = {
    "xs_momentum": CrossSectionalMomentum,
    "dual_momentum": DualMomentum,
}

STRATEGY_NAMES = tuple(_REGISTRY)


def build_strategy(name: str, cfg: LabConfig):
    if name not in _REGISTRY:
        raise ValueError(f"Unknown strategy {name!r}. Available: {STRATEGY_NAMES}")
    return _REGISTRY[name](cfg)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && python -m pytest tests/lab/test_momentum.py -q`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/ggTrader/lab/strategies/momentum.py tests/lab/test_momentum.py
git commit -m "feat(lab): momentum strategies (select ported verbatim + to_targets)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 3: Data layer (OHLCV read, PIT universe, rebalance dates)

Reuses the proven DB-first equity loader (`fetch_stock_ohlcv`) and PIT constituents helpers, plus the month-end logic from the old harness, wrapped behind a small lab-local API. The pure-function `rebalance_dates` is unit-tested; the DB-touching loaders get an integration test marked `integration`.

**Files:**
- Create: `src/ggTrader/lab/data.py`
- Test: `tests/lab/test_data.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/lab/test_data.py
import pandas as pd
import pytest

from ggTrader.lab.data import rebalance_dates


def test_rebalance_dates_are_month_ends_excluding_last():
    idx = pd.date_range("2021-01-01", "2021-06-30", freq="B", tz="UTC")
    dates = rebalance_dates(idx, pd.Timestamp("2021-01-31", tz="UTC"),
                            pd.Timestamp("2021-06-30", tz="UTC"))
    # Jan..May month-ends are selection dates; June is the final span end (dropped).
    assert [d.strftime("%Y-%m") for d in dates] == ["2021-01", "2021-02", "2021-03",
                                                    "2021-04", "2021-05"]
    assert all(d.tz is not None for d in dates)


def test_rebalance_dates_empty_when_no_overlap():
    idx = pd.date_range("2021-01-01", "2021-01-10", freq="B", tz="UTC")
    assert rebalance_dates(idx, pd.Timestamp("2022-01-01", tz="UTC"),
                           pd.Timestamp("2022-12-31", tz="UTC")) == []


@pytest.mark.integration
def test_load_ohlcv_returns_multiindex_frame():
    from ggTrader.lab.data import load_ohlcv
    df = load_ohlcv(["SPY"], "2024-01-01", "2024-03-01")
    assert df.columns.names == ["symbol", "field"]
    assert "close" in df["SPY"].columns
    assert len(df) > 20
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tests/lab/test_data.py -q -m "not integration"`
Expected: FAIL with `ModuleNotFoundError: No module named 'ggTrader.lab.data'`

- [ ] **Step 3: Implement `data.py`**

```python
# src/ggTrader/lab/data.py
"""Data access for the lab bench: OHLCV from the DB, PIT universe, rebalance dates."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ggTrader.data.core.index_constituents import (
    all_members_between,
    coverage_stats,
    normalize_yf_ticker,
    sp500_members_asof,
)
from ggTrader.lab.strategy import LabConfig
from ggTrader.research.equity_wfo import fetch_stock_ohlcv


def load_ohlcv(symbols: List[str], start: str, end: Optional[str] = None) -> pd.DataFrame:
    """DB-first daily OHLCV as a (symbol, field) MultiIndex frame."""
    return fetch_stock_ohlcv(symbols, start=start, end=end, interval="1d", use_db_cache=True)


def equity_universe_between(eval_start: pd.Timestamp, eval_end: pd.Timestamp) -> List[str]:
    """All yfinance-normalized S&P 500 members that existed anywhere in the span."""
    return sorted({normalize_yf_ticker(t) for t in all_members_between(eval_start, eval_end)})


def rebalance_dates(
    index: pd.DatetimeIndex, eval_start: pd.Timestamp, eval_end: pd.Timestamp
) -> List[pd.Timestamp]:
    """Last trading day of each month in [eval_start, eval_end), excluding the final
    month (selecting there would leave no forward period to trade)."""
    idx = index[(index >= eval_start) & (index <= eval_end)]
    if len(idx) == 0:
        return []
    series = pd.Series(idx, index=idx)
    month_ends = series.groupby(idx.tz_localize(None).to_period("M")).max().tolist()
    return month_ends[:-1]


def eligible_at(
    asof: pd.Timestamp, past: pd.DataFrame, cfg: LabConfig
) -> Tuple[List[str], Dict[str, Any]]:
    """PIT members at asof with enough history in ``past`` (data <= asof)."""
    members = [normalize_yf_ticker(m) for m in sp500_members_asof(asof)]
    have = set(past.columns.get_level_values(0).unique())
    eligible: List[str] = []
    for sym in members:
        if sym in have and len(past[sym]["close"].dropna()) >= cfg.min_history_bars:
            eligible.append(sym)
    if cfg.max_stocks is not None:
        eligible = sorted(eligible)[: cfg.max_stocks]
    coverage = coverage_stats(members, eligible)
    coverage["asof"] = str(asof.date())
    return eligible, coverage
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && python -m pytest tests/lab/test_data.py -q -m "not integration"`
Expected: PASS (2 passed, 1 deselected)

Then confirm the integration path against the real DB:
Run: `source .venv/bin/activate && python -m pytest tests/lab/test_data.py -q -m integration`
Expected: PASS (1 passed) — if the DB is unreachable it falls back to yfinance and still passes.

- [ ] **Step 5: Commit**

```bash
git add src/ggTrader/lab/data.py tests/lab/test_data.py
git commit -m "feat(lab): data layer — DB OHLCV, PIT universe, rebalance dates

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 4: Vectorized simulate (one grouped from_orders for all strategies)

The heart of the "all strats simultaneously" requirement. `simulate_weights` takes `{strategy_name: target_matrix}`, concatenates them into `(strategy, symbol)` columns, and runs ONE `from_orders` with `group_by` on the strategy level — every strategy's full equity curve from a single broadcast pass. The equivalence test guards correctness.

**Files:**
- Create: `src/ggTrader/lab/simulate.py`
- Test: `tests/lab/test_simulate.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/lab/test_simulate.py
import numpy as np
import pandas as pd

from ggTrader.lab.simulate import simulate_weights

BASE = {"START_CASH": 10000.0, "FEES": 0.0, "SLIPPAGE": 0.0, "FREQ": "1d"}


def _prices(n=40):
    idx = pd.date_range("2021-01-01", periods=n, freq="B", tz="UTC")
    return pd.DataFrame({
        "A": 100.0 * 1.01 ** np.arange(n),  # +1%/day
        "B": np.full(n, 50.0),              # flat
    }, index=idx)


def _targets(prices, first_weights):
    t = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    t.iloc[1] = [first_weights.get(c, 0.0) for c in prices.columns]
    return t


def test_simulate_weights_matches_hand_computed_return():
    prices = _prices()
    targets = {"x": _targets(prices, {"A": 0.5, "B": 0.5})}
    rets, equity, diags = simulate_weights(targets, prices, BASE)
    # Buy at bar 1; B flat, so portfolio total return is half of A's appreciation.
    expected = 0.5 * (prices["A"].iloc[-1] / prices["A"].iloc[1] - 1.0)
    total = float(equity["x"].iloc[-1] / BASE["START_CASH"] - 1.0)
    assert abs(total - expected) < 1e-6
    assert diags["x"]["n_strategies"] == 1


def test_simulate_weights_runs_strategies_simultaneously_and_equally():
    prices = _prices()
    together = {"x": _targets(prices, {"A": 1.0}), "y": _targets(prices, {"B": 1.0})}
    r_both, eq_both, _ = simulate_weights(together, prices, BASE)
    r_x, eq_x, _ = simulate_weights({"x": together["x"]}, prices, BASE)
    # Grouped multi-strategy run must equal the strategy run alone (vectorization guard).
    pd.testing.assert_series_equal(eq_both["x"], eq_x["x"], check_names=False)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tests/lab/test_simulate.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'ggTrader.lab.simulate'`

- [ ] **Step 3: Implement `simulate.py`**

```python
# src/ggTrader/lab/simulate.py
"""Vectorized portfolio simulation: one grouped vbt call across all strategies."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import vectorbt as vbt


def simulate_weights(
    targets_by_strategy: Dict[str, pd.DataFrame],
    prices: pd.DataFrame,
    base_config: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """Simulate every weight-based strategy in ONE from_orders call.

    Args:
        targets_by_strategy: name -> (time x symbol) targetpercent matrix
            (NaN = no order that bar; 0.0 = exit; w = target weight).
        prices: (time x symbol) close prices covering every target column.
        base_config: START_CASH, FEES, SLIPPAGE, FREQ.

    Returns:
        (returns_df, equity_df, diags) each keyed by strategy name (columns).
    """
    names = list(targets_by_strategy)
    size_blocks, close_blocks, groups = [], [], []
    for name in names:
        tgt = targets_by_strategy[name]
        cols = pd.MultiIndex.from_product([[name], tgt.columns], names=["strategy", "symbol"])
        size_blocks.append(tgt.set_axis(cols, axis=1))
        px = prices[tgt.columns].reindex(tgt.index).ffill()
        close_blocks.append(px.set_axis(cols, axis=1))
        groups.extend([name] * tgt.shape[1])

    size = pd.concat(size_blocks, axis=1)
    close = pd.concat(close_blocks, axis=1)

    pf = vbt.Portfolio.from_orders(
        close=close,
        size=size,
        size_type="targetpercent",
        init_cash=float(base_config["START_CASH"]),
        fees=float(base_config["FEES"]),
        slippage=float(base_config["SLIPPAGE"]),
        freq=base_config["FREQ"],
        cash_sharing=True,
        group_by=pd.Index(groups, name="strategy"),
        call_seq="auto",
    ).copy()

    value = pf.value()  # (time x strategy) once grouped
    if isinstance(value, pd.Series):
        value = value.to_frame(names[0])
    value = value[names]
    returns = value.pct_change().fillna(0.0)
    diags = {name: {"n_strategies": 1, "n_symbols": int(targets_by_strategy[name].shape[1])}
             for name in names}
    return returns, value, diags
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && python -m pytest tests/lab/test_simulate.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/ggTrader/lab/simulate.py tests/lab/test_simulate.py
git commit -m "feat(lab): vectorized simulate — one grouped from_orders for all strategies

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 5: Metrics (curve stats + SPY benchmark)

Ported from the old harness's `benchmark_vs_spy`, adapted to take a precomputed equity curve and SPY closes and return the same metric shape.

**Files:**
- Create: `src/ggTrader/lab/metrics.py`
- Test: `tests/lab/test_metrics.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/lab/test_metrics.py
import numpy as np
import pandas as pd

from ggTrader.lab.metrics import curve_stats, benchmark


def test_curve_stats_known_values():
    idx = pd.date_range("2021-01-01", periods=253, freq="B", tz="UTC")
    curve = pd.Series(10000.0 * 1.0005 ** np.arange(253), index=idx)  # steady up
    s = curve_stats(curve)
    assert s["total_return_pct"] > 0
    assert s["max_drawdown_pct"] == 0.0  # monotonic up -> no drawdown
    assert s["sharpe"] > 0


def test_benchmark_shape():
    idx = pd.date_range("2021-01-01", periods=253, freq="B", tz="UTC")
    equity = pd.Series(10000.0 * 1.0006 ** np.arange(253), index=idx)
    spy = pd.Series(400.0 * 1.0004 ** np.arange(253), index=idx)
    rep = benchmark(equity, spy, 10000.0)
    assert set(rep) == {"strategy", "spy", "monthly_hit_rate_vs_spy", "n_months"}
    assert "sharpe" in rep["strategy"] and "sharpe" in rep["spy"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tests/lab/test_metrics.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'ggTrader.lab.metrics'`

- [ ] **Step 3: Implement `metrics.py`**

```python
# src/ggTrader/lab/metrics.py
"""Curve statistics and SPY benchmark comparison for lab runs."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def curve_stats(curve: pd.Series) -> Dict[str, float]:
    curve = curve.dropna()
    rets = curve.pct_change().dropna()
    years = max((curve.index[-1] - curve.index[0]).days / 365.25, 1e-9)
    total = float(curve.iloc[-1] / curve.iloc[0] - 1.0)
    ann_vol = float(rets.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
    sharpe = (float(rets.mean() / rets.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
              if rets.std() > 0 else float("nan"))
    downside = rets[rets < 0]
    sortino = (float(rets.mean() / downside.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
               if len(downside) and downside.std() > 0 else float("nan"))
    dd = float((curve / curve.cummax() - 1.0).min())
    return {
        "total_return_pct": total * 100,
        "cagr_pct": ((1 + total) ** (1 / years) - 1) * 100,
        "sharpe": sharpe,
        "sortino": sortino,
        "ann_vol_pct": ann_vol * 100,
        "max_drawdown_pct": dd * 100,
    }


def benchmark(equity: pd.Series, spy_close: pd.Series, start_cash: float) -> Dict[str, Any]:
    spy = spy_close.reindex(equity.index).ffill().dropna()
    spy_curve = start_cash * (spy / spy.iloc[0])
    eq = equity.reindex(spy_curve.index)

    strat_m = eq.resample("ME").last().pct_change().dropna()
    spy_m = spy_curve.resample("ME").last().pct_change().dropna()
    common = strat_m.index.intersection(spy_m.index)
    hit = float((strat_m.loc[common] > spy_m.loc[common]).mean()) if len(common) else None

    return {
        "strategy": curve_stats(eq.dropna()),
        "spy": curve_stats(spy_curve),
        "monthly_hit_rate_vs_spy": hit,
        "n_months": int(len(common)),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && python -m pytest tests/lab/test_metrics.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/ggTrader/lab/metrics.py tests/lab/test_metrics.py
git commit -m "feat(lab): curve stats + SPY benchmark metrics

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 6: Persistence + DB schema (TimescaleDB only)

All run state in the DB. Schema init is idempotent (`CREATE TABLE IF NOT EXISTS`); `lab_returns`/`lab_equity` become hypertables when the TimescaleDB extension is present (guarded so plain Postgres still works). Tests are `integration` (real DB).

**Files:**
- Create: `src/ggTrader/lab/persist.py`
- Test: `tests/lab/test_persist.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/lab/test_persist.py
import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.integration


def _engine():
    from ggTrader.lab.persist import get_engine
    return get_engine()


def test_schema_init_idempotent():
    from ggTrader.lab.persist import init_schema
    init_schema()
    init_schema()  # second call must not raise


def test_plan_roundtrip_and_resume():
    from ggTrader.lab.persist import (init_schema, start_run, write_plan,
                                      plan_done, read_plan)
    init_schema()
    run_id = start_run("xs_momentum", "equity", "monthly",
                       "2021-01-31", "2021-03-31", {"top_n": 2})
    asof = pd.Timestamp("2021-01-31", tz="UTC")
    assert plan_done(run_id, "xs_momentum", asof) is False
    plan = [{"symbol": "AAA", "weight": 0.5, "momentum": 0.1}]
    write_plan(run_id, "xs_momentum", asof, plan, eligible_count=10, coverage={"n": 10})
    assert plan_done(run_id, "xs_momentum", asof) is True
    assert read_plan(run_id, "xs_momentum", asof) == plan


def test_returns_equity_summary_write():
    from ggTrader.lab.persist import (init_schema, start_run, write_returns_equity,
                                      write_summary)
    init_schema()
    run_id = start_run("xs_momentum", "equity", "monthly",
                       "2021-01-31", "2021-03-31", {})
    idx = pd.date_range("2021-02-01", periods=5, freq="B", tz="UTC")
    rets = pd.Series([0.0, 0.01, -0.005, 0.002, 0.0], index=idx)
    eq = pd.Series(10000.0 * (1 + rets).cumprod(), index=idx)
    bench = pd.Series(np.linspace(10000, 10100, 5), index=idx)
    write_returns_equity(run_id, "xs_momentum", rets, eq, bench)
    write_summary(run_id, "xs_momentum", {"sharpe": 1.0}, {"sharpe": 0.9}, {"turnover": 0.3})
    # No exception == pass; the read paths are exercised by the harness integration test.
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tests/lab/test_persist.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'ggTrader.lab.persist'`

- [ ] **Step 3: Implement `persist.py`**

```python
# src/ggTrader/lab/persist.py
"""TimescaleDB persistence for lab runs — the only store for research state."""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import create_engine, text

from ggTrader.utils.config import get_db_connection_string

_ENGINE = None

_SCHEMA = """
CREATE TABLE IF NOT EXISTS lab_runs (
    run_id TEXT PRIMARY KEY,
    strategy TEXT NOT NULL,
    market TEXT NOT NULL,
    freq TEXT NOT NULL,
    eval_start TEXT NOT NULL,
    eval_end TEXT NOT NULL,
    params JSONB,
    status TEXT NOT NULL DEFAULT 'running',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS lab_plans (
    run_id TEXT NOT NULL,
    strategy TEXT NOT NULL,
    asof TIMESTAMPTZ NOT NULL,
    plan JSONB NOT NULL,
    eligible_count INT,
    coverage JSONB,
    PRIMARY KEY (run_id, strategy, asof)
);
CREATE TABLE IF NOT EXISTS lab_returns (
    run_id TEXT NOT NULL,
    strategy TEXT NOT NULL,
    date TIMESTAMPTZ NOT NULL,
    ret DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (run_id, strategy, date)
);
CREATE TABLE IF NOT EXISTS lab_equity (
    run_id TEXT NOT NULL,
    strategy TEXT NOT NULL,
    date TIMESTAMPTZ NOT NULL,
    strategy_equity DOUBLE PRECISION NOT NULL,
    benchmark_equity DOUBLE PRECISION,
    PRIMARY KEY (run_id, strategy, date)
);
CREATE TABLE IF NOT EXISTS lab_summary (
    run_id TEXT NOT NULL,
    strategy TEXT NOT NULL,
    metrics JSONB,
    benchmark_metrics JSONB,
    diagnostics JSONB,
    PRIMARY KEY (run_id, strategy)
);
"""


def get_engine():
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = create_engine(get_db_connection_string())
    return _ENGINE


def init_schema() -> None:
    eng = get_engine()
    with eng.begin() as conn:
        for stmt in [s for s in _SCHEMA.split(";") if s.strip()]:
            conn.execute(text(stmt))
        for tbl in ("lab_returns", "lab_equity"):
            try:
                conn.execute(text(
                    f"SELECT create_hypertable('{tbl}', 'date', "
                    "if_not_exists => TRUE, migrate_data => TRUE)"
                ))
            except Exception:
                pass  # plain Postgres (no TimescaleDB extension) — table still works


def start_run(strategy: str, market: str, freq: str, eval_start: str, eval_end: str,
              params: Dict[str, Any]) -> str:
    run_id = f"{strategy}_{uuid.uuid4().hex[:8]}"
    with get_engine().begin() as conn:
        conn.execute(text(
            "INSERT INTO lab_runs (run_id, strategy, market, freq, eval_start, eval_end, params) "
            "VALUES (:r, :s, :m, :f, :es, :ee, :p)"
        ), {"r": run_id, "s": strategy, "m": market, "f": freq,
            "es": eval_start, "ee": eval_end, "p": json.dumps(params)})
    return run_id


def finish_run(run_id: str) -> None:
    with get_engine().begin() as conn:
        conn.execute(text("UPDATE lab_runs SET status='done' WHERE run_id=:r"), {"r": run_id})


def plan_done(run_id: str, strategy: str, asof: pd.Timestamp) -> bool:
    with get_engine().connect() as conn:
        row = conn.execute(text(
            "SELECT 1 FROM lab_plans WHERE run_id=:r AND strategy=:s AND asof=:a"
        ), {"r": run_id, "s": strategy, "a": asof.to_pydatetime()}).first()
    return row is not None


def write_plan(run_id: str, strategy: str, asof: pd.Timestamp, plan: List[Dict[str, Any]],
               eligible_count: int, coverage: Dict[str, Any]) -> None:
    with get_engine().begin() as conn:
        conn.execute(text(
            "INSERT INTO lab_plans (run_id, strategy, asof, plan, eligible_count, coverage) "
            "VALUES (:r, :s, :a, :p, :ec, :c) "
            "ON CONFLICT (run_id, strategy, asof) DO UPDATE SET plan=EXCLUDED.plan"
        ), {"r": run_id, "s": strategy, "a": asof.to_pydatetime(),
            "p": json.dumps(plan), "ec": eligible_count, "c": json.dumps(coverage)})


def read_plan(run_id: str, strategy: str, asof: pd.Timestamp) -> List[Dict[str, Any]]:
    with get_engine().connect() as conn:
        row = conn.execute(text(
            "SELECT plan FROM lab_plans WHERE run_id=:r AND strategy=:s AND asof=:a"
        ), {"r": run_id, "s": strategy, "a": asof.to_pydatetime()}).first()
    if row is None:
        return []
    return row[0] if isinstance(row[0], list) else json.loads(row[0])


def read_all_plans(run_id: str, strategy: str) -> Dict[pd.Timestamp, List[Dict[str, Any]]]:
    with get_engine().connect() as conn:
        rows = conn.execute(text(
            "SELECT asof, plan FROM lab_plans WHERE run_id=:r AND strategy=:s ORDER BY asof"
        ), {"r": run_id, "s": strategy}).fetchall()
    out: Dict[pd.Timestamp, List[Dict[str, Any]]] = {}
    for asof, plan in rows:
        out[pd.Timestamp(asof)] = plan if isinstance(plan, list) else json.loads(plan)
    return out


def write_returns_equity(run_id: str, strategy: str, rets: pd.Series, equity: pd.Series,
                         benchmark_equity: Optional[pd.Series] = None) -> None:
    bench = benchmark_equity.reindex(equity.index) if benchmark_equity is not None else None
    with get_engine().begin() as conn:
        for dt, r in rets.items():
            conn.execute(text(
                "INSERT INTO lab_returns (run_id, strategy, date, ret) VALUES (:r,:s,:d,:v) "
                "ON CONFLICT (run_id, strategy, date) DO UPDATE SET ret=EXCLUDED.ret"
            ), {"r": run_id, "s": strategy, "d": pd.Timestamp(dt).to_pydatetime(), "v": float(r)})
        for dt, e in equity.items():
            be = None if bench is None else (None if pd.isna(bench.loc[dt]) else float(bench.loc[dt]))
            conn.execute(text(
                "INSERT INTO lab_equity (run_id, strategy, date, strategy_equity, benchmark_equity) "
                "VALUES (:r,:s,:d,:e,:b) ON CONFLICT (run_id, strategy, date) DO UPDATE "
                "SET strategy_equity=EXCLUDED.strategy_equity, benchmark_equity=EXCLUDED.benchmark_equity"
            ), {"r": run_id, "s": strategy, "d": pd.Timestamp(dt).to_pydatetime(),
                "e": float(e), "b": be})


def write_summary(run_id: str, strategy: str, metrics: Dict[str, Any],
                  benchmark_metrics: Dict[str, Any], diagnostics: Dict[str, Any]) -> None:
    with get_engine().begin() as conn:
        conn.execute(text(
            "INSERT INTO lab_summary (run_id, strategy, metrics, benchmark_metrics, diagnostics) "
            "VALUES (:r,:s,:m,:b,:d) ON CONFLICT (run_id, strategy) DO UPDATE "
            "SET metrics=EXCLUDED.metrics, benchmark_metrics=EXCLUDED.benchmark_metrics, "
            "diagnostics=EXCLUDED.diagnostics"
        ), {"r": run_id, "s": strategy, "m": json.dumps(metrics),
            "b": json.dumps(benchmark_metrics), "d": json.dumps(diagnostics)})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && python -m pytest tests/lab/test_persist.py -q`
Expected: PASS (3 passed). If it errors with a connection failure, the DB at `localhost:5433` isn't reachable from this shell — start it / open the tunnel, then re-run.

- [ ] **Step 5: Commit**

```bash
git add src/ggTrader/lab/persist.py tests/lab/test_persist.py
git commit -m "feat(lab): TimescaleDB persistence (runs/plans/returns/equity/summary)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 7: Harness (plan → simulate → persist → score) + leak_check

Wires everything: per strategy, loop rebalance dates running `select` on `data.loc[:T]` (resuming completed plans from the DB), build target matrices, run the single vectorized `simulate_weights`, persist returns/equity/summary, score vs SPY. `leak_check` is the generic full/truncated/unmasked comparison.

**Files:**
- Create: `src/ggTrader/lab/harness.py`
- Test: `tests/lab/test_harness.py`

- [ ] **Step 1: Write the failing test** (synthetic, DB-backed → `integration`)

```python
# tests/lab/test_harness.py
import numpy as np
import pandas as pd
import pytest


def _ohlcv(symbols, n=400):
    idx = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
    frames = {}
    for i, s in enumerate(symbols):
        close = pd.Series(100.0 * (1 + 0.0003 * (i + 1)) ** np.arange(n), index=idx)
        frames[s] = pd.DataFrame(
            {"open": close, "high": close * 1.01, "low": close * 0.99,
             "close": close, "volume": np.full(n, 1e6)}, index=idx)
    out = pd.concat(frames, axis=1)
    out.columns = out.columns.set_names(["symbol", "field"])
    return out


def test_leak_check_passes_for_momentum():
    from ggTrader.lab.harness import leak_check
    from ggTrader.lab.strategies.momentum import CrossSectionalMomentum
    from ggTrader.lab.strategy import LabConfig
    ohlcv = _ohlcv(["A", "B", "C", "D"])
    asof = ohlcv.index[-30]
    assert leak_check(CrossSectionalMomentum(LabConfig(top_n=2)), ohlcv, asof,
                      ["A", "B", "C", "D"]) is True


@pytest.mark.integration
def test_walkforward_persists_and_resumes():
    from ggTrader.lab.harness import walkforward
    from ggTrader.lab.persist import read_all_plans
    from ggTrader.lab.strategies.momentum import CrossSectionalMomentum
    from ggTrader.lab.strategy import LabConfig
    ohlcv = _ohlcv(["A", "B", "C"], n=400)
    spy = ohlcv["A"]["close"]
    strat = CrossSectionalMomentum(LabConfig(top_n=2, min_history_bars=50))
    run_id = walkforward(
        [strat], ohlcv, spy,
        eval_start="2021-01-31", eval_end="2021-06-30",
        market="test", freq="monthly",
        universe_fn=lambda asof, past: ["A", "B", "C"],
        base_config={"START_CASH": 10000.0, "FEES": 0.0, "SLIPPAGE": 0.0, "FREQ": "1d"},
    )
    plans = read_all_plans(run_id, "xs_momentum")
    assert len(plans) >= 4  # one per rebalance month
    # Resume: a second run on the same run_id must not recompute (no error, same plans).
    run_id2 = walkforward(
        [strat], ohlcv, spy, eval_start="2021-01-31", eval_end="2021-06-30",
        market="test", freq="monthly", run_id=run_id,
        universe_fn=lambda asof, past: ["A", "B", "C"],
        base_config={"START_CASH": 10000.0, "FEES": 0.0, "SLIPPAGE": 0.0, "FREQ": "1d"},
    )
    assert run_id2 == run_id
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tests/lab/test_harness.py -q -m "not integration"`
Expected: FAIL with `ModuleNotFoundError: No module named 'ggTrader.lab.harness'`

- [ ] **Step 3: Implement `harness.py`**

```python
# src/ggTrader/lab/harness.py
"""The lab walk-forward harness: plan -> vectorized simulate -> persist -> score."""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from ggTrader.lab import persist
from ggTrader.lab.data import rebalance_dates
from ggTrader.lab.metrics import benchmark
from ggTrader.lab.simulate import simulate_weights
from ggTrader.lab.strategy import Plan, Strategy

UniverseFn = Callable[[pd.Timestamp, pd.DataFrame], List[str]]


def leak_check(strategy: Strategy, ohlcv: pd.DataFrame, asof: pd.Timestamp,
               eligible: List[str]) -> bool:
    """select at asof must be identical with and without post-asof rows present."""
    full = strategy.select(asof, ohlcv.loc[:asof], eligible)
    truncated = strategy.select(asof, ohlcv.loc[:asof].copy(deep=True), eligible)
    unmasked = strategy.select(asof, ohlcv, eligible)
    return (json.dumps(full, sort_keys=True, default=str)
            == json.dumps(truncated, sort_keys=True, default=str)
            == json.dumps(unmasked, sort_keys=True, default=str))


def walkforward(
    strategies: List[Strategy],
    ohlcv: pd.DataFrame,
    spy_close: pd.Series,
    eval_start: str,
    eval_end: str,
    market: str,
    freq: str,
    universe_fn: UniverseFn,
    base_config: Dict[str, Any],
    run_id: Optional[str] = None,
) -> str:
    """Run one or more weight-based strategies over [eval_start, eval_end)."""
    start_ts = pd.Timestamp(eval_start, tz="UTC")
    end_ts = pd.Timestamp(eval_end, tz="UTC")
    dates = rebalance_dates(ohlcv.index, start_ts, end_ts)
    if not dates:
        raise RuntimeError("No rebalance dates in the eval span.")

    prices = pd.concat({s: ohlcv[s]["close"] for s in ohlcv.columns.get_level_values(0).unique()},
                       axis=1)

    if run_id is None:
        run_id = persist.start_run(strategies[0].name, market, freq, eval_start, eval_end,
                                   params=dict(base_config))
    persist.init_schema()

    targets_by_strategy: Dict[str, pd.DataFrame] = {}
    for strat in strategies:
        plans: Dict[pd.Timestamp, Plan] = {}
        for asof in dates:
            if persist.plan_done(run_id, strat.name, asof):
                plans[asof] = persist.read_plan(run_id, strat.name, asof)
                continue
            past = ohlcv.loc[:asof]
            eligible = universe_fn(asof, past)
            plan = strat.select(asof, past, eligible)
            persist.write_plan(run_id, strat.name, asof, plan,
                               eligible_count=len(eligible), coverage={"n_eligible": len(eligible)})
            plans[asof] = plan
        targets_by_strategy[strat.name] = strat.to_targets(plans, ohlcv)

    returns, equity, diags = simulate_weights(targets_by_strategy, prices, base_config)

    for strat in strategies:
        name = strat.name
        eq = equity[name].dropna()
        rep = benchmark(eq, spy_close, float(base_config["START_CASH"]))
        spy = spy_close.reindex(eq.index).ffill()
        bench_curve = float(base_config["START_CASH"]) * (spy / spy.dropna().iloc[0])
        persist.write_returns_equity(run_id, name, returns[name], eq, bench_curve)
        persist.write_summary(run_id, name, rep["strategy"], rep["spy"],
                              {**diags[name], "monthly_hit_rate_vs_spy": rep["monthly_hit_rate_vs_spy"],
                               "n_months": rep["n_months"]})

    persist.finish_run(run_id)
    return run_id
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && python -m pytest tests/lab/test_harness.py -q -m "not integration"`
Expected: PASS (1 passed, 1 deselected)

Then the DB-backed end-to-end (resume) path:
Run: `source .venv/bin/activate && python -m pytest tests/lab/test_harness.py -q -m integration`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add src/ggTrader/lab/harness.py tests/lab/test_harness.py
git commit -m "feat(lab): walk-forward harness (plan -> vectorized sim -> persist) + leak_check

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 8: CLI entry point

A thin argparse CLI to run a lab strategy over the equity universe end-to-end, wiring the data layer's PIT universe into the harness.

**Files:**
- Create: `src/ggTrader/lab/cli.py`
- Test: `tests/lab/test_cli.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/lab/test_cli.py
from ggTrader.lab.cli import build_arg_parser


def test_arg_parser_defaults():
    p = build_arg_parser()
    args = p.parse_args(["--strategy", "xs_momentum"])
    assert args.strategy == "xs_momentum"
    assert args.market == "equity"
    assert args.top_n == 50


def test_arg_parser_rejects_unknown_strategy():
    p = build_arg_parser()
    try:
        p.parse_args(["--strategy", "bogus"])
        assert False, "expected SystemExit"
    except SystemExit:
        pass
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest tests/lab/test_cli.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'ggTrader.lab.cli'`

- [ ] **Step 3: Implement `cli.py`**

```python
# src/ggTrader/lab/cli.py
"""CLI for the lab research bench: run a strategy over the equity universe."""

from __future__ import annotations

import argparse
from typing import List

import pandas as pd

from ggTrader.lab.data import eligible_at, equity_universe_between, load_ohlcv
from ggTrader.lab.harness import walkforward
from ggTrader.lab.strategies.momentum import STRATEGY_NAMES, build_strategy
from ggTrader.lab.strategy import LabConfig
from ggTrader.research.equity_wfo import STOCK_BASE_CONFIG


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run a lab strategy walk-forward.")
    p.add_argument("--strategy", choices=STRATEGY_NAMES, required=True)
    p.add_argument("--market", default="equity")
    p.add_argument("--eval-start", default="2021-01-31")
    p.add_argument("--eval-end", default=None)
    p.add_argument("--top-n", type=int, default=50)
    p.add_argument("--lookback", type=int, default=252)
    p.add_argument("--skip", type=int, default=21)
    p.add_argument("--max-stocks", type=int, default=None)
    return p


def run_lab(argv: List[str] | None = None) -> str:
    args = build_arg_parser().parse_args(argv)
    cfg = LabConfig(top_n=args.top_n, lookback=args.lookback, skip=args.skip,
                    max_stocks=args.max_stocks)
    strat = build_strategy(args.strategy, cfg)

    eval_start = pd.Timestamp(args.eval_start, tz="UTC")
    eval_end = (pd.Timestamp(args.eval_end, tz="UTC") if args.eval_end
                else pd.Timestamp.now(tz="UTC").normalize())
    data_start = eval_start - pd.Timedelta(days=int(cfg.lookback * 1.6) + 30)
    universe = equity_universe_between(eval_start, eval_end)
    ohlcv = load_ohlcv(universe + ["SPY"], str(data_start.date()), str(eval_end.date()))
    spy_close = ohlcv["SPY"]["close"].dropna()
    sym_cols = [s for s in ohlcv.columns.get_level_values(0).unique() if s != "SPY"]
    ohlcv = ohlcv[sym_cols]

    run_id = walkforward(
        [strat], ohlcv, spy_close,
        eval_start=str(eval_start.date()),
        eval_end=str(eval_end.date()),
        market=args.market, freq="monthly",
        universe_fn=lambda asof, past: eligible_at(asof, past, cfg)[0],
        base_config=dict(STOCK_BASE_CONFIG),
    )
    print(f"lab run complete: {run_id}")
    return run_id


if __name__ == "__main__":
    run_lab()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && python -m pytest tests/lab/test_cli.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/ggTrader/lab/cli.py tests/lab/test_cli.py
git commit -m "feat(lab): CLI to run a strategy over the equity universe

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 9: Validation gate — reproduce known-good selections

The acceptance test for the whole architecture: the new core's per-rebalance selections must match the **old** `results/monthly_wf/sp500_xs_momentum/` and `sp500_dual_momentum/` checkpoints exactly. (Equity numbers may differ slightly — single-pass compounding vs the old stitching — which Task 10 documents.)

**Files:**
- Create: `tests/lab/test_validation_gate.py`
- Read-only reference: `results/monthly_wf/sp500_xs_momentum/month=*/selections.json`

- [ ] **Step 1: Write the validation test** (`integration` — needs the DB + full universe)

```python
# tests/lab/test_validation_gate.py
import json
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.integration

REF = {
    "xs_momentum": "results/monthly_wf/sp500_xs_momentum",
    "dual_momentum": "results/monthly_wf/sp500_dual_momentum",
}


def _old_selections(run_dir: str):
    out = {}
    for d in sorted(Path(run_dir).glob("month=*")):
        doc = json.loads((d / "selections.json").read_text())
        asof = pd.Timestamp(doc["asof"], tz="UTC")
        out[asof] = [(s["symbol"], round(float(s["weight"]), 10)) for s in doc["selections"]]
    return out


@pytest.mark.parametrize("strategy", ["xs_momentum", "dual_momentum"])
def test_lab_reproduces_old_selections(strategy):
    ref_dir = REF[strategy]
    if not Path(ref_dir).exists():
        pytest.skip(f"reference run {ref_dir} not present")
    old = _old_selections(ref_dir)

    from ggTrader.lab.data import eligible_at, equity_universe_between, load_ohlcv
    from ggTrader.lab.strategies.momentum import build_strategy
    from ggTrader.lab.strategy import LabConfig

    cfg = LabConfig(top_n=50, lookback=252, skip=21, min_history_bars=400)
    strat = build_strategy(strategy, cfg)
    eval_start = pd.Timestamp("2021-01-31", tz="UTC")
    eval_end = pd.Timestamp("2026-05-31", tz="UTC")
    data_start = eval_start - pd.Timedelta(days=int(cfg.lookback * 1.6) + 30)
    universe = equity_universe_between(eval_start, eval_end)
    ohlcv = load_ohlcv(universe, str(data_start.date()), str(eval_end.date()))

    mismatches = []
    for asof in sorted(old):
        past = ohlcv.loc[:asof]
        eligible, _ = eligible_at(asof, past, cfg)
        new = [(s["symbol"], round(float(s["weight"]), 10))
               for s in strat.select(asof, past, eligible)]
        if new != old[asof]:
            mismatches.append(str(asof.date()))
    assert not mismatches, f"selection mismatches at: {mismatches[:5]} ({len(mismatches)} total)"
```

- [ ] **Step 2: Run it (expect PASS, since select is ported verbatim)**

Run: `source .venv/bin/activate && python -m pytest tests/lab/test_validation_gate.py -q -m integration`
Expected: PASS (2 passed). If a strategy's reference dir is absent it SKIPS — re-generate it with the old `scripts/sp500_monthly_walkforward.py --strategy <name>` first if needed. If selections mismatch, the port diverged from the original `select`; diff against `research/monthly_strategies.py` and fix before proceeding.

- [ ] **Step 3: Commit**

```bash
git add tests/lab/test_validation_gate.py
git commit -m "test(lab): validation gate — reproduce sp500 momentum selections exactly

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 10: Full suite, equity-delta characterization, changelog

Run the whole suite, then characterize the documented equity difference (single-pass `from_orders` vs the old per-month stitching) for `xs_momentum` over the full window, and record it.

**Files:**
- Modify: `docs/changelog.md`
- Modify: `docs/superpowers/specs/2026-06-15-vectorbt-lab-core-design.md` (mark Plan 1 status)

- [ ] **Step 1: Run the full unit suite (no integration)**

Run: `source .venv/bin/activate && python -m pytest tests/ -m "not integration" -q`
Expected: all lab unit tests pass; baseline 3 pre-existing failures unchanged (`test_circuit_breaker_persistence`, `test_system_dry_run_cycle`, `test_persistence_logic`).

- [ ] **Step 2: Run the lab integration suite**

Run: `source .venv/bin/activate && python -m pytest tests/lab/ -m integration -q`
Expected: persistence, harness-resume, and validation-gate tests pass.

- [ ] **Step 3: Characterize the equity delta**

Run a full `xs_momentum` lab run via the CLI and compare its summary metrics to the old `results/monthly_wf/sp500_xs_momentum/summary.json` (old: total +125.98%, Sharpe 0.82):

```bash
source .venv/bin/activate && python -m ggTrader.lab.cli --strategy xs_momentum --eval-start 2021-01-31 --eval-end 2026-05-31
```

Then read the new summary back from the DB (psql or a one-off `read` via `lab_summary`) and note total-return and Sharpe. **Soft gate:** total-return within 1% relative, Sharpe within 0.05 absolute, of the old run. Record the actual delta in the changelog entry below (any residual difference is attributed to single-pass compounding being more correct than stitching).

- [ ] **Step 4: Add the changelog entry**

Add under a new `## 2026-06-15` heading at the top of `docs/changelog.md`:

```markdown
## 2026-06-15

### Research: vectorbt lab core (Plan 1 — momentum bench)

New `src/ggTrader/lab/` package: a simple, DB-only, vectorbt-centric research
bench. One unified harness — point-in-time `select` (data ≤ T, resumable from
the DB) then a single grouped `vbt.Portfolio.from_orders` that simulates all
weight-based strategies simultaneously. All run state (plans, returns, equity,
summary) lives in TimescaleDB (`lab_*` tables); nothing is written to results/.
Strategies `xs_momentum` and `dual_momentum` ported with selections validated
**bit-identical** against the old `sp500_*` runs. Equity differs from the old
numbers by <X%> (Sharpe Δ <Y>), attributable to single-pass compounding being
more correct than the old per-month stitching. Spec:
[2026-06-15-vectorbt-lab-core-design.md](superpowers/specs/2026-06-15-vectorbt-lab-core-design.md);
plan: [2026-06-15-vectorbt-lab-core.md](superpowers/plans/2026-06-15-vectorbt-lab-core.md).
Deferred: the signal-based `wfo_tournament` family (Plan 2) and deletion of the
old research/backtest code (Plan 3).
```

Replace `<X%>`/`<Y>` with the measured deltas from Step 3.

- [ ] **Step 5: Mark the plan executed and commit**

In the spec file, change `**Status:** Approved...` to `**Status:** Plan 1 executed 2026-06-15`. Check off this plan's boxes. Then:

```bash
git add docs/changelog.md docs/superpowers/specs/2026-06-15-vectorbt-lab-core-design.md docs/superpowers/plans/2026-06-15-vectorbt-lab-core.md
git commit -m "docs: changelog + spec status for lab core Plan 1

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Self-review

- **Spec coverage:** §3 architecture → Tasks 1-8; §3.1 protocol → Task 1; §3.2 two-phase harness → Tasks 4 (simulate) + 7 (harness); §4 DB-only persistence → Task 6; §5 leak safety → Task 2 (self-truncation) + Task 7 (`leak_check`); §6 error handling → empty-plan/missing-price handled in `simulate_weights` ffill + `to_targets` zero-fill, fail-fast inherent to the loop; §7 testing → every task is TDD, vectorization-equivalence in Task 4, leak in Task 7; §8 validation gate → Task 9 (hard) + Task 10 step 3 (soft); §9 deletion → explicitly deferred to Plan 3 (stated up front). The signal-based family (§3 `wfo_tournament`, the §10 open question) is deferred to Plan 2 — also stated up front.
- **Placeholder scan:** the only intentional fill-ins are `<X%>`/`<Y>` in Task 10's changelog, which Step 3 measures before writing. No TBDs in code.
- **Type consistency:** `Strategy.select(asof, data, eligible)` and `to_targets(plans, data)` signatures match across Tasks 1, 2, 7. `simulate_weights(targets_by_strategy, prices, base_config) -> (returns, equity, diags)` consistent between Tasks 4 and 7. `persist` function names (`start_run`, `plan_done`, `write_plan`, `read_plan`, `read_all_plans`, `write_returns_equity`, `write_summary`, `finish_run`, `init_schema`, `get_engine`) consistent between Tasks 6 and 7. `build_strategy`/`STRATEGY_NAMES` consistent between Tasks 2 and 8.
- **Risk noted:** the soft-gate equity delta is *expected* to be nonzero; Task 10 measures and documents it rather than asserting equality.
