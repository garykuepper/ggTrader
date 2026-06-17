# Lab Plan 2: Signal-Based Strategy Family (from_signals harness + WFO tournament)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the lab bench to support signal-based strategies (entry/exit boolean signals via `vbt.Portfolio.from_signals`), add `EmaCrossSignal` as a concrete simple strategy, add `WfoTournamentSignal` which runs a mini IS/OOS EMA-combo tournament each rebalance to pick the best signal params, and wire everything into the existing `walkforward` harness alongside the existing weight-based strategies.

**Architecture:** `simulate_signals` (parallel to `simulate_weights`) runs one grouped `from_signals` call for all signal strategies simultaneously. `walkforward` detects `target_kind` and splits strategies into two groups — weight strategies go to `simulate_weights`, signal strategies go to `simulate_signals` — then merges their equity/returns dicts before persisting. Signal strategies return a `SignalTargets(entries, exits)` NamedTuple from `to_targets` instead of a plain weight DataFrame. The WFO tournament in `WfoTournamentSignal.select` tries 4 EMA combos on the IS window and picks the best combo by portfolio Sharpe; `to_targets` generates piecewise signals for each rebalance period using the period's winning params.

**Tech Stack:** Python, vectorbt 0.28.5, pandas/numpy, TimescaleDB, pytest. Activate venv: `source .venv/bin/activate`.

**Spec:** `docs/superpowers/specs/2026-06-15-vectorbt-lab-core-design.md` §3.2 (harness two phases), §10 (open question resolved: two grouped calls, merged metrics).

**Conventions:** ruff line length 100; absolute imports from `ggTrader`; PostToolUse hook runs ruff autofix. Commit trailer: `Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>`.

---

## File Map

| File | Action | Reason |
|---|---|---|
| `src/ggTrader/lab/strategy.py` | Modify | Add `SignalTargets` NamedTuple; update `Strategy.to_targets` return annotation |
| `src/ggTrader/lab/simulate.py` | Modify | Add `simulate_signals()` parallel to `simulate_weights()` |
| `src/ggTrader/lab/strategies/signals.py` | Create | `EmaCrossSignal`, `WfoTournamentSignal`, `build_signal_strategy` |
| `src/ggTrader/lab/harness.py` | Modify | Split strategies by `target_kind`, merge results from two sim calls |
| `src/ggTrader/lab/cli.py` | Modify | Add signal strategy names to `--strategy` choices |
| `tests/lab/test_simulate_signals.py` | Create | Unit tests for `simulate_signals` |
| `tests/lab/test_signals.py` | Create | Unit tests for `EmaCrossSignal` and `WfoTournamentSignal` |
| `tests/lab/test_harness_signals.py` | Create | Integration test for mixed weight+signal `walkforward` |
| `docs/superpowers/specs/2026-06-15-vectorbt-lab-core-design.md` | Modify | Mark Plan 2 executed |
| `docs/changelog.md` | Modify | Add 2026-06-16 Plan 2 entry |

---

## Task 1: `SignalTargets` NamedTuple + `simulate_signals`

`SignalTargets` is the return type of signal strategies' `to_targets`. `simulate_signals` runs ONE grouped `vbt.Portfolio.from_signals` call across all signal strategies simultaneously, mirroring the `simulate_weights` structure exactly.

**Files:**
- Modify: `src/ggTrader/lab/strategy.py`
- Modify: `src/ggTrader/lab/simulate.py`
- Create: `tests/lab/test_simulate_signals.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/lab/test_simulate_signals.py`:

```python
# tests/lab/test_simulate_signals.py
import numpy as np
import pandas as pd
import pytest

from ggTrader.lab.strategy import SignalTargets
from ggTrader.lab.simulate import simulate_signals

BASE = {"START_CASH": 10000.0, "FEES": 0.0, "SLIPPAGE": 0.0, "FREQ": "1d",
        "SIGNAL_POSITION_SIZE": 0.5}


def _prices(n=40):
    idx = pd.date_range("2021-01-01", periods=n, freq="B", tz="UTC")
    return pd.DataFrame({
        "A": 100.0 * 1.01 ** np.arange(n),
        "B": np.full(n, 50.0),
    }, index=idx)


def test_simulate_signals_buy_and_hold_a():
    prices = _prices(40)
    # Buy A on bar 2, never exit
    entries = pd.DataFrame(False, index=prices.index, columns=prices.columns)
    exits = pd.DataFrame(False, index=prices.index, columns=prices.columns)
    entries.iloc[2, 0] = True  # buy A
    st = SignalTargets(entries=entries, exits=exits)

    rets, equity, diags = simulate_signals({"strat_a": st}, prices, BASE)
    assert "strat_a" in equity.columns
    assert equity["strat_a"].iloc[-1] > BASE["START_CASH"]  # A is rising
    assert diags["strat_a"]["n_symbols"] == 2


def test_simulate_signals_two_strategies_independent():
    prices = _prices(40)
    e1 = pd.DataFrame(False, index=prices.index, columns=prices.columns)
    x1 = pd.DataFrame(False, index=prices.index, columns=prices.columns)
    e1.iloc[2, 0] = True  # strat1: buy A

    e2 = pd.DataFrame(False, index=prices.index, columns=prices.columns)
    x2 = pd.DataFrame(False, index=prices.index, columns=prices.columns)
    e2.iloc[2, 1] = True  # strat2: buy B (flat)

    rets_both, eq_both, _ = simulate_signals(
        {"s1": SignalTargets(e1, x1), "s2": SignalTargets(e2, x2)}, prices, BASE
    )
    rets_s1, eq_s1, _ = simulate_signals({"s1": SignalTargets(e1, x1)}, prices, BASE)

    # Running together must not change individual equity curves
    pd.testing.assert_series_equal(eq_both["s1"], eq_s1["s1"], check_names=False)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tests/lab/test_simulate_signals.py -q`
Expected: FAIL with `ImportError: cannot import name 'SignalTargets' from 'ggTrader.lab.strategy'`

- [ ] **Step 3: Add `SignalTargets` to `strategy.py`**

In `src/ggTrader/lab/strategy.py`, add after the `Plan` type alias (line 10) and update the imports:

```python
from dataclasses import dataclass
from typing import Any, Dict, List, NamedTuple, Protocol, Union

import pandas as pd

Plan = List[Dict[str, Any]]


class SignalTargets(NamedTuple):
    """Return type for signal-based strategies' to_targets method."""
    entries: pd.DataFrame   # (time x symbol) boolean — True = entry bar
    exits: pd.DataFrame     # (time x symbol) boolean — True = exit bar
```

Also update the `Strategy` protocol's `to_targets` return annotation to `Union[pd.DataFrame, "SignalTargets"]`:

```python
    def to_targets(
        self, plans: Dict[pd.Timestamp, Plan], data: pd.DataFrame
    ) -> "Union[pd.DataFrame, SignalTargets]":
        """Whole-window target matrix from per-rebalance plans.

        Weight strategies return pd.DataFrame (time x symbol, weight values).
        Signal strategies return SignalTargets(entries, exits) boolean frames.
        """
        ...
```

- [ ] **Step 4: Add `simulate_signals` to `simulate.py`**

Append to `src/ggTrader/lab/simulate.py` (after `simulate_weights`):

```python
from ggTrader.lab.strategy import SignalTargets  # add to imports at top


def simulate_signals(
    targets_by_strategy: Dict[str, "SignalTargets"],
    prices: pd.DataFrame,
    base_config: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """Simulate every signal-based strategy in ONE from_signals call.

    Args:
        targets_by_strategy: name -> SignalTargets(entries, exits) where each
            frame is (time x symbol) boolean — True = entry/exit on that bar.
        prices: (time x symbol) close prices covering every target column.
        base_config: START_CASH, FEES, SLIPPAGE, FREQ.
            Optional SIGNAL_POSITION_SIZE (fraction of portfolio per entry, default 0.02).

    Returns:
        (returns_df, equity_df, diags) each keyed by strategy name (columns).
    """
    names = list(targets_by_strategy)
    entry_blocks, exit_blocks, close_blocks, groups = [], [], [], []

    for name in names:
        st = targets_by_strategy[name]
        cols = pd.MultiIndex.from_product(
            [[name], st.entries.columns], names=["strategy", "symbol"]
        )
        entry_blocks.append(st.entries.set_axis(cols, axis=1))
        exit_blocks.append(st.exits.set_axis(cols, axis=1))
        px = prices[st.entries.columns].reindex(st.entries.index).ffill()
        close_blocks.append(px.set_axis(cols, axis=1))
        groups.extend([name] * st.entries.shape[1])

    entries = pd.concat(entry_blocks, axis=1).fillna(False)
    exits = pd.concat(exit_blocks, axis=1).fillna(False)
    close = pd.concat(close_blocks, axis=1)

    pf = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        size=float(base_config.get("SIGNAL_POSITION_SIZE", 0.02)),
        size_type="percent",
        init_cash=float(base_config["START_CASH"]),
        fees=float(base_config["FEES"]),
        slippage=float(base_config["SLIPPAGE"]),
        freq=base_config["FREQ"],
        cash_sharing=True,
        group_by=pd.Index(groups, name="strategy"),
    ).copy()

    value = pf.value()
    if isinstance(value, pd.Series):
        value = value.to_frame(names[0])
    value = value[names]
    returns = value.pct_change().fillna(0.0)
    diags = {
        name: {"n_strategies": 1, "n_symbols": int(targets_by_strategy[name].entries.shape[1])}
        for name in names
    }
    return returns, value, diags
```

Note: the import of `SignalTargets` at the top of simulate.py must use a string annotation or a `TYPE_CHECKING` guard to avoid circular imports since `strategy.py` doesn't import `simulate.py`. Use a plain string in the type hint for the function signature (shown above with quotes).

Actually, avoid the circular import entirely: `simulate_signals` takes a Dict but the type annotation can remain loose. Change the function signature to:

```python
def simulate_signals(
    targets_by_strategy: Dict[str, Any],  # values are SignalTargets instances
    prices: pd.DataFrame,
    base_config: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, Any]]]:
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `source .venv/bin/activate && python -m pytest tests/lab/test_simulate_signals.py -v`
Expected: PASS (2 passed)

- [ ] **Step 6: Run full lab unit suite**

Run: `source .venv/bin/activate && python -m pytest tests/lab/ -m "not integration" -q`
Expected: all existing tests still pass (no regressions).

- [ ] **Step 7: Commit**

```bash
git add src/ggTrader/lab/strategy.py src/ggTrader/lab/simulate.py tests/lab/test_simulate_signals.py
git commit -m "feat(lab): SignalTargets NamedTuple + simulate_signals (from_signals grouped call)

Adds the signal-strategy simulation path parallel to simulate_weights.
One grouped from_signals call handles all signal strategies simultaneously.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 2: `EmaCrossSignal` strategy

A signal strategy that generates buy signals when the fast EMA crosses above the slow EMA and sell signals on the reverse cross. `select` returns all eligible symbols with fixed params; `to_targets` computes whole-window EMA signals using pandas ewm.

**Files:**
- Create: `src/ggTrader/lab/strategies/signals.py`
- Create: `tests/lab/test_signals.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/lab/test_signals.py`:

```python
# tests/lab/test_signals.py
import numpy as np
import pandas as pd

from ggTrader.lab.strategy import LabConfig, SignalTargets
from ggTrader.lab.strategies.signals import EmaCrossSignal, build_signal_strategy


def _idx(n, start="2020-01-01"):
    return pd.date_range(start, periods=n, freq="B", tz="UTC")


def _ohlcv(symbols, n=600):
    idx = _idx(n)
    frames = {}
    for i, s in enumerate(symbols):
        close = pd.Series(100.0 * (1 + 0.0003 * (i + 1)) ** np.arange(n), index=idx)
        frames[s] = pd.DataFrame(
            {"open": close, "high": close * 1.01, "low": close * 0.99,
             "close": close, "volume": np.full(n, 1e6)}, index=idx
        )
    out = pd.concat(frames, axis=1)
    out.columns = out.columns.set_names(["symbol", "field"])
    return out


def test_ema_cross_select_returns_eligible_symbols():
    ohlcv = _ohlcv(["A", "B", "C"])
    strat = EmaCrossSignal(LabConfig(min_history_bars=400))
    asof = ohlcv.index[-1]
    sels = strat.select(asof, ohlcv, ["A", "B", "C"])
    assert [s["symbol"] for s in sels] == ["A", "B", "C"]
    assert all("ema_fast" in s and "ema_slow" in s for s in sels)
    assert all(s["weight"] == 0.0 for s in sels)


def test_ema_cross_select_respects_min_history():
    ohlcv = _ohlcv(["A"], n=200)  # fewer bars than min_history_bars=400
    strat = EmaCrossSignal(LabConfig(min_history_bars=400))
    sels = strat.select(ohlcv.index[-1], ohlcv, ["A"])
    assert sels == []


def test_ema_cross_to_targets_returns_signal_targets():
    ohlcv = _ohlcv(["A", "B"])
    strat = EmaCrossSignal(LabConfig(min_history_bars=100))
    asof1 = ohlcv.index[300]
    asof2 = ohlcv.index[450]
    plans = {
        asof1: [{"symbol": "A", "weight": 0.0, "ema_fast": 20, "ema_slow": 50}],
        asof2: [{"symbol": "A", "weight": 0.0, "ema_fast": 20, "ema_slow": 50},
                {"symbol": "B", "weight": 0.0, "ema_fast": 20, "ema_slow": 50}],
    }
    result = strat.to_targets(plans, ohlcv)
    assert isinstance(result, SignalTargets)
    assert result.entries.shape == result.exits.shape
    assert set(result.entries.columns) == {"A", "B"}
    assert result.entries.dtype == bool
    assert result.exits.dtype == bool


def test_ema_cross_no_lookahead():
    ohlcv = _ohlcv(["A"])
    strat = EmaCrossSignal(LabConfig(min_history_bars=100))
    asof = ohlcv.index[-30]
    full = strat.select(asof, ohlcv.loc[:asof], ["A"])
    truncated = strat.select(asof, ohlcv.loc[:asof].copy(), ["A"])
    unmasked = strat.select(asof, ohlcv, ["A"])
    import json
    assert (json.dumps(full, sort_keys=True)
            == json.dumps(truncated, sort_keys=True)
            == json.dumps(unmasked, sort_keys=True))


def test_build_signal_strategy_dispatch():
    cfg = LabConfig()
    assert build_signal_strategy("ema_cross", cfg).name == "ema_cross"
    try:
        build_signal_strategy("bogus", cfg)
        assert False, "expected ValueError"
    except ValueError:
        pass
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tests/lab/test_signals.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'ggTrader.lab.strategies.signals'`

- [ ] **Step 3: Implement `strategies/signals.py`**

Create `src/ggTrader/lab/strategies/signals.py`:

```python
# src/ggTrader/lab/strategies/signals.py
"""Signal-based lab strategies: entry/exit boolean signals via from_signals."""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from ggTrader.lab.strategy import LabConfig, Plan, SignalTargets


class EmaCrossSignal:
    """EMA crossover signal strategy.

    select() returns all eligible symbols with fixed EMA params.
    to_targets() computes whole-window entry/exit signals using pandas ewm.
    """

    name = "ema_cross"
    target_kind = "signals"

    def __init__(self, cfg: LabConfig, ema_fast: int = 20, ema_slow: int = 50) -> None:
        self.cfg = cfg
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow

    def select(self, asof: pd.Timestamp, data: pd.DataFrame, eligible: List[str]) -> Plan:
        """All eligible symbols with enough history — fixed EMA params."""
        data = data.loc[:asof]
        have = set(data.columns.get_level_values(0).unique())
        return [
            {
                "symbol": s,
                "weight": 0.0,
                "ema_fast": self.ema_fast,
                "ema_slow": self.ema_slow,
            }
            for s in eligible
            if s in have and len(data[s]["close"].dropna()) >= self.cfg.min_history_bars
        ]

    def to_targets(
        self, plans: Dict[pd.Timestamp, Plan], data: pd.DataFrame
    ) -> SignalTargets:
        """Compute EMA cross signals over the full window for all selected symbols."""
        symbols = sorted({s["symbol"] for plan in plans.values() for s in plan})
        close = pd.concat(
            {s: data[s]["close"] for s in symbols if s in data.columns.get_level_values(0)},
            axis=1,
        )

        ema_f = close.ewm(span=self.ema_fast, adjust=False).mean()
        ema_s = close.ewm(span=self.ema_slow, adjust=False).mean()

        entries = ((ema_f > ema_s) & (ema_f.shift(1) <= ema_s.shift(1))).fillna(False)
        exits = ((ema_f < ema_s) & (ema_f.shift(1) >= ema_s.shift(1))).fillna(False)

        return SignalTargets(entries=entries.astype(bool), exits=exits.astype(bool))


_SIGNAL_REGISTRY = {
    "ema_cross": EmaCrossSignal,
}

SIGNAL_STRATEGY_NAMES = tuple(_SIGNAL_REGISTRY)


def build_signal_strategy(name: str, cfg: LabConfig) -> EmaCrossSignal:
    if name not in _SIGNAL_REGISTRY:
        raise ValueError(f"Unknown signal strategy {name!r}. Available: {SIGNAL_STRATEGY_NAMES}")
    return _SIGNAL_REGISTRY[name](cfg)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && python -m pytest tests/lab/test_signals.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Run full lab unit suite**

Run: `source .venv/bin/activate && python -m pytest tests/lab/ -m "not integration" -q`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/ggTrader/lab/strategies/signals.py tests/lab/test_signals.py
git commit -m "feat(lab): EmaCrossSignal strategy + signal strategy registry

Implements the first signal-based lab strategy using pandas EMA crossover.
select() returns eligible symbols with fixed params; to_targets() computes
whole-window boolean entry/exit signals.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 3: Extend `walkforward` for signal strategies

The harness currently only handles weight strategies. Extend it to detect `target_kind`, split strategies into weight/signal groups, run the appropriate sim, and merge results for persistence.

**Files:**
- Modify: `src/ggTrader/lab/harness.py`
- Create: `tests/lab/test_harness_signals.py`

- [ ] **Step 1: Write the failing integration test**

Create `tests/lab/test_harness_signals.py`:

```python
# tests/lab/test_harness_signals.py
import numpy as np
import pandas as pd
import pytest


def _ohlcv(symbols, n=600):
    idx = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
    frames = {}
    for i, s in enumerate(symbols):
        close = pd.Series(100.0 * (1 + 0.0002 * (i + 1)) ** np.arange(n), index=idx)
        frames[s] = pd.DataFrame(
            {"open": close, "high": close * 1.01, "low": close * 0.99,
             "close": close, "volume": np.full(n, 1e6)}, index=idx
        )
    out = pd.concat(frames, axis=1)
    out.columns = out.columns.set_names(["symbol", "field"])
    return out


@pytest.mark.integration
def test_walkforward_signal_strategy_persists_and_resumes():
    from ggTrader.lab.harness import walkforward
    from ggTrader.lab.persist import read_all_plans
    from ggTrader.lab.strategies.signals import EmaCrossSignal
    from ggTrader.lab.strategy import LabConfig

    ohlcv = _ohlcv(["A", "B", "C"], n=600)
    spy = ohlcv["A"]["close"]
    strat = EmaCrossSignal(LabConfig(min_history_bars=50))

    run_id = walkforward(
        [strat], ohlcv, spy,
        eval_start="2022-01-31", eval_end="2022-06-30",
        market="test", freq="monthly",
        universe_fn=lambda asof, past: ["A", "B", "C"],
        base_config={
            "START_CASH": 10000.0, "FEES": 0.0, "SLIPPAGE": 0.0,
            "FREQ": "1d", "SIGNAL_POSITION_SIZE": 0.3,
        },
    )
    plans = read_all_plans(run_id, "ema_cross")
    assert len(plans) >= 4

    # Resume: second call with same run_id must not fail
    run_id2 = walkforward(
        [strat], ohlcv, spy,
        eval_start="2022-01-31", eval_end="2022-06-30",
        market="test", freq="monthly", run_id=run_id,
        universe_fn=lambda asof, past: ["A", "B", "C"],
        base_config={
            "START_CASH": 10000.0, "FEES": 0.0, "SLIPPAGE": 0.0,
            "FREQ": "1d", "SIGNAL_POSITION_SIZE": 0.3,
        },
    )
    assert run_id2 == run_id


@pytest.mark.integration
def test_walkforward_mixed_weight_and_signal_strategies():
    from ggTrader.lab.harness import walkforward
    from ggTrader.lab.strategies.momentum import CrossSectionalMomentum
    from ggTrader.lab.strategies.signals import EmaCrossSignal
    from ggTrader.lab.strategy import LabConfig

    ohlcv = _ohlcv(["A", "B", "C", "D"], n=600)
    spy = ohlcv["A"]["close"]
    weight_strat = CrossSectionalMomentum(LabConfig(top_n=2, min_history_bars=50))
    signal_strat = EmaCrossSignal(LabConfig(min_history_bars=50))

    run_id = walkforward(
        [weight_strat, signal_strat], ohlcv, spy,
        eval_start="2022-01-31", eval_end="2022-06-30",
        market="test", freq="monthly",
        universe_fn=lambda asof, past: ["A", "B", "C", "D"],
        base_config={
            "START_CASH": 10000.0, "FEES": 0.0, "SLIPPAGE": 0.0,
            "FREQ": "1d", "SIGNAL_POSITION_SIZE": 0.25,
        },
    )
    assert run_id is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tests/lab/test_harness_signals.py -q -m integration`
Expected: FAIL because `walkforward` passes signal strategies to `simulate_weights` which doesn't handle `SignalTargets`.

- [ ] **Step 3: Update `harness.py`**

Replace `src/ggTrader/lab/harness.py` with this updated version:

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
from ggTrader.lab.simulate import simulate_signals, simulate_weights
from ggTrader.lab.strategy import Plan, SignalTargets, Strategy

UniverseFn = Callable[[pd.Timestamp, pd.DataFrame], List[str]]


def leak_check(
    strategy: Strategy, ohlcv: pd.DataFrame, asof: pd.Timestamp, eligible: List[str]
) -> bool:
    """select at asof must be identical with and without post-asof rows present."""
    full = strategy.select(asof, ohlcv.loc[:asof], eligible)
    truncated = strategy.select(asof, ohlcv.loc[:asof].copy(deep=True), eligible)
    unmasked = strategy.select(asof, ohlcv, eligible)
    return (
        json.dumps(full, sort_keys=True, default=str)
        == json.dumps(truncated, sort_keys=True, default=str)
        == json.dumps(unmasked, sort_keys=True, default=str)
    )


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
    """Run one or more strategies (weight or signal) over [eval_start, eval_end).

    Weight strategies (target_kind='weights') are simulated via from_orders.
    Signal strategies (target_kind='signals') are simulated via from_signals.
    Both groups run in a single grouped vbt call each; results are merged.
    """
    start_ts = pd.Timestamp(eval_start, tz="UTC")
    end_ts = pd.Timestamp(eval_end, tz="UTC")
    dates = rebalance_dates(ohlcv.index, start_ts, end_ts)
    if not dates:
        raise RuntimeError("No rebalance dates in the eval span.")

    prices = pd.concat(
        {s: ohlcv[s]["close"] for s in ohlcv.columns.get_level_values(0).unique()}, axis=1
    )

    persist.init_schema()
    if run_id is None:
        run_name = strategies[0].name if len(strategies) == 1 else "multi"
        run_id = persist.start_run(
            run_name, market, freq, eval_start, eval_end, params=dict(base_config)
        )

    # Phase 1: plan phase (point-in-time select, resumable)
    weight_targets: Dict[str, pd.DataFrame] = {}
    signal_targets: Dict[str, SignalTargets] = {}

    for strat in strategies:
        plans: Dict[pd.Timestamp, Plan] = {}
        for asof in dates:
            if persist.plan_done(run_id, strat.name, asof):
                plans[asof] = persist.read_plan(run_id, strat.name, asof)
                continue
            past = ohlcv.loc[:asof]
            eligible = universe_fn(asof, past)
            plan = strat.select(asof, past, eligible)
            persist.write_plan(
                run_id, strat.name, asof, plan,
                eligible_count=len(eligible),
                coverage={"n_eligible": len(eligible)},
            )
            plans[asof] = plan

        targets = strat.to_targets(plans, ohlcv)
        if strat.target_kind == "signals":
            signal_targets[strat.name] = targets  # type: ignore[assignment]
        else:
            weight_targets[strat.name] = targets  # type: ignore[assignment]

    # Phase 2: vectorized simulation (one grouped vbt call per family)
    all_returns: Dict[str, pd.Series] = {}
    all_equity: Dict[str, pd.Series] = {}
    all_diags: Dict[str, Dict[str, Any]] = {}

    if weight_targets:
        w_rets, w_eq, w_diags = simulate_weights(weight_targets, prices, base_config)
        for name in weight_targets:
            all_returns[name] = w_rets[name]
            all_equity[name] = w_eq[name]
            all_diags[name] = w_diags[name]

    if signal_targets:
        s_rets, s_eq, s_diags = simulate_signals(signal_targets, prices, base_config)
        for name in signal_targets:
            all_returns[name] = s_rets[name]
            all_equity[name] = s_eq[name]
            all_diags[name] = s_diags[name]

    # Score only the traded span (trim warmup cash prefix)
    forward = ohlcv.index[ohlcv.index > dates[0]]
    trade_start = forward[0] if len(forward) else dates[0]

    for strat in strategies:
        name = strat.name
        eq = all_equity[name].loc[trade_start:].dropna()
        rets = all_returns[name].loc[trade_start:]
        rep = benchmark(eq, spy_close, float(base_config["START_CASH"]))
        spy = spy_close.reindex(eq.index).ffill()
        bench_curve = float(base_config["START_CASH"]) * (spy / spy.dropna().iloc[0])
        persist.write_returns_equity(run_id, name, rets, eq, bench_curve)
        persist.write_summary(
            run_id, name, rep["strategy"], rep["spy"],
            {
                **all_diags[name],
                "monthly_hit_rate_vs_spy": rep["monthly_hit_rate_vs_spy"],
                "n_months": rep["n_months"],
            },
        )

    persist.finish_run(run_id)
    return run_id
```

- [ ] **Step 4: Run integration tests**

Run: `source .venv/bin/activate && python -m pytest tests/lab/test_harness_signals.py -q -m integration`
Expected: PASS (2 passed)

- [ ] **Step 5: Run full lab suite (unit + integration)**

Run: `source .venv/bin/activate && python -m pytest tests/lab/ -m "not integration" -q`
Expected: all existing unit tests pass (the existing harness test for weight strategies must still pass).

Run: `source .venv/bin/activate && python -m pytest tests/lab/ -m integration -q`
Expected: all lab integration tests pass (8 from Plan 1 + 2 new).

- [ ] **Step 6: Commit**

```bash
git add src/ggTrader/lab/harness.py tests/lab/test_harness_signals.py
git commit -m "feat(lab): extend walkforward for signal strategies (two-call, merged results)

Detects target_kind per strategy, routes weight strategies to simulate_weights
and signal strategies to simulate_signals, then merges equity/returns dicts
before persisting. Mixed weight+signal runs now work end-to-end.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 4: `WfoTournamentSignal` strategy

A signal strategy that runs a mini EMA-combo IS tournament each rebalance. `select` evaluates 4 (ema_fast, ema_slow) combos on an IS window (70% of data up to `asof`) and picks the globally best combo by equal-weight portfolio IS Sharpe. `to_targets` stitches per-period signals using the winning params at each rebalance date.

**Files:**
- Modify: `src/ggTrader/lab/strategies/signals.py`
- Modify: `tests/lab/test_signals.py`

- [ ] **Step 1: Add failing tests for WfoTournamentSignal**

Add to `tests/lab/test_signals.py`:

```python
def test_wfo_tournament_select_returns_plan_with_params():
    ohlcv = _ohlcv(["A", "B", "C"])
    strat = WfoTournamentSignal(LabConfig(top_n=3, min_history_bars=100))
    asof = ohlcv.index[-1]
    sels = strat.select(asof, ohlcv, ["A", "B", "C"])
    assert len(sels) <= 3
    if sels:
        assert "ema_fast" in sels[0] and "ema_slow" in sels[0]
        assert "is_sharpe" in sels[0]
        # params must come from the known combo list
        assert sels[0]["ema_fast"] in (5, 10, 20, 50)


def test_wfo_tournament_select_no_lookahead():
    ohlcv = _ohlcv(["A"])
    strat = WfoTournamentSignal(LabConfig(min_history_bars=100))
    asof = ohlcv.index[-30]
    import json
    full = strat.select(asof, ohlcv.loc[:asof], ["A"])
    unmasked = strat.select(asof, ohlcv, ["A"])
    assert json.dumps(full, sort_keys=True) == json.dumps(unmasked, sort_keys=True)


def test_wfo_tournament_to_targets_returns_signal_targets():
    ohlcv = _ohlcv(["A", "B"])
    strat = WfoTournamentSignal(LabConfig(min_history_bars=100))
    asof1 = ohlcv.index[300]
    asof2 = ohlcv.index[450]
    plans = {
        asof1: [{"symbol": "A", "weight": 0.0, "ema_fast": 20, "ema_slow": 50,
                 "is_sharpe": 0.5}],
        asof2: [{"symbol": "A", "weight": 0.0, "ema_fast": 10, "ema_slow": 30,
                 "is_sharpe": 0.7},
                {"symbol": "B", "weight": 0.0, "ema_fast": 10, "ema_slow": 30,
                 "is_sharpe": 0.7}],
    }
    result = strat.to_targets(plans, ohlcv)
    assert isinstance(result, SignalTargets)
    assert "A" in result.entries.columns
    assert result.entries.dtype == bool
    # Only the forward period for asof1 should have signals for A (not before)
    forward1 = ohlcv.index[ohlcv.index > asof1]
    # At least some bars in the first period may have signals
    assert result.entries.shape[0] == len(ohlcv)


def test_build_signal_strategy_dispatch_wfo():
    from ggTrader.lab.strategies.signals import build_signal_strategy
    strat = build_signal_strategy("wfo_tournament", LabConfig())
    assert strat.name == "wfo_tournament"
```

Also add this import to `tests/lab/test_signals.py`:
```python
from ggTrader.lab.strategies.signals import EmaCrossSignal, WfoTournamentSignal, build_signal_strategy
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tests/lab/test_signals.py::test_wfo_tournament_select_returns_plan_with_params -v`
Expected: FAIL with `ImportError: cannot import name 'WfoTournamentSignal'`

- [ ] **Step 3: Implement `WfoTournamentSignal` in `strategies/signals.py`**

Add the following to `src/ggTrader/lab/strategies/signals.py` (after the `EmaCrossSignal` class, before `_SIGNAL_REGISTRY`):

```python
import numpy as np  # add to imports at top
import vectorbt as vbt  # add to imports at top

_EMA_COMBOS = [
    {"ema_fast": 5, "ema_slow": 20},
    {"ema_fast": 10, "ema_slow": 30},
    {"ema_fast": 20, "ema_slow": 50},
    {"ema_fast": 50, "ema_slow": 200},
]


def _ema_combo_is_sharpe(close_is: pd.DataFrame, ema_fast: int, ema_slow: int) -> float:
    """Equal-weight portfolio IS Sharpe for a single EMA combo."""
    ema_f = close_is.ewm(span=ema_fast, adjust=False).mean()
    ema_s = close_is.ewm(span=ema_slow, adjust=False).mean()
    entries = ((ema_f > ema_s) & (ema_f.shift(1) <= ema_s.shift(1))).fillna(False)
    exits = ((ema_f < ema_s) & (ema_f.shift(1) >= ema_s.shift(1))).fillna(False)
    n_syms = close_is.shape[1]
    if n_syms == 0:
        return float("-inf")
    try:
        pf = vbt.Portfolio.from_signals(
            close=close_is,
            entries=entries,
            exits=exits,
            size=1.0 / n_syms,
            size_type="percent",
            init_cash=10000.0,
            fees=0.0,
            freq="1d",
            group_by=np.zeros(n_syms, dtype=int),
            cash_sharing=True,
        ).copy()
        sharpe = pf.sharpe_ratio()
        val = float(sharpe.iloc[0] if hasattr(sharpe, "iloc") else sharpe)
        return val if np.isfinite(val) else float("-inf")
    except Exception:
        return float("-inf")


class WfoTournamentSignal:
    """EMA combo tournament: pick best (fast, slow) params on IS data each rebalance.

    select() evaluates 4 EMA combos on a 70% IS window and picks the best combo
    by equal-weight portfolio Sharpe. to_targets() generates piecewise signals
    using the per-period winning params.
    """

    name = "wfo_tournament"
    target_kind = "signals"

    def __init__(self, cfg: LabConfig, is_fraction: float = 0.7) -> None:
        self.cfg = cfg
        self.is_fraction = is_fraction

    def select(self, asof: pd.Timestamp, data: pd.DataFrame, eligible: List[str]) -> Plan:
        data = data.loc[:asof]
        have = set(data.columns.get_level_values(0).unique())
        syms = [
            s for s in eligible
            if s in have and len(data[s]["close"].dropna()) >= self.cfg.min_history_bars
        ]
        if not syms:
            return []

        close_all = pd.concat({s: data[s]["close"] for s in syms}, axis=1).ffill()
        is_end = max(1, int(len(close_all) * self.is_fraction))
        close_is = close_all.iloc[:is_end].dropna(axis=1, how="all")

        if len(close_is) < self.cfg.min_history_bars or close_is.shape[1] == 0:
            return []

        best_sharpe = float("-inf")
        best_combo = _EMA_COMBOS[2]  # default: 20/50
        for combo in _EMA_COMBOS:
            sharpe = _ema_combo_is_sharpe(close_is, combo["ema_fast"], combo["ema_slow"])
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_combo = combo

        return [
            {
                "symbol": s,
                "weight": 0.0,
                "ema_fast": best_combo["ema_fast"],
                "ema_slow": best_combo["ema_slow"],
                "is_sharpe": round(best_sharpe, 6),
            }
            for s in syms
        ][: self.cfg.top_n if self.cfg.top_n else None]

    def to_targets(
        self, plans: Dict[pd.Timestamp, Plan], data: pd.DataFrame
    ) -> SignalTargets:
        """Piecewise signals: each period uses the params selected at its rebalance date."""
        all_symbols = sorted({s["symbol"] for plan in plans.values() for s in plan})
        entries = pd.DataFrame(False, index=data.index, columns=all_symbols)
        exits = pd.DataFrame(False, index=data.index, columns=all_symbols)

        sorted_dates = sorted(plans.keys())
        have = set(data.columns.get_level_values(0).unique())

        for i, asof in enumerate(sorted_dates):
            next_asof = sorted_dates[i + 1] if i + 1 < len(sorted_dates) else data.index[-1]
            period_mask = (data.index > asof) & (data.index <= next_asof)
            period_index = data.index[period_mask]
            if len(period_index) == 0:
                continue

            active = {s["symbol"] for s in plans[asof]}

            # Exit dropped symbols at the start of this period
            if i > 0:
                prev_active = {s["symbol"] for s in plans[sorted_dates[i - 1]]}
                for sym in prev_active - active:
                    if sym in exits.columns and len(period_index) > 0:
                        exits.loc[period_index[0], sym] = True

            # Generate signals for active symbols using this period's params
            for sel in plans[asof]:
                sym = sel["symbol"]
                if sym not in have:
                    continue
                ema_fast = int(sel.get("ema_fast", 20))
                ema_slow = int(sel.get("ema_slow", 50))

                close = data[sym]["close"].dropna()
                ema_f = close.ewm(span=ema_fast, adjust=False).mean()
                ema_s = close.ewm(span=ema_slow, adjust=False).mean()
                sym_entries = (ema_f > ema_s) & (ema_f.shift(1) <= ema_s.shift(1))
                sym_exits = (ema_f < ema_s) & (ema_f.shift(1) >= ema_s.shift(1))

                entries.loc[period_index, sym] = (
                    sym_entries.reindex(period_index).fillna(False).values
                )
                exits.loc[period_index, sym] = (
                    sym_exits.reindex(period_index).fillna(False).values
                )

        return SignalTargets(entries=entries.astype(bool), exits=exits.astype(bool))
```

Also update `_SIGNAL_REGISTRY` and `SIGNAL_STRATEGY_NAMES`:

```python
_SIGNAL_REGISTRY = {
    "ema_cross": EmaCrossSignal,
    "wfo_tournament": WfoTournamentSignal,
}

SIGNAL_STRATEGY_NAMES = tuple(_SIGNAL_REGISTRY)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && python -m pytest tests/lab/test_signals.py -v`
Expected: PASS (all 9 tests pass)

- [ ] **Step 5: Run full lab unit suite**

Run: `source .venv/bin/activate && python -m pytest tests/lab/ -m "not integration" -q`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/ggTrader/lab/strategies/signals.py tests/lab/test_signals.py
git commit -m "feat(lab): WfoTournamentSignal — EMA combo IS tournament per rebalance

select() evaluates 4 EMA fast/slow combos on 70% IS window, picks best by
equal-weight portfolio Sharpe. to_targets() generates piecewise per-period
signals using the per-rebalance winning params.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 5: CLI extension + docs

Add signal strategies to the `--strategy` CLI choices. Update spec and changelog.

**Files:**
- Modify: `src/ggTrader/lab/cli.py`
- Modify: `docs/superpowers/specs/2026-06-15-vectorbt-lab-core-design.md`
- Modify: `docs/changelog.md`

- [ ] **Step 1: Update `cli.py` to include signal strategies**

Read `src/ggTrader/lab/cli.py`. The `build_arg_parser` function currently uses `choices=STRATEGY_NAMES` (weight-only registry). Change it to include signal strategy names as additional choices.

In `src/ggTrader/lab/cli.py`:

Add to imports at the top:
```python
from ggTrader.lab.strategies.signals import SIGNAL_STRATEGY_NAMES, build_signal_strategy
```

Update `build_arg_parser` — change the `--strategy` argument:
```python
    p.add_argument(
        "--strategy",
        choices=tuple(STRATEGY_NAMES) + tuple(SIGNAL_STRATEGY_NAMES),
        required=True,
    )
```

Update `run_lab` to route signal strategies to `build_signal_strategy`:
```python
    if args.strategy in SIGNAL_STRATEGY_NAMES:
        strat = build_signal_strategy(args.strategy, cfg)
    else:
        strat = build_strategy(args.strategy, cfg)
```

- [ ] **Step 2: Verify CLI shows all strategies**

Run: `source .venv/bin/activate && cd /home/flynn/ggTrader && python -m ggTrader.lab.cli --help`
Expected: `--strategy` choices include `xs_momentum`, `dual_momentum`, `ema_cross`, `wfo_tournament`.

- [ ] **Step 3: Run CLI unit test to verify parser still works**

Run: `source .venv/bin/activate && python -m pytest tests/lab/test_cli.py -v`
Expected: PASS (note: existing test passes `xs_momentum`; it should still pass).

- [ ] **Step 4: Run full lab unit suite**

Run: `source .venv/bin/activate && python -m pytest tests/lab/ -m "not integration" -q`
Expected: all tests pass.

- [ ] **Step 5: Update spec status**

In `docs/superpowers/specs/2026-06-15-vectorbt-lab-core-design.md`, update the Status line to:

```
**Status:** Plan 1 executed 2026-06-15 (momentum bench + validation gate). Plan 3 executed 2026-06-16 (CachedYFinanceLoader bug fixed, S&P 500 backfilled into DB, research/WFO code cluster deleted). Plan 2 executed 2026-06-16 (signal-based strategies: EmaCrossSignal + WfoTournamentSignal; from_signals harness; open §10 question resolved — two grouped calls + merged metrics).
```

Also update §10 in the spec to mark the open question resolved:
```
**Resolved (Plan 2):** Weight and signal strategies use two separate grouped vbt calls
(`from_orders` for weights, `from_signals` for signals) with metrics merged at the
harness level before persisting. This cleanly separates the two simulation APIs.
```

- [ ] **Step 6: Add changelog entry for Plan 2**

In `docs/changelog.md`, add under the existing `## 2026-06-16` section:

```markdown
### Research: Lab Plan 2 — signal-based strategy family

- **Added** `SignalTargets(entries, exits)` NamedTuple to the Strategy protocol.
- **Added** `simulate_signals()` — one grouped `vbt.Portfolio.from_signals` call for all
  signal strategies simultaneously (parallel to `simulate_weights`).
- **Extended** `walkforward()` to detect `target_kind`, route strategies to the appropriate
  sim call, and merge results. Mixed weight + signal runs now work end-to-end.
- **Added** `EmaCrossSignal` — simple whole-window EMA crossover signal strategy.
- **Added** `WfoTournamentSignal` — evaluates 4 EMA combos on a 70% IS window each
  rebalance, picks the best by equal-weight portfolio Sharpe, generates piecewise
  signals for each forward period. Resolves spec §10 open question.
- **Extended** `ggt lab` CLI: `--strategy ema_cross` and `--strategy wfo_tournament` now work.
```

- [ ] **Step 7: Commit**

```bash
git add src/ggTrader/lab/cli.py \
        docs/changelog.md \
        docs/superpowers/specs/2026-06-15-vectorbt-lab-core-design.md
git commit -m "feat(lab): CLI signal strategy choices + Plan 2 docs

Adds ema_cross and wfo_tournament to --strategy choices. Updates spec
status (§10 open question resolved) and changelog.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**

- §3.2 Phase 2 (signal strategies via `from_signals`) → Task 1 (`simulate_signals`) + Task 3 (harness extension) ✅
- §3.1 `Strategy.to_targets` return type → Task 1 (`SignalTargets` NamedTuple) ✅
- §10 open question (weight + signal grouped calls) → resolved in Task 3 (two calls, merged) ✅
- `wfo_tournament` strategy → Task 4 (`WfoTournamentSignal`) ✅
- CLI `--strategy` extension → Task 5 ✅
- Leak safety for signal strategies → Task 2 (`test_ema_cross_no_lookahead`) + self-truncation in both strategies' `select` ✅
- Integration test for mixed strategies → Task 3 (`test_walkforward_mixed_weight_and_signal_strategies`) ✅

**Placeholder scan:** No TBDs. All code is complete. The note about `np` and `vbt` imports in Task 4 is an action item, not a placeholder — the implementer must add those imports to the top of `signals.py`.

**Type consistency:**
- `SignalTargets(entries, exits)` defined in Task 1 strategy.py, consumed in Task 2 (`EmaCrossSignal.to_targets`), Task 4 (`WfoTournamentSignal.to_targets`), and Task 3 harness
- `simulate_signals(targets_by_strategy, prices, base_config)` signature matches its callers in harness.py
- `SIGNAL_STRATEGY_NAMES` defined in Task 2 strategies/signals.py, imported in Task 5 cli.py
- `build_signal_strategy(name, cfg)` defined in Task 2, imported and called in Task 5 cli.py
- `_ema_combo_is_sharpe(close_is, ema_fast, ema_slow)` defined and used only within Task 4

**Risk notes:**
- `_ema_combo_is_sharpe` wraps the vbt call in `try/except Exception` to handle degenerate data (all NaN, zero variance). This is correct — `select` should not crash on a single bad symbol.
- `WfoTournamentSignal.to_targets` loops over symbols per period — O(n_periods × n_symbols) iterations with pandas EMA each time. For 64 months × 50 stocks, this is ~3200 iterations, each doing 2 ewm calls on a ~600-bar series. Acceptable for research (a few seconds), not production.
- The `is_fraction=0.7` split means the IS window uses 70% of data up to `asof`. For early rebalances (where `asof` is near `data_start`), the IS window may be short. The `min_history_bars` guard in `select` prevents this from causing problems.
