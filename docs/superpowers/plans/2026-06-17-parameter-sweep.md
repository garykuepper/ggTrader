# Parameter Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add vectorbt-native parameter sweep tooling — generate signals for all param combos in one vectorized pass and simulate them in a single `vbt.Portfolio` call with `group_by`.

**Architecture:** Each strategy declares sweepable params via `sweep_params()`. Signal strategies implement `sweep_signals()` to compute all combos' entry/exit signals vectorized. A new `sweep.py` module orchestrates grid generation, the batched vbt call, metrics extraction, DB persistence (`lab_sweeps` + `lab_sweep_combos`), and a ranked CLI table. The CLI adds `--sweep` and `--sweep-param` flags to the existing `ggt lab` command.

**Tech Stack:** Python 3.11+, vectorbt 1.0, pandas, numpy, SQLAlchemy (TimescaleDB), argparse, pytest.

## Global Constraints

- All code under `src/ggTrader/`, absolute imports from `ggTrader.*`.
- Ruff linting (88-100 char lines, PEP 8, sorted imports).
- Type hints on all function signatures.
- Vectorization first — no row iteration for signal computation.
- Tests run via `pytest tests/` from repo root.
- Docker research: `docker compose run --rm ggtrader_live python ggt.py lab ...`

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/ggTrader/lab/strategy.py` | Modify | Add `sweep_params` to Strategy protocol |
| `src/ggTrader/lab/strategies/signals.py` | Modify | Add `sweep_params()` + `sweep_signals()` to EmaCrossSignal and WfoTournamentSignal |
| `src/ggTrader/lab/strategies/momentum.py` | Modify | Add `sweep_params()` to CrossSectionalMomentum and DualMomentum |
| `src/ggTrader/lab/persist.py` | Modify | Add `lab_sweeps` + `lab_sweep_combos` schema and CRUD helpers |
| `src/ggTrader/lab/sweep.py` | Create | Grid generation, vectorized sweep orchestration, results table |
| `src/ggTrader/lab/cli.py` | Modify | Add `--sweep` and `--sweep-param` flags, wire to sweep module |
| `tests/lab/test_sweep.py` | Create | Tests for grid, combo naming, signal stacking, metrics, CLI args |

---

### Task 1: Strategy Protocol — Add `sweep_params`

**Files:**
- Modify: `src/ggTrader/lab/strategy.py`
- Modify: `src/ggTrader/lab/strategies/signals.py`
- Modify: `src/ggTrader/lab/strategies/momentum.py`
- Test: `tests/lab/test_sweep.py` (new file, first tests)

**Interfaces:**
- Consumes: existing `LabConfig`, `Strategy` protocol, `EmaCrossSignal`, `WfoTournamentSignal`, `CrossSectionalMomentum`, `DualMomentum`
- Produces:
  - `Strategy.sweep_params(cls) -> dict[str, list]` (protocol method)
  - `EmaCrossSignal.sweep_params()` returns `{"ema_fast": [5, 10, 20, 50], "ema_slow": [20, 30, 50, 100, 200]}`
  - `WfoTournamentSignal.sweep_params()` returns `{"is_fraction": [0.5, 0.6, 0.7, 0.8]}`
  - `CrossSectionalMomentum.sweep_params()` returns `{"top_n": [10, 20, 50], "lookback": [126, 252], "skip": [0, 21]}`
  - `DualMomentum.sweep_params()` inherits from `CrossSectionalMomentum`

- [ ] **Step 1: Write failing tests for `sweep_params`**

Create `tests/lab/test_sweep.py`:

```python
"""Tests for parameter sweep tooling."""

from ggTrader.lab.strategies.momentum import CrossSectionalMomentum, DualMomentum
from ggTrader.lab.strategies.signals import EmaCrossSignal, WfoTournamentSignal
from ggTrader.lab.strategy import LabConfig


def test_ema_cross_sweep_params_returns_fast_and_slow():
    params = EmaCrossSignal.sweep_params()
    assert "ema_fast" in params
    assert "ema_slow" in params
    assert all(isinstance(v, list) and len(v) > 1 for v in params.values())


def test_wfo_tournament_sweep_params_returns_is_fraction():
    params = WfoTournamentSignal.sweep_params()
    assert "is_fraction" in params
    assert all(0.0 < f < 1.0 for f in params["is_fraction"])


def test_xs_momentum_sweep_params_returns_labconfig_params():
    params = CrossSectionalMomentum.sweep_params()
    assert "top_n" in params
    assert "lookback" in params
    assert "skip" in params


def test_dual_momentum_inherits_sweep_params():
    params = DualMomentum.sweep_params()
    assert "top_n" in params
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/lab/test_sweep.py -v`
Expected: FAIL — `sweep_params` not defined on any strategy class.

- [ ] **Step 3: Add `sweep_params` to the Strategy protocol**

In `src/ggTrader/lab/strategy.py`, add to the `Strategy` protocol class (after the existing methods, before the file ends at line 54):

```python
    @classmethod
    def sweep_params(cls) -> dict[str, list]:
        """Parameter names → candidate values for grid sweep."""
        ...
```

Also add `classmethod` to the typing imports — change line 6 from:

```python
from typing import Any, Dict, List, NamedTuple, Protocol, Union
```

to:

```python
from typing import Any, ClassVar, Dict, List, NamedTuple, Protocol, Union
```

(Note: `ClassVar` import is for future use if needed; `classmethod` is a builtin decorator.)

- [ ] **Step 4: Add `sweep_params()` to EmaCrossSignal and WfoTournamentSignal**

In `src/ggTrader/lab/strategies/signals.py`, add to `EmaCrossSignal` class (after `__init__`, before `select`):

```python
    @classmethod
    def sweep_params(cls) -> dict[str, list]:
        return {
            "ema_fast": [5, 10, 20, 50],
            "ema_slow": [20, 30, 50, 100, 200],
        }
```

Add to `WfoTournamentSignal` class (after `__init__`, before `select`):

```python
    @classmethod
    def sweep_params(cls) -> dict[str, list]:
        return {
            "is_fraction": [0.5, 0.6, 0.7, 0.8],
        }
```

- [ ] **Step 5: Add `sweep_params()` to momentum strategies**

In `src/ggTrader/lab/strategies/momentum.py`, add to `CrossSectionalMomentum` class (after `__init__`, before `select`):

```python
    @classmethod
    def sweep_params(cls) -> dict[str, list]:
        return {
            "top_n": [10, 20, 50],
            "lookback": [126, 252],
            "skip": [0, 21],
        }
```

`DualMomentum` inherits it — no change needed there.

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/lab/test_sweep.py -v`
Expected: 4 PASS

- [ ] **Step 7: Commit**

```bash
git add src/ggTrader/lab/strategy.py src/ggTrader/lab/strategies/signals.py \
       src/ggTrader/lab/strategies/momentum.py tests/lab/test_sweep.py
git commit -m "feat(lab): add sweep_params() to strategy protocol and all strategies"
```

---

### Task 2: Grid Generation & Combo Naming

**Files:**
- Create: `src/ggTrader/lab/sweep.py`
- Test: `tests/lab/test_sweep.py` (append tests)

**Interfaces:**
- Consumes: `Strategy.sweep_params()` from Task 1
- Produces:
  - `build_grid(strategy_cls, overrides: dict[str, list] | None) -> list[dict[str, Any]]` — Cartesian product of params, with invalid combos filtered
  - `combo_name(strategy_name: str, params: dict) -> str` — deterministic label like `ema_cross__f5_s20`
  - `validate_combo(params: dict) -> bool` — filters e.g. `ema_fast >= ema_slow`

- [ ] **Step 1: Write failing tests for grid generation and combo naming**

Append to `tests/lab/test_sweep.py`:

```python
from ggTrader.lab.sweep import build_grid, combo_name


def test_build_grid_cartesian_product():
    from ggTrader.lab.strategies.signals import EmaCrossSignal

    grid = build_grid(EmaCrossSignal)
    # 4 fast x 5 slow = 20, minus invalid (fast >= slow)
    # Invalid: (50, 20), (50, 30), (50, 50), (20, 20), (10, 10 — not in slow), ...
    # fast=5: all 5 slow valid → 5
    # fast=10: slow 20,30,50,100,200 → 5
    # fast=20: slow 30,50,100,200 → 4 (skip 20)
    # fast=50: slow 100,200 → 2 (skip 20,30,50)
    # Total: 5+5+4+2 = 16
    assert len(grid) == 16
    assert all("ema_fast" in c and "ema_slow" in c for c in grid)
    # No combo has fast >= slow
    assert all(c["ema_fast"] < c["ema_slow"] for c in grid)


def test_build_grid_with_overrides():
    from ggTrader.lab.strategies.signals import EmaCrossSignal

    grid = build_grid(EmaCrossSignal, overrides={"ema_fast": [5, 10], "ema_slow": [50, 100]})
    assert len(grid) == 4  # 2 x 2, all valid
    assert all(c["ema_fast"] in (5, 10) for c in grid)


def test_build_grid_no_constraint_strategies():
    from ggTrader.lab.strategies.momentum import CrossSectionalMomentum

    grid = build_grid(CrossSectionalMomentum)
    # 3 top_n x 2 lookback x 2 skip = 12, no constraint filtering
    assert len(grid) == 12


def test_combo_name_deterministic():
    assert combo_name("ema_cross", {"ema_fast": 5, "ema_slow": 20}) == "ema_cross__ema_fast5_ema_slow20"
    assert combo_name("ema_cross", {"ema_slow": 20, "ema_fast": 5}) == "ema_cross__ema_fast5_ema_slow20"


def test_combo_name_single_param():
    assert combo_name("wfo_tournament", {"is_fraction": 0.7}) == "wfo_tournament__is_fraction0.7"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/lab/test_sweep.py::test_build_grid_cartesian_product -v`
Expected: FAIL — `ImportError: cannot import name 'build_grid' from 'ggTrader.lab.sweep'`

- [ ] **Step 3: Implement grid generation and combo naming**

Create `src/ggTrader/lab/sweep.py`:

```python
"""Parameter sweep: grid generation, vectorized orchestration, results display."""

from __future__ import annotations

from itertools import product
from typing import Any, Dict, List, Optional, Type


def _is_valid_combo(params: Dict[str, Any]) -> bool:
    """Filter combos where a 'fast' param is >= a corresponding 'slow' param."""
    fast_keys = sorted(k for k in params if "fast" in k)
    slow_keys = sorted(k for k in params if "slow" in k)
    for fk, sk in zip(fast_keys, slow_keys):
        if params[fk] >= params[sk]:
            return False
    return True


def build_grid(
    strategy_cls: Type,
    overrides: Optional[Dict[str, list]] = None,
) -> List[Dict[str, Any]]:
    """Cartesian product of sweep params, filtering invalid combos."""
    raw = strategy_cls.sweep_params()
    if overrides:
        raw = {**raw, **overrides}
    keys = sorted(raw.keys())
    combos = [dict(zip(keys, vals)) for vals in product(*(raw[k] for k in keys))]
    return [c for c in combos if _is_valid_combo(c)]


def combo_name(strategy_name: str, params: Dict[str, Any]) -> str:
    """Deterministic label from strategy name + sorted param key-value pairs."""
    parts = [f"{k}{v}" for k, v in sorted(params.items())]
    return strategy_name + "__" + "_".join(parts)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/lab/test_sweep.py -v`
Expected: 9 PASS (4 from Task 1 + 5 new)

- [ ] **Step 5: Commit**

```bash
git add src/ggTrader/lab/sweep.py tests/lab/test_sweep.py
git commit -m "feat(lab): grid generation and combo naming for parameter sweeps"
```

---

### Task 3: Vectorized `sweep_signals()` on Signal Strategies

**Files:**
- Modify: `src/ggTrader/lab/strategies/signals.py`
- Test: `tests/lab/test_sweep.py` (append tests)

**Interfaces:**
- Consumes: `build_grid()`, `combo_name()` from Task 2; existing `EmaCrossSignal`, `WfoTournamentSignal`
- Produces:
  - `EmaCrossSignal.sweep_signals(combos: list[dict], symbols: list[str], data: DataFrame) -> dict[str, SignalTargets]` — keyed by combo name
  - `WfoTournamentSignal.sweep_signals(combos: list[dict], symbols: list[str], data: DataFrame) -> dict[str, SignalTargets]` — keyed by combo name

- [ ] **Step 1: Write failing tests for `sweep_signals`**

Append to `tests/lab/test_sweep.py`:

```python
import numpy as np
import pandas as pd

from ggTrader.lab.strategy import LabConfig, SignalTargets


def _ohlcv(symbols, n=600):
    idx = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
    frames = {}
    for i, s in enumerate(symbols):
        close = pd.Series(100.0 * (1 + 0.0003 * (i + 1)) ** np.arange(n), index=idx)
        frames[s] = pd.DataFrame(
            {"open": close, "high": close * 1.01, "low": close * 0.99,
             "close": close, "volume": np.full(n, 1e6)},
            index=idx,
        )
    out = pd.concat(frames, axis=1)
    out.columns = out.columns.set_names(["symbol", "field"])
    return out


def test_ema_cross_sweep_signals_returns_all_combos():
    from ggTrader.lab.strategies.signals import EmaCrossSignal

    ohlcv = _ohlcv(["A", "B"])
    combos = [
        {"ema_fast": 5, "ema_slow": 20},
        {"ema_fast": 10, "ema_slow": 50},
    ]
    strat = EmaCrossSignal(LabConfig(min_history_bars=100))
    result = strat.sweep_signals(combos, ["A", "B"], ohlcv)
    assert len(result) == 2
    for key, st in result.items():
        assert isinstance(st, SignalTargets)
        assert set(st.entries.columns) == {"A", "B"}
        assert st.entries.dtype == bool
        assert st.exits.dtype == bool


def test_ema_cross_sweep_signals_matches_single_run():
    """Vectorized sweep must produce identical signals to single-combo to_targets."""
    from ggTrader.lab.strategies.signals import EmaCrossSignal

    ohlcv = _ohlcv(["A", "B"])
    cfg = LabConfig(min_history_bars=100)
    fast, slow = 10, 50
    strat = EmaCrossSignal(cfg, ema_fast=fast, ema_slow=slow)
    plans = {ohlcv.index[200]: [
        {"symbol": "A", "weight": 0.0, "ema_fast": fast, "ema_slow": slow},
        {"symbol": "B", "weight": 0.0, "ema_fast": fast, "ema_slow": slow},
    ]}
    single = strat.to_targets(plans, ohlcv)

    combos = [{"ema_fast": fast, "ema_slow": slow}]
    sweep_result = strat.sweep_signals(combos, ["A", "B"], ohlcv)
    sweep_st = list(sweep_result.values())[0]

    pd.testing.assert_frame_equal(single.entries, sweep_st.entries)
    pd.testing.assert_frame_equal(single.exits, sweep_st.exits)


def test_ema_cross_sweep_signals_different_combos_differ():
    from ggTrader.lab.strategies.signals import EmaCrossSignal

    ohlcv = _ohlcv(["A"])
    combos = [
        {"ema_fast": 5, "ema_slow": 20},
        {"ema_fast": 50, "ema_slow": 200},
    ]
    strat = EmaCrossSignal(LabConfig(min_history_bars=100))
    result = strat.sweep_signals(combos, ["A"], ohlcv)
    keys = list(result.keys())
    assert not result[keys[0]].entries.equals(result[keys[1]].entries)


def test_wfo_tournament_sweep_signals_returns_all_combos():
    from ggTrader.lab.strategies.signals import WfoTournamentSignal

    ohlcv = _ohlcv(["A", "B"])
    combos = [{"is_fraction": 0.5}, {"is_fraction": 0.8}]
    strat = WfoTournamentSignal(LabConfig(min_history_bars=100))
    result = strat.sweep_signals(combos, ["A", "B"], ohlcv)
    assert len(result) == 2
    for st in result.values():
        assert isinstance(st, SignalTargets)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/lab/test_sweep.py::test_ema_cross_sweep_signals_returns_all_combos -v`
Expected: FAIL — `AttributeError: 'EmaCrossSignal' object has no attribute 'sweep_signals'`

- [ ] **Step 3: Implement `EmaCrossSignal.sweep_signals()`**

In `src/ggTrader/lab/strategies/signals.py`, add this import near the top (after the existing imports):

```python
from ggTrader.lab.sweep import combo_name
```

Add to `EmaCrossSignal` class (after `to_targets`, before the class ends):

```python
    def sweep_signals(
        self,
        combos: list[dict],
        symbols: list[str],
        data: pd.DataFrame,
    ) -> dict[str, "SignalTargets"]:
        """Vectorized signal generation for all (ema_fast, ema_slow) combos at once."""
        close = pd.concat(
            {s: data[s]["close"] for s in symbols if s in data.columns.get_level_values(0)},
            axis=1,
        )
        unique_spans = sorted({v for c in combos for v in c.values()})
        emas: dict[int, pd.DataFrame] = {
            span: close.ewm(span=span, adjust=False).mean() for span in unique_spans
        }
        result: dict[str, SignalTargets] = {}
        for combo in combos:
            fast, slow = int(combo["ema_fast"]), int(combo["ema_slow"])
            ema_f, ema_s = emas[fast], emas[slow]
            entries = ((ema_f > ema_s) & (ema_f.shift(1) <= ema_s.shift(1))).fillna(False)
            exits = ((ema_f < ema_s) & (ema_f.shift(1) >= ema_s.shift(1))).fillna(False)
            key = combo_name(self.name, combo)
            result[key] = SignalTargets(entries=entries.astype(bool), exits=exits.astype(bool))
        return result
```

- [ ] **Step 4: Implement `WfoTournamentSignal.sweep_signals()`**

Add to `WfoTournamentSignal` class (after `to_targets`, before the class ends):

```python
    def sweep_signals(
        self,
        combos: list[dict],
        symbols: list[str],
        data: pd.DataFrame,
    ) -> dict[str, "SignalTargets"]:
        """Sweep over is_fraction values — each gets its own IS/OOS split and tournament."""
        from ggTrader.lab.sweep import combo_name

        close = pd.concat(
            {s: data[s]["close"] for s in symbols if s in data.columns.get_level_values(0)},
            axis=1,
        ).ffill()
        unique_spans = sorted(
            {v for combo_list in _EMA_COMBOS for v in combo_list.values()}
        )
        emas: dict[int, pd.DataFrame] = {
            span: close.ewm(span=span, adjust=False).mean() for span in unique_spans
        }

        result: dict[str, SignalTargets] = {}
        for combo in combos:
            is_frac = float(combo["is_fraction"])
            is_end = max(1, int(len(close) * is_frac))
            close_is = close.iloc[:is_end].dropna(axis=1, how="all")
            best_combo = _EMA_COMBOS[2]  # default 20/50
            best_sharpe = float("-inf")
            for ec in _EMA_COMBOS:
                sharpe = _ema_combo_is_sharpe(close_is, ec["ema_fast"], ec["ema_slow"])
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_combo = ec
            fast, slow = best_combo["ema_fast"], best_combo["ema_slow"]
            ema_f, ema_s = emas[fast], emas[slow]
            entries = ((ema_f > ema_s) & (ema_f.shift(1) <= ema_s.shift(1))).fillna(False)
            exits = ((ema_f < ema_s) & (ema_f.shift(1) >= ema_s.shift(1))).fillna(False)
            key = combo_name(self.name, combo)
            result[key] = SignalTargets(entries=entries.astype(bool), exits=exits.astype(bool))
        return result
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/lab/test_sweep.py -v`
Expected: 13 PASS (9 from Tasks 1-2 + 4 new)

- [ ] **Step 6: Commit**

```bash
git add src/ggTrader/lab/strategies/signals.py tests/lab/test_sweep.py
git commit -m "feat(lab): vectorized sweep_signals() for EmaCross and WfoTournament"
```

---

### Task 4: Sweep Persistence — DB Schema & Helpers

**Files:**
- Modify: `src/ggTrader/lab/persist.py`
- Test: `tests/lab/test_sweep.py` (append tests)

**Interfaces:**
- Consumes: existing `get_engine()`, `init_schema()`, `_sanitize()` from persist.py
- Produces:
  - `start_sweep(strategy: str, market: str, param_grid: dict, n_combos: int) -> str` — returns sweep_id
  - `write_sweep_combo(sweep_id: str, combo_name: str, params: dict, metrics: dict, benchmark_metrics: dict, diagnostics: dict) -> None`
  - `finish_sweep(sweep_id: str) -> None`

- [ ] **Step 1: Write failing tests for sweep persistence**

Append to `tests/lab/test_sweep.py`:

```python
import pytest


@pytest.mark.integration
def test_sweep_persistence_roundtrip():
    from ggTrader.lab.persist import (
        finish_sweep,
        get_engine,
        init_schema,
        start_sweep,
        write_sweep_combo,
    )
    from sqlalchemy import text

    init_schema()
    sweep_id = start_sweep("ema_cross", "equity", {"ema_fast": [5, 10], "ema_slow": [20, 50]}, 4)
    assert sweep_id.startswith("sweep_ema_cross_")

    write_sweep_combo(
        sweep_id, "ema_cross__ema_fast5_ema_slow20",
        {"ema_fast": 5, "ema_slow": 20},
        {"sharpe": 0.42, "cagr_pct": 3.1},
        {"sharpe": 0.85},
        {"n_symbols": 50},
    )
    finish_sweep(sweep_id)

    with get_engine().connect() as conn:
        row = conn.execute(
            text("SELECT status FROM lab_sweeps WHERE sweep_id = :s"),
            {"s": sweep_id},
        ).first()
        assert row[0] == "done"
        combo_row = conn.execute(
            text("SELECT params, metrics FROM lab_sweep_combos WHERE sweep_id = :s"),
            {"s": sweep_id},
        ).first()
        assert combo_row[0]["ema_fast"] == 5
        assert combo_row[1]["sharpe"] == 0.42
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/lab/test_sweep.py::test_sweep_persistence_roundtrip -v -m integration`
Expected: FAIL — `ImportError: cannot import name 'start_sweep'`

- [ ] **Step 3: Add sweep schema and helpers to persist.py**

In `src/ggTrader/lab/persist.py`, append to the `_SCHEMA` string (before the closing `"""`), add:

```sql
CREATE TABLE IF NOT EXISTS lab_sweeps (
    sweep_id TEXT PRIMARY KEY,
    strategy TEXT NOT NULL,
    market TEXT NOT NULL,
    param_grid JSONB NOT NULL,
    n_combos INT NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS lab_sweep_combos (
    sweep_id TEXT NOT NULL,
    combo_name TEXT NOT NULL,
    params JSONB NOT NULL,
    metrics JSONB,
    benchmark_metrics JSONB,
    diagnostics JSONB,
    PRIMARY KEY (sweep_id, combo_name)
);
```

Add the following functions at the end of `persist.py`:

```python
def start_sweep(strategy: str, market: str, param_grid: Dict[str, Any], n_combos: int) -> str:
    sweep_id = f"sweep_{strategy}_{uuid.uuid4().hex[:8]}"
    with get_engine().begin() as conn:
        conn.execute(
            text(
                "INSERT INTO lab_sweeps"
                " (sweep_id, strategy, market, param_grid, n_combos)"
                " VALUES (:s, :st, :m, :pg, :n)"
            ),
            {
                "s": sweep_id,
                "st": strategy,
                "m": market,
                "pg": json.dumps(param_grid),
                "n": n_combos,
            },
        )
    return sweep_id


def write_sweep_combo(
    sweep_id: str,
    combo_name: str,
    params: Dict[str, Any],
    metrics: Dict[str, Any],
    benchmark_metrics: Dict[str, Any],
    diagnostics: Dict[str, Any],
) -> None:
    with get_engine().begin() as conn:
        conn.execute(
            text(
                "INSERT INTO lab_sweep_combos"
                " (sweep_id, combo_name, params, metrics, benchmark_metrics, diagnostics)"
                " VALUES (:s, :cn, :p, :m, :bm, :d)"
                " ON CONFLICT (sweep_id, combo_name) DO UPDATE"
                " SET metrics=EXCLUDED.metrics, benchmark_metrics=EXCLUDED.benchmark_metrics,"
                " diagnostics=EXCLUDED.diagnostics"
            ),
            {
                "s": sweep_id,
                "cn": combo_name,
                "p": json.dumps(_sanitize(params)),
                "m": json.dumps(_sanitize(metrics)),
                "bm": json.dumps(_sanitize(benchmark_metrics)),
                "d": json.dumps(_sanitize(diagnostics)),
            },
        )


def finish_sweep(sweep_id: str) -> None:
    with get_engine().begin() as conn:
        conn.execute(
            text("UPDATE lab_sweeps SET status='done' WHERE sweep_id=:s"),
            {"s": sweep_id},
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/lab/test_sweep.py::test_sweep_persistence_roundtrip -v -m integration`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ggTrader/lab/persist.py tests/lab/test_sweep.py
git commit -m "feat(lab): lab_sweeps + lab_sweep_combos DB schema and persistence helpers"
```

---

### Task 5: Sweep Orchestrator — Batched vbt Simulation & Results Table

**Files:**
- Modify: `src/ggTrader/lab/sweep.py`
- Test: `tests/lab/test_sweep.py` (append tests)

**Interfaces:**
- Consumes:
  - `build_grid()`, `combo_name()` from Task 2
  - `EmaCrossSignal.sweep_signals()` from Task 3
  - `simulate_signals(targets_by_strategy, prices, base_config)` from `simulate.py`
  - `curve_stats(equity_series)`, `benchmark(equity, spy_close, start_cash)` from `metrics.py`
  - `start_sweep()`, `write_sweep_combo()`, `finish_sweep()` from Task 4
- Produces:
  - `run_sweep(strategy_name: str, strategy_cls: Type, cfg: LabConfig, ohlcv: DataFrame, spy_close: Series, eval_start: str, eval_end: str, market: str, base_config: dict, grid: list[dict]) -> str` — returns sweep_id
  - `format_results_table(rows: list[dict], strategy_name: str, n_combos: int, eval_start: str, eval_end: str, sweep_id: str, spy_metrics: dict) -> str` — formatted CLI output

- [ ] **Step 1: Write failing tests for the orchestrator**

Append to `tests/lab/test_sweep.py`:

```python
from ggTrader.lab.sweep import build_grid, combo_name, format_results_table


def test_format_results_table_renders_header_and_rows():
    rows = [
        {"combo": "ema_cross__ema_fast5_ema_slow20", "sharpe": 0.42, "cagr_pct": 3.1,
         "max_drawdown_pct": -18.2, "sortino": 0.61, "total_return_pct": 12.4},
        {"combo": "ema_cross__ema_fast10_ema_slow30", "sharpe": 0.38, "cagr_pct": 2.7,
         "max_drawdown_pct": -21.0, "sortino": 0.54, "total_return_pct": 10.8},
    ]
    spy = {"cagr_pct": 18.2, "sharpe": 0.85, "max_drawdown_pct": -24.5}
    table = format_results_table(
        rows, "ema_cross", 2, "2021-01-31", "2026-06-17", "sweep_ema_cross_abc123", spy
    )
    assert "ema_cross" in table
    assert "2 combos" in table
    assert "Sharpe" in table
    assert "SPY" in table
    assert "0.42" in table


def test_format_results_table_sorted_by_sharpe():
    rows = [
        {"combo": "a", "sharpe": -0.1, "cagr_pct": 0, "max_drawdown_pct": 0,
         "sortino": 0, "total_return_pct": 0},
        {"combo": "b", "sharpe": 0.5, "cagr_pct": 0, "max_drawdown_pct": 0,
         "sortino": 0, "total_return_pct": 0},
    ]
    table = format_results_table(rows, "x", 2, "2021", "2026", "id", {"cagr_pct": 0, "sharpe": 0, "max_drawdown_pct": 0})
    lines = table.strip().split("\n")
    # First data row (after header lines) should be the higher-sharpe combo
    data_lines = [l for l in lines if l.strip().startswith(("1", "2"))]
    assert "b" in data_lines[0]
    assert "a" in data_lines[1]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/lab/test_sweep.py::test_format_results_table_renders_header_and_rows -v`
Expected: FAIL — `ImportError: cannot import name 'format_results_table'`

- [ ] **Step 3: Implement `format_results_table` and `run_sweep`**

Add to `src/ggTrader/lab/sweep.py`:

```python
import pandas as pd

from ggTrader.lab import persist
from ggTrader.lab.data import eligible_at, rebalance_dates
from ggTrader.lab.metrics import curve_stats
from ggTrader.lab.simulate import simulate_signals, simulate_weights
from ggTrader.lab.strategy import LabConfig, SignalTargets


def format_results_table(
    rows: List[Dict[str, Any]],
    strategy_name: str,
    n_combos: int,
    eval_start: str,
    eval_end: str,
    sweep_id: str,
    spy_metrics: Dict[str, Any],
) -> str:
    """Ranked results table sorted by Sharpe descending."""
    sorted_rows = sorted(rows, key=lambda r: r.get("sharpe", float("-inf")), reverse=True)
    lines = [
        f"Sweep complete: {strategy_name} | {n_combos} combos | {eval_start} → {eval_end}",
        f"sweep_id: {sweep_id}",
        "",
        f"{'Rank':<6}{'Combo':<40}{'Sharpe':>8}{'CAGR%':>8}{'MaxDD%':>8}{'Sortino':>9}{'TotRet%':>9}",
        "─" * 88,
    ]
    for i, r in enumerate(sorted_rows, 1):
        lines.append(
            f"{i:<6}{r['combo']:<40}{r['sharpe']:>8.2f}{r['cagr_pct']:>7.1f}%"
            f"{r['max_drawdown_pct']:>7.1f}%{r['sortino']:>9.2f}{r['total_return_pct']:>8.1f}%"
        )
    lines.append("")
    lines.append(
        f"SPY baseline: CAGR {spy_metrics['cagr_pct']:.1f}%"
        f" | Sharpe {spy_metrics['sharpe']:.2f}"
        f" | MaxDD {spy_metrics['max_drawdown_pct']:.1f}%"
    )
    return "\n".join(lines)


def run_sweep(
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
    """Run a full parameter sweep: vectorized signals → batched vbt sim → persist + print."""
    persist.init_schema()
    param_grid = strategy_cls.sweep_params()
    sweep_id = persist.start_sweep(strategy_name, market, param_grid, len(grid))

    prices = pd.concat(
        {s: ohlcv[s]["close"] for s in ohlcv.columns.get_level_values(0).unique()},
        axis=1,
    )

    # Determine symbols from the universe (all available in ohlcv)
    symbols = sorted(ohlcv.columns.get_level_values(0).unique())

    # Generate signals for all combos
    strat_instance = strategy_cls(cfg)
    if hasattr(strat_instance, "sweep_signals"):
        all_targets = strat_instance.sweep_signals(grid, symbols, ohlcv)
    else:
        # Weight strategies: build per-combo, simulate together
        all_targets = {}
        for combo_params in grid:
            merged = {**{"top_n": cfg.top_n, "lookback": cfg.lookback, "skip": cfg.skip}, **combo_params}
            combo_cfg = LabConfig(
                top_n=int(merged.get("top_n", cfg.top_n)),
                lookback=int(merged.get("lookback", cfg.lookback)),
                skip=int(merged.get("skip", cfg.skip)),
                min_history_bars=cfg.min_history_bars,
                max_stocks=cfg.max_stocks,
            )
            start_ts = pd.Timestamp(eval_start, tz="UTC")
            end_ts = pd.Timestamp(eval_end, tz="UTC")
            dates = rebalance_dates(ohlcv.index, start_ts, end_ts)
            strat = strategy_cls(combo_cfg)
            plans = {}
            for asof in dates:
                past = ohlcv.loc[:asof]
                elig = eligible_at(asof, past, combo_cfg)[0]
                plans[asof] = strat.select(asof, past, elig)
            targets = strat.to_targets(plans, ohlcv)
            key = combo_name(strategy_name, combo_params)
            all_targets[key] = targets

    # Batched simulation — one vbt call
    start_cash = float(base_config["START_CASH"])
    spy_stats = curve_stats(start_cash * (spy_close / spy_close.dropna().iloc[0]))

    if isinstance(next(iter(all_targets.values())), SignalTargets):
        rets_df, eq_df, diags = simulate_signals(all_targets, prices, base_config)
    else:
        rets_df, eq_df, diags = simulate_weights(all_targets, prices, base_config)

    # Score each combo and persist
    result_rows: List[Dict[str, Any]] = []
    for key in all_targets:
        eq = eq_df[key].dropna()
        if len(eq) < 2:
            continue
        metrics = curve_stats(eq)
        combo_params = next(c for c in grid if combo_name(strategy_name, c) == key)
        persist.write_sweep_combo(
            sweep_id, key, combo_params, metrics, spy_stats, diags.get(key, {})
        )
        result_rows.append({"combo": key, **metrics})

    persist.finish_sweep(sweep_id)

    table = format_results_table(
        result_rows, strategy_name, len(grid), eval_start, eval_end, sweep_id, spy_stats
    )
    print(table)
    return sweep_id
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/lab/test_sweep.py -v -k "not integration"`
Expected: All non-integration tests PASS (format_results_table tests + earlier tests).

- [ ] **Step 5: Commit**

```bash
git add src/ggTrader/lab/sweep.py tests/lab/test_sweep.py
git commit -m "feat(lab): sweep orchestrator — batched vbt simulation and results table"
```

---

### Task 6: CLI Integration — `--sweep` and `--sweep-param` Flags

**Files:**
- Modify: `src/ggTrader/lab/cli.py`
- Test: `tests/lab/test_sweep.py` (append tests)

**Interfaces:**
- Consumes:
  - `run_sweep()`, `build_grid()` from Tasks 2/5
  - `build_signal_strategy()`, `SIGNAL_STRATEGY_NAMES` from `signals.py`
  - `build_strategy()`, `STRATEGY_NAMES` from `momentum.py`
  - `EmaCrossSignal`, `WfoTournamentSignal` class refs
  - `CrossSectionalMomentum`, `DualMomentum` class refs
- Produces: Updated `build_arg_parser()` with `--sweep` and `--sweep-param` flags; updated `run_lab()` that branches into sweep mode.

- [ ] **Step 1: Write failing tests for CLI sweep args**

Append to `tests/lab/test_sweep.py`:

```python
def test_cli_parser_accepts_sweep_flag():
    from ggTrader.lab.cli import build_arg_parser

    p = build_arg_parser()
    args = p.parse_args(["--strategy", "ema_cross", "--sweep"])
    assert args.sweep is True


def test_cli_parser_accepts_sweep_param():
    from ggTrader.lab.cli import build_arg_parser

    p = build_arg_parser()
    args = p.parse_args([
        "--strategy", "ema_cross", "--sweep",
        "--sweep-param", "ema_fast=5,10", "--sweep-param", "ema_slow=50,100",
    ])
    assert args.sweep_param == ["ema_fast=5,10", "ema_slow=50,100"]


def test_cli_parser_sweep_param_without_sweep_is_ok():
    from ggTrader.lab.cli import build_arg_parser

    p = build_arg_parser()
    args = p.parse_args(["--strategy", "ema_cross"])
    assert args.sweep is False
    assert args.sweep_param == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/lab/test_sweep.py::test_cli_parser_accepts_sweep_flag -v`
Expected: FAIL — argparse error, `--sweep` unrecognized.

- [ ] **Step 3: Add `--sweep` and `--sweep-param` to the arg parser**

In `src/ggTrader/lab/cli.py`, add these lines inside `build_arg_parser()`, after the `--max-stocks` argument (line 31):

```python
    p.add_argument("--sweep", action="store_true", default=False,
                   help="Run parameter sweep instead of single walk-forward.")
    p.add_argument("--sweep-param", action="append", default=[],
                   help="Override sweep range: --sweep-param ema_fast=5,10,20")
```

- [ ] **Step 4: Update `run_lab()` to branch into sweep mode**

Replace the `run_lab` function body in `src/ggTrader/lab/cli.py` with:

```python
def run_lab(argv: List[str] | None = None) -> str:
    args = build_arg_parser().parse_args(argv)
    cfg = LabConfig(
        top_n=args.top_n, lookback=args.lookback, skip=args.skip, max_stocks=args.max_stocks
    )

    eval_start = pd.Timestamp(args.eval_start, tz="UTC")
    eval_end = (
        pd.Timestamp(args.eval_end, tz="UTC")
        if args.eval_end
        else pd.Timestamp.now(tz="UTC").normalize()
    )
    warmup_days = int(max(cfg.lookback, cfg.min_history_bars) * 1.6) + 60
    data_start = eval_start - pd.Timedelta(days=warmup_days)
    universe = equity_universe_between(eval_start, eval_end)
    ohlcv = load_ohlcv(universe + ["SPY"], str(data_start.date()), str(eval_end.date()))
    spy_close = ohlcv["SPY"]["close"].dropna()
    sym_cols = [s for s in ohlcv.columns.get_level_values(0).unique() if s != "SPY"]
    ohlcv = ohlcv[sym_cols]

    if args.sweep:
        from ggTrader.lab.strategies.momentum import CrossSectionalMomentum, DualMomentum
        from ggTrader.lab.strategies.signals import EmaCrossSignal, WfoTournamentSignal
        from ggTrader.lab.sweep import build_grid, run_sweep

        cls_map = {
            "ema_cross": EmaCrossSignal,
            "wfo_tournament": WfoTournamentSignal,
            "xs_momentum": CrossSectionalMomentum,
            "dual_momentum": DualMomentum,
        }
        strategy_cls = cls_map[args.strategy]
        overrides = _parse_sweep_params(args.sweep_param)
        grid = build_grid(strategy_cls, overrides=overrides if overrides else None)
        print(f"Sweep: {args.strategy} | {len(grid)} param combos")
        return run_sweep(
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

    if args.strategy in SIGNAL_STRATEGY_NAMES:
        strat = build_signal_strategy(args.strategy, cfg)
    else:
        strat = build_strategy(args.strategy, cfg)

    run_id = walkforward(
        [strat],
        ohlcv,
        spy_close,
        eval_start=str(eval_start.date()),
        eval_end=str(eval_end.date()),
        market=args.market,
        freq="monthly",
        universe_fn=lambda asof, past: eligible_at(asof, past, cfg)[0],
        base_config=dict(STOCK_BASE_CONFIG),
    )
    print(f"lab run complete: {run_id}")
    return run_id


def _parse_sweep_params(raw: List[str]) -> dict[str, list]:
    """Parse CLI '--sweep-param key=v1,v2,v3' into {key: [v1, v2, v3]}."""
    result: dict[str, list] = {}
    for item in raw:
        key, _, vals = item.partition("=")
        if not key or not vals:
            raise ValueError(f"Invalid --sweep-param: {item!r} (expected key=v1,v2,...)")
        parsed = []
        for v in vals.split(","):
            v = v.strip()
            try:
                parsed.append(int(v))
            except ValueError:
                try:
                    parsed.append(float(v))
                except ValueError:
                    parsed.append(v)
        result[key.strip()] = parsed
    return result
```

Also add `List` to the typing import on line 5 if not already there (it is: `from typing import List`).

- [ ] **Step 5: Run all tests to verify they pass**

Run: `pytest tests/lab/test_sweep.py -v -k "not integration"`
Expected: All non-integration tests PASS.

- [ ] **Step 6: Run the full existing test suite to check for regressions**

Run: `pytest tests/ -v -k "not integration" --tb=short`
Expected: All existing tests PASS. No regressions.

- [ ] **Step 7: Commit**

```bash
git add src/ggTrader/lab/cli.py tests/lab/test_sweep.py
git commit -m "feat(lab): CLI --sweep and --sweep-param flags for parameter sweeps"
```

---

### Task 7: Integration Smoke Test — End-to-End Sweep in Docker

**Files:**
- Test: `tests/lab/test_sweep.py` (append integration test)

**Interfaces:**
- Consumes: All previous tasks — full `run_sweep()` pipeline
- Produces: One passing integration test that exercises the full path

- [ ] **Step 1: Write integration smoke test**

Append to `tests/lab/test_sweep.py`:

```python
@pytest.mark.integration
def test_sweep_end_to_end_ema_cross_small_grid(tmp_path):
    """Full sweep: 2 combos, synthetic OHLCV, hits DB."""
    from ggTrader.lab.data import STOCK_BASE_CONFIG
    from ggTrader.lab.persist import get_engine, init_schema
    from ggTrader.lab.strategies.signals import EmaCrossSignal
    from ggTrader.lab.sweep import build_grid, run_sweep
    from sqlalchemy import text

    init_schema()
    ohlcv = _ohlcv(["A", "B"], n=600)
    spy_idx = ohlcv.index
    spy_close = pd.Series(100.0 * 1.0004 ** np.arange(len(spy_idx)), index=spy_idx)

    grid = build_grid(EmaCrossSignal, overrides={"ema_fast": [5, 10], "ema_slow": [50]})
    assert len(grid) == 2

    cfg = LabConfig(min_history_bars=100, top_n=2)
    sweep_id = run_sweep(
        "ema_cross", EmaCrossSignal, cfg, ohlcv, spy_close,
        eval_start=str(ohlcv.index[200].date()),
        eval_end=str(ohlcv.index[-1].date()),
        market="equity",
        base_config=dict(STOCK_BASE_CONFIG),
        grid=grid,
    )
    assert sweep_id.startswith("sweep_ema_cross_")

    with get_engine().connect() as conn:
        sweep_row = conn.execute(
            text("SELECT status, n_combos FROM lab_sweeps WHERE sweep_id = :s"),
            {"s": sweep_id},
        ).first()
        assert sweep_row[0] == "done"
        assert sweep_row[1] == 2

        combo_count = conn.execute(
            text("SELECT count(*) FROM lab_sweep_combos WHERE sweep_id = :s"),
            {"s": sweep_id},
        ).scalar()
        assert combo_count == 2
```

- [ ] **Step 2: Run integration test in Docker**

Run: `docker compose run --rm ggtrader_live pytest tests/lab/test_sweep.py::test_sweep_end_to_end_ema_cross_small_grid -v -m integration`
Expected: PASS — full pipeline exercises grid → sweep_signals → simulate_signals → persist → format.

- [ ] **Step 3: Run full test suite in Docker**

Run: `docker compose run --rm ggtrader_live pytest tests/ -v --tb=short`
Expected: All tests PASS. No regressions.

- [ ] **Step 4: Commit**

```bash
git add tests/lab/test_sweep.py
git commit -m "test(lab): end-to-end integration test for parameter sweep pipeline"
```
