# Unified Registry + `--blend` Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collapse the triplicated strategy registry to one source of truth, and add a first-class `ggt lab --blend strategy@universe,...` path that blends sleeves through the gated WFO via the validated inverse-vol/target-vol overlay, persisted as a normal lab run.

**Architecture:** Part A — `STRATEGY_REGISTRY` stays the sole map; a new `registry.py` derives names/builders lazily; `signals.py`/`momentum.py` keep their public names as PEP-562 `__getattr__` shims (zero call-site edits). Part B — a pure `blend_curves` math helper + a `run_blend` orchestrator (load → `run_wfo` per sleeve → `combine_sleeves` → persist), wired into the CLI; the two research scripts are retired.

**Tech Stack:** Python 3.12, pandas, vectorbt (via existing `simulate`/`wfo`), pytest. Native `.venv`.

## Global Constraints

- Absolute imports from `ggTrader` (no relative imports in package code, except the existing `from .x import` style already used inside `strategies/__init__.py`).
- Vectorization-first: no per-symbol Python loops over price data; a loop over sleeves or rebalance dates is fine.
- Strict ruff: run `.venv/bin/ruff check <files>` before each commit.
- Tests live under `tests/lab/`; run with `.venv/bin/python -m pytest tests/lab/<file> -v`.
- `STRATEGY_REGISTRY` in `strategies/__init__.py` is the ONE source of truth; all names/builders derive from it via each class's `target_kind` attribute.
- The blend uses ONLY the validated inverse-vol → target-vol scheme in `allocation.combine_sleeves` (defaults `target_vol=0.068`, `window=60`, `max_leverage=2.0`). No fixed-weight modes.
- A sleeve is `strategy@universe`; `strategy ∈ all_strategy_names()`, `universe ∈ UNIVERSE_CHOICES = ("sp500","nasdaq100","russell2000","midcap400")`.
- Do NOT modify strategy classes or `allocation.py` (both already validated/tested).

---

### Task 1: Single-source registry helpers

**Files:**
- Create: `src/ggTrader/lab/strategies/registry.py`
- Modify: `src/ggTrader/lab/strategies/__init__.py` (re-export helpers; extend `__all__`)
- Test: `tests/lab/test_registry.py` (create)

**Interfaces:**
- Produces: `signal_strategy_names() -> tuple[str,...]`, `weight_strategy_names() -> tuple[str,...]`, `all_strategy_names() -> tuple[str,...]`, `signal_registry() -> dict[str, type]`, `build_strategy(name: str, cfg: LabConfig) -> Any`. Importable from both `ggTrader.lab.strategies.registry` and `ggTrader.lab.strategies`.

- [ ] **Step 1: Write the failing test**

```python
# tests/lab/test_registry.py
"""The strategy registry is a single source of truth; names/builders derive from it."""

import pytest

from ggTrader.lab.strategies import (
    STRATEGY_REGISTRY,
    all_strategy_names,
    build_strategy,
    signal_strategy_names,
    weight_strategy_names,
)
from ggTrader.lab.strategy import LabConfig


def test_registry_keys_match_class_name_and_kind():
    for name, cls in STRATEGY_REGISTRY.items():
        assert cls.name == name, f"key {name!r} != cls.name {cls.name!r}"
        assert cls.target_kind in {"signals", "weights"}


def test_name_views_partition_the_registry():
    sig, wt = set(signal_strategy_names()), set(weight_strategy_names())
    assert sig.isdisjoint(wt)
    assert sig | wt == set(STRATEGY_REGISTRY)
    assert set(all_strategy_names()) == set(STRATEGY_REGISTRY)


def test_build_strategy_builds_every_registered_name():
    cfg = LabConfig()
    for name in all_strategy_names():
        assert build_strategy(name, cfg).name == name


def test_build_strategy_unknown_raises():
    with pytest.raises(ValueError):
        build_strategy("nope", LabConfig())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/lab/test_registry.py -v`
Expected: FAIL with `ImportError: cannot import name 'all_strategy_names'`

- [ ] **Step 3: Create `registry.py`**

```python
"""Single-source derivation of strategy names/builders from STRATEGY_REGISTRY.

Helpers import STRATEGY_REGISTRY lazily (inside functions) to avoid an import
cycle: strategies/__init__.py imports the strategy modules to build the
registry, so those modules cannot read it at their own import time.
"""

from __future__ import annotations

from typing import Any

from ggTrader.lab.strategy import LabConfig


def _registry() -> dict[str, Any]:
    from ggTrader.lab.strategies import STRATEGY_REGISTRY

    return STRATEGY_REGISTRY


def signal_strategy_names() -> tuple[str, ...]:
    return tuple(n for n, c in _registry().items() if c.target_kind == "signals")


def weight_strategy_names() -> tuple[str, ...]:
    return tuple(n for n, c in _registry().items() if c.target_kind == "weights")


def all_strategy_names() -> tuple[str, ...]:
    return tuple(_registry())


def signal_registry() -> dict[str, Any]:
    return {n: c for n, c in _registry().items() if c.target_kind == "signals"}


def build_strategy(name: str, cfg: LabConfig) -> Any:
    reg = _registry()
    if name not in reg:
        raise ValueError(f"Unknown strategy {name!r}. Available: {tuple(reg)}")
    return reg[name](cfg)
```

- [ ] **Step 4: Re-export from `__init__.py`**

Add after the `STRATEGY_REGISTRY = {...}` block in `src/ggTrader/lab/strategies/__init__.py`:

```python
from .registry import (
    all_strategy_names,
    build_strategy,
    signal_registry,
    signal_strategy_names,
    weight_strategy_names,
)
```

And add these names to the `__all__` list:

```python
    "all_strategy_names",
    "build_strategy",
    "signal_registry",
    "signal_strategy_names",
    "weight_strategy_names",
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/lab/test_registry.py -v`
Expected: PASS (4 passed). If `test_build_strategy_builds_every_registered_name` fails, a strategy constructor needs more than `cfg` — surface it (do not paper over).

- [ ] **Step 6: Lint + commit**

```bash
.venv/bin/ruff check src/ggTrader/lab/strategies/registry.py src/ggTrader/lab/strategies/__init__.py tests/lab/test_registry.py
git add src/ggTrader/lab/strategies/registry.py src/ggTrader/lab/strategies/__init__.py tests/lab/test_registry.py
git commit -m "feat(lab): single-source strategy registry helpers"
```

---

### Task 2: `signals.py` lazy shims

**Files:**
- Modify: `src/ggTrader/lab/strategies/signals.py` (delete the literal registry block ~604-653; add `__getattr__`)
- Test: existing `tests/lab/test_ensemble_ic.py`, `test_conviction.py`, `test_ensemble_conviction.py`, `test_macd_signals.py`, `test_mtf_signals.py`, `test_reversion_signals.py` must still pass.

**Interfaces:**
- Consumes: `registry.signal_strategy_names`, `registry.build_strategy`, `registry.signal_registry` (Task 1).
- Produces (lazily): `signals.SIGNAL_STRATEGY_NAMES`, `signals.build_signal_strategy`, `signals._get_registry` — unchanged names/behavior for all existing importers.

- [ ] **Step 1: Delete the literal registry block**

In `src/ggTrader/lab/strategies/signals.py`, delete `_build_signal_registry()`, `_SIGNAL_REGISTRY`, `_get_registry()`, the `SIGNAL_STRATEGY_NAMES = (...)` tuple, and `build_signal_strategy()` (the block spanning roughly lines 604–653, ending the file's previous tail). Confirm first with `grep -n "_build_signal_registry\|_get_registry\|SIGNAL_STRATEGY_NAMES\|build_signal_strategy" src/ggTrader/lab/strategies/signals.py` that none are referenced earlier in the file (they are not — they are only defined at the tail).

- [ ] **Step 2: Add the `__getattr__` shim at the end of `signals.py`**

```python
def __getattr__(name: str):  # PEP 562 — derive public names from the single registry
    from ggTrader.lab.strategies import registry

    if name == "SIGNAL_STRATEGY_NAMES":
        return registry.signal_strategy_names()
    if name == "build_signal_strategy":
        return registry.build_strategy
    if name == "_get_registry":
        return registry.signal_registry
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

- [ ] **Step 3: Run the affected tests to verify they still pass**

Run: `.venv/bin/python -m pytest tests/lab/test_ensemble_ic.py tests/lab/test_conviction.py tests/lab/test_ensemble_conviction.py tests/lab/test_macd_signals.py tests/lab/test_mtf_signals.py tests/lab/test_reversion_signals.py -v`
Expected: PASS (all). These exercise `build_signal_strategy(...)`, `SIGNAL_STRATEGY_NAMES`, and `_get_registry()` via the new shim.

- [ ] **Step 4: Lint + commit**

```bash
.venv/bin/ruff check src/ggTrader/lab/strategies/signals.py
git add src/ggTrader/lab/strategies/signals.py
git commit -m "refactor(lab): signals.py registry names derive from single source via __getattr__"
```

---

### Task 3: `momentum.py` lazy shims

**Files:**
- Modify: `src/ggTrader/lab/strategies/momentum.py` (delete `_REGISTRY`/`STRATEGY_NAMES`/`build_strategy` literals ~75-86; add `__getattr__`)
- Test: existing `tests/lab/test_momentum.py` must still pass.

**Interfaces:**
- Consumes: `registry.weight_strategy_names`, `registry.build_strategy` (Task 1).
- Produces (lazily): `momentum.STRATEGY_NAMES`, `momentum.build_strategy` — unchanged for importers (`cli.py`, `test_momentum.py`).

- [ ] **Step 1: Delete the literal block**

In `src/ggTrader/lab/strategies/momentum.py`, delete `_REGISTRY = {...}`, `STRATEGY_NAMES = tuple(_REGISTRY)`, and `def build_strategy(...)` (≈ lines 75-86). Keep the `CrossSectionalMomentum` / `DualMomentum` class definitions.

- [ ] **Step 2: Add the `__getattr__` shim at the end of `momentum.py`**

```python
def __getattr__(name: str):  # PEP 562 — derive public names from the single registry
    from ggTrader.lab.strategies import registry

    if name == "STRATEGY_NAMES":
        return registry.weight_strategy_names()
    if name == "build_strategy":
        return registry.build_strategy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

- [ ] **Step 3: Run the affected test**

Run: `.venv/bin/python -m pytest tests/lab/test_momentum.py -v`
Expected: PASS. `test_build_strategy_dispatch` builds `xs_momentum`/`dual_momentum` and expects `ValueError` for `"nope"` — the unified `build_strategy` satisfies all three.

- [ ] **Step 4: Lint + commit**

```bash
.venv/bin/ruff check src/ggTrader/lab/strategies/momentum.py
git add src/ggTrader/lab/strategies/momentum.py
git commit -m "refactor(lab): momentum.py registry names derive from single source via __getattr__"
```

---

### Task 4: `blend_curves` pure helper

**Files:**
- Create: `src/ggTrader/lab/blend.py`
- Test: `tests/lab/test_blend.py` (create)

**Interfaces:**
- Consumes: `allocation.combine_sleeves`, `metrics.curve_stats`, `data.STOCK_BASE_CONFIG`.
- Produces: `blend_curves(curves: dict[str, pd.Series], *, target_vol: float = 0.068, window: int = 60, max_leverage: float = 2.0) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]` returning `(blend_equity, returns_df, diag)`. `blend_equity` is a cumprod equity curve starting at `START_CASH`; `returns_df` is the aligned per-sleeve daily returns (intersection of dates); `diag` is the `combine_sleeves` diagnostics.

- [ ] **Step 1: Write the failing test**

```python
# tests/lab/test_blend.py
"""Tests for the portfolio-blend helper and orchestrator."""

import numpy as np
import pandas as pd

from ggTrader.lab.blend import blend_curves


def _idx(n, start="2021-01-01"):
    return pd.date_range(start, periods=n, freq="B", tz="UTC")


def _equity_from_returns(rets: pd.Series, start=100000.0) -> pd.Series:
    return (1.0 + rets).cumprod() * start


def test_blend_curves_aligns_on_intersection_and_blends():
    idx = _idx(400)
    np.random.seed(0)
    a = _equity_from_returns(pd.Series(np.random.normal(0.0004, 0.01, 400), index=idx))
    # b starts 50 bars later -> intersection trims the blend to the common span
    b = _equity_from_returns(pd.Series(np.random.normal(0.0004, 0.01, 350), index=idx[50:]))
    blend_eq, returns_df, diag = blend_curves({"A@sp500": a, "B@nasdaq100": b})
    assert list(returns_df.columns) == ["A@sp500", "B@nasdaq100"]
    assert returns_df.index.min() >= idx[50]  # trimmed to the later start
    assert blend_eq.notna().all()
    assert (blend_eq > 0).all()


def test_blend_curves_equal_vol_gives_balanced_weights():
    """Two sleeves with the same vol get ~50/50 inverse-vol weight (diag)."""
    idx = _idx(400)
    rng = np.random.default_rng(1)
    a = _equity_from_returns(pd.Series(rng.normal(0.0003, 0.012, 400), index=idx))
    b = _equity_from_returns(pd.Series(rng.normal(0.0003, 0.012, 400), index=idx))
    _, _, diag = blend_curves({"A@sp500": a, "B@nasdaq100": b}, window=60)
    last = diag.iloc[-1]
    assert abs(last["w_A@sp500"] - last["w_B@nasdaq100"]) < 0.15  # near-balanced
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/lab/test_blend.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ggTrader.lab.blend'`

- [ ] **Step 3: Create `blend.py` with the pure helper**

```python
"""Portfolio-of-sleeves blend: run sleeves through the gated WFO and combine
their OOS curves with the validated inverse-vol / target-vol overlay.

blend_curves is the pure math (no I/O); run_blend orchestrates data load, WFO,
blend, and persistence.
"""

from __future__ import annotations

from functools import reduce
from typing import Any, NamedTuple

import pandas as pd

from ggTrader.lab.allocation import combine_sleeves
from ggTrader.lab.data import STOCK_BASE_CONFIG


def blend_curves(
    curves: dict[str, pd.Series],
    *,
    target_vol: float = 0.068,
    window: int = 60,
    max_leverage: float = 2.0,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Align sleeve OOS equity curves on common dates, blend to a target vol.

    Returns (blend_equity, returns_df, diag). blend_equity is a cumprod curve
    starting at START_CASH; returns_df is the aligned per-sleeve daily returns.
    """
    common = reduce(lambda a, b: a.intersection(b), (c.index for c in curves.values()))
    returns_df = pd.DataFrame(
        {label: curves[label].reindex(common).pct_change() for label in curves}
    ).dropna()
    blended, diag = combine_sleeves(
        returns_df, target_vol=target_vol, window=window, max_leverage=max_leverage
    )
    start_cash = float(STOCK_BASE_CONFIG["START_CASH"])
    blend_equity = (1.0 + blended).cumprod() * start_cash
    return blend_equity, returns_df, diag
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/lab/test_blend.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Lint + commit**

```bash
.venv/bin/ruff check src/ggTrader/lab/blend.py tests/lab/test_blend.py
git add src/ggTrader/lab/blend.py tests/lab/test_blend.py
git commit -m "feat(lab): blend_curves pure helper (align + inverse-vol/target-vol)"
```

---

### Task 5: `run_blend` orchestration + persistence

**Files:**
- Modify: `src/ggTrader/lab/blend.py` (add `BlendResult`, `run_blend`)
- Test: `tests/lab/test_blend.py` (append)

**Interfaces:**
- Consumes: `blend_curves` (Task 4); `data.equity_universe_between`, `data.load_ohlcv`; `wfo.run_wfo` (returns `WfoResult(oos_equity, fold_results, live_params, table)`); `sweep.build_grid`; `strategies.STRATEGY_REGISTRY`; `metrics.curve_stats`; `persist` (`init_schema`, `start_run`, `write_returns_equity`, `write_summary`, `finish_run`); `LabConfig`.
- Produces: `BlendResult = NamedTuple(blended_equity: pd.Series, sleeve_equity: dict[str,pd.Series], diag: pd.DataFrame, table: str, run_id: str)`; `run_blend(sleeves: list[tuple[str,str]], cfg: LabConfig, eval_start: str, eval_end: str, *, market: str, base_config: dict, target_vol: float = 0.068, window: int = 60, max_leverage: float = 2.0) -> BlendResult`.

- [ ] **Step 1: Write the failing test (patches WFO/data/persist)**

```python
# append to tests/lab/test_blend.py
import ggTrader.lab.blend as blend_mod
from ggTrader.lab.strategy import LabConfig
from ggTrader.lab.wfo import WfoResult


def test_run_blend_orchestrates_and_persists(monkeypatch):
    idx = _idx(400)
    rng = np.random.default_rng(2)
    eqs = {
        "ensemble@sp500": _equity_from_returns(pd.Series(rng.normal(0.0004, 0.011, 400), index=idx)),
        "xs_momentum@nasdaq100": _equity_from_returns(
            pd.Series(rng.normal(0.0003, 0.013, 400), index=idx)
        ),
    }
    spy = _equity_from_returns(pd.Series(rng.normal(0.0003, 0.01, 400), index=idx))
    spy_ohlcv = pd.concat({"SPY": pd.DataFrame({"close": spy})}, axis=1)
    spy_ohlcv.columns.names = ["symbol", "field"]

    # universe membership + ohlcv: return a frame containing SPY + 1 dummy symbol
    monkeypatch.setattr(blend_mod, "equity_universe_between", lambda *a, **k: ["AAA"])

    def _fake_load(symbols, start, end, **k):
        frames = {"SPY": pd.DataFrame({"close": spy}, index=idx)}
        frames["AAA"] = pd.DataFrame({"close": spy.values}, index=idx)
        df = pd.concat(frames, axis=1)
        df.columns.names = ["symbol", "field"]
        return df

    monkeypatch.setattr(blend_mod, "load_ohlcv", _fake_load)
    monkeypatch.setattr(blend_mod, "build_grid", lambda cls: [{}])

    labels = iter(eqs.values())

    def _fake_wfo(name, cls, cfg, ohlcv, spy_close, **k):
        return WfoResult(oos_equity=next(labels), fold_results=[], live_params={}, table="t")

    monkeypatch.setattr(blend_mod, "run_wfo", _fake_wfo)

    calls = {"start": 0, "returns": 0, "summary": 0, "finish": 0}
    monkeypatch.setattr(blend_mod.persist, "init_schema", lambda: None)
    monkeypatch.setattr(
        blend_mod.persist, "start_run", lambda *a, **k: calls.__setitem__("start", calls["start"] + 1) or "run123"
    )
    monkeypatch.setattr(
        blend_mod.persist, "write_returns_equity", lambda *a, **k: calls.__setitem__("returns", calls["returns"] + 1)
    )
    monkeypatch.setattr(
        blend_mod.persist, "write_summary", lambda *a, **k: calls.__setitem__("summary", calls["summary"] + 1)
    )
    monkeypatch.setattr(
        blend_mod.persist, "finish_run", lambda *a, **k: calls.__setitem__("finish", calls["finish"] + 1)
    )

    result = run_blend(
        [("ensemble", "sp500"), ("xs_momentum", "nasdaq100")],
        LabConfig(),
        "2021-01-01",
        "2022-07-01",
        market="equity",
        base_config=dict(blend_mod.STOCK_BASE_CONFIG),
    )
    assert result.run_id == "run123"
    assert set(result.sleeve_equity) == {"ensemble@sp500", "xs_momentum@nasdaq100"}
    assert result.blended_equity.notna().all()
    assert calls["start"] == 1 and calls["finish"] == 1
    assert calls["returns"] == 3  # 2 sleeves + 1 blend
    assert calls["summary"] == 1
    assert "blend" in result.table.lower()


from ggTrader.lab.blend import run_blend  # noqa: E402
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/lab/test_blend.py::test_run_blend_orchestrates_and_persists -v`
Expected: FAIL with `ImportError: cannot import name 'run_blend'`

- [ ] **Step 3: Append `BlendResult` + `run_blend` (and module-level imports) to `blend.py`**

Add these imports to the top of `blend.py` (with the existing ones):

```python
from ggTrader.lab import persist
from ggTrader.lab.data import equity_universe_between, load_ohlcv
from ggTrader.lab.metrics import curve_stats
from ggTrader.lab.strategies import STRATEGY_REGISTRY
from ggTrader.lab.strategy import LabConfig
from ggTrader.lab.sweep import build_grid
from ggTrader.lab.wfo import run_wfo
```

Then append:

```python
class BlendResult(NamedTuple):
    blended_equity: pd.Series
    sleeve_equity: dict[str, pd.Series]
    diag: pd.DataFrame
    table: str
    run_id: str


def _row(label: str, s: dict[str, Any]) -> str:
    return (
        f"| {label} | {s['cagr_pct']:.2f}% | {s['sharpe']:.2f} | {s['sortino']:.2f} "
        f"| {s['ann_vol_pct']:.2f}% | {s['max_drawdown_pct']:.2f}% |"
    )


def run_blend(
    sleeves: list[tuple[str, str]],
    cfg: LabConfig,
    eval_start: str,
    eval_end: str,
    *,
    market: str,
    base_config: dict,
    target_vol: float = 0.068,
    window: int = 60,
    max_leverage: float = 2.0,
) -> BlendResult:
    """Run each (strategy, universe) sleeve through the gated WFO, blend the OOS
    curves (inverse-vol -> target-vol), persist as a lab run, return BlendResult.
    """
    es = pd.Timestamp(eval_start, tz="UTC")
    ee = pd.Timestamp(eval_end, tz="UTC")
    warmup_days = int(max(cfg.lookback, cfg.min_history_bars) * 1.6) + 60
    data_start = str((es - pd.Timedelta(days=warmup_days)).date())

    universes = sorted({u for _, u in sleeves})
    members = {u: equity_universe_between(es, ee, universe=u) for u in universes}
    all_symbols = sorted({s for ms in members.values() for s in ms} | {"SPY"})
    ohlcv = load_ohlcv(all_symbols, data_start, eval_end, use_negative_cache=True)
    available = set(ohlcv.columns.get_level_values(0))
    spy_close = ohlcv["SPY"]["close"].dropna()

    curves: dict[str, pd.Series] = {}
    for strategy, universe in sleeves:
        label = f"{strategy}@{universe}"
        syms = [x for x in members[universe] if x in available]
        cls = STRATEGY_REGISTRY[strategy]
        result = run_wfo(
            strategy, cls, cfg, ohlcv[syms], spy_close,
            eval_start=eval_start, eval_end=eval_end, market=market,
            base_config=base_config, grid=build_grid(cls),
        )
        curves[label] = result.oos_equity

    blend_eq, returns_df, diag = blend_curves(
        curves, target_vol=target_vol, window=window, max_leverage=max_leverage
    )
    common = returns_df.index
    start_cash = float(base_config["START_CASH"])
    spy_common = spy_close.reindex(common).ffill()
    spy_bench = start_cash * (spy_common / spy_common.dropna().iloc[0])
    spy_stats = curve_stats(spy_common)

    # Build the report table.
    rows = ["| Strategy | CAGR | Sharpe | Sortino | Vol | Max DD |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |"]
    for label in curves:
        rows.append(_row(label, curve_stats(curves[label].reindex(common))))
    rows.append(_row("Inverse-vol + target-vol blend", curve_stats(blend_eq)))
    rows.append(_row("SPY", spy_stats))
    table = "\n".join(rows)

    # Persist as a normal lab run.
    persist.init_schema()
    labels = list(curves)
    run_id = persist.start_run(
        f"blend:{','.join(labels)}", market, "blend", eval_start, eval_end,
        params={"sleeves": labels, "target_vol": target_vol, "window": window,
                "max_leverage": max_leverage},
    )
    for label in labels:
        sleeve_eq = curves[label].reindex(common)
        persist.write_returns_equity(run_id, label, sleeve_eq.pct_change().dropna(), sleeve_eq, spy_bench)
    persist.write_returns_equity(run_id, "blend", blend_eq.pct_change().dropna(), blend_eq, spy_bench)
    persist.write_summary(
        run_id, "blend", curve_stats(blend_eq), spy_stats,
        {"avg_leverage": float(diag["scale"].mean()), "max_leverage": float(diag["scale"].max()),
         "sleeves": labels},
    )
    persist.finish_run(run_id)

    return BlendResult(
        blended_equity=blend_eq,
        sleeve_equity={label: curves[label].reindex(common) for label in labels},
        diag=diag, table=table, run_id=run_id,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/lab/test_blend.py -v`
Expected: PASS (3 passed). If `write_returns_equity` count != 3, check the per-sleeve + blend write loop.

- [ ] **Step 5: Lint + commit**

```bash
.venv/bin/ruff check src/ggTrader/lab/blend.py tests/lab/test_blend.py
git add src/ggTrader/lab/blend.py tests/lab/test_blend.py
git commit -m "feat(lab): run_blend orchestration + lab-run persistence"
```

---

### Task 6: CLI `--blend` wiring + retire scripts

**Files:**
- Modify: `src/ggTrader/lab/cli.py` (args + dispatch branch + sleeve parser)
- Delete: `scripts/multi_sleeve_research.py`, `scripts/portfolio_blend.py`
- Test: `tests/lab/test_cli.py` (append)

**Interfaces:**
- Consumes: `blend.run_blend` (Task 5); `all_strategy_names` (Task 1); `UNIVERSE_CHOICES` (cli.py).
- Produces: `--blend "s@u,..."`, `--target-vol`, `--blend-window`, `--max-leverage` CLI args; a module-level `_parse_blend_sleeves(spec: str) -> list[tuple[str,str]]`.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/lab/test_cli.py
import pytest

from ggTrader.lab.cli import _parse_blend_sleeves, build_arg_parser


def test_parse_blend_sleeves_ok():
    assert _parse_blend_sleeves("ensemble@sp500, xs_momentum@nasdaq100") == [
        ("ensemble", "sp500"),
        ("xs_momentum", "nasdaq100"),
    ]


def test_parse_blend_sleeves_unknown_strategy():
    with pytest.raises(SystemExit):
        _parse_blend_sleeves("nope@sp500")


def test_parse_blend_sleeves_unknown_universe():
    with pytest.raises(SystemExit):
        _parse_blend_sleeves("ensemble@mars")


def test_parse_blend_sleeves_bad_format():
    with pytest.raises(SystemExit):
        _parse_blend_sleeves("ensemble_sp500")


def test_blend_is_mutually_exclusive_with_wfo():
    parser = build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--strategy", "ensemble", "--wfo", "--blend", "ensemble@sp500"])
```

(Note: `--strategy` is `required=True`; `--blend` runs may still pass a dummy `--strategy` since the blend branch ignores it. The mutual-exclusion test passes both `--wfo` and `--blend` to trigger the argparse error.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/lab/test_cli.py -k blend -v`
Expected: FAIL with `ImportError: cannot import name '_parse_blend_sleeves'`

- [ ] **Step 3: Add args to `build_arg_parser`**

In `src/ggTrader/lab/cli.py`, add `--blend` to the existing mutually-exclusive `mode` group (next to `--sweep`/`--wfo`), and add the three overlay args after it:

```python
    mode.add_argument(
        "--blend",
        default=None,
        metavar="S@U,...",
        help="Blend sleeves: --blend ensemble@sp500,xs_momentum@nasdaq100",
    )
    p.add_argument("--target-vol", type=float, default=0.068)
    p.add_argument("--blend-window", type=int, default=60)
    p.add_argument("--max-leverage", type=float, default=2.0)
```

- [ ] **Step 4: Add the sleeve parser (module level, near the top of cli.py)**

`UNIVERSE_CHOICES` is already defined at module level in `cli.py` (near the top) — reference it directly, do not re-import it.

```python
def _parse_blend_sleeves(spec: str) -> List[tuple[str, str]]:
    """Parse 'strat@univ,strat@univ' into [(strategy, universe), ...]; validate."""
    from ggTrader.lab.strategies import all_strategy_names

    valid_strats = set(all_strategy_names())
    sleeves: List[tuple[str, str]] = []
    for part in spec.split(","):
        item = part.strip()
        if item.count("@") != 1:
            raise SystemExit(f"Bad sleeve {item!r}; expected 'strategy@universe'")
        strategy, universe = item.split("@")
        if strategy not in valid_strats:
            raise SystemExit(f"Unknown strategy {strategy!r}; choices: {sorted(valid_strats)}")
        if universe not in UNIVERSE_CHOICES:
            raise SystemExit(f"Unknown universe {universe!r}; choices: {UNIVERSE_CHOICES}")
        sleeves.append((strategy, universe))
    return sleeves
```

- [ ] **Step 5: Add the dispatch branch in `run_lab`**

In `run_lab`, immediately AFTER `eval_end` is computed and BEFORE the `warmup_days`/single-universe `equity_universe_between(...)`/`load_ohlcv(...)` block, insert:

```python
    if args.blend:
        from ggTrader.lab.blend import run_blend

        sleeves = _parse_blend_sleeves(args.blend)
        result = run_blend(
            sleeves, cfg, str(eval_start.date()), str(eval_end.date()),
            market=args.market, base_config=dict(STOCK_BASE_CONFIG),
            target_vol=args.target_vol, window=args.blend_window,
            max_leverage=args.max_leverage,
        )
        print(result.table)
        print(f"blend run complete: {result.run_id}")
        return result.run_id
```

This early-returns before the single-universe load, so that load only runs for non-blend modes.

- [ ] **Step 6: Run the CLI tests**

Run: `.venv/bin/python -m pytest tests/lab/test_cli.py -k blend -v`
Expected: PASS (5 passed)

- [ ] **Step 7: Retire the two research scripts**

```bash
git rm scripts/multi_sleeve_research.py scripts/portfolio_blend.py
```
Confirm nothing imports them: `grep -rn "multi_sleeve_research\|portfolio_blend" --include=*.py src/ scripts/ tests/` should return nothing.

- [ ] **Step 8: Full lab suite + lint + commit**

```bash
.venv/bin/python -m pytest tests/lab -q
.venv/bin/ruff check src/ggTrader/lab/cli.py tests/lab/test_cli.py
git add -A
git commit -m "feat(lab): ggt lab --blend strategy@universe path; retire research scripts"
```

---

## Self-Review

**Spec coverage:** registry single-source + helpers (T1) ✓; `__getattr__` shims signals/momentum (T2–T3) ✓; sync test (T1) ✓; `blend_curves` pure + `run_blend` orchestrate + persist (T4–T5) ✓; CLI `--blend` + flags + validation + mutual-exclusion (T6) ✓; retire both scripts (T6) ✓; non-goals (no new weighting modes, no class changes, no decorator) respected ✓.

**Placeholder scan:** none — every code/test step is complete. The one conditional note (`UNIVERSE_CHOICES` import source) is a verify-and-reuse instruction, not a placeholder.

**Type consistency:** `build_strategy(name, cfg)` signature identical T1/T2/T3; `blend_curves` return triple matches T4 def and T5 consumer; `run_blend` signature identical T5 def and T6 call site; `BlendResult` fields (`blended_equity`, `sleeve_equity`, `diag`, `table`, `run_id`) consistent T5/T6; `WfoResult.oos_equity` consumed as defined in `wfo.py`.
