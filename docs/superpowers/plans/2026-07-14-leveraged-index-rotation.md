# Leveraged/Inverse Index Rotation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and WFO-test a breadth-driven leveraged/inverse index rotation
strategy across three universes (SP500, Nasdaq-100, Russell 2000), both
leverage tiers (2x/3x), and produce a GO/NO-GO research verdict.

**Architecture:** A shared `_LeveragedRotationBase` weight-strategy holds all
breadth + monthly-rotation + hysteresis logic; three thin per-universe
subclasses fix each universe's ETF-pair constants. A dedicated orchestration
script (`scripts/leveraged_rotation_research.py`, mirroring `blend.py`'s
`run_blend()`) loads combined OHLCV (universe constituents + all 4 ETF
tickers) and drives `run_wfo()` with a fixed `universe_fn`, since the generic
`--wfo` CLI path has no hook for a custom eligible-set function.

**Tech Stack:** Python, pandas, existing `lab/` research modules
(`strategy.py`, `wfo.py`, `sweep.py`, `data.py`), pytest.

## Global Constraints

- Rebalance cadence is monthly (the harness's `rebalance_dates()` is
  hardcoded monthly) — this is a monthly regime-timing strategy, not daily.
- `EnsembleSignal(cfg)` for breadth computation must be constructed with no
  extra params beyond the shared `LabConfig` — same construction as the
  deployed core, so the breadth measure describes the same validated signal.
- MidCap400 and Dow are out of scope (see spec:
  `docs/superpowers/specs/2026-07-14-leveraged-index-rotation-design.md`).
- No live-trading wiring in this plan — research only.
- `wfo.py` constructs strategies as `strategy_cls(cfg)` (bare) in some code
  paths (anchor-set computation) and `strategy_cls(combo_cfg, **extra_kwargs)`
  (swept params only) in others — every constructor arg not in
  `sweep_params()` must have a class-level default, never be required.

---

### Task 1: Breadth and hysteresis-rotation helpers

**Files:**
- Create: `src/ggTrader/lab/strategies/leveraged_rotation.py`
- Test: `tests/lab/test_leveraged_rotation.py`

**Interfaces:**
- Produces: `compute_breadth(entries: pd.DataFrame) -> pd.Series` — fraction
  of columns with `True` per row, fixed denominator = `entries.shape[1]`.
  `rotate_positions(breadth: pd.Series, upper_threshold: float,
  lower_threshold: float, min_hold_months: int) -> pd.Series` — per-date
  state in `{"long", "inverse", "cash"}`, with hysteresis: a state change
  only takes effect once the new raw signal has held for `min_hold_months`
  consecutive dates in the input series; the very first date's state takes
  effect immediately (no prior state to hold against).

- [ ] **Step 1: Write the failing tests**

Create `tests/lab/test_leveraged_rotation.py`:

```python
"""Tests for the leveraged/inverse index rotation strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _idx(n, start="2020-01-31", freq="ME"):
    return pd.date_range(start, periods=n, freq=freq, tz="UTC")


class TestComputeBreadth:
    def test_fraction_of_true_columns(self):
        from ggTrader.lab.strategies.leveraged_rotation import compute_breadth

        entries = pd.DataFrame(
            {
                "A": [True, False, True],
                "B": [True, False, False],
                "C": [False, False, False],
                "D": [True, False, False],
            },
            index=_idx(3),
        )
        breadth = compute_breadth(entries)
        assert list(breadth) == [0.75, 0.0, 0.25]

    def test_empty_columns_returns_empty_series(self):
        from ggTrader.lab.strategies.leveraged_rotation import compute_breadth

        entries = pd.DataFrame(index=_idx(3))
        breadth = compute_breadth(entries)
        assert len(breadth) == 3
        assert breadth.isna().all() or (breadth == 0.0).all()


class TestRotatePositions:
    def test_first_state_takes_effect_immediately(self):
        from ggTrader.lab.strategies.leveraged_rotation import rotate_positions

        breadth = pd.Series([0.7], index=_idx(1))
        states = rotate_positions(breadth, upper_threshold=0.6, lower_threshold=0.4, min_hold_months=3)
        assert states.iloc[0] == "long"

    def test_min_hold_one_flips_immediately(self):
        from ggTrader.lab.strategies.leveraged_rotation import rotate_positions

        breadth = pd.Series([0.7, 0.3, 0.7], index=_idx(3))
        states = rotate_positions(breadth, upper_threshold=0.6, lower_threshold=0.4, min_hold_months=1)
        assert list(states) == ["long", "inverse", "long"]

    def test_min_hold_three_requires_confirmation(self):
        from ggTrader.lab.strategies.leveraged_rotation import rotate_positions

        # Starts long, dips to inverse-territory for only 2 straight readings,
        # then bounces back to long-territory -- the flip to inverse should
        # never actually take effect (never held 3 consecutive readings).
        breadth = pd.Series([0.7, 0.3, 0.3, 0.7, 0.7], index=_idx(5))
        states = rotate_positions(breadth, upper_threshold=0.6, lower_threshold=0.4, min_hold_months=3)
        assert list(states) == ["long", "long", "long", "long", "long"]

    def test_min_hold_three_flips_after_three_consecutive(self):
        from ggTrader.lab.strategies.leveraged_rotation import rotate_positions

        breadth = pd.Series([0.7, 0.3, 0.3, 0.3, 0.3], index=_idx(5))
        states = rotate_positions(breadth, upper_threshold=0.6, lower_threshold=0.4, min_hold_months=3)
        assert list(states) == ["long", "long", "long", "inverse", "inverse"]

    def test_between_thresholds_is_cash(self):
        from ggTrader.lab.strategies.leveraged_rotation import rotate_positions

        breadth = pd.Series([0.5], index=_idx(1))
        states = rotate_positions(breadth, upper_threshold=0.6, lower_threshold=0.4, min_hold_months=1)
        assert states.iloc[0] == "cash"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && pytest tests/lab/test_leveraged_rotation.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ggTrader.lab.strategies.leveraged_rotation'`.

- [ ] **Step 3: Implement the helpers**

Create `src/ggTrader/lab/strategies/leveraged_rotation.py`:

```python
"""Breadth-driven leveraged/inverse index rotation (weight-based, monthly).

Rotates between a universe's leveraged-long ETF, its inverse ETF, and cash,
driven by the breadth of the existing validated EnsembleSignal across that
universe's own constituent stocks -- a repurposing of a stock-picking signal
as an index-timing feature. See
docs/superpowers/specs/2026-07-14-leveraged-index-rotation-design.md.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from ggTrader.lab.strategies.ensemble import EnsembleSignal
from ggTrader.lab.strategy import LabConfig, Plan


def compute_breadth(entries: pd.DataFrame) -> pd.Series:
    """Fraction of columns with an active (True) entry signal per row.

    Fixed denominator = entries.shape[1] (the full breadth-universe size),
    not the count of symbols currently past warmup -- see spec for why.
    """
    if entries.shape[1] == 0:
        return pd.Series(0.0, index=entries.index)
    return entries.sum(axis=1) / entries.shape[1]


def rotate_positions(
    breadth: pd.Series,
    upper_threshold: float,
    lower_threshold: float,
    min_hold_months: int,
) -> pd.Series:
    """Per-date state in {"long", "inverse", "cash"} with hysteresis.

    A state change only takes effect once the new raw signal (breadth vs.
    thresholds) has held for min_hold_months consecutive dates in the input
    series. The first date's state takes effect immediately.
    """
    raw = pd.Series("cash", index=breadth.index, dtype=object)
    raw[breadth > upper_threshold] = "long"
    raw[breadth < lower_threshold] = "inverse"

    states: list[str] = []
    current: str | None = None
    streak_value: str | None = None
    streak_len = 0
    for val in raw:
        if val == streak_value:
            streak_len += 1
        else:
            streak_value = val
            streak_len = 1
        if current is None:
            current = val
        elif val != current and streak_len >= min_hold_months:
            current = val
        states.append(current)
    return pd.Series(states, index=breadth.index)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/lab/test_leveraged_rotation.py -v`
Expected: PASS (all `TestComputeBreadth`/`TestRotatePositions` cases).

- [ ] **Step 5: Commit**

```bash
git add src/ggTrader/lab/strategies/leveraged_rotation.py tests/lab/test_leveraged_rotation.py
git commit -m "feat(lab): breadth + hysteresis-rotation helpers

Pure functions: compute_breadth (fraction of a universe with an active
entry signal) and rotate_positions (long/inverse/cash state machine
with N-consecutive-reading hysteresis). First building block of the
leveraged/inverse index rotation research strategy."
```

---

### Task 2: `_LeveragedRotationBase` strategy class

**Files:**
- Modify: `src/ggTrader/lab/strategies/leveraged_rotation.py`
- Test: `tests/lab/test_leveraged_rotation.py`

**Interfaces:**
- Consumes: `compute_breadth`, `rotate_positions` (Task 1);
  `EnsembleSignal(cfg).to_targets(plans, data) -> SignalTargets` (existing,
  unmodified); `LabConfig`, `Plan` (existing).
- Produces: `_LeveragedRotationBase` with `target_kind = "weights"`, class
  attributes `PAIR_3X: tuple[str, str]` and `PAIR_2X: tuple[str, str]`
  (must be set by subclasses — Task 3), `__init__(cfg, upper_threshold=0.6,
  lower_threshold=0.4, min_hold_months=1, leverage_tier="3x")`,
  `sweep_params()`, `select(asof, data, eligible) -> Plan`,
  `to_targets(plans, data) -> pd.DataFrame`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/lab/test_leveraged_rotation.py`:

```python
def _ohlcv_from_returns(returns: pd.DataFrame) -> pd.DataFrame:
    frames = {}
    for col in returns.columns:
        close = 100.0 * (1.0 + returns[col].fillna(0.0)).cumprod()
        frames[col] = pd.DataFrame(
            {
                "open": close,
                "high": close * 1.001,
                "low": close * 0.999,
                "close": close,
                "volume": np.full(len(close), 1e6),
            },
            index=returns.index,
        )
    out = pd.concat(frames, axis=1)
    out.columns = out.columns.set_names(["symbol", "field"])
    return out


def _daily_returns(symbols, n=500, seed=0, start="2020-01-01"):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n, tz="UTC")
    data = {s: rng.normal(0.0003, 0.01, n) for s in symbols}
    return pd.DataFrame(data, index=idx)


def _concrete_cls():
    """A concrete subclass for testing the base class directly."""
    from ggTrader.lab.strategies.leveraged_rotation import _LeveragedRotationBase

    class _Concrete(_LeveragedRotationBase):
        name = "leveraged_rotation_test"
        PAIR_3X = ("LONG3X", "INV3X")
        PAIR_2X = ("LONG2X", "INV2X")

    return _Concrete


class TestLeveragedRotationBaseSelect:
    def test_select_returns_active_tier_pair_only(self):
        from ggTrader.lab.strategy import LabConfig

        strat = _concrete_cls()(LabConfig(), leverage_tier="3x")
        eligible = ["LONG3X", "INV3X", "LONG2X", "INV2X"]
        plan = strat.select(pd.Timestamp("2020-06-30", tz="UTC"), pd.DataFrame(), eligible)
        symbols = {s["symbol"] for s in plan}
        assert symbols == {"LONG3X", "INV3X"}

    def test_select_2x_tier(self):
        from ggTrader.lab.strategy import LabConfig

        strat = _concrete_cls()(LabConfig(), leverage_tier="2x")
        eligible = ["LONG3X", "INV3X", "LONG2X", "INV2X"]
        plan = strat.select(pd.Timestamp("2020-06-30", tz="UTC"), pd.DataFrame(), eligible)
        symbols = {s["symbol"] for s in plan}
        assert symbols == {"LONG2X", "INV2X"}


class TestLeveragedRotationBaseToTargets:
    def test_to_targets_shape_and_columns(self):
        from ggTrader.lab.strategy import LabConfig

        stocks = [f"S{i}" for i in range(20)]
        etfs = ["LONG3X", "INV3X", "LONG2X", "INV2X"]
        returns = _daily_returns(stocks + etfs, n=400, seed=1)
        ohlcv = _ohlcv_from_returns(returns)

        strat = _concrete_cls()(
            LabConfig(min_history_bars=60), leverage_tier="3x",
            upper_threshold=0.6, lower_threshold=0.4, min_hold_months=1,
        )
        rebalance_dates = ohlcv.index[[200, 260, 320]]
        eligible = etfs
        plans = {d: strat.select(d, ohlcv.loc[:d], eligible) for d in rebalance_dates}
        targets = strat.to_targets(plans, ohlcv)

        assert isinstance(targets, pd.DataFrame)
        assert set(targets.columns) == {"LONG3X", "INV3X"}
        assert targets.index.equals(ohlcv.index)

    def test_to_targets_never_selects_inactive_tier(self):
        from ggTrader.lab.strategy import LabConfig

        stocks = [f"S{i}" for i in range(20)]
        etfs = ["LONG3X", "INV3X", "LONG2X", "INV2X"]
        returns = _daily_returns(stocks + etfs, n=400, seed=2)
        ohlcv = _ohlcv_from_returns(returns)

        strat = _concrete_cls()(LabConfig(min_history_bars=60), leverage_tier="2x")
        rebalance_dates = ohlcv.index[[200, 260]]
        plans = {d: strat.select(d, ohlcv.loc[:d], etfs) for d in rebalance_dates}
        targets = strat.to_targets(plans, ohlcv)

        assert set(targets.columns) == {"LONG2X", "INV2X"}

    def test_empty_plans_returns_empty_frame(self):
        from ggTrader.lab.strategy import LabConfig

        strat = _concrete_cls()(LabConfig())
        targets = strat.to_targets({}, pd.DataFrame())
        assert isinstance(targets, pd.DataFrame)
        assert len(targets) == 0

    def test_sweep_params_grid(self):
        params = _concrete_cls().sweep_params()
        assert "upper_threshold" in params
        assert "lower_threshold" in params
        assert "min_hold_months" in params
        assert "leverage_tier" in params
        assert set(params["leverage_tier"]) == {"2x", "3x"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && pytest tests/lab/test_leveraged_rotation.py -v -k Base`
Expected: FAIL — `_LeveragedRotationBase` doesn't exist.

- [ ] **Step 3: Implement the base class**

Append to `src/ggTrader/lab/strategies/leveraged_rotation.py`:

```python
class _LeveragedRotationBase:
    """Rotates between a leveraged-long ETF, an inverse ETF, and cash,
    driven by monthly breadth of EnsembleSignal across the universe's own
    constituent stocks. Subclasses fix PAIR_3X/PAIR_2X (see Task 3)."""

    name: str
    target_kind = "weights"
    PAIR_3X: tuple[str, str]
    PAIR_2X: tuple[str, str]

    def __init__(
        self,
        cfg: LabConfig,
        upper_threshold: float = 0.6,
        lower_threshold: float = 0.4,
        min_hold_months: int = 1,
        leverage_tier: str = "3x",
    ) -> None:
        self.cfg = cfg
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self.min_hold_months = min_hold_months
        self.leverage_tier = leverage_tier

    @classmethod
    def sweep_params(cls) -> dict[str, list]:
        return {
            "upper_threshold": [0.55, 0.60, 0.65],
            "lower_threshold": [0.35, 0.40, 0.45],
            "min_hold_months": [1, 2, 3],
            "leverage_tier": ["2x", "3x"],
        }

    def _pair(self) -> tuple[str, str]:
        return self.PAIR_3X if self.leverage_tier == "3x" else self.PAIR_2X

    def _all_etf_tickers(self) -> set[str]:
        return set(self.PAIR_3X) | set(self.PAIR_2X)

    def select(self, asof: pd.Timestamp, data: pd.DataFrame, eligible: List[str]) -> Plan:
        long_t, inv_t = self._pair()
        return [{"symbol": s, "weight": 0.0} for s in (long_t, inv_t) if s in eligible]

    def to_targets(self, plans: Dict[pd.Timestamp, Plan], data: pd.DataFrame) -> pd.DataFrame:
        long_t, inv_t = self._pair()
        rebalance_dates = sorted(plans.keys())
        if not rebalance_dates or data.empty:
            return pd.DataFrame(columns=[long_t, inv_t])

        have = set(data.columns.get_level_values(0).unique())
        breadth_symbols = sorted(have - self._all_etf_tickers())

        ensemble = EnsembleSignal(self.cfg)
        placeholder = [{"symbol": s, "weight": 0.0} for s in breadth_symbols]
        signal_targets = ensemble.to_targets({data.index[0]: placeholder}, data)
        breadth = compute_breadth(signal_targets.entries)

        monthly_breadth = breadth.reindex(rebalance_dates)
        states = rotate_positions(
            monthly_breadth, self.upper_threshold, self.lower_threshold, self.min_hold_months
        )

        targets = pd.DataFrame(np.nan, index=data.index, columns=[long_t, inv_t])
        for asof in rebalance_dates:
            forward = data.index[data.index > asof]
            if len(forward) == 0:
                continue
            bar = forward[0]
            state = states.loc[asof]
            targets.loc[bar, long_t] = 1.0 if state == "long" else 0.0
            targets.loc[bar, inv_t] = 1.0 if state == "inverse" else 0.0
        return targets
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/lab/test_leveraged_rotation.py -v`
Expected: PASS (all of Task 1 + Task 2's tests).

- [ ] **Step 5: Commit**

```bash
git add src/ggTrader/lab/strategies/leveraged_rotation.py tests/lab/test_leveraged_rotation.py
git commit -m "feat(lab): _LeveragedRotationBase strategy class

select() records the active leverage tier's ETF pair; to_targets()
does one vectorized EnsembleSignal pass over the breadth-universe
columns of the combined data frame, computes monthly breadth at each
rebalance date, and applies hysteresis-gated rotation. Mirrors
EnsembleSignal's own select()-is-a-placeholder / to_targets()-does-the-
real-work pattern, since breadth is a full-window vectorized
computation, not a per-date pick."
```

---

### Task 3: Per-universe subclasses + registry

**Files:**
- Modify: `src/ggTrader/lab/strategies/leveraged_rotation.py`
- Modify: `src/ggTrader/lab/strategies/__init__.py`
- Test: `tests/lab/test_leveraged_rotation.py`

**Interfaces:**
- Produces: `LeveragedRotationSp500`, `LeveragedRotationNasdaq100`,
  `LeveragedRotationRussell2000` classes, each with `BREADTH_UNIVERSE: str`
  (the `lab/data.py` universe name used only by the orchestrator in Task 4,
  not by the strategy's own `select`/`to_targets`), `PAIR_3X`, `PAIR_2X` set
  per the spec's mapping table. Registered in `STRATEGY_REGISTRY` under
  `"leveraged_rotation_sp500"`, `"leveraged_rotation_nasdaq100"`,
  `"leveraged_rotation_russell2000"`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/lab/test_leveraged_rotation.py`:

```python
class TestPerUniverseSubclasses:
    def test_sp500_pairs(self):
        from ggTrader.lab.strategies.leveraged_rotation import LeveragedRotationSp500

        assert LeveragedRotationSp500.PAIR_3X == ("UPRO", "SPXU")
        assert LeveragedRotationSp500.PAIR_2X == ("SSO", "SDS")
        assert LeveragedRotationSp500.BREADTH_UNIVERSE == "sp500"
        assert LeveragedRotationSp500.name == "leveraged_rotation_sp500"

    def test_nasdaq100_pairs(self):
        from ggTrader.lab.strategies.leveraged_rotation import LeveragedRotationNasdaq100

        assert LeveragedRotationNasdaq100.PAIR_3X == ("TQQQ", "SQQQ")
        assert LeveragedRotationNasdaq100.PAIR_2X == ("QLD", "QID")
        assert LeveragedRotationNasdaq100.BREADTH_UNIVERSE == "nasdaq100"

    def test_russell2000_pairs(self):
        from ggTrader.lab.strategies.leveraged_rotation import LeveragedRotationRussell2000

        assert LeveragedRotationRussell2000.PAIR_3X == ("TNA", "TZA")
        assert LeveragedRotationRussell2000.PAIR_2X == ("UWM", "TWM")
        assert LeveragedRotationRussell2000.BREADTH_UNIVERSE == "russell2000"

    def test_bare_construction_works_without_extra_args(self):
        """wfo.py calls strategy_cls(cfg) with no extra args in some paths
        (anchor-set computation) -- every subclass must support this."""
        from ggTrader.lab.strategies.leveraged_rotation import (
            LeveragedRotationNasdaq100,
            LeveragedRotationRussell2000,
            LeveragedRotationSp500,
        )
        from ggTrader.lab.strategy import LabConfig

        for cls in (LeveragedRotationSp500, LeveragedRotationNasdaq100, LeveragedRotationRussell2000):
            strat = cls(LabConfig())
            assert strat.leverage_tier == "3x"


def test_all_three_registered():
    from ggTrader.lab.strategies import STRATEGY_REGISTRY
    from ggTrader.lab.strategies.leveraged_rotation import (
        LeveragedRotationNasdaq100,
        LeveragedRotationRussell2000,
        LeveragedRotationSp500,
    )

    assert STRATEGY_REGISTRY["leveraged_rotation_sp500"] is LeveragedRotationSp500
    assert STRATEGY_REGISTRY["leveraged_rotation_nasdaq100"] is LeveragedRotationNasdaq100
    assert STRATEGY_REGISTRY["leveraged_rotation_russell2000"] is LeveragedRotationRussell2000
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && pytest tests/lab/test_leveraged_rotation.py -v -k "Subclasses or registered"`
Expected: FAIL — subclasses don't exist yet.

- [ ] **Step 3: Implement the subclasses**

Append to `src/ggTrader/lab/strategies/leveraged_rotation.py`:

```python
class LeveragedRotationSp500(_LeveragedRotationBase):
    name = "leveraged_rotation_sp500"
    BREADTH_UNIVERSE = "sp500"
    PAIR_3X = ("UPRO", "SPXU")
    PAIR_2X = ("SSO", "SDS")


class LeveragedRotationNasdaq100(_LeveragedRotationBase):
    name = "leveraged_rotation_nasdaq100"
    BREADTH_UNIVERSE = "nasdaq100"
    PAIR_3X = ("TQQQ", "SQQQ")
    PAIR_2X = ("QLD", "QID")


class LeveragedRotationRussell2000(_LeveragedRotationBase):
    name = "leveraged_rotation_russell2000"
    BREADTH_UNIVERSE = "russell2000"
    PAIR_3X = ("TNA", "TZA")
    PAIR_2X = ("UWM", "TWM")
```

Edit `src/ggTrader/lab/strategies/__init__.py`: add the import and registry
entries.

```python
from .idio_vol import IdioVolStrategy
from .leveraged_rotation import (
    LeveragedRotationNasdaq100,
    LeveragedRotationRussell2000,
    LeveragedRotationSp500,
)
from .momentum import CrossSectionalMomentum, DualMomentum
```

```python
    "idio_vol": IdioVolStrategy,
    "leveraged_rotation_sp500": LeveragedRotationSp500,
    "leveraged_rotation_nasdaq100": LeveragedRotationNasdaq100,
    "leveraged_rotation_russell2000": LeveragedRotationRussell2000,
```

Add the three class names to `__all__` alongside `IdioVolStrategy`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/lab/test_leveraged_rotation.py -v`
Expected: PASS (all tests in this file).

Also run: `source .venv/bin/activate && pytest -q` — full suite must stay green (registry changes are shared infra).

- [ ] **Step 5: Commit**

```bash
git add src/ggTrader/lab/strategies/leveraged_rotation.py src/ggTrader/lab/strategies/__init__.py tests/lab/test_leveraged_rotation.py
git commit -m "feat(lab): per-universe leveraged-rotation subclasses + registry

LeveragedRotationSp500/Nasdaq100/Russell2000 fix their own ETF-pair
constants (UPRO/SPXU, TQQQ/SQQQ, TNA/TZA at 3x; SSO/SDS, QLD/QID,
UWM/TWM at 2x) and BREADTH_UNIVERSE, registered in STRATEGY_REGISTRY.
Three subclasses rather than one universe-parametrized class because
wfo.py constructs strategies as strategy_cls(cfg) with no extra args
in some paths -- the universe binding has to live on the class."
```

---

### Task 4: Orchestration script

**Files:**
- Create: `scripts/leveraged_rotation_research.py`
- Test: `tests/scripts/test_leveraged_rotation_research.py`

**Interfaces:**
- Consumes: `ggTrader.lab.data.STOCK_BASE_CONFIG`,
  `ggTrader.lab.data.equity_universe_between`, `ggTrader.lab.data.load_ohlcv`
  (all existing, unmodified), `ggTrader.lab.strategy.LabConfig`,
  `ggTrader.lab.sweep.build_grid`, `ggTrader.lab.wfo.WfoResult`,
  `ggTrader.lab.wfo.run_wfo` (all existing, unmodified), the three
  subclasses from Task 3.
- Produces: `UNIVERSES: dict[str, type]` (universe name -> subclass),
  `run_universe(universe: str, eval_start: str, eval_end: str, cfg:
  LabConfig) -> str` (one universe's WFO run; returns a summary line), `main()`
  (loops all three universes, prints each summary).

- [ ] **Step 1: Write the failing test**

Create `tests/scripts/test_leveraged_rotation_research.py` (create the
`tests/scripts/` directory if it doesn't exist yet):

```python
"""Tests for the leveraged-rotation research orchestration script."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestUniversesMapping:
    def test_maps_all_three_universes(self):
        from scripts.leveraged_rotation_research import UNIVERSES
        from ggTrader.lab.strategies.leveraged_rotation import (
            LeveragedRotationNasdaq100,
            LeveragedRotationRussell2000,
            LeveragedRotationSp500,
        )

        assert UNIVERSES["sp500"] is LeveragedRotationSp500
        assert UNIVERSES["nasdaq100"] is LeveragedRotationNasdaq100
        assert UNIVERSES["russell2000"] is LeveragedRotationRussell2000


class TestRunUniverse:
    @patch("scripts.leveraged_rotation_research.run_wfo")
    @patch("scripts.leveraged_rotation_research.load_ohlcv")
    @patch("scripts.leveraged_rotation_research.equity_universe_between")
    def test_calls_run_wfo_with_fixed_universe_fn(self, mock_members, mock_load, mock_run_wfo):
        import pandas as pd

        from ggTrader.lab.strategy import LabConfig
        from ggTrader.lab.wfo import WfoResult
        from scripts.leveraged_rotation_research import run_universe

        mock_members.return_value = ["AAPL", "MSFT"]
        ohlcv = MagicMock()
        ohlcv.__getitem__.return_value = MagicMock()
        mock_load.return_value = ohlcv
        mock_run_wfo.return_value = WfoResult(
            oos_equity=pd.Series(dtype=float),
            fold_results=[],
            live_params={},
            table="fake table",
        )

        result = run_universe("sp500", "2010-06-30", "2020-01-01", LabConfig())

        assert "sp500" in result
        mock_run_wfo.assert_called_once()
        call_kwargs = mock_run_wfo.call_args
        universe_fn = call_kwargs.kwargs["universe_fn"]
        # Fixed regardless of asof/past -- always all 4 ETF tickers for sp500.
        eligible = universe_fn(pd.Timestamp("2015-01-01", tz="UTC"), None)
        assert set(eligible) == {"UPRO", "SPXU", "SSO", "SDS"}

    @patch("scripts.leveraged_rotation_research.run_wfo")
    @patch("scripts.leveraged_rotation_research.load_ohlcv")
    @patch("scripts.leveraged_rotation_research.equity_universe_between")
    def test_no_valid_folds_reports_gracefully(self, mock_members, mock_load, mock_run_wfo):
        from ggTrader.lab.strategy import LabConfig
        from scripts.leveraged_rotation_research import run_universe

        mock_members.return_value = ["AAPL"]
        mock_load.return_value = MagicMock()
        mock_run_wfo.return_value = "WFO: leveraged_rotation_sp500 | no valid folds"

        result = run_universe("sp500", "2010-06-30", "2010-07-01", LabConfig())
        assert "no valid folds" in result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/scripts/test_leveraged_rotation_research.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.leveraged_rotation_research'`
(if `tests/scripts/` or `scripts/` lack `__init__.py`, also add an empty
`tests/scripts/__init__.py` so pytest can import the test module by path.)

- [ ] **Step 3: Implement**

Create `scripts/leveraged_rotation_research.py`:

```python
"""Leveraged/inverse index rotation research: WFO each universe's
breadth-driven rotation strategy against real leveraged/inverse ETF price
history. See docs/superpowers/specs/2026-07-14-leveraged-index-rotation-design.md.
"""

from __future__ import annotations

import pandas as pd

from ggTrader.lab.data import STOCK_BASE_CONFIG, equity_universe_between, load_ohlcv
from ggTrader.lab.strategies.leveraged_rotation import (
    LeveragedRotationNasdaq100,
    LeveragedRotationRussell2000,
    LeveragedRotationSp500,
)
from ggTrader.lab.strategy import LabConfig
from ggTrader.lab.sweep import build_grid
from ggTrader.lab.wfo import WfoResult, run_wfo

UNIVERSES: dict[str, type] = {
    "sp500": LeveragedRotationSp500,
    "nasdaq100": LeveragedRotationNasdaq100,
    "russell2000": LeveragedRotationRussell2000,
}


def _fixed_universe_fn(tickers: tuple[str, ...]):
    """universe_fn is called once per (asof, combo) without seeing which
    combo is active, so it can't be leverage-tier-aware -- it always
    returns the full 4-ticker union; the strategy picks its own 2 relevant
    tickers internally via self.leverage_tier."""

    def _fn(asof: pd.Timestamp, past: pd.DataFrame | None) -> list[str]:
        return list(tickers)

    return _fn


def run_universe(universe: str, eval_start: str, eval_end: str, cfg: LabConfig) -> str:
    cls = UNIVERSES[universe]
    es = pd.Timestamp(eval_start, tz="UTC")
    ee = pd.Timestamp(eval_end, tz="UTC")
    warmup_days = int(max(cfg.lookback, cfg.min_history_bars) * 1.6) + 60
    data_start = str((es - pd.Timedelta(days=warmup_days)).date())

    members = equity_universe_between(es, ee, universe=universe)
    etf_tickers = sorted(set(cls.PAIR_3X) | set(cls.PAIR_2X))
    all_symbols = sorted(set(members) | set(etf_tickers) | {"SPY"})
    ohlcv = load_ohlcv(all_symbols, data_start, eval_end, use_negative_cache=True)
    spy_close = ohlcv["SPY"]["close"].dropna()

    result = run_wfo(
        cls.name,
        cls,
        cfg,
        ohlcv,
        spy_close,
        eval_start=eval_start,
        eval_end=eval_end,
        market="equity",
        base_config=dict(STOCK_BASE_CONFIG),
        grid=build_grid(cls),
        universe_fn=_fixed_universe_fn(tuple(etf_tickers)),
    )
    if not isinstance(result, WfoResult):
        return f"{universe}: {result}"
    return f"{universe}: WFO complete\n{result.table}"


def main() -> None:
    cfg = LabConfig(min_history_bars=400)
    for universe in UNIVERSES:
        print(run_universe(universe, "2010-06-30", str(pd.Timestamp.now().date()), cfg))
        print()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/scripts/test_leveraged_rotation_research.py -v`
Expected: PASS.

Also run: `source .venv/bin/activate && pytest -q` — full suite green.

- [ ] **Step 5: Commit**

```bash
git add scripts/leveraged_rotation_research.py tests/scripts/test_leveraged_rotation_research.py
git commit -m "feat(lab): leveraged-rotation research orchestration script

Mirrors blend.py's run_blend() shape: loads combined OHLCV (universe
constituents + all 4 ETF tickers) per universe and drives run_wfo()
with a fixed universe_fn, since the generic --wfo CLI path has no hook
to inject a custom eligible-set function. main() runs all three
universes end to end."
```

---

## Post-plan: run the research and write it up (not automated, not part of this plan)

This plan ships the strategy and orchestration tooling only. Per the spec,
running the actual WFO (`python scripts/leveraged_rotation_research.py`,
likely several minutes per universe given ~15 years of daily data and a
36-combo grid per universe) and writing up the result is a separate,
deliberate step:

1. Run `python scripts/leveraged_rotation_research.py`, capturing each
   universe's `table` output (gate pass rate, per-fold winner stability,
   OOS Sharpe/CAGR/MaxDD vs. that universe's own SPY benchmark).
2. Write a research report following `docs/research/TEMPLATE`'s 6-section
   structure, covering all three universes and both leverage tiers, with a
   GO/NO-GO verdict per universe plus a combined verdict — watching
   specifically for the "winner selected in only 3/17 folds" noise tell
   that closed eight prior hypotheses in this project.
3. Add a dated `roadmap.md` entry recording the result, win or lose.
4. If any universe clears WFO, live-wiring onto the already-provisioned
   second Alpaca paper account is a separate, later plan — not automated
   here.

## Verification (after all tasks)

- Full test suite: `source .venv/bin/activate && pytest -q` — expect all
  green, including `tests/lab/test_leveraged_rotation.py` and
  `tests/scripts/test_leveraged_rotation_research.py`.
- Manual smoke test (safe, no live trading involved — pure research code):
  `source .venv/bin/activate && python scripts/leveraged_rotation_research.py`
  and confirm all three universes print a WFO table without raising.
