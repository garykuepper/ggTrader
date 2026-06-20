# Edge Research Phase 2: Vol Targeting, Signal Ensemble, Conviction Sizing

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Find a deployable equity strategy that beats SPY risk-adjusted in WFO by layering vol targeting, signal ensembles, and conviction-weighted sizing on top of the mean-reversion signals that already show edge (bb_reversion Sharpe 0.80 vs SPY 0.59).

**Architecture:** All work lives in the existing `src/ggTrader/lab/` package. New strategies implement the same `Strategy` protocol (name, target_kind, select, to_targets, sweep_params, sweep_signals). Simulation uses vectorbt `Portfolio.from_signals` with `cash_sharing=True` and `group_by`. The lab CLI (`ggt.py lab`) drives everything — `--sweep` for parameter grids, `--wfo` for walk-forward validation. The equity universe is point-in-time S&P 500 via `equity_universe_between()`, daily bars from TimescaleDB-cached yfinance.

**Tech Stack:** Python 3.11, vectorbt, pandas, numpy, pytest. Research runs via `docker compose run --rm ggtrader_live python ggt.py lab --strategy <name> --wfo`.

## Global Constraints

- All strategies must implement `sweep_params()` and `sweep_signals()` for grid sweep compatibility
- Signal strategies return `SignalTargets(entries, exits)` boolean DataFrames
- No lookahead: `select()` must be a pure function of `data <= asof`; `to_targets()` uses EMA warmup from full history but only scores the eval window
- Tests use synthetic price data, not DB — keep tests fast and deterministic
- Absolute imports from `src` (e.g., `from ggTrader.lab.strategy import ...`)
- ruff linting must pass

## Research context

**What's been proven:**
- Trend-following (EMA cross, WFO tournament): tracks SPY, no alpha
- Cross-sectional/dual momentum: market-like, no risk-adjusted edge
- Trailing stops: destructive on both trend and reversion strategies
- Fee/gate/universe tuning: exhausted, no deployable edge
- bb_reversion: Sharpe 0.80, CAGR 17%+ (beats SPY 0.59/13%) — **first real signal**
- rsi_reversion: also beats SPY in WFO

**Thesis:** Reversion signals have edge. Improve them via (1) risk overlay that scales position size by realized vol, (2) combining orthogonal signals to smooth the equity curve, (3) sizing by conviction instead of fixed 2%.

---

### Task 1: Commit vol targeting + run sweep on bb_reversion

Vol targeting code is already written (uncommitted in `simulate.py`, `sweep.py`) with 12 passing tests in `tests/lab/test_vol_target.py`. This task commits that work, then runs a parameter sweep to find optimal vol target settings for bb_reversion.

**Files:**
- Already modified: `src/ggTrader/lab/simulate.py` (compute_vol_scalar, vol targeting in simulate_signals)
- Already modified: `src/ggTrader/lab/sweep.py` (VOL_PARAMS, OVERLAY_PARAMS, split_params)
- Already created: `tests/lab/test_vol_target.py` (12 tests)

**Interfaces:**
- Consumes: `simulate_signals()` from simulate.py, `split_params()` from sweep.py
- Produces: Committed vol targeting overlay; sweep results for bb_reversion + vol_target combos

- [ ] **Step 1: Run existing vol targeting tests**

Run: `docker compose run --rm ggtrader_live python -m pytest tests/lab/test_vol_target.py -v`
Expected: All 12 tests pass.

- [ ] **Step 2: Run full lab test suite to confirm no regressions**

Run: `docker compose run --rm ggtrader_live python -m pytest tests/lab/ -v`
Expected: All 88+ tests pass.

- [ ] **Step 3: Commit vol targeting**

```bash
git add src/ggTrader/lab/simulate.py src/ggTrader/lab/sweep.py tests/lab/test_vol_target.py
git commit -m "feat(lab): add vol targeting overlay for signal strategies"
```

- [ ] **Step 4: Run bb_reversion sweep with vol targeting params**

Run:
```bash
docker compose run --rm ggtrader_live python ggt.py lab \
  --strategy bb_reversion --sweep \
  --sweep-param bb_period=10,15,20,30 \
  --sweep-param bb_std=1.5,2.0,2.5 \
  --sweep-param vol_target=0.10,0.12,0.15,0.20 \
  --sweep-param vol_lookback=20,40,60
```

Record the top 5 combos by Sharpe. Compare best vol-targeted Sharpe vs baseline bb_reversion (0.80). If vol targeting improves Sharpe, use the best vol params as defaults for subsequent tasks.

- [ ] **Step 5: Run WFO on best vol-targeted bb_reversion**

Using the best vol_target/vol_lookback from the sweep:
```bash
docker compose run --rm ggtrader_live python ggt.py lab \
  --strategy bb_reversion --wfo \
  --sweep-param vol_target=<best> \
  --sweep-param vol_lookback=<best>
```

Document: OOS Sharpe, CAGR, MaxDD vs SPY. Does vol targeting improve the WFO result?

---

### Task 2: Signal ensemble strategy (majority-vote)

Build `EnsembleSignal` — a strategy that runs bb_reversion, rsi_reversion, and ema_cross independently, then enters when at least `min_agree` of them fire on the same bar+symbol. This is the core edge-discovery task: diversified signals should smooth the equity curve.

**Files:**
- Create: `src/ggTrader/lab/strategies/ensemble.py`
- Modify: `src/ggTrader/lab/strategies/__init__.py` (register)
- Modify: `src/ggTrader/lab/cli.py` (add to CLI choices + cls_map)
- Test: `tests/lab/test_ensemble.py`

**Interfaces:**
- Consumes: `_bb_signals()`, `_rsi_signals()` from `strategies/signals.py`; `SignalTargets`, `LabConfig`, `Plan` from `strategy.py`
- Produces: `EnsembleSignal` class with name="ensemble", target_kind="signals", sweep_params, sweep_signals

- [ ] **Step 1: Write core test — ensemble enters only when 2-of-3 agree**

```python
# tests/lab/test_ensemble.py
import numpy as np
import pandas as pd

from ggTrader.lab.strategies.ensemble import EnsembleSignal
from ggTrader.lab.strategy import LabConfig


def _idx(n, start="2020-01-01"):
    return pd.date_range(start, periods=n, freq="B", tz="UTC")


def _ohlcv(n=300, n_syms=3, seed=42):
    """Synthetic OHLCV with (symbol, field) MultiIndex columns."""
    np.random.seed(seed)
    idx = _idx(n)
    frames = {}
    for i in range(n_syms):
        sym = f"S{i}"
        close = 100.0 * np.exp(np.cumsum(np.random.normal(0.0003, 0.015, n)))
        frames[sym] = pd.DataFrame(
            {
                "open": close * 0.999,
                "high": close * 1.005,
                "low": close * 0.995,
                "close": close,
                "volume": np.random.randint(1000, 10000, n).astype(float),
            },
            index=idx,
        )
    df = pd.concat(frames, axis=1)
    df.columns.names = ["symbol", "field"]
    return df


def test_ensemble_no_entry_when_fewer_than_min_agree():
    """If only 1 sub-signal fires, ensemble should NOT enter."""
    cfg = LabConfig(min_history_bars=50)
    strat = EnsembleSignal(cfg, min_agree=3)
    ohlcv = _ohlcv(n=300)
    symbols = sorted(ohlcv.columns.get_level_values(0).unique())
    close = pd.concat({s: ohlcv[s]["close"] for s in symbols}, axis=1)
    plans = {close.index[100]: [{"symbol": s, "weight": 0.0} for s in symbols]}
    targets = strat.to_targets(plans, ohlcv)
    # With min_agree=3 (all must agree), entries should be very rare or zero
    # on random synthetic data where signals are uncorrelated
    assert targets.entries.sum().sum() <= targets.entries.shape[0] * len(symbols) * 0.01
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker compose run --rm ggtrader_live python -m pytest tests/lab/test_ensemble.py::test_ensemble_no_entry_when_fewer_than_min_agree -v`
Expected: FAIL — `EnsembleSignal` does not exist yet.

- [ ] **Step 3: Implement EnsembleSignal**

```python
# src/ggTrader/lab/strategies/ensemble.py
"""Signal ensemble: enter when N-of-M sub-signals agree on the same bar+symbol."""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from ggTrader.lab.strategies.signals import _bb_signals, _rsi_signals
from ggTrader.lab.strategy import LabConfig, Plan, SignalTargets


def _ema_signals(
    close: pd.DataFrame, ema_fast: int, ema_slow: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """EMA crossover entry/exit signals."""
    ema_f = close.ewm(span=ema_fast, adjust=False).mean()
    ema_s = close.ewm(span=ema_slow, adjust=False).mean()
    entries = ((ema_f > ema_s) & (ema_f.shift(1) <= ema_s.shift(1))).fillna(False)
    exits = ((ema_f < ema_s) & (ema_f.shift(1) >= ema_s.shift(1))).fillna(False)
    return entries.astype(bool), exits.astype(bool)


class EnsembleSignal:
    """Majority-vote ensemble: enter when >= min_agree sub-signals fire together.

    Sub-signals: bb_reversion, rsi_reversion, ema_cross.
    Exit: when >= min_agree sub-signals fire an exit.
    """

    name = "ensemble"
    target_kind = "signals"

    def __init__(
        self,
        cfg: LabConfig,
        min_agree: int = 2,
        bb_period: int = 20,
        bb_std: float = 2.0,
        rsi_period: int = 14,
        rsi_oversold: int = 30,
        rsi_exit: int = 50,
        ema_fast: int = 20,
        ema_slow: int = 50,
    ) -> None:
        self.cfg = cfg
        self.min_agree = min_agree
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_exit = rsi_exit
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow

    @classmethod
    def sweep_params(cls) -> dict[str, list]:
        return {
            "min_agree": [2, 3],
            "bb_period": [15, 20],
            "bb_std": [2.0, 2.5],
            "rsi_period": [7, 14],
            "rsi_oversold": [25, 30],
            "ema_fast": [10, 20],
            "ema_slow": [50, 100],
        }

    def select(self, asof: pd.Timestamp, data: pd.DataFrame, eligible: List[str]) -> Plan:
        data = data.loc[:asof]
        have = set(data.columns.get_level_values(0).unique())
        return [
            {"symbol": s, "weight": 0.0}
            for s in eligible
            if s in have and len(data[s]["close"].dropna()) >= self.cfg.min_history_bars
        ]

    def _generate_signals(self, close: pd.DataFrame) -> SignalTargets:
        """Run all 3 sub-signals, sum entry/exit votes, threshold at min_agree."""
        bb_ent, bb_ext = _bb_signals(close, self.bb_period, self.bb_std)
        rsi_ent, rsi_ext = _rsi_signals(
            close, self.rsi_period, self.rsi_oversold, self.rsi_exit
        )
        ema_ent, ema_ext = _ema_signals(close, self.ema_fast, self.ema_slow)

        entry_votes = bb_ent.astype(int) + rsi_ent.astype(int) + ema_ent.astype(int)
        exit_votes = bb_ext.astype(int) + rsi_ext.astype(int) + ema_ext.astype(int)

        entries = (entry_votes >= self.min_agree).astype(bool)
        exits = (exit_votes >= self.min_agree).astype(bool)
        return SignalTargets(entries=entries, exits=exits)

    def to_targets(self, plans: Dict[pd.Timestamp, Plan], data: pd.DataFrame) -> SignalTargets:
        symbols = sorted({s["symbol"] for plan in plans.values() for s in plan})
        close = pd.concat(
            {s: data[s]["close"] for s in symbols if s in data.columns.get_level_values(0)},
            axis=1,
        )
        return self._generate_signals(close)

    def sweep_signals(
        self,
        combos: list[dict],
        symbols: list[str],
        data: pd.DataFrame,
    ) -> dict[str, SignalTargets]:
        from ggTrader.lab.sweep import combo_name

        close = pd.concat(
            {s: data[s]["close"] for s in symbols if s in data.columns.get_level_values(0)},
            axis=1,
        )
        result: dict[str, SignalTargets] = {}
        for combo in combos:
            strat = EnsembleSignal(
                self.cfg,
                min_agree=int(combo.get("min_agree", self.min_agree)),
                bb_period=int(combo.get("bb_period", self.bb_period)),
                bb_std=float(combo.get("bb_std", self.bb_std)),
                rsi_period=int(combo.get("rsi_period", self.rsi_period)),
                rsi_oversold=int(combo.get("rsi_oversold", self.rsi_oversold)),
                rsi_exit=int(combo.get("rsi_exit", self.rsi_exit)),
                ema_fast=int(combo.get("ema_fast", self.ema_fast)),
                ema_slow=int(combo.get("ema_slow", self.ema_slow)),
            )
            targets = strat._generate_signals(close)
            key = combo_name(self.name, combo)
            result[key] = targets
        return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker compose run --rm ggtrader_live python -m pytest tests/lab/test_ensemble.py::test_ensemble_no_entry_when_fewer_than_min_agree -v`
Expected: PASS

- [ ] **Step 5: Write additional tests**

Add to `tests/lab/test_ensemble.py`:

```python
def test_ensemble_enters_when_all_agree():
    """With min_agree=1, any single signal triggers entry."""
    cfg = LabConfig(min_history_bars=50)
    strat = EnsembleSignal(cfg, min_agree=1)
    ohlcv = _ohlcv(n=300)
    symbols = sorted(ohlcv.columns.get_level_values(0).unique())
    close = pd.concat({s: ohlcv[s]["close"] for s in symbols}, axis=1)
    plans = {close.index[100]: [{"symbol": s, "weight": 0.0} for s in symbols]}
    targets = strat.to_targets(plans, ohlcv)
    # min_agree=1 means any sub-signal fires -> should have more entries
    assert targets.entries.sum().sum() > 0


def test_ensemble_sweep_params_returns_dict():
    assert "min_agree" in EnsembleSignal.sweep_params()
    assert "bb_period" in EnsembleSignal.sweep_params()


def test_ensemble_sweep_signals_keys_match_combos():
    from ggTrader.lab.sweep import combo_name

    cfg = LabConfig(min_history_bars=50)
    strat = EnsembleSignal(cfg)
    ohlcv = _ohlcv(n=200)
    symbols = sorted(ohlcv.columns.get_level_values(0).unique())
    combos = [{"min_agree": 2, "bb_period": 20, "bb_std": 2.0, "rsi_period": 14,
               "rsi_oversold": 30, "ema_fast": 20, "ema_slow": 50}]
    result = strat.sweep_signals(combos, symbols, ohlcv)
    expected_key = combo_name("ensemble", combos[0])
    assert expected_key in result
    assert hasattr(result[expected_key], "entries")
    assert hasattr(result[expected_key], "exits")


def test_ensemble_select_respects_asof():
    """select() must not look at data past asof."""
    cfg = LabConfig(min_history_bars=50)
    strat = EnsembleSignal(cfg)
    ohlcv = _ohlcv(n=300)
    symbols = sorted(ohlcv.columns.get_level_values(0).unique())
    mid = ohlcv.index[150]
    plan_full = strat.select(mid, ohlcv.loc[:mid], symbols)
    plan_trunc = strat.select(mid, ohlcv, symbols)
    assert len(plan_full) == len(plan_trunc)


def test_ensemble_min_agree_2_fewer_entries_than_1():
    """Higher min_agree -> fewer entries (more filtering)."""
    cfg = LabConfig(min_history_bars=50)
    ohlcv = _ohlcv(n=300, seed=123)
    symbols = sorted(ohlcv.columns.get_level_values(0).unique())
    close = pd.concat({s: ohlcv[s]["close"] for s in symbols}, axis=1)

    strat1 = EnsembleSignal(cfg, min_agree=1)
    strat2 = EnsembleSignal(cfg, min_agree=2)
    plans = {close.index[100]: [{"symbol": s, "weight": 0.0} for s in symbols]}

    t1 = strat1.to_targets(plans, ohlcv)
    t2 = strat2.to_targets(plans, ohlcv)
    assert t1.entries.sum().sum() >= t2.entries.sum().sum()
```

- [ ] **Step 6: Run all ensemble tests**

Run: `docker compose run --rm ggtrader_live python -m pytest tests/lab/test_ensemble.py -v`
Expected: All pass.

- [ ] **Step 7: Register ensemble in CLI**

Modify `src/ggTrader/lab/strategies/__init__.py` — if empty, add the import.

Modify `src/ggTrader/lab/cli.py`:
- Add `"ensemble"` to the `choices` tuple in `build_arg_parser()`
- Add import: `from ggTrader.lab.strategies.ensemble import EnsembleSignal`
- Add to `cls_map` in both the `--sweep` and `--wfo` blocks: `"ensemble": EnsembleSignal`
- Add `"ensemble"` to `SIGNAL_STRATEGY_NAMES` imports or handle it in the signal registry

Also modify `src/ggTrader/lab/strategies/signals.py`:
- Add `EnsembleSignal` to `_SIGNAL_REGISTRY` and update `SIGNAL_STRATEGY_NAMES`, OR
- Keep ensemble in its own module and update `cli.py` to check both registries

Preferred approach: add `"ensemble"` to the `_SIGNAL_REGISTRY` dict in `signals.py` by importing from `ensemble.py`. This keeps the CLI code simple (it already reads from `SIGNAL_STRATEGY_NAMES`).

```python
# At the bottom of src/ggTrader/lab/strategies/signals.py, update:
from ggTrader.lab.strategies.ensemble import EnsembleSignal

_SIGNAL_REGISTRY = {
    "ema_cross": EmaCrossSignal,
    "wfo_tournament": WfoTournamentSignal,
    "bb_reversion": BollingerReversionSignal,
    "rsi_reversion": RsiReversionSignal,
    "ensemble": EnsembleSignal,
}
```

Then update `cli.py` cls_map in both `--sweep` and `--wfo` blocks to include:
```python
"ensemble": EnsembleSignal,
```

- [ ] **Step 8: Run full lab test suite**

Run: `docker compose run --rm ggtrader_live python -m pytest tests/lab/ -v`
Expected: All tests pass (88 existing + new ensemble tests).

- [ ] **Step 9: Commit**

```bash
git add src/ggTrader/lab/strategies/ensemble.py tests/lab/test_ensemble.py \
  src/ggTrader/lab/strategies/signals.py src/ggTrader/lab/cli.py
git commit -m "feat(lab): add signal ensemble strategy (majority-vote bb+rsi+ema)"
```

- [ ] **Step 10: Run ensemble sweep**

```bash
docker compose run --rm ggtrader_live python ggt.py lab \
  --strategy ensemble --sweep \
  --sweep-param min_agree=2,3 \
  --sweep-param bb_period=15,20 \
  --sweep-param bb_std=2.0,2.5 \
  --sweep-param rsi_period=7,14 \
  --sweep-param rsi_oversold=25,30 \
  --sweep-param ema_fast=10,20 \
  --sweep-param ema_slow=50,100
```

Record top combos. Compare Sharpe vs standalone bb_reversion (0.80).

- [ ] **Step 11: Run WFO on best ensemble params**

```bash
docker compose run --rm ggtrader_live python ggt.py lab \
  --strategy ensemble --wfo \
  --sweep-param min_agree=<best> \
  --sweep-param bb_period=<best> \
  --sweep-param bb_std=<best> \
  --sweep-param rsi_period=<best> \
  --sweep-param rsi_oversold=<best> \
  --sweep-param ema_fast=<best> \
  --sweep-param ema_slow=<best>
```

Document OOS Sharpe, CAGR, MaxDD vs SPY. Compare vs bb_reversion WFO results.

---

### Task 3: Ensemble + vol targeting combined WFO

Combine the best ensemble params with vol targeting and run WFO. This is the full-stack test: does the combination beat both individual components?

**Files:**
- No new code — uses existing vol targeting overlay + ensemble strategy

**Interfaces:**
- Consumes: `EnsembleSignal` from Task 2, vol targeting overlay from Task 1
- Produces: WFO results comparing ensemble alone vs ensemble + vol targeting

- [ ] **Step 1: Run ensemble WFO with vol targeting**

```bash
docker compose run --rm ggtrader_live python ggt.py lab \
  --strategy ensemble --wfo \
  --sweep-param min_agree=<best> \
  --sweep-param bb_period=<best> \
  --sweep-param bb_std=<best> \
  --sweep-param rsi_period=<best> \
  --sweep-param rsi_oversold=<best> \
  --sweep-param ema_fast=<best> \
  --sweep-param ema_slow=<best> \
  --sweep-param vol_target=<best_from_task1> \
  --sweep-param vol_lookback=<best_from_task1>
```

- [ ] **Step 2: Document results**

Create a results summary comparing all variants:
| Variant | OOS Sharpe | CAGR | MaxDD | vs SPY |
|---------|-----------|------|-------|--------|
| bb_reversion baseline | 0.80 | 17%+ | ? | beats |
| bb_reversion + vol target | ? | ? | ? | ? |
| ensemble | ? | ? | ? | ? |
| ensemble + vol target | ? | ? | ? | ? |
| SPY | 0.59 | 13% | ? | — |

---

### Task 4: Conviction-weighted position sizing

Replace fixed 2% position sizing with signal-strength-weighted sizing. For bb_reversion, size proportional to how far price is below the lower band. For rsi_reversion, size proportional to how far RSI is below the threshold. This requires extending `simulate_signals` to accept per-bar, per-symbol size matrices computed from indicator values.

**Files:**
- Create: `src/ggTrader/lab/strategies/conviction.py`
- Modify: `src/ggTrader/lab/strategies/signals.py` (register)
- Modify: `src/ggTrader/lab/cli.py` (add to choices)
- Modify: `src/ggTrader/lab/simulate.py` (accept size from SignalTargets)
- Modify: `src/ggTrader/lab/strategy.py` (extend SignalTargets)
- Test: `tests/lab/test_conviction.py`

**Interfaces:**
- Consumes: `_bb_signals()`, `_rsi_signals()` from signals.py, `simulate_signals()` from simulate.py
- Produces: `ConvictionBBSignal` class that returns SignalTargets with an optional `sizes` DataFrame

**Design decision:** Extend `SignalTargets` to optionally carry a `sizes` DataFrame (time x symbol, float). When present, `simulate_signals` uses it instead of the fixed `SIGNAL_POSITION_SIZE`. This keeps backward compatibility — existing strategies that return `SignalTargets(entries, exits)` continue to work because NamedTuple fields are positional and the new field has a default.

- [ ] **Step 1: Write failing test for conviction sizing**

```python
# tests/lab/test_conviction.py
import numpy as np
import pandas as pd

from ggTrader.lab.strategies.conviction import ConvictionBBSignal
from ggTrader.lab.strategy import LabConfig


def _idx(n, start="2020-01-01"):
    return pd.date_range(start, periods=n, freq="B", tz="UTC")


def _ohlcv(n=300, n_syms=3, seed=42):
    np.random.seed(seed)
    idx = _idx(n)
    frames = {}
    for i in range(n_syms):
        sym = f"S{i}"
        close = 100.0 * np.exp(np.cumsum(np.random.normal(0.0003, 0.015, n)))
        frames[sym] = pd.DataFrame(
            {
                "open": close * 0.999,
                "high": close * 1.005,
                "low": close * 0.995,
                "close": close,
                "volume": np.random.randint(1000, 10000, n).astype(float),
            },
            index=idx,
        )
    df = pd.concat(frames, axis=1)
    df.columns.names = ["symbol", "field"]
    return df


def test_conviction_bb_sizes_vary_with_depth():
    """Deeper oversold = larger position size."""
    cfg = LabConfig(min_history_bars=50)
    strat = ConvictionBBSignal(cfg)
    ohlcv = _ohlcv(n=300)
    symbols = sorted(ohlcv.columns.get_level_values(0).unique())
    close = pd.concat({s: ohlcv[s]["close"] for s in symbols}, axis=1)
    plans = {close.index[100]: [{"symbol": s, "weight": 0.0} for s in symbols]}
    targets = strat.to_targets(plans, ohlcv)
    assert targets.sizes is not None
    # Where entries fire, sizes should be > 0 and variable (not all identical)
    entry_sizes = targets.sizes[targets.entries].dropna()
    if len(entry_sizes) > 1:
        assert entry_sizes.std() > 0, "Conviction sizes should vary"


def test_conviction_bb_sizes_bounded():
    """Sizes must be within [min_size, max_size] range."""
    cfg = LabConfig(min_history_bars=50)
    strat = ConvictionBBSignal(cfg, min_size=0.01, max_size=0.05)
    ohlcv = _ohlcv(n=300)
    symbols = sorted(ohlcv.columns.get_level_values(0).unique())
    close = pd.concat({s: ohlcv[s]["close"] for s in symbols}, axis=1)
    plans = {close.index[100]: [{"symbol": s, "weight": 0.0} for s in symbols]}
    targets = strat.to_targets(plans, ohlcv)
    assert targets.sizes is not None
    valid = targets.sizes[targets.entries].dropna()
    if len(valid) > 0:
        assert valid.min() >= 0.01 - 1e-10
        assert valid.max() <= 0.05 + 1e-10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker compose run --rm ggtrader_live python -m pytest tests/lab/test_conviction.py -v`
Expected: FAIL — `ConvictionBBSignal` does not exist.

- [ ] **Step 3: Extend SignalTargets to support optional sizes**

Modify `src/ggTrader/lab/strategy.py`:

```python
class SignalTargets(NamedTuple):
    """Return type for signal-based strategies' to_targets method."""

    entries: pd.DataFrame   # (time x symbol) boolean
    exits: pd.DataFrame     # (time x symbol) boolean
    sizes: pd.DataFrame | None = None  # (time x symbol) float, optional per-bar sizing
```

- [ ] **Step 4: Update simulate_signals to use per-bar sizes when provided**

In `src/ggTrader/lab/simulate.py`, in `simulate_signals()`, after the vol targeting block, add:

```python
    # --- Per-bar conviction sizing (overrides vol targeting if present) ---
    for name in names:
        st = targets_by_strategy[name]
        if hasattr(st, "sizes") and st.sizes is not None:
            cols = pd.MultiIndex.from_product(
                [[name], st.sizes.columns], names=["strategy", "symbol"]
            )
            conviction_sizes = st.sizes.set_axis(cols, axis=1)
            if isinstance(size_param, pd.DataFrame):
                size_param.update(conviction_sizes.reindex(
                    index=size_param.index, columns=conviction_sizes.columns
                ).fillna(size_param))
            else:
                # Replace flat scalar with DataFrame for this strategy's columns
                if not isinstance(size_param, pd.DataFrame):
                    size_param = pd.DataFrame(
                        base_size, index=close.index, columns=close.columns
                    )
                size_param.update(conviction_sizes.reindex(
                    index=size_param.index, columns=conviction_sizes.columns
                ))
```

- [ ] **Step 5: Implement ConvictionBBSignal**

```python
# src/ggTrader/lab/strategies/conviction.py
"""Conviction-weighted signal strategies: size proportional to signal strength."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ggTrader.lab.strategies.signals import _bb_signals
from ggTrader.lab.strategy import LabConfig, Plan, SignalTargets


class ConvictionBBSignal:
    """Bollinger Band reversion with conviction-weighted sizing.

    Position size scales with how far price is below the lower band:
    deeper oversold = larger position (up to max_size).
    """

    name = "conviction_bb"
    target_kind = "signals"

    def __init__(
        self,
        cfg: LabConfig,
        bb_period: int = 20,
        bb_std: float = 2.0,
        min_size: float = 0.01,
        max_size: float = 0.05,
    ) -> None:
        self.cfg = cfg
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.min_size = min_size
        self.max_size = max_size

    @classmethod
    def sweep_params(cls) -> dict[str, list]:
        return {
            "bb_period": [10, 15, 20, 30],
            "bb_std": [1.5, 2.0, 2.5],
            "min_size": [0.01],
            "max_size": [0.03, 0.05, 0.08],
        }

    def select(self, asof: pd.Timestamp, data: pd.DataFrame, eligible: List[str]) -> Plan:
        data = data.loc[:asof]
        have = set(data.columns.get_level_values(0).unique())
        return [
            {"symbol": s, "weight": 0.0}
            for s in eligible
            if s in have and len(data[s]["close"].dropna()) >= self.cfg.min_history_bars
        ]

    def _compute_conviction_sizes(
        self, close: pd.DataFrame, entries: pd.DataFrame
    ) -> pd.DataFrame:
        """Size = linear interpolation based on distance below lower band."""
        sma = close.rolling(window=self.bb_period, min_periods=self.bb_period).mean()
        rolling_std = close.rolling(window=self.bb_period, min_periods=self.bb_period).std()
        lower = sma - self.bb_std * rolling_std
        # depth = how far below the lower band (0 = at band, 1 = one band-width below)
        band_width = self.bb_std * rolling_std
        depth = ((lower - close) / band_width.replace(0, np.nan)).clip(lower=0.0, upper=1.0)
        sizes = self.min_size + depth * (self.max_size - self.min_size)
        # Only set sizes where entries fire; NaN elsewhere
        sizes = sizes.where(entries, np.nan)
        return sizes

    def to_targets(self, plans: Dict[pd.Timestamp, Plan], data: pd.DataFrame) -> SignalTargets:
        symbols = sorted({s["symbol"] for plan in plans.values() for s in plan})
        close = pd.concat(
            {s: data[s]["close"] for s in symbols if s in data.columns.get_level_values(0)},
            axis=1,
        )
        entries, exits = _bb_signals(close, self.bb_period, self.bb_std)
        sizes = self._compute_conviction_sizes(close, entries)
        return SignalTargets(entries=entries, exits=exits, sizes=sizes)

    def sweep_signals(
        self,
        combos: list[dict],
        symbols: list[str],
        data: pd.DataFrame,
    ) -> dict[str, SignalTargets]:
        from ggTrader.lab.sweep import combo_name

        close = pd.concat(
            {s: data[s]["close"] for s in symbols if s in data.columns.get_level_values(0)},
            axis=1,
        )
        result: dict[str, SignalTargets] = {}
        for combo in combos:
            period = int(combo["bb_period"])
            std = float(combo["bb_std"])
            min_s = float(combo.get("min_size", self.min_size))
            max_s = float(combo.get("max_size", self.max_size))
            entries, exits = _bb_signals(close, period, std)
            strat = ConvictionBBSignal(
                self.cfg, bb_period=period, bb_std=std, min_size=min_s, max_size=max_s
            )
            sizes = strat._compute_conviction_sizes(close, entries)
            key = combo_name(self.name, combo)
            result[key] = SignalTargets(entries=entries, exits=exits, sizes=sizes)
        return result
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `docker compose run --rm ggtrader_live python -m pytest tests/lab/test_conviction.py -v`
Expected: All pass.

- [ ] **Step 7: Register in CLI and run full test suite**

Add `"conviction_bb": ConvictionBBSignal` to `_SIGNAL_REGISTRY` in signals.py (import from conviction.py). Update cli.py cls_map.

Run: `docker compose run --rm ggtrader_live python -m pytest tests/lab/ -v`
Expected: All pass.

- [ ] **Step 8: Commit**

```bash
git add src/ggTrader/lab/strategy.py src/ggTrader/lab/simulate.py \
  src/ggTrader/lab/strategies/conviction.py src/ggTrader/lab/strategies/signals.py \
  src/ggTrader/lab/cli.py tests/lab/test_conviction.py
git commit -m "feat(lab): add conviction-weighted BB sizing strategy"
```

- [ ] **Step 9: Run conviction_bb sweep and WFO**

```bash
docker compose run --rm ggtrader_live python ggt.py lab \
  --strategy conviction_bb --sweep \
  --sweep-param bb_period=10,15,20,30 \
  --sweep-param bb_std=1.5,2.0,2.5 \
  --sweep-param max_size=0.03,0.05,0.08

docker compose run --rm ggtrader_live python ggt.py lab \
  --strategy conviction_bb --wfo \
  --sweep-param bb_period=<best> \
  --sweep-param bb_std=<best> \
  --sweep-param max_size=<best>
```

Compare vs fixed-size bb_reversion. Does conviction sizing improve Sharpe?

---

### Task 5: Results assessment and next direction

After Tasks 1-4, assess which combination produces the best WFO results and decide the next research direction.

- [ ] **Step 1: Compile results table**

| Strategy | OOS Sharpe | CAGR % | MaxDD % | WFE avg | Verdict |
|----------|-----------|--------|---------|---------|---------|
| bb_reversion (baseline) | 0.80 | 17+ | ? | ? | BEAT SPY |
| bb_reversion + vol target | ? | ? | ? | ? | ? |
| ensemble (2-of-3) | ? | ? | ? | ? | ? |
| ensemble + vol target | ? | ? | ? | ? | ? |
| conviction_bb | ? | ? | ? | ? | ? |
| SPY | 0.59 | 13 | ? | ? | — |

- [ ] **Step 2: Decision gate**

If any strategy has OOS Sharpe > 0.90 and CAGR > 15% with WFE > 0.50:
→ Consider it a deployment candidate. Next step: expanded reversion signals (MACD divergence, volume-confirmed) to diversify the ensemble further.

If all strategies are in the 0.70-0.90 Sharpe range:
→ The improvement is incremental. Next step: try more aggressive signal diversification — add MACD divergence and volume-confirmed reversion to the ensemble.

If nothing beats the bb_reversion baseline:
→ The overlays don't help. Next step: focus entirely on new signals (multi-timeframe, volume-confirmed) rather than risk overlays.

- [ ] **Step 3: Update roadmap and memory with findings**

Update `docs/roadmap.md` §2d status markers based on results. Save a project memory with the key findings.
