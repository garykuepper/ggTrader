# Overnight Gap Reversion Strategy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Candidate A from `docs/2026-07-06_strategy_recommendations.md` — an
overnight-gap reversion signal strategy (`overnight_gap`) — as a new lab strategy that
plugs into the existing signal-strategy registry, WFO harness, and NDH/DSR gates with
zero changes to `simulate.py`, `wfo.py`, or `gates.py`.

**Architecture:** Follows the exact `RsiReversionSignal`/`BollingerReversionSignal`
pattern in `src/ggTrader/lab/strategies/signals.py`: a pure-pandas indicator function in
`indicators.py` that returns `(entries, exits)` boolean DataFrames, wrapped in a
`Strategy`-protocol class with `select()`/`to_targets()`/`sweep_signals()`/
`sweep_params()`, registered in `STRATEGY_REGISTRY`. Entries/exits fire on the bar the
gap Z-score crosses a threshold and are filled at that bar's **close** — the same
same-bar-close-fill convention every existing voter uses — so no new price-plumbing is
needed in `simulate_signals`/`Portfolio.from_signals`.

**Tech Stack:** Python, pandas, pytest, vectorbt 0.28.5 (via the existing
`simulate_signals`/`Portfolio.from_signals` path — untouched by this plan).

## Global Constraints

- Follow `ggTrader/agents.md` coding standards: strict ruff linting, vectorization-first
  (no per-row Python loops over time), absolute imports from `src`.
- All new pandas boolean signal frames must be `.fillna(False).astype(bool)` before
  return, matching every existing indicator function in `indicators.py`.
- No changes to `simulate.py`, `wfo.py`, or `gates.py` — this strategy must be provably
  a drop-in addition to the existing signal-strategy registry, reusing the same
  close-fill convention as every other voter (per the Task right-sizing rationale
  above).
- Run `pytest tests/lab/ -q` and `ruff check src/ggTrader/lab/` before every commit.

---

## File Structure

- **Modify `src/ggTrader/lab/strategies/indicators.py`**: add `extract_open()` (mirrors
  `extract_close()`/`extract_volume()`) and `overnight_gap_signals()` (mirrors
  `rsi_signals()` — rolling Z-score crossing thresholds instead of RSI crossing
  thresholds).
- **Modify `src/ggTrader/lab/strategies/signals.py`**: add `OvernightGapReversionSignal`
  class, following `RsiReversionSignal`'s structure exactly (constructor params,
  `sweep_params`, `select`, `to_targets`, `sweep_signals`).
- **Modify `src/ggTrader/lab/strategies/__init__.py`**: import and register
  `OvernightGapReversionSignal` in `STRATEGY_REGISTRY` and `__all__`. No change needed
  in `registry.py` or `cli.py` — both derive strategy names/choices from
  `STRATEGY_REGISTRY` automatically (`cli.py:49` builds `--strategy` choices from
  `STRATEGY_NAMES + SIGNAL_STRATEGY_NAMES`, which is sourced from the registry).
- **Create `tests/lab/test_overnight_gap_signals.py`**: mirrors
  `tests/lab/test_reversion_signals.py` — indicator-function tests, strategy-class
  tests, registry/CLI wiring tests.

---

### Task 1: `extract_open` + `overnight_gap_signals` indicator functions

**Files:**
- Modify: `src/ggTrader/lab/strategies/indicators.py`
- Test: `tests/lab/test_overnight_gap_signals.py` (new file)

**Interfaces:**
- Produces: `extract_open(data: pd.DataFrame, symbols: list[str]) -> pd.DataFrame`
- Produces: `overnight_gap_signals(close: pd.DataFrame, open_: pd.DataFrame, gap_lookback: int, gap_z_entry: float, gap_z_exit: float) -> tuple[pd.DataFrame, pd.DataFrame]`

- [ ] **Step 1: Write the failing tests**

```python
# tests/lab/test_overnight_gap_signals.py
"""Tests for overnight-gap reversion indicator functions and signal class."""

import numpy as np
import pandas as pd

from ggTrader.lab.strategies.indicators import extract_open, overnight_gap_signals


def _idx(n, start="2020-01-01"):
    return pd.date_range(start, periods=n, freq="B", tz="UTC")


def _ohlcv(symbols, n=300, seed=42):
    rng = np.random.default_rng(seed)
    idx = _idx(n)
    frames = {}
    for i, s in enumerate(symbols):
        close = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n)))
        open_ = close + rng.normal(0, 0.05, n)  # tiny day-to-day gaps by default
        frames[s] = pd.DataFrame(
            {
                "open": open_,
                "high": np.maximum(open_, close) * 1.005,
                "low": np.minimum(open_, close) * 0.995,
                "close": close,
                "volume": np.full(n, 1e6),
            },
            index=idx,
        )
    out = pd.concat(frames, axis=1)
    out.columns = out.columns.set_names(["symbol", "field"])
    return out


class TestExtractOpen:
    def test_shape_matches_close(self):
        ohlcv = _ohlcv(["A", "B"], n=100)
        opens = extract_open(ohlcv, ["A", "B"])
        assert opens.shape == (100, 2)
        assert list(opens.columns) == ["A", "B"]

    def test_missing_symbol_skipped(self):
        ohlcv = _ohlcv(["A"], n=50)
        opens = extract_open(ohlcv, ["A", "NOSUCH"])
        assert "A" in opens.columns
        assert "NOSUCH" not in opens.columns


class TestOvernightGapSignals:
    def test_output_shape_and_dtype(self):
        ohlcv = _ohlcv(["A"], n=200)
        close = ohlcv["A"]["close"].to_frame("A")
        open_ = ohlcv["A"]["open"].to_frame("A")
        entries, exits = overnight_gap_signals(close, open_, 20, -1.5, -0.5)
        assert entries.shape == close.shape
        assert exits.shape == close.shape
        assert entries.dtypes.eq(bool).all()
        assert exits.dtypes.eq(bool).all()

    def test_entry_on_extreme_gap_down(self):
        """A large overnight gap-down should trigger an entry that same bar."""
        idx = _idx(100)
        rng = np.random.default_rng(1)
        close_vals = np.full(100, 100.0)
        open_vals = close_vals + rng.normal(0, 0.05, 100)  # tiny gaps most days
        open_vals[50] = close_vals[49] * 0.85  # 15% overnight gap down on day 50
        close = pd.DataFrame({"A": close_vals}, index=idx)
        open_ = pd.DataFrame({"A": open_vals}, index=idx)
        entries, _ = overnight_gap_signals(close, open_, gap_lookback=20, gap_z_entry=-1.5, gap_z_exit=-0.5)
        assert entries["A"].iloc[50], "Should enter on extreme overnight gap-down"

    def test_no_entries_in_warmup(self):
        """No signals before gap_lookback bars of history exist."""
        idx = _idx(50)
        close = pd.DataFrame({"A": np.full(50, 100.0)}, index=idx)
        open_ = pd.DataFrame({"A": np.full(50, 100.0)}, index=idx)
        entries, _ = overnight_gap_signals(close, open_, gap_lookback=20, gap_z_entry=-1.5, gap_z_exit=-0.5)
        assert not entries["A"].iloc[:20].any()

    def test_stricter_threshold_fewer_entries(self):
        """A more extreme (more negative) gap_z_entry should produce fewer entries."""
        idx = _idx(300)
        rng = np.random.default_rng(7)
        close_vals = np.full(300, 100.0)
        open_vals = close_vals + rng.normal(0, 1.0, 300)
        close = pd.DataFrame({"A": close_vals}, index=idx)
        open_ = pd.DataFrame({"A": open_vals}, index=idx)
        entries_loose, _ = overnight_gap_signals(close, open_, 20, -1.0, -0.5)
        entries_strict, _ = overnight_gap_signals(close, open_, 20, -2.5, -0.5)
        assert entries_loose["A"].sum() >= entries_strict["A"].sum()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/lab/test_overnight_gap_signals.py -v`
Expected: FAIL with `ImportError: cannot import name 'extract_open'` (or
`'overnight_gap_signals'`) from `ggTrader.lab.strategies.indicators`.

- [ ] **Step 3: Implement the indicator functions**

Add to `src/ggTrader/lab/strategies/indicators.py` (near `extract_volume`, after
`extract_close`):

```python
def extract_open(data: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
    """Extract a (time x symbol) open-price DataFrame from multi-level OHLCV data."""
    have = set(data.columns.get_level_values(0))
    return pd.concat({s: data[s]["open"] for s in symbols if s in have}, axis=1)
```

Add a new indicator function (place it near `rsi_signals`, after `bb_signals`):

```python
def overnight_gap_signals(
    close: pd.DataFrame,
    open_: pd.DataFrame,
    gap_lookback: int,
    gap_z_entry: float,
    gap_z_exit: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Vectorized overnight-gap reversion entry/exit signals.

    gap_t = (open_t - close_{t-1}) / close_{t-1} — known as soon as open_t prints,
    so it is causal at the close of bar t (the fill bar). Entry fires when the
    rolling Z-score of the gap crosses below gap_z_entry (an unusually large
    overnight gap-down); exit fires when the Z-score normalizes back above
    gap_z_exit. Both are filled at close_t, matching every other voter's
    same-bar-close-fill convention (see simulate_signals).
    """
    prev_close = close.shift(1)
    gap = (open_ - prev_close) / prev_close

    gap_mean = gap.rolling(window=gap_lookback, min_periods=gap_lookback).mean()
    gap_std = gap.rolling(window=gap_lookback, min_periods=gap_lookback).std()
    gap_z = (gap - gap_mean) / gap_std.replace(0, np.nan)

    prev_above_entry = gap_z.shift(1) >= gap_z_entry
    now_below_entry = gap_z < gap_z_entry
    entries = (prev_above_entry & now_below_entry).fillna(False).astype(bool)

    prev_below_exit = gap_z.shift(1) < gap_z_exit
    now_above_exit = gap_z >= gap_z_exit
    exits = (prev_below_exit & now_above_exit).fillna(False).astype(bool)

    return entries, exits
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/lab/test_overnight_gap_signals.py -v`
Expected: PASS on all 6 tests. If `test_entry_on_extreme_gap_down` or
`test_stricter_threshold_fewer_entries` fails on the exact assertion, inspect the
printed gap Z-score series (`overnight_gap_signals` intermediate `gap_z` — add a
temporary `print(gap_z["A"].iloc[45:55])` while debugging) and adjust the test's
injected gap magnitude (`0.85` multiplier) or `gap_lookback`/threshold values so the
engineered event is clearly outside the rolling distribution — do not change the
production threshold defaults to make a test pass.

- [ ] **Step 5: Commit**

```bash
git add src/ggTrader/lab/strategies/indicators.py tests/lab/test_overnight_gap_signals.py
git commit -m "feat(lab): add overnight-gap reversion indicator functions"
```

---

### Task 2: `OvernightGapReversionSignal` strategy class (select/to_targets)

**Files:**
- Modify: `src/ggTrader/lab/strategies/signals.py`
- Test: `tests/lab/test_overnight_gap_signals.py` (append)

**Interfaces:**
- Consumes: `extract_close`, `extract_open`, `eligible_symbols`, `overnight_gap_signals`
  (Task 1); `LabConfig`, `Plan`, `SignalTargets` from `ggTrader.lab.strategy`.
- Produces: `OvernightGapReversionSignal` class with `name = "overnight_gap"`,
  `target_kind = "signals"`, constructor `(cfg, gap_lookback=20, gap_z_entry=-1.5,
  gap_z_exit=-0.5)`, `select()`, `to_targets()`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/lab/test_overnight_gap_signals.py`:

```python
from ggTrader.lab.strategies.signals import OvernightGapReversionSignal
from ggTrader.lab.strategy import LabConfig, SignalTargets


def test_overnight_gap_select_returns_eligible():
    ohlcv = _ohlcv(["A", "B", "C"], n=500)
    strat = OvernightGapReversionSignal(LabConfig(min_history_bars=400))
    sels = strat.select(ohlcv.index[-1], ohlcv, ["A", "B", "C"])
    assert [s["symbol"] for s in sels] == ["A", "B", "C"]
    assert all("gap_lookback" in s and "gap_z_entry" in s and "gap_z_exit" in s for s in sels)


def test_overnight_gap_select_respects_min_history():
    ohlcv = _ohlcv(["A"], n=200)
    strat = OvernightGapReversionSignal(LabConfig(min_history_bars=400))
    sels = strat.select(ohlcv.index[-1], ohlcv, ["A"])
    assert sels == []


def test_overnight_gap_to_targets_returns_signal_targets():
    ohlcv = _ohlcv(["A", "B"], n=500)
    strat = OvernightGapReversionSignal(LabConfig(min_history_bars=100))
    plans = {
        ohlcv.index[300]: [
            {"symbol": "A", "weight": 0.0, "gap_lookback": 20, "gap_z_entry": -1.5, "gap_z_exit": -0.5},
            {"symbol": "B", "weight": 0.0, "gap_lookback": 20, "gap_z_entry": -1.5, "gap_z_exit": -0.5},
        ],
    }
    result = strat.to_targets(plans, ohlcv)
    assert isinstance(result, SignalTargets)
    assert result.entries.shape == result.exits.shape
    assert set(result.entries.columns) == {"A", "B"}
    assert result.entries.dtypes.eq(bool).all()
    assert result.exits.dtypes.eq(bool).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/lab/test_overnight_gap_signals.py -v`
Expected: FAIL with `ImportError: cannot import name 'OvernightGapReversionSignal'`.

- [ ] **Step 3: Implement the strategy class**

Add to `src/ggTrader/lab/strategies/signals.py`'s import block:

```python
from ggTrader.lab.strategies.indicators import (
    bb_signals,
    eligible_symbols,
    ema_signals,
    extract_close,
    extract_open,
    extract_volume,
    macd_signals,
    mtf_signals,
    overnight_gap_signals,
    rsi_signals,
    volume_bb_signals,
)
```

Add the class after `RsiReversionSignal` (before `MACDDivergenceSignal`):

```python
class OvernightGapReversionSignal:
    """Overnight-gap fade: enter after an extreme overnight gap-down Z-score.

    Entry: gap Z-score (open vs. prior close, rolling-normalized) crosses below
    gap_z_entry — an unusually large overnight gap-down (capitulation open).
    Exit: gap Z-score normalizes back above gap_z_exit.
    """

    name = "overnight_gap"
    target_kind = "signals"

    def __init__(
        self,
        cfg: LabConfig,
        gap_lookback: int = 20,
        gap_z_entry: float = -1.5,
        gap_z_exit: float = -0.5,
    ) -> None:
        self.cfg = cfg
        self.gap_lookback = gap_lookback
        self.gap_z_entry = gap_z_entry
        self.gap_z_exit = gap_z_exit

    @classmethod
    def sweep_params(cls) -> dict[str, list]:
        return {
            "gap_lookback": [10, 20, 30],
            "gap_z_entry": [-2.5, -2.0, -1.5],
            "gap_z_exit": [-1.0, -0.5, 0.0],
        }

    def select(self, asof: pd.Timestamp, data: pd.DataFrame, eligible: List[str]) -> Plan:
        data = data.loc[:asof]
        return [
            {
                "symbol": s,
                "weight": 0.0,
                "gap_lookback": self.gap_lookback,
                "gap_z_entry": self.gap_z_entry,
                "gap_z_exit": self.gap_z_exit,
            }
            for s in eligible_symbols(data, eligible, self.cfg.min_history_bars)
        ]

    def to_targets(self, plans: Dict[pd.Timestamp, Plan], data: pd.DataFrame) -> SignalTargets:
        symbols = sorted({s["symbol"] for plan in plans.values() for s in plan})
        close = extract_close(data, symbols)
        open_ = extract_open(data, symbols)
        entries, exits = overnight_gap_signals(
            close, open_, self.gap_lookback, self.gap_z_entry, self.gap_z_exit
        )
        return SignalTargets(entries=entries, exits=exits)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/lab/test_overnight_gap_signals.py -v`
Expected: PASS on all 9 tests so far.

- [ ] **Step 5: Commit**

```bash
git add src/ggTrader/lab/strategies/signals.py tests/lab/test_overnight_gap_signals.py
git commit -m "feat(lab): add OvernightGapReversionSignal select/to_targets"
```

---

### Task 3: `sweep_signals` for grid-search support

**Files:**
- Modify: `src/ggTrader/lab/strategies/signals.py`
- Test: `tests/lab/test_overnight_gap_signals.py` (append)

**Interfaces:**
- Consumes: `combo_name` from `ggTrader.lab.sweep` (existing helper, signature
  `combo_name(strategy_name: str, params: dict) -> str`).
- Produces: `OvernightGapReversionSignal.sweep_signals(combos: list[dict], symbols:
  list[str], data: pd.DataFrame) -> dict[str, SignalTargets]`.

- [ ] **Step 1: Write the failing test**

Append to `tests/lab/test_overnight_gap_signals.py`:

```python
def test_overnight_gap_sweep_params():
    params = OvernightGapReversionSignal.sweep_params()
    assert "gap_lookback" in params
    assert "gap_z_entry" in params
    assert "gap_z_exit" in params
    assert len(params["gap_lookback"]) >= 3


def test_overnight_gap_sweep_signals_produces_all_combos():
    ohlcv = _ohlcv(["A", "B"], n=500)
    strat = OvernightGapReversionSignal(LabConfig(min_history_bars=100))
    combos = [
        {"gap_lookback": 10, "gap_z_entry": -2.0, "gap_z_exit": -1.0},
        {"gap_lookback": 20, "gap_z_entry": -1.5, "gap_z_exit": -0.5},
    ]
    result = strat.sweep_signals(combos, ["A", "B"], ohlcv)
    assert len(result) == 2
    for st in result.values():
        assert isinstance(st, SignalTargets)
        assert set(st.entries.columns) == {"A", "B"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/lab/test_overnight_gap_signals.py -v`
Expected: FAIL — `sweep_params` passes (already implemented in Task 2), but
`test_overnight_gap_sweep_signals_produces_all_combos` fails with
`AttributeError: 'OvernightGapReversionSignal' object has no attribute 'sweep_signals'`.

- [ ] **Step 3: Implement `sweep_signals`**

Add to the `OvernightGapReversionSignal` class, after `to_targets`:

```python
    def sweep_signals(
        self,
        combos: list[dict],
        symbols: list[str],
        data: pd.DataFrame,
    ) -> dict[str, "SignalTargets"]:
        from ggTrader.lab.sweep import combo_name

        close = extract_close(data, symbols)
        open_ = extract_open(data, symbols)
        cache: dict[tuple[int, float, float], tuple[pd.DataFrame, pd.DataFrame]] = {}
        result: dict[str, SignalTargets] = {}
        for combo in combos:
            lookback = int(combo["gap_lookback"])
            z_entry = float(combo["gap_z_entry"])
            z_exit = float(combo["gap_z_exit"])
            key = (lookback, z_entry, z_exit)
            if key not in cache:
                cache[key] = overnight_gap_signals(close, open_, lookback, z_entry, z_exit)
            ent, ext = cache[key]
            result[combo_name(self.name, combo)] = SignalTargets(entries=ent, exits=ext)
        return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/lab/test_overnight_gap_signals.py -v`
Expected: PASS on all 11 tests.

- [ ] **Step 5: Commit**

```bash
git add src/ggTrader/lab/strategies/signals.py tests/lab/test_overnight_gap_signals.py
git commit -m "feat(lab): add sweep_signals to OvernightGapReversionSignal"
```

---

### Task 4: Registry + CLI wiring

**Files:**
- Modify: `src/ggTrader/lab/strategies/__init__.py`
- Test: `tests/lab/test_overnight_gap_signals.py` (append)

**Interfaces:**
- Consumes: `OvernightGapReversionSignal` (Task 2/3).
- Produces: `STRATEGY_REGISTRY["overnight_gap"] = OvernightGapReversionSignal`; the CLI's
  `--strategy` choices and `ggt lab --wfo --strategy overnight_gap` support this
  automatically since `cli.py:49` builds choices from `STRATEGY_NAMES +
  SIGNAL_STRATEGY_NAMES`, both derived from `STRATEGY_REGISTRY` — no `cli.py` edit
  needed.

- [ ] **Step 1: Write the failing tests**

Append to `tests/lab/test_overnight_gap_signals.py`:

```python
def test_overnight_gap_registered():
    from ggTrader.lab.strategies.signals import _get_registry

    assert "overnight_gap" in _get_registry()


def test_build_overnight_gap_strategy():
    from ggTrader.lab.strategies.signals import build_signal_strategy

    strat = build_signal_strategy("overnight_gap", LabConfig())
    assert strat.name == "overnight_gap"
    assert strat.target_kind == "signals"


def test_cli_accepts_overnight_gap():
    from ggTrader.lab.cli import build_arg_parser

    parser = build_arg_parser()
    args = parser.parse_args(["--strategy", "overnight_gap"])
    assert args.strategy == "overnight_gap"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/lab/test_overnight_gap_signals.py -v`
Expected: FAIL — `"overnight_gap" in _get_registry()` is `False`, and
`build_signal_strategy("overnight_gap", ...)` raises `ValueError: Unknown strategy`.

- [ ] **Step 3: Register the strategy**

In `src/ggTrader/lab/strategies/__init__.py`, add the import:

```python
from .signals import (
    BollingerReversionSignal,
    EmaCrossSignal,
    MACDDivergenceSignal,
    MultiTimeframeReversionSignal,
    OvernightGapReversionSignal,
    RsiReversionSignal,
    VolumeBBReversionSignal,
    WfoTournamentSignal,
)
```

Add to `STRATEGY_REGISTRY`:

```python
STRATEGY_REGISTRY: dict[str, Any] = {
    "ema_cross": EmaCrossSignal,
    "wfo_tournament": WfoTournamentSignal,
    "bb_reversion": BollingerReversionSignal,
    "rsi_reversion": RsiReversionSignal,
    "macd_divergence": MACDDivergenceSignal,
    "volume_bb_reversion": VolumeBBReversionSignal,
    "mtf_reversion": MultiTimeframeReversionSignal,
    "overnight_gap": OvernightGapReversionSignal,
    "ensemble": EnsembleSignal,
    "ensemble_ic": EnsembleICSignal,
    "ensemble_kelly": EnsembleKellySignal,
    "conviction_bb": ConvictionBBSignal,
    "ensemble_conviction": EnsembleConvictionSignal,
    "xs_momentum": CrossSectionalMomentum,
    "dual_momentum": DualMomentum,
}
```

Add `"OvernightGapReversionSignal"` to `__all__` (next to `"RsiReversionSignal"`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/lab/test_overnight_gap_signals.py -v`
Expected: PASS on all 14 tests.

- [ ] **Step 5: Commit**

```bash
git add src/ggTrader/lab/strategies/__init__.py tests/lab/test_overnight_gap_signals.py
git commit -m "feat(lab): register overnight_gap strategy in STRATEGY_REGISTRY"
```

---

### Task 5: Full-suite verification and lint

**Files:** none new — verification only.

- [ ] **Step 1: Run the full lab test suite**

Run: `pytest tests/lab/ -q`
Expected: All tests pass (417 pre-existing + 14 new = 431), zero failures.

- [ ] **Step 2: Run ruff**

Run: `ruff check src/ggTrader/lab/`
Expected: No new lint errors introduced by `indicators.py`, `signals.py`, or
`strategies/__init__.py`. Fix any reported issues (import ordering, unused imports)
before proceeding.

- [ ] **Step 3: Commit (only if ruff required fixes)**

```bash
git add -u
git commit -m "chore(lab): fix lint issues from overnight_gap addition"
```

(Skip this step entirely if Step 2 reported no issues.)

---

### Task 6: WFO smoke run — confirm end-to-end wiring through the real gates

**Files:** none — this is a research run, not a code change.

**Purpose:** Tasks 1–5 only prove the strategy is correctly wired into the registry and
passes unit tests with synthetic data. This task runs it once through the real
`ggt lab --wfo` harness against the SP500 universe to confirm it produces a result at
all (executes without exceptions, produces at least one non-empty fold) — this is a
wiring/smoke check, **not** the actual WFO validation research question ("does this
strategy beat the 1.12 Sharpe baseline?"), which is a separate, longer research task
that should be scoped on its own once this smoke run confirms the plumbing works.

- [ ] **Step 1: Run a smoke WFO pass**

Run:
```bash
docker compose run --rm ggtrader_live python ggt.py lab --wfo --strategy overnight_gap --universe sp500
```
Expected: The command completes without raising an exception and prints per-fold WFO
output (fold train/test date ranges, gate pass/fail, chosen params) for at least one
fold — matching the console output shape already produced by
`--strategy rsi_reversion --wfo` runs.

- [ ] **Step 2: Record the smoke-run result**

If the command fails, treat it as a Task 1–4 bug (not a research finding) and fix the
code before re-running — a crash means the wiring is broken, not that the strategy
lacks edge. If it completes, note the resulting OOS Sharpe/drawdown/gate-pass-rate
figures in a short note appended to `docs/roadmap.md`'s research-directions list (do
not draw a GO/NO-GO conclusion from a single smoke run — that requires the full
walk-forward research pass described below).

- [ ] **Step 3: Commit the roadmap note only**

```bash
git add docs/roadmap.md
git commit -m "docs(roadmap): note overnight_gap smoke-run wiring confirmed"
```

---

## Follow-on Roadmap (not part of this plan — scope for future plans)

Per `docs/2026-07-06_strategy_recommendations.md` §4, once Candidate A (this plan)
validates through a full WFO research pass (multiple SP500 folds, NDH/DSR gate
pass-rate, comparison against the 1.12 Sharpe baseline — a research task, not a coding
task, run via the same `ggt lab --wfo` harness with the full param grid from
`sweep_params()`), the next three candidates each warrant their own
`docs/superpowers/plans/` document when their turn comes:

1. **Candidate B — Cross-Sectional Idiosyncratic Volatility.** A new *weight-based*
   strategy (`target_kind = "weights"`), following the `momentum.py` pattern instead of
   `signals.py`, since it uses `simulate_weights`/`Portfolio.from_orders`. Needs a
   rolling-OLS-vs-SPY residual-variance function and cross-sectional quantile ranking —
   distinct enough machinery to deserve its own plan.
2. **Candidate C — Sector-Neutral Relative-Value Reversion.** Should be planned only
   after the in-progress, currently-uncommitted sector-constraint work
   (`data/universe/sp500_sectors.json`, `tests/lab/test_sector_constraints.py`) is
   committed and merged — this strategy builds directly on that infrastructure.
3. **Candidate D — Calendar/Seasonality Effects.** Small enough to fold into a single
   short plan (or even a single subagent-driven task) once A is validated — a boolean
   day-of-month/day-of-week matrix multiplied against existing weight logic.

## Verification (this plan)

- `pytest tests/lab/test_overnight_gap_signals.py -v` — all new tests pass (Tasks 1–4).
- `pytest tests/lab/ -q` — full lab suite still passes, no regressions (Task 5).
- `ruff check src/ggTrader/lab/` — clean (Task 5).
- `ggt lab --wfo --strategy overnight_gap --universe sp500` completes without
  exceptions and produces fold output (Task 6) — confirms the strategy is genuinely
  wired into the WFO/gate pipeline end-to-end, not just unit-testable in isolation.
