# Kelly-Criterion Position Sizing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a walk-forward research experiment testing whether Kelly-criterion position sizing beats the flat-3% baseline on the SP500 core ensemble reversion strategy.

**Architecture:** A new pure module (`kelly.py`) derives a pooled, causal, expanding-window Kelly fraction from a strategy's own closed trades. A new strategy variant (`EnsembleKellySignal`) reuses `EnsembleSignal`'s entries/exits unchanged and layers Kelly-derived per-bar sizes on top via the existing `SignalTargets.sizes` extension point (the same mechanism `EnsembleConvictionSignal` and `ConvictionBBSignal` already use) — no changes to `simulate.py`, `wfo.py`, or `gates.py` are needed.

**Tech Stack:** Python, pandas, numpy, pytest, existing `ggTrader.lab` WFO/gate/persistence pipeline.

## Global Constraints

- Kelly fraction: `f* = W - (1-W)/R` (W = win rate, R = avg-win/avg-loss), pooled across all symbols, never per-symbol.
- Estimation window: expanding, not rolling — recomputed causally from every trade closed strictly before the query time.
- Sweep grid: `kelly_multiplier ∈ {0.25, 0.5, 1.0}` — no other new tunables.
- Hard cap: position size ≤ `max_concentration_pct` (5%, matches the live risk guard in `src/ggTrader/paper/risk.py`).
- Fallback: if there's no measurable positive edge yet (fewer than `min_trades` closed trades, or `f* <= 0`), size at the flat-3% baseline (`base_size`) rather than zero.
- GO bar: OOS Sharpe > 1.12, drawdown no worse than -11%, and the winning `k` stable across a majority of the 17 SP500 folds.
- Spec: `docs/superpowers/specs/2026-07-06-kelly-position-sizing-design.md`.

---

## Task 1: Trade extraction (`extract_trades`)

**Files:**
- Create: `src/ggTrader/lab/kelly.py`
- Test: `tests/lab/test_kelly.py`

**Interfaces:**
- Produces: `extract_trades(entries: pd.DataFrame, exits: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame` with columns `["symbol", "entry_time", "exit_time", "ret"]`, sorted by `exit_time` ascending (ties broken by `symbol`). Empty DataFrame (same columns) if no trade closes. Later tasks depend on this exact column set and sort order.

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for ggTrader.lab.kelly — pooled expanding Kelly-fraction sizing."""

import numpy as np
import pandas as pd
import pytest

from ggTrader.lab.kelly import extract_trades


def _idx(n, start="2020-01-01"):
    return pd.date_range(start, periods=n, freq="B", tz="UTC")


class TestExtractTrades:
    def test_single_symbol_single_trade(self):
        idx = _idx(5)
        entries = pd.DataFrame({"A": [True, False, False, False, False]}, index=idx)
        exits = pd.DataFrame({"A": [False, False, True, False, False]}, index=idx)
        close = pd.DataFrame({"A": [100.0, 101, 110, 111, 112]}, index=idx)
        trades = extract_trades(entries, exits, close)
        assert len(trades) == 1
        row = trades.iloc[0]
        assert row["symbol"] == "A"
        assert row["entry_time"] == idx[0]
        assert row["exit_time"] == idx[2]
        assert row["ret"] == pytest.approx(0.10)

    def test_redundant_entries_while_in_position_are_ignored(self):
        idx = _idx(6)
        entries = pd.DataFrame({"A": [True, True, False, False, False, False]}, index=idx)
        exits = pd.DataFrame({"A": [False, False, False, True, False, False]}, index=idx)
        close = pd.DataFrame({"A": [100.0, 100, 100, 105, 105, 105]}, index=idx)
        trades = extract_trades(entries, exits, close)
        assert len(trades) == 1
        assert trades.iloc[0]["entry_time"] == idx[0]

    def test_unrealized_trade_at_end_is_dropped(self):
        idx = _idx(4)
        entries = pd.DataFrame({"A": [False, True, False, False]}, index=idx)
        exits = pd.DataFrame({"A": [False, False, False, False]}, index=idx)
        close = pd.DataFrame({"A": [100.0, 100, 100, 100]}, index=idx)
        trades = extract_trades(entries, exits, close)
        assert trades.empty

    def test_multiple_symbols_pooled_and_sorted_by_exit_time(self):
        idx = _idx(6)
        entries = pd.DataFrame(
            {
                "A": [True, False, False, False, False, False],
                "B": [False, True, False, False, False, False],
            },
            index=idx,
        )
        exits = pd.DataFrame(
            {
                "A": [False, False, False, True, False, False],
                "B": [False, False, True, False, False, False],
            },
            index=idx,
        )
        close = pd.DataFrame(
            {
                "A": [100.0, 100, 100, 105, 105, 105],
                "B": [50.0, 50, 55, 55, 55, 55],
            },
            index=idx,
        )
        trades = extract_trades(entries, exits, close)
        assert list(trades["symbol"]) == ["B", "A"]
        assert trades["exit_time"].is_monotonic_increasing

    def test_new_entry_after_close_opens_a_new_trade(self):
        idx = _idx(6)
        entries = pd.DataFrame({"A": [True, False, False, True, False, False]}, index=idx)
        exits = pd.DataFrame({"A": [False, True, False, False, True, False]}, index=idx)
        close = pd.DataFrame({"A": [100.0, 110, 110, 110, 121, 121]}, index=idx)
        trades = extract_trades(entries, exits, close)
        assert len(trades) == 2
        assert trades.iloc[0]["entry_time"] == idx[0]
        assert trades.iloc[1]["entry_time"] == idx[3]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/flynn/ggTrader && .venv/bin/pytest tests/lab/test_kelly.py -v`
Expected: FAIL/ERROR — `ggTrader.lab.kelly` does not exist yet.

- [ ] **Step 3: Implement `extract_trades`**

```python
"""Pooled, causal Kelly-criterion sizing for signal-based lab strategies.

Given a strategy's own (entries, exits, close) signal matrices, this module
reconstructs the round-trip trades those signals imply, estimates a pooled
expanding-window Kelly fraction from them, and turns that into a per-bar,
per-symbol position-size matrix suitable for SignalTargets.sizes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _symbol_trades(
    entry_positions: np.ndarray, exit_positions: np.ndarray, prices: np.ndarray
) -> list[tuple[int, int, float]]:
    """Pair one symbol's entry/exit bar positions into closed round trips.

    Mirrors vbt's from_signals semantics: while a position is open, further
    entries are ignored; a trade closes on the next exit at or after its
    entry. A trailing entry with no closing exit left is unrealized and
    dropped (it can't contribute a win/loss to the edge estimate yet).
    """
    trades: list[tuple[int, int, float]] = []
    ei = 0
    xi = 0
    n_entries = len(entry_positions)
    n_exits = len(exit_positions)
    in_position = False
    entry_idx = -1
    while True:
        if not in_position:
            if ei >= n_entries:
                break
            entry_idx = entry_positions[ei]
            ei += 1
            in_position = True
            while xi < n_exits and exit_positions[xi] < entry_idx:
                xi += 1
        else:
            if xi >= n_exits:
                break  # no closing exit remains -> unrealized, drop it
            exit_idx = exit_positions[xi]
            xi += 1
            ret = float(prices[exit_idx] / prices[entry_idx] - 1.0)
            trades.append((entry_idx, exit_idx, ret))
            in_position = False
            while ei < n_entries and entry_positions[ei] <= exit_idx:
                ei += 1
    return trades


def extract_trades(
    entries: pd.DataFrame, exits: pd.DataFrame, close: pd.DataFrame
) -> pd.DataFrame:
    """Pair each symbol's entry/exit signal bars into completed round-trip trades.

    Returns columns [symbol, entry_time, exit_time, ret], sorted by exit_time
    ascending (ties broken by symbol) — the ordering the rest of the
    Kelly-sizing pipeline assumes.
    """
    index = entries.index
    records = []
    for col in entries.columns:
        entry_positions = np.flatnonzero(entries[col].to_numpy())
        exit_positions = np.flatnonzero(exits[col].to_numpy())
        prices = close[col].to_numpy()
        for entry_idx, exit_idx, ret in _symbol_trades(entry_positions, exit_positions, prices):
            records.append(
                {
                    "symbol": col,
                    "entry_time": index[entry_idx],
                    "exit_time": index[exit_idx],
                    "ret": ret,
                }
            )
    trades = pd.DataFrame.from_records(
        records, columns=["symbol", "entry_time", "exit_time", "ret"]
    )
    if trades.empty:
        return trades
    return trades.sort_values(["exit_time", "symbol"]).reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/flynn/ggTrader && .venv/bin/pytest tests/lab/test_kelly.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/ggTrader/lab/kelly.py tests/lab/test_kelly.py
git commit -m "feat(lab): extract_trades — pair signal entries/exits into round-trip trades"
```

---

## Task 2: Expanding Kelly fraction (`expanding_kelly_fraction`, `kelly_fraction_asof`)

**Files:**
- Modify: `src/ggTrader/lab/kelly.py`
- Test: `tests/lab/test_kelly.py`

**Interfaces:**
- Consumes: the `trades` DataFrame shape from Task 1 (`symbol`, `entry_time`, `exit_time`, `ret`; sorted by `exit_time`).
- Produces:
  - `expanding_kelly_fraction(trades: pd.DataFrame, min_trades: int = 10) -> pd.Series` — indexed by `exit_time`, one value per row of `trades`, `NaN` until `min_trades` trades have closed or while wins/losses are both required-and-missing.
  - `kelly_fraction_asof(f_star: pd.Series, asof: pd.Timestamp) -> float` — the last `f_star` value strictly before `asof`; `NaN` if none.

- [ ] **Step 1: Write the failing tests**

Append to `tests/lab/test_kelly.py`:

```python
from ggTrader.lab.kelly import expanding_kelly_fraction, kelly_fraction_asof


class TestExpandingKellyFraction:
    def test_nan_before_min_trades(self):
        trades = pd.DataFrame({"ret": [0.05, -0.02, 0.03], "exit_time": _idx(3)})
        f_star = expanding_kelly_fraction(trades, min_trades=5)
        assert f_star.isna().all()

    def test_nan_when_only_wins_or_only_losses(self):
        trades = pd.DataFrame({"ret": [0.05] * 10, "exit_time": _idx(10)})
        f_star = expanding_kelly_fraction(trades, min_trades=3)
        assert f_star.isna().all()

    def test_matches_hand_computed_value(self):
        # 6 wins of +0.10, 4 losses of -0.05 -> W=0.6, avg_win=0.10, avg_loss=0.05, R=2
        # f* = W - (1-W)/R = 0.6 - 0.4/2 = 0.4
        rets = [0.10] * 6 + [-0.05] * 4
        trades = pd.DataFrame({"ret": rets, "exit_time": _idx(10)})
        f_star = expanding_kelly_fraction(trades, min_trades=3)
        assert f_star.iloc[-1] == pytest.approx(0.4)

    def test_is_causal_expanding(self):
        """f*.iloc[i] must be unaffected by trades after position i."""
        rets = [0.10, -0.05, 0.10, -0.05, 0.10, -0.05]
        trades = pd.DataFrame({"ret": rets, "exit_time": _idx(6)})
        full = expanding_kelly_fraction(trades, min_trades=2)
        prefix = expanding_kelly_fraction(trades.iloc[:4], min_trades=2)
        pd.testing.assert_series_equal(full.iloc[:4], prefix, check_names=False)

    def test_empty_trades_returns_empty_series(self):
        trades = pd.DataFrame(columns=["symbol", "entry_time", "exit_time", "ret"])
        f_star = expanding_kelly_fraction(trades)
        assert f_star.empty


class TestKellyFractionAsof:
    def test_uses_only_trades_strictly_before_asof(self):
        idx = _idx(5)
        f_star = pd.Series([0.1, 0.2, 0.3], index=idx[[0, 2, 4]])
        assert np.isnan(kelly_fraction_asof(f_star, idx[0]))
        assert kelly_fraction_asof(f_star, idx[1]) == pytest.approx(0.1)
        assert kelly_fraction_asof(f_star, idx[3]) == pytest.approx(0.2)

    def test_empty_series_returns_nan(self):
        assert np.isnan(kelly_fraction_asof(pd.Series(dtype=float), pd.Timestamp("2020-01-01")))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/flynn/ggTrader && .venv/bin/pytest tests/lab/test_kelly.py -v -k "ExpandingKellyFraction or KellyFractionAsof"`
Expected: FAIL — functions not defined.

- [ ] **Step 3: Implement `expanding_kelly_fraction` and `kelly_fraction_asof`**

Append to `src/ggTrader/lab/kelly.py`:

```python
def expanding_kelly_fraction(trades: pd.DataFrame, min_trades: int = 10) -> pd.Series:
    """Pooled expanding Kelly fraction f* = W - (1-W)/R, one value per row of
    `trades` (must be pre-sorted by exit_time ascending, as extract_trades
    returns). f*.iloc[i] is computed only from trades[0:i+1] — it never
    looks at a later trade.

    NaN until `min_trades` trades have closed, or while there are zero wins
    or zero losses in the pool so far (the payoff ratio R is undefined
    without both).
    """
    if trades.empty:
        return pd.Series(dtype=float)
    ret = trades["ret"].to_numpy()
    is_win = ret > 0
    is_loss = ret < 0
    n = np.arange(1, len(ret) + 1)
    win_count = np.cumsum(is_win)
    loss_count = np.cumsum(is_loss)
    win_rate = win_count / n
    sum_win = np.cumsum(np.where(is_win, ret, 0.0))
    sum_loss = np.cumsum(np.where(is_loss, -ret, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        avg_win = np.where(win_count > 0, sum_win / np.maximum(win_count, 1), np.nan)
        avg_loss = np.where(loss_count > 0, sum_loss / np.maximum(loss_count, 1), np.nan)
        payoff_ratio = avg_win / avg_loss
        f_star = win_rate - (1.0 - win_rate) / payoff_ratio
    valid = (n >= min_trades) & (win_count > 0) & (loss_count > 0)
    f_star = np.where(valid, f_star, np.nan)
    return pd.Series(f_star, index=trades["exit_time"], name="f_star")


def kelly_fraction_asof(f_star: pd.Series, asof: pd.Timestamp) -> float:
    """f* using only trades closed strictly before `asof`; NaN if none qualify.

    `f_star` must be indexed by exit_time, sorted ascending (the output of
    expanding_kelly_fraction).
    """
    if f_star.empty:
        return float("nan")
    pos = f_star.index.searchsorted(asof, side="left")
    if pos == 0:
        return float("nan")
    return float(f_star.iloc[pos - 1])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/flynn/ggTrader && .venv/bin/pytest tests/lab/test_kelly.py -v`
Expected: PASS (all tests so far)

- [ ] **Step 5: Commit**

```bash
git add src/ggTrader/lab/kelly.py tests/lab/test_kelly.py
git commit -m "feat(lab): expanding_kelly_fraction + kelly_fraction_asof — causal pooled edge estimate"
```

---

## Task 3: Sizing orchestrator (`kelly_sizes`) + causality integration test

**Files:**
- Modify: `src/ggTrader/lab/kelly.py`
- Test: `tests/lab/test_kelly.py`

**Interfaces:**
- Consumes: `extract_trades`, `expanding_kelly_fraction`, `kelly_fraction_asof` from Tasks 1-2.
- Produces: `kelly_sizes(entries: pd.DataFrame, exits: pd.DataFrame, close: pd.DataFrame, *, kelly_multiplier: float, base_size: float, max_size: float, min_trades: int = 10) -> pd.DataFrame` — (time x symbol) float, `NaN` where `entries` is `False`. This is the function Task 4's strategy class calls directly.

- [ ] **Step 1: Write the failing tests**

Append to `tests/lab/test_kelly.py`:

```python
from ggTrader.lab.kelly import kelly_sizes


class TestKellySizes:
    def test_nan_where_no_entry(self):
        idx = _idx(5)
        entries = pd.DataFrame({"A": [True, False, False, False, False]}, index=idx)
        exits = pd.DataFrame({"A": [False, True, False, False, False]}, index=idx)
        close = pd.DataFrame({"A": [100.0, 101, 102, 103, 104]}, index=idx)
        sizes = kelly_sizes(
            entries, exits, close, kelly_multiplier=0.5, base_size=0.03, max_size=0.05
        )
        no_entry = ~entries
        assert sizes[no_entry].isna().all().all()

    def test_falls_back_to_base_size_without_measurable_edge(self):
        idx = _idx(5)
        entries = pd.DataFrame({"A": [True, False, False, False, False]}, index=idx)
        exits = pd.DataFrame({"A": [False, True, False, False, False]}, index=idx)
        close = pd.DataFrame({"A": [100.0, 101, 102, 103, 104]}, index=idx)
        sizes = kelly_sizes(
            entries, exits, close, kelly_multiplier=0.5, base_size=0.03, max_size=0.05
        )
        assert sizes.at[idx[0], "A"] == pytest.approx(0.03)

    def test_capped_at_max_size(self):
        idx = _idx(40)
        entries = pd.DataFrame({"A": [False] * 40}, index=idx)
        exits = pd.DataFrame({"A": [False] * 40}, index=idx)
        prices = [100.0] * 40
        # 6 winning round-trips (+10%) then 4 losing round-trips (-5%).
        for i in range(10):
            entry_bar, exit_bar = 2 * i, 2 * i + 1
            entries.iloc[entry_bar, 0] = True
            exits.iloc[exit_bar, 0] = True
            prices[exit_bar] = 110.0 if i < 6 else 95.0
        entries.iloc[35, 0] = True
        exits.iloc[36, 0] = True
        close = pd.DataFrame({"A": prices}, index=idx)

        sizes = kelly_sizes(
            entries,
            exits,
            close,
            kelly_multiplier=5.0,
            base_size=0.03,
            max_size=0.05,
            min_trades=3,
        )
        # W=0.6, avg_win=0.10, avg_loss=0.05, R=2 -> f*=0.4; k*f*=2.0, must cap.
        assert sizes.at[idx[35], "A"] == pytest.approx(0.05)


class TestKellySizesCausality:
    def test_future_trades_do_not_affect_earlier_sizes(self):
        """Appending more trades/bars to the end of the data must not change
        the Kelly size computed for an earlier entry — the no-look-ahead
        property this sizing mechanism relies on for honest walk-forward."""
        idx = _idx(30)
        entries = pd.DataFrame({"A": [False] * 30}, index=idx)
        exits = pd.DataFrame({"A": [False] * 30}, index=idx)
        prices = [100.0] * 30
        for i in range(8):
            entry_bar, exit_bar = 2 * i, 2 * i + 1
            entries.iloc[entry_bar, 0] = True
            exits.iloc[exit_bar, 0] = True
            prices[exit_bar] = 110.0 if i % 2 == 0 else 95.0
        entries.iloc[20, 0] = True
        exits.iloc[21, 0] = True
        close_short = pd.DataFrame({"A": prices}, index=idx)
        sizes_short = kelly_sizes(
            entries,
            exits,
            close_short,
            kelly_multiplier=0.5,
            base_size=0.03,
            max_size=0.05,
            min_trades=3,
        )

        idx_long = _idx(38)
        entries_long = entries.reindex(idx_long, fill_value=False)
        exits_long = exits.reindex(idx_long, fill_value=False)
        prices_long = prices + [100.0] * 8
        entries_long.iloc[36, 0] = True
        exits_long.iloc[37, 0] = True
        prices_long[37] = 50.0  # a huge future loss
        close_long = pd.DataFrame({"A": prices_long}, index=idx_long)
        sizes_long = kelly_sizes(
            entries_long,
            exits_long,
            close_long,
            kelly_multiplier=0.5,
            base_size=0.03,
            max_size=0.05,
            min_trades=3,
        )
        assert sizes_long.at[idx[20], "A"] == pytest.approx(sizes_short.at[idx[20], "A"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/flynn/ggTrader && .venv/bin/pytest tests/lab/test_kelly.py -v -k "KellySizes"`
Expected: FAIL — `kelly_sizes` not defined.

- [ ] **Step 3: Implement `kelly_sizes`**

Append to `src/ggTrader/lab/kelly.py`:

```python
def kelly_sizes(
    entries: pd.DataFrame,
    exits: pd.DataFrame,
    close: pd.DataFrame,
    *,
    kelly_multiplier: float,
    base_size: float,
    max_size: float,
    min_trades: int = 10,
) -> pd.DataFrame:
    """Per-bar, per-symbol position size (time x symbol) for SignalTargets.sizes.

    NaN everywhere `entries` is False. Where `entries` is True:
    `kelly_multiplier * f*`, capped at `max_size`, using only the causal
    pooled expanding Kelly fraction as of that entry's time. Falls back to
    `base_size` whenever there isn't yet a positive measurable edge (fewer
    than `min_trades` closed trades, or f* <= 0).
    """
    trades = extract_trades(entries, exits, close)
    f_star = expanding_kelly_fraction(trades, min_trades=min_trades)

    sizes = pd.DataFrame(np.nan, index=entries.index, columns=entries.columns)
    for col in entries.columns:
        entry_times = entries.index[entries[col].to_numpy()]
        for t in entry_times:
            f = kelly_fraction_asof(f_star, t)
            if np.isnan(f) or f <= 0:
                size = base_size
            else:
                size = min(kelly_multiplier * f, max_size)
            sizes.at[t, col] = size
    return sizes
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/flynn/ggTrader && .venv/bin/pytest tests/lab/test_kelly.py -v`
Expected: PASS (all tests in the file)

- [ ] **Step 5: Commit**

```bash
git add src/ggTrader/lab/kelly.py tests/lab/test_kelly.py
git commit -m "feat(lab): kelly_sizes — capped, fallback-safe position sizing from pooled edge"
```

---

## Task 4: `EnsembleKellySignal` strategy

**Files:**
- Create: `src/ggTrader/lab/strategies/ensemble_kelly.py`
- Test: `tests/lab/test_ensemble_kelly.py`

**Interfaces:**
- Consumes: `EnsembleSignal`, `DEFAULT_VOTERS`, `_validate_voters` from `src/ggTrader/lab/strategies/ensemble.py`; `kelly_sizes` from Task 3; `eligible_symbols`, `extract_close`, `extract_volume` from `src/ggTrader/lab/strategies/indicators.py`; `LabConfig`, `Plan`, `SignalTargets` from `src/ggTrader/lab/strategy.py`; `combo_name` from `src/ggTrader/lab/sweep.py`.
- Produces: class `EnsembleKellySignal` with `name = "ensemble_kelly"`, `target_kind = "signals"`, `sweep_params()`, `select()`, `to_targets()`, `sweep_signals()` — the exact `Strategy` protocol shape every other strategy in `STRATEGY_REGISTRY` implements. Task 5 imports this class.

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for EnsembleKellySignal — Kelly-criterion-sized ensemble."""

import numpy as np
import pandas as pd

from ggTrader.lab.strategies.ensemble import EnsembleSignal
from ggTrader.lab.strategies.ensemble_kelly import EnsembleKellySignal
from ggTrader.lab.strategy import LabConfig, SignalTargets


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


def _plans(ohlcv):
    symbols = sorted(ohlcv.columns.get_level_values(0).unique())
    return {ohlcv.index[100]: [{"symbol": s, "weight": 0.0} for s in symbols]}


class TestEnsembleKellySignal:
    def test_returns_signal_targets_with_sizes(self):
        cfg = LabConfig(min_history_bars=50)
        strat = EnsembleKellySignal(cfg)
        ohlcv = _ohlcv(n=300)
        targets = strat.to_targets(_plans(ohlcv), ohlcv)
        assert isinstance(targets, SignalTargets)
        assert targets.sizes is not None
        assert targets.sizes.shape == targets.entries.shape

    def test_entries_exits_match_plain_ensemble(self):
        """Entry/exit logic must be identical to EnsembleSignal — only sizing differs."""
        cfg = LabConfig(min_history_bars=50)
        ohlcv = _ohlcv(n=300, seed=77)

        plain = EnsembleSignal(cfg, min_agree=2)
        kelly = EnsembleKellySignal(cfg, min_agree=2)

        t_plain = plain.to_targets(_plans(ohlcv), ohlcv)
        t_kelly = kelly.to_targets(_plans(ohlcv), ohlcv)

        pd.testing.assert_frame_equal(t_plain.entries, t_kelly.entries)
        pd.testing.assert_frame_equal(t_plain.exits, t_kelly.exits)

    def test_sizes_bounded_by_max_size(self):
        cfg = LabConfig(min_history_bars=50)
        strat = EnsembleKellySignal(
            cfg, min_agree=1, kelly_multiplier=1.0, base_size=0.03, max_size=0.05
        )
        ohlcv = _ohlcv(n=300)
        targets = strat.to_targets(_plans(ohlcv), ohlcv)
        valid = targets.sizes[targets.entries].dropna()
        if len(valid) > 0:
            assert valid.max() <= 0.05 + 1e-10

    def test_sizes_nan_where_no_entry(self):
        cfg = LabConfig(min_history_bars=50)
        strat = EnsembleKellySignal(cfg)
        ohlcv = _ohlcv(n=300)
        targets = strat.to_targets(_plans(ohlcv), ohlcv)
        no_entry = ~targets.entries
        assert targets.sizes[no_entry].isna().all().all()

    def test_sweep_params_is_kelly_multiplier_only(self):
        params = EnsembleKellySignal.sweep_params()
        assert params == {"kelly_multiplier": [0.25, 0.5, 1.0]}

    def test_name_and_target_kind(self):
        assert EnsembleKellySignal.name == "ensemble_kelly"
        assert EnsembleKellySignal.target_kind == "signals"

    def test_select_delegates_to_eligible(self):
        cfg = LabConfig(min_history_bars=50)
        strat = EnsembleKellySignal(cfg)
        ohlcv = _ohlcv(n=300)
        symbols = sorted(ohlcv.columns.get_level_values(0).unique())
        plan = strat.select(ohlcv.index[200], ohlcv, symbols)
        assert len(plan) == len(symbols)

    def test_sweep_signals_returns_sizes_for_each_multiplier(self):
        cfg = LabConfig(min_history_bars=50)
        strat = EnsembleKellySignal(cfg, min_agree=1)
        ohlcv = _ohlcv(n=300)
        symbols = sorted(ohlcv.columns.get_level_values(0).unique())
        combos = [{"kelly_multiplier": 0.25}, {"kelly_multiplier": 0.5}, {"kelly_multiplier": 1.0}]
        result = strat.sweep_signals(combos, symbols, ohlcv)
        assert len(result) == 3
        for targets in result.values():
            assert targets.sizes is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/flynn/ggTrader && .venv/bin/pytest tests/lab/test_ensemble_kelly.py -v`
Expected: FAIL — `ggTrader.lab.strategies.ensemble_kelly` does not exist yet.

- [ ] **Step 3: Implement `EnsembleKellySignal`**

```python
"""Kelly-criterion-sized ensemble: same entries/exits as EnsembleSignal, but
position size scales with a pooled, causal, expanding Kelly fraction
estimated from the strategy's own closed-trade history (see
ggTrader.lab.kelly). See docs/superpowers/specs/2026-07-06-kelly-position-sizing-design.md.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from ggTrader.lab.kelly import kelly_sizes
from ggTrader.lab.strategies.ensemble import DEFAULT_VOTERS, EnsembleSignal, _validate_voters
from ggTrader.lab.strategies.indicators import eligible_symbols, extract_close, extract_volume
from ggTrader.lab.strategy import LabConfig, Plan, SignalTargets


class EnsembleKellySignal:
    """EnsembleSignal entries/exits with Kelly-criterion position sizing.

    Falls back to `base_size` (the deployed flat-3% baseline) whenever
    there isn't yet a positive measurable edge; always capped at `max_size`.
    """

    name = "ensemble_kelly"
    target_kind = "signals"

    def __init__(
        self,
        cfg: LabConfig,
        min_agree: int = 2,
        min_agree_exit: int | None = None,
        bb_period: int = 20,
        bb_std: float = 2.0,
        rsi_period: int = 14,
        rsi_oversold: int = 30,
        rsi_exit: int = 50,
        ema_fast: int = 20,
        ema_slow: int = 50,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        divergence_window: int = 20,
        vol_period: int = 20,
        vol_mult: float = 2.0,
        weekly_rsi_period: int = 14,
        weekly_rsi_oversold: int = 30,
        weekly_rsi_exit: int = 50,
        kelly_multiplier: float = 0.5,
        base_size: float = 0.03,
        max_size: float = 0.05,
        min_trades: int = 10,
        voters: tuple[str, ...] | list[str] = DEFAULT_VOTERS,
    ) -> None:
        self.voters = _validate_voters(voters)
        self.cfg = cfg
        self.min_agree = min_agree
        self.min_agree_exit = min_agree_exit if min_agree_exit is not None else min_agree
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_exit = rsi_exit
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.divergence_window = divergence_window
        self.vol_period = vol_period
        self.vol_mult = vol_mult
        self.weekly_rsi_period = weekly_rsi_period
        self.weekly_rsi_oversold = weekly_rsi_oversold
        self.weekly_rsi_exit = weekly_rsi_exit
        self.kelly_multiplier = kelly_multiplier
        self.base_size = base_size
        self.max_size = max_size
        self.min_trades = min_trades

    @classmethod
    def sweep_params(cls) -> dict[str, list]:
        # Only the Kelly multiplier is swept; ensemble params pinned at baseline.
        return {"kelly_multiplier": [0.25, 0.5, 1.0]}

    def _base_ensemble(self) -> EnsembleSignal:
        return EnsembleSignal(
            self.cfg,
            min_agree=self.min_agree,
            min_agree_exit=self.min_agree_exit,
            bb_period=self.bb_period,
            bb_std=self.bb_std,
            rsi_period=self.rsi_period,
            rsi_oversold=self.rsi_oversold,
            rsi_exit=self.rsi_exit,
            ema_fast=self.ema_fast,
            ema_slow=self.ema_slow,
            macd_fast=self.macd_fast,
            macd_slow=self.macd_slow,
            macd_signal=self.macd_signal,
            divergence_window=self.divergence_window,
            vol_period=self.vol_period,
            vol_mult=self.vol_mult,
            weekly_rsi_period=self.weekly_rsi_period,
            weekly_rsi_oversold=self.weekly_rsi_oversold,
            weekly_rsi_exit=self.weekly_rsi_exit,
            voters=self.voters,
        )

    def select(self, asof: pd.Timestamp, data: pd.DataFrame, eligible: List[str]) -> Plan:
        data = data.loc[:asof]
        syms = eligible_symbols(data, eligible, self.cfg.min_history_bars)
        max_sec = self.cfg.max_sector_count
        if max_sec is not None:
            from ggTrader.lab.strategies.registry import apply_sector_constraints

            syms = apply_sector_constraints(syms, max_sec)
        return [{"symbol": s, "weight": 0.0} for s in syms]

    def _generate_signals_with_sizes(
        self, close: pd.DataFrame, volume: pd.DataFrame
    ) -> SignalTargets:
        base_targets = self._base_ensemble()._generate_signals(close, volume)
        sizes = kelly_sizes(
            base_targets.entries,
            base_targets.exits,
            close,
            kelly_multiplier=self.kelly_multiplier,
            base_size=self.base_size,
            max_size=self.max_size,
            min_trades=self.min_trades,
        )
        return SignalTargets(entries=base_targets.entries, exits=base_targets.exits, sizes=sizes)

    def to_targets(self, plans: Dict[pd.Timestamp, Plan], data: pd.DataFrame) -> SignalTargets:
        symbols = sorted({s["symbol"] for plan in plans.values() for s in plan})
        close = extract_close(data, symbols)
        volume = extract_volume(data, symbols)
        return self._generate_signals_with_sizes(close, volume)

    def sweep_signals(
        self, combos: list[dict], symbols: list[str], data: pd.DataFrame
    ) -> dict[str, SignalTargets]:
        from ggTrader.lab.sweep import combo_name

        close = extract_close(data, symbols)
        volume = extract_volume(data, symbols)
        result: dict[str, SignalTargets] = {}
        for combo in combos:
            strat = EnsembleKellySignal(
                self.cfg,
                min_agree=self.min_agree,
                min_agree_exit=self.min_agree_exit,
                bb_period=self.bb_period,
                bb_std=self.bb_std,
                rsi_period=self.rsi_period,
                rsi_oversold=self.rsi_oversold,
                rsi_exit=self.rsi_exit,
                ema_fast=self.ema_fast,
                ema_slow=self.ema_slow,
                macd_fast=self.macd_fast,
                macd_slow=self.macd_slow,
                macd_signal=self.macd_signal,
                divergence_window=self.divergence_window,
                vol_period=self.vol_period,
                vol_mult=self.vol_mult,
                weekly_rsi_period=self.weekly_rsi_period,
                weekly_rsi_oversold=self.weekly_rsi_oversold,
                weekly_rsi_exit=self.weekly_rsi_exit,
                kelly_multiplier=float(combo.get("kelly_multiplier", self.kelly_multiplier)),
                base_size=self.base_size,
                max_size=self.max_size,
                min_trades=self.min_trades,
                voters=self.voters,
            )
            result[combo_name(self.name, combo)] = strat._generate_signals_with_sizes(
                close, volume
            )
        return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/flynn/ggTrader && .venv/bin/pytest tests/lab/test_ensemble_kelly.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add src/ggTrader/lab/strategies/ensemble_kelly.py tests/lab/test_ensemble_kelly.py
git commit -m "feat(lab): EnsembleKellySignal — Kelly-criterion-sized ensemble strategy"
```

---

## Task 5: Registry wiring

**Files:**
- Modify: `src/ggTrader/lab/strategies/__init__.py`
- Test: `tests/lab/test_ensemble_kelly.py` (append)

**Interfaces:**
- Consumes: `EnsembleKellySignal` from Task 4.
- Produces: `"ensemble_kelly"` present in `STRATEGY_REGISTRY`, `all_strategy_names()`, `signal_strategy_names()`, and `SIGNAL_STRATEGY_NAMES` (all derived automatically from `STRATEGY_REGISTRY` per the existing single-source pattern — no other files need editing).

- [ ] **Step 1: Write the failing tests**

Append to `tests/lab/test_ensemble_kelly.py`:

```python
def test_registered_in_strategy_registry():
    from ggTrader.lab.strategies import STRATEGY_REGISTRY

    assert "ensemble_kelly" in STRATEGY_REGISTRY
    assert STRATEGY_REGISTRY["ensemble_kelly"] is EnsembleKellySignal


def test_registered_in_signal_strategy_names():
    from ggTrader.lab.strategies.registry import signal_strategy_names

    assert "ensemble_kelly" in signal_strategy_names()


def test_build_signal_strategy():
    from ggTrader.lab.strategies.signals import build_signal_strategy

    strat = build_signal_strategy("ensemble_kelly", LabConfig())
    assert strat.name == "ensemble_kelly"


def test_cli_accepts_ensemble_kelly():
    from ggTrader.lab.cli import build_arg_parser

    parser = build_arg_parser()
    args = parser.parse_args(["--strategy", "ensemble_kelly"])
    assert args.strategy == "ensemble_kelly"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/flynn/ggTrader && .venv/bin/pytest tests/lab/test_ensemble_kelly.py -v -k "registered or build_signal or cli_accepts"`
Expected: FAIL — `"ensemble_kelly"` not yet in the registry.

- [ ] **Step 3: Wire the strategy into `STRATEGY_REGISTRY`**

In `src/ggTrader/lab/strategies/__init__.py`:

```python
from .ensemble_ic import EnsembleICSignal
from .ensemble_kelly import EnsembleKellySignal
```

```python
STRATEGY_REGISTRY: dict[str, Any] = {
    "ema_cross": EmaCrossSignal,
    "wfo_tournament": WfoTournamentSignal,
    "bb_reversion": BollingerReversionSignal,
    "rsi_reversion": RsiReversionSignal,
    "macd_divergence": MACDDivergenceSignal,
    "volume_bb_reversion": VolumeBBReversionSignal,
    "mtf_reversion": MultiTimeframeReversionSignal,
    "ensemble": EnsembleSignal,
    "ensemble_ic": EnsembleICSignal,
    "ensemble_kelly": EnsembleKellySignal,
    "conviction_bb": ConvictionBBSignal,
    "ensemble_conviction": EnsembleConvictionSignal,
    "xs_momentum": CrossSectionalMomentum,
    "dual_momentum": DualMomentum,
}
```

```python
__all__ = [
    "STRATEGY_REGISTRY",
    "all_strategy_names",
    "apply_sector_constraints",
    "build_strategy",
    "signal_registry",
    "signal_strategy_names",
    "weight_strategy_names",
    "CrossSectionalMomentum",
    "DualMomentum",
    "EmaCrossSignal",
    "WfoTournamentSignal",
    "BollingerReversionSignal",
    "RsiReversionSignal",
    "MACDDivergenceSignal",
    "VolumeBBReversionSignal",
    "MultiTimeframeReversionSignal",
    "EnsembleSignal",
    "EnsembleICSignal",
    "EnsembleKellySignal",
    "ConvictionBBSignal",
    "EnsembleConvictionSignal",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/flynn/ggTrader && .venv/bin/pytest tests/lab/test_ensemble_kelly.py tests/lab/test_kelly.py tests/lab/test_registry.py tests/lab/test_cli.py -v`
Expected: PASS (all)

- [ ] **Step 5: Run the full test suite**

Run: `cd /home/flynn/ggTrader && .venv/bin/pytest tests/ -q`
Expected: PASS, 313+ tests (305 previous + new Kelly tests), zero failures.

- [ ] **Step 6: Commit**

```bash
git add src/ggTrader/lab/strategies/__init__.py tests/lab/test_ensemble_kelly.py
git commit -m "feat(lab): register ensemble_kelly in STRATEGY_REGISTRY + CLI wiring"
```

---

## Task 6: Run the WFO experiment and record the verdict

**Files:**
- Modify: `docs/roadmap.md`

**Interfaces:**
- Consumes: `ensemble_kelly` strategy registered in Task 5, existing `ggt lab --strategy <name>@<universe> --wfo` CLI path (`src/ggTrader/lab/cli.py`), existing NDH/DSR gates (`src/ggTrader/lab/gates.py`), existing persistence (`lab_runs`/`lab_summary` tables).
- Produces: a roadmap entry recording the GO/NO-GO verdict, following the same format as the June 28 entries.

- [ ] **Step 1: Run the WFO sweep on the SP500 core universe**

Run: `cd /home/flynn/ggTrader && docker compose run --rm ggtrader_live python ggt.py lab --strategy ensemble_kelly@sp500 --wfo`

This sweeps `kelly_multiplier ∈ {0.25, 0.5, 1.0}` across the same 17 SP500 folds used to validate the deployed baseline, gated through NDH + DSR, and persists results to `lab_runs`/`lab_summary`.

- [ ] **Step 2: Query the persisted results**

Run (via the `postgres` MCP tool or `psql`):

```sql
SELECT * FROM lab_summary
WHERE strategy = 'ensemble_kelly'
ORDER BY created_at DESC
LIMIT 1;
```

Record: OOS Sharpe, drawdown, and which `kelly_multiplier` won each fold (fold-level detail is in the run's persisted `lab_runs`/diagnostics JSONB — check `lab_runs.diagnostics` for the same fold-winner breakdown the `ensemble_ic` run recorded).

- [ ] **Step 3: Apply the verdict bar from the spec**

GO only if **all** of:
- OOS Sharpe > 1.12 (the SP500 core baseline)
- Drawdown no worse than -11%
- The winning `kelly_multiplier` is stable across a majority of the 17 folds

Otherwise: NO-GO.

- [ ] **Step 4: Record the result in the roadmap**

Add a dated bullet to `docs/roadmap.md`'s "In-Flight Tasks" changelog section (same place the June 26-29 entries live), in the format of the prior three verdicts, e.g.:

```
* **<date>, 2026**: Ran the Kelly-criterion position-sizing experiment (`ensemble_kelly`, kelly_multiplier swept 0.25/0.5/1.0, 17 SP500 folds). <GO: Sharpe X.XX > 1.12, DD -X.X%, winning multiplier k=X stable in N/17 folds — deployed. | NO-GO: Sharpe X.XX <= 1.12 [and/or] DD worse than -11% [and/or] winning multiplier unstable (N/17 folds) — not deployed.>
```

Also update the "Roadmap at a Glance" table's "Future Research" row to reflect that Kelly sizing has now been tested (mirroring how the table already reflects the ML-filter and exit-rule verdicts).

- [ ] **Step 5: Commit**

```bash
git add docs/roadmap.md
git commit -m "docs(roadmap): record ensemble_kelly WFO verdict"
```
