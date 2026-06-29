# IC-Weighted Voting Ensemble (`ensemble_ic`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new `ensemble_ic` lab strategy that weights the validated 5-voter pool by each voter's trailing cross-sectional Spearman IC, leaving the going-live `ensemble` baseline untouched.

**Architecture:** Three focused units — raw-value extractors (indicators.py), a pure `ic_weights` module (causal trailing-window IC → quarterly weight schedule), and an `EnsembleICSignal` strategy that thresholds the IC-weighted vote sum. Weights are a pure function of past data (leak-safe by construction); the harness/WFO are not modified.

**Tech Stack:** Python 3, pandas, numpy, vectorbt (via existing simulate layer), pytest. Native `.venv` for research runs (Docker is live-only).

## Global Constraints

- Absolute imports from `ggTrader` (no relative imports in package code).
- Vectorization-first: no per-symbol Python loops over price data; a loop over the ~quarterly rebalance dates is acceptable.
- Strict ruff: keep lines within the project limit; run `ruff check` before each commit.
- Tests live under `tests/lab/`; synthetic-OHLCV fixture pattern follows `tests/lab/test_ensemble.py` (`(symbol, field)` MultiIndex columns, `freq="B"`, tz `UTC`).
- The validated 5-voter set is `FIVE_VOTERS = ("bb", "rsi", "ema", "macd", "vbb")` (= `DEFAULT_VOTERS`), exported from `ggTrader.lab.strategies.ensemble`.
- **Directional convention:** every raw-value extractor is oriented so a higher value = a more bullish / deeper-oversold reading, so a positive IC means the voter is predictive.
- Do NOT modify `EnsembleSignal` or `EnsembleConviction Signal` — the baseline is going live.
- Run tests with: `.venv/bin/python -m pytest tests/lab/<file> -v`

---

### Task 1: Raw-value extractors

**Files:**
- Modify: `src/ggTrader/lab/strategies/indicators.py` (append 5 functions)
- Test: `tests/lab/test_raw_values.py` (create)

**Interfaces:**
- Produces: `rsi_raw(close, period) -> DataFrame`, `bb_raw(close, period, std) -> DataFrame`, `ema_raw(close, fast, slow) -> DataFrame`, `macd_raw(close, fast, slow, signal_period) -> DataFrame`, `vbb_raw(close, volume, period, std, vol_period) -> DataFrame`. All `(time × symbol)` float, unclipped, point-in-time, higher = more bullish.

- [ ] **Step 1: Write the failing test**

```python
# tests/lab/test_raw_values.py
"""Tests for raw-value extractors feeding the IC weighting."""

import numpy as np
import pandas as pd

from ggTrader.lab.strategies.indicators import (
    bb_raw,
    ema_raw,
    macd_raw,
    rsi_raw,
    vbb_raw,
)


def _idx(n, start="2020-01-01"):
    return pd.date_range(start, periods=n, freq="B", tz="UTC")


def _close(n=120, n_syms=4, seed=7):
    np.random.seed(seed)
    idx = _idx(n)
    cols = {}
    for i in range(n_syms):
        cols[f"S{i}"] = 100.0 * np.exp(np.cumsum(np.random.normal(0, 0.012, n)))
    return pd.DataFrame(cols, index=idx)


def _volume(close, seed=8):
    np.random.seed(seed)
    return pd.DataFrame(
        np.random.randint(1000, 9000, size=close.shape).astype(float),
        index=close.index,
        columns=close.columns,
    )


def test_rsi_raw_higher_when_more_oversold():
    """A monotonically falling series (oversold) ranks ABOVE a rising one."""
    idx = _idx(60)
    falling = pd.Series(np.linspace(100, 60, 60), index=idx)
    rising = pd.Series(np.linspace(60, 100, 60), index=idx)
    close = pd.DataFrame({"DOWN": falling, "UP": rising})
    raw = rsi_raw(close, period=14)
    assert raw["DOWN"].iloc[-1] > raw["UP"].iloc[-1]


def test_bb_raw_higher_below_lower_band():
    """Negated %b: a price far below its mean ranks above one at its mean."""
    close = _close()
    raw = bb_raw(close, period=20, std=2.0)
    assert raw.shape == close.shape
    assert raw.notna().any().any()


def test_ema_raw_positive_in_uptrend():
    idx = _idx(80)
    up = pd.Series(np.linspace(50, 150, 80), index=idx)
    close = pd.DataFrame({"UP": up})
    raw = ema_raw(close, fast=20, slow=50)
    assert raw["UP"].iloc[-1] > 0


def test_macd_raw_shape_and_finite():
    close = _close()
    raw = macd_raw(close, fast=12, slow=26, signal_period=9)
    assert raw.shape == close.shape
    assert np.isfinite(raw.iloc[-1].to_numpy()).all()


def test_vbb_raw_shape():
    close = _close()
    vol = _volume(close)
    raw = vbb_raw(close, vol, period=20, std=2.0, vol_period=20)
    assert raw.shape == close.shape
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/lab/test_raw_values.py -v`
Expected: FAIL with `ImportError: cannot import name 'rsi_raw'`

- [ ] **Step 3: Append the implementation to `indicators.py`**

```python
def rsi_raw(close: pd.DataFrame, period: int) -> pd.DataFrame:
    """Raw RSI level, NEGATED so higher = more oversold (bullish)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return -rsi


def bb_raw(close: pd.DataFrame, period: int, std: float) -> pd.DataFrame:
    """Negated Bollinger %b; higher = deeper below the lower band (bullish)."""
    sma = close.rolling(window=period, min_periods=period).mean()
    rolling_std = close.rolling(window=period, min_periods=period).std()
    upper = sma + std * rolling_std
    lower = sma - std * rolling_std
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    return -pct_b


def ema_raw(close: pd.DataFrame, fast: int, slow: int) -> pd.DataFrame:
    """Signed EMA gap (fast - slow) / slow; higher = stronger bullish trend."""
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    return (ema_f - ema_s) / ema_s.replace(0, np.nan)


def macd_raw(close: pd.DataFrame, fast: int, slow: int, signal_period: int) -> pd.DataFrame:
    """MACD histogram (macd - signal); higher = stronger bullish momentum."""
    macd_line = (
        close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    )
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    return macd_line - signal_line


def vbb_raw(
    close: pd.DataFrame, volume: pd.DataFrame, period: int, std: float, vol_period: int
) -> pd.DataFrame:
    """Negated %b scaled by volume ratio; higher = deep oversold on heavy volume."""
    sma = close.rolling(window=period, min_periods=period).mean()
    rolling_std = close.rolling(window=period, min_periods=period).std()
    upper = sma + std * rolling_std
    lower = sma - std * rolling_std
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    vol_avg = volume.rolling(window=vol_period, min_periods=vol_period).mean()
    vol_ratio = (volume / vol_avg.replace(0, np.nan)).clip(upper=3.0)
    return (-pct_b) * vol_ratio
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/lab/test_raw_values.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Lint + commit**

```bash
.venv/bin/ruff check src/ggTrader/lab/strategies/indicators.py tests/lab/test_raw_values.py
git add src/ggTrader/lab/strategies/indicators.py tests/lab/test_raw_values.py
git commit -m "feat(lab): raw-value extractors for IC weighting"
```

---

### Task 2: `forward_returns` + `daily_cross_sectional_ic`

**Files:**
- Create: `src/ggTrader/lab/strategies/ic_weights.py`
- Test: `tests/lab/test_ic_weights.py` (create)

**Interfaces:**
- Produces: `forward_returns(close: DataFrame, horizon: int = 3) -> DataFrame`; `daily_cross_sectional_ic(raw: DataFrame, fwd: DataFrame, min_names: int = 10) -> Series` (per-date mean-Spearman IC across symbols; NaN where < `min_names` valid pairs).
- Consumes (later): used by `ic_weight_schedule` (Task 3) and `EnsembleICSignal` (Task 4).

- [ ] **Step 1: Write the failing test**

```python
# tests/lab/test_ic_weights.py
"""Tests for the causal IC weight schedule."""

import numpy as np
import pandas as pd

from ggTrader.lab.strategies.ic_weights import (
    daily_cross_sectional_ic,
    forward_returns,
)


def _idx(n, start="2020-01-01"):
    return pd.date_range(start, periods=n, freq="B", tz="UTC")


def test_forward_returns_shifts_up_by_horizon():
    idx = _idx(5)
    close = pd.DataFrame({"A": [10.0, 11.0, 12.0, 13.0, 14.0]}, index=idx)
    fwd = forward_returns(close, horizon=1)
    # fwd[t] = close[t+1]/close[t]-1
    assert abs(fwd["A"].iloc[0] - (11.0 / 10.0 - 1.0)) < 1e-9
    assert pd.isna(fwd["A"].iloc[-1])  # no t+1 for the last bar


def test_daily_ic_perfect_positive_rank():
    """raw ranking that matches forward-return ranking gives IC ~ +1."""
    idx = _idx(1)
    raw = pd.DataFrame({"A": [1.0], "B": [2.0], "C": [3.0]}, index=idx)
    fwd = pd.DataFrame({"A": [0.01], "B": [0.02], "C": [0.03]}, index=idx)
    ic = daily_cross_sectional_ic(raw, fwd, min_names=3)
    assert abs(ic.iloc[0] - 1.0) < 1e-9


def test_daily_ic_perfect_negative_rank():
    idx = _idx(1)
    raw = pd.DataFrame({"A": [3.0], "B": [2.0], "C": [1.0]}, index=idx)
    fwd = pd.DataFrame({"A": [0.01], "B": [0.02], "C": [0.03]}, index=idx)
    ic = daily_cross_sectional_ic(raw, fwd, min_names=3)
    assert abs(ic.iloc[0] - (-1.0)) < 1e-9


def test_daily_ic_nan_below_min_names():
    idx = _idx(1)
    raw = pd.DataFrame({"A": [1.0], "B": [2.0]}, index=idx)
    fwd = pd.DataFrame({"A": [0.01], "B": [0.02]}, index=idx)
    ic = daily_cross_sectional_ic(raw, fwd, min_names=3)
    assert pd.isna(ic.iloc[0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/lab/test_ic_weights.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ggTrader.lab.strategies.ic_weights'`

- [ ] **Step 3: Create `ic_weights.py` with these two functions**

```python
"""Causal cross-sectional IC weighting for the ensemble_ic strategy.

All functions are pure functions of their inputs. forward_returns peeks ahead
by `horizon` bars BY DESIGN; the leak guard lives in ic_weight_schedule, which
never consumes a forward return that is not yet realized at a rebalance date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def forward_returns(close: pd.DataFrame, horizon: int = 3) -> pd.DataFrame:
    """Forward return close[t+horizon]/close[t] - 1 per (date, symbol)."""
    return close.shift(-horizon) / close - 1.0


def daily_cross_sectional_ic(
    raw: pd.DataFrame, fwd: pd.DataFrame, min_names: int = 10
) -> pd.Series:
    """Per-date Spearman rank IC across symbols between `raw` and `fwd`.

    Spearman = Pearson on per-row ranks. Days with fewer than `min_names`
    jointly-valid symbols return NaN.
    """
    raw, fwd = raw.align(fwd, join="inner")
    valid = raw.notna() & fwd.notna()
    n = valid.sum(axis=1)

    rr = raw.where(valid).rank(axis=1)
    fr = fwd.where(valid).rank(axis=1)
    rm = rr.sub(rr.mean(axis=1), axis=0)
    fm = fr.sub(fr.mean(axis=1), axis=0)
    cov = (rm * fm).sum(axis=1)
    denom = np.sqrt((rm**2).sum(axis=1) * (fm**2).sum(axis=1))
    ic = cov / denom.replace(0, np.nan)
    ic[n < min_names] = np.nan
    return ic
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/lab/test_ic_weights.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Lint + commit**

```bash
.venv/bin/ruff check src/ggTrader/lab/strategies/ic_weights.py tests/lab/test_ic_weights.py
git add src/ggTrader/lab/strategies/ic_weights.py tests/lab/test_ic_weights.py
git commit -m "feat(lab): forward returns + daily cross-sectional IC"
```

---

### Task 3: `ic_weight_schedule` (causal, quarterly, leak-safe)

**Files:**
- Modify: `src/ggTrader/lab/strategies/ic_weights.py` (append `ic_weight_schedule`)
- Test: `tests/lab/test_ic_weights.py` (append)

**Interfaces:**
- Consumes: `forward_returns`, `daily_cross_sectional_ic` (Task 2).
- Produces: `ic_weight_schedule(raw_by_voter: dict[str, DataFrame], close: DataFrame, *, lookback_months: int, horizon: int = 3, rebalance: str = "Q", min_names: int = 10) -> DataFrame` — a `(time × voter)` weight schedule, rows sum to 1.0, forward-filled between quarterly rebalances, equal weights during warmup / all-non-positive-IC.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/lab/test_ic_weights.py
from ggTrader.lab.strategies.ic_weights import ic_weight_schedule


def _ramp_close(n=900, n_syms=12, seed=3):
    np.random.seed(seed)
    idx = _idx(n)
    cols = {f"S{i}": 100.0 * np.exp(np.cumsum(np.random.normal(0, 0.01, n))) for i in range(n_syms)}
    return pd.DataFrame(cols, index=idx)


def test_weights_sum_to_one_each_row():
    close = _ramp_close()
    raw = {"a": close.pct_change(), "b": -close.pct_change()}
    w = ic_weight_schedule(raw, close, lookback_months=6)
    row_sums = w.sum(axis=1)
    assert np.allclose(row_sums.to_numpy(), 1.0, atol=1e-9)


def test_warmup_is_equal_weight():
    close = _ramp_close()
    raw = {"a": close.pct_change(), "b": -close.pct_change()}
    w = ic_weight_schedule(raw, close, lookback_months=6)
    # First row (no full trailing window yet) is equal-weight.
    assert np.allclose(w.iloc[0].to_numpy(), 0.5, atol=1e-9)


def test_all_nonpositive_ic_falls_back_to_equal():
    """Two voters that both anti-predict -> clip(0) zeros both -> equal weights."""
    close = _ramp_close()
    fwd = forward_returns(close, 3)
    anti = -fwd  # perfectly anti-correlated raw -> negative IC
    raw = {"a": anti, "b": anti}
    w = ic_weight_schedule(raw, close, lookback_months=6)
    assert np.allclose(w.iloc[-1].to_numpy(), 0.5, atol=1e-9)


def test_predictive_voter_gets_more_weight():
    """A voter whose raw == forward return should out-weight a noise voter."""
    close = _ramp_close()
    fwd = forward_returns(close, 3)
    np.random.seed(99)
    noise = pd.DataFrame(
        np.random.normal(size=close.shape), index=close.index, columns=close.columns
    )
    raw = {"good": fwd.fillna(0.0), "noise": noise}
    w = ic_weight_schedule(raw, close, lookback_months=6)
    assert w["good"].iloc[-1] > w["noise"].iloc[-1]


def test_truncation_invariance_leak_guard():
    """Weights up to date d are identical whether or not post-d rows exist."""
    close = _ramp_close()
    raw = {"a": close.pct_change(), "b": -close.pct_change()}
    d = close.index[600]
    full = ic_weight_schedule(raw, close, lookback_months=6)
    raw_trunc = {k: v.loc[:d] for k, v in raw.items()}
    trunc = ic_weight_schedule(raw_trunc, close.loc[:d], lookback_months=6)
    aligned = full.loc[:d]
    assert np.allclose(aligned.to_numpy(), trunc.to_numpy(), atol=1e-9, equal_nan=True)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/lab/test_ic_weights.py -k schedule_or_weights -v` (or run the whole file)
Expected: FAIL with `ImportError: cannot import name 'ic_weight_schedule'`

- [ ] **Step 3: Append `ic_weight_schedule` to `ic_weights.py`**

```python
def ic_weight_schedule(
    raw_by_voter: dict[str, pd.DataFrame],
    close: pd.DataFrame,
    *,
    lookback_months: int,
    horizon: int = 3,
    rebalance: str = "Q",
    min_names: int = 10,
) -> pd.DataFrame:
    """Causal (time x voter) weight schedule, recomputed each rebalance date.

    At each rebalance date t_k: take the trailing `lookback_months` window,
    DROP its last `horizon` bars (their forward returns are not realized by
    t_k -> the leak guard), average each voter's daily IC over the window, and
    set w_j = max(0, IC_j) / sum_k max(0, IC_k). Warmup and all-non-positive
    windows fall back to equal weights. Weights forward-fill until the next
    rebalance date.
    """
    voters = list(raw_by_voter)
    eq = 1.0 / len(voters)
    fwd = forward_returns(close, horizon)
    daily_ic = pd.DataFrame(
        {j: daily_cross_sectional_ic(raw_by_voter[j], fwd, min_names) for j in voters}
    )

    index = close.index
    rebal_dates = pd.Series(index, index=index).resample(rebalance).last().dropna()
    lookback = pd.DateOffset(months=lookback_months)

    weights = pd.DataFrame(index=index, columns=voters, dtype=float)
    for t_k in rebal_dates:
        window_start = t_k - lookback
        cutoff = index[index <= t_k]
        # leak guard: last usable bar is `horizon` bars before t_k
        usable_end = cutoff[-(horizon + 1)] if len(cutoff) > horizon else None
        if usable_end is None or window_start < index[0]:
            w = pd.Series(eq, index=voters)  # warmup: not a full window yet
        else:
            win = daily_ic.loc[
                (daily_ic.index > window_start) & (daily_ic.index <= usable_end)
            ]
            ic = win.mean()
            pos = ic.clip(lower=0.0)
            total = pos.sum()
            w = pos / total if total > 0 else pd.Series(eq, index=voters)
        weights.loc[t_k:] = w.values

    return weights.fillna(eq)  # pre-first-rebalance rows -> equal weight
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/lab/test_ic_weights.py -v`
Expected: PASS (9 passed)

- [ ] **Step 5: Lint + commit**

```bash
.venv/bin/ruff check src/ggTrader/lab/strategies/ic_weights.py tests/lab/test_ic_weights.py
git add src/ggTrader/lab/strategies/ic_weights.py tests/lab/test_ic_weights.py
git commit -m "feat(lab): causal quarterly IC weight schedule with leak guard"
```

---

### Task 4: `EnsembleICSignal` strategy

**Files:**
- Create: `src/ggTrader/lab/strategies/ensemble_ic.py`
- Test: `tests/lab/test_ensemble_ic.py` (create)

**Interfaces:**
- Consumes: `*_signals` + `*_raw` (indicators.py), `ic_weight_schedule` (ic_weights.py), `FIVE_VOTERS`/`DEFAULT_VOTERS` (ensemble.py), `LabConfig`/`SignalTargets` (strategy.py), `extract_close`/`extract_volume` (indicators.py).
- Produces: `EnsembleICSignal(cfg, *, consensus_threshold=0.4, ic_lookback_months=12, min_agree_exit=2, ...)` with `name="ensemble_ic"`, `target_kind="signals"`, methods `select`, `to_targets`, `sweep_params`, `sweep_signals`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/lab/test_ensemble_ic.py
"""Tests for the IC-weighted EnsembleICSignal strategy."""

import numpy as np
import pandas as pd

import ggTrader.lab.strategies.ensemble_ic as eic_mod
from ggTrader.lab.strategies.ensemble import EnsembleSignal
from ggTrader.lab.strategies.ensemble_ic import EnsembleICSignal
from ggTrader.lab.strategy import LabConfig


def _idx(n, start="2020-01-01"):
    return pd.date_range(start, periods=n, freq="B", tz="UTC")


def _ohlcv(n=400, n_syms=12, seed=42):
    np.random.seed(seed)
    idx = _idx(n)
    frames = {}
    for i in range(n_syms):
        close = 100.0 * np.exp(np.cumsum(np.random.normal(0.0003, 0.015, n)))
        frames[f"S{i}"] = pd.DataFrame(
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


def test_reduces_to_baseline_under_equal_weights(monkeypatch):
    """Equal weights + threshold 0.4 == baseline 2-of-5 entries."""
    ohlcv = _ohlcv()
    symbols = sorted(ohlcv.columns.get_level_values(0).unique())

    def _equal_weights(raw_by_voter, close, **kwargs):
        eq = 1.0 / len(raw_by_voter)
        return pd.DataFrame(eq, index=close.index, columns=list(raw_by_voter))

    monkeypatch.setattr(eic_mod, "ic_weight_schedule", _equal_weights)

    cfg = LabConfig(min_history_bars=50)
    ic = EnsembleICSignal(cfg, consensus_threshold=0.4)
    base = EnsembleSignal(cfg, min_agree=2)
    ic_t = ic.to_targets(_plans(ohlcv), ohlcv)
    base_t = base.to_targets(_plans(ohlcv), ohlcv)
    pd.testing.assert_frame_equal(ic_t.entries, base_t.entries)


def test_higher_threshold_is_subset_of_lower(monkeypatch):
    ohlcv = _ohlcv()

    def _equal_weights(raw_by_voter, close, **kwargs):
        eq = 1.0 / len(raw_by_voter)
        return pd.DataFrame(eq, index=close.index, columns=list(raw_by_voter))

    monkeypatch.setattr(eic_mod, "ic_weight_schedule", _equal_weights)
    cfg = LabConfig(min_history_bars=50)
    low = EnsembleICSignal(cfg, consensus_threshold=0.4).to_targets(_plans(ohlcv), ohlcv)
    high = EnsembleICSignal(cfg, consensus_threshold=0.8).to_targets(_plans(ohlcv), ohlcv)
    # every high-threshold entry is also a low-threshold entry
    assert (high.entries & ~low.entries).sum().sum() == 0


def test_exits_match_baseline():
    ohlcv = _ohlcv()
    cfg = LabConfig(min_history_bars=50)
    ic = EnsembleICSignal(cfg).to_targets(_plans(ohlcv), ohlcv)
    base = EnsembleSignal(cfg, min_agree=2).to_targets(_plans(ohlcv), ohlcv)
    pd.testing.assert_frame_equal(ic.exits, base.exits)


def test_to_targets_truncation_invariance():
    """Entries up to date d unchanged when post-d rows are removed (leak test)."""
    ohlcv = _ohlcv()
    cfg = LabConfig(min_history_bars=50)
    strat = EnsembleICSignal(cfg, ic_lookback_months=6)
    full = strat.to_targets(_plans(ohlcv), ohlcv)
    d = ohlcv.index[300]
    trunc = strat.to_targets(_plans(ohlcv.loc[:d]), ohlcv.loc[:d])
    pd.testing.assert_frame_equal(full.entries.loc[:d], trunc.entries.loc[:d])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/lab/test_ensemble_ic.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ggTrader.lab.strategies.ensemble_ic'`

- [ ] **Step 3: Create `ensemble_ic.py`**

```python
"""IC-weighted voting ensemble: weight the 5-voter pool by trailing Spearman IC."""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from ggTrader.lab.strategies.ensemble import DEFAULT_VOTERS, _validate_voters
from ggTrader.lab.strategies.ic_weights import ic_weight_schedule
from ggTrader.lab.strategies.indicators import (
    bb_raw,
    bb_signals,
    ema_raw,
    ema_signals,
    eligible_symbols,
    extract_close,
    extract_volume,
    macd_raw,
    macd_signals,
    rsi_raw,
    rsi_signals,
    vbb_raw,
    volume_bb_signals,
)
from ggTrader.lab.strategy import LabConfig, Plan, SignalTargets


class EnsembleICSignal:
    """Enter when the IC-weighted sum of voter entries clears a consensus threshold.

    Weights come from a causal trailing-window cross-sectional Spearman IC,
    recomputed quarterly (see ic_weights.ic_weight_schedule). Exits reuse the
    baseline EnsembleSignal logic. The validated EnsembleSignal is untouched.
    """

    name = "ensemble_ic"
    target_kind = "signals"

    def __init__(
        self,
        cfg: LabConfig,
        *,
        consensus_threshold: float = 0.4,
        ic_lookback_months: int = 12,
        ic_horizon: int = 3,
        ic_rebalance: str = "Q",
        ic_min_names: int = 10,
        min_agree_exit: int = 2,
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
        td_stop: int | None = None,
        exits_enabled: bool = True,
        voters: tuple[str, ...] | list[str] = DEFAULT_VOTERS,
    ) -> None:
        self.voters = _validate_voters(voters)
        self.cfg = cfg
        self.consensus_threshold = consensus_threshold
        self.ic_lookback_months = ic_lookback_months
        self.ic_horizon = ic_horizon
        self.ic_rebalance = ic_rebalance
        self.ic_min_names = ic_min_names
        self.min_agree_exit = min_agree_exit
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
        self.td_stop = td_stop
        self.exits_enabled = exits_enabled

    @classmethod
    def sweep_params(cls) -> dict[str, list]:
        # Only the two IC axes are swept; indicator params pinned at baseline.
        return {
            "consensus_threshold": [0.3, 0.4, 0.5, 0.6, 0.7],
            "ic_lookback_months": [3, 6, 12],
        }

    def select(self, asof: pd.Timestamp, data: pd.DataFrame, eligible: List[str]) -> Plan:
        data = data.loc[:asof]
        return [
            {"symbol": s, "weight": 0.0}
            for s in eligible_symbols(data, eligible, self.cfg.min_history_bars)
        ]

    def _entries_exits_raw(self, close: pd.DataFrame, volume: pd.DataFrame):
        ent: Dict[str, pd.DataFrame] = {}
        ext: Dict[str, pd.DataFrame] = {}
        raw: Dict[str, pd.DataFrame] = {}
        if "bb" in self.voters:
            ent["bb"], ext["bb"] = bb_signals(close, self.bb_period, self.bb_std)
            raw["bb"] = bb_raw(close, self.bb_period, self.bb_std)
        if "rsi" in self.voters:
            ent["rsi"], ext["rsi"] = rsi_signals(
                close, self.rsi_period, self.rsi_oversold, self.rsi_exit
            )
            raw["rsi"] = rsi_raw(close, self.rsi_period)
        if "ema" in self.voters:
            ent["ema"], ext["ema"] = ema_signals(close, self.ema_fast, self.ema_slow)
            raw["ema"] = ema_raw(close, self.ema_fast, self.ema_slow)
        if "macd" in self.voters:
            ent["macd"], ext["macd"] = macd_signals(
                close, self.macd_fast, self.macd_slow, self.macd_signal, self.divergence_window
            )
            raw["macd"] = macd_raw(close, self.macd_fast, self.macd_slow, self.macd_signal)
        if "vbb" in self.voters:
            ent["vbb"], ext["vbb"] = volume_bb_signals(
                close, volume, self.bb_period, self.bb_std, self.vol_period, self.vol_mult
            )
            raw["vbb"] = vbb_raw(close, volume, self.bb_period, self.bb_std, self.vol_period)
        return ent, ext, raw

    def _apply_time_stop(self, entries: pd.DataFrame, exits: pd.DataFrame) -> pd.DataFrame:
        if self.td_stop is None:
            return exits.astype(bool)
        timed = entries.shift(self.td_stop, fill_value=False).astype(bool)
        return (exits.astype(bool) | timed).astype(bool)

    def _generate_signals(self, close: pd.DataFrame, volume: pd.DataFrame) -> SignalTargets:
        ent, ext, raw = self._entries_exits_raw(close, volume)

        weights = ic_weight_schedule(
            raw,
            close,
            lookback_months=self.ic_lookback_months,
            horizon=self.ic_horizon,
            rebalance=self.ic_rebalance,
            min_names=self.ic_min_names,
        )
        # weighted_score[d, s] = sum_j w_j[d] * ent_j[d, s]; rows of w sum to 1.
        score = sum(ent[j].astype(float).mul(weights[j], axis=0) for j in ent)
        entries = (score >= self.consensus_threshold).astype(bool)

        exit_votes = sum(df.astype(int) for df in ext.values())
        if self.exits_enabled:
            independent_exit = ext["rsi"] if "rsi" in ext else False
            exits = independent_exit | (exit_votes >= self.min_agree_exit)
        else:
            exits = pd.DataFrame(False, index=entries.index, columns=entries.columns)
        exits = self._apply_time_stop(entries, exits)
        return SignalTargets(entries=entries, exits=exits.astype(bool))

    def to_targets(self, plans: Dict[pd.Timestamp, Plan], data: pd.DataFrame) -> SignalTargets:
        symbols = sorted({s["symbol"] for plan in plans.values() for s in plan})
        close = extract_close(data, symbols)
        volume = extract_volume(data, symbols)
        return self._generate_signals(close, volume)

    def sweep_signals(
        self, combos: list[dict], symbols: list[str], data: pd.DataFrame
    ) -> dict[str, SignalTargets]:
        from ggTrader.lab.sweep import combo_name

        close = extract_close(data, symbols)
        volume = extract_volume(data, symbols)
        result: dict[str, SignalTargets] = {}
        for combo in combos:
            strat = EnsembleICSignal(
                self.cfg,
                consensus_threshold=float(
                    combo.get("consensus_threshold", self.consensus_threshold)
                ),
                ic_lookback_months=int(combo.get("ic_lookback_months", self.ic_lookback_months)),
                min_agree_exit=int(combo.get("min_agree_exit", self.min_agree_exit)),
                voters=self.voters,
            )
            result[combo_name(self.name, combo)] = strat._generate_signals(close, volume)
        return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/lab/test_ensemble_ic.py -v`
Expected: PASS (4 passed). If `test_reduces_to_baseline_under_equal_weights` fails, check that the baseline default voter set is `FIVE_VOTERS` and `min_agree=2` matches threshold `0.4 == 2/5`.

- [ ] **Step 5: Lint + commit**

```bash
.venv/bin/ruff check src/ggTrader/lab/strategies/ensemble_ic.py tests/lab/test_ensemble_ic.py
git add src/ggTrader/lab/strategies/ensemble_ic.py tests/lab/test_ensemble_ic.py
git commit -m "feat(lab): EnsembleICSignal IC-weighted voting strategy"
```

---

### Task 5: Register `ensemble_ic` and wire the CLI

**Files:**
- Modify: `src/ggTrader/lab/strategies/__init__.py`
- Modify: `src/ggTrader/lab/strategies/signals.py:605-648`
- Test: `tests/lab/test_ensemble_ic.py` (append wiring test)

**Interfaces:**
- Consumes: `EnsembleICSignal` (Task 4), `build_signal_strategy` / `SIGNAL_STRATEGY_NAMES` (signals.py), `STRATEGY_REGISTRY` (__init__.py).
- Produces: `ensemble_ic` resolvable via `build_signal_strategy("ensemble_ic", cfg)` and present in `SIGNAL_STRATEGY_NAMES` and `STRATEGY_REGISTRY`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/lab/test_ensemble_ic.py
from ggTrader.lab.strategies import STRATEGY_REGISTRY
from ggTrader.lab.strategies.signals import SIGNAL_STRATEGY_NAMES, build_signal_strategy


def test_ensemble_ic_registered():
    assert "ensemble_ic" in SIGNAL_STRATEGY_NAMES
    assert "ensemble_ic" in STRATEGY_REGISTRY
    cfg = LabConfig(min_history_bars=50)
    strat = build_signal_strategy("ensemble_ic", cfg)
    assert strat.name == "ensemble_ic"
    assert strat.target_kind == "signals"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/lab/test_ensemble_ic.py::test_ensemble_ic_registered -v`
Expected: FAIL — `assert "ensemble_ic" in SIGNAL_STRATEGY_NAMES`

- [ ] **Step 3: Register in both files**

In `src/ggTrader/lab/strategies/__init__.py`: add the import and registry/exports entries next to the existing `ensemble` ones.

```python
# import line (next to: from .ensemble import EnsembleConvictionSignal, EnsembleSignal)
from .ensemble_ic import EnsembleICSignal

# inside STRATEGY_REGISTRY dict, after "ensemble": EnsembleSignal,
    "ensemble_ic": EnsembleICSignal,

# inside __all__, after "EnsembleSignal",
    "EnsembleICSignal",
```

In `src/ggTrader/lab/strategies/signals.py` (within `build_signal_strategy`'s local map near line 605, and the `SIGNAL_STRATEGY_NAMES` tuple near line 631):

```python
# in the import inside build_signal_strategy
    from ggTrader.lab.strategies.ensemble_ic import EnsembleICSignal

# in the local builders map, after "ensemble": EnsembleSignal,
        "ensemble_ic": EnsembleICSignal,

# in SIGNAL_STRATEGY_NAMES tuple, after "ensemble",
    "ensemble_ic",
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/lab/test_ensemble_ic.py::test_ensemble_ic_registered -v`
Expected: PASS

- [ ] **Step 5: Full lab test sweep + lint + commit**

```bash
.venv/bin/python -m pytest tests/lab -q
.venv/bin/ruff check src/ggTrader/lab
git add src/ggTrader/lab/strategies/__init__.py src/ggTrader/lab/strategies/signals.py tests/lab/test_ensemble_ic.py
git commit -m "feat(lab): register ensemble_ic strategy + CLI wiring"
```

---

### Task 6: WFO validation run (Go/No-Go)

**Files:** none (research run + result capture).

This task produces the decision, not code. It is a manual run; record the output.

- [ ] **Step 1: Confirm the baseline still passes end-to-end**

Run: `.venv/bin/python -m pytest tests/lab -q`
Expected: all green (no regressions in the existing `ensemble` tests).

- [ ] **Step 2: Run the gate-honest WFO on SP500**

Run (native `.venv`, per project convention — Docker is live-only):
```bash
.venv/bin/python ggt.py lab --strategy ensemble_ic --universe sp500 --wfo
```
(Match the exact universe/date flags used for the 1.12 baseline run; mirror them from the most recent `ensemble` WFO invocation so the comparison is apples-to-apples.)

- [ ] **Step 3: Capture and compare**

Record OOS Sharpe, CAGR, max drawdown, fold pass rate, and the DSR verdict (accounting for the 2 swept axes: `consensus_threshold`, `ic_lookback_months`). Compare against the baseline (Sharpe 1.12 / CAGR 16.3% / DD -11% / 16-of-17).

- [ ] **Step 4: Decision**

- **GO** only if `ensemble_ic` beats 1.12 OOS Sharpe with the DSR gate still passing under the wider search.
- **NO-GO** otherwise → per report §5/§6, the equity selection book is closed; record the result in a `project_ensemble_ic_*` memory + roadmap, and do not deploy. No live deploy in either case until it also beats the live baseline with statistical significance.

---

## Self-Review

**Spec coverage:** raw values (T1) ✓; causal trailing IC + estimator + leak guard (T2–T3) ✓; `EnsembleICSignal` entry-weighting + baseline exits + reduces-to-baseline (T4) ✓; registration/CLI (T5) ✓; WFO validation + Go/No-Go (T6) ✓; clustering explicitly deferred (non-goal) ✓.

**Placeholder scan:** none — every code/test step is complete.

**Type consistency:** `ic_weight_schedule` signature identical in Tasks 3, 4; `daily_cross_sectional_ic` / `forward_returns` signatures identical in Tasks 2, 3; `*_raw` names match Tasks 1 and 4; registry symbol `EnsembleICSignal` consistent across Tasks 4–5.
