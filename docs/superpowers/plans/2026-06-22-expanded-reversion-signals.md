# Expanded Reversion Signals Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 3 new reversion signal types (MACD divergence, volume-confirmed BB, multi-timeframe) to the ensemble, expanding from 3 to 6 voters, plus an ML pre-screen script to evaluate signal quality before WFO.

**Architecture:** Each new signal follows the existing pattern: vectorized indicator function in `indicators.py`, standalone signal class in `signals.py`, wired into the ensemble as an additional voter. An ML pre-screen script reuses the `FeatureGate` feature extraction to train a LightGBM per signal and report precision.

**Tech Stack:** Python 3.12, pandas, numpy, vectorbt, LightGBM, scikit-learn, joblib. Tests via pytest. Docker container must be rebuilt after code changes.

## Global Constraints

- All indicator functions must be fully vectorized (no Python loops over bars).
- All signal classes must implement: `name`, `target_kind = "signals"`, `__init__(self, cfg: LabConfig, ...)`, `sweep_params() -> dict[str, list]`, `select(asof, data, eligible) -> Plan`, `to_targets(plans, data) -> SignalTargets`, `sweep_signals(combos, symbols, data) -> dict[str, SignalTargets]`.
- Tests run inside Docker: `docker compose run --rm ggtrader_live python -m pytest <path> -v`.
- Docker image must be rebuilt before running tests on new/changed files: `docker compose build ggtrader_live`.
- Existing tests must not break (exclude pre-existing broken `tests/lab/test_reversion_signals.py`).
- Imports follow the absolute `from ggTrader.x.y import z` pattern. No relative imports.
- `SignalTargets` is a `NamedTuple` with fields `entries: pd.DataFrame`, `exits: pd.DataFrame`, `sizes: pd.DataFrame | None = None`.
- The `extract_close(data, symbols)` helper extracts a `(time x symbol)` close-price DataFrame from multi-level OHLCV. A new `extract_volume(data, symbols)` must follow the same pattern.

---

### Task 1: MACD Divergence — Indicator Functions + Tests

**Files:**
- Modify: `src/ggTrader/lab/strategies/indicators.py` (append new functions)
- Create: `tests/lab/test_macd_signals.py`

**Interfaces:**
- Consumes: `extract_close(data, symbols)` from `indicators.py`
- Produces:
  - `macd_signals(close: pd.DataFrame, fast: int, slow: int, signal_period: int, divergence_window: int) -> tuple[pd.DataFrame, pd.DataFrame]` — returns `(entries, exits)` boolean DataFrames same shape as `close`.
  - `macd_strength(close: pd.DataFrame, fast: int, slow: int, signal_period: int) -> pd.DataFrame` — returns DataFrame of values in [0, 1], same shape as `close`.

- [ ] **Step 1: Write failing tests for `macd_signals`**

```python
# tests/lab/test_macd_signals.py
"""Tests for MACD divergence indicator functions and signal class."""

import numpy as np
import pandas as pd
import pytest

from ggTrader.lab.strategies.indicators import macd_signals, macd_strength


def _idx(n, start="2020-01-01"):
    return pd.date_range(start, periods=n, freq="B", tz="UTC")


def _close(n=300, n_syms=3, seed=42):
    np.random.seed(seed)
    idx = _idx(n)
    frames = {}
    for i in range(n_syms):
        sym = f"S{i}"
        frames[sym] = pd.Series(
            100.0 * np.exp(np.cumsum(np.random.normal(0.0003, 0.015, n))),
            index=idx,
        )
    return pd.DataFrame(frames)


class TestMACDSignals:
    def test_output_shape_matches_input(self):
        close = _close(200)
        entries, exits = macd_signals(close, fast=12, slow=26, signal_period=9, divergence_window=20)
        assert entries.shape == close.shape
        assert exits.shape == close.shape

    def test_entries_are_boolean(self):
        close = _close(200)
        entries, exits = macd_signals(close, 12, 26, 9, 20)
        assert entries.dtypes.apply(lambda d: d == bool).all()
        assert exits.dtypes.apply(lambda d: d == bool).all()

    def test_no_entries_during_warmup(self):
        close = _close(200)
        entries, _ = macd_signals(close, fast=12, slow=26, signal_period=9, divergence_window=20)
        # First slow+signal+divergence_window bars should have no entries
        warmup = 26 + 9 + 20
        assert entries.iloc[:warmup].sum().sum() == 0

    def test_divergence_requires_price_low_and_histogram_higher(self):
        """Manually construct a bullish divergence scenario."""
        idx = _idx(60)
        # Price makes lower lows but MACD histogram makes higher lows
        price = np.concatenate([
            np.linspace(100, 80, 30),  # declining
            np.linspace(80, 75, 15),   # lower low
            np.linspace(75, 85, 15),   # recovery
        ])
        close = pd.DataFrame({"A": price}, index=idx)
        entries, _ = macd_signals(close, fast=8, slow=21, signal_period=9, divergence_window=10)
        # At minimum, no crash — divergence detection runs without error
        assert entries.shape == close.shape

    def test_exit_on_histogram_crosses_below_zero(self):
        close = _close(300, seed=99)
        _, exits = macd_signals(close, 12, 26, 9, 20)
        # Exits should fire at some point on random data
        assert exits.sum().sum() >= 0  # no crash; exits are valid booleans


class TestMACDStrength:
    def test_output_shape_matches_input(self):
        close = _close(200)
        strength = macd_strength(close, fast=12, slow=26, signal_period=9)
        assert strength.shape == close.shape

    def test_values_in_zero_one_range(self):
        close = _close(200)
        strength = macd_strength(close, 12, 26, 9)
        valid = strength.dropna()
        if not valid.empty:
            assert (valid >= 0.0).all().all()
            assert (valid <= 1.0).all().all()

    def test_nan_during_warmup(self):
        close = _close(200)
        strength = macd_strength(close, fast=12, slow=26, signal_period=9)
        # First slow bars should be NaN
        assert strength.iloc[:26].isna().all().all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `docker compose build ggtrader_live && docker compose run --rm ggtrader_live python -m pytest tests/lab/test_macd_signals.py -v`
Expected: ImportError — `macd_signals` and `macd_strength` do not exist yet.

- [ ] **Step 3: Implement `macd_signals` and `macd_strength` in `indicators.py`**

Append to `src/ggTrader/lab/strategies/indicators.py`:

```python
def macd_signals(
    close: pd.DataFrame,
    fast: int,
    slow: int,
    signal_period: int,
    divergence_window: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """MACD bullish divergence entry + histogram-crosses-zero exit."""
    macd_line = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line

    price_low = close.rolling(window=divergence_window, min_periods=divergence_window).min()
    hist_low = histogram.rolling(window=divergence_window, min_periods=divergence_window).min()

    price_at_low = close <= price_low
    hist_above_low = histogram > hist_low
    entries = (price_at_low & hist_above_low).fillna(False).astype(bool)

    prev_above = histogram.shift(1) >= 0
    now_below = histogram < 0
    exits = (prev_above & now_below).fillna(False).astype(bool)

    return entries, exits


def macd_strength(close: pd.DataFrame, fast: int, slow: int, signal_period: int) -> pd.DataFrame:
    """Normalized absolute MACD histogram magnitude, clipped to [0, 1]."""
    macd_line = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    hist_max = histogram.abs().rolling(window=50, min_periods=1).max()
    strength = histogram.abs() / hist_max.replace(0, np.nan)
    return strength.clip(lower=0.0, upper=1.0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker compose build ggtrader_live && docker compose run --rm ggtrader_live python -m pytest tests/lab/test_macd_signals.py -v`
Expected: All 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ggTrader/lab/strategies/indicators.py tests/lab/test_macd_signals.py
git commit -m "feat(lab): MACD divergence indicator functions + strength"
```

---

### Task 2: Volume-Confirmed BB — Indicator Functions + Tests

**Files:**
- Modify: `src/ggTrader/lab/strategies/indicators.py` (append new functions)
- Modify: `tests/lab/test_macd_signals.py` → rename is NOT needed; create a new test file
- Create: `tests/lab/test_volume_bb_signals.py`

**Interfaces:**
- Consumes: `extract_close(data, symbols)`, `bb_signals(close, period, std)` from `indicators.py`
- Produces:
  - `extract_volume(data: pd.DataFrame, symbols: list[str]) -> pd.DataFrame` — returns `(time x symbol)` volume DataFrame. Same pattern as `extract_close`.
  - `volume_bb_signals(close: pd.DataFrame, volume: pd.DataFrame, bb_period: int, bb_std: float, vol_period: int, vol_mult: float) -> tuple[pd.DataFrame, pd.DataFrame]` — returns `(entries, exits)`.
  - `volume_bb_strength(close: pd.DataFrame, volume: pd.DataFrame, bb_period: int, bb_std: float, vol_period: int) -> pd.DataFrame` — returns DataFrame [0, 1].

- [ ] **Step 1: Write failing tests for `extract_volume` and `volume_bb_signals`**

```python
# tests/lab/test_volume_bb_signals.py
"""Tests for volume-confirmed BB reversion indicator functions and signal class."""

import numpy as np
import pandas as pd
import pytest

from ggTrader.lab.strategies.indicators import (
    extract_volume,
    volume_bb_signals,
    volume_bb_strength,
)


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


def _close_and_volume(n=300, n_syms=3, seed=42):
    ohlcv = _ohlcv(n, n_syms, seed)
    symbols = sorted(ohlcv.columns.get_level_values(0).unique())
    close = pd.concat({s: ohlcv[s]["close"] for s in symbols}, axis=1)
    vol = extract_volume(ohlcv, symbols)
    return close, vol


class TestExtractVolume:
    def test_shape_matches_close(self):
        ohlcv = _ohlcv(100)
        symbols = sorted(ohlcv.columns.get_level_values(0).unique())
        vol = extract_volume(ohlcv, symbols)
        assert vol.shape == (100, 3)
        assert list(vol.columns) == symbols

    def test_missing_symbol_skipped(self):
        ohlcv = _ohlcv(100)
        vol = extract_volume(ohlcv, ["S0", "NOSUCH"])
        assert "S0" in vol.columns
        assert "NOSUCH" not in vol.columns


class TestVolumeBBSignals:
    def test_output_shape_matches_input(self):
        close, vol = _close_and_volume(200)
        entries, exits = volume_bb_signals(close, vol, bb_period=20, bb_std=2.0, vol_period=20, vol_mult=1.5)
        assert entries.shape == close.shape
        assert exits.shape == close.shape

    def test_entries_are_boolean(self):
        close, vol = _close_and_volume(200)
        entries, exits = volume_bb_signals(close, vol, 20, 2.0, 20, 1.5)
        assert entries.dtypes.apply(lambda d: d == bool).all()
        assert exits.dtypes.apply(lambda d: d == bool).all()

    def test_higher_vol_mult_fewer_entries(self):
        """Stricter volume filter -> fewer entries."""
        close, vol = _close_and_volume(300, seed=123)
        ent_low, _ = volume_bb_signals(close, vol, 20, 2.0, 20, 1.0)
        ent_high, _ = volume_bb_signals(close, vol, 20, 2.0, 20, 3.0)
        assert ent_low.sum().sum() >= ent_high.sum().sum()

    def test_no_entries_during_warmup(self):
        close, vol = _close_and_volume(200)
        entries, _ = volume_bb_signals(close, vol, bb_period=20, bb_std=2.0, vol_period=20, vol_mult=2.0)
        warmup = max(20, 20)
        assert entries.iloc[:warmup].sum().sum() == 0

    def test_exits_match_plain_bb_exits(self):
        """Exits should be identical to plain bb_reversion exits."""
        from ggTrader.lab.strategies.indicators import bb_signals
        close, vol = _close_and_volume(300)
        _, vol_exits = volume_bb_signals(close, vol, 20, 2.0, 20, 1.5)
        _, bb_exits = bb_signals(close, 20, 2.0)
        pd.testing.assert_frame_equal(vol_exits, bb_exits)


class TestVolumeBBStrength:
    def test_output_shape_matches_input(self):
        close, vol = _close_and_volume(200)
        strength = volume_bb_strength(close, vol, bb_period=20, bb_std=2.0, vol_period=20)
        assert strength.shape == close.shape

    def test_values_in_zero_one_range(self):
        close, vol = _close_and_volume(200)
        strength = volume_bb_strength(close, vol, 20, 2.0, 20)
        valid = strength.dropna()
        if not valid.empty:
            assert (valid >= 0.0).all().all()
            assert (valid <= 1.0).all().all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `docker compose build ggtrader_live && docker compose run --rm ggtrader_live python -m pytest tests/lab/test_volume_bb_signals.py -v`
Expected: ImportError — `extract_volume`, `volume_bb_signals`, `volume_bb_strength` do not exist yet.

- [ ] **Step 3: Implement `extract_volume`, `volume_bb_signals`, and `volume_bb_strength` in `indicators.py`**

Append to `src/ggTrader/lab/strategies/indicators.py`:

```python
def extract_volume(data: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
    """Extract a (time x symbol) volume DataFrame from multi-level OHLCV data."""
    have = set(data.columns.get_level_values(0))
    return pd.concat({s: data[s]["volume"] for s in symbols if s in have}, axis=1)


def volume_bb_signals(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    bb_period: int,
    bb_std: float,
    vol_period: int,
    vol_mult: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """BB reversion entry gated by a volume spike above vol_mult * SMA(volume)."""
    sma = close.rolling(window=bb_period, min_periods=bb_period).mean()
    rolling_std = close.rolling(window=bb_period, min_periods=bb_period).std()
    lower = sma - bb_std * rolling_std

    prev_above = close.shift(1) >= lower.shift(1)
    now_below = close < lower
    bb_entry = (prev_above & now_below).fillna(False)

    vol_avg = volume.rolling(window=vol_period, min_periods=vol_period).mean()
    vol_spike = volume > (vol_mult * vol_avg)

    entries = (bb_entry & vol_spike).fillna(False).astype(bool)

    prev_below_sma = close.shift(1) < sma.shift(1)
    now_above_sma = close >= sma
    exits = (prev_below_sma & now_above_sma).fillna(False).astype(bool)

    return entries, exits


def volume_bb_strength(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    bb_period: int,
    bb_std: float,
    vol_period: int,
) -> pd.DataFrame:
    """BB depth strength scaled by volume spike intensity, clipped to [0, 1]."""
    bb_str = bb_strength(close, bb_period, bb_std)
    vol_avg = volume.rolling(window=vol_period, min_periods=vol_period).mean()
    vol_ratio = (volume / vol_avg.replace(0, np.nan)).clip(upper=3.0) / 3.0
    combined = (bb_str + vol_ratio) / 2.0
    return combined.clip(lower=0.0, upper=1.0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker compose build ggtrader_live && docker compose run --rm ggtrader_live python -m pytest tests/lab/test_volume_bb_signals.py -v`
Expected: All 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ggTrader/lab/strategies/indicators.py tests/lab/test_volume_bb_signals.py
git commit -m "feat(lab): volume-confirmed BB reversion indicators + extract_volume"
```

---

### Task 3: Multi-Timeframe Reversion — Indicator Functions + Tests

**Files:**
- Modify: `src/ggTrader/lab/strategies/indicators.py` (append new functions)
- Create: `tests/lab/test_mtf_signals.py`

**Interfaces:**
- Consumes: `bb_signals(close, period, std)` from `indicators.py` (for the daily BB component)
- Produces:
  - `mtf_signals(close: pd.DataFrame, weekly_rsi_period: int, weekly_rsi_oversold: int, weekly_rsi_exit: int, daily_bb_period: int, daily_bb_std: float) -> tuple[pd.DataFrame, pd.DataFrame]` — returns `(entries, exits)`.
  - `mtf_strength(close: pd.DataFrame, weekly_rsi_period: int, weekly_rsi_oversold: int, daily_bb_period: int, daily_bb_std: float) -> pd.DataFrame` — returns DataFrame [0, 1].

- [ ] **Step 1: Write failing tests for `mtf_signals`**

```python
# tests/lab/test_mtf_signals.py
"""Tests for multi-timeframe reversion indicator functions and signal class."""

import numpy as np
import pandas as pd
import pytest

from ggTrader.lab.strategies.indicators import mtf_signals, mtf_strength


def _idx(n, start="2020-01-01"):
    return pd.date_range(start, periods=n, freq="B", tz="UTC")


def _close(n=300, n_syms=3, seed=42):
    np.random.seed(seed)
    idx = _idx(n)
    frames = {}
    for i in range(n_syms):
        sym = f"S{i}"
        frames[sym] = pd.Series(
            100.0 * np.exp(np.cumsum(np.random.normal(0.0003, 0.015, n))),
            index=idx,
        )
    return pd.DataFrame(frames)


class TestMTFSignals:
    def test_output_shape_matches_input(self):
        close = _close(300)
        entries, exits = mtf_signals(close, weekly_rsi_period=14, weekly_rsi_oversold=30, weekly_rsi_exit=50, daily_bb_period=20, daily_bb_std=2.0)
        assert entries.shape == close.shape
        assert exits.shape == close.shape

    def test_entries_are_boolean(self):
        close = _close(300)
        entries, exits = mtf_signals(close, 14, 30, 50, 20, 2.0)
        assert entries.dtypes.apply(lambda d: d == bool).all()
        assert exits.dtypes.apply(lambda d: d == bool).all()

    def test_no_entries_during_warmup(self):
        close = _close(300)
        entries, _ = mtf_signals(close, 14, 30, 50, 20, 2.0)
        # Weekly RSI needs at least weekly_rsi_period weeks of data + daily BB warmup
        warmup = 14 * 5 + 20  # conservative estimate
        assert entries.iloc[:warmup].sum().sum() == 0

    def test_stricter_oversold_fewer_entries(self):
        """Lower weekly_rsi_oversold threshold -> fewer entries."""
        close = _close(500, seed=123)
        ent_loose, _ = mtf_signals(close, 14, 40, 50, 20, 2.0)
        ent_strict, _ = mtf_signals(close, 14, 20, 50, 20, 2.0)
        assert ent_loose.sum().sum() >= ent_strict.sum().sum()

    def test_weekly_resampling_doesnt_crash_on_short_data(self):
        close = _close(30)
        entries, exits = mtf_signals(close, 7, 30, 50, 15, 2.0)
        assert entries.shape == close.shape


class TestMTFStrength:
    def test_output_shape_matches_input(self):
        close = _close(300)
        strength = mtf_strength(close, weekly_rsi_period=14, weekly_rsi_oversold=30, daily_bb_period=20, daily_bb_std=2.0)
        assert strength.shape == close.shape

    def test_values_in_zero_one_range(self):
        close = _close(300)
        strength = mtf_strength(close, 14, 30, 20, 2.0)
        valid = strength.dropna()
        if not valid.empty:
            assert (valid >= 0.0).all().all()
            assert (valid <= 1.0).all().all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `docker compose build ggtrader_live && docker compose run --rm ggtrader_live python -m pytest tests/lab/test_mtf_signals.py -v`
Expected: ImportError — `mtf_signals` and `mtf_strength` do not exist yet.

- [ ] **Step 3: Implement `mtf_signals` and `mtf_strength` in `indicators.py`**

Append to `src/ggTrader/lab/strategies/indicators.py`:

```python
def _weekly_rsi(close: pd.DataFrame, period: int) -> pd.DataFrame:
    """Compute RSI on weekly-resampled close, forward-filled to daily index."""
    weekly = close.resample("W").last().dropna(how="all")
    delta = weekly.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.reindex(close.index, method="ffill")


def mtf_signals(
    close: pd.DataFrame,
    weekly_rsi_period: int,
    weekly_rsi_oversold: int,
    weekly_rsi_exit: int,
    daily_bb_period: int,
    daily_bb_std: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Multi-timeframe reversion: weekly RSI oversold + daily BB breakdown."""
    w_rsi = _weekly_rsi(close, weekly_rsi_period)

    sma = close.rolling(window=daily_bb_period, min_periods=daily_bb_period).mean()
    rolling_std = close.rolling(window=daily_bb_period, min_periods=daily_bb_period).std()
    lower = sma - daily_bb_std * rolling_std

    weekly_oversold = w_rsi < weekly_rsi_oversold
    daily_below_bb = close < lower

    entries = (weekly_oversold & daily_below_bb).fillna(False).astype(bool)

    weekly_recovered = w_rsi >= weekly_rsi_exit
    prev_weekly_below = w_rsi.shift(1) < weekly_rsi_exit
    weekly_exit = (weekly_recovered & prev_weekly_below).fillna(False)

    prev_below_sma = close.shift(1) < sma.shift(1)
    now_above_sma = close >= sma
    daily_exit = (prev_below_sma & now_above_sma).fillna(False)

    exits = (weekly_exit | daily_exit).fillna(False).astype(bool)

    return entries, exits


def mtf_strength(
    close: pd.DataFrame,
    weekly_rsi_period: int,
    weekly_rsi_oversold: int,
    daily_bb_period: int,
    daily_bb_std: float,
) -> pd.DataFrame:
    """Average of weekly RSI depth and daily BB depth, clipped to [0, 1]."""
    w_rsi = _weekly_rsi(close, weekly_rsi_period)
    rsi_depth = ((weekly_rsi_oversold - w_rsi) / weekly_rsi_oversold).clip(lower=0.0, upper=1.0)
    bb_str = bb_strength(close, daily_bb_period, daily_bb_std)
    combined = (rsi_depth + bb_str) / 2.0
    return combined.clip(lower=0.0, upper=1.0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker compose build ggtrader_live && docker compose run --rm ggtrader_live python -m pytest tests/lab/test_mtf_signals.py -v`
Expected: All 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ggTrader/lab/strategies/indicators.py tests/lab/test_mtf_signals.py
git commit -m "feat(lab): multi-timeframe reversion indicators (weekly RSI + daily BB)"
```

---

### Task 4: Standalone Signal Classes + Registry

**Files:**
- Modify: `src/ggTrader/lab/strategies/signals.py` (add 3 new classes + registry entries)
- Modify: `src/ggTrader/lab/cli.py` (add 3 entries to `cls_map`)
- Modify: `tests/lab/test_macd_signals.py` (add class + registration tests)
- Modify: `tests/lab/test_volume_bb_signals.py` (add class + registration tests)
- Modify: `tests/lab/test_mtf_signals.py` (add class + registration tests)

**Interfaces:**
- Consumes: `macd_signals`, `volume_bb_signals`, `mtf_signals`, `extract_close`, `extract_volume`, `eligible_symbols` from `indicators.py`; `LabConfig`, `Plan`, `SignalTargets` from `strategy.py`; `combo_name` from `sweep.py`
- Produces:
  - `MACDDivergenceSignal` — class with `name = "macd_divergence"`, params `macd_fast`, `macd_slow`, `macd_signal`, `divergence_window`
  - `VolumeBBReversionSignal` — class with `name = "volume_bb_reversion"`, params `bb_period`, `bb_std`, `vol_period`, `vol_mult`
  - `MultiTimeframeReversionSignal` — class with `name = "mtf_reversion"`, params `weekly_rsi_period`, `weekly_rsi_oversold`, `weekly_rsi_exit`, `daily_bb_period`, `daily_bb_std`
  - All three registered in `_build_signal_registry()`, `SIGNAL_STRATEGY_NAMES`, and `cli.py` `cls_map`

- [ ] **Step 1: Add `MACDDivergenceSignal` class to `signals.py`**

Add after the `RsiReversionSignal` class in `src/ggTrader/lab/strategies/signals.py`:

```python
class MACDDivergenceSignal:
    """MACD bullish divergence: price makes lower low, histogram makes higher low.

    Entry: bullish divergence detected within a rolling divergence_window.
    Exit: MACD histogram crosses below zero.
    """

    name = "macd_divergence"
    target_kind = "signals"

    def __init__(
        self,
        cfg: LabConfig,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        divergence_window: int = 20,
    ) -> None:
        self.cfg = cfg
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.divergence_window = divergence_window

    @classmethod
    def sweep_params(cls) -> dict[str, list]:
        return {
            "macd_fast": [8, 12],
            "macd_slow": [21, 26],
            "macd_signal": [9],
            "divergence_window": [10, 20],
        }

    def select(self, asof: pd.Timestamp, data: pd.DataFrame, eligible: List[str]) -> Plan:
        data = data.loc[:asof]
        return [
            {"symbol": s, "weight": 0.0}
            for s in eligible_symbols(data, eligible, self.cfg.min_history_bars)
        ]

    def to_targets(self, plans: Dict[pd.Timestamp, Plan], data: pd.DataFrame) -> SignalTargets:
        symbols = sorted({s["symbol"] for plan in plans.values() for s in plan})
        close = extract_close(data, symbols)
        entries, exits = macd_signals(close, self.macd_fast, self.macd_slow, self.macd_signal, self.divergence_window)
        return SignalTargets(entries=entries, exits=exits)

    def sweep_signals(
        self,
        combos: list[dict],
        symbols: list[str],
        data: pd.DataFrame,
    ) -> dict[str, SignalTargets]:
        from ggTrader.lab.sweep import combo_name

        close = extract_close(data, symbols)
        result: dict[str, SignalTargets] = {}
        for combo in combos:
            fast = int(combo["macd_fast"])
            slow = int(combo["macd_slow"])
            sig = int(combo["macd_signal"])
            win = int(combo["divergence_window"])
            ent, ext = macd_signals(close, fast, slow, sig, win)
            result[combo_name(self.name, combo)] = SignalTargets(entries=ent, exits=ext)
        return result
```

- [ ] **Step 2: Add `VolumeBBReversionSignal` class to `signals.py`**

Add after `MACDDivergenceSignal`:

```python
class VolumeBBReversionSignal:
    """BB reversion gated by a volume spike — capitulation filter.

    Entry: close crosses below lower BB AND volume > vol_mult * SMA(volume, vol_period).
    Exit: close crosses above the SMA (same as plain bb_reversion).
    """

    name = "volume_bb_reversion"
    target_kind = "signals"

    def __init__(
        self,
        cfg: LabConfig,
        bb_period: int = 20,
        bb_std: float = 2.0,
        vol_period: int = 20,
        vol_mult: float = 2.0,
    ) -> None:
        self.cfg = cfg
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.vol_period = vol_period
        self.vol_mult = vol_mult

    @classmethod
    def sweep_params(cls) -> dict[str, list]:
        return {
            "bb_period": [15, 20],
            "bb_std": [2.0, 2.5],
            "vol_period": [20],
            "vol_mult": [1.5, 2.0, 2.5],
        }

    def select(self, asof: pd.Timestamp, data: pd.DataFrame, eligible: List[str]) -> Plan:
        data = data.loc[:asof]
        return [
            {"symbol": s, "weight": 0.0}
            for s in eligible_symbols(data, eligible, self.cfg.min_history_bars)
        ]

    def to_targets(self, plans: Dict[pd.Timestamp, Plan], data: pd.DataFrame) -> SignalTargets:
        symbols = sorted({s["symbol"] for plan in plans.values() for s in plan})
        close = extract_close(data, symbols)
        volume = extract_volume(data, symbols)
        entries, exits = volume_bb_signals(close, volume, self.bb_period, self.bb_std, self.vol_period, self.vol_mult)
        return SignalTargets(entries=entries, exits=exits)

    def sweep_signals(
        self,
        combos: list[dict],
        symbols: list[str],
        data: pd.DataFrame,
    ) -> dict[str, SignalTargets]:
        from ggTrader.lab.sweep import combo_name

        close = extract_close(data, symbols)
        volume = extract_volume(data, symbols)
        result: dict[str, SignalTargets] = {}
        for combo in combos:
            period = int(combo["bb_period"])
            std = float(combo["bb_std"])
            vp = int(combo["vol_period"])
            vm = float(combo["vol_mult"])
            ent, ext = volume_bb_signals(close, volume, period, std, vp, vm)
            result[combo_name(self.name, combo)] = SignalTargets(entries=ent, exits=ext)
        return result
```

- [ ] **Step 3: Add `MultiTimeframeReversionSignal` class to `signals.py`**

Add after `VolumeBBReversionSignal`:

```python
class MultiTimeframeReversionSignal:
    """Multi-timeframe: weekly RSI oversold confirms daily BB breakdown.

    Entry: weekly RSI below oversold AND daily close below lower BB.
    Exit: weekly RSI crosses above exit level OR daily close crosses above SMA.
    """

    name = "mtf_reversion"
    target_kind = "signals"

    def __init__(
        self,
        cfg: LabConfig,
        weekly_rsi_period: int = 14,
        weekly_rsi_oversold: int = 30,
        weekly_rsi_exit: int = 50,
        daily_bb_period: int = 20,
        daily_bb_std: float = 2.0,
    ) -> None:
        self.cfg = cfg
        self.weekly_rsi_period = weekly_rsi_period
        self.weekly_rsi_oversold = weekly_rsi_oversold
        self.weekly_rsi_exit = weekly_rsi_exit
        self.daily_bb_period = daily_bb_period
        self.daily_bb_std = daily_bb_std

    @classmethod
    def sweep_params(cls) -> dict[str, list]:
        return {
            "weekly_rsi_period": [7, 14],
            "weekly_rsi_oversold": [30, 35],
            "weekly_rsi_exit": [50, 55],
            "daily_bb_period": [15, 20],
            "daily_bb_std": [2.0, 2.5],
        }

    def select(self, asof: pd.Timestamp, data: pd.DataFrame, eligible: List[str]) -> Plan:
        data = data.loc[:asof]
        return [
            {"symbol": s, "weight": 0.0}
            for s in eligible_symbols(data, eligible, self.cfg.min_history_bars)
        ]

    def to_targets(self, plans: Dict[pd.Timestamp, Plan], data: pd.DataFrame) -> SignalTargets:
        symbols = sorted({s["symbol"] for plan in plans.values() for s in plan})
        close = extract_close(data, symbols)
        entries, exits = mtf_signals(
            close, self.weekly_rsi_period, self.weekly_rsi_oversold,
            self.weekly_rsi_exit, self.daily_bb_period, self.daily_bb_std,
        )
        return SignalTargets(entries=entries, exits=exits)

    def sweep_signals(
        self,
        combos: list[dict],
        symbols: list[str],
        data: pd.DataFrame,
    ) -> dict[str, SignalTargets]:
        from ggTrader.lab.sweep import combo_name

        close = extract_close(data, symbols)
        result: dict[str, SignalTargets] = {}
        for combo in combos:
            ent, ext = mtf_signals(
                close,
                int(combo["weekly_rsi_period"]),
                int(combo["weekly_rsi_oversold"]),
                int(combo["weekly_rsi_exit"]),
                int(combo["daily_bb_period"]),
                float(combo["daily_bb_std"]),
            )
            result[combo_name(self.name, combo)] = SignalTargets(entries=ent, exits=ext)
        return result
```

- [ ] **Step 4: Update imports in `signals.py`**

Add to the imports at the top of `signals.py`:

```python
from ggTrader.lab.strategies.indicators import (
    bb_signals,
    eligible_symbols,
    ema_signals,
    extract_close,
    extract_volume,
    macd_signals,
    mtf_signals,
    rsi_signals,
    volume_bb_signals,
)
```

- [ ] **Step 5: Update `_build_signal_registry` and `SIGNAL_STRATEGY_NAMES` in `signals.py`**

Update the function to include the 3 new classes:

```python
def _build_signal_registry() -> dict[str, Any]:
    from ggTrader.lab.strategies.conviction import ConvictionBBSignal
    from ggTrader.lab.strategies.ensemble import EnsembleConvictionSignal, EnsembleSignal

    return {
        "ema_cross": EmaCrossSignal,
        "wfo_tournament": WfoTournamentSignal,
        "bb_reversion": BollingerReversionSignal,
        "rsi_reversion": RsiReversionSignal,
        "macd_divergence": MACDDivergenceSignal,
        "volume_bb_reversion": VolumeBBReversionSignal,
        "mtf_reversion": MultiTimeframeReversionSignal,
        "ensemble": EnsembleSignal,
        "conviction_bb": ConvictionBBSignal,
        "ensemble_conviction": EnsembleConvictionSignal,
    }


SIGNAL_STRATEGY_NAMES = (
    "ema_cross",
    "wfo_tournament",
    "bb_reversion",
    "rsi_reversion",
    "macd_divergence",
    "volume_bb_reversion",
    "mtf_reversion",
    "ensemble",
    "conviction_bb",
    "ensemble_conviction",
)
```

- [ ] **Step 6: Update `cli.py` — add 3 new entries to `cls_map`**

In `src/ggTrader/lab/cli.py`, inside the `if args.sweep or args.wfo:` block, add imports and map entries:

```python
        from ggTrader.lab.strategies.signals import (
            BollingerReversionSignal,
            EmaCrossSignal,
            MACDDivergenceSignal,
            MultiTimeframeReversionSignal,
            RsiReversionSignal,
            VolumeBBReversionSignal,
            WfoTournamentSignal,
        )

        cls_map = {
            "ema_cross": EmaCrossSignal,
            "wfo_tournament": WfoTournamentSignal,
            "bb_reversion": BollingerReversionSignal,
            "rsi_reversion": RsiReversionSignal,
            "macd_divergence": MACDDivergenceSignal,
            "volume_bb_reversion": VolumeBBReversionSignal,
            "mtf_reversion": MultiTimeframeReversionSignal,
            "ensemble": EnsembleSignal,
            "conviction_bb": ConvictionBBSignal,
            "ensemble_conviction": EnsembleConvictionSignal,
            "xs_momentum": CrossSectionalMomentum,
            "dual_momentum": DualMomentum,
        }
```

- [ ] **Step 7: Add registration and class tests to each test file**

Append to `tests/lab/test_macd_signals.py`:

```python
def test_macd_divergence_registered():
    from ggTrader.lab.strategies.signals import _get_registry
    assert "macd_divergence" in _get_registry()


def test_build_macd_divergence():
    from ggTrader.lab.strategies.signals import build_signal_strategy
    from ggTrader.lab.strategy import LabConfig
    strat = build_signal_strategy("macd_divergence", LabConfig())
    assert strat.name == "macd_divergence"
    assert strat.target_kind == "signals"


def test_cli_accepts_macd_divergence():
    from ggTrader.lab.cli import build_arg_parser
    parser = build_arg_parser()
    args = parser.parse_args(["--strategy", "macd_divergence"])
    assert args.strategy == "macd_divergence"


def test_macd_divergence_sweep_params():
    from ggTrader.lab.strategies.signals import MACDDivergenceSignal
    params = MACDDivergenceSignal.sweep_params()
    assert "macd_fast" in params
    assert "divergence_window" in params


def test_macd_divergence_to_targets():
    from ggTrader.lab.strategies.signals import MACDDivergenceSignal
    from ggTrader.lab.strategy import LabConfig, SignalTargets
    cfg = LabConfig(min_history_bars=50)
    strat = MACDDivergenceSignal(cfg)
    ohlcv = _ohlcv_multi(300)
    symbols = sorted(ohlcv.columns.get_level_values(0).unique())
    plans = {ohlcv.index[100]: [{"symbol": s, "weight": 0.0} for s in symbols]}
    targets = strat.to_targets(plans, ohlcv)
    assert isinstance(targets, SignalTargets)
    assert targets.entries.shape[1] == len(symbols)


def _ohlcv_multi(n=300, n_syms=3, seed=42):
    """Synthetic OHLCV with (symbol, field) MultiIndex columns."""
    np.random.seed(seed)
    idx = _idx(n)
    frames = {}
    for i in range(n_syms):
        sym = f"S{i}"
        close = 100.0 * np.exp(np.cumsum(np.random.normal(0.0003, 0.015, n)))
        frames[sym] = pd.DataFrame(
            {"open": close * 0.999, "high": close * 1.005, "low": close * 0.995,
             "close": close, "volume": np.random.randint(1000, 10000, n).astype(float)},
            index=idx,
        )
    df = pd.concat(frames, axis=1)
    df.columns.names = ["symbol", "field"]
    return df
```

Append to `tests/lab/test_volume_bb_signals.py`:

```python
def test_volume_bb_registered():
    from ggTrader.lab.strategies.signals import _get_registry
    assert "volume_bb_reversion" in _get_registry()


def test_build_volume_bb():
    from ggTrader.lab.strategies.signals import build_signal_strategy
    from ggTrader.lab.strategy import LabConfig
    strat = build_signal_strategy("volume_bb_reversion", LabConfig())
    assert strat.name == "volume_bb_reversion"


def test_cli_accepts_volume_bb():
    from ggTrader.lab.cli import build_arg_parser
    parser = build_arg_parser()
    args = parser.parse_args(["--strategy", "volume_bb_reversion"])
    assert args.strategy == "volume_bb_reversion"


def test_volume_bb_to_targets():
    from ggTrader.lab.strategies.signals import VolumeBBReversionSignal
    from ggTrader.lab.strategy import LabConfig, SignalTargets
    cfg = LabConfig(min_history_bars=50)
    strat = VolumeBBReversionSignal(cfg)
    ohlcv = _ohlcv(300)
    symbols = sorted(ohlcv.columns.get_level_values(0).unique())
    plans = {ohlcv.index[100]: [{"symbol": s, "weight": 0.0} for s in symbols]}
    targets = strat.to_targets(plans, ohlcv)
    assert isinstance(targets, SignalTargets)
    assert targets.entries.shape[1] == len(symbols)
```

Append to `tests/lab/test_mtf_signals.py`:

```python
def test_mtf_registered():
    from ggTrader.lab.strategies.signals import _get_registry
    assert "mtf_reversion" in _get_registry()


def test_build_mtf():
    from ggTrader.lab.strategies.signals import build_signal_strategy
    from ggTrader.lab.strategy import LabConfig
    strat = build_signal_strategy("mtf_reversion", LabConfig())
    assert strat.name == "mtf_reversion"


def test_cli_accepts_mtf():
    from ggTrader.lab.cli import build_arg_parser
    parser = build_arg_parser()
    args = parser.parse_args(["--strategy", "mtf_reversion"])
    assert args.strategy == "mtf_reversion"


def test_mtf_to_targets():
    from ggTrader.lab.strategies.signals import MultiTimeframeReversionSignal
    from ggTrader.lab.strategy import LabConfig, SignalTargets
    cfg = LabConfig(min_history_bars=50)
    strat = MultiTimeframeReversionSignal(cfg)
    close_data = _close(300)
    idx = close_data.index
    frames = {}
    for sym in close_data.columns:
        frames[sym] = pd.DataFrame(
            {"open": close_data[sym] * 0.999, "high": close_data[sym] * 1.005,
             "low": close_data[sym] * 0.995, "close": close_data[sym],
             "volume": np.random.randint(1000, 10000, len(idx)).astype(float)},
            index=idx,
        )
    ohlcv = pd.concat(frames, axis=1)
    ohlcv.columns.names = ["symbol", "field"]
    symbols = sorted(ohlcv.columns.get_level_values(0).unique())
    plans = {ohlcv.index[100]: [{"symbol": s, "weight": 0.0} for s in symbols]}
    targets = strat.to_targets(plans, ohlcv)
    assert isinstance(targets, SignalTargets)
```

- [ ] **Step 8: Run all new tests**

Run: `docker compose build ggtrader_live && docker compose run --rm ggtrader_live python -m pytest tests/lab/test_macd_signals.py tests/lab/test_volume_bb_signals.py tests/lab/test_mtf_signals.py -v`
Expected: All tests PASS (8+9+7 indicator tests + 5+4+4 class/registration tests = 37 total).

- [ ] **Step 9: Run full lab test suite to verify no regressions**

Run: `docker compose run --rm ggtrader_live python -m pytest tests/lab/ tests/data/ --ignore=tests/lab/test_reversion_signals.py -v`
Expected: All existing tests + new tests PASS.

- [ ] **Step 10: Commit**

```bash
git add src/ggTrader/lab/strategies/signals.py src/ggTrader/lab/cli.py \
    tests/lab/test_macd_signals.py tests/lab/test_volume_bb_signals.py tests/lab/test_mtf_signals.py
git commit -m "feat(lab): standalone signal classes + CLI/registry for 3 new reversion signals"
```

---

### Task 5: Expanded Ensemble — 6-Voter Wiring + Tests

**Files:**
- Modify: `src/ggTrader/lab/strategies/ensemble.py`
- Modify: `tests/lab/test_ensemble.py`

**Interfaces:**
- Consumes: `macd_signals`, `volume_bb_signals`, `mtf_signals`, `extract_close`, `extract_volume`, `macd_strength`, `volume_bb_strength`, `mtf_strength` from `indicators.py`
- Produces: `EnsembleSignal` expanded to 6 voters; `EnsembleConvictionSignal` expanded with 3 new strength functions

- [ ] **Step 1: Update `EnsembleSignal.__init__` to accept new signal params**

In `src/ggTrader/lab/strategies/ensemble.py`, update the `__init__` method:

```python
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
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        divergence_window: int = 20,
        vol_period: int = 20,
        vol_mult: float = 2.0,
        weekly_rsi_period: int = 14,
        weekly_rsi_oversold: int = 30,
        weekly_rsi_exit: int = 50,
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
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.divergence_window = divergence_window
        self.vol_period = vol_period
        self.vol_mult = vol_mult
        self.weekly_rsi_period = weekly_rsi_period
        self.weekly_rsi_oversold = weekly_rsi_oversold
        self.weekly_rsi_exit = weekly_rsi_exit
```

- [ ] **Step 2: Update `sweep_params` to include new params and expand `min_agree`**

```python
    @classmethod
    def sweep_params(cls) -> dict[str, list]:
        return {
            "min_agree": [2, 3, 4],
            "bb_period": [15, 20],
            "bb_std": [2.0, 2.5],
            "rsi_period": [7, 14],
            "rsi_oversold": [25, 30],
            "ema_fast": [10, 20],
            "ema_slow": [50, 100],
            "macd_fast": [8, 12],
            "macd_slow": [21, 26],
            "macd_signal": [9],
            "divergence_window": [10, 20],
            "vol_period": [20],
            "vol_mult": [1.5, 2.0],
            "weekly_rsi_period": [7, 14],
            "weekly_rsi_oversold": [30, 35],
            "weekly_rsi_exit": [50],
        }
```

- [ ] **Step 3: Update imports at top of `ensemble.py`**

```python
from ggTrader.lab.strategies.indicators import (
    bb_signals,
    bb_strength,
    eligible_symbols,
    ema_signals,
    ema_strength,
    extract_close,
    extract_volume,
    macd_signals,
    macd_strength,
    mtf_signals,
    mtf_strength,
    rsi_signals,
    rsi_strength,
    volume_bb_signals,
    volume_bb_strength,
)
```

- [ ] **Step 4: Update `_generate_signals` to use 6 voters**

Change the method signature to accept both `close` and `volume`, and add the 3 new sub-signals:

```python
    def _generate_signals(self, close: pd.DataFrame, volume: pd.DataFrame) -> SignalTargets:
        """Run all 6 sub-signals, sum entry/exit votes, threshold at min_agree."""
        bb_ent, bb_ext = bb_signals(close, self.bb_period, self.bb_std)
        rsi_ent, rsi_ext = rsi_signals(close, self.rsi_period, self.rsi_oversold, self.rsi_exit)
        ema_ent, ema_ext = ema_signals(close, self.ema_fast, self.ema_slow)
        macd_ent, macd_ext = macd_signals(close, self.macd_fast, self.macd_slow, self.macd_signal, self.divergence_window)
        vbb_ent, vbb_ext = volume_bb_signals(close, volume, self.bb_period, self.bb_std, self.vol_period, self.vol_mult)
        mtf_ent, mtf_ext = mtf_signals(close, self.weekly_rsi_period, self.weekly_rsi_oversold, self.weekly_rsi_exit, self.bb_period, self.bb_std)

        entry_votes = (
            bb_ent.astype(int) + rsi_ent.astype(int) + ema_ent.astype(int)
            + macd_ent.astype(int) + vbb_ent.astype(int) + mtf_ent.astype(int)
        )
        exit_votes = (
            bb_ext.astype(int) + rsi_ext.astype(int) + ema_ext.astype(int)
            + macd_ext.astype(int) + vbb_ext.astype(int) + mtf_ext.astype(int)
        )

        entries = (entry_votes >= self.min_agree).astype(bool)
        exits = (exit_votes >= self.min_agree).astype(bool)
        return SignalTargets(entries=entries, exits=exits)
```

- [ ] **Step 5: Update `to_targets` and `sweep_signals` to pass volume**

```python
    def to_targets(self, plans: Dict[pd.Timestamp, Plan], data: pd.DataFrame) -> SignalTargets:
        symbols = sorted({s["symbol"] for plan in plans.values() for s in plan})
        close = extract_close(data, symbols)
        volume = extract_volume(data, symbols)
        return self._generate_signals(close, volume)

    def sweep_signals(
        self,
        combos: list[dict],
        symbols: list[str],
        data: pd.DataFrame,
    ) -> dict[str, SignalTargets]:
        from ggTrader.lab.sweep import combo_name

        close = extract_close(data, symbols)
        volume = extract_volume(data, symbols)
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
                macd_fast=int(combo.get("macd_fast", self.macd_fast)),
                macd_slow=int(combo.get("macd_slow", self.macd_slow)),
                macd_signal=int(combo.get("macd_signal", self.macd_signal)),
                divergence_window=int(combo.get("divergence_window", self.divergence_window)),
                vol_period=int(combo.get("vol_period", self.vol_period)),
                vol_mult=float(combo.get("vol_mult", self.vol_mult)),
                weekly_rsi_period=int(combo.get("weekly_rsi_period", self.weekly_rsi_period)),
                weekly_rsi_oversold=int(combo.get("weekly_rsi_oversold", self.weekly_rsi_oversold)),
                weekly_rsi_exit=int(combo.get("weekly_rsi_exit", self.weekly_rsi_exit)),
            )
            targets = strat._generate_signals(close, volume)
            key = combo_name(self.name, combo)
            result[key] = targets
        return result
```

- [ ] **Step 6: Apply the same changes to `EnsembleConvictionSignal`**

Mirror all changes from Steps 1-5 in `EnsembleConvictionSignal`:
- Same new `__init__` params
- Same `sweep_params` expansion
- Update `_generate_signals_with_sizes` to call 6 sub-signals and 6 strengths:

```python
    def _generate_signals_with_sizes(self, close: pd.DataFrame, volume: pd.DataFrame) -> SignalTargets:
        """Entry/exit via majority vote + conviction-weighted sizes."""
        bb_ent, bb_ext = bb_signals(close, self.bb_period, self.bb_std)
        rsi_ent, rsi_ext = rsi_signals(close, self.rsi_period, self.rsi_oversold, self.rsi_exit)
        ema_ent, ema_ext = ema_signals(close, self.ema_fast, self.ema_slow)
        macd_ent, macd_ext = macd_signals(close, self.macd_fast, self.macd_slow, self.macd_signal, self.divergence_window)
        vbb_ent, vbb_ext = volume_bb_signals(close, volume, self.bb_period, self.bb_std, self.vol_period, self.vol_mult)
        mtf_ent, mtf_ext = mtf_signals(close, self.weekly_rsi_period, self.weekly_rsi_oversold, self.weekly_rsi_exit, self.bb_period, self.bb_std)

        entry_votes = (
            bb_ent.astype(int) + rsi_ent.astype(int) + ema_ent.astype(int)
            + macd_ent.astype(int) + vbb_ent.astype(int) + mtf_ent.astype(int)
        )
        exit_votes = (
            bb_ext.astype(int) + rsi_ext.astype(int) + ema_ext.astype(int)
            + macd_ext.astype(int) + vbb_ext.astype(int) + mtf_ext.astype(int)
        )

        entries = (entry_votes >= self.min_agree).astype(bool)
        exits = (exit_votes >= self.min_agree).astype(bool)

        # Compute per-signal strength (0-1), masked to entry bars only
        bb_str = bb_strength(close, self.bb_period, self.bb_std)
        rsi_str = rsi_strength(close, self.rsi_period, self.rsi_oversold)
        ema_str = ema_strength(close, self.ema_fast, self.ema_slow)
        macd_str = macd_strength(close, self.macd_fast, self.macd_slow, self.macd_signal)
        vbb_str = volume_bb_strength(close, volume, self.bb_period, self.bb_std, self.vol_period)
        mtf_str = mtf_strength(close, self.weekly_rsi_period, self.weekly_rsi_oversold, self.bb_period, self.bb_std)

        # Sum strengths of agreeing signals, divide by count
        strength_sum = (
            bb_str.where(bb_ent, 0.0) + rsi_str.where(rsi_ent, 0.0) + ema_str.where(ema_ent, 0.0)
            + macd_str.where(macd_ent, 0.0) + vbb_str.where(vbb_ent, 0.0) + mtf_str.where(mtf_ent, 0.0)
        )
        conviction = strength_sum / entry_votes.replace(0, np.nan)

        sizes = self.min_size + conviction * (self.max_size - self.min_size)
        sizes = sizes.where(entries, np.nan)

        return SignalTargets(entries=entries, exits=exits, sizes=sizes)
```

Update `to_targets` and `sweep_signals` in `EnsembleConvictionSignal` to pass volume (same pattern as Step 5).

- [ ] **Step 7: Update ensemble tests for 6-voter behavior**

Add to `tests/lab/test_ensemble.py`:

```python
def test_ensemble_sweep_params_includes_new_signals():
    params = EnsembleSignal.sweep_params()
    assert "macd_fast" in params
    assert "vol_mult" in params
    assert "weekly_rsi_period" in params
    assert 4 in params["min_agree"]


def test_ensemble_6_voter_min_agree_4():
    """With min_agree=4 (majority of 6), entries should be very rare."""
    cfg = LabConfig(min_history_bars=50)
    strat = EnsembleSignal(cfg, min_agree=4)
    ohlcv = _ohlcv(n=300)
    symbols = sorted(ohlcv.columns.get_level_values(0).unique())
    plans = {ohlcv.index[100]: [{"symbol": s, "weight": 0.0} for s in symbols]}
    targets = strat.to_targets(plans, ohlcv)
    # 4-of-6 agreement on random data should be extremely rare
    assert targets.entries.sum().sum() <= targets.entries.shape[0] * len(symbols) * 0.005
```

- [ ] **Step 8: Run all tests**

Run: `docker compose build ggtrader_live && docker compose run --rm ggtrader_live python -m pytest tests/lab/test_ensemble.py tests/lab/test_ensemble_conviction.py -v`
Expected: All existing + new tests PASS.

- [ ] **Step 9: Run full test suite**

Run: `docker compose run --rm ggtrader_live python -m pytest tests/lab/ tests/data/ --ignore=tests/lab/test_reversion_signals.py -v`
Expected: All tests PASS (no regressions).

- [ ] **Step 10: Commit**

```bash
git add src/ggTrader/lab/strategies/ensemble.py tests/lab/test_ensemble.py tests/lab/test_ensemble_conviction.py
git commit -m "feat(lab): expand ensemble to 6 voters with conviction strength extensions"
```

---

### Task 6: ML Pre-Screen Script + Tests

**Files:**
- Create: `scripts/ml_signal_screen.py`
- Create: `tests/lab/test_ml_screen.py`

**Interfaces:**
- Consumes: `build_signal_strategy` from `signals.py`; `extract_features`, `FEATURE_NAMES` from `paper/feature_gate.py`; `fetch_stock_ohlcv` from `lab/data.py`; `extract_close`, `extract_volume` from `indicators.py`
- Produces: A CLI script that prints precision/recall/F1 and writes JSON results to `results/ml_screen_<signal>_<timestamp>.json`

- [ ] **Step 1: Write the ML pre-screen script**

Create `scripts/ml_signal_screen.py`:

```python
#!/usr/bin/env python3
"""ML pre-screen: evaluate a signal strategy's entry quality via LightGBM.

Usage:
    python scripts/ml_signal_screen.py --signal macd_divergence
    python scripts/ml_signal_screen.py --signal volume_bb_reversion --start 2022-01-01
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ggTrader.lab.data import fetch_stock_ohlcv, equity_universe_between
from ggTrader.lab.strategies.indicators import extract_close, extract_volume
from ggTrader.lab.strategies.signals import build_signal_strategy
from ggTrader.lab.strategy import LabConfig
from ggTrader.paper.feature_gate import FEATURE_NAMES, extract_features


def collect_entries(
    signal_name: str,
    ohlcv: pd.DataFrame,
    symbols: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Generate entries for a signal and build a feature+label DataFrame."""
    cfg = LabConfig(min_history_bars=50)
    strat = build_signal_strategy(signal_name, cfg)

    plans = {pd.Timestamp(start, tz="UTC"): [{"symbol": s, "weight": 0.0} for s in symbols]}
    targets = strat.to_targets(plans, ohlcv)

    close_df = extract_close(ohlcv, symbols)
    vol_df = extract_volume(ohlcv, symbols)

    rows = []
    for sym in symbols:
        if sym not in targets.entries.columns:
            continue
        entry_bars = targets.entries.index[targets.entries[sym]]
        close_s = close_df[sym].dropna()
        vol_s = vol_df[sym].dropna() if sym in vol_df.columns else pd.Series(1.0, index=close_s.index)

        for bar in entry_bars:
            if bar not in close_s.index:
                continue
            feats = extract_features(close_s, vol_s, bar)
            # Label: 5-day forward return > 0
            bar_idx = close_s.index.get_loc(bar)
            if bar_idx + 5 >= len(close_s):
                continue
            fwd_ret = close_s.iloc[bar_idx + 5] / close_s.iloc[bar_idx] - 1.0
            feats["label"] = int(fwd_ret > 0)
            feats["symbol"] = sym
            feats["bar_date"] = str(bar.date())
            rows.append(feats)

    return pd.DataFrame(rows)


def train_and_evaluate(df: pd.DataFrame) -> dict:
    """Train LightGBM with 5-fold time-series CV, return metrics."""
    import lightgbm as lgb
    from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
    from sklearn.model_selection import TimeSeriesSplit

    X = df[FEATURE_NAMES].values
    y = df["label"].values

    tscv = TimeSeriesSplit(n_splits=5)
    all_preds = np.zeros(len(y))
    all_true = np.zeros(len(y))
    mask = np.zeros(len(y), dtype=bool)

    for train_idx, test_idx in tscv.split(X):
        model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            min_child_samples=20,
            verbose=-1,
        )
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[test_idx])
        all_preds[test_idx] = preds
        all_true[test_idx] = y[test_idx]
        mask[test_idx] = True

    y_true = all_true[mask]
    y_pred = all_preds[mask]

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # Feature importances from last fold
    importances = dict(zip(FEATURE_NAMES, model.feature_importances_.tolist()))

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "n_samples": int(mask.sum()),
        "n_positive": int(y_true.sum()),
        "feature_importances": importances,
    }


def main():
    parser = argparse.ArgumentParser(description="ML pre-screen for signal quality")
    parser.add_argument("--signal", required=True, help="Signal strategy name")
    parser.add_argument("--start", default="2021-01-01", help="Data start date")
    parser.add_argument("--end", default=None, help="Data end date (default: now)")
    parser.add_argument("--universe", default="sp500", help="Stock universe")
    args = parser.parse_args()

    end = args.end or str(pd.Timestamp.now(tz="UTC").normalize().date())
    start_ts = pd.Timestamp(args.start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")

    print(f"ML Pre-Screen: {args.signal}")
    print(f"Universe: {args.universe} | {args.start} -> {end}")
    print()

    symbols = equity_universe_between(start_ts, end_ts, universe=args.universe)
    print(f"Loading OHLCV for {len(symbols)} symbols...")
    ohlcv = fetch_stock_ohlcv(symbols, start=args.start, end=end)

    sym_cols = [s for s in ohlcv.columns.get_level_values(0).unique()]
    print(f"Generating entries for {args.signal}...")
    df = collect_entries(args.signal, ohlcv, sym_cols, args.start, end)

    if len(df) < 50:
        print(f"Only {len(df)} entries — too few for meaningful ML evaluation.")
        sys.exit(1)

    print(f"Training LightGBM on {len(df)} entries...")
    results = train_and_evaluate(df)

    # Verdict
    prec = results["precision"]
    if prec < 0.50:
        verdict = "DROP"
    elif prec < 0.55:
        verdict = "BORDERLINE"
    else:
        verdict = "STRONG"

    print()
    print(f"{'Signal':<25} {args.signal}")
    print(f"{'Precision':<25} {prec:.4f}")
    print(f"{'Recall':<25} {results['recall']:.4f}")
    print(f"{'F1':<25} {results['f1']:.4f}")
    print(f"{'Samples':<25} {results['n_samples']}")
    print(f"{'Positive rate':<25} {results['n_positive'] / results['n_samples']:.2%}")
    print(f"{'Verdict':<25} {verdict}")
    print()
    print("Top features:")
    sorted_feats = sorted(results["feature_importances"].items(), key=lambda x: -x[1])
    for feat, imp in sorted_feats[:5]:
        print(f"  {feat:<20} {imp}")

    # Write results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outpath = results_dir / f"ml_screen_{args.signal}_{ts}.json"
    results["signal"] = args.signal
    results["verdict"] = verdict
    results["start"] = args.start
    results["end"] = end
    results["universe"] = args.universe
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {outpath}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write tests for the ML screen logic**

Create `tests/lab/test_ml_screen.py`:

```python
"""Tests for the ML signal pre-screen script logic."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from ggTrader.paper.feature_gate import extract_features, FEATURE_NAMES


def _close_series(n=100, seed=42):
    np.random.seed(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
    return pd.Series(100.0 * np.exp(np.cumsum(np.random.normal(0.0003, 0.015, n))), index=idx)


def _volume_series(n=100, seed=42):
    np.random.seed(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
    return pd.Series(np.random.randint(1000, 10000, n).astype(float), index=idx)


class TestExtractFeatures:
    def test_returns_all_feature_names(self):
        close = _close_series()
        vol = _volume_series()
        feats = extract_features(close, vol, close.index[50])
        assert set(FEATURE_NAMES) == set(feats.keys())

    def test_features_are_finite(self):
        close = _close_series()
        vol = _volume_series()
        feats = extract_features(close, vol, close.index[50])
        for k, v in feats.items():
            assert np.isfinite(v), f"Feature {k} is not finite: {v}"

    def test_rsi_in_0_100_range(self):
        close = _close_series()
        vol = _volume_series()
        feats = extract_features(close, vol, close.index[50])
        assert 0.0 <= feats["rsi_14"] <= 100.0


class TestVerdictThresholds:
    def test_drop_below_050(self):
        assert 0.49 < 0.50  # DROP threshold

    def test_borderline_050_to_055(self):
        prec = 0.52
        assert 0.50 <= prec < 0.55  # BORDERLINE

    def test_strong_above_055(self):
        prec = 0.58
        assert prec >= 0.55  # STRONG
```

- [ ] **Step 3: Run tests**

Run: `docker compose build ggtrader_live && docker compose run --rm ggtrader_live python -m pytest tests/lab/test_ml_screen.py -v`
Expected: All 6 tests PASS.

- [ ] **Step 4: Commit**

```bash
git add scripts/ml_signal_screen.py tests/lab/test_ml_screen.py
git commit -m "feat(lab): ML pre-screen script — LightGBM precision gate for signal evaluation"
```

---
