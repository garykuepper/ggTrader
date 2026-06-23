# Conviction Ensemble Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add conviction-weighted position sizing to the ensemble strategy — size entries by sub-signal strength instead of flat 2%.

**Architecture:** Three new vectorized indicator functions (`bb_strength`, `rsi_strength`, `ema_strength`) in `indicators.py`. A new `EnsembleConvictionSignal` class in `ensemble.py` that reuses the ensemble's entry/exit logic but adds a `sizes` DataFrame to `SignalTargets`. Registered in `signals.py` so the CLI picks it up automatically.

**Tech Stack:** pandas, numpy, vectorbt (simulation only), pytest

## Global Constraints

- All indicator functions must be vectorized (no per-bar loops)
- Strength values clipped to [0, 1]
- Position sizes bounded to [min_size, max_size]
- `sizes` must be NaN where `entries` is False (simulate_signals contract)
- Follow existing patterns: NamedTuple `SignalTargets`, `sweep_signals` method, `combo_name` keys

---

### Task 1: Strength indicator functions

**Files:**
- Modify: `src/ggTrader/lab/strategies/indicators.py` (append 3 functions)
- Create: `tests/lab/test_strength_indicators.py`

**Interfaces:**
- Consumes: `bb_signals`, `rsi_signals`, `ema_signals` patterns from same file
- Produces:
  - `bb_strength(close: pd.DataFrame, period: int, std: float) -> pd.DataFrame` — (time x symbol) float [0, 1]
  - `rsi_strength(close: pd.DataFrame, period: int, oversold: int) -> pd.DataFrame` — (time x symbol) float [0, 1]
  - `ema_strength(close: pd.DataFrame, ema_fast: int, ema_slow: int) -> pd.DataFrame` — (time x symbol) float [0, 1]

- [ ] **Step 1: Write failing tests for `bb_strength`**

```python
# tests/lab/test_strength_indicators.py
"""Tests for conviction strength indicator functions."""

import numpy as np
import pandas as pd
import pytest

from ggTrader.lab.strategies.indicators import bb_strength, ema_strength, rsi_strength


def _close(n=200, n_syms=2, seed=42):
    np.random.seed(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
    data = 100.0 * np.exp(np.cumsum(np.random.normal(0.0003, 0.015, (n, n_syms)), axis=0))
    return pd.DataFrame(data, index=idx, columns=[f"S{i}" for i in range(n_syms)])


class TestBBStrength:
    def test_output_shape_matches_input(self):
        close = _close()
        result = bb_strength(close, period=20, std=2.0)
        assert result.shape == close.shape
        assert list(result.columns) == list(close.columns)

    def test_values_in_zero_one_range(self):
        close = _close()
        result = bb_strength(close, period=20, std=2.0)
        valid = result.dropna()
        if len(valid) > 0:
            assert valid.min().min() >= -1e-10
            assert valid.max().max() <= 1.0 + 1e-10

    def test_nan_during_warmup(self):
        close = _close(n=50)
        result = bb_strength(close, period=20, std=2.0)
        assert result.iloc[:19].isna().all().all()

    def test_zero_when_price_at_lower_band(self):
        """Price exactly at lower band -> strength ~ 0."""
        idx = pd.date_range("2020-01-01", periods=100, freq="B", tz="UTC")
        close = pd.DataFrame({"A": np.full(100, 100.0)}, index=idx)
        # Flat price => std=0 => band_width=0 => should return 0 (not inf)
        result = bb_strength(close, period=20, std=2.0)
        valid = result.dropna()
        assert (valid == 0.0).all().all()


class TestRSIStrength:
    def test_output_shape_matches_input(self):
        close = _close()
        result = rsi_strength(close, period=14, oversold=30)
        assert result.shape == close.shape

    def test_values_in_zero_one_range(self):
        close = _close()
        result = rsi_strength(close, period=14, oversold=30)
        valid = result.dropna()
        if len(valid) > 0:
            assert valid.min().min() >= -1e-10
            assert valid.max().max() <= 1.0 + 1e-10

    def test_zero_when_rsi_above_oversold(self):
        """RSI above oversold threshold -> strength = 0."""
        close = _close(seed=99)
        result = rsi_strength(close, period=14, oversold=30)
        # Compute RSI to verify
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1.0 / 14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / 14, min_periods=14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        above_mask = rsi >= 30
        strength_above = result[above_mask].dropna()
        if len(strength_above) > 0:
            assert (strength_above == 0.0).all().all()


class TestEMAStrength:
    def test_output_shape_matches_input(self):
        close = _close()
        result = ema_strength(close, ema_fast=20, ema_slow=50)
        assert result.shape == close.shape

    def test_values_in_zero_one_range(self):
        close = _close()
        result = ema_strength(close, ema_fast=20, ema_slow=50)
        valid = result.dropna()
        if len(valid) > 0:
            assert valid.min().min() >= -1e-10
            assert valid.max().max() <= 1.0 + 1e-10

    def test_zero_when_fast_below_slow(self):
        """Fast EMA below slow -> strength = 0."""
        close = _close()
        result = ema_strength(close, ema_fast=20, ema_slow=50)
        ema_f = close.ewm(span=20, adjust=False).mean()
        ema_s = close.ewm(span=50, adjust=False).mean()
        below_mask = ema_f < ema_s
        strength_below = result[below_mask].dropna()
        if len(strength_below) > 0:
            assert (strength_below == 0.0).all().all()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/flynn/ggTrader && python -m pytest tests/lab/test_strength_indicators.py -v
```

Expected: `ImportError: cannot import name 'bb_strength'`

- [ ] **Step 3: Implement the three strength functions**

Add to the end of `src/ggTrader/lab/strategies/indicators.py`:

```python
def bb_strength(close: pd.DataFrame, period: int, std: float) -> pd.DataFrame:
    """Normalized depth below the lower Bollinger Band, clipped to [0, 1]."""
    sma = close.rolling(window=period, min_periods=period).mean()
    rolling_std = close.rolling(window=period, min_periods=period).std()
    band_width = std * rolling_std
    depth = (sma - std * rolling_std - close) / band_width.replace(0, np.nan)
    return depth.clip(lower=0.0, upper=1.0)


def rsi_strength(close: pd.DataFrame, period: int, oversold: int) -> pd.DataFrame:
    """Normalized RSI depth below oversold threshold, clipped to [0, 1]."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    depth = (oversold - rsi) / oversold
    return depth.clip(lower=0.0, upper=1.0)


def ema_strength(close: pd.DataFrame, ema_fast: int, ema_slow: int) -> pd.DataFrame:
    """Normalized EMA gap (fast above slow) / slow, clipped to [0, 1]."""
    ema_f = close.ewm(span=ema_fast, adjust=False).mean()
    ema_s = close.ewm(span=ema_slow, adjust=False).mean()
    gap = (ema_f - ema_s) / ema_s.replace(0, np.nan)
    return gap.clip(lower=0.0, upper=1.0)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/flynn/ggTrader && python -m pytest tests/lab/test_strength_indicators.py -v
```

Expected: all 9 tests PASS

- [ ] **Step 5: Run existing indicator tests to check for regressions**

```bash
cd /home/flynn/ggTrader && python -m pytest tests/lab/test_reversion_signals.py tests/lab/test_signals.py -v
```

Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add src/ggTrader/lab/strategies/indicators.py tests/lab/test_strength_indicators.py
git commit -m "feat(lab): add bb_strength, rsi_strength, ema_strength indicator functions"
```

---

### Task 2: EnsembleConvictionSignal strategy class

**Files:**
- Modify: `src/ggTrader/lab/strategies/ensemble.py` (add class after `EnsembleSignal`)
- Create: `tests/lab/test_ensemble_conviction.py`

**Interfaces:**
- Consumes: `bb_strength`, `rsi_strength`, `ema_strength` from Task 1; `bb_signals`, `rsi_signals`, `ema_signals`, `extract_close`, `eligible_symbols` from `indicators.py`
- Produces: `EnsembleConvictionSignal` class with:
  - `name = "ensemble_conviction"`
  - `target_kind = "signals"`
  - `__init__(self, cfg, min_agree=2, bb_period=20, bb_std=2.0, rsi_period=14, rsi_oversold=30, rsi_exit=50, ema_fast=20, ema_slow=50, min_size=0.01, max_size=0.04)`
  - `sweep_params() -> dict[str, list]`
  - `select(asof, data, eligible) -> Plan`
  - `to_targets(plans, data) -> SignalTargets` (with `sizes` populated)
  - `sweep_signals(combos, symbols, data) -> dict[str, SignalTargets]`

- [ ] **Step 1: Write failing tests**

```python
# tests/lab/test_ensemble_conviction.py
"""Tests for EnsembleConvictionSignal — conviction-weighted ensemble sizing."""

import numpy as np
import pandas as pd

from ggTrader.lab.strategies.ensemble import EnsembleConvictionSignal, EnsembleSignal
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


class TestEnsembleConvictionSignal:
    def test_returns_signal_targets_with_sizes(self):
        cfg = LabConfig(min_history_bars=50)
        strat = EnsembleConvictionSignal(cfg)
        ohlcv = _ohlcv(n=300)
        symbols = sorted(ohlcv.columns.get_level_values(0).unique())
        plans = {ohlcv.index[100]: [{"symbol": s, "weight": 0.0} for s in symbols]}
        targets = strat.to_targets(plans, ohlcv)
        assert isinstance(targets, SignalTargets)
        assert targets.sizes is not None
        assert targets.sizes.shape == targets.entries.shape

    def test_sizes_bounded(self):
        cfg = LabConfig(min_history_bars=50)
        strat = EnsembleConvictionSignal(cfg, min_size=0.01, max_size=0.04)
        ohlcv = _ohlcv(n=300)
        symbols = sorted(ohlcv.columns.get_level_values(0).unique())
        plans = {ohlcv.index[100]: [{"symbol": s, "weight": 0.0} for s in symbols]}
        targets = strat.to_targets(plans, ohlcv)
        valid = targets.sizes[targets.entries].dropna()
        if len(valid) > 0:
            assert valid.min() >= 0.01 - 1e-10
            assert valid.max() <= 0.04 + 1e-10

    def test_sizes_nan_where_no_entry(self):
        cfg = LabConfig(min_history_bars=50)
        strat = EnsembleConvictionSignal(cfg)
        ohlcv = _ohlcv(n=300)
        symbols = sorted(ohlcv.columns.get_level_values(0).unique())
        plans = {ohlcv.index[100]: [{"symbol": s, "weight": 0.0} for s in symbols]}
        targets = strat.to_targets(plans, ohlcv)
        no_entry = ~targets.entries
        assert targets.sizes[no_entry].isna().all().all()

    def test_entries_exits_match_plain_ensemble(self):
        """Entry/exit logic must be identical to EnsembleSignal."""
        cfg = LabConfig(min_history_bars=50)
        ohlcv = _ohlcv(n=300, seed=77)
        symbols = sorted(ohlcv.columns.get_level_values(0).unique())
        plans = {ohlcv.index[100]: [{"symbol": s, "weight": 0.0} for s in symbols]}

        plain = EnsembleSignal(cfg, min_agree=2)
        conviction = EnsembleConvictionSignal(cfg, min_agree=2)

        t_plain = plain.to_targets(plans, ohlcv)
        t_conv = conviction.to_targets(plans, ohlcv)

        pd.testing.assert_frame_equal(t_plain.entries, t_conv.entries)
        pd.testing.assert_frame_equal(t_plain.exits, t_conv.exits)

    def test_sizes_vary(self):
        """Conviction sizes should not all be identical."""
        cfg = LabConfig(min_history_bars=50)
        strat = EnsembleConvictionSignal(cfg, min_agree=1)
        ohlcv = _ohlcv(n=300)
        symbols = sorted(ohlcv.columns.get_level_values(0).unique())
        plans = {ohlcv.index[100]: [{"symbol": s, "weight": 0.0} for s in symbols]}
        targets = strat.to_targets(plans, ohlcv)
        entry_sizes = targets.sizes[targets.entries].dropna()
        if len(entry_sizes) > 2:
            assert entry_sizes.std() > 0, "Conviction sizes should vary across entries"

    def test_sweep_params_includes_ensemble_keys(self):
        params = EnsembleConvictionSignal.sweep_params()
        assert "min_agree" in params
        assert "bb_period" in params

    def test_sweep_signals_returns_sizes(self):
        from ggTrader.lab.sweep import combo_name

        cfg = LabConfig(min_history_bars=50)
        strat = EnsembleConvictionSignal(cfg)
        ohlcv = _ohlcv(n=200)
        symbols = sorted(ohlcv.columns.get_level_values(0).unique())
        combos = [
            {
                "min_agree": 2,
                "bb_period": 20,
                "bb_std": 2.0,
                "rsi_period": 14,
                "rsi_oversold": 30,
                "ema_fast": 20,
                "ema_slow": 50,
            }
        ]
        result = strat.sweep_signals(combos, symbols, ohlcv)
        key = list(result.keys())[0]
        assert result[key].sizes is not None

    def test_name_and_target_kind(self):
        assert EnsembleConvictionSignal.name == "ensemble_conviction"
        assert EnsembleConvictionSignal.target_kind == "signals"

    def test_select_delegates_to_eligible(self):
        cfg = LabConfig(min_history_bars=50)
        strat = EnsembleConvictionSignal(cfg)
        ohlcv = _ohlcv(n=300)
        symbols = sorted(ohlcv.columns.get_level_values(0).unique())
        plan = strat.select(ohlcv.index[200], ohlcv, symbols)
        assert len(plan) == len(symbols)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/flynn/ggTrader && python -m pytest tests/lab/test_ensemble_conviction.py -v
```

Expected: `ImportError: cannot import name 'EnsembleConvictionSignal'`

- [ ] **Step 3: Implement `EnsembleConvictionSignal`**

Add to `src/ggTrader/lab/strategies/ensemble.py` after the `EnsembleSignal` class:

```python
from ggTrader.lab.strategies.indicators import (
    bb_strength,
    ema_strength,
    rsi_strength,
)

# (Add to existing imports at top of file, alongside the existing indicator imports)


class EnsembleConvictionSignal:
    """Majority-vote ensemble with conviction-weighted position sizing.

    Same entry/exit logic as EnsembleSignal, but sizes positions by the
    average strength of the agreeing sub-signals on each entry bar.
    """

    name = "ensemble_conviction"
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
        min_size: float = 0.01,
        max_size: float = 0.04,
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
        self.min_size = min_size
        self.max_size = max_size

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
        return [
            {"symbol": s, "weight": 0.0}
            for s in eligible_symbols(data, eligible, self.cfg.min_history_bars)
        ]

    def _generate_signals_with_sizes(self, close: pd.DataFrame) -> SignalTargets:
        """Entry/exit via majority vote + conviction-weighted sizes."""
        import numpy as np

        bb_ent, bb_ext = bb_signals(close, self.bb_period, self.bb_std)
        rsi_ent, rsi_ext = rsi_signals(close, self.rsi_period, self.rsi_oversold, self.rsi_exit)
        ema_ent, ema_ext = ema_signals(close, self.ema_fast, self.ema_slow)

        entry_votes = bb_ent.astype(int) + rsi_ent.astype(int) + ema_ent.astype(int)
        exit_votes = bb_ext.astype(int) + rsi_ext.astype(int) + ema_ext.astype(int)

        entries = (entry_votes >= self.min_agree).astype(bool)
        exits = (exit_votes >= self.min_agree).astype(bool)

        # Compute per-signal strength (0-1), masked to entry bars only
        bb_str = bb_strength(close, self.bb_period, self.bb_std)
        rsi_str = rsi_strength(close, self.rsi_period, self.rsi_oversold)
        ema_str = ema_strength(close, self.ema_fast, self.ema_slow)

        # Sum strengths of agreeing signals, divide by count of agreeing signals
        strength_sum = (
            bb_str.where(bb_ent, 0.0) + rsi_str.where(rsi_ent, 0.0) + ema_str.where(ema_ent, 0.0)
        )
        conviction = strength_sum / entry_votes.replace(0, np.nan)

        sizes = self.min_size + conviction * (self.max_size - self.min_size)
        sizes = sizes.where(entries, np.nan)

        return SignalTargets(entries=entries, exits=exits, sizes=sizes)

    def to_targets(self, plans: Dict[pd.Timestamp, Plan], data: pd.DataFrame) -> SignalTargets:
        symbols = sorted({s["symbol"] for plan in plans.values() for s in plan})
        return self._generate_signals_with_sizes(extract_close(data, symbols))

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
            strat = EnsembleConvictionSignal(
                self.cfg,
                min_agree=int(combo.get("min_agree", self.min_agree)),
                bb_period=int(combo.get("bb_period", self.bb_period)),
                bb_std=float(combo.get("bb_std", self.bb_std)),
                rsi_period=int(combo.get("rsi_period", self.rsi_period)),
                rsi_oversold=int(combo.get("rsi_oversold", self.rsi_oversold)),
                rsi_exit=int(combo.get("rsi_exit", self.rsi_exit)),
                ema_fast=int(combo.get("ema_fast", self.ema_fast)),
                ema_slow=int(combo.get("ema_slow", self.ema_slow)),
                min_size=self.min_size,
                max_size=self.max_size,
            )
            targets = strat._generate_signals_with_sizes(close)
            key = combo_name(self.name, combo)
            result[key] = targets
        return result
```

Note: add `bb_strength`, `rsi_strength`, `ema_strength` to the import block at the top of `ensemble.py`, alongside the existing `bb_signals`, `ema_signals`, `extract_close`, `rsi_signals` imports. Also add `import numpy as np` at the top of the file if not already present (remove the local `import numpy as np` inside the method).

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/flynn/ggTrader && python -m pytest tests/lab/test_ensemble_conviction.py -v
```

Expected: all 9 tests PASS

- [ ] **Step 5: Run existing ensemble tests for regressions**

```bash
cd /home/flynn/ggTrader && python -m pytest tests/lab/test_ensemble.py tests/lab/test_conviction.py -v
```

Expected: all PASS (EnsembleSignal untouched)

- [ ] **Step 6: Commit**

```bash
git add src/ggTrader/lab/strategies/ensemble.py tests/lab/test_ensemble_conviction.py
git commit -m "feat(lab): EnsembleConvictionSignal — conviction-weighted ensemble sizing"
```

---

### Task 3: CLI registration + integration test

**Files:**
- Modify: `src/ggTrader/lab/strategies/signals.py` (register in `_build_signal_registry` and `SIGNAL_STRATEGY_NAMES`)
- Modify: `src/ggTrader/lab/cli.py` (add to `cls_map` in sweep/wfo block)
- Modify: `tests/lab/test_ensemble_conviction.py` (add registration + simulation tests)

**Interfaces:**
- Consumes: `EnsembleConvictionSignal` from Task 2
- Produces: `ensemble_conviction` available via CLI `--strategy ensemble_conviction`, in signal registry, and passing simulation

- [ ] **Step 1: Write failing tests for registration and simulation**

Append to `tests/lab/test_ensemble_conviction.py`:

```python
def test_registered_in_signal_registry():
    from ggTrader.lab.strategies.signals import _get_registry

    assert "ensemble_conviction" in _get_registry()


def test_build_signal_strategy():
    from ggTrader.lab.strategies.signals import build_signal_strategy

    strat = build_signal_strategy("ensemble_conviction", LabConfig())
    assert strat.name == "ensemble_conviction"


def test_cli_accepts_ensemble_conviction():
    from ggTrader.lab.cli import build_arg_parser

    parser = build_arg_parser()
    args = parser.parse_args(["--strategy", "ensemble_conviction"])
    assert args.strategy == "ensemble_conviction"


def test_simulate_signals_with_conviction_sizes():
    """Conviction sizes flow through simulate_signals and produce different equity than flat."""
    from ggTrader.lab.simulate import simulate_signals

    cfg = LabConfig(min_history_bars=50)
    ohlcv = _ohlcv(n=300, seed=42)
    symbols = sorted(ohlcv.columns.get_level_values(0).unique())
    close = pd.concat({s: ohlcv[s]["close"] for s in symbols}, axis=1)

    # Generate conviction targets
    conv_strat = EnsembleConvictionSignal(cfg, min_agree=1, min_size=0.01, max_size=0.04)
    plans = {ohlcv.index[100]: [{"symbol": s, "weight": 0.0} for s in symbols]}
    conv_targets = conv_strat.to_targets(plans, ohlcv)

    # Generate plain ensemble targets (no sizes)
    plain_strat = EnsembleSignal(cfg, min_agree=1)
    plain_targets = plain_strat.to_targets(plans, ohlcv)

    config = {
        "START_CASH": 100000.0,
        "FEES": 0.001,
        "SLIPPAGE": 0.0005,
        "FREQ": "1d",
        "SIGNAL_POSITION_SIZE": 0.02,
    }

    if conv_targets.entries.sum().sum() > 0:
        _, eq_conv, _ = simulate_signals({"conv": conv_targets}, close, config)
        _, eq_plain, _ = simulate_signals({"plain": plain_targets}, close, config)
        # They use different sizing, so equity curves should differ
        assert not eq_conv["conv"].equals(eq_plain["plain"])
```

- [ ] **Step 2: Run to verify failures**

```bash
cd /home/flynn/ggTrader && python -m pytest tests/lab/test_ensemble_conviction.py::test_registered_in_signal_registry tests/lab/test_ensemble_conviction.py::test_cli_accepts_ensemble_conviction -v
```

Expected: FAIL — not yet registered

- [ ] **Step 3: Register in signals.py**

In `src/ggTrader/lab/strategies/signals.py`, modify `_build_signal_registry`:

```python
def _build_signal_registry() -> dict[str, Any]:
    from ggTrader.lab.strategies.conviction import ConvictionBBSignal
    from ggTrader.lab.strategies.ensemble import EnsembleConvictionSignal, EnsembleSignal

    return {
        "ema_cross": EmaCrossSignal,
        "wfo_tournament": WfoTournamentSignal,
        "bb_reversion": BollingerReversionSignal,
        "rsi_reversion": RsiReversionSignal,
        "ensemble": EnsembleSignal,
        "conviction_bb": ConvictionBBSignal,
        "ensemble_conviction": EnsembleConvictionSignal,
    }
```

Update `SIGNAL_STRATEGY_NAMES`:

```python
SIGNAL_STRATEGY_NAMES = (
    "ema_cross",
    "wfo_tournament",
    "bb_reversion",
    "rsi_reversion",
    "ensemble",
    "conviction_bb",
    "ensemble_conviction",
)
```

- [ ] **Step 4: Register in cli.py**

In `src/ggTrader/lab/cli.py`, add to the `cls_map` dict inside the `if args.sweep or args.wfo:` block (around line 101):

```python
from ggTrader.lab.strategies.ensemble import EnsembleConvictionSignal, EnsembleSignal

# In cls_map:
"ensemble_conviction": EnsembleConvictionSignal,
```

The import of `EnsembleSignal` already exists — just add `EnsembleConvictionSignal` to it. Add the dict entry after `"conviction_bb": ConvictionBBSignal,`.

- [ ] **Step 5: Run all tests**

```bash
cd /home/flynn/ggTrader && python -m pytest tests/lab/test_ensemble_conviction.py tests/lab/test_ensemble.py tests/lab/test_conviction.py tests/lab/test_strength_indicators.py -v
```

Expected: all PASS

- [ ] **Step 6: Run full lab test suite**

```bash
cd /home/flynn/ggTrader && python -m pytest tests/lab/ -v
```

Expected: all PASS, no regressions

- [ ] **Step 7: Commit**

```bash
git add src/ggTrader/lab/strategies/signals.py src/ggTrader/lab/cli.py tests/lab/test_ensemble_conviction.py
git commit -m "feat(lab): register ensemble_conviction in CLI and signal registry"
```
