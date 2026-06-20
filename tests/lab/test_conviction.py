"""Tests for conviction-weighted BB sizing strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ggTrader.lab.strategies.conviction import ConvictionBBSignal
from ggTrader.lab.strategies.signals import build_signal_strategy
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


def test_signal_targets_backward_compatible():
    """Existing code creating SignalTargets(entries=..., exits=...) still works."""
    idx = _idx(10)
    df = pd.DataFrame(False, index=idx, columns=["A", "B"])
    st = SignalTargets(entries=df, exits=df)
    assert st.sizes is None
    # Positional also works
    st2 = SignalTargets(df, df)
    assert st2.sizes is None


def test_conviction_bb_returns_signal_targets_with_sizes():
    """to_targets returns a SignalTargets with a non-None sizes DataFrame."""
    cfg = LabConfig(min_history_bars=50)
    strat = ConvictionBBSignal(cfg)
    ohlcv = _ohlcv(n=300)
    symbols = sorted(ohlcv.columns.get_level_values(0).unique())
    plans = {ohlcv.index[100]: [{"symbol": s, "weight": 0.0} for s in symbols]}
    targets = strat.to_targets(plans, ohlcv)
    assert isinstance(targets, SignalTargets)
    assert targets.sizes is not None
    assert targets.sizes.shape == targets.entries.shape


def test_conviction_bb_sizes_nan_where_no_entry():
    """Sizes should be NaN where entries are False."""
    cfg = LabConfig(min_history_bars=50)
    strat = ConvictionBBSignal(cfg)
    ohlcv = _ohlcv(n=300)
    symbols = sorted(ohlcv.columns.get_level_values(0).unique())
    plans = {ohlcv.index[100]: [{"symbol": s, "weight": 0.0} for s in symbols]}
    targets = strat.to_targets(plans, ohlcv)
    no_entry = ~targets.entries
    assert targets.sizes[no_entry].isna().all().all()


def test_conviction_bb_sweep_params():
    """sweep_params returns expected keys."""
    params = ConvictionBBSignal.sweep_params()
    assert "bb_period" in params
    assert "bb_std" in params
    assert "max_size" in params


def test_conviction_bb_select():
    """select returns eligible symbols."""
    cfg = LabConfig(min_history_bars=50)
    strat = ConvictionBBSignal(cfg)
    ohlcv = _ohlcv(n=300)
    symbols = sorted(ohlcv.columns.get_level_values(0).unique())
    plan = strat.select(ohlcv.index[200], ohlcv, symbols)
    assert len(plan) == len(symbols)
    assert all(p["symbol"] in symbols for p in plan)


def test_conviction_bb_sweep_signals():
    """sweep_signals returns results keyed by combo name."""
    cfg = LabConfig(min_history_bars=50)
    strat = ConvictionBBSignal(cfg)
    ohlcv = _ohlcv(n=300)
    symbols = sorted(ohlcv.columns.get_level_values(0).unique())
    combos = [{"bb_period": 20, "bb_std": 2.0, "min_size": 0.01, "max_size": 0.05}]
    results = strat.sweep_signals(combos, symbols, ohlcv)
    assert len(results) == 1
    key = list(results.keys())[0]
    assert results[key].sizes is not None


def test_conviction_bb_registered():
    """conviction_bb is in the signal registry."""
    from ggTrader.lab.strategies.signals import _get_registry

    assert "conviction_bb" in _get_registry()


def test_conviction_bb_build():
    """build_signal_strategy can create ConvictionBBSignal."""

    strat = build_signal_strategy("conviction_bb", LabConfig())
    assert strat.name == "conviction_bb"


def test_cli_accepts_conviction_bb():
    """CLI parser accepts conviction_bb as a strategy choice."""
    from ggTrader.lab.cli import build_arg_parser

    parser = build_arg_parser()
    args = parser.parse_args(["--strategy", "conviction_bb"])
    assert args.strategy == "conviction_bb"


def test_conviction_sizes_override_vol_target():
    """When both vol_target and conviction sizes are present, conviction wins on entry bars."""
    from ggTrader.lab.simulate import simulate_signals

    np.random.seed(42)
    idx = _idx(200)
    prices = pd.DataFrame(
        100.0 * np.exp(np.cumsum(np.random.normal(0.0003, 0.015, (200, 2)), axis=0)),
        index=idx,
        columns=["S0", "S1"],
    )
    # Create a simple signal with entries every 30 bars
    idx = prices.index
    cols = prices.columns
    entries = pd.DataFrame(False, index=idx, columns=cols)
    exits = pd.DataFrame(False, index=idx, columns=cols)
    sizes = pd.DataFrame(np.nan, index=idx, columns=cols)
    for i in range(50, len(idx), 30):
        entries.iloc[i] = True
        sizes.iloc[i] = 0.04  # fixed conviction size
        if i + 5 < len(idx):
            exits.iloc[i + 5] = True

    targets_with_sizes = SignalTargets(
        entries=entries.astype(bool), exits=exits.astype(bool), sizes=sizes
    )
    targets_no_sizes = SignalTargets(entries=entries.astype(bool), exits=exits.astype(bool))

    config_vol = {
        "START_CASH": 10000.0,
        "FEES": 0.0,
        "SLIPPAGE": 0.0,
        "FREQ": "1d",
        "SIGNAL_POSITION_SIZE": 0.02,
        "vol_target": 0.15,
        "vol_lookback": 20,
    }
    # With conviction sizes: sizes override vol-scaled base on entry bars
    _, eq_conviction, _ = simulate_signals({"test": targets_with_sizes}, prices, config_vol)
    # Without conviction sizes: vol targeting scales the flat 2%
    _, eq_vol_only, _ = simulate_signals({"test": targets_no_sizes}, prices, config_vol)

    # Equity curves must differ (conviction uses 4% vs vol-scaled 2%)
    assert not eq_conviction["test"].equals(eq_vol_only["test"]), (
        "Conviction sizes should produce different equity than vol-only"
    )
