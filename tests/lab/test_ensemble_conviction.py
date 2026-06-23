"""Tests for EnsembleConvictionSignal — conviction-weighted ensemble sizing."""

import numpy as np
import pandas as pd

from ggTrader.lab.simulate import simulate_signals
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
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
                "divergence_window": 20,
                "vol_period": 20,
                "vol_mult": 2.0,
                "weekly_rsi_period": 14,
                "weekly_rsi_oversold": 30,
                "weekly_rsi_exit": 50,
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
