"""Tests for parameter sweep tooling."""

from ggTrader.lab.strategies.momentum import CrossSectionalMomentum, DualMomentum
from ggTrader.lab.strategies.signals import EmaCrossSignal, WfoTournamentSignal


def test_ema_cross_sweep_params_returns_fast_and_slow():
    params = EmaCrossSignal.sweep_params()
    assert "ema_fast" in params
    assert "ema_slow" in params
    assert all(isinstance(v, list) and len(v) > 1 for v in params.values())


def test_wfo_tournament_sweep_params_returns_is_fraction():
    params = WfoTournamentSignal.sweep_params()
    assert "is_fraction" in params
    assert all(0.0 < f < 1.0 for f in params["is_fraction"])


def test_xs_momentum_sweep_params_returns_labconfig_params():
    params = CrossSectionalMomentum.sweep_params()
    assert "top_n" in params
    assert "lookback" in params
    assert "skip" in params


def test_dual_momentum_inherits_sweep_params():
    params = DualMomentum.sweep_params()
    assert "top_n" in params
