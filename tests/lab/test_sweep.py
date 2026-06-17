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


def test_build_grid_cartesian_product():
    from ggTrader.lab.sweep import build_grid

    grid = build_grid(EmaCrossSignal)
    # 4 fast x 5 slow = 20, minus invalid (fast >= slow)
    # Invalid: (50, 20), (50, 30), (50, 50), (20, 20), (10, 10 — not in slow), ...
    # fast=5: all 5 slow valid → 5
    # fast=10: slow 20,30,50,100,200 → 5
    # fast=20: slow 30,50,100,200 → 4 (skip 20)
    # fast=50: slow 100,200 → 2 (skip 20,30,50)
    # Total: 5+5+4+2 = 16
    assert len(grid) == 16
    assert all("ema_fast" in c and "ema_slow" in c for c in grid)
    # No combo has fast >= slow
    assert all(c["ema_fast"] < c["ema_slow"] for c in grid)


def test_build_grid_with_overrides():
    from ggTrader.lab.sweep import build_grid

    grid = build_grid(EmaCrossSignal, overrides={"ema_fast": [5, 10], "ema_slow": [50, 100]})
    assert len(grid) == 4  # 2 x 2, all valid
    assert all(c["ema_fast"] in (5, 10) for c in grid)


def test_build_grid_no_constraint_strategies():
    from ggTrader.lab.sweep import build_grid

    grid = build_grid(CrossSectionalMomentum)
    # 3 top_n x 2 lookback x 2 skip = 12, no constraint filtering
    assert len(grid) == 12


def test_combo_name_deterministic():
    from ggTrader.lab.sweep import combo_name

    assert (
        combo_name("ema_cross", {"ema_fast": 5, "ema_slow": 20})
        == "ema_cross__ema_fast5_ema_slow20"
    )
    assert (
        combo_name("ema_cross", {"ema_slow": 20, "ema_fast": 5})
        == "ema_cross__ema_fast5_ema_slow20"
    )


def test_combo_name_single_param():
    from ggTrader.lab.sweep import combo_name

    assert combo_name("wfo_tournament", {"is_fraction": 0.7}) == "wfo_tournament__is_fraction0.7"
