"""Tests for parameter sweep tooling."""

import numpy as np
import pandas as pd

from ggTrader.lab.strategies.momentum import CrossSectionalMomentum, DualMomentum
from ggTrader.lab.strategies.signals import EmaCrossSignal, WfoTournamentSignal
from ggTrader.lab.strategy import LabConfig, SignalTargets


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


def _ohlcv(symbols, n=600):
    idx = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
    frames = {}
    for i, s in enumerate(symbols):
        close = pd.Series(100.0 * (1 + 0.0003 * (i + 1)) ** np.arange(n), index=idx)
        frames[s] = pd.DataFrame(
            {
                "open": close,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": np.full(n, 1e6),
            },
            index=idx,
        )
    out = pd.concat(frames, axis=1)
    out.columns = out.columns.set_names(["symbol", "field"])
    return out


def test_ema_cross_sweep_signals_returns_all_combos():
    ohlcv = _ohlcv(["A", "B"])
    combos = [
        {"ema_fast": 5, "ema_slow": 20},
        {"ema_fast": 10, "ema_slow": 50},
    ]
    strat = EmaCrossSignal(LabConfig(min_history_bars=100))
    result = strat.sweep_signals(combos, ["A", "B"], ohlcv)
    assert len(result) == 2
    for key, st in result.items():
        assert isinstance(st, SignalTargets)
        assert set(st.entries.columns) == {"A", "B"}
        assert (st.entries.dtypes == bool).all()
        assert (st.exits.dtypes == bool).all()


def test_ema_cross_sweep_signals_matches_single_run():
    """Vectorized sweep must produce identical signals to single-combo to_targets."""
    ohlcv = _ohlcv(["A", "B"])
    cfg = LabConfig(min_history_bars=100)
    fast, slow = 10, 50
    strat = EmaCrossSignal(cfg, ema_fast=fast, ema_slow=slow)
    plans = {
        ohlcv.index[200]: [
            {"symbol": "A", "weight": 0.0, "ema_fast": fast, "ema_slow": slow},
            {"symbol": "B", "weight": 0.0, "ema_fast": fast, "ema_slow": slow},
        ]
    }
    single = strat.to_targets(plans, ohlcv)

    combos = [{"ema_fast": fast, "ema_slow": slow}]
    sweep_result = strat.sweep_signals(combos, ["A", "B"], ohlcv)
    sweep_st = list(sweep_result.values())[0]

    pd.testing.assert_frame_equal(single.entries, sweep_st.entries)
    pd.testing.assert_frame_equal(single.exits, sweep_st.exits)


def test_ema_cross_sweep_signals_different_combos_differ():
    # Use oscillating data so different EMA combos produce distinct crossovers
    idx = pd.date_range("2020-01-01", periods=600, freq="B", tz="UTC")
    close = pd.Series(100.0 + 10.0 * np.sin(np.arange(600) * 2 * np.pi / 60), index=idx)
    ohlcv = pd.concat(
        {
            "A": pd.DataFrame(
                {
                    "open": close,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": np.full(600, 1e6),
                },
                index=idx,
            )
        },
        axis=1,
    )
    ohlcv.columns = ohlcv.columns.set_names(["symbol", "field"])
    combos = [
        {"ema_fast": 5, "ema_slow": 20},
        {"ema_fast": 50, "ema_slow": 200},
    ]
    strat = EmaCrossSignal(LabConfig(min_history_bars=100))
    result = strat.sweep_signals(combos, ["A"], ohlcv)
    keys = list(result.keys())
    assert not result[keys[0]].entries.equals(result[keys[1]].entries)


def test_wfo_tournament_sweep_signals_returns_all_combos():
    ohlcv = _ohlcv(["A", "B"])
    combos = [{"is_fraction": 0.5}, {"is_fraction": 0.8}]
    strat = WfoTournamentSignal(LabConfig(min_history_bars=100))
    result = strat.sweep_signals(combos, ["A", "B"], ohlcv)
    assert len(result) == 2
    for st in result.values():
        assert isinstance(st, SignalTargets)
