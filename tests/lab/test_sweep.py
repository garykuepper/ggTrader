"""Tests for parameter sweep tooling."""

import numpy as np
import pandas as pd
import pytest

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


@pytest.mark.integration
def test_sweep_persistence_roundtrip():
    from sqlalchemy import text

    from ggTrader.lab.persist import (
        finish_sweep,
        get_engine,
        init_schema,
        start_sweep,
        write_sweep_combo,
    )

    init_schema()
    sweep_id = start_sweep(
        "ema_cross",
        "equity",
        {"ema_fast": [5, 10], "ema_slow": [20, 50]},
        4,
    )
    assert sweep_id.startswith("sweep_ema_cross_")

    write_sweep_combo(
        sweep_id,
        "ema_cross__ema_fast5_ema_slow20",
        {"ema_fast": 5, "ema_slow": 20},
        {"sharpe": 0.42, "cagr_pct": 3.1},
        {"sharpe": 0.85},
        {"n_symbols": 50},
    )
    finish_sweep(sweep_id)

    with get_engine().connect() as conn:
        row = conn.execute(
            text("SELECT status FROM lab_sweeps WHERE sweep_id = :s"),
            {"s": sweep_id},
        ).first()
        assert row[0] == "done"
        combo_row = conn.execute(
            text("SELECT params, metrics FROM lab_sweep_combos WHERE sweep_id = :s"),
            {"s": sweep_id},
        ).first()
        assert combo_row[0]["ema_fast"] == 5
        assert combo_row[1]["sharpe"] == 0.42


def test_format_results_table_renders_header_and_rows():
    from ggTrader.lab.sweep import format_results_table

    rows = [
        {
            "combo": "ema_cross__ema_fast5_ema_slow20",
            "sharpe": 0.42,
            "cagr_pct": 3.1,
            "max_drawdown_pct": -18.2,
            "sortino": 0.61,
            "total_return_pct": 12.4,
        },
        {
            "combo": "ema_cross__ema_fast10_ema_slow30",
            "sharpe": 0.38,
            "cagr_pct": 2.7,
            "max_drawdown_pct": -21.0,
            "sortino": 0.54,
            "total_return_pct": 10.8,
        },
    ]
    spy = {"cagr_pct": 18.2, "sharpe": 0.85, "max_drawdown_pct": -24.5}
    table = format_results_table(
        rows, "ema_cross", 2, "2021-01-31", "2026-06-17", "sweep_ema_cross_abc123", spy
    )
    assert "ema_cross" in table
    assert "2 combos" in table
    assert "Sharpe" in table
    assert "SPY" in table
    assert "0.42" in table


def test_format_results_table_sorted_by_sharpe():
    from ggTrader.lab.sweep import format_results_table

    rows = [
        {
            "combo": "a",
            "sharpe": -0.1,
            "cagr_pct": 0,
            "max_drawdown_pct": 0,
            "sortino": 0,
            "total_return_pct": 0,
        },
        {
            "combo": "b",
            "sharpe": 0.5,
            "cagr_pct": 0,
            "max_drawdown_pct": 0,
            "sortino": 0,
            "total_return_pct": 0,
        },
    ]
    table = format_results_table(
        rows,
        "x",
        2,
        "2021",
        "2026",
        "id",
        {"cagr_pct": 0, "sharpe": 0, "max_drawdown_pct": 0},
    )
    lines = table.strip().split("\n")
    # First data row (after header lines) should be the higher-sharpe combo
    data_lines = [l for l in lines if l.strip().startswith(("1", "2"))]
    assert "b" in data_lines[0]
    assert "a" in data_lines[1]


def test_cli_parser_accepts_sweep_flag():
    from ggTrader.lab.cli import build_arg_parser

    p = build_arg_parser()
    args = p.parse_args(["--strategy", "ema_cross", "--sweep"])
    assert args.sweep is True


def test_cli_parser_accepts_sweep_param():
    from ggTrader.lab.cli import build_arg_parser

    p = build_arg_parser()
    args = p.parse_args(
        [
            "--strategy",
            "ema_cross",
            "--sweep",
            "--sweep-param",
            "ema_fast=5,10",
            "--sweep-param",
            "ema_slow=50,100",
        ]
    )
    assert args.sweep_param == ["ema_fast=5,10", "ema_slow=50,100"]


def test_cli_parser_sweep_param_without_sweep_is_ok():
    from ggTrader.lab.cli import build_arg_parser

    p = build_arg_parser()
    args = p.parse_args(["--strategy", "ema_cross"])
    assert args.sweep is False
    assert args.sweep_param == []


def test_sweep_param_coerces_none_and_bool():
    """Exit-sweep grids need None / True / False, not the literal strings."""
    from ggTrader.lab.cli import _parse_sweep_params

    out = _parse_sweep_params(
        ["td_stop=None,5,10", "exits_enabled=True,False", "tp_stop=None,0.03"]
    )
    assert out["td_stop"] == [None, 5, 10]
    assert out["exits_enabled"] == [True, False]
    assert out["tp_stop"] == [None, 0.03]


@pytest.mark.integration
def test_sweep_end_to_end_ema_cross_small_grid():
    """Full sweep: 2 combos, synthetic OHLCV, hits DB."""
    from sqlalchemy import text

    from ggTrader.lab.data import STOCK_BASE_CONFIG
    from ggTrader.lab.persist import get_engine, init_schema
    from ggTrader.lab.strategies.signals import EmaCrossSignal
    from ggTrader.lab.sweep import build_grid, run_sweep

    init_schema()
    ohlcv = _ohlcv(["A", "B"], n=600)
    spy_idx = ohlcv.index
    spy_close = pd.Series(100.0 * 1.0004 ** np.arange(len(spy_idx)), index=spy_idx)

    grid = build_grid(EmaCrossSignal, overrides={"ema_fast": [5, 10], "ema_slow": [50]})
    assert len(grid) == 2

    cfg = LabConfig(min_history_bars=100, top_n=2)
    sweep_id = run_sweep(
        "ema_cross",
        EmaCrossSignal,
        cfg,
        ohlcv,
        spy_close,
        eval_start=str(ohlcv.index[200].date()),
        eval_end=str(ohlcv.index[-1].date()),
        market="equity",
        base_config=dict(STOCK_BASE_CONFIG),
        grid=grid,
    )
    assert sweep_id.startswith("sweep_ema_cross_")

    with get_engine().connect() as conn:
        sweep_row = conn.execute(
            text("SELECT status, n_combos FROM lab_sweeps WHERE sweep_id = :s"),
            {"s": sweep_id},
        ).first()
        assert sweep_row[0] == "done"
        assert sweep_row[1] == 2

        combo_count = conn.execute(
            text("SELECT count(*) FROM lab_sweep_combos WHERE sweep_id = :s"),
            {"s": sweep_id},
        ).scalar()
        assert combo_count == 2


def test_split_params_separates_signal_and_stop():
    from ggTrader.lab.sweep import split_params

    combo = {"ema_fast": 10, "ema_slow": 50, "ts_stop": 0.03}
    signal, stop = split_params(combo)
    assert signal == {"ema_fast": 10, "ema_slow": 50}
    assert stop == {"ts_stop": 0.03}


def test_split_params_no_stop_params():
    from ggTrader.lab.sweep import split_params

    combo = {"ema_fast": 10, "ema_slow": 50}
    signal, stop = split_params(combo)
    assert signal == {"ema_fast": 10, "ema_slow": 50}
    assert stop == {}


def test_split_params_atr_params():
    from ggTrader.lab.sweep import split_params

    combo = {"ema_fast": 10, "ema_slow": 50, "atr_period": 14, "atr_mult": 2.0}
    signal, stop = split_params(combo)
    assert signal == {"ema_fast": 10, "ema_slow": 50}
    assert stop == {"atr_period": 14, "atr_mult": 2.0}


def test_stop_params_constant():
    from ggTrader.lab.sweep import STOP_PARAMS

    assert "ts_stop" in STOP_PARAMS
    assert "atr_period" in STOP_PARAMS
    assert "atr_mult" in STOP_PARAMS
    assert "ema_fast" not in STOP_PARAMS


def test_grid_rejects_ts_stop_and_atr_mult_together():
    from ggTrader.lab.sweep import _is_valid_combo

    assert _is_valid_combo({"ema_fast": 10, "ema_slow": 50, "ts_stop": 0.03}) is True
    assert _is_valid_combo({"ema_fast": 10, "ema_slow": 50, "atr_mult": 2.0}) is True
    assert _is_valid_combo({"ts_stop": 0.03, "atr_mult": 2.0}) is False
    assert _is_valid_combo({"ema_fast": 10, "ts_stop": 0.03, "atr_mult": 2.0}) is False


def test_tp_stop_in_stop_params():
    from ggTrader.lab.sweep import STOP_PARAMS

    assert "tp_stop" in STOP_PARAMS


def test_split_params_routes_tp_stop_to_overlay():
    """tp_stop is a portfolio-side stop, not a signal param."""
    from ggTrader.lab.sweep import split_params

    signal, overlay = split_params({"min_agree": 2, "tp_stop": 0.05})
    assert signal == {"min_agree": 2}
    assert overlay == {"tp_stop": 0.05}


def test_split_params_keeps_td_stop_and_exits_enabled_as_signal():
    """td_stop and exits_enabled are signal params (go to the strategy ctor)."""
    from ggTrader.lab.sweep import split_params

    signal, overlay = split_params(
        {"min_agree": 2, "td_stop": 5, "exits_enabled": False, "tp_stop": 0.05}
    )
    assert signal == {"min_agree": 2, "td_stop": 5, "exits_enabled": False}
    assert overlay == {"tp_stop": 0.05}


def test_valid_combo_rejects_strategy_that_never_exits():
    """exits_enabled=False with neither a time-stop nor a take-profit can never
    close a position -> invalid."""
    from ggTrader.lab.sweep import _is_valid_combo

    assert _is_valid_combo({"min_agree": 2, "exits_enabled": False}) is False
    assert _is_valid_combo({"min_agree": 2, "exits_enabled": False, "td_stop": 5}) is True
    assert _is_valid_combo({"min_agree": 2, "exits_enabled": False, "tp_stop": 0.05}) is True
    # td_stop / tp_stop explicitly None still count as "no exit"
    assert (
        _is_valid_combo({"min_agree": 2, "exits_enabled": False, "td_stop": None, "tp_stop": None})
        is False
    )
    # exits_enabled True (default) always has the indicator exits available
    assert _is_valid_combo({"min_agree": 2, "exits_enabled": True}) is True
    assert _is_valid_combo({"min_agree": 2}) is True


@pytest.mark.integration
def test_sweep_with_fixed_trailing_stop():
    """Sweep with ts_stop produces different results than without."""
    from sqlalchemy import text

    from ggTrader.lab.data import STOCK_BASE_CONFIG
    from ggTrader.lab.persist import get_engine, init_schema
    from ggTrader.lab.strategies.signals import EmaCrossSignal
    from ggTrader.lab.sweep import run_sweep

    init_schema()
    ohlcv = _ohlcv(["A", "B"], n=600)
    spy_idx = ohlcv.index
    spy_close = pd.Series(100.0 * 1.0004 ** np.arange(len(spy_idx)), index=spy_idx)

    # 2 entry combos x 2 stop values = 4 total combos
    grid = [
        {"ema_fast": 5, "ema_slow": 50},
        {"ema_fast": 5, "ema_slow": 50, "ts_stop": 0.03},
        {"ema_fast": 10, "ema_slow": 50},
        {"ema_fast": 10, "ema_slow": 50, "ts_stop": 0.03},
    ]

    cfg = LabConfig(min_history_bars=100, top_n=2)
    sweep_id = run_sweep(
        "ema_cross",
        EmaCrossSignal,
        cfg,
        ohlcv,
        spy_close,
        eval_start=str(ohlcv.index[200].date()),
        eval_end=str(ohlcv.index[-1].date()),
        market="equity",
        base_config=dict(STOCK_BASE_CONFIG),
        grid=grid,
    )

    with get_engine().connect() as conn:
        combo_count = conn.execute(
            text("SELECT count(*) FROM lab_sweep_combos WHERE sweep_id = :s"),
            {"s": sweep_id},
        ).scalar()
        assert combo_count == 4


@pytest.mark.integration
def test_sweep_with_atr_trailing_stop():
    """Sweep with atr_mult produces results and persists to DB."""
    from sqlalchemy import text

    from ggTrader.lab.data import STOCK_BASE_CONFIG
    from ggTrader.lab.persist import get_engine, init_schema
    from ggTrader.lab.strategies.signals import EmaCrossSignal
    from ggTrader.lab.sweep import run_sweep

    init_schema()
    ohlcv = _ohlcv(["A", "B"], n=600)
    spy_idx = ohlcv.index
    spy_close = pd.Series(100.0 * 1.0004 ** np.arange(len(spy_idx)), index=spy_idx)

    grid = [
        {"ema_fast": 10, "ema_slow": 50, "atr_mult": 2.0, "atr_period": 14},
        {"ema_fast": 10, "ema_slow": 50, "atr_mult": 3.0, "atr_period": 14},
        {"ema_fast": 10, "ema_slow": 50},  # no stop baseline
    ]

    cfg = LabConfig(min_history_bars=100, top_n=2)
    sweep_id = run_sweep(
        "ema_cross",
        EmaCrossSignal,
        cfg,
        ohlcv,
        spy_close,
        eval_start=str(ohlcv.index[200].date()),
        eval_end=str(ohlcv.index[-1].date()),
        market="equity",
        base_config=dict(STOCK_BASE_CONFIG),
        grid=grid,
    )

    with get_engine().connect() as conn:
        sweep_row = conn.execute(
            text("SELECT status, n_combos FROM lab_sweeps WHERE sweep_id = :s"),
            {"s": sweep_id},
        ).first()
        assert sweep_row[0] == "done"
        assert sweep_row[1] == 3

        combos = conn.execute(
            text(
                "SELECT combo_name, params FROM lab_sweep_combos"
                " WHERE sweep_id = :s ORDER BY combo_name"
            ),
            {"s": sweep_id},
        ).fetchall()
        assert len(combos) == 3
        # At least one combo should have atr_mult in its params
        atr_combos = [c for c in combos if c[1].get("atr_mult") is not None]
        assert len(atr_combos) == 2
