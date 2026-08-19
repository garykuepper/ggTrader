"""Tests for the EnsembleSignal majority-vote strategy."""

import numpy as np
import pandas as pd

from ggTrader.lab.strategies.ensemble import EnsembleSignal
from ggTrader.lab.strategy import LabConfig


def _idx(n, start="2020-01-01"):
    return pd.date_range(start, periods=n, freq="B", tz="UTC")


def _ohlcv(n=300, n_syms=3, seed=42):
    """Synthetic OHLCV with (symbol, field) MultiIndex columns."""
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


def test_ensemble_no_entry_when_fewer_than_min_agree():
    """If only 1 sub-signal fires, ensemble should NOT enter."""
    cfg = LabConfig(min_history_bars=50)
    strat = EnsembleSignal(cfg, min_agree=3)
    ohlcv = _ohlcv(n=300)
    symbols = sorted(ohlcv.columns.get_level_values(0).unique())
    close = pd.concat({s: ohlcv[s]["close"] for s in symbols}, axis=1)
    plans = {close.index[100]: [{"symbol": s, "weight": 0.0} for s in symbols]}
    targets = strat.to_targets(plans, ohlcv)
    # With min_agree=3 (all must agree), entries should be very rare or zero
    # on random synthetic data where signals are uncorrelated
    assert targets.entries.sum().sum() <= targets.entries.shape[0] * len(symbols) * 0.01


def test_ensemble_no_entries_before_symbol_eligibility_start():
    """PIT eligibility: a symbol added to the plan pool mid-sample must not
    trade before the rebalance date it first appears in `plans`.

    Regression test for the item-11 PIT-eligibility-masking fix
    (docs/research/2026-08-18-wfo-anchor-leakage-fix.md, ensemble.py to_targets).
    """
    cfg = LabConfig(min_history_bars=20)
    strat = EnsembleSignal(cfg, min_agree=1)
    ohlcv = _ohlcv(n=300, n_syms=2)
    symbols = sorted(ohlcv.columns.get_level_values(0).unique())
    assert symbols == ["S0", "S1"]

    late_start = ohlcv.index[200]
    plans = {
        # S0 eligible from the start of the plan sequence.
        ohlcv.index[30]: [{"symbol": "S0", "weight": 0.0}],
        # S1 only enters the eligible pool at bar 200 (e.g. a late index add).
        late_start: [{"symbol": "S0", "weight": 0.0}, {"symbol": "S1", "weight": 0.0}],
    }
    targets = strat.to_targets(plans, ohlcv)

    # S1 must have zero entries strictly before its first eligible date.
    pre_period = targets.entries.loc[: ohlcv.index[199], "S1"]
    assert not pre_period.any(), "S1 traded before it was eligible"
    # Sanity: S1 does get to trade once eligible (min_agree=1 on a long
    # random-walk sample should produce at least one entry post-eligibility).
    post_period = targets.entries.loc[late_start:, "S1"]
    assert post_period.any(), "S1 never traded even after becoming eligible"
    # S0, eligible from bar 30, should also have pre-200 entries -- proves
    # the mask isn't just blanking everything before bar 200.
    assert targets.entries.loc[ohlcv.index[30] : ohlcv.index[199], "S0"].any()


def test_ensemble_enters_when_all_agree():
    """With min_agree=1, any single signal triggers entry."""
    cfg = LabConfig(min_history_bars=50)
    strat = EnsembleSignal(cfg, min_agree=1)
    ohlcv = _ohlcv(n=300)
    symbols = sorted(ohlcv.columns.get_level_values(0).unique())
    close = pd.concat({s: ohlcv[s]["close"] for s in symbols}, axis=1)
    plans = {close.index[100]: [{"symbol": s, "weight": 0.0} for s in symbols]}
    targets = strat.to_targets(plans, ohlcv)
    # min_agree=1 means any sub-signal fires -> should have more entries
    assert targets.entries.sum().sum() > 0


def test_ensemble_sweep_params_returns_dict():
    assert "min_agree" in EnsembleSignal.sweep_params()
    assert "bb_period" in EnsembleSignal.sweep_params()


def test_ensemble_sweep_signals_keys_match_combos():
    from ggTrader.lab.sweep import combo_name

    cfg = LabConfig(min_history_bars=50)
    strat = EnsembleSignal(cfg)
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
    expected_key = combo_name("ensemble", combos[0])
    assert expected_key in result
    assert hasattr(result[expected_key], "entries")
    assert hasattr(result[expected_key], "exits")


def test_ensemble_select_respects_asof():
    """select() must not look at data past asof."""
    cfg = LabConfig(min_history_bars=50)
    strat = EnsembleSignal(cfg)
    ohlcv = _ohlcv(n=300)
    symbols = sorted(ohlcv.columns.get_level_values(0).unique())
    mid = ohlcv.index[150]
    plan_full = strat.select(mid, ohlcv.loc[:mid], symbols)
    plan_trunc = strat.select(mid, ohlcv, symbols)
    assert len(plan_full) == len(plan_trunc)


def test_ensemble_min_agree_2_fewer_entries_than_1():
    """Higher min_agree -> fewer entries (more filtering)."""
    cfg = LabConfig(min_history_bars=50)
    ohlcv = _ohlcv(n=300, seed=123)
    symbols = sorted(ohlcv.columns.get_level_values(0).unique())
    close = pd.concat({s: ohlcv[s]["close"] for s in symbols}, axis=1)

    strat1 = EnsembleSignal(cfg, min_agree=1)
    strat2 = EnsembleSignal(cfg, min_agree=2)
    plans = {close.index[100]: [{"symbol": s, "weight": 0.0} for s in symbols]}

    t1 = strat1.to_targets(plans, ohlcv)
    t2 = strat2.to_targets(plans, ohlcv)
    assert t1.entries.sum().sum() >= t2.entries.sum().sum()


def test_ensemble_target_kind():
    """Ensemble must declare target_kind='signals'."""
    assert EnsembleSignal.target_kind == "signals"
    assert EnsembleSignal.name == "ensemble"


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
    assert targets.entries.sum().sum() <= targets.entries.shape[0] * len(symbols) * 0.005


def test_asymmetric_exit_more_exits_than_symmetric():
    """min_agree_exit=1 should produce >= exits than min_agree_exit=min_agree."""
    cfg = LabConfig(min_history_bars=50)
    ohlcv = _ohlcv(n=300, seed=99)
    symbols = sorted(ohlcv.columns.get_level_values(0).unique())
    plans = {ohlcv.index[100]: [{"symbol": s, "weight": 0.0} for s in symbols]}

    sym = EnsembleSignal(cfg, min_agree=2, min_agree_exit=2)
    asym = EnsembleSignal(cfg, min_agree=2, min_agree_exit=1)

    t_sym = sym.to_targets(plans, ohlcv)
    t_asym = asym.to_targets(plans, ohlcv)

    assert t_asym.exits.sum().sum() >= t_sym.exits.sum().sum()


def test_min_agree_exit_defaults_to_min_agree():
    cfg = LabConfig(min_history_bars=50)
    strat = EnsembleSignal(cfg, min_agree=3)
    assert strat.min_agree_exit == 3


def test_sweep_params_includes_min_agree_exit():
    params = EnsembleSignal.sweep_params()
    assert "min_agree_exit" in params
    assert 1 in params["min_agree_exit"]


# ── Configurable voters ────────────────────────────────────────────────


def test_voters_default_is_validated_five_voter():
    """Default = ablation-validated 5-voter (2026-06-24): all voters minus MTF."""
    cfg = LabConfig(min_history_bars=50)
    strat = EnsembleSignal(cfg)
    assert set(strat.voters) == {"bb", "rsi", "ema", "macd", "vbb"}
    assert "mtf" not in strat.voters


def test_three_voter_matches_manual_bb_rsi_ema():
    """voters=(bb,rsi,ema) must equal a hand-computed 3-voter vote."""
    from ggTrader.lab.strategies.indicators import (
        bb_signals,
        eligibility_mask,
        ema_signals,
        extract_close,
        rsi_signals,
    )

    cfg = LabConfig(min_history_bars=50)
    ohlcv = _ohlcv(n=300, seed=7)
    symbols = sorted(ohlcv.columns.get_level_values(0).unique())
    close = extract_close(ohlcv, symbols)

    strat = EnsembleSignal(cfg, min_agree=2, min_agree_exit=2, voters=("bb", "rsi", "ema"))
    plans = {ohlcv.index[100]: [{"symbol": s, "weight": 0.0} for s in symbols]}
    got = strat.to_targets(plans, ohlcv)

    bb_ent, bb_ext = bb_signals(close, strat.bb_period, strat.bb_std)
    rsi_ent, rsi_ext = rsi_signals(close, strat.rsi_period, strat.rsi_oversold, strat.rsi_exit)
    ema_ent, ema_ext = ema_signals(close, strat.ema_fast, strat.ema_slow)
    entry_votes = bb_ent.astype(int) + rsi_ent.astype(int) + ema_ent.astype(int)
    exit_votes = bb_ext.astype(int) + rsi_ext.astype(int) + ema_ext.astype(int)
    # PIT eligibility: the plan's single asof (bar 100) means entries before
    # that bar are masked out -- see eligibility_mask.
    mask = eligibility_mask(plans, symbols, ohlcv.index)
    exp_entries = (entry_votes >= 2).astype(bool) & mask
    exp_exits = rsi_ext | (exit_votes >= 2)

    pd.testing.assert_frame_equal(got.entries, exp_entries)
    pd.testing.assert_frame_equal(got.exits, exp_exits)


def test_three_voter_differs_from_six_voter():
    cfg = LabConfig(min_history_bars=50)
    ohlcv = _ohlcv(n=300, seed=11)
    symbols = sorted(ohlcv.columns.get_level_values(0).unique())
    plans = {ohlcv.index[100]: [{"symbol": s, "weight": 0.0} for s in symbols]}

    three = EnsembleSignal(cfg, min_agree=2, voters=("bb", "rsi", "ema")).to_targets(plans, ohlcv)
    six = EnsembleSignal(cfg, min_agree=2).to_targets(plans, ohlcv)
    assert not three.entries.equals(six.entries)


def test_voters_threaded_through_sweep_signals():
    """sweep_signals must not silently drop the voters config."""
    from ggTrader.lab.sweep import combo_name

    cfg = LabConfig(min_history_bars=50)
    ohlcv = _ohlcv(n=200, seed=5)
    symbols = sorted(ohlcv.columns.get_level_values(0).unique())

    strat = EnsembleSignal(cfg, voters=("bb", "rsi", "ema"))
    combo = {"min_agree": 2, "min_agree_exit": 2, "bb_std": 2.0, "rsi_oversold": 30, "ema_fast": 20}
    swept = strat.sweep_signals([combo], symbols, ohlcv)
    key = combo_name("ensemble", combo)

    direct = EnsembleSignal(
        cfg, min_agree=2, min_agree_exit=2, voters=("bb", "rsi", "ema")
    ).to_targets({ohlcv.index[50]: [{"symbol": s, "weight": 0.0} for s in symbols]}, ohlcv)
    pd.testing.assert_frame_equal(swept[key].entries, direct.entries)


def test_invalid_voter_name_raises():
    cfg = LabConfig(min_history_bars=50)
    try:
        EnsembleSignal(cfg, voters=("bb", "nope"))
    except ValueError as exc:
        assert "nope" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown voter")


# ── Exit-rule controls: time-stop (td_stop) + exits_enabled ────────────


def _entry_plans(ohlcv):
    symbols = sorted(ohlcv.columns.get_level_values(0).unique())
    return symbols, {ohlcv.index[100]: [{"symbol": s, "weight": 0.0} for s in symbols]}


def test_td_stop_and_exits_enabled_defaults():
    """New exit knobs default to off: td_stop=None, exits_enabled=True."""
    cfg = LabConfig(min_history_bars=50)
    strat = EnsembleSignal(cfg)
    assert strat.td_stop is None
    assert strat.exits_enabled is True


def test_td_stop_none_matches_baseline_exits():
    """td_stop=None must leave the exits matrix byte-identical to before."""
    cfg = LabConfig(min_history_bars=50)
    ohlcv = _ohlcv(n=300, seed=3)
    _, plans = _entry_plans(ohlcv)
    base = EnsembleSignal(cfg, min_agree=1).to_targets(plans, ohlcv)
    td = EnsembleSignal(cfg, min_agree=1, td_stop=None).to_targets(plans, ohlcv)
    pd.testing.assert_frame_equal(base.exits, td.exits)


def test_td_stop_adds_exit_n_bars_after_each_entry():
    """With td_stop=N, every entry has a forced exit exactly N bars later."""
    cfg = LabConfig(min_history_bars=50)
    ohlcv = _ohlcv(n=300, seed=3)
    _, plans = _entry_plans(ohlcv)
    strat = EnsembleSignal(cfg, min_agree=1, td_stop=5)
    t = strat.to_targets(plans, ohlcv)
    assert t.entries.sum().sum() > 0  # meaningful
    shifted = t.entries.shift(5, fill_value=False).astype(bool)
    # every shifted entry must be present in exits (time-stop OR'd in)
    assert int((shifted & ~t.exits).sum().sum()) == 0


def test_td_stop_only_adds_exits_never_removes():
    """Time-stop is additive: baseline exits remain a subset."""
    cfg = LabConfig(min_history_bars=50)
    ohlcv = _ohlcv(n=300, seed=3)
    _, plans = _entry_plans(ohlcv)
    base = EnsembleSignal(cfg, min_agree=1).to_targets(plans, ohlcv)
    td = EnsembleSignal(cfg, min_agree=1, td_stop=5).to_targets(plans, ohlcv)
    assert int((base.exits & ~td.exits).sum().sum()) == 0


def test_exits_enabled_false_suppresses_indicator_exits():
    """Replacement arm with no stops: no exits at all (position would not close)."""
    cfg = LabConfig(min_history_bars=50)
    ohlcv = _ohlcv(n=300, seed=3)
    _, plans = _entry_plans(ohlcv)
    strat = EnsembleSignal(cfg, min_agree=1, exits_enabled=False)
    t = strat.to_targets(plans, ohlcv)
    assert int(t.exits.sum().sum()) == 0


def test_exits_enabled_false_with_td_stop_exits_only_on_time():
    """Replacement arm: exits are exactly the time-stop shifts, nothing else."""
    cfg = LabConfig(min_history_bars=50)
    ohlcv = _ohlcv(n=300, seed=3)
    _, plans = _entry_plans(ohlcv)
    strat = EnsembleSignal(cfg, min_agree=1, exits_enabled=False, td_stop=5)
    t = strat.to_targets(plans, ohlcv)
    expected = t.entries.shift(5, fill_value=False).astype(bool)
    pd.testing.assert_frame_equal(t.exits, expected)


def test_exits_enabled_false_does_not_change_entries():
    """Suppressing exits must not touch the entry signals."""
    cfg = LabConfig(min_history_bars=50)
    ohlcv = _ohlcv(n=300, seed=3)
    _, plans = _entry_plans(ohlcv)
    on = EnsembleSignal(cfg, min_agree=1).to_targets(plans, ohlcv)
    off = EnsembleSignal(cfg, min_agree=1, exits_enabled=False).to_targets(plans, ohlcv)
    pd.testing.assert_frame_equal(on.entries, off.entries)


def test_exit_controls_threaded_through_sweep_signals():
    """sweep_signals must honor td_stop and exits_enabled from the combo."""
    from ggTrader.lab.sweep import combo_name

    cfg = LabConfig(min_history_bars=50)
    ohlcv = _ohlcv(n=200, seed=5)
    symbols = sorted(ohlcv.columns.get_level_values(0).unique())
    strat = EnsembleSignal(cfg, min_agree=1)
    combo = {"min_agree": 1, "td_stop": 5, "exits_enabled": False}
    swept = strat.sweep_signals([combo], symbols, ohlcv)
    key = combo_name("ensemble", combo)

    # sweep_signals has no per-date PIT eligibility concept (no `plans` dict);
    # to_targets does, and masks entries to bars on/after the plan's asof
    # date (see eligibility_mask). Use index[0] here so the mask covers the
    # whole window and this test compares exit-control wiring, not masking.
    direct = EnsembleSignal(cfg, min_agree=1, td_stop=5, exits_enabled=False).to_targets(
        {ohlcv.index[0]: [{"symbol": s, "weight": 0.0} for s in symbols]}, ohlcv
    )
    pd.testing.assert_frame_equal(swept[key].exits, direct.exits)
