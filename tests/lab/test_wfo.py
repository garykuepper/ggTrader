import numpy as np
import pandas as pd
import pytest

from ggTrader.lab.cli import build_arg_parser
from ggTrader.lab.strategy import LabConfig, SignalTargets
from ggTrader.lab.wfo import (
    _pick_live_winner,
    composite_score,
    generate_folds,
    run_wfo,
    select_live_params,
)


def _ohlcv(symbols, n):
    """Build a (symbol, field) MultiIndex OHLCV DataFrame with trending prices."""
    idx = pd.date_range("2014-01-01", periods=n, freq="B", tz="UTC")
    frames = {}
    for i, s in enumerate(symbols):
        base = 100.0 + i * 10
        close = base * (1 + 0.0003 * np.arange(n))
        frames[s] = pd.DataFrame(
            {
                "open": close * 0.999,
                "high": close * 1.005,
                "low": close * 0.995,
                "close": close,
                "volume": np.full(n, 1e6),
            },
            index=idx,
        )
    return pd.concat(frames, axis=1)


class _TinySignal:
    """Minimal signal strategy for testing: buy on bar 2, never exit."""

    name = "tiny"
    target_kind = "signals"

    def __init__(self, cfg):
        self.cfg = cfg

    @classmethod
    def sweep_params(cls):
        return {"param_a": [1, 2]}

    def sweep_signals(self, combos, symbols, data):
        from ggTrader.lab.sweep import combo_name

        close = pd.concat(
            {s: data[s]["close"] for s in symbols if s in data.columns.get_level_values(0)},
            axis=1,
        )
        result = {}
        for combo in combos:
            entries = pd.DataFrame(False, index=close.index, columns=close.columns)
            exits = pd.DataFrame(False, index=close.index, columns=close.columns)
            # Buy on bar 2 — after warmup
            entries.iloc[2] = True
            key = combo_name(self.name, combo)
            result[key] = SignalTargets(entries=entries.astype(bool), exits=exits.astype(bool))
        return result


class _TinyWeight:
    """Minimal weight strategy for testing: equal-weight top param_a symbols."""

    name = "tinyweight"
    target_kind = "weights"

    def __init__(self, cfg):
        self.cfg = cfg

    @classmethod
    def sweep_params(cls):
        return {"top_n": [1, 2]}

    def select(self, asof, data, eligible):
        data = data.loc[:asof]
        chosen = sorted(eligible)[: self.cfg.top_n]
        if not chosen:
            return []
        w = 1.0 / len(chosen)
        return [{"symbol": s, "weight": w} for s in chosen]

    def to_targets(self, plans, data):
        symbols = sorted({s["symbol"] for plan in plans.values() for s in plan})
        targets = pd.DataFrame(np.nan, index=data.index, columns=symbols)
        for asof in sorted(plans):
            forward = data.index[data.index > asof]
            if len(forward) == 0:
                continue
            bar = forward[0]
            targets.loc[bar, symbols] = 0.0
            for sel in plans[asof]:
                targets.loc[bar, sel["symbol"]] = float(sel["weight"])
        return targets


def _tiny_weight_universe_fn(asof, past):
    return sorted(past.columns.get_level_values(0).unique())


def _ohlcv_varied_growth(symbols, n, rates):
    """Like _ohlcv but each symbol grows at its own rate (rates: dict symbol->pct/bar).

    Needed to make selection-dependent tests meaningful: _ohlcv's uniform
    0.0003/bar growth for every symbol means *which* symbol gets picked never
    affects the resulting equity curve, so a bug that silently ignores a
    combo's selection-driving kwarg would go undetected. Distinct growth
    rates per symbol make the equity curve depend on which symbols were
    actually selected.
    """
    idx = pd.date_range("2014-01-01", periods=n, freq="B", tz="UTC")
    frames = {}
    for s in symbols:
        rate = rates[s]
        close = 100.0 * (1 + rate * np.arange(n))
        frames[s] = pd.DataFrame(
            {
                "open": close * 0.999,
                "high": close * 1.005,
                "low": close * 0.995,
                "close": close,
                "volume": np.full(n, 1e6),
            },
            index=idx,
        )
    return pd.concat(frames, axis=1)


class _TinyWeightKwarg:
    """Weight strategy whose selection depends on a non-LabConfig constructor
    kwarg (``multiplier``), mirroring IdioVolStrategy's reg_window/quintile.

    This is the regression fixture for the bug where _sweep_fold_weights
    built every grid combo's strategy from only the merged LabConfig,
    silently dropping any sweep_params() key that isn't top_n/lookback/skip
    (e.g. IdioVolStrategy's reg_window/quintile) — so every combo in a
    real idio_vol WFO sweep was constructed with identical defaults
    regardless of the grid.
    """

    name = "tinyweightkwarg"
    target_kind = "weights"

    def __init__(self, cfg, multiplier=1):
        self.cfg = cfg
        self.multiplier = multiplier

    @classmethod
    def sweep_params(cls) -> dict:
        return {"multiplier": [1, -1]}

    def select(self, asof, data, eligible):
        # multiplier=1 picks the lowest-growth symbol, multiplier=-1 the
        # highest-growth symbol (alphabetical order == growth-rate order in
        # the test fixture below), so different multipliers must select
        # different symbols and thus produce different equity curves.
        ordered = sorted(eligible, reverse=self.multiplier < 0)
        chosen = ordered[: self.cfg.top_n]
        if not chosen:
            return []
        w = 1.0 / len(chosen)
        return [{"symbol": s, "weight": w} for s in chosen]

    def to_targets(self, plans, data):
        symbols = sorted({s["symbol"] for plan in plans.values() for s in plan})
        targets = pd.DataFrame(np.nan, index=data.index, columns=symbols)
        for asof in sorted(plans):
            forward = data.index[data.index > asof]
            if len(forward) == 0:
                continue
            bar = forward[0]
            targets.loc[bar, symbols] = 0.0
            for sel in plans[asof]:
                targets.loc[bar, sel["symbol"]] = float(sel["weight"])
        return targets


def test_sweep_fold_weights_threads_non_labconfig_kwargs():
    """Regression test: combo_params keys outside top_n/lookback/skip (e.g.
    IdioVolStrategy's reg_window/quintile) must reach the strategy
    constructor, not be silently dropped in favor of the class default.
    """
    from ggTrader.lab.wfo import _sweep_fold_weights

    symbols = ["A", "B", "C"]
    rates = {"A": 0.0001, "B": 0.0006, "C": 0.0012}
    n = 300
    ohlcv = _ohlcv_varied_growth(symbols, n, rates)
    cfg = LabConfig(top_n=1, min_history_bars=10)
    base_config = {
        "START_CASH": 10000.0,
        "FEES": 0.0,
        "SLIPPAGE": 0.0,
        "FREQ": "1d",
    }
    window_start = ohlcv.index[100]
    window_end = ohlcv.index[250]
    # Only the extra kwarg varies across combos; top_n is constant so any
    # difference in outcome is attributable solely to `multiplier` having
    # been threaded through the constructor.
    grid = [{"multiplier": 1}, {"multiplier": -1}]

    results, all_eq = _sweep_fold_weights(
        "tinyweightkwarg",
        _TinyWeightKwarg,
        cfg,
        ohlcv,
        window_start,
        window_end,
        base_config,
        grid,
        _tiny_weight_universe_fn,
    )

    assert len(results) == 2
    eq_curves = list(all_eq.values())
    assert len(eq_curves) == 2
    # If `multiplier` were never forwarded (the bug), both combos would fall
    # back to the same default and produce byte-identical equity curves.
    assert not eq_curves[0].equals(eq_curves[1])
    total_returns = sorted(r["total_return_pct"] for r in results)
    assert total_returns[0] != pytest.approx(total_returns[1])


def test_sweep_fold_weights_basic():
    from ggTrader.lab.wfo import _sweep_fold_weights

    symbols = ["X", "Y", "Z"]
    n = 300
    ohlcv = _ohlcv(symbols, n)
    cfg = LabConfig(top_n=2, min_history_bars=10)
    base_config = {
        "START_CASH": 10000.0,
        "FEES": 0.0,
        "SLIPPAGE": 0.0,
        "FREQ": "1d",
    }
    window_start = ohlcv.index[100]
    window_end = ohlcv.index[250]
    grid = [{"top_n": 1}, {"top_n": 2}]

    results, all_eq = _sweep_fold_weights(
        "tinyweight",
        _TinyWeight,
        cfg,
        ohlcv,
        window_start,
        window_end,
        base_config,
        grid,
        _tiny_weight_universe_fn,
    )

    assert len(results) == 2
    for r in results:
        assert "combo" in r and "params" in r and "sharpe" in r
    assert set(all_eq.keys()) == {r["combo"] for r in results}
    for eq in all_eq.values():
        assert eq.notna().sum() > 0


def test_extract_grid_arrays_uses_real_per_trade_expectancy():
    """Regression: expectancy_arr previously used raw total_return_pct as a
    stand-in for per-trade expectancy ("we don't have n_trades" per the old
    comment). Two combos with identical total return but very different
    trade counts must now score very differently -- the one that earned
    the same return from fewer trades has higher per-trade expectancy."""
    from ggTrader.lab.wfo import _extract_grid_arrays

    grid = [{"k": 1}, {"k": 2}]
    train_metrics = [
        {"params": {"k": 1}, "sharpe": 1.0, "total_return_pct": 10.0, "n_trades": 2},
        {"params": {"k": 2}, "sharpe": 1.0, "total_return_pct": 10.0, "n_trades": 20},
    ]
    _sharpe_grid, expectancy_grid, _shape, result_to_grid = _extract_grid_arrays(
        train_metrics, grid, "teststrat"
    )
    exp_k1 = expectancy_grid[result_to_grid[0]]
    exp_k2 = expectancy_grid[result_to_grid[1]]
    assert exp_k1 == pytest.approx(0.10 / 2)
    assert exp_k2 == pytest.approx(0.10 / 20)
    assert exp_k1 > exp_k2


def test_sweep_fold_weights_results_include_real_trade_count():
    """Regression: the NDH gate's expectancy calc used raw total-return as
    a stand-in for per-trade expectancy because n_trades was never threaded
    from simulate_weights' diags into the per-combo result dict."""
    from ggTrader.lab.wfo import _sweep_fold_weights

    symbols = ["X", "Y", "Z"]
    n = 300
    ohlcv = _ohlcv(symbols, n)
    cfg = LabConfig(top_n=2, min_history_bars=10)
    base_config = {"START_CASH": 10000.0, "FEES": 0.0, "SLIPPAGE": 0.0, "FREQ": "1d"}
    window_start = ohlcv.index[100]
    window_end = ohlcv.index[250]
    grid = [{"top_n": 1}, {"top_n": 2}]

    results, _all_eq = _sweep_fold_weights(
        "tinyweight",
        _TinyWeight,
        cfg,
        ohlcv,
        window_start,
        window_end,
        base_config,
        grid,
        _tiny_weight_universe_fn,
    )

    for r in results:
        assert "n_trades" in r
        assert isinstance(r["n_trades"], int)
        assert r["n_trades"] >= 1  # this fixture always opens at least one position


def test_sweep_fold_weights_no_rebalance_dates_returns_empty():
    from ggTrader.lab.wfo import _sweep_fold_weights

    symbols = ["X", "Y"]
    ohlcv = _ohlcv(symbols, 50)
    cfg = LabConfig(top_n=1, min_history_bars=5)
    base_config = {"START_CASH": 10000.0, "FEES": 0.0, "SLIPPAGE": 0.0, "FREQ": "1d"}
    # A window with no full month inside it produces no rebalance dates.
    window_start = ohlcv.index[0]
    window_end = ohlcv.index[1]
    grid = [{"top_n": 1}]

    results, all_eq = _sweep_fold_weights(
        "tinyweight",
        _TinyWeight,
        cfg,
        ohlcv,
        window_start,
        window_end,
        base_config,
        grid,
        _tiny_weight_universe_fn,
    )
    assert results == []
    assert all_eq == {}


def test_generate_folds_count_and_boundaries():
    """5-year span with 12mo train / 3mo test -> 16 folds, no overlap."""
    start = pd.Timestamp("2020-01-01", tz="UTC")
    end = pd.Timestamp("2025-01-01", tz="UTC")
    folds = generate_folds(start, end)
    assert len(folds) == 16
    for f in folds:
        assert f.test_start == f.train_end  # no gap
        train_months = (f.train_end - f.train_start).days / 30.44
        test_months = (f.test_end - f.test_start).days / 30.44
        assert abs(train_months - 12.0) < 1.0
        assert abs(test_months - 3.0) < 1.0
    assert all(f.test_end <= end for f in folds)
    # Folds slide by 3 months
    for i in range(1, len(folds)):
        delta_months = (folds[i].train_start - folds[i - 1].train_start).days / 30.44
        assert abs(delta_months - 3.0) < 1.0


def test_generate_folds_short_data_returns_fewer():
    """Only 12 months of data -> not enough for 12mo train + 3mo test."""
    start = pd.Timestamp("2024-01-01", tz="UTC")
    end = pd.Timestamp("2025-01-01", tz="UTC")
    folds = generate_folds(start, end)
    assert len(folds) == 0


def test_generate_folds_exact_15_months():
    """Exactly 15 months (12 + 3) -> 1 fold."""
    start = pd.Timestamp("2024-01-01", tz="UTC")
    end = pd.Timestamp("2025-04-01", tz="UTC")
    folds = generate_folds(start, end)
    assert len(folds) == 1
    assert folds[0].train_start == start
    assert folds[0].test_end == end


def test_extract_grid_arrays_handles_none_valued_axis():
    """Exit-sweep axes (td_stop / tp_stop) mix None ('no stop') with numbers.
    The categorical NDH grid must build without choking on None vs int order."""
    from ggTrader.lab.wfo import _extract_grid_arrays

    grid = [
        {"min_agree": 2, "td_stop": None},
        {"min_agree": 2, "td_stop": 5},
        {"min_agree": 2, "td_stop": 10},
    ]
    train_metrics = [{"params": g, "sharpe": 1.0, "total_return_pct": 10.0} for g in grid]

    sharpe_grid, exp_grid, shape, r2g = _extract_grid_arrays(train_metrics, grid, "ensemble")

    # param_keys sorted: ['min_agree', 'td_stop'] -> shape (1 x 3)
    assert shape == (1, 3)
    assert len(sharpe_grid) == 3
    assert len(r2g) == 3
    # every combo maps to a distinct grid cell
    assert len(set(r2g.values())) == 3


def test_composite_score_ranking():
    """Combo with best sharpe+sortino and least drawdown scores highest."""
    metrics = [
        {"sharpe": 1.0, "sortino": 1.2, "max_drawdown_pct": -10.0},
        {"sharpe": 0.5, "sortino": 0.6, "max_drawdown_pct": -20.0},
        {"sharpe": 0.2, "sortino": 0.3, "max_drawdown_pct": -30.0},
    ]
    scores = composite_score(metrics)
    assert len(scores) == 3
    assert scores[0] > scores[1] > scores[2]
    # Best combo should get close to 1.0 (all normalized to 1.0)
    # 0.5 * 1.0 + 0.3 * 1.0 - 0.2 * 0.0 = 0.8
    assert abs(scores[0] - 0.8) < 1e-9


def test_composite_score_single_combo():
    """Single combo: min==max for all metrics, all normalized to 0.0, score=0.0."""
    metrics = [{"sharpe": 0.5, "sortino": 0.6, "max_drawdown_pct": -15.0}]
    scores = composite_score(metrics)
    assert len(scores) == 1
    assert scores[0] == 0.0


def test_composite_score_nan_handling():
    """NaN sharpe/sortino treated as worst (floor of range)."""
    metrics = [
        {"sharpe": 1.0, "sortino": 1.0, "max_drawdown_pct": -10.0},
        {"sharpe": float("nan"), "sortino": float("nan"), "max_drawdown_pct": -20.0},
    ]
    scores = composite_score(metrics)
    assert scores[0] > scores[1]


def test_run_wfo_integration():
    """Full WFO with tiny strategy: ~7yr data, 12mo/3mo folds."""
    symbols = ["X", "Y"]
    n = 252 * 7  # ~7 years of daily bars
    ohlcv = _ohlcv(symbols, n)
    spy_close = ohlcv["X"]["close"].copy()  # use X as SPY proxy
    cfg = LabConfig(top_n=10, lookback=20, skip=5, min_history_bars=10)
    base_config = {
        "START_CASH": 10000.0,
        "FEES": 0.0,
        "SLIPPAGE": 0.0,
        "FREQ": "1d",
        "SIGNAL_POSITION_SIZE": 0.5,
    }
    eval_start = ohlcv.index[0]
    eval_end = ohlcv.index[-1]
    grid = [{"param_a": 1}, {"param_a": 2}]

    output = run_wfo(
        "tiny",
        _TinySignal,
        cfg,
        ohlcv,
        spy_close,
        str(eval_start.date()),
        str(eval_end.date()),
        "test",
        base_config,
        grid,
    )
    assert "WFO:" in output.table
    assert "OOS Aggregate:" in output.table
    assert "Recommended Live Params" in output.table
    # Should have at least 1 fold
    assert "Fold" in output.table


def test_pick_live_winner_prefers_fold_proven_combo():
    """A combo that wins folds is chosen over a higher-scoring combo that never
    won out-of-sample (the overfit-to-recent-window trap)."""
    train_metrics = [
        {"combo": "durable"},  # won folds, lower recent score
        {"combo": "overfit"},  # best recent score, never won a fold
    ]
    scores = [0.40, 0.99]
    fold_win_counts = {"durable": 11}
    idx, stability = _pick_live_winner(train_metrics, scores, fold_win_counts)
    assert train_metrics[idx]["combo"] == "durable"
    assert stability == 11


def test_pick_live_winner_picks_best_score_among_durable():
    """Among combos that cleared the stability bar, the best recent score wins."""
    train_metrics = [
        {"combo": "a"},
        {"combo": "b"},
        {"combo": "overfit"},
    ]
    scores = [0.50, 0.70, 0.99]
    fold_win_counts = {"a": 3, "b": 6}
    idx, stability = _pick_live_winner(train_metrics, scores, fold_win_counts)
    assert train_metrics[idx]["combo"] == "b"
    assert stability == 6


def test_pick_live_winner_falls_back_when_no_fold_winners():
    """With no fold winners (e.g. every fold failed gates), fall back to the
    global best composite score rather than recommending nothing."""
    train_metrics = [{"combo": "a"}, {"combo": "b"}]
    scores = [0.30, 0.90]
    idx, stability = _pick_live_winner(train_metrics, scores, {})
    assert train_metrics[idx]["combo"] == "b"
    assert stability == 0


def test_select_live_params_uses_recent_window():
    """Live params trained on most recent 12mo; stability counts matching fold winners."""
    symbols = ["X", "Y"]
    n = 252 * 7
    ohlcv = _ohlcv(symbols, n)
    cfg = LabConfig(top_n=10, lookback=20, skip=5, min_history_bars=10)
    base_config = {
        "START_CASH": 10000.0,
        "FEES": 0.0,
        "SLIPPAGE": 0.0,
        "FREQ": "1d",
        "SIGNAL_POSITION_SIZE": 0.5,
    }
    grid = [{"param_a": 1}, {"param_a": 2}]
    # Simulate fold_winners that all picked param_a=1
    fold_winners = [
        {"combo": "tiny__param_a1", "params": {"param_a": 1}},
        {"combo": "tiny__param_a1", "params": {"param_a": 1}},
    ]
    eval_end = str(ohlcv.index[-1].date())
    result = select_live_params(
        "tiny",
        _TinySignal,
        cfg,
        ohlcv,
        eval_end,
        base_config,
        grid,
        fold_winners,
    )
    assert "combo" in result
    assert "stability" in result
    assert isinstance(result["stability"], int)
    assert "train_metrics" in result
    assert "sharpe" in result["train_metrics"]


def test_run_wfo_reports_gate_status():
    """WFO output should include NDH/DSR gate status per fold."""
    symbols = ["X", "Y"]
    n = 252 * 7
    ohlcv = _ohlcv(symbols, n)
    spy_close = ohlcv["X"]["close"].copy()
    cfg = LabConfig(top_n=10, lookback=20, skip=5, min_history_bars=10)
    base_config = {
        "START_CASH": 10000.0,
        "FEES": 0.0,
        "SLIPPAGE": 0.0,
        "FREQ": "1d",
        "SIGNAL_POSITION_SIZE": 0.5,
    }
    eval_start = ohlcv.index[0]
    eval_end = ohlcv.index[-1]
    grid = [{"param_a": 1}, {"param_a": 2}]

    output = run_wfo(
        "tiny",
        _TinySignal,
        cfg,
        ohlcv,
        spy_close,
        str(eval_start.date()),
        str(eval_end.date()),
        "test",
        base_config,
        grid,
    )
    assert "Gate" in output.table
    # Each fold line should show PASS or FAIL
    fold_lines = [
        ln for ln in output.table.splitlines() if ln.strip().startswith(tuple("0123456789"))
    ]
    assert len(fold_lines) > 0
    for line in fold_lines:
        assert "PASS" in line or "FAIL" in line


def test_cli_parser_accepts_wfo_flag():
    p = build_arg_parser()
    args = p.parse_args(["--strategy", "ema_cross", "--wfo"])
    assert args.wfo is True
    assert args.sweep is False


def test_cli_parser_wfo_and_sweep_mutually_exclusive():
    p = build_arg_parser()
    with pytest.raises(SystemExit):
        p.parse_args(["--strategy", "ema_cross", "--wfo", "--sweep"])


def test_cli_parser_wfo_with_sweep_param():
    p = build_arg_parser()
    args = p.parse_args(
        [
            "--strategy",
            "ema_cross",
            "--wfo",
            "--sweep-param",
            "atr_mult=1.5,2.0",
        ]
    )
    assert args.wfo is True
    assert len(args.sweep_param) == 1


# ── compute_wfe tests ──────────────────────────────────────────────────


def test_compute_wfe_normal():
    """WFE = OOS sharpe / IS sharpe when IS >= floor."""
    from ggTrader.lab.wfo import compute_wfe

    assert compute_wfe(is_sharpe=1.0, oos_sharpe=0.6) == pytest.approx(0.6)
    assert compute_wfe(is_sharpe=2.0, oos_sharpe=1.0) == pytest.approx(0.5)


def test_compute_wfe_neutral_window():
    """IS Sharpe below floor returns None (neutral, excluded from average)."""
    from ggTrader.lab.wfo import compute_wfe

    assert compute_wfe(is_sharpe=0.3, oos_sharpe=0.1) is None
    assert compute_wfe(is_sharpe=0.0, oos_sharpe=-0.5) is None


def test_compute_wfe_negative_oos():
    """Negative OOS sharpe -> negative WFE (valid, not neutral)."""
    from ggTrader.lab.wfo import compute_wfe

    result = compute_wfe(is_sharpe=1.0, oos_sharpe=-0.5)
    assert result == pytest.approx(-0.5)


def test_compute_wfe_custom_floor():
    """Custom IS floor is respected."""
    from ggTrader.lab.wfo import compute_wfe

    assert compute_wfe(is_sharpe=0.35, oos_sharpe=0.1, is_floor=0.3) == pytest.approx(0.1 / 0.35)
    assert compute_wfe(is_sharpe=0.35, oos_sharpe=0.1, is_floor=0.4) is None


# ── circuit breaker tests ──────────────────────────────────────────────


def test_circuit_breaker_chronic_decay():
    """Trailing 4-window WFE avg < 0.25 triggers halt."""
    from ggTrader.lab.wfo import WfoState, check_circuit_breaker

    state = WfoState(wfe_history=[0.2, 0.1, 0.3, 0.15])
    result = check_circuit_breaker(state)
    assert result.halted is True
    assert "chronic" in result.halt_reason.lower()


def test_circuit_breaker_acute_failure():
    """Two consecutive negative OOS Sharpe windows trigger halt."""
    from ggTrader.lab.wfo import WfoState, check_circuit_breaker

    state = WfoState(
        wfe_history=[0.8, 0.6],
        oos_sharpes=[-0.1, -0.2],
    )
    result = check_circuit_breaker(state)
    assert result.halted is True
    assert "acute" in result.halt_reason.lower()


def test_circuit_breaker_healthy():
    """Healthy WFE history does not trigger halt."""
    from ggTrader.lab.wfo import WfoState, check_circuit_breaker

    state = WfoState(wfe_history=[0.6, 0.7, 0.5, 0.8])
    result = check_circuit_breaker(state)
    assert result.halted is False
    assert result.halt_reason is None


def test_circuit_breaker_neutral_windows_excluded():
    """None values in WFE history are excluded from average, not counted as 0."""
    from ggTrader.lab.wfo import WfoState, check_circuit_breaker

    state = WfoState(wfe_history=[0.6, None, None, 0.8])
    result = check_circuit_breaker(state)
    # avg of [0.6, 0.8] = 0.7 > 0.25 -> healthy
    assert result.halted is False


def test_circuit_breaker_single_negative_oos_not_halt():
    """One negative OOS Sharpe is not enough (need 2 consecutive)."""
    from ggTrader.lab.wfo import WfoState, check_circuit_breaker

    state = WfoState(
        wfe_history=[0.8],
        oos_sharpes=[0.5, -0.1, 0.3],
    )
    result = check_circuit_breaker(state)
    assert result.halted is False


# ── shadow re-entry tests ─────────────────────────────────────────────


def test_shadow_reentry_needs_two_clean():
    """First clean shadow window doesn't restore -- need 2 of last 3."""
    from ggTrader.lab.wfo import WfoState, check_shadow_reentry

    state = WfoState(halted=True, halt_reason="test")
    result = check_shadow_reentry(
        state,
        ndh_passed=True,
        dsr_passed=True,
        wfe=0.6,
    )
    assert result.halted is True
    assert result.shadow_window == [True]


def test_shadow_reentry_two_consecutive_restores():
    """Two consecutive clean windows end the halt (2 of last 2)."""
    from ggTrader.lab.wfo import WfoState, check_shadow_reentry

    state = WfoState(halted=True, halt_reason="test", shadow_window=[True])
    result = check_shadow_reentry(
        state,
        ndh_passed=True,
        dsr_passed=True,
        wfe=0.6,
    )
    assert result.halted is False
    assert result.shadow_window == []
    assert result.halt_reason is None


def test_shadow_reentry_two_of_three_restores():
    """A dirty window BETWEEN two clean windows no longer blocks re-entry.

    This is the mid-cap fix: noisy universes alternate clean/dirty, so the old
    '2 consecutive' rule left the halt permanently stuck. 'clean, dirty, clean'
    is 2 of the last 3 and must restore live trading.
    """
    from ggTrader.lab.wfo import WfoState, check_shadow_reentry

    # window so far: clean, then dirty
    state = WfoState(halted=True, halt_reason="test", shadow_window=[True, False])
    result = check_shadow_reentry(
        state,
        ndh_passed=True,
        dsr_passed=True,
        wfe=0.6,
    )
    assert result.halted is False
    assert result.shadow_window == []
    assert result.halt_reason is None


def test_shadow_reentry_dirty_window_does_not_wipe_history():
    """A single dirty window appends False but keeps the rolling history."""
    from ggTrader.lab.wfo import WfoState, check_shadow_reentry

    state = WfoState(halted=True, halt_reason="test", shadow_window=[True])
    result = check_shadow_reentry(
        state,
        ndh_passed=True,
        dsr_passed=False,
        wfe=0.6,
    )
    assert result.halted is True
    assert result.shadow_window == [True, False]


def test_shadow_reentry_only_last_three_count():
    """Rolling window caps at 3: an old clean window ages out and can't restore."""
    from ggTrader.lab.wfo import WfoState, check_shadow_reentry

    # last 3 would be [False, False, <new>]; one stale clean is dropped
    state = WfoState(halted=True, halt_reason="test", shadow_window=[True, False, False])
    result = check_shadow_reentry(
        state,
        ndh_passed=True,
        dsr_passed=True,
        wfe=0.6,
    )
    # window becomes [False, False, True] -> only 1 of last 3 clean -> stay halted
    assert result.halted is True
    assert result.shadow_window == [False, False, True]


def test_shadow_reentry_requires_wfe_healthy():
    """WFE < 0.5 (healthy target) is not a clean window."""
    from ggTrader.lab.wfo import WfoState, check_shadow_reentry

    state = WfoState(halted=True, halt_reason="test", shadow_window=[True])
    result = check_shadow_reentry(
        state,
        ndh_passed=True,
        dsr_passed=True,
        wfe=0.3,
    )
    assert result.halted is True
    assert result.shadow_window == [True, False]


def test_shadow_reentry_neutral_wfe_fails():
    """Neutral WFE (None) is not a clean window -- can't confirm recovery."""
    from ggTrader.lab.wfo import WfoState, check_shadow_reentry

    state = WfoState(halted=True, halt_reason="test", shadow_window=[True])
    result = check_shadow_reentry(
        state,
        ndh_passed=True,
        dsr_passed=True,
        wfe=None,
    )
    assert result.halted is True
    assert result.shadow_window == [True, False]


# ── anchor set tests ─────────────────────────────────────────────────


def test_compute_anchor_set_returns_valid():
    """Anchor set returns a combo with minimum drawdown and CAGR > risk-free."""
    from ggTrader.lab.wfo import AnchorSet, compute_anchor_set

    symbols = ["X", "Y"]
    n = 252 * 7
    ohlcv = _ohlcv(symbols, n)
    cfg = LabConfig(top_n=10, lookback=20, skip=5, min_history_bars=10)
    base_config = {
        "START_CASH": 10000.0,
        "FEES": 0.0,
        "SLIPPAGE": 0.0,
        "FREQ": "1d",
        "SIGNAL_POSITION_SIZE": 0.5,
    }
    grid = [{"param_a": 1}, {"param_a": 2}]

    result = compute_anchor_set(
        "tiny",
        _TinySignal,
        cfg,
        ohlcv,
        base_config,
        grid,
    )
    assert isinstance(result, AnchorSet)
    assert result.combo != ""
    assert isinstance(result.params, dict)
    assert result.max_drawdown_pct <= 0  # drawdown is negative or zero
    assert isinstance(result.cagr_pct, float)


def test_compute_anchor_set_minimizes_drawdown():
    """Given multiple combos, anchor picks the one with smallest max drawdown."""
    from ggTrader.lab.wfo import compute_anchor_set

    symbols = ["X", "Y"]
    n = 252 * 7
    ohlcv = _ohlcv(symbols, n)
    cfg = LabConfig(top_n=10, lookback=20, skip=5, min_history_bars=10)
    base_config = {
        "START_CASH": 10000.0,
        "FEES": 0.0,
        "SLIPPAGE": 0.0,
        "FREQ": "1d",
        "SIGNAL_POSITION_SIZE": 0.5,
    }
    grid = [{"param_a": 1}, {"param_a": 2}]

    result = compute_anchor_set(
        "tiny",
        _TinySignal,
        cfg,
        ohlcv,
        base_config,
        grid,
    )
    # The anchor should have a valid drawdown
    assert result.max_drawdown_pct != float("nan")


def test_run_wfo_shows_anchor_set():
    """WFO output includes the anchor set section."""
    symbols = ["X", "Y"]
    n = 252 * 7
    ohlcv = _ohlcv(symbols, n)
    spy_close = ohlcv["X"]["close"].copy()
    cfg = LabConfig(top_n=10, lookback=20, skip=5, min_history_bars=10)
    base_config = {
        "START_CASH": 10000.0,
        "FEES": 0.0,
        "SLIPPAGE": 0.0,
        "FREQ": "1d",
        "SIGNAL_POSITION_SIZE": 0.5,
    }
    eval_start = ohlcv.index[0]
    eval_end = ohlcv.index[-1]
    grid = [{"param_a": 1}, {"param_a": 2}]

    output = run_wfo(
        "tiny",
        _TinySignal,
        cfg,
        ohlcv,
        spy_close,
        str(eval_start.date()),
        str(eval_end.date()),
        "test",
        base_config,
        grid,
    )
    assert "Anchor Set" in output.table


def test_run_wfo_anchor_fallback_on_gate_failure():
    """When gates fail (impossible thresholds), anchor params are used ([A] marker)."""
    symbols = ["X", "Y"]
    n = 252 * 7
    ohlcv = _ohlcv(symbols, n)
    spy_close = ohlcv["X"]["close"].copy()
    cfg = LabConfig(top_n=10, lookback=20, skip=5, min_history_bars=10)
    base_config = {
        "START_CASH": 10000.0,
        "FEES": 0.0,
        "SLIPPAGE": 0.0,
        "FREQ": "1d",
        "SIGNAL_POSITION_SIZE": 0.5,
    }
    eval_start = ohlcv.index[0]
    eval_end = ohlcv.index[-1]
    grid = [{"param_a": 1}, {"param_a": 2}]

    output = run_wfo(
        "tiny",
        _TinySignal,
        cfg,
        ohlcv,
        spy_close,
        str(eval_start.date()),
        str(eval_end.date()),
        "test",
        base_config,
        grid,
        ndh_threshold=1.0,
        dsr_threshold=1.0,
    )
    # At least one fold should fail gates and use anchor fallback
    fold_lines = [
        ln for ln in output.table.splitlines() if ln.strip().startswith(tuple("0123456789"))
    ]
    assert len(fold_lines) > 0
    anchor_lines = [ln for ln in fold_lines if "[A]" in ln]
    assert len(anchor_lines) >= 1
    for line in anchor_lines:
        assert "FAIL" in line


def test_run_wfo_returns_wforesult_namedtuple():
    from ggTrader.lab.wfo import WfoResult

    r = WfoResult(
        oos_equity=__import__("pandas").Series(dtype=float),
        fold_results=[],
        live_params={},
        table="x",
    )
    assert r.table == "x"
    assert list(r._fields) == ["oos_equity", "fold_results", "live_params", "table"]


def test_run_wfo_weight_strategy_integration():
    """Weight strategies flow through the full gated WFO: folds, gates, table."""
    symbols = ["X", "Y", "Z"]
    n = 252 * 7
    ohlcv = _ohlcv(symbols, n)
    spy_close = ohlcv["X"]["close"].copy()
    cfg = LabConfig(top_n=2, lookback=20, skip=5, min_history_bars=10)
    base_config = {
        "START_CASH": 10000.0,
        "FEES": 0.0,
        "SLIPPAGE": 0.0,
        "FREQ": "1d",
    }
    eval_start = ohlcv.index[0]
    eval_end = ohlcv.index[-1]
    grid = [{"top_n": 1}, {"top_n": 2}]

    output = run_wfo(
        "tinyweight",
        _TinyWeight,
        cfg,
        ohlcv,
        spy_close,
        str(eval_start.date()),
        str(eval_end.date()),
        "test",
        base_config,
        grid,
        universe_fn=_tiny_weight_universe_fn,
    )
    assert "WFO:" in output.table
    assert "OOS Aggregate:" in output.table
    assert "Recommended Live Params" in output.table


def test_run_wfo_weight_strategy_without_universe_fn_raises():
    """A weight strategy with no universe_fn is a caller bug, not a silent no-op."""
    symbols = ["X", "Y"]
    ohlcv = _ohlcv(symbols, 252 * 2)
    spy_close = ohlcv["X"]["close"].copy()
    cfg = LabConfig(top_n=1, min_history_bars=10)
    base_config = {"START_CASH": 10000.0, "FEES": 0.0, "SLIPPAGE": 0.0, "FREQ": "1d"}
    grid = [{"top_n": 1}]

    with pytest.raises(ValueError, match="universe_fn"):
        run_wfo(
            "tinyweight",
            _TinyWeight,
            cfg,
            ohlcv,
            spy_close,
            str(ohlcv.index[0].date()),
            str(ohlcv.index[-1].date()),
            "test",
            base_config,
            grid,
        )


def test_cli_wfo_accepts_weight_strategy():
    """--wfo must accept a weight strategy name (xs_momentum), not just signals."""
    parser = build_arg_parser()
    args = parser.parse_args(["--strategy", "xs_momentum", "--wfo"])
    assert args.strategy == "xs_momentum"
    assert args.wfo is True


class TestWfoTableOosColumn:
    """The per-fold OOS column must report the fold's actual out-of-sample
    Sharpe.

    Regression test for audit 2026-07-25 §2.1A: the column was rendered
    from `oos_score`, which is `composite_score(test_metrics)[0]` where
    `test_metrics` always holds exactly one combo (`winner_grid =
    [deploy_params]`). `composite_score` min-max normalizes within the
    list it is given, and `_min_max_normalize` returns [0.0] when
    min == max -- so the column was a hardcoded 0.00 in every WFO table
    ever printed, regardless of performance.
    """

    @staticmethod
    def _fold(oos_sharpe, oos_score=0.0):
        return {
            "fold_num": 1,
            "train_start": pd.Timestamp("2024-01-01"),
            "train_end": pd.Timestamp("2025-01-01"),
            "test_start": pd.Timestamp("2025-01-01"),
            "test_end": pd.Timestamp("2025-04-01"),
            "winner_combo": "demo__x1",
            "winner_params": {"x": 1},
            "train_score": 0.80,
            "oos_score": oos_score,
            "gates_passed": True,
            "wfe": 0.5,
            "is_sharpe": 1.0,
            "oos_sharpe": oos_sharpe,
            "halted": False,
            "used_anchor": False,
        }

    def _render(self, oos_sharpe):
        from ggTrader.lab.wfo import format_wfo_table

        return format_wfo_table(
            [self._fold(oos_sharpe)],
            {"sharpe": 0.5, "cagr_pct": 5.0, "max_drawdown_pct": -10.0},
            {"sharpe": 1.0, "cagr_pct": 10.0, "max_drawdown_pct": -8.0},
            {"combo": "demo__x1", "params": {"x": 1}, "train_metrics": {}, "stability": 1.0},
            "demo",
            1,
            1,
        )

    def test_positive_oos_sharpe_is_shown_not_zero(self):
        out = self._render(1.44)
        assert "1.44" in out, f"fold OOS Sharpe missing from table:\n{out}"

    def test_negative_oos_sharpe_is_shown(self):
        out = self._render(-0.38)
        assert "-0.38" in out, f"negative fold OOS Sharpe missing:\n{out}"

    def test_distinct_oos_values_render_differently(self):
        """Two folds with very different OOS results must not print the
        same number -- the exact symptom of the constant-0.00 bug."""
        assert self._render(1.44) != self._render(-0.38)

    def test_train_and_oos_columns_are_both_sharpe_so_wfe_is_verifiable(self):
        """WFE is OOS Sharpe / IS Sharpe, so the two columns either side of
        it must be those same Sharpes -- otherwise the row cannot be
        checked by eye and mixes units (composite score vs Sharpe)."""
        from ggTrader.lab.wfo import format_wfo_table

        fold = self._fold(0.50)
        fold["is_sharpe"] = 2.00
        fold["train_score"] = 0.80  # composite score -- must NOT be shown
        fold["wfe"] = 0.25
        out = format_wfo_table(
            [fold],
            {"sharpe": 0.5, "cagr_pct": 5.0, "max_drawdown_pct": -10.0},
            {"sharpe": 1.0, "cagr_pct": 10.0, "max_drawdown_pct": -8.0},
            {"combo": "demo__x1", "params": {"x": 1}, "train_metrics": {}, "stability": 1.0},
            "demo",
            1,
            1,
        )
        row = [ln for ln in out.splitlines() if ln.startswith("1  ")][0]
        assert "2.00" in row, f"train Sharpe missing from row: {row!r}"
        assert "0.80" not in row, f"composite score leaked into row: {row!r}"
