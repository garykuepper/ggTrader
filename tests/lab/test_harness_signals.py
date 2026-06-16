# tests/lab/test_harness_signals.py
import numpy as np
import pandas as pd
import pytest


def _ohlcv(symbols, n=600):
    idx = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
    frames = {}
    for i, s in enumerate(symbols):
        close = pd.Series(100.0 * (1 + 0.0002 * (i + 1)) ** np.arange(n), index=idx)
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


@pytest.mark.integration
def test_walkforward_signal_strategy_persists_and_resumes():
    from ggTrader.lab.harness import walkforward
    from ggTrader.lab.persist import read_all_plans
    from ggTrader.lab.strategies.signals import EmaCrossSignal
    from ggTrader.lab.strategy import LabConfig

    ohlcv = _ohlcv(["A", "B", "C"], n=700)  # 700 bars reaches past 2022-06-30
    spy = ohlcv["A"]["close"]
    strat = EmaCrossSignal(LabConfig(min_history_bars=50))

    run_id = walkforward(
        [strat],
        ohlcv,
        spy,
        eval_start="2022-01-31",
        eval_end="2022-06-30",
        market="test",
        freq="monthly",
        universe_fn=lambda asof, past: ["A", "B", "C"],
        base_config={
            "START_CASH": 10000.0,
            "FEES": 0.0,
            "SLIPPAGE": 0.0,
            "FREQ": "1d",
            "SIGNAL_POSITION_SIZE": 0.3,
        },
    )
    plans = read_all_plans(run_id, "ema_cross")
    assert len(plans) >= 4

    # Resume: second call with same run_id must not fail
    run_id2 = walkforward(
        [strat],
        ohlcv,
        spy,
        eval_start="2022-01-31",
        eval_end="2022-06-30",
        market="test",
        freq="monthly",
        run_id=run_id,
        universe_fn=lambda asof, past: ["A", "B", "C"],
        base_config={
            "START_CASH": 10000.0,
            "FEES": 0.0,
            "SLIPPAGE": 0.0,
            "FREQ": "1d",
            "SIGNAL_POSITION_SIZE": 0.3,
        },
    )
    assert run_id2 == run_id


@pytest.mark.integration
def test_walkforward_mixed_weight_and_signal_strategies():
    from ggTrader.lab.harness import walkforward
    from ggTrader.lab.strategies.momentum import CrossSectionalMomentum
    from ggTrader.lab.strategies.signals import EmaCrossSignal
    from ggTrader.lab.strategy import LabConfig

    ohlcv = _ohlcv(["A", "B", "C", "D"], n=600)
    spy = ohlcv["A"]["close"]
    weight_strat = CrossSectionalMomentum(LabConfig(top_n=2, min_history_bars=50))
    signal_strat = EmaCrossSignal(LabConfig(min_history_bars=50))

    run_id = walkforward(
        [weight_strat, signal_strat],
        ohlcv,
        spy,
        eval_start="2022-01-31",
        eval_end="2022-06-30",
        market="test",
        freq="monthly",
        universe_fn=lambda asof, past: ["A", "B", "C", "D"],
        base_config={
            "START_CASH": 10000.0,
            "FEES": 0.0,
            "SLIPPAGE": 0.0,
            "FREQ": "1d",
            "SIGNAL_POSITION_SIZE": 0.25,
        },
    )
    assert run_id is not None
