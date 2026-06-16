import numpy as np
import pandas as pd
import pytest


def _ohlcv(symbols, n=400):
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


def test_leak_check_passes_for_momentum():
    from ggTrader.lab.harness import leak_check
    from ggTrader.lab.strategies.momentum import CrossSectionalMomentum
    from ggTrader.lab.strategy import LabConfig

    ohlcv = _ohlcv(["A", "B", "C", "D"])
    asof = ohlcv.index[-30]
    assert (
        leak_check(CrossSectionalMomentum(LabConfig(top_n=2)), ohlcv, asof, ["A", "B", "C", "D"])
        is True
    )


@pytest.mark.integration
def test_walkforward_persists_and_resumes():
    from ggTrader.lab.harness import walkforward
    from ggTrader.lab.persist import read_all_plans
    from ggTrader.lab.strategies.momentum import CrossSectionalMomentum
    from ggTrader.lab.strategy import LabConfig

    ohlcv = _ohlcv(["A", "B", "C"], n=400)
    spy = ohlcv["A"]["close"]
    strat = CrossSectionalMomentum(LabConfig(top_n=2, min_history_bars=50))
    run_id = walkforward(
        [strat],
        ohlcv,
        spy,
        eval_start="2021-01-31",
        eval_end="2021-06-30",
        market="test",
        freq="monthly",
        universe_fn=lambda asof, past: ["A", "B", "C"],
        base_config={"START_CASH": 10000.0, "FEES": 0.0, "SLIPPAGE": 0.0, "FREQ": "1d"},
    )
    plans = read_all_plans(run_id, "xs_momentum")
    assert len(plans) >= 4  # one per rebalance month

    # The stored equity must NOT include the warmup/cash prefix — it starts at the
    # first traded bar (after the first rebalance), never at the data window start.
    from sqlalchemy import text

    from ggTrader.lab.persist import get_engine

    with get_engine().connect() as conn:
        emin = conn.execute(
            text("SELECT min(date) FROM lab_equity WHERE run_id=:r"), {"r": run_id}
        ).scalar()
    assert pd.Timestamp(emin) >= pd.Timestamp("2021-02-01", tz="UTC")

    run_id2 = walkforward(
        [strat],
        ohlcv,
        spy,
        eval_start="2021-01-31",
        eval_end="2021-06-30",
        market="test",
        freq="monthly",
        run_id=run_id,
        universe_fn=lambda asof, past: ["A", "B", "C"],
        base_config={"START_CASH": 10000.0, "FEES": 0.0, "SLIPPAGE": 0.0, "FREQ": "1d"},
    )
    assert run_id2 == run_id
