import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.integration


def _engine():
    from ggTrader.lab.persist import get_engine

    return get_engine()


def test_schema_init_idempotent():
    from ggTrader.lab.persist import init_schema

    init_schema()
    init_schema()  # second call must not raise


def test_plan_roundtrip_and_resume():
    from ggTrader.lab.persist import init_schema, plan_done, read_plan, start_run, write_plan

    init_schema()
    run_id = start_run("xs_momentum", "equity", "monthly", "2021-01-31", "2021-03-31", {"top_n": 2})
    asof = pd.Timestamp("2021-01-31", tz="UTC")
    assert plan_done(run_id, "xs_momentum", asof) is False
    plan = [{"symbol": "AAA", "weight": 0.5, "momentum": 0.1}]
    write_plan(run_id, "xs_momentum", asof, plan, eligible_count=10, coverage={"n": 10})
    assert plan_done(run_id, "xs_momentum", asof) is True
    assert read_plan(run_id, "xs_momentum", asof) == plan


def test_returns_equity_summary_write():
    from ggTrader.lab.persist import init_schema, start_run, write_returns_equity, write_summary

    init_schema()
    run_id = start_run("xs_momentum", "equity", "monthly", "2021-01-31", "2021-03-31", {})
    idx = pd.date_range("2021-02-01", periods=5, freq="B", tz="UTC")
    rets = pd.Series([0.0, 0.01, -0.005, 0.002, 0.0], index=idx)
    eq = pd.Series(10000.0 * (1 + rets).cumprod(), index=idx)
    bench = pd.Series(np.linspace(10000, 10100, 5), index=idx)
    write_returns_equity(run_id, "xs_momentum", rets, eq, bench)
    write_summary(run_id, "xs_momentum", {"sharpe": 1.0}, {"sharpe": 0.9}, {"turnover": 0.3})
