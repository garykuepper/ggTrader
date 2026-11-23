"""Published synthetic release through the close-only RSI strategy and WFO."""

from datetime import date

import pandas as pd
import pytest

from ggTrader.data.bars import load_bars
from ggTrader.data.kaggle_import.release import build_release
from ggTrader.lab.dead_stocks import forced_exit_frames, simulation_close
from ggTrader.lab.strategies.contract import ParamGrid, simulate_signal_out
from ggTrader.lab.strategies.registry import get
from ggTrader.lab.strategy import LabConfig
from ggTrader.lab.wfo.run import run_wfo
from tests.data.kaggle_import.test_release import _inputs
from tests.data.test_bars_combined import conn  # noqa: F401
from tests.data.test_bars_combined import published as release_fixture
from tests.lab.wfo.helpers import TOY_CONFIG

published = release_fixture
pytestmark = [pytest.mark.integration, pytest.mark.db]


def test_combined_release_smoke(published):
    db, load, _ = published
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO market.daily_bars(load_id,permno,spell,date,close,abs_prc,midpoint_flag) "
            "SELECT %s, p, 1, d::date, CASE WHEN p=91002 AND d::date >= '2019-03-01' "
            "THEN 50 WHEN p=91002 THEN 100 + extract(doy from d) ELSE 100 END, 100, false "
            "FROM generate_series('2019-01-01'::timestamp, '2022-06-30'::timestamp, '1 day') d "
            "CROSS JOIN unnest(ARRAY[91001,91002,91003]) p "
            "WHERE extract(isodow from d) < 6 AND (p<>91002 OR d::date <= '2021-06-30') "
            "AND (p<>91003 OR d::date <= '2020-01-06')",
            (load,),
        )
        cur.execute(
            "INSERT INTO public.ohlcv(timestamp,symbol,interval,venue,open,high,low,close,volume) "
            "SELECT d,'SYNTH','1d','yfinance',100,100,100,100,1000 FROM "
            "generate_series('2020-01-06'::timestamp,'2022-06-30'::timestamp,'1 day') d "
            "WHERE extract(isodow from d) < 6"
        )
    db.commit()
    links, settings = _inputs()
    links.loc[0, ["symbol", "permno"]] = ["SYNTH", 91003]
    rid = build_release(db, crsp_bars_load=load, mcgill_load=None, links=links, settings=settings)
    bars = load_bars(["PERMNO-91001", "PERMNO-91002", "SYNTH"], source="combined", release_id=rid)
    assert len(bars.symbols) == 3
    assert bars.open["PERMNO-91002"].isna().all()
    assert set(bars.flags.loc[(slice(None), "SYNTH"), "source"]) == {"crsp", "yfinance"}
    assert bars.spans.set_index("symbol").loc["PERMNO-91002", "last_date"] == date(2021, 6, 30)
    strategy = get("rsi_reversion")
    grid = ParamGrid.single({"rsi_period": 7, "rsi_oversold": 30, "rsi_exit": 50})
    spy = pd.Series(100.0, index=bars.close.index)
    for loss in (0.0, 0.5):
        result = run_wfo(
            "rsi_reversion",
            type(strategy),
            LabConfig(),
            bars,
            spy,
            "2019-01-01",
            "2022-06-30",
            "equity",
            TOY_CONFIG,
            grid,
            allow_unmasked=True,
            dead_stock_loss=loss,
            verbose=False,
        )
        assert result.diagnostics["n_symbols"] == 3
        assert result.diagnostics["dead_stock_rule"]["n_ended_in_window"] == 1
        out = strategy.build(bars, grid)
        sim = simulate_signal_out(
            out,
            simulation_close(bars.close, bars.spans, loss),
            TOY_CONFIG,
            dead=forced_exit_frames(bars.close, bars.spans),
        )
        dead_col = out.symbols.index("PERMNO-91002")
        trades = sim.trades.loc[sim.trades.col == dead_col]
        assert len(sim.trades) == len(trades) == 1
        assert int(trades["status"].iloc[0]) == 1  # Closed: no remaining shares.
        assert trades["size"].iloc[0] == pytest.approx(100.0, abs=1e-9, rel=0)
        assert trades["entry_price"].iloc[0] == pytest.approx(50.0, abs=1e-9, rel=0)
        assert trades["exit_price"].iloc[0] == pytest.approx(50.0 * (1 - loss), abs=1e-9, rel=0)
        assert bars.close.index[int(trades.exit_idx.iloc[0])].date() == date(2021, 6, 30)
        assert sim.value.iloc[-1, 0] == pytest.approx(10000 - 5000 * loss, abs=1e-9, rel=0)
        assert sim.returns.iloc[int(trades.exit_idx.iloc[0]), 0] == pytest.approx(
            -0.5 * loss, abs=1e-9
        )
