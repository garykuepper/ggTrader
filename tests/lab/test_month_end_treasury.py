"""Tests for month_end_treasury (candidate A10) calendar placement."""

import pandas as pd

from ggTrader.lab.strategies.month_end_treasury import (
    MONTH_END_TREASURY_UNIVERSE,
    MonthEndTreasuryStrategy,
    month_end_signals,
)
from ggTrader.lab.strategy import LabConfig
from ggTrader.lab.sweep import build_grid, combo_name

SYMS = ["EDV", "IEF", "TLT"]


def _index(end: str) -> pd.DatetimeIndex:
    # Business days with a 16:00 UTC stamp (bars carry a time of day), minus
    # Good Friday 2024-03-29 so March's real last trading day is the 28th.
    idx = pd.bdate_range("2024-02-01", end, tz="UTC") + pd.Timedelta(hours=16)
    return idx[idx.normalize() != pd.Timestamp("2024-03-29", tz="UTC")]


def test_last_three_trading_days_use_real_calendar():
    idx = _index("2024-04-30")
    st = month_end_signals(idx, SYMS, "IEF", 3)
    entry_days = [str(d.date()) for d in idx[st.entries["IEF"].to_numpy()]]
    exit_days = [str(d.date()) for d in idx[st.exits["IEF"].to_numpy()]]
    # Feb: last=29th -> entry 26th; Mar: last=28th (Good Friday) -> entry 25th.
    assert entry_days == ["2024-02-26", "2024-03-25", "2024-04-25"]
    assert exit_days == ["2024-02-29", "2024-03-28", "2024-04-30"]
    assert not st.entries[["EDV", "TLT"]].to_numpy().any()


def test_truncated_final_month_is_skipped():
    st = month_end_signals(_index("2024-04-29"), SYMS, "TLT", 3)
    assert st.entries["TLT"].sum() == 2 and st.exits["TLT"].sum() == 2


def test_sweep_signals_route_instrument_by_duration_rank():
    data = pd.concat(
        {s: pd.DataFrame({"close": 1.0}, index=_index("2024-04-30")) for s in SYMS}, axis=1
    )
    strat = MonthEndTreasuryStrategy(LabConfig())
    grid = build_grid(MonthEndTreasuryStrategy)
    assert len(grid) == 12
    out = strat.sweep_signals(grid, SYMS, data)
    for c in grid:
        sym = MONTH_END_TREASURY_UNIVERSE[c["duration_rank"]]
        t = out[combo_name(strat.name, c)]
        assert t.entries.sum().to_dict() == {s: (3 if s == sym else 0) for s in SYMS}
