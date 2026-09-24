"""Month-end Treasury duration sleeve -- candidate A10 from WEB_RESEARCH_CANDIDATES.md's
2026-09-23 batch. Long one Treasury ETF over the last N trading days of each
calendar month, cash otherwise.

Source: Jonathan Hartley & Krista Schwarz, "Predictable End-of-Month Treasury
Returns," SSRN 3440417 (2019). Bond indexes extend duration at month-end, so
index trackers buy duration on predictable dates; the paper's frozen rule is
long the last 3 trading days (enter at the close of the 4th-to-last trading
day, exit at the close of the last).

Same shape as fomc_drift: `target_kind="signals"` because a 3-day hold can't be
expressed by the monthly weights path; select() is a fixed-universe
pass-through and to_targets() places entries/exits from the trading calendar.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from ggTrader.lab.strategies.indicators import extract_close
from ggTrader.lab.strategy import LabConfig, Plan, SignalTargets

#: Ordered by duration (IEF 7-10y < TLT 20y+ < EDV 20-30y STRIPS) so the NDH
#: neighborhood on `duration_rank` compares adjacent durations, not an
#: alphabetical accident.
MONTH_END_TREASURY_UNIVERSE = ["IEF", "TLT", "EDV"]


class MonthEndTreasuryStrategy:
    name = "month_end_treasury"
    target_kind = "signals"

    def __init__(self, cfg: LabConfig, days_before_end: int = 3, duration_rank: int = 0) -> None:
        self.cfg = cfg
        #: Hold over the last N trading days' returns (enter N bars before month-end).
        self.days_before_end = days_before_end
        #: Index into MONTH_END_TREASURY_UNIVERSE.
        self.duration_rank = duration_rank

    @classmethod
    def sweep_params(cls) -> dict[str, list]:
        return {"days_before_end": [2, 3, 4, 5], "duration_rank": [0, 1, 2]}

    def select(self, asof: pd.Timestamp, data: pd.DataFrame, eligible: List[str]) -> Plan:
        have = set(data.columns.get_level_values(0).unique())
        return [{"symbol": s, "weight": 0.0} for s in MONTH_END_TREASURY_UNIVERSE if s in have]

    def to_targets(self, plans: Dict[pd.Timestamp, Plan], data: pd.DataFrame) -> SignalTargets:
        symbols = sorted({s["symbol"] for plan in plans.values() for s in plan})
        close = extract_close(data, symbols)
        return month_end_signals(
            close.index,
            symbols,
            MONTH_END_TREASURY_UNIVERSE[self.duration_rank],
            self.days_before_end,
        )

    def sweep_signals(
        self, combos: list[dict], symbols: list[str], data: pd.DataFrame
    ) -> dict[str, SignalTargets]:
        from ggTrader.lab.sweep import combo_name

        index = extract_close(data, symbols).index
        return {
            combo_name(self.name, c): month_end_signals(
                index,
                symbols,
                MONTH_END_TREASURY_UNIVERSE[int(c["duration_rank"])],
                int(c["days_before_end"]),
            )
            for c in combos
        }


def month_end_signals(
    index: pd.DatetimeIndex, symbols: List[str], instrument: str, days_before_end: int
) -> SignalTargets:
    """Entry `days_before_end` bars before each month's last trading bar, exit on it.

    Month-ends come from the bar index itself (the real trading calendar,
    holidays included), grouped by calendar date -- bars carry a time of day.
    The final month is skipped when the index stops before that month's last
    weekday: a truncated slice would otherwise mislabel a mid-month bar as
    month-end.
    """
    entries = pd.DataFrame(False, index=index, columns=symbols)
    exits = pd.DataFrame(False, index=index, columns=symbols)
    if instrument not in symbols or len(index) == 0:
        return SignalTargets(entries=entries, exits=exits)

    dates = index.tz_localize(None).normalize() if index.tz is not None else index.normalize()
    pos = pd.Series(range(len(index)), index=dates)
    last_pos = pos.groupby(dates.to_period("M")).max()
    # ponytail: weekday check, not an NYSE calendar -- a month whose last weekday is
    # a holiday (e.g. Good Friday 2024-03-29) is dropped only if it is also the
    # truncated final month of the slice.
    if dates[-1] < dates[-1] + pd.offsets.BMonthEnd(0):
        last_pos = last_pos.iloc[:-1]

    col = symbols.index(instrument)
    for p in last_pos:
        entry = p - days_before_end
        if entry >= 0:
            entries.iat[entry, col] = True
            exits.iat[p, col] = True
    return SignalTargets(entries=entries, exits=exits)
