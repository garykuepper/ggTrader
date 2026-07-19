"""Congressional (House STOCK Act) trade-mirroring signal.

Mirrors House members' disclosed open-market purchases (Periodic
Transaction Report transaction type "P"), holding for hold_days. House
only (v1) -- see house_ptr_data.py for the Senate-scope-out rationale.
Event-driven, same pattern as index_deletion.py/insider_cluster.py; a
mirrored symbol is normally still a current index member, so no
"eligible"-bypass side channel is needed.

The original candidate's own caveat (WEB_RESEARCH_CANDIDATES.md #9):
post-STOCK-Act evidence for a reliable, broad-based edge is weak/contested
in the literature -- this v1 mirrors the FULL disclosed-purchase feed
(not restricted to committee-leadership-tied trades, which would need a
separate committee-assignment data source), matching the original
candidate's less-favored "blanket mirror" framing rather than its
recommended committee-restricted refinement.
"""

from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from ggTrader.lab.house_ptr_data import load_house_ptr_transactions
from ggTrader.lab.strategies.indicators import eligible_symbols
from ggTrader.lab.strategy import LabConfig, Plan

#: Conservative gate: STOCK Act allows up to 45 days between a trade and
#: its public disclosure -- use the filing_date (when the PTR was actually
#: published), not the transaction_date, as the point-in-time-available
#: signal, with a small extra buffer for same-day tradeability.
REPORT_LAG_DAYS = 1

#: How far back to pull PTR history per select() call.
_LOOKBACK_DAYS = 400

TxLoader = Callable[[List[str], str, str], pd.DataFrame]


class CongressTradeMirrorStrategy:
    """Long-only event-driven sleeve: equal-weight every symbol with a
    disclosed House-member open-market purchase still within hold_days,
    rebalanced monthly.
    """

    name = "congress_trades"
    target_kind = "weights"

    def __init__(
        self,
        cfg: LabConfig,
        hold_days: int = 252,
        report_lag_days: int = REPORT_LAG_DAYS,
        _tx_loader: TxLoader | None = None,
    ) -> None:
        self.cfg = cfg
        self.hold_days = hold_days
        self.report_lag_days = report_lag_days
        self._tx_loader: TxLoader = _tx_loader or load_house_ptr_transactions
        self._tx_cache: Dict[tuple, pd.DataFrame] = {}

    @classmethod
    def sweep_params(cls) -> dict[str, list]:
        return {"hold_days": [126, 189, 252]}

    def _load_tx(self, symbols: List[str], start: str, end: str) -> pd.DataFrame:
        key = (tuple(sorted(symbols)), start, end)
        if key not in self._tx_cache:
            self._tx_cache[key] = self._tx_loader(symbols, start, end)
        return self._tx_cache[key]

    def select(self, asof: pd.Timestamp, data: pd.DataFrame, eligible: List[str]) -> Plan:
        data = data.loc[:asof]
        elig = eligible_symbols(data, eligible, self.cfg.min_history_bars)
        if not elig:
            return []

        start = (asof - pd.Timedelta(days=_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
        end = asof.strftime("%Y-%m-%d")
        tx = self._load_tx(elig, start, end)
        if tx.empty:
            return []

        purchases = tx[tx["transaction_type"] == "P"]
        if purchases.empty:
            return []

        asof_naive = (
            pd.Timestamp(asof).tz_localize(None) if pd.Timestamp(asof).tz else pd.Timestamp(asof)
        )
        filing_date = pd.to_datetime(purchases["filing_date"])
        if filing_date.dt.tz is not None:
            filing_date = filing_date.dt.tz_localize(None)
        known_by = filing_date + pd.Timedelta(days=self.report_lag_days)
        age_days = (asof_naive - known_by).dt.days
        active = purchases[age_days >= 0]
        if active.empty:
            return []

        trade_date = pd.to_datetime(active["transaction_date"])
        if trade_date.dt.tz is not None:
            trade_date = trade_date.dt.tz_localize(None)
        hold_age = (asof_naive - trade_date).dt.days
        active = active[hold_age <= self.hold_days]
        if active.empty:
            return []

        symbols = sorted(active["symbol"].unique())
        available_symbols = set(data.columns.get_level_values(0).unique())
        selected: List[str] = []
        for sym in symbols:
            if sym not in available_symbols:
                continue
            close = data[sym]["close"]
            if close.empty or pd.isna(close.iloc[-1]):
                continue
            selected.append(sym)
        if not selected:
            return []

        weight = 1.0 / len(selected)
        return [{"symbol": s, "weight": weight} for s in selected]

    def to_targets(self, plans: Dict[pd.Timestamp, Plan], data: pd.DataFrame) -> pd.DataFrame:
        symbols = sorted({s["symbol"] for plan in plans.values() for s in plan})
        targets = pd.DataFrame(np.nan, index=data.index, columns=symbols)
        for asof in sorted(plans):
            forward = data.index[data.index > asof]
            if len(forward) == 0:
                continue
            bar = forward[0]
            targets.loc[bar, symbols] = 0.0  # default: exit anything not re-selected
            for sel in plans[asof]:
                targets.loc[bar, sel["symbol"]] = float(sel["weight"])
        return targets
