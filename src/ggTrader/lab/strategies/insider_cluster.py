"""Insider cluster-buying signal (SEC Form 4).

Flags stocks where 3+ distinct insiders make open-market purchases
(transaction code "P") within a compressed window, excluding scheduled
10b5-1 plan trades and non-purchase transactions (option exercises,
grants). Event-driven, like index_deletion.py, but a cluster-buy symbol
is normally still a current index member -- unlike a deletion, it doesn't
need to bypass the "eligible" argument via a side channel.
"""

from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from ggTrader.lab.form4_data import load_form4_transactions
from ggTrader.lab.strategies.indicators import eligible_symbols
from ggTrader.lab.strategy import LabConfig, Plan

TxLoader = Callable[[List[str], str, str], pd.DataFrame]


def cluster_events(
    transactions: pd.DataFrame, cluster_window_days: int = 14, min_insiders: int = 3
) -> pd.DataFrame:
    """Detect insider cluster-buy events per symbol: open-market purchases
    (code "P", excluding 10b5-1 plans) by >= min_insiders distinct insiders
    within a trailing cluster_window_days window. One event per cluster --
    firing resets the window so a subsequent purchase inside the same
    already-triggered cluster doesn't re-fire the signal.
    """
    empty = pd.DataFrame(columns=["symbol", "event_date"])
    if transactions.empty:
        return empty

    df = transactions[
        (transactions["transaction_code"] == "P") & (~transactions["is_10b5_1_plan"].astype(bool))
    ].copy()
    if df.empty:
        return empty
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])

    events: List[dict] = []
    for symbol, g in df.sort_values("transaction_date").groupby("symbol"):
        window: List[tuple] = []
        for _, row in g.iterrows():
            date = row["transaction_date"]
            insider = row["insider_cik"]
            window = [(d, i) for d, i in window if (date - d).days <= cluster_window_days]
            window.append((date, insider))
            distinct = {i for _, i in window}
            if len(distinct) >= min_insiders:
                events.append({"symbol": symbol, "event_date": date})
                window = []  # cluster consumed -- next purchase starts a fresh one
    return pd.DataFrame(events) if events else empty


class InsiderClusterBuyStrategy:
    """Long-only event-driven sleeve: equal-weight every symbol still
    within hold_days of an insider cluster-buy event, rebalanced monthly.
    """

    name = "insider_cluster_buy"
    target_kind = "weights"

    def __init__(
        self,
        cfg: LabConfig,
        cluster_window_days: int = 14,
        min_insiders: int = 3,
        hold_days: int = 252,
        _tx_loader: TxLoader | None = None,
    ) -> None:
        self.cfg = cfg
        self.cluster_window_days = cluster_window_days
        self.min_insiders = min_insiders
        self.hold_days = hold_days
        self._tx_loader: TxLoader = _tx_loader or load_form4_transactions
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

        lookback_days = self.hold_days + self.cluster_window_days + 30
        start = (asof - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        end = asof.strftime("%Y-%m-%d")
        tx = self._load_tx(elig, start, end)
        if tx.empty:
            return []

        events = cluster_events(tx, self.cluster_window_days, self.min_insiders)
        if events.empty:
            return []

        asof_naive = (
            pd.Timestamp(asof).tz_localize(None) if pd.Timestamp(asof).tz else pd.Timestamp(asof)
        )
        event_date = pd.to_datetime(events["event_date"])
        if event_date.dt.tz is not None:
            event_date = event_date.dt.tz_localize(None)
        age_days = (asof_naive - event_date).dt.days
        active = events[(age_days >= 0) & (age_days <= self.hold_days)]
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
