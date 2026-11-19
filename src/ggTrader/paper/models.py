"""Typed records of the paper trader; `TypedDict`s describe mappings that are stored as JSON,
dataclasses describe values the trader builds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class SubmittedOrder:
    """An order handed to the broker this run, kept until it settles.

    Attributes:
        order_id: The broker's id for the order.
        side: ``"BUY"`` or ``"SELL"``.
        symbol: The ticker.
        amount: Intended notional for a BUY or sweep funding SELL; position market value for
            other SELLs. It is what the alert shows when the order has not filled.
        reason: ``None`` for a strategy order, otherwise the ledger tag:
            ``cash_sweep.SWEEP_TRADE_REASON`` or ``catastrophe_stop.CATASTROPHE_STOP_REASON``.
    """

    order_id: str
    side: Literal["BUY", "SELL"]
    symbol: str
    amount: float
    reason: str | None
