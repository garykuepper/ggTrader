"""Typed records of the paper trader; `TypedDict`s describe mappings that are stored as JSON,
dataclasses describe values the trader builds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


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


@dataclass(frozen=True)
class RunResult:
    """What one trader run returns; `as_dict()` is the mapping `run()` hands its callers.

    Attributes:
        buys: Symbols bought (or planned, in a dry run).
        sells: Symbols sold (or planned).
        errors: Messages of the problems the run met.
        gate: The gate that ended the run early, when one did (``"late_run"``).
        untradable: Symbols skipped as untradable; only the final result carries it.
        skipped_writes: Names of the writes a read-only dry run skipped; only the final result
            carries it.
    """

    buys: list[str]
    sells: list[str]
    errors: list[str]
    gate: str | None = None
    untradable: list[str] | None = None
    skipped_writes: list[str] | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return today's keys in today's order; a field that is None is absent.

        The lists are the fields themselves, not copies, as the callers have always received them.
        """
        result: dict[str, Any] = {"buys": self.buys, "sells": self.sells, "errors": self.errors}
        if self.gate is not None:
            result["gate"] = self.gate
        if self.untradable is not None:
            result["untradable"] = self.untradable
        if self.skipped_writes is not None:
            result["skipped_writes"] = self.skipped_writes
        return result
