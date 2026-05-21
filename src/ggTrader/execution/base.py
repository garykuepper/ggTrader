"""Broker Protocol: domain-level interface to an execution venue.

Adapters in this package (``KrakenSpotBroker`` etc.) implement this Protocol
by wrapping their venue SDK. Strategy / sizer / portfolio code must depend
only on ``Broker`` — never on ccxt, alpaca-py, or other concrete SDKs.

Methods are synchronous in Phase 1; async support is deferred until needed.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, Protocol, runtime_checkable

from ggTrader.core.balance import Balance
from ggTrader.core.instrument import Instrument, Venue
from ggTrader.core.order import Fill, Order
from ggTrader.core.ticker import Ticker


@runtime_checkable
class Broker(Protocol):
    venue: Venue

    def submit_order(self, order: Order) -> str:
        """Submit an order; return the venue-assigned order_id.

        Raises ``InsufficientFunds``, ``OrderRejected``, or ``VenueUnavailable``
        from :mod:`ggTrader.execution.errors` on failure.
        """
        ...

    def cancel_order(self, order_id: str, instrument: Instrument) -> bool:
        """Cancel an open order. Returns True if a cancel was issued.

        Raises ``OrderNotFound`` if the venue does not recognize the id.
        """
        ...

    def fetch_order(self, order_id: str, instrument: Instrument) -> Optional[Order]:
        """Return the current state of an order, or None if not found."""
        ...

    def fetch_open_orders(self, instrument: Optional[Instrument] = None) -> list[Order]:
        """List open orders, optionally filtered by instrument."""
        ...

    def fetch_fills(
        self,
        instrument: Optional[Instrument] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[Fill]:
        """Return executed fills, newest-first."""
        ...

    def fetch_balance(self) -> dict[str, Balance]:
        """Return account balances keyed by currency code."""
        ...

    def fetch_ticker(self, instrument: Instrument) -> Ticker:
        """Return the current top-of-book / last-trade snapshot."""
        ...
