"""Domain exceptions raised by Broker adapters.

Adapters translate venue-specific errors (ccxt exceptions, HTTP errors, etc.)
into these stable domain types so strategy/orchestration code can catch by
intent rather than by venue SDK.
"""

from __future__ import annotations


class BrokerError(Exception):
    """Base for all broker-side failures."""


class InsufficientFunds(BrokerError):
    """Order rejected because account balance is too low."""


class OrderNotFound(BrokerError):
    """Lookup or cancel referenced an unknown order_id."""


class OrderRejected(BrokerError):
    """Order rejected for any reason other than funds (e.g., price band, size)."""


class VenueUnavailable(BrokerError):
    """Network or venue-side outage; the request may be safe to retry."""
