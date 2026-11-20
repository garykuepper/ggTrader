"""The two ports of the paper trader: ``Broker`` and ``Notifier``, as structural protocols.

Public names: ``Broker`` and ``Notifier``. ``Broker`` lists the 11 public methods of
``AlpacaBroker`` that the trader calls (``get_account``, ``get_positions``, ``get_untradable``,
``submit_buy``, ``submit_sell``, ``get_order``, ``get_order_by_client_id``, ``get_clock``,
``get_split_evidence``, ``get_split_corrections``, ``get_dividend_corrections``); its
``_normalize_order`` and ``_get_*`` helpers are private and excluded. ``Notifier`` lists
``send``, ``trade_alert`` and ``daily_summary`` of ``TelegramNotifier``. The one-method
``Notifier`` of ``data/membership_weekly.py`` is a different, narrower protocol: ``data`` must
not import ``paper``.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Protocol, runtime_checkable

from ggTrader.paper.models import Account, MarketClock, OrderStatus, Position


@runtime_checkable
class Broker(Protocol):
    """Order, account and corporate-action access (``AlpacaBroker`` is the implementation).

    Port contract: ``get_account`` and ``get_positions`` return fresh objects that the caller
    may keep. ``PortfolioView`` holds frozen copies of them for the whole run, so a broker must
    not return a shared dictionary that it mutates afterwards.
    """

    def get_account(self) -> Account:
        """Return a fresh account record the caller may keep (never a shared, mutated one)."""
        ...

    def get_positions(self) -> dict[str, Position]:
        """Return fresh position records the caller may keep (never a shared, mutated dict)."""
        ...

    def get_untradable(self, symbols: list[str]) -> set[str]:
        """Return the symbols the broker marks as not tradable."""
        ...

    def submit_buy(self, symbol: str, notional: float, client_order_id: str) -> str:
        """Submit a market buy for ``notional`` dollars; return the broker order id."""
        ...

    def submit_sell(self, symbol: str, qty: float, client_order_id: str) -> str:
        """Submit a market sell of ``qty`` shares; return the broker order id."""
        ...

    def get_order(self, order_id: str) -> OrderStatus:
        """Return the normalized status of the order with this broker order id."""
        ...

    def get_order_by_client_id(self, client_order_id: str) -> OrderStatus:
        """Return the normalized status of the order with this client order id."""
        ...

    def get_clock(self) -> MarketClock:
        """Return the market clock (whether the market is open, and the next open and close)."""
        ...

    def get_split_evidence(self, symbols: list[str], since: date) -> dict[str, Any]:
        """Return the broker's raw split evidence for ``symbols`` since ``since``."""
        ...

    def get_split_corrections(self, symbols: list[str], since: date) -> dict[str, float]:
        """Return a correction factor per symbol whose split was never applied to the account."""
        ...

    def get_dividend_corrections(self, symbols: list[str], since: date | str) -> dict[str, Any]:
        """Return the cash-dividend evidence for ``symbols`` since ``since``."""
        ...


@runtime_checkable
class Notifier(Protocol):
    """Alerts and the daily summary (``TelegramNotifier`` is the implementation)."""

    def send(self, message: str) -> bool:
        """Send a message and return whether it was delivered."""
        ...

    def trade_alert(
        self,
        side: str,
        symbol: str,
        amount: float,
        order_id: str,
        *,
        qty: float | None = None,
        price: float | None = None,
        status: str | None = None,
    ) -> bool:
        """Send an alert for one order and return whether it was delivered."""
        ...

    def daily_summary(
        self,
        portfolio_value: float | None,
        daily_pnl: float | None,
        positions: dict[str, Position],
        total_pnl: float | None = None,
        unrealized_pnl: float | None = None,
        split_corrections: dict[str, float] | None = None,
        booked_equity: float | None = None,
        dividend_total: float | None = None,
        new_dividend_accruals: list[dict] | None = None,
        errors: list[str] | None = None,
        starting_capital: float | None = None,
    ) -> bool:
        """Send the end-of-run summary and return whether it was delivered."""
        ...
