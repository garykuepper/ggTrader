"""Kraken Futures broker adapter — wraps ccxt.krakenfutures behind ``Broker``.

Mirrors ``KrakenSpotBroker`` for the futures venue. Same error translation,
same Pydantic round-tripping, same testability pattern (inject a mock ccxt
exchange for unit tests).

Kraken Futures contract symbols look like ``BTC/USD:USD-260626`` (linear,
USD-margined, expiring 2026-06-26). The strategy passes the contract's
expiry string via ``Instrument.expiry`` and we encode it on the
``Instrument.venue_specific_id``.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Optional

import ccxt  # type: ignore[import-untyped]

from ggTrader.core.balance import Balance
from ggTrader.core.instrument import Instrument, Venue
from ggTrader.core.order import Fill, Order
from ggTrader.core.signal import Direction
from ggTrader.core.ticker import Ticker
from ggTrader.execution.errors import (
    BrokerError,
    InsufficientFunds,
    OrderNotFound,
    OrderRejected,
    VenueUnavailable,
)


def _to_ccxt_symbol(instrument: Instrument) -> str:
    if instrument.venue_specific_id:
        return instrument.venue_specific_id
    # Futures canonical → ccxt: "BTC-USD-260626" → "BTC/USD:USD-260626"
    if instrument.expiry:
        return f"{instrument.base_currency}/{instrument.quote_currency}:{instrument.quote_currency}-{instrument.expiry}"
    return f"{instrument.base_currency}/{instrument.quote_currency}:{instrument.quote_currency}"


def _ts_from_ms(ms: Optional[int]) -> datetime:
    if ms is None:
        return datetime.now(tz=timezone.utc)
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def _dec(value: Any) -> Decimal:
    if value is None:
        return Decimal("0")
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


class KrakenFuturesBroker:
    venue: Venue = Venue.KRAKEN_FUTURES

    def __init__(self, exchange: Optional[ccxt.Exchange] = None) -> None:
        if exchange is None:
            exchange = ccxt.krakenfutures(
                {
                    "apiKey": os.getenv("KRAKEN_FUTURES_KEY"),
                    "secret": os.getenv("KRAKEN_FUTURES_SECRET"),
                    "enableRateLimit": True,
                }
            )
        self._exchange = exchange

    def submit_order(self, order: Order) -> str:
        symbol = _to_ccxt_symbol(order.instrument)
        side = "buy" if order.side is Direction.LONG else "sell"
        amount = float(order.quantity)

        try:
            if order.order_type == "market":
                raw = self._exchange.create_order(symbol, "market", side, amount)
            elif order.order_type == "limit":
                if order.limit_price is None:
                    raise OrderRejected("limit order requires limit_price")
                raw = self._exchange.create_order(
                    symbol, "limit", side, amount, float(order.limit_price)
                )
            else:
                raise OrderRejected(f"unsupported order_type: {order.order_type}")
        except ccxt.InsufficientFunds as exc:
            raise InsufficientFunds(str(exc)) from exc
        except ccxt.NetworkError as exc:
            raise VenueUnavailable(str(exc)) from exc
        except ccxt.ExchangeError as exc:
            raise OrderRejected(str(exc)) from exc

        return str(raw["id"])

    def cancel_order(self, order_id: str, instrument: Instrument) -> bool:
        symbol = _to_ccxt_symbol(instrument)
        try:
            self._exchange.cancel_order(order_id, symbol)
        except ccxt.OrderNotFound as exc:
            raise OrderNotFound(str(exc)) from exc
        except ccxt.NetworkError as exc:
            raise VenueUnavailable(str(exc)) from exc
        except ccxt.ExchangeError as exc:
            raise BrokerError(str(exc)) from exc
        return True

    def fetch_order(self, order_id: str, instrument: Instrument) -> Optional[Order]:
        symbol = _to_ccxt_symbol(instrument)
        try:
            raw = self._exchange.fetch_order(order_id, symbol)
        except ccxt.OrderNotFound:
            return None
        except ccxt.NetworkError as exc:
            raise VenueUnavailable(str(exc)) from exc
        return self._raw_to_order(raw, instrument)

    def fetch_open_orders(self, instrument: Optional[Instrument] = None) -> list[Order]:
        symbol = _to_ccxt_symbol(instrument) if instrument is not None else None
        try:
            raws = self._exchange.fetch_open_orders(symbol)
        except ccxt.NetworkError as exc:
            raise VenueUnavailable(str(exc)) from exc
        results: list[Order] = []
        for raw in raws:
            if instrument is not None:
                results.append(self._raw_to_order(raw, instrument))
        return results

    def fetch_fills(
        self,
        instrument: Optional[Instrument] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[Fill]:
        symbol = _to_ccxt_symbol(instrument) if instrument is not None else None
        since_ms = int(since.timestamp() * 1000) if since is not None else None
        try:
            raws = self._exchange.fetch_my_trades(symbol, since=since_ms, limit=limit)
        except ccxt.NetworkError as exc:
            raise VenueUnavailable(str(exc)) from exc
        fills: list[Fill] = []
        for raw in raws:
            if instrument is None:
                continue
            fills.append(
                Fill(
                    ts=_ts_from_ms(raw.get("timestamp")),
                    order_id=str(raw.get("order") or ""),
                    instrument=instrument,
                    side=Direction.LONG if raw.get("side") == "buy" else Direction.SHORT,
                    quantity=_dec(raw.get("amount")),
                    price=_dec(raw.get("price")),
                    fee=_dec((raw.get("fee") or {}).get("cost")),
                    fee_currency=(raw.get("fee") or {}).get("currency")
                    or instrument.quote_currency,
                )
            )
        return fills

    def fetch_balance(self) -> dict[str, Balance]:
        try:
            raw = self._exchange.fetch_balance()
        except ccxt.NetworkError as exc:
            raise VenueUnavailable(str(exc)) from exc

        out: dict[str, Balance] = {}
        for ccy, entry in raw.items():
            if not isinstance(entry, dict):
                continue
            if not {"free", "used", "total"}.issubset(entry.keys()):
                continue
            free = _dec(entry["free"])
            used = _dec(entry["used"])
            total = _dec(entry["total"])
            if free + used != total:
                total = free + used
            out[ccy] = Balance(currency=ccy, free=free, used=used, total=total)
        return out

    def fetch_ticker(self, instrument: Instrument) -> Ticker:
        symbol = _to_ccxt_symbol(instrument)
        try:
            raw = self._exchange.fetch_ticker(symbol)
        except ccxt.NetworkError as exc:
            raise VenueUnavailable(str(exc)) from exc
        return Ticker(
            ts=_ts_from_ms(raw.get("timestamp")),
            instrument=instrument,
            bid=_dec(raw["bid"]) if raw.get("bid") is not None else None,
            ask=_dec(raw["ask"]) if raw.get("ask") is not None else None,
            last=_dec(raw.get("last") or raw.get("close") or 0),
            volume_24h=_dec(raw["baseVolume"]) if raw.get("baseVolume") is not None else None,
        )

    @staticmethod
    def _raw_to_order(raw: dict[str, Any], instrument: Instrument) -> Order:
        side = Direction.LONG if raw.get("side") == "buy" else Direction.SHORT
        limit_price = raw.get("price")
        return Order(
            ts=_ts_from_ms(raw.get("timestamp")),
            instrument=instrument,
            side=side,
            quantity=_dec(raw.get("amount")),
            order_type=str(raw.get("type") or "market"),
            limit_price=_dec(limit_price) if limit_price is not None else None,
            time_in_force=str(raw.get("timeInForce") or "GTC"),
            client_order_id=str(raw.get("clientOrderId") or raw.get("id") or uuid.uuid4()),
        )
