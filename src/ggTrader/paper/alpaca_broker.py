"""Alpaca paper trading broker adapter."""

from __future__ import annotations

import os

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from ggTrader.utils.config import _load_env


class AlpacaBroker:
    """Thin wrapper around alpaca-py TradingClient for paper trading."""

    def __init__(self) -> None:
        _load_env()
        key = os.environ.get("APCA_API_KEY_ID")
        secret = os.environ.get("APCA_API_SECRET_KEY")
        if not key or not secret:
            raise ValueError("APCA_API_KEY_ID and APCA_API_SECRET_KEY must be set in .env")
        self._client = TradingClient(key, secret, paper=True)

    def get_account(self) -> dict:
        acct = self._client.get_account()
        return {
            "cash": float(acct.cash),
            "portfolio_value": float(acct.portfolio_value),
            "buying_power": float(acct.buying_power),
        }

    def get_positions(self) -> dict[str, dict]:
        positions = self._client.get_all_positions()
        return {
            p.symbol: {
                "qty": float(p.qty),
                "market_value": float(p.market_value),
                "avg_entry": float(p.avg_entry_price),
                "unrealized_pl": float(p.unrealized_pl),
            }
            for p in positions
        }

    def submit_buy(self, symbol: str, notional: float) -> str:
        req = MarketOrderRequest(
            symbol=symbol,
            notional=round(notional, 2),
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )
        order = self._client.submit_order(req)
        return str(order.id)

    def submit_sell(self, symbol: str, qty: float) -> str:
        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        order = self._client.submit_order(req)
        return str(order.id)

    def get_clock(self) -> dict:
        clock = self._client.get_clock()
        return {
            "is_open": clock.is_open,
            "next_open": str(clock.next_open),
            "next_close": str(clock.next_close),
        }
