"""Alpaca paper trading broker adapter."""

from __future__ import annotations

import logging
import os
from datetime import date

from alpaca.data.enums import CorporateActionsType
from alpaca.data.historical.corporate_actions import CorporateActionsClient
from alpaca.data.requests import CorporateActionsRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from ggTrader.paper.dividend_check import normalize_since
from ggTrader.paper.split_check import compute_split_corrections
from ggTrader.utils.config import _load_env

_log = logging.getLogger(__name__)


class AlpacaBroker:
    """Thin wrapper around alpaca-py TradingClient for paper trading."""

    def __init__(self) -> None:
        _load_env()
        key = os.environ.get("APCA_API_KEY_ID")
        secret = os.environ.get("APCA_API_SECRET_KEY")
        if not key or not secret:
            raise ValueError("APCA_API_KEY_ID and APCA_API_SECRET_KEY must be set in .env")
        self._key = key
        self._secret = secret
        self._client = TradingClient(key, secret, paper=True)

    def get_account(self) -> dict:
        acct = self._client.get_account()
        return {
            "cash": float(acct.cash),
            "portfolio_value": float(acct.portfolio_value),
            "buying_power": float(acct.buying_power),
            "multiplier": float(acct.multiplier),
        }

    def get_positions(self) -> dict[str, dict]:
        positions = self._client.get_all_positions()
        return {
            p.symbol: {
                "qty": float(p.qty),
                "market_value": float(p.market_value),
                "current_price": float(p.current_price),
                "avg_entry": float(p.avg_entry_price),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc),
                "change_today": float(p.change_today),
                "cost_basis": float(p.cost_basis),
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

    def get_order(self, order_id: str) -> dict:
        o = self._client.get_order_by_id(order_id)
        return {
            "id": str(o.id),
            "symbol": o.symbol,
            "side": o.side.value if hasattr(o.side, "value") else str(o.side),
            "qty": float(o.qty) if o.qty is not None else None,
            "notional": float(o.notional) if o.notional is not None else None,
            "filled_qty": float(o.filled_qty) if o.filled_qty is not None else 0.0,
            "filled_avg_price": float(o.filled_avg_price)
            if o.filled_avg_price is not None
            else 0.0,
            "status": o.status.value if hasattr(o.status, "value") else str(o.status),
        }

    def get_clock(self) -> dict:
        clock = self._client.get_clock()
        return {
            "is_open": clock.is_open,
            "next_open": str(clock.next_open),
            "next_close": str(clock.next_close),
        }

    def _get_corporate_action_splits(
        self, symbols: list[str], since: date
    ) -> dict[str, list[tuple[date, float]]]:
        """Query Alpaca's corporate-actions feed for forward/reverse splits.

        Returns `{symbol: [(ex_date, factor), ...]}` where `factor` is
        `new_rate / old_rate`. Raises on failure -- the caller
        (`get_split_corrections`) is the fail-soft boundary.
        """
        client = CorporateActionsClient(self._key, self._secret)
        req = CorporateActionsRequest(
            symbols=symbols,
            types=[CorporateActionsType.FORWARD_SPLIT, CorporateActionsType.REVERSE_SPLIT],
            start=since,
        )
        result = client.get_corporate_actions(req)
        data = result.data
        splits: dict[str, list[tuple[date, float]]] = {}
        for key in ("forward_splits", "reverse_splits"):
            for action in data.get(key, []):
                factor = float(action.new_rate) / float(action.old_rate)
                splits.setdefault(action.symbol, []).append((action.ex_date, factor))
        return splits

    def _get_split_activity_symbols(self, since: date) -> set[str]:
        """Symbols with a SPLIT activity already booked against this account
        on/after `since`. Raises on failure -- see `_get_corporate_action_splits`."""
        activities = self._client.get("/account/activities/SPLIT", {"after": since.isoformat()})
        return {a["symbol"] for a in activities if a.get("symbol")}

    def get_split_corrections(self, symbols: list[str], since: date) -> dict[str, float]:
        """Cross-reference the broker's corporate-actions feed against this
        account's own SPLIT activities to find splits the broker knows about
        for `symbols` but never applied to this account.

        Returns `{symbol: correction_factor}` (see
        `split_check.compute_split_corrections`). Fails soft: any API error
        is logged as a warning and this returns `{}` ("no corrections
        known") rather than raising -- this must never abort a trading run.
        """
        if not symbols:
            return {}
        try:
            corp_splits = self._get_corporate_action_splits(symbols, since)
            if not corp_splits:
                return {}
            applied_symbols = self._get_split_activity_symbols(since)
            return compute_split_corrections(corp_splits, applied_symbols)
        except Exception as exc:
            _log.warning("Split correction check failed (non-fatal): %s", exc)
            return {}

    def _get_corporate_action_dividends(
        self, symbols: list[str], since: date
    ) -> dict[str, list[tuple[date, float]]]:
        """Query Alpaca's corporate-actions feed for cash dividends.

        Returns `{symbol: [(ex_date, rate), ...]}` where `rate` is the
        per-share cash amount. Raises on failure -- the caller
        (`get_dividend_corrections`) is the fail-soft boundary.
        """
        client = CorporateActionsClient(self._key, self._secret)
        req = CorporateActionsRequest(
            symbols=symbols,
            types=[CorporateActionsType.CASH_DIVIDEND],
            start=since,
        )
        result = client.get_corporate_actions(req)
        data = result.data
        dividends: dict[str, list[tuple[date, float]]] = {}
        for action in data.get("cash_dividends", []):
            dividends.setdefault(action.symbol, []).append((action.ex_date, float(action.rate)))
        return dividends

    def _get_dividend_activity_keys(self, since: date) -> set[tuple[str, date]]:
        """`(symbol, date)` pairs already credited to this account via a
        real DIV activity on/after `since`. Raises on failure -- see
        `_get_corporate_action_dividends`."""
        activities = self._client.get("/account/activities/DIV", {"after": since.isoformat()})
        keys: set[tuple[str, date]] = set()
        for a in activities:
            symbol = a.get("symbol")
            raw_date = a.get("date")
            if not symbol or not raw_date:
                continue
            keys.add((symbol, date.fromisoformat(raw_date)))
        return keys

    def get_dividend_corrections(
        self, symbols: list[str], since: date | str
    ) -> dict[str, dict[str, list[tuple[date, float]]] | set[tuple[str, date]]]:
        """Cross-reference the broker's corporate-actions feed against this
        account's own DIV activities to find cash dividends the broker knows
        about for `symbols` but never credited to this account.

        `since` is coerced via `dividend_check.normalize_since` *before* the
        fail-soft boundary below, so a caller passing the wrong type (e.g.
        a `str` where a split-check-style caller might expect `date`) fails
        loudly here instead of being silently swallowed by the `except`
        below into a "no dividends known" `{}` -- see
        `dividend_check.normalize_since`'s docstring for the split-check
        landmine this avoids.

        Returns `{"corp_dividends": {symbol: [(ex_date, rate), ...]},
        "credited_keys": {(symbol, ex_date), ...}}`. Fails soft on any API
        error: logs a warning and returns empty collections rather than
        raising -- this must never abort a trading run.
        """
        since = normalize_since(since)
        empty: dict[str, dict | set] = {"corp_dividends": {}, "credited_keys": set()}
        if not symbols:
            return empty
        try:
            corp_dividends = self._get_corporate_action_dividends(symbols, since)
            if not corp_dividends:
                return empty
            credited_keys = self._get_dividend_activity_keys(since)
            return {"corp_dividends": corp_dividends, "credited_keys": credited_keys}
        except Exception as exc:
            _log.warning("Dividend correction check failed (non-fatal): %s", exc)
            return empty
