"""Order execution and recovery for paper trading."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from decimal import Decimal
from typing import Protocol

from ggTrader.paper import cash_sweep
from ggTrader.paper.constants import STALE_PENDING_DAYS as _STALE_PENDING_DAYS
from ggTrader.paper.constants import TERMINAL_STATUSES, TRADER_LOGGER_NAME
from ggTrader.paper.ledger import PaperLedger, RunLock
from ggTrader.paper.models import IntentKind, OrderIntent, OrderStatus, Position, SubmittedOrder
from ggTrader.paper.ports import Broker, Notifier
from ggTrader.paper.seams import RunHooks

_log = logging.getLogger(TRADER_LOGGER_NAME)


class Executor(Protocol):
    """Run-path order execution contract."""

    submits_orders: bool

    def recover_intents(self) -> None:
        """Resolve the order intents a previous run left unresolved.

        Live: attaches broker ids, books terminal fills, and raises (after an alert attempt) on
        working orders or a recovery error; earlier writes may already be committed. Dry run: no-op.
        """
        ...

    def submit(self, intent: OrderIntent) -> str | None:
        """Submit the intent's order; return the broker order id, or None when none is sent.

        Live: commits the intent, verifies the run lock, submits, then attaches the broker id; a
        failure at any stage alerts and raises, even after the broker accepted the order.
        """
        ...

    def announce(self, intent: OrderIntent) -> None:
        """Alert the operator: live only for a catastrophe stop; dry run for every intent kind."""
        ...

    def poll(self, orders: Sequence[SubmittedOrder]) -> dict[str, OrderStatus]:
        """Poll the submitted orders; return their order info keyed by order id."""
        ...

    def after_sells(self, filled: Mapping[str, OrderStatus]) -> None:
        """Follow up after the sell orders have been polled.

        Live: reconciles the intents when there are results. Dry run: no-op.
        """
        ...

    def after_orders(self, orders: Sequence[SubmittedOrder]) -> None:
        """Follow up after the strategy order batch has been polled, before any trailing sweep buy.

        Live: reconciles the intents when there are orders. Dry run: no-op.
        """
        ...

    def unfilled_exit_errors(
        self,
        orders: Sequence[SubmittedOrder],
        filled: Mapping[str, OrderStatus],
        broker_positions: Mapping[str, Position],
    ) -> list[str]:
        """Return an error message for each sell that did not fill in full.

        Any message blocks the run's strategy buys. Live skips the cash-sweep sell; dry run
        returns none.
        """
        ...


class OrderExecutor:
    """Execute and recover orders through injected collaborators."""

    submits_orders = True

    def __init__(
        self,
        *,
        broker: Broker,
        notifier: Notifier,
        ledger: PaperLedger,
        hooks: RunHooks,
        lock: Callable[[], RunLock | None],
        read_only: bool,
        current: Callable[[], OrderExecutor] | None = None,
    ) -> None:
        """Wire the executor.

        Args:
            current: builds a fresh executor from the trader's state at the moment it is called.
                The submit, poll and reconcile entry points and the stale-order alert run on
                what it returns, so a collaborator or flag the trader replaced mid-run applies to
                the next call, as it did when each call went through a trader method that rebuilt
                its executor. ``None`` (a hand-built executor) runs on itself.
        """
        self._broker = broker
        self._notifier = notifier
        self._ledger = ledger
        self._hooks = hooks
        self._lock = lock
        self._read_only = read_only
        self._current = current if current is not None else lambda: self

    def recover_intents(self) -> None:
        """Reconcile the order intents through the current executor.

        Attaches broker ids and books terminal fills; raises, after an alert attempt, on working
        orders or a recovery error, with earlier writes possibly already committed.
        """
        self._current().reconcile_order_intents()

    def submit(self, intent: OrderIntent) -> str | None:
        """Submit the intent's order through the current executor; return the broker order id.

        Commits the intent, verifies the run lock, submits, then attaches the id. A failure at any
        stage alerts and raises; attachment can fail after the broker accepted the order.
        """
        return self._current().submit_order(intent.side, intent.symbol, intent.size, intent.reason)

    def poll(self, orders: Sequence[SubmittedOrder]) -> dict[str, OrderStatus]:
        """Poll the orders through the current executor; return their info keyed by order id."""
        return self._current().poll_orders(list(orders))

    def after_sells(self, filled: Mapping[str, OrderStatus]) -> None:
        """Reconcile the order intents when there are sell results; do nothing if there are none.

        Reconciliation attaches broker ids, books terminal fills and can raise after an alert.
        """
        if filled:
            self._current().reconcile_order_intents()

    def after_orders(self, orders: Sequence[SubmittedOrder]) -> None:
        """Reconcile the order intents when there are orders; do nothing for an empty sequence.

        Reconciliation attaches broker ids, books terminal fills and can raise after an alert.
        """
        if orders:
            self._current().reconcile_order_intents()

    def announce(self, intent: OrderIntent) -> None:
        """Send the catastrophe-stop alert for a stop intent; every other intent sends nothing."""
        if intent.kind == IntentKind.STOP:
            self._notifier.send(
                f"<b>🛑 Catastrophe stop:</b> {intent.symbol} "
                f"unrealized {intent.unrealized_pct:.1%} "
                f"breached {intent.stop_threshold:.1%} floor — selling qty {intent.size}."
            )

    def unfilled_exit_errors(
        self,
        orders: Sequence[SubmittedOrder],
        filled: Mapping[str, OrderStatus],
        broker_positions: Mapping[str, Position],
    ) -> list[str]:
        """Return a message for each sell that is not filled in full.

        The cash-sweep sell is skipped. A sell counts as filled only when its status is "filled" and
        its filled quantity is within a `math.isclose` tolerance of the broker position quantity.
        Any message blocks the run's strategy buys.
        """
        errors = []
        for order in orders:
            if order.side != "SELL" or order.reason == cash_sweep.SWEEP_TRADE_REASON:
                continue
            info = filled.get(order.order_id, {})
            qty = info.get("filled_qty")
            expected = broker_positions[order.symbol]["qty"]
            if (
                info.get("status") != "filled"
                or qty is None
                or not math.isfinite(qty)
                or not math.isclose(qty, expected, rel_tol=1e-12, abs_tol=1e-9)
            ):
                errors.append(
                    f"SELL {order.symbol} not fully filled "
                    f"({info.get('status', 'unknown')}); "
                    "new strategy buys blocked"
                )
        return errors

    def _pending_load_failed(self, exc: Exception) -> set[str]:
        self._notifier.send(f"<b>🛑 Could not load pending orders — no orders placed:</b> {exc}")
        raise exc

    def submit_order(self, side: str, symbol: str, amount: float, reason: str | None) -> str:
        """Persist the identity before submitting an order under the run lock."""
        if side not in ("BUY", "SELL"):
            raise ValueError("Invalid order side")
        if self._read_only:
            # Fail closed: a read-only ledger would skip the intent write, so the order would
            # reach the broker with no durable identity. Nothing below may run.
            raise RuntimeError("read-only trader must not submit orders")
        client_id = str(self._hooks.new_uuid())
        submit = self._broker.submit_buy if side == "BUY" else self._broker.submit_sell
        stage = "intent"
        try:
            self._ledger.create_order_intent(
                client_id,
                self._hooks.today_et().isoformat(),
                side,
                symbol,
                amount,
                "notional" if side == "BUY" else "qty",
                reason,
            )
            stage = "lock"
            self._lock().verify()
            stage = "submit"
            order_id = submit(symbol, amount, client_id)
            stage = "attach"
            self._ledger.attach_broker_order(client_id, order_id)
        except Exception:
            try:
                self._notifier.send(f"Order {client_id} failed at {stage}; investigation required")
            except Exception:
                _log.exception("Failed to notify about order %s at %s", client_id, stage)
            raise
        return order_id

    def reconcile_order_intents(self) -> set[str]:
        """Resolve terminal intents and block uncertain or working orders."""
        try:
            intents = self._ledger.get_unresolved_order_intents()
            legacy_ids = {p["order_id"] for p in self._ledger.get_pending_orders()}
            working = set()
            for intent in intents:
                client_id = intent["client_order_id"]
                info = self._broker.get_order_by_client_id(client_id)
                oid = info.get("id")
                if (
                    not isinstance(oid, str)
                    or not oid.strip()
                    or info.get("client_order_id") != client_id
                    or (intent.get("broker_order_id") and intent["broker_order_id"] != oid)
                    or oid in legacy_ids
                    or str(info.get("symbol", "")).upper() != intent["symbol"].upper()
                    or str(info.get("side", "")).upper() != intent["side"].upper()
                ):
                    raise ValueError("Order identity mismatch or duplicate legacy order")
                kind = intent["amount_kind"]
                actual, expected = Decimal(str(info.get(kind))), Decimal(str(intent["amount"]))
                tolerance = Decimal("0.01") if kind == "notional" else Decimal("0.000000001")
                if (
                    not actual.is_finite()
                    or not expected.is_finite()
                    or abs(actual - expected) > tolerance
                ):
                    raise ValueError("Order amount mismatch")
                qty = Decimal(str(info.get("filled_qty")))
                price_value = info.get("filled_avg_price")
                price = Decimal(str(price_value)) if price_value is not None else None
                if (
                    not qty.is_finite()
                    or qty < 0
                    or (price is not None and (not price.is_finite() or price < 0))
                    or (qty > 0 and (price is None or price <= 0))
                ):
                    raise ValueError("Invalid order fill")
                status = info.get("status")
                terminal = set(TERMINAL_STATUSES)
                active = {
                    "new",
                    "accepted",
                    "pending_new",
                    "partially_filled",
                    "pending_cancel",
                    "pending_replace",
                    "accepted_for_bidding",
                    "held",
                    "stopped",
                    "calculated",
                }
                if status == "filled" and qty == 0:
                    raise ValueError("Filled order has no fill")
                if status not in terminal | active:
                    raise ValueError("Unknown order status")
                self._ledger.attach_broker_order(client_id, oid)
                if status in terminal:
                    self._ledger.settle_order_intent(
                        client_id,
                        oid,
                        float(qty),
                        float(price) if price is not None else None,
                        status,
                    )
                else:
                    working.add(intent["symbol"])
            if working:
                raise RuntimeError("Unresolved working orders block submissions")
            return working
        except Exception as exc:
            try:
                self._notifier.send(
                    f"<b>🛑 Order recovery/pending orders failed:</b> {exc}. "
                    "Orders may already have been submitted; check their status."
                )
            except Exception:
                _log.exception("Could not send order recovery failure notice")
            raise

    def flag_if_stale(self, pending_order: dict, now: datetime) -> None:
        """Send a one-time "stale pending order" alert once an order has been
        open past `_STALE_PENDING_DAYS`. Idempotent via the `flagged_stale`
        column -- reconciliation keeps polling the order regardless."""
        if pending_order.get("flagged_stale"):
            return
        created_at = pending_order.get("created_at")
        if created_at is None:
            return
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        age_days = (now - created_at).total_seconds() / 86400
        if age_days < _STALE_PENDING_DAYS:
            return
        self._notifier.send(
            f"<b>⚠️ Stale pending order:</b> {pending_order['side']} "
            f"{pending_order['symbol']} (order <code>{pending_order['order_id']}</code>) "
            f"has been open {age_days:.1f} days without filling."
        )
        try:
            self._ledger.mark_pending_order_stale(pending_order["order_id"])
        except Exception as exc:
            _log.warning(
                "Could not mark pending order %s stale (non-fatal): %s",
                pending_order["order_id"],
                exc,
            )

    def reconcile_pending_orders(self) -> set[str]:
        """Settle orders that were still working at the end of a prior run.

        Orders queued after the close may fill at the next session, long after
        that run exited. Each run re-checks them: a completed fill is booked at
        its real executed value under the original order date; a terminally-failed order is
        dropped; anything still working is left for a later run.

        Returns the set of symbols that remain pending after this pass, so
        the caller can avoid submitting a duplicate/conflicting order for
        them this run.
        """
        still_pending: set[str] = set()
        try:
            pending = self._ledger.get_pending_orders()
        except Exception as exc:
            return self._pending_load_failed(exc)

        if self._read_only:
            # Booking a fill or dropping an order are writes: leave every pending order for the
            # next real run, and only keep its symbol out of today's intended orders.
            return {po["symbol"] for po in pending}

        now = self._hooks.utcnow()
        for po in pending:
            oid = po["order_id"]
            try:
                info = self._broker.get_order(oid)
            except Exception as exc:
                _log.warning("Reconcile: get_order(%s) failed: %s", oid, exc)
                raise

            status = info.get("status", "pending")
            filled_qty = info.get("filled_qty")
            filled_price = info.get("filled_avg_price")
            terminal = status in TERMINAL_STATUSES
            if not terminal:
                still_pending.add(po["symbol"])
                self._current().flag_if_stale(po, now)
                continue  # still working — try again next run

            try:
                self._ledger.settle_legacy_pending_order(oid, filled_qty, filled_price, status)
            except Exception as exc:
                self._notifier.send(
                    f"<b>🛑 Could not settle pending order {oid} — no orders placed:</b> {exc}"
                )
                raise
            if filled_qty > 0:
                amount = filled_qty * filled_price
                self._notifier.trade_alert(
                    po["side"],
                    po["symbol"],
                    amount,
                    oid,
                    qty=filled_qty,
                    price=filled_price,
                    status=f"{status} (reconciled)",
                )
        return still_pending

    def poll_orders(self, orders: list[SubmittedOrder]) -> dict[str, OrderStatus]:
        """Poll submitted orders until they reach a terminal status (or a
        wait budget expires), returning `{order_id: order_info}`.

        Shared by the main per-run order batch (strategy sells/buys, plus a
        cash-sweep sell if one was submitted) and the trailing cash-sweep buy,
        which is sized from the account's real post-trade cash and so must be
        submitted and polled separately, after the main batch resolves.
        """
        filled_orders: dict[str, OrderStatus] = {}
        if not orders:
            return filled_orders

        start_time = self._hooks.clock_time()
        is_open = True
        try:
            is_open = self._broker.get_clock()["is_open"]
        except Exception as exc:
            _log.debug("Broker clock lookup skipped: %s", type(exc).__name__)

        max_wait = 15.0 if is_open else 2.0
        remaining = {item.order_id: item for item in orders}

        while remaining and (self._hooks.clock_time() - start_time) < max_wait:
            for oid in list(remaining.keys()):
                try:
                    order_info = self._broker.get_order(oid)
                    status = order_info["status"]
                    # A partial fill is NOT terminal: keep polling so it can
                    # complete (or time out) rather than booking the partial.
                    if status in TERMINAL_STATUSES:
                        filled_orders[oid] = order_info
                        remaining.pop(oid)
                except Exception as e:
                    _log.warning("Error polling order %s: %s", oid, e)
            if remaining:
                self._hooks.sleep(1.0)

        # Query final statuses for remaining orders
        for oid, item in remaining.items():
            try:
                filled_orders[oid] = self._broker.get_order(oid)
            except Exception:
                filled_orders[oid] = {
                    "id": oid,
                    "symbol": item.symbol,
                    "side": item.side,
                    "qty": None,
                    "notional": item.amount if item.side == "BUY" else None,
                    "filled_qty": 0.0,
                    "filled_avg_price": 0.0,
                    "status": "pending",
                }
        return filled_orders


class DryRunExecutor(OrderExecutor):
    """Announce intentions without submitting orders."""

    submits_orders = False

    def _pending_load_failed(self, exc: Exception) -> set[str]:
        _log.warning("Could not load pending orders; dry-run intentions may conflict: %s", exc)
        return set()

    def recover_intents(self) -> None:
        """Do nothing: a dry run neither submits orders nor recovers intents."""
        pass

    def submit(self, intent: OrderIntent) -> None:
        """Submit nothing and return None."""
        return None

    def after_sells(self, filled: Mapping[str, OrderStatus]) -> None:
        """Do nothing."""
        pass

    def after_orders(self, orders: Sequence[SubmittedOrder]) -> None:
        """Do nothing."""
        pass

    def unfilled_exit_errors(
        self,
        orders: Sequence[SubmittedOrder],
        filled: Mapping[str, OrderStatus],
        broker_positions: Mapping[str, Position],
    ) -> list[str]:
        """Return no errors: a dry run submits no sells."""
        return []

    def announce(self, intent: OrderIntent) -> None:
        """Send a DRY RUN alert that describes the intent by kind.

        The kinds are stop, exit, sweep sell, buy and sweep buy.
        """
        symbol, size = intent.symbol, intent.size
        if intent.kind == IntentKind.STOP:
            message = (
                f"<b>🛑 DRY RUN catastrophe stop:</b> {symbol} (qty {size}, "
                f"unrealized {intent.unrealized_pct:.1%} "
                f"breached {intent.stop_threshold:.1%} floor)"
            )
        elif intent.kind == IntentKind.EXIT:
            message = f"<b>🔍 DRY RUN sell:</b> {symbol} (qty {size})"
        elif intent.kind == IntentKind.SWEEP_SELL:
            message = (
                f"<b>🔍 DRY RUN sweep sell:</b> {symbol} "
                f"(${intent.book_amount:.0f}, funding strategy buys)"
            )
        elif intent.kind == IntentKind.BUY:
            message = f"<b>🔍 DRY RUN buy:</b> {symbol} (${size:.0f}, sleeve={intent.sleeve})"
        else:
            message = f"<b>🔍 DRY RUN sweep buy:</b> {symbol} (${size:.0f})"
        self._notifier.send(message)
