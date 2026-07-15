"""Paper trading orchestrator: signals -> orders -> risk guardrails -> notifications."""

from __future__ import annotations

import logging
import time

from ggTrader.paper.alpaca_broker import AlpacaBroker
from ggTrader.paper.notifier import TelegramNotifier
from ggTrader.paper.persist import (
    clear_pending_order,
    get_latest_snapshot,
    get_pending_orders,
    init_paper_schema,
    log_pending_order,
    log_snapshot,
    log_trade,
)
from ggTrader.paper.risk import RiskConfig, RiskGuard
from ggTrader.paper.signal_runner import generate_blended_signals

_log = logging.getLogger(__name__)


class PaperTrader:
    """Orchestrates daily paper trading: generate signals, execute, notify."""

    def __init__(
        self,
        broker: AlpacaBroker,
        notifier: TelegramNotifier,
        risk_cfg: RiskConfig | None = None,
        dry_run: bool = True,
    ) -> None:
        self._broker = broker
        self._notifier = notifier
        self._risk = RiskGuard(risk_cfg)
        self._dry_run = dry_run

    def _reconcile_pending_orders(self) -> None:
        """Settle orders that were still working at the end of a prior run.

        Orders queued after the close (the 21:30 stocks cron submits ~30 min
        post-close) fill at the next session, long after that run exited. Each
        run re-checks them: a completed fill is booked at its real executed
        value under the original order date; a terminally-failed order is
        dropped; anything still working is left for a later run.
        """
        try:
            pending = get_pending_orders()
        except Exception as exc:
            _log.warning("Could not load pending orders (non-fatal): %s", exc)
            return

        for po in pending:
            oid = po["order_id"]
            try:
                info = self._broker.get_order(oid)
            except Exception as exc:
                _log.warning("Reconcile: get_order(%s) failed: %s", oid, exc)
                continue

            status = info.get("status", "pending")
            filled_qty = info.get("filled_qty") or 0.0
            filled_price = info.get("filled_avg_price") or 0.0
            filled = filled_qty > 0 and filled_price > 0
            terminal = status in ("filled", "canceled", "rejected", "expired")
            if not terminal:
                continue  # still working — try again next run

            if filled:
                amount = filled_qty * filled_price
                log_trade(po["run_date"], po["side"], po["symbol"], amount, oid)
                self._notifier.trade_alert(
                    po["side"],
                    po["symbol"],
                    amount,
                    oid,
                    qty=filled_qty,
                    price=filled_price,
                    status=f"{status} (reconciled)",
                )
            try:
                clear_pending_order(oid)
            except Exception as exc:
                _log.warning("Reconcile: clear_pending_order(%s) failed: %s", oid, exc)

    def run(self) -> dict:
        try:
            init_paper_schema()
        except Exception as exc:
            _log.warning("DB schema init failed (non-fatal): %s", exc)

        # Settle any orders that filled after a prior run exited.
        self._reconcile_pending_orders()

        try:
            blend = generate_blended_signals()
        except Exception as exc:
            self._notifier.send(f"Paper trading failed: signal generation error\n{exc}")
            raise

        if blend["fallback_used"]:
            self._notifier.send(
                "<b>⚠️ Overlay fallback:</b> rebalance-day data fetch failed; "
                "reusing last month's sleeve weights."
            )

        weights, scale = blend["weights"], blend["scale"]
        all_buys: list[tuple[str, str]] = []  # (symbol, sleeve)
        all_sells: list[str] = []
        gate_infos: dict[str, dict] = {}
        for universe, sleeve_signals in blend["sleeves"].items():
            for sym in sleeve_signals["buys"]:
                all_buys.append((sym, universe))
            all_sells.extend(sleeve_signals["sells"])
            gate_infos[universe] = sleeve_signals.get("gate", {})

        signals = {
            "buys": [sym for sym, _u in all_buys],
            "sells": sorted(set(all_sells)),
            "as_of": next(iter(blend["sleeves"].values()))["as_of"],
        }

        for universe, gate in gate_infos.items():
            if gate.get("gate_enabled") and gate.get("scores"):
                scores = gate["scores"]
                kept = gate.get("kept_buys", 0)
                raw = gate.get("raw_buys", 0)
                score_strs = [
                    f"{s}({v:.2f})" for s, v in sorted(scores.items(), key=lambda x: -x[1])
                ]
                self._notifier.send(
                    f"<b>🤖 ML Gate ({universe}):</b> kept {kept}/{raw} signals\n"
                    + ", ".join(score_strs)
                )
            elif gate.get("gate_enabled") is False:
                _log.info("ML gate disabled for %s (no model file)", universe)

        account = self._broker.get_account()
        positions = self._broker.get_positions()
        portfolio_value = account["portfolio_value"]

        # Track peak for drawdown calculation
        self._risk.update_peak(portfolio_value)

        # Check max drawdown halt
        halted, halt_reason = self._risk.check_drawdown_halt(portfolio_value)
        if halted:
            self._notifier.send(f"<b>🛑 HALTED:</b> {halt_reason}")
            return {"buys": [], "sells": [], "errors": [halt_reason]}

        # Check daily loss limit
        prev_snapshot_value = None
        try:
            prev_snapshot_value = get_latest_snapshot()
        except Exception:
            pass
        day_start = prev_snapshot_value or portfolio_value

        daily_halted, daily_reason = self._risk.check_daily_loss(portfolio_value, day_start)
        if daily_halted:
            self._notifier.send(f"<b>⚠️ Daily limit:</b> {daily_reason}")
            return {"buys": [], "sells": [], "errors": [daily_reason]}

        executed_sells: list[str] = []
        executed_buys: list[str] = []
        errors: list[str] = []
        pending_orders: list[tuple[str, str, str, float]] = []

        # Sells first — free up position slots
        for symbol in signals["sells"]:
            if symbol not in positions:
                continue
            qty = positions[symbol]["qty"]
            if self._dry_run:
                executed_sells.append(symbol)
                self._notifier.send(f"<b>🔍 DRY RUN sell:</b> {symbol} (qty {qty})")
                continue
            try:
                oid = self._broker.submit_sell(symbol, qty)
                executed_sells.append(symbol)
                pending_orders.append((oid, "SELL", symbol, positions[symbol]["market_value"]))
            except Exception as exc:
                errors.append(f"SELL {symbol}: {exc}")

        # Buys — respect global + per-sleeve position limits. Each position is
        # sized as a fixed fraction of its own sleeve's allocated capital
        # (weight * scale * portfolio_value), independent of how many signals
        # fire that day within the sleeve — sleeve_slot_caps governs sleeve
        # concurrency, sleeve_position_notional governs per-trade size.
        slot_caps = self._risk.sleeve_slot_caps(weights)
        sleeve_open_count = {u: 0 for u in weights}
        buys_by_sleeve: dict[str, list[str]] = {}
        for symbol, universe in all_buys:
            buys_by_sleeve.setdefault(universe, []).append(symbol)

        slots_available = self._risk.max_new_positions(len(positions) - len(executed_sells))
        buys_attempted = 0
        for universe, syms in buys_by_sleeve.items():
            sleeve_notional = self._risk.sleeve_position_notional(
                portfolio_value, weights.get(universe, 0.0), scale
            )
            sleeve_cap = slot_caps.get(universe, 0)
            for symbol in syms:
                if symbol in positions:
                    continue
                if buys_attempted >= slots_available:
                    _log.info(
                        "Max positions reached (%d), skipping %s",
                        self._risk.cfg.max_positions,
                        symbol,
                    )
                    break
                if sleeve_open_count[universe] >= sleeve_cap:
                    _log.info("Sleeve %s slot cap reached, skipping %s", universe, symbol)
                    break
                if self._risk.check_concentration(
                    symbol, positions, portfolio_value, prospective_notional=sleeve_notional
                ):
                    _log.info("Concentration limit for %s, skipping", symbol)
                    continue
                if self._dry_run:
                    executed_buys.append(symbol)
                    buys_attempted += 1
                    sleeve_open_count[universe] += 1
                    self._notifier.send(
                        f"<b>🔍 DRY RUN buy:</b> {symbol} "
                        f"(${sleeve_notional:.0f}, sleeve={universe})"
                    )
                    continue
                try:
                    oid = self._broker.submit_buy(symbol, sleeve_notional)
                    executed_buys.append(symbol)
                    buys_attempted += 1
                    sleeve_open_count[universe] += 1
                    pending_orders.append((oid, "BUY", symbol, sleeve_notional))
                except Exception as exc:
                    errors.append(f"BUY {symbol}: {exc}")

        # Poll submitted orders until they fill (or timeout)
        filled_orders: dict[str, dict] = {}
        if pending_orders:
            start_time = time.time()
            is_open = True
            try:
                is_open = self._broker.get_clock()["is_open"]
            except Exception:
                pass

            max_wait = 15.0 if is_open else 2.0
            remaining = {item[0]: item for item in pending_orders}

            while remaining and (time.time() - start_time) < max_wait:
                for oid in list(remaining.keys()):
                    try:
                        order_info = self._broker.get_order(oid)
                        status = order_info["status"]
                        # A partial fill is NOT terminal: keep polling so it can
                        # complete (or time out) rather than booking the partial.
                        if status in ("filled", "canceled", "rejected", "expired"):
                            filled_orders[oid] = order_info
                            remaining.pop(oid)
                    except Exception as e:
                        _log.warning("Error polling order %s: %s", oid, e)
                if remaining:
                    time.sleep(1.0)

            # Query final statuses for remaining orders
            for oid, item in remaining.items():
                try:
                    filled_orders[oid] = self._broker.get_order(oid)
                except Exception:
                    filled_orders[oid] = {
                        "id": oid,
                        "symbol": item[2],
                        "side": item[1],
                        "qty": None,
                        "notional": item[3] if item[1] == "BUY" else None,
                        "filled_qty": 0.0,
                        "filled_avg_price": 0.0,
                        "status": "pending",
                    }

        # Alert on every submitted order, but only book the trade ledger at a
        # real executed value — never the intended notional. Orders still
        # working at run end (queued after the close, or a partial fill not yet
        # complete) are persisted so the next run reconciles their final fill;
        # terminally-failed orders are simply dropped.
        for oid, side, symbol, amount in pending_orders:
            info = filled_orders.get(oid, {})
            status = info.get("status", "pending")
            filled_qty = info.get("filled_qty") or 0.0
            filled_price = info.get("filled_avg_price") or 0.0
            filled = filled_qty > 0 and filled_price > 0
            terminal = status in ("filled", "canceled", "rejected", "expired")

            trade_amount = filled_qty * filled_price if filled else amount
            self._notifier.trade_alert(
                side,
                symbol,
                trade_amount,
                oid,
                qty=filled_qty if filled else None,
                price=filled_price if filled else None,
                status=status,
            )
            if terminal:
                if filled:
                    log_trade(signals["as_of"], side, symbol, trade_amount, oid)
                else:
                    _log.info(
                        "Order %s (%s %s) terminal unfilled (status=%s); no ledger entry",
                        oid,
                        side,
                        symbol,
                        status,
                    )
            else:
                # accepted / new / pending / partially_filled — settle next run
                log_pending_order(signals["as_of"], side, symbol, amount, oid)
                _log.info(
                    "Order %s (%s %s) still working (status=%s); queued for reconciliation",
                    oid,
                    side,
                    symbol,
                    status,
                )

        new_account = self._broker.get_account()
        new_value = new_account["portfolio_value"]
        updated_positions = self._broker.get_positions()

        daily_pnl = new_value - day_start

        weight_str = ", ".join(f"{u}={w:.0%}" for u, w in weights.items())
        risk_line = (
            f"Positions: {len(updated_positions)}/{self._risk.cfg.max_positions} | "
            f"Scale: {scale:.2f}x | Weights: {weight_str}"
        )
        self._notifier.daily_summary(new_value, daily_pnl, updated_positions)
        self._notifier.send(f"<b>📊 Risk:</b> {risk_line}")

        try:
            log_snapshot(
                signals["as_of"],
                new_value,
                new_account["cash"],
                updated_positions,
            )
        except Exception as exc:
            _log.warning("DB snapshot failed (non-fatal): %s", exc)

        return {"buys": executed_buys, "sells": executed_sells, "errors": errors}


def run_paper_trading(dry_run: bool = True) -> dict:
    """Convenience entry point: wire up broker + notifier and run.

    Before submitting real orders (dry_run=False), checks the account isn't
    margin-enabled, since the blend's target-vol overlay assumes
    max_leverage=1.0 (unlevered). Dry-run never submits real orders, so it
    is allowed regardless of account type -- gating it too would defeat its
    purpose as a safe burn-in/smoke-test mode."""
    broker = AlpacaBroker()
    account = broker.get_account()
    if not dry_run and account["multiplier"] > 1.0:
        raise RuntimeError(
            f"Account multiplier is {account['multiplier']}x (margin-enabled); "
            "the blend overlay assumes an unlevered (1.0x) account."
        )
    notifier = TelegramNotifier()
    trader = PaperTrader(broker, notifier, dry_run=dry_run)
    return trader.run()
