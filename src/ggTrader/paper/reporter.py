"""Report paper-trading results and persist the daily snapshot."""

import logging
import math
from datetime import date

from ggTrader.paper.constants import TRADER_LOGGER_NAME
from ggTrader.paper.ledger import PaperLedger
from ggTrader.paper.models import RunContext, RunResult, StrategySignals
from ggTrader.paper.order_flow import FlowOutcome
from ggTrader.paper.portfolio import PortfolioView
from ggTrader.paper.ports import Broker, Notifier
from ggTrader.paper.risk import RiskGuard
from ggTrader.paper.split_check import apply_corrections_to_positions
from ggTrader.utils.json_safe import json_safe

_log = logging.getLogger(TRADER_LOGGER_NAME)


class RunReporter:
    """Report the run in the same order as execution."""

    def __init__(
        self,
        *,
        broker: Broker,
        notifier: Notifier,
        ledger: PaperLedger,
        risk: RiskGuard,
        skipped_writes: list[str],
        dry_run: bool,
    ) -> None:
        self.broker = broker
        self.notifier = notifier
        self.ledger = ledger
        self.risk = risk
        self.skipped_writes = skipped_writes
        self.dry_run = dry_run

    def previous_value(self, today_et: date) -> float | None:
        """Read and validate the previous reporting snapshot."""
        try:
            prev_snapshot_value = self.ledger.get_latest_snapshot(before=today_et)
        except Exception as exc:
            _log.warning("Could not load prior reporting snapshot: %s", exc)
            prev_snapshot_value = None
        if prev_snapshot_value is not None and (
            not math.isfinite(prev_snapshot_value) or prev_snapshot_value <= 0
        ):
            _log.warning("Invalid prior reporting snapshot; daily P&L is unavailable")
            prev_snapshot_value = None

        return prev_snapshot_value

    def finish(
        self,
        ctx: RunContext,
        view: PortfolioView,
        signals: StrategySignals,
        outcome: FlowOutcome,
        previous_value: float | None,
    ) -> RunResult:
        """Send alerts and summary, then attempt the snapshot."""
        today_et = ctx.today_et
        weights, scale = signals.weights, signals.scale
        split_corrections = dict(view.split_corrections)
        dividend_info = dict(view.dividends)
        untradable_seen = view.untradable_seen
        loss_halts = list(outcome.buy_halts)
        executed_sells, executed_buys = outcome.book.sells, outcome.book.buys
        errors, pending_orders = outcome.book.errors, outcome.book.orders
        filled_orders = outcome.book.filled
        prev_snapshot_value = previous_value

        # Alert on every submitted order, but only book the trade ledger at a
        # real executed value — never the intended notional. Orders still
        # working at run end (queued after the close, or a partial fill not yet
        # complete) are persisted so the next run reconciles their final fill;
        # terminally-failed orders are simply dropped.
        for order in pending_orders:
            oid, side, symbol, amount = order.order_id, order.side, order.symbol, order.amount
            info = filled_orders.get(oid, {})
            status = info.get("status", "pending")
            filled_qty = info.get("filled_qty") or 0.0
            filled_price = info.get("filled_avg_price") or 0.0
            filled = filled_qty > 0 and filled_price > 0
            trade_amount = filled_qty * filled_price if filled else amount
            self.notifier.trade_alert(
                side,
                symbol,
                trade_amount,
                oid,
                qty=filled_qty if filled else None,
                price=filled_price if filled else None,
                status=status,
            )
            # Leave durable intents for the next reconciliation pass; do not
            # create a second legacy representation or book a separate trade.

        if errors:
            self.notifier.send(f"<b>❌ {len(errors)} order(s) failed:</b>\n" + "\n".join(errors))

        try:
            new_account = self.broker.get_account()
            updated_positions = self.broker.get_positions()
        except Exception as exc:
            _log.error(
                "Post-trade account or positions read failed (%s); snapshot not saved",
                type(exc).__name__,
            )
            try:
                self.notifier.send(
                    "<b>⚠️ Account or positions data unavailable after trading:</b> "
                    + (
                        "dry run; no orders submitted"
                        if self.dry_run
                        else "orders may have been submitted"
                    )
                    + "; snapshot not saved."
                )
            except Exception:
                _log.exception("Could not send post-trade data failure notice")
            raise
        new_value = new_account["portfolio_value"]  # broker's own ("booked") figure, uncorrected

        # Re-apply the same split_corrections computed at the top of this
        # run (the correction factor doesn't change intra-run) to the
        # post-trade position snapshot, so every downstream reporting number
        # -- unrealized P&L, daily/total P&L, and the persisted snapshot --
        # reflects true economic value rather than the broker's uncorrected
        # figures. NOTE: `updated_positions` (uncorrected) is what must be
        # used for anything that actually transacts against the broker (see
        # submit_sell above, which already ran against the broker's real
        # qty) -- never replace that with the corrected numbers below.
        corrected_updated_positions = apply_corrections_to_positions(
            updated_positions, split_corrections
        )
        correction_delta = sum(
            corrected_updated_positions[sym]["market_value"]
            - updated_positions[sym]["market_value"]
            for sym in split_corrections
            if sym in updated_positions
        )
        # dividend_info["total"] is the cumulative (all-time) accrual, a
        # reporting-only correction for cash the broker never credited (see
        # dividend_check.py) -- it is NOT part of new_value/new_account["cash"]
        # and never will be, so it must be added back here for every equity
        # figure reported downstream.
        corrected_value = new_value + correction_delta + dividend_info["total"]
        if not math.isfinite(corrected_value) or corrected_value <= 0:
            corrected_value = None
            message = "Invalid post-trade portfolio value; snapshot not saved"
            _log.warning(message)
            errors.append(message)

        daily_pnl = (
            corrected_value - prev_snapshot_value
            if corrected_value is not None and prev_snapshot_value is not None
            else None
        )

        pnl_values = [
            json_safe(p.get("unrealized_pl")) for p in corrected_updated_positions.values()
        ]
        unrealized_pnl = (
            sum(pnl_values)
            if all(isinstance(p, (int, float)) and not isinstance(p, bool) for p in pnl_values)
            else None
        )
        if unrealized_pnl is not None and not math.isfinite(unrealized_pnl):
            unrealized_pnl = None
        try:
            starting_capital = self.ledger.get_earliest_snapshot()
        except Exception as exc:
            _log.warning("Starting snapshot read failed (%s)", type(exc).__name__)
            errors.append("Starting snapshot unavailable; total P&L unavailable")
            starting_capital = None
        total_pnl = (
            corrected_value - starting_capital
            if corrected_value is not None and starting_capital
            else None
        )

        weight_str = ", ".join(f"{u}={w:.0%}" for u, w in weights.items())
        risk_line = (
            f"Positions: {len(updated_positions)}/{self.risk.cfg.max_positions} | "
            f"Scale: {scale:.2f}x | Weights: {weight_str}"
        )
        self.notifier.daily_summary(
            corrected_value,
            daily_pnl,
            corrected_updated_positions,
            total_pnl=total_pnl,
            unrealized_pnl=unrealized_pnl,
            split_corrections=split_corrections,
            booked_equity=new_value if split_corrections else None,
            dividend_total=dividend_info["total"],
            new_dividend_accruals=dividend_info["new_accruals"],
            errors=loss_halts + errors,
            starting_capital=starting_capital,
        )
        self.notifier.send(f"<b>📊 Risk:</b> {risk_line}")

        try:
            # stamp the actual ET session date of this run, not the lagged
            # signals["as_of"] date.
            if corrected_value is not None:
                self.ledger.log_snapshot(
                    today_et.isoformat(),
                    corrected_value,
                    new_account["cash"],
                    corrected_updated_positions,
                )
        except Exception as exc:
            _log.warning("DB snapshot failed (non-fatal): %s", exc)
            message = "Daily snapshot could not be saved"
            errors.append(message)
            try:
                self.notifier.send(f"<b>⚠️ {message}.</b>")
            except Exception:
                _log.exception("Could not send snapshot failure notice")

        return RunResult(
            executed_buys,
            executed_sells,
            loss_halts + errors,
            untradable=sorted(untradable_seen),
            skipped_writes=list(self.skipped_writes),
        )
