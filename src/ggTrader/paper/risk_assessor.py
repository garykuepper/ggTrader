"""Assess risk for a paper-trading run.

Public names: ``RiskAssessor``.
"""

import logging
import math

from ggTrader.paper.constants import TRADER_LOGGER_NAME
from ggTrader.paper.ledger import PaperLedger
from ggTrader.paper.models import RiskVerdict, RunContext
from ggTrader.paper.portfolio import PortfolioView
from ggTrader.paper.ports import Notifier
from ggTrader.paper.risk import RiskGuard

_log = logging.getLogger(TRADER_LOGGER_NAME)


class RiskAssessor:
    """Assess risk limits."""

    def __init__(self, *, risk: RiskGuard, ledger: PaperLedger, notifier: Notifier) -> None:
        self.risk = risk
        self.ledger = ledger
        self.notifier = notifier

    def assess(self, ctx: RunContext, view: PortfolioView) -> RiskVerdict:
        """Assess a portfolio and send risk notices."""
        portfolio_value = view.portfolio_value
        account = view.account
        split_corrections = view.split_corrections
        # Seed the risk guard's peak from what was persisted by a prior
        # process (each cron run constructs a fresh PaperTrader, so without
        # this the halt below always compares against a same-run peak and
        # can never fire).
        try:
            persisted_peak = self.ledger.get_peak_value()
        except Exception as exc:
            _log.warning("Could not load persisted peak: %s", exc)
            reason = "Risk history unavailable: could not read persisted peak"
            self.notifier.send(f"<b>🛑 HALTED:</b> {reason}")
            return RiskVerdict(reason, ())
        if persisted_peak is not None:
            try:
                self.risk.update_peak(persisted_peak)
            except ValueError as exc:
                reason = f"Invalid stored risk peak: {exc}"
                self.notifier.send(f"<b>🛑 HALTED:</b> {reason}")
                return RiskVerdict(reason, ())

        # Check max drawdown halt BEFORE updating the peak with today's
        # value, so a real drawdown against the persisted peak is actually
        # measured (updating first would always show 0% drawdown).
        halted, halt_reason = self.risk.check_drawdown_halt(portfolio_value)
        loss_halts: list[str] = []
        if halted:
            if (
                portfolio_value is None
                or not math.isfinite(portfolio_value)
                or portfolio_value <= 0
            ):
                self.notifier.send(f"<b>🛑 HALTED:</b> {halt_reason}")
                return RiskVerdict(halt_reason, ())
            loss_halts.append(halt_reason)

        if not loss_halts:
            self.risk.update_peak(portfolio_value)
            try:
                if self.risk.peak_value is not None:
                    self.ledger.save_peak_value(self.risk.peak_value)
            except Exception as exc:
                _log.warning("Could not persist peak value: %s", exc)
                reason = "Risk history unavailable: could not persist drawdown peak"
                self.notifier.send(f"<b>🛑 HALTED:</b> {reason}")
                return RiskVerdict(reason, ())

        # Booked equity can understate a holding whose split the broker has not applied yet,
        # so a harmless split can look like a loss. That false halt is accepted on purpose: it
        # only blocks buys and sweeps for the day (exits stay allowed), while adding the
        # correction back could hide a real loss when the stored correction is stale.
        last_equity = account.get("last_equity")
        if last_equity is None:
            reason = "Invalid broker previous-close equity; daily loss cannot be assessed"
            self.notifier.send(f"<b>🛑 HALTED:</b> {reason}")
            return RiskVerdict(reason, ())
        daily_halted, daily_reason = self.risk.check_daily_loss(portfolio_value, last_equity)
        if daily_halted:
            if not math.isfinite(last_equity) or last_equity <= 0:
                self.notifier.send(f"<b>🛑 HALTED:</b> {daily_reason}")
                return RiskVerdict(daily_reason, ())
            if split_corrections:
                symbols = ", ".join(sorted(map(str, split_corrections)))
                daily_reason += (
                    f" (split correction on record for {symbols}: it may explain part of the "
                    "drop, but a real loss cannot be ruled out; buys stay blocked today)"
                )
            loss_halts.append(daily_reason)

        if loss_halts:
            try:
                self.notifier.send(f"<b>🛑 HALTED buys; exits only:</b> {'; '.join(loss_halts)}")
            except Exception:
                _log.exception("Could not send loss-halt notice")
            if ctx.pending_symbols:
                try:
                    self.notifier.send(
                        "<b>⚠️ Pending orders may still fill:</b> no cancellations were attempted."
                    )
                except Exception:
                    _log.exception("Could not send pending-order notice")

        return RiskVerdict(None, tuple(loss_halts))
