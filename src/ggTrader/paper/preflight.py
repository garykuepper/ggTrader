"""Market, signal and freshness checks for a paper run."""

import logging
from dataclasses import dataclass
from datetime import date

from ggTrader.paper.constants import MAX_SIGNAL_AGE_DAYS, TRADER_LOGGER_NAME
from ggTrader.paper.gates import SignalGateError, late_run_reason
from ggTrader.paper.models import MarketClock, RunResult, SignalBundle, StrategySignals
from ggTrader.paper.planning import strategy_signals
from ggTrader.paper.ports import Broker, Notifier
from ggTrader.paper.seams import RunHooks

_log = logging.getLogger(TRADER_LOGGER_NAME)


@dataclass(frozen=True)
class Halted:
    """A preflight gate stopped the run."""

    reason: str
    result: RunResult


@dataclass(frozen=True)
class PreflightResult:
    """Signals and run date after all preflight checks pass."""

    bundle: SignalBundle
    signals: StrategySignals
    today_et: date


class PreflightGate:
    """Check the market, signals, run time and freshness in order."""

    def __init__(
        self,
        *,
        broker: Broker,
        notifier: Notifier,
        hooks: RunHooks,
        membership: str | None,
        read_only: bool,
        dry_run: bool,
        skipped_writes: list[str],
    ) -> None:
        self.broker = broker
        self.notifier = notifier
        self.hooks = hooks
        self.membership = membership
        self.read_only = read_only
        self.dry_run = dry_run
        self.skipped_writes = skipped_writes

    def check(self) -> PreflightResult | Halted:
        """Run the checks and return signals or the reason for stopping."""
        # Market-open gate: cron has no calendar awareness, so a holiday or
        # half-day would otherwise submit DAY orders into a closed market.
        # Fail open (assume the market is open) on a get_clock() error, same
        # non-fatal-degrade convention used elsewhere in this file.
        market_open = True
        next_close: str | None = None
        try:
            clock: MarketClock = self.broker.get_clock()
            market_open = clock.get("is_open", True)
            next_close = clock.get("next_close")
        except Exception as exc:
            _log.warning("get_clock() failed (non-fatal, assuming market open): %s", exc)
        if not market_open:
            _log.info("Market closed; skipping run")
            self.notifier.send("<b>ℹ️ Market closed</b> — no trading this run.")
            return Halted("market_closed", RunResult([], [], []))

        try:
            blend: SignalBundle = self.hooks.generate_signals(
                persist_tape=not self.read_only,
                membership=self.hooks.membership_source(self.membership),
            )
        except SignalGateError as exc:
            # Fail closed: a failed engine self-check or sanity gate means no orders at all.
            self.notifier.send(f"<b>🛑 Signal gate — no orders placed:</b>\n{exc}")
            raise
        except Exception as exc:
            self.notifier.send(f"Paper trading failed: signal generation error\n{exc}")
            raise

        for universe, sleeve in blend["sleeves"].items():
            _log.info(
                "signals %s: engine=%s bars=%s buys=%d sells=%d as_of=%s",
                universe,
                sleeve.get("engine"),
                sleeve.get("bars"),
                len(sleeve["buys"]),
                len(sleeve["sells"]),
                sleeve["as_of"],
            )
            membership_gate = sleeve.get("membership_gate") or {}
            if membership_gate.get("level") == "warn":
                self.notifier.send(f"<b>⚠️ Membership list:</b> {membership_gate['message']}")

        # Late-run gate: a DAY order sent after the close queues to the next open, the
        # overnight-gap slippage the 12:45 schedule exists to avoid. A dry run only warns, so
        # the owner can rehearse the path at any hour.
        late = late_run_reason(self.hooks.now_pt(), next_close)
        if late is not None:
            if not self.dry_run:
                self.notifier.send(f"<b>⏰ Late run — no orders placed:</b> {late}")
                return Halted("late_run", RunResult([], [], [], gate="late_run"))
            _log.warning("dry run past the late-run cutoff (a live run would stop): %s", late)
            self.notifier.send(f"<b>⏰ DRY RUN late-run notice:</b> {late}")

        # Side job: keep SPY and the macro ETFs' tape current for the lab (an `ohlcv`
        # write, so a read-only run skips it). Wrapped inside the function; a failure here
        # logs and moves on.
        if self.read_only:
            self.skipped_writes.append("refresh_benchmark_tape")
            _log.info("read-only: skipped write refresh_benchmark_tape")
        else:
            refreshed = self.hooks.refresh_tape()
            if not refreshed:
                _log.warning("benchmark tape refresh returned no symbols")

        if blend["fallback_used"]:
            self.notifier.send(
                "<b>⚠️ Overlay fallback:</b> rebalance-day data fetch failed; "
                "reusing last month's sleeve weights."
            )

        signals = strategy_signals(blend)

        # Freshness gate: refuse to trade on data that is stale beyond the
        # normal one-completed-session lag (see MAX_SIGNAL_AGE_DAYS).
        today_et = self.hooks.today_et()
        for universe, sleeve in blend["sleeves"].items():
            as_of = sleeve.get("as_of")
            try:
                as_of_date = date.fromisoformat(as_of)
            except (TypeError, ValueError):
                as_of_date = None
            age_days = None if as_of_date is None else (today_et - as_of_date).days
            if age_days is None or not 0 <= age_days <= MAX_SIGNAL_AGE_DAYS:
                if age_days is None:
                    age_description = "unparseable"
                elif age_days < 0:
                    age_description = f"{-age_days}d in the future"
                else:
                    age_description = f"{age_days}d old"
                self.notifier.send(
                    f"<b>⚠️ Stale signal data:</b> {universe} as_of={as_of} is "
                    f"{age_description} "
                    f"(today {today_et.isoformat()}, max {MAX_SIGNAL_AGE_DAYS}d); "
                    f"no trades submitted this run."
                )
                return Halted("stale_signals", RunResult([], [], []))

        return PreflightResult(blend, signals, today_et)
