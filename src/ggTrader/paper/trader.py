"""Paper trading orchestrator: signals -> orders -> risk guardrails -> notifications."""

from __future__ import annotations

import logging
import time
from datetime import date, datetime, timezone
from uuid import uuid4

from ggTrader.paper import (
    cash_sweep,  # noqa: F401 - patch target
    catastrophe_stop,  # noqa: F401 - patch target
)
from ggTrader.paper.alpaca_broker import AlpacaBroker
from ggTrader.paper.config import TraderConfig
from ggTrader.paper.constants import ET as _ET
from ggTrader.paper.constants import MAX_SIGNAL_AGE_DAYS as _MAX_SIGNAL_AGE_DAYS  # noqa: F401 - compatibility alias
from ggTrader.paper.constants import PT as _PT
from ggTrader.paper.constants import SPLIT_LOOKBACK_DAYS as _SPLIT_LOOKBACK_DAYS  # noqa: F401 - compatibility alias
from ggTrader.paper.constants import STALE_PENDING_DAYS as _STALE_PENDING_DAYS  # noqa: F401 - compatibility alias
from ggTrader.paper.constants import TERMINAL_STATUSES  # noqa: F401 - compatibility alias
from ggTrader.paper.corporate_actions import CorporateActions, missed_exits
from ggTrader.paper.execution import DryRunExecutor, OrderExecutor
from ggTrader.paper.ledger import PaperLedger, ReadOnlyLedger
from ggTrader.paper.models import (
    OrderStatus,
    RunContext,
    RunResult,
    SubmittedOrder,
)
from ggTrader.paper.notifier import DryRunNotifier, TelegramNotifier
from ggTrader.paper.order_flow import OrderFlow
from ggTrader.paper.persist import PgPaperLedger
from ggTrader.paper.portfolio import PortfolioView
from ggTrader.paper.ports import Broker, Notifier
from ggTrader.paper.preflight import Halted, PreflightGate
from ggTrader.paper.reporter import RunReporter
from ggTrader.paper.risk import RiskConfig, RiskGuard, config_from_env
from ggTrader.paper.risk_assessor import RiskAssessor
from ggTrader.paper.run_helpers import account_read_halt as _account_read_halt
from ggTrader.paper.run_helpers import as_date as _as_date  # noqa: F401 - compatibility alias
from ggTrader.paper.seams import RunHooks, TraderSeams
from ggTrader.paper.signal_runner import (
    generate_core_signals,
    live_membership_source,
    refresh_benchmark_tape,
    require_live_membership_spec,
)
from ggTrader.paper.split_check import (
    find_split_applied_symbols,
)

_log = logging.getLogger(__name__)

# How far back to check held symbols for split events the broker's own
# corporate-actions feed may know about but the account never applied (see
# split_check.py / AlpacaBroker.get_split_corrections). Daily runs only need
# a window wide enough to survive a missed run or two.

# A pending order still open this many days after submission gets a one-time
# "stale" alert (see _flag_if_stale) instead of being silently re-polled
# forever with no visibility. Reconciliation itself keeps retrying past this
# point -- only the repeated Telegram noise is capped.


#: The S&P 500 list `ggt paper` reads: the latest database vintage, behind the freshness gate of
#: `ggTrader.data.membership_gate` (slice M4). If the database cannot be read, or the list is too
#: old, the run stops with no orders; it never falls back to the frozen file.
LIVE_MEMBERSHIP_SPEC = "db"

# Signals are built from the most recent *completed* daily bar, so on a healthy
# run `as_of` is the prior trading session, not today -- requiring today's date
# would block every single run. This cap exists only to catch a genuinely stale
# feed (e.g. a yfinance outage serving week-old bars). Four calendar days covers
# the worst normal case: a Friday close still being the latest completed bar on
# a Tuesday run after a Monday holiday.
#
# NOTE: `as_of` is used ONLY for this staleness check. `paper_trades.run_date` /
# `paper_snapshots.run_date` are stamped from `_today_et()` -- the ET calendar
# date of the run itself -- not from `as_of`. Persisting `as_of` conflated "the
# date the input data is dated" with "the date this run/order/snapshot
# happened", and compounded with the (now-fixed) OHLCV day-shift bug
# (cached_yfinance_loader.py) to land `run_date` a full session-plus-a-day
# early -- the weekday census of `paper_snapshots.run_date` was Sun-Thu with
# zero Fridays. See scripts/migrate_paper_dates_20260822.py for the historical
# row fix.


def _today_et() -> date:
    """Current calendar date in the exchange's timezone (America/New_York)."""
    return datetime.now(_ET).date()


def _now_pt() -> datetime:
    """Current time in Pacific, the zone of the 12:45 cron and the late-run cutoff."""
    return datetime.now(_PT)


class PaperTrader:
    """Orchestrates daily paper trading: generate signals, execute, notify."""

    def __init__(
        self,
        broker: Broker,
        notifier: Notifier,
        risk_cfg: RiskConfig | None = None,
        dry_run: bool = True,
        read_only: bool | None = None,
        membership: str | None = None,
        *,
        ledger: PaperLedger | None = None,
    ) -> None:
        """Wire the trader.

        Args:
            broker: order and account access.
            notifier: Telegram (or a stand-in).
            risk_cfg: risk limits (default: from the environment).
            dry_run: log intended orders instead of submitting them.
            read_only: write nothing: no DDL, no ledger, snapshot, peak, split or dividend row,
                no tape refresh and no reconcile booking. Defaults to ``dry_run``, so a dry run
                is read-only unless the caller opts out (``ggt paper --dry-run-writes``).
                Only a dry run may be read-only.
            membership: membership spec for the S&P 500 list (``fixture:<name>``, ``db``,
                ``db:<id>``); ``None`` defers to ``GGT_MEMBERSHIP`` and fails if that is unset.
                ``run_paper_trading`` always passes one.
            ledger: where durable state lives (default: the PostgreSQL-backed ledger).
                Injecting a ledger replaces only the database; the broker, notifier, signals and
                clock are still the caller's to provide. A read-only run wraps whichever ledger
                it is given.
        """
        require_live_membership_spec(membership, dry_run=dry_run)
        self._broker = broker
        self._notifier = notifier
        self._risk = RiskGuard(risk_cfg or config_from_env())
        self._dry_run = dry_run
        self._read_only = dry_run if read_only is None else read_only
        self._membership = membership
        if self._read_only and not dry_run:
            raise ValueError("read_only requires dry_run: a live run must book its trades")
        #: Names of the persistence calls a read-only run skipped (for the run record).
        self.skipped_writes: list[str] = []
        base = ledger if ledger is not None else PgPaperLedger()
        # One list: the ledger fills what the run record later returns.
        self._ledger = ReadOnlyLedger(base, self.skipped_writes, _log) if self._read_only else base
        self._paper_lock = None
        #: The settings snapshot of the current run; ``run()`` replaces it at its first line.
        self._cfg: TraderConfig | None = None

    def _hooks(self) -> RunHooks:
        """Resolve module collaborators at call time."""
        return RunHooks(
            today_et=lambda: _today_et(),
            now_pt=lambda: _now_pt(),
            utcnow=lambda: datetime.now(timezone.utc),
            new_uuid=lambda: uuid4(),
            clock_time=lambda: time.time(),
            sleep=lambda seconds: time.sleep(seconds),
            generate_signals=lambda *args, **kwargs: generate_core_signals(*args, **kwargs),
            membership_source=lambda spec: live_membership_source(spec),
            refresh_tape=lambda: refresh_benchmark_tape(),
            find_split_applied=lambda *args: find_split_applied_symbols(*args),
        )

    def _seams(self) -> TraderSeams:
        """Resolve private methods on this instance at call time."""
        return TraderSeams(
            submit_order=lambda *args: self._submit_order(*args),
            poll_orders=lambda orders: self._poll_orders(orders),
            missed_exits=lambda *args: self._missed_exits(*args),
            split_corrections=lambda *args: self._compute_split_corrections(*args),
            accrue_dividends=lambda positions: self._accrue_dividends(positions),
        )

    def _executor_for_call(self) -> OrderExecutor:
        """Build an executor from current collaborators."""
        cls = DryRunExecutor if self._dry_run else OrderExecutor
        return cls(
            broker=self._broker,
            notifier=self._notifier,
            ledger=self._ledger,
            hooks=self._hooks(),
            lock=lambda: self._paper_lock,
            read_only=self._read_only,
            seams=self._seams(),
        )

    def _submit_order(self, side: str, symbol: str, amount: float, reason: str | None) -> str:
        """Persist the identity before submitting an order under the run lock."""
        return self._executor_for_call().submit_order(side, symbol, amount, reason)

    def _missed_exits(
        self, positions: dict, last_exit: dict[str, str], already: set[str]
    ) -> list[str]:
        """Find held symbols with unacted-on exit signals."""
        return missed_exits(self._ledger, positions, last_exit, already)

    def _poll_orders(self, orders: list[SubmittedOrder]) -> dict[str, OrderStatus]:
        """Poll submitted orders until they reach a terminal status (or a
        wait budget expires), returning `{order_id: order_info}`.

        Shared by the main per-run order batch (strategy sells/buys, plus a
        cash-sweep sell if one was submitted) and the trailing cash-sweep buy,
        which is sized from the account's real post-trade cash and so must be
        submitted and polled separately, after the main batch resolves.
        """
        return self._executor_for_call().poll_orders(orders)

    def _compute_split_corrections(
        self, positions: dict[str, dict], split_since: date, today: date
    ) -> dict[str, float]:
        """Determine unapplied split corrections."""
        return CorporateActions(
            broker=self._broker, ledger=self._ledger, hooks=self._hooks()
        ).split_corrections(positions, split_since, today)

    def _accrue_dividends(self, positions: dict[str, dict]) -> dict:
        """Accrue uncredited dividends."""
        return CorporateActions(
            broker=self._broker, ledger=self._ledger, hooks=self._hooks()
        ).accrue_dividends(positions)

    def run(self) -> dict:
        """Hold the live lock; returned buy/sell lists are submitted or dry-run plans."""
        # One snapshot of the environment for the whole run, taken here (not at construction)
        # so a setting changed between runs applies to the next one.
        cfg = TraderConfig.from_env(dry_run=self._dry_run, read_only=self._read_only)
        self._cfg = cfg
        threshold = cfg.stop_threshold
        if cfg.stop_enabled and cfg.stop_threshold_error is not None:
            exc = cfg.stop_threshold_error
            try:
                self._notifier.send(f"<b>🛑 HALTED:</b> {exc}")
            except Exception:
                _log.exception("Could not send invalid catastrophe-stop threshold alert")
            raise exc
        if self._dry_run:
            return self._run(stop_threshold=threshold)
        with self._ledger.paper_run_lock() as lock:
            self._paper_lock = lock
            try:
                return self._run(stop_threshold=threshold)
            finally:
                self._paper_lock = None

    def _run(self, *, stop_threshold: float | None) -> dict:
        cfg = self._cfg
        if cfg is None:
            raise RuntimeError("_run needs the settings snapshot that run() takes first")
        try:
            self._ledger.init_paper_schema()
        except Exception as exc:
            _log.warning("DB schema init failed (non-fatal): %s", exc)
        if not self._dry_run:
            self._executor_for_call().reconcile_order_intents()
        pending_symbols = self._executor_for_call().reconcile_pending_orders()
        pre = PreflightGate(
            broker=self._broker, notifier=self._notifier, hooks=self._hooks(),
            membership=self._membership, read_only=self._read_only, dry_run=self._dry_run,
            skipped_writes=self.skipped_writes,
        ).check()  # fmt: skip
        if isinstance(pre, Halted):
            return pre.result.as_dict()
        ctx = RunContext(pre.today_et, cfg, frozenset(pending_symbols))
        view = PortfolioView.load(
            broker=self._broker, notifier=self._notifier, seams=self._seams(),
            ctx=ctx, signals=pre.signals,
        )  # fmt: skip
        verdict = RiskAssessor(
            risk=self._risk, ledger=self._ledger, notifier=self._notifier
        ).assess(ctx, view)
        if verdict.hard_stop is not None:
            return RunResult([], [], [verdict.hard_stop]).as_dict()
        reporter = RunReporter(
            broker=self._broker, notifier=self._notifier, ledger=self._ledger, risk=self._risk,
            skipped_writes=self.skipped_writes, dry_run=self._dry_run,
        )  # fmt: skip
        previous = reporter.previous_value(ctx.today_et)
        outcome = OrderFlow(
            executor=self._executor_for_call(), broker=self._broker, notifier=self._notifier,
            risk=self._risk, ctx=ctx, stop_threshold=stop_threshold,
        ).run(view, pre.signals, verdict)  # fmt: skip
        return reporter.finish(ctx, view, pre.signals, outcome, previous).as_dict()


def run_paper_trading(
    dry_run: bool = True,
    read_only: bool | None = None,
    notify: bool = False,
    *,
    membership: str | None = None,
) -> dict:
    """Convenience entry point: wire up broker + notifier and run.

    Before submitting real orders (dry_run=False), checks the account isn't
    margin-enabled, since the blend's target-vol overlay assumes
    max_leverage=1.0 (unlevered). Dry-run never submits real orders, so it
    is allowed regardless of account type -- gating it too would defeat its
    purpose as a safe burn-in/smoke-test mode.

    A dry run is read-only by default: it writes no ledger, snapshot, peak, split, dividend or
    tape row and sends nothing to Telegram unless ``notify`` (``ggt paper --notify``), which
    forwards only the free-form lines with a ``[DRY RUN]`` prefix. ``read_only=False`` restores
    the pre-rebuild dry run that persisted state (``--dry-run-writes``).

    ``membership`` is the spec of the S&P 500 list (the CLI passes ``LIVE_MEMBERSHIP_SPEC``);
    ``None`` defers to ``GGT_MEMBERSHIP`` and fails if that is unset, never to a file.
    """
    effective_read_only = dry_run if read_only is None else read_only
    if effective_read_only and not dry_run:
        raise ValueError("--dry-run-writes/read-only only applies to a dry run, not --live")
    require_live_membership_spec(membership, dry_run=dry_run)  # before any broker call
    broker = AlpacaBroker()
    notifier = TelegramNotifier()
    if effective_read_only:
        notifier = DryRunNotifier(notifier, forward=notify)
    try:
        account = broker.get_account()
    except Exception as exc:
        _account_read_halt(notifier, exc)
        raise
    if not dry_run and account["multiplier"] > 1.0:
        raise RuntimeError(
            f"Account multiplier is {account['multiplier']}x (margin-enabled); "
            "the blend overlay assumes an unlevered (1.0x) account."
        )
    trader = PaperTrader(
        broker, notifier, dry_run=dry_run, read_only=effective_read_only, membership=membership
    )
    return trader.run()
