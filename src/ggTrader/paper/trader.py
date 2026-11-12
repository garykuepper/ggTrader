"""Paper trading orchestrator: signals -> orders -> risk guardrails -> notifications."""

from __future__ import annotations

import logging
import math
import time
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from uuid import uuid4
from zoneinfo import ZoneInfo

from ggTrader.paper import cash_sweep, catastrophe_stop
from ggTrader.paper.alpaca_broker import AlpacaBroker
from ggTrader.paper.dividend_check import DIVIDEND_BACKFILL_START, compute_dividend_accruals
from ggTrader.paper.gates import SignalGateError, late_run_reason
from ggTrader.paper.notifier import DryRunNotifier, TelegramNotifier
from ggTrader.paper.persist import (
    attach_broker_order,
    clear_pending_order,  # noqa: F401 - legacy test compatibility
    clear_split_correction,
    create_order_intent,
    get_accrued_dividend_keys,
    get_earliest_snapshot,
    get_last_buy_dates,
    get_latest_snapshot,
    get_open_split_states,
    get_peak_value,
    get_pending_orders,
    get_snapshot_history,
    get_total_dividend_accrual,
    get_trade_history_dates,
    get_unresolved_order_intents,
    init_paper_schema,
    log_dividend_accrual,
    log_pending_order,  # noqa: F401 - compatibility for existing test patches
    log_snapshot,
    log_trade,  # noqa: F401 - legacy test compatibility
    mark_pending_order_stale,
    paper_run_lock,
    save_peak_value,
    save_split_correction,
    settle_legacy_pending_order,
    settle_order_intent,
)
from ggTrader.paper.risk import RiskConfig, RiskGuard, config_from_env
from ggTrader.paper.signal_runner import (
    generate_core_signals,
    live_membership_source,
    refresh_benchmark_tape,
    require_live_membership_spec,
)
from ggTrader.paper.split_check import (
    apply_corrections_to_positions,
    compute_split_corrections,
    find_split_applied_symbols,
)
from ggTrader.paper.untradable import (
    configured_untradable,
    is_untradable_error,
    resolve_untradable,
)
from ggTrader.utils.json_safe import json_safe

_log = logging.getLogger(__name__)

# How far back to check held symbols for split events the broker's own
# corporate-actions feed may know about but the account never applied (see
# split_check.py / AlpacaBroker.get_split_corrections). Daily runs only need
# a window wide enough to survive a missed run or two.
_SPLIT_LOOKBACK_DAYS = 14

# A pending order still open this many days after submission gets a one-time
# "stale" alert (see _flag_if_stale) instead of being silently re-polled
# forever with no visibility. Reconciliation itself keeps retrying past this
# point -- only the repeated Telegram noise is capped.
_STALE_PENDING_DAYS = 5

_ET = ZoneInfo("America/New_York")
_PT = ZoneInfo("America/Los_Angeles")

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
_MAX_SIGNAL_AGE_DAYS = 4


def _today_et() -> date:
    """Current calendar date in the exchange's timezone (America/New_York)."""
    return datetime.now(_ET).date()


def _now_pt() -> datetime:
    """Current time in Pacific, the zone of the 12:45 cron and the late-run cutoff."""
    return datetime.now(_PT)


def _as_date(value) -> date:
    """`paper_trades.run_date` as a `date` (drivers may return a datetime)."""
    return value.date() if isinstance(value, datetime) else value


def _account_read_halt(notifier, exc: Exception) -> None:
    """Report an unavailable account before any orders are submitted."""
    _log.error("Account read failed (%s); stopping before orders", type(exc).__name__)
    reason = "Account data unavailable; no orders submitted"
    try:
        notifier.send(f"<b>🛑 HALTED:</b> {reason}")
    except Exception:
        _log.exception("Could not send account read failure notice")


class PaperTrader:
    """Orchestrates daily paper trading: generate signals, execute, notify."""

    def __init__(
        self,
        broker: AlpacaBroker,
        notifier: TelegramNotifier,
        risk_cfg: RiskConfig | None = None,
        dry_run: bool = True,
        read_only: bool | None = None,
        membership: str | None = None,
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
        self._paper_lock = None

    def _write(self, fn, *args, **kwargs):
        """Call a persistence function, or record and skip it in read-only mode."""
        if self._read_only:
            # `_mock_name`: a patched persist function (tests) has no `__name__`.
            name = getattr(fn, "__name__", None) or getattr(fn, "_mock_name", None) or repr(fn)
            self.skipped_writes.append(name)
            _log.info("read-only: skipped write %s", name)
            return None
        return fn(*args, **kwargs)

    def _submit_order(self, side: str, symbol: str, amount: float, reason: str | None) -> str:
        """Persist the identity before submitting an order under the run lock."""
        if side not in ("BUY", "SELL"):
            raise ValueError("Invalid order side")
        client_id = str(uuid4())
        submit = self._broker.submit_buy if side == "BUY" else self._broker.submit_sell
        stage = "intent"
        try:
            create_order_intent(
                client_id,
                _today_et().isoformat(),
                side,
                symbol,
                amount,
                "notional" if side == "BUY" else "qty",
                reason,
            )
            stage = "lock"
            self._paper_lock.verify()
            stage = "submit"
            order_id = submit(symbol, amount, client_id)
            stage = "attach"
            attach_broker_order(client_id, order_id)
        except Exception:
            try:
                self._notifier.send(f"Order {client_id} failed at {stage}; investigation required")
            except Exception:
                _log.exception("Failed to notify about order %s at %s", client_id, stage)
            raise
        return order_id

    def _reconcile_order_intents(self) -> set[str]:
        """Resolve terminal intents and block uncertain or working orders."""
        try:
            intents = get_unresolved_order_intents()
            legacy_ids = {p["order_id"] for p in get_pending_orders()}
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
                terminal = {"filled", "canceled", "rejected", "expired"}
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
                attach_broker_order(client_id, oid)
                if status in terminal:
                    settle_order_intent(
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

    def _missed_exits(
        self, positions: dict, last_exit: dict[str, str], already: set[str]
    ) -> list[str]:
        """Held symbols whose most recent exit bar falls after their entry.

        Exits are crossover events and `signals["sells"]` sees only today's
        partial bar, so a cross completing after the 12:45 PT run, or on a
        day the run was skipped, was never acted on -- 6 of 26 names were
        still held on 2026-09-28 past a close-based RSI exit. The backtest
        closes on the first exit bar after entry; this restores that. An
        exit on the entry bar itself doesn't count (vbt won't exit the bar
        it entered). A symbol with no BUY record, or a lookup failure, falls
        back to last-bar-only behavior.
        """
        candidates = [s for s in positions if s in last_exit and s not in already]
        if not candidates:
            return []
        try:
            entries = get_last_buy_dates()
        except Exception as exc:
            _log.warning("last-buy lookup failed, skipping exit catch-up: %s", exc)
            return []
        return sorted(
            s
            for s in candidates
            if s in entries and date.fromisoformat(last_exit[s]) > _as_date(entries[s])
        )

    def _flag_if_stale(self, pending_order: dict, now: datetime) -> None:
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
            mark_pending_order_stale(pending_order["order_id"])
        except Exception as exc:
            _log.warning(
                "Could not mark pending order %s stale (non-fatal): %s",
                pending_order["order_id"],
                exc,
            )

    def _reconcile_pending_orders(self) -> set[str]:
        """Settle orders that were still working at the end of a prior run.

        Orders queued after the close (the 21:30 stocks cron submits ~30 min
        post-close) fill at the next session, long after that run exited. Each
        run re-checks them: a completed fill is booked at its real executed
        value under the original order date; a terminally-failed order is
        dropped; anything still working is left for a later run.

        Returns the set of symbols that remain pending after this pass, so
        the caller can avoid submitting a duplicate/conflicting order for
        them this run.
        """
        still_pending: set[str] = set()
        try:
            pending = get_pending_orders()
        except Exception as exc:
            if not self._dry_run:
                self._notifier.send(
                    f"<b>🛑 Could not load pending orders — no orders placed:</b> {exc}"
                )
                raise
            _log.warning("Could not load pending orders; dry-run intentions may conflict: %s", exc)
            return still_pending

        if self._read_only:
            # Booking a fill or dropping an order are writes: leave every pending order for the
            # next real run, and only keep its symbol out of today's intended orders.
            return {po["symbol"] for po in pending}

        now = datetime.now(timezone.utc)
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
            terminal = status in ("filled", "canceled", "rejected", "expired")
            if not terminal:
                still_pending.add(po["symbol"])
                self._flag_if_stale(po, now)
                continue  # still working — try again next run

            try:
                settle_legacy_pending_order(oid, filled_qty, filled_price, status)
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

    def _poll_orders(
        self, orders: list[tuple[str, str, str, float, str | None]]
    ) -> dict[str, dict]:
        """Poll submitted orders until they reach a terminal status (or a
        wait budget expires), returning `{order_id: order_info}`.

        Shared by the main per-run order batch (strategy sells/buys, plus a
        cash-sweep sell if one was submitted) and the trailing cash-sweep buy,
        which is sized from the account's real post-trade cash and so must be
        submitted and polled separately, after the main batch resolves.
        """
        filled_orders: dict[str, dict] = {}
        if not orders:
            return filled_orders

        start_time = time.time()
        is_open = True
        try:
            is_open = self._broker.get_clock()["is_open"]
        except Exception:
            pass

        max_wait = 15.0 if is_open else 2.0
        remaining = {item[0]: item for item in orders}

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
        return filled_orders

    def _compute_split_corrections(
        self, positions: dict[str, dict], split_since: date, today: date
    ) -> dict[str, float]:
        """Determine unapplied-split corrections for currently held
        `positions`, using `paper_snapshots`/`paper_trades` history as the
        primary evidence and the account's own SPLIT activities as a
        secondary, fallback signal -- see `split_check.py`'s module
        docstring for why the activities feed alone is no longer trusted
        (it is a no-op on this paper account, so a real applied split would
        otherwise be double-corrected).

        Merges in `paper_split_state` (see `persist.get_open_split_states`)
        so a split detected on a prior run keeps being corrected even once
        its ex_date falls outside `split_since` -- the rolling lookback
        window only bounds *new* detection from the broker's feed, not how
        long an already-known correction is honored (the MNST incident:
        `_SPLIT_LOOKBACK_DAYS` expired 2026-08-25 while still held, see
        `docs/next_steps.md`).

        A persisted, currently-held symbol keeps being re-verified against
        snapshot evidence on every run even once the broker's live feed
        stops mentioning it (its ex_date ages out of the feed's own
        lookback window): a synthetic `corp_splits`-shaped entry is built
        from the persisted `(ex_date, factor)` and merged into the set
        `find_split_applied_symbols` checks. Without this, a persisted
        correction could never be confirmed applied or cleared once the
        broker feed moved on -- if Alpaca ever did belatedly apply the
        split, the correction would double-count it forever (see this
        review's Finding 1). Persisted state is cleared once snapshot
        evidence confirms the broker applied the split, or the position is
        no longer held.

        Returns `{symbol: correction_factor}`, same shape as
        `AlpacaBroker.get_split_corrections` (see
        `split_check.apply_corrections_to_positions`). Fails soft at every
        step -- any error degrades toward the broker's corporate-actions +
        activities view alone, never aborts the run.
        """
        held_symbols = set(positions)

        try:
            persisted_states = get_open_split_states()
        except Exception as exc:
            _log.warning("Could not load persisted split state (non-fatal): %s", exc)
            persisted_states = {}

        # A symbol no longer held has nothing left to correct -- drop its
        # persisted row so the table doesn't grow unboundedly.
        for symbol in list(persisted_states):
            if symbol not in held_symbols:
                try:
                    self._write(clear_split_correction, symbol)
                except Exception as exc:
                    _log.warning(
                        "Could not clear stale split state for %s (non-fatal): %s", symbol, exc
                    )
                persisted_states.pop(symbol)

        persisted = {symbol: info["factor"] for symbol, info in persisted_states.items()}

        evidence = self._broker.get_split_evidence(list(positions), split_since)
        corp_splits = dict(evidence.get("corp_splits", {}))
        activity_applied = evidence.get("activity_applied", set())

        # Re-verify every persisted, still-held symbol every run, even ones
        # the broker's live feed no longer reports -- see the docstring.
        for symbol, info in persisted_states.items():
            if symbol not in corp_splits:
                corp_splits[symbol] = [(info["ex_date"], info["factor"])]

        if not corp_splits:
            return persisted

        try:
            snapshot_history = get_snapshot_history()
        except Exception as exc:
            _log.warning("Could not load snapshot history for split check (non-fatal): %s", exc)
            snapshot_history = []

        trade_dates_by_symbol: dict[str, list] = {}
        for symbol in corp_splits:
            try:
                trade_dates_by_symbol[symbol] = get_trade_history_dates(symbol)
            except Exception as exc:
                _log.warning(
                    "Could not load trade history for %s split check (non-fatal): %s",
                    symbol,
                    exc,
                )
                trade_dates_by_symbol[symbol] = []

        snapshot_applied, unresolved = find_split_applied_symbols(
            corp_splits, snapshot_history, trade_dates_by_symbol, today
        )
        if unresolved:
            _log.warning(
                "No snapshot evidence to confirm split status for %s -- falling back to "
                "the activities feed / correcting by default",
                sorted(unresolved),
            )

        applied_symbols = snapshot_applied | set(activity_applied)
        for symbol in applied_symbols:
            if symbol in persisted:
                try:
                    self._write(clear_split_correction, symbol)
                except Exception as exc:
                    _log.warning(
                        "Could not clear confirmed-applied split state for %s (non-fatal): %s",
                        symbol,
                        exc,
                    )
                persisted.pop(symbol, None)

        new_corrections = compute_split_corrections(corp_splits, applied_symbols)
        for symbol, factor in new_corrections.items():
            ex_date = min(ex for ex, _ in corp_splits[symbol])
            try:
                self._write(save_split_correction, symbol, str(ex_date), factor)
            except Exception as exc:
                _log.warning("Could not persist split state for %s (non-fatal): %s", symbol, exc)

        return {**persisted, **new_corrections}

    def _accrue_dividends(self, positions: dict[str, dict]) -> dict:
        """Credit cash dividends Alpaca's corporate-actions feed knows about
        but this account's own DIV activity log never booked (see
        dividend_check.py's module docstring for the incident). Reporting
        only -- never touches buying power, position sizing, or cash; see
        `AlpacaBroker.get_dividend_corrections` and `log_dividend_accrual`.

        Backfills from `DIVIDEND_BACKFILL_START` every run (idempotent via
        `paper_dividend_accruals`'s `(symbol, ex_date)` primary key, so
        re-checking the same window is always safe and cheap) rather than a
        rolling lookback, so a dividend missed weeks ago still gets caught.

        Fails soft at every step -- any error degrades to "nothing new
        accrued this run" (the already-persisted cumulative total is still
        returned where available) and never aborts the run.

        Returns `{"total": cumulative_$, "new_accruals": [...], "skipped": [...]}`.
        """
        result: dict = {"total": 0.0, "new_accruals": [], "skipped": []}
        try:
            result["total"] = get_total_dividend_accrual()
        except Exception as exc:
            _log.warning("Could not load dividend accrual total (non-fatal): %s", exc)
            return result

        try:
            snapshot_history = get_snapshot_history()
        except Exception as exc:
            _log.warning(
                "Could not load snapshot history for dividend accrual (non-fatal): %s", exc
            )
            return result

        # Symbols the account has ever held (not just today's), since a
        # dividend can be owed on a position that was later sold.
        all_symbols = set(positions) | {s for _, pos in snapshot_history for s in pos}
        if not all_symbols:
            return result

        div_data = self._broker.get_dividend_corrections(
            sorted(all_symbols), DIVIDEND_BACKFILL_START
        )
        corp_dividends = div_data.get("corp_dividends", {})
        if not corp_dividends:
            return result

        try:
            already_accrued_keys = get_accrued_dividend_keys()
        except Exception as exc:
            _log.warning("Could not load accrued dividend keys (non-fatal): %s", exc)
            return result

        accruals, skipped = compute_dividend_accruals(
            corp_dividends,
            div_data.get("credited_keys", set()),
            already_accrued_keys,
            snapshot_history,
        )

        for acc in accruals:
            try:
                self._write(
                    log_dividend_accrual,
                    acc["symbol"],
                    acc["ex_date"],
                    acc["rate"],
                    acc["qty"],
                    acc["amount"],
                )
            except Exception as exc:
                _log.warning(
                    "Could not persist dividend accrual for %s/%s (non-fatal): %s",
                    acc["symbol"],
                    acc["ex_date"],
                    exc,
                )

        if accruals:
            _log.info(
                "Accrued %d dividend(s) totaling $%.2f: %s",
                len(accruals),
                sum(a["amount"] for a in accruals),
                [(a["symbol"], str(a["ex_date"])) for a in accruals],
            )
        if skipped:
            _log.info(
                "Skipped %d dividend event(s), not attributed: %s",
                len(skipped),
                [(s["symbol"], str(s["ex_date"]), s["reason"]) for s in skipped],
            )

        try:
            result["total"] = get_total_dividend_accrual()
        except Exception as exc:
            _log.warning("Could not reload dividend accrual total (non-fatal): %s", exc)
        result["new_accruals"] = accruals
        result["skipped"] = skipped
        return result

    @staticmethod
    def _order_failed(
        errors: list[str], untradable_hit: set[str], label: str, symbol: str, exc: Exception
    ) -> None:
        """Record a rejected order: an inactive/untradable asset is skipped and alerted once
        per run (OD17); any other failure is a real error."""
        if is_untradable_error(exc):
            untradable_hit.add(symbol)
            _log.warning("%s %s skipped, broker will not trade the asset: %s", label, symbol, exc)
            return
        errors.append(f"{label} {symbol}: {exc}")

    def run(self) -> dict:
        """Hold the live lock; returned buy/sell lists are submitted or dry-run plans."""
        threshold = None
        if catastrophe_stop.catastrophe_stop_enabled():
            try:
                threshold = catastrophe_stop.catastrophe_stop_pct()
            except ValueError as exc:
                try:
                    self._notifier.send(f"<b>🛑 HALTED:</b> {exc}")
                except Exception:
                    _log.exception("Could not send invalid catastrophe-stop threshold alert")
                raise
        if self._dry_run:
            return self._run(stop_threshold=threshold)
        with paper_run_lock() as lock:
            self._paper_lock = lock
            try:
                return self._run(stop_threshold=threshold)
            finally:
                self._paper_lock = None

    def _run(self, *, stop_threshold: float | None) -> dict:
        try:
            self._write(init_paper_schema)
        except Exception as exc:
            _log.warning("DB schema init failed (non-fatal): %s", exc)

        # Settle any orders that filled after a prior run exited, and note
        # which symbols still have an order open so we don't double up on
        # them below.
        if not self._dry_run:
            self._reconcile_order_intents()
        pending_symbols = self._reconcile_pending_orders()

        # Market-open gate: cron has no calendar awareness, so a holiday or
        # half-day would otherwise submit DAY orders into a closed market.
        # Fail open (assume the market is open) on a get_clock() error, same
        # non-fatal-degrade convention used elsewhere in this file.
        market_open = True
        next_close: str | None = None
        try:
            clock = self._broker.get_clock()
            market_open = clock.get("is_open", True)
            next_close = clock.get("next_close")
        except Exception as exc:
            _log.warning("get_clock() failed (non-fatal, assuming market open): %s", exc)
        if not market_open:
            _log.info("Market closed; skipping run")
            self._notifier.send("<b>ℹ️ Market closed</b> — no trading this run.")
            return {"buys": [], "sells": [], "errors": []}

        try:
            blend = generate_core_signals(
                persist_tape=not self._read_only,
                membership=live_membership_source(self._membership),
            )
        except SignalGateError as exc:
            # Fail closed: a failed engine self-check or sanity gate means no orders at all.
            self._notifier.send(f"<b>🛑 Signal gate — no orders placed:</b>\n{exc}")
            raise
        except Exception as exc:
            self._notifier.send(f"Paper trading failed: signal generation error\n{exc}")
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
                self._notifier.send(f"<b>⚠️ Membership list:</b> {membership_gate['message']}")

        # Late-run gate: a DAY order sent after the close queues to the next open, the
        # overnight-gap slippage the 12:45 schedule exists to avoid. A dry run only warns, so
        # the owner can rehearse the path at any hour.
        late = late_run_reason(_now_pt(), next_close)
        if late is not None:
            if not self._dry_run:
                self._notifier.send(f"<b>⏰ Late run — no orders placed:</b> {late}")
                return {"buys": [], "sells": [], "errors": [], "gate": "late_run"}
            _log.warning("dry run past the late-run cutoff (a live run would stop): %s", late)
            self._notifier.send(f"<b>⏰ DRY RUN late-run notice:</b> {late}")

        # Side job: keep SPY and the macro ETFs' tape current for the lab (an `ohlcv`
        # write, so a read-only run skips it). Wrapped inside the function; a failure here
        # logs and moves on.
        if self._read_only:
            self.skipped_writes.append("refresh_benchmark_tape")
            _log.info("read-only: skipped write refresh_benchmark_tape")
        else:
            refreshed = refresh_benchmark_tape()
            if not refreshed:
                _log.warning("benchmark tape refresh returned no symbols")

        if blend["fallback_used"]:
            self._notifier.send(
                "<b>⚠️ Overlay fallback:</b> rebalance-day data fetch failed; "
                "reusing last month's sleeve weights."
            )

        weights, scale = blend["weights"], blend["scale"]
        all_buys: list[tuple[str, str]] = []  # (symbol, sleeve)
        all_sells: list[str] = []
        last_exit: dict[str, str] = {}
        for universe, sleeve_signals in blend["sleeves"].items():
            for sym in sleeve_signals["buys"]:
                all_buys.append((sym, universe))
            all_sells.extend(sleeve_signals["sells"])
            last_exit.update(sleeve_signals.get("last_exit", {}))

        signals = {
            "buys": [sym for sym, _u in all_buys],
            "sells": sorted(set(all_sells)),
            "as_of": next(iter(blend["sleeves"].values()))["as_of"],
        }

        # Freshness gate: refuse to trade on data that is stale beyond the
        # normal one-completed-session lag (see _MAX_SIGNAL_AGE_DAYS).
        today_et = _today_et()
        for universe, sleeve in blend["sleeves"].items():
            as_of = sleeve.get("as_of")
            try:
                as_of_date = date.fromisoformat(as_of)
            except (TypeError, ValueError):
                as_of_date = None
            age_days = None if as_of_date is None else (today_et - as_of_date).days
            if age_days is None or not 0 <= age_days <= _MAX_SIGNAL_AGE_DAYS:
                if age_days is None:
                    age_description = "unparseable"
                elif age_days < 0:
                    age_description = f"{-age_days}d in the future"
                else:
                    age_description = f"{age_days}d old"
                self._notifier.send(
                    f"<b>⚠️ Stale signal data:</b> {universe} as_of={as_of} is "
                    f"{age_description} "
                    f"(today {today_et.isoformat()}, max {_MAX_SIGNAL_AGE_DAYS}d); "
                    f"no trades submitted this run."
                )
                return {"buys": [], "sells": [], "errors": []}

        try:
            account = self._broker.get_account()
        except Exception as exc:
            _account_read_halt(self._notifier, exc)
            raise
        try:
            positions = self._broker.get_positions()
        except Exception as exc:
            _log.error("Position read failed (%s); stopping before orders", type(exc).__name__)
            try:
                self._notifier.send(
                    "<b>🛑 HALTED:</b> Position data unavailable; no orders submitted"
                )
            except Exception:
                _log.exception("Could not send position read failure notice")
            raise
        portfolio_value = account["portfolio_value"]

        # Cash sweep (feature-flagged, default OFF -- see cash_sweep.py). The
        # sweep ETF position (if any) must be invisible to the strategy: not a
        # strategy position, not counted against RiskGuard slot caps, and not
        # sellable by strategy exit signals. `strategy_positions` is what all
        # downstream slot/count/sell logic uses instead of the raw `positions`
        # from the broker; when the flag is off this is just `positions`
        # unchanged, so behavior is identical to before this feature existed.
        sweep_on = cash_sweep.sweep_enabled()
        sweep_sym = cash_sweep.sweep_symbol() if sweep_on else None
        strategy_positions = (
            {sym: pos for sym, pos in positions.items() if sym != sweep_sym}
            if sweep_on
            else positions
        )

        # Holdings the broker won't trade (merger/delisting, e.g. AVB -> EQR)
        # can't be sold, so they must not take a slot or draw a rejected sell
        # every run (OD17). The set is the owner's UNTRADABLE_SYMBOLS plus the broker's
        # tradable=false status for held names AND buy candidates (a dead ticker can still
        # produce a buy signal from a stale tape). Fails soft: on a lookup error only the
        # configured set is excluded.
        untradable_seen = resolve_untradable(
            self._broker,
            set(strategy_positions) | {sym for sym, _u in all_buys},
            configured_untradable(),
        )
        frozen = untradable_seen & set(strategy_positions)
        if frozen:
            _log.warning("Untradable holdings excluded from strategy: %s", sorted(frozen))
            self._notifier.send(
                "<b>⚠️ Untradable holdings:</b> "
                f"{', '.join(sorted(frozen))} (broker tradable=false; excluded from "
                "slots and sells — resolve manually)"
            )
            strategy_positions = {s: p for s, p in strategy_positions.items() if s not in frozen}
        buy_symbols = {sym for sym, _u in all_buys}
        blocked_buys = sorted((untradable_seen & buy_symbols) - frozen)
        if blocked_buys:
            _log.warning("Untradable buy candidates skipped: %s", blocked_buys)
            self._notifier.send(
                f"<b>⚠️ Untradable buy candidates skipped:</b> {', '.join(blocked_buys)} "
                "(asset not tradable at the broker)"
            )
        # Also drops a held-but-frozen name's own buy signal: it is no longer in
        # `strategy_positions`, so the buy loop's "already held" check would not catch it.
        all_buys = [(sym, u) for sym, u in all_buys if sym not in untradable_seen]
        untradable_hit: set[str] = set()

        # Determine unapplied-split corrections for currently-held symbols.
        # Primary evidence is paper_snapshots/paper_trades history (see
        # _compute_split_corrections and split_check.py's module docstring);
        # the broker's own SPLIT activities feed is kept only as a secondary
        # signal, since it is a no-op on this paper account. Fails soft
        # internally -- an API or DB error here degrades toward {} ("no
        # corrections known"), never an exception.
        split_since = today_et - timedelta(days=_SPLIT_LOOKBACK_DAYS)
        split_corrections = self._compute_split_corrections(positions, split_since, today_et)
        if split_corrections:
            _log.warning("Unapplied broker split corrections: %s", split_corrections)
        # Concentration checks (below, in the buy loop) must see true
        # economic exposure -- the corrected market_value -- not the
        # broker's uncorrected figure for a split-flagged holding. Built from
        # `strategy_positions`, not raw `positions`: the sweep ETF position is
        # invisible to all strategy signal/slot logic, and that includes the
        # concentration check (its exposure is cash-in-waiting, not a
        # strategy bet).
        corrected_positions = apply_corrections_to_positions(strategy_positions, split_corrections)

        # Catch up exits the last-bar check missed (see _missed_exits). Added
        # here, before the sweep's anticipated_sells, so freed slots count.
        missed = self._missed_exits(strategy_positions, last_exit, set(signals["sells"]))
        if missed:
            _log.info("Catching up missed exits: %s", missed)
            signals["sells"] = sorted(set(signals["sells"]) | set(missed))

        # Cross-reference the broker's corporate-actions feed against this
        # account's DIV activities to credit (as a reporting-only accrual --
        # see dividend_check.py) cash dividends the broker never paid out.
        # Fails soft internally, same convention as the split check above.
        dividend_info = self._accrue_dividends(positions)

        # Seed the risk guard's peak from what was persisted by a prior
        # process (each cron run constructs a fresh PaperTrader, so without
        # this the halt below always compares against a same-run peak and
        # can never fire).
        try:
            persisted_peak = get_peak_value()
        except Exception as exc:
            _log.warning("Could not load persisted peak: %s", exc)
            reason = "Risk history unavailable: could not read persisted peak"
            self._notifier.send(f"<b>🛑 HALTED:</b> {reason}")
            return {"buys": [], "sells": [], "errors": [reason]}
        if persisted_peak is not None:
            try:
                self._risk.update_peak(persisted_peak)
            except ValueError as exc:
                reason = f"Invalid stored risk peak: {exc}"
                self._notifier.send(f"<b>🛑 HALTED:</b> {reason}")
                return {"buys": [], "sells": [], "errors": [reason]}

        # Check max drawdown halt BEFORE updating the peak with today's
        # value, so a real drawdown against the persisted peak is actually
        # measured (updating first would always show 0% drawdown).
        halted, halt_reason = self._risk.check_drawdown_halt(portfolio_value)
        loss_halts: list[str] = []
        if halted:
            if (
                portfolio_value is None
                or not math.isfinite(portfolio_value)
                or portfolio_value <= 0
            ):
                self._notifier.send(f"<b>🛑 HALTED:</b> {halt_reason}")
                return {"buys": [], "sells": [], "errors": [halt_reason]}
            loss_halts.append(halt_reason)

        if not loss_halts:
            self._risk.update_peak(portfolio_value)
            try:
                if self._risk.peak_value is not None:
                    self._write(save_peak_value, self._risk.peak_value)
            except Exception as exc:
                _log.warning("Could not persist peak value: %s", exc)
                reason = "Risk history unavailable: could not persist drawdown peak"
                self._notifier.send(f"<b>🛑 HALTED:</b> {reason}")
                return {"buys": [], "sells": [], "errors": [reason]}

        # Booked equity can understate a holding whose split the broker has not applied yet,
        # so a harmless split can look like a loss. That false halt is accepted on purpose: it
        # only blocks buys and sweeps for the day (exits stay allowed), while adding the
        # correction back could hide a real loss when the stored correction is stale.
        last_equity = account.get("last_equity")
        if last_equity is None:
            reason = "Invalid broker previous-close equity; daily loss cannot be assessed"
            self._notifier.send(f"<b>🛑 HALTED:</b> {reason}")
            return {"buys": [], "sells": [], "errors": [reason]}
        daily_halted, daily_reason = self._risk.check_daily_loss(portfolio_value, last_equity)
        if daily_halted:
            if not math.isfinite(last_equity) or last_equity <= 0:
                self._notifier.send(f"<b>🛑 HALTED:</b> {daily_reason}")
                return {"buys": [], "sells": [], "errors": [daily_reason]}
            if split_corrections:
                daily_reason += (
                    " (known unapplied split correction for "
                    f"{', '.join(sorted(split_corrections))}: booked equity may understate the "
                    "true value; buys stay blocked today)"
                )
            loss_halts.append(daily_reason)

        if loss_halts:
            all_buys = []
            try:
                self._notifier.send(f"<b>🛑 HALTED buys; exits only:</b> {'; '.join(loss_halts)}")
            except Exception:
                _log.exception("Could not send loss-halt notice")
            if pending_symbols:
                try:
                    self._notifier.send(
                        "<b>⚠️ Pending orders may still fill:</b> no cancellations were attempted."
                    )
                except Exception:
                    _log.exception("Could not send pending-order notice")

        # Prior reporting snapshot is used only for the daily summary, not the risk check.
        try:
            prev_snapshot_value = get_latest_snapshot(before=today_et)
        except Exception as exc:
            _log.warning("Could not load prior reporting snapshot: %s", exc)
            prev_snapshot_value = None
        if prev_snapshot_value is not None and (
            not math.isfinite(prev_snapshot_value) or prev_snapshot_value <= 0
        ):
            _log.warning("Invalid prior reporting snapshot; daily P&L is unavailable")
            prev_snapshot_value = None

        executed_sells: list[str] = []
        executed_buys: list[str] = []
        errors: list[str] = []
        # 5th element is the ledger `reason` tag: None for an ordinary
        # strategy order, cash_sweep.SWEEP_TRADE_REASON for a sweep order.
        pending_orders: list[tuple[str, str, str, float, str | None]] = []

        # Catastrophe stop (feature-flagged, default OFF -- see
        # catastrophe_stop.py). A position the strategy's own RSI exit never
        # fires on can otherwise decay indefinitely (see the module
        # docstring for the NXPI incident); this is a risk backstop,
        # evaluated BEFORE any strategy buy/sell signal below, so a stop-out
        # frees its slot/cash the same run. Reads `corrected_positions`
        # (split-corrected via cost_basis, built from `strategy_positions`
        # above -- the sweep ETF position is never in it), so an unapplied
        # broker split can't fake a phantom loss and the sweep symbol can
        # never be stopped out. Stopped symbols are removed from
        # `strategy_positions` immediately after, so the sells loop below
        # sees them as no longer held and never submits a second, coincident
        # strategy-exit sell for the same symbol.
        # Hoisted out of the flag check: the buy loop below consults this set
        # so a just-stopped symbol can never be re-bought in the same run.
        stopped_symbols: set[str] = set()
        invalid_exit_symbols: set[str] = set()
        if stop_threshold is not None:
            unassessable = set()
            for symbol, position in corrected_positions.items():
                try:
                    if catastrophe_stop.unrealized_pct(position) is None:
                        unassessable.add(symbol)
                except ValueError:
                    unassessable.add(symbol)
            if unassessable:
                reason = "Cannot assess catastrophe stop for: " + ", ".join(sorted(unassessable))
                loss_halts.append(reason)
                all_buys = []
                _log.warning("%s; new buys and sweeps blocked", reason)
                try:
                    self._notifier.send(f"<b>🛑 HALTED buys; exits only:</b> {reason}")
                except Exception:
                    _log.exception("Could not send unassessable catastrophe-stop alert")
                if pending_symbols:
                    try:
                        self._notifier.send(
                            "<b>⚠️ Pending orders may still fill:</b> "
                            "no cancellations were attempted."
                        )
                    except Exception:
                        _log.exception("Could not send pending-order notice")
            assessable = {
                symbol: position
                for symbol, position in corrected_positions.items()
                if symbol not in unassessable
            }
            for symbol in catastrophe_stop.find_catastrophe_stops(assessable, stop_threshold):
                qty = strategy_positions[symbol]["qty"]
                if qty is None or not math.isfinite(qty) or qty <= 0:
                    errors.append(
                        f"Cannot submit risk-reducing SELL {symbol}: invalid long quantity; "
                        "new strategy buys blocked"
                    )
                    invalid_exit_symbols.add(symbol)
                    continue
                if symbol in pending_symbols:
                    _log.info("Skipping catastrophe stop %s: order already pending", symbol)
                    continue
                pct = catastrophe_stop.unrealized_pct(corrected_positions[symbol])
                if self._dry_run:
                    stopped_symbols.add(symbol)
                    executed_sells.append(symbol)
                    self._notifier.send(
                        f"<b>🛑 DRY RUN catastrophe stop:</b> {symbol} (qty {qty}, "
                        f"unrealized {pct:.1%} breached {stop_threshold:.1%} floor)"
                    )
                    continue
                oid = self._submit_order(
                    "SELL", symbol, qty, catastrophe_stop.CATASTROPHE_STOP_REASON
                )
                stopped_symbols.add(symbol)
                executed_sells.append(symbol)
                pending_orders.append(
                    (
                        oid,
                        "SELL",
                        symbol,
                        strategy_positions[symbol]["market_value"],
                        catastrophe_stop.CATASTROPHE_STOP_REASON,
                    )
                )
                self._notifier.send(
                    f"<b>🛑 Catastrophe stop:</b> {symbol} unrealized {pct:.1%} "
                    f"breached {stop_threshold:.1%} floor — selling qty {qty}."
                )
            if stopped_symbols:
                strategy_positions = {
                    sym: pos
                    for sym, pos in strategy_positions.items()
                    if sym not in stopped_symbols
                }

        # Check requested strategy exits before estimating sweep funding for buys.
        for symbol in signals["sells"]:
            # A pending order does not excuse an invalid quantity: the account state is
            # still untrusted, so it must still block new buys.
            if symbol not in strategy_positions or symbol in invalid_exit_symbols:
                continue
            qty = strategy_positions[symbol]["qty"]
            if qty is None or not math.isfinite(qty) or qty <= 0:
                errors.append(
                    f"Cannot submit risk-reducing SELL {symbol}: invalid long quantity; "
                    "new strategy buys blocked"
                )
                invalid_exit_symbols.add(symbol)
        if invalid_exit_symbols:
            all_buys = []

        # Buys — respect global + per-sleeve position limits. Computed here
        # (before the sells loop) rather than just before the buy loop below,
        # because the cash-sweep funding check needs the same sleeve notional
        # sizing to estimate how much cash the day's strategy buys could
        # consume. Each position is sized as a fixed fraction of its own
        # sleeve's allocated capital (weight * scale * portfolio_value),
        # independent of how many signals fire that day within the sleeve —
        # sleeve_slot_caps governs sleeve concurrency, sleeve_position_notional
        # governs per-trade size.
        slot_caps = self._risk.sleeve_slot_caps(weights)
        sleeve_open_count = {u: 0 for u in weights}
        buys_by_sleeve: dict[str, list[str]] = {}
        for symbol, universe in all_buys:
            buys_by_sleeve.setdefault(universe, []).append(symbol)
        sleeve_notional_by_universe = {
            universe: self._risk.sleeve_position_notional(
                portfolio_value, weights.get(universe, 0.0), scale
            )
            for universe in buys_by_sleeve
        }

        if sweep_on and not loss_halts:
            # Sell the sweep position down FIRST (before strategy sells/buys
            # execute) if the day's anticipated strategy buys would need more
            # cash than is currently on hand. `prospective_buy_notional` is a
            # deliberate upper-bound estimate (see
            # cash_sweep.estimate_prospective_buy_notional) computed BEFORE
            # the sells loop runs, so it conservatively assumes the slots that
            # today's sells will free are already free.
            anticipated_sells = [
                s for s in signals["sells"] if s in strategy_positions and s not in pending_symbols
            ]
            slots_estimate = self._risk.max_new_positions(
                len(strategy_positions) - len(anticipated_sells)
            )
            prospective_buy_notional = cash_sweep.estimate_prospective_buy_notional(
                buys_by_sleeve, sleeve_notional_by_universe, slot_caps, slots_estimate
            )
            sweep_position = positions.get(sweep_sym)
            sweep_position_value = sweep_position["market_value"] if sweep_position else 0.0
            sell_action = cash_sweep.compute_sweep_sell_for_funding(
                cash_available=account["cash"],
                portfolio_value=portfolio_value,
                prospective_buy_notional=prospective_buy_notional,
                current_sweep_position_value=sweep_position_value,
                reserve_pct=cash_sweep.reserve_pct(),
            )
            if sell_action.side == "sell" and sweep_position:
                sweep_price = sweep_position["current_price"]
                sell_qty = round(sell_action.notional / sweep_price, 4) if sweep_price > 0 else 0.0
                if sell_qty > 0:
                    if self._dry_run:
                        self._notifier.send(
                            f"<b>🔍 DRY RUN sweep sell:</b> {sweep_sym} "
                            f"(${sell_action.notional:.0f}, funding strategy buys)"
                        )
                    else:
                        # Alpaca paper credits a market sell's proceeds to
                        # cash immediately (no T+1/T+2 settlement lag), so
                        # it's safe to treat this cash as available to the
                        # strategy buys submitted later in this same run.
                        # A same-run sell -> buy sequence would NOT be
                        # safe on a broker with real settlement delay.
                        oid = self._submit_order(
                            "SELL", sweep_sym, sell_qty, cash_sweep.SWEEP_TRADE_REASON
                        )
                        pending_orders.append(
                            (
                                oid,
                                "SELL",
                                sweep_sym,
                                sell_action.notional,
                                cash_sweep.SWEEP_TRADE_REASON,
                            )
                        )

        # Sells first — free up position slots. `regular_sell_count` (not
        # `executed_sells`) is what feeds the slots_available math below: a
        # catastrophe stop already shrank `strategy_positions` by removing
        # its symbol (see the block above), so that freed slot is already
        # reflected there. `executed_sells` also records catastrophe-stop
        # sells (for the return value / reporting), so subtracting its full
        # length here would count every stop's freed slot twice -- once via
        # the shrunken `strategy_positions`, again via `executed_sells` --
        # letting the buy loop open one extra position per stop and breach
        # `max_positions`.
        regular_sell_count = 0
        for symbol in signals["sells"]:
            if symbol not in strategy_positions or symbol in invalid_exit_symbols:
                continue
            if symbol in pending_symbols:
                _log.info("Skipping SELL %s: order already pending", symbol)
                continue
            qty = strategy_positions[symbol]["qty"]
            if self._dry_run:
                executed_sells.append(symbol)
                regular_sell_count += 1
                self._notifier.send(f"<b>🔍 DRY RUN sell:</b> {symbol} (qty {qty})")
                continue
            oid = self._submit_order("SELL", symbol, qty, None)
            executed_sells.append(symbol)
            regular_sell_count += 1
            pending_orders.append(
                (oid, "SELL", symbol, strategy_positions[symbol]["market_value"], None)
            )

        # Wait for every sell submitted so far (catastrophe stop, sweep funding,
        # strategy exits) to fill before buying: Alpaca credits proceeds only on
        # fill, and on 2026-09-25 buys sent while the funding sell was still
        # working were rejected for insufficient buying power (6 of 8).
        filled_orders: dict[str, dict] = self._poll_orders(
            [o for o in pending_orders if o[1] == "SELL"]
        )

        if filled_orders and not self._dry_run:
            self._reconcile_order_intents()

        # A submitted exit does not free a position slot unless it fully fills.
        if not self._dry_run:
            for oid, side, symbol, _, reason in pending_orders:
                if side != "SELL" or reason == cash_sweep.SWEEP_TRADE_REASON:
                    continue
                info = filled_orders.get(oid, {})
                qty = info.get("filled_qty")
                expected = positions[symbol]["qty"]
                if (
                    info.get("status") != "filled"
                    or qty is None
                    or not math.isfinite(qty)
                    or not math.isclose(qty, expected, rel_tol=1e-12, abs_tol=1e-9)
                ):
                    errors.append(
                        f"SELL {symbol} not fully filled ({info.get('status', 'unknown')}); "
                        "new strategy buys blocked"
                    )
                    buys_by_sleeve.clear()

        slots_available = self._risk.max_new_positions(len(strategy_positions) - regular_sell_count)
        buys_attempted = 0
        for universe, syms in buys_by_sleeve.items():
            sleeve_notional = sleeve_notional_by_universe[universe]
            sleeve_cap = slot_caps.get(universe, 0)
            for symbol in syms:
                if symbol in strategy_positions:
                    continue
                # A symbol the catastrophe stop just sold is gone from
                # `strategy_positions`, so the held-check above can't see it —
                # without this guard a coincident buy signal would re-open the
                # position the backstop just closed, in the same run.
                if symbol in stopped_symbols:
                    _log.info("Skipping BUY %s: catastrophe-stopped this run", symbol)
                    continue
                if symbol in pending_symbols:
                    _log.info("Skipping BUY %s: order already pending", symbol)
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
                    symbol,
                    corrected_positions,
                    portfolio_value,
                    prospective_notional=sleeve_notional,
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
                oid = self._submit_order("BUY", symbol, sleeve_notional, None)
                executed_buys.append(symbol)
                buys_attempted += 1
                sleeve_open_count[universe] += 1
                pending_orders.append((oid, "BUY", symbol, sleeve_notional, None))

        # Poll the remaining (buy) orders until they fill (or timeout)
        filled_orders.update(
            self._poll_orders([o for o in pending_orders if o[0] not in filled_orders])
        )

        if pending_orders and not self._dry_run:
            self._reconcile_order_intents()

        # An invalid exit quantity means the account state is not trusted: no sweep buy either.
        if sweep_on and not loss_halts and not invalid_exit_symbols:
            # Sweep BUY: deploy whatever cash is left over after the strategy's
            # own orders for the day, using the account's REAL post-trade cash
            # (not an estimate) now that strategy sells/buys above have been
            # submitted and polled to a terminal status.
            try:
                mid_account = self._broker.get_account()
                buy_action = cash_sweep.compute_sweep_buy(
                    cash_after_strategy_orders=mid_account["cash"],
                    portfolio_value=mid_account["portfolio_value"],
                    reserve_pct=cash_sweep.reserve_pct(),
                    min_clip=cash_sweep.min_clip_usd(),
                    buy_trigger_pct=cash_sweep.buy_trigger_pct(),
                )
            except Exception as exc:
                _log.warning("Cash sweep buy sizing failed (non-fatal): %s", exc)
                buy_action = cash_sweep.SweepAction(None, 0.0)
            if buy_action.side == "buy":
                if self._dry_run:
                    self._notifier.send(
                        f"<b>🔍 DRY RUN sweep buy:</b> {sweep_sym} (${buy_action.notional:.0f})"
                    )
                else:
                    oid = self._submit_order(
                        "BUY", sweep_sym, buy_action.notional, cash_sweep.SWEEP_TRADE_REASON
                    )
                    sweep_buy_order = (
                        oid,
                        "BUY",
                        sweep_sym,
                        buy_action.notional,
                        cash_sweep.SWEEP_TRADE_REASON,
                    )
                    filled_orders.update(self._poll_orders([sweep_buy_order]))
                    pending_orders.append(sweep_buy_order)
                    self._reconcile_order_intents()

        # Alert on every submitted order, but only book the trade ledger at a
        # real executed value — never the intended notional. Orders still
        # working at run end (queued after the close, or a partial fill not yet
        # complete) are persisted so the next run reconciles their final fill;
        # terminally-failed orders are simply dropped.
        for oid, side, symbol, amount, reason in pending_orders:
            info = filled_orders.get(oid, {})
            status = info.get("status", "pending")
            filled_qty = info.get("filled_qty") or 0.0
            filled_price = info.get("filled_avg_price") or 0.0
            filled = filled_qty > 0 and filled_price > 0
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
            # Leave durable intents for the next reconciliation pass; do not
            # create a second legacy representation or book a separate trade.

        if untradable_hit:
            self._notifier.send(
                "<b>⚠️ Untradable (order skipped):</b> "
                f"{', '.join(sorted(untradable_hit))} — the broker says the asset is not "
                "active/tradable; excluded, not an error — resolve manually"
            )
        if errors:
            self._notifier.send(f"<b>❌ {len(errors)} order(s) failed:</b>\n" + "\n".join(errors))

        try:
            new_account = self._broker.get_account()
            updated_positions = self._broker.get_positions()
        except Exception as exc:
            _log.error(
                "Post-trade account or positions read failed (%s); snapshot not saved",
                type(exc).__name__,
            )
            try:
                self._notifier.send(
                    "<b>⚠️ Account or positions data unavailable after trading:</b> "
                    + (
                        "dry run; no orders submitted"
                        if self._dry_run
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
            starting_capital = get_earliest_snapshot()
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
            f"Positions: {len(updated_positions)}/{self._risk.cfg.max_positions} | "
            f"Scale: {scale:.2f}x | Weights: {weight_str}"
        )
        self._notifier.daily_summary(
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
        )
        self._notifier.send(f"<b>📊 Risk:</b> {risk_line}")

        try:
            # Same run_date reasoning as log_trade / log_pending_order above:
            # stamp the actual ET session date of this run, not the lagged
            # signals["as_of"] date.
            if corrected_value is not None:
                self._write(
                    log_snapshot,
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
                self._notifier.send(f"<b>⚠️ {message}.</b>")
            except Exception:
                _log.exception("Could not send snapshot failure notice")

        return {
            "buys": executed_buys,
            "sells": executed_sells,
            "errors": loss_halts + errors,
            "untradable": sorted(untradable_seen | untradable_hit),
            "skipped_writes": list(self.skipped_writes),
        }


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
