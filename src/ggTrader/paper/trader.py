"""Paper trading orchestrator: signals -> orders -> risk guardrails -> notifications."""

from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from ggTrader.paper import cash_sweep, catastrophe_stop
from ggTrader.paper.alpaca_broker import AlpacaBroker
from ggTrader.paper.dividend_check import DIVIDEND_BACKFILL_START, compute_dividend_accruals
from ggTrader.paper.notifier import TelegramNotifier
from ggTrader.paper.persist import (
    clear_pending_order,
    get_accrued_dividend_keys,
    get_earliest_snapshot,
    get_latest_snapshot,
    get_peak_value,
    get_pending_orders,
    get_snapshot_history,
    get_total_dividend_accrual,
    init_paper_schema,
    log_dividend_accrual,
    log_pending_order,
    log_snapshot,
    log_trade,
    mark_pending_order_stale,
    save_peak_value,
)
from ggTrader.paper.risk import RiskConfig, RiskGuard
from ggTrader.paper.signal_runner import generate_blended_signals
from ggTrader.paper.split_check import apply_corrections_to_positions

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
            _log.warning("Could not load pending orders (non-fatal): %s", exc)
            return still_pending

        now = datetime.now(timezone.utc)
        for po in pending:
            oid = po["order_id"]
            try:
                info = self._broker.get_order(oid)
            except Exception as exc:
                _log.warning("Reconcile: get_order(%s) failed: %s", oid, exc)
                still_pending.add(po["symbol"])
                continue

            status = info.get("status", "pending")
            filled_qty = info.get("filled_qty") or 0.0
            filled_price = info.get("filled_avg_price") or 0.0
            filled = filled_qty > 0 and filled_price > 0
            terminal = status in ("filled", "canceled", "rejected", "expired")
            if not terminal:
                still_pending.add(po["symbol"])
                self._flag_if_stale(po, now)
                continue  # still working — try again next run

            if filled:
                amount = filled_qty * filled_price
                log_trade(po["run_date"], po["side"], po["symbol"], amount, oid, po.get("reason"))
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
                log_dividend_accrual(
                    acc["symbol"], acc["ex_date"], acc["rate"], acc["qty"], acc["amount"]
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

    def run(self) -> dict:
        try:
            init_paper_schema()
        except Exception as exc:
            _log.warning("DB schema init failed (non-fatal): %s", exc)

        # Settle any orders that filled after a prior run exited, and note
        # which symbols still have an order open so we don't double up on
        # them below.
        pending_symbols = self._reconcile_pending_orders()

        # Market-open gate: cron has no calendar awareness, so a holiday or
        # half-day would otherwise submit DAY orders into a closed market.
        # Fail open (assume the market is open) on a get_clock() error, same
        # non-fatal-degrade convention used elsewhere in this file.
        market_open = True
        try:
            market_open = self._broker.get_clock().get("is_open", True)
        except Exception as exc:
            _log.warning("get_clock() failed (non-fatal, assuming market open): %s", exc)
        if not market_open:
            _log.info("Market closed; skipping run")
            self._notifier.send("<b>ℹ️ Market closed</b> — no trading this run.")
            return {"buys": [], "sells": [], "errors": []}

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

        # Freshness gate: refuse to trade on data that is stale beyond the
        # normal one-completed-session lag (see _MAX_SIGNAL_AGE_DAYS).
        today_et = _today_et()
        try:
            as_of_date = date.fromisoformat(signals["as_of"])
        except (TypeError, ValueError):
            as_of_date = None
        age_days = None if as_of_date is None else (today_et - as_of_date).days
        if age_days is None or age_days > _MAX_SIGNAL_AGE_DAYS:
            self._notifier.send(
                f"<b>⚠️ Stale signal data:</b> as_of={signals['as_of']} is "
                f"{'unparseable' if age_days is None else f'{age_days}d old'} "
                f"(today {today_et.isoformat()}, max {_MAX_SIGNAL_AGE_DAYS}d); "
                f"no trades submitted this run."
            )
            return {"buys": [], "sells": [], "errors": []}

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

        # Cross-reference the broker's own corporate-actions feed against
        # this account's SPLIT activities for currently-held symbols to find
        # splits the broker knows about but never applied to the account
        # (see AlpacaBroker.get_split_corrections / split_check.py). Fails
        # soft internally -- an API error here yields {} ("no corrections
        # known"), never an exception.
        split_since = today_et - timedelta(days=_SPLIT_LOOKBACK_DAYS)
        split_corrections = self._broker.get_split_corrections(list(positions), split_since)
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
            _log.warning("Could not load persisted peak (non-fatal): %s", exc)
            persisted_peak = None
        if persisted_peak is not None:
            self._risk.update_peak(persisted_peak)

        # Check max drawdown halt BEFORE updating the peak with today's
        # value, so a real drawdown against the persisted peak is actually
        # measured (updating first would always show 0% drawdown).
        halted, halt_reason = self._risk.check_drawdown_halt(portfolio_value)
        if halted:
            self._notifier.send(f"<b>🛑 HALTED:</b> {halt_reason}")
            return {"buys": [], "sells": [], "errors": [halt_reason]}

        self._risk.update_peak(portfolio_value)
        try:
            if self._risk.peak_value is not None:
                save_peak_value(self._risk.peak_value)
        except Exception as exc:
            _log.warning("Could not persist peak value (non-fatal): %s", exc)

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
        if catastrophe_stop.catastrophe_stop_enabled():
            threshold = catastrophe_stop.catastrophe_stop_pct()
            for symbol in catastrophe_stop.find_catastrophe_stops(corrected_positions, threshold):
                if symbol in pending_symbols:
                    _log.info("Skipping catastrophe stop %s: order already pending", symbol)
                    continue
                qty = strategy_positions[symbol]["qty"]
                pct = catastrophe_stop.unrealized_pct(corrected_positions[symbol])
                if self._dry_run:
                    stopped_symbols.add(symbol)
                    executed_sells.append(symbol)
                    self._notifier.send(
                        f"<b>🛑 DRY RUN catastrophe stop:</b> {symbol} (qty {qty}, "
                        f"unrealized {pct:.1%} breached {threshold:.1%} floor)"
                    )
                    continue
                try:
                    oid = self._broker.submit_sell(symbol, qty)
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
                        f"breached {threshold:.1%} floor — selling qty {qty}."
                    )
                except Exception as exc:
                    errors.append(f"CATASTROPHE STOP {symbol}: {exc}")
            if stopped_symbols:
                strategy_positions = {
                    sym: pos
                    for sym, pos in strategy_positions.items()
                    if sym not in stopped_symbols
                }

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

        if sweep_on:
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
                        try:
                            # Alpaca paper credits a market sell's proceeds to
                            # cash immediately (no T+1/T+2 settlement lag), so
                            # it's safe to treat this cash as available to the
                            # strategy buys submitted later in this same run.
                            # A same-run sell -> buy sequence would NOT be
                            # safe on a broker with real settlement delay.
                            oid = self._broker.submit_sell(sweep_sym, sell_qty)
                            pending_orders.append(
                                (
                                    oid,
                                    "SELL",
                                    sweep_sym,
                                    sell_action.notional,
                                    cash_sweep.SWEEP_TRADE_REASON,
                                )
                            )
                        except Exception as exc:
                            errors.append(f"SWEEP SELL {sweep_sym}: {exc}")

        # Sells first — free up position slots
        for symbol in signals["sells"]:
            if symbol not in strategy_positions:
                continue
            if symbol in pending_symbols:
                _log.info("Skipping SELL %s: order already pending", symbol)
                continue
            qty = strategy_positions[symbol]["qty"]
            if self._dry_run:
                executed_sells.append(symbol)
                self._notifier.send(f"<b>🔍 DRY RUN sell:</b> {symbol} (qty {qty})")
                continue
            try:
                oid = self._broker.submit_sell(symbol, qty)
                executed_sells.append(symbol)
                pending_orders.append(
                    (oid, "SELL", symbol, strategy_positions[symbol]["market_value"], None)
                )
            except Exception as exc:
                errors.append(f"SELL {symbol}: {exc}")

        slots_available = self._risk.max_new_positions(
            len(strategy_positions) - len(executed_sells)
        )
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
                try:
                    oid = self._broker.submit_buy(symbol, sleeve_notional)
                    executed_buys.append(symbol)
                    buys_attempted += 1
                    sleeve_open_count[universe] += 1
                    pending_orders.append((oid, "BUY", symbol, sleeve_notional, None))
                except Exception as exc:
                    errors.append(f"BUY {symbol}: {exc}")

        # Poll submitted orders until they fill (or timeout)
        filled_orders: dict[str, dict] = self._poll_orders(pending_orders)

        if sweep_on:
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
                    try:
                        oid = self._broker.submit_buy(sweep_sym, buy_action.notional)
                        sweep_buy_order = (
                            oid,
                            "BUY",
                            sweep_sym,
                            buy_action.notional,
                            cash_sweep.SWEEP_TRADE_REASON,
                        )
                        filled_orders.update(self._poll_orders([sweep_buy_order]))
                        pending_orders.append(sweep_buy_order)
                    except Exception as exc:
                        errors.append(f"SWEEP BUY {sweep_sym}: {exc}")

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
                    # run_date is the ET calendar date of THIS run (the session the
                    # order was actually placed/filled in), not signals["as_of"]
                    # (the date of the last completed bar the signal was computed
                    # from, which trails by one session and was the vector for the
                    # OHLCV day-shift bug -- see _today_et() and
                    # scripts/migrate_paper_dates_20260822.py).
                    log_trade(today_et.isoformat(), side, symbol, trade_amount, oid, reason)
                else:
                    _log.info(
                        "Order %s (%s %s) terminal unfilled (status=%s); no ledger entry",
                        oid,
                        side,
                        symbol,
                        status,
                    )
            else:
                # accepted / new / pending / partially_filled — settle next run.
                # Same run_date reasoning as log_trade above.
                log_pending_order(today_et.isoformat(), side, symbol, amount, oid, reason)
                _log.info(
                    "Order %s (%s %s) still working (status=%s); queued for reconciliation",
                    oid,
                    side,
                    symbol,
                    status,
                )

        if errors:
            self._notifier.send(f"<b>❌ {len(errors)} order(s) failed:</b>\n" + "\n".join(errors))

        new_account = self._broker.get_account()
        new_value = new_account["portfolio_value"]  # broker's own ("booked") figure, uncorrected
        updated_positions = self._broker.get_positions()

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

        daily_pnl = corrected_value - day_start

        unrealized_pnl = sum(
            p.get("unrealized_pl", 0.0) for p in corrected_updated_positions.values()
        )
        starting_capital = get_earliest_snapshot()
        total_pnl = (corrected_value - starting_capital) if starting_capital else None

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
            errors=errors,
        )
        self._notifier.send(f"<b>📊 Risk:</b> {risk_line}")

        try:
            # Same run_date reasoning as log_trade / log_pending_order above:
            # stamp the actual ET session date of this run, not the lagged
            # signals["as_of"] date.
            log_snapshot(
                today_et.isoformat(),
                corrected_value,
                new_account["cash"],
                corrected_updated_positions,
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
