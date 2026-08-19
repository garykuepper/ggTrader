"""Paper trading orchestrator: signals -> orders -> risk guardrails -> notifications."""

from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta, timezone

from zoneinfo import ZoneInfo

from ggTrader.paper.alpaca_broker import AlpacaBroker
from ggTrader.paper.notifier import TelegramNotifier
from ggTrader.paper.persist import (
    clear_pending_order,
    get_earliest_snapshot,
    get_latest_snapshot,
    get_latest_snapshot_positions,
    get_latest_snapshot_run_date,
    get_peak_value,
    get_pending_orders,
    init_paper_schema,
    log_pending_order,
    log_snapshot,
    log_trade,
    mark_pending_order_stale,
    save_peak_value,
)
from ggTrader.paper.risk import RiskConfig, RiskGuard
from ggTrader.paper.signal_runner import generate_blended_signals
from ggTrader.paper.split_check import find_unadjusted_split_symbols, get_recent_splits

_log = logging.getLogger(__name__)

# How far back to check held symbols for split events the broker may not have
# applied to their qty/avg_entry (see split_check.py). Daily runs only need a
# window wide enough to survive a missed run or two.
_SPLIT_LOOKBACK_DAYS = 14

# A pending order still open this many days after submission gets a one-time
# "stale" alert (see _flag_if_stale) instead of being silently re-polled
# forever with no visibility. Reconciliation itself keeps retrying past this
# point -- only the repeated Telegram noise is capped.
_STALE_PENDING_DAYS = 5

_ET = ZoneInfo("America/New_York")


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
        return still_pending

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

        # Freshness gate: only trade on today's bars. A stale as_of (e.g. a
        # yfinance outage serving yesterday's close) becomes a no-trade alert
        # instead of silently trading on old data.
        today_et = _today_et()
        if signals["as_of"] != today_et.isoformat():
            self._notifier.send(
                f"<b>⚠️ Stale signal data:</b> as_of={signals['as_of']} != today "
                f"({today_et.isoformat()}); no trades submitted this run."
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
        pending_orders: list[tuple[str, str, str, float]] = []

        # Sells first — free up position slots
        for symbol in signals["sells"]:
            if symbol not in positions:
                continue
            if symbol in pending_symbols:
                _log.info("Skipping SELL %s: order already pending", symbol)
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

        slots_available = self._risk.max_new_positions(
            len(positions) - len(executed_sells), weights
        )
        total_position_cap = sum(slot_caps.values())
        buys_attempted = 0
        for universe, syms in buys_by_sleeve.items():
            sleeve_notional = self._risk.sleeve_position_notional(
                portfolio_value, weights.get(universe, 0.0), scale
            )
            sleeve_cap = slot_caps.get(universe, 0)
            for symbol in syms:
                if symbol in positions:
                    continue
                if symbol in pending_symbols:
                    _log.info("Skipping BUY %s: order already pending", symbol)
                    continue
                if buys_attempted >= slots_available:
                    _log.info(
                        "Max positions reached (%d), skipping %s",
                        total_position_cap,
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

        if errors:
            self._notifier.send(f"<b>❌ {len(errors)} order(s) failed:</b>\n" + "\n".join(errors))

        new_account = self._broker.get_account()
        new_value = new_account["portfolio_value"]
        updated_positions = self._broker.get_positions()

        daily_pnl = new_value - day_start

        # Detect positions where the broker never applied a stock split's qty
        # adjustment (observed on MNST's 2026-08-11 2-for-1 split in Alpaca's
        # paper environment) — its unrealized_pl is bogus and would otherwise
        # silently corrupt the aggregate below.
        prev_positions = get_latest_snapshot_positions() or {}
        prev_snapshot_date: date | None = None
        try:
            prev_run_date_raw = get_latest_snapshot_run_date()
            if prev_run_date_raw:
                prev_snapshot_date = date.fromisoformat(str(prev_run_date_raw)[:10])
        except Exception as exc:
            _log.warning("Could not parse previous snapshot date (non-fatal): %s", exc)

        flagged_splits: list[str] = []
        splits_unavailable = False
        if prev_positions and prev_snapshot_date is not None:
            since = today_et - timedelta(days=_SPLIT_LOOKBACK_DAYS)
            try:
                splits, all_failed = get_recent_splits(list(updated_positions), since.isoformat())
            except Exception as exc:
                _log.warning("Split check failed (non-fatal): %s", exc)
                splits, all_failed = {}, False
            if all_failed:
                splits_unavailable = True
                _log.warning("Split check unavailable: all symbol lookups failed")
            else:
                flagged_splits = find_unadjusted_split_symbols(
                    prev_positions,
                    updated_positions,
                    splits,
                    prev_snapshot_date=prev_snapshot_date,
                    today=today_et,
                )
        if flagged_splits:
            _log.warning("Unadjusted broker split detected: %s", flagged_splits)

        # Estimated phantom P&L contributed by flagged symbols -- the
        # broker's own (bogus) unrealized_pl for each is the most defensible
        # available number without recovering the true post-split cost basis.
        phantom_estimate = sum(
            updated_positions.get(sym, {}).get("unrealized_pl", 0.0) for sym in flagged_splits
        )

        # Cumulative P&L breakdown. When any symbol is split-flagged, the
        # whole summary (daily/total/realized/unrealized) is suspect by
        # roughly phantom_estimate -- see notifier.daily_summary's banner --
        # so these aggregates are computed over ALL positions rather than
        # silently excluding the flagged one (which used to just shift the
        # corruption into the derived "realized" line).
        unrealized_pnl = sum(p.get("unrealized_pl", 0.0) for p in updated_positions.values())
        starting_capital = get_earliest_snapshot()
        total_pnl = (new_value - starting_capital) if starting_capital else None

        weight_str = ", ".join(f"{u}={w:.0%}" for u, w in weights.items())
        total_position_cap = sum(self._risk.sleeve_slot_caps(weights).values())
        risk_line = (
            f"Positions: {len(updated_positions)}/{total_position_cap} | "
            f"Scale: {scale:.2f}x | Weights: {weight_str}"
        )
        self._notifier.daily_summary(
            new_value,
            daily_pnl,
            updated_positions,
            total_pnl=total_pnl,
            unrealized_pnl=unrealized_pnl,
            flagged_splits=flagged_splits,
            phantom_estimate=phantom_estimate,
            errors=errors,
        )
        self._notifier.send(f"<b>📊 Risk:</b> {risk_line}")
        if splits_unavailable:
            self._notifier.send(
                "<b>⚠️ Split check unavailable:</b> all symbol lookups failed this run; "
                "unable to confirm P&L is unaffected by an unreflected split."
            )

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
