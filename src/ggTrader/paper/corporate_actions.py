"""Reconcile splits, dividends and missed exit signals."""

import logging
from datetime import date

from ggTrader.paper.constants import TRADER_LOGGER_NAME
from ggTrader.paper.dividend_check import DIVIDEND_BACKFILL_START, compute_dividend_accruals
from ggTrader.paper.ledger import PaperLedger
from ggTrader.paper.ports import Broker
from ggTrader.paper.run_helpers import as_date as _as_date
from ggTrader.paper.seams import RunHooks
from ggTrader.paper.split_check import compute_split_corrections

_log = logging.getLogger(TRADER_LOGGER_NAME)


class CorporateActions:
    """Handle corporate action reconciliation using injected collaborators."""

    def __init__(self, *, broker: Broker, ledger: PaperLedger, hooks: RunHooks) -> None:
        self._broker = broker
        self._ledger = ledger
        self._hooks = hooks

    def split_corrections(
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
            persisted_states = self._ledger.get_open_split_states()
        except Exception as exc:
            _log.warning("Could not load persisted split state (non-fatal): %s", exc)
            persisted_states = {}

        # A symbol no longer held has nothing left to correct -- drop its
        # persisted row so the table doesn't grow unboundedly.
        for symbol in list(persisted_states):
            if symbol not in held_symbols:
                try:
                    self._ledger.clear_split_correction(symbol)
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
            snapshot_history = self._ledger.get_snapshot_history()
        except Exception as exc:
            _log.warning("Could not load snapshot history for split check (non-fatal): %s", exc)
            snapshot_history = []

        trade_dates_by_symbol: dict[str, list] = {}
        for symbol in corp_splits:
            try:
                trade_dates_by_symbol[symbol] = self._ledger.get_trade_history_dates(symbol)
            except Exception as exc:
                _log.warning(
                    "Could not load trade history for %s split check (non-fatal): %s",
                    symbol,
                    exc,
                )
                trade_dates_by_symbol[symbol] = []

        snapshot_applied, unresolved = self._hooks.find_split_applied(
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
                    self._ledger.clear_split_correction(symbol)
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
                self._ledger.save_split_correction(symbol, str(ex_date), factor)
            except Exception as exc:
                _log.warning("Could not persist split state for %s (non-fatal): %s", symbol, exc)

        return {**persisted, **new_corrections}

    def accrue_dividends(self, positions: dict[str, dict]) -> dict:
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
            result["total"] = self._ledger.get_total_dividend_accrual()
        except Exception as exc:
            _log.warning("Could not load dividend accrual total (non-fatal): %s", exc)
            return result

        try:
            snapshot_history = self._ledger.get_snapshot_history()
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
            already_accrued_keys = self._ledger.get_accrued_dividend_keys()
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
                self._ledger.log_dividend_accrual(
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
            result["total"] = self._ledger.get_total_dividend_accrual()
        except Exception as exc:
            _log.warning("Could not reload dividend accrual total (non-fatal): %s", exc)
        result["new_accruals"] = accruals
        result["skipped"] = skipped
        return result


def missed_exits(
    ledger: PaperLedger, positions: dict, last_exit: dict[str, str], already: set[str]
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
        entries = ledger.get_last_buy_dates()
    except Exception as exc:
        _log.warning("last-buy lookup failed, skipping exit catch-up: %s", exc)
        return []
    return sorted(
        s
        for s in candidates
        if s in entries and date.fromisoformat(last_exit[s]) > _as_date(entries[s])
    )
