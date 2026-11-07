"""Fail-closed gates for the paper trader's signal path (spec 8.3, S7 section 4).

Every gate answers one question: is it safe to turn these signals into orders? A failed gate
raises :class:`SignalGateError` (or returns a reason, for the late-run gate), and the trader
places **no orders** and sends a notifier message. There is no fallback engine and no repair:
a bad input stops the run, because a wrong order is worse than a missed one (missed exits
self-heal through ``_missed_exits``; a wrong order does not).

The bounds below are ESTIMATES tuned on the 120-session live golden (P0): the universe is
500-502 loaded symbols, a session has at most 37 buys and 127 sells, and the last bar of a
healthy tape is complete for every loaded symbol. They are expected to be tuned from the
dry-run record (spec 11).

Incomplete-bar rule (the distinction the spec asks for)
-------------------------------------------------------
The 12:45 PT bar is a partial intraday bar, and a data vendor can lag on it: a snapshot day of
the research tape had a close for 1 of 646 symbols on its final date. That is a bar that is
*not yet complete*, not a corrupt tape, and the two need different handling:

* **Not-yet-complete bar**: the final bar is dated today (ET) or later, its coverage is below
  the floor, **and** the bar before it is fully healthy. The final bar is dropped and the
  signals are computed on the previous completed session (the trader's freshness gate already
  allows that lag). The result records ``trimmed_partial_bar`` so the run log shows it.
* **Bad tape**: anything else below the floor (the thin bar is an older date, or the previous
  bar is thin too, or too few symbols loaded, or duplicates, weekend stamps, too little
  history). The run stops.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import pandas as pd

from ggTrader.data.bars import Bars

PT = ZoneInfo("America/Los_Angeles")
#: Signals ready after this Pacific time place no orders: the market closes at 13:00 PT, and a
#: DAY order sent after the close queues to the next open (the slippage the 12:45 cron avoids).
LATE_RUN_CUTOFF = time(12, 55)
#: Margin kept before the broker-reported close when it is earlier than the cutoff (half days).
CLOSE_MARGIN = pd.Timedelta(minutes=5)


class SignalGateError(RuntimeError):
    """The signal inputs or outputs failed a sanity gate; the run must place no orders.

    Attributes:
        code: short machine-readable reason (``empty_frame``, ``duplicate_days``, ...).
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code


@dataclass(frozen=True)
class SanityConfig:
    """Bounds of the input and output sanity gates (all ESTIMATES; see the module docstring)."""

    #: Rows needed for the indicator seeds to wash out (a 365-day window has ~252 bars).
    min_bars: int = 240
    #: Symbols that must come back with at least one bar (live universe ~503, golden 500-502).
    min_universe: int = 450
    #: Loaded / requested symbols.
    min_loaded_share: float = 0.90
    #: Share of loaded symbols with a close on the final bar.
    min_last_bar_coverage: float = 0.95
    #: Absurd-output bounds (golden maxima 37 and 127 over 120 sessions).
    max_buys: int = 100
    max_sells: int = 250

    @classmethod
    def permissive(cls) -> SanityConfig:
        """Bounds that only reject an empty or structurally broken frame (small test panels)."""
        return cls(
            min_bars=1,
            min_universe=1,
            min_loaded_share=0.0,
            min_last_bar_coverage=0.0,
            max_buys=10**9,
            max_sells=10**9,
        )


@dataclass(frozen=True)
class BarsVerdict:
    """Outcome of :func:`vet_bars`: the bars to compute on, and what was checked."""

    bars: Bars
    last_bar_coverage: float
    trimmed_partial_bar: bool = False
    dropped_bar: str | None = None


def _coverage(close: pd.DataFrame, row: int) -> float:
    return float(close.iloc[row].notna().mean()) if close.shape[1] else 0.0


def vet_bars(bars: Bars, today_et: date, cfg: SanityConfig | None = None) -> BarsVerdict:
    """Check the live bars; trim a not-yet-complete final bar; raise on a bad tape.

    Args:
        bars: bars from the live loader (duplicates must still be present, not collapsed).
        today_et: the exchange-calendar date of this run, to recognise the in-progress session.
        cfg: gate bounds (default: production).

    Raises:
        SignalGateError: any structural problem or a final bar that is thin for a reason other
            than "today's bar is not complete yet".
    """
    cfg = cfg or SanityConfig()
    close = bars.close
    if close.empty or close.shape[1] == 0:
        raise SignalGateError("empty_frame", "the loader returned no bars")
    index = close.index
    if not index.is_unique:
        n_dup = int(index.duplicated().sum())
        raise SignalGateError("duplicate_days", f"{n_dup} duplicate day(s) in the bar index")
    if not index.is_monotonic_increasing:
        raise SignalGateError("unsorted_index", "bar index is not sorted ascending")
    weekend = index[index.day_of_week >= 5]
    if len(weekend):
        raise SignalGateError(
            "weekend_bars", f"{len(weekend)} weekend stamp(s), e.g. {weekend[0].date()}"
        )
    loaded = close.shape[1]
    asked = max(len(bars.requested), loaded)
    if loaded < cfg.min_universe or loaded / asked < cfg.min_loaded_share:
        raise SignalGateError(
            "small_universe",
            f"{loaded} of {asked} requested symbols loaded (need >= {cfg.min_universe} and "
            f">= {cfg.min_loaded_share:.0%})",
        )
    if len(close) < cfg.min_bars:
        raise SignalGateError(
            "short_history", f"{len(close)} bars (need >= {cfg.min_bars} for the seeds to wash out)"
        )

    cov_last = _coverage(close, -1)
    if cov_last >= cfg.min_last_bar_coverage:
        return BarsVerdict(bars=bars, last_bar_coverage=cov_last)

    last_day = close.index[-1].date()
    cov_prev = _coverage(close, -2) if len(close) > 1 else 0.0
    in_progress = last_day >= today_et
    if in_progress and cov_prev >= cfg.min_last_bar_coverage:
        trimmed = _drop_last_bar(bars)
        return BarsVerdict(
            bars=trimmed,
            last_bar_coverage=cov_prev,
            trimmed_partial_bar=True,
            dropped_bar=str(last_day),
        )
    raise SignalGateError(
        "last_bar_coverage",
        f"final bar {last_day} has closes for {cov_last:.1%} of {loaded} symbols "
        f"(floor {cfg.min_last_bar_coverage:.0%}); previous bar {cov_prev:.1%}; "
        f"{'today' if in_progress else 'not today'}: treated as a bad tape",
    )


def _drop_last_bar(bars: Bars) -> Bars:
    """The same bars without the final row (a not-yet-complete session)."""
    cut = {name: bars.frame(name).iloc[:-1] for name in ("close", "open", "high", "low", "volume")}
    last_valid = cut["close"].apply(lambda col: col.last_valid_index())
    last_valid.name = "last_valid_bar"
    return replace(bars, last_valid_bar=last_valid, **cut)


def vet_signal_counts(n_buys: int, n_sells: int, cfg: SanityConfig | None = None) -> None:
    """Refuse an absurd number of signals (a degenerate all-True frame would buy alphabetically).

    Raises:
        SignalGateError: buys or sells exceed the configured bounds.
    """
    cfg = cfg or SanityConfig()
    if n_buys > cfg.max_buys or n_sells > cfg.max_sells:
        raise SignalGateError(
            "absurd_signal_count",
            f"{n_buys} buys / {n_sells} sells exceed the bounds "
            f"({cfg.max_buys} / {cfg.max_sells}); the golden maxima are 37 / 127",
        )


def late_run_reason(now: datetime, next_close: str | None = None) -> str | None:
    """Why signals ready at ``now`` are too late to trade, or ``None`` if they are on time.

    Args:
        now: timezone-aware time the signals became ready.
        next_close: the broker clock's ``next_close`` (any ``pd.Timestamp``-parsable text);
            when it falls earlier the same Pacific day (a half day) the cutoff moves to
            five minutes before it. An unparsable value is ignored.
    """
    local = now.astimezone(PT)
    cutoff = datetime.combine(local.date(), LATE_RUN_CUTOFF, tzinfo=PT)
    if next_close:
        try:
            close_at = pd.Timestamp(next_close).tz_convert(PT)
        except (ValueError, TypeError):
            close_at = None
        if close_at is not None and close_at.date() == local.date():
            cutoff = min(cutoff, (close_at - CLOSE_MARGIN).to_pydatetime())
    if local > cutoff:
        return (
            f"signals ready at {local:%H:%M} PT, after the {cutoff:%H:%M} PT cutoff; "
            "a DAY order now would queue to the next open"
        )
    return None
