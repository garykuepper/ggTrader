"""Generate today's ensemble signals from a given equity universe on the vectorbt engine."""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from ggTrader.data.core.index_constituents import normalize_yf_ticker, universe_members_asof
from ggTrader.data.membership import (
    MembershipError,
    MembershipIntegrityError,
    MembershipSource,
    MembershipUnavailable,
    resolve_source,
    selected_source,
)
from ggTrader.data.membership_gate import (
    CODE_CONFIG,
    CODE_UNAVAILABLE,
    decide_gate,
    gate_applies,
)
from ggTrader.engine import selfcheck as engine_selfcheck
from ggTrader.lab.data import fetch_stock_ohlcv
from ggTrader.paper.gates import (
    SanityConfig,
    SignalGateError,
    vet_bars,
    vet_signal_counts,
)
from ggTrader.paper.live_bars import bars_from_legacy_frame
from ggTrader.paper.live_signals import MIN_HISTORY_BARS, live_ensemble_frames
from ggTrader.paper.overlay import (
    SLEEVE_UNIVERSES,
    compute_sleeve_curve,
    compute_weights_and_scale,
    should_rebalance,
)
from ggTrader.paper.persist import get_rebalance_state, save_rebalance_state

_log = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York")

#: Production bounds of the input/output sanity gates (see ``paper/gates.py``).
SANITY = SanityConfig()

#: Symbols no sleeve trades but the lab benchmarks against or builds
#: sleeves from. Nothing else on the box fetches them daily, so the
#: paper run keeps their tape alive (SPY's went dead 2026-08-21 when the
#: old benchmark-only writer was retired; TLT/GLD/DBC stalled 2026-07-20).
BENCHMARK_SYMBOLS: tuple[str, ...] = ("SPY", "TLT", "GLD", "DBC", "IEF")


def refresh_benchmark_tape(lookback_days: int = 30) -> list[str]:
    """Re-fetch the trailing window of the benchmark/ETF tape and upsert it.

    Deliberately bypasses `CachedYFinanceLoader.fetch_ohlcv`'s freshness
    check: this runs at 12:45 PT while the session is open, so the newest
    bar written is always partial, and the freshness check would judge it
    current the next day and never refetch it. Overwriting the trailing
    window every run (the writer upserts on timestamp/symbol) means only
    the current day's bar is ever partial. Never raises: this is a side job
    of the live run and must not block trading. Returns the symbols that
    came back with data (empty on any failure).
    """
    today = pd.Timestamp.now(tz="UTC").normalize()
    start = today - pd.Timedelta(days=lookback_days)
    try:
        from ggTrader.data.live.cached_yfinance_loader import CachedYFinanceLoader
        from ggTrader.data.live.yfinance_loader import YFinanceDataLoader

        df = YFinanceDataLoader().fetch_ohlcv(
            list(BENCHMARK_SYMBOLS), "1d", start_date=start, end_date=today
        )
        if df.empty:
            return []
        CachedYFinanceLoader()._cache_to_db(df, "1d")
    except Exception as exc:
        _log.warning("benchmark tape refresh failed (non-fatal): %s", exc)
        return []
    return sorted(df.columns.get_level_values(0).unique().tolist())


def require_rust_engine() -> dict[str, Any]:
    """Prove the Rust engine is the one that runs, before any data is fetched.

    The trader has no fallback engine (spec 5.5, 8.3): numba would cost 20-60 s of JIT inside
    a 15-minute window, so any engine problem means no orders.

    Returns:
        The ``engine.selfcheck`` result (versions, engine, Rust call count).

    Raises:
        SignalGateError: the self-check failed or the configured engine is not ``rust``.
    """
    try:
        result = engine_selfcheck.selfcheck()
    except Exception as exc:  # noqa: BLE001 - any engine problem must stop the run
        raise SignalGateError("engine_selfcheck", f"{type(exc).__name__}: {exc}") from exc
    if result.get("engine") != "rust":
        raise SignalGateError(
            "engine_not_rust",
            f"the trader requires the rust engine, the self-check ran {result.get('engine')!r}",
        )
    return result


def live_membership_source(spec: str | None) -> MembershipSource:
    """The S&P 500 list the live path reads, or a fail-closed ``SignalGateError``.

    An unreadable database, an empty table or a vintage that fails its integrity check is
    ``membership_unavailable``: no orders, and never a fallback to the frozen file. A missing
    or malformed selection still raises its own ``MembershipError`` (a configuration error).
    """
    try:
        return selected_source(spec)
    except (MembershipUnavailable, MembershipIntegrityError) as exc:
        raise SignalGateError(
            CODE_UNAVAILABLE,
            f"{exc}. The live trader never falls back to the frozen file. Fix: check the "
            "database, run `ggt db membership status`, and load a vintage with "
            "`ggt db load-sp500` if the table is empty",
        ) from None


def _membership_gate(source: MembershipSource, asof: date, wall_clock: bool) -> dict[str, Any]:
    """Apply the freshness gate of spec 3.4/D8; raise ``SignalGateError`` on a refusal."""
    if not gate_applies(source, wall_clock=wall_clock):
        return {"applied": False, "level": "skipped"}
    try:
        decision = decide_gate(source.vintage(), asof)
    except MembershipError as exc:
        raise SignalGateError(CODE_CONFIG, str(exc)) from exc
    if decision.level == "refuse":
        raise SignalGateError(decision.code or CODE_UNAVAILABLE, decision.message)
    if decision.level == "warn":
        _log.warning("membership list warning: %s", decision.message)
    return decision.as_dict()


def _fetch_live_frame(
    symbols: list[str], start: pd.Timestamp, end: pd.Timestamp, persist_tape: bool
) -> pd.DataFrame:
    """The live loader's ``(symbol, field)`` frame; ``persist_tape=False`` never writes a row.

    The default is the same DB-first loader chain the trader has always used (it upserts the
    bars it fetches into ``ohlcv``). The non-persisting path, used by the read-only dry run,
    asks yfinance directly and keeps the result in memory only.
    """
    if persist_tape:
        return fetch_stock_ohlcv(symbols, start=str(start.date()), end=str(end.date()))
    from ggTrader.data.live.yfinance_loader import YFinanceDataLoader

    return YFinanceDataLoader().fetch_ohlcv(symbols, "1d", start_date=start, end_date=end)


def generate_signals(
    universe: str = "sp500",
    lookback_days: int = 365,
    *,
    sanity: SanityConfig | None = None,
    now: datetime | None = None,
    persist_tape: bool = True,
    membership: MembershipSource | None = None,
) -> dict:
    """Fetch recent data for a PIT universe and return today's ensemble signals.

    The signals come from the validated ``Ensemble.build()`` with the one-combination grid of the
    live defaults (spec OD14: five voters, ``min_agree=2``, ``min_agree_exit=2``, divergence
    window 20, volume multiple 2.0), read off the last bar. P0's live golden (120 sessions of
    buys, sells and ``last_exit``) is its acceptance test (``tests/paper/test_live_golden.py``).

    ``lookback_days=365`` (~252 bars) lets the indicator seeds wash out, so the signals match the
    backtest's long-history indicators (research/paper-system/indicator-warmup/
    2026-09-28-live-indicator-warmup.md).

    Fails closed: the engine self-check runs first, then the input and output sanity gates
    (:mod:`ggTrader.paper.gates`); any failure raises :class:`SignalGateError`, which the trader
    turns into "no orders" plus a notifier message. There is no legacy fallback.

    Args:
        universe: universe name (``sp500`` for the deployed core).
        lookback_days: calendar-day length of the bar window.
        sanity: gate bounds (default: production bounds).
        now: the run time (default: the wall clock); injectable for tests.
        persist_tape: ``False`` keeps every fetched bar in memory only (read-only dry run).
        membership: the S&P 500 list source; the trader passes its explicit choice. ``None`` means
            the source named by ``GGT_MEMBERSHIP`` (an error if unset), never a file default.
            A wall-clock run on the latest ``db`` vintage passes the membership gate first
            (:mod:`ggTrader.data.membership_gate`); fixtures, pinned vintages and an injected
            ``now`` skip it.

    Returns:
        Dict with keys buys, sells, last_exit, as_of, universe_size, engine, bars,
        membership_gate (the gate's decision, or ``{"applied": False}``).

    Raises:
        SignalGateError: any gate above, including a stale or lagging membership list.
    """
    cfg = sanity or SANITY
    wall_clock = now is None
    now = now or datetime.now(timezone.utc)
    engine_info = require_rust_engine()

    today = pd.Timestamp(now).tz_convert("UTC").normalize()
    today_et = pd.Timestamp(now).tz_convert(_ET).date()
    start = today - pd.Timedelta(days=lookback_days)

    source = resolve_source(membership)
    gate = _membership_gate(source, today.date(), wall_clock)
    members = universe_members_asof(universe, today, membership=source)
    symbols = sorted({normalize_yf_ticker(t) for t in members})

    ohlcv = _fetch_live_frame(symbols, start, today, persist_tape)
    bars = bars_from_legacy_frame(ohlcv, symbols)
    verdict = vet_bars(bars, today_et, cfg)
    bars = verdict.bars
    if verdict.trimmed_partial_bar:
        _log.warning(
            "final bar %s is not complete yet (coverage below the floor); signals computed on %s",
            verdict.dropped_bar,
            bars.close.index[-1].date(),
        )
        ohlcv = ohlcv.iloc[:-1]

    last_bar = bars.close.index[-1]
    counts = bars.close.notna().sum()
    eligible = sorted(counts[counts >= MIN_HISTORY_BARS].index.tolist())
    if not eligible:
        raise SignalGateError(
            "no_eligible_symbols", f"no symbol has {MIN_HISTORY_BARS} valid closes"
        )

    all_entries, all_exits = live_ensemble_frames(bars)
    entries, exits = all_entries[eligible], all_exits[eligible]

    last_entries = entries.loc[last_bar]
    last_exits = exits.loc[last_bar]
    buys = sorted(last_entries[last_entries].index.tolist())
    sells = sorted(last_exits[last_exits].index.tolist())
    vet_signal_counts(len(buys), len(sells), cfg)

    # Exits are crossover events, and `sells` sees only the last (intraday)
    # bar, so a cross completing after the run or on a skipped day is lost.
    # The trader compares these dates with entry dates to catch those up.
    latest = exits.iloc[::-1].idxmax()[exits.any()]
    last_exit = {sym: str(ts.date()) for sym, ts in latest.items()}

    return {
        "buys": buys,
        "sells": sells,
        "as_of": str(last_bar.date()),
        "universe_size": len(bars.symbols),
        "membership_gate": gate,
        "last_exit": last_exit,
        "engine": {k: engine_info.get(k) for k in ("engine", "vectorbt", "vectorbt_rust")},
        "bars": {
            "n_bars": len(bars.close),
            "requested": len(bars.requested),
            "missing_symbols": len(bars.missing_symbols),
            "last_bar_coverage": round(verdict.last_bar_coverage, 4),
            "trimmed_partial_bar": verdict.trimmed_partial_bar,
            "dropped_bar": verdict.dropped_bar,
            "eligible": len(eligible),
        },
    }


def generate_blended_signals(membership: MembershipSource | None = None) -> dict:
    """Generate today's signals for all three sleeves, recomputing the
    inverse-vol/target-vol overlay monthly. On any failure to recompute
    (e.g. an OHLCV fetch error on a rebalance date), falls back to the last
    stored weights/scale rather than raising."""
    today = pd.Timestamp.now(tz="UTC").normalize()
    sleeves = {
        universe: generate_signals(universe=universe, membership=membership)
        for universe in SLEEVE_UNIVERSES
    }

    state = get_rebalance_state()
    rebalanced_today = False
    fallback_used = False

    if should_rebalance(state["rebalance_date"] if state else None, today):
        try:
            curves = {
                u: compute_sleeve_curve(u, today, membership=membership) for u in SLEEVE_UNIVERSES
            }
            weights, scale = compute_weights_and_scale(curves)
        except Exception:
            if state is None:
                raise  # no fallback available on the very first run
            weights, scale = state["weights"], state["scale"]
            fallback_used = True
        else:
            # Only a recompute failure (fetch/compute) falls back to stale
            # state. A persistence failure here is a genuine error and
            # should propagate rather than be swallowed into fallback_used.
            save_rebalance_state(str(today.date()), weights, scale)
            rebalanced_today = True
    else:
        weights, scale = state["weights"], state["scale"]

    return {
        "sleeves": sleeves,
        "weights": weights,
        "scale": scale,
        "rebalanced_today": rebalanced_today,
        "fallback_used": fallback_used,
    }


def generate_core_signals(
    persist_tape: bool = True, membership: MembershipSource | None = None
) -> dict:
    """Generate today's signals for the standalone SP500 core strategy,
    wrapped in the same shape `generate_blended_signals()` returns so
    `trader.py`'s sleeve-iteration/buy-sizing logic needs no changes.

    Deployed 2026-09 in place of the 3-sleeve blend: the corrected-tape
    pinned-window re-baseline
    (`research/audits/rebaseline/_rebaseline_corrected_tape_20260822.json`) shows the
    blend underperforming this standalone core (Sharpe 0.69 vs 0.99, third
    independent confirmation) -- see `docs/next_steps.md`. Kept alongside
    `generate_blended_signals` (not deleted) since the blend's WFO/research
    infrastructure (`ggt lab --blend`) is still valid tooling for any future
    diversification-sleeve candidate that actually clears the bar.

    Args:
        persist_tape: ``False`` (read-only dry run) keeps every fetched bar in memory only.
        membership: the S&P 500 list source, as in :func:`generate_signals`.
    """
    core_signals = generate_signals(
        universe="sp500", persist_tape=persist_tape, membership=membership
    )
    return {
        "sleeves": {"sp500": core_signals},
        "weights": {"sp500": 1.0},
        "scale": 1.0,
        "rebalanced_today": False,
        "fallback_used": False,
    }
