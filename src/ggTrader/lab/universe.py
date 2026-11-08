"""Vectorised point-in-time (PIT) universe masks and optional exit policies; pure, no I/O.

Replaces the per-bar loop ``lab.sweep.pit_entry_mask`` (one ``lab.data.eligible_at`` call per
bar, O(T^2), about 65% of profiled walk-forward time) with one array computation. Re-derived
from the S6 prototype and tested against ``eligible_at`` itself (spec
docs/superpowers/specs/2026-11-06-vectorbt-engine-design.md, sections 5.1, 5.7, OD5-OD7).

Eligibility on bar ``d`` for symbol ``s`` is exactly ``eligible_at``'s rule, in order:

1. ``s`` is a member on ``d``: the constituent snapshot in force is the latest row dated on or
   before ``d`` (same-day effective), tickers normalised ``BRK.B`` -> ``BRK-B``. A bar earlier
   than the first snapshot raises, as ``sp500_members_asof`` does.
2. ``s`` is a column of the price frame and has at least ``min_history_bars`` non-NaN closes up
   to and including ``d`` (NaN gaps do not count; a dead ticker keeps its count).
3. Optional ``max_stocks``: of the symbols passing 1-2 on that bar, keep the first
   ``max_stocks`` in alphabetical order.

Not reproduced: the sector cap (``max_sector_count``), which the sweep mask path never applied
(it lives in ``EnsembleSignal.select``), and the rebalance-granularity ``eligibility_mask``.

Exit policies are opt-in helpers that change numbers by policy and are off by default:

* :func:`removal_exits` (OD6): exit at the first bar after a symbol leaves the index. Today the
  mask gates entries only and a position runs to its normal exit.
* :func:`terminal_exits` (OD7): exit at the last valid bar of a series whose data has ended, and
  drop a same-bar entry (vectorbt ignores an entry and exit on one bar, which would leave the
  position open at its last price forever). The fill is the last traded price, which
  **overstates recovery for failures**: results using it are optimistic, and the returned
  metadata says so.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from ggTrader.data.core.index_constituents import normalize_yf_ticker
from ggTrader.data.membership import MembershipSource, age_since_verified, resolve_source

#: Constituent-file age (days since its last snapshot row) beyond which the builder warns.
#: ESTIMATE per spec 5.7 and OD5; tune from dry-run data.
DEFAULT_MAX_AGE_DAYS = 45
STALE_POLICIES = ("warn", "raise", "ignore")


class StaleConstituentsWarning(UserWarning):
    """The constituent file's last snapshot is older than the allowed age."""


class StaleConstituentsError(RuntimeError):
    """Raised instead of the warning when ``on_stale='raise'``."""


# --------------------------------------------------------------------------- age check


def check_constituent_age(
    asof: Any,
    membership_history: Optional[pd.DataFrame] = None,
    *,
    max_age_days: int = DEFAULT_MAX_AGE_DAYS,
    on_stale: str = "warn",
    membership: Optional[MembershipSource] = None,
) -> dict[str, Any]:
    """How old is the constituent list at ``asof``? Warn, raise or ignore when too old.

    With a bare ``membership_history`` table (synthetic tests, old callers) age is ``asof`` minus
    the date of the last snapshot row: a quiet stretch with no index changes also ages it, so this
    is a prompt to refresh (OD5), not proof of error.

    With a source (``membership``, else the one selected by ``GGT_MEMBERSHIP``) the spec 3.4
    rule applies: ``age_days`` is still ``asof - coverage_end`` (information only) and the list is
    *stale* when ``asof - verified_at`` exceeds ``max_age_days``, i.e. when our copy was last
    proved equal to upstream too long ago. A vintage verified after ``asof`` is never stale, so
    replaying an old backtest is not penalised.

    Returns a JSON-able record for the run provenance.
    """
    if on_stale not in STALE_POLICIES:
        raise ValueError(f"on_stale must be one of {STALE_POLICIES}, got {on_stale!r}")
    ts = pd.Timestamp(asof)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    extra: dict[str, Any] = {}
    if membership_history is not None:
        last = membership_history.index[-1]
        age = int((ts - last).days)
        stale = age > max_age_days
        what, shown = "last snapshot", age
        record_last = str(last.date())
    else:
        vintage = resolve_source(membership).vintage()
        age = (ts.date() - vintage.coverage_end).days
        verified = age_since_verified(vintage, ts.date())
        stale = verified > max_age_days
        what, shown = "last verification", verified
        record_last = str(vintage.coverage_end)
        extra = {
            "age_since_verified_days": verified,
            "verified_at": vintage.verified_at.isoformat(),
            "membership_sha256": vintage.content_sha256,
        }
    record = {
        "last_snapshot": record_last,
        "asof": str(ts.date()),
        "age_days": age,
        "max_age_days": max_age_days,
        "stale": stale,
        **extra,
    }
    if stale and on_stale != "ignore":
        msg = (
            f"constituent file is stale: {what} is {shown} days before {record['asof']} "
            f"(limit {max_age_days}; last snapshot {record_last}); refresh per OD5"
        )
        if on_stale == "raise":
            raise StaleConstituentsError(msg)
        warnings.warn(msg, StaleConstituentsWarning, stacklevel=3)
    return record


# --------------------------------------------------------------------------- membership


def _utc_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Naive stamps are read as UTC (the stored convention); aware stamps are converted."""
    return index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")


def _check_columns(symbols: Sequence[str]) -> list[str]:
    cols = list(symbols)
    if len(set(cols)) != len(cols):
        raise ValueError("duplicate symbols in columns")
    return cols


def membership_mask(
    index: pd.DatetimeIndex,
    symbols: Sequence[str],
    membership_history: Optional[pd.DataFrame] = None,
    *,
    membership: Optional[MembershipSource] = None,
) -> pd.DataFrame:
    """True where ``symbol`` is a PIT index member on the bar (``dates x symbols``, bool).

    ``membership_history`` is the ``load_sp500_history()`` table (UTC date index, a ``tickers``
    list per row); without it the table comes from ``membership`` (else the source selected by
    ``GGT_MEMBERSHIP``; there is no file default). Column names are matched as given; only the
    history's tickers are normalised, as in ``eligible_at``.
    """
    hist = (
        membership_history
        if membership_history is not None
        else resolve_source(membership).history()
    )
    cols = _check_columns(symbols)
    idx = _utc_index(index)
    if len(idx) and idx[0] < hist.index[0]:
        raise ValueError(
            f"asof={idx[0]} predates constituent history (starts {hist.index[0].date()})"
        )
    pos = hist.index.searchsorted(idx, side="right") - 1  # snapshot row in force per bar
    used = np.unique(pos)
    col_pos = {c: j for j, c in enumerate(cols)}
    table = np.zeros((len(used), len(cols)), dtype=bool)
    for row, p in enumerate(used):
        for ticker in hist["tickers"].iloc[p]:
            j = col_pos.get(normalize_yf_ticker(ticker))
            if j is not None:
                table[row, j] = True
    return pd.DataFrame(table[np.searchsorted(used, pos)], index=index, columns=cols)


def static_membership_mask(
    index: pd.DatetimeIndex, symbols: Sequence[str], members: Sequence[str]
) -> pd.DataFrame:
    """Constant membership for a non-PIT universe (``snapshot_members``): same list every bar."""
    cols = _check_columns(symbols)
    wanted = {normalize_yf_ticker(m) for m in members}
    row = np.array([c in wanted for c in cols], dtype=bool)
    return pd.DataFrame(
        np.broadcast_to(row, (len(index), len(cols))).copy(), index=index, columns=cols
    )


# --------------------------------------------------------------------------- PIT mask


def pit_mask(
    close: pd.DataFrame,
    membership_history: Optional[pd.DataFrame] = None,
    *,
    membership: Optional[MembershipSource] = None,
    min_history_bars: int = 400,
    max_stocks: Optional[int] = None,
    static_members: Optional[Sequence[str]] = None,
    max_age_days: int = DEFAULT_MAX_AGE_DAYS,
    on_stale: str = "warn",
) -> pd.DataFrame:
    """Daily PIT eligibility for every bar and symbol; equals ``eligible_at`` on every date.

    Args:
        close: ``dates x symbols`` closes (NaN where no bar). Index must be strictly increasing.
        membership_history: PIT constituent table (tests); without it, ``membership``.
        membership: the source of the table (spec 3.3); ``None`` uses the one selected by
            ``GGT_MEMBERSHIP`` and raises ``MembershipSourceNotSelected`` if none is.
        min_history_bars: required non-NaN closes up to the bar (lab default 400; live/sweeps 60).
        max_stocks: keep only the first ``max_stocks`` eligible symbols alphabetically per bar.
        static_members: for a non-PIT universe, the fixed member list (skips the age check).
        max_age_days, on_stale: constituent-file age policy, see :func:`check_constituent_age`.

    Returns:
        Boolean frame shaped like ``close`` (same index and column order).
    """
    if not close.index.is_monotonic_increasing or not close.index.is_unique:
        raise ValueError("close index must be unique and strictly increasing")
    if max_stocks is not None and max_stocks < 0:
        raise ValueError("max_stocks must be >= 0 (a negative slice would drop the last names)")
    cols = list(close.columns)
    if static_members is not None:
        member = static_membership_mask(close.index, cols, static_members)
    else:
        if len(close):
            check_constituent_age(
                close.index[-1],
                membership_history,
                max_age_days=max_age_days,
                on_stale=on_stale,
                membership=membership,
            )
        member = membership_mask(close.index, cols, membership_history, membership=membership)

    enough = np.cumsum(close.notna().to_numpy(), axis=0) >= min_history_bars
    mask = member.to_numpy() & enough
    if max_stocks is not None:
        order = sorted(range(len(cols)), key=cols.__getitem__)  # eligible_at sorts by name
        ranked = mask[:, order]
        mask[:, order] = ranked & (np.cumsum(ranked, axis=1) <= max_stocks)
    return pd.DataFrame(mask, index=close.index, columns=close.columns)


# --------------------------------------------------------------------------- exit policies


def removal_exits(member: pd.DataFrame) -> pd.DataFrame:
    """Exit signal on the first bar a symbol is no longer a member (OD6, optional).

    Pass the :func:`membership_mask`, not a capped :func:`pit_mask`: a ``max_stocks`` cut can
    drop a symbol without any index change. A symbol that is a member on the first bar has no
    removal event; re-entry after removal is allowed.
    """
    before = member.shift(1, fill_value=False).astype(bool)
    return before & ~member.astype(bool)


@dataclass(frozen=True, eq=False)
class TerminalExits:
    """Result of :func:`terminal_exits`.

    Attributes:
        exits: True on each dead series' last valid bar.
        entries: the supplied entries with the terminal-bar entry removed (``None`` if none given).
        meta: JSON-able record: the dead-ticker list, the suppressed-entry count and a ``biases``
            block labelling both the optimistic exit price and the entry suppression (which, like
            the forced exit, uses knowledge that the series ends).
    """

    exits: pd.DataFrame
    entries: Optional[pd.DataFrame]
    meta: dict[str, Any]


def terminal_exits(
    close: pd.DataFrame,
    entries: Optional[pd.DataFrame] = None,
    *,
    min_trailing_nan_bars: int = 1,
) -> TerminalExits:
    """Force an exit at each ended series' last valid bar (OD7, optional, optimistic).

    A series is "dead" when its last valid close is followed by at least
    ``min_trailing_nan_bars`` NaN bars up to the end of the index (all-NaN columns are ignored).
    The default of 1 treats any series missing on the final bar as dead; drop a ragged final
    bar (a few symbols updated) from ``close`` first, or raise the threshold. Interior NaN gaps
    never count. If ``entries`` is given, an entry on the same bar as the forced exit is removed.

    Two biases, both recorded under ``meta['biases']``:

    * **Exit price (optimistic).** The exit fills at the last traded price. For delistings after
      failure that overstates what a holder recovered.
    * **Same-bar entry suppression (favourable lookahead).** Removing an entry on the dead
      series' last bar avoids buying a dying name, but only because we know, on bar t, that t is
      the last bar of the series. The forced exit itself uses that same knowledge. The
      suppression is an accounting repair, acceptable for a stated OD7 run, and the count of
      suppressed entries must be reported with the result.
    """
    valid = close.notna().to_numpy()
    n_bars = valid.shape[0]
    has_any = valid.any(axis=0)
    last_pos = n_bars - 1 - valid[::-1].argmax(axis=0)
    trailing = (n_bars - 1) - last_pos
    dead = has_any & (trailing >= max(1, min_trailing_nan_bars))

    exit_arr = np.zeros(valid.shape, dtype=bool)
    dead_cols = np.flatnonzero(dead)
    exit_arr[last_pos[dead_cols], dead_cols] = True
    exits = pd.DataFrame(exit_arr, index=close.index, columns=close.columns)

    kept = None
    suppressed = 0
    if entries is not None:
        both = entries.to_numpy(dtype=bool) & exit_arr
        suppressed = int(both.sum())
        kept = pd.DataFrame(
            entries.to_numpy(dtype=bool) & ~exit_arr, index=entries.index, columns=entries.columns
        )
    meta = {
        "policy": "terminal_exit_at_last_valid_bar",
        "optimistic": True,
        "note": "exit at last traded price overstates recovery for failed or delisted names",
        "lookahead": True,
        "uses_knowledge_of_series_end": ["exit_on_last_bar", "entry_suppression"],
        "lookahead_note": (
            "the forced exit and the suppression of a same-bar entry use knowledge, on bar t, "
            "that the series ends at t; a live system cannot know this in advance"
        ),
        "biases": {
            "exit_price": {
                "direction": "optimistic",
                "uses_knowledge_of_series_end": True,
                "what": "exit at the last traded price overstates recovery for failed names",
            },
            "same_bar_entry_suppression": {
                "direction": "favourable",
                "uses_knowledge_of_series_end": True,
                "entries_suppressed": suppressed,
                "what": (
                    "lookahead: an entry on the series' last bar is dropped because the series "
                    "is known to end there, avoiding a purchase of a dying name"
                ),
            },
        },
        "min_trailing_nan_bars": min_trailing_nan_bars,
        "n_dead": int(dead.sum()),
        "dead_symbols": {
            str(close.columns[j]): str(close.index[last_pos[j]].date()) for j in dead_cols
        },
        "same_bar_entries_suppressed": suppressed,
    }
    return TerminalExits(exits=exits, entries=kept, meta=meta)


# --------------------------------------------------------------------------- reporting


def coverage_by_year(member: pd.DataFrame, close: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Per calendar year: member-days and the share that has a close that day (survivorship gap).

    ``close`` is reindexed to ``member``'s columns, so requested symbols with no bars at all
    (``Bars.missing_symbols``) count as uncovered member-days.
    """
    have = close.reindex(index=member.index, columns=member.columns).notna().to_numpy()
    mem = member.to_numpy(dtype=bool)
    years = np.asarray(_utc_index(member.index).year)
    out: dict[str, dict[str, float]] = {}
    for year in np.unique(years):
        rows = years == year
        total = int(mem[rows].sum())
        covered = int((mem[rows] & have[rows]).sum())
        out[str(int(year))] = {
            "member_days": total,
            "coverage_pct": round(100.0 * covered / total, 2) if total else 0.0,
        }
    return out
