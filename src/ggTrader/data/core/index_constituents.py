"""Point-in-time S&P 500 constituent membership.

Source: fja05680/sp500 "S&P 500 Historical Components & Changes (Updated)"
— one snapshot row per membership-change date (1996 -> present), with tickers
mapped to *current* symbology so they resolve against yfinance. Since the
membership-database work (spec docs/superpowers/specs/2026-11-07-membership-database-design.md)
the list is read through a :class:`~ggTrader.data.membership.MembershipSource`
(the database, or a frozen fixture file), chosen by name and never guessed:
every lookup here takes ``membership=`` (or an already-loaded ``history`` table)
and, when given neither, uses the source named by ``GGT_MEMBERSHIP`` or raises
``MembershipSourceNotSelected``. Nothing in ``src/`` writes a membership file;
``ggt db load-sp500`` is the only writer.

``load_sp500_history(csv_path)`` with a path stays a pure file reader, for
frozen studies that name their file (the event-timing audit).

Honest limitation: point-in-time membership removes selection-survivorship
(picking from today's index), but yfinance lacks price data for many delisted
tickers, so residual survivorship remains in the *data*. Quantify it per
period with ``coverage_stats()`` rather than ignoring it.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional

import pandas as pd

if TYPE_CHECKING:  # data.membership imports this module's siblings; import it lazily at runtime
    from ggTrader.data.membership import MembershipSource

PROJECT_ROOT = Path(__file__).resolve().parents[4]
#: The frozen vintage-1 file (fixture ``sp500_v1_2026-06-02``); its bytes are pinned by studies.
SP500_HISTORY_CSV = PROJECT_ROOT / "data" / "universe" / "sp500_constituents_history.csv.gz"


def load_sp500_history(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """The snapshot table: DatetimeIndex (UTC) -> sorted list of tickers.

    With ``csv_path`` this reads that file (pure file reader, no source selection). Without it
    the table comes from the source selected by ``GGT_MEMBERSHIP``
    (:func:`~ggTrader.data.membership.selected_source`); there is no implicit file default.
    """
    if csv_path is None:
        from ggTrader.data.membership import selected_source

        return selected_source().history()
    if not csv_path.exists():
        raise FileNotFoundError(f"S&P 500 constituent history not found at {csv_path}.")
    raw = pd.read_csv(csv_path)
    if list(raw.columns) != ["date", "tickers"]:
        raise ValueError(f"Unexpected constituent CSV columns: {list(raw.columns)}")
    return pd.DataFrame(
        {
            "tickers": [
                sorted(t.strip() for t in row.split(",") if t.strip()) for row in raw["tickers"]
            ]
        },
        index=pd.DatetimeIndex(pd.to_datetime(raw["date"]), tz="UTC", name="date"),
    ).sort_index()


def _history(
    history: Optional[pd.DataFrame], membership: Optional["MembershipSource"]
) -> pd.DataFrame:
    """The table to look up in: an explicit ``history``, else the source's (or selected) one."""
    if history is not None:
        return history
    from ggTrader.data.membership import resolve_source

    return resolve_source(membership).history()


def sp500_members_asof(
    asof: pd.Timestamp,
    history: Optional[pd.DataFrame] = None,
    membership: Optional["MembershipSource"] = None,
) -> List[str]:
    """Members at ``asof``: the latest snapshot row dated on or before it."""
    hist = _history(history, membership)
    ts = pd.Timestamp(asof)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    eligible = hist.index[hist.index <= ts]
    if len(eligible) == 0:
        raise ValueError(
            f"asof={asof} predates constituent history (starts {hist.index[0].date()})"
        )
    return list(hist.loc[eligible[-1], "tickers"])


def all_members_between(
    start: pd.Timestamp,
    end: pd.Timestamp,
    history: Optional[pd.DataFrame] = None,
    membership: Optional["MembershipSource"] = None,
) -> List[str]:
    """Union of members over [start, end] — the data-download universe.

    Includes the snapshot in force at ``start`` (the latest row <= start), plus
    every change snapshot inside the window.
    """
    hist = _history(history, membership)

    def _utc(ts: pd.Timestamp) -> pd.Timestamp:
        ts = pd.Timestamp(ts)
        return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")

    start_ts, end_ts = _utc(start), _utc(end)
    union: set[str] = set(sp500_members_asof(start_ts, hist))
    in_window = hist[(hist.index > start_ts) & (hist.index <= end_ts)]
    for tickers in in_window["tickers"]:
        union.update(tickers)
    return sorted(union)


def normalize_yf_ticker(ticker: str) -> str:
    """Map index-style tickers to yfinance form ('BRK.B' -> 'BRK-B')."""
    return ticker.strip().upper().replace(".", "-")


def coverage_stats(members: Iterable[str], available: Iterable[str]) -> Dict[str, object]:
    """Quantify how much of a point-in-time universe has price data.

    ``available`` is compared in yfinance ticker form. The residual
    survivorship bias of a backtest is bounded by the missing fraction.
    """
    member_set = {normalize_yf_ticker(m) for m in members}
    avail_set = {normalize_yf_ticker(a) for a in available}
    missing = sorted(member_set - avail_set)
    n = len(member_set)
    return {
        "n_members": n,
        "n_with_data": n - len(missing),
        "coverage_pct": 100.0 * (n - len(missing)) / n if n else 0.0,
        "missing": missing,
    }


# ---------------------------------------------------------------------------
# Static-snapshot universe loader (Nasdaq-100, Russell 2000, etc.)
# ---------------------------------------------------------------------------

_SNAPSHOT_DIR = PROJECT_ROOT / "data" / "universe"


def _load_snapshot(filename: str) -> List[str]:
    """Load a newline-delimited ticker snapshot file."""
    path = _SNAPSHOT_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Universe snapshot not found: {path}")
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


_SNAPSHOT_REGISTRY: Dict[str, str] = {
    "nasdaq100": "nasdaq100_tickers_snapshot_2026-06-09.txt",
    "russell2000": "russell2000_tickers_snapshot_2026-06-09.txt",
    "midcap400": "midcap400_tickers_snapshot_2026-06-24.txt",
    # Fixed basket, not an index -- no point-in-time membership changes to
    # track, but reuses the snapshot-universe path since it's the same
    # "static ticker list" shape. Currency-hedged/unhedged ETF pairs for
    # fx_hedge_overlay: EWJ/DXJ (Japan, JPY, active since 2006) and
    # EZU/HEZU (Eurozone, EUR, active since 2014). Several other
    # single-country hedged ETFs (HEWG, HEWU) were delisted 2023-2024 --
    # deliberately excluded, only currently-active pairs included.
    "fx_hedge": "fx_hedge_tickers_snapshot_2026-07-20.txt",
    # Long-duration Treasury ETFs for fomc_drift (candidate A7).
    "fomc_treasury": "fomc_treasury_tickers_snapshot_2026-07-20.txt",
    # Liquid single-commodity ETFs for commodity_trend (candidate A3):
    # metals (GLD/SLV/CPER/PALL/PPLT/DBB), energy (USO/UNG/UGA), ags
    # (DBA/CORN/WEAT/SOYB/CANE). BAL and JJC were checked and found
    # delisted 2023 -- excluded. Broad multi-commodity baskets (DBC/GSG)
    # deliberately excluded from the cross-sectional universe itself --
    # they'd dilute the ranking signal, not add a distinct commodity.
    "commodity_trend": "commodity_trend_tickers_snapshot_2026-07-20.txt",
    # Treasury duration-bucket ETFs for treasury_curve (candidate A5):
    # SHY (1-3yr), IEF (7-10yr), TLT (20yr+) -- the register's own named
    # example for an ETF-approximation of Filipovic/Pelger/Ye's 4-factor
    # term-structure model.
    "treasury_curve": "treasury_curve_tickers_snapshot_2026-07-20.txt",
    # SPY core + TLT/GLD/PDBC sleeve + BIL off-asset for cross_asset_sleeve
    # (RESEARCH_SNAPSHOT §6 Tier 1 #1). PDBC, not DBC: DBC issues a K-1.
    "cross_asset_sleeve": "cross_asset_sleeve_tickers_snapshot_2026-09-28.txt",
}


def snapshot_members(universe: str) -> List[str]:
    """Return members from a static snapshot file.

    Unlike ``sp500_members_asof``, these are NOT point-in-time — they use a
    single snapshot date. Backtests on these universes carry survivorship bias
    from the snapshot date.
    """
    filename = _SNAPSHOT_REGISTRY.get(universe)
    if filename is None:
        raise ValueError(
            f"Unknown snapshot universe {universe!r}. Available: {sorted(_SNAPSHOT_REGISTRY)}"
        )
    return _load_snapshot(filename)


def universe_members_asof(
    universe: str,
    asof: pd.Timestamp,
    history: Optional[pd.DataFrame] = None,
    membership: Optional["MembershipSource"] = None,
) -> List[str]:
    """Unified universe lookup: PIT for sp500, static snapshot for others."""
    if universe == "sp500":
        return sp500_members_asof(asof, history, membership)
    return snapshot_members(universe)


def universe_all_between(
    universe: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    membership: Optional["MembershipSource"] = None,
) -> List[str]:
    """All members across [start, end]: PIT union for sp500, snapshot for others."""
    if universe == "sp500":
        return all_members_between(start, end, membership=membership)
    return sorted({normalize_yf_ticker(t) for t in snapshot_members(universe)})
