"""Build the death roster for the dead-stock-early-filings audit (no I/O, no network).

Reproduces the exact death definition of the momentum-deaths study using this study's
frozen ``ef_*`` engine copies: every stock in universe A that was usable on a month-end
decision day (eligible, finite momentum score, series not already ended) and whose series
ended inside the holding month opening on that decision day, excluding successors and
CRSP file-end censored rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import ef_dead_stock_first_test as fp
import numpy as np
import pandas as pd
from ef_replay import (
    _aligned_eligibility,
    _decision_days,
    _localize,
    span_end_dates,
)

from ggTrader.lab.strategies.xs_momentum import momentum_scores

TOP_N = fp.TOP_N
LOOKBACK = fp.LOOKBACK
SKIP = fp.SKIP
PERMNO_PREFIX = "PERMNO-"
RANK_BANDS: tuple[tuple[str, int, int | None], ...] = (
    ("1-50", 1, 50),
    ("51-150", 51, 150),
    ("151-500", 151, 500),
    ("501+", 501, None),
)
ROSTER_COLUMNS = [
    "column",
    "permno",
    "symbol",
    "last_bar",
    "decision_date",
    "rank_band",
    "held_by_rule",
]


@dataclass(frozen=True)
class DecisionSet:
    """One decision day: when it acts, usable columns in column and score-rank order, scores."""

    decision_date: pd.Timestamp
    date: pd.Timestamp
    usable: np.ndarray
    ranked: np.ndarray
    scores: np.ndarray


@dataclass(frozen=True)
class HoldingMonth:
    """One holding month from execution day through the day before the next execution day."""

    decision_date: pd.Timestamp
    execution_date: pd.Timestamp
    hold_end: pd.Timestamp
    ranked: np.ndarray
    scores: np.ndarray


def base_symbol(column: str) -> str:
    """Strip a ``~k`` spell suffix from a release column name, keeping its base symbol."""
    head, sep, tail = column.rpartition("~")
    if sep and head and tail.isdigit():
        return head
    return column


def _rank_band(rank: int) -> str:
    """Map a 1-based momentum rank on the decision day to its registered rank band label."""
    for label, lo_rank, hi_rank in RANK_BANDS:
        if rank >= lo_rank and (hi_rank is None or rank <= hi_rank):
            return label
    raise ValueError(f"rank must be at least 1, got {rank}")


def _extract_permno_lookup(
    release_spans: pd.DataFrame | None, span_rows: pd.DataFrame
) -> dict[str, int]:
    """Build a ``column -> permno`` map from ``release_spans`` (preferred) and ``span_rows``."""
    lookup: dict[str, int] = {}
    for frame in (span_rows, release_spans):
        if frame is None or not {"symbol", "permno"}.issubset(frame.columns):
            continue
        for sym, raw_permno in zip(frame["symbol"], frame["permno"], strict=False):
            if pd.notna(raw_permno):
                lookup[str(sym)] = int(raw_permno)
    return lookup


def _resolve_permno(column: str, symbol: str, permno_lookup: dict[str, int]) -> Any:
    """Resolve a column's PERMNO from release/span metadata or a ``PERMNO-<id>`` base symbol."""
    if column in permno_lookup:
        return permno_lookup[column]
    if symbol.startswith(PERMNO_PREFIX):
        rest = symbol[len(PERMNO_PREFIX) :]
        if rest.isdigit():
            return int(rest)
    return pd.NA


def prepare_deaths(
    close: pd.DataFrame,
    span_rows: pd.DataFrame,
    end: str,
    *,
    crsp_end: str | None = None,
) -> tuple[pd.DataFrame, pd.Series, frozenset[str], pd.Series, int]:
    """Normalise spans, drop CRSP file-end rows, and split real deaths from successors.

    Returns:
        ``(spans, ends, successors, death_dates, n_censored)`` where ``death_dates`` maps
        only real dying columns (no successor, not file-end censored) to their last bar date.
    """
    spans, n_censored = fp.drop_file_end_rows(
        fp._normalise_spans(span_rows), end, crsp_end=crsp_end
    )
    ends = span_end_dates(close, spans)
    flagged = spans.drop_duplicates("symbol").set_index("symbol")["has_successor"]
    successors = frozenset(s for s in ends.index if bool(flagged.get(s, False)))
    death_dates = ends[~ends.index.isin(successors)]
    return spans, ends, successors, death_dates, n_censored


def eligible_sets(
    close: pd.DataFrame,
    eligible: pd.DataFrame,
    spans: pd.DataFrame,
    *,
    lookback: int = LOOKBACK,
    skip: int = SKIP,
    start: str,
    end: str,
) -> list[DecisionSet]:
    """Build decision days with usable sets and score ranks matching ``ef_replay.plan_trades``.

    ``spans`` should already be normalised with file-end rows dropped (via ``prepare_deaths``).
    """
    index = close.index
    days = _decision_days(index, start, end)
    if not days:
        return []
    scores = momentum_scores(close, lookback, skip).reindex(pd.DatetimeIndex(days)).to_numpy()
    allowed = _aligned_eligibility(eligible, close, days)
    ended = span_end_dates(close, spans).reindex(close.columns)
    last_exec = _localize(pd.Timestamp(end), index)
    sets: list[DecisionSet] = []
    for i, decision in enumerate(days):
        pos = int(index.get_loc(decision)) + 1
        if pos >= len(index) or index[pos] > last_exec:
            continue
        already_dead = (ended < decision).to_numpy()
        usable = np.flatnonzero(allowed[i] & np.isfinite(scores[i]) & ~already_dead)
        order = np.argsort(-scores[i][usable], kind="stable")
        ranked = usable[order]
        sets.append(DecisionSet(decision, index[pos], usable, ranked, scores[i]))
    return sets


def holding_months(
    sets: list[DecisionSet], index: pd.DatetimeIndex, end: str
) -> list[HoldingMonth]:
    """Build each decision day's holding month: execution day through day before next execution.

    For the last decision day, the holding month ends at ``end`` (the window end, inclusive).
    """
    window_end = _localize(pd.Timestamp(end), index)
    months: list[HoldingMonth] = []
    for i, item in enumerate(sets):
        hold_end = sets[i + 1].date - pd.Timedelta(days=1) if i + 1 < len(sets) else window_end
        months.append(
            HoldingMonth(
                decision_date=item.decision_date,
                execution_date=item.date,
                hold_end=hold_end,
                ranked=item.ranked,
                scores=item.scores,
            )
        )
    return months


def deaths_in_month(month: HoldingMonth, symbols: np.ndarray, death_dates: pd.Series) -> set[int]:
    """Column indices of usable stocks in ``month`` whose real death falls inside the month."""
    if death_dates.empty or len(month.ranked) == 0:
        return set()
    death_map = death_dates.to_dict()
    dying: set[int] = set()
    for col in month.ranked:
        last_bar = death_map.get(str(symbols[col]))
        if last_bar is not None and month.execution_date <= last_bar <= month.hold_end:
            dying.add(int(col))
    return dying


def build_death_roster(
    close: pd.DataFrame,
    eligible: pd.DataFrame,
    span_rows: pd.DataFrame,
    *,
    start: str,
    end: str,
    release_spans: pd.DataFrame | None = None,
    top_n: int = TOP_N,
    lookback: int = LOOKBACK,
    skip: int = SKIP,
    crsp_end: str | None = None,
) -> pd.DataFrame:
    """List every usable stock whose real death falls inside a holding month of ``[start, end]``.

    Args:
        close: Daily closes, trading days x release columns, including momentum warm-up bars.
        eligible: Boolean point-in-time eligibility frame (days x columns).
        span_rows: Ended-series rows (``symbol``, ``last_date``, ``has_successor``, ``total_loss``;
            may also carry ``permno``).
        start: First date of the window, ``YYYY-MM-DD``.
        end: Last date of the window, ``YYYY-MM-DD``.
        release_spans: Optional release spans frame (``symbol``, ``permno``, ...) used to attach
            PERMNOs to non-placeholder columns.
        top_n: Number of top-ranked stocks held by the momentum rule (default 50).
        lookback: Trading days back for the older close in the momentum score.
        skip: Trading days back for the newer close in the momentum score.
        crsp_end: Optional CRSP file-end censoring date (default
            ``ef_dead_stock_first_test.CRSP_END``).

    Returns:
        DataFrame with one row per death and columns ``column``, ``permno`` (nullable ``Int64``),
        ``symbol`` (base symbol without any ``~k`` spell suffix), ``last_bar`` (``Timestamp``),
        ``decision_date`` (``Timestamp``), ``rank_band`` (``"1-50"``, ``"51-150"``,
        ``"151-500"``, ``"501+"``), and ``held_by_rule`` (``bool``).
    """
    spans, _ends, _successors, death_dates, _n_censored = prepare_deaths(
        close, span_rows, end, crsp_end=crsp_end
    )
    sets = eligible_sets(
        close,
        eligible,
        spans,
        lookback=lookback,
        skip=skip,
        start=start,
        end=end,
    )
    months = holding_months(sets, close.index, end)
    columns = np.array(close.columns)
    death_map = death_dates.to_dict()
    permno_lookup = _extract_permno_lookup(release_spans, span_rows)

    rows: list[dict[str, Any]] = []
    for month in months:
        month_deaths = deaths_in_month(month, columns, death_dates)
        if not month_deaths:
            continue
        for rank_idx, col_raw in enumerate(month.ranked):
            col = int(col_raw)
            if col not in month_deaths:
                continue
            rank = rank_idx + 1
            col_name = str(columns[col])
            sym = base_symbol(col_name)
            rows.append(
                {
                    "column": col_name,
                    "permno": _resolve_permno(col_name, sym, permno_lookup),
                    "symbol": sym,
                    "last_bar": death_map[col_name],
                    "decision_date": month.decision_date,
                    "rank_band": _rank_band(rank),
                    "held_by_rule": bool(rank <= top_n),
                }
            )

    if not rows:
        dt_dtype = close.index.dtype
        return pd.DataFrame(
            {
                "column": pd.Series(dtype=object),
                "permno": pd.Series(dtype="Int64"),
                "symbol": pd.Series(dtype=object),
                "last_bar": pd.Series(dtype=dt_dtype),
                "decision_date": pd.Series(dtype=dt_dtype),
                "rank_band": pd.Series(dtype=object),
                "held_by_rule": pd.Series(dtype=bool),
            },
            columns=ROSTER_COLUMNS,
        )

    frame = pd.DataFrame(rows, columns=ROSTER_COLUMNS)
    frame["permno"] = frame["permno"].astype("Int64")
    frame["held_by_rule"] = frame["held_by_rule"].astype(bool)
    return frame


__all__ = [
    "DecisionSet",
    "HoldingMonth",
    "LOOKBACK",
    "PERMNO_PREFIX",
    "RANK_BANDS",
    "ROSTER_COLUMNS",
    "SKIP",
    "TOP_N",
    "base_symbol",
    "build_death_roster",
    "deaths_in_month",
    "eligible_sets",
    "holding_months",
    "prepare_deaths",
]
