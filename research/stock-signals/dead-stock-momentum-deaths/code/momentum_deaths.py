"""Pure measures for the dead-stock-momentum-deaths audit (no I/O, no ledger simulation).

Computes section 1 of the frozen brief (A1 to A5, B, C, D) directly from daily closes,
point-in-time eligibility, and ended-series spans, reusing the first test's exact planner
conditions, successor flag, and CRSP file-end censoring rule.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import dm_dead_stock_first_test as fp
import dm_replay
import numpy as np
import pandas as pd
from dm_replay import (
    TargetStep,
    _aligned_eligibility,
    _decision_days,
    _localize,
    span_end_dates,
)

from ggTrader.lab.strategies.xs_momentum import momentum_scores

TOP_N = fp.TOP_N
LOOKBACK = fp.LOOKBACK
SKIP = fp.SKIP
BANDS: tuple[tuple[str, int, int | None], ...] = (
    ("1_to_50", 1, 50),
    ("51_to_150", 51, 150),
    ("151_to_500", 151, 500),
    ("501_plus", 501, None),
)
BOOTSTRAP_SEED = 20261125
BOOTSTRAP_RESAMPLES = 2_000
RECENT_RETURN_DAYS = 63
JUMP_THRESHOLD = 0.20
JUMP_MAX_DAYS_BEFORE = 252
JUMP_MIN_DAYS_BEFORE = 21
JUMP_NOTE = (
    "Descriptive count only: a one-day close-to-close gain of 20 percent or more is not a "
    "buy-out (earnings, news and dead-cat bounces also cause jumps)."
)


@dataclass(frozen=True)
class DecisionSet:
    """One decision day: when it acts, usable columns in column and score-rank order, scores."""

    decision_date: pd.Timestamp
    date: pd.Timestamp
    usable: np.ndarray  # column positions of usable stocks (column order)
    ranked: np.ndarray  # column positions of usable stocks (highest score first, stable)
    scores: np.ndarray  # momentum score of every column on decision_date


@dataclass(frozen=True)
class HoldingMonth:
    """One holding month from execution day through the day before the next execution day."""

    decision_date: pd.Timestamp
    execution_date: pd.Timestamp
    hold_end: pd.Timestamp
    ranked: np.ndarray  # column positions sorted highest score first (stable)
    scores: np.ndarray  # momentum scores on decision_date


def prepare_deaths(
    close: pd.DataFrame,
    span_rows: pd.DataFrame,
    end: str,
    *,
    crsp_end: str | None = None,
) -> tuple[pd.DataFrame, pd.Series, frozenset[str], pd.Series, int]:
    """Normalise spans, drop CRSP file-end rows, and split real deaths from successors.

    Follows ``dm_dead_stock_first_test.run_study`` verbatim: normalise span columns, drop
    rows ending on or after ``crsp_end`` (default ``fp.CRSP_END``), find every remaining
    series' last bar via ``span_end_dates``, and mark series with ``has_successor`` so they
    are excluded from death counts while still blocking stale selection after they end.

    Returns:
        ``(spans, ends, successors, death_dates, n_censored)`` where ``death_dates`` maps
        only real dying symbols (no successor, not file-end censored) to their last bar date.
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
    """Build decision days with usable sets and score ranks matching ``dm_replay.plan_trades``.

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


def rule_ranking_plan(
    sets: list[DecisionSet], symbols: np.ndarray, *, top_n: int = TOP_N
) -> list[TargetStep]:
    """Rebuild the momentum rule's equal-weight top-N plan from ``sets``."""
    steps: list[TargetStep] = []
    for item in sets:
        chosen = item.ranked[:top_n]
        targets = {str(s): 1.0 / len(chosen) for s in symbols[chosen]} if len(chosen) else {}
        steps.append(TargetStep(item.decision_date, item.date, targets))
    return steps


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


def _rate(deaths: int, stock_months: int) -> float | None:
    return float(deaths / stock_months) if stock_months > 0 else None


def bootstrap_rate_ratio(
    top_n_counts: np.ndarray,
    top_deaths: np.ndarray,
    all_counts: np.ndarray,
    all_deaths: np.ndarray,
    *,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Top-50 to all-usable death rate ratio with a 95 percent month-bootstrap interval."""
    top_n = np.asarray(top_n_counts, dtype=float)
    top_d = np.asarray(top_deaths, dtype=float)
    all_n = np.asarray(all_counts, dtype=float)
    all_d = np.asarray(all_deaths, dtype=float)
    n_months = len(all_n)

    top_sm, top_tot = int(top_n.sum()), int(top_d.sum())
    all_sm, all_tot = int(all_n.sum()), int(all_d.sum())
    top_rate = _rate(top_tot, top_sm)
    all_rate = _rate(all_tot, all_sm)
    point = (
        float(top_rate / all_rate)
        if top_rate is not None and all_rate is not None and all_rate > 0.0
        else None
    )

    ci_low: float | None = None
    ci_high: float | None = None
    n_valid = 0
    if n_months > 0 and n_resamples > 0:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, n_months, size=(n_resamples, n_months))
        b_top_n = top_n[idx].sum(axis=1)
        b_top_d = top_d[idx].sum(axis=1)
        b_all_n = all_n[idx].sum(axis=1)
        b_all_d = all_d[idx].sum(axis=1)
        valid = (b_top_n > 0.0) & (b_all_n > 0.0) & (b_all_d > 0.0)
        n_valid = int(valid.sum())
        if n_valid > 0:
            ratios = (b_top_d[valid] / b_top_n[valid]) / (b_all_d[valid] / b_all_n[valid])
            lo, hi = np.percentile(ratios, [2.5, 97.5])
            ci_low, ci_high = float(lo), float(hi)

    return {
        "rate_ratio": point,
        "top_50_rate": top_rate,
        "all_usable_rate": all_rate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_resamples": int(n_resamples),
        "n_valid_resamples": n_valid,
        "seed": int(seed),
    }


def one_day_jump_flags(close: pd.DataFrame, *, threshold: float = JUMP_THRESHOLD) -> np.ndarray:
    """Boolean matrix ``(days, cols)``: True where ``close[j] / close[j-1] >= 1 + threshold``.

    NaN gaps (missing current or prior close) evaluate to False and never bridge across holes.
    """
    vals = close.to_numpy(dtype=float)
    flags = np.zeros(vals.shape, dtype=bool)
    if vals.shape[0] < 2:
        return flags
    prev = vals[:-1]
    curr = vals[1:]
    valid = np.isfinite(prev) & np.isfinite(curr) & (prev > 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(valid, curr / prev, 0.0)
    flags[1:] = valid & (ratio >= 1.0 + threshold)
    return flags


def scoring_window_has_jump(
    valid_bars: np.ndarray, jump_cum_col: np.ndarray, dec_pos: int, *, far: int, near: int
) -> bool:
    """True if a flagged one-day jump falls inside the span the momentum score reads.

    The score looks ``near`` and ``far`` closes back on the stock's OWN valid closes (holes in
    its history do not count), starting from its last valid close on or before the decision
    day (``xs_momentum.momentum_scores``). The span runs from the bar ``far`` valid closes back
    to the bar ``near`` valid closes back, both ends included. ``jump_cum_col`` is the running
    count of flags for the column (``jump_cum_col[j + 1] - jump_cum_col[j]`` is bar ``j``).
    A stock without ``far`` valid closes before its last one has no span: False.
    """
    last = int(np.searchsorted(valid_bars, dec_pos, side="right")) - 1
    if last < far:
        return False
    lo = int(valid_bars[last - far])
    hi = int(valid_bars[last - near])
    return bool(jump_cum_col[hi + 1] - jump_cum_col[lo] > 0)


def quartiles(values: list[float]) -> dict[str, Any]:
    """Count, lower quartile, median, and upper quartile of finite values."""
    finite = np.array([v for v in values if v is not None and math.isfinite(v)], dtype=float)
    if len(finite) == 0:
        return {"n": 0, "p25": None, "p50": None, "p75": None}
    q25, q50, q75 = np.percentile(finite, [25.0, 50.0, 75.0])
    return {"n": int(len(finite)), "p25": float(q25), "p50": float(q50), "p75": float(q75)}


def _profile_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarise one group of dying stocks for measure B."""
    n = len(records)
    bucket_counts = {name: 0 for name in fp.RATIO_BUCKETS}
    for rec in records:
        bucket_counts[rec["bucket"]] += 1
    buckets = {
        name: {"count": count, "share": float(count / n) if n > 0 else None}
        for name, count in bucket_counts.items()
    }
    return {
        "n": n,
        "momentum_score": quartiles([r["score"] for r in records]),
        "rank_share": quartiles([r["rank_share"] for r in records]),
        "final_month_move_buckets": buckets,
        "final_month_ratio": quartiles([r["ratio"] for r in records if r["ratio"] is not None]),
        "return_63d": quartiles([r["return_63d"] for r in records if r["return_63d"] is not None]),
    }


def _jump_group_summary(n: int, n_flagged: int) -> dict[str, Any]:
    return {
        "n": int(n),
        "n_flagged": int(n_flagged),
        "share_flagged": float(n_flagged / n) if n > 0 else None,
    }


def check_against_held_at_death(top_50_deaths: int, held_at_death: int) -> dict[str, Any]:
    """Measure D: compare top-50 holding-month deaths against the first test's held_at_death."""
    if held_at_death < 0 or top_50_deaths < 0:
        raise ValueError("death counts must be non-negative")
    diff = top_50_deaths - held_at_death
    abs_diff = abs(diff)
    if held_at_death > 0:
        ratio: float | None = float(top_50_deaths / held_at_death)
        rel_diff: float | None = float(abs_diff / held_at_death)
    else:
        ratio = None
        rel_diff = 0.0 if top_50_deaths == 0 else None
    # Exact integer check for strictly greater than 10 percent (avoids float rounding at 10%).
    exceeds_10pct = bool(abs_diff * 10 > held_at_death)
    return {
        "top_50_deaths": int(top_50_deaths),
        "first_test_held_at_death": int(held_at_death),
        "difference": int(diff),
        "abs_difference": int(abs_diff),
        "ratio": ratio,
        "relative_difference": rel_diff,
        "exceeds_10pct_flag": exceeds_10pct,
        "threshold_pct": 10.0,
    }


def compute_measures(
    close: pd.DataFrame,
    eligible: pd.DataFrame,
    span_rows: pd.DataFrame,
    *,
    start: str,
    end: str,
    held_at_death: int | None = None,
    random_baseline_median: float | None = None,
    top_n: int = TOP_N,
    lookback: int = LOOKBACK,
    skip: int = SKIP,
    bands: tuple[tuple[str, int, int | None], ...] = BANDS,
    recent_return_days: int = RECENT_RETURN_DAYS,
    jump_threshold: float = JUMP_THRESHOLD,
    jump_max_days_before: int = JUMP_MAX_DAYS_BEFORE,
    jump_min_days_before: int = JUMP_MIN_DAYS_BEFORE,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    crsp_end: str | None = None,
) -> dict[str, Any]:
    """Compute every measure of brief section 1 (A1 to A5, B, C, D) for one window."""
    spans, _ends, _successors, death_dates, n_censored = prepare_deaths(
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
    symbols = np.array(close.columns)
    death_map = death_dates.to_dict()
    close_vals = close.to_numpy(dtype=float)
    jump_matrix = one_day_jump_flags(close, threshold=jump_threshold)
    # Running count of flags per column, so "any jump between two bars" is one subtraction.
    jump_cum = np.zeros((jump_matrix.shape[0] + 1, jump_matrix.shape[1]), dtype=np.int32)
    np.cumsum(jump_matrix, axis=0, out=jump_cum[1:])
    valid_bars = [np.flatnonzero(np.isfinite(close_vals[:, c])) for c in range(close.shape[1])]

    band_sm = {name: 0 for name, _, _ in bands}
    band_deaths = {name: 0 for name, _, _ in bands}
    monthly_top_n: list[int] = []
    monthly_top_d: list[int] = []
    monthly_all_n: list[int] = []
    monthly_all_d: list[int] = []
    expected_total = 0.0
    by_year: dict[str, dict[str, Any]] = {}

    dying_top_records: list[dict[str, Any]] = []
    dying_rest_records: list[dict[str, Any]] = []

    c_counts = {
        "top_50_died": [0, 0],
        "top_50_did_not_die": [0, 0],
        "all_usable": [0, 0],
    }

    for month in months:
        n_usable = int(len(month.ranked))
        month_deaths = deaths_in_month(month, symbols, death_dates)
        n_deaths = len(month_deaths)

        top_cols = month.ranked[:top_n]
        n_top = int(len(top_cols))
        n_top_deaths = sum(1 for c in top_cols if int(c) in month_deaths)

        monthly_top_n.append(n_top)
        monthly_top_d.append(n_top_deaths)
        monthly_all_n.append(n_usable)
        monthly_all_d.append(n_deaths)

        month_expected = float(min(top_n, n_usable) * n_deaths / n_usable) if n_usable > 0 else 0.0
        expected_total += month_expected

        year_key = str(month.execution_date.year)
        if year_key not in by_year:
            by_year[year_key] = {
                "n_months": 0,
                "usable_stock_months": 0,
                "all_usable_deaths": 0,
                "top_50_deaths": 0,
                "expected_random_50_deaths": 0.0,
            }
        yr = by_year[year_key]
        yr["n_months"] += 1
        yr["usable_stock_months"] += n_usable
        yr["all_usable_deaths"] += n_deaths
        yr["top_50_deaths"] += n_top_deaths
        yr["expected_random_50_deaths"] += month_expected

        for name, lo_rank, hi_rank in bands:
            slice_cols = month.ranked[lo_rank - 1 : hi_rank]
            band_sm[name] += int(len(slice_cols))
            band_deaths[name] += sum(1 for c in slice_cols if int(c) in month_deaths)

        dec_pos = int(close.index.get_loc(month.decision_date))

        for rank_idx, col_raw in enumerate(month.ranked):
            col = int(col_raw)
            rank = rank_idx + 1
            in_top = rank <= top_n
            died = col in month_deaths
            jumped = scoring_window_has_jump(
                valid_bars[col],
                jump_cum[:, col],
                dec_pos,
                far=jump_max_days_before,
                near=jump_min_days_before,
            )

            c_counts["all_usable"][0] += 1
            c_counts["all_usable"][1] += int(jumped)
            if in_top:
                key = "top_50_died" if died else "top_50_did_not_die"
                c_counts[key][0] += 1
                c_counts[key][1] += int(jumped)

            if died:
                sym = str(symbols[col])
                last_bar = death_map[sym]
                last_pos = int(close.index.get_loc(last_bar))
                dec_close = close_vals[dec_pos, col]
                last_close = close_vals[last_pos, col]
                if math.isfinite(dec_close) and dec_close > 0.0 and math.isfinite(last_close):
                    ratio: float | None = float(last_close / dec_close)
                else:
                    ratio = None
                ret_63d: float | None = None
                if dec_pos >= recent_return_days:
                    past_close = close_vals[dec_pos - recent_return_days, col]
                    if math.isfinite(dec_close) and math.isfinite(past_close) and past_close > 0.0:
                        ret_63d = float(dec_close / past_close - 1.0)
                rec = {
                    "score": float(month.scores[col]),
                    "rank_share": float(rank / n_usable),
                    "ratio": ratio,
                    "bucket": fp.ratio_bucket(ratio),
                    "return_63d": ret_63d,
                }
                if in_top:
                    dying_top_records.append(rec)
                else:
                    dying_rest_records.append(rec)

    total_sm = int(sum(monthly_all_n))
    total_deaths = int(sum(monthly_all_d))
    actual_top_deaths = int(sum(monthly_top_d))

    a1 = {
        "stock_months": total_sm,
        "deaths": total_deaths,
        "death_rate": _rate(total_deaths, total_sm),
        "n_months": len(months),
    }
    a2 = {
        name: {
            "stock_months": band_sm[name],
            "deaths": band_deaths[name],
            "death_rate": _rate(band_deaths[name], band_sm[name]),
        }
        for name, _, _ in bands
    }
    a3 = bootstrap_rate_ratio(
        np.array(monthly_top_n),
        np.array(monthly_top_d),
        np.array(monthly_all_n),
        np.array(monthly_all_d),
        n_resamples=n_resamples,
        seed=bootstrap_seed,
    )
    a4 = {
        "expected_random_50_deaths": float(expected_total),
        "actual_top_50_deaths": actual_top_deaths,
        "excess_deaths": float(actual_top_deaths - expected_total),
        "actual_to_expected_ratio": (
            float(actual_top_deaths / expected_total) if expected_total > 0.0 else None
        ),
        "random_baseline_fresh_draw_median": random_baseline_median,
    }
    a5 = {year: by_year[year] for year in sorted(by_year)}
    b_block = {
        "in_top_50": _profile_summary(dying_top_records),
        "usable_not_top_50": _profile_summary(dying_rest_records),
    }
    c_block = {
        "threshold": float(jump_threshold),
        "window_valid_closes_before": [int(jump_max_days_before), int(jump_min_days_before)],
        "window_note": "counted on each stock's own valid closes, as the momentum score is",
        "note": JUMP_NOTE,
        "top_50_died": _jump_group_summary(*c_counts["top_50_died"]),
        "top_50_did_not_die": _jump_group_summary(*c_counts["top_50_did_not_die"]),
        "all_usable": _jump_group_summary(*c_counts["all_usable"]),
    }
    d_block = (
        check_against_held_at_death(actual_top_deaths, held_at_death)
        if held_at_death is not None
        else None
    )

    return {
        "A1_totals": a1,
        "A2_by_band": a2,
        "A3_rate_ratio": a3,
        "A4_expected_vs_actual": a4,
        "A5_by_year": a5,
        "B_dying_profile": b_block,
        "C_one_day_jumps": c_block,
        "D_first_test_check": d_block,
        "n_file_end_censored": int(n_censored),
    }


__all__ = [
    "BANDS",
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "DecisionSet",
    "HoldingMonth",
    "JUMP_MAX_DAYS_BEFORE",
    "JUMP_MIN_DAYS_BEFORE",
    "JUMP_THRESHOLD",
    "LOOKBACK",
    "RECENT_RETURN_DAYS",
    "SKIP",
    "TOP_N",
    "bootstrap_rate_ratio",
    "check_against_held_at_death",
    "compute_measures",
    "deaths_in_month",
    "dm_replay",
    "eligible_sets",
    "holding_months",
    "one_day_jump_flags",
    "prepare_deaths",
    "quartiles",
    "rule_ranking_plan",
]
