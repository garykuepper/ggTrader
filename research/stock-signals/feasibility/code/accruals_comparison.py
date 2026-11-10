"""Calculate descriptive portfolio comparisons from aligned monthly returns."""

from __future__ import annotations

import math

import pandas as pd
from scipy.stats import norm

from ggTrader.utils.significance import newey_west_se


def monthly_returns(prices: pd.Series) -> pd.Series:
    """Convert adjusted closing prices to monthly returns without filling gaps."""
    if prices.empty or prices.index.has_duplicates or not prices.index.is_monotonic_increasing:
        raise ValueError("Prices need unique, ordered dates")
    month_end = prices.resample("ME").last()
    if len(month_end) < 2 or not all(math.isfinite(x) and x > 0 for x in month_end):
        raise ValueError("Missing or invalid monthly prices")
    return month_end.pct_change(fill_method=None).iloc[1:]


def compare(candidate: pd.Series, benchmark: pd.Series) -> dict:
    """Compare identical monthly periods; reject missing or invalid observations."""
    if not candidate.index.equals(benchmark.index) or candidate.empty:
        raise ValueError("Candidate and benchmark must share a nonempty monthly index")
    expected = pd.date_range(candidate.index[0], candidate.index[-1], freq="ME")
    if not candidate.index.equals(expected):
        raise ValueError("Monthly periods must be unique, ordered and continuous")
    if not all(math.isfinite(x) and x > -1 for series in (candidate, benchmark) for x in series):
        raise ValueError("Returns must be finite with positive wealth factors")

    def stats(series: pd.Series) -> dict:
        wealth = (1 + series).cumprod()
        peak = wealth.cummax().clip(lower=1.0)
        return {
            "terminal_wealth_per_1": float(wealth.iloc[-1]),
            "annualized_return": float(wealth.iloc[-1] ** (12 / len(series)) - 1),
            "max_month_end_drawdown": float((wealth / peak - 1).min()),
        }

    return {
        "months": len(candidate),
        "candidate": stats(candidate),
        "benchmark": stats(benchmark),
    }


def screen(candidate: pd.Series, benchmark: pd.Series, annual_cost: float) -> dict:
    """Calculate paired evidence with an explicit illustrative annual cost haircut."""
    if not math.isfinite(annual_cost) or annual_cost < 0:
        raise ValueError("Annual cost must be finite and nonnegative")
    net = candidate - annual_cost / 12
    comparison = compare(net, benchmark)
    if len(net) <= 12:
        raise ValueError("Twelve-lag inference requires more than twelve months")
    excess = net - benchmark
    se = newey_west_se(excess.to_numpy(), 12) if excess.nunique() > 1 else 0.0
    t = float(excess.mean() / se) if se > 0 else None
    return {
        "comparison": comparison,
        "annual_cost_assumption": annual_cost,
        "mean_excess_t": t,
        "one_sided_p": float(norm.sf(t)) if t is not None else None,
        "inference_note": "12-lag Newey–West; undefined when excess returns have zero variance",
    }


def evaluate(candidate: pd.Series, benchmark: pd.Series) -> dict:
    """Evaluate the proposed fixed windows without assigning a verdict."""
    result = {}
    for name, start in (("full", "1963-07-31"), ("since_1994", "1994-01-31")):
        index = pd.date_range(start, "2026-04-30", freq="ME")
        result[name] = screen(candidate.reindex(index), benchmark.reindex(index), annual_cost=0.01)
        result[f"{name}_stress"] = screen(
            candidate.reindex(index), benchmark.reindex(index), annual_cost=0.02
        )
    return result
