"""Newey-West excess-return statistics for the pre-registered audit."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ggTrader.lab.metrics import curve_stats

T_BAR = 3.0
NW_LAGS = 5
CAGR_BAR_PP = 1.0


@dataclass(frozen=True)
class ExcessStats:
    """Aligned daily excess-return evidence against a passive benchmark."""

    n_obs: int
    t_stat: float
    ann_excess_pct: float
    tracking_error_pct: float
    information_ratio: float
    cagr_diff_pp: float


def newey_west_se(x: np.ndarray, lags: int) -> float:
    """Return the Bartlett-kernel Newey-West standard error of the mean."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all() or lags < 0:
        raise ValueError("Newey-West requires a finite nonempty vector and nonnegative lags")
    n = len(x)
    d = x - x.mean()
    variance = float(d @ d / n)
    for lag in range(1, min(lags, n - 1) + 1):
        variance += 2 * (1 - lag / (lags + 1)) * float(d[lag:] @ d[:-lag] / n)
    return math.sqrt(max(variance, 0.0) / n)


def excess_stats(
    strategy_equity: pd.Series,
    benchmark_equity: pd.Series,
    lags: int = NW_LAGS,
    periods_per_year: int = 252,
) -> ExcessStats:
    """Measure excess returns after aligning curves, using the shared CAGR calculation."""
    curves = (
        pd.concat(
            [strategy_equity.rename("strategy"), benchmark_equity.rename("benchmark")],
            axis=1,
            join="inner",
        )
        .dropna()
        .sort_index()
    )
    if len(curves) < 3 or not np.isfinite(curves.to_numpy()).all() or (curves <= 0).any().any():
        raise ValueError("Excess statistics require at least three aligned positive finite NAVs")
    if periods_per_year <= 0 or curves.index.has_duplicates:
        raise ValueError("Excess statistics require positive annualization and unique dates")
    returns = curves.pct_change().dropna()
    excess = returns.strategy - returns.benchmark
    mean = float(excess.mean())
    se = newey_west_se(excess.to_numpy(), lags)
    t_stat = mean / se if se > 0 else (math.copysign(math.inf, mean) if mean else 0.0)
    annual_excess = mean * periods_per_year * 100
    tracking_error = float(excess.std(ddof=1) * math.sqrt(periods_per_year) * 100)
    return ExcessStats(
        n_obs=len(excess),
        t_stat=t_stat,
        ann_excess_pct=annual_excess,
        tracking_error_pct=tracking_error,
        information_ratio=annual_excess / tracking_error if tracking_error > 0 else 0.0,
        cagr_diff_pp=curve_stats(curves.strategy)["cagr_pct"]
        - curve_stats(curves.benchmark)["cagr_pct"],
    )


def years_for_t(information_ratio: float, target_t: float = 2.0) -> float:
    """Return the years needed for a positive IR to reach the target t-statistic."""
    return (target_t / information_ratio) ** 2 if information_ratio > 0 else math.inf


def passes_pretax_bar(
    stats: ExcessStats, t_bar: float = T_BAR, cagr_bar_pp: float = CAGR_BAR_PP
) -> bool:
    """Require both the pre-registered statistical and economic thresholds."""
    return stats.t_stat >= t_bar and stats.cagr_diff_pp >= cagr_bar_pp
