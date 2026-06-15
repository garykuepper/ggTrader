"""Curve statistics and SPY benchmark comparison for lab runs."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def curve_stats(curve: pd.Series) -> Dict[str, float]:
    curve = curve.dropna()
    rets = curve.pct_change().dropna()
    years = max((curve.index[-1] - curve.index[0]).days / 365.25, 1e-9)
    total = float(curve.iloc[-1] / curve.iloc[0] - 1.0)
    ann_vol = float(rets.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
    sharpe = (
        float(rets.mean() / rets.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
        if rets.std() > 0
        else float("nan")
    )
    downside = rets[rets < 0]
    sortino = (
        float(rets.mean() / downside.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
        if len(downside) and downside.std() > 0
        else float("nan")
    )
    dd = float((curve / curve.cummax() - 1.0).min())
    return {
        "total_return_pct": total * 100,
        "cagr_pct": ((1 + total) ** (1 / years) - 1) * 100,
        "sharpe": sharpe,
        "sortino": sortino,
        "ann_vol_pct": ann_vol * 100,
        "max_drawdown_pct": dd * 100,
    }


def benchmark(equity: pd.Series, spy_close: pd.Series, start_cash: float) -> Dict[str, Any]:
    spy = spy_close.reindex(equity.index).ffill().dropna()
    spy_curve = start_cash * (spy / spy.iloc[0])
    eq = equity.reindex(spy_curve.index)

    strat_m = eq.resample("ME").last().pct_change().dropna()
    spy_m = spy_curve.resample("ME").last().pct_change().dropna()
    common = strat_m.index.intersection(spy_m.index)
    hit = float((strat_m.loc[common] > spy_m.loc[common]).mean()) if len(common) else None

    return {
        "strategy": curve_stats(eq.dropna()),
        "spy": curve_stats(spy_curve),
        "monthly_hit_rate_vs_spy": hit,
        "n_months": int(len(common)),
    }
