"""Oracle: the pre-rebuild ``lab.metrics.curve_stats`` (hand-written Sharpe/vol/drawdown).

Kept verbatim so the engine statistics (``engine.stats``, ``wfo.report.curve_summary``) can be
checked against an independent implementation of the Sharpe, volatility and drawdown conventions
(252 bars, ddof=1, returns from the second bar, drawdown from the base bar). Sortino and CAGR are
*not* comparable: vectorbt's definitions replaced the hand-written ones (owner decision
2026-11-06), so tests compare only the shared keys. Not part of the application.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def curve_stats(curve: pd.Series) -> Dict[str, float]:
    """Total return, CAGR (calendar years), Sharpe, Sortino, volatility and drawdown of a curve."""
    curve = curve.dropna()
    rets = curve.pct_change(fill_method=None).dropna()
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
