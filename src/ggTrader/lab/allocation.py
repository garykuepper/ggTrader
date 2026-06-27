"""Out-of-sample portfolio overlay math for the multi-sleeve research harness.

Pure functions only — no I/O, no DB, no WFO. All volatility estimates use
trailing data so a value at date t never depends on t's own future.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def trailing_realized_vol(returns: pd.Series, window: int = 60) -> pd.Series:
    """Rolling annualized realized volatility from daily returns.

    Returns NaN for the first ``window - 1`` observations (warmup).
    """
    return returns.rolling(window).std() * np.sqrt(TRADING_DAYS)


def inverse_vol_weights(vols: dict[str, float]) -> dict[str, float]:
    """Risk-parity weights: w_i = (1/vol_i) / sum(1/vol_j), summing to 1.0.

    Sleeves with non-positive or NaN vol are dropped. If none are valid,
    fall back to equal weights across the original keys.
    """
    valid = {k: v for k, v in vols.items() if v is not None and np.isfinite(v) and v > 0}
    if not valid:
        n = len(vols)
        return {k: 1.0 / n for k in vols}
    inv = {k: 1.0 / v for k, v in valid.items()}
    total = sum(inv.values())
    return {k: x / total for k, x in inv.items()}


def target_vol_scale(
    blend_trailing_vol: float, target_vol: float, max_leverage: float = 2.0
) -> float:
    """Exposure multiplier to bring a blend's trailing vol to target_vol.

    clip(target_vol / blend_trailing_vol, 0.0, max_leverage). Returns 0.0 when
    blend vol is non-positive or NaN (cannot size safely).
    """
    if not np.isfinite(blend_trailing_vol) or blend_trailing_vol <= 0:
        return 0.0
    return float(np.clip(target_vol / blend_trailing_vol, 0.0, max_leverage))


def combine_sleeves(
    sleeve_returns: pd.DataFrame,
    target_vol: float = 0.068,
    window: int = 60,
    max_leverage: float = 2.0,
    rebalance: str = "ME",
) -> tuple[pd.Series, pd.DataFrame]:
    """Blend sleeve return streams with rolling inverse-vol weights scaled to
    a target volatility. Returns (blended_daily_returns, diagnostics).

    Out-of-sample: at each rebalance date, only returns strictly BEFORE that
    date inform the weights/scale that are then applied to forward days.
    Warmup (insufficient history before the first rebalance) uses equal weights
    at scale 1.0.
    """
    df = sleeve_returns.sort_index()
    sleeves = list(df.columns)
    equal = {s: 1.0 / len(sleeves) for s in sleeves}

    # Rebalance dates that fall within the sample.
    reb_dates = df.resample(rebalance).last().index
    reb_dates = [d for d in reb_dates if d <= df.index[-1]]

    blended = pd.Series(0.0, index=df.index)
    diag_rows: list[dict] = []

    # Build held (weights, scale) decisions per rebalance using only past data.
    decisions: list[tuple[pd.Timestamp, pd.Series, float]] = []
    for d in reb_dates:
        past = df.loc[df.index < d]
        if len(past) < window:
            w, sc = pd.Series(equal), 1.0
        else:
            vols = {s: float(trailing_realized_vol(past[s], window).iloc[-1]) for s in sleeves}
            wd = inverse_vol_weights(vols)
            w = pd.Series({s: wd.get(s, 0.0) for s in sleeves})
            prov = (past[sleeves] * w).sum(axis=1)
            blend_vol = float(trailing_realized_vol(prov, window).iloc[-1])
            sc = target_vol_scale(blend_vol, target_vol, max_leverage)
        decisions.append((d, w, sc))
        row = {f"w_{s}": w[s] for s in sleeves}
        row["blend_vol"] = (
            float(trailing_realized_vol((past[sleeves] * w).sum(axis=1), window).iloc[-1])
            if len(past) >= window
            else float("nan")
        )
        row["scale"] = sc
        diag_rows.append(row)

    # Apply each decision forward until the next rebalance.
    for i, (d, w, sc) in enumerate(decisions):
        nxt = decisions[i + 1][0] if i + 1 < len(decisions) else None
        mask = df.index >= d if nxt is None else (df.index >= d) & (df.index < nxt)
        blended.loc[mask] = (df.loc[mask, sleeves] * w).sum(axis=1) * sc

    # Days before the first rebalance: equal weight, scale 1.0.
    if decisions:
        pre = df.index < decisions[0][0]
        blended.loc[pre] = (df.loc[pre, sleeves] * pd.Series(equal)).sum(axis=1)
    else:
        blended = (df[sleeves] * pd.Series(equal)).sum(axis=1)

    diag = pd.DataFrame(diag_rows, index=pd.Index([d for d, _, _ in decisions], name="rebalance"))
    return blended, diag
