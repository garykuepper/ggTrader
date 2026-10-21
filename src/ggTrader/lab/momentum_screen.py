"""Run the pre-registered monthly French size-by-momentum screen."""

import math

import numpy as np
import pandas as pd
from scipy.stats import norm

from ggTrader.lab.significance import newey_west_se

FULL = ("1927-01", "2026-04")
POST = ("1994-01", "2026-04")
DIAG = [("1927-01", "1993-12"), ("1994-01", "2012-12"), ("2013-01", "2026-04")]
REBOUND_2009 = ("2009-03", "2009-12")
BASE_COST = {"big": 0.010, "small": 0.030}
STRESS_COST = {"big": 0.020, "small": 0.060}
NW_LAGS_MONTHLY = 12
HOLM_ALPHA = 0.00135
POST_BAR_PP = 1.0
NEXT = {
    (False, False): "Momentum is closed and reported as NO-GO. No paid data.",
    (True, False): "Phase 2 on the S&P 500.",
    (False, True): (
        "Skip Phase 2, since we have no small-cap PIT data. Phase 3 runs on paid small-cap "
        "data, and paper trading waits until Phase 3 passes."
    ),
    (True, True): (
        "Phases 2–3 test the S&P 500 version as the primary, and only it can authorize "
        "paper or real money. Small caps run in Phase 3 as a diagnostic only; deploying "
        "them needs a brief of their own."
    ),
}


def _aligned(left: pd.Series, right: pd.Series) -> pd.DataFrame:
    """Align common observations without silently dropping invalid returns."""
    if left.index.has_duplicates or right.index.has_duplicates:
        raise ValueError("Monthly returns require unique dates")
    frame = pd.concat([left.rename("left"), right.rename("right")], axis=1, join="inner")
    if frame.empty or not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Monthly returns require nonempty aligned data without NaN or infinity")
    return frame.sort_index()


def net_excess(winners: pd.Series, market: pd.Series, annual_cost: float) -> pd.Series:
    """Subtract aligned market returns and one twelfth of the annual haircut."""
    frame = _aligned(winners, market)
    return frame.left - frame.right - annual_cost / 12


def nw_t(x: pd.Series, lags: int = NW_LAGS_MONTHLY) -> float:
    """Return the mean divided by its Bartlett Newey-West standard error."""
    se = newey_west_se(x.to_numpy(), lags)
    mean = float(x.mean())
    return mean / se if se > 0 else (math.copysign(math.inf, mean) if mean else 0.0)


def one_sided_p(t: float) -> float:
    """Return the upper normal-tail probability for the registered one-sided test."""
    return float(norm.sf(t))


def holm(pvals: dict[str, float], alpha: float = HOLM_ALPHA) -> dict[str, bool]:
    """Apply Holm's step-down family-wise rejection rule."""
    if any(not 0 <= p <= 1 for p in pvals.values()):
        raise ValueError("Holm requires finite p-values between zero and one")
    rejected = dict.fromkeys(pvals, False)
    for i, (group, p) in enumerate(sorted(pvals.items(), key=lambda item: item[1])):
        if p > alpha / (len(pvals) - i):
            break
        rejected[group] = True
    return rejected


def cagr_diff_pp(winners: pd.Series, market: pd.Series, annual_cost: float) -> float:
    """Subtract monthly-compounded market CAGR from net winners' CAGR in points."""
    frame = _aligned(winners, market)
    gross = 1 + frame - pd.Series({"left": annual_cost / 12, "right": 0.0})
    if (gross <= 0).any().any():
        raise ValueError("Monthly compounding requires returns greater than -100%")
    annualized = np.expm1(np.log(gross).sum() * 12 / len(gross))
    return float((annualized.left - annualized.right) * 100)


def capm(win_ex_rf: pd.Series, mkt_ex_rf: pd.Series) -> dict:
    """Fit OLS annualized alpha and market beta without an intercept t-test."""
    frame = _aligned(win_ex_rf, mkt_ex_rf)
    design = np.column_stack([np.ones(len(frame)), frame.right.to_numpy()])
    coefficients, _, rank, _ = np.linalg.lstsq(design, frame.left.to_numpy(), rcond=None)
    if rank < 2:
        raise ValueError("CAPM requires a varying market excess return")
    alpha, beta = coefficients
    return {
        "alpha_ann_pct": float(alpha * 12 * 100),
        "beta": float(beta),
    }


def _relative_wealth(winners: pd.Series, market: pd.Series, annual_cost: float) -> pd.Series:
    """Compound net winners' wealth divided by market wealth."""
    net = _aligned(winners, market) - pd.Series({"left": annual_cost / 12, "right": 0.0})
    gross = 1 + net
    if (gross <= 0).any().any():
        raise ValueError("Relative wealth requires returns greater than -100% after costs")
    wealth = gross.cumprod()
    return wealth.left / wealth.right


def relative_drawdown(winners: pd.Series, market: pd.Series, annual_cost: float) -> dict:
    """Find the worst relative drawdown and trough-to-recovery calendar months.

    Include initial wealth of one before the first return. For tied peaks,
    use the last peak before the trough; recovery is its first later regain.
    """
    wealth = _relative_wealth(winners, market, annual_cost)
    initial = pd.Series([1.0], index=[wealth.index[0] - 1])
    wealth = pd.concat([initial, wealth])
    drawdown = wealth / wealth.cummax() - 1
    trough = drawdown.idxmin()
    prior = wealth.loc[:trough]
    peak = prior[prior == prior.max()].index[-1]
    later = wealth.loc[wealth.index > trough]
    recovered = later[later >= wealth[peak]]
    recovery = recovered.index[0] if not recovered.empty else None
    return {
        "max_drawdown_pct": float(drawdown[trough] * 100),
        "peak": str(peak),
        "trough": str(trough),
        "recovery": str(recovery) if recovery is not None else None,
        "recovery_months": recovery.ordinal - trough.ordinal if recovery is not None else None,
    }


def worst_rolling_relative(winners: pd.Series, market: pd.Series, months: int = 12) -> dict:
    """Find the worst compounded winners-to-market wealth ratio over a window."""
    frame = _aligned(winners, market)
    if months < 1 or len(frame) < months or (frame <= -1).any().any():
        raise ValueError("Rolling relative returns require a valid window and returns above -100%")
    wealth = np.exp(np.log1p(frame).rolling(months).sum())
    relative = wealth.left / wealth.right - 1
    end = relative.idxmin()
    start = frame.index[frame.index.get_loc(end) - months + 1]
    return {
        "start": str(start)[:7],
        "end": str(end)[:7],
        "relative_pct": float(relative[end] * 100),
    }


def _window(frame: pd.DataFrame, window: tuple[str, str] = FULL) -> pd.DataFrame:
    """Require every registered month once and all required values finite."""
    frame = frame.copy()
    frame.index = pd.PeriodIndex(frame.index, freq="M")
    expected = pd.period_range(*window, freq="M")
    frame = frame.loc[(frame.index >= expected[0]) & (frame.index <= expected[-1])]
    if frame.index.has_duplicates:
        raise ValueError("Duplicate month in the registered screen window")
    missing = expected.difference(frame.index)
    if len(missing):
        raise ValueError(f"Missing months in registered screen window: {list(map(str, missing))}")
    frame = frame.reindex(expected)
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("NaN or infinity in the registered screen window")
    return frame


def _stats(winners: pd.Series, market: pd.Series, annual_cost: float) -> dict:
    """Collect the same monthly excess-return evidence for each diagnostic."""
    t = nw_t(net_excess(winners, market, annual_cost))
    return {"t": t, "p": one_sided_p(t), "cagr_diff_pp": cagr_diff_pp(winners, market, annual_cost)}


def screen(sm: pd.DataFrame, mk: pd.DataFrame) -> dict:
    """Evaluate only the fixed significance and post-publication economic gates.

    Stress uses the registered stress haircut. CAPM, size-matched and crash
    diagnostics describe gross winners' returns over FULL. Rebound and relative
    drawdown definitions were added after the result to fulfill brief §10;
    neither affects the registered gates.
    """
    columns = [f"{size}_{rank}" for size in ("small", "big") for rank in ("lo", "mid", "hi")]
    sm = _window(sm[columns])
    mk = _window(mk[["mkt", "rf"]])
    groups = {}
    for group in ("big", "small"):
        winners = sm[f"{group}_hi"]
        cost = BASE_COST[group]
        full = _stats(winners, mk.mkt, cost)
        stress = _stats(winners, mk.mkt, STRESS_COST[group])
        size_market = sm[[f"{group}_{rank}" for rank in ("lo", "mid", "hi")]].mean(axis=1)
        rebound_winners = winners.loc[slice(*REBOUND_2009)]
        rebound_market = mk.mkt.loc[slice(*REBOUND_2009)]
        groups[group] = {
            "full_t": full["t"],
            "full_p": full["p"],
            "post_cagr_diff_pp": cagr_diff_pp(
                winners.loc[slice(*POST)], mk.mkt.loc[slice(*POST)], cost
            ),
            "diag": [
                {
                    "start": start,
                    "end": end,
                    **_stats(winners.loc[start:end], mk.mkt.loc[start:end], cost),
                }
                for start, end in DIAG
            ],
            "stress": {
                "full_t": stress["t"],
                "full_p": stress["p"],
                "post_cagr_diff_pp": cagr_diff_pp(
                    winners.loc[slice(*POST)], mk.mkt.loc[slice(*POST)], STRESS_COST[group]
                ),
            },
            "capm": capm(winners - mk.rf, mk.mkt - mk.rf),
            "size_matched": _stats(winners, size_market, 0.0),
            "worst_12m": worst_rolling_relative(winners, mk.mkt),
            "rebound_2009": {
                label: float(
                    (_relative_wealth(rebound_winners, rebound_market, haircut).iloc[-1] - 1) * 100
                )
                for label, haircut in (("net_pct", cost), ("gross_pct", 0.0))
            },
            "relative_drawdown": relative_drawdown(winners, mk.mkt, cost),
        }
    rejected = holm({group: values["full_p"] for group, values in groups.items()})
    passed = {
        group: rejected[group] and values["post_cagr_diff_pp"] >= POST_BAR_PP
        for group, values in groups.items()
    }
    return {
        "groups": groups,
        "holm": rejected,
        "pass": passed,
        "next": NEXT[passed["big"], passed["small"]],
    }
