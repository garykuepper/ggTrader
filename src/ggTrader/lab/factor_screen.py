"""Evaluate the four pre-registered long-only French factor hypotheses."""

import pandas as pd

from ggTrader.lab.momentum_screen import (
    HOLM_ALPHA,
    POST,
    POST_BAR_PP,
    _stats,
    _window,
    cagr_diff_pp,
    capm,
    holm,
    net_excess,
    nw_t,
    one_sided_p,
    relative_drawdown,
    worst_rolling_relative,
)

FULL = ("1963-07", "2026-04")
HISTORY = ("1927-01", "2026-04")
DIAG = [("1963-07", "1993-12"), ("1994-01", "2012-12"), ("2013-01", "2026-04")]
HYPOTHESES = {
    "H1": {"factor": "BIG HiOP", "base_cost": 0.01, "stress_cost": 0.02},
    "H2": {"factor": "BIG HiBM", "base_cost": 0.01, "stress_cost": 0.02},
    "H3": {"factor": "BIG LoINV", "base_cost": 0.01, "stress_cost": 0.02},
    "H4": {"factor": "50% BIG HiBM + 50% BIG HiPRIOR", "base_cost": 0.015, "stress_cost": 0.03},
}
SMALL_BASE_COST = 0.03
SMALL_STRESS_COST = 0.06


def value_momentum_combo(value: pd.Series, momentum: pd.Series) -> pd.Series:
    """Form monthly rebalanced gross H4 returns before charging its haircut once."""
    return 0.5 * value + 0.5 * momentum


def _require_wealth(winners: pd.Series, market: pd.Series, annual_cost: float) -> None:
    """Stop before statistics if any required monthly wealth factor is non-positive."""
    if (winners <= annual_cost / 12 - 1).any() or (market <= -1).any():
        raise ValueError("Non-positive monthly wealth factor after costs")


def _diagnostics(
    winners: pd.Series, market: pd.DataFrame, base_cost: float, stress_cost: float
) -> dict:
    """Describe the registered common-window behavior without adding pass criteria."""
    stress_t = nw_t(net_excess(winners, market.mkt, stress_cost))
    return {
        "subperiods": [
            {
                "start": start,
                "end": end,
                "n_months": len(winners.loc[start:end]),
                **_stats(winners.loc[start:end], market.mkt.loc[start:end], base_cost),
            }
            for start, end in DIAG
        ],
        "post_2013_cagr_diff_pp": cagr_diff_pp(
            winners.loc[slice(*DIAG[-1])], market.mkt.loc[slice(*DIAG[-1])], base_cost
        ),
        "stress": {
            "annual_cost": stress_cost,
            "full_t": stress_t,
            "full_p": one_sided_p(stress_t),
            "post_cagr_diff_pp": cagr_diff_pp(
                winners.loc[slice(*POST)], market.mkt.loc[slice(*POST)], stress_cost
            ),
        },
        "capm": capm(winners - market.rf, market.mkt - market.rf),
        "worst_12m": worst_rolling_relative(winners, market.mkt),
        "relative_drawdown": relative_drawdown(winners, market.mkt, base_cost),
    }


def screen(
    value: pd.DataFrame,
    profitability: pd.DataFrame,
    investment: pd.DataFrame,
    momentum: pd.DataFrame,
    market: pd.DataFrame,
) -> dict:
    """Validate all required histories before evaluating the fixed A1/A2 family."""
    common_value = _window(value[["small", "big"]], FULL)
    value = _window(value[["big"]], HISTORY)
    momentum = _window(momentum[["big_hi"]], HISTORY)
    market_history = _window(market[["mkt", "rf"]], HISTORY)
    common_market = _window(market_history, FULL)
    profitability = _window(profitability[["small", "big"]], FULL)
    investment = _window(investment[["small", "big"]], FULL)
    combo = value_momentum_combo(value.big, momentum.big_hi)
    history = {"H2": value.big, "H4": combo}
    big = {
        "H1": profitability.big,
        "H2": common_value.big,
        "H3": investment.big,
        "H4": combo.loc[slice(*FULL)],
    }
    small = {
        "SMALL HiOP": profitability.small,
        "SMALL HiBM": common_value.small,
        "SMALL LoINV": investment.small,
    }
    # Validate every diagnostic and stress input before computing any test statistic.
    for hypothesis, winners in big.items():
        _require_wealth(winners, common_market.mkt, HYPOTHESES[hypothesis]["stress_cost"])
    for hypothesis, winners in history.items():
        _require_wealth(winners, market_history.mkt, HYPOTHESES[hypothesis]["base_cost"])
    _require_wealth(momentum.big_hi, market_history.mkt, 0.0)
    for winners in small.values():
        _require_wealth(winners, common_market.mkt, SMALL_STRESS_COST)

    hypotheses = {}
    for hypothesis, winners in big.items():
        config = HYPOTHESES[hypothesis]
        cost = config["base_cost"]
        t = nw_t(net_excess(winners, common_market.mkt, cost))
        delta = cagr_diff_pp(winners.loc[slice(*POST)], common_market.mkt.loc[slice(*POST)], cost)
        diagnostics = _diagnostics(winners, common_market, cost, config["stress_cost"])
        if hypothesis in history:
            diagnostics["history_1927"] = {
                "start": HISTORY[0],
                "end": HISTORY[1],
                "n_months": len(history[hypothesis]),
                **_stats(history[hypothesis], market_history.mkt, cost),
            }
        hypotheses[hypothesis] = {
            **config,
            "A1": {"t": t, "p": one_sided_p(t)},
            "A2": {"cagr_diff_pp": delta, "pass": delta >= POST_BAR_PP},
            "diagnostics": diagnostics,
        }

    pvals = {key: result["A1"]["p"] for key, result in hypotheses.items()}
    rejected = holm(pvals, alpha=HOLM_ALPHA)
    for rank, hypothesis in enumerate(sorted(pvals, key=pvals.get), start=1):
        result = hypotheses[hypothesis]
        result["A1"].update(
            holm_rank=rank,
            holm_threshold=HOLM_ALPHA / (len(pvals) - rank + 1),
            pass_holm=rejected[hypothesis],
        )
        result["overall_pass"] = rejected[hypothesis] and result["A2"]["pass"]
    passers = [key for key, result in hypotheses.items() if result["overall_pass"]]
    advancing = min(
        passers,
        key=lambda key: (pvals[key], -hypotheses[key]["A2"]["cagr_diff_pp"]),
        default=None,
    )
    return {
        "inputs": {
            "common_window": {"start": FULL[0], "end": FULL[1], "n_months": len(common_market)},
            "A2_window": {
                "start": POST[0],
                "end": POST[1],
                "n_months": len(common_market.loc[slice(*POST)]),
            },
            "nw_lags": 12,
            "holm_alpha": HOLM_ALPHA,
            "A2_bar_pp": POST_BAR_PP,
        },
        "hypotheses": hypotheses,
        "advancing_factor": advancing,
        "next": (
            "Factor screen closed NO-GO, no paid data. Propose the next research step."
            if advancing is None
            else "Separate Phase 2 brief for the single advancing factor: paid 2000–2020 "
            "survivorship-free data, exact S&P 500 rules, after-tax vs SPY >= +1.0 pt/yr."
        ),
        "small_cap_diagnostics": {
            name: {
                "base_cost": SMALL_BASE_COST,
                **_stats(winners, common_market.mkt, SMALL_BASE_COST),
                "post_cagr_diff_pp": cagr_diff_pp(
                    winners.loc[slice(*POST)], common_market.mkt.loc[slice(*POST)], SMALL_BASE_COST
                ),
                **_diagnostics(winners, common_market, SMALL_BASE_COST, SMALL_STRESS_COST),
            }
            for name, winners in small.items()
        },
    }
