"""Evaluate the pre-registered long-only BIG net-repurchaser screen."""

from pathlib import Path

import pandas as pd

from ggTrader.lab.french_data import FRENCH_DIR, _read_frozen_csv, parse_french_monthly
from ggTrader.lab.momentum_screen import (
    _window,
    cagr_diff_pp,
    capm,
    net_excess,
    nw_t,
    one_sided_p,
    relative_drawdown,
    worst_rolling_relative,
)

SIZE_NI_FILE = "25_Portfolios_ME_NI_5x5_CSV.zip"
UNI_NI_FILE = "Portfolios_Formed_on_NI_CSV.zip"
SIZE_NI_SECTION = "Average Value Weighted Returns -- Monthly"
UNI_NI_SECTION = "Value Weighted Returns -- Monthly"
BIG_COLUMNS = (
    "BIG NegNI",
    "BIG ZeroNI",
    "BIG LoNI",
    "ME5 NI2",
    "ME5 NI3",
    "ME5 NI4",
    "BIG HiNI",
)
FULL = ("1963-07", "2026-04")
POST = ("1994-01", "2026-04")
POST_PUBLICATION = ("2009-01", "2026-04")
SUBPERIODS = (("1963-07", "1993-12"), ("1994-01", "2008-12"), POST_PUBLICATION)
BASE_COST = 0.01
STRESS_COST = 0.02
A1_T_BAR = 3.0
A1_P_BAR = 0.00135
A2_BAR_PP = 1.0


def load_net_issuance_vw(dest: Path = FRENCH_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read both verified value-weighted tables with their exact registered headers."""
    size = parse_french_monthly(_read_frozen_csv(SIZE_NI_FILE, dest), SIZE_NI_SECTION)
    if list(size.columns[-len(BIG_COLUMNS) :]) != list(BIG_COLUMNS):
        raise ValueError(f"Unexpected French BIG net-issuance headers: {list(size.columns)}")

    text = _read_frozen_csv(UNI_NI_FILE, dest)
    lines = text.splitlines()
    section = next((i for i, line in enumerate(lines) if line.strip() == UNI_NI_SECTION), None)
    if (
        section is None
        or section + 1 == len(lines)
        or lines[section + 1].split(",")[1:2] != ["< 0 "]
    ):
        raise ValueError("Missing or changed French univariate '< 0 ' header")
    univariate = parse_french_monthly(text, UNI_NI_SECTION)
    if not len(univariate.columns) or univariate.columns[0] != "< 0":
        raise ValueError("Missing or changed French univariate '< 0 ' column")
    return size[list(BIG_COLUMNS)], univariate[["< 0"]]


def _cell_stats(winners: pd.Series, market: pd.Series) -> dict:
    """Report both registered costs over full, since-1994, and post-2008 windows."""
    result = {}
    for label, cost in (("base", BASE_COST), ("stress", STRESS_COST)):
        t = nw_t(net_excess(winners, market, cost))
        result[label] = {
            "full_t": t,
            "full_p": one_sided_p(t),
            "since_1994_cagr_diff_pp": cagr_diff_pp(
                winners.loc[slice(*POST)], market.loc[slice(*POST)], cost
            ),
            "post_2008_cagr_diff_pp": cagr_diff_pp(
                winners.loc[slice(*POST_PUBLICATION)],
                market.loc[slice(*POST_PUBLICATION)],
                cost,
            ),
        }
    return result


def screen(size: pd.DataFrame, univariate: pd.DataFrame, market: pd.DataFrame) -> dict:
    """Stop on bad inputs before applying the single H1 significance and CAGR gates."""
    size = _window(size[list(BIG_COLUMNS)], FULL)
    univariate = _window(univariate[["< 0"]], FULL)
    market = _window(market[["mkt", "rf"]], FULL)
    if (market.mkt <= -1).any() or (size <= STRESS_COST / 12 - 1).any().any():
        raise ValueError("Non-positive monthly market or BIG wealth factor after stress cost")
    if (univariate["< 0"] <= STRESS_COST / 12 - 1).any():
        raise ValueError("Non-positive monthly univariate wealth factor after stress cost")

    winners = size["BIG NegNI"]
    cells = {name: _cell_stats(size[name], market.mkt) for name in BIG_COLUMNS}
    uni_stats = _cell_stats(univariate["< 0"], market.mkt)
    t = cells["BIG NegNI"]["base"]["full_t"]
    p = cells["BIG NegNI"]["base"]["full_p"]
    delta = cells["BIG NegNI"]["base"]["since_1994_cagr_diff_pp"]
    a1_pass = t >= A1_T_BAR and p <= A1_P_BAR
    a2_pass = delta >= A2_BAR_PP
    overall_pass = a1_pass and a2_pass
    return {
        "inputs": {
            "full_window": {"start": FULL[0], "end": FULL[1], "n_months": len(market)},
            "A2_window": {
                "start": POST[0],
                "end": POST[1],
                "n_months": len(market.loc[slice(*POST)]),
            },
            "post_2008_window": {
                "start": POST_PUBLICATION[0],
                "end": POST_PUBLICATION[1],
                "n_months": len(market.loc[slice(*POST_PUBLICATION)]),
            },
            "nw_lags": 12,
            "base_cost": BASE_COST,
            "stress_cost": STRESS_COST,
            "A1_t_bar": A1_T_BAR,
            "A1_p_bar": A1_P_BAR,
            "A2_bar_pp": A2_BAR_PP,
        },
        "H1": {
            "portfolio": "BIG NegNI",
            "A1": {"t": t, "p": p, "pass": a1_pass},
            "A2": {"cagr_diff_pp": delta, "pass": a2_pass},
            "overall_pass": overall_pass,
            "diagnostics": {
                "post_2008_cagr_diff_pp": cells["BIG NegNI"]["base"]["post_2008_cagr_diff_pp"],
                "subperiods": [
                    {
                        "start": start,
                        "end": end,
                        "n_months": len(winners.loc[start:end]),
                        "t": (
                            t_sub := nw_t(
                                net_excess(
                                    winners.loc[start:end], market.mkt.loc[start:end], BASE_COST
                                )
                            )
                        ),
                        "p": one_sided_p(t_sub),
                        "cagr_diff_pp": cagr_diff_pp(
                            winners.loc[start:end], market.mkt.loc[start:end], BASE_COST
                        ),
                    }
                    for start, end in SUBPERIODS
                ],
                "stress": {
                    "full_t": cells["BIG NegNI"]["stress"]["full_t"],
                    "full_p": cells["BIG NegNI"]["stress"]["full_p"],
                    "since_1994_cagr_diff_pp": cells["BIG NegNI"]["stress"][
                        "since_1994_cagr_diff_pp"
                    ],
                },
                "capm": capm(winners - market.rf, market.mkt - market.rf),
                "worst_12m": worst_rolling_relative(winners, market.mkt),
                "relative_drawdown": relative_drawdown(winners, market.mkt, BASE_COST),
            },
        },
        "issuance_shape": cells,
        "univariate_repurchasers": {"portfolio": "< 0 ", "includes_small_caps": True, **uni_stats},
        "next": (
            "Write a separate fixed-rules candidate brief, then B0 free PIT kill-only replay and "
            "B1 free forward shadow study under the approved standard. No paid data."
            if overall_pass
            else "Close this net-issuance screen NO-GO. Publish failed gates and diagnostics, "
            "use free data only, and propose the next research step."
        ),
    }
