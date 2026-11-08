"""After-tax simulation of static SPY/SSO fractional leverage (backlog A23) vs SPY buy-and-hold.

Lot-level tax accounting (short/long-term by holding period, distributions, HIFO
long-term-first sells, tax paid the following April from the portfolio, terminal
liquidation), band rebalancing checked at month-end and traded on the next bar, a
labelled synthetic SSO for 1993-2006 calibrated on the real-SSO overlap, and the A24
monthly-SMA sleeve rider. Brief:
research/allocation/static-leverage/2026-09-30-static-fractional-leverage-brief.md.

Not a vectorbt/WFO run: lot accounting needs a stateful loop, and the brief's bar is
after-tax wealth vs SPY, not the lab's Sharpe gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ggTrader.utils.study import bootstrap

ROOT = bootstrap(__file__)

from a23_backtest import load_prices, run

from ggTrader.lab.a23_rebalance import COST_BPS

OUT_JSON = ROOT / "research/allocation/static-leverage/_static_leverage_aftertax_results.json"

BRACKETS = {"24/15": (0.24, 0.15), "40.8/23.8": (0.408, 0.238)}
STRESS = {
    "2000-02 dot-com (synthetic SSO)": ("2000-03-24", "2002-10-09"),
    "2008 GFC (real SSO)": ("2007-10-09", "2009-03-09"),
    "2011": ("2011-04-29", "2011-10-03"),
    "2015-16": ("2015-05-20", "2016-02-11"),
    "2020 COVID": ("2020-02-19", "2020-03-23"),
    "2022": ("2022-01-03", "2022-10-12"),
}


def stress(navs: dict[str, pd.Series]) -> list[dict]:
    rows = []
    for name, (a, b) in STRESS.items():
        row = {"episode": name}
        for k, nav in navs.items():
            seg = nav.loc[a:b]
            if len(seg) < 2:
                row[k] = None
                continue
            row[k] = round(float(seg.iloc[-1] / seg.iloc[0] - 1), 4)
        rows.append(row)
    return rows


def recovery_days(nav: pd.Series, trough: str) -> int | None:
    t = pd.Timestamp(trough)
    peak = nav.loc[:t].max()
    after = nav.loc[t:]
    hit = after[after >= peak]
    return int(len(after.loc[: hit.index[0]])) if len(hit) else None


# ----------------------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--spread", type=float, default=0.0075, help="financing spread over T-bill")
    p.add_argument("--main-start", default="2006-06-21")
    p.add_argument("--end", default="2026-09-29")
    p.add_argument(
        "--calib-end",
        default="2026-09-30",
        help="last day of the real-SSO calibration overlap (pinned for reproducibility)",
    )
    p.add_argument("--holdout", nargs=2, default=["1993-02-01", "2006-06-20"])
    p.add_argument("--rf-offset", type=float, default=0.0, help="add to T-bill rate (annual)")
    p.add_argument("--main-only", action="store_true", help="skip holdout/stitched windows")
    p.add_argument("--arms", default="", help="comma list of arm names to run (default all)")
    p.add_argument("--out", default=str(OUT_JSON))
    p.add_argument("--frozen-inputs", type=Path, help="checksummed A23 input directory")
    p.add_argument(
        "--sso-dist-tax",
        choices=["lt", "st"],
        default="lt",
        help="tax treatment of SSO distributions (lt or st, default lt)",
    )
    p.add_argument(
        "--bil-dist-tax",
        choices=["lt", "st"],
        default="lt",
        help="tax treatment of BIL distributions (lt or st, default lt)",
    )
    a = p.parse_args()

    close, dist, calib = load_prices(a.spread, a.rf_offset, a.calib_end, a.frozen_inputs)
    close, dist = close.loc[: a.end], dist.loc[: a.end]
    print("synthetic calibration:", calib)

    dist_tax = {"SSO": a.sso_dist_tax, "BIL": a.bil_dist_tax}

    arms = {
        "SPY buy-and-hold": dict(w_sso=0.0, band=9.9, rebalance="never"),
        "static 1.25x (75/25) band 8pp": dict(w_sso=0.25, band=0.08),
        "static 1.3x (70/30) band 8pp": dict(w_sso=0.30, band=0.08),
        "static 1.5x (50/50) band 10pp": dict(w_sso=0.50, band=0.10),
        "static 2.0x (SSO) no rebalance": dict(w_sso=1.0, band=9.9, rebalance="never"),
        "idealized 1.3x daily rebalance": dict(w_sso=0.30, band=0.0, rebalance="daily"),
        "A24 rider: 60 SPY + 40 sleeve (10m SMA +/-2%)": dict(
            w_sso=0.40, band=0.08, rider_a24=True
        ),
    }
    windows = {
        "main_real_sso": (a.main_start, a.end),
        "holdout_synthetic": tuple(a.holdout),
        "full_1993_2026_stitched": (a.holdout[0], a.end),
    }
    if a.arms:
        keep = [x.strip() for x in a.arms.split(",")]
        arms = {k: v for k, v in arms.items() if k in keep}
    if a.main_only:
        windows = {"main_real_sso": windows["main_real_sso"]}
    results: dict = {
        "calibration": calib,
        "brackets": BRACKETS,
        "cost_bps_per_side": COST_BPS,
        "dist_tax": dist_tax,
        "windows": {},
    }
    navs_full: dict[str, pd.Series] = {}
    for wname, (s, e) in windows.items():
        results["windows"][wname] = {}
        for arm, kw in arms.items():
            row = {}
            for bname, (st, lt) in BRACKETS.items():
                for wash in (False, True):
                    res = run(
                        close,
                        dist,
                        s,
                        e,
                        st=st,
                        lt=lt,
                        wash=wash,
                        dist_tax=dist_tax,
                        **kw,
                    )
                    if wname == "full_1993_2026_stitched" and bname == "24/15" and not wash:
                        navs_full[arm] = res.pop("_nav")
                    else:
                        res.pop("_nav")
                    row[f"{bname}{' wash' if wash else ''}"] = res
            results["windows"][wname][arm] = row
            base = row["24/15"]
            hi = row["40.8/23.8"]
            print(
                f"{wname:24s} {arm:48s} preCAGR {base['pre_tax_cagr']:6.2%} "
                f"Sharpe {base['pre_tax_sharpe']:5.2f} DD {base['max_dd']:7.2%} | "
                f"afterTax CAGR 24/15 {base['after_tax_cagr']:6.2%} "
                f"40.8/23.8 {hi['after_tax_cagr']:6.2%} | tax {base['taxes_paid_total']:>9,.0f}"
            )
    # beta-matched expectation on the main window (pre-tax)
    m = results["windows"]["main_real_sso"]
    rf_ann = float(close.loc[a.main_start : a.end, "rf"].mean() * 252)
    spy = m["SPY buy-and-hold"]["24/15"]["pre_tax_cagr"]
    results["beta_matched_pretax_cagr"] = {
        "rf_ann": round(rf_ann, 4),
        "1.3x": round(rf_ann + 1.3 * (spy - rf_ann), 4),
        "1.5x": round(rf_ann + 1.5 * (spy - rf_ann), 4),
    }
    if not navs_full:
        Path(a.out).write_text(json.dumps(results, indent=2, default=str))
        print("wrote", a.out)
        return
    results["stress_panel_pretax"] = stress(navs_full)
    results["recovery_days"] = {
        arm: {
            "2009-03-09": recovery_days(nav, "2009-03-09"),
            "2020-03-23": recovery_days(nav, "2020-03-23"),
        }
        for arm, nav in navs_full.items()
    }
    # worst rolling 10y pre-tax relative CAGR vs SPY (full stitched)
    spy_nav = navs_full["SPY buy-and-hold"]
    roll = {}
    for arm, nav in navs_full.items():
        rel = (nav / nav.shift(2520)) ** (1 / 10) - (spy_nav / spy_nav.shift(2520)) ** (1 / 10)
        rel = rel.dropna()
        roll[arm] = {
            "worst_10y_rel_cagr": round(float(rel.min()), 4),
            "worst_10y_end": str(rel.idxmin().date()),
            "share_10y_windows_behind_spy": round(float((rel < 0).mean()), 3),
        }
    results["rolling_10y_pretax_vs_spy"] = roll
    Path(a.out).write_text(json.dumps(results, indent=2, default=str))
    print("\nstress (pre-tax, full stitched):")
    print(pd.DataFrame(results["stress_panel_pretax"]).to_string(index=False))
    print("\nrolling 10y vs SPY:", json.dumps(roll, indent=1))
    print("recovery days:", results["recovery_days"])
    print("beta-matched:", results["beta_matched_pretax_cagr"])
    print("wrote", a.out)


if __name__ == "__main__":
    main()
