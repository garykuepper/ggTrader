"""After-tax simulation of static SPY/SSO fractional leverage (backlog A23) vs SPY buy-and-hold.

Lot-level tax accounting (short/long-term by holding period, distributions, HIFO
long-term-first sells, tax paid the following April from the portfolio, terminal
liquidation), band rebalancing checked at month-end and traded on the next bar, a
labelled synthetic SSO for 1993-2006 calibrated on the real-SSO overlap, and the A24
monthly-SMA sleeve rider. Brief: docs/research/briefs/2026-09-30-static-fractional-leverage.md.

Not a vectorbt/WFO run: lot accounting needs a stateful loop, and the brief's bar is
after-tax wealth vs SPY, not the lab's Sharpe gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, "src")
from ggTrader.lab.fred_data import load_fred_series  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "docs" / "research" / "_static_leverage_aftertax_results.json"

SSO_INCEPTION = pd.Timestamp("2006-06-21")  # first yfinance bar
SSO_ER = 0.0088
COST_BPS = 1.0  # per side
BRACKETS = {"24/15": (0.24, 0.15), "40.8/23.8": (0.408, 0.238)}
STRESS = {
    "2000-02 dot-com (synthetic SSO)": ("2000-03-24", "2002-10-09"),
    "2008 GFC (real SSO)": ("2007-10-09", "2009-03-09"),
    "2011": ("2011-04-29", "2011-10-03"),
    "2015-16": ("2015-05-20", "2016-02-11"),
    "2020 COVID": ("2020-02-19", "2020-03-23"),
    "2022": ("2022-01-03", "2022-10-12"),
}


# ----------------------------------------------------------------------------- data
def load_prices(spread: float, rf_offset: float = 0.0) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Unadjusted closes + per-share distributions for SPY/SSO/BIL, with synthetic SSO pre-2006.

    Returns (close, dist, calib). Synthetic SSO has zero distributions (all return in price),
    which slightly favours it on tax deferral; noted in the report.
    """
    close, dist = {}, {}
    for s in ("SPY", "SSO", "BIL"):
        h = yf.Ticker(s).history(start="1993-01-01", auto_adjust=False)
        h.index = h.index.tz_localize(None).normalize()
        close[s], dist[s] = h["Close"], h["Dividends"].fillna(0.0)
    close, dist = pd.DataFrame(close), pd.DataFrame(dist).fillna(0.0)
    close = close.dropna(subset=["SPY"])
    dist = dist.reindex(close.index).fillna(0.0)

    rf = load_fred_series("DTB3", "1990-01-01", "2030-01-01")
    rf = rf.set_index(pd.to_datetime(rf["date"]))["value"].astype(float) / 100.0
    rf = rf.reindex(close.index, method="ffill").bfill()
    rf = (rf + rf_offset) / 252.0  # rf_offset: funding-regime sensitivity (annual, added)
    # SPY total return (price + distribution)
    r_spy = (close["SPY"] + dist["SPY"]) / close["SPY"].shift(1) - 1
    synth = 2.0 * r_spy - (rf + spread / 252.0) - SSO_ER / 252.0
    # calibrate on the real overlap
    r_sso = (close["SSO"] + dist["SSO"]) / close["SSO"].shift(1) - 1
    ov = pd.concat([r_sso, synth], axis=1, keys=["real", "synth"]).dropna()
    resid = ov["real"] - ov["synth"]
    calib = {
        "overlap": [str(ov.index[0].date()), str(ov.index[-1].date())],
        "residual_ann_mean": round(float(resid.mean() * 252), 5),
        "residual_ann_vol": round(float(resid.std() * np.sqrt(252)), 5),
        "corr_real_synth": round(float(ov["real"].corr(ov["synth"])), 5),
        "spread_used": spread,
        "haircut_applied_pre2006_ann": round(float(min(resid.mean(), 0.0) * 252), 5),
    }
    haircut = min(float(resid.mean()), 0.0)  # only ever penalise the synthetic
    pre = close.index < SSO_INCEPTION
    synth_px = (1 + synth.where(pre, 0.0).fillna(0.0) + np.where(pre, haircut, 0.0)).cumprod()
    # splice: scale synthetic level to meet the first real SSO close
    first_real = close.loc[~pre, "SSO"].iloc[0]
    synth_px = synth_px * first_real / synth_px[pre].iloc[-1] / (1 + synth[~pre].iloc[0])
    sso = close["SSO"].copy()
    sso[pre] = synth_px[pre]
    if rf_offset:
        # funding-regime sensitivity on the REAL series too: SSO finances ~1x NAV, so a
        # higher bill rate is ~1:1 extra drag on the 2x fund. Applied as a daily haircut.
        drag = pd.Series(1.0 - rf_offset / 252.0, index=close.index).where(~pre, 1.0).cumprod()
        sso = sso * drag / drag[~pre].iloc[0]
    close["SSO"] = sso
    dist.loc[pre, "SSO"] = 0.0
    # cash leg: BIL where it exists, else accrue T-bill in a synthetic "BIL"
    bil_first = close["BIL"].first_valid_index()
    cash_px = (1 + rf).cumprod()
    cash_px = cash_px * close.loc[bil_first, "BIL"] / cash_px.loc[bil_first]
    close.loc[close.index < bil_first, "BIL"] = cash_px[close.index < bil_first]
    dist["BIL"] = dist["BIL"].fillna(0.0)
    close["rf"] = rf
    return close.dropna(subset=["SSO", "BIL"]), dist, calib


# ----------------------------------------------------------------------------- tax book
@dataclass
class Lot:
    date: pd.Timestamp
    qty: float
    basis: float  # per share


@dataclass
class Book:
    st_rate: float
    lt_rate: float
    wash: bool
    lots: dict[str, list[Lot]] = field(default_factory=lambda: {"SPY": [], "SSO": [], "BIL": []})
    cash: float = 0.0
    year_st: float = 0.0
    year_lt: float = 0.0  # includes qualified dividends / distributions
    loss_cf: float = 0.0
    tax_due: float = 0.0
    taxes_paid: float = 0.0
    realized_st_total: float = 0.0
    realized_lt_total: float = 0.0
    recent_buys: dict[str, list[pd.Timestamp]] = field(
        default_factory=lambda: {"SPY": [], "SSO": [], "BIL": []}
    )
    disallowed: float = 0.0

    def value(self, px: pd.Series) -> float:
        return self.cash + sum(
            lot.qty * px[s] for s, ls in self.lots.items() for lot in ls if lot.qty > 0
        )

    def qty(self, s: str) -> float:
        return sum(lot.qty for lot in self.lots[s])

    def buy(self, s: str, dollars: float, px: float, date: pd.Timestamp) -> None:
        if dollars <= 0:
            return
        fill = px * (1 + COST_BPS / 1e4)
        self.lots[s].append(Lot(date, dollars / fill, fill))
        self.cash -= dollars
        self.recent_buys[s].append(date)

    def sell(self, s: str, dollars: float, px: float, date: pd.Timestamp) -> float:
        """Sell `dollars` of s, HIFO with long-term lots first. Returns dollars raised."""
        if dollars <= 0 or not self.lots[s]:
            return 0.0
        fill = px * (1 - COST_BPS / 1e4)
        need = min(dollars / fill, self.qty(s))
        one_year = pd.DateOffset(years=1)
        order = sorted(
            self.lots[s], key=lambda lot: ((lot.date + one_year) >= date, -lot.basis)
        )  # LT (False) first, then highest basis
        raised = 0.0
        for lot in order:
            if need <= 1e-12:
                break
            q = min(lot.qty, need)
            gain = q * (fill - lot.basis)
            is_lt = (lot.date + one_year) < date
            if (
                gain < 0
                and self.wash
                and any((date - b).days <= 30 and b != lot.date for b in self.recent_buys[s])
            ):
                # loss disallowed: fold it into the newest lot's basis
                self.disallowed += -gain
                newest = max(self.lots[s], key=lambda x: x.date)
                if newest.qty > 0:
                    newest.basis += -gain / newest.qty
                gain = 0.0
            if is_lt:
                self.year_lt += gain
                self.realized_lt_total += gain
            else:
                self.year_st += gain
                self.realized_st_total += gain
            lot.qty -= q
            need -= q
            raised += q * fill
        self.lots[s] = [lot for lot in self.lots[s] if lot.qty > 1e-12]
        self.cash += raised
        return raised

    def distributions(self, s: str, per_share: float, px: float, date: pd.Timestamp) -> None:
        q = self.qty(s)
        if q <= 0 or per_share <= 0:
            return
        amt = q * per_share
        self.year_lt += amt  # qualified dividends / LT cap-gain distributions at the LT rate
        self.cash += amt
        self.buy(s, amt, px, date)  # reinvest

    def year_end(self) -> None:
        net = self.year_st + self.year_lt - self.loss_cf
        if net <= 0:
            self.loss_cf = -net
            self.tax_due += 0.0
        else:
            self.loss_cf = 0.0
            # losses offset the higher-taxed bucket first
            st = max(self.year_st, 0.0)
            lt = net - st if net > st else 0.0
            st = min(st, net)
            self.tax_due += st * self.st_rate + lt * self.lt_rate
        self.year_st = self.year_lt = 0.0

    def pay_tax(self, px: pd.Series, date: pd.Timestamp) -> None:
        due = self.tax_due
        if due <= 0:
            return
        if self.cash < due:  # raise pro-rata
            total = self.value(px) - self.cash
            for s in ("SPY", "SSO", "BIL"):
                v = self.qty(s) * px[s]
                if v > 0:
                    self.sell(s, (due - self.cash) * v / total * 1.002, px[s], date)
        self.cash -= due
        self.taxes_paid += due
        self.tax_due = 0.0

    def liquidate(self, px: pd.Series, date: pd.Timestamp) -> float:
        for s in ("SPY", "SSO", "BIL"):
            self.sell(s, 1e18, px[s], date)
        self.year_end()
        self.pay_tax(px, date)
        return self.cash


# ----------------------------------------------------------------------------- rules
def run(
    close: pd.DataFrame,
    dist: pd.DataFrame,
    start: str,
    end: str,
    w_sso: float,
    band: float,
    st: float,
    lt: float,
    wash: bool,
    rebalance: str = "band",  # band | daily | never
    rider_a24: bool = False,
    initial: float = 100_000.0,
) -> dict:
    idx = close.loc[start:end].index
    book = Book(st, lt, wash)
    px0 = close.loc[idx[0]]
    sleeve_on = True  # A24: sleeve starts in SSO
    sma = close["SPY"].resample("ME").last().rolling(10).mean()
    m_close = close["SPY"].resample("ME").last()
    targets = {"SPY": 1 - w_sso, "SSO": w_sso, "BIL": 0.0}
    for s, w in targets.items():
        book.buy(s, initial * w, px0[s], idx[0])
    book.cash = 0.0
    nav, pending, last_year = [], None, idx[0].year
    month_ends = set(
        close.loc[start:end].resample("ME").last().index.map(lambda t: idx[idx <= t][-1])
    )
    for i, t in enumerate(idx):
        px = close.loc[t]
        if pending is not None:  # trade decided at prior close, executed at this close
            _trade_to(book, pending, px, t)
            pending = None
        for s in ("SPY", "SSO", "BIL"):
            if dist.at[t, s] > 0:
                book.distributions(s, dist.at[t, s], px[s], t)
        if t.year != last_year:
            book.year_end()
            last_year = t.year
        if t.month == 4 and t.day >= 15 and book.tax_due > 0:
            book.pay_tax(px, t)
        v = book.value(px)
        nav.append(v)
        # decide next trade
        if rebalance == "daily":
            pending = targets
        elif t in month_ends and rebalance == "band":
            tg = dict(targets)
            if rider_a24:
                me = m_close.index[m_close.index >= t]
                me = me[0] if len(me) else None
                if me is not None and not np.isnan(sma.get(me, np.nan)):
                    ratio = m_close[me] / sma[me]
                    if ratio >= 1.02:
                        sleeve_on = True
                    elif ratio <= 0.98:
                        sleeve_on = False
                tg = {
                    "SPY": 1 - w_sso + (0 if sleeve_on else w_sso),
                    "SSO": w_sso if sleeve_on else 0.0,
                    "BIL": 0.0,
                }
            cur = {s: book.qty(s) * px[s] / v for s in ("SPY", "SSO", "BIL")}
            if any(abs(cur[s] - tg[s]) > band for s in tg):
                pending = tg
    nav = pd.Series(nav, index=idx)
    r = nav.pct_change().dropna()
    yrs = len(idx) / 252
    pre_cagr = (nav.iloc[-1] / initial) ** (1 / yrs) - 1
    terminal = book.liquidate(close.loc[idx[-1]], idx[-1])
    return {
        "window": [str(idx[0].date()), str(idx[-1].date())],
        "years": round(yrs, 2),
        "pre_tax_cagr": round(float(pre_cagr), 4),
        "pre_tax_sharpe": round(float(r.mean() / r.std() * np.sqrt(252)), 3),
        "pre_tax_vol": round(float(r.std() * np.sqrt(252)), 4),
        "max_dd": round(float((nav / nav.cummax() - 1).min()), 4),
        "after_tax_terminal": round(float(terminal), 0),
        "after_tax_cagr": round(float((terminal / initial) ** (1 / yrs) - 1), 4),
        "taxes_paid_total": round(float(book.taxes_paid), 0),
        "realized_st": round(float(book.realized_st_total), 0),
        "realized_lt": round(float(book.realized_lt_total), 0),
        "wash_disallowed": round(float(book.disallowed), 0),
        "_nav": nav,
    }


def _trade_to(book: Book, tg: dict, px: pd.Series, t: pd.Timestamp) -> None:
    v = book.value(px)
    for s in ("SPY", "SSO", "BIL"):  # sells first
        d = book.qty(s) * px[s] - tg[s] * v
        if d > 1.0:
            book.sell(s, d, px[s], t)
    for s in ("SPY", "SSO", "BIL"):
        d = tg[s] * v - book.qty(s) * px[s]
        if d > 1.0:
            book.buy(s, min(d, book.cash), px[s], t)


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
    p.add_argument("--holdout", nargs=2, default=["1993-02-01", "2006-06-20"])
    p.add_argument("--rf-offset", type=float, default=0.0, help="add to T-bill rate (annual)")
    p.add_argument("--main-only", action="store_true", help="skip holdout/stitched windows")
    p.add_argument("--arms", default="", help="comma list of arm names to run (default all)")
    p.add_argument("--out", default=str(OUT_JSON))
    a = p.parse_args()

    close, dist, calib = load_prices(a.spread, a.rf_offset)
    close, dist = close.loc[: a.end], dist.loc[: a.end]
    print("synthetic calibration:", calib)

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
        "windows": {},
    }
    navs_full: dict[str, pd.Series] = {}
    for wname, (s, e) in windows.items():
        results["windows"][wname] = {}
        for arm, kw in arms.items():
            row = {}
            for bname, (st, lt) in BRACKETS.items():
                for wash in (False, True):
                    res = run(close, dist, s, e, st=st, lt=lt, wash=wash, **kw)
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
