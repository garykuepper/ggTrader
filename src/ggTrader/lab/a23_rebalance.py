"""Pure A23 lot accounting and advisory rebalancing rules."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

COST_BPS = 1.0  # per side, unchanged from the frozen A23 study
TARGETS = {"SPY": 0.70, "SSO": 0.30, "BIL": 0.0}
SSO_BAND = 0.08


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

    def distributions(
        self, s: str, per_share: float, px: float, date: pd.Timestamp, taxed_as: str = "lt"
    ) -> None:
        if taxed_as not in ("lt", "st"):
            raise ValueError(f"Invalid tax label: {taxed_as!r}")
        q = self.qty(s)
        if q <= 0 or per_share <= 0:
            return
        amt = q * per_share
        if taxed_as == "st":
            self.year_st += amt
        else:
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


def _trade_to(
    book: Book, tg: dict, px: pd.Series, t: pd.Timestamp, *, skip_buy_sold: bool = False
) -> None:
    v = book.value(px)
    sold = set()
    for s in ("SPY", "SSO", "BIL"):  # sells first
        d = book.qty(s) * px[s] - tg[s] * v
        if d > 1.0:
            book.sell(s, d, px[s], t)
            sold.add(s)
    for s in ("SPY", "SSO", "BIL"):
        if skip_buy_sold and s in sold:
            continue
        d = tg[s] * v - book.qty(s) * px[s]
        if d > 1.0:
            book.buy(s, min(d, book.cash), px[s], t)


def band_trigger(weight: float, target: float, band: float = SSO_BAND) -> bool:
    """Return whether a weight is outside the inclusive A23 band."""
    return weight < target - band or weight > target + band


def check_rebalance(
    lots: dict[str, list[Lot]],
    prices: dict[str, float],
    asof: pd.Timestamp,
    next_session: pd.Timestamp,
    month_end: bool,
    cash: float = 0.0,
    new_cash: float = 0.0,
    st_rate: float = 0.24,
    lt_rate: float = 0.15,
    execution_prices: dict[str, float] | None = None,
) -> dict:
    """Build an indicative A23 plan without changing the supplied holdings."""
    asof, next_session = pd.Timestamp(asof).normalize(), pd.Timestamp(next_session).normalize()
    if next_session <= asof:
        raise ValueError("next_session must follow asof")
    if any(not math.isfinite(x) or x < 0 for x in (cash, new_cash)):
        raise ValueError("cash and new_cash must be finite and nonnegative")
    if any(not math.isfinite(x) or not 0 <= x <= 1 for x in (st_rate, lt_rate)):
        raise ValueError("tax rates must be finite fractions between 0 and 1")
    if set(lots) - {"SPY", "SSO"}:
        raise ValueError("lots may contain only SPY and SSO")
    marks = execution_prices or prices
    if any(s not in prices or s not in marks for s in ("SPY", "SSO")):
        raise ValueError("SPY and SSO prices are required")
    if any(
        not math.isfinite(px[s]) or px[s] <= 0 for px in (prices, marks) for s in ("SPY", "SSO")
    ):
        raise ValueError("SPY and SSO prices must be finite and positive")

    book = Book(st_rate, lt_rate, wash=False)
    original: dict[str, list[tuple[Lot, float]]] = {"SPY": [], "SSO": []}
    for symbol in ("SPY", "SSO"):
        for lot in lots.get(symbol, []):
            if lot.date > asof or lot.qty <= 0 or lot.basis < 0:
                raise ValueError(f"invalid {symbol} lot: acquisition, shares, or basis")
            if not math.isfinite(lot.qty) or not math.isfinite(lot.basis):
                raise ValueError(f"invalid {symbol} lot: non-finite shares or basis")
            copy = Lot(pd.Timestamp(lot.date).normalize(), lot.qty, lot.basis)
            book.lots[symbol].append(copy)
            original[symbol].append((copy, copy.qty))

    invested = sum(book.qty(s) * prices[s] for s in ("SPY", "SSO"))
    if invested <= 0:
        raise ValueError("at least one SPY or SSO share is required")
    weight = book.qty("SSO") * prices["SSO"] / invested
    result = {
        "asof": str(asof.date()),
        "next_session": str(next_session.date()),
        "month_end": month_end,
        "sso_weight": weight,
        "band": [0.22, 0.38],
        "status": "no action until month-end" if not month_end else "within band",
        "buys": [],
        "sales": [],
        "warnings": [],
        "st_fallback": False,
        "leftover_cash": cash + new_cash,
    }
    if not month_end or not band_trigger(weight, TARGETS["SSO"]):
        return result

    marked_value = sum(book.qty(s) * marks[s] for s in ("SPY", "SSO"))
    marked_sso = book.qty("SSO") * marks["SSO"]
    if marked_sso / marked_value < TARGETS["SSO"]:
        needed = (TARGETS["SSO"] * marked_value - marked_sso) / TARGETS["SPY"]
    else:
        needed = marked_sso / TARGETS["SSO"] - marked_value
    book.cash = min(cash + new_cash, max(needed, 0.0))
    unused_cash = cash + new_cash - book.cash
    _trade_to(book, TARGETS, pd.Series({**marks, "BIL": 1.0}), next_session, skip_buy_sold=True)
    result["leftover_cash"] = unused_cash + book.cash

    for symbol in ("SPY", "SSO"):
        for lot, initial_qty in original[symbol]:
            sold = initial_qty - lot.qty
            if sold <= 1e-12:
                continue
            gain = sold * (marks[symbol] * (1 - COST_BPS / 1e4) - lot.basis)
            term = "LT" if lot.date + pd.DateOffset(years=1) < next_session else "ST"
            result["sales"].append(
                {
                    "symbol": symbol,
                    "acquired_date": str(lot.date.date()),
                    "cost_basis_per_share": lot.basis,
                    "shares": sold,
                    "estimated_proceeds": sold * marks[symbol] * (1 - COST_BPS / 1e4),
                    "term": term,
                    "estimated_gain": gain,
                    "estimated_tax": gain * (lt_rate if term == "LT" else st_rate),
                }
            )
            if term == "ST":
                result["st_fallback"] = True
        for lot in book.lots[symbol]:
            if lot.date == next_session:
                result["buys"].append(
                    {
                        "symbol": symbol,
                        "shares": lot.qty,
                        "estimated_cost": lot.qty * marks[symbol] * (1 + COST_BPS / 1e4),
                    }
                )

    result["sales"].sort(key=lambda sale: (sale["term"] == "ST", -sale["cost_basis_per_share"]))

    # shortcut: open lots omit closed buys and reinvestments; upgrade if history is supplied.
    bought = {buy["symbol"] for buy in result["buys"]}
    for symbol in sorted(
        {sale["symbol"] for sale in result["sales"] if sale["estimated_gain"] < 0}
    ):
        other = "SPY" if symbol == "SSO" else "SSO"
        findings = []
        for candidate in (symbol, other):
            recent = any(
                0 <= (next_session - lot.date).days <= 30
                and (candidate != symbol or lot.qty == initial_qty)
                for lot, initial_qty in original[candidate]
            )
            if recent or candidate in bought:
                qualifier = (
                    "potential substantially identical" if candidate == other else "same symbol"
                )
                findings.append(f"{candidate} ({qualifier}) bought recently or in this plan")
        result["warnings"].append(
            f"WASH-SALE WARNING: {symbol} loss sale; review purchases/reinvestments "
            f"(including dividend reinvestment) of {symbol} and {other} "
            "(review for substantially identical) within 30 days BEFORE or AFTER "
            f"the trade date. Flagged: {', '.join(findings) if findings else 'none'}; "
            "review manually."
        )
    if result["sales"] or result["buys"]:
        result["status"] = "rebalance proposed"
    else:
        result["status"] = "no action: trades below $1 minimum"
    if result["st_fallback"]:
        short_gain = sum(s["estimated_gain"] for s in result["sales"] if s["term"] == "ST")
        result["warnings"].append(
            f"SHORT-TERM LOTS REQUIRED: estimated ST gain/loss ${short_gain:,.2f}."
        )
    return result
