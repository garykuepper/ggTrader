"""Frozen daily inputs and taxable static leverage mechanics for the century study."""

from __future__ import annotations

import io
import json
import zipfile
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.static_leverage_aftertax import Book, _trade_to

START = "1926-07-01"
END = "2026-06-30"
FRENCH_FILE = "F-F_Research_Data_Factors_daily_CSV.zip"
SHILLER_FILE = "ie_data.xls"
INITIAL = 100_000.0
MIXES = (0.0, 0.15, 0.30, 0.45, 0.60)
BRACKETS = {"24/15": (0.24, 0.15), "40.8/23.8": (0.408, 0.238)}
WINDOWS = {
    "full": (START, END),
    "1926-1945": (START, "1945-12-31"),
    "1946-1972": ("1946-01-01", "1972-12-31"),
    "1973-1981": ("1973-01-01", "1981-12-31"),
    "1982-1999": ("1982-01-01", "1999-12-31"),
    "2000-2026-06": ("2000-01-01", END),
}
KNOWN_CLOSURES = {
    (pd.Timestamp("1933-03-03"), pd.Timestamp("1933-03-15")),
    (pd.Timestamp("2001-09-10"), pd.Timestamp("2001-09-17")),
}


def checked_file(directory: Path, filename: str) -> tuple[Path, dict]:
    """Verify a frozen input and return its manifest record."""
    path = directory / filename
    manifest = json.loads((directory / "MANIFEST.json").read_text())
    record = manifest[filename]
    if sha256(path.read_bytes()).hexdigest() != record["sha256"]:
        raise ValueError(f"Frozen input checksum mismatch: {path}")
    return path, record


def parse_french_daily(text: str) -> pd.DataFrame:
    """Parse the sole daily factor section, rejecting malformed dates and returns."""
    lines = text.splitlines()
    header = ",Mkt-RF,SMB,HML,RF"
    if lines.count(header) != 1:
        raise ValueError("French daily header changed or repeated")
    first = lines.index(header)
    last = next((i for i in range(first + 1, len(lines)) if not lines[i].strip()), None)
    if last is None:
        raise ValueError("French daily section has no terminator")
    frame = pd.read_csv(io.StringIO("\n".join(lines[first:last])), index_col=0)
    if list(frame.columns) != ["Mkt-RF", "SMB", "HML", "RF"]:
        raise ValueError("French daily columns changed")
    labels = frame.index.astype(str)
    if frame.empty or not pd.Index(labels).str.fullmatch(r"\d{8}").all():
        raise ValueError("French daily dates are malformed")
    frame.index = pd.to_datetime(labels, format="%Y%m%d", errors="raise")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("French daily dates are duplicated or unsorted")
    gaps = zip(frame.index[:-1], frame.index[1:], strict=True)
    for before, after in gaps:
        if (after - before).days > 5 and (before, after) not in KNOWN_CLOSURES:
            raise ValueError(f"Unexplained French daily date gap: {before} to {after}")
    required = frame[["Mkt-RF", "RF"]].to_numpy(dtype=float)
    if not np.isfinite(required).all():
        raise ValueError("French daily market or RF contains a non-finite value")
    out = pd.DataFrame(index=frame.index)
    out["mkt"] = (frame["Mkt-RF"].astype(float) + frame["RF"].astype(float)) / 100
    out["rf"] = frame["RF"].astype(float) / 100
    if (1 + out["mkt"] <= 0).any():
        raise ValueError("French market has a nonpositive daily wealth factor")
    return out


def load_french_daily(directory: Path) -> pd.DataFrame:
    """Read the verified French daily archive through the study endpoint."""
    path, _ = checked_file(directory, FRENCH_FILE)
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != ["F-F_Research_Data_Factors_daily.csv"]:
            raise ValueError("French daily archive member changed")
        frame = parse_french_daily(archive.read(archive.namelist()[0]).decode("utf-8-sig"))
    frame = frame.loc[START:END]
    if frame.empty or frame.index[0] != pd.Timestamp(START) or frame.index[-1] != pd.Timestamp(END):
        raise ValueError("French daily window is incomplete")
    return frame


def load_shiller_yield(directory: Path) -> pd.Series:
    """Read monthly annualized S&P dividend yield from frozen Shiller P and D."""
    path, _ = checked_file(directory, SHILLER_FILE)
    raw = pd.read_excel(path, sheet_name="Data", header=None).iloc[8:, [0, 1, 2]].copy()
    raw.columns = ["date", "price", "dividend"]
    raw = raw[pd.to_numeric(raw["date"], errors="coerce").notna()]
    labels = raw["date"].astype(float)
    year = labels.astype(int)
    month = ((labels - year) * 100).round().astype(int)
    if not month.between(1, 12).all():
        raise ValueError("Shiller monthly labels are malformed")
    raw.index = pd.to_datetime(dict(year=year, month=month, day=1))
    raw = raw.loc[START:END]
    expected = pd.date_range("1926-07-01", "2026-06-01", freq="MS")
    if not raw.index.equals(expected):
        raise ValueError("Shiller monthly dates are missing or duplicated")
    values = raw[["price", "dividend"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values[:, 0] <= 0).any() or (values[:, 1] < 0).any():
        raise ValueError("Shiller P/D contains a missing or invalid required value")
    return (raw["dividend"].astype(float) / raw["price"].astype(float)).rename("annual_yield")


def build_legs(
    daily: pd.DataFrame,
    yields: pd.Series,
    fee: float,
    spread: float,
    zero_distribution: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build synthetic 1x price/distribution and daily-reset 2x price legs."""
    dates = daily.index
    month_end = dates.to_series().groupby(dates.to_period("M")).tail(1).index
    dividend_rate = pd.Series(0.0, index=dates)
    if not zero_distribution:
        dividend_rate.loc[month_end] = (
            yields.loc[month_end.to_period("M").to_timestamp()].to_numpy() / 12
        )
    r1 = daily["mkt"] - 0.000945 / 252
    r2 = 2 * daily["mkt"] - daily["rf"] - (fee + spread) / 252
    price_return_1 = r1 - dividend_rate
    if (1 + price_return_1 <= 0).any() or (1 + r2 <= 0).any():
        raise ValueError("Synthetic leg has a nonpositive daily wealth factor")
    close = pd.DataFrame(index=dates)
    close["SPY"] = (1 + price_return_1).cumprod() / (1 + price_return_1.iloc[0])
    close["SSO"] = (1 + r2).cumprod() / (1 + r2.iloc[0])
    close["BIL"] = 1.0
    dist = pd.DataFrame(0.0, index=dates, columns=["SPY", "SSO", "BIL"])
    dist["SPY"] = close["SPY"].shift(1) * dividend_rate
    dist.iloc[0, dist.columns.get_loc("SPY")] = 0.0
    return close, dist


def band_trigger(weight: float, target: float, band: float = 0.08) -> bool:
    """Return whether the SSO weight left its strict target band."""
    return weight < target - band or weight > target + band


def _event_positions(index: pd.DatetimeIndex) -> list[int]:
    """Select dates that can change holdings or tax balances."""
    months = index.to_period("M")
    month_ends = np.flatnonzero(np.r_[months[1:] != months[:-1], True])
    years = index.year
    new_year = np.flatnonzero(np.r_[False, years[1:] != years[:-1]])
    april = np.flatnonzero((index.month == 4) & (index.day >= 15))
    april_first = [i for i in april if i == 0 or index[i - 1].month != 4 or index[i - 1].day < 15]
    return sorted(
        set([0, len(index) - 1, *month_ends, *(month_ends[:-1] + 1), *new_year, *april_first])
    )


def simulate(
    close: pd.DataFrame,
    dist: pd.DataFrame,
    start: str,
    end: str,
    mix: float,
    st_rate: float,
    lt_rate: float,
    wash: bool,
    track_nav: bool = True,
) -> dict:
    """Run A23's tax book at all transaction/tax events; reinitialize every window."""
    idx = close.loc[start:end].index
    if len(idx) < 2:
        raise ValueError(f"Window has fewer than two sessions: {start} to {end}")
    prices = close.loc[idx, ["SPY", "SSO", "BIL"]]
    payouts = dist.loc[idx, ["SPY", "SSO", "BIL"]]
    book = Book(st_rate, lt_rate, wash)
    targets = {"SPY": 1 - mix, "SSO": mix, "BIL": 0.0}
    for symbol, weight in targets.items():
        book.buy(symbol, INITIAL * weight, prices.iloc[0][symbol], idx[0])
    book.cash = 0.0
    nav = np.empty(len(idx)) if track_nav else None
    positions = _event_positions(idx)
    months = idx.to_period("M")
    pending = False
    last_year = idx[0].year
    previous = -1
    for pos in positions:
        if nav is not None and previous + 1 < pos:
            block = prices.iloc[previous + 1 : pos]
            nav[previous + 1 : pos] = (
                block.to_numpy() * np.array([book.qty(s) for s in targets])
            ).sum(axis=1) + book.cash
        date = idx[pos]
        px = prices.iloc[pos]
        if pending:
            _trade_to(book, targets, px, date)
            pending = False
        if payouts.iloc[pos]["SPY"] > 0:
            book.distributions("SPY", payouts.iloc[pos]["SPY"], px["SPY"], date)
        if date.year != last_year:
            book.year_end()
            last_year = date.year
        if date.month == 4 and date.day >= 15 and book.tax_due > 0:
            book.pay_tax(px, date)
        value = book.value(px)
        if nav is not None:
            nav[pos] = value
        if mix and (pos == len(idx) - 1 or months[pos] != months[pos + 1]):
            current = {s: book.qty(s) * px[s] / value for s in targets}
            pending = any(band_trigger(current[s], target) for s, target in targets.items())
        previous = pos
    terminal = book.liquidate(prices.iloc[-1], idx[-1])
    days = (idx[-1] - idx[0]).days
    result = {
        "window": [str(idx[0].date()), str(idx[-1].date())],
        "sessions": len(idx),
        "calendar_days": days,
        "after_tax_terminal": float(terminal),
        "after_tax_cagr": float((terminal / INITIAL) ** (365.2425 / days) - 1),
        "taxes_paid_total": float(book.taxes_paid),
        "realized_st": float(book.realized_st_total),
        "realized_lt": float(book.realized_lt_total),
        "wash_disallowed": float(book.disallowed),
    }
    if nav is not None:
        peaks = np.maximum.accumulate(np.r_[INITIAL, nav])[1:]
        result["max_drawdown"] = float(np.min(nav / peaks - 1))
        result["ruin"] = bool(result["max_drawdown"] <= -0.90)
        completed, peak, peak_pos, underwater = 0, INITIAL, -1, False
        for i, value in enumerate(nav):
            if value >= peak:
                if underwater:
                    completed = max(completed, i - peak_pos)
                peak, peak_pos, underwater = value, i, False
            else:
                underwater = True
        result["longest_completed_recovery_sessions"] = completed
        result["unrecovered_peak_elapsed_sessions"] = (
            len(nav) - 1 - peak_pos if underwater else None
        )
    return result


def recommendation_rule(rows: dict[str, dict]) -> dict:
    """Evaluate every pre-registered condition on unrounded primary rows."""
    baseline = rows["30"]
    low_extra = baseline["full"]["ruin"] or (
        baseline["full"]["after_tax_cagr"] < rows["0"]["full"]["after_tax_cagr"]
    )
    alternatives = {}
    for mix in ("0", "15", "45", "60"):
        full, era = rows[mix]["full"], rows[mix]["1973-1981"]
        checks = {
            "full_cagr_plus_0_5pp": full["after_tax_cagr"] - baseline["full"]["after_tax_cagr"]
            >= 0.005,
            "high_rate_cagr_not_below_30": era["after_tax_cagr"]
            >= baseline["1973-1981"]["after_tax_cagr"],
            "drawdown_within_10pp": full["max_drawdown"] >= baseline["full"]["max_drawdown"] - 0.10,
            "no_ruin": not full["ruin"],
            "below_30_extra": low_extra if int(mix) < 30 else True,
        }
        alternatives[mix] = {"conditions": checks, "passes": all(checks.values())}
    passing = [mix for mix, row in alternatives.items() if row["passes"]]
    return {
        "alternatives": alternatives,
        "passing_alternatives": passing,
        "proposal": max(passing, key=lambda m: rows[m]["full"]["after_tax_terminal"])
        if passing
        else "keep 30",
    }
