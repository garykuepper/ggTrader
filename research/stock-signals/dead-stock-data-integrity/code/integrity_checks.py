"""Run data-quality checks on CRSP daily rows and match flags against held intervals."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

FLAG_COLUMNS = ["check", "permno", "date", "detail"]
INTERVAL_COLUMNS = ["permno", "start", "end"]
# Reference cell from the first test: zero dead-stock loss, 18.3 bps per-side cost, low tax.
REFERENCE_CELL = "loss0_cost18.3bps_taxlow"
PERMNO_PREFIX = "PERMNO-"
# Positions open above 1e-9 shares and close when running quantity returns within 1e-9 of zero.
QTY_TOLERANCE = 1e-9

SPIKE_THRESHOLD = 0.5
SPIKE_REVERSE_BAND = (0.9, 1.1)
GAP_MIN_DAYS = 5
STALE_RUN = 10

# Trade files serialized numpy scalars as text such as "np.float64(337.4789998835294)".
_NUMPY_SCALAR_RE = re.compile(r"^\s*np\.(?:float|int)\d+\((.+)\)\s*$")


def _empty_flags() -> pd.DataFrame:
    """Return an empty flag frame with the standard columns and dtypes."""
    return pd.DataFrame(
        {
            "check": pd.Series(dtype="object"),
            "permno": pd.Series(dtype="int64"),
            "date": pd.Series(dtype="datetime64[ns]"),
            "detail": pd.Series(dtype="object"),
        },
        columns=FLAG_COLUMNS,
    )


def _empty_intervals() -> pd.DataFrame:
    """Return an empty holding-intervals frame with the standard columns and dtypes."""
    return pd.DataFrame(
        {
            "permno": pd.Series(dtype="int64"),
            "start": pd.Series(dtype="datetime64[ns]"),
            "end": pd.Series(dtype="datetime64[ns]"),
        },
        columns=INTERVAL_COLUMNS,
    )


def _build_flags(
    check: str,
    permnos: Any,
    dates: Any,
    details: Any,
) -> pd.DataFrame:
    """Assemble a non-empty flag frame with consistent column order and dtypes."""
    out = pd.DataFrame(
        {
            "check": check,
            "permno": permnos,
            "date": dates,
            "detail": details,
        },
        columns=FLAG_COLUMNS,
    ).reset_index(drop=True)
    if out.empty:
        return _empty_flags()
    out["permno"] = out["permno"].astype("int64")
    out["date"] = pd.to_datetime(out["date"])
    return out


def check_duplicates(rows: pd.DataFrame) -> pd.DataFrame:
    """Flag repeated rows sharing the same (permno, date) pair.

    Args:
        rows: CRSP daily rows with columns ``permno, date, prc, ret, shrout``.

    Returns:
        Flag frame containing every copy after the first for each duplicate ``(permno, date)``.
    """
    if rows.empty:
        return _empty_flags()

    work = rows[["permno", "date"]].copy()
    work["date"] = pd.to_datetime(work["date"])
    dups = work[work.duplicated(subset=["permno", "date"], keep="first")]
    if dups.empty:
        return _empty_flags()

    return _build_flags("duplicate", dups["permno"], dups["date"], "duplicate (permno, date)")


def check_bad_price(rows: pd.DataFrame) -> pd.DataFrame:
    """Flag rows whose price is missing or zero while return is present.

    Negative CRSP prices indicate bid-ask midpoints rather than errors and are allowed.

    Args:
        rows: CRSP daily rows with columns ``permno, date, prc, ret, shrout``.

    Returns:
        Flag frame listing rows with null or zero ``prc`` and non-null ``ret``.
    """
    if rows.empty:
        return _empty_flags()

    prc = pd.to_numeric(rows["prc"], errors="coerce")
    ret = pd.to_numeric(rows["ret"], errors="coerce")
    is_null = prc.isna()
    is_zero = prc == 0.0
    mask = (is_null | is_zero) & ret.notna()
    if not mask.any():
        return _empty_flags()

    details = [
        "prc is null with non-null ret" if null_flag else "prc is zero with non-null ret"
        for null_flag in is_null[mask]
    ]
    return _build_flags("bad_price", rows.loc[mask, "permno"], rows.loc[mask, "date"], details)


def check_spikes(
    rows: pd.DataFrame,
    threshold: float = SPIKE_THRESHOLD,
    reverse_band: tuple[float, float] = SPIKE_REVERSE_BAND,
) -> pd.DataFrame:
    """Flag one-day return spikes that immediately reverse on the next trading row.

    A row is flagged when ``abs(ret) > threshold`` and the two-day compounded gross return
    ``(1 + ret_t) * (1 + ret_{t+1})`` for the same stock lies inside ``reverse_band``.

    Args:
        rows: CRSP daily rows with columns ``permno, date, prc, ret, shrout``.
        threshold: Minimum absolute one-day simple return to treat as a candidate spike.
        reverse_band: Inclusive ``(low, high)`` bounds for the two-day gross return.

    Returns:
        Flag frame containing the spike day (not the reversal day) for each matching event.
    """
    if rows.empty:
        return _empty_flags()

    work = rows[["permno", "date", "ret"]].copy()
    work["date"] = pd.to_datetime(work["date"])
    work["ret"] = pd.to_numeric(work["ret"], errors="coerce")
    work = work.sort_values(["permno", "date"], kind="mergesort")

    next_ret = work.groupby("permno", sort=False)["ret"].shift(-1)
    two_day = (1.0 + work["ret"]) * (1.0 + next_ret)
    low, high = reverse_band
    mask = (work["ret"].abs() > threshold) & (two_day >= low) & (two_day <= high)
    if not mask.any():
        return _empty_flags()

    flagged = work.loc[mask]
    details = [
        f"ret={r1:.4f}, next_ret={r2:.4f}, 2d={td:.4f}"
        for r1, r2, td in zip(flagged["ret"], next_ret.loc[mask], two_day.loc[mask])
    ]
    return _build_flags("spike", flagged["permno"], flagged["date"], details)


def check_gaps(
    rows: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    min_days: int = GAP_MIN_DAYS,
) -> pd.DataFrame:
    """Flag runs of missing trading days inside each stock's own first-to-last span.

    Args:
        rows: CRSP daily rows with columns ``permno, date, prc, ret, shrout``.
        calendar: Reference trading calendar to compare each stock's dates against.
        min_days: Minimum consecutive missing calendar days required to raise a flag.

    Returns:
        Flag frame with one row per gap run, dated on the first missing calendar day and
        ``detail`` formatted as ``"N days missing"``.
    """
    if rows.empty:
        return _empty_flags()

    cal = pd.DatetimeIndex(pd.to_datetime(calendar)).unique().sort_values()
    if cal.empty:
        return _empty_flags()
    # Compare at nanosecond precision so datetime64 unit differences never break searchsorted.
    cal_ns = cal.to_numpy(dtype="datetime64[ns]")

    work = rows[["permno", "date"]].copy()
    work["date"] = pd.to_datetime(work["date"])
    work = work.dropna(subset=["date"])
    if work.empty:
        return _empty_flags()

    records: list[dict[str, Any]] = []
    for permno, group in work.groupby("permno", sort=True):
        stock_dates = np.unique(group["date"].to_numpy(dtype="datetime64[ns]"))
        if len(stock_dates) < 2:
            continue

        lo = int(np.searchsorted(cal_ns, stock_dates[0], side="left"))
        hi = int(np.searchsorted(cal_ns, stock_dates[-1], side="right"))
        span_len = hi - lo
        if span_len < min_days:
            continue

        present = np.zeros(span_len, dtype=bool)
        pos = np.searchsorted(cal_ns, stock_dates)
        in_span = (pos >= lo) & (pos < hi)
        pos_in_span = pos[in_span]
        matched = pos_in_span[cal_ns[pos_in_span] == stock_dates[in_span]]
        present[matched - lo] = True

        missing = ~present
        if not missing.any():
            continue

        padded = np.empty(span_len + 2, dtype=bool)
        padded[0] = False
        padded[1:-1] = missing
        padded[-1] = False
        diff = np.diff(padded.view(np.int8))
        starts = np.flatnonzero(diff == 1)
        ends = np.flatnonzero(diff == -1)

        for start_idx, end_idx in zip(starts, ends):
            run_len = int(end_idx - start_idx)
            if run_len >= min_days:
                records.append(
                    {
                        "check": "gap",
                        "permno": int(permno),
                        "date": cal[lo + int(start_idx)],
                        "detail": f"{run_len} days missing",
                    }
                )

    if not records:
        return _empty_flags()

    out = pd.DataFrame(records, columns=FLAG_COLUMNS)
    out["permno"] = out["permno"].astype("int64")
    out["date"] = pd.to_datetime(out["date"])
    return out


def check_stale(rows: pd.DataFrame, run: int = STALE_RUN) -> pd.DataFrame:
    """Flag stocks whose absolute price stays identical for at least ``run`` rows in a row.

    CRSP sign is stripped via ``abs(prc)`` so toggling between a trade close and a matching
    bid-ask midpoint still counts as an unchanged price. Missing prices break runs.

    Args:
        rows: CRSP daily rows with columns ``permno, date, prc, ret, shrout``.
        run: Minimum number of consecutive identical absolute prices to flag.

    Returns:
        Flag frame dated on the first row of each stale price run.
    """
    if rows.empty:
        return _empty_flags()

    work = rows[["permno", "date", "prc"]].copy()
    work["date"] = pd.to_datetime(work["date"])
    work["abs_prc"] = pd.to_numeric(work["prc"], errors="coerce").abs()
    work = work.sort_values(["permno", "date"], kind="mergesort").reset_index(drop=True)

    same_as_prev = (
        (work["permno"] == work["permno"].shift(1))
        & work["abs_prc"].notna()
        & (work["abs_prc"] == work["abs_prc"].shift(1))
    )
    work["run_id"] = (~same_as_prev).cumsum()

    valid = work[work["abs_prc"].notna()]
    if valid.empty:
        return _empty_flags()

    runs = valid.groupby("run_id", sort=False).agg(
        permno=("permno", "first"),
        date=("date", "first"),
        price=("abs_prc", "first"),
        length=("abs_prc", "size"),
    )
    flagged = runs[runs["length"] >= run]
    if flagged.empty:
        return _empty_flags()

    details = [
        f"{int(length)} rows at {float(price):g}"
        for length, price in zip(flagged["length"], flagged["price"])
    ]
    return _build_flags("stale", flagged["permno"], flagged["date"], details)


def check_shares(rows: pd.DataFrame) -> pd.DataFrame:
    """Flag month-end rows that have a valid price but missing or non-positive shares.

    Only the last available trading row per stock in each calendar month is checked because
    size rankings depend on month-end market cap.

    Args:
        rows: CRSP daily rows with columns ``permno, date, prc, ret, shrout``.

    Returns:
        Flag frame listing month-end rows where ``abs(prc) > 0`` and ``shrout`` is null or <= 0.
    """
    if rows.empty:
        return _empty_flags()

    work = rows[["permno", "date", "prc", "shrout"]].copy()
    work["date"] = pd.to_datetime(work["date"])
    work["prc"] = pd.to_numeric(work["prc"], errors="coerce")
    work["shrout"] = pd.to_numeric(work["shrout"], errors="coerce")
    work = work.dropna(subset=["date"])
    if work.empty:
        return _empty_flags()

    work = work.sort_values(["permno", "date"], kind="mergesort")
    last = work.groupby(
        [work["permno"], work["date"].dt.year, work["date"].dt.month],
        sort=False,
    ).tail(1)

    valid_price = last["prc"].notna() & (last["prc"].abs() > 0.0)
    bad_shares = last["shrout"].isna() | (last["shrout"] <= 0.0)
    flagged = last[valid_price & bad_shares]
    if flagged.empty:
        return _empty_flags()

    details = [
        "shrout is null on month-end" if pd.isna(val) else f"shrout={float(val):g} on month-end"
        for val in flagged["shrout"]
    ]
    return _build_flags("shares", flagged["permno"], flagged["date"], details)


def _parse_scalar_float(value: Any) -> float:
    """Convert plain numeric text or ``np.float64(...)`` text into a Python float."""
    if isinstance(value, (int, float, np.number)):
        return float(value)
    if pd.isna(value):
        return float("nan")
    text = str(value).strip()
    match = _NUMPY_SCALAR_RE.match(text)
    if match:
        text = match.group(1).strip()
    return float(text)


def read_trades(path: Path) -> pd.DataFrame:
    """Read a first-test trade file and parse numpy-formatted quantities into floats.

    Args:
        path: Path to a ``.csv`` or ``.csv.gz`` trade log produced by the first test.

    Returns:
        Trade log DataFrame with ``qty`` (and ``price`` when present) as ``float64`` and
        ``date`` parsed as ``datetime64[ns]``.
    """
    df = pd.read_csv(Path(path))
    if "qty" in df.columns:
        df["qty"] = pd.Series(
            [_parse_scalar_float(v) for v in df["qty"]],
            index=df.index,
            dtype="float64",
        )
    if "price" in df.columns:
        df["price"] = pd.Series(
            [_parse_scalar_float(v) for v in df["price"]],
            index=df.index,
            dtype="float64",
        )
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def holding_intervals(trades: pd.DataFrame, cell: str) -> pd.DataFrame:
    """Reconstruct open-to-close holding intervals for ``PERMNO-<n>`` symbols in one cell.

    A position opens on the first trade where running quantity exceeds ``QTY_TOLERANCE``
    (``1e-9``) and closes when running quantity returns within ``QTY_TOLERANCE`` of zero.
    If a position remains open at the end of the trade log, its ``end`` is ``NaT``.

    Args:
        trades: Trade log DataFrame with columns ``cell, date, symbol, qty``.
        cell: Exact cell label to filter on (for example ``REFERENCE_CELL``).

    Returns:
        DataFrame with columns ``permno, start, end``.
    """
    if trades.empty:
        return _empty_intervals()

    sub = trades[trades["cell"] == cell]
    if sub.empty:
        return _empty_intervals()

    symbols = sub["symbol"].astype(str)
    is_permno = symbols.str.startswith(PERMNO_PREFIX)
    sub = sub[is_permno]
    if sub.empty:
        return _empty_intervals()

    work = pd.DataFrame(
        {
            "permno": symbols[is_permno].str.slice(len(PERMNO_PREFIX)).astype("int64"),
            "date": pd.to_datetime(sub["date"]),
            "qty": pd.to_numeric(sub["qty"], errors="raise").astype("float64"),
        }
    ).sort_values(["permno", "date"], kind="mergesort")

    records: list[dict[str, Any]] = []
    for permno, group in work.groupby("permno", sort=True):
        running = 0.0
        open_start: pd.Timestamp | None = None
        for trade_date, delta in zip(group["date"], group["qty"]):
            running += float(delta)
            if open_start is None:
                if running > QTY_TOLERANCE:
                    open_start = trade_date
            elif abs(running) <= QTY_TOLERANCE:
                records.append({"permno": int(permno), "start": open_start, "end": trade_date})
                open_start = None
                # Zero out tiny floating-point dust so a later re-entry starts clean.
                running = 0.0
        if open_start is not None:
            records.append({"permno": int(permno), "start": open_start, "end": pd.NaT})

    if not records:
        return _empty_intervals()

    out = pd.DataFrame(records, columns=INTERVAL_COLUMNS)
    out["permno"] = out["permno"].astype("int64")
    out["start"] = pd.to_datetime(out["start"])
    out["end"] = pd.to_datetime(out["end"])
    return out


def flags_inside_holdings(flags: pd.DataFrame, intervals: pd.DataFrame) -> pd.DataFrame:
    """Return the subset of flags whose date falls inside a held interval for the same stock.

    Both interval endpoints ``start`` and ``end`` are inclusive; an interval with ``end``
    equal to ``NaT`` is treated as still open from ``start`` onward. Each matching flag row
    is returned at most once, preserving the original order and columns of ``flags``.

    Args:
        flags: Flag DataFrame with at least ``permno`` and ``date`` columns.
        intervals: Output of ``holding_intervals`` with ``permno, start, end``.

    Returns:
        Copy of the matching rows from ``flags`` (empty frame with ``flags``'s columns if none).
    """
    if flags.empty or intervals.empty:
        return flags.iloc[0:0].copy()

    flag_keys = pd.DataFrame(
        {
            "permno": flags["permno"].astype("int64").to_numpy(),
            "date": pd.to_datetime(flags["date"]).to_numpy(),
            "_flag_idx": np.arange(len(flags)),
        }
    )
    iv_keys = pd.DataFrame(
        {
            "permno": intervals["permno"].astype("int64").to_numpy(),
            "start": pd.to_datetime(intervals["start"]).to_numpy(),
            "end": pd.to_datetime(intervals["end"]).to_numpy(),
        }
    )

    merged = flag_keys.merge(iv_keys, on="permno", how="inner")
    if merged.empty:
        return flags.iloc[0:0].copy()

    in_interval = (merged["date"] >= merged["start"]) & (
        merged["end"].isna() | (merged["date"] <= merged["end"])
    )
    matched_idx = np.sort(merged.loc[in_interval, "_flag_idx"].unique())
    return flags.iloc[matched_idx].reset_index(drop=True).copy()
