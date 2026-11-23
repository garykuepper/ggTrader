"""Match Yahoo symbols to CRSP stocks using shared daily returns."""

import hashlib
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LinkSettings:
    """Thresholds and dates fixed by calibration before building a release."""

    min_sessions: int
    min_sessions_short: int
    agreement_bar: float
    margin: float
    window_start: date
    window_end: date
    hold_first_days: int


def pairwise_correlation(
    Y: np.ndarray, C: np.ndarray, min_shared: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Pearson correlations on each pair's shared observations."""
    y = np.asarray(Y, dtype=np.float64)
    c = np.asarray(C, dtype=np.float64)
    if y.ndim != 2 or c.ndim != 2 or y.shape[0] != c.shape[0]:
        raise ValueError("returns must be two matrices with matching rows")
    ym = np.isfinite(y).astype(np.float64)
    y = np.where(np.isfinite(y), y, 0.0)
    counts = np.empty((y.shape[1], c.shape[1]), dtype=np.int64)
    correlations = np.full(counts.shape, np.nan)
    for start in range(0, c.shape[1], 2000):
        block = c[:, start : start + 2000]
        cm = np.isfinite(block).astype(np.float64)
        block = np.where(np.isfinite(block), block, 0.0)
        n = ym.T @ cm
        sy, sc = y.T @ cm, ym.T @ block
        ssy, ssc = (y * y).T @ cm, ym.T @ (block * block)
        cross = y.T @ block
        with np.errstate(divide="ignore", invalid="ignore"):
            numerator = cross - sy * sc / n
            denom = np.sqrt(np.maximum(ssy - sy * sy / n, 0) * np.maximum(ssc - sc * sc / n, 0))
            result = numerator / denom
        result[(n < min_shared) | (denom == 0)] = np.nan
        counts[:, start : start + block.shape[1]] = n.astype(np.int64)
        correlations[:, start : start + block.shape[1]] = np.clip(result, -1, 1)
    return counts, correlations


def top_candidates(corr: np.ndarray, k: int = 5) -> np.ndarray:
    """Return best candidate indices, padding unavailable slots with -1."""
    values = np.asarray(corr)
    result = np.full((values.shape[0], k), -1, dtype=int)
    if not values.shape[1] or k <= 0:
        return result
    order = np.argsort(-np.where(np.isfinite(values), values, -np.inf), axis=1, kind="stable")[
        :, :k
    ]
    selected = np.take_along_axis(values, order, axis=1)
    result[:, : order.shape[1]] = np.where(np.isfinite(selected), order, -1)
    return result


def agreement(yahoo_ret: pd.Series, crsp_ret: pd.Series, *, tolerance: float = 0.005) -> dict:
    """Summarize agreement over shared dates and the two date-ordered halves."""
    shared = pd.concat([yahoo_ret.rename("y"), crsp_ret.rename("c")], axis=1).dropna().sort_index()
    shared = shared[np.isfinite(shared).all(axis=1)]

    def stats(frame: pd.DataFrame) -> dict:
        return {
            "n_shared": len(frame),
            "corr": frame.y.corr(frame.c) if len(frame) >= 2 else np.nan,
            "share_within_tol": (frame.y - frame.c).abs().le(tolerance).mean()
            if len(frame)
            else np.nan,
        }

    midpoint = (len(shared) + 1) // 2
    return {
        **stats(shared),
        "first_half": stats(shared.iloc[:midpoint]),
        "second_half": stats(shared.iloc[midpoint:]),
    }


def find_links(
    yahoo_returns: pd.DataFrame,
    crsp_returns: pd.DataFrame,
    yahoo_first: pd.Series,
    crsp_first: pd.Series,
    yahoo_last: pd.Series,
    crsp_last: pd.Series,
    settings: LinkSettings,
    calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Find links while refusing weak, ambiguous, duplicate and outliving matches."""
    if not calendar.is_monotonic_increasing or not calendar.is_unique:
        raise ValueError("calendar must be sorted and unique")
    dates = yahoo_returns.index.union(crsp_returns.index).sort_values()
    y = yahoo_returns.reindex(dates)
    c = crsp_returns.reindex(dates)
    counts, corr = pairwise_correlation(y.to_numpy(), c.to_numpy(), settings.min_sessions_short)
    candidates = top_candidates(corr)
    rows = []
    for i, symbol in enumerate(y.columns):
        row = dict(
            symbol=symbol,
            permno=None,
            status="unlinked",
            corr=np.nan,
            corr_first_half=np.nan,
            corr_second_half=np.nan,
            n_shared=0,
            second_best_corr=np.nan,
            reason="no_candidate",
        )
        indices = candidates[i]
        if indices[0] >= 0:
            j = indices[0]
            permno = c.columns[j]
            details = agreement(y.iloc[:, i], c.iloc[:, j])
            second = corr[i, indices[1]] if len(indices) > 1 and indices[1] >= 0 else np.nan
            row.update(
                permno=permno,
                corr=corr[i, j],
                corr_first_half=details["first_half"]["corr"],
                corr_second_half=details["second_half"]["corr"],
                n_shared=int(counts[i, j]),
                second_best_corr=second,
            )
            yf = pd.Timestamp(yahoo_first.loc[symbol])
            cf = pd.Timestamp(crsp_first.loc[permno])
            yl = pd.Timestamp(yahoo_last.loc[symbol])
            cl = pd.Timestamp(crsp_last.loc[permno])
            life = (
                (calendar >= max(yf, pd.Timestamp(settings.window_start)))
                & (calendar <= min(yl, pd.Timestamp(settings.window_end)))
            ).sum()
            start_gap = calendar.searchsorted(cf, side="right") - calendar.searchsorted(
                yf, side="right"
            )
            enough = counts[i, j] >= settings.min_sessions or (
                life < settings.min_sessions
                and counts[i, j] >= settings.min_sessions_short
                and cf >= yf
                and start_gap <= 1
            )
            if (
                calendar.searchsorted(cl, side="right") - calendar.searchsorted(yl, side="right")
                > 20
            ):
                row["reason"] = "crsp_outlives_yahoo"
            elif not enough:
                row["reason"] = "insufficient_sessions"
            elif not (
                row["corr_first_half"] >= settings.agreement_bar
                and row["corr_second_half"] >= settings.agreement_bar
            ):
                row["reason"] = "agreement_below_bar"
            elif np.isfinite(second) and row["corr"] - second < settings.margin:
                row.update(status="ambiguous", reason="margin")
            else:
                row.update(status="linked", reason="")
        rows.append(row)
    result = pd.DataFrame(
        rows,
        columns=[
            "symbol",
            "permno",
            "status",
            "corr",
            "corr_first_half",
            "corr_second_half",
            "n_shared",
            "second_best_corr",
            "reason",
        ],
    )
    linked = result[result.status == "linked"].sort_values("corr", ascending=False, kind="stable")
    duplicates = linked.index[linked.permno.duplicated()]
    result.loc[duplicates, ["status", "reason"]] = ["duplicate", "duplicate_permno"]
    return result


def join_returns_hash(yahoo_close: pd.Series, first_yf_date: date, k: int) -> str:
    """Hash the first k Yahoo returns after the join, independent of price scale."""
    closes = yahoo_close.sort_index()
    returns = (
        closes.pct_change(fill_method=None)
        .loc[lambda s: s.index > pd.Timestamp(first_yf_date)]
        .iloc[:k]
    )
    encoded = "\n".join(f"{value:.8f}" for value in returns).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()
