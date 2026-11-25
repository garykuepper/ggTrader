"""Study logic for the dead-stock first test: the fixed grid, the sanity checks, the guard.

Everything here is pure (frames in, a plain dict out) except the holdout guard
(``resolve_output``, ``claim_holdout``), which touches the results folder.
``run.py`` wires the database reads and writes the JSON.

The fixed grid (the brief's registered values, never tuned): dead-stock loss 1.0, 0.5, 0.0
(the share of its last price a holder loses when a stock stops trading), one-way cost 6.4,
18.3 and 40 basis points of traded value (a basis point is 0.01 percent) and two tax sets.
That is 18 strategy cells per window and universe, plus 6 SPY cells (SPY does not depend on
the loss). The planner does not depend on the loss either, so it runs once.

Turnover is the yearly traded dollars (every buy and sell except the final liquidation, which
only closes the experiment) divided by that year's average pre-tax equity.
"""

from __future__ import annotations

import gzip
import io
import json
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dm_ledger_run import BookResult, TaxParams, run_book, run_spy
from dm_replay import (
    DEAD_EXIT_REASON,
    LIQUIDATION_REASON,
    TargetStep,
    exit_close,
    plan_trades,
    span_end_dates,
)

from ggTrader.lab.wfo.report import curve_summary
from ggTrader.utils.io import atomic_write_bytes

WINDOWS: dict[str, tuple[str, str]] = {
    "dev": ("2005-01-03", "2019-12-31"),
    "holdout": ("2020-01-02", "2024-12-31"),
}
CRSP_END = "2024-12-31"
WARMUP_CALENDAR_DAYS = 500  # closes are read this far before the window for 252 + 21 days
TOP_N = 50
LOOKBACK = 252
SKIP = 21
START_CASH = 100_000.0
LOSSES = (1.0, 0.5, 0.0)
COSTS = (0.00064, 0.00183, 0.0040)  # one way: 6.4, 18.3 and 40 basis points
TAXES: dict[str, tuple[float, float]] = {"low": (0.24, 0.15), "high": (0.35, 0.20)}
REFERENCE = (1.0, 0.00183, "low")  # the cell that coverage and the death count are read from
SPY_TOLERANCE = 1e-3
MONOTONE_TOLERANCE = 1e-9  # relative; a rise above this as the loss grows is reported
HELD_QTY = 1e-9  # shares below this are float dust, not a holding
RESULTS_FOLDER = Path("results") / "dead_stock_first_test"
PIN_RELEASE_ID = 1  # the holdout runs on this release and this raw CRSP load only
PIN_CRSP_LOAD = 1
DECOMPOSITION_COST = 0.00183  # the cost and tax set of the gap decomposition and the audits
DECOMPOSITION_TAX = "low"
HOLDINGS_LOSSES = (1.0, 0.5, 0.0)
YAHOO_KINDS = ("linked_yahoo", "yahoo")
CRSP_ONLY_KINDS = ("placeholder", "spell")
SUSPECT_KIND = "linked_yahoo"  # a confirmed link: the only kind that can be a false death
INFORMATIONAL_ROWS = 20
AUDIT_GRACE_DAYS = 7  # CRSP may run this many calendar days past a Yahoo end without a flag
SPAN_ROW_COLUMNS = ["symbol", "last_date", "has_successor", "total_loss"]


def _bps(cost: float) -> str:
    return f"{cost * 1e4:g}"


def cell_key(loss: float, cost: float, tax: str) -> str:
    """The dictionary key of one strategy cell."""
    return f"loss{loss:g}_cost{_bps(cost)}bps_tax{tax}"


def spy_key(cost: float, tax: str) -> str:
    """The dictionary key of one SPY cell."""
    return f"cost{_bps(cost)}bps_tax{tax}"


REFUSED = (
    "a holdout attempt was already started; only the owner may delete the claim. "
    "The holdout is looked at once."
)


def holdout_files(universe: str, results_dir: Path) -> tuple[Path, Path]:
    """The holdout's result file and its claim file in ``results_dir``."""
    return results_dir / f"holdout-{universe}.json", results_dir / f"holdout-{universe}.claim"


def resolve_output(window: str, universe: str, out: Path | None, results_dir: Path) -> Path:
    """Where this run writes, refusing a holdout that was already started.

    The holdout always writes ``holdout-<universe>.json`` in ``results_dir`` (``out`` is
    ignored) and stops with ``SystemExit`` naming the file when the result or its claim
    exists. A development run may be repeated and has no claim.
    """
    if window == "holdout":
        target, claim = holdout_files(universe, results_dir)
        for path in (target, claim):
            if path.exists():
                raise SystemExit(f"refusing to run: {REFUSED} Found {path}.")
        return target
    return out if out is not None else results_dir / f"dev-{universe}.json"


def claim_holdout(universe: str, results_dir: Path, *, git_head: str | None, command: list[str]):
    """Claim the holdout before any holdout data is read: an exclusive create of the claim file.

    Two runs started together cannot both succeed (``O_EXCL``). The claim is never deleted by
    code, so a crash after it leaves it in place on purpose: the owner decides.

    Returns:
        The claim path.

    Raises:
        SystemExit: the claim or the result already exists, naming the file.
    """
    target, claim = holdout_files(universe, results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise SystemExit(f"refusing to run: {REFUSED} Found {target}.")
    try:
        fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        raise SystemExit(f"refusing to run: {REFUSED} Found {claim}.") from None
    note = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "git_head": git_head,
        "command": command,
    }
    with os.fdopen(fd, "w") as handle:
        json.dump(note, handle, indent=2)
        handle.write("\n")
    return claim


def check_holdout_pins(release_id: int, crsp_load: int, *, git_dirty: bool) -> None:
    """Holdout only: refuse unless the pinned release and CRSP load are used on a clean tree.

    Raises:
        SystemExit: naming the failed condition.
    """
    if release_id != PIN_RELEASE_ID:
        raise SystemExit(
            f"holdout refused: --release-id must be {PIN_RELEASE_ID}, got {release_id}"
        )
    if crsp_load != PIN_CRSP_LOAD:
        raise SystemExit(f"holdout refused: --crsp-load must be {PIN_CRSP_LOAD}, got {crsp_load}")
    if git_dirty:
        raise SystemExit(
            "holdout refused: the git working tree has uncommitted changes to tracked files; "
            "commit them first so the run is tied to one commit"
        )


def git_tree_dirty(root: Path) -> bool:
    """True when a tracked file differs from HEAD (``git status --untracked-files=no``)."""
    done = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return bool(done.stdout.strip())


def trades_file(window: str, universe: str, results_dir: Path) -> Path:
    """The side file with every cell's full trade list."""
    return results_dir / f"{window}-{universe}-trades.csv.gz"


def audit_file(window: str, universe: str, results_dir: Path) -> Path:
    """The file the false-death audit writes on its own (``--audit-only``)."""
    return results_dir / f"{window}-{universe}-audit.json"


def refuse_existing_trades_file(window: str, universe: str, results_dir: Path) -> None:
    """Holdout only: the trades side file is written once too.

    Raises:
        SystemExit: the file exists, naming it.
    """
    path = trades_file(window, universe, results_dir)
    if window == "holdout" and path.exists():
        raise SystemExit(f"refusing to run: {REFUSED} Found {path}.")


def write_trades_csv(path: Path, trades: dict[str, list]) -> Path:
    """Write every cell's fills as one gzip CSV (cell, date, symbol, qty, price, reason)."""
    lines = ["cell,date,symbol,qty,price,reason"]
    for cell, fills in trades.items():
        for t in fills:
            lines.append(
                f"{cell},{pd.Timestamp(t.date).date()},{t.symbol},{t.qty!r},{t.price!r},{t.reason}"
            )
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as handle:
        handle.write(("\n".join(lines) + "\n").encode())
    return atomic_write_bytes(path, buffer.getvalue())


WINDOW_EDGE_DAYS = 7  # the window's first and last trading day must lie this close to its ends
MAX_SPY_NAN_RUN = 5  # trading days


def _window_days(index: pd.DatetimeIndex, start: str, end: str) -> pd.DatetimeIndex:
    days = _naive_index(pd.DatetimeIndex(index))
    return days[(days >= pd.Timestamp(start)) & (days <= pd.Timestamp(end))]


def validate_window_inputs(
    close: pd.DataFrame, spy_close: pd.Series, start: str, end: str
) -> dict[str, Any]:
    """Refuse inputs whose strategy and SPY histories do not cover the same window.

    Checks, before any cell runs: (a) both indexes have a first trading day within 7 calendar
    days after ``start`` and a last within 7 days before ``end``; (b) the two first days and
    the two last days inside the window are identical; (c) SPY has a close on its first and
    last day and no run of more than 5 missing closes inside the window.

    Returns:
        Small facts for the results file (the dates and the SPY missing-close count).

    Raises:
        ValueError: naming the problem.
    """
    low, high = pd.Timestamp(start), pd.Timestamp(end)
    edge = pd.Timedelta(days=WINDOW_EDGE_DAYS)
    ends: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for name, index in (("strategy close", close.index), ("SPY close", spy_close.index)):
        days = _window_days(index, start, end)
        if len(days) == 0:
            raise ValueError(f"{name} has no trading day inside {start}..{end}")
        if days[0] > low + edge:
            raise ValueError(f"{name} starts late: first day {days[0].date()} for window {start}")
        if days[-1] < high - edge:
            raise ValueError(f"{name} ends early: last day {days[-1].date()} for window {end}")
        ends[name] = (days[0], days[-1])
    if ends["strategy close"][0] != ends["SPY close"][0]:
        raise ValueError(
            f"first day differs: strategy {ends['strategy close'][0].date()}, "
            f"SPY {ends['SPY close'][0].date()}"
        )
    if ends["strategy close"][1] != ends["SPY close"][1]:
        raise ValueError(
            f"last day differs: strategy {ends['strategy close'][1].date()}, "
            f"SPY {ends['SPY close'][1].date()}"
        )
    stamps = _naive_index(pd.DatetimeIndex(spy_close.index))
    inside = (stamps >= low) & (stamps <= high)
    missing = spy_close.isna().to_numpy()[inside]
    if missing[0] or missing[-1]:
        raise ValueError("SPY has no close on the first or last day of the window")
    longest = run = 0
    for flag in missing:
        run = run + 1 if flag else 0
        longest = max(longest, run)
    if longest > MAX_SPY_NAN_RUN:
        raise ValueError(f"SPY is missing {longest} closes in a row inside the window")
    return {
        "first_day": str(ends["SPY close"][0].date()),
        "last_day": str(ends["SPY close"][1].date()),
        "spy_missing_closes": int(missing.sum()),
    }


def audit_false_deaths(
    conn: Any,
    release_spans: pd.DataFrame,
    *,
    crsp_load: int,
    window_start: str,
    window_end: str,
    crsp_end: str,
) -> dict[str, Any]:
    """Which ended release series might not have died: CRSP keeps going after the series ended.

    Metadata only: one grouped read-only query for each permno's last CRSP date (no prices or
    returns). A span is checked when it has a permno and its last date lies inside
    ``[window_start, window_end - 1 day]`` and before ``crsp_end``. It is flagged when CRSP runs
    more than 7 calendar days past that date. Only a flagged ``linked_yahoo`` span (a confirmed
    link) with no later spell of the same company number is a suspect; every other flagged row
    (spell columns with a later spell, unlinked ``yahoo`` columns) is informational and never
    blocks.

    Returns:
        ``n_checked``, ``n_suspect``, ``suspects``, ``n_informational``, ``informational``
        (counts by kind and the first 20 rows) and ``multi_symbol_permnos`` (company numbers with
        more than one linked Yahoo column).
    """
    spans = release_spans.copy()
    spans["permno"] = pd.to_numeric(spans["permno"], errors="coerce")
    spans["last_date"] = pd.to_datetime(spans["last_date"])
    low = pd.Timestamp(window_start)
    high = pd.Timestamp(window_end) - pd.Timedelta(days=1)
    stop = pd.Timestamp(crsp_end)
    checked = spans[
        spans["permno"].notna()
        & (spans["last_date"] >= low)
        & (spans["last_date"] <= high)
        & (spans["last_date"] < stop)
    ]
    permnos = sorted({int(p) for p in checked["permno"]})
    crsp_last: dict[int, pd.Timestamp] = {}
    if permnos:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT permno, max(date) FROM kaggle_crsp.daily_of(%s) "
                "WHERE permno = ANY(%s) GROUP BY permno",
                (int(crsp_load), permnos),
            )
            crsp_last = {int(p): pd.Timestamp(d) for p, d in cur.fetchall() if d is not None}
    spell = spans["spell"] if "spell" in spans.columns else pd.Series(0, index=spans.index)
    latest_spell = spell.groupby(spans["permno"]).max()
    suspects: list[dict[str, Any]] = []
    informational: list[dict[str, Any]] = []
    grace = pd.Timedelta(days=AUDIT_GRACE_DAYS)
    for row in checked.itertuples():
        seen = crsp_last.get(int(row.permno))
        if seen is None or seen <= row.last_date + grace:
            continue
        has_later = bool(
            "spell" in spans.columns and latest_spell.get(row.permno, row.spell) > row.spell
        )
        entry = {
            "symbol": row.symbol,
            "permno": int(row.permno),
            "kind": row.kind,
            "last_date": str(row.last_date.date()),
            "crsp_last_date": str(seen.date()),
            "has_later_spell": has_later,
        }
        # Only a confirmed link with no later spell can be a Yahoo series that stopped while
        # the company lives. A spell column ends because the company continues in a later
        # spell, and an unlinked Yahoo column's permno is only a rejected link candidate.
        if row.kind == SUSPECT_KIND and not has_later:
            suspects.append(entry)
        else:
            informational.append(entry)
    by_kind: dict[str, int] = {}
    for entry in informational:
        by_kind[entry["kind"]] = by_kind.get(entry["kind"], 0) + 1
    linked = spans[(spans["kind"] == SUSPECT_KIND) & spans["permno"].notna()]
    names = linked.groupby("permno")["symbol"].apply(lambda col: sorted(set(col)))
    multi = [{"permno": int(p), "symbols": syms} for p, syms in names.items() if len(syms) > 1]
    unlinked = spans[(spans["kind"] == "yahoo") & spans["permno"].notna()]
    unlinked_names = unlinked.groupby("permno")["symbol"].nunique()
    return {
        "n_checked": int(len(checked)),
        "n_suspect": len(suspects),
        "suspects": suspects,
        "n_informational": len(informational),
        "informational": {
            "label": (
                "informational only, never blocks: spell columns that end because a later spell "
                "of the same company continues, and unlinked yahoo columns whose permno is only "
                "a rejected link candidate"
            ),
            "counts_by_kind": by_kind,
            "first_rows": informational[:INFORMATIONAL_ROWS],
        },
        "multi_symbol_permnos": multi,
        "n_unlinked_candidate_permnos_with_several_symbols": int((unlinked_names > 1).sum()),
        "grace_calendar_days": AUDIT_GRACE_DAYS,
        "window": {"start": window_start, "end": window_end},
        "crsp_end": crsp_end,
    }


def _naive_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return index.tz_convert(None) if index.tz is not None else index


def _normalise_spans(span_rows: pd.DataFrame) -> pd.DataFrame:
    """Ended-series rows with every column the replay reads (an empty read has none)."""
    frame = span_rows.copy()
    if "symbol" not in frame.columns:
        return pd.DataFrame({name: pd.Series(dtype=object) for name in SPAN_ROW_COLUMNS})
    for name in ("has_successor", "total_loss"):
        if name not in frame.columns:
            frame[name] = False
        frame[name] = frame[name].fillna(False).astype(bool)
    return frame


def yearly_turnover(result: BookResult) -> dict[str, float]:
    """Traded dollars per calendar year over that year's average pre-tax equity."""
    traded: dict[int, float] = {}
    for trade in result.trades:
        if trade.reason == LIQUIDATION_REASON:
            continue
        traded[trade.date.year] = traded.get(trade.date.year, 0.0) + abs(trade.qty * trade.price)
    equity = result.pretax_equity
    mean_equity = equity.groupby(equity.index.year).mean()
    return {
        str(int(year)): float(traded.get(int(year), 0.0) / level) if level > 0 else math.nan
        for year, level in mean_equity.items()
    }


def month_end_series(curve: pd.Series) -> list[list[Any]]:
    """The last value of each calendar month as ``[date, value]`` pairs."""
    stamps = _naive_index(pd.DatetimeIndex(curve.index))
    last = pd.Series(curve.to_numpy(dtype=float), index=stamps)
    last = last.groupby(stamps.to_period("M")).last()
    dates = pd.Series(stamps, index=stamps).groupby(stamps.to_period("M")).last()
    return [[str(dates[period].date()), float(value)] for period, value in last.items()]


def _curve_stats(curve: pd.Series) -> dict[str, Any]:
    """Sharpe and largest fall of a value curve; a curve that reaches zero or below is flagged.

    The return-based statistics are not defined for a non-positive value, so such a curve
    reports no Sharpe and a fall of 100 percent instead of crashing.
    """
    values = curve.to_numpy(dtype=float)
    if len(values) >= 2 and np.isfinite(values).all() and (values > 0.0).all():
        summary = curve_summary(curve)
        return {
            "sharpe": summary["sharpe"],
            "max_drawdown_pct": summary["max_drawdown_pct"],
            "nonpositive": False,
        }
    return {"sharpe": math.nan, "max_drawdown_pct": -100.0, "nonpositive": True}


def _cell(result: BookResult) -> dict[str, Any]:
    """The reported fields of one finished book.

    ``sharpe`` and ``max_drawdown_pct`` come from ``account_value`` (marks plus cash minus tax
    still owed). The ``*_pretax_curve`` pair comes from ``pretax_equity``, which adds the taxes
    already paid back: it is not an independent no-tax run.
    """
    account = _curve_stats(result.account_value)
    pretax = _curve_stats(result.pretax_equity)
    turnover = yearly_turnover(result)
    return {
        "after_tax_final": result.after_tax_final,
        "cagr_pretax": result.cagr_pretax,
        "cagr_after_tax": result.cagr_after_tax,
        "tax_paid": result.tax_paid,
        "cost_paid": result.cost_paid,
        "unpaid_tax_left": result.unpaid_tax_left,
        "unpaid_tax_flag": result.unpaid_tax_flag,
        "n_dead_exits": result.n_dead_exits,
        "dead_exit_writeoff": result.dead_exit_writeoff,
        "skipped_buys": result.skipped_buys,
        "sharpe": account["sharpe"],
        "max_drawdown_pct": account["max_drawdown_pct"],
        "account_value_nonpositive": account["nonpositive"],
        "sharpe_pretax_curve": pretax["sharpe"],
        "max_drawdown_pct_pretax_curve": pretax["max_drawdown_pct"],
        "turnover_by_year": turnover,
        "mean_yearly_turnover": float(np.nanmean(list(turnover.values()))) if turnover else 0.0,
        "n_trades": len(result.trades),
        "chain_verified": result.chain_verified,
        "min_cash": result.min_cash,
        "account_value_month_end": month_end_series(result.account_value),
    }


def permno_double_holds(trades: list, release_spans: pd.DataFrame | None) -> dict[str, Any]:
    """Days on which one PERMNO (one company) was held in two columns at the same time.

    Args:
        trades: A book's fills, in order.
        release_spans: Release spans with ``symbol`` and ``permno``; ``None`` skips the check
            and says so (``checked`` is False).

    Returns:
        ``checked`` and the ``violations`` (first day, permno, symbols), never hidden.
    """
    if release_spans is None or not {"symbol", "permno"}.issubset(release_spans.columns):
        return {"checked": False, "violations": []}
    permno = pd.to_numeric(release_spans.drop_duplicates("symbol").set_index("symbol")["permno"])
    qty: dict[str, float] = {}
    seen: set[tuple[int, tuple[str, ...]]] = set()
    violations: list[dict[str, Any]] = []

    def check(day: pd.Timestamp) -> None:
        by_company: dict[int, list[str]] = {}
        for symbol, held in qty.items():
            company = permno.get(symbol)
            if held > HELD_QTY and company is not None and np.isfinite(company):
                by_company.setdefault(int(company), []).append(symbol)
        for company, symbols in by_company.items():
            names = tuple(sorted(symbols))
            if len(names) > 1 and (company, names) not in seen:
                seen.add((company, names))
                violations.append(
                    {"first_day": str(day.date()), "permno": company, "symbols": list(names)}
                )

    previous = None
    for trade in trades:
        if previous is not None and trade.date != previous:
            check(previous)
        previous = trade.date
        qty[trade.symbol] = qty.get(trade.symbol, 0.0) + trade.qty
    if previous is not None:
        check(previous)
    return {"checked": True, "violations": violations}


def coverage_block(
    eligible: pd.DataFrame,
    close: pd.DataFrame,
    ends: pd.Series,
    successors: frozenset[str],
    trades: list,
    start: str,
    end: str,
) -> dict[str, dict[str, Any]]:
    """Per calendar year: how many stocks were eligible, ended, and were held when they ended.

    ``successors`` are series that end with a later CRSP spell of the same company number: a data
    break (a halt or an exchange move), not a verified buy-out or merger.
    """
    days = _naive_index(eligible.index)
    inside = (days >= pd.Timestamp(start)) & (days <= pd.Timestamp(end))
    window = eligible.set_axis(days, axis=0).loc[inside].reindex(columns=close.columns)
    window = window.fillna(False).astype(bool)
    counts = window.sum(axis=1)
    month_end = counts.groupby(counts.index.to_period("M")).last()
    per_month = month_end.groupby(month_end.index.year).mean()
    ever = set(window.columns[window.to_numpy().any(axis=0)])
    stamps = _naive_index(close.index)
    years: dict[str, dict[str, Any]] = {
        str(int(y)): {
            "eligible_per_month_mean": float(per_month[y]),
            "ever_eligible_dead": 0,
            "ever_eligible_later_spell": 0,
            "held_at_death": 0,
            "held_at_later_spell": 0,
            "dollars_written_off": 0.0,
        }
        for y in per_month.index
    }
    low, high = pd.Timestamp(start), pd.Timestamp(end)
    for symbol, last in ends.items():
        when = pd.Timestamp(last)
        when = when.tz_convert(None) if when.tzinfo else when
        key = str(when.year)
        if symbol in ever and low <= when <= high and key in years:
            field = "ever_eligible_later_spell" if symbol in successors else "ever_eligible_dead"
            years[key][field] += 1
    for trade in trades:
        if trade.reason != DEAD_EXIT_REASON or str(trade.date.year) not in years:
            continue
        year = years[str(trade.date.year)]
        if trade.symbol in successors:
            year["held_at_later_spell"] += 1
            continue
        year["held_at_death"] += 1
        row = stamps.get_loc(trade.date)
        last_close = float(close.iat[row, close.columns.get_loc(trade.symbol)])
        year["dollars_written_off"] += abs(trade.qty) * (last_close - trade.price)
    return years


def spy_zero_cost_check(spy_close: pd.Series, start: str, end: str) -> dict[str, Any]:
    """SPY through the book at zero cost and no tax against SPY's own close ratio."""
    result = run_spy(spy_close, params=TaxParams(0.0, 0.0), one_way_cost=0.0, start=start, end=end)
    stamps = _naive_index(spy_close.index)
    prices = spy_close.to_numpy(dtype=float)[
        (stamps >= pd.Timestamp(start)) & (stamps <= pd.Timestamp(end))
    ]
    own = float(prices[-1] / prices[0])
    book = result.after_tax_final / START_CASH
    difference = book / own - 1.0
    return {
        "book_growth": book,
        "close_ratio": own,
        "relative_difference": difference,
        "within_tolerance": bool(abs(difference) <= SPY_TOLERANCE),
        "tolerance": SPY_TOLERANCE,
    }


def loss_monotone_check(strategy: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """After-tax final value must not rise as the loss rises (0.0 to 0.5 to 1.0)."""
    violations = []
    for cost in COSTS:
        for tax in TAXES:
            finals = [strategy[cell_key(loss, cost, tax)]["after_tax_final"] for loss in LOSSES]
            for lower_loss, higher_loss, a, b in zip(
                LOSSES[::-1], LOSSES[::-1][1:], finals[::-1], finals[::-1][1:], strict=False
            ):
                if b > a * (1.0 + MONOTONE_TOLERANCE):
                    violations.append(
                        {
                            "cost": cost,
                            "tax": tax,
                            "lower_loss": lower_loss,
                            "higher_loss": higher_loss,
                            "final_at_lower_loss": a,
                            "final_at_higher_loss": b,
                        }
                    )
    return {"violations": violations}


RATIO_BUCKETS = ("lt_0.5", "0.5_to_0.9", "0.9_to_1.1", "ge_1.1", "no_history")
DEATH_PATH_DAYS = 21  # trading days before the last close
DEATH_PATH_NOTE = (
    "Descriptive only: a flat or rising price just before the end suggests a buy-out, a "
    "collapse suggests a failure; this never feeds any result cell and chooses nothing."
)


def ratio_bucket(ratio: float | None) -> str:
    """The bucket of last close over the close 21 trading days earlier.

    Lower edges belong to the upper bucket: exactly 0.5 is ``0.5_to_0.9``, exactly 0.9 is
    ``0.9_to_1.1`` and exactly 1.1 is ``ge_1.1``. A missing or unusable ratio is ``no_history``.
    """
    if ratio is None or not math.isfinite(ratio):
        return "no_history"
    if ratio < 0.5:
        return "lt_0.5"
    if ratio < 0.9:
        return "0.5_to_0.9"
    if ratio < 1.1:
        return "0.9_to_1.1"
    return "ge_1.1"


def death_price_paths(
    trades: list, close: pd.DataFrame, successors: frozenset[str]
) -> dict[str, Any]:
    """What the price did before each dead stock the rule held (descriptive, never a cell input).

    The CRSP file has no delisting codes, so a buy-out (holders paid) cannot be told from a
    failure; this block only shows the run-up to each death.
    """
    stamps = _naive_index(close.index)
    exits: list[dict[str, Any]] = []
    buckets = {name: {"count": 0, "dollars": 0.0} for name in RATIO_BUCKETS}
    for trade in trades:
        if trade.reason != DEAD_EXIT_REASON or trade.symbol in successors:
            continue
        row = stamps.get_loc(trade.date)
        col = close.columns.get_loc(trade.symbol)
        last = float(close.iat[row, col])
        earlier = float(close.iat[row - DEATH_PATH_DAYS, col]) if row >= DEATH_PATH_DAYS else None
        usable = earlier is not None and math.isfinite(earlier) and earlier > 0.0
        ratio = last / earlier if usable else None
        dollars = abs(trade.qty) * last
        name = ratio_bucket(ratio)
        buckets[name]["count"] += 1
        buckets[name]["dollars"] += dollars
        exits.append(
            {
                "symbol": trade.symbol,
                "exit_date": str(trade.date.date()),
                "last_close": last,
                "close_21_trading_days_earlier": earlier if usable else None,
                "ratio": ratio,
                "dollars_at_last_close": dollars,
                "bucket": name,
            }
        )
    return {
        "descriptive_only": True,
        "note": DEATH_PATH_NOTE,
        "reference_cell": cell_key(*REFERENCE),
        "exits": exits,
        "buckets": buckets,
    }


def drop_file_end_rows(
    spans: pd.DataFrame, end: str, crsp_end: str | None = None
) -> tuple[pd.DataFrame, int]:
    """Remove ended-series rows that end on or after the CRSP file's last day (censored).

    A stock that is alive on the file's last day has an unknown future: its series did not die,
    it was cut off. Such a row must not be sold at the dead-stock loss; the stock is simply
    held until the window's last day like any other.

    Returns:
        The kept rows and the number dropped that fall inside the window (``last_date`` from
        the CRSP end through the window end; zero for a window that ends before the file does).
    """
    limit = pd.Timestamp(CRSP_END if crsp_end is None else crsp_end)
    last = pd.to_datetime(spans["last_date"])
    censored = last >= limit
    in_window = censored & (last <= pd.Timestamp(end))
    return spans.loc[~censored].reset_index(drop=True), int(in_window.sum())


def holdings_audit(
    trades: list,
    month_ends: list[pd.Timestamp],
    ends: pd.Series,
    successors: frozenset[str],
    kinds: dict[str, str] | None,
) -> dict[str, Any]:
    """Month-end holdings: how many, how many later die, and where their history comes from.

    A holding is a symbol with shares left after the month-end's last trade. "Later die" means
    its series ends without a later spell after that month-end (inside the window, since the
    book never holds past it). The split puts each held stock-month into Yahoo-history columns
    (release kind ``linked_yahoo`` or ``yahoo``) or CRSP-only columns (``placeholder`` or
    ``spell``); without release kinds it is left empty.
    """
    death = {str(s): _naive_stamp(t) for s, t in ends.items() if str(s) not in successors}
    qty: dict[str, float] = {}
    pos = 0
    rows = []
    for stamp in month_ends:
        while pos < len(trades) and _naive_stamp(trades[pos].date) <= stamp:
            qty[trades[pos].symbol] = qty.get(trades[pos].symbol, 0.0) + trades[pos].qty
            pos += 1
        held = sorted(sym for sym, q in qty.items() if q > HELD_QTY)
        dying = [sym for sym in held if sym in death and death[sym] > stamp]
        row: dict[str, Any] = {
            "date": str(stamp.date()),
            "n_held": len(held),
            "n_dead_holdings": len(dying),
            "n_yahoo_history": None,
            "n_crsp_only": None,
        }
        if kinds is not None:
            row["n_yahoo_history"] = sum(kinds.get(sym) in YAHOO_KINDS for sym in held)
            row["n_crsp_only"] = sum(kinds.get(sym) in CRSP_ONLY_KINDS for sym in held)
        rows.append(row)
    totals = {
        "held_stock_months": sum(r["n_held"] for r in rows),
        "dead_holding_months": sum(r["n_dead_holdings"] for r in rows),
        "yahoo_history_stock_months": None,
        "crsp_only_stock_months": None,
    }
    if kinds is not None:
        totals["yahoo_history_stock_months"] = sum(r["n_yahoo_history"] for r in rows)
        totals["crsp_only_stock_months"] = sum(r["n_crsp_only"] for r in rows)
    return {"months": rows, "totals": totals}


def _naive_stamp(stamp: Any) -> pd.Timestamp:
    value = pd.Timestamp(stamp)
    return value.tz_convert(None) if value.tzinfo is not None else value


def decomposition_for_loss(
    *,
    spy_zero: float,
    rule_loss0_zero: float,
    rule_zero: float,
    rule_cost_no_tax: float,
    rule_final: float,
    spy_final: float,
) -> dict[str, Any]:
    """Sequential steps in annual growth, in percentage points, from SPY to the rule.

    Each argument is a yearly growth rate in percent: SPY with no cost and no tax; the rule at
    loss 0.0 with none; the rule at this loss with none; the same with cost; the same with cost
    and tax (the reported cell); and SPY with the same cost and tax. The four steps sum to
    ``rule_final - spy_zero``; SPY's own cost and tax drag is a separate line, so the steps plus
    the drag equal the reported gap exactly.
    """
    steps = {
        "selection_effect_pp": rule_loss0_zero - spy_zero,
        "dead_stock_effect_pp": rule_zero - rule_loss0_zero,
        "trading_cost_effect_pp": rule_cost_no_tax - rule_zero,
        "tax_effect_pp": rule_final - rule_cost_no_tax,
    }
    total = sum(steps.values())
    spy_drag = spy_zero - spy_final
    gap = rule_final - spy_final
    return {
        **steps,
        "steps_sum_pp": total,
        "rule_minus_spy_zero_cost_zero_tax_pp": rule_final - spy_zero,
        "spy_cost_and_tax_drag_pp": spy_drag,
        "total_gap_pp": gap,
        "reconciles": bool(abs(total + spy_drag - gap) < 1e-9),
        "annual_growth_pct": {
            "spy_zero_cost_zero_tax": spy_zero,
            "rule_loss0_zero_cost_zero_tax": rule_loss0_zero,
            "rule_zero_cost_zero_tax": rule_zero,
            "rule_cost_zero_tax": rule_cost_no_tax,
            "rule_cost_and_tax": rule_final,
            "spy_cost_and_tax": spy_final,
        },
    }


def run_study(
    close: pd.DataFrame,
    spy_close: pd.Series,
    eligible: pd.DataFrame,
    span_rows: pd.DataFrame,
    *,
    start: str,
    end: str,
    release_spans: pd.DataFrame | None = None,
    start_cash: float = START_CASH,
    trade_sink: dict[str, list] | None = None,
) -> dict[str, Any]:
    """Run the fixed grid over one window and return every cell, the coverage and the checks.

    Args:
        close: Daily closes, trading days x release columns, with warm-up history.
        spy_close: SPY daily closes.
        eligible: Boolean eligibility (``universe_frames.eligibility``).
        span_rows: The ended-series rows the reader returns (``symbol``, ``last_date``,
            ``has_successor``, ``total_loss``): what the replay needs to find dead stocks.
        start: First day of the window.
        end: Last day of the window.
        release_spans: The release's span rows with ``permno`` and ``kind`` (for the
            one-company-one-column check and the Yahoo versus CRSP-only split); ``None`` skips
            those and says so.
        start_cash: Opening cash.
        trade_sink: If given, filled with every cell's fills (key: the cell key, SPY cells
            prefixed ``spy_``); the side file is written from it.

    Returns:
        ``strategy`` (18 cells), ``spy`` (6 cells), ``coverage``, ``sanity``, the holdings
        audit, the gap decomposition and the other blocks listed in the module notes.
    """
    window_check = validate_window_inputs(close, spy_close, start, end)
    spans, n_censored = drop_file_end_rows(_normalise_spans(span_rows), end)
    if pd.Timestamp(end) < pd.Timestamp(CRSP_END) and n_censored:
        raise AssertionError("a window that ends before the CRSP file ends has censored rows")
    steps: list[TargetStep] = plan_trades(  # the plan does not depend on the loss: once
        close,
        eligible,
        spans,
        top_n=TOP_N,
        lookback=LOOKBACK,
        skip=SKIP,
        dead_loss=LOSSES[0],
        start=start,
        end=end,
    )
    ends = span_end_dates(close, spans)
    flagged = spans.drop_duplicates("symbol").set_index("symbol")["has_successor"]
    successors = frozenset(s for s in ends.index if bool(flagged.get(s, False)))
    taxes = {name: TaxParams(*rates) for name, rates in TAXES.items()}
    no_tax = TaxParams(0.0, 0.0)
    prices = {loss: exit_close(close, spans, loss) for loss in LOSSES}

    def book_of(loss: float, cost: float, params: TaxParams) -> BookResult:
        return run_book(
            close,
            steps,
            ends,
            prices[loss],
            params=params,
            one_way_cost=cost,
            start=start,
            end=end,
            successors=successors,
            start_cash=start_cash,
        )

    strategy: dict[str, dict[str, Any]] = {}
    spy: dict[str, dict[str, Any]] = {}
    all_trades: dict[str, list] = {}
    for cost in COSTS:
        for tax, params in taxes.items():
            book = run_spy(
                spy_close,
                params=params,
                one_way_cost=cost,
                start=start,
                end=end,
                start_cash=start_cash,
            )
            spy[spy_key(cost, tax)] = {"one_way_cost": cost, "tax": tax, **_cell(book)}
            all_trades[f"spy_{spy_key(cost, tax)}"] = book.trades
    doubles: list[dict[str, Any]] = []
    checked = False
    for loss in LOSSES:
        for cost in COSTS:
            for tax, params in taxes.items():
                book = book_of(loss, cost, params)
                comparison = spy[spy_key(cost, tax)]["cagr_after_tax"]
                strategy[cell_key(loss, cost, tax)] = {
                    "loss": loss,
                    "one_way_cost": cost,
                    "tax": tax,
                    **_cell(book),
                    "gap_cagr_after_tax_pp": (book.cagr_after_tax - comparison) * 100.0,
                }
                all_trades[cell_key(loss, cost, tax)] = book.trades
                check = permno_double_holds(book.trades, release_spans)
                checked = checked or check["checked"]
                doubles += [
                    {"loss": loss, "cost": cost, "tax": tax, **v} for v in check["violations"]
                ]
    reference_trades = all_trades[cell_key(*REFERENCE)]
    held_at_death = sum(
        1 for t in reference_trades if t.reason == DEAD_EXIT_REASON and t.symbol not in successors
    )
    sanity = {
        "spy_zero_cost_growth": spy_zero_cost_check(spy_close, start, end),
        "loss_monotone": loss_monotone_check(strategy),
        "held_at_death": held_at_death,
        "no_dead_stock_held": held_at_death == 0,
        "one_permno_one_column": {"checked": checked, "violations": doubles},
        "reference_cell": cell_key(*REFERENCE),
    }
    paths = death_price_paths(reference_trades, close, successors)  # after the cells, read-only
    # Extra, labelled runs (not part of the 18 + 6 grid): the same plan and exit prices with
    # no cost and/or no tax, to split the gap to SPY into steps.
    spy_zero = (
        run_spy(
            spy_close, params=no_tax, one_way_cost=0.0, start=start, end=end, start_cash=start_cash
        ).cagr_after_tax
        * 100.0
    )
    zero_cost: dict[float, float] = {}
    for loss in sorted(set(LOSSES) | {0.0}):
        zero_cost[loss] = book_of(loss, 0.0, no_tax).cagr_after_tax * 100.0
    spy_final = spy[spy_key(DECOMPOSITION_COST, DECOMPOSITION_TAX)]["cagr_after_tax"] * 100.0
    decomposition = {}
    for loss in LOSSES:
        final_cell = strategy[cell_key(loss, DECOMPOSITION_COST, DECOMPOSITION_TAX)]
        part = decomposition_for_loss(
            spy_zero=spy_zero,
            rule_loss0_zero=zero_cost[0.0],
            rule_zero=zero_cost[loss],
            rule_cost_no_tax=book_of(loss, DECOMPOSITION_COST, no_tax).cagr_after_tax * 100.0,
            rule_final=final_cell["cagr_after_tax"] * 100.0,
            spy_final=spy_final,
        )
        part["matches_cell_gap"] = bool(
            abs(part["total_gap_pp"] - final_cell["gap_cagr_after_tax_pp"]) < 1e-9
        )
        decomposition[f"loss{loss:g}"] = part
    gap_decomposition = {
        "cost": DECOMPOSITION_COST,
        "tax": DECOMPOSITION_TAX,
        "note": (
            "Steps in yearly growth, percentage points: SPY with no cost and no tax, then the "
            "rule's stock picking (loss 0.0, no cost, no tax), then dead-stock losses, then "
            "trading costs, then taxes. SPY's own cost and tax drag is its own line."
        ),
        "by_loss": decomposition,
    }
    days = _window_days(close.index, start, end)
    month_ends = list(pd.Series(days, index=days).groupby(days.to_period("M")).last())
    kinds = None
    if release_spans is not None and "kind" in release_spans.columns:
        kinds = dict(zip(release_spans["symbol"], release_spans["kind"], strict=False))
    holdings = {
        cell_key(loss, DECOMPOSITION_COST, DECOMPOSITION_TAX): holdings_audit(
            all_trades[cell_key(loss, DECOMPOSITION_COST, DECOMPOSITION_TAX)],
            month_ends,
            ends,
            successors,
            kinds,
        )
        for loss in HOLDINGS_LOSSES
    }
    if trade_sink is not None:
        trade_sink.update(all_trades)
    low, high = pd.Timestamp(start), pd.Timestamp(end)
    in_window = [t for t in ends if low <= _naive_stamp(t) <= high]
    return {
        "strategy": strategy,
        "spy": spy,
        "coverage": coverage_block(eligible, close, ends, successors, reference_trades, start, end),
        "sanity": sanity,
        "window_check": window_check,
        "death_price_paths": paths,
        "gap_decomposition": gap_decomposition,
        "holdings_audit": holdings,
        "n_file_end_censored": n_censored,
        "crsp_end": CRSP_END,
        "n_rebalance_steps": len(steps),
        "n_columns": int(close.shape[1]),
        "n_ended_series_in_window": len(in_window),
        "n_ended_series_in_read": int(len(ends)),
        "skipped_buys_note": (
            "skipped_buys counts every target that could not be bought (no price that day, or "
            "its series had ended); the ledger does not split the reasons."
        ),
    }
