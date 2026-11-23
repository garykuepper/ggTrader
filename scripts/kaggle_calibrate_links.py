"""Calibrate return-based links against dated McGill ticker observations."""

import argparse
import itertools
import json
import os
import random
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ggTrader.data.bars import connect_readonly
from ggTrader.data.kaggle_import.links import LinkSettings, find_links, pairwise_correlation
from ggTrader.data.kaggle_import.spells import calendar_from_file

KNOWN = (
    "AAPL",
    "MSFT",
    "IBM",
    "XOM",
    "META",
    "FB",
    "GOOG",
    "GOOGL",
    "PX",
    "LIN",
    "GM",
    "VIAC",
    "PARA",
)


def truth_from_rows(rows: pd.DataFrame) -> tuple[dict, dict, dict]:
    """Build unambiguous ticker truth and both directions of observed identities."""
    rows = rows.dropna(subset=["date", "permno", "stock_ticker"]).copy()
    rows["date"] = pd.to_datetime(rows["date"].astype(str), format="%Y%m%d", errors="raise")
    rows["permno"] = rows["permno"].astype(int)
    ticker = rows.groupby("stock_ticker")["permno"].agg(lambda x: sorted(set(x))).to_dict()
    reverse = rows.groupby("permno")["stock_ticker"].agg(lambda x: sorted(set(x))).to_dict()
    return {k: v[0] for k, v in ticker.items() if len(v) == 1}, reverse, ticker


def split_truth(symbols: list[str]) -> tuple[list[str], list[str]]:
    """Split sorted symbols reproducibly without overlap."""
    symbols = sorted(symbols)
    random.Random(20261122).shuffle(symbols)
    n = len(symbols) * 80 // 100
    return symbols[:n], symbols[n:]


def score(frame: pd.DataFrame, truth: dict, symbols: list[str]) -> dict:
    """Score links and unresolved statuses for a specified independent set."""
    subset = frame.set_index("symbol").loc[symbols]
    linked = subset[subset.status == "linked"]
    correct = sum(int(row.permno == truth[symbol]) for symbol, row in linked.iterrows())
    return {
        "correct": correct,
        "false": len(linked) - correct,
        **{
            status: int((subset.status == status).sum())
            for status in ("unlinked", "ambiguous", "duplicate")
        },
    }


def choose(scores: list[dict]) -> dict:
    """Choose using calibration scores alone, preferring fewer errors and stricter ties."""
    return min(scores, key=lambda s: (s["false"], -s["correct"], -s["agreement_bar"], -s["margin"]))


def yahoo_data(url: str, start: pd.Timestamp, end: pd.Timestamp) -> tuple:
    """Read Yahoo prices and whole-history lifetimes in a read-only session."""
    conn = connect_readonly(url)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT symbol, timestamp, close FROM ohlcv WHERE venue = 'yfinance' "
                "AND interval = '1d' AND timestamp >= %s AND timestamp < %s",
                (start - pd.Timedelta(days=30), end + pd.Timedelta(days=1)),
            )
            rows = cur.fetchall()
            cur.execute(
                "SELECT symbol, min(timestamp)::date, max(timestamp)::date FROM ohlcv "
                "WHERE venue = 'yfinance' AND interval = '1d' GROUP BY symbol"
            )
            lives = cur.fetchall()
    finally:
        conn.close()
    prices = pd.DataFrame(rows, columns=["symbol", "date", "close"])
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.pivot(index="date", columns="symbol", values="close").sort_index()
    sessions = prices.index[prices["SPY"].notna()]
    prices = prices.reindex(sessions)
    returns = prices.pct_change(fill_method=None).loc[start:end]
    life = pd.DataFrame(lives, columns=["symbol", "first", "last"]).set_index("symbol")
    return returns, life["first"], life["last"]


def crsp_data(path: Path, sessions: pd.DatetimeIndex, sample: int | None) -> tuple:
    """Stream twice: count observations and lifetimes, then fill the return matrix."""
    parquet = pq.ParquetFile(path)
    counts, first, last = {}, {}, {}
    for i in range(parquet.num_row_groups):
        rows = (
            parquet.read_row_group(i, columns=["permno", "date", "ret"])
            .to_pandas()
            .dropna(subset=["permno", "date"])
        )
        rows["date"] = pd.to_datetime(rows["date"])
        for permno, group in rows.groupby("permno"):
            permno = int(permno)
            first[permno] = min(first.get(permno, group.date.min()), group.date.min())
            last[permno] = max(last.get(permno, group.date.max()), group.date.max())
            counts[permno] = counts.get(permno, 0) + int(
                (group.date.isin(sessions) & np.isfinite(group.ret)).sum()
            )
    ids = sorted(p for p, count in counts.items() if count >= 250)
    if sample is not None:
        ids = ids[:sample]
    matrix = np.full((len(sessions), len(ids)), np.nan, dtype=np.float32)
    positions = pd.Index(ids)
    for i in range(parquet.num_row_groups):
        rows = (
            parquet.read_row_group(i, columns=["permno", "date", "ret"])
            .to_pandas()
            .dropna(subset=["permno", "date"])
        )
        dates = sessions.get_indexer(pd.to_datetime(rows.date))
        columns = positions.get_indexer(rows.permno)
        mask = (dates >= 0) & (columns >= 0)
        matrix[dates[mask], columns[mask]] = rows.ret.to_numpy()[mask]
    return pd.DataFrame(matrix, index=sessions, columns=ids), pd.Series(first), pd.Series(last)


def run(args: argparse.Namespace) -> dict:
    """Load data, calibrate on one partition, then evaluate the held-out partition."""
    started = time.monotonic()
    start, end = pd.Timestamp(args.window_start), pd.Timestamp(args.window_end)
    y, yf, yl = yahoo_data(os.environ[args.yahoo_url_env], start, end)
    c, cf, cl = crsp_data(args.crsp_file, y.index, args.sample_permnos)
    calendar = calendar_from_file(args.crsp_file)
    truth, reverse, tickers = truth_from_rows(
        pd.read_csv(args.mcgill_file, usecols=["date", "permno", "stock_ticker"])
    )
    calibration, held_out = split_truth(list(set(y.columns) & truth.keys()))
    pairwise = pairwise_correlation(y.to_numpy(), c.to_numpy(), 40)
    scores, results = [], {}
    for bar, margin in itertools.product((0.99, 0.995, 0.998, 0.999), (0.0005, 0.001, 0.003, 0.01)):
        settings = LinkSettings(250, 40, bar, margin, start.date(), end.date(), 60)
        result = find_links(y, c, yf, cf, yl, cl, settings, calendar, pairwise=pairwise)
        scores.append({"agreement_bar": bar, "margin": margin, **score(result, truth, calibration)})
        results[(bar, margin)] = result
    chosen = choose(scores)
    result = results[chosen["agreement_bar"], chosen["margin"]]
    held_score = score(result, truth, held_out)
    indexed = result.set_index("symbol")
    known = [
        {
            "symbol": s,
            "yahoo_has_symbol": s in y.columns,
            "mcgill_permnos": tickers.get(s, []),
            "link": indexed.loc[s].to_dict() if s in indexed.index else None,
        }
        for s in KNOWN
    ]
    disagreements = []
    for row in result.itertuples(index=False):
        if row.status == "linked" and row.symbol in truth and row.permno != truth[row.symbol]:
            disagreements.append(
                {
                    "symbol": row.symbol,
                    "linked_permno": row.permno,
                    "mcgill_permno": truth[row.symbol],
                    "corr": row.corr,
                    "corr_first_half": row.corr_first_half,
                    "corr_second_half": row.corr_second_half,
                    "second_best_corr": row.second_best_corr,
                }
            )
    return {
        "grid_scores": scores,
        "chosen_settings": chosen,
        "zero_false_match_available": chosen["false"] == 0,
        "held_out": held_score,
        "known_matches": known,
        "disagreements": disagreements,
        "status_counts": result.status.value_counts().to_dict(),
        "mcgill_permno_to_tickers": reverse,
        "runtime_seconds": time.monotonic() - started,
        "peak_memory_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }


def json_safe(value):
    """Convert numerical scalar values and missing correlations to JSON values."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and write calibration results without exposing credentials."""
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("crsp-file", "mcgill-file", "out"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--window-start", default="2005-01-01")
    parser.add_argument("--window-end", default="2024-12-31")
    parser.add_argument("--yahoo-url-env", default="GGT_YAHOO_URL")
    parser.add_argument("--sample-permnos", type=int)
    args = parser.parse_args(argv)
    try:
        output = run(args)
        args.out.write_text(json.dumps(json_safe(output), indent=2, allow_nan=False) + "\n")
    except Exception:
        print("Calibration failed; check inputs and database access.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
