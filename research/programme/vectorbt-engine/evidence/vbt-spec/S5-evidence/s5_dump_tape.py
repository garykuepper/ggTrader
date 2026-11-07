"""S5 prototype step 1: dump a read-only snapshot of the cached SP500 tape to parquet.

Runs in the ggTrader project venv. The DB session is forced read-only through
PGOPTIONS (set by the caller), so any loader write attempt fails and is logged.
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2026-10-30")
    a = ap.parse_args()
    assert "default_transaction_read_only=on" in os.environ.get("PGOPTIONS", ""), "need RO"

    import psycopg2

    from ggTrader.lab.data import equity_universe_between, load_ohlcv
    from ggTrader.utils.config import get_db_connection_string

    uni = equity_universe_between(
        pd.Timestamp("2021-01-31", tz="UTC"), pd.Timestamp(a.end, tz="UTC"), universe="sp500"
    )
    dsn = get_db_connection_string().replace("postgresql+psycopg2://", "postgresql://")
    with psycopg2.connect(dsn) as c, c.cursor() as cur:
        cur.execute(
            "select symbol, count(*) from ohlcv where timestamp >= %s and timestamp <= %s "
            "group by symbol",
            (a.start, a.end),
        )
        counts = dict(cur.fetchall())
    # Deterministic sample: every k-th PIT-window member that is in the cache.
    # Includes short-history names on purpose (warm-up edge cases).
    cand = [u for u in uni if counts.get(u, 0) >= 120 and u != "SPY"]
    step = max(1, len(cand) // a.n)
    syms = cand[::step][: a.n]
    print(f"universe {len(uni)} cached {len(cand)} sampled {len(syms)}", flush=True)
    ohlcv = load_ohlcv(syms, a.start, a.end, use_negative_cache=True)
    have = sorted(ohlcv.columns.get_level_values(0).unique())
    close = pd.concat({s: ohlcv[s]["close"] for s in have}, axis=1).astype("float64")
    volume = pd.concat({s: ohlcv[s]["volume"] for s in have}, axis=1).astype("float64")
    close.index = close.index.tz_convert(None)
    volume.index = volume.index.tz_convert(None)
    close.to_parquet(os.path.join(a.out, "tape_close.parquet"))
    volume.to_parquet(os.path.join(a.out, "tape_volume.parquet"))
    h = hashlib.sha256(np.ascontiguousarray(close.fillna(-1).to_numpy()).tobytes()).hexdigest()
    meta = {
        "start": a.start,
        "end": a.end,
        "n_symbols": len(have),
        "n_bars": len(close),
        "first_bar": str(close.index[0]),
        "last_bar": str(close.index[-1]),
        "nan_frac": float(close.isna().to_numpy().mean()),
        "short_history_symbols": int((close.notna().sum() < 500).sum()),
        "interior_nan_cells": int(
            (close.isna() & close.ffill().notna() & close.bfill().notna()).to_numpy().sum()
        ),
        "close_sha256": h,
        "symbols": have,
        "pandas": pd.__version__,
        "python": sys.version.split()[0],
    }
    with open(os.path.join(a.out, "tape_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print({k: v for k, v in meta.items() if k != "symbols"})


if __name__ == "__main__":
    main()
