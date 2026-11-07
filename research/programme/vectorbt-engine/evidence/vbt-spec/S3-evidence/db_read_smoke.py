"""READ-ONLY DB smoke: CachedYFinanceLoader._db_fetch + lab.data helpers, dump frame facts.

Usage: db_read_smoke.py <outdir>   (run once per venv; compare with db_read_compare.py)
No writes: the connection is forced to a read-only session and the write helpers are
replaced by a recorder that captures what WOULD be written.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)

import psycopg2  # noqa: E402

from ggTrader.data.live import cached_yfinance_loader as cyl  # noqa: E402

_orig_connect = cyl.CachedYFinanceLoader._connect


def _ro_connect(self):
    c = _orig_connect(self)
    c.set_session(readonly=True)
    return c


cyl.CachedYFinanceLoader._connect = _ro_connect

SYMS = ["SPY", "AAPL", "MSFT", "JPM", "XOM", "MNST", "BRK-B"]
loader = cyl.CachedYFinanceLoader()
start = pd.Timestamp("2022-01-03", tz="UTC")
end = pd.Timestamp("2022-12-30", tz="UTC")
wide = loader._db_fetch(SYMS, "1d", start, end)

facts = {
    "pandas": pd.__version__,
    "psycopg2": psycopg2.__version__,
    "shape": list(wide.shape),
    "index_type": type(wide.index).__name__,
    "index_dtype": str(wide.index.dtype),
    "index_tz": str(wide.index.tz),
    "index_unit": getattr(wide.index, "unit", None),
    "index_first": str(wide.index[0]),
    "index_last": str(wide.index[-1]),
    "index_freq": str(wide.index.freq),
    "col_type": type(wide.columns).__name__,
    "col_level0_dtype": str(wide.columns.levels[0].dtype),
    "col_level1_dtype": str(wide.columns.levels[1].dtype),
    "col_level0_vals": list(map(str, wide.columns.levels[0])),
    "col_level1_vals": list(map(str, wide.columns.levels[1])),
    "dtypes": sorted({str(d) for d in wide.dtypes}),
    "nan_count": int(wide.isna().sum().sum()),
    "weekday_hist": {int(k): int(v) for k, v in wide.index.dayofweek.value_counts().sort_index().items()},
    "epoch_first_astype_int64": int(wide.index.astype("int64")[0]),
    "col_level0_is_object": bool(wide.columns.levels[0].dtype == object),
}
(out / "facts.json").write_text(json.dumps(facts, indent=1, default=str))
wide.to_parquet(out / "wide.parquet")
print(json.dumps(facts, indent=1, default=str))

# What WOULD _cache_to_db write for a yfinance-style frame? Capture, do not execute.
captured = []


class _Cur:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Conn:
    autocommit = False

    def cursor(self):
        return _Cur()

    def close(self):
        pass


def _fake_execute_values(cur, query, records, page_size=100):
    captured.extend(records)


cyl.execute_values = _fake_execute_values
cyl.CachedYFinanceLoader._connect = lambda self: _Conn()

# Build a yfinance-like frame from the DB frame: tz-aware UTC index at 00:00 (what YFinanceDataLoader emits)
for label, frame in {
    "utc_midnight": wide[SYMS[:2] and [c for c in wide.columns if c[0] in SYMS[:2]]],
}.items():
    captured.clear()
    loader._cache_to_db(frame, "1d")
    rec = captured[:2] + captured[-1:]
    facts_w = {
        "n_records": len(captured),
        "first_ts": repr(captured[0][0]),
        "ts_type": type(captured[0][0]).__name__,
        "ts_tzinfo": str(captured[0][0].tzinfo),
        "sample_tuple": [repr(x) for x in captured[0]],
        "nan_to_none_types": sorted({type(x).__name__ for r in captured for x in r[4:9]}),
        "sym_types": sorted({type(r[1]).__name__ for r in captured}),
    }
    (out / f"write_capture_{label}.json").write_text(json.dumps(facts_w, indent=1, default=str))
    print(label, json.dumps(facts_w, indent=1, default=str))

# New: sub-second/unit variants of the same index -> the written timestamps must be identical
base_ts = None
for unit in ("ns", "us", "s"):
    fr = wide[[c for c in wide.columns if c[0] == "SPY"]].copy()
    try:
        fr.index = fr.index.as_unit(unit)
    except Exception as e:  # pandas 2.x has as_unit too; fall through if not
        print("as_unit failed", unit, e)
        continue
    captured.clear()
    loader._cache_to_db(fr, "1d")
    ts = [r[0] for r in captured]
    if base_ts is None:
        base_ts = ts
    print("unit", unit, "write timestamps identical to ns:", ts == base_ts, ts[0], ts[-1])
    (out / f"write_unit_{unit}.txt").write_text(f"{ts == base_ts} {ts[0]!r} {ts[-1]!r}\n")
