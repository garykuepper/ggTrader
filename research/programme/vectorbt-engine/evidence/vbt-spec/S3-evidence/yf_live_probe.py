"""Tiny network probe (no DB): YFinanceDataLoader output frame facts for 2 symbols, 8 days."""
import json
import sys
import warnings

import pandas as pd
import yfinance

warnings.simplefilter("ignore")
from ggTrader.data.live.yfinance_loader import YFinanceDataLoader  # noqa: E402

start = pd.Timestamp("2026-10-20", tz="UTC")
end = pd.Timestamp("2026-10-30", tz="UTC")
raw = yfinance.download(["SPY", "AAPL"], start="2026-10-20", end="2026-10-30", progress=False, auto_adjust=True)
df = YFinanceDataLoader().fetch_ohlcv(["SPY", "AAPL"], "1d", start_date=start, end_date=end)
facts = {
    "pandas": pd.__version__,
    "yfinance": yfinance.__version__,
    "raw_index_dtype": str(raw.index.dtype),
    "raw_columns_names": list(raw.columns.names),
    "raw_col_level0": list(map(str, raw.columns.levels[0])),
    "loader_shape": list(df.shape),
    "loader_index_dtype": str(df.index.dtype),
    "loader_index_first": str(df.index[0]) if len(df) else None,
    "loader_index_last": str(df.index[-1]) if len(df) else None,
    "loader_col_names": list(df.columns.names),
    "loader_col_level0_dtype": str(df.columns.levels[0].dtype),
    "loader_dtypes": sorted({str(d) for d in df.dtypes}),
    "loader_col_sample": [tuple(c) for c in df.columns[:6]],
    "SPY_close_first": float(df["SPY"]["close"].iloc[0]) if len(df) else None,
    "SPY_close_last": float(df["SPY"]["close"].iloc[-1]) if len(df) else None,
}
print(json.dumps(facts, indent=1, default=str))
json.dump(facts, open(sys.argv[1], "w"), indent=1, default=str)
