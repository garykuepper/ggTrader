"""Load full SP500-union tape via TimescaleData; time Data ops; run vbt indicators; save parquet."""
import sys, time, resource, warnings
sys.path.insert(0, '.')
import numpy as np, pandas as pd, vectorbt as vbt
from ggTrader.lab.data import equity_universe_between
from tsdata import TimescaleData, ReadOnlyLoader

START = "2019-03-01"
uni = equity_universe_between(pd.Timestamp(START, tz="UTC"), pd.Timestamp("2026-11-05", tz="UTC"))
t = time.time()
d = TimescaleData.download(uni + ["SPY"], start=START)
print("download->Data", round(time.time()-t,1), "s; symbols", len(d.symbols), "missing", len(d.missing_symbols), flush=True)
print("missing_symbols (no rows in cache):", d.missing_symbols)
print("index", d.wrapper.index.tz, d.wrapper.index[0], d.wrapper.index[-1], len(d.wrapper.index), "dups", d.wrapper.index.duplicated().sum())
print("RSS MB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024, flush=True)

t = time.time(); close = d.get("Close"); print("get('Close') [vectorised concat override]", round(time.time()-t,2), "s", close.shape)
t = time.time(); hi, lo = d.get(["High", "Low"]); print("get(['High','Low'])", round(time.time()-t,2), "s")
# stock Data.concat for comparison (100 symbols; base class implementation called explicitly)
syms100 = d.symbols[:100]
d100 = vbt.Data.from_data({s: d.data[s] for s in syms100}, missing_index="nan", missing_columns="nan", download_kwargs={})
t0 = time.time(); out = vbt.Data.concat.func(d100) if hasattr(vbt.Data.concat, "func") else vbt.Data.concat(d100)
print("stock vbt.Data.concat, 100 symbols:", round(time.time()-t0,2), "s (x6.5 for 646 symbols ~", round((time.time()-t0)*6.46,1), "s)")
same = np.allclose(out["Close"].values, d.get("Close")[syms100].values, equal_nan=True)
print("stock concat == override on Close:", same)

# indicators over the Data
t = time.time(); rsi = vbt.RSI.run(close, window=14); print("vbt.RSI.run(close,14)", round(time.time()-t,2), "s", rsi.rsi.shape)
t = time.time(); ma = vbt.MA.run(close, window=[20, 50]); print("vbt.MA.run(close,[20,50])", round(time.time()-t,2), "s", ma.ma.shape)
t = time.time(); atr = vbt.ATR.run(hi, lo, close, window=14); print("vbt.ATR.run", round(time.time()-t,2), "s")
# pandas parity of MA(20) for one symbol
a = close["AAPL"]
print("MA20 diff vs pandas rolling", float((ma.ma[(20, "AAPL")] - a.rolling(20).mean()).abs().max()))
print("RSI AAPL tail", float(rsi.rsi[(14, "AAPL")].iloc[-1]))
# wrapper slicing: Data indexing by date, and by symbol
sl = d.loc["2024-01-01":"2024-12-31"] if hasattr(d, "loc") else None
print("Data.loc slice:", sl.wrapper.index[0], sl.wrapper.index[-1], len(sl.symbols))
close.to_parquet("tape_close.parquet"); 
for f in ("Open","High","Low","Volume"):
    d.get(f).to_parquet(f"tape_{f.lower()}.parquet")
print("saved parquet; RSS MB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
print("Data has .run?", hasattr(d, "run"), "| has .close?", hasattr(d, "close"), "| has .features?", hasattr(d, "features"), "| has fetch_symbol/pull?", hasattr(d, "fetch_symbol"), hasattr(d, "pull"))
print("Data public attrs:", [a for a in dir(vbt.Data) if not a.startswith("_")][:60])
# indexing by symbol (columns of wrapper = single-col?), Data.__getitem__?
try:
    print("d['AAPL']:", type(d["AAPL"]))
except Exception as e:
    print("d['AAPL'] ->", type(e).__name__, str(e)[:80])
print("d.data['AAPL'] cols:", list(d.data["AAPL"].columns), d.data["AAPL"].dtypes.iloc[0])
