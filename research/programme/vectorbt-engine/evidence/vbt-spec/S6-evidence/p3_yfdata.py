"""Evaluate vbt.YFData: tz, mixed ETF/stock, delisted, speed. Network read only; nothing persisted."""
import time, warnings, sys
import pandas as pd, vectorbt as vbt
warnings.filterwarnings("ignore")
def tryit(label, f):
    t=time.time()
    try:
        r=f(); print(f"[{label}] OK {round(time.time()-t,1)}s", r, flush=True)
    except Exception as e:
        print(f"[{label}] {type(e).__name__}: {str(e)[:160]} ({round(time.time()-t,1)}s)", flush=True)
def one(sym, **kw):
    d = vbt.YFData.download(sym, start="2024-01-02", end="2024-02-02", **kw)
    x = d.get()
    return dict(cols=list(x.columns), tz=str(x.index.tz), first=str(x.index[0]), last=str(x.index[-1]), n=len(x))
tryit("SPY (ETF)", lambda: one("SPY"))
tryit("AAPL", lambda: one("AAPL"))
tryit("[SPY,AAPL] default", lambda: vbt.YFData.download(["SPY","AAPL"], start="2024-01-02", end="2024-02-02").get().shape)
tryit("[SPY,AAPL] missing_columns=drop", lambda: vbt.YFData.download(["SPY","AAPL"], start="2024-01-02", end="2024-02-02", missing_columns="drop").get("Close").shape)
tryit("delisted SIVB", lambda: one("SIVB"))
tryit("renamed ANTM", lambda: one("ANTM"))
tryit("[AAPL,SIVB,MSFT] mixed delisted", lambda: vbt.YFData.download(["AAPL","SIVB","MSFT"], start="2024-01-02", end="2024-02-02").symbols)
# start/end handling: asks for [2024-01-02, 2024-01-05] -> are both end days included? local-tz mktime issue
d = vbt.YFData.download("AAPL", start="2024-01-02", end="2024-01-05")
print("start=2024-01-02 end=2024-01-05 ->", [str(i) for i in d.get().index], flush=True)
# auto_adjust default / Adj close
x = vbt.YFData.download("AAPL", start="2024-01-02", end="2024-01-05").get()
print("AAPL close 2024-01-02 raw from YFData:", float(x['Close'].iloc[0]), flush=True)
# speed: 12 symbols serial
syms = ["AAPL","MSFT","JPM","XOM","KO","PG","JNJ","WMT","CVX","HD","MRK","PEP"]
t=time.time(); dd = vbt.YFData.download(syms, start="2019-03-01", missing_columns="drop"); 
print("12 symbols x full history serial:", round(time.time()-t,1), "s", dd.get("Close").shape, "tz", dd.get("Close").index.tz, "first idx", dd.get("Close").index[0], flush=True)
