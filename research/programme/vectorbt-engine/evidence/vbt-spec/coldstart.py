import time, sys, gc, os
t0=time.time()
import numpy as np, pandas as pd
import vectorbt as vbt
t1=time.time()
import numba
print("versions", vbt.__version__, pd.__version__, np.__version__, numba.__version__, "import_s=%.2f"%(t1-t0))
from vectorbt import _engine
print("rust_available", _engine.is_rust_available())
eng = sys.argv[1]
vbt.settings["engine"]=eng
rng=np.random.default_rng(0)
idx=pd.bdate_range("2025-01-01",periods=252)
close=pd.DataFrame(100*np.exp(np.cumsum(rng.normal(0,0.01,(252,500)),0)),index=idx,columns=[f"S{i}" for i in range(500)])
names=set()
def ndisp():
    n=0
    for o in gc.get_objects():
        try:
            if type(o).__name__=="CPUDispatcher" and len(o.signatures)>0: n+=1; names.add(getattr(o,"__name__","?"))
        except Exception: pass
    return n
def step(name,fn):
    t=time.time(); r=fn(); dt=time.time()-t
    print("%-28s %7.2fs  compiled_dispatchers=%d"%(name,dt,ndisp())); return r
rsi=step("RSI ewm w=27", lambda: vbt.RSI.run(close,window=27,ewm=True).rsi)
bb=step("BBANDS w20 ddof1", lambda: vbt.BBANDS.run(close,window=20,alpha=2.0,ddof=1))
ma=step("MA ewm 20", lambda: vbt.MA.run(close,window=20,ewm=True).ma)
mc=step("MACD ewm", lambda: vbt.MACD.run(close,fast_window=12,slow_window=26,signal_window=9,macd_ewm=True,signal_ewm=True))
lo=pd.DataFrame(bb.lower.values,index=close.index,columns=close.columns); mi=pd.DataFrame(bb.middle.values,index=close.index,columns=close.columns); ent=close<lo; ex=close>mi
pf=step("from_signals tiny", lambda: vbt.Portfolio.from_signals(close,ent,ex,size=0.03,size_type="percent",init_cash=1e5,cash_sharing=True,group_by=True))
step("pf.value()", lambda: pf.value())
step("pf.trades.count()", lambda: pf.trades.count())
step("signals accessor fshift/AND", lambda: ent.vbt.signals.AND(ex))
step("returns sharpe", lambda: pf.returns().vbt.returns(freq="D",year_freq="252 days").sharpe_ratio())
print(sorted(names)); print("total %.2f"%(time.time()-t0))
