"""Independent cross-check: 25-line numpy ledger vs vbt from_signals (single symbol, all-in, fees, next-bar? same-bar close fills)."""
import numpy as np, pandas as pd, vectorbt as vbt, quantstats as qs, empyrical as ep, ffn
from t_common import make_data
o,h,l,c,v = make_data(n=1000, k=1); px = c.iloc[:,0]
fast, slow = px.rolling(10).mean(), px.rolling(40).mean()
ent, ext = (fast>slow)&(fast.shift()<=slow.shift()), (fast<slow)&(fast.shift()>=slow.shift())
fee=0.001; cash=100000.0; sh=0.0; eq=[]
for p,e,x in zip(px.values, ent.values, ext.values):
    if e and sh==0: sh=cash/(p*(1+fee)); cash=0.0
    elif x and sh>0: cash=sh*p*(1-fee); sh=0.0
    eq.append(cash+sh*p)
eq=pd.Series(eq,index=px.index)
pf=vbt.Portfolio.from_signals(px, ent, ext, init_cash=100000.0, fees=fee, freq="D", accumulate=False)
print("OK   numpy-ledger (fee convention: shares=cash/(p*(1+fee))) vs vbt from_signals max |equity diff| = %.3e" % float((eq-pf.value()).abs().max()))
r=pf.returns()
a=r.vbt.returns(freq="D",year_freq="252 days").sharpe_ratio(); b=ep.sharpe_ratio(r); q=qs.stats.sharpe(r); f_=ffn.calc_stats((1+r).cumprod()*100).stats["daily_sharpe"]
d=r.vbt.returns(freq="D").sharpe_ratio()
print("OK   Sharpe: vbt(252d)=%.6f empyrical=%.6f quantstats=%.6f ffn=%.6f | vbt default(365d)=%.6f" % (a,b,q,float(f_),d))
# pf.stats() Sharpe vs returns accessor
print("INFO pf.sharpe_ratio() default =", round(float(pf.sharpe_ratio()),6))
