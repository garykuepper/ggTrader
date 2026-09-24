"""Quick, non-WFO sanity checks on simple timing rules from the 2026-09-23 research batch."""
import subprocess, io, numpy as np, pandas as pd
from ggTrader.lab.fomc_calendar import historical_fomc_announcement_dates

def load(sym):
    q=f"copy (select timestamp::date d, close from ohlcv where venue='yfinance' and interval='1d' and symbol='{sym}' order by 1) to stdout csv"
    out=subprocess.run(["docker","exec","ggtrader_db","psql","-U","ggtrader","-d","ggtrader","-Atc",q],capture_output=True,text=True,check=True).stdout
    s=pd.read_csv(io.StringIO(out),names=["d","c"],parse_dates=["d"]).drop_duplicates("d").set_index("d")["c"]
    return s
px=pd.DataFrame({s:load(s) for s in ["SPY","IEF","TLT"]}).dropna()
r=px.pct_change().dropna()
COST=0.0001  # 1 bp per side

def stats(ret, pos=None):
    ret=ret.dropna()
    if pos is not None:
        trades=pos.diff().abs().fillna(0).reindex(ret.index).fillna(0)
        ret=ret-trades*COST
    ann=(1+ret).prod()**(252/len(ret))-1
    sh=ret.mean()/ret.std()*np.sqrt(252) if ret.std()>0 else np.nan
    eq=(1+ret).cumprod(); dd=(eq/eq.cummax()-1).min()
    return f"CAGR {ann*100:5.1f}%  Sharpe {sh:5.2f}  MaxDD {dd*100:6.1f}%"

idx=r.index
ym=idx.to_period("M")
pos_in_month=pd.Series(idx,index=idx).groupby(ym).cumcount()
n_in_month=pd.Series(idx,index=idx).groupby(ym).transform("count")
from_end=n_in_month-pos_in_month  # 1 = last trading day
fomc=pd.DatetimeIndex(historical_fomc_announcement_dates(2011))
# day -1 return = return on the trading day before the announcement day
day_before=set()
for f in fomc:
    loc=idx.searchsorted(f)
    if loc<len(idx) and idx[loc]==f and loc>=1: day_before.add(idx[loc-1])
pre_fomc=pd.Series(idx.isin(list(day_before)),index=idx)

for name,(a,b) in {"PINNED 2021-02-01..2026-04-30":("2021-02-01","2026-04-30"),"LONG 2011-03-01..2026-04-30":("2011-03-01","2026-04-30")}.items():
    w=slice(a,b); print(f"\n=== {name} (daily returns, cash earns 0, 1bp/side cost on switches) ===")
    for s in ["SPY","IEF","TLT"]: print(f"  buy&hold {s:4s}            ", stats(r[s][w]))
    for s in ["IEF","TLT"]:
        p=(from_end<=3).astype(float); print(f"  month-end last3 {s}      ", stats((r[s]*p)[w],p[w]), f" exposure {p[w].mean()*100:.0f}%")
        p=pre_fomc.astype(float); print(f"  pre-FOMC day-1 {s}       ", stats((r[s]*p)[w],p[w]), f" exposure {p[w].mean()*100:.1f}%  events {int(p[w].sum())}")
    # turn of month SPY: last 1 + first 3 days (Lakonishok-Smidt) and last 4 + first 4 (8-day)
    for lbl,(L,F) in {"ToM [-1,+3]":(1,3),"ToM [-4,+4]":(4,4)}.items():
        p=((from_end<=L)|(pos_in_month<F)).astype(float); print(f"  {lbl} SPY else cash     ", stats((r['SPY']*p)[w],p[w]), f" exposure {p[w].mean()*100:.0f}%")
        # mean daily return in vs out
        print(f"      SPY mean bp/day in-window {r['SPY'][w][p[w]==1].mean()*1e4:5.1f}  out {r['SPY'][w][p[w]==0].mean()*1e4:5.1f}")
    # rebalancing tilt: last 4 days hold IEF if SPY beat IEF month-to-date (through day -5), else SPY; SPY otherwise
    mtd=(1+r).groupby(ym).cumprod()-1
    sig=(mtd["SPY"]-mtd["IEF"]).where(from_end==5).groupby(ym).transform("max")
    hold_ief=((from_end<=4)&(sig>0))
    ret=r["SPY"].where(~hold_ief,r["IEF"]); p=hold_ief.astype(float)
    print("  rebal tilt SPY->IEF      ", stats(ret[w],p[w]))
    # vol-managed SPY: w=min(1, c/var21), c = expanding median of var21 (no lookahead), lagged 1 day
    var=r["SPY"].rolling(21).var()*252; c=var.expanding(252).median()
    wv=(c/var).clip(upper=1).shift(1)
    print("  vol-managed SPY (1/var)  ", stats((r["SPY"]*wv)[w],wv[w]), f" avg exposure {wv[w].mean()*100:.0f}%")
    wv2=(np.sqrt(c)/np.sqrt(var)).clip(upper=1).shift(1)
    print("  vol-managed SPY (1/vol)  ", stats((r["SPY"]*wv2)[w],wv2[w]), f" avg exposure {wv2[w].mean()*100:.0f}%")
