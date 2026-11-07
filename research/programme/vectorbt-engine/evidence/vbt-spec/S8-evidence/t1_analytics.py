import matplotlib; matplotlib.use("Agg")
import numpy as np, pandas as pd, vectorbt as vbt
from t_common import *
o,h,l,c,v = make_data()
spy = c.iloc[:,0].pct_change().dropna(); spy.name="SPY"
pf = vbt.Portfolio.from_signals(c, c.vbt.rsi.__class__ and (c.pct_change()< -0.01), (c.pct_change()>0.01), size=0.05, size_type="percent", group_by=True, cash_sharing=True, freq="D") if False else None
fast=c.rolling(10).mean(); slow=c.rolling(50).mean()
pf = vbt.Portfolio.from_signals(c, fast>slow, fast<slow, size=0.08, size_type="percent", group_by=True, cash_sharing=True, freq="D", init_cash=100000)
ret = pf.returns(); ret.name="strat"
print("vbt", vbt.__version__, "ret type", type(ret), ret.index.tz, len(ret))
import quantstats as qs
run("quantstats.stats.sharpe", lambda: qs.stats.sharpe(ret))
run("quantstats.stats.max_drawdown", lambda: qs.stats.max_drawdown(ret))
run("quantstats.stats.cagr", lambda: qs.stats.cagr(ret))
run("quantstats.reports.metrics(benchmark)", lambda: qs.reports.metrics(ret, benchmark=spy, display=False, mode="basic").shape)
run("quantstats.reports.html", lambda: (qs.reports.html(ret, benchmark=spy, output="/tmp/claude-1000/qs_test.html", title="t"), "written")[1])
import empyrical as ep
run("empyrical.sharpe_ratio", lambda: ep.sharpe_ratio(ret))
run("empyrical.max_drawdown", lambda: ep.max_drawdown(ret))
run("empyrical.alpha_beta", lambda: ep.alpha_beta(ret, spy))
run("empyrical.annual_return", lambda: ep.annual_return(ret))
import pyfolio as pf_
run("pyfolio.timeseries.perf_stats", lambda: pf_.timeseries.perf_stats(ret).round(3).to_dict())
run("pyfolio.create_returns_tear_sheet", lambda: (pf_.create_returns_tear_sheet(ret, benchmark_rets=spy, return_fig=True), "fig")[1])
import ffn
px = (1+ret).cumprod()*100
run("ffn.calc_stats", lambda: round(float(ffn.calc_stats(px).stats["daily_sharpe"]),3))
run("ffn.to_drawdown_series", lambda: float(ffn.to_drawdown_series(px).min()))
run("vbt returns accessor sharpe", lambda: ret.vbt.returns(freq="D", year_freq="252 days").sharpe_ratio())
run("vbt pf.stats", lambda: len(pf.stats()))
run("vbt returns qs_adapter?", lambda: hasattr(ret.vbt.returns(freq='D'), 'qs'))
