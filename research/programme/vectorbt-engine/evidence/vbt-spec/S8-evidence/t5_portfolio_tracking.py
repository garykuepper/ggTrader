import numpy as np, pandas as pd, vectorbt as vbt, tempfile, os
from t_common import *
o,h,l,c,v = make_data(n=800, k=8)
rets = c.pct_change().dropna()
def sk():
    from skfolio.optimization import MeanRisk, InverseVolatility, HierarchicalRiskParity
    from skfolio import RiskMeasure
    m = HierarchicalRiskParity().fit(rets); return f"HRP w sum={m.weights_.sum():.3f}"
run("skfolio HRP", sk)
def sk2():
    from skfolio.optimization import MeanRisk
    from skfolio import RiskMeasure
    m = MeanRisk(risk_measure=RiskMeasure.CVAR).fit(rets); return f"MeanRisk CVaR sum={m.weights_.sum():.3f}"
run("skfolio MeanRisk CVaR (cvxpy backend)", sk2)
def rp():
    import riskfolio as rf
    port = rf.Portfolio(returns=rets); port.assets_stats(method_mu='hist', method_cov='hist')
    w = port.optimization(model='Classic', rm='MV', obj='Sharpe', rf=0, l=0, hist=True); return f"Riskfolio sum={float(w.sum().iloc[0]):.3f}"
run("riskfolio-lib MV Sharpe", rp)
def pyp():
    from pypfopt import EfficientFrontier, risk_models, expected_returns
    mu = expected_returns.mean_historical_return(c); S = risk_models.sample_cov(c)
    ef = EfficientFrontier(mu, S); w = ef.max_sharpe(); return f"PyPortfolioOpt sum={sum(w.values()):.3f}"
run("PyPortfolioOpt max_sharpe (prices)", pyp)
def vbt_pf_weights():
    from skfolio.optimization import HierarchicalRiskParity
    w = HierarchicalRiskParity().fit(rets).weights_
    pf = vbt.Portfolio.from_orders(c, size=pd.Series(w, index=c.columns).to_frame().T.reindex(c.index).ffill().iloc[:1].reindex(c.index), size_type="targetpercent", group_by=True, cash_sharing=True, freq="D")
    return f"vbt from_orders targetpercent total_return={pf.total_return():.3f}"
run("skfolio weights -> vbt from_orders targetpercent", vbt_pf_weights)
def optuna_vbt():
    import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)
    cl = c.iloc[:,0]
    def obj(t):
        f=t.suggest_int("f",5,30); s=t.suggest_int("s",31,120)
        e = vbt.MA.run(cl,f).ma_crossed_above(vbt.MA.run(cl,s)); x = vbt.MA.run(cl,f).ma_crossed_below(vbt.MA.run(cl,s))
        return vbt.Portfolio.from_signals(cl,e,x,freq="D").sharpe_ratio()
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=1)); st.optimize(obj, n_trials=25)
    return f"optuna best={st.best_value:.3f} {st.best_params}"
run("optuna + vbt objective", optuna_vbt)
def duck():
    import duckdb
    df = rets.copy(); df.index.name="date"; df=df.reset_index()
    p = "/tmp/claude-1000/s8_test.parquet"; df.to_parquet(p)
    r = duckdb.sql(f"select count(*), avg(S0) from '{p}'").fetchall(); 
    r2 = duckdb.sql("select count(*) from df").fetchall()  # replacement scan on pandas 3 frame
    return f"parquet {r} df-scan {r2}"
run("duckdb parquet + pandas3 df scan", duck)
def duck_pg():
    import duckdb; con = duckdb.connect(); con.sql("INSTALL postgres"); return "postgres ext install ok"
run("duckdb postgres ext (timescale read)", duck_pg)
def bt_():
    import bt
    s = bt.Strategy("t", [bt.algos.RunMonthly(), bt.algos.SelectAll(), bt.algos.WeighEqually(), bt.algos.Rebalance()])
    res = bt.run(bt.Backtest(s, c)); return f"bt total_return={float(res.stats.loc['total_return'].iloc[0]):.3f}"
run("bt monthly equal-weight", bt_)
run("edgartools import", lambda: __import__('edgar').__name__)
run("fredapi import", lambda: __import__('fredapi').__name__)
run("yfinance import", lambda: __import__('yfinance').__version__)
