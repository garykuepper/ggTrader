import numpy as np, pandas as pd, vectorbt as vbt
from t_common import *
o,h,l,c,v = make_data(n=800, k=8)
rets = c.pct_change().dropna()
from skfolio.optimization import HierarchicalRiskParity
# monthly rebalance: weights frame dated at month-end, NaN elsewhere = no order (vbt convention)
me = c.groupby(c.index.to_period("M")).tail(1).index[12:]
W = pd.DataFrame(np.nan, index=c.index, columns=c.columns)
for d in me:
    w = HierarchicalRiskParity().fit(rets.loc[:d].tail(252)).weights_
    W.loc[d] = w
pf = vbt.Portfolio.from_orders(c, size=W, size_type="targetpercent", group_by=True, cash_sharing=True, call_seq="auto", freq="D", fees=0.0005)
print("OK   skfolio rolling HRP weights -> vbt from_orders targetpercent: total_return=%.3f orders=%d" % (pf.total_return(), pf.orders.count()))
