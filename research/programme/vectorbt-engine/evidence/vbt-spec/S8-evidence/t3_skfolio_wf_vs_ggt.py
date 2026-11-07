"""Can skfolio.WalkForward reproduce ggTrader's generate_folds (12m train / 3m test / 3m step, eval 2021-01-31..2026-04-30)?"""
import pandas as pd, numpy as np
from skfolio.model_selection import WalkForward, CombinatorialPurgedCV
def ggt_folds(es, ee, tm=12, te=3):
    out=[]; cur=es
    while True:
        tr_end = cur+pd.DateOffset(months=tm); te_end=tr_end+pd.DateOffset(months=te)
        if te_end>ee: break
        out.append((cur,tr_end,te_end)); cur+=pd.DateOffset(months=te)
    return out
es,ee=pd.Timestamp("2021-01-31"),pd.Timestamp("2026-04-30")
g=ggt_folds(es,ee); print("ggTrader folds:",len(g), g[0], g[-1])
idx=pd.bdate_range("2019-01-01","2026-04-30"); X=pd.DataFrame(np.random.default_rng(0).normal(0,.01,(len(idx),3)),index=idx)
Xe=X.loc[es-pd.DateOffset(months=0):ee]
# calendar test: test_size=3 'MS'-periods stepping by test? try freq=3MS with test_size=1, train_size=DateOffset(months=12)
for kw in [dict(test_size=1,train_size=pd.DateOffset(months=12),freq=pd.offsets.MonthBegin(3)),
           dict(test_size=1,train_size=pd.DateOffset(months=12),freq="QS")]:
    cv=WalkForward(**kw); sp=list(cv.split(Xe))
    print(kw["freq"], "folds",len(sp))
    for tr,te in sp[:2]+sp[-1:]:
        print("  train",Xe.index[tr[0]].date(),Xe.index[tr[-1]].date(),"test",Xe.index[te[0]].date(),Xe.index[te[-1]].date())
# purge demo: purged_size=5 => drops last 5 train obs
cv=WalkForward(test_size=1,train_size=pd.DateOffset(months=12),freq="QS",purged_size=5)
tr,te=list(cv.split(Xe))[0]; print("purged_size=5 train_end",Xe.index[tr[-1]].date(),"test_start",Xe.index[te[0]].date())
cv=WalkForward(test_size=1,train_size=pd.DateOffset(months=12),freq="QS",purged_size=5,expand_train=True)
sp=list(cv.split(Xe)); print("expand_train folds",len(sp), len(sp[0][0]), len(sp[-1][0]))
