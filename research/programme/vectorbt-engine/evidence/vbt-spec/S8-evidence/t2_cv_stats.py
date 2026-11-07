import numpy as np, pandas as pd
from t_common import *
o,h,l,c,v = make_data(n=1000)
X = c.pct_change().fillna(0); idx = X.index
run("sklearn TimeSeriesSplit(gap)", lambda: [(len(a),len(b)) for a,b in __import__('sklearn.model_selection',fromlist=['x']).TimeSeriesSplit(5,gap=5,test_size=63).split(X)])
def sk():
    from skfolio.model_selection import CombinatorialPurgedCV, WalkForward
    cv = CombinatorialPurgedCV(n_folds=8, n_test_folds=2, purged_size=5, embargo_size=5)
    n = cv.get_n_splits(X); sp = list(cv.split(X))
    return f"CPCV n_splits={n} paths={cv.n_test_paths} first_train={len(sp[0][0])} test_parts={len(sp[0][1])}"
run("skfolio CombinatorialPurgedCV", sk)
def skwf():
    from skfolio.model_selection import WalkForward
    cv = WalkForward(test_size=1, train_size=12, freq="MS", purged_size=0)  # calendar-month freq
    sp = list(cv.split(X)); return f"WalkForward freq=MS folds={len(sp)} train_len0={len(sp[0][0])} test_len0={len(sp[0][1])}"
run("skfolio WalkForward calendar freq", skwf)
def pcv():
    import purgedcv, inspect
    return [n for n in dir(purgedcv) if not n.startswith('_')][:25]
run("purgedcv exports", pcv)
def pcv2():
    from purgedcv import PurgedKFold, CombinatorialPurgedCV
    t1 = pd.Series(idx.shift(5, freq="B") if False else idx[np.minimum(np.arange(len(idx))+5,len(idx)-1)], index=idx)
    k = PurgedKFold(n_splits=5, embargo_pct=0.01) if 'embargo_pct' in PurgedKFold.__init__.__code__.co_varnames else PurgedKFold(n_splits=5)
    return str(PurgedKFold.__init__.__code__.co_varnames[:8]) + f" splits={len(list(k.split(X.iloc[:, :1], groups=None)) if False else [])}"
run("purgedcv PurgedKFold sig", pcv2)
def tscv():
    from timeseriescv import cross_validation as cv
    return [n for n in dir(cv) if 'Split' in n or 'CV' in n]
run("timeseriescv", tscv)
def js():
    import jsharpe
    return [n for n in dir(jsharpe) if not n.startswith('_')]
run("jsharpe exports", js)
def ds():
    import deflated_sharpe as d
    return [n for n in dir(d) if not n.startswith('_')]
run("deflated_sharpe exports", ds)
def arch_spa():
    from arch.bootstrap import SPA, StepM, MCS, IIDBootstrap, StationaryBootstrap
    rng = np.random.default_rng(1)
    bench = rng.normal(0,1,500); models = rng.normal(0.0,1,(500,20)); models[:,0]+=0.2
    spa = SPA(bench, models, reps=300, seed=1); spa.compute()
    return f"SPA pvalues={spa.pvalues.round(3).to_dict()}"
run("arch SPA", arch_spa)
def arch_boot():
    from arch.bootstrap import StationaryBootstrap
    r = c.iloc[:,0].pct_change().dropna().values
    sh = lambda x: x.mean()/x.std(ddof=1)*np.sqrt(252)
    ci = StationaryBootstrap(21, r, seed=1).conf_int(sh, reps=300, method="percentile")
    return f"Sharpe 95% CI {ci.ravel().round(2)}"
run("arch StationaryBootstrap CI", arch_boot)
def scipy_boot():
    from scipy.stats import bootstrap
    r = c.iloc[:,0].pct_change().dropna().values
    res = bootstrap((r,), lambda x: x.mean()/x.std(ddof=1)*np.sqrt(252), method="percentile", n_resamples=300, random_state=1)
    return f"scipy bootstrap CI {res.confidence_interval}"
run("scipy.stats.bootstrap", scipy_boot)
