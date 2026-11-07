"""Cross-library DSR / PSR / PBO consistency on one synthetic 48-config grid (null: no skill) and one with a planted winner."""
import numpy as np, pandas as pd, scipy.stats as st, purgedcv, jsharpe, deflated_sharpe as dsr_mod
rng=np.random.default_rng(7); T=1000; N=48
null=rng.normal(0,0.01,(N,T)); planted=null.copy(); planted[5]+=0.0012
def manual_dsr(R, n_trials):
    sr=R.mean(1)/R.std(1,ddof=1)                      # per-period Sharpe of each config
    best=int(np.argmax(sr)); r=R[best]; T=len(r)
    g3=st.skew(r); g4=st.kurtosis(r,fisher=False)     # NON-excess kurtosis (the trap in vbt 0.28.5)
    v=np.var(sr,ddof=1); emc=0.5772156649
    sr0=np.sqrt(v)*((1-emc)*st.norm.ppf(1-1/n_trials)+emc*st.norm.ppf(1-1/(n_trials*np.e)))
    z=(sr[best]-sr0)*np.sqrt(T-1)/np.sqrt(1-g3*sr[best]+(g4-1)/4*sr[best]**2)
    return best, st.norm.cdf(z), sr[best], sr0, np.sqrt(v)
for name,R in (("null",null),("planted",planted)):
    best,m,srb,sr0,sdv=manual_dsr(R,N)
    p=purgedcv.deflated_sharpe_ratio(R[best],n_trials=N,var_sharpe=sdv**2)
    out=[f"{name}: best={best} per-period SR={srb:.4f} SR0={sr0:.4f}  DSR manual={m:.4f} purgedcv={p:.4f}"]
    try:
        g3=st.skew(R[best]); g4=st.kurtosis(R[best],fisher=False)
        out.append(f"jsharpe PSR(SR,SR0)={jsharpe.probabilistic_sharpe_ratio(srb,sr0,T=T,gamma3=g3,gamma4=g4):.4f}")
    except Exception as e: out.append(f"jsharpe ERR {e}")
    try:
        d=dsr_mod.deflated_sharpe_ratio(observed_sr=srb*np.sqrt(252),num_trials=N,num_obs=T,skewness=g3,kurtosis=g4); out.append(f"deflated-sharpe(mnemox)={d} [takes ANNUALISED SR: sqrt(252) scaling inside? unverified semantics]")
    except Exception as e: out.append(f"deflated-sharpe ERR {e}")
    print("OK  ", " | ".join(out))
    pbo=purgedcv.probability_of_backtest_overfitting(R,n_splits=10)
    print("OK   PBO(%s) via purgedcv CSCV: %s" % (name, getattr(pbo,'pbo',pbo)))
