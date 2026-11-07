import time,sys
t=time.time()
import numpy as np, pandas as pd, vectorbt as vbt
t1=time.time()
c=pd.DataFrame(np.linspace(10,12,60)[:,None]*[1,1.1]); e=pd.DataFrame(False,index=c.index,columns=c.columns); e.iloc[3]=True
pf=vbt.Portfolio.from_signals(c,e,~e,size=0.5,size_type="percent",sl_stop=0.05,sl_trail=True,cash_sharing=True,group_by=True,freq="1D",engine=sys.argv[1])
pf.value(); pf.trades.count()
print(sys.argv[1],"import %.1fs first-call %.1fs"%(t1-t,time.time()-t1))
