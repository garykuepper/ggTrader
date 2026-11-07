import json,sys,math
a,b=sys.argv[1],sys.argv[2]
A=json.load(open(f"probes/out/{a}.json"));B=json.load(open(f"probes/out/{b}.json"))
def same(x,y):
    if isinstance(x,dict) and isinstance(y,dict):
        if (x['n_valid'],x['n_nan'],x['first_valid'])!=(y['n_valid'],y['n_nan'],y['first_valid']): return False
        return all((x[k]==y[k]) or abs((x[k] or 0)-(y[k] or 0))<=1e-9*max(1,abs(x[k] or 0)) for k in ('sum','last'))
    if isinstance(x,float) and isinstance(y,float):
        return x==y or (math.isnan(x) and math.isnan(y)) or abs(x-y)<=1e-9*max(1,abs(x))
    return x==y
def f(v):
    if isinstance(v,dict): return "n=%s nan=%s sum=%.10g last=%.10g"%(v['n_valid'],v['n_nan'],v['sum'] if v['sum'] is not None else float('nan'),v['last'] if v['last'] is not None else float('nan'))
    return "%.12g"%v if isinstance(v,float) else str(v)
d=[q for q in A if not q.startswith('_') and not same(A[q]['v'],B[q]['v'])]
print(f"--- {a} vs {b}: {len(d)} differ of {len([q for q in A if not q.startswith('_')])}")
for q in d: print("  %-32s %s | %s"%(q,f(A[q]['v']),f(B[q]['v'])))
