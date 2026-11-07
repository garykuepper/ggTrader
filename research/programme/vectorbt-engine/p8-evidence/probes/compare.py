import json,sys
names=sys.argv[1:]
d={n:json.load(open(f"probes/out/{n}.json")) for n in names}
base=d[names[0]]
def fmt(v):
    if isinstance(v,dict): return "n=%s nan=%s sum=%.10g last=%.10g fv=%s"%(v['n_valid'],v['n_nan'],v['sum'] if v['sum'] is not None else float('nan'),v['last'] if v['last'] is not None else float('nan'),v['first_valid'])
    if isinstance(v,float): return "%.12g"%v
    return str(v)
def same(a,b):
    if isinstance(a,dict) and isinstance(b,dict):
        if (a['n_valid'],a['n_nan'],a['first_valid'])!=(b['n_valid'],b['n_nan'],b['first_valid']): return False
        return all(abs((a[k] or 0)-(b[k] or 0))<=1e-9*max(1,abs(a[k] or 0)) for k in ('sum','last'))
    if isinstance(a,float) and isinstance(b,float):
        import math
        if math.isnan(a) and math.isnan(b): return True
        return (a==b or abs(a-b)<=1e-9*max(1,abs(a)))
    return a==b
for q in sorted(base):
    if q.startswith('_'): continue
    row=[(n,d[n].get(q,{}).get('v')) for n in names]
    allsame=all(same(row[0][1],v) for _,v in row[1:])
    print(("SAME   " if allsame else "DIFFER ")+q)
    if not allsame:
        for n,v in row: print("      %-8s %s"%(n,fmt(v)))
