import json, sys, math
A=json.load(open(sys.argv[1])); B=json.load(open(sys.argv[2]))
mx={"abs":0.0,"rel":0.0}; disc=[]
def walk(a,b,path=""):
    if isinstance(a,dict):
        for k in a: walk(a[k],b.get(k),path+"/"+k)
    elif isinstance(a,list):
        if len(a)!=len(b): disc.append((path,"len",len(a),len(b))); return
        for i,(x,y) in enumerate(zip(a,b)): walk(x,y,path+f"[{i}]" if not isinstance(x,(int,float)) else path)
        if a and isinstance(a[0],(int,float)):
            for x,y in zip(a,b):
                if x is None or y is None: 
                    if x!=y: disc.append((path,"none")); 
                    continue
                if isinstance(x,float) and math.isnan(x) and math.isnan(y): continue
                d=abs(x-y); mx["abs"]=max(mx["abs"],d); 
                if x!=0: mx["rel"]=max(mx["rel"],d/abs(x))
                if isinstance(x,int) and x!=y: disc.append((path,"int",x,y))
    elif isinstance(a,(int,float)):
        d=abs(a-b); mx["abs"]=max(mx["abs"],d)
        if a!=0: mx["rel"]=max(mx["rel"],d/abs(a))
        if a!=b: disc.append((path,"val",a,b))
    elif a!=b: disc.append((path,"other",a,b))
walk(A,B)
print(f"max_abs={mx['abs']:.3e} max_rel={mx['rel']:.3e} discrete_or_scalar_diffs={len(disc)}")
for d in disc[:8]: print("   ",d)
