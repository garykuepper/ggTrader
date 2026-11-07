"""Extract vbt.* symbols and Portfolio method/kwarg names from the marketcalls skill and test them against installed vectorbt 1.2.0."""
import re, glob, inspect, collections, vectorbt as vbt, sys
root = sys.argv[1]
import os
files = [os.path.join(d,f) for d,_,fs in os.walk(root) for f in fs if f.endswith((".md",".py"))]
txt = "\n".join(open(f, errors="ignore").read() for f in files)
syms = collections.Counter(re.findall(r"\bvbt\.([A-Za-z_][A-Za-z_0-9]*(?:\.[A-Za-z_][A-Za-z_0-9]*)*)", txt))
bad = []
for s, n in syms.items():
    o = vbt
    try:
        for p in s.split("."): o = getattr(o, p)
    except AttributeError:
        bad.append((s, n))
print(f"files={len(files)} distinct vbt.* symbols={len(syms)} missing_in_{vbt.__version__}={len(bad)}")
for s, n in sorted(bad): print("  MISSING", s, n)
# kwargs passed to from_signals / from_orders
for fn in ("from_signals", "from_orders"):
    sig = set(inspect.signature(getattr(vbt.Portfolio, fn)).parameters)
    used = collections.Counter()
    for m in re.finditer(r"Portfolio\." + fn + r"\((.*?)\n\s*\)", txt, re.S):
        used.update(re.findall(r"\b([a-z_]+)\s*=", m.group(1)))
    unk = {k: v for k, v in used.items() if k not in sig}
    print(fn, "kwargs used", len(used), "unknown to 1.2.0 signature:", unk)
pfm = collections.Counter(re.findall(r"\bpf\.([a-z_]+)\(", txt))
pfbad = {k: v for k, v in pfm.items() if not hasattr(vbt.Portfolio, k)}
print("pf.<method>( distinct", len(pfm), "missing on Portfolio:", pfbad)
