"""Fetch PyPI latest version, upload date, licence and pandas/numpy requirement lines for a package list."""
import json, sys, urllib.request, re
pk = sys.argv[1:]
for p in pk:
    try:
        d = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/{p}/json", timeout=30))
    except Exception as e:
        print(f"{p}\tNOT-ON-PYPI ({e})"); continue
    i = d["info"]; v = i["version"]; up = max((f["upload_time"] for f in d["urls"]), default="?")[:10]
    req = [r.split(";")[0].strip() for r in (i.get("requires_dist") or []) if re.match(r"(pandas|numpy|numba)\b", r, re.I) and "extra ==" not in r]
    lic = (i.get("license_expression") or i.get("license") or "")[:30].replace("\n", " ")
    print(f"{p}\t{v}\t{up}\tpy{i.get('requires_python')}\t{lic}\t{'; '.join(req)}")
