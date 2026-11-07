"""Compare old-stack vs new-stack e2e artefacts (run under the NEW venv so pandas 3 reads both)."""
import json
import sys

import numpy as np
import pandas as pd

E = sys.argv[1]


def jd(path):
    return json.load(open(path))


print("=== run_wfo (2 folds, 60 symbols) ===")
o, n = jd(f"{E}/e2e_old/summary.json"), jd(f"{E}/e2e_new/summary.json")
for k in ("pandas", "n_symbols", "rows", "folds", "grid", "load_s", "wfo_s", "peak_rss_mb", "oos_equity_len", "oos_equity_last"):
    print(f"{k:18} old={o.get(k)!s:24} new={n.get(k)!s}")
print("fold_results identical:", o["fold_results"] == n["fold_results"])
print("live_params identical:", o["live_params"] == n["live_params"])
eo = pd.read_csv(f"{E}/e2e_old/oos_equity.csv", index_col=0)
en = pd.read_csv(f"{E}/e2e_new/oos_equity.csv", index_col=0)
print("oos_equity index equal:", list(eo.index) == list(en.index), "| max abs diff:", float(np.abs(eo.iloc[:, 0] - en.iloc[:, 0]).max()))
if o["fold_results"] != n["fold_results"]:
    for i, (a, b) in enumerate(zip(o["fold_results"], n["fold_results"])):
        for k in a:
            if a[k] != b.get(k):
                print(f"fold {i} key {k}: old={str(a[k])[:100]} new={str(b.get(k))[:100]}")

for tag in ("signal", "weight"):
    print(f"=== sweep: {tag} path ===")
    try:
        mo = pd.read_csv(f"{E}/sweep_old/{tag}_metrics.csv").set_index("combo")
        mn = pd.read_csv(f"{E}/sweep_new/{tag}_metrics.csv").set_index("combo")
    except FileNotFoundError as e:
        print("missing", e)
        continue
    print("combos old/new:", len(mo), len(mn), "same set:", set(mo.index) == set(mn.index))
    common = mo.index.intersection(mn.index)
    d = (mo.loc[common] - mn.loc[common]).abs()
    print("max abs metric diff by column:")
    print(d.max().to_string())
    qo = pd.read_parquet(f"{E}/sweep_old/{tag}_equity.parquet")
    qn = pd.read_parquet(f"{E}/sweep_new/{tag}_equity.parquet")
    qo.index = qo.index.tz_convert("UTC").as_unit("ns")
    qn.index = qn.index.tz_convert("UTC").as_unit("ns")
    cc = qo.columns.intersection(qn.columns)
    print("equity index equal:", qo.index.equals(qn.index), "| curves compared:", len(cc))
    diff = (qo[cc] - qn[cc]).abs()
    print("max abs equity diff:", float(np.nanmax(diff.to_numpy())), "| bit-identical:", bool(np.array_equal(qo[cc].to_numpy(), qn[cc].to_numpy(), equal_nan=True)))
    rel = (diff / qo[cc].abs()).to_numpy()
    print("max rel equity diff:", float(np.nanmax(rel)))
    if tag == "signal":
        mk_o = pd.read_parquet(f"{E}/sweep_old/pit_mask.parquet")
        mk_n = pd.read_parquet(f"{E}/sweep_new/pit_mask.parquet")
        print("PIT mask identical:", bool(np.array_equal(mk_o.to_numpy(), mk_n.to_numpy())), mk_o.shape, mk_n.shape)
so, sn = jd(f"{E}/sweep_old/summary.json"), jd(f"{E}/sweep_new/summary.json")
for k in ("pit_mask_s", "sweep_signal_s", "sweep_weight_s", "signal_combos", "weight_combos"):
    print(f"{k:16} old={so.get(k)} new={sn.get(k)}")
