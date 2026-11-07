import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, "/home/flynn/ggTrader-vbt12/src")
from ggTrader.data.bars import load_bars, to_legacy_frame
from ggTrader.lab.data import eligible_at, equity_universe_between
from ggTrader.lab.replica import build_signal_panel, run_replica
from ggTrader.lab.strategy import LabConfig

OLD = np.load(sys.argv[1], allow_pickle=True)
es, ee = pd.Timestamp("2024-01-02", tz="UTC"), pd.Timestamp("2026-04-29", tz="UTC")
uni = equity_universe_between(es, ee, universe="sp500")
syms = sorted(uni)[:120]
bars = load_bars(sorted(set(syms) | {"SPY"}), start="2022-01-01", end="2026-04-30", source="snapshot")
legacy = to_legacy_frame(bars)
cfg = LabConfig(min_history_bars=60)
fn = lambda d, past: eligible_at(d, past, cfg, universe="sp500")[0]
panel = build_signal_panel(legacy, es, ee, fn)
res = run_replica(panel.close, panel.entries, panel.exits, panel.spy_close)
oe, ox = OLD["entries"], OLD["exits"]
ne, nx = panel.entries.to_numpy(), panel.exits.to_numpy()
assert list(OLD["cols"]) == list(panel.entries.columns), "columns differ"
assert (OLD["idx"] == panel.entries.index.astype("int64").to_numpy()).all(), "index differs"
out = {
 "shape": list(ne.shape),
 "entries_legacy": int(oe.sum()), "entries_new": int(ne.sum()), "entry_cells_differ": int((oe != ne).sum()),
 "exits_legacy": int(ox.sum()), "exits_new": int(nx.sum()), "exit_cells_differ": int((ox != nx).sum()),
 "equity_final_legacy": float(OLD["equity"][-1]), "equity_final_new": float(res.equity.iloc[-1]),
 "equity_max_abs_diff": float(np.abs(OLD["equity"] - res.equity.to_numpy()).max()),
 "equity_max_rel_diff": float((np.abs(OLD["equity"] - res.equity.to_numpy())/OLD["equity"]).max()),
}
diff_e = np.argwhere(oe != ne); diff_x = np.argwhere(ox != nx)
out["first_entry_diffs"] = [[str(panel.entries.index[i].date()), str(panel.entries.columns[j])] for i,j in diff_e[:10]]
out["first_exit_diffs"] = [[str(panel.entries.index[i].date()), str(panel.entries.columns[j])] for i,j in diff_x[:10]]
print(json.dumps(out, indent=1))
json.dump(out, open(sys.argv[2], "w"), indent=1)
