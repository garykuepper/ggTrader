"""S5 prototype step 3: signal-diff report across arms, with one-at-a-time attribution."""

import argparse
import json
import os

import numpy as np
import pandas as pd

VOTERS = ("bb", "rsi", "ema", "macd", "vbb")
ARMS = ["A_ggt_pd2", "B_ggt_pd3", "C_vbt_matched", "D_vbt_matched_vbtcross", "E_vbt_default"]
BURN = 200


def load(ev, arm, key, side):
    return pd.read_parquet(os.path.join(ev, "arms", arm, f"{key}_{side}.parquet"))


def diff_stats(a, b, valid, post):
    a, b = a.reindex_like(valid).fillna(False), b.reindex_like(valid).fillna(False)
    x = (a ^ b) & valid
    union = (a | b) & valid
    first = None
    if x.to_numpy().any():
        r, c = np.argwhere(x.to_numpy())[0]
        first = f"{x.index[r].date()} {x.columns[c]} ref={bool(a.iat[r, c])}"
    return {
        "ref_true": int((a & valid).to_numpy().sum()),
        "new_true": int((b & valid).to_numpy().sum()),
        "xor_cells": int(x.to_numpy().sum()),
        "xor_frac_of_valid_cells": float(x.to_numpy().sum() / valid.to_numpy().sum()),
        "xor_frac_of_signal_cells": float(x.to_numpy().sum() / max(1, union.to_numpy().sum())),
        "xor_cells_after_burnin": int((x & post).to_numpy().sum()),
        "symbols_affected": int(x.any().sum()),
        "first_diff": first,
    }


def ens(v, ma=2, mx=2):
    ev = sum(v[k][0].astype(int) for k in VOTERS)
    xv = sum(v[k][1].astype(int) for k in VOTERS)
    return ev >= ma, v["rsi"][1] | (xv >= mx)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", required=True)
    a = ap.parse_args()
    close = pd.read_parquet(os.path.join(a.evidence, "tape_close.parquet"))
    valid = close.notna()
    post = valid.cumsum() > BURN  # per-symbol: beyond the first 200 valid bars
    rep = {"burnin_bars": BURN, "valid_cells": int(valid.to_numpy().sum()), "pairs": {},
           "attribution_ensemble": {}, "live_window": {}}
    sig = {arm: {k: (load(a.evidence, arm, k, "entries"), load(a.evidence, arm, k, "exits"))
                 for k in VOTERS + ("ENS",)} for arm in ARMS}
    pairs = [("A_ggt_pd2", arm) for arm in ARMS[1:]] + list(zip(ARMS[1:-1], ARMS[2:]))
    for r, n in pairs:
        key = f"{r} -> {n}"
        rep["pairs"][key] = {}
        for k in VOTERS + ("ENS",):
            for i, side in enumerate(("entries", "exits")):
                rep["pairs"][key][f"{k}.{side}"] = diff_stats(sig[r][k][i], sig[n][k][i], valid, post)
    # One-at-a-time: swap ONE voter from arm X into the legacy voter set, re-vote.
    base = {k: sig["A_ggt_pd2"][k] for k in VOTERS}
    be, bx = ens(base)
    for arm in ARMS[2:]:
        for k in VOTERS:
            v = dict(base)
            v[k] = sig[arm][k]
            e, x = ens(v)
            rep["attribution_ensemble"][f"{arm}:{k}"] = {
                "entries": diff_stats(be, e, valid, post),
                "exits": diff_stats(bx, x, valid, post),
            }
    for arm in ARMS[1:]:
        out = {}
        for side in ("entries", "exits"):
            r = load(a.evidence, "A_ggt_pd2", "LIVE", side)
            n = load(a.evidence, arm, "LIVE", side).reindex_like(r).fillna(False)
            x = r ^ n
            out[side] = {
                "sessions": int(len(r)),
                "ref_signals": int(r.to_numpy().sum()),
                "new_signals": int(n.to_numpy().sum()),
                "differing_cells": int(x.to_numpy().sum()),
                "sessions_with_any_diff": int(x.any(axis=1).sum()),
                "examples": [f"{d.date()} {s} ref={bool(r.at[d, s])}"
                             for d, s in zip(*np.where(x.to_numpy()))
                             for d, s in [(r.index[d], r.columns[s])]][:5],
            }
        rep["live_window"][arm] = out
    with open(os.path.join(a.evidence, "signal_diff_report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    # Compact text table
    lines = []
    for key, d in rep["pairs"].items():
        lines.append(f"\n== {key}")
        lines.append(f"{'voter.side':16s} {'ref':>6s} {'new':>6s} {'xor':>6s} {'xor/sig':>8s} "
                     f"{'post200':>7s} {'syms':>4s}  first")
        for k, s in d.items():
            lines.append(f"{k:16s} {s['ref_true']:6d} {s['new_true']:6d} {s['xor_cells']:6d} "
                         f"{s['xor_frac_of_signal_cells']:8.4f} {s['xor_cells_after_burnin']:7d} "
                         f"{s['symbols_affected']:4d}  {s['first_diff']}")
    lines.append("\n== one-at-a-time attribution (swap one voter into legacy set; ensemble XOR)")
    for k, s in rep["attribution_ensemble"].items():
        lines.append(f"{k:40s} ent xor {s['entries']['xor_cells']:5d} (post {s['entries']['xor_cells_after_burnin']:4d})"
                     f"  exit xor {s['exits']['xor_cells']:5d} (post {s['exits']['xor_cells_after_burnin']:4d})")
    lines.append("\n== live window (365 cal days, last bar, last N sessions)")
    for arm, d in rep["live_window"].items():
        lines.append(f"{arm:26s} " + "  ".join(
            f"{s}: ref {v['ref_signals']} new {v['new_signals']} diff {v['differing_cells']} "
            f"sessions {v['sessions_with_any_diff']}/{v['sessions']}" for s, v in d.items()))
    txt = "\n".join(lines)
    open(os.path.join(a.evidence, "signal_diff_report.txt"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
