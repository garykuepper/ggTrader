import pstats
import sys

E = sys.argv[1]
for tag in ("old", "new"):
    st = pstats.Stats(f"{E}/52_prof_{tag}.prof")
    print(f"===== {tag}: total {st.total_tt:.1f}s, calls {st.total_calls}")
    rows = []
    for (fn, ln, name), (cc, nc, tt, ct, callers) in st.stats.items():
        rows.append((ct, tt, nc, fn, ln, name))
    rows.sort(reverse=True)
    n = 0
    for ct, tt, nc, fn, ln, name in rows:
        if "ggTrader" in fn and ".venv" not in fn and "site-packages" not in fn:
            print(f"cum {ct:6.2f}s  tot {tt:5.2f}s  n={nc:7d}  {fn.split('src/')[-1]}:{ln}({name})")
            n += 1
            if n >= 14:
                break
    # vectorbt cumulative
    for ct, tt, nc, fn, ln, name in rows:
        if "vectorbt" in fn and name in ("from_signals", "simulate_from_signals", "value", "copy", "wrapper_h", "_add_trade_counts"):
            print(f"  vbt cum {ct:6.2f}s n={nc} {fn.split('site-packages/')[-1]}:{ln}({name})")
