"""One textual mutation per scratch copy of the worktree; run tests/engine; record killers."""
import json, os, shutil, subprocess, sys, time
from pathlib import Path
S = Path(sys.argv[1]); BASE = S/"base"
E = "src/ggTrader/engine/"
M = [
 ("zero_trade_guard_series", E+"stats.py", ".fillna(0) > 0, np.nan)", ".fillna(0) >= 0, np.nan)"),
 ("zero_trade_guard_scalar", E+"stats.py", "== 0:\n        return float(\"nan\")\n    return out", "== -1:\n        return float(\"nan\")\n    return out"),
 ("trade_count_window_end_inclusive", E+"stats.py", '(trades["entry_idx"] < end)', '(trades["entry_idx"] <= end)'),
 ("mask_after_time_stop", E+"portfolio.py", "    if mask is not None:\n        entries = entries & _as_bool(\n            mask.reindex(index=entries.index, columns=entries.columns, fill_value=False)\n        )\n    if td_stop is not None:\n        exits = exits | sig.time_stop(entries, int(td_stop))\n", "    raw_entries = entries\n    if mask is not None:\n        entries = entries & _as_bool(\n            mask.reindex(index=entries.index, columns=entries.columns, fill_value=False)\n        )\n    if td_stop is not None:\n        exits = exits | sig.time_stop(raw_entries, int(td_stop))\n"),
 ("contract_diff_comparable", E+"provenance.py", '_COMPARABLE = {"SAME_STACK", "ENGINE_DIFF", "PATCH_DIFF", "VBT_VERSION_DIFF"}', '_COMPARABLE = {"SAME_STACK", "ENGINE_DIFF", "PATCH_DIFF", "VBT_VERSION_DIFF", "CONTRACT_DIFF"}'),
 ("no_version_mismatch_error", E+"config.py", '    warnings.filterwarnings("error", message=_MISMATCH_MESSAGE.pattern, category=UserWarning)\n', ''),
 ("selfcheck_reports_configured_engine", E+"selfcheck.py", '"engine": "rust" if calls["n"] > 0 else "numba"', '"engine": vbt.settings["engine"]'),
 # new in the fix-up
 ("window_includes_first_return", E+"stats.py", "window = frame.iloc[start + 1 : end]", "window = frame.iloc[start:end]"),
 ("drawdown_without_base_bar", E+"stats.py", '"max_drawdown": max_drawdown(base),', '"max_drawdown": max_drawdown(window),'),
 ("legacy_sortino_is_vbt", E+"stats.py", "sortino_values = legacy_defs.sortino(window)", "sortino_values = sortino(window)"),
 ("tape_ignored_in_compare", E+"provenance.py", '_IGNORED_KEYS = {"created_utc", "git_sha", "git_dirty", "lock_sha256", "rust_calls"}', '_IGNORED_KEYS = {"created_utc", "git_sha", "git_dirty", "lock_sha256", "rust_calls", "tape_sha256"}'),
 ("chunk_kept_alive", E+"portfolio.py", "        del pf, value, records\n        _release_memory()\n", ""),
 ("vol_keys_ignored", E+"portfolio.py", "    _reject_unsupported(base_config)\n    _check_shapes(close, entries=entries, exits=exits)", "    _check_shapes(close, entries=entries, exits=exits)"),
]
only = sys.argv[2:]
outp = S/"mutation_results.json"
res = json.loads(outp.read_text()) if outp.exists() else {}
for name, f, old, new in M:
    if only and name not in only: continue
    d = S/"run"/name
    if d.exists(): shutil.rmtree(d)
    shutil.copytree(BASE, d, symlinks=True)
    p = d/f; txt = p.read_text()
    n = txt.count(old)
    if n != 1:
        res[name] = {"error": f"pattern count {n}"}; print(name, "PATTERN", n, flush=True); shutil.rmtree(d); continue
    p.write_text(txt.replace(old, new))
    t = time.time()
    r = subprocess.run(["scripts/run_tests.sh","tests/engine","-q","-p","no:cacheprovider","-o","addopts="], cwd=d, capture_output=True, text=True, env={**os.environ, "GGT_PYTHON":"/home/flynn/ggTrader-vbt12/.venv/bin/python"})
    tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-300:]
    failed = sorted({l.split(" ")[1] for l in r.stdout.splitlines() if l.startswith(("FAILED","ERROR"))})
    res[name] = {"survived": r.returncode == 0, "rc": r.returncode, "summary": tail, "failing_tests": failed, "files": sorted({x.split("::")[0] for x in failed}), "secs": round(time.time()-t)}
    print(name, "SURVIVED" if r.returncode == 0 else "killed", len(failed), res[name]["files"], flush=True)
    outp.write_text(json.dumps(res, indent=1))
    shutil.rmtree(d)
