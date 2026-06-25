import importlib.util

import pandas as pd

spec = importlib.util.spec_from_file_location("equity_backfill", "scripts/equity_backfill.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_resolve_symbols_includes_benchmark_and_members():
    start = pd.Timestamp("2021-01-01", tz="UTC")
    end = pd.Timestamp("2026-06-01", tz="UTC")
    syms = mod.resolve_symbols("midcap400", start, end, benchmark="MDY")
    assert "MDY" in syms
    assert 380 <= len([s for s in syms if s != "MDY"]) <= 420
    assert syms == sorted(set(syms))  # deduped + sorted
