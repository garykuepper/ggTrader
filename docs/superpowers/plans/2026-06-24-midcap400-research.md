# MidCap 400 Research (Bias-Quantified) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether the 5-voter + 3%-exposure strategy is deployable on the S&P MidCap 400, with an explicit *measured* survivorship-bias bound (since clean PIT midcap data doesn't exist publicly).

**Architecture:** Slot `midcap400` into the existing snapshot-universe machinery (current Wikipedia ticker list → `_SNAPSHOT_REGISTRY`). Backfill its OHLCV + the MDY benchmark. Self-calibrate the survivorship bias by running the *same* WFO on the SP500 span-union (≈PIT) vs the SP500 current snapshot — `run_wfo` trades whatever symbols are loaded, so the universe is set purely by which tickers go into the OHLCV. Then run the midcap WFO vs MDY and apply the calibrated haircut.

**Tech Stack:** Python 3.12, pandas, numpy, vectorbt, TimescaleDB, pytest. Native `.venv` for research (Docker is live-only). Absolute imports from `ggTrader`.

## Global Constraints

- Run natively: `source .venv/bin/activate` before any python/pytest. Docker is live-only.
- Strict ruff lint must pass; vectorized pandas/numpy.
- Absolute imports from `ggTrader`.
- Strategy under test: `EnsembleSignal` defaults (the 5-voter `bb,rsi,ema,macd,vbb`) at `SIGNAL_POSITION_SIZE=0.03` (already the `STOCK_BASE_CONFIG` default).
- WFO settings match the SP500 work: `eval_start=2021-01-31`, rolling 12mo/3mo, 17 folds, `run_wfo` from `ggTrader.lab.wfo`.
- Ticker normalization: always pass raw tickers through `normalize_yf_ticker` (handles `MOG.A` → `MOG-A`).
- Benchmark: MDY (SPDR S&P MidCap 400) for the midcap run; SPY reported as cross-reference.
- Snapshot universes carry survivorship bias by construction — never report a midcap number without the calibrated haircut from Task 3.

---

### Task 1: MidCap 400 snapshot + universe wiring

Source the current S&P 400 ticker list, commit it as a snapshot file, and register `midcap400` so `--universe midcap400` flows through the existing machinery.

**Files:**
- Create: `data/universe/midcap400_tickers_snapshot_2026-06-24.txt` (generated)
- Create: `scripts/fetch_midcap400_snapshot.py` (one-time generator, committed for reproducibility)
- Modify: `src/ggTrader/data/core/index_constituents.py` (`_SNAPSHOT_REGISTRY`)
- Modify: `src/ggTrader/lab/cli.py:22` (`UNIVERSE_CHOICES`)
- Test: `tests/lab/test_midcap_universe.py`

**Interfaces:**
- Consumes: `normalize_yf_ticker`, `snapshot_members`, `_SNAPSHOT_REGISTRY` (all in `ggTrader.data.core.index_constituents`).
- Produces: `snapshot_members("midcap400") -> List[str]` (normalized tickers); `"midcap400"` present in `UNIVERSE_CHOICES`.

- [ ] **Step 1: Write the fetch generator script**

```python
# scripts/fetch_midcap400_snapshot.py
"""One-time: scrape the current S&P MidCap 400 tickers from Wikipedia into a
snapshot file (same format as the nasdaq100/russell2000 snapshots). Survivorship
note: this is a CURRENT snapshot, not point-in-time — see the midcap400 research
spec for the bias-calibration that bounds the resulting bias.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "src")
from ggTrader.data.core.index_constituents import normalize_yf_ticker  # noqa: E402

URL = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
OUT = Path("data/universe/midcap400_tickers_snapshot_2026-06-24.txt")


def main() -> None:
    tables = pd.read_html(URL)
    # The constituents table is the one with a "Symbol" column.
    const = next(t for t in tables if "Symbol" in t.columns)
    raw = [str(s).strip() for s in const["Symbol"].tolist() if str(s).strip()]
    tickers = sorted({normalize_yf_ticker(t) for t in raw})
    if not (380 <= len(tickers) <= 420):
        raise SystemExit(f"Expected ~400 tickers, got {len(tickers)} — check the page structure")
    OUT.write_text("\n".join(tickers) + "\n")
    print(f"wrote {len(tickers)} tickers to {OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the generator to produce the snapshot file**

Run: `source .venv/bin/activate && python scripts/fetch_midcap400_snapshot.py`
Expected: `wrote ~400 tickers to data/universe/midcap400_tickers_snapshot_2026-06-24.txt`. If the count gate fails, inspect `pd.read_html(URL)` tables and adjust the table selection.

- [ ] **Step 3: Write the failing test**

```python
# tests/lab/test_midcap_universe.py
from ggTrader.data.core.index_constituents import snapshot_members
from ggTrader.lab.cli import UNIVERSE_CHOICES


def test_midcap400_registered_in_universe_choices():
    assert "midcap400" in UNIVERSE_CHOICES


def test_midcap400_snapshot_loads_normalized():
    members = snapshot_members("midcap400")
    assert 380 <= len(members) <= 420
    # normalized: no dotted class tickers (MOG.A -> MOG-A)
    assert all("." not in m for m in members)
    assert members == sorted(members)
```

- [ ] **Step 4: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/lab/test_midcap_universe.py -v`
Expected: FAIL — `"midcap400"` not in `UNIVERSE_CHOICES`; `snapshot_members("midcap400")` raises `ValueError` (unknown snapshot universe).

- [ ] **Step 5: Register midcap400**

In `src/ggTrader/data/core/index_constituents.py`, add to `_SNAPSHOT_REGISTRY`:

```python
_SNAPSHOT_REGISTRY: Dict[str, str] = {
    "nasdaq100": "nasdaq100_tickers_snapshot_2026-06-09.txt",
    "russell2000": "russell2000_tickers_snapshot_2026-06-09.txt",
    "midcap400": "midcap400_tickers_snapshot_2026-06-24.txt",
}
```

In `src/ggTrader/lab/cli.py:22`, extend `UNIVERSE_CHOICES`:

```python
UNIVERSE_CHOICES = ("sp500", "nasdaq100", "russell2000", "midcap400")
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/lab/test_midcap_universe.py -v`
Expected: PASS (both tests).

- [ ] **Step 7: Lint + commit**

```bash
source .venv/bin/activate && ruff check scripts/fetch_midcap400_snapshot.py tests/lab/test_midcap_universe.py src/ggTrader/data/core/index_constituents.py src/ggTrader/lab/cli.py
git add scripts/fetch_midcap400_snapshot.py data/universe/midcap400_tickers_snapshot_2026-06-24.txt src/ggTrader/data/core/index_constituents.py src/ggTrader/lab/cli.py tests/lab/test_midcap_universe.py
git commit -m "feat(lab): add midcap400 snapshot universe"
```

---

### Task 2: OHLCV backfill (midcap400 members + MDY)

Extend the backfill to accept any universe and always include a benchmark, then backfill midcap400 + MDY into TimescaleDB.

**Files:**
- Modify: `scripts/equity_backfill.py`
- Test: `tests/lab/test_backfill_universe.py`

**Interfaces:**
- Consumes: `universe_all_between(universe, start, end)`, `normalize_yf_ticker` (from `index_constituents`); `fetch_stock_ohlcv` (from `ggTrader.lab.data`); `snapshot_members("midcap400")` (Task 1).
- Produces: `python scripts/equity_backfill.py --universe midcap400` backfills all midcap400 members + MDY. (No new importable symbol; the deliverable is populated DB rows + a runnable CLI.)

- [ ] **Step 1: Write the failing test for universe arg + benchmark inclusion**

The current script hardcodes SP500 via `all_members_between`. Factor member-resolution into a testable function so we can assert it without downloading.

```python
# tests/lab/test_backfill_universe.py
import importlib.util
from pathlib import Path

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/lab/test_backfill_universe.py -v`
Expected: FAIL — `AttributeError: module 'equity_backfill' has no attribute 'resolve_symbols'`.

- [ ] **Step 3: Add `resolve_symbols` + `--universe`/`--benchmark` args**

In `scripts/equity_backfill.py`, add the helper and wire the args:

```python
from ggTrader.data.core.index_constituents import (
    normalize_yf_ticker,
    universe_all_between,
)


def resolve_symbols(universe, start_ts, end_ts, benchmark):
    """All members of `universe` across [start, end] plus the benchmark, normalized + sorted."""
    members = {normalize_yf_ticker(t) for t in universe_all_between(universe, start_ts, end_ts)}
    members.add(normalize_yf_ticker(benchmark))
    return sorted(members)
```

In `main()`, add args and use the helper:

```python
    p.add_argument("--universe", default="sp500", help="Universe to backfill (default: sp500)")
    p.add_argument("--benchmark", default=None,
                   help="Benchmark ticker to include (default: SPY for sp500, MDY for midcap400)")
    ...
    benchmark = args.benchmark or ("MDY" if args.universe == "midcap400" else "SPY")
    members = resolve_symbols(args.universe, start_ts, end_ts, benchmark)
```

(Replace the existing hardcoded `members = sorted({normalize_yf_ticker(t) for t in all_members_between(...)})` line.)

- [ ] **Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest tests/lab/test_backfill_universe.py -v`
Expected: PASS. (This resolves members offline; no download.)

- [ ] **Step 5: Run the actual backfill (data task)**

Run: `source .venv/bin/activate && python scripts/equity_backfill.py --universe midcap400 --start 2018-01-01 2>&1 | tee midcap_backfill.log`
Expected: backfills ~400 symbols + MDY into TimescaleDB (several minutes; some yfinance gaps on smaller names are normal). Record the count of symbols that returned data vs requested — this is the coverage figure the research run will cite.

- [ ] **Step 6: Lint + commit**

```bash
source .venv/bin/activate && ruff check scripts/equity_backfill.py tests/lab/test_backfill_universe.py
git add scripts/equity_backfill.py tests/lab/test_backfill_universe.py
git commit -m "feat(lab): backfill any universe + benchmark; backfill midcap400 + MDY"
```

---

### Task 3: Survivorship-bias calibration harness

Quantify the snapshot-vs-PIT gap on SP500 (where we have both) under the exact strategy, yielding the haircut for the midcap result.

**Files:**
- Create: `scripts/midcap_bias_calibration.py`

**Interfaces:**
- Consumes: `run_wfo` (`ggTrader.lab.wfo`), `EnsembleSignal`, `build_grid`, `STOCK_BASE_CONFIG`, `equity_universe_between`, `load_ohlcv`, `LabConfig`; `ablation_voters.parse_table` to read `run_wfo` output; the SP500 snapshot file `data/universe/sp500_tickers_snapshot_2026-06-09.txt`.
- Produces: stdout — SP500-PIT vs SP500-snapshot OOS (CAGR, Sharpe) and `Δ = snapshot − pit`. Writes the Δ to `bias_calibration.log`.

- [ ] **Step 1: Write the harness**

Two WFO runs differing only in which SP500 symbols load:
- **PIT (span-union):** `universe = equity_universe_between(eval_start, eval_end, universe="sp500")`.
- **Snapshot:** `universe = [normalize_yf_ticker(t) for t in Path(".../sp500_tickers_snapshot_2026-06-09.txt").read_text().split()]`.

For each, load OHLCV (members + SPY), drop SPY into `spy_close`, run `run_wfo("ensemble", EnsembleSignal, cfg, ohlcv, spy_close, eval_start, eval_end, "equity", dict(STOCK_BASE_CONFIG), build_grid(EnsembleSignal))`, parse the OOS line. Reuse the load/parse pattern from `scripts/ablation_voters.py`. Then:

```python
delta_cagr = snap["oos_cagr"] - pit["oos_cagr"]
delta_sharpe = snap["oos_sharpe"] - pit["oos_sharpe"]
print(f"SP500 PIT:      CAGR {pit['oos_cagr']:.1f}% Sharpe {pit['oos_sharpe']:.2f}")
print(f"SP500 snapshot: CAGR {snap['oos_cagr']:.1f}% Sharpe {snap['oos_sharpe']:.2f}")
print(f"Survivorship haircut  Δ: CAGR {delta_cagr:+.1f}pp  Sharpe {delta_sharpe:+.2f}")
```

- [ ] **Step 2: Run it and record the haircut**

Run: `source .venv/bin/activate && python scripts/midcap_bias_calibration.py 2>&1 | tee bias_calibration.log`
Expected: prints both runs + Δ. The PIT run should reproduce ≈ the known SP500 result (CAGR 16.2% / Sharpe 1.09) — if it doesn't, stop and reconcile before trusting Δ. Record Δ for Task 4.

- [ ] **Step 3: Commit**

```bash
source .venv/bin/activate && ruff check scripts/midcap_bias_calibration.py
git add scripts/midcap_bias_calibration.py
git commit -m "feat(lab): survivorship-bias calibration (SP500 PIT vs snapshot)"
```

---

### Task 4: MidCap research run + verdict

Run the strategy on midcap400 vs MDY, apply the haircut, and record the verdict.

**Files:**
- Create: `scripts/midcap_research.py`
- Modify: `docs/roadmap.md` (research direction G result)

**Interfaces:**
- Consumes: `run_wfo`, `EnsembleSignal`, `build_grid`, `STOCK_BASE_CONFIG`, `equity_universe_between`, `load_ohlcv`, `LabConfig`; `ablation_voters.parse_table`; the Task 3 Δ (read from `bias_calibration.log` or passed as a constant after recording).
- Produces: stdout — midcap400 OOS (raw + haircut-adjusted) vs MDY and SPY; a PASS/FAIL verdict on "beats MDY after haircut".

- [ ] **Step 1: Write the research script**

Load `universe = equity_universe_between(eval_start, eval_end, universe="midcap400")` + `MDY` + `SPY`. Split MDY and SPY out of the OHLCV; pass **MDY's close** as the benchmark series to `run_wfo` (this is where SP500 passes SPY). Parse the OOS line. Report coverage (members requested vs with data). Apply the Task-3 haircut:

```python
adj_cagr = mid["oos_cagr"] - delta_cagr      # delta_cagr from Task 3 (snapshot inflation)
adj_sharpe = mid["oos_sharpe"] - delta_sharpe
print(f"midcap400 raw:       CAGR {mid['oos_cagr']:.1f}% Sharpe {mid['oos_sharpe']:.2f}")
print(f"midcap400 haircut:   CAGR {adj_cagr:.1f}% Sharpe {adj_sharpe:.2f}  (Δ from SP500 calibration)")
print(f"MDY benchmark:       CAGR {mid['spy_cagr']:.1f}% Sharpe {mid['spy_sharpe']:.2f}")
beats = adj_cagr > mid["spy_cagr"] and adj_sharpe > mid["spy_sharpe"]
print(f"VERDICT: beats MDY after haircut -> {'PASS' if beats else 'FAIL'}")
```

Also print the SPY cross-reference run for context.

- [ ] **Step 2: Run it**

Run: `source .venv/bin/activate && python scripts/midcap_research.py 2>&1 | tee midcap_research.log`
Expected: prints raw + haircut-adjusted midcap metrics, MDY/SPY benchmarks, and the verdict. If coverage is poor (<~85% of members have data), flag it — the result is unreliable and that is itself the finding.

- [ ] **Step 3: Record the verdict in the roadmap**

In `docs/roadmap.md`, update research direction G with the numbers and verdict (raw, haircut-adjusted, vs MDY/SPY, coverage, and the two-way-churn caveat that the haircut is a conservative bound).

- [ ] **Step 4: Commit**

```bash
source .venv/bin/activate && ruff check scripts/midcap_research.py
git add scripts/midcap_research.py docs/roadmap.md
git commit -m "feat(lab): midcap400 WFO vs MDY with calibrated bias haircut + verdict"
```

---

## Self-Review

**Spec coverage:**
- Component 1 (snapshot + wiring) → Task 1. ✅
- Component 2 (OHLCV backfill + MDY) → Task 2. ✅
- Component 3 (bias calibration, SP500 PIT vs snapshot) → Task 3. ✅
- Component 4 (midcap run vs MDY + haircut + verdict) → Task 4. ✅
- Coverage reporting (`coverage_stats` / requested-vs-have) → Task 2 Step 5 + Task 4 Step 2. ✅
- Benchmark = MDY, SPY cross-reference → Task 4. ✅

**Placeholder scan:** No TBD/TODO; all code steps show real code; the snapshot date `2026-06-24` is concrete. ✅

**Type consistency:** `snapshot_members`/`resolve_symbols`/`parse_table` outputs are used consistently; `parse_table` keys (`oos_cagr`, `oos_sharpe`, `spy_cagr`, `spy_sharpe`) match `scripts/ablation_voters.py`'s parser across Tasks 3-4. ✅

**Note:** Task 4 consumes the Δ from Task 3 — record the two Δ numbers when Task 3 runs and substitute them as constants (or read `bias_calibration.log`) in Task 4; they are real values, not placeholders, once Task 3 has run.
