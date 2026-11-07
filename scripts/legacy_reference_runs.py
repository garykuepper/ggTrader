"""Legacy-engine reference runs on the frozen OHLCV tape (vectorbt-engine programme, P0).

Runs the code at git tag ``legacy-engine-vbt0285`` (extract it with
``scripts/legacy_env.sh``) under the legacy interpreter (vectorbt 0.28.5, pandas 2.3.3), reading
bars only from the hashed Parquet snapshot in ``data/snapshots/``. Each run writes one JSON file
into ``research/programme/vectorbt-engine/legacy-references/``; a run whose JSON already exists
with ``"ok": true`` is skipped (resume) unless ``--force``.

Database and network access are blocked for the whole process: ``psycopg2.connect`` and the
yfinance loader raise, so a stray read fails loudly instead of fetching or writing. The one
strategy that normally reads a table (retail_attention) gets an empty interest frame, which is
what the database holds (the table is empty).

Example:
    legacy-engine/venv/bin/python scripts/legacy_reference_runs.py \\
        --legacy-src legacy-engine/src/src --runs smoke2_ensemble_sp500
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
# Scratch by default; the committed references live in
# research/programme/vectorbt-engine/legacy-references and are rewritten only by --out-dir.
OUT_DIR = Path(tempfile.gettempdir()) / "ggtrader_evidence" / "legacy-references"
SNAPSHOT_DIR = REPO_ROOT / "data" / "snapshots"
LEGACY_TAG = "legacy-engine-vbt0285"

EVAL_START = "2021-01-31"
EVAL_END_17 = "2026-04-30"  # exactly 17 folds (anchor-fix reproduction script)
EVAL_END_SMOKE = "2022-07-31"  # 2 folds
EVAL_END_ONE_FOLD = "2022-04-30"  # 1 fold (fallback for slow strategies)
UNIVERSE = "sp500"

#: Strategies ported in P4 (spec Appendix D Q8) other than the ensemble itself.
STRATEGY_RUNS = (
    "conviction_bb",
    "cross_asset_sleeve",
    "retail_attention",
    "xs_momentum",
    "bb_reversion",
    "rsi_reversion",
    "ema_cross",
    "macd_divergence",
    "volume_bb_reversion",
)
GOLDEN_SYMBOLS = 150
VOTERS = ("bb", "rsi", "ema", "macd", "vbb")


# --------------------------------------------------------------------------- bootstrap


def bootstrap_legacy(legacy_src: Path) -> None:
    """Put the extracted tag first on sys.path and prove ggTrader comes from there."""
    sys.path.insert(0, str(legacy_src))
    import ggTrader  # noqa: PLC0415

    where = Path(ggTrader.__file__).resolve()
    if legacy_src.resolve() not in where.parents:
        raise RuntimeError(f"ggTrader imported from {where}, not from {legacy_src}")


def block_io() -> None:
    """Make any database or network access raise (the tape is frozen and the DB is not read)."""
    import psycopg2  # noqa: PLC0415

    def _no_db(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("database access is blocked in frozen-tape reference runs")

    psycopg2.connect = _no_db  # type: ignore[assignment]
    from ggTrader.data.live.yfinance_loader import YFinanceDataLoader  # noqa: PLC0415

    def _no_net(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("network fetch is blocked in frozen-tape reference runs")

    YFinanceDataLoader.fetch_ohlcv = _no_net  # type: ignore[method-assign]


def import_snapshot_helpers() -> Any:
    """Import scripts/make_tape_snapshot.py (shared snapshot reader) by path."""
    spec = importlib.util.spec_from_file_location(
        "make_tape_snapshot", SCRIPTS_DIR / "make_tape_snapshot.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["make_tape_snapshot"] = mod
    spec.loader.exec_module(mod)
    return mod


def pick_snapshot(arg: Optional[str]) -> tuple[Path, dict[str, Any]]:
    """Resolve the snapshot file and its manifest entry; refuse a hash mismatch."""
    manifest = json.loads((SNAPSHOT_DIR / "MANIFEST.json").read_text())
    entries = sorted(manifest["snapshots"], key=lambda e: e["created"])
    if arg:
        entry = next(e for e in entries if e["path"].endswith(Path(arg).name))
    else:
        entry = entries[-1]
    path = REPO_ROOT / entry["path"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != entry["sha256"]:
        raise RuntimeError(f"snapshot hash mismatch for {path}: {digest} != {entry['sha256']}")
    return path, entry


# --------------------------------------------------------------------------- helpers


def jsonable(obj: Any) -> Any:
    """Recursively convert numpy / pandas scalars to plain JSON types."""
    import numpy as np  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return val if np.isfinite(val) else None
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    return obj


def versions() -> dict[str, str]:
    """Interpreter and key library versions."""
    import numba  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415
    import vectorbt  # noqa: PLC0415

    return {
        "python": sys.version.split()[0],
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "vectorbt": vectorbt.__version__,
        "numba": numba.__version__,
    }


def rss_mb() -> tuple[float, float]:
    """Current and peak resident set size of this process in MB (Linux /proc)."""
    cur = peak = 0.0
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                cur = int(line.split()[1]) / 1024
            elif line.startswith("VmHWM:"):
                peak = int(line.split()[1]) / 1024
    return cur, peak


def install_memory_hygiene(trace: list[dict[str, float]]) -> None:
    """Collect garbage and return freed heap to the OS after each fold step; log RSS.

    Numerically inert (no value is touched). The legacy walk-forward holds many vectorbt
    portfolio objects whose reference cycles keep large arrays alive until a collection.
    """
    import ctypes  # noqa: PLC0415
    import gc  # noqa: PLC0415

    import ggTrader.lab.wfo as wfo_mod  # noqa: PLC0415

    try:
        trim = ctypes.CDLL("libc.so.6").malloc_trim
    except OSError:  # pragma: no cover - non-glibc
        trim = None

    def wrap(name: str) -> None:
        inner = getattr(wfo_mod, name)

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            out = inner(*args, **kwargs)
            gc.collect()
            if trim is not None:
                trim(0)
            cur, peak = rss_mb()
            trace.append({"after": name, "rss_mb": round(cur), "peak_mb": round(peak)})
            return out

        setattr(wfo_mod, name, wrapped)

    wrap("compute_anchor_set")
    wrap("_sweep_fold_dispatch")


def legacy_commit() -> str:
    """Commit the legacy tag points at."""
    out = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", f"{LEGACY_TAG}^{{commit}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return out.stdout.strip() or "unknown"


def stats_block(curve: Any) -> dict[str, Any]:
    """Sharpe / CAGR / max drawdown (and extras) of an equity curve via the legacy curve_stats."""
    from ggTrader.lab.metrics import curve_stats  # noqa: PLC0415

    return jsonable(curve_stats(curve)) if len(curve.dropna()) > 1 else {}


def fold_table(fold_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-fold record (dates, winner, gates, anchor use, WFE, in- and out-of-sample Sharpe)."""
    keys = (
        "fold_num",
        "train_start",
        "train_end",
        "test_start",
        "test_end",
        "winner_combo",
        "winner_params",
        "train_score",
        "oos_score",
        "ndh_passed",
        "dsr_passed",
        "gates_passed",
        "ndh_density",
        "dsr_value",
        "wfe",
        "is_sharpe",
        "oos_sharpe",
        "halted",
        "halted_entering",
        "used_anchor",
    )
    return [jsonable({k: r.get(k) for k in keys}) for r in fold_results]


# --------------------------------------------------------------------------- run specs

RUN_SPECS: dict[str, dict[str, Any]] = {
    "smoke2_ensemble_sp500": {
        "kind": "wfo",
        "strategy": "ensemble",
        "universe": UNIVERSE,
        "eval_start": EVAL_START,
        "eval_end": EVAL_END_SMOKE,
        "note": "2-fold smoke profile, full 48-combo grid (matches the P0 prof_wfo.py profile)",
    },
    "core17_ensemble_sp500": {
        "kind": "wfo",
        "strategy": "ensemble",
        "universe": UNIVERSE,
        "eval_start": EVAL_START,
        "eval_end": EVAL_END_17,
        "note": "17-fold SP500 core, STOCK_BASE_CONFIG, leverage 1.0 (no overlay), PIT universe",
    },
    "spy_bh_smoke_window": {
        "kind": "spy",
        "eval_start": EVAL_START,
        "eval_end": EVAL_END_SMOKE,
        "oos_ref": "smoke2_ensemble_sp500",
    },
    "spy_bh_17fold_window": {
        "kind": "spy",
        "eval_start": EVAL_START,
        "eval_end": EVAL_END_17,
        "oos_ref": "core17_ensemble_sp500",
    },
    "golden_ensemble_signals": {"kind": "golden"},
    "golden_live_signals": {
        "kind": "live_golden",
        "universe": UNIVERSE,
        "sessions": 120,
        "lookback_days": 365,
    },
}
for _name in STRATEGY_RUNS:
    RUN_SPECS[f"ref_{_name}"] = {
        "kind": "wfo",
        "strategy": _name,
        "universe": "cross_asset" if _name == "cross_asset_sleeve" else UNIVERSE,
        "eval_start": EVAL_START,
        "eval_end": EVAL_END_SMOKE,
        "note": "one pinned 2-fold window (smallest multi-fold honest window)",
    }


RUN_SPECS["core17_ensemble_sp500_dsr0"] = {
    "kind": "wfo",
    "strategy": "ensemble",
    "universe": UNIVERSE,
    "eval_start": EVAL_START,
    "eval_end": EVAL_END_17,
    "run_wfo_kwargs": {"dsr_threshold": 0.0},
    "note": (
        "core17 with the DSR threshold set to 0 so the DSR gate always passes: the default run "
        "deploys the anchor in 17/17 folds, so this variant is the reference that exercises the "
        "winner-deployment path (diagnostic, not a production configuration)"
    ),
}

RUN_SPECS["ref_cross_asset_sleeve_17fold"] = {
    "kind": "wfo",
    "strategy": "cross_asset_sleeve",
    "universe": "cross_asset",
    "eval_start": EVAL_START,
    "eval_end": EVAL_END_17,
    "note": "cheap weight strategy, so the full pinned 17-fold window is also recorded",
}


# --------------------------------------------------------------------------- run bodies


class Tape:
    """Loads frames from the frozen snapshot with the legacy ``load_ohlcv`` semantics."""

    def __init__(self, path: Path, entry: dict[str, Any], helpers: Any) -> None:
        self.path, self.entry, self.helpers = path, entry, helpers

    def load(self, symbols: list[str], start: str, end: Optional[str]) -> Any:
        """Equivalent of ``lab.data.load_ohlcv`` (tz-aware UTC index, (symbol, field) columns)."""
        return self.helpers.load_snapshot_wide(
            self.path, symbols, start, end, session_tz=self.entry["db_session_timezone"]
        )


def _common_header(name: str, spec: dict[str, Any], tape: Tape) -> dict[str, Any]:
    return {
        "run": name,
        "engine": "legacy",
        "legacy_tag": LEGACY_TAG,
        "legacy_commit": legacy_commit(),
        "spec": spec,
        "versions": versions(),
        "tape": {
            "path": tape.entry["path"],
            "sha256": tape.entry["sha256"],
            "content_sha256": tape.entry["content_sha256"],
            "rows": tape.entry["rows"],
            "symbols": tape.entry["symbols"],
        },
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def run_wfo_case(name: str, spec: dict[str, Any], tape: Tape, out_dir: Path) -> dict[str, Any]:
    """Walk-forward run for one strategy, mirroring ``run_core`` of the anchor-fix driver."""
    import joblib  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415
    from ggTrader.lab.strategies.cross_asset_sleeve import (  # noqa: PLC0415
        CROSS_ASSET_SLEEVE_UNIVERSE,
    )
    from ggTrader.lab.sweep import build_grid  # noqa: PLC0415
    from ggTrader.lab.wfo import run_wfo  # noqa: PLC0415

    from ggTrader.lab.data import (  # noqa: PLC0415
        STOCK_BASE_CONFIG,
        eligible_at,
        equity_universe_between,
    )
    from ggTrader.lab.strategies import STRATEGY_REGISTRY  # noqa: PLC0415
    from ggTrader.lab.strategy import LabConfig  # noqa: PLC0415

    t0 = time.time()
    result = _common_header(name, spec, tape)
    cfg = LabConfig()
    es = pd.Timestamp(spec["eval_start"], tz="UTC")
    ee = pd.Timestamp(spec["eval_end"], tz="UTC")
    warmup_days = int(max(cfg.lookback, cfg.min_history_bars) * 1.6) + 60
    data_start = str((es - pd.Timedelta(days=warmup_days)).date())

    if spec["universe"] == "cross_asset":
        universe = list(CROSS_ASSET_SLEEVE_UNIVERSE)
        universe_fn: Callable[..., Any] = lambda asof, past: list(CROSS_ASSET_SLEEVE_UNIVERSE)  # noqa: E731
    else:
        universe = equity_universe_between(es, ee, universe=spec["universe"])
        uni = spec["universe"]
        universe_fn = lambda asof, past: eligible_at(asof, past, cfg, universe=uni)[0]  # noqa: E731
    t_load = time.time()
    ohlcv = tape.load(sorted(set(universe) | {"SPY"}), data_start, spec["eval_end"])
    load_s = time.time() - t_load
    spy_close = ohlcv["SPY"]["close"].dropna()
    cols = [s for s in ohlcv.columns.get_level_values(0).unique() if s != "SPY"]
    if spec["universe"] == "cross_asset":
        cols = list(CROSS_ASSET_SLEEVE_UNIVERSE)
    ohlcv = ohlcv[cols]

    if spec["strategy"] == "retail_attention":
        # The strategy normally reads google_trends_interest from the DB (with a CREATE TABLE
        # IF NOT EXISTS, which a read-only session refuses). The table holds 0 rows (checked
        # read-only on 2026-11-06), so the faithful frozen input is an empty frame.
        import ggTrader.lab.strategies.retail_attention as ra  # noqa: PLC0415

        ra.load_search_interest = lambda symbols, start, end: pd.DataFrame(
            columns=["symbol", "date", "search_interest"]
        )
        result["input_note"] = "google_trends_interest empty: strategy never selects"
    cls = STRATEGY_REGISTRY[spec["strategy"]]
    grid = build_grid(cls)
    result["config"] = {
        "base_config": dict(STOCK_BASE_CONFIG),
        "lab_config": jsonable(vars(cfg)),
        "grid_size": len(grid),
        "grid": jsonable(grid),
        "universe_requested": len(universe),
        "universe_loaded": len(cols),
        "data_start": data_start,
        "max_leverage": 1.0,
        "wfo_policy": "WfoPolicy() defaults",
        "memory_hygiene": "gc.collect + malloc_trim after anchor and sweep steps (inert)",
    }
    print(f"[{name}] {spec['strategy']} grid={len(grid)} symbols={len(cols)} load={load_s:.1f}s")
    mem_trace: list[dict[str, float]] = []
    install_memory_hygiene(mem_trace)
    # Threads (not loky worker processes) so the in-process patches above reach the workers.
    with joblib.parallel_backend("threading" if spec["strategy"] == "retail_attention" else "loky"):
        wfo = run_wfo(
            spec["strategy"],
            cls,
            cfg,
            ohlcv,
            spy_close,
            eval_start=spec["eval_start"],
            eval_end=spec["eval_end"],
            market="stock",
            base_config=dict(STOCK_BASE_CONFIG),
            grid=grid,
            universe_fn=universe_fn,
            **spec.get("run_wfo_kwargs", {}),
        )
    if isinstance(wfo, str):
        raise RuntimeError(f"run_wfo returned no result: {wfo}")
    rows = fold_table(wfo.fold_results)
    oos = wfo.oos_equity
    spy_common = spy_close.reindex(oos.index).ffill().dropna()
    spy_stats = stats_block(spy_common * (100000.0 / spy_common.iloc[0])) if len(spy_common) else {}
    n_pass = sum(1 for r in rows if r["gates_passed"])
    result.update(
        ok=True,
        oos=stats_block(oos),
        spy_oos_index=spy_stats,
        n_folds=len(rows),
        gates_passed=n_pass,
        gates=f"{n_pass}/{len(rows)}",
        anchor_folds=sum(1 for r in rows if r["used_anchor"]),
        winners=[r["winner_combo"] for r in rows],
        fold_table=rows,
        live_params=jsonable(wfo.live_params),
        skipped_folds=jsonable(list(wfo.skipped_folds)),
        table=wfo.table,
        oos_first=str(oos.index[0]) if len(oos) else None,
        oos_last=str(oos.index[-1]) if len(oos) else None,
        oos_bars=int(len(oos)),
        load_seconds=round(load_s, 1),
        wall_seconds=round(time.time() - t0, 1),
        peak_rss_mb=round(rss_mb()[1]),
        rss_trace=mem_trace,
    )
    oos.rename("oos_equity").to_csv(out_dir / f"{name}_oos_equity.csv", header=True)
    return result


def run_spy_case(name: str, spec: dict[str, Any], tape: Tape, out_dir: Path) -> dict[str, Any]:
    """SPY buy-and-hold over the eval window, and over the OOS index of the matching WFO run."""
    import pandas as pd  # noqa: PLC0415

    t0 = time.time()
    result = _common_header(name, spec, tape)
    spy = tape.load(["SPY"], "2019-03-01", spec["eval_end"])["SPY"]["close"].dropna()
    window = spy[(spy.index >= pd.Timestamp(spec["eval_start"], tz="UTC"))]
    result["eval_window"] = {
        "first": str(window.index[0]),
        "last": str(window.index[-1]),
        "bars": int(len(window)),
        "stats": stats_block(window * (100000.0 / window.iloc[0])),
    }
    ref = out_dir / f"{spec['oos_ref']}_oos_equity.csv"
    if ref.exists():
        idx = pd.read_csv(ref, index_col=0, parse_dates=True).index
        idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
        common = spy.reindex(idx).ffill().dropna()
        result["oos_index_of"] = {
            "run": spec["oos_ref"],
            "first": str(common.index[0]),
            "last": str(common.index[-1]),
            "stats": stats_block(common * (100000.0 / common.iloc[0])),
        }
    else:
        result["oos_index_of"] = {"run": spec["oos_ref"], "missing": True}
    result.update(ok=True, wall_seconds=round(time.time() - t0, 1))
    return result


def run_golden_case(name: str, spec: dict[str, Any], tape: Tape, out_dir: Path) -> dict[str, Any]:
    """Boolean entry/exit frames for the five voters and the ensemble (first 150 symbols)."""
    import numpy as np  # noqa: PLC0415
    from ggTrader.lab.strategies.ensemble import EnsembleSignal  # noqa: PLC0415
    from ggTrader.lab.strategies.indicators import (  # noqa: PLC0415
        bb_signals,
        ema_signals,
        macd_signals,
        rsi_signals,
        volume_bb_signals,
    )

    from ggTrader.lab.strategy import LabConfig  # noqa: PLC0415

    t0 = time.time()
    result = _common_header(name, spec, tape)
    wide = tape.load(None, tape.entry["date_min"], None)
    symbols = sorted(wide.columns.get_level_values(0).unique())[:GOLDEN_SYMBOLS]
    close = wide.xs("close", axis=1, level="field")[symbols]
    volume = wide.xs("volume", axis=1, level="field")[symbols]
    p = {"bb": (20, 2.0), "rsi": (14, 30, 50), "ema": (20, 50), "macd": (12, 26, 9, 20)}
    p["vbb"] = (20, 2.0, 20, 2.0)
    frames: dict[str, Any] = {}
    frames["bb"] = bb_signals(close, *p["bb"])
    frames["rsi"] = rsi_signals(close, *p["rsi"])
    frames["ema"] = ema_signals(close, *p["ema"])
    frames["macd"] = macd_signals(close, *p["macd"])
    frames["vbb"] = volume_bb_signals(close, volume, *p["vbb"])
    ens = EnsembleSignal(LabConfig())._generate_signals(close, volume)
    frames["ensemble"] = (ens.entries, ens.exits)

    arrays: dict[str, Any] = {
        "index_ns": close.index.tz_convert(None).asi8,
        "columns": np.array(symbols),
    }
    digests: dict[str, str] = {}
    counts: dict[str, dict[str, int]] = {}
    for key, (ent, ext) in frames.items():
        for side, frame in (("entries", ent), ("exits", ext)):
            arr = np.asarray(frame.reindex(index=close.index, columns=symbols)).astype(bool)
            arrays[f"{key}_{side}"] = arr
            digests[f"{key}_{side}"] = hashlib.sha256(np.packbits(arr).tobytes()).hexdigest()
            counts.setdefault(key, {})[side] = int(arr.sum())
    npz = out_dir / "signal_golden_legacy.npz"
    np.savez_compressed(npz, **arrays)
    result.update(
        ok=True,
        params=p,
        defaults_note="EnsembleSignal(LabConfig()) defaults: min_agree=2, exit=2, 5 voters",
        symbols=symbols,
        bars=int(len(close)),
        first=str(close.index[0]),
        last=str(close.index[-1]),
        signal_counts=counts,
        array_sha256=digests,
        npz=str(npz.relative_to(REPO_ROOT)) if npz.is_relative_to(REPO_ROOT) else str(npz),
        npz_sha256=hashlib.sha256(npz.read_bytes()).hexdigest(),
        wall_seconds=round(time.time() - t0, 1),
    )
    return result


def run_live_golden_case(
    name: str, spec: dict[str, Any], tape: Tape, out_dir: Path
) -> dict[str, Any]:
    """Live golden (spec 7.3 item 4): what ``signal_runner.generate_signals`` returns per session.

    For each of the last ``sessions`` bars: PIT SP500 members of that day, bars from 365 calendar
    days earlier up to and including that day, ``LabConfig(min_history_bars=60)``, default
    ``EnsembleSignal``; ``buys``/``sells`` on the last bar and ``last_exit`` from the whole
    window, exactly as the live code derives them (the ML gate is off by default and skipped).
    """
    import pandas as pd  # noqa: PLC0415
    from ggTrader.lab.strategies.ensemble import EnsembleSignal  # noqa: PLC0415

    from ggTrader.data.core.index_constituents import (  # noqa: PLC0415
        normalize_yf_ticker,
        universe_members_asof,
    )
    from ggTrader.lab.strategy import LabConfig  # noqa: PLC0415

    t0 = time.time()
    result = _common_header(name, spec, tape)
    wide = tape.load(None, tape.entry["date_min"], None)
    sessions = list(wide["SPY"]["close"].dropna().index[-spec["sessions"] :])
    cfg = LabConfig(min_history_bars=60)
    records: list[dict[str, Any]] = []
    for day in sessions:
        members = universe_members_asof(spec["universe"], day)
        symbols = sorted({normalize_yf_ticker(t) for t in members})
        have = set(wide.columns.get_level_values(0))
        cols = [c for c in wide.columns if c[0] in set(symbols) & have]
        start = day - pd.Timedelta(days=spec["lookback_days"])
        window = wide.loc[(wide.index >= start) & (wide.index <= day), cols]
        sym_cols = list(window.columns.get_level_values(0).unique())
        close = pd.concat({sym: window[sym]["close"] for sym in sym_cols}, axis=1)
        last_bar = close.index[-1]
        ensemble = EnsembleSignal(cfg)
        plan = ensemble.select(last_bar, window, sym_cols)
        buys: list[str] = []
        sells: list[str] = []
        last_exit: dict[str, str] = {}
        if plan:
            targets = ensemble.to_targets({last_bar: plan}, window)
            last_entries = targets.entries.loc[last_bar]
            last_exits = targets.exits.loc[last_bar]
            buys = sorted(last_entries[last_entries].index.tolist())
            sells = sorted(last_exits[last_exits].index.tolist())
            exits = targets.exits
            latest = exits.iloc[::-1].idxmax()[exits.any()]
            last_exit = {sym: str(ts.date()) for sym, ts in latest.items()}
        n_short = sum(1 for sym in sym_cols if 60 <= window[sym]["close"].count() < 200)
        records.append(
            {
                "as_of": str(last_bar.date()),
                "session": str(day.date()),
                "universe_members": len(symbols),
                "universe_size": len(sym_cols),
                "symbols_60_to_199_bars": n_short,
                "buys": buys,
                "sells": sells,
                "last_exit": last_exit,
            }
        )
    payload = json.dumps(jsonable(records), sort_keys=True, separators=(",", ":"))
    out = out_dir / "live_golden_legacy.json"
    out.write_text(payload + "\n")
    result.update(
        ok=True,
        sessions=len(records),
        first=records[0]["as_of"],
        last=records[-1]["as_of"],
        total_buys=sum(len(r["buys"]) for r in records),
        total_sells=sum(len(r["sells"]) for r in records),
        sessions_with_last_bar_mismatch=sum(r["as_of"] != r["session"] for r in records),
        short_history_symbol_sessions=sum(r["symbols_60_to_199_bars"] for r in records),
        file=_rel_repo(out),
        file_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
        wall_seconds=round(time.time() - t0, 1),
    )
    return result


def _rel_repo(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)


KIND_RUNNERS: dict[str, Callable[..., dict[str, Any]]] = {
    "live_golden": run_live_golden_case,
    "wfo": run_wfo_case,
    "spy": run_spy_case,
    "golden": run_golden_case,
}


# --------------------------------------------------------------------------- main


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse the command line."""
    p = argparse.ArgumentParser(
        description="Legacy-engine reference runs on the frozen OHLCV snapshot (resumable).",
    )
    p.add_argument("--legacy-src", help="extracted tag's source root (contains ggTrader/)")
    p.add_argument("--snapshot", help="snapshot parquet (default: newest in MANIFEST.json)")
    p.add_argument("--runs", default="all", help="comma list of run names, or 'all'")
    p.add_argument("--out-dir", default=str(OUT_DIR), help="JSON output directory")
    p.add_argument("--force", action="store_true", help="redo runs that already have ok JSON")
    p.add_argument("--list", action="store_true", help="list run names and exit")
    p.add_argument("--eval-end", help="override eval_end of the selected wfo runs (for fallbacks)")
    p.add_argument("--suffix", default="", help="suffix appended to the run name when overriding")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Run the selected reference runs sequentially; failures are recorded, not raised."""
    args = parse_args(argv)
    if args.list:
        for key, spec in RUN_SPECS.items():
            print(f"{key:28s} {spec['kind']}")
        return 0
    if not args.legacy_src:
        print("--legacy-src is required", file=sys.stderr)
        return 2
    names = list(RUN_SPECS) if args.runs == "all" else [r for r in args.runs.split(",") if r]
    unknown = [n for n in names if n not in RUN_SPECS]
    if unknown:
        print(f"unknown runs: {unknown}", file=sys.stderr)
        return 2

    bootstrap_legacy(Path(args.legacy_src))
    block_io()
    helpers = import_snapshot_helpers()
    path, entry = pick_snapshot(args.snapshot)
    tape = Tape(path, entry, helpers)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    failures = 0
    for name in names:
        spec = dict(RUN_SPECS[name])
        out_name = name
        if args.eval_end and spec["kind"] == "wfo":
            spec["eval_end"] = args.eval_end
            out_name = name + (args.suffix or f"_to{args.eval_end}")
        target = out_dir / f"_legacy_reference_{out_name}.json"
        if target.exists() and not args.force:
            if json.loads(target.read_text()).get("ok"):
                print(f"[skip] {out_name}: ok JSON exists")
                continue
        t0 = time.time()
        try:
            result = KIND_RUNNERS[spec["kind"]](out_name, spec, tape, out_dir)
        except Exception as exc:  # noqa: BLE001 - recorded in the JSON, run continues
            failures += 1
            result = _common_header(out_name, spec, tape)
            result.update(
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(),
                wall_seconds=round(time.time() - t0, 1),
            )
            print(f"[FAIL] {out_name}: {result['error']}", flush=True)
        target.write_text(json.dumps(jsonable(result), indent=2) + "\n")
        print(f"[done] {out_name} ok={result.get('ok')} {result['wall_seconds']}s -> {target}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
