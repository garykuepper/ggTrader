"""P1 parity driver: old stack vs new stack on real cached data, bit-for-bit.

Three sub-commands (all READ-ONLY; nothing here writes to the database):

    extract  Pull the 2-fold ensemble profile's OHLCV from TimescaleDB with a read-only
             session into a flat parquet file, plus wide close panels for the pct_change probes.
             Needs DB_* environment variables (use a wrapper that exports only those).
    run      Run the profile from the extract (no database access at all: pandas.read_sql and the
             loader's connection are replaced) on the interpreter that executes this script.
             Writes per-combo equity curves, per-fold WFO results, strategy and live-path probe
             outputs and a pct_change call-site probe into the output directory.
    compare  Diff two run directories exactly (np.array_equal and max abs difference).

Run it with `--src PATH` to import ggTrader from a different source tree (the original
HEAD export for the legacy side), and `--engine {numba,rust}` on the vectorbt 1.x stack.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import resource
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EVAL_START = "2021-01-31"
EVAL_END = "2022-07-31"
N_SYMBOLS = 150
N_GAP_SYMBOLS = 12
PROBE_WINDOWS = {"2021_22": ("2021-01-31", "2022-07-31"), "2025_26": ("2025-01-02", "2026-10-30")}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------------------
# extract
# --------------------------------------------------------------------------------------
def cmd_extract(a: argparse.Namespace) -> None:
    import psycopg2

    from ggTrader.lab.data import equity_universe_between
    from ggTrader.lab.strategy import LabConfig
    from ggTrader.utils.config import get_db_connection_string

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    es, ee = pd.Timestamp(EVAL_START, tz="UTC"), pd.Timestamp(EVAL_END, tz="UTC")
    cfg = LabConfig()
    warm = int(max(cfg.lookback, cfg.min_history_bars) * 1.6) + 60
    start = (es - pd.Timedelta(days=warm)).date()
    dsn = get_db_connection_string().replace("postgresql+psycopg2://", "postgresql://")
    universe = equity_universe_between(es, ee, universe="sp500")

    def query(symbols: list[str], d0: str, d1: str) -> pd.DataFrame:
        q = (
            "SELECT timestamp, symbol, open, high, low, close, volume FROM ohlcv "
            "WHERE symbol = ANY(%s) AND interval = '1d' AND venue = 'yfinance' "
            "AND timestamp >= %s AND timestamp <= %s ORDER BY timestamp ASC"
        )
        with psycopg2.connect(dsn, options="-c default_transaction_read_only=on") as conn:
            return pd.read_sql(q, conn, params=[symbols, d0, d1], parse_dates=["timestamp"])

    with psycopg2.connect(dsn, options="-c default_transaction_read_only=on") as conn:
        with conn.cursor() as cur:
            cur.execute("SHOW transaction_read_only")
            assert cur.fetchone()[0] == "on", "session is not read-only"
            cur.execute(
                "SELECT symbol FROM ohlcv WHERE venue='yfinance' AND interval='1d' "
                "AND timestamp >= %s AND timestamp < %s GROUP BY symbol HAVING count(*) > 700",
                (str(start), EVAL_END),
            )
            have = {r[0] for r in cur.fetchall()}
    cand = sorted(u for u in universe if u in have and u != "SPY")
    flat_all = query(cand + ["SPY"], str(start), EVAL_END)
    close = flat_all.pivot(index="timestamp", columns="symbol", values="close")
    in_win = close.loc[close.index >= str(es.date())]
    # symbols with interior or trailing NaN gaps inside the evaluation window
    gappy = sorted(
        s
        for s in in_win.columns
        if s != "SPY" and (in_win[s].isna() & in_win[s].notna().cummax()).any()
    )
    step = max(1, len(cand) // (N_SYMBOLS - N_GAP_SYMBOLS))
    spread = cand[::step]
    chosen = list(dict.fromkeys(gappy[:N_GAP_SYMBOLS] + spread))[:N_SYMBOLS]
    chosen = sorted(chosen)
    flat = flat_all[flat_all["symbol"].isin(chosen + ["SPY"])].reset_index(drop=True)
    flat.to_parquet(out / "flat.parquet")
    (out / "symbols.json").write_text(
        json.dumps(
            {"symbols": chosen, "gappy_in_window": gappy, "n_candidates": len(cand)}, indent=1
        )
    )
    # wide close panels for the pct_change probes (all candidates, two windows)
    for name, (d0, d1) in PROBE_WINDOWS.items():
        syms = cand
        if name != "2021_22":
            ps = [
                s
                for s in equity_universe_between(
                    pd.Timestamp(d0, tz="UTC"), pd.Timestamp(d1, tz="UTC")
                )
                if s != "SPY"
            ]
            syms = sorted(set(ps))
        f = query(syms, d0, d1)
        f.pivot(index="timestamp", columns="symbol", values="close").to_parquet(
            out / f"probe_close_{name}.parquet"
        )
    meta = {
        "n_flat_rows": int(len(flat)),
        "flat_sha256": _sha(out / "flat.parquet"),
        "n_chosen": len(chosen),
        "n_gappy_in_window": len(gappy),
        "max_timestamp": str(flat["timestamp"].max()),
        "min_timestamp": str(flat["timestamp"].min()),
        "extracted_with_pandas": pd.__version__,
    }
    (out / "extract_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


# --------------------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------------------
class _NoDb:
    """Connection stand-in: any attempt to use it raises, proving the run never hits the DB."""

    def __enter__(self) -> "_NoDb":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def __getattr__(self, name: str) -> Any:
        raise RuntimeError(f"database access attempted via .{name}")


def _install_pct_probe(records: dict[str, collections.Counter]) -> None:
    """Record, per caller line, how many pct_change calls saw non-leading NaN input."""
    from pandas.core.generic import NDFrame

    orig = NDFrame.pct_change

    def probe(self, *args: Any, **kwargs: Any):  # noqa: ANN202
        fr = sys._getframe(1)
        site = f"{Path(fr.f_code.co_filename).name}:{fr.f_lineno}"
        try:
            na = self.isna().to_numpy()
            seen = np.logical_or.accumulate(~na, axis=0)
            nonleading = bool((na & seen).any())
            records["calls"][site] += 1
            if nonleading:
                records["calls_with_nonleading_nan"][site] += 1
            kwargs_fill = "fill_method" in kwargs
            records["explicit_fill_method"][site] += int(kwargs_fill)
        except Exception:  # noqa: BLE001 - the probe must never alter behaviour
            pass
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return orig(self, *args, **kwargs)

    NDFrame.pct_change = probe  # type: ignore[method-assign]


def _clean(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if o != o else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, pd.Timestamp):
        return str(o)
    return o if isinstance(o, (int, str, bool, type(None))) else str(o)


def _ns_index(index: pd.Index) -> np.ndarray:
    """Index as int64 nanoseconds since the epoch, whatever resolution the stack uses."""
    di = pd.DatetimeIndex(index)
    di = di.tz_localize("UTC") if di.tz is None else di.tz_convert("UTC")
    return di.as_unit("ns").asi8


def _save_curves(path: Path, curves: dict[str, pd.Series]) -> None:
    keys = sorted(curves)
    arrays = {f"v{i}": np.asarray(curves[k], dtype="float64") for i, k in enumerate(keys)}
    idx = {f"i{i}": _ns_index(curves[k].index) for i, k in enumerate(keys)}
    np.savez(path, keys=np.array(keys), **arrays, **idx)


def cmd_run(a: argparse.Namespace) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ext = Path(a.extract)
    warnings.simplefilter("ignore")
    if a.src:
        sys.path.insert(0, a.src)
    probe_records: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    _install_pct_probe(probe_records)

    summary: dict[str, Any] = {
        "tag": a.tag,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "src": a.src or "(installed)",
        "engine_requested": a.engine,
    }
    import numba
    import vectorbt as vbt

    summary["numba"] = numba.__version__
    summary["vectorbt"] = vbt.__version__
    if a.engine != "default":
        vbt.settings["engine"] = a.engine
    try:
        from vectorbt import _engine

        summary["rust_available"] = bool(_engine.is_rust_available())
    except Exception:  # noqa: BLE001
        summary["rust_available"] = None
    summary["vbt_engine_setting"] = (
        str(vbt.settings["engine"]) if "engine" in vbt.settings else None
    )

    import ggTrader
    from ggTrader.data.live import cached_yfinance_loader as cyl
    from ggTrader.data.live import yfinance_loader as yfl

    summary["ggtrader_file"] = str(Path(ggTrader.__file__).resolve())
    flat = pd.read_parquet(ext / "flat.parquet")
    # the DB driver hands back naive UTC timestamps; keep that contract
    flat["timestamp"] = pd.DatetimeIndex(flat["timestamp"]).tz_localize(None)

    def fake_read_sql(query: str, conn: Any, params: list | None = None, **kw: Any) -> pd.DataFrame:
        syms, _interval, _venue, *rest = params
        df = flat[flat["symbol"].isin(list(syms))]
        i = 0
        if "timestamp >=" in query:
            lo = pd.Timestamp(rest[i]).tz_convert("UTC").tz_localize(None)
            df = df[df["timestamp"] >= lo]
            i += 1
        if "timestamp <=" in query:
            hi = pd.Timestamp(rest[i]).tz_convert("UTC").tz_localize(None)
            df = df[df["timestamp"] <= hi]
        return df.sort_values("timestamp", kind="stable").reset_index(drop=True).copy()

    def no_db_init(self: Any, connection_string: str | None = None) -> None:
        yfl.YFinanceDataLoader.__init__(self)
        self.connection_string = "postgresql://unused"

    pd.read_sql = fake_read_sql  # type: ignore[assignment]
    cyl.CachedYFinanceLoader.__init__ = no_db_init  # type: ignore[method-assign]
    cyl.CachedYFinanceLoader._connect = lambda self: _NoDb()  # type: ignore[method-assign]
    cyl.CachedYFinanceLoader.fetch_ohlcv = (  # type: ignore[method-assign]
        lambda self, symbols, interval, start_date=None, end_date=None, limit=None: self._db_fetch(
            symbols, interval, start_date, end_date
        )
    )
    cyl.CachedYFinanceLoader._cache_to_db = lambda self, df, interval: None  # type: ignore[method-assign]
    cyl.CachedYFinanceLoader._record_inceptions = lambda self, *x, **k: None  # type: ignore[method-assign]
    yfl.YFinanceDataLoader.fetch_ohlcv = lambda self, *x, **k: pd.DataFrame()  # type: ignore[method-assign]

    from ggTrader.lab.data import (
        STOCK_BASE_CONFIG,
        eligible_at,
        load_ohlcv,
    )
    from ggTrader.lab.strategies import STRATEGY_REGISTRY
    from ggTrader.lab.strategy import LabConfig
    from ggTrader.lab.sweep import build_grid, pit_entry_mask
    from ggTrader.lab.wfo import _sweep_fold, _sweep_fold_weights, generate_folds, run_wfo

    symbols = json.loads((ext / "symbols.json").read_text())["symbols"]
    es, ee = pd.Timestamp(EVAL_START, tz="UTC"), pd.Timestamp(EVAL_END, tz="UTC")
    cfg = LabConfig()
    warm = int(max(cfg.lookback, cfg.min_history_bars) * 1.6) + 60
    t0 = time.time()
    ohlcv = load_ohlcv(symbols + ["SPY"], str((es - pd.Timedelta(days=warm)).date()), EVAL_END)
    summary["load_s"] = time.time() - t0
    summary["ohlcv_index_dtype"] = str(ohlcv.index.dtype)
    spy = ohlcv["SPY"]["close"].dropna()
    ohlcv = ohlcv[[s for s in ohlcv.columns.get_level_values(0).unique() if s != "SPY"]]
    summary["n_symbols"] = len(ohlcv.columns.get_level_values(0).unique())
    summary["rows"] = len(ohlcv)
    uni_fn = lambda asof, past: eligible_at(asof, past, cfg, universe="sp500")[0]  # noqa: E731
    base = dict(STOCK_BASE_CONFIG)
    cls = STRATEGY_REGISTRY["ensemble"]
    grid = build_grid(cls)
    summary["grid"] = len(grid)

    # ---- 1. per-combo signal sweep (48 combos) + weight path (6 combos) --------------------
    t = time.time()
    mask = pit_entry_mask(ohlcv, uni_fn)
    summary["pit_mask_s"] = time.time() - t
    np.save(out / "pit_mask.npy", mask.to_numpy(dtype=bool))
    t = time.time()
    res, eq = _sweep_fold("ensemble", cls(cfg), ohlcv, es, ee, base, grid, entry_mask=mask)
    summary["sweep_signal_s"] = time.time() - t
    (out / "signal_metrics.json").write_text(json.dumps(_clean(res), indent=1))
    _save_curves(out / "signal_equity.npz", dict(eq))
    wcls = STRATEGY_REGISTRY["xs_momentum"]
    wgrid = build_grid(wcls)[:6]
    t = time.time()
    wres, weq = _sweep_fold_weights("xs_momentum", wcls, cfg, ohlcv, es, ee, base, wgrid, uni_fn)
    summary["sweep_weight_s"] = time.time() - t
    (out / "weight_metrics.json").write_text(json.dumps(_clean(wres), indent=1))
    _save_curves(out / "weight_equity.npz", dict(weq))

    # ---- 2. full 2-fold run_wfo -----------------------------------------------------------
    t = time.time()
    wfo = run_wfo(
        "ensemble",
        cls,
        cfg,
        ohlcv,
        spy,
        eval_start=EVAL_START,
        eval_end=EVAL_END,
        market="stock",
        base_config=base,
        grid=grid,
        universe_fn=uni_fn,
    )
    summary["wfo_s"] = time.time() - t
    summary["folds"] = len(generate_folds(es, ee))
    if isinstance(wfo, str):
        summary["wfo_result_str"] = wfo
    else:
        (out / "wfo_fold_results.json").write_text(json.dumps(_clean(wfo.fold_results), indent=1))
        (out / "wfo_live_params.json").write_text(json.dumps(_clean(wfo.live_params), indent=1))
        (out / "wfo_table.txt").write_text(wfo.table)
        _save_curves(out / "wfo_oos_equity.npz", {"oos": wfo.oos_equity})
        summary["wfo_oos_len"] = int(len(wfo.oos_equity))
        summary["wfo_skipped_folds"] = _clean(list(wfo.skipped_folds))

    # ---- 3. pct_change-affected paths on real data ---------------------------------------
    probes: dict[str, Any] = {}
    close_all = pd.concat(
        {s: ohlcv[s]["close"] for s in ohlcv.columns.get_level_values(0).unique()}, axis=1
    )
    # (a) price-panel gaps, two windows, all candidate symbols
    gap_report = {}
    from ggTrader.lab import event_study as es_mod

    for name in PROBE_WINDOWS:
        panel = pd.read_parquet(ext / f"probe_close_{name}.parquet")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if pd.__version__.startswith("2."):
                legacy = panel.pct_change()  # pandas 2 default: pad
            else:
                legacy = panel.ffill().pct_change(fill_method=None)
            bare_fm_none = panel.pct_change(fill_method=None)
        cells_changed = int((legacy.fillna(-9.0) != bare_fm_none.fillna(-9.0)).to_numpy().sum())
        gap_report[name] = {
            "symbols": int(panel.shape[1]),
            "symbols_with_nonleading_gaps": int(
                (panel.isna() & panel.notna().cummax()).any().sum()
            ),
            "cells_where_pad_differs_from_no_fill": cells_changed,
        }
        np.save(out / f"probe_legacy_pct_{name}.npy", legacy.to_numpy(dtype="float64"))
        try:
            from ggTrader.utils.returns import padded_pct_change
        except ImportError:  # original source tree has no helper
            padded_pct_change = None
        if padded_pct_change is not None:
            np.save(
                out / f"probe_padded_pct_{name}.npy",
                padded_pct_change(panel).to_numpy(dtype="float64"),
            )
    probes["gap_report"] = gap_report

    # (b) event study on real closes with deterministic synthetic in-position signals
    entries = pd.DataFrame(False, index=close_all.index, columns=close_all.columns)
    exits = entries.copy()
    for j, col in enumerate(close_all.columns):
        for i in range(30 + j % 20, len(close_all) - 10, 37):
            entries.iloc[i, j] = True
            exits.iloc[i + 5, j] = True
    r = es_mod.run_event_study_from_signals(close_all, entries, exits)
    probes["event_study"] = _clean(r._asdict())

    # (c) idio_vol / max_effect selections at several as-of dates
    from ggTrader.lab.strategies.idio_vol import IdioVolStrategy
    from ggTrader.lab.strategies.max_effect import MaxEffectStrategy

    syms = list(close_all.columns)
    asofs = [
        ohlcv.index[i]
        for i in (len(ohlcv) - 1, len(ohlcv) - 60, len(ohlcv) - 130, len(ohlcv) - 200)
    ]
    sel: dict[str, Any] = {}
    for nm, strat in (
        ("idio_vol", IdioVolStrategy(LabConfig(min_history_bars=100), reg_window=20, quintile=5)),
        ("max_effect", MaxEffectStrategy(LabConfig(min_history_bars=100), window=21, quintile=5)),
    ):
        sel[nm] = {str(d): _clean(strat.select(d, ohlcv, syms)) for d in asofs}
    probes["selections"] = sel

    # (d) live-path: feature_gate features and overlay weights/scale
    from ggTrader.paper.feature_gate import extract_features
    from ggTrader.paper.overlay import compute_weights_and_scale

    feats = {}
    gap_syms = [s for s in syms if (close_all[s].isna() & close_all[s].notna().cummax()).any()][:8]
    for s in list(dict.fromkeys(gap_syms + syms[:4])):
        c = ohlcv[s]["close"]
        v = ohlcv[s]["volume"]
        for d in asofs[:2]:
            feats[f"{s}@{d.date()}"] = _clean(
                extract_features(c, v, d, high=ohlcv[s]["high"], low=ohlcv[s]["low"])
            )
    probes["feature_gate"] = feats
    ks = sorted(eq)[:2]
    curves = {k: eq[k].dropna() for k in ks}
    w, sc = compute_weights_and_scale(curves, target_vol=0.068, window=60, max_leverage=1.0)
    probes["overlay"] = {"weights": _clean(w), "scale": float(sc), "curves": ks}

    # (e) curve metrics on real equity curves
    from ggTrader.lab.metrics import curve_stats

    probes["curve_stats"] = {k: _clean(curve_stats(eq[k].dropna())) for k in sorted(eq)[:6]}
    (out / "probes.json").write_text(json.dumps(probes, indent=1))

    summary["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    summary["pct_change_probe"] = {k: dict(v) for k, v in probe_records.items()}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k != "pct_change_probe"}, indent=1, default=str
        )
    )


# --------------------------------------------------------------------------------------
# compare
# --------------------------------------------------------------------------------------
def _diff_json(a: Any, b: Any, path: str = "", out: list | None = None, tol: float = 0.0) -> list:
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append({"path": f"{path}/{k}", "kind": "missing", "a": k in a, "b": k in b})
            else:
                _diff_json(a[k], b[k], f"{path}/{k}", out, tol)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append({"path": path, "kind": "length", "a": len(a), "b": len(b)})
        for i, (x, y) in enumerate(zip(a, b)):
            _diff_json(x, y, f"{path}[{i}]", out, tol)
    elif isinstance(a, float) and isinstance(b, float):
        if a != b:
            out.append({"path": path, "kind": "float", "a": a, "b": b, "abs": abs(a - b)})
    elif a != b:
        out.append({"path": path, "kind": "value", "a": a, "b": b})
    return out


def _compare_npz(pa: Path, pb: Path) -> dict[str, Any]:
    A, B = np.load(pa), np.load(pb)
    ka, kb = list(A["keys"]), list(B["keys"])
    res: dict[str, Any] = {
        "same_keys": ka == kb,
        "n_curves": len(ka),
        "n_identical": 0,
        "max_abs": 0.0,
        "index_identical": True,
        "different": [],
    }
    if ka != kb:
        return res
    for i, k in enumerate(ka):
        va, vb = A[f"v{i}"], B[f"v{i}"]
        ia, ib = A[f"i{i}"], B[f"i{i}"]
        same = va.shape == vb.shape and bool(np.array_equal(va, vb, equal_nan=True))
        if not np.array_equal(ia, ib):
            res["index_identical"] = False
        if same:
            res["n_identical"] += 1
        else:
            res["different"].append(k)
            if va.shape == vb.shape:
                res["max_abs"] = max(res["max_abs"], float(np.nanmax(np.abs(va - vb))))
    return res


def cmd_compare(a: argparse.Namespace) -> None:
    A, B = Path(a.a), Path(a.b)
    rep: dict[str, Any] = {"a": a.a, "b": b_name(B), "a_summary": {}, "b_summary": {}, "checks": {}}
    sa = json.loads((A / "summary.json").read_text())
    sb = json.loads((B / "summary.json").read_text())
    keep = (
        "tag",
        "pandas",
        "numpy",
        "numba",
        "vectorbt",
        "engine_requested",
        "rust_available",
        "src",
    )
    rep["a_summary"] = {k: sa.get(k) for k in keep}
    rep["b_summary"] = {k: sb.get(k) for k in keep}
    c = rep["checks"]
    for name in ("signal_equity", "weight_equity", "wfo_oos_equity"):
        if (A / f"{name}.npz").exists() and (B / f"{name}.npz").exists():
            c[name] = _compare_npz(A / f"{name}.npz", B / f"{name}.npz")
    c["pit_mask_identical"] = bool(
        np.array_equal(np.load(A / "pit_mask.npy"), np.load(B / "pit_mask.npy"))
    )
    for name in (
        "signal_metrics",
        "weight_metrics",
        "wfo_fold_results",
        "wfo_live_params",
        "probes",
    ):
        fa, fb = A / f"{name}.json", B / f"{name}.json"
        if fa.exists() and fb.exists():
            d = _diff_json(json.loads(fa.read_text()), json.loads(fb.read_text()))
            c[name] = {
                "n_diffs": len(d),
                "max_abs_float_diff": max([x.get("abs", 0.0) for x in d], default=0.0),
                "non_float_diffs": [x for x in d if x["kind"] != "float"][:10],
                "float_diffs_sample": [x for x in d if x["kind"] == "float"][:10],
            }
    ta, tb = A / "wfo_table.txt", B / "wfo_table.txt"
    if ta.exists() and tb.exists():
        c["wfo_table_identical"] = ta.read_text() == tb.read_text()
    for nm in sorted(p.name for p in A.glob("probe_*_pct_*.npy")):
        if (B / nm).exists():
            xa, xb = np.load(A / nm), np.load(B / nm)
            c[nm] = {
                "identical": bool(np.array_equal(xa, xb, equal_nan=True)),
                "max_abs": float(np.nanmax(np.abs(xa - xb))) if xa.shape == xb.shape else None,
            }
    Path(a.out).write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps(rep, indent=1, default=str))


def b_name(p: Path) -> str:
    return str(p)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract", help="read-only DB extract")
    e.add_argument("out")
    r = sub.add_parser("run", help="run the profile from the extract")
    r.add_argument("--extract", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--tag", required=True)
    r.add_argument("--src", help="prepend this directory to sys.path (alternative ggTrader source)")
    r.add_argument("--engine", choices=["default", "numba", "rust"], default="default")
    c = sub.add_parser("compare", help="diff two run directories")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    {"extract": cmd_extract, "run": cmd_run, "compare": cmd_compare}[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
