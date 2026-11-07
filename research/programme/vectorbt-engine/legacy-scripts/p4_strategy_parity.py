"""P4 evidence: ported strategies against the legacy implementations on the frozen tape.

Sections (each writes its own JSON under ``p4-evidence/`` and logs and continues on an error):

* ``grids``     grid values, sizes and validity rules against the legacy ``build_grid`` (f).
* ``signals``   entry/exit boolean frames, every valid combination plus the defaults, two
                200-symbol samples over all 1,929 bars; every difference classified as
                burn-in / tie / real (a); ensemble differences are attributed to voters.
* ``simulator`` identical signals into ``engine.run_signals`` and ``lab.simulate`` (b), and the
                equity effect of the residual burn-in differences.
* ``golden``    the P0 signal golden (npz) for the five voters and the ensemble (c).
* ``live``      the live-defaults control config against P0's live golden, 120 sessions (d).
* ``leak``      truncation and perturbation lookahead checks on real bars (e).
* ``weights``   xs_momentum and cross_asset_sleeve targets and equity against the legacy.
* ``memory``    the full 48-combination ensemble grid on all loaded symbols, chunked run (g).

Reads the frozen Parquet snapshot only (no database, no network, no writes to ``ohlcv``).

Usage:
    systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 \\
        .venv/bin/python scripts/p4_strategy_parity.py --section all
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from ggTrader.data.bars import BAR_FIELDS, Bars, load_bars, to_legacy_frame
from ggTrader.utils.results import write_result_json

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "research" / "programme" / "vectorbt-engine" / "p4-evidence"
GOLDEN_DIR = REPO_ROOT / "research" / "programme" / "vectorbt-engine" / "legacy-references"
EVAL_SPAN = ("2021-01-31", "2026-04-30")
BAR_WINDOW = ("2019-03-01", "2026-10-30")
EXTRAS = ("SPY", "TLT", "GLD", "PDBC", "BIL")
BURN_IN = 200
TIE_TOLERANCE = 1e-9
N_SYMBOLS = 200
EXAMPLES = 6
SECTIONS = ("grids", "signals", "simulator", "golden", "live", "leak", "weights", "memory")


# --------------------------------------------------------------------------- shared helpers


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_tape() -> Bars:
    """All PIT-union SP500 symbols of the 17-fold window plus the sleeve ETFs, every bar."""
    return load_bars(
        universe="sp500",
        universe_span=EVAL_SPAN,
        extra_symbols=EXTRAS,
        start=BAR_WINDOW[0],
        end=BAR_WINDOW[1],
    )


def subset(bars: Bars, symbols: list[str]) -> Bars:
    """Bars restricted to ``symbols`` (column order as given)."""
    return replace(bars, **{f: bars.frame(f)[symbols] for f in BAR_FIELDS})


def samples(bars: Bars) -> dict[str, list[str]]:
    """Sample A: first 200 symbols alphabetically; sample B: seeded random 200 of the rest."""
    syms = sorted(bars.symbols)
    first = syms[:N_SYMBOLS]
    rest = syms[N_SYMBOLS:]
    rng = np.random.default_rng(20261106)
    pick = sorted(rng.choice(rest, size=min(N_SYMBOLS, len(rest)), replace=False).tolist())
    return {"A": first, "B": pick}


def settled_mask(close: pd.DataFrame, burn_in: int = BURN_IN) -> np.ndarray:
    """True on bars at least ``burn_in`` bars after each symbol's own first valid close."""
    first = close.notna().to_numpy().argmax(axis=0)
    return np.arange(len(close))[:, None] >= (first + burn_in)[None, :]


def classify(
    new: np.ndarray,
    old: np.ndarray,
    close: pd.DataFrame,
    margin: Callable[[], np.ndarray] | None,
) -> dict[str, Any]:
    """Split the cells where two boolean frames differ into burn-in, tie and real.

    A cell after the burn-in is a *tie* when the compared quantity is within ``TIE_TOLERANCE`` of
    its threshold on the flip bar or the bar before (vectorbt crossovers need a strict dip first,
    the legacy ``prev <= b`` form does not; a float-rounding difference in a mean can flip a
    comparison that sits exactly on the line). Everything else after the burn-in is *real*.
    ``margin`` returns the absolute distance to the threshold, evaluated only when needed.
    """
    new = np.asarray(new, dtype=bool)
    old = np.asarray(old, dtype=bool)
    xor = new != old
    ok = settled_mask(close)
    late = xor & ok
    rows, cols = np.nonzero(late)
    ties, real_cells, tie_cells = 0, [], []
    if len(rows):
        m = margin() if margin is not None else None
        for r, c in zip(rows, cols, strict=True):
            near = m is not None and (
                m[r, c] < TIE_TOLERANCE or (r > 0 and m[r - 1, c] < TIE_TOLERANCE)
            )
            if near:
                ties += 1
                if len(tie_cells) < EXAMPLES:
                    tie_cells.append((int(r), int(c), bool(new[r, c]), float(m[r, c])))
            else:
                real_cells.append((int(r), int(c), bool(new[r, c])))
    examples = [
        {
            "symbol": str(close.columns[c]),
            "date": str(close.index[r].date()),
            "new": new_v,
            "legacy": not new_v,
        }
        for r, c, new_v in real_cells[:EXAMPLES]
    ]
    return {
        "cells_legacy": int(old.sum()),
        "cells_new": int(new.sum()),
        "xor_total": int(xor.sum()),
        "burn_in": int((xor & ~ok).sum()),
        "tie": ties,
        "real": len(real_cells),
        "real_examples": examples,
        "tie_examples": [
            {
                "symbol": str(close.columns[c]),
                "date": str(close.index[r].date()),
                "new": new_v,
                "legacy": not new_v,
                "distance_to_threshold": dist,
            }
            for r, c, new_v, dist in tie_cells
        ],
    }


def merge_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ("cells_legacy", "cells_new", "xor_total", "burn_in", "tie", "real")
    out = {k: int(sum(r[k] for r in rows)) for k in keys}
    out["real_examples"] = [e for r in rows for e in r["real_examples"]][:EXAMPLES]
    out["tie_examples"] = [e for r in rows for e in r.get("tie_examples", [])][:EXAMPLES]
    return out


# ---- legacy-form margins (distance of the compared quantity to its threshold) ------------


def margin_bb(close: pd.DataFrame, period: int, std: float, which: str) -> np.ndarray:
    sma = close.rolling(period, min_periods=period).mean()
    sd = close.rolling(period, min_periods=period).std()
    ref = sma - std * sd if which == "entry" else sma
    return (close - ref).abs().to_numpy()


def margin_rsi(close: pd.DataFrame, period: int, level: float) -> np.ndarray:
    delta = close.diff()
    gain, loss = delta.clip(lower=0.0), -delta.clip(upper=0.0)
    ag = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    al = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rsi = 100.0 - 100.0 / (1.0 + ag / al.replace(0, np.nan))
    return (rsi - level).abs().to_numpy()


def margin_ema(close: pd.DataFrame, fast: int, slow: int) -> np.ndarray:
    f = close.ewm(span=fast, adjust=False).mean()
    s = close.ewm(span=slow, adjust=False).mean()
    return (f - s).abs().to_numpy()


def margin_macd(close: pd.DataFrame, p: tuple, which: str) -> np.ndarray:
    fast, slow, signal, window = (int(x) for x in p)
    line = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    hist = line - line.ewm(span=signal, adjust=False).mean()
    if which == "exit":
        return hist.abs().to_numpy()
    low = hist.rolling(window, min_periods=window).min()
    return (hist - low).abs().to_numpy()


def margin_vbb(close: pd.DataFrame, volume: pd.DataFrame, p: tuple, which: str) -> np.ndarray:
    period, std, vol_period, vol_mult = p
    if which == "exit":
        return margin_bb(close, int(period), float(std), "exit")
    band = margin_bb(close, int(period), float(std), "entry")
    avg = volume.rolling(int(vol_period), min_periods=int(vol_period)).mean()
    spike = (volume - float(vol_mult) * avg).abs().to_numpy()
    return np.fmin(band, spike)


def voter_margin(voter: str, params: tuple, which: str, bars: Bars) -> np.ndarray:
    """Legacy-form distance to threshold of one voter's entry or exit comparison."""
    c, v = bars.close, bars.volume
    if voter == "bb":
        return margin_bb(c, int(params[0]), float(params[1]), which)
    if voter == "rsi":
        return margin_rsi(c, int(params[0]), float(params[1] if which == "entry" else params[2]))
    if voter == "ema":
        return margin_ema(c, int(params[0]), int(params[1]))
    if voter == "macd":
        return margin_macd(c, params, which)
    return margin_vbb(c, v, params, which)


def legacy_voter(voter: str, params: tuple, bars: Bars) -> tuple[np.ndarray, np.ndarray]:
    """Legacy ``*_signals`` function output for one voter and parameter tuple."""
    from ggTrader.lab.strategies import indicators as leg

    c, v = bars.close, bars.volume
    if voter == "bb":
        e, x = leg.bb_signals(c, int(params[0]), float(params[1]))
    elif voter == "rsi":
        e, x = leg.rsi_signals(c, int(params[0]), int(params[1]), int(params[2]))
    elif voter == "ema":
        e, x = leg.ema_signals(c, int(params[0]), int(params[1]))
    elif voter == "macd":
        e, x = leg.macd_signals(c, *(int(p) for p in params))
    else:
        e, x = leg.volume_bb_signals(
            c, v, int(params[0]), float(params[1]), int(params[2]), float(params[3])
        )
    return e.to_numpy(), x.to_numpy()


# --------------------------------------------------------------------------- section: grids


def section_grids() -> dict[str, Any]:
    from ggTrader.lab.strategies import STRATEGY_REGISTRY
    from ggTrader.lab.strategies.v2 import REGISTRY_V2
    from ggTrader.lab.sweep import build_grid

    rows = {}
    for name, cls in REGISTRY_V2.items():
        legacy = build_grid(STRATEGY_REGISTRY[name])
        new = cls.grid.combos()
        rows[name] = {
            "kind": cls.kind,
            "raw_combinations": int(np.prod([len(v) for v in cls.grid.values.values()])),
            "valid_new": len(new),
            "valid_legacy": len(legacy),
            "combos_equal_in_order": new == legacy,
            "rules": [f"{r.small} < {r.large}" for r in cls.grid.rules],
            "swept_levels": cls.grid.swept(),
        }
    ens = REGISTRY_V2["ensemble"]
    ens_grid_has_live = any(
        all(c.get(k) == v for k, v in ens.live.items() if k in c) for c in ens.grid.combos()
    )
    not_ported = sorted(set(STRATEGY_REGISTRY) - set(REGISTRY_V2))
    return {
        "section": "grids",
        "strategies": rows,
        "all_equal": all(r["combos_equal_in_order"] for r in rows.values()),
        "ensemble_live_defaults": ens.live,
        "ensemble_live_inside_wfo_grid": ens_grid_has_live,
        "ported": sorted(REGISTRY_V2),
        "not_ported_legacy_only": not_ported,
        "n_legacy_registry": len(STRATEGY_REGISTRY),
    }


# --------------------------------------------------------------------------- section: signals


def legacy_combo_signals(
    old_cls: type, combos: list[dict[str, Any]], bars: Bars, frame: pd.DataFrame
) -> dict[str, Any]:
    from ggTrader.lab.strategy import LabConfig

    return old_cls(LabConfig()).sweep_signals(combos, list(bars.close.columns), frame)


def voter_level_report(voter: str, params_list: list[tuple], bars: Bars) -> dict[str, Any]:
    """New voter arrays against the legacy voter function for each parameter tuple."""
    from ggTrader.lab.strategies.v2.voters import VOTER_SPECS

    new_arrays = VOTER_SPECS[voter].compute(bars, params_list)
    rows = []
    for params in params_list:
        old_e, old_x = legacy_voter(voter, params, bars)
        new_e, new_x = new_arrays[params]
        rows.append(
            {
                "params": list(params),
                "entries": classify(
                    new_e, old_e, bars.close, lambda p=params: voter_margin(voter, p, "entry", bars)
                ),
                "exits": classify(
                    new_x, old_x, bars.close, lambda p=params: voter_margin(voter, p, "exit", bars)
                ),
            }
        )
    return {
        "voter": voter,
        "n_param_tuples": len(rows),
        "entries": merge_counts([r["entries"] for r in rows]),
        "exits": merge_counts([r["exits"] for r in rows]),
        "per_tuple": rows,
    }


def ensemble_report(bars: Bars, frame: pd.DataFrame) -> dict[str, Any]:
    """Ensemble: all 48 grid combos plus the live-defaults control, with voter attribution."""
    from ggTrader.lab.strategies.ensemble import EnsembleSignal
    from ggTrader.lab.strategies.v2.ensemble import Ensemble
    from ggTrader.lab.strategies.v2.voters import VOTER_SPECS, voter_arrays
    from ggTrader.lab.sweep import combo_name

    out: dict[str, Any] = {}
    n = len(bars.symbols)
    close = bars.close
    for label, grid, legacy_combos in (
        ("grid_48", Ensemble.grid, Ensemble.grid.combos()),
        ("live_defaults_control", Ensemble.live_grid(), [dict(Ensemble.live)]),
    ):
        t0 = time.perf_counter()
        new = Ensemble().build(bars, grid)
        t_new = time.perf_counter() - t0
        t0 = time.perf_counter()
        legacy = legacy_combo_signals(EnsembleSignal, legacy_combos, bars, frame)
        t_old = time.perf_counter() - t0
        new_voters = {
            v: voter_arrays(v, bars, list(new.combos)) for v in ("bb", "rsi", "ema", "macd", "vbb")
        }
        vcache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
        rows = []
        for i, combo in enumerate(new.combos):
            key = combo_name("ensemble", {k: combo[k] for k in legacy_combos[0]})
            old = legacy[key]
            sl = slice(i * n, (i + 1) * n)
            res: dict[str, Any] = {"params": {k: combo[k] for k in grid.swept()}}
            for side, new_arr, old_arr in (
                ("entries", new.entries.to_numpy()[:, sl], old.entries.to_numpy()),
                ("exits", new.exits.to_numpy()[:, sl], old.exits.to_numpy()),
            ):

                def margin(side=side, combo=combo, new_arr=new_arr, old_arr=old_arr) -> np.ndarray:
                    return attribute_ensemble(
                        side, combo, new_arr, old_arr, bars, new_voters, vcache, VOTER_SPECS
                    )

                res[side] = classify(new_arr, old_arr, close, margin)
            rows.append(res)
        out[label] = {
            "n_combos": len(rows),
            "seconds_new_build": round(t_new, 2),
            "seconds_legacy": round(t_old, 2),
            "entries": merge_counts([r["entries"] for r in rows]),
            "exits": merge_counts([r["exits"] for r in rows]),
            "per_combo": rows,
        }
    return out


def attribute_ensemble(
    side: str,
    combo: dict[str, Any],
    new_arr: np.ndarray,
    old_arr: np.ndarray,
    bars: Bars,
    new_voters: dict,
    vcache: dict,
    specs: dict,
) -> np.ndarray:
    """Margin matrix for ensemble flips: the smallest voter margin among voters that differ.

    A flipped cell is only explainable when some voter (entry or, for exits, exit) differs from
    its legacy output at that cell. Cells where no voter differs get margin +inf, i.e. *real*
    (the vote logic itself would then be wrong).
    """
    shape = new_arr.shape
    best = np.full(shape, np.inf)
    which = "entry" if side == "entries" else "exit"
    for voter, arrays in new_voters.items():
        params = tuple(combo[k] for k in specs[voter].keys)
        key = (voter, params)
        if key not in vcache:
            vcache[key] = legacy_voter(voter, params, bars)
        old_e, old_x = vcache[key]
        new_e, new_x = arrays.arrays[params]
        diff = (new_e != old_e) if which == "entry" else (new_x != old_x)
        if not diff.any():
            continue
        margin = voter_margin(voter, params, which, bars)
        best = np.where(diff, np.fmin(best, np.nan_to_num(margin, nan=np.inf)), best)
    return best


def section_signals(bars: Bars) -> dict[str, Any]:
    from ggTrader.lab.strategies import STRATEGY_REGISTRY
    from ggTrader.lab.strategies.v2 import REGISTRY_V2
    from ggTrader.lab.strategies.v2.voters import VOTER_SPECS

    result: dict[str, Any] = {"section": "signals", "burn_in_bars": BURN_IN}
    for sample, syms in samples(bars).items():
        sub = subset(bars, syms)
        frame = to_legacy_frame(sub)
        log(f"signals sample {sample}: {len(syms)} symbols x {len(sub.close)} bars")
        sample_out: dict[str, Any] = {
            "symbols": len(syms),
            "bars": len(sub.close),
            "first_symbol": syms[0],
            "last_symbol": syms[-1],
        }
        voter_of = {
            "bb_reversion": "bb",
            "rsi_reversion": "rsi",
            "ema_cross": "ema",
            "macd_divergence": "macd",
            "volume_bb_reversion": "vbb",
        }
        for name, voter in voter_of.items():
            cls = REGISTRY_V2[name]
            combos = [{**cls.defaults, **c} for c in cls.grid.combos()]
            tuples = list(
                dict.fromkeys(tuple(c[k] for k in VOTER_SPECS[voter].keys) for c in combos)
            )
            defaults_tuple = tuple(cls.defaults[k] for k in VOTER_SPECS[voter].keys)
            if defaults_tuple not in tuples:
                tuples.append(defaults_tuple)
            rep = voter_level_report(voter, tuples, sub)
            # Strategy-level check through the real legacy sweep path on the same combos.
            rep["strategy_level"] = strategy_level(name, cls, STRATEGY_REGISTRY[name], sub, frame)
            sample_out[name] = rep
            log(
                f"  {name}: tuples={rep['n_param_tuples']} entries={rep['entries']['xor_total']} "
                f"(real {rep['entries']['real']}, tie {rep['entries']['tie']}) "
                f"exits={rep['exits']['xor_total']} (real {rep['exits']['real']}, tie "
                f"{rep['exits']['tie']})"
            )
        ens = ensemble_report(sub, frame)
        sample_out["ensemble"] = ens
        for label, rep in ens.items():
            log(
                f"  ensemble {label}: combos={rep['n_combos']} entries real="
                f"{rep['entries']['real']} tie={rep['entries']['tie']} exits real="
                f"{rep['exits']['real']} tie={rep['exits']['tie']}"
            )
        cb = conviction_report(sub, frame)
        sample_out["conviction_bb"] = cb
        log(f"  conviction_bb: {cb['entries']} {cb['exits']} size max diff {cb['max_size_diff']}")
        result[f"sample_{sample}"] = sample_out
    return result


def strategy_level(name: str, cls: Any, old_cls: type, bars: Bars, frame: pd.DataFrame) -> dict:
    """Standalone strategy build against the legacy ``sweep_signals`` for every grid combo."""
    from ggTrader.lab.sweep import combo_name

    combos = cls.grid.combos()
    new = cls().build(bars)
    legacy = legacy_combo_signals(old_cls, combos, bars, frame)
    n = len(bars.close.columns)
    ok = settled_mask(bars.close)
    e_diff = x_diff = e_burn = x_burn = 0
    for i, combo in enumerate(new.combos):
        old = legacy[combo_name(name, {k: combo[k] for k in cls.grid.values})]
        sl = slice(i * n, (i + 1) * n)
        de = new.entries.to_numpy()[:, sl] != old.entries.to_numpy()
        dx = new.exits.to_numpy()[:, sl] != old.exits.to_numpy()
        e_diff += int((de & ok).sum())
        x_diff += int((dx & ok).sum())
        e_burn += int((de & ~ok).sum())
        x_burn += int((dx & ~ok).sum())
    return {
        "n_combos": len(new.combos),
        "entries_xor_after_burn_in": e_diff,
        "exits_xor_after_burn_in": x_diff,
        "entries_xor_within_burn_in": e_burn,
        "exits_xor_within_burn_in": x_burn,
    }


def conviction_report(bars: Bars, frame: pd.DataFrame) -> dict[str, Any]:
    from ggTrader.lab.strategies.conviction import ConvictionBBSignal
    from ggTrader.lab.strategies.v2.conviction_bb import ConvictionBb
    from ggTrader.lab.sweep import combo_name

    cls = ConvictionBb
    out = cls().build(bars)
    legacy = legacy_combo_signals(ConvictionBBSignal, cls.grid.combos(), bars, frame)
    n = len(bars.close.columns)
    rows_e, rows_x, worst = [], [], 0.0
    ok = settled_mask(bars.close)
    for i, combo in enumerate(out.combos):
        old = legacy[combo_name("conviction_bb", {k: combo[k] for k in cls.grid.values})]
        sl = slice(i * n, (i + 1) * n)
        key = (combo["bb_period"], combo["bb_std"])
        rows_e.append(
            classify(
                out.entries.to_numpy()[:, sl],
                old.entries.to_numpy(),
                bars.close,
                lambda k=key: margin_bb(bars.close, int(k[0]), float(k[1]), "entry"),
            )
        )
        rows_x.append(
            classify(
                out.exits.to_numpy()[:, sl],
                old.exits.to_numpy(),
                bars.close,
                lambda k=key: margin_bb(bars.close, int(k[0]), float(k[1]), "exit"),
            )
        )
        both = ok & out.entries.to_numpy()[:, sl] & old.entries.to_numpy()
        a, b = out.size.to_numpy()[:, sl][both], old.sizes.to_numpy()[both]
        if a.size:
            worst = max(worst, float(np.nanmax(np.abs(a - b))))
    return {
        "n_combos": len(out.combos),
        "entries": merge_counts(rows_e),
        "exits": merge_counts(rows_x),
        "max_size_diff": worst,
    }


# --------------------------------------------------------------------------- section: simulator

SIM_CASES = {
    "ensemble": [0, 9, 18, 27, 36, 47],
    "conviction_bb": [0, 14, 35],
    "bb_reversion": [0, 7, 15],
    "rsi_reversion": [0, 13, 26],
    "ema_cross": [0, 8, 15],
    "macd_divergence": [0, 4, 7],
    "volume_bb_reversion": [0, 6, 11],
}


def section_simulator(bars: Bars) -> dict[str, Any]:
    from ggTrader.lab.data import STOCK_BASE_CONFIG
    from ggTrader.lab.simulate import simulate_signals
    from ggTrader.lab.strategies import STRATEGY_REGISTRY
    from ggTrader.lab.strategies.v2 import REGISTRY_V2
    from ggTrader.lab.strategies.v2.contract import simulate_signal_out
    from ggTrader.lab.strategy import SignalTargets
    from ggTrader.lab.sweep import combo_name
    from ggTrader.lab.universe import pit_mask

    syms = samples(bars)["A"]
    sub = subset(bars, syms)
    frame = to_legacy_frame(sub)
    mask = pit_mask(sub.close, min_history_bars=400, on_stale="ignore")
    cfg = dict(STOCK_BASE_CONFIG)
    out: dict[str, Any] = {
        "section": "simulator",
        "base_config": {k: cfg[k] for k in ("START_CASH", "FEES", "SLIPPAGE", "FREQ")}
        | {"SIGNAL_POSITION_SIZE": cfg["SIGNAL_POSITION_SIZE"]},
        "mask": "pit_mask(min_history_bars=400, sp500 PIT), applied before the time stop",
        "symbols": len(syms),
        "bars": len(sub.close),
        "cases": {},
    }
    for name, picks in SIM_CASES.items():
        cls = REGISTRY_V2[name]
        combos_all = cls.grid.combos()
        picks = [p for p in picks if p < len(combos_all)]
        combos = [combos_all[p] for p in picks]
        full = cls().build(sub)  # whole grid once; the picked groups are sliced out below
        n = len(syms)
        rows = []
        for p, combo in zip(picks, combos, strict=True):
            keep_cols = slice(p * n, (p + 1) * n)
            e = full.entries.iloc[:, keep_cols]
            x = full.exits.iloc[:, keep_cols]
            size = None if full.size is None else full.size.iloc[:, keep_cols]
            one = type(full)(
                entries=e, exits=x, combos=(full.combos[p],), size=size, td_stop=full.td_stop
            )
            new_res = simulate_signal_out(one, sub.close, cfg, mask=mask)
            ent_plain, ext_plain = one.group(0)
            masked_entries = ent_plain & mask
            tgt = SignalTargets(
                entries=masked_entries,
                exits=ext_plain,
                sizes=None if size is None else size.set_axis(sub.close.columns, axis=1),
            )
            _, old_same, diag_same = simulate_signals({"g": tgt}, sub.close, cfg)
            same_equal = bool(
                np.array_equal(new_res.value.iloc[:, 0].to_numpy(), old_same["g"].to_numpy())
            )
            same_trades = int(new_res.n_trades.iloc[0]) == int(diag_same["g"]["n_trades"])
            # Legacy signals through the legacy simulator, new signals through the engine.
            legacy = legacy_combo_signals(STRATEGY_REGISTRY[name], [combo], sub, frame)
            lkey = combo_name(name, combo)
            lt = legacy[lkey]
            ltgt = SignalTargets(
                entries=lt.entries & mask, exits=lt.exits, sizes=getattr(lt, "sizes", None)
            )
            _, old_legacy, diag_old = simulate_signals({"g": ltgt}, sub.close, cfg)
            a = new_res.value.iloc[:, 0].to_numpy()
            b = old_legacy["g"].to_numpy()
            rows.append(
                {
                    "combo_index": p,
                    "params": {k: full.combos[p][k] for k in cls.grid.swept()},
                    "same_signals_equity_bit_identical": same_equal,
                    "same_signals_trade_counts_equal": same_trades,
                    "n_trades_new": int(new_res.n_trades.iloc[0]),
                    "legacy_signals_vs_new_signals": {
                        "equity_bit_identical": bool(np.array_equal(a, b)),
                        "max_abs_equity_diff": float(np.abs(a - b).max()),
                        "final_equity_new": float(a[-1]),
                        "final_equity_legacy_signals": float(b[-1]),
                        "n_trades_new": int(new_res.n_trades.iloc[0]),
                        "n_trades_legacy_signals": int(diag_old["g"]["n_trades"]),
                    },
                }
            )
        out["cases"][name] = rows
        log(
            f"simulator {name}: same-signal bit-identical "
            f"{sum(r['same_signals_equity_bit_identical'] for r in rows)}/{len(rows)}; "
            f"legacy-vs-new signals bit-identical "
            f"{sum(r['legacy_signals_vs_new_signals']['equity_bit_identical'] for r in rows)}"
            f"/{len(rows)}"
        )
    out["all_same_signal_equity_bit_identical"] = all(
        r["same_signals_equity_bit_identical"] and r["same_signals_trade_counts_equal"]
        for rows in out["cases"].values()
        for r in rows
    )
    return out


# --------------------------------------------------------------------------- section: golden


def section_golden(bars: Bars) -> dict[str, Any]:
    from ggTrader.lab.strategies.v2.ensemble import Ensemble
    from ggTrader.lab.strategies.v2.voters import VOTER_DEFAULTS, VOTER_SPECS

    golden = np.load(GOLDEN_DIR / "signal_golden_legacy.npz", allow_pickle=False)
    syms = [str(s) for s in golden["columns"]]
    sub = subset(bars, syms)
    assert len(sub.close) == len(golden["index_ns"]), "golden bar count differs from the tape"
    assert (sub.close.index.tz_convert(None).asi8 == golden["index_ns"]).all()
    out: dict[str, Any] = {
        "section": "golden",
        "golden_file": "legacy-references/signal_golden_legacy.npz",
        "symbols": len(syms),
        "bars": len(sub.close),
        "config": "live defaults (bb 20/2.0, rsi 14/30/50, ema 20/50, macd 12/26/9 div 20, vbb "
        "20/2.0/20/2.0), min_agree 2, min_agree_exit 2",
        "voters": {},
    }
    defaults = {**VOTER_DEFAULTS}
    for voter in ("bb", "rsi", "ema", "macd", "vbb"):
        keys = VOTER_SPECS[voter].keys
        params = tuple(defaults[k] for k in keys)
        arrays = VOTER_SPECS[voter].compute(sub, [params])[params]
        out["voters"][voter] = {
            "entries": classify(
                arrays[0],
                golden[f"{voter}_entries"],
                sub.close,
                lambda v=voter, p=params: voter_margin(v, p, "entry", sub),
            ),
            "exits": classify(
                arrays[1],
                golden[f"{voter}_exits"],
                sub.close,
                lambda v=voter, p=params: voter_margin(v, p, "exit", sub),
            ),
        }
    ens = Ensemble().build(sub, Ensemble.live_grid())
    from ggTrader.lab.strategies.v2.voters import voter_arrays

    new_voters = {
        v: voter_arrays(v, sub, list(ens.combos)) for v in ("bb", "rsi", "ema", "macd", "vbb")
    }
    vcache: dict = {}
    combo = ens.combos[0]
    res: dict[str, Any] = {}
    for side in ("entries", "exits"):
        new_arr = getattr(ens, side).to_numpy()
        gold = golden[f"ensemble_{side}"]

        def margin(side: str = side) -> np.ndarray:
            return attribute_ensemble(
                side, combo, new_arr, gold, sub, new_voters, vcache, VOTER_SPECS
            )

        res[side] = classify(new_arr, gold, sub.close, margin)
    out["ensemble"] = res
    out["ensemble_totals_golden"] = {
        "entries": int(golden["ensemble_entries"].sum()),
        "exits": int(golden["ensemble_exits"].sum()),
    }
    for name, rep in out["voters"].items():
        log(
            f"golden {name}: entries {rep['entries']['xor_total']} real {rep['entries']['real']}; "
            f"exits {rep['exits']['xor_total']} real {rep['exits']['real']}"
        )
    log(
        f"golden ensemble: {res['entries']['xor_total']}/{res['exits']['xor_total']} xor, real "
        f"{res['entries']['real']}/{res['exits']['real']}, tie {res['entries']['tie']}/"
        f"{res['exits']['tie']}"
    )
    return out


# --------------------------------------------------------------------------- section: live


def section_live(bars: Bars) -> dict[str, Any]:
    from ggTrader.data.core.index_constituents import normalize_yf_ticker, universe_members_asof
    from ggTrader.lab.strategies.v2.ensemble import Ensemble

    golden = json.loads((GOLDEN_DIR / "live_golden_legacy.json").read_text())
    have = set(bars.close.columns)
    min_hist = 60
    rows, mismatches = [], []
    strat = Ensemble()
    grid = Ensemble.live_grid()
    t0 = time.perf_counter()
    for rec in golden:
        day = pd.Timestamp(rec["session"], tz="UTC")
        members = universe_members_asof("sp500", day)
        symbols = sorted({normalize_yf_ticker(t) for t in members} & have)
        start = day - pd.Timedelta(days=365)
        idx = bars.close.index
        sel = (idx >= start) & (idx <= day)
        window = replace(bars, **{f: bars.frame(f).loc[sel, symbols] for f in BAR_FIELDS})
        count = window.close.notna().sum()
        eligible = count[count >= min_hist].index.tolist()
        out = strat.build(window, grid)
        last = window.close.index[-1]
        ent = out.entries.iloc[:, :].set_axis(window.symbols, axis=1)
        ext = out.exits.iloc[:, :].set_axis(window.symbols, axis=1)
        buys = sorted(s for s in eligible if bool(ent.at[last, s]))
        sells = sorted(s for s in eligible if bool(ext.at[last, s]))
        last_exit = {}
        for s in eligible:
            col = ext[s].to_numpy()
            if col.any():
                last_exit[s] = str(ext.index[len(col) - 1 - int(col[::-1].argmax())].date())
        short = {s for s in eligible if count[s] < 200}
        diff = {
            "as_of": rec["as_of"],
            "buys_only_new": sorted(set(buys) - set(rec["buys"])),
            "buys_only_legacy": sorted(set(rec["buys"]) - set(buys)),
            "sells_only_new": sorted(set(sells) - set(rec["sells"])),
            "sells_only_legacy": sorted(set(rec["sells"]) - set(sells)),
            "last_exit_diff": {
                s: {"new": last_exit.get(s), "legacy": rec["last_exit"].get(s)}
                for s in set(last_exit) | set(rec["last_exit"])
                if last_exit.get(s) != rec["last_exit"].get(s)
            },
        }
        diff["short_history_symbols_involved"] = sorted(
            (
                set(diff["buys_only_new"])
                | set(diff["buys_only_legacy"])
                | set(diff["sells_only_new"])
                | set(diff["sells_only_legacy"])
                | set(diff["last_exit_diff"])
            )
            & short
        )
        bad = any(diff[k] for k in diff if k not in ("as_of", "short_history_symbols_involved"))
        rows.append(
            {
                "as_of": rec["as_of"],
                "universe": len(symbols),
                "eligible": len(eligible),
                "buys": len(buys),
                "sells": len(sells),
                "short_history_symbols": len(short),
                "differs": bad,
            }
        )
        if bad:
            mismatches.append(diff)
    n_bad = len(mismatches)
    total_ld = sum(len(m["last_exit_diff"]) for m in mismatches)
    result = {
        "section": "live",
        "sessions": len(golden),
        "first": golden[0]["as_of"],
        "last": golden[-1]["as_of"],
        "sessions_with_any_difference": n_bad,
        "sessions_with_buy_or_sell_difference": sum(
            1
            for m in mismatches
            if m["buys_only_new"]
            or m["buys_only_legacy"]
            or m["sells_only_new"]
            or m["sells_only_legacy"]
        ),
        "symbol_sessions_last_exit_differs": total_ld,
        "total_buys_golden": int(sum(len(r["buys"]) for r in golden)),
        "total_sells_golden": int(sum(len(r["sells"]) for r in golden)),
        "total_buys_new": int(sum(r["buys"] for r in rows)),
        "total_sells_new": int(sum(r["sells"] for r in rows)),
        "short_history_symbol_sessions": int(sum(r["short_history_symbols"] for r in rows)),
        "seconds": round(time.perf_counter() - t0, 1),
        "mismatches": mismatches[:25],
        "per_session": rows,
    }
    log(
        f"live golden: {n_bad}/{len(golden)} sessions differ; buys/sells differ in "
        f"{result['sessions_with_buy_or_sell_difference']}; last_exit diffs {total_ld}"
    )
    return result


# --------------------------------------------------------------------------- section: leak


def section_leak(bars: Bars) -> dict[str, Any]:
    from ggTrader.lab.data import rebalance_dates
    from ggTrader.lab.strategies.v2 import REGISTRY_V2
    from ggTrader.lab.strategies.v2.contract import ParamGrid
    from ggTrader.lab.strategies.v2.ensemble import Ensemble
    from ggTrader.lab.strategies.v2.leak import signal_leak_check, weight_leak_check

    syms = samples(bars)["A"][:80]
    sub = subset(bars, syms)
    rng = np.random.default_rng(7)
    pos = sorted(rng.choice(np.arange(250, len(sub.close) - 2), size=40, replace=False).tolist())
    dates = [sub.close.index[p] for p in pos]
    out: dict[str, Any] = {
        "section": "leak",
        "symbols": len(syms),
        "bars": len(sub.close),
        "dates_tested": len(dates),
        "variants": ["truncate (rows after t removed)", "perturb (rows after t rescaled)"],
        "signals": {},
        "weights": {},
    }
    cases: dict[str, tuple[Any, ParamGrid]] = {"ensemble_live": (Ensemble(), Ensemble.live_grid())}
    small_ens = ParamGrid.from_sweep(
        {**Ensemble.grid.values, "min_agree": [2, 3], "ema_fast": [10, 20]}
    )
    cases["ensemble_grid_slice"] = (Ensemble(), small_ens)
    for name in (
        "bb_reversion",
        "rsi_reversion",
        "ema_cross",
        "macd_divergence",
        "volume_bb_reversion",
        "conviction_bb",
    ):
        cls = REGISTRY_V2[name]
        cases[name] = (cls(), ParamGrid.single(cls.defaults))
    for label, (strategy, grid) in cases.items():
        t0 = time.perf_counter()
        problems = signal_leak_check(strategy, sub, grid, dates)
        out["signals"][label] = {
            "violations": problems,
            "seconds": round(time.perf_counter() - t0, 1),
        }
        log(f"leak {label}: {len(problems)} violations")
    # Weights: eligibility is part of the input, so perturb bars only.
    index = bars.close.index
    rebal = rebalance_dates(
        index, pd.Timestamp(EVAL_SPAN[0], tz="UTC"), pd.Timestamp(EVAL_SPAN[1], tz="UTC")
    )
    pick = [rebal[i] for i in (3, 20, 40, 60)]
    wbars = subset(bars, sorted(set(samples(bars)["A"]) | set(EXTRAS)))
    mom_grid = ParamGrid.from_sweep({"top_n": [20], "lookback": [126, 252], "skip": [0, 21]})
    out["weights"]["xs_momentum"] = weight_leak_check(
        REGISTRY_V2["xs_momentum"](), wbars, mom_grid, pick
    )
    out["weights"]["cross_asset_sleeve"] = weight_leak_check(
        REGISTRY_V2["cross_asset_sleeve"](), wbars, REGISTRY_V2["cross_asset_sleeve"].grid, pick
    )
    out["all_clean"] = not any(v["violations"] for v in out["signals"].values()) and not any(
        out["weights"].values()
    )
    log(f"leak weights: {out['weights']}; all clean {out['all_clean']}")
    return out


# --------------------------------------------------------------------------- section: weights


def section_weights(bars: Bars) -> dict[str, Any]:
    from ggTrader.engine import portfolio as engine_portfolio
    from ggTrader.lab.data import STOCK_BASE_CONFIG, rebalance_dates
    from ggTrader.lab.simulate import simulate_weights
    from ggTrader.lab.strategies.cross_asset_sleeve import CrossAssetSleeveStrategy
    from ggTrader.lab.strategies.momentum import CrossSectionalMomentum
    from ggTrader.lab.strategies.v2.contract import ParamGrid, tile_columns
    from ggTrader.lab.strategies.v2.cross_asset_sleeve import CrossAssetSleeve
    from ggTrader.lab.strategies.v2.xs_momentum import XsMomentum
    from ggTrader.lab.strategy import LabConfig
    from ggTrader.lab.universe import pit_mask

    cfg = dict(STOCK_BASE_CONFIG)
    index = bars.close.index
    rebal = rebalance_dates(
        index, pd.Timestamp(EVAL_SPAN[0], tz="UTC"), pd.Timestamp(EVAL_SPAN[1], tz="UTC")
    )
    data = to_legacy_frame(bars)
    out: dict[str, Any] = {"section": "weights", "rebalance_dates": len(rebal), "bars": len(index)}

    # ---- xs_momentum on the PIT universe (min history 400)
    eligible = pit_mask(bars.close, min_history_bars=400, on_stale="ignore")
    t0 = time.perf_counter()
    w = XsMomentum().build(bars, rebal=pd.DatetimeIndex(rebal), eligible=eligible)
    t_new = time.perf_counter() - t0
    legacy_targets: dict[str, pd.DataFrame] = {}
    rows = []
    t0 = time.perf_counter()
    for i, combo in enumerate(XsMomentum.grid.combos()):
        strat = CrossSectionalMomentum(
            LabConfig(top_n=combo["top_n"], lookback=combo["lookback"], skip=combo["skip"])
        )
        plans = {
            a: strat.select(a, data.loc[:a], eligible.columns[eligible.loc[a].to_numpy()].tolist())
            for a in rebal
        }
        legacy = strat.to_targets(plans, data)
        legacy_targets[f"c{i}"] = legacy
        syms = list(dict.fromkeys(w.columns.get_level_values(-1)))
        block = w.iloc[:, i * len(syms) : (i + 1) * len(syms)].set_axis(syms, axis=1)
        new = block[list(legacy.columns)]
        extra = [s for s in syms if s not in legacy.columns]
        same = bool(np.array_equal(new.to_numpy(), legacy.to_numpy(), equal_nan=True))
        rows.append(
            {
                "params": {k: combo[k] for k in ("top_n", "lookback", "skip")},
                "targets_equal": same,
                "max_abs_diff": float(np.nanmax(np.abs(new.to_numpy() - legacy.to_numpy())))
                if legacy.size
                else 0.0,
                "legacy_columns": legacy.shape[1],
                "extra_columns_in_union_are_all_zero_or_nan": bool(
                    np.nan_to_num(block[extra].to_numpy()).sum() == 0.0
                ),
            }
        )
    t_old = time.perf_counter() - t0
    new_res = engine_portfolio.run_weights(
        w, tile_columns(bars.close, w.columns), cfg, groups=[k[:-1] for k in w.columns.to_list()]
    )
    _, old_value, old_diag = simulate_weights(legacy_targets, bars.close, cfg)
    eq_equal = bool(np.array_equal(new_res.value.to_numpy(), old_value.to_numpy()))
    out["xs_momentum"] = {
        "combos": rows,
        "all_targets_equal": all(r["targets_equal"] for r in rows),
        "equity_bit_identical": eq_equal,
        "max_abs_equity_diff": float(np.abs(new_res.value.to_numpy() - old_value.to_numpy()).max()),
        "trade_counts_equal": [int(x) for x in new_res.n_trades]
        == [int(old_diag[f"c{i}"]["n_trades"]) for i in range(len(rows))],
        "seconds_new_build": round(t_new, 2),
        "seconds_legacy_select_and_targets": round(t_old, 1),
        "weight_columns": w.shape[1],
    }
    log(
        f"weights xs_momentum: targets equal {out['xs_momentum']['all_targets_equal']}, equity "
        f"bit-identical {eq_equal}"
    )

    # ---- cross_asset_sleeve (trend grid and the static arm)
    sleeve_out = {}
    for label, grid, trend in (
        ("trend_grid", CrossAssetSleeve.grid, True),
        ("static", ParamGrid.single({"trend": False}), False),
    ):
        wk = CrossAssetSleeve().build(bars, grid, pd.DatetimeIndex(rebal))
        syms = list(dict.fromkeys(wk.columns.get_level_values(-1)))
        targets, rows = {}, []
        for i, combo in enumerate(grid.combos()):
            strat = CrossAssetSleeveStrategy(
                LabConfig(lookback=int(combo.get("lookback", 252))), trend=trend
            )
            plans = {a: strat.select(a, data.loc[:a], []) for a in rebal}
            legacy = strat.to_targets(plans, data)
            targets[f"c{i}"] = legacy
            block = wk.iloc[:, i * len(syms) : (i + 1) * len(syms)].set_axis(syms, axis=1)[
                list(legacy.columns)
            ]
            rows.append(
                {
                    "params": {k: combo[k] for k in grid.swept()} or {"trend": trend},
                    "targets_equal": bool(
                        np.array_equal(block.to_numpy(), legacy.to_numpy(), equal_nan=True)
                    ),
                }
            )
        new_res = engine_portfolio.run_weights(
            wk,
            tile_columns(bars.close, wk.columns),
            cfg,
            groups=[k[:-1] for k in wk.columns.to_list()],
        )
        _, old_value, _ = simulate_weights(targets, bars.close, cfg)
        sleeve_out[label] = {
            "combos": rows,
            "all_targets_equal": all(r["targets_equal"] for r in rows),
            "equity_bit_identical": bool(
                np.array_equal(new_res.value.to_numpy(), old_value.to_numpy())
            ),
            "max_abs_equity_diff": float(
                np.abs(new_res.value.to_numpy() - old_value.to_numpy()).max()
            ),
        }
        log(
            f"weights sleeve {label}: targets equal {sleeve_out[label]['all_targets_equal']}, "
            f"equity bit-identical {sleeve_out[label]['equity_bit_identical']}"
        )
    out["cross_asset_sleeve"] = sleeve_out
    out["retail_attention_note"] = (
        "google_trends_interest has 0 rows, so the real book is flat cash. The port is tested "
        "on synthetic interest data (tests/lab/strategies_v2/test_weights.py: targets equal the "
        "legacy for all 6 combinations; empty data gives no columns). No real-data run exists."
    )
    return out


# --------------------------------------------------------------------------- section: memory


def section_memory(bars: Bars, max_gb: float = 5.0) -> dict[str, Any]:
    from ggTrader.engine import portfolio as engine_portfolio
    from ggTrader.lab.data import STOCK_BASE_CONFIG
    from ggTrader.lab.strategies.v2.contract import simulate_signal_out
    from ggTrader.lab.strategies.v2.ensemble import Ensemble
    from ggTrader.lab.universe import pit_mask

    def rss_mb() -> float:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    n_sym = len(bars.close.columns)
    out: dict[str, Any] = {
        "section": "memory",
        "symbols": n_sym,
        "bars": len(bars.close),
        "grid_combos": len(Ensemble.grid),
        "rss_mb_after_load": round(rss_mb()),
    }
    t0 = time.perf_counter()
    sig_out = Ensemble().build(bars)
    out["build_seconds"] = round(time.perf_counter() - t0, 2)
    out["rss_mb_after_build"] = round(rss_mb())
    out["signal_columns"] = sig_out.entries.shape[1]
    out["signal_frames_mb"] = round(
        (sig_out.entries.to_numpy().nbytes + sig_out.exits.to_numpy().nbytes) / 1e6, 1
    )
    mask = pit_mask(bars.close, min_history_bars=400, on_stale="ignore")
    est = engine_portfolio.estimate_peak_bytes(len(bars.close), sig_out.entries.shape[1], 1)
    out["engine_estimate_peak_gb_single_call"] = round(est / 1024**3, 2)
    max_bytes = int(max_gb * 1024**3)
    out["engine_budget_gb"] = max_gb
    t0 = time.perf_counter()
    res = simulate_signal_out(
        sig_out, bars.close, dict(STOCK_BASE_CONFIG), mask=mask, max_bytes=max_bytes
    )
    out["run_signals_seconds"] = round(time.perf_counter() - t0, 1)
    out["n_chunks"] = res.n_chunks
    out["groups"] = int(res.value.shape[1])
    out["trades"] = int(len(res.trades))
    out["rss_mb_peak"] = round(rss_mb())
    out["finite_equity"] = bool(np.isfinite(res.value.to_numpy()).all())
    log(f"memory: {out}")
    return out


# --------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--section", default="all", choices=("all", *SECTIONS))
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--max-gb", type=float, default=5.0, help="engine chunking budget for the memory section"
    )
    args = parser.parse_args(argv)
    wanted = SECTIONS if args.section == "all" else (args.section,)
    status = 0
    bars: Bars | None = None
    for name in wanted:
        log(f"=== {name}")
        t0 = time.perf_counter()
        try:
            if name == "grids":
                payload = section_grids()
            else:
                if bars is None:
                    bars = load_tape()
                    log(f"tape: {bars.close.shape}, missing {len(bars.missing_symbols)}")
                if name == "memory":
                    payload = section_memory(bars, args.max_gb)
                else:
                    payload = globals()[f"section_{name}"](bars)
            if bars is not None:
                payload["tape"] = {
                    **bars.provenance,
                    "symbols_loaded": len(bars.symbols),
                    "missing_symbols": list(bars.missing_symbols),
                }
            payload["seconds"] = round(time.perf_counter() - t0, 1)
            payload["peak_rss_mb"] = round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            )
            suffix = f"_budget_{args.max_gb:g}gb" if name == "memory" and args.max_gb != 5.0 else ""
            path = write_result_json(args.out_dir / f"p4_{name}{suffix}.json", payload)
            log(f"wrote {path}")
        except Exception:  # noqa: BLE001 - one failed section must not lose the others
            traceback.print_exc()
            status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
