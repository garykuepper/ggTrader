"""Measure live-vs-backtest signal drift from the paper trader's short indicator window.

`paper/signal_runner.generate_signals` loads `lookback_days` calendar days of
OHLCV (default 120, ~82 bars) and computes the 5-voter ensemble on that
window. EMA/MACD/RSI are recursive (``adjust=False``) and seeded at the
window's first bar, so their values -- and crossover events -- can differ
from the backtest, which computes them over years of history.

For every trading day d in the evaluation span and every candidate
``lookback_days`` L, this recomputes the ensemble on ``[d - L, d]`` (as live
does) and compares day-d entries/exits, and the "most recent exit date"
the trader's missed-exit catch-up uses, with a long-history reference. The
reference is computed once: every indicator is causal, so its value at d
depends only on bars <= d.

Read-only against the DB except for whatever `load_ohlcv` itself backfills.
Results -> docs/research/_live_indicator_warmup_results.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from ggTrader.data.core.index_constituents import normalize_yf_ticker, universe_members_asof
from ggTrader.lab.data import equity_universe_between, load_ohlcv
from ggTrader.lab.strategies.ensemble import EnsembleSignal
from ggTrader.lab.strategies.indicators import (
    bb_signals,
    ema_signals,
    extract_close,
    extract_volume,
    macd_signals,
    rsi_signals,
    volume_bb_signals,
)
from ggTrader.lab.strategy import LabConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "docs/research/_live_indicator_warmup_results.json"

REF_START = "2016-01-01"
EVAL_START = "2026-03-30"
EVAL_END = "2026-09-25"  # last complete session before the 09-28 partial bar
LOOKBACKS = [120, 180, 250, 365, 500, 730]
#: Live's eligibility floor (signal_runner: LabConfig(min_history_bars=60)).
LIVE_MIN_BARS = 60


def voter_signals(ens: EnsembleSignal, close: pd.DataFrame, volume: pd.DataFrame) -> dict:
    """Per-voter (entries, exits), same calls and params as EnsembleSignal._generate_signals."""
    return {
        "bb": bb_signals(close, ens.bb_period, ens.bb_std),
        "rsi": rsi_signals(close, ens.rsi_period, ens.rsi_oversold, ens.rsi_exit),
        "ema": ema_signals(close, ens.ema_fast, ens.ema_slow),
        "macd": macd_signals(
            close, ens.macd_fast, ens.macd_slow, ens.macd_signal, ens.divergence_window
        ),
        "vbb": volume_bb_signals(
            close, volume, ens.bb_period, ens.bb_std, ens.vol_period, ens.vol_mult
        ),
    }


def last_exit_dates(exits: pd.DataFrame, since: pd.Timestamp) -> pd.Series:
    """Most recent exit bar per symbol on/after ``since`` (NaT if none) -- live's `last_exit`."""
    ex = exits.loc[since:]
    has = ex.any()
    out = pd.Series(pd.NaT, index=ex.columns, dtype="datetime64[ns, UTC]")
    out[has] = ex.loc[:, has].iloc[::-1].idxmax()
    return out


def tally() -> dict:
    return {"ref": 0, "live": 0, "both": 0}


def add(t: dict, ref: pd.Series, live: pd.Series) -> None:
    t["ref"] += int(ref.sum())
    t["live"] += int(live.sum())
    t["both"] += int((ref & live).sum())


def summarize(t: dict) -> dict:
    missed, spurious = t["ref"] - t["both"], t["live"] - t["both"]
    union = t["ref"] + t["live"] - t["both"]
    return {
        **t,
        "missed": missed,
        "spurious": spurious,
        "mismatch_pct_of_union": 100 * (missed + spurious) / union if union else 0.0,
        "recall_pct": 100 * t["both"] / t["ref"] if t["ref"] else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--lookbacks", type=int, nargs="+", default=LOOKBACKS)
    args = ap.parse_args()

    eval_start, eval_end = pd.Timestamp(EVAL_START, tz="UTC"), pd.Timestamp(EVAL_END, tz="UTC")
    symbols = equity_universe_between(eval_start, eval_end, "sp500")
    ohlcv = load_ohlcv(symbols, REF_START, str((eval_end + pd.Timedelta(days=1)).date()), True)
    ohlcv = ohlcv.loc[:eval_end]
    have = sorted(ohlcv.columns.get_level_values(0).unique())
    close, volume = extract_close(ohlcv, have), extract_volume(ohlcv, have)
    days = close.index[(close.index >= eval_start) & (close.index <= eval_end)]

    ens = EnsembleSignal(LabConfig(min_history_bars=LIVE_MIN_BARS))
    ref = ens._generate_signals(close, volume)
    ref_v = voter_signals(ens, close, volume)

    members = {
        d: sorted({normalize_yf_ticker(m) for m in universe_members_asof("sp500", d)} & set(have))
        for d in days
    }

    out: dict = {
        "meta": {
            "eval_days": len(days),
            "eval_span": [str(days[0].date()), str(days[-1].date())],
            "reference": f"indicators over full history from {REF_START}",
            "universe": "S&P 500 members as of each day (PIT), symbols present in ohlcv",
            "n_symbols_loaded": len(have),
            "live_min_history_bars": LIVE_MIN_BARS,
            "ensemble": "EnsembleSignal defaults (5 voters, min_agree 2, RSI exit independent)",
        },
        "by_lookback": {},
    }

    for L in args.lookbacks:
        t0 = time.time()
        ent_t, ext_t, le_t = tally(), tally(), {"compared": 0, "differ": 0}
        voter_t = {v: {"entries": tally(), "exits": tally()} for v in ref_v}
        bars = []
        for d in days:
            lo = d - pd.Timedelta(days=L)
            win_c, win_v = close.loc[lo:d], volume.loc[lo:d]
            bars.append(len(win_c))
            elig = [s for s in members[d] if win_c[s].notna().sum() >= LIVE_MIN_BARS]
            if not elig:
                continue
            wc, wv = win_c[elig], win_v[elig]
            live = ens._generate_signals(wc, wv)
            add(ent_t, ref.entries.loc[d, elig], live.entries.loc[d, elig])
            add(ext_t, ref.exits.loc[d, elig], live.exits.loc[d, elig])
            lv = voter_signals(ens, wc, wv)
            for v in ref_v:
                add(voter_t[v]["entries"], ref_v[v][0].loc[d, elig], lv[v][0].loc[d, elig])
                add(voter_t[v]["exits"], ref_v[v][1].loc[d, elig], lv[v][1].loc[d, elig])
            # Catch-up input: latest exit inside the live window, per symbol.
            first = wc.index[0]
            r_le = last_exit_dates(ref.exits[elig].loc[:d], first)
            l_le = last_exit_dates(live.exits, first)
            either = r_le.notna() | l_le.notna()
            le_t["compared"] += int(either.sum())
            le_t["differ"] += int((either & (r_le != l_le)).sum())
        out["by_lookback"][str(L)] = {
            "median_bars": float(pd.Series(bars).median()),
            "entries": summarize(ent_t),
            "exits": summarize(ext_t),
            "last_exit_date": {
                **le_t,
                "differ_pct": 100 * le_t["differ"] / le_t["compared"] if le_t["compared"] else 0,
            },
            "voters": {v: {k: summarize(t) for k, t in vt.items()} for v, vt in voter_t.items()},
            "elapsed_s": round(time.time() - t0, 1),
        }
        e, x = out["by_lookback"][str(L)]["entries"], out["by_lookback"][str(L)]["exits"]
        print(
            f"L={L:4d} bars~{out['by_lookback'][str(L)]['median_bars']:.0f} "
            f"entries mismatch {e['mismatch_pct_of_union']:.1f}% (miss {e['missed']}, "
            f"spur {e['spurious']}) | exits mismatch {x['mismatch_pct_of_union']:.1f}% "
            f"(miss {x['missed']}, spur {x['spurious']})",
            flush=True,
        )
        args.out.write_text(json.dumps(out, indent=2, default=str))

    # The reported anecdote: AVB's 2026-08-14 RSI exit.
    d = pd.Timestamp("2026-08-14", tz="UTC")
    if "AVB" in have and d in close.index:
        lo = d - pd.Timedelta(days=120)
        wc = close.loc[lo:d, ["AVB"]]

        def rsi_at(c: pd.DataFrame) -> list:
            delta = c["AVB"].diff()
            g = delta.clip(lower=0).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
            ls = (-delta.clip(upper=0)).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
            rsi = 100 - 100 / (1 + g / ls)
            return [round(float(v), 2) for v in rsi.iloc[-2:]]

        out["avb_2026_08_14"] = {
            "ref_rsi_prev_and_day": rsi_at(close.loc[:d, ["AVB"]]),
            "live120_rsi_prev_and_day": rsi_at(wc),
            "ref_rsi_exit": bool(ref_v["rsi"][1].loc[d, "AVB"]),
            "live120_rsi_exit": bool(
                rsi_signals(wc, 14, ens.rsi_oversold, ens.rsi_exit)[1].loc[d, "AVB"]
            ),
            "live120_bars": len(wc),
        }
        print("AVB", out["avb_2026_08_14"])
    args.out.write_text(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
